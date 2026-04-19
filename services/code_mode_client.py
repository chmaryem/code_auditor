"""
code_mode_client.py — Client API pour le MCP Code Mode.

FIX v6 — Correction du bug "GitHubClient has no attribute get_pull_request" :

  CAUSE DU BUG :
    Gemini générait du code utilisant les noms MCP bruts :
      github.get_pull_request()         → n'existe pas (nom MCP, pas le wrapper)
      github.get_pull_request_files()   → n'existe pas
      github.create_pull_request_review() → n'existe pas
    Car le system prompt disait "MCP tools available" et Gemini utilisait ces noms.

  DOUBLE FIX :
    1. Alias défensifs dans GitHubClient :
       get_pull_request = get_pr_info                    (alias)
       get_pull_request_files = get_pr_files             (alias)
       create_pull_request_review = post_review          (alias)
       get_file_contents = get_file_content              (alias, MCP name has 's')
       push_files = push_file                            (alias, MCP name has 's')
       Ainsi les DEUX nommages fonctionnent — quel que soit ce que Gemini génère.

    2. System prompt corrigé dans code_mode_agent.py pour imposer les noms wrappers.

  AUTRES FIXES v6 (hérités de v5) :
    - RAGAnalyzer.analyze() : cache SQLite par hash contenu (0 token si cache hit)
    - RAGAnalyzer.analyze() : fallback statique si quota 429 (score réel, pas 0)
    - rag.count_severity(text) : parse un texte existant, 0 token LLM
    - get_pr_mergeable_status() : délègue à mcp_github_service → mergeableState fiable
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Event loop manager (inchangé)
# ─────────────────────────────────────────────────────────────────────────────

class _LoopManager:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock  = threading.Lock()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or (self._thread and not self._thread.is_alive()):
                self._ready.clear()
                self._thread = threading.Thread(
                    target=self._run_loop, daemon=True, name="mcp-event-loop"
                )
                self._thread.start()
                self._ready.wait(timeout=5)
        return self._loop

    def run(self, coro, timeout: int = 180):
        loop = self.get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def shutdown(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


_loop_manager = _LoopManager()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback statique (activé uniquement si quota 429 sur tous les modèles)
# ─────────────────────────────────────────────────────────────────────────────

class _StaticFallbackAnalyzer:
    """
    Analyse statique par regex.
    Appelée UNIQUEMENT quand RAGAnalyzer.analyze() reçoit un 429 irrecupérable.
    Ne remplace pas le pipeline RAG — c'est un filet de sécurité.
    """

    JAVA_RULES = [
        (r'"\s*\+\s*(username|user|input|query|id)\b',           "CRITICAL", "SQL Injection: string concat in query"),
        (r'Statement\.executeQuery\s*\(\s*"[^"]*"\s*\+',         "CRITICAL", "SQL Injection: raw Statement"),
        (r'(password|passwd|secret|apikey)\s*=\s*"[^"]{4,}"',    "CRITICAL", "Hardcoded credential"),
        (r'\.equals\s*\("admin"\).*&&.*\.equals\s*\("',          "CRITICAL", "Hardcoded admin backdoor"),
        (r'MessageDigest\.getInstance\s*\("MD5"\)',               "CRITICAL", "MD5 for passwords (broken)"),
        (r'return\s+password\s*;',                                "CRITICAL", "Password returned in plaintext"),
        (r'"hashed_"\s*\+',                                       "CRITICAL", "Fake placeholder password hash"),
        (r'MessageDigest\.getInstance\s*\("SHA-1"\)',             "HIGH",     "SHA-1 insufficient for passwords"),
        (r'SELECT \* FROM \w+(?!\s+WHERE)',                       "HIGH",     "SELECT * without WHERE clause"),
        (r'catch\s*\(\s*(Exception|Throwable)\s+\w+\s*\)\s*\{\s*\}', "HIGH", "Exception swallowed silently"),
        (r'static\s+(List|Map|Set|ArrayList|HashMap)\s*<',        "HIGH",     "Mutable static state (thread-unsafe)"),
        (r'System\.out\.println\s*\(',                            "MEDIUM",   "System.out.println in production"),
        (r'e\.printStackTrace\s*\(\s*\)',                         "MEDIUM",   "printStackTrace in production"),
    ]

    PYTHON_RULES = [
        (r'cursor\.execute\s*\(\s*["\'].*%|cursor\.execute\s*\(\s*f["\']', "CRITICAL", "SQL Injection"),
        (r'(password|secret|api_key)\s*=\s*["\'][^"\']{4,}["\']',          "CRITICAL", "Hardcoded credential"),
        (r'eval\s*\(|exec\s*\(',                                            "CRITICAL", "eval/exec arbitrary code"),
        (r'except\s*:\s*pass|except\s+Exception\s*:\s*pass',               "HIGH",     "Exception swallowed"),
    ]

    def analyze(self, code: str, language: str) -> dict:
        lang  = language.lower()
        rules = self.JAVA_RULES if lang == "java" else self.PYTHON_RULES
        findings, c, h, m = [], 0, 0, 0
        for pattern, sev, desc in rules:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                findings.append(f"[{sev}] {desc}")
                if sev == "CRITICAL": c += 1
                elif sev == "HIGH":   h += 1
                else:                 m += 1
        score = c * 10.0 + h * 3.0 + m * 1.0
        text  = "[STATIC ANALYSIS — LLM quota exceeded]\n" + ("\n".join(findings) or "No obvious issues.")
        return {"analysis": text, "critical": c, "high": h, "medium": m,
                "score": score, "relevant_knowledge": [], "source": "static_fallback"}


_static_fallback = _StaticFallbackAnalyzer()


# ─────────────────────────────────────────────────────────────────────────────
# GitHubClient — wrappers MCP avec ALIAS défensifs
# ─────────────────────────────────────────────────────────────────────────────

class GitHubClient:
    """
    Client GitHub via MCP GitHub Server.

    NOMMAGE DOUBLE (v6) :
      Chaque opération a deux noms :
        1. Nom wrapper (recommandé) : get_pr_info, get_pr_files, post_review…
        2. Alias MCP brut : get_pull_request, get_pull_request_files, create_pull_request_review…

      Les alias garantissent que le code généré par Gemini fonctionne
      indépendamment du nommage qu'il choisit.
    """

    def __init__(self):
        self._service = None

    def _get_service(self):
        if self._service is None:
            from services.mcp_github_service import MCPGitHubService
            self._service = MCPGitHubService()
        return self._service

    def _ensure_connected(self):
        svc = self._get_service()
        if svc._session is None:
            _loop_manager.run(svc.connect())
        return svc

    def disconnect(self):
        if self._service and self._service._session:
            try:
                _loop_manager.run(self._service.disconnect(), timeout=10)
            except Exception:
                pass

    # ── Discovery ─────────────────────────────────────────────────────────────

    def get_available_tools(self) -> List[str]:
        return sorted(self._ensure_connected()._available_tools)

    def get_tool_mapping(self) -> Dict[str, str]:
        return self._ensure_connected().get_tool_mapping()

    # ── Pull Requests — wrappers ───────────────────────────────────────────────

    def get_pr_info(self, owner: str, repo: str, pr_number: int) -> dict:
        """Infos PR. Returns: {title, state, body, mergeable, base:{ref}, head:{ref,sha}}"""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_pull_request(owner, repo, pr_number))
        return r if isinstance(r, dict) else {}

    def get_pr_mergeable_status(self, owner: str, repo: str, pr_number: int) -> dict:
        """
        Détecte les conflits de manière fiable (polling + mergeableState).
        Returns: {has_conflicts, mergeable, conflict_files, base_ref, head_ref, head_sha, pr_data}
        """
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_pr_mergeable_status(owner, repo, pr_number), timeout=90)
        return r if isinstance(r, dict) else {
            "has_conflicts": False, "mergeable": None,
            "conflict_files": [], "base_ref": "main",
            "head_ref": "", "head_sha": "", "pr_data": {},
        }

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list:
        """Fichiers modifiés dans une PR. Returns: [{filename, status, patch, additions, deletions}]"""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_pr_files(owner, repo, pr_number))
        return r if isinstance(r, list) else []

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """Contenu d'un fichier GitHub (base64 décodé automatiquement)."""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_file_content(owner, repo, path, ref))
        return r if isinstance(r, str) else ""

    def post_review(self, owner: str, repo: str, pr_number: int,
                    body: str, event: str, comments: list = None) -> dict:
        """Soumet un review. event: 'APPROVE' | 'REQUEST_CHANGES' | 'COMMENT'"""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.create_pr_review(owner, repo, pr_number, body, event, comments))
        return r if isinstance(r, dict) else {}

    def post_comment(self, owner: str, repo: str, pr_number: int, body: str) -> dict:
        """Poste un commentaire général sur une PR."""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.post_pr_comment(owner, repo, pr_number, body))
        return r if isinstance(r, dict) else {}

    def get_check_runs(self, owner: str, repo: str, ref: str) -> list:
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_check_runs(owner, repo, ref))
        return r if isinstance(r, list) else []

    def get_pr_reviews(self, owner: str, repo: str, pr_number: int) -> list:
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_pr_reviews(owner, repo, pr_number))
        return r if isinstance(r, list) else []

    def create_branch(self, owner: str, repo: str, branch: str, from_ref: str = "main") -> dict:
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.create_branch(owner, repo, branch, from_ref))
        return r if isinstance(r, dict) else {}

    def push_file(self, owner: str, repo: str, path: str, content: str,
                  message: str, branch: str) -> dict:
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.create_or_update_file(owner, repo, path, content, message, branch))
        return r if isinstance(r, dict) else {}

    def create_pull_request(self, owner: str, repo: str, title: str,
                             body: str, head: str, base: str = "main") -> dict:
        """Crée une nouvelle PR. Returns: {number, html_url, title, state}"""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.create_pull_request(owner, repo, title, body, head, base))
        return r if isinstance(r, dict) else {}

    # ── ALIAS DÉFENSIFS (v6) ───────────────────────────────────────────────────
    # Gemini génère parfois les noms MCP bruts au lieu des noms wrappers.
    # Ces alias font fonctionner les deux nommages.

    # get_pull_request → get_pr_info
    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> dict:
        return self.get_pr_info(owner, repo, pr_number)

    # get_pull_request_files → get_pr_files
    def get_pull_request_files(self, owner: str, repo: str, pr_number: int) -> list:
        return self.get_pr_files(owner, repo, pr_number)

    # get_file_contents (MCP a un 's') → get_file_content
    def get_file_contents(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        return self.get_file_content(owner, repo, path, ref)

    # create_pull_request_review → post_review
    def create_pull_request_review(self, owner: str, repo: str, pr_number: int,
                                   body: str, event: str, comments: list = None) -> dict:
        return self.post_review(owner, repo, pr_number, body, event, comments)

    # push_files (MCP a un 's') → push_file
    def push_files(self, owner: str, repo: str, path: str, content: str,
                   message: str, branch: str) -> dict:
        return self.push_file(owner, repo, path, content, message, branch)

    # add_issue_comment → post_comment (parfois utilisé pour les PR aussi)
    def add_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict:
        return self.post_comment(owner, repo, issue_number, body)

    # get_pull_request_reviews → get_pr_reviews
    def get_pull_request_reviews(self, owner: str, repo: str, pr_number: int) -> list:
        return self.get_pr_reviews(owner, repo, pr_number)

    # get_pull_request_status → get_pr_mergeable_status
    def get_pull_request_status(self, owner: str, repo: str, pr_number: int) -> dict:
        return self.get_pr_mergeable_status(owner, repo, pr_number)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAnalyzer — Pipeline complet ChromaDB + KG + Gemini PRÉSERVÉ
# + Cache-first par hash contenu + Fallback statique si quota
# ─────────────────────────────────────────────────────────────────────────────

class RAGAnalyzer:
    """
    Pipeline RAG complet (ARCHITECTURE PRÉSERVÉE).

    Ordre d'exécution :
      1. Cache SQLite par hash SHA256 du contenu → 0 token si hit
      2. Pipeline RAG : ChromaDB + KG + Gemini (assistant_agent.analyze_code_with_rag)
      3. Si quota 429 → fallback statique (_StaticFallbackAnalyzer)
    """

    def _check_content_cache(self, code: str) -> Optional[dict]:
        """Vérifie le cache SQLite par hash du contenu (indépendant du chemin)."""
        try:
            import sqlite3
            from config import config
            content_hash = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
            cache_db = config.CACHE_DIR / "analysis_cache.db"
            if not cache_db.exists():
                return None
            conn = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
            row = conn.execute(
                "SELECT analysis_text FROM file_cache WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
            conn.close()
            if not row or not row[0]:
                return None
            from smart_git.git_hook import _count_severity_from_blocks
            c, h, m, score = _count_severity_from_blocks(row[0])
            logger.info("Cache hit SHA=%s... score=%.0f", content_hash[:8], score)
            return {
                "analysis": f"[FROM WATCH CACHE]\n{row[0][:800]}",
                "critical": c, "high": h, "medium": m, "score": score,
                "relevant_knowledge": [], "source": "watch_cache",
            }
        except Exception as e:
            logger.debug("Content cache: %s", e)
            return None

    def analyze(self, code: str, file_path: str, language: str, patch: str = "") -> dict:
        """
        Pipeline RAG complet avec cache-first et fallback statique.

        1. Cache SQLite par hash contenu (0 token)
        2. Pipeline RAG : ChromaDB + KG + Gemini
        3. Si 429 → fallback statique
        """
        # Couche 1 : Cache (0 token LLM)
        cached = self._check_content_cache(code)
        if cached is not None:
            return cached

        # Couche 2 : Pipeline RAG complet
        try:
            from services.llm_service import assistant_agent
            context = {"file_path": file_path, "language": language}
            if patch:
                context["pr_patch"] = patch
                context["post_solution_hint"] = (
                    "Focus on the CHANGED lines from the patch/diff, "
                    "but consider the full file for context."
                )
            result       = assistant_agent.analyze_code_with_rag(code=code, context=context)
            analysis_text = result.get("analysis", "")

            # FIX v6.1 — Détecter le 429 caché dans le texte d'analyse.
            # assistant_agent peut capturer le 429 en interne et retourner
            # analysis="Error: 429 RESOURCE_EXHAUSTED..." au lieu de lever une exception.
            # Dans ce cas, _count_severity_from_blocks() ne trouve aucun bloc → score=0.
            # On bascule alors sur le fallback statique pour avoir un vrai score.
            _quota_markers = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")
            if analysis_text and any(m in analysis_text for m in _quota_markers):
                logger.warning("RAG returned quota error in text → static fallback for %s", file_path)
                r = _static_fallback.analyze(code, language)
                r["error"] = analysis_text[:120]
                return r

            from smart_git.git_hook import _count_severity_from_blocks
            c, h, m, score = _count_severity_from_blocks(analysis_text)
            return {
                "analysis": analysis_text, "critical": c, "high": h, "medium": m,
                "score": score, "relevant_knowledge": result.get("relevant_knowledge", []),
                "source": "rag",
            }
        except Exception as e:
            err = str(e)
            is_quota = "429" in err or "RESOURCE_EXHAUSTED" in err
            if is_quota:
                # Couche 3 : Fallback statique (si quota épuisé sur tous les modèles)
                logger.warning("RAG quota exception → static fallback for %s", file_path)
                r = _static_fallback.analyze(code, language)
                r["error"] = err[:120]
                return r
            logger.error("RAG analyze: %s", e)
            return {"analysis": f"Error: {e}", "critical": 0, "high": 0, "medium": 0, "score": 0}

    def count_severity(self, analysis_text: str) -> dict:
        """
        Parse un texte d'analyse existant pour extraire les counts.
        0 token LLM. Utilisé quand cache.read_analysis() retourne un résultat.
        """
        try:
            from smart_git.git_hook import _count_severity_from_blocks
            c, h, m, score = _count_severity_from_blocks(analysis_text)
            return {"critical": c, "high": h, "medium": m, "score": score, "analysis": analysis_text}
        except Exception:
            return {"critical": 0, "high": 0, "medium": 0, "score": 0, "analysis": analysis_text}


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeGraphClient, CacheClient, ConflictResolver (inchangés)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphClient:
    def detect_patterns(self, code: str, language: str) -> list:
        try:
            from services.knowledge_graph import knowledge_graph
            if not knowledge_graph._built:
                return []
            detected = knowledge_graph.detect_patterns(code, language, parsed_entities=[])
            return [name for name, _ in detected]
        except Exception as e:
            logger.debug("KG detect_patterns: %s", e)
            return []

    def has_pattern(self, code: str, language: str) -> bool:
        return len(self.detect_patterns(code, language)) > 0


class CacheClient:
    def read_analysis(self, file_path: str) -> Optional[str]:
        """
        Lit l'analyse depuis le cache SQLite du mode Watch.
        Appeler AVANT rag.analyze() pour économiser les tokens LLM.
        """
        try:
            from config import config
            from smart_git.git_hook import _read_analysis_fresh
            cache_db = config.CACHE_DIR / "analysis_cache.db"
            return _read_analysis_fresh(file_path, cache_db)
        except Exception as e:
            logger.debug("Cache read: %s", e)
            return None

    def get_recurring_patterns(self, file_path: str, min_count: int = 2) -> list:
        try:
            from config import config
            from smart_git.git_hook import _get_recurring_patterns
            cache_db = config.CACHE_DIR / "analysis_cache.db"
            return _get_recurring_patterns(file_path, cache_db, min_count)
        except Exception as e:
            logger.debug("Recurring patterns: %s", e)
            return []


class ConflictResolver:
    def resolve(self, file_path: str, conflicted_content: str,
                ours_content: str, theirs_content: str,
                project_context: str = "") -> Optional[str]:
        try:
            from smart_git.git_conflict_resolver import resolve_single_file
            return resolve_single_file(
                file_path=file_path,
                conflicted_content=conflicted_content,
                ours_content=ours_content,
                theirs_content=theirs_content,
                project_context=project_context,
            )
        except Exception as e:
            logger.error("Conflict resolution: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Instances pré-initialisées (importées dans le sandbox)
# ─────────────────────────────────────────────────────────────────────────────

github   = GitHubClient()
rag      = RAGAnalyzer()
kg       = KnowledgeGraphClient()
cache    = CacheClient()
resolver = ConflictResolver()


# ─────────────────────────────────────────────────────────────────────────────
# API_DOCUMENTATION — injectée dans les system prompts (v6)
# ─────────────────────────────────────────────────────────────────────────────

API_DOCUMENTATION = """
## Available API — code_mode_client (v6)

### RÈGLE CRITIQUE — Noms exacts des méthodes github.*
UTILISE EXACTEMENT ces noms. NE PAS utiliser les noms MCP bruts.

### github (GitHubClient) — NOMS EXACTS À UTILISER :
- github.get_pr_info(owner, repo, pr_number)          → dict {title, state, base:{ref}, head:{ref,sha}}
- github.get_pr_mergeable_status(owner, repo, pr_number) → dict {has_conflicts, base_ref, head_ref, head_sha}
- github.get_pr_files(owner, repo, pr_number)          → list[{filename, status, patch}]
- github.get_file_content(owner, repo, path, ref)      → str
- github.post_review(owner, repo, pr_number, body, event, comments=[]) → dict
  event = "APPROVE" | "REQUEST_CHANGES" | "COMMENT"
- github.post_comment(owner, repo, pr_number, body)    → dict
- github.get_check_runs(owner, repo, ref)              → list[dict]
- github.get_pr_reviews(owner, repo, pr_number)        → list[dict]
- github.create_branch(owner, repo, branch, from_ref)  → dict
- github.push_file(owner, repo, path, content, message, branch) → dict
- github.create_pull_request(owner, repo, title, body, head, base) → dict {html_url}

### rag (RAGAnalyzer) — Pipeline ChromaDB + KG + Gemini :
- rag.analyze(code, file_path, language, patch="")     → dict {analysis, critical, high, medium, score}
- rag.count_severity(analysis_text)                    → dict {critical, high, medium, score}  [0 tokens]

### kg (KnowledgeGraphClient) :
- kg.detect_patterns(code, language)                   → list[str]

### cache (CacheClient) — SQLite Watch cache :
- cache.read_analysis(file_path)                       → str | None   [CHECK FIRST before rag.analyze]
- cache.get_recurring_patterns(file_path, min_count=2) → list[dict]

### resolver (ConflictResolver) :
- resolver.resolve(file_path, conflicted, ours, theirs) → str | None

### PATTERN CACHE-FIRST (toujours utiliser) :
  cached = cache.read_analysis(filename)
  if cached:
      result = rag.count_severity(cached)   # 0 tokens LLM
  else:
      result = rag.analyze(content, filename, language, patch)  # RAG complet
"""