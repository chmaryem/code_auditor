"""
code_mode_client.py — Client API pour le MCP Code Mode.
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

class GitHubClient:
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

    def get_available_tools(self) -> List[str]:
        return sorted(self._ensure_connected()._available_tools)

    def get_tool_mapping(self) -> Dict[str, str]:
        return self._ensure_connected().get_tool_mapping()

  
    def get_pr_info(self, owner: str, repo: str, pr_number: int) -> dict:
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_pull_request(owner, repo, pr_number))
        return r if isinstance(r, dict) else {}

    def get_pr_mergeable_status(self, owner: str, repo: str, pr_number: int) -> dict:
       
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

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main",
                         max_chars: int = 8000) -> str:
        """Contenu d'un fichier GitHub (base64 décodé automatiquement).

        max_chars=0 → fichier entier (requis pour la résolution de conflits ;
        une troncature corromprait le merge 3-way)."""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_file_content(owner, repo, path, ref, max_chars))
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

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> dict:
        return self.get_pr_info(owner, repo, pr_number)

    def get_pull_request_files(self, owner: str, repo: str, pr_number: int) -> list:
        return self.get_pr_files(owner, repo, pr_number)

    def get_file_contents(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        return self.get_file_content(owner, repo, path, ref)

    def list_open_prs(self, owner: str, repo: str, base: str = "") -> list:
        """Liste les PRs ouvertes. Returns: [{number, title, head:{ref,sha}, base:{ref}, ...}]"""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.list_open_prs(owner, repo, base))
        return r if isinstance(r, list) else []

    def get_pr_commits(self, owner: str, repo: str, pr_number: int) -> list:
        """Commits d'une PR. Returns: [{sha, commit:{message, author:{name}}}]"""
        svc = self._ensure_connected()
        r = _loop_manager.run(svc.get_pr_commits(owner, repo, pr_number))
        return r if isinstance(r, list) else []

    def create_pull_request_review(self, owner: str, repo: str, pr_number: int,
                                   body: str, event: str, comments: list = None) -> dict:
        return self.post_review(owner, repo, pr_number, body, event, comments)


    def push_files(self, owner: str, repo: str, path: str, content: str,
                   message: str, branch: str) -> dict:
        return self.push_file(owner, repo, path, content, message, branch)

    def add_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict:
        return self.post_comment(owner, repo, issue_number, body)

   
    def get_pull_request_reviews(self, owner: str, repo: str, pr_number: int) -> list:
        return self.get_pr_reviews(owner, repo, pr_number)

   
    def get_pull_request_status(self, owner: str, repo: str, pr_number: int) -> dict:
        return self.get_pr_mergeable_status(owner, repo, pr_number)


class RAGAnalyzer:
  

    def _check_content_cache(self, code: str) -> Optional[dict]:
        """Vérifie le cache Redis par hash du contenu (indépendant du chemin)."""
        try:
            from services.mcp_redis_service import get_mcp_redis, key_hash, KEY_PREFIX
            content_hash = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
            redis = get_mcp_redis()
            
            path_hash = redis.get(f"{KEY_PREFIX}fch:{content_hash[:16]}")
            if not path_hash:
                return None
            # Lire l'analyse depuis le hash principal
            redis_key = f"{KEY_PREFIX}fc:{path_hash}"
            analysis_text = redis.hget(redis_key, "analysis_text")
            if not analysis_text:
                return None
            from smart_git.git_hook import _count_severity_from_blocks
            c, h, m, score = _count_severity_from_blocks(analysis_text)
            logger.info("Cache hit SHA=%s... score=%.0f", content_hash[:8], score)
            return {
                "analysis": f"[FROM WATCH CACHE]\n{analysis_text[:800]}",
                "critical": c, "high": h, "medium": m, "score": score,
                "relevant_knowledge": [], "source": "watch_cache",
            }
        except Exception as e:
            logger.debug("Content cache: %s", e)
            return None

    def analyze(self, code: str, file_path: str, language: str, patch: str = "") -> dict:
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
           
            if len(code) > 5000:
                result = assistant_agent.analyze_code_chunked(code=code, context=context)
            else:
                result = assistant_agent.analyze_code_with_rag(code=code, context=context)
            analysis_text = result.get("analysis", "")

    
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
        
        try:
            from smart_git.git_hook import _count_severity_from_blocks
            c, h, m, score = _count_severity_from_blocks(analysis_text)
            return {"critical": c, "high": h, "medium": m, "score": score, "analysis": analysis_text}
        except Exception:
            return {"critical": 0, "high": 0, "medium": 0, "score": 0, "analysis": analysis_text}

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



github   = GitHubClient()
rag      = RAGAnalyzer()
kg       = KnowledgeGraphClient()
cache    = CacheClient()
resolver = ConflictResolver()


API_DOCUMENTATION = """

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