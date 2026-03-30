"""
FeedbackProcessor — Self-Improving Graph RAG  (mode batch différé).

NOUVEAU comportement vs l'ancienne version :

  AVANT : interrompait le développeur après CHAQUE analyse
    → 19 prompts d'affilée pendant qu'il regarde les résultats
    → timeout auto-skip sur tout si absent 20s

  APRÈS : deux modes séparés

    Mode 1 — Auto-promotion (immédiate, silencieuse)
      Déclenché si severity == CRITICAL OU pattern critique connu
      → règle promue sans demander, message discret dans le log
      → le développeur n'est jamais interrompu

    Mode 2 — Batch différé (à la fin de session)
      Tout le reste s'accumule dans _pending pendant la session
      → présenté UNE SEULE FOIS quand le dev tape Ctrl+C
      → interface claire : tout accepter / choisir / ignorer

Pipeline complet (inchangé en interne) :
  1. déduplication ChromaDB
  2. LLM généralise le fix → règle .md
  3. écriture dans data/knowledge_base/auto_learned/<language>/
  4. rechargement ChromaDB
  5. mise à jour KnowledgeGraph
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ajouter après les imports existants
from services.web_search_client import web_search_client


logger = logging.getLogger(__name__)

#  Constantes ANSI 
_R  = "\033[0m"
_BD = "\033[1m"
_DM = "\033[2m"
_GR = "\033[92m"
_YL = "\033[93m"
_CY = "\033[96m"
_OR = "\033[38;5;208m"
_RD = "\033[91m"
_W  = 68

_SEV_COLOR = {
    "CRITICAL": _RD,
    "HIGH":     _OR,
    "MEDIUM":   _YL,
    "LOW":      "\033[94m",
}

#  Seuils et critères ────────────────────────────────────────────────────────

# Score cosinus en-dessous duquel une règle est considérée "déjà connue"
DEDUP_THRESHOLD = 0.35   # plus permissif que 0.15 pour absorber les variations

# Limite d'auto-promotions par session (appels LLM = précieux sur free tier)
# Les règles au-delà de cette limite vont dans le batch différé
MAX_AUTO_PER_SESSION = 3

# Patterns de problèmes qui déclenchent l'auto-promotion (sans demander)
# Match sur problem.lower() ou why.lower()
AUTO_PROMOTE_PATTERNS = {
    "sql injection", "sqli", "sql concatenation",
    "plain text password", "cleartext password", "password in plain",
    "hardcoded credential", "hardcoded password", "hardcoded secret",
    "authentication bypass", "remote code execution", "rce",
    "path traversal", "directory traversal",
    "xxe", "xml injection", "deserializ",
}


@dataclass
class _Candidate:
    """Un fix candidat accumulé pendant la session."""
    block:       Dict
    code_before: str
    language:    str
    file_name:   str
    occurrences: int  = 1    # compté si même problème dans plusieurs fichiers
    auto:        bool = False # True = auto-approuvé (pas encore généralisé)


class FeedbackProcessor:
    """
    Accumule les fixes pendant la session, auto-promeut les critiques,
    et présente un bilan UNE FOIS à la fin (Ctrl+C).
    """

    def __init__(self, llm, vector_store, kb_dir: Path, kb_loader, knowledge_graph):
        self._llm        = llm
        self._store      = vector_store
        self._kb_dir     = Path(kb_dir)
        self._kb_loader  = kb_loader
        self._kg         = knowledge_graph
        self._auto_dir   = self._kb_dir / "auto_learned"
        self._auto_dir.mkdir(parents=True, exist_ok=True)

        # Candidats accumulés pendant la session (batch différé)
        self._pending: List[_Candidate] = []
        self._pending_lock = threading.Lock()

        self._stats = {
            "auto_promoted": 0,   # promus automatiquement (CRITIQUE)
            "batch_promoted": 0,  # promus via bilan de fin de session
            "rejected":      0,
            "deduped":       0,
            "errors":        0,
        }

        # Délai minimum entre deux appels LLM dans le feedback processor
        # Évite les 429 en partageant le quota avec l'analyse principale
        self._last_llm_call: float = 0.0
        self._llm_min_delay: float = 12.0  # secondes (free tier = 5 req/min)

        # CORRECTION 1 : compteur d'auto-promotions (limité à MAX_AUTO_PER_SESSION)
        self._auto_count: int = 0

        # CORRECTION 3 : garde contre les appels multiples de flush_session()
        self._flushed: bool = False

    # ── API publique — appelée par LearningAgent ──────────────────────────────

    def collect_feedback(
        self,
        blocks:          List[Dict],
        code_before:     str,
        language:        str,
        file_name:       str,
        project_indexer  = None,
        dependency_graph = None,
    ) -> None:
        """
        Triage immédiat de chaque bloc :
          CRITIQUE ou pattern grave → auto-promotion silencieuse
          Reste                     → accumulation dans le batch
        """
        if not blocks:
            return

        valid = [b for b in blocks if b.get("problem")]
        if not valid:
            return

        for block in valid:
            try:
                self._triage(block, code_before, language, file_name)
            except Exception as e:
                logger.error("FeedbackProcessor triage erreur : %s", e)
                self._stats["errors"] += 1

    def flush_session(self) -> None:
        """
        Appelé UNE SEULE FOIS depuis LearningAgent.stop() quand le dev tape Ctrl+C.
        Présente le bilan de session et demande les actions en une seule interaction.
        """
        # CORRECTION 3 : protection contre les appels multiples
        if self._flushed:
            logger.debug("flush_session appelé plusieurs fois — ignoré")
            return
        self._flushed = True

        with self._pending_lock:
            pending = list(self._pending)

        # Dédupliquation finale (double-check ChromaDB sur les candidats accumulés)
        fresh = []
        for c in pending:
            if not self._rule_already_exists(c.block.get("problem", ""), c.language):
                fresh.append(c)
            else:
                self._stats["deduped"] += 1

        # Séparer auto-approuvés et batch manuel
        auto_fresh  = [c for c in fresh if c.auto]
        batch_fresh = [c for c in fresh if not c.auto]

        # Promouvoir les auto-approuvés (appels LLM espacés)
        if auto_fresh:
            print(f"\n  {_DM}Généralisation des {len(auto_fresh)} règle(s) auto-approuvée(s)...{_R}")
            for c in auto_fresh:
                ok = self._promote_to_kb_throttled(c.block, c.language, c.code_before)
                if ok:
                    self._stats["auto_promoted"] += 1
                    problem = c.block.get("problem", "")
                    print(f"  {_GR}✓{_R} {_DM}{problem[:60]}{_R}")
            print()

        if not batch_fresh:
            if self._stats["auto_promoted"] > 0:
                print(f"  {_DM}KB session : {self._stats['auto_promoted']} règle(s) "
                      f"auto-promue(s), aucun candidat batch restant.{_R}")
            self._print_session_footer()
            return

        if not self._is_interactive():
            logger.info("flush_session : stdin non interactif — %d candidats batch ignorés",
                        len(batch_fresh))
            self._print_session_footer()
            return

        self._print_session_summary(batch_fresh)
        choice = self._ask_batch_choice()

        if choice == "a":
            for c in batch_fresh:
                ok = self._promote_to_kb_throttled(c.block, c.language, c.code_before)
                if ok:
                    self._stats["batch_promoted"] += 1
            print(f"\n  {_GR}✓ {self._stats['batch_promoted']} règle(s) batch ajoutée(s){_R}\n")

        elif choice == "c":
            self._interactive_batch(batch_fresh)

        else:
            self._stats["rejected"] += len(batch_fresh)
            print(f"\n  {_DM}Bilan batch ignoré.{_R}\n")

        self._print_session_footer()

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ── Triage immédiat ───────────────────────────────────────────────────────

    def _triage(self, block: Dict, code_before: str, language: str, file_name: str):
        """Décide : auto-promouvoir maintenant ou accumuler pour le batch."""
        problem  = block.get("problem",  "").lower()
        why      = block.get("why",      "").lower()
        severity = block.get("severity", "MEDIUM").upper()

        # Déduplication préalable (ne rien faire si déjà connu)
        if self._rule_already_exists(block.get("problem", ""), language):
            self._stats["deduped"] += 1
            return

        # Critère d'auto-promotion
        is_critical_severity = (severity == "CRITICAL")
        is_critical_pattern  = any(p in problem or p in why
                                   for p in AUTO_PROMOTE_PATTERNS)

        if (is_critical_severity or is_critical_pattern) and self._auto_count < MAX_AUTO_PER_SESSION:
            self._auto_promote(block, language, code_before, file_name)
        else:
            # Limite atteinte → batch différé au lieu d'appeler Gemini maintenant
            self._add_to_batch(block, code_before, language, file_name)

    def _auto_promote(self, block: Dict, language: str, code_before: str, file_name: str):
        """
        Marque le bloc comme "auto-approuvé" sans appel LLM immédiat.
        Le message discret est affiché maintenant.
        La généralisation LLM se fait à flush_session() pour ne pas
        concurrencer l'analyse principale sur le quota Gemini.
        """
        problem  = block.get("problem", "")
        severity = block.get("severity", "CRITICAL")
        color    = _SEV_COLOR.get(severity, _RD)

        # Ajouter dans le batch avec flag auto=True (sera promu à flush_session)
        with self._pending_lock:
            for c in self._pending:
                if (c.block.get("problem", "") == problem and c.language == language):
                    c.occurrences += 1
                    return
            self._pending.append(_Candidate(
                block=block, code_before=code_before,
                language=language, file_name=file_name,
                auto=True,
            ))

        self._auto_count += 1

        remaining = MAX_AUTO_PER_SESSION - self._auto_count
        limit_tag = f"  {_DM}({remaining} restant(s)){_R}" if remaining >= 0 else ""

        # Affichage discret immédiat (sans LLM)
        print(f"  {_DM}KB auto ← [{color}{severity}{_R}{_DM}] "
              f"{problem[:60]}{'…' if len(problem) > 60 else ''} "
              f"({file_name}){limit_tag}{_R}")

    def _add_to_batch(self, block: Dict, code_before: str, language: str, file_name: str):
        """Ajoute un candidat au batch, ou incrémente son compteur si déjà présent."""
        problem = block.get("problem", "")
        with self._pending_lock:
            for c in self._pending:
                if (c.block.get("problem", "") == problem
                        and c.language == language):
                    c.occurrences += 1
                    return
            self._pending.append(_Candidate(
                block=block, code_before=code_before,
                language=language, file_name=file_name,
            ))

    # ── Interface batch de fin de session ─────────────────────────────────────

    def _print_session_summary(self, candidates: List[_Candidate]) -> None:
        """Affiche le tableau récapitulatif des candidats batch (hors auto)."""
        print(f"\n{'═'*_W}")
        print(f"  {_CY}{_BD}KB — Candidats batch{_R}")
        print(f"{'═'*_W}")

        print(f"\n  {len(candidates)} règle(s) en attente de validation :\n")

        for i, c in enumerate(candidates, 1):
            severity = c.block.get("severity", "MEDIUM")
            problem  = c.block.get("problem",  "")
            color    = _SEV_COLOR.get(severity, _YL)
            occ_tag  = f"  {_DM}(×{c.occurrences}){_R}" if c.occurrences > 1 else ""
            print(f"  {_DM}{i:2d}.{_R} [{_BD}{color}{severity:<8}{_R}]  "
                  f"{problem[:48]:<48}  {_DM}{c.language}  {c.file_name}{_R}{occ_tag}")

        print()

    def _ask_batch_choice(self) -> str:
        """Demande l'action globale. Pas de timeout — c'est la fin de session."""
        prompt = (f"  {_CY}→ Que faire ?{_R}  "
                  f"{_BD}[a]{_R}ll (tout promouvoir)  "
                  f"{_BD}[c]{_R}hoisir (une par une)  "
                  f"{_BD}[n]{_R}on (ignorer) : ")
        try:
            print(prompt, end="", flush=True)
            answer = sys.stdin.readline().strip().lower()
            return answer if answer in ("a", "c", "n") else "n"
        except (EOFError, KeyboardInterrupt):
            return "n"

    def _interactive_batch(self, candidates: List[_Candidate]) -> None:
        """Présente chaque candidat individuellement pour accept/reject."""
        for i, c in enumerate(candidates, 1):
            severity = c.block.get("severity", "MEDIUM")
            problem  = c.block.get("problem",  "")
            why      = (c.block.get("why", "") or "").replace("\n", " ").strip()
            color    = _SEV_COLOR.get(severity, _YL)

            print(f"\n  {_DM}[{i}/{len(candidates)}]{_R} [{_BD}{color}{severity}{_R}] "
                  f"{_BD}{problem}{_R}")
            if why:
                short = why[:120] + "…" if len(why) > 120 else why
                print(f"  {_DM}{short}{_R}")

            try:
                print(f"  {_CY}→ [a]ccept  [r]eject  [s]kip reste : {_R}",
                      end="", flush=True)
                answer = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if answer == "a":
                ok = self._promote_to_kb(c.block, c.language, c.code_before)
                if ok:
                    self._stats["batch_promoted"] += 1
                    print(f"  {_GR}✓ Ajoutée{_R}")
                else:
                    print(f"  {_YL}⚠ Échouée{_R}")
            elif answer == "s":
                print(f"  {_DM}Suite ignorée.{_R}")
                break
            else:
                self._stats["rejected"] += 1
                print(f"  {_DM}Rejetée.{_R}")

    def _print_session_footer(self) -> None:
        s = self._stats
        parts = []
        if s["auto_promoted"]:  parts.append(f"{_GR}{s['auto_promoted']} auto{_R}")
        if s["batch_promoted"]: parts.append(f"{_GR}{s['batch_promoted']} batch{_R}")
        if s["deduped"]:        parts.append(f"{_DM}{s['deduped']} déjà connus{_R}")
        if s["rejected"]:       parts.append(f"{_DM}{s['rejected']} rejetés{_R}")
        total = s["auto_promoted"] + s["batch_promoted"]
        if total:
            print(f"  {_GR}KB enrichie de {total} règle(s) cette session.{_R}")
        if parts:
            print(f"  {_DM}Détail : {' · '.join(parts)}{_R}")
        print(f"{'═'*_W}\n")

    # ── Promotion → KB ───────────────────────────────────────────────────────

    def _promote_to_kb_throttled(self, block: Dict, language: str, code_before: str) -> bool:
        """
        Comme _promote_to_kb() mais attend entre les appels LLM pour respecter
        le quota Gemini free tier (5 req/min = 12s minimum entre appels).
        """
        import time
        elapsed = time.time() - self._last_llm_call
        if elapsed < self._llm_min_delay:
            wait = self._llm_min_delay - elapsed
            logger.debug("LLM throttle : attente %.1fs", wait)
            time.sleep(wait)
        result = self._promote_to_kb(block, language, code_before)
        self._last_llm_call = time.time()
        return result

    def _promote_to_kb(self, block: Dict, language: str, code_before: str) -> bool:
        """Généralise → écrit .md → recharge ChromaDB → update KG."""
        if self._rule_already_exists(block.get("problem", ""), language):
            self._stats["deduped"] += 1
            return False

        rule_md = self._generalise_to_rule(block, language, code_before)
        if not rule_md:
            logger.error("FeedbackProcessor : généralisation LLM échouée")
            return False

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        lang_dir = self._auto_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        rule_file = lang_dir / f"rule_{ts}.md"
        try:
            rule_file.write_text(rule_md, encoding="utf-8")
            logger.info("FeedbackProcessor : règle écrite → %s", rule_file.name)
        except Exception as e:
            logger.error("Écriture règle KB : %s", e)
            return False

        self._reload_chromadb(rule_file)
        self._update_kg(rule_file)
        return True

    def _reload_chromadb(self, md_path: Path) -> bool:
        if self._kb_loader is None or self._store is None:
            return False
        try:
            docs = self._kb_loader.process_file(md_path)
            if not docs:
                return False
            try:
                self._store._collection.delete(where={"source_file": md_path.name})
            except Exception:
                pass
            self._store.add_documents(docs)
            logger.info("ChromaDB reload : %d chunk(s) de %s", len(docs), md_path.name)
            return True
        except Exception as e:
            logger.error("ChromaDB reload erreur : %s", e)
            return False

    def _update_kg(self, md_path: Path) -> bool:
        if self._kg is None:
            return False
        try:
            if hasattr(self._kg, "reload_kb_file"):
                added = self._kg.reload_kb_file(md_path)
                if added:
                    n = self._kg._graph.number_of_nodes()
                    e = self._kg._graph.number_of_edges()
                    logger.debug("KG après reload : %d nœuds, %d arêtes", n, e)
                return added
            else:
                from config import config
                self._kg._builder.build_from_kb(config.KNOWLEDGE_BASE_DIR)
                self._kg._save()
                return True
        except Exception as e:
            logger.error("KG update erreur : %s", e)
            return False

    # ── LLM : généraliser le fix ──────────────────────────────────────────────

    def _generalise_to_rule(self, block: Dict, language: str, code_before: str) -> Optional[str]:
        problem = block.get('problem', '')

    # ── NOUVEAU : Recherche MCP ──────────────────────────────────────────
        web_context = self._fetch_documentation(problem, language)

    # Section doc web (optionnelle — vide si MCP indisponible)
        web_section = ''
        if web_context:
          web_section = f'''
    DOCUMENTATION OFFICIELLE (MCP Brave Search) :
          {web_context}
→ Intègre ces références dans la règle générée.
'''
        logger.info('MCP Web Search : %d chars de doc ajoutés au prompt', len(web_context))

        if self._llm is None:
            return None

        problem      = block.get("problem",      "")
        current_code = (block.get("current_code", "") or "")[:400]
        fixed_code   = (block.get("fixed_code",   "") or "")[:400]
        why          = block.get("why",            "")
        severity     = block.get("severity",       "MEDIUM")

        prompt = f"""Génère une règle KB réutilisable depuis ce fix

=== Correction à généraliser ===
{web_section}
Langage  : {language}
Sévérité : {severity}
Problème : {problem}
Pourquoi : {why}
Code problématique :
{current_code}
Code corrigé :
{fixed_code}

=== Format EXACT attendu (réponds UNIQUEMENT avec ce .md, sans balises ```) ===
---
title: [titre court, max 8 mots]
language: {language}
category: [security|quality|performance|patterns|architecture]
severity: {severity}
tags: [{language}-best-practice, <tag1>, <tag2>]
kg_nodes:
  - name: [NomConceptCamelCase]
    type: [vulnerability|fix|concept]
    severity: {severity}
    languages: [{language}]
    kb_queries:
      {language}: "[mots-clés recherche vectorielle]"
kg_relations:
  - [NomConcept, FIXED_BY, NomDuFix]
pattern_map:
  {language}:
    "[token_dangereux_exact]": NomConcept
---

## Problème
[2-3 phrases]

## Pourquoi c'est dangereux
[Impact concret]

## Code à éviter
```{language}
{current_code}
```

## Code correct
```{language}
{fixed_code}
```

## Pourquoi
[Explication technique]

## Références
[liens OWASP / CVE / doc officielle si trouvés]

"""
        try:
            response = self._llm.invoke(prompt)
            content  = response.content if hasattr(response, "content") else str(response)
            content  = content.strip()
            if content.startswith("```"):
                lines   = content.splitlines()
                content = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```"
                                    else lines[1:])
            return content.strip()
        except Exception as e:
            logger.error("LLM généralisation règle : %s", e)
            return None

    # ── Déduplication ─────────────────────────────────────────────────────────

    def _rule_already_exists(self, problem: str, language: str) -> bool:
        if not problem or self._store is None:
            return False
        try:
            results = self._store.similarity_search_with_score(
                f"{problem} {language}", k=1)
            return bool(results) and results[0][1] < DEDUP_THRESHOLD
        except Exception:
            return False
        
    def _fetch_documentation(self, problem: str, language: str) -> str:
        """
         Cherche doc officielle.
        """
        if not problem:
               return ''

        # Construire une requête ciblée
        query = f'{problem} {language} OWASP best practice fix 2024'
        logger.debug('WEB Search query : %s', query)

    # Appel synchrone (FeedbackProcessor tourne dans un thread non-async)
        result = web_search_client.search_sync(query, count=3)

    # Limiter pour ne pas exploser le prompt (800 chars max)
        return result[:800] if result else ''


    @staticmethod
    def _is_interactive() -> bool:
        return sys.stdin.isatty()