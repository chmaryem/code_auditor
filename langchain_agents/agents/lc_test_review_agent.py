"""
lc_test_review_agent.py — LangChain TestReviewAgent (Critic).

4 Pillars:
  LLM     : OpenRouter/Groq/Gemini cascade via RunnableWithFallbacks (reuses
            _build_llm_with_fallback() from lc_analysis_agent.py — no raw HTTP)
  Tools   : N/A — one primitive per job (review(), diagnose_failure()), exposed
            as bare public methods, no @tool wrapper (Pattern B already
            established for this module; see plan decision 1.2)
  Memory  : AgentRedisMemory("test_review_agent") — write-only for now
            (remember_last_diagnosis / recall_last_diagnosis provided as a
            ready-to-use API, but the recall is NOT wired into any prompt yet —
            keeps the scope of this migration minimal per the approved plan)
  Planning: fixed role (Pattern B) — Critic pattern, verdicts are PURELY
            INFORMATIVE, NEVER change routing/validated/retry/cache (contrat
            non-négociable validé par l'utilisateur)

Role: pattern Generator + Critic (Reflexion, CodeT, AlphaCodium). Two jobs:
  (a) review()          — semantic critique of generated test code, called from
                           validation_node.py AFTER the mechanical check. Never
                           influences validated/_val_route.
  (b) diagnose_failure() — root-cause diagnosis of a runtime test failure,
                           called from execution_node.py. Its output only
                           enriches the TEXT sent to the runtime-retry LLM
                           prompt — never changes whether/how many times a
                           retry happens (still governed by run_result.success
                           + retry_runtime, exactly as before this agent
                           existed).

Zero-regression contract: both methods return None on any failure (LLM error,
malformed response) — fail-silent, never raises. A parsing failure is NEVER
translated into a negative verdict by default (that would fabricate a false
signal).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VERDICT_RE = re.compile(r"VERDICT:\s*(ok|concerns)", re.IGNORECASE)
_ISSUE_LINE_RE = re.compile(r"^\s*-\s*(.+)$", re.MULTILINE)


class TestReviewAgent:
    """
    LangChain TestReviewAgent — Critic, purely informative semantic review +
    runtime failure diagnosis.
    """

    def __init__(self):
        from langchain_agents.memory.redis_memory import AgentRedisMemory

        self.memory = AgentRedisMemory("test_review_agent")
        self._llm = None

    @property
    def llm(self):
        """Lazy-init LLM with fallback cascade (LLM pillar)."""
        if self._llm is None:
            from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
            self._llm = _build_llm_with_fallback()
        return self._llm

    # ── Job (a) : revue sémantique pré-exécution ────────────────────────────

    def review(
        self,
        generated_code: str,
        source_code: str,
        signatures: List[Dict[str, Any]],
        framework: str,
        attempt: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Juge sémantiquement le code de test généré (tautologie, assertions
        triviales, mock de la mauvaise dépendance, etc.). PUREMENT INFORMATIF —
        n'affecte jamais `validated` ni le routing du graphe.

        Retourne {"verdict": "ok"|"concerns", "issues": [...], "attempt": attempt}
        ou None si le LLM échoue / la réponse est malformée (fail-silent — un
        échec de parsing ne devient JAMAIS un verdict "concerns" par défaut).
        """
        try:
            prompt = self._build_review_prompt(generated_code, source_code, signatures, framework)
            raw = self.llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)
            return self._parse_review_response(text, attempt)
        except Exception as e:
            logger.debug("TestReviewAgent.review erreur (fail-silent) : %s", e)
            return None

    def _build_review_prompt(
        self,
        generated_code: str,
        source_code: str,
        signatures: List[Dict[str, Any]],
        framework: str,
    ) -> str:
        sig_names = ", ".join(s.get("name", "") for s in signatures[:10]) or "(none)"
        return f"""You are a senior code reviewer critiquing a generated unit test file
(framework: {framework}). Methods under test: {sig_names}.

Source code being tested:
```
{source_code[:2000]}
```

Generated test code:
```
{generated_code[:3000]}
```

Look for REAL quality problems only: tautological assertions (e.g. assert True,
assertTrue(true)), assertions that can never fail, mocking the wrong
dependency, missing assertions, or tests that don't actually exercise the
methods listed above. Do NOT flag style preferences or minor nitpicks.

Respond with EXACTLY this format, nothing else:
VERDICT: ok|concerns
ISSUES:
- issue 1 (omit this section if no issues)"""

    def _parse_review_response(self, text: str, attempt: int) -> Optional[Dict[str, Any]]:
        match = _VERDICT_RE.search(text or "")
        if not match:
            return None
        verdict = match.group(1).lower()
        issues = [m.strip() for m in _ISSUE_LINE_RE.findall(text or "")]
        return {"verdict": verdict, "issues": issues, "attempt": attempt}

    # ── Job (b) : diagnostic post-exécution ─────────────────────────────────

    def diagnose_failure(
        self,
        generated_code: str,
        run_result: Any,
        source_code: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Diagnostique la cause racine d'un échec d'exécution runtime, mieux que
        le résumé regex de TestRunner. PUREMENT ADDITIF — n'affecte JAMAIS si
        un retry a lieu ni combien de fois (déjà décidé par run_result.success
        + retry_runtime, inchangé).

        Retourne {"root_cause": "...", "confidence": "low"|"medium"|"high"}
        ou None si le LLM échoue / réponse malformée (fail-silent).
        """
        try:
            prompt = self._build_diagnosis_prompt(generated_code, run_result, source_code)
            raw = self.llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)
            return self._parse_diagnosis_response(text)
        except Exception as e:
            logger.debug("TestReviewAgent.diagnose_failure erreur (fail-silent) : %s", e)
            return None

    def _build_diagnosis_prompt(self, generated_code: str, run_result: Any, source_code: str) -> str:
        error_summary = getattr(run_result, "error_summary", "") or ""
        return f"""A generated unit test failed at execution. Diagnose the ROOT CAUSE in 1-3 sentences.

Source code being tested:
```
{source_code[:1500]}
```

Generated test code:
```
{generated_code[:2000]}
```

Execution error:
```
{error_summary[:2000]}
```

Respond with EXACTLY this format, nothing else:
ROOT_CAUSE: <1-3 sentence diagnosis>
CONFIDENCE: low|medium|high"""

    def _parse_diagnosis_response(self, text: str) -> Optional[Dict[str, Any]]:
        root_cause_match = re.search(r"ROOT_CAUSE:\s*(.+?)(?:\nCONFIDENCE:|$)", text or "", re.DOTALL | re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*(low|medium|high)", text or "", re.IGNORECASE)
        if not root_cause_match:
            return None
        root_cause = root_cause_match.group(1).strip()
        if not root_cause:
            return None
        confidence = confidence_match.group(1).lower() if confidence_match else "medium"
        return {"root_cause": root_cause, "confidence": confidence}

    # ── Memory : diagnostic write-only (API prête, pas encore rappelée) ─────

    def remember_last_diagnosis(self, source_path: str, diagnosis: str) -> None:
        """Mémorise le dernier diagnostic runtime pour ce fichier (Memory pillar)."""
        self.memory.set(f"last_diagnosis:{source_path}", diagnosis, expire_seconds=3600)

    def recall_last_diagnosis(self, source_path: str) -> Optional[str]:
        """Récupère le dernier diagnostic connu pour ce fichier, ou None."""
        return self.memory.get(f"last_diagnosis:{source_path}")


test_review_agent = TestReviewAgent()
