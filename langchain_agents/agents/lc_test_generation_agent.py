
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LCTestGenerationAgent:
    """
    LangChain TestGenerationAgent — génération de tests avec LLM réel + memory.
    """

    def __init__(self):
        from langchain_agents.memory.redis_memory import AgentRedisMemory

        self.memory = AgentRedisMemory("test_generation_agent")
        self._llm = None

    @property
    def llm(self):
        """Lazy-init LLM avec cascade de fallback (pillar LLM)."""
        if self._llm is None:
            from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
            self._llm = _build_llm_with_fallback()
        return self._llm

    # ── Génération ───────────────────────────────────────────────────────────

    def generate(self, prompt: str, file_name: str = "") -> Optional[str]:
        """
        Appelle le LLM (cascade Groq → OpenRouter → Gemini) et extrait le code
        de test de la réponse.

        Même contrat que l'ancien _helpers.call_llm() : retourne None sur échec
        (silencieux, logué), sinon le code extrait.
        """
        from langchain_agents.graphs.test_gen_nodes._helpers import extract_code_from_response

        try:
            raw = self.llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)
            if not text:
                return None
            return extract_code_from_response(text)
        except Exception as e:
            logger.error("LLM test generation erreur: %s", e)
            return None

    # ── Memory : raison du dernier retry par fichier ────────────────────────

    def remember_retry_reason(self, source_path: str, reason: str) -> None:
        """Mémorise la raison du dernier retry (structurel ou runtime) pour ce fichier."""
        self.memory.set(f"last_retry_reason:{source_path}", reason, expire_seconds=3600)

    def recall_retry_reason(self, source_path: str) -> Optional[str]:
        """Récupère la raison du dernier retry connu pour ce fichier, ou None."""
        return self.memory.get(f"last_retry_reason:{source_path}")


lc_test_generation_agent = LCTestGenerationAgent()
