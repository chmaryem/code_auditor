
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BlackboardAgent:
    """Classe de base. Par défaut : toujours actif, contribution vide."""

    name: str = "agent"
    #: Vague d'exécution. 0 = séquentiel avant les autres (produit le contexte
    #: fichier) ; 1 = parallèle. Voir node_gather.
    wave: int = 1

    def should_activate(self, state: Dict[str, Any]) -> bool:  # pragma: no cover - trivial
        return True

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        return {}


_REGISTRY: List[BlackboardAgent] | None = None


def get_registry() -> List[BlackboardAgent]:
    """Instances singletons des agents (stateless). Construites paresseusement."""
    global _REGISTRY
    if _REGISTRY is None:
        from langchain_agents.chat_agents.agents import (
            FileAgent, RagAgent, GitAgent, CiAgent, ProjectStateAgent, SummaryAgent,
        )
        _REGISTRY = [
            FileAgent(),          # wave 0 — produit file_code / dependencies
            RagAgent(),           # wave 1
            GitAgent(),           # wave 1
            CiAgent(),            # wave 1
            ProjectStateAgent(),  # wave 1
            SummaryAgent(),       # wave 1
        ]
    return _REGISTRY
