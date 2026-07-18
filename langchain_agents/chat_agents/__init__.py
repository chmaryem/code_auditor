"""
chat_agents — Agents spécialisés déterministes du ChatGraph (architecture blackboard).

Migration SMA — Phase 2. Chaque agent :
  - décide LOCALEMENT de sa pertinence (should_activate) selon le message + le scope
    (dashboard | extension) → autonomie décisionnelle, 0 token ;
  - contribue sa section au blackboard partagé (ChatState) via contribute() → 0 token.

La SEULE étape LLM du flux blackboard est la synthèse finale (node_synthesize).
Ce package n'est utilisé que lorsque CHAT_ORCHESTRATOR=blackboard (défaut : legacy).
"""
from langchain_agents.chat_agents.base import BlackboardAgent, get_registry

__all__ = ["BlackboardAgent", "get_registry"]
