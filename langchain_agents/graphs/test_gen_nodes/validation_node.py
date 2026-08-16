
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_state import MAX_RETRY_STRUCTURAL, MAX_RETRY_REFLEXION
from langchain_agents.graphs.test_gen_nodes._helpers import validate_generated_test

logger = logging.getLogger(__name__)


def node_validation(state: Dict[str, Any]) -> Dict[str, Any]:
    generated_code = state["generated_code"]
    source_path = state["source_path"]
    signatures = state["signatures"]   # ALL signatures (private-call detection)

    validated = validate_generated_test(generated_code, source_path, signatures=signatures)

   
    review_notes: Dict[str, Any] = dict(state.get("review_notes") or {})
    try:
        from langchain_agents.agents.lc_test_review_agent import test_review_agent
        attempt = state.get("retry_structural", 0) + 1
        semantic = test_review_agent.review(
            generated_code, state["source_code"], signatures,
            state["framework"], attempt=attempt,
        )
        review_notes["semantic_review"] = semantic   # None si l'agent a échoué
    except Exception as e:
        logger.debug("TestReviewAgent.review erreur (fail-silent) : %s", e)
        review_notes["semantic_review"] = None
    review_notes.setdefault("runtime_diagnosis", None)

    if validated:
        # ── Reflexion (Generator ↔ Critic) ─────────────────────────────────────
        # Le code passe la validation MÉCANIQUE, mais le Critic le juge faible
        # (verdict "concerns"). Il déclenche UNE régénération réflexive, bornée.
        # C'est ici que le verdict du Critic influence enfin le Generator — la
        # coordination inter-agents qui fait de ce module un vrai MAS.
        semantic = review_notes.get("semantic_review")
        concerns = bool(
            semantic
            and semantic.get("verdict") == "concerns"
            and semantic.get("issues")
        )
        if concerns and state.get("retry_reflexion", 0) < MAX_RETRY_REFLEXION:
            return {
                "validated": True,               # la vérité mécanique ne change pas
                "validation_error": None,
                "_val_route": "reflexion",
                "review_notes": review_notes,
                "_reflexion_issues": list(semantic["issues"]),
                "retry_reflexion": state.get("retry_reflexion", 0) + 1,
            }
        return {
            "validated": True, "validation_error": None, "_val_route": "proceed",
            "review_notes": review_notes,
        }

    # ── Invalide ────────────────────────────────────────────────────────────────
    # Sécurité Reflexion : si une v2 réflexive est invalide, restaurer la version
    # VALIDE d'avant la reflexion (la reflexion ne dégrade jamais un test valide).
    pre_reflexion = state.get("_pre_reflexion_code")
    if pre_reflexion is not None:
        return {
            "validated": True,
            "validation_error": None,
            "_val_route": "proceed",
            "review_notes": review_notes,
            "generated_code": pre_reflexion,
            "_pre_reflexion_code": None,
        }

    # Invalid — one structural retry allowed
    if state.get("retry_structural", 0) < MAX_RETRY_STRUCTURAL:
        return {
            "validated": False,
            "validation_error": "structural validation failed",
            "_val_route": "generate",
            "review_notes": review_notes,
        }

    # Retry exhausted → keep the ORIGINAL first-pass code, proceed with validated=False
    updates: Dict[str, Any] = {
        "validated": False,
        "validation_error": None,
        "_val_route": "proceed",
        "review_notes": review_notes,
    }
    pre = state.get("_pre_retry_code")
    if pre is not None:
        updates["generated_code"] = pre
    return updates
