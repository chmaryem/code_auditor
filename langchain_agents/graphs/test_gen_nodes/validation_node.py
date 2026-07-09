"""
validation_node.py — ValidationAgent.

Wraps step 11 (_run_validation's validation half). Validates the generated code
per language:
  - Python : compile()
  - Java   : structural (balanced braces/parens, no private-method calls, no reflection)
  - JS/TS  : structural (balanced delimiters + at least one test function)

Fidelity to the original _run_validation:
  - On first failure, route back to GenerationAgent for exactly ONE structural
    retry (MAX_RETRY_STRUCTURAL = 1).
  - If the retry still fails, the ORIGINAL first-pass code is restored (the
    original returned `test_code`, not the failed v2) and the pipeline PROCEEDS
    anyway (validation only gated caching, never stopped write/execution).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_state import MAX_RETRY_STRUCTURAL
from langchain_agents.graphs.test_gen_nodes._helpers import validate_generated_test

logger = logging.getLogger(__name__)


def node_validation(state: Dict[str, Any]) -> Dict[str, Any]:
    generated_code = state["generated_code"]
    source_path = state["source_path"]
    signatures = state["signatures"]   # ALL signatures (private-call detection)

    validated = validate_generated_test(generated_code, source_path, signatures=signatures)

    # TestReviewAgent job (a) — revue sémantique, PUREMENT INFORMATIVE.
    # N'affecte JAMAIS `validated` ni `_val_route` (contrat non-négociable).
    # Fail-silent à 2 niveaux : review() ne lève jamais, et ce try/except
    # protège en plus contre toute exception imprévue (ex. import cassé).
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
        return {
            "validated": True, "validation_error": None, "_val_route": "proceed",
            "review_notes": review_notes,
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
