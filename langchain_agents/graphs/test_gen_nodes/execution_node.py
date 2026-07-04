"""
execution_node.py — ExecutionAgent (+ finalize).

Wraps step 12 of generate_for_file() and the result/cache assembly:

  node_execution — only reached when write=True. Writes the test to disk,
      runs it (pytest / mvn|gradle / jest via TestRunner), and on failure routes
      back to GenerationAgent for one runtime retry (MAX_RETRY_RUNTIME = 1). A
      passing RETRY run flips validated → True, mirroring the original.

  node_finalize — builds the exact result dict returned by generate_for_file()
      and writes the Redis cache (Memory out) only when validated AND the tests
      passed (or were never run, i.e. write=False → run_result is None).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_state import MAX_RETRY_RUNTIME

logger = logging.getLogger(__name__)


def node_execution(state: Dict[str, Any]) -> Dict[str, Any]:
    target_path = state["target_path"]
    language = state["language"]
    generated_code = state["generated_code"]
    runner = state["_runner"]

    # Write the generated test to disk (step 12).
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(generated_code, encoding="utf-8")
    except Exception as e:
        return {
            "final_output": {"error": f"Écriture impossible : {e}", "test_file": None},
            "_exec_route": "end",
        }

    # Exécution des tests générés — détecte les erreurs runtime (ImportError, etc.)
    run_result = runner.run(target_path, language)
    updates: Dict[str, Any] = {"run_result": run_result}

    if run_result.success:
        # A passing RETRY run marks the code validated (original line 230).
        if state.get("retry_runtime", 0) > 0:
            updates["validated"] = True
            logger.info("Retry runtime réussi : tests passent maintenant")
        updates["_exec_route"] = "finalize"
        return updates

    logger.info(
        "Exécution échouée pour %s : %s",
        target_path.name, run_result.error_summary[:200],
    )

    if state.get("retry_runtime", 0) < MAX_RETRY_RUNTIME:
        # Retry LLM avec l'erreur d'exécution comme contexte
        updates["run_error"] = run_result
        updates["_exec_route"] = "generate"
    else:
        logger.info("Retry runtime épuisé : %s", run_result.error_summary[:150])
        updates["_exec_route"] = "finalize"
    return updates


def node_finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    run_result = state.get("run_result")
    rag_context = state.get("rag_context", {}) or {}

    result = {
        "test_file":          state["target_path"],
        "test_code":          state["generated_code"],
        "framework":          state["framework"],
        "error":              None,
        "rag_docs_used":      rag_context.get("total_docs", 0),
        "validated":          state.get("validated", False),
        "incremental":        state.get("is_incremental", False),
        "run_success":        run_result.success if run_result else None,
        "run_error_summary":  run_result.error_summary if run_result and not run_result.success else None,
    }

    # Mettre en cache uniquement si le test est valide ET passe à l'exécution
    cache_ok = result["validated"] and (run_result is None or run_result.success)
    if cache_ok:
        from services.test_generation_cache import test_generation_cache
        test_generation_cache.set(state["source_path"], state.get("incremental", False), result)

    return {"final_output": result}
