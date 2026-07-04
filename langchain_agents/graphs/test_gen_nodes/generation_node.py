"""
generation_node.py — GenerationAgent.

Wraps steps 8–10 of generate_for_file() plus the two retry helpers. The node is
re-entered by the graph's conditional edges and operates in three modes:

  INITIAL         (no error flags set):
      redact secrets → build prompt (memorised in state) → call LLM →
      incremental merge. No code → error final_output.

  STRUCTURAL retry (validation_error set):
      _retry_with_error(previous_code) → keep v2 only if it will validate,
      otherwise the ORIGINAL first-pass code is restored by ValidationAgent.
      (Fidelity: exactly ONE structural retry, original kept on failure.)

  RUNTIME retry   (run_error set):
      _retry_with_runtime_error(previous_code, run_result). If a fix is
      produced → back to ExecutionAgent (no re-validation, as in the original).
      If no fix → finalize with the original failing run_result.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_nodes._helpers import (
    build_prompt,
    call_llm,
    merge_incremental,
    retry_with_error,
    retry_with_runtime_error,
)

logger = logging.getLogger(__name__)


def node_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]

    # ── RUNTIME retry mode ─────────────────────────────────────────────────────
    if state.get("run_error") is not None:
        return _generate_runtime_retry(state)

    # ── STRUCTURAL retry mode ──────────────────────────────────────────────────
    if state.get("validation_error"):
        return _generate_structural_retry(state)

    # ── INITIAL generation ─────────────────────────────────────────────────────
    return _generate_initial(state)


def _generate_initial(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]
    source_code = state["source_code"]
    framework = state["framework"]
    language = state["language"]
    is_incremental = state.get("is_incremental", False)
    existing_test_code = state.get("existing_test_code", "")
    untested_signatures = state["untested_signatures"]
    imports = state["imports"]
    rag_context = state["rag_context"]
    kg_context = state["kg_context"]

    # Security gate (Phase A): redact before source_code enters the LLM prompt.
    # Signatures/imports extraction (AnalysisAgent) already used the real source_code.
    safe_source_code = source_code
    try:
        from services.secret_redactor import redact_secrets
        safe_source_code, _n = redact_secrets(source_code)
    except Exception:
        pass

    # 8. Prompt LLM enrichi
    prompt = build_prompt(
        source_path=source_path,
        source_code=safe_source_code,
        signatures=untested_signatures,
        imports=imports,
        framework=framework,
        rag_context=rag_context,
        kg_context=kg_context,
        language=language,
        incremental=is_incremental,
        existing_test_code=existing_test_code,
    )

    # 9. Génération LLM directe
    test_code = call_llm(prompt, source_path.name)
    if not test_code:
        return {
            "prompt": prompt,
            "safe_source_code": safe_source_code,
            "_gen_route": "end",
            "final_output": {"error": "Le LLM n'a pas généré de code de test.", "test_file": None},
        }

    # 10. En mode incrémentiel : fusionner avec le fichier existant
    if is_incremental:
        test_code = merge_incremental(existing_test_code, test_code, language)

    return {
        "prompt": prompt,
        "safe_source_code": safe_source_code,
        "generated_code": test_code,
        "_gen_route": "validate",
    }


def _generate_structural_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]
    prompt = state["prompt"]
    previous_code = state["generated_code"]

    logger.info("Test non valide, retry avec feedback d'erreur...")
    test_code_v2 = retry_with_error(previous_code, source_path, prompt)

    updates: Dict[str, Any] = {
        "_pre_retry_code":  previous_code,   # kept for restore if v2 also fails
        "retry_structural": state.get("retry_structural", 0) + 1,
        "validation_error": None,            # let ValidationAgent re-evaluate
        "_gen_route":       "validate",
    }
    if test_code_v2:
        updates["generated_code"] = test_code_v2
    # else: keep the previous (original) generated_code unchanged
    return updates


def _generate_runtime_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]
    prompt = state["prompt"]
    previous_code = state["generated_code"]
    run_result = state["run_error"]

    fixed_code = retry_with_runtime_error(previous_code, source_path, prompt, run_result)

    updates: Dict[str, Any] = {
        "retry_runtime": state.get("retry_runtime", 0) + 1,
        "run_error":     None,
    }
    if fixed_code:
        updates["generated_code"] = fixed_code
        updates["_runtime_fix"] = True
        updates["_gen_route"] = "execute"
    else:
        # No fix produced → the original failing run_result stands (the original
        # code did NOT re-run in this case).
        updates["_runtime_fix"] = False
        updates["_gen_route"] = "finalize"
    return updates
