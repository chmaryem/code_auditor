"""
analysis_tools.py — LangChain @tool wrappers for LLM analysis operations.

Tools:
  - tool_build_context        : Build enriched context dict for LLM prompt
  - tool_build_system_impact  : Build [IMPACT SUR LE SYSTÈME] section
  - tool_validate_fix_blocks  : Validate proposed fix blocks against source code
  - tool_llm_analyze          : Invoke LLM with cascade fallback
"""
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool


@tool
def tool_build_context(
    file_path: str,
    neighborhood: dict,
    change_info: dict,
) -> Dict[str, Any]:
    """Build the enriched context dict passed to the LLM.

    Assembles: file path, language, dependencies, dependents, criticality,
    system impact section, project context, change info.

    Args:
        file_path: Absolute file path.
        neighborhood: Graph neighborhood dict.
        change_info: Change analysis result dict.

    Returns:
        Context dict for LLM prompt construction.
    """
    from agents.analysis_agent import build_context

    # project_indexer is injected via graph state, not here
    return build_context(
        file_path=Path(file_path),
        neighborhood=neighborhood,
        project_indexer=None,
        change_info=change_info,
    )


@tool
def tool_validate_fix_blocks(
    raw_analysis: str,
    source_code: str,
    language: str,
) -> List[Dict[str, Any]]:
    """Parse and validate each ---FIX START--- block proposed by the LLM.

    Each block is checked against the source code for correctness.

    Args:
        raw_analysis: Raw LLM response text.
        source_code: Original source code of the analyzed file.
        language: Programming language.

    Returns:
        List of validated block dicts with is_valid and validation_reason.
    """
    from agents.analysis_agent import AnalysisAgent
    agent = AnalysisAgent()
    return agent._validate_blocks(raw_analysis, source_code, language)


@tool
def tool_llm_analyze(
    code: str,
    context: dict,
    docs: list,
    scores: list,
) -> Dict[str, Any]:
    """Analyze code with LLM using RAG context and cascade fallback.

    Pipeline: prompt construction → LLM invoke (OpenRouter→Gemini) → parse response.

    Args:
        code: Source code to analyze.
        context: Enriched context dict.
        docs: Pre-computed RAG documents.
        scores: RAG similarity scores.

    Returns:
        Analysis result dict with 'analysis' text, 'relevant_knowledge',
        'validated_blocks'.
    """
    from services.llm_service import assistant_agent
    from langchain_core.documents import Document

    # ── Security gate (Phase A) ───────────────────────────────────────────────
    # Redact hardcoded secrets BEFORE the code leaves the machine for an external
    # LLM. The real source is anchored separately in the watch graph, so redacting
    # this copy never affects fix application — it only protects what we transmit.
    import logging
    from services.secret_redactor import redact_secrets

    safe_code, n_secrets = redact_secrets(code)
    if n_secrets:
        logging.getLogger(__name__).warning(
            "[SECURITY] redacted %d secret(s) from %s before sending to the LLM",
            n_secrets,
            (context or {}).get("file_path", "code") if isinstance(context, dict) else "code",
        )

    # Reconstruct Document objects from serialized dicts
    doc_objects = []
    for d in (docs or []):
        if isinstance(d, dict):
            doc_objects.append(Document(
                page_content=d.get("content", ""),
                metadata=d.get("metadata", {}),
            ))
        elif isinstance(d, Document):
            doc_objects.append(d)

    result = assistant_agent.analyze_code_with_rag(
        code=safe_code,
        context=context,
        precomputed_docs=doc_objects if doc_objects else None,
        precomputed_scores=scores if scores else None,
    )
    return result
