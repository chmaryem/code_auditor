"""
code_tools.py — LangChain @tool wrappers for CodeAgent operations.

Tools (deterministic, no LLM):
  - tool_parse_file      : AST parsing → entities, imports, language
  - tool_analyze_change  : Diff analysis → significant/minor decision
  - tool_detect_language : File extension → language string
"""
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool


@tool
def tool_parse_file(file_path: str) -> Dict[str, Any]:
    """Parse a source file and extract AST entities, imports, and language.

    Args:
        file_path: Absolute path to the source file.

    Returns:
        Dict with keys: file_path, language, entities, imports, code, line_count, error.
    """
    from agents.code_agent import code_agent
    return code_agent.parse(Path(file_path))


@tool
def tool_analyze_change(old_content: str, new_content: str) -> Dict[str, Any]:
    """Compare old and new file content to determine if LLM analysis is needed.

    Args:
        old_content: Previous file content (empty string for new files).
        new_content: Current file content.

    Returns:
        Dict with keys: significant (bool), score (0-100), lines_changed,
        change_type, reason.
    """
    from agents.code_agent import code_agent
    return code_agent.analyze_change(old_content, new_content)


@tool
def tool_detect_language(file_path: str) -> str:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the file.

    Returns:
        Language string: python, java, javascript, typescript, or unknown.
    """
    from agents.code_agent import code_agent
    return code_agent.detect_language(Path(file_path))
