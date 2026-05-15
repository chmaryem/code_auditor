"""
test_gap_tools.py — LangChain @tool wrappers for Test Gap Detection.

Tools:
  - tool_test_gap_check : Detect if a source file is missing tests.
  - tool_is_test_file   : Check if a path is itself a test file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool
def tool_test_gap_check(
    file_path: str,
    code: str,
    language: str,
    parsed_entities: list,
    change_info: Optional[dict] = None,
    project_path: str = ".",
) -> Optional[Dict[str, Any]]:
    """Check whether a source file has missing tests and compute impact score.

    Skips test files (e.g. *Test.java, test_*.py, *.spec.ts).
    Uses TestDiscoveryService + TestGapAgent (0 token, no LLM).

    Args:
        file_path: Absolute path to the source file.
        code: Source code content (used for impact scoring).
        language: Programming language.
        parsed_entities: List of public entities extracted by the parser.
        change_info: Optional change analysis dict from the CodeAgent.
        project_path: Project root directory (for test discovery).

    Returns:
        Dict with test-gap metadata if attention is needed, otherwise None.
    """
    from langchain_agents.agents.lc_test_gap_agent import LCTestGapAgent

    agent = LCTestGapAgent(Path(project_path))
    return agent.check(
        source_file=file_path,
        parsed_entities=parsed_entities,
        change_info=change_info,
        language=language,
    )


@tool
def tool_is_test_file(file_path: str, language: str = "") -> bool:
    """Return True if the given file path is itself a test file.

    Patterns:
      Java     : *Test.java, Test*.java
      Python   : test_*.py, *_test.py
      JS/TS    : *.test.js, *.spec.js, *.test.ts, *.spec.ts

    Args:
        file_path: Absolute file path.
        language: Optional language hint for stricter matching.

    Returns:
        bool — True if this is a test file (not a source file).
    """
    from langchain_agents.agents.lc_test_gap_agent import _is_test_file
    return _is_test_file(Path(file_path), language)
