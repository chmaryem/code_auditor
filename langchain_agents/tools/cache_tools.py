"""
cache_tools.py — LangChain @tool wrappers for Cache operations.

Memory backend: Redis via MCPRedisService (replaces SQLite).

Tools:
  - tool_cache_has_changed  : Check if file hash changed since last analysis
  - tool_cache_read         : Read cached analysis result
  - tool_cache_write        : Write analysis result to cache
  - tool_cache_remove       : Remove a file from cache
"""
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool
def tool_cache_has_changed(file_path: str) -> bool:
    """Check if a file's content has changed since the last cached analysis.

    Uses SHA-256 hash comparison. Returns True if the file needs re-analysis.

    Args:
        file_path: Absolute path to the file.

    Returns:
        True if the file has changed or has no cached hash.
    """
    from services.cache_service import CacheService
    cache = CacheService()
    return cache.has_file_changed(Path(file_path))


@tool
def tool_cache_read(file_path: str) -> Optional[Dict[str, Any]]:
    """Read cached analysis result for a file.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Cached analysis dict or None if not found.
    """
    from services.cache_service import CacheService
    cache = CacheService()
    return cache.get_cached_analysis(Path(file_path))


@tool
def tool_cache_write(
    file_path: str,
    analysis: dict,
    dependencies: list,
    dependents: list,
) -> bool:
    """Write analysis result to cache.

    Args:
        file_path: Absolute path to the file.
        analysis: Analysis result dict.
        dependencies: List of dependency file paths.
        dependents: List of dependent file paths.

    Returns:
        True if cache write succeeded.
    """
    from services.cache_service import CacheService
    cache = CacheService()
    try:
        cache.update_file_cache(Path(file_path), analysis, dependencies, dependents)
        return True
    except Exception:
        return False


@tool
def tool_cache_remove(file_path: str) -> bool:
    """Remove a file from the analysis cache.

    Args:
        file_path: Absolute path to the file.

    Returns:
        True if removal succeeded.
    """
    from services.cache_service import CacheService
    cache = CacheService()
    try:
        cache.remove_file_from_cache(Path(file_path))
        return True
    except Exception:
        return False
