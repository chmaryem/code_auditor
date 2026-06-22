"""
services/history_cache_service.py

Redis cache layer for the History dashboard.

Strategy:
  - history:summary:{user_id}                   → stats cards  (TTL 300s)
  - history:timeline:{user_id}:{version}:{hash} → timeline page (TTL 120s)
  - history:version:{user_id}                   → monotonic counter (invalidation token)

Version-based invalidation: instead of scanning Redis keys with KEYS/SCAN
(slow, O(N)), we increment a per-user version counter on every write event.
The timeline cache keys include this version, so stale pages are never served
even though old keys linger until their TTL expires.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SUMMARY_TTL  = 300   # 5 min
_TIMELINE_TTL = 120   # 2 min
_VERSION_TTL  = 86400 # 24 h — version counter lives longer than timeline entries


def _redis():
    from services.mcp_redis_service import get_mcp_redis
    return get_mcp_redis()


def _summary_key(user_id: str) -> str:
    return f"history:summary:{user_id}"


def _version_key(user_id: str) -> str:
    return f"history:version:{user_id}"


def _timeline_key(user_id: str, version: int, filters_hash: str) -> str:
    return f"history:timeline:{user_id}:{version}:{filters_hash}"


def _filters_hash(params: Dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def get_current_version(user_id: str) -> int:
    try:
        r = _redis()
        val = r.get(_version_key(user_id))
        if val is None:
            return 0
        return int(val) if str(val).lstrip("-").isdigit() else 0
    except Exception:
        return 0


def get_summary(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = _redis()
        raw = r.get(_summary_key(user_id))
        if raw and isinstance(raw, str):
            return json.loads(raw)
    except Exception as e:
        logger.debug("history cache get_summary miss: %s", e)
    return None


def set_summary(user_id: str, data: Dict[str, Any]) -> None:
    try:
        r = _redis()
        r.set(_summary_key(user_id), json.dumps(data))
        r.expire(_summary_key(user_id), _SUMMARY_TTL)
    except Exception as e:
        logger.debug("history cache set_summary error: %s", e)


def get_timeline(user_id: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        version = get_current_version(user_id)
        fh = _filters_hash(params)
        r = _redis()
        raw = r.get(_timeline_key(user_id, version, fh))
        if raw and isinstance(raw, str):
            return json.loads(raw)
    except Exception as e:
        logger.debug("history cache get_timeline miss: %s", e)
    return None


def set_timeline(user_id: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
    try:
        version = get_current_version(user_id)
        fh = _filters_hash(params)
        key = _timeline_key(user_id, version, fh)
        r = _redis()
        r.set(key, json.dumps(data, default=str))
        r.expire(key, _TIMELINE_TTL)
    except Exception as e:
        logger.debug("history cache set_timeline error: %s", e)


def invalidate(user_id: str) -> None:
    """
    Bump the version counter so all existing timeline cache entries are
    logically stale on the next request. Also delete the summary cache
    immediately since it is cheap and frequently referenced.
    """
    try:
        r = _redis()
        # Bump version
        vk = _version_key(user_id)
        r.incr(vk)
        r.expire(vk, _VERSION_TTL)
        # Eagerly delete summary (small, single key)
        r.delete(_summary_key(user_id))
    except Exception as e:
        logger.debug("history cache invalidate error: %s", e)
