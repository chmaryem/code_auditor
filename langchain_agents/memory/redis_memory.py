"""
redis_memory.py — Redis-backed memory for LangChain agents.

Provides persistent memory using the existing MCPRedisService:
  - AgentRedisMemory  : Key-value memory for any agent (old content, counters, etc.)
  - PatternMemory     : Pattern frequency tracker for LearningAgent
  - AnalysisCacheMemory: Analysis result cache for AnalysisAgent

All operations are synchronous (MCPRedisService handles async internally).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_redis():
    """Lazy import of Redis singleton."""
    from services.mcp_redis_service import get_mcp_redis
    return get_mcp_redis()


class AgentRedisMemory:
    """
    Generic key-value memory backed by Redis hashes.

    Each agent gets its own Redis hash namespace:
      ca:mem:{agent_name}:{key} → value

    Pillar: MEMORY (stores old_content for CodeAgent, solution flags for AnalysisAgent)
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._prefix = f"ca:mem:{agent_name}"

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> Optional[str]:
        """Read a value from Redis."""
        try:
            redis = _get_redis()
            return redis.get(self._key(key))
        except Exception as e:
            logger.debug("[%s] Redis get failed: %s", self.agent_name, e)
            return None

    def set(self, key: str, value: str, expire_seconds: Optional[int] = None) -> None:
        """Write a value to Redis."""
        try:
            redis = _get_redis()
            redis.set(self._key(key), value, expire_seconds=expire_seconds)
        except Exception as e:
            logger.debug("[%s] Redis set failed: %s", self.agent_name, e)

    def get_json(self, key: str) -> Optional[Any]:
        """Read a JSON-serialized value."""
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set_json(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> None:
        """Write a JSON-serialized value."""
        self.set(key, json.dumps(value, ensure_ascii=False, default=str), expire_seconds)

    def delete(self, key: str) -> None:
        """Delete a key."""
        try:
            redis = _get_redis()
            redis.delete(self._key(key))
        except Exception as e:
            logger.debug("[%s] Redis delete failed: %s", self.agent_name, e)


class PatternMemory:
    """
    Redis-backed pattern frequency tracker for LearningAgent.

    Uses Redis Sorted Sets:
      ca:patterns:{language} → ZADD pattern_name score(=count)

    Pillar: MEMORY (episodic — tracks how many times a pattern was seen)
    """

    KEY_PREFIX = "ca:patterns"

    def record_pattern(self, language: str, pattern_name: str) -> int:
        """Record a pattern occurrence. Returns the new count."""
        try:
            redis = _get_redis()
            key = f"{self.KEY_PREFIX}:{language}"
            # Get current score, increment
            current_items = redis.zrange(key, 0, -1, with_scores=True)
            current_score = 0
            for member, score in current_items:
                if member == pattern_name:
                    current_score = int(score)
                    break
            new_score = current_score + 1
            redis.zadd(key, float(new_score), pattern_name)
            return new_score
        except Exception as e:
            logger.debug("PatternMemory.record failed: %s", e)
            return 1

    def get_frequency(self, language: str, pattern_name: str) -> int:
        """Get how many times a pattern was seen."""
        try:
            redis = _get_redis()
            key = f"{self.KEY_PREFIX}:{language}"
            items = redis.zrange(key, 0, -1, with_scores=True)
            for member, score in items:
                if member == pattern_name:
                    return int(score)
            return 0
        except Exception:
            return 0

    def get_top_patterns(self, language: str, n: int = 10) -> List[Dict[str, Any]]:
        """Get top-N most frequent patterns for a language."""
        try:
            redis = _get_redis()
            key = f"{self.KEY_PREFIX}:{language}"
            items = redis.zrevrange(key, 0, n - 1, with_scores=True)
            return [{"pattern": m, "count": int(s)} for m, s in items]
        except Exception:
            return []


class AnalysisCacheMemory:
    """
    Redis-backed analysis result cache.

    Uses Redis Hashes:
      ca:analysis:{hash(file_path)} → {result, timestamp, hash}

    Pillar: MEMORY (caches LLM results to avoid re-analysis)
    """

    KEY_PREFIX = "ca:analysis"

    @staticmethod
    def _file_key(file_path: str) -> str:
        from services.mcp_redis_service import key_hash
        return f"{AnalysisCacheMemory.KEY_PREFIX}:{key_hash(file_path)}"

    def get_cached(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis for a file."""
        try:
            redis = _get_redis()
            data = redis.hgetall(self._file_key(file_path))
            if not data or "result" not in data:
                return None
            result = data.get("result", "")
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return {"analysis": result}
        except Exception:
            return None

    def set_cached(
        self,
        file_path: str,
        result: Dict[str, Any],
        content_hash: str,
        expire_seconds: int = 86400,
    ) -> None:
        """Cache an analysis result."""
        try:
            redis = _get_redis()
            redis.hset_dict(self._file_key(file_path), {
                "result": json.dumps(result, ensure_ascii=False, default=str),
                "hash": content_hash,
                "timestamp": str(int(time.time())),
                "file_path": file_path,
            }, expire_seconds=expire_seconds)
        except Exception as e:
            logger.debug("AnalysisCacheMemory.set failed: %s", e)

    def is_stale(self, file_path: str, current_hash: str) -> bool:
        """Check if cached result is outdated based on content hash."""
        try:
            redis = _get_redis()
            cached_hash = redis.hget(self._file_key(file_path), "hash")
            return cached_hash != current_hash
        except Exception:
            return True
