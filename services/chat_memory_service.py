"""
chat_memory_service.py — Redis conversation memory for ChatAgent.

Phase 1:
  - store per-session turns in Redis
  - keep only the last N turns
  - store lightweight metadata
  - graceful fallback if Redis/MCP is not available

Phase 2 — Session history (sidebar):
  - every session is indexed per project so a developer can list and
    reopen past conversations, not just the one still held in memory
  - the index outlives the raw turn history (turns expire faster than
    the lightweight "which sessions exist" pointer)

Redis keys:
  ca:chat:{session_id}:history     -> JSON list of turns (TTL: HISTORY_TTL_SECONDS)
  ca:chat:{session_id}:meta        -> JSON metadata incl. title (TTL: META_TTL_SECONDS)
  ca:chat:idx:{project_hash}       -> sorted set, member=session_id, score=updated_at
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _project_hash(project_path: str) -> str:
    """Stable short id for a project root, used to scope the session index."""
    resolved = str(Path(project_path or ".").resolve()).lower()
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]


def _make_title(message: str, limit: int = 60) -> str:
    """Derive a short, human-readable session title from the first user message."""
    title = " ".join((message or "").split())
    if len(title) > limit:
        title = title[: limit - 1].rstrip() + "…"
    return title or "New conversation"


class ChatMemoryService:
    """Small Redis-backed chat history service."""

    HISTORY_TTL_SECONDS = 24 * 3600
    META_TTL_SECONDS = 30 * 24 * 3600
    MAX_TURNS = 12
    MAX_SESSIONS_PER_PROJECT = 200

    def __init__(self, max_turns: int = MAX_TURNS, ttl_seconds: int = HISTORY_TTL_SECONDS):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._fallback: Dict[str, List[Dict[str, Any]]] = {}
        self._fallback_meta: Dict[str, Dict[str, Any]] = {}
        self._fallback_index: Dict[str, Dict[str, float]] = {}

    # ── Keys ────────────────────────────────────────────────────────────────

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def history_key(session_id: str) -> str:
        return f"ca:chat:{session_id}:history"

    @staticmethod
    def meta_key(session_id: str) -> str:
        return f"ca:chat:{session_id}:meta"

    @staticmethod
    def index_key(project_path: str) -> str:
        return f"ca:chat:idx:{_project_hash(project_path)}"

    def _redis(self):
        try:
            from services.mcp_redis_service import get_mcp_redis
            return get_mcp_redis()
        except Exception as e:
            logger.debug("ChatMemoryService Redis unavailable: %s", e)
            return None

    # ── Public API ──────────────────────────────────────────────────────────

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Return recent conversation turns for a session.

        Reads both backends and keeps whichever has more turns: the MCP Redis
        client fails silently (returns None) when Redis is unreachable, so an
        empty Redis read does not necessarily mean the session has no history.
        """
        if not session_id:
            return []

        fallback_history = self._fallback.get(session_id, [])

        redis = self._redis()
        key = self.history_key(session_id)
        redis_history: List[Dict[str, Any]] = []

        if redis:
            try:
                raw = redis.get(key)
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        redis_history = data
            except Exception as e:
                logger.debug("load_history Redis error: %s", e)

        best = redis_history if len(redis_history) >= len(fallback_history) else fallback_history
        return best[-self.max_turns:]

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one message turn to the session history."""
        if not session_id:
            return

        turn = {
            "role": role,
            "content": content,
            "ts": int(time.time()),
            "metadata": metadata or {},
        }

        history = self.load_history(session_id)
        history.append(turn)
        history = history[-self.max_turns:]

        # Write-through: always keep the in-memory fallback current. The MCP
        # Redis client returns None on failure instead of raising (circuit
        # breaker), so we can't rely on "no exception" to mean "persisted".
        self._fallback[session_id] = history

        redis = self._redis()
        key = self.history_key(session_id)

        if redis:
            try:
                redis.set(
                    key,
                    json.dumps(history, ensure_ascii=False, default=str),
                    expire_seconds=self.ttl_seconds,
                )
            except Exception as e:
                logger.debug("append_turn Redis error: %s", e)

    def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        project_path: str = "",
    ) -> None:
        """Save user + assistant messages as one exchange and index the session."""
        self.append_turn(session_id, "user", user_message, metadata)
        self.append_turn(session_id, "assistant", assistant_response, metadata)

        existing_meta = self.get_meta(session_id)
        title = existing_meta.get("title") or _make_title(user_message)
        self.set_meta(
            session_id,
            {
                "title": title,
                "project_path": project_path or existing_meta.get("project_path", ""),
                "intent": (metadata or {}).get("intent", existing_meta.get("intent", "")),
            },
        )

        if project_path or existing_meta.get("project_path"):
            self.touch_session(session_id, project_path or existing_meta.get("project_path", ""))

    def set_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        """Store lightweight session metadata."""
        if not session_id:
            return
        payload = {**meta, "updated_at": int(time.time())}

        # Write-through (see append_turn for why we can't trust "no exception").
        self._fallback_meta[session_id] = payload

        redis = self._redis()
        if redis:
            try:
                redis.set(
                    self.meta_key(session_id),
                    json.dumps(payload, ensure_ascii=False, default=str),
                    expire_seconds=self.META_TTL_SECONDS,
                )
            except Exception as e:
                logger.debug("set_meta Redis error: %s", e)

    def get_meta(self, session_id: str) -> Dict[str, Any]:
        """Read session metadata, preferring whichever backend has the freshest copy."""
        if not session_id:
            return {}

        fallback = self._fallback_meta.get(session_id, {})

        redis = self._redis()
        redis_meta: Dict[str, Any] = {}
        if redis:
            try:
                raw = redis.get(self.meta_key(session_id))
                if raw:
                    redis_meta = json.loads(raw)
            except Exception:
                pass

        if redis_meta.get("updated_at", 0) >= fallback.get("updated_at", 0):
            return redis_meta or fallback
        return fallback

    # ── Session index (sidebar) ────────────────────────────────────────────

    def touch_session(self, session_id: str, project_path: str) -> None:
        """Record/refresh a session in its project's sorted-set index."""
        if not session_id or not project_path:
            return
        now = time.time()
        idx_key = self.index_key(project_path)

        # Write-through fallback index.
        bucket = self._fallback_index.setdefault(idx_key, {})
        bucket[session_id] = now

        redis = self._redis()
        if redis:
            try:
                redis.zadd(idx_key, now, session_id)
            except Exception as e:
                logger.debug("touch_session Redis error: %s", e)

    def list_sessions(self, project_path: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List sessions for a project, most recently updated first."""
        idx_key = self.index_key(project_path)
        redis = self._redis()
        redis_members: List[str] = []

        if redis:
            try:
                pairs = redis.zrevrange(idx_key, 0, limit - 1, with_scores=True)
                redis_members = [m for m, _ in pairs] if pairs else []
            except Exception as e:
                logger.debug("list_sessions Redis error: %s", e)

        fallback_bucket = self._fallback_index.get(idx_key, {})
        fallback_members = sorted(fallback_bucket, key=lambda sid: fallback_bucket[sid], reverse=True)

        # Merge both sources (most recent first) without duplicates — either
        # backend may be missing entries the other has written successfully.
        seen: set = set()
        members: List[str] = []
        for sid in [*redis_members, *fallback_members]:
            if sid not in seen:
                seen.add(sid)
                members.append(sid)
        members = members[:limit]

        sessions = []
        for sid in members:
            meta = self.get_meta(sid)
            if not meta:
                continue
            history = self.load_history(sid)
            sessions.append({
                "session_id": sid,
                "title": meta.get("title", "New conversation"),
                "updated_at": meta.get("updated_at", 0),
                "turn_count": len(history),
            })
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def clear_session(self, session_id: str, project_path: str = "") -> None:
        """Delete a session's history + metadata and drop it from the index."""
        if not session_id:
            return

        meta = self.get_meta(session_id)
        project_path = project_path or meta.get("project_path", "")

        redis = self._redis()
        if redis:
            try:
                redis.delete(self.history_key(session_id))
                redis.delete(self.meta_key(session_id))
                if project_path:
                    redis.zrem(self.index_key(project_path), session_id)
            except Exception as e:
                logger.debug("clear_session Redis error: %s", e)

        self._fallback.pop(session_id, None)
        self._fallback_meta.pop(session_id, None)
        if project_path:
            self._fallback_index.get(self.index_key(project_path), {}).pop(session_id, None)


chat_memory_service = ChatMemoryService()
