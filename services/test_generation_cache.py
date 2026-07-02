
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KEY_PREFIX = "ca:tg:"
DEFAULT_TTL = 86400  # 24 heures


def _content_hash(source_path: Path) -> str:
    """SHA-256 du contenu du fichier source, tronqué à 16 hex chars."""
    try:
        data = source_path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(source_path).encode()).hexdigest()[:16]


def _cache_key(content_hash: str, incremental: bool) -> str:
    mode = "inc" if incremental else "full"
    return f"{KEY_PREFIX}{content_hash}:{mode}"


class TestGenerationCache:
    """
    Cache Redis dédié à la génération de tests.
    Thread-safe, résilient aux pannes Redis (fail-silently).
    """

    def __init__(self, ttl: int = DEFAULT_TTL):
        self._ttl = ttl
        self._redis = None
        self._disabled = False

    @property
    def redis(self):
        if self._disabled:
            return None
        if self._redis is None:
            try:
                from services.mcp_redis_service import get_mcp_redis
                self._redis = get_mcp_redis()
            except Exception as e:
                logger.warning("TestGenerationCache: Redis indisponible (%s) — cache désactivé", e)
                self._disabled = True
        return self._redis

    # ── Lecture ──────────────────────────────────────────────────────────────

    def get(self, source_path: Path, incremental: bool) -> Optional[Dict[str, Any]]:
        """
        Retourne le résultat mis en cache ou None si absent/expiré.

        Returns:
            dict avec les clés : test_code, framework, validated,
                                 rag_docs_used, incremental, cached=True
            None si cache miss ou Redis indisponible.
        """
        if self.redis is None:
            return None

        ch = _content_hash(source_path)
        key = _cache_key(ch, incremental)

        try:
            data = self.redis.hgetall(key)
        except Exception as e:
            logger.debug("Cache get erreur: %s", e)
            return None

        if not data or "test_code" not in data:
            return None

        # Vérifier le TTL manuel
        expires_at = float(data.get("expires_at", "0"))
        if expires_at and time.time() > expires_at:
            logger.debug("Cache expiré pour %s", source_path.name)
            self._delete(key)
            return None

        logger.info("Cache hit test generation : %s (mode=%s)", source_path.name,
                    "incremental" if incremental else "complet")
        return {
            "test_code":     data.get("test_code", ""),
            "framework":     data.get("framework", ""),
            "validated":     data.get("validated", "false").lower() == "true",
            "rag_docs_used": int(data.get("rag_docs_used", "0")),
            "incremental":   data.get("incremental", "false").lower() == "true",
            "test_file":     Path(data["test_file"]) if data.get("test_file") else None,
            "error":         None,
            "cached":        True,
        }

    # ── Écriture ─────────────────────────────────────────────────────────────

    def set(self, source_path: Path, incremental: bool, result: Dict[str, Any]) -> bool:
        """
        Met en cache un résultat de génération réussi.
        Ne met PAS en cache les résultats avec error != None.

        Returns:
            True si mise en cache réussie, False sinon.
        """
        if self.redis is None:
            return False

        if result.get("error"):
            return False

        test_code = result.get("test_code", "")
        if not test_code:
            return False

        ch = _content_hash(source_path)
        key = _cache_key(ch, incremental)

        test_file = result.get("test_file")

        mapping = {
            "test_code":     test_code,
            "framework":     str(result.get("framework", "")),
            "validated":     str(result.get("validated", False)).lower(),
            "rag_docs_used": str(result.get("rag_docs_used", 0)),
            "incremental":   str(result.get("incremental", False)).lower(),
            "test_file":     str(test_file) if test_file else "",
            "source_name":   source_path.name,
            "generated_at":  str(time.time()),
            "expires_at":    str(time.time() + self._ttl),
        }

        try:
            self.redis.hset_dict(key, mapping)
            logger.info(
                "Cache set test generation : %s (mode=%s, ttl=%dh)",
                source_path.name,
                "incremental" if incremental else "complet",
                self._ttl // 3600,
            )
            return True
        except Exception as e:
            logger.debug("Cache set erreur: %s", e)
            return False

    # ── Invalidation ─────────────────────────────────────────────────────────

    def invalidate(self, source_path: Path) -> None:
        """Invalide toutes les entrées cache pour ce fichier source (full + incremental)."""
        if self.redis is None:
            return

        ch = _content_hash(source_path)
        for mode in ("full", "inc"):
            key = f"{KEY_PREFIX}{ch}:{mode}"
            self._delete(key)
        logger.debug("Cache invalidé pour %s", source_path.name)

    def _delete(self, key: str) -> None:
        try:
            self.redis.delete(key)
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le cache de génération."""
        if self.redis is None:
            return {"status": "disabled", "entries": 0}

        try:
            keys = self.redis.scan_keys(f"{KEY_PREFIX}*")
            now = time.time()
            active = 0
            expired = 0
            for k in keys:
                try:
                    exp = self.redis.hget(k, "expires_at")
                    if exp and float(exp) > now:
                        active += 1
                    else:
                        expired += 1
                except Exception:
                    pass
            return {"status": "active", "entries": active, "expired": expired}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton utilisé par TestGeneratorAgent
test_generation_cache = TestGenerationCache()
