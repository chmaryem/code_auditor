"""
CacheService — Redis MCP thread-safe.

Migration SQLite → Redis via MCP (redis/mcp-redis officiel).

Architecture :
  Toutes les opérations passent par MCPRedisService qui communique
  avec redis-mcp-server via le protocole MCP (stdio).

  L'interface publique est 100% identique à l'ancienne version SQLite.
  Aucun consommateur n'a besoin de changer.

Schéma Redis :
  ca:fc:{path_hash}                    → Hash (file_cache)
  ca:fch:{content_hash}               → String (content_hash → path_hash)
  ca:em:{path_hash}:{pattern_hash}    → Hash (episode_memory)
  ca:em:hotspots                       → Sorted Set (score = total_occurrences)
  ca:gm:{id}                           → Hash (git_memory entry)
  ca:gm:file:{path_hash}              → Sorted Set (score = timestamp)
  ca:gm:next_id                        → String (auto-increment counter)
  ca:ss:{session_id}                   → Hash (session_stats)
  ca:meta:{key}                        → String (metadata)
"""
import hashlib
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config
from services.mcp_redis_service import get_mcp_redis, key_hash, KEY_PREFIX


class CacheService:

    def __init__(self, cache_dir: Path = None):
        # cache_dir conservé pour compatibilité mais non utilisé par Redis
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._redis = None  # Lazy init
        print(f" Cache Redis MCP : {config.redis.url}")

    @property
    def redis(self):
        """Lazy init du client MCP Redis."""
        if self._redis is None:
            self._redis = get_mcp_redis()
        return self._redis

    # ── Key helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fc_key(file_path: str) -> str:
        """Clé Redis pour le cache d'un fichier."""
        return f"{KEY_PREFIX}fc:{key_hash(file_path)}"

    @staticmethod
    def _fch_key(content_hash: str) -> str:
        """Clé Redis pour l'index content_hash → path_hash."""
        return f"{KEY_PREFIX}fch:{content_hash[:16]}"

    @staticmethod
    def _em_key(file_path: str, pattern_type: str) -> str:
        """Clé Redis pour un pattern épisodique."""
        return f"{KEY_PREFIX}em:{key_hash(file_path)}:{key_hash(pattern_type)}"

    @staticmethod
    def _em_pattern(file_path: str) -> str:
        """Pattern SCAN pour tous les patterns d'un fichier."""
        return f"{KEY_PREFIX}em:{key_hash(file_path)}:*"

    @staticmethod
    def _gm_key(entry_id: int) -> str:
        """Clé Redis pour une entrée git_memory."""
        return f"{KEY_PREFIX}gm:{entry_id}"

    @staticmethod
    def _gm_file_key(file_path: str) -> str:
        """Clé Redis pour le sorted set git_memory par fichier."""
        return f"{KEY_PREFIX}gm:file:{key_hash(file_path)}"

    # ── Hash ──────────────────────────────────────────────────────────────────

    def compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f" Erreur hash {file_path}: {e}")
            return ""

    # ── Vérification changement ──────────────────────────────────────────────

    def has_file_changed(self, file_path: Path) -> bool:
        file_key = str(file_path)
        redis_key = self._fc_key(file_key)
        try:
            cached_hash = self.redis.hget(redis_key, "content_hash")
        except Exception:
            return True
        if cached_hash is None:
            return True
        return self.compute_file_hash(file_path) != cached_hash

    # ── Lecture ───────────────────────────────────────────────────────────────

    def get_cached_analysis(self, file_path: Path) -> Optional[Dict[str, Any]]:
        file_key = str(file_path)
        redis_key = self._fc_key(file_key)
        try:
            data = self.redis.hgetall(redis_key)
        except Exception:
            return None
        if not data or "analysis_text" not in data:
            return None
        knowledge = data.get("relevant_knowledge", "[]")
        try:
            knowledge_parsed = json.loads(knowledge)
        except (json.JSONDecodeError, TypeError):
            knowledge_parsed = []
        return {
            "analysis":           data.get("analysis_text", ""),
            "relevant_knowledge": knowledge_parsed,
        }

    def get_file_dependencies(self, file_path: Path) -> Dict[str, list]:
        file_key = str(file_path)
        redis_key = self._fc_key(file_key)
        try:
            data = self.redis.hgetall(redis_key)
        except Exception:
            return {"dependencies": [], "dependents": []}
        if not data:
            return {"dependencies": [], "dependents": []}
        try:
            deps = json.loads(data.get("dependencies", "[]"))
        except (json.JSONDecodeError, TypeError):
            deps = []
        try:
            depts = json.loads(data.get("dependents", "[]"))
        except (json.JSONDecodeError, TypeError):
            depts = []
        return {"dependencies": deps, "dependents": depts}

    # ── Écriture ─────────────────────────────────────────────────────────────

    def update_file_cache(
        self,
        file_path:    Path,
        analysis:     Dict[str, Any],
        dependencies: list = None,
        dependents:   list = None,
    ):
        file_key     = str(file_path)
        current_hash = self.compute_file_hash(file_path)
        mtime        = file_path.stat().st_mtime if file_path.exists() else 0.0
        now          = datetime.now().isoformat()
        redis_key    = self._fc_key(file_key)

        mapping = {
            "file_path":          file_key,
            "content_hash":       current_hash,
            "last_modified":      str(mtime),
            "analysis_text":      analysis.get("analysis", ""),
            "relevant_knowledge": json.dumps(analysis.get("relevant_knowledge", [])),
            "dependencies":       json.dumps(dependencies or []),
            "dependents":         json.dumps(dependents or []),
            "updated_at":         now,
        }

        with self._lock:
            self.redis.hset_dict(redis_key, mapping)
            # Index secondaire : content_hash → path_hash
            if current_hash:
                self.redis.set(self._fch_key(current_hash), key_hash(file_key))

    def remove_file_from_cache(self, file_path: Path):
        file_key = str(file_path)
        redis_key = self._fc_key(file_key)
        # Supprimer l'index content_hash d'abord
        try:
            old_hash = self.redis.hget(redis_key, "content_hash")
            if old_hash:
                self.redis.delete(self._fch_key(old_hash))
        except Exception:
            pass
        self.redis.delete(redis_key)
        print(f" Retiré du cache : {file_path.name}")

    def update_dependencies(self, file_path: Path, dependencies: list, dependents: list):
        file_key = str(file_path)
        redis_key = self._fc_key(file_key)
        with self._lock:
            self.redis.hset(redis_key, "dependencies", json.dumps(dependencies))
            self.redis.hset(redis_key, "dependents", json.dumps(dependents))

    # ── save()/load() — no-ops pour compatibilité ────────────────────────────

    def save(self):
        """Compatibilité — Redis persiste automatiquement (appendonly yes)."""
        pass

    def load(self):
        """Compatibilité — Redis est toujours disponible."""
        pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        try:
            keys = self.redis.scan_keys(f"{KEY_PREFIX}fc:*")
        except Exception:
            keys = []
        return {
            "total_files":      len(keys),
            "cache_file":       config.redis.url,
            "cache_size_bytes": 0,  # Redis n'a pas de taille fichier
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f" Cache Redis : {stats['total_files']} fichiers")

    def clear(self):
        try:
            keys = self.redis.scan_keys(f"{KEY_PREFIX}fc:*")
            for k in keys:
                self.redis.delete(k)
            # Nettoyer les index content_hash aussi
            fch_keys = self.redis.scan_keys(f"{KEY_PREFIX}fch:*")
            for k in fch_keys:
                self.redis.delete(k)
        except Exception:
            pass
        print(" Cache effacé")

    # ── Episode Memory ───────────────────────────────────────────────────────

    def record_pattern(
        self,
        file_path:    str,
        pattern_type: str,
        severity:     str,
        session_id:   str = None
    ):
        """
        Enregistre une occurrence de pattern détecté.
        Si le pattern existe déjà → incrémente le compteur.
        """
        em_key = self._em_key(file_path, pattern_type)
        now = datetime.now().isoformat()

        with self._lock:
            existing = self.redis.hgetall(em_key)

            if existing and "pattern_type" in existing:
                # Pattern existe → incrémenter
                count = int(existing.get("occurrence_count", "1")) + 1
                self.redis.hset(em_key, "occurrence_count", str(count))
                self.redis.hset(em_key, "last_seen", now)
                self.redis.hset(em_key, "severity", severity)
                if session_id:
                    self.redis.hset(em_key, "last_session", session_id)
            else:
                # Nouveau pattern
                mapping = {
                    "file_path":        file_path,
                    "pattern_type":     pattern_type,
                    "severity":         severity,
                    "occurrence_count": "1",
                    "first_seen":       now,
                    "last_seen":        now,
                    "last_session":     session_id or "",
                    "promoted_to_kb":   "0",
                }
                self.redis.hset_dict(em_key, mapping)

            # Mettre à jour le sorted set hotspots
            total = self._get_file_total_occurrences(file_path)
            self.redis.zadd(f"{KEY_PREFIX}em:hotspots", float(total), key_hash(file_path))

    def _get_file_total_occurrences(self, file_path: str) -> int:
        """Calcule le total d'occurrences pour un fichier donné."""
        pattern = self._em_pattern(file_path)
        try:
            keys = self.redis.scan_keys(pattern)
        except Exception:
            return 0
        total = 0
        for k in keys:
            try:
                data = self.redis.hgetall(k)
                total += int(data.get("occurrence_count", "0"))
            except Exception:
                pass
        return total

    def get_recurring_patterns(
        self,
        file_path:   str,
        min_count:   int = 2
    ) -> list:
        """
        Retourne les patterns récurrents d'un fichier.
        Utile pour afficher : "Ce bug existe depuis 3 jours."
        """
        pattern = self._em_pattern(file_path)
        try:
            keys = self.redis.scan_keys(pattern)
        except Exception:
            return []

        results = []
        for k in keys:
            try:
                data = self.redis.hgetall(k)
                count = int(data.get("occurrence_count", "0"))
                if count >= min_count:
                    results.append({
                        "pattern":    data.get("pattern_type", ""),
                        "severity":   data.get("severity", "MEDIUM"),
                        "count":      count,
                        "first_seen": data.get("first_seen", ""),
                        "last_seen":  data.get("last_seen", ""),
                        "in_kb":      data.get("promoted_to_kb", "0") == "1",
                    })
            except Exception:
                continue

        # Trier par count décroissant (comme ORDER BY occurrence_count DESC)
        results.sort(key=lambda x: x["count"], reverse=True)
        return results

    def get_hotspot_files(self, top_n: int = 5) -> list:
        """
        Fichiers avec le plus de patterns récurrents non corrigés.
        Utilise le sorted set em:hotspots pour le tri.
        """
        hotspot_key = f"{KEY_PREFIX}em:hotspots"
        try:
            top_members = self.redis.zrevrange(hotspot_key, 0, top_n - 1, with_scores=True)
        except Exception:
            return []

        results = []
        for member_data in top_members:
            if isinstance(member_data, (list, tuple)) and len(member_data) >= 2:
                file_hash = str(member_data[0])
                total_occ = int(float(member_data[1]))
            else:
                continue

            # Récupérer les patterns de ce fichier
            file_pattern = f"{KEY_PREFIX}em:{file_hash}:*"
            try:
                pattern_keys = self.redis.scan_keys(file_pattern)
            except Exception:
                continue

            file_path = ""
            pattern_count = 0
            has_critical = False

            for pk in pattern_keys:
                try:
                    data = self.redis.hgetall(pk)
                    if data.get("promoted_to_kb", "0") == "0":
                        pattern_count += 1
                        if not file_path:
                            file_path = data.get("file_path", "")
                        if data.get("severity") == "CRITICAL":
                            has_critical = True
                except Exception:
                    continue

            if file_path and pattern_count > 0:
                results.append({
                    "file":        file_path,
                    "patterns":    pattern_count,
                    "occurrences": total_occ,
                    "critical":    has_critical,
                })

        return results

    def mark_pattern_promoted(
        self,
        file_path:    str,
        pattern_type: str
    ):
        """Marque un pattern comme promu dans la KB."""
        em_key = self._em_key(file_path, pattern_type)
        self.redis.hset(em_key, "promoted_to_kb", "1")

    # ── Git Memory ───────────────────────────────────────────────────────────

    def save_commit_analysis(self, commit_hash: str, branch: str,
                             author: str, file_path: str,
                             critical: int, high: int, medium: int,
                             blocked: bool):
        """Persiste le résultat d'analyse d'un fichier du commit."""
        now = datetime.now().isoformat()
        now_ts = time.time()

        # Auto-increment ID
        entry_id = self.redis.incr(f"{KEY_PREFIX}gm:next_id")
        gm_key = self._gm_key(entry_id)

        mapping = {
            "commit_hash":     commit_hash,
            "branch":          branch,
            "author":          author,
            "file_path":       file_path,
            "issues_critical": str(critical),
            "issues_high":     str(high),
            "issues_medium":   str(medium),
            "blocked":         str(int(blocked)),
            "analyzed_at":     now,
        }

        with self._lock:
            self.redis.hset_dict(gm_key, mapping)
            # Index par fichier (sorted set trié par timestamp)
            self.redis.zadd(self._gm_file_key(file_path), now_ts, str(entry_id))

    def get_file_history(self, file_path: str, limit: int = 10) -> list:
        """Retourne l'historique des analyses d'un fichier."""
        zset_key = self._gm_file_key(file_path)
        try:
            # Récupérer les IDs les plus récents (score = timestamp)
            all_ids = self.redis.zrevrange(zset_key, 0, limit - 1)
        except Exception:
            return []

        rows = []
        for entry_id in all_ids:
            eid = str(entry_id)
            if isinstance(entry_id, (list, tuple)):
                eid = str(entry_id[0])
            try:
                data = self.redis.hgetall(self._gm_key(int(eid)))
                if data:
                    rows.append((
                        data.get("branch", ""),
                        int(data.get("issues_critical", "0")),
                        int(data.get("issues_high", "0")),
                        int(data.get("issues_medium", "0")),
                        int(data.get("blocked", "0")),
                        data.get("analyzed_at", ""),
                    ))
            except Exception:
                continue
        return rows

    def is_recurring_issue(self, file_path: str) -> bool:
        """True si ce fichier a eu des CRITICAL dans les 2 dernières semaines."""
        zset_key = self._gm_file_key(file_path)
        cutoff = time.time() - (14 * 86400)  # 14 jours

        try:
            # Récupérer les entrées des 14 derniers jours
            all_ids = self.redis.zrange(zset_key, 0, -1, with_scores=True)
        except Exception:
            return False

        critical_count = 0
        for item in all_ids:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entry_id, score = str(item[0]), float(item[1])
            else:
                continue
            if score < cutoff:
                continue
            try:
                data = self.redis.hgetall(self._gm_key(int(entry_id)))
                if int(data.get("issues_critical", "0")) > 0:
                    critical_count += 1
            except Exception:
                continue

        return critical_count >= 2


# Instance globale
cache_service = CacheService()