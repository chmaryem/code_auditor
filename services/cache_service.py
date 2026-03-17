"""
CacheService — SQLite thread-safe.
Remplace l'ancien cache_manager.py (pickle).

Pourquoi ce changement ?
  L'ancien .pkl pouvait se corrompre si deux analyses écrivaient
  en même temps (mode watch). SQLite gère ça avec des transactions ACID.

Compatibilité totale : CacheManager = CacheService (alias).
"""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config import config


class CacheService:

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        db_path = self.cache_dir / "analysis_cache.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()
        print(f" Cache SQLite : {db_path}")

    # ── Schéma ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_path          TEXT PRIMARY KEY,
                    content_hash       TEXT NOT NULL,
                    last_modified      REAL,
                    analysis_text      TEXT,
                    relevant_knowledge TEXT,
                    dependencies       TEXT,
                    dependents         TEXT,
                    updated_at         TEXT
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            self._conn.commit()

    #  Hash

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

    # Vérification changement

    def has_file_changed(self, file_path: Path) -> bool:
        file_key = str(file_path)
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM file_cache WHERE file_path = ?",
                (file_key,),
            ).fetchone()
        if row is None:
            return True
        return self.compute_file_hash(file_path) != row[0]

    # Lecture 

    def get_cached_analysis(self, file_path: Path) -> Optional[Dict[str, Any]]:
        file_key = str(file_path)
        with self._lock:
            row = self._conn.execute(
                "SELECT analysis_text, relevant_knowledge FROM file_cache WHERE file_path = ?",
                (file_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "analysis":           row[0] or "",
            "relevant_knowledge": json.loads(row[1] or "[]"),
        }

    def get_file_dependencies(self, file_path: Path) -> Dict[str, list]:
        file_key = str(file_path)
        with self._lock:
            row = self._conn.execute(
                "SELECT dependencies, dependents FROM file_cache WHERE file_path = ?",
                (file_key,),
            ).fetchone()
        if row is None:
            return {"dependencies": [], "dependents": []}
        return {
            "dependencies": json.loads(row[0] or "[]"),
            "dependents":   json.loads(row[1] or "[]"),
        }

    # Écriture

    def update_file_cache(
        self,
        file_path:    Path,
        analysis:     Dict[str, Any],
        dependencies: list = None,
        dependents:   list = None,
    ):
        file_key     = str(file_path)
        current_hash = self.compute_file_hash(file_path)
        mtime        = file_path.stat().st_mtime if file_path.exists() else None
        now          = datetime.now().isoformat()

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO file_cache
                    (file_path, content_hash, last_modified,
                     analysis_text, relevant_knowledge,
                     dependencies, dependents, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_key,
                    current_hash,
                    mtime,
                    analysis.get("analysis", ""),
                    json.dumps(analysis.get("relevant_knowledge", [])),
                    json.dumps(dependencies or []),
                    json.dumps(dependents   or []),
                    now,
                ),
            )
            self._conn.commit()

    def remove_file_from_cache(self, file_path: Path):
        file_key = str(file_path)
        with self._lock:
            self._conn.execute(
                "DELETE FROM file_cache WHERE file_path = ?", (file_key,)
            )
            self._conn.commit()
        print(f" Retiré du cache : {file_path.name}")

    def update_dependencies(self, file_path: Path, dependencies: list, dependents: list):
        file_key = str(file_path)
        with self._lock:
            self._conn.execute(
                "UPDATE file_cache SET dependencies=?, dependents=? WHERE file_path=?",
                (json.dumps(dependencies), json.dumps(dependents), file_key),
            )
            self._conn.commit()

    # ── save()/load() deviennent des no-ops — SQLite est déjà durable ─────────

    def save(self):
        """Compatibilité avec l'ancien code — SQLite n'a pas besoin de flush manuel."""
        pass

    def load(self):
        """Compatibilité — la connexion SQLite est ouverte dès __init__."""
        pass

    # Stats

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM file_cache").fetchone()[0]
        db_path = self.cache_dir / "analysis_cache.db"
        return {
            "total_files":      total,
            "cache_file":       str(db_path),
            "cache_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f" Cache : {stats['total_files']} fichiers — {stats['cache_size_bytes']/1024:.2f} KB")

    def clear(self):
        with self._lock:
            self._conn.execute("DELETE FROM file_cache")
            self._conn.commit()
        print(" Cache effacé")


# Alias pour compatibilité totale avec l'ancien nom
CacheManager = CacheService

# Instance globale
cache_service = CacheService()