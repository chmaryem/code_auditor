"""
CacheService — SQLite thread-safe.


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
                -- mémoire épisodique cross-session
            CREATE TABLE IF NOT EXISTS episode_memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path        TEXT NOT NULL,
                pattern_type     TEXT NOT NULL,
                severity         TEXT NOT NULL DEFAULT 'MEDIUM',
                occurrence_count INTEGER DEFAULT 1,
                first_seen       TEXT DEFAULT (datetime('now')),
                last_seen        TEXT DEFAULT (datetime('now')),
                last_session     TEXT,
                promoted_to_kb   INTEGER DEFAULT 0,
                UNIQUE(file_path, pattern_type)
            );
            
            --  stats par session
            CREATE TABLE IF NOT EXISTS session_stats (
                session_id    TEXT PRIMARY KEY,
                started_at    TEXT,
                ended_at      TEXT,
                files_analyzed INTEGER DEFAULT 0,
                kb_rules_added INTEGER DEFAULT 0,
                patterns_found TEXT  -- JSON list
            );
            CREATE TABLE IF NOT EXISTS git_memory (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash      TEXT,
                branch           TEXT,
                author           TEXT,
                file_path        TEXT,
                issues_critical  INTEGER DEFAULT 0,
                issues_high      INTEGER DEFAULT 0,
                issues_medium    INTEGER DEFAULT 0,
                blocked          INTEGER DEFAULT 0,   -- 1 si commit bloque
                analyzed_at      TEXT
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

    #  save()/load() deviennent des no-ops — SQLite est déjà durable

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
        with self._lock:
         self._conn.execute("""
            INSERT INTO episode_memory
                (file_path, pattern_type, severity, last_session)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path, pattern_type) DO UPDATE SET
                occurrence_count = occurrence_count + 1,
                last_seen        = datetime('now'),
                last_session     = excluded.last_session,
                severity         = excluded.severity
        """, (file_path, pattern_type, severity, session_id))
        self._conn.commit()


    def get_recurring_patterns(
      self,
      file_path:   str,
      min_count:   int = 2
    ) -> list:
      """
    Retourne les patterns récurrents d'un fichier.
    Utile pour afficher : "Ce bug existe depuis 3 jours."
    """
      with self._lock:
        rows = self._conn.execute("""
            SELECT pattern_type, severity,
                   occurrence_count, first_seen, last_seen,
                   promoted_to_kb
            FROM episode_memory
            WHERE file_path = ?
              AND occurrence_count >= ?
            ORDER BY occurrence_count DESC
        """, (file_path, min_count)).fetchall()
      return [
        {
            "pattern":    r[0],
            "severity":   r[1],
            "count":      r[2],
            "first_seen": r[3],
            "last_seen":  r[4],
            "in_kb":      bool(r[5])
        }
        for r in rows
    ]
      
    def get_hotspot_files(self, top_n: int = 5) -> list:
     """
     Fichiers avec le plus de patterns récurrents non corrigés.
     Utile pour le rapport de session.
     """
     with self._lock:
        rows = self._conn.execute("""
            SELECT file_path,
                   COUNT(DISTINCT pattern_type) as pattern_count,
                   SUM(occurrence_count)         as total_occurrences,
                   MAX(CASE WHEN severity='CRITICAL' THEN 1 ELSE 0 END) as has_critical
            FROM episode_memory
            WHERE promoted_to_kb = 0
            GROUP BY file_path
            ORDER BY has_critical DESC, total_occurrences DESC
            LIMIT ?
        """, (top_n,)).fetchall()
     return [
        {
            "file":        r[0],
            "patterns":    r[1],
            "occurrences": r[2],
            "critical":    bool(r[3])
        }
        for r in rows
    ]
     
    def mark_pattern_promoted(
    self,
    file_path:    str,
    pattern_type: str 
    ):
     """Marque un pattern comme promu dans la KB."""
     with self._lock:
        self._conn.execute("""
            UPDATE episode_memory
            SET promoted_to_kb = 1
            WHERE file_path = ? AND pattern_type = ?
        """, (file_path, pattern_type))
        self._conn.commit()
        
    def save_commit_analysis(self, commit_hash: str, branch: str,
                          author: str, file_path: str,
                          critical: int, high: int, medium: int,
                          blocked: bool):
     """Persiste le resultat d'analyse d'un fichier du commit."""
     now = datetime.now().isoformat()
     with self._lock:
        self._conn.execute(
            """INSERT INTO git_memory
               (commit_hash, branch, author, file_path,
                issues_critical, issues_high, issues_medium,
                blocked, analyzed_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (commit_hash, branch, author, file_path,
             critical, high, medium, int(blocked), now)
        )
        self._conn.commit()
 
    def get_file_history(self, file_path: str, limit: int = 10) -> list:
      """Retourne l'historique des analyses d'un fichier."""
      with self._lock:
         rows = self._conn.execute(
            """SELECT branch, issues_critical, issues_high,
                      issues_medium, blocked, analyzed_at
               FROM git_memory WHERE file_path = ?
               ORDER BY analyzed_at DESC LIMIT ?""",
            (file_path, limit)
        ).fetchall()
      return rows
 
    def is_recurring_issue(self, file_path: str) -> bool:
      """True si ce fichier a eu des CRITICAL dans les 2 dernieres semaines."""
      with self._lock:
        count = self._conn.execute(
            """SELECT COUNT(*) FROM git_memory
               WHERE file_path = ? AND issues_critical > 0
               AND analyzed_at > datetime('now', '-14 days')""",
            (file_path,)
        ).fetchone()[0]
      return count >= 2



#CacheManager = CacheService

# Instance globale
cache_service = CacheService()