"""
git_session_tracker.py — Surveillance proactive de la session de développement.

Rôle dans l'architecture Smart Git :
  Le FileWatcher (mode watch) analyse chaque fichier sauvegardé et stocke
  les résultats dans SQLite (analysis_cache.db).

  Le GitSessionTracker ne refait PAS ces analyses — il LIT le cache SQLite
  et croise ces données avec 'git diff HEAD' pour répondre à :
    "Le développeur accumule-t-il des bugs non corrigés dans sa session ?"

  Il tourne en thread daemon (arrêt automatique à la fermeture du programme)
  et se réveille toutes les CHECK_INTERVAL secondes pour recalculer le score.

Calcul du score de risque :
  score = Σ(bugs pondérés) × facteur_temps

  Poids par sévérité :
    CRITICAL → 10 pts
    HIGH     →  3 pts
    MEDIUM   →  1 pt
    LOW      →  0 pt (informatif seulement)

  Facteur temps (multiplicateur) :
    < 30 min   → ×1.0  (normal)
    30–60 min  → ×1.2  (attention)
    60–120 min → ×1.5  (risque)
    > 120 min  → ×2.0  (critique — commits trop espacés)

  Seuils de niveau :
    CLEAN    : score == 0        → tout va bien
    WATCH    : 0  < score < 15  → information légère
    WARN     : 15 ≤ score < 35  → rapport intermédiaire recommandé
    CRITICAL : score ≥ 35       → correction urgente avant commit

Hysteresis :
  Le tracker notifie uniquement quand le niveau MONTE (pas descend).
  Évite le spam si le dev corrige un bug mais que le score reste en WARN.
  La descente est notifiée seulement quand on repasse à CLEAN (commit fait).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


CHECK_INTERVAL = 180       
SEVERITY_WEIGHTS = {
    "CRITICAL": 10,
    "HIGH":      3,
    "MEDIUM":    1,
    "LOW":       0,
}
TIME_MULTIPLIERS = [       
    (120, 2.0),
    (60,  1.5),
    (30,  1.2),
    (0,   1.0),
]
LEVEL_THRESHOLDS = {      
    "CRITICAL": 35,
    "WARN":     15,
    "WATCH":     1,
    "CLEAN":     0,
}
WATCHED_EXTENSIONS = {".java", ".py", ".ts", ".js", ".tsx", ".jsx"}

# FIX 3 — How often (seconds) to re-fire notification if level doesn't drop
REMINDER_INTERVALS: Dict[str, int] = {
    "WATCH":    600,   # remind every 10 min if still at WATCH
    "WARN":     180,   # remind every 3 min  if still at WARN
    "CRITICAL":  60,   # remind every 1 min  if still at CRITICAL
}



# Structures de données


@dataclass
class FileRisk:
    """
    Représente le risque associé à un fichier dans la session courante.
    Produit par _assess_file_risk() en croisant git diff + cache SQLite.
    """
    path:          str
    status:        str          
    staged:        bool
    bugs_critical: int = 0
    bugs_high:     int = 0
    bugs_medium:   int = 0
    bugs_low:      int = 0
    score:         float = 0.0
    has_analysis:  bool = False   

    @property
    def total_bugs(self) -> int:
        return self.bugs_critical + self.bugs_high + self.bugs_medium + self.bugs_low

    @property
    def max_severity(self) -> str:
        if self.bugs_critical: return "CRITICAL"
        if self.bugs_high:     return "HIGH"
        if self.bugs_medium:   return "MEDIUM"
        if self.bugs_low:      return "LOW"
        return "CLEAN"


@dataclass
class SessionSnapshot:
    """
    Photographie complète de la session à un instant T.
    Produite par calculate_session_score() et passée au GitNotifier.
    """
    score:               float
    level:               str                   # CLEAN / WATCH / WARN / CRITICAL
    files_at_risk:       List[FileRisk]
    files_unanalyzed:    List[str]             # modifiés mais sans analyse SQLite
    minutes_since_commit: int
    time_multiplier:     float
    stats: Dict[str, Any] = field(default_factory=dict)
    test_gaps:           List[Any] = field(default_factory=list)  # TestGapStatus

    @property
    def total_critical(self) -> int:
        return sum(f.bugs_critical for f in self.files_at_risk)

    @property
    def total_high(self) -> int:
        return sum(f.bugs_high for f in self.files_at_risk)

    @property
    def total_bugs(self) -> int:
        return sum(f.total_bugs for f in self.files_at_risk)


# ─────────────────────────────────────────────────────────────────────────────
# GitSessionTracker
# ─────────────────────────────────────────────────────────────────────────────

class GitSessionTracker:
    """
    Surveille la session de développement en arrière-plan.

    Usage typique (dans Orchestrator.initialize()) :
        tracker = GitSessionTracker(
            project_path = self.project_path,
            cache_db     = Path("data/cache/analysis_cache.db"),
            notifier     = git_notifier,
        )
        tracker.start()    # lance le thread daemon
        # ... mode watch ...
        tracker.stop()     # arrêt propre à Ctrl+C
    """

    def __init__(
        self,
        project_path:   Path,
        cache_db:       Path,
        notifier=None,                
        check_interval: int = CHECK_INTERVAL,
    ):
        self.project_path   = project_path
        self.cache_db       = cache_db
        self._notifier      = notifier
        self._interval      = check_interval

        self._thread:  Optional[threading.Thread] = None
        self._running: bool = False
        self._lock     = threading.Lock()

        # Dernier snapshot calculé (accessible depuis l'extérieur)
        self._last_snapshot: Optional[SessionSnapshot] = None
        # Dernier niveau notifié (hysteresis)
        self._last_level: str = "CLEAN"

        # FIX 3 — reminder timers
        self._level_since:         Dict[str, float] = {}
        self._last_reminder_fired: Dict[str, float] = {}

    # ── Interface publique ────────────────────────────────────────────────────

    def start(self):
        """Lance le thread de surveillance en arrière-plan."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="GitSessionTracker", daemon=True
        )
        self._thread.start()
        logger.info("GitSessionTracker démarré (intervalle %ds)", self._interval)

    def stop(self):
        """Arrêt propre — attend la fin du cycle en cours."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._interval + 5)
        logger.info("GitSessionTracker arrêté")

    def get_snapshot(self) -> Optional[SessionSnapshot]:
        """Retourne le dernier snapshot calculé (thread-safe)."""
        with self._lock:
            return self._last_snapshot

    def force_check(self) -> Optional[SessionSnapshot]:
        """
        Déclenche un calcul immédiat (hors cycle).
        Utile pour le pre-commit hook qui veut le score à l'instant T.
        """
        return self._run_check(notify=False)

    # ── Boucle principale ─────────────────────────────────────────────────────

    def _loop(self):
        """
        Tourne en arrière-plan, se réveille toutes les _interval secondes.
        Utilise time.sleep() par tranches de 1s pour réagir rapidement à stop().
        """
        elapsed = 0
        while self._running:
            time.sleep(1)
            elapsed += 1
            if elapsed >= self._interval:
                elapsed = 0
                self._run_check(notify=True)

    def _run_check(self, notify: bool) -> Optional[SessionSnapshot]:
        """
        Calcule le snapshot courant et notifie si nécessaire.
        Retourne le snapshot pour usage direct (force_check).
        """
        try:
            snapshot = self.calculate_session_score()
            with self._lock:
                self._last_snapshot = snapshot

            if notify and self._notifier:
                self._maybe_notify(snapshot)

            return snapshot
        except Exception as e:
            logger.error("GitSessionTracker._run_check erreur : %s", e)
            return None

    # Calcul du score ───────────────────────────────────────────────────────

    def calculate_session_score(self) -> SessionSnapshot:
        """
        Fonction principale — produit le SessionSnapshot complet.

        Séquence :
          1. Récupère les fichiers uncommités via git diff HEAD
          2. Pour chaque fichier, lit l'analyse dans SQLite
          3. Calcule le score pondéré (bugs × sévérité × temps)
          4. Détermine le niveau (CLEAN / WATCH / WARN / CRITICAL)
          5. Retourne le snapshot avec tous les détails
        """
        from smart_git.git_diff_parser import get_uncommitted_files, get_session_stats, is_git_repo

        # Vérification préalable — repo git valide ?
        if not is_git_repo(self.project_path):
            return SessionSnapshot(
                score=0, level="CLEAN", files_at_risk=[],
                files_unanalyzed=[], minutes_since_commit=0, time_multiplier=1.0,
            )

        # Étape 1 : fichiers modifiés depuis le dernier commit
        uncommitted = get_uncommitted_files(self.project_path)
        # Filtrer sur les extensions surveillées seulement
        uncommitted = [
            f for f in uncommitted
            if Path(f["path"]).suffix.lower() in WATCHED_EXTENSIONS
            and f["status"] != "D" 
        ]

        # Étape 2 : stats globales de session
        stats = get_session_stats(self.project_path)
        minutes = stats.get("minutes_since_commit", 0)

        # Étape 3 : facteur temps
        time_mult = _time_multiplier(minutes)

        # Étape 4 : évaluer chaque fichier depuis le cache SQLite
        files_at_risk:    List[FileRisk] = []
        files_unanalyzed: List[str]      = []
        total_score = 0.0

        for file_info in uncommitted:
            file_path_abs = self.project_path / file_info["path"]
            risk = self._assess_file_risk(file_path_abs, file_info)

            if not risk.has_analysis:
                files_unanalyzed.append(file_info["path"])
                continue

            if risk.total_bugs > 0:
                files_at_risk.append(risk)
                total_score += risk.score

        # Score final avec facteur temps
        final_score = total_score * time_mult

        # Étape 5 : détecter les test gaps (0 token — scan rapide)
        test_gaps = []
        try:
            from services.test_discovery import TestDiscoveryService
            discovery = TestDiscoveryService(self.project_path)
            for file_info in uncommitted:
                file_path_abs = self.project_path / file_info["path"]
                if not discovery.has_test_file(file_path_abs):
                    test_gaps.append({
                        "source_file": file_path_abs,
                        "test_file": None,
                        "missing": True,
                        "impact_score": 50,
                        "reason": f"aucun test pour {file_path_abs.name}",
                    })
        except Exception as e:
            logger.debug("TestDiscoveryService erreur : %s", e)

        # Étape 6 : déterminer le niveau
        level = _score_to_level(final_score)

        return SessionSnapshot(
            score                = round(final_score, 1),
            level                = level,
            files_at_risk        = sorted(files_at_risk, key=lambda f: f.score, reverse=True),
            files_unanalyzed     = files_unanalyzed,
            minutes_since_commit = minutes,
            time_multiplier      = time_mult,
            stats                = stats,
            test_gaps            = test_gaps,
        )

    def _assess_file_risk(self, file_path_abs: Path, file_info: Dict) -> FileRisk:
        """
        Lit la dernière analyse d'un fichier depuis SQLite et calcule son score.

        Pourquoi SQLite et pas re-analyser avec le LLM ?
        → Le mode Watch a DÉJÀ analysé ce fichier à chaque Ctrl+S.
        → Le cache contient le résultat. Relancer le LLM ici doublerait
          le quota Gemini et ralentirait l'expérience développeur.

        Si le fichier n'a jamais été analysé (pas dans le cache) :
        → FileRisk.has_analysis = False → ajouté à files_unanalyzed
        → Le GitNotifier peut suggérer de lancer une analyse manuellement.
        """
        risk = FileRisk(
            path=file_info["path"],
            status=file_info["status"],
            staged=file_info.get("staged", False),
        )

        analysis_text = self._read_analysis_from_cache(str(file_path_abs))
        if not analysis_text:
            return risk   # has_analysis = False par défaut

        risk.has_analysis = True

        # FIX 1 — Use structured block counting (same as git_hook.py v3)
        from smart_git.git_hook import _count_severity_from_blocks
        c, h, m, score = _count_severity_from_blocks(analysis_text)
        risk.bugs_critical = c
        risk.bugs_high     = h
        risk.bugs_medium   = m
        risk.score         = score
        return risk

    def _read_analysis_from_cache(self, file_path: str) -> Optional[str]:
        """
        Lit le texte d'analyse depuis SQLite pour un fichier donné.
        Retourne None si le fichier n'est pas dans le cache.

        Utilise une connexion SQLite séparée (lecture seule) pour éviter
        tout conflit avec la connexion d'écriture de CacheService.
        """
        if not self.cache_db.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{self.cache_db}?mode=ro", uri=True)
            row  = conn.execute(
                "SELECT analysis_text FROM file_cache WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            logger.debug("SQLite read erreur pour %s : %s", file_path, e)
            return None

    #  Notification avec hysteresis + FIX 3 reminder timer

    def _maybe_notify(self, snapshot: SessionSnapshot):
        """
        Notifie le GitNotifier sur changement de niveau (hysteresis)
        OU quand le reminder interval expire au même niveau.

        Règles d'hysteresis :
          - Monte        : notifier toujours
          - Descend      : notifier seulement si → CLEAN
          - Même niveau  : fire reminder si interval expiré (FIX 3)
        """
        new_level  = snapshot.level
        prev_level = self._last_level
        now        = time.time()

        levels_order = ["CLEAN", "WATCH", "WARN", "CRITICAL"]
        new_idx  = levels_order.index(new_level)
        prev_idx = levels_order.index(prev_level)

        # Level change logic (original hysteresis)
        level_changed = False
        if new_idx > prev_idx:
            level_changed = True
        elif new_idx < prev_idx and new_level == "CLEAN":
            level_changed = True

        if level_changed:
            self._last_level = new_level
            self._level_since[new_level]         = now
            self._last_reminder_fired[new_level] = now
            try:
                self._notifier.notify(snapshot)
            except Exception as e:
                logger.debug("GitNotifier.notify erreur : %s", e)
            return

        # FIX 3 — Reminder: same level, check if interval expired
        if new_level in ("WATCH", "WARN", "CRITICAL") and new_level == prev_level:
            interval   = REMINDER_INTERVALS.get(new_level, 999999)
            last_fired = self._last_reminder_fired.get(new_level, 0)

            if now - last_fired >= interval:
                self._last_reminder_fired[new_level] = now
                entered_at = self._level_since.get(new_level, now)
                minutes_at_level = int((now - entered_at) / 60)

                try:
                    self._notifier.notify_reminder(snapshot, minutes_at_level)
                except AttributeError:
                    # Fallback if notifier doesn't have notify_reminder
                    self._notifier.notify(snapshot)
                except Exception as e:
                    logger.debug("GitNotifier.notify_reminder erreur : %s", e)

        # Notification des test gaps (pas liée au niveau de risque)
        if snapshot.test_gaps:
            try:
                from agents.test_proposal_notifier import TestProposalNotifier
                if isinstance(self._notifier, TestProposalNotifier):
                    self._notifier.notify_batch(snapshot.test_gaps)
                elif hasattr(self._notifier, "notify_test_gaps"):
                    self._notifier.notify_test_gaps(snapshot.test_gaps)
            except Exception as e:
                logger.debug("Test gap notification erreur : %s", e)

    # Accesseurs utiles

    def set_notifier(self, notifier) -> None:
        """Injecte le notifier après construction (évite import circulaire)."""
        self._notifier = notifier

    def get_last_level(self) -> str:
        """Dernier niveau calculé — utilisé par le pre-commit hook."""
        return self._last_level


# Fonctions utilitaires pures

def _time_multiplier(minutes: int) -> float:
    """
    Retourne le multiplicateur de score selon le temps écoulé depuis le dernier commit.
    Plus le développeur attend, plus les bugs accumulés pèsent lourd dans le score.
    """
    for threshold, mult in TIME_MULTIPLIERS:
        if minutes >= threshold:
            return mult
    return 1.0


def _score_to_level(score: float) -> str:
    """Convertit un score numérique en niveau textuel."""
    if score >= LEVEL_THRESHOLDS["CRITICAL"]: return "CRITICAL"
    if score >= LEVEL_THRESHOLDS["WARN"]:     return "WARN"
    if score >= LEVEL_THRESHOLDS["WATCH"]:    return "WATCH"
    return "CLEAN"