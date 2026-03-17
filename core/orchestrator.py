"""
Orchestrator — Chef d'orchestre du projet.

Rôle UNIQUE : recevoir un Event et décider quels composants appeler, dans quel ordre.
Il ne fait PAS d'analyse lui-même. Il délègue.

Phase actuelle : délègue à IncrementalAnalyzer (code existant, 100% fonctionnel).
Phase suivante : les 4 agents spécialisés remplaceront IncrementalAnalyzer progressivement.
"""
import logging
from pathlib import Path
from typing import Optional

from core.events import Event, EventType

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Point d'entrée unique pour toute demande d'analyse.

    Usage :
        orch = Orchestrator(project_path)
        orch.initialize()
        orch.handle(file_changed_event(Path("mon_fichier.py")))
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._analyzer    = None

    def initialize(self):
        """Démarre tous les composants."""
        from core.incremental_analyzer import IncrementalAnalyzer
        print(" Orchestrateur : initialisation...")
        self._analyzer = IncrementalAnalyzer(self.project_path)
        self._analyzer.initialize()
        print(" Orchestrateur : prêt\n")

    def handle(self, event: Event):
        """Reçoit un événement et le traite."""
        if event.type in (EventType.FILE_CHANGED, EventType.MANUAL_ANALYZE):
            self._on_file_changed(event.file_path, deleted=False)

        elif event.type == EventType.FILE_DELETED:
            self._on_file_changed(event.file_path, deleted=True)

        elif event.type == EventType.GIT_COMMIT:
            self._on_git_commit(event)

    def stop(self):
        if self._analyzer:
            self._analyzer.stop()

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_file_changed(self, file_path: Path, deleted: bool):
        if self._analyzer:
            self._analyzer.queue_analysis(file_path, deleted=deleted)

    def _on_git_commit(self, event: Event):
        """Analyse tous les fichiers modifiés dans un commit Git."""
        changed_files = event.payload.get("changed_files", [])
        for file_info in changed_files:
            path = Path(file_info.get("path", ""))
            if path.exists() and self._analyzer:
                self._analyzer.queue_analysis(path, deleted=False)