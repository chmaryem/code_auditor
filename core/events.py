"""
events.py — Messages qui circulent dans le système.

Principe : quand quelque chose se passe (un fichier change, un commit Git...),
on crée un Event et on le passe à l'Orchestrateur.
Personne n'appelle directement les autres — ils s'envoient des Events.

v2 — Ajout du champ `priority` pour l'asyncio.PriorityQueue.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# ── Priorités ────────────────────────────────────────────────────────────────
# Plus la valeur est basse, plus l'événement est traité tôt.

CRITICAL_PRIORITY = 0    # Fichiers avec sévérité critique connue
HIGH_PRIORITY     = 10   # Changements de logique, nouvelles fonctions
NORMAL_PRIORITY   = 50   # Changements standards
LOW_PRIORITY      = 90   # Analyses de dépendants, tâches secondaires


class EventType(Enum):
    FILE_CHANGED   = "file_changed"
    FILE_DELETED   = "file_deleted"
    FILE_CREATED   = "file_created"
    GIT_COMMIT     = "git_commit"
    MANUAL_ANALYZE = "manual_analyze"
    CODE_PARSED    = "code_parsed"
    KB_RETRIEVED   = "kb_retrieved"
    ANALYSIS_DONE  = "analysis_done"
    RESULT_READY   = "result_ready"
    FEEDBACK_GIVEN = "feedback_given"


@dataclass
class Event:
    type:      EventType
    payload:   Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime       = field(default_factory=datetime.now)
    source:    str            = "unknown"
    priority:  int            = NORMAL_PRIORITY

    # ── Comparaison pour asyncio.PriorityQueue ────────────────────────────
    # PriorityQueue compare les éléments avec < ; sans __lt__ le dataclass
    # crasherait dès que deux Event ont la même priorité.
    def __lt__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        # À priorité égale, le plus ancien passe en premier (FIFO)
        return self.timestamp < other.timestamp

    @property
    def file_path(self) -> Optional[Path]:
        p = self.payload.get("file_path")
        return Path(p) if p else None

    @property
    def language(self) -> str:
        return self.payload.get("language", "unknown")


# ── Constructeurs pratiques ───────────────────────────────────────────────────

def file_changed_event(file_path: Path, deleted: bool = False,
                       priority: int = NORMAL_PRIORITY) -> Event:
    etype = EventType.FILE_DELETED if deleted else EventType.FILE_CHANGED
    return Event(
        type=etype,
        payload={"file_path": str(file_path)},
        source="file_watcher",
        priority=priority,
    )

def git_commit_event(commit_hash: str, changed_files: list,
                     repo_path: Path = None) -> Event:
    return Event(
        type    = EventType.GIT_COMMIT,
        payload = {
            "commit_hash":   commit_hash,
            "changed_files": changed_files,
            "repo_path":     repo_path or Path("."),
        },
        source  = "git_hook",
        priority = HIGH_PRIORITY,
    )

def manual_analyze_event(file_path: Path) -> Event:
    return Event(
        type    = EventType.MANUAL_ANALYZE,
        payload = {"file_path": str(file_path)},
        source  = "cli",
        priority = HIGH_PRIORITY,
    )