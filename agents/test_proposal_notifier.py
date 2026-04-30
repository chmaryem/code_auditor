"""
test_proposal_notifier.py — Affichage des propositions de tests (style GitNotifier).

Principes de design (cohérents avec git_notifier.py) :
  1. Non-bloquant : print() simple, jamais d'input() ou de popup.
  2. Progressif : plus le gap est grave, plus le message est visible.
  3. Respectueux du terminal : utilise le print_lock de l'orchestrator.
  4. Actionnable : chaque message propose une commande CLI concrète.

Niveaux :
  INFO     → test manquant sur fichier non critique (1 ligne, dim)
  WARN     → fichier critique sans test ou couverture < 50% (ligne jaune)
  URGENT   → nouvelle entité publique complexe, 0% coverage (bloc visible)
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.test_gap_agent import TestGapStatus

# ── Couleurs ANSI (cohérentes avec console_renderer.py / git_notifier.py) ──
_R  = "\033[0m"
_B  = "\033[1m"
_DM = "\033[2m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_MG = "\033[95m"


class TestProposalNotifier:
    """
    Notifie le développeur des gaps de tests détectés.
    S'utilise dans l'Orchestrator (mode watch) et dans le GitSessionTracker.
    """

    def __init__(self, print_lock: threading.Lock = None):
        self._lock = print_lock or threading.Lock()

    # ── Point d'entrée principal ───────────────────────────────────────────

    def notify(self, status: "TestGapStatus") -> None:
        """Affiche la proposition adaptée au niveau de criticité."""
        if not status.needs_attention:
            self._notify_info(status)
            return

        if status.impact_score >= 75:
            self._notify_urgent(status)
        elif status.impact_score >= 50:
            self._notify_warn(status)
        else:
            self._notify_info(status)

    def notify_batch(self, statuses: list["TestGapStatus"]) -> None:
        """
        Notification groupée — utilisée par le GitSessionTracker
        quand plusieurs fichiers modifiés ont des gaps de tests.
        """
        urgent = [s for s in statuses if s.impact_score >= 75]
        warn   = [s for s in statuses if 50 <= s.impact_score < 75]
        info   = [s for s in statuses if s.impact_score < 50]

        with self._lock:
            if urgent:
                self._print_batch_urgent(urgent)
            if warn:
                self._print_batch_warn(warn)
            if info and not urgent and not warn:
                self._print_batch_info(info)

    # ── Niveaux individuels ──────────────────────────────────────────────────

    def _notify_info(self, status: "TestGapStatus") -> None:
        """Niveau INFO : une ligne discrète."""
        with self._lock:
            print(
                f"  {_DM}🧪 Test gap : {status.source_file.name} — "
                f"{status.reason}  [{status.impact_score}]{_R}"
            )

    def _notify_warn(self, status: "TestGapStatus") -> None:
        """Niveau WARN : ligne visible avec commande."""
        with self._lock:
            print(
                f"  {_YL}🧪 Test gap : {status.source_file.name} — "
                f"{status.reason}{_R}"
            )
            print(
                f"  {_DM}   → python main.py generate-tests "
                f"{status.source_file.name}{_R}"
            )

    def _notify_urgent(self, status: "TestGapStatus") -> None:
        """Niveau URGENT : bloc bien visible."""
        with self._lock:
            print()
            print(f"  {_MG}{_B}{'─' * 62}{_R}")
            print(
                f"  {_MG}{_B}🧪  TESTS MANQUANTS — {status.source_file.name}{_R}"
            )
            print(f"  {_MG}{_B}{'─' * 62}{_R}")
            print(f"  {status.reason}")

            if status.untested_entities:
                ents = ", ".join(status.untested_entities[:5])
                if len(status.untested_entities) > 5:
                    ents += f" +{len(status.untested_entities) - 5}"
                print(f"  Entités sans test : {_B}{ents}{_R}")

            print()
            print(
                f"  {_CY}→ Générer les tests : python main.py generate-tests "
                f"{status.source_file.name}{_R}"
            )
            print(
                f"  {_DM}   Framework détecté : {status.framework or 'inconnu'}{_R}"
            )
            print(f"  {_MG}{_B}{'─' * 62}{_R}")
            print()

    # ── Batch (GitSessionTracker) ────────────────────────────────────────────

    def _print_batch_urgent(self, statuses: list) -> None:
        print()
        print(f"  {_MG}{_B}{'═' * 62}{_R}")
        print(f"  {_MG}{_B}🧪  PLUSIEURS FICHIERS SANS TESTS DE COUVERTURE{_R}")
        print(f"  {_MG}{_B}{'═' * 62}{_R}")
        for s in statuses:
            ents = len(s.untested_entities)
            print(
                f"  • {s.source_file.name}  —  {_RD}{ents} entité(s) non testée(s){_R}"
            )
        print()
        print(
            f"  {_CY}→ Générer tous : python main.py generate-tests --all-uncommitted{_R}"
        )
        print(f"  {_MG}{_B}{'═' * 62}{_R}")
        print()

    def _print_batch_warn(self, statuses: list) -> None:
        print()
        print(f"  {_YL}{_B}🧪  Tests manquants sur {len(statuses)} fichier(s){_R}")
        for s in statuses:
            print(f"  • {s.source_file.name}  —  {s.reason}")
        print()

    def _print_batch_info(self, statuses: list) -> None:
        names = ", ".join(s.source_file.name for s in statuses[:3])
        if len(statuses) > 3:
            names += f" +{len(statuses) - 3}"
        print(
            f"  {_DM}🧪 Tests optionnels manquants : {names}{_R}"
        )
