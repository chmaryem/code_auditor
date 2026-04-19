"""
git_notifier.py — Affichage des alertes de session dans le terminal.

Reçoit un SessionSnapshot et choisit le format d'affichage adapté
selon le niveau de risque : CLEAN / WATCH / WARN / CRITICAL.

Principes de design :
  1. Non-bloquant : toutes les notifications sont des print() simples.
     Le développeur n'est jamais interrompu dans son flow de travail.

  2. Progressif : plus le niveau est haut, plus l'affichage est détaillé.
     CLEAN    → silence (pas de notification)
     WATCH    → une seule ligne, discrète
     WARN     → tableau par fichier + recommandation
     CRITICAL → bloc bien visible + liste des bugs critiques + action urgente

  3. Respectueux du terminal : utilise print_lock pour ne pas interrompre
     un affichage de solution du mode Watch en cours.

  4. Formaté comme le reste du système : même palette ANSI que console_renderer.py.

"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smart_git.git_session_tracker import SessionSnapshot

# ── Couleurs ANSI (cohérentes avec console_renderer.py) ──────────────────────
_R  = "\033[0m"        # reset
_B  = "\033[1m"        # bold
_DM = "\033[2m"        # dim
_GR = "\033[92m"       # vert
_YL = "\033[93m"       # jaune
_RD = "\033[91m"       # rouge
_CY = "\033[96m"       # cyan
_MG = "\033[95m"       # magenta

# Icônes de sévérité
_SEV_ICON = {
    "CRITICAL": f"{_RD}●{_R} CRITICAL",
    "HIGH":     f"{_YL}●{_R} HIGH",
    "MEDIUM":   f"{_CY}●{_R} MEDIUM",
    "LOW":      f"{_DM}●{_R} LOW",
    "CLEAN":    f"{_GR}●{_R} CLEAN",
}

# Niveau → couleur de l'en-tête
_LEVEL_COLOR = {
    "WATCH":    _CY,
    "WARN":     _YL,
    "CRITICAL": _RD,
    "CLEAN":    _GR,
}


class GitNotifier:
   
    def __init__(self, print_lock: threading.Lock = None):
        # Si aucun lock fourni, on en crée un local (mode standalone)
        self._lock = print_lock or threading.Lock()

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def notify(self, snapshot: "SessionSnapshot") -> None:
        """
        Affiche la notification adaptée au niveau du snapshot.
        Appelé par GitSessionTracker._maybe_notify() lors d'un changement de niveau.
        """
        level = snapshot.level

        if level == "CLEAN":
            self._notify_clean(snapshot)
        elif level == "WATCH":
            self._notify_watch(snapshot)
        elif level == "WARN":
            self._notify_warn(snapshot)
        elif level == "CRITICAL":
            self._notify_critical(snapshot)

    def notify_reminder(self, snapshot: "SessionSnapshot", minutes_at_level: int) -> None:
        """
        FIX 3 — Reminder notification after N minutes at same level.

        Different from notify(): uses a compact, inline format so it
        doesn't look like a new event. Makes clear this is a reminder
        that issues are STILL unresolved.
        """
        level = snapshot.level
        color = _LEVEL_COLOR.get(level, _YL)

        with self._lock:
            c = snapshot.total_critical
            h = snapshot.total_high
            nb_files = len(snapshot.files_at_risk)

            now = datetime.now().strftime("%H:%M")

            print(f"\n  {color}⏰  [{now}] RAPPEL — {level} depuis {minutes_at_level} min"
                  f" — {nb_files} fichier(s) non corrigés"
                  f"  (🔴{c} 🟠{h}  score {snapshot.score}){_R}")

            # For CRITICAL, repeat the file list
            if level == "CRITICAL" and snapshot.files_at_risk:
                for fr in snapshot.files_at_risk[:3]:
                    name = fr.path.split("/")[-1]
                    print(f"    {_RD}●{_R}  {name}  —  {fr.bugs_critical}C {fr.bugs_high}H")

            print(f"  {_DM}Appliquez les corrections Watch ou git commit --no-verify.{_R}\n")

    def notify_unanalyzed(self, file_names: list) -> None:
        """
        Notifie que des fichiers modifiés n'ont pas encore été analysés.
        Suggestion : lancer le mode watch ou analyser manuellement.
        Affiché seulement si > 2 fichiers sans analyse.
        """
        if len(file_names) <= 2:
            return
        with self._lock:
            print(f"\n  {_YL}⚠  {len(file_names)} fichier(s) modifiés sans analyse récente{_R}")
            for name in file_names[:5]:
                print(f"     {_DM}→ {name}{_R}")
            if len(file_names) > 5:
                print(f"     {_DM}... +{len(file_names)-5} autre(s){_R}")
            print(f"  {_DM}Conseil : le mode watch analyse automatiquement à chaque Ctrl+S.{_R}\n")

    #  Niveaux de notification

    def _notify_clean(self, snapshot: "SessionSnapshot") -> None:
        """
        Retour à CLEAN après avoir été en WARN ou CRITICAL.
        Indique que le commit a résolu les problèmes.
        """
        with self._lock:
            print(f"\n  {_GR}{_B}✓  Session propre — aucun problème en attente{_R}")
            print(f"  {_DM}Dernier commit : il y a {snapshot.minutes_since_commit} min{_R}\n")

    def _notify_watch(self, snapshot: "SessionSnapshot") -> None:
        """
        Niveau WATCH : 1-2 bugs mineurs détectés.
        Notification discrète — une seule ligne.
        Objectif : informer sans interrompre.
        """
        nb_files = len(snapshot.files_at_risk)
        nb_bugs  = snapshot.total_bugs
        mins     = snapshot.minutes_since_commit

        with self._lock:
            print(
                f"\n  {_CY}⚡  Session : {nb_bugs} problème(s) dans {nb_files} fichier(s) "
                f"— {mins} min depuis le dernier commit  (score {snapshot.score}){_R}\n"
            )

    def _notify_warn(self, snapshot: "SessionSnapshot") -> None:
        """
        Niveau WARN : plusieurs bugs ou au moins un HIGH.
        Affiche un tableau par fichier + recommandation de commit.
        """
        col_w = 28   # largeur colonne fichier

        with self._lock:
            print()
            c = _LEVEL_COLOR["WARN"]
            print(f"  {c}{_B}⚡  Session Watch — {snapshot.total_bugs} problème(s) détecté(s){_R}")
            print(f"  {_DM}Temps depuis le dernier commit : {snapshot.minutes_since_commit} min"
                  f"  ·  Multiplicateur temps : ×{snapshot.time_multiplier}{_R}")
            print(f"  {'─' * 62}")
            print(f"  {'Fichier':<{col_w}}  {'Bugs':>4}  Sévérité max")
            print(f"  {'─' * 62}")

            for fr in snapshot.files_at_risk:
                name   = fr.path.split("/")[-1][:col_w]
                icon   = _SEV_ICON.get(fr.max_severity, fr.max_severity)
                staged = f"{_GR}[S]{_R}" if fr.staged else "   "
                print(f"  {name:<{col_w}}  {fr.total_bugs:>4}  {icon}  {staged}")

            print(f"  {'─' * 62}")
            print(f"  {_YL}→  Recommandation : corriger les HIGH/CRITICAL avant le prochain commit.{_R}")

            if snapshot.files_unanalyzed:
                print(f"  {_DM}  {len(snapshot.files_unanalyzed)} fichier(s) modifié(s) sans analyse récente.{_R}")

            print()

    def _notify_critical(self, snapshot: "SessionSnapshot") -> None:
        """
        Niveau CRITICAL : bug(s) CRITICAL détecté(s) ou accumulation importante.
        Bloc d'alerte bien visible avec liste des bugs critiques.
        Action urgente suggérée.
        """
        border = f"  {'═' * 62}"

        with self._lock:
            print()
            print(f"  {_RD}{_B}{border}{_R}")
            print(f"  {_RD}{_B}  ✗  ALERTE SESSION — ACTION REQUISE AVANT LE COMMIT{_R}")
            print(f"  {_RD}{_B}{border}{_R}")
            print(f"  Score de risque : {_B}{snapshot.score}{_R}  "
                  f"·  Temps : {snapshot.minutes_since_commit} min  "
                  f"·  Multiplicateur : ×{snapshot.time_multiplier}")
            print()

            # Liste des fichiers critiques
            critical_files = [f for f in snapshot.files_at_risk if f.bugs_critical > 0]
            high_files     = [f for f in snapshot.files_at_risk if f.bugs_high > 0 and f.bugs_critical == 0]

            if critical_files:
                print(f"  {_RD}Bugs CRITICAL à corriger :{_R}")
                for fr in critical_files:
                    name = fr.path.split("/")[-1]
                    print(f"    {_RD}●{_R}  {name}  —  {fr.bugs_critical} CRITICAL, {fr.bugs_high} HIGH")

            if high_files:
                print(f"  {_YL}Bugs HIGH à corriger :{_R}")
                for fr in high_files:
                    name = fr.path.split("/")[-1]
                    print(f"    {_YL}●{_R}  {name}  —  {fr.bugs_high} HIGH")

            print()
            print(f"  {_B}Actions recommandées :{_R}")
            print(f"    1. Corriger les CRITICAL ci-dessus (les solutions sont dans le terminal Watch)")
            print(f"    2. Sauvegarder les fichiers corrigés → le Watch re-analysera automatiquement")
            print(f"    3. Commiter une fois le score redescendu sous WARN")
            print()
            print(f"  {_DM}  Forcer le commit malgré tout : git commit --no-verify{_R}")
            print(f"  {_RD}{_B}{border}{_R}")
            print()