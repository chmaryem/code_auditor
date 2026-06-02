"""
lc_git_synthesis_agent.py — Smart Git Synthesis Agent.

Role:
  Convert structured Smart Git results into a developer-friendly answer.

This agent does not call Git.
It only formats the final response.
"""

from __future__ import annotations

from typing import Any, Dict, List


class LCGitSynthesisAgent:
    def synthesize(self, state: Dict[str, Any]) -> str:
        intent = state.get("intent", "git_status")

        if intent == "can_commit":
            return self._can_commit(state)

        if intent == "commit_message":
            return self._commit_message(state)

        if intent == "summarize_changes":
            return self._summarize_changes(state)

        if intent == "branch_readiness":
            return self._branch_readiness(state)

        if intent == "pr_readiness":
            return self._pr_readiness(state)

        if intent == "pr_review":
            return self._pr_review(state)

        if intent == "conflict_resolution_dry_run":
            return self._conflicts(state)

        return self._status(state)

    # ── Generic status ──────────────────────────────────────────────────────

    def _status(self, state: Dict[str, Any]) -> str:
        snapshot = state.get("session_snapshot") or {}

        if not snapshot.get("success"):
            return (
                "## Git Status\n\n"
                f"Impossible de lire le statut Git : "
                f"{snapshot.get('error', 'unknown error')}"
            )

        files_at_risk = snapshot.get("files_at_risk", [])
        files_unanalyzed = snapshot.get("files_unanalyzed", [])

        lines = [
            "## Git Session Status",
            "",
            f"- Niveau : **{snapshot.get('level')}**",
            f"- Score : **{snapshot.get('score')}**",
            f"- Critical : **{snapshot.get('total_critical', 0)}**",
            f"- High : **{snapshot.get('total_high', 0)}**",
            f"- Fichiers à risque : **{len(files_at_risk)}**",
            f"- Fichiers non analysés : **{len(files_unanalyzed)}**",
        ]

        if files_at_risk:
            lines += ["", "### Fichiers à risque"]
            for file in files_at_risk[:6]:
                lines.append(
                    f"- `{file.get('path')}` — "
                    f"C:{file.get('critical')} "
                    f"H:{file.get('high')} "
                    f"M:{file.get('medium')} "
                    f"score:{file.get('score')}"
                )

        return "\n".join(lines)

    # ── Can commit ──────────────────────────────────────────────────────────

    def _can_commit(self, state: Dict[str, Any]) -> str:
        snapshot = state.get("session_snapshot") or {}
        changes = state.get("changes") or {}

        if not snapshot.get("success"):
            return (
                "## Commit Readiness\n\n"
                f"Impossible de calculer le risque : "
                f"{snapshot.get('error', 'unknown error')}"
            )

        level = snapshot.get("level")
        score = snapshot.get("score", 0)
        files_at_risk = snapshot.get("files_at_risk", [])
        files_unanalyzed = snapshot.get("files_unanalyzed", [])

        if level in ("CLEAN", "WATCH"):
            verdict = "✅ Tu peux commit, avec risque faible."
        elif level == "WARN":
            verdict = (
                "⚠️ Commit possible, mais je recommande de corriger "
                "les fichiers à risque avant."
            )
        else:
            verdict = (
                "❌ Je déconseille le commit maintenant. "
                "Risque critique détecté."
            )

        lines = [
            "## Commit Readiness",
            "",
            verdict,
            "",
            f"- Niveau : **{level}**",
            f"- Score : **{score}**",
            f"- Fichiers staged : **{len(changes.get('staged_files', []))}**",
            f"- Fichiers modifiés : **{len(changes.get('uncommitted_files', []))}**",
        ]

        if files_at_risk:
            lines += ["", "### Fichiers à risque"]
            for file in files_at_risk[:6]:
                lines.append(
                    f"- `{file.get('path')}` — "
                    f"C:{file.get('critical')} "
                    f"H:{file.get('high')} "
                    f"M:{file.get('medium')} "
                    f"score:{file.get('score')}"
                )

        if files_unanalyzed:
            lines += ["", "### Fichiers non analysés"]
            for file in files_unanalyzed[:6]:
                lines.append(f"- `{file}`")

        return "\n".join(lines)

    # ── Commit message ──────────────────────────────────────────────────────

    def _commit_message(self, state: Dict[str, Any]) -> str:
        message = state.get("commit_message", "")

        if not message:
            changes = state.get("changes") or {}
            error = changes.get("error", "No staged diff found")
            return (
                "## Message de commit\n\n"
                f"Je n’ai pas pu générer de message : `{error}`.\n\n"
                "Vérifie que tu as des fichiers staged avec :\n\n"
                "```bash\n"
                "git diff --staged\n"
                "```"
            )

        return (
            "## Message de commit proposé\n\n"
            "```text\n"
            f"{message}\n"
            "```\n\n"
            "Tu peux l’utiliser tel quel ou l’ajuster avant commit."
        )

    # ── Summarize changes ───────────────────────────────────────────────────

    def _summarize_changes(self, state: Dict[str, Any]) -> str:
        changes = state.get("changes") or {}

        if not changes.get("success"):
            return (
                "## Changements Git\n\n"
                f"Erreur : `{changes.get('error', 'unknown error')}`"
            )

        files = changes.get("uncommitted_files", [])
        staged = changes.get("staged_files", [])
        stats = changes.get("session_stats", {})

        lines = [
            "## Résumé des changements",
            "",
            f"- Fichiers modifiés : **{len(files)}**",
            f"- Fichiers staged : **{len(staged)}**",
            f"- Lignes ajoutées : **{stats.get('lines_added', 0)}**",
            f"- Lignes supprimées : **{stats.get('lines_removed', 0)}**",
            f"- Dernier commit : **{stats.get('last_commit_msg', '')}**",
        ]

        if staged:
            lines += ["", "### Fichiers staged"]
            for file in staged[:10]:
                lines.append(f"- `{file.get('status')}` `{file.get('path')}`")

        if files:
            lines += ["", "### Tous les fichiers modifiés"]
            for file in files[:10]:
                staged_label = "staged" if file.get("staged") else "unstaged"
                lines.append(
                    f"- `{file.get('status')}` `{file.get('path')}` — {staged_label}"
                )

        return "\n".join(lines)

    # ── Branch readiness ────────────────────────────────────────────────────

    def _branch_readiness(self, state: Dict[str, Any]) -> str:
        report = state.get("branch_report") or {}

        if not report.get("success"):
            return (
                "## Branch Readiness\n\n"
                f"Erreur : `{report.get('error', 'unknown error')}`"
            )

        lines = [
            f"## Branch Readiness — `{report.get('branch')}` → `{report.get('base')}`",
            "",
            f"- Verdict : **{report.get('verdict')}**",
            f"- Score total : **{report.get('total_score')}**",
            f"- Critical : **{report.get('total_critical')}**",
            f"- High : **{report.get('total_high')}**",
            f"- Conflits potentiels : **{len(report.get('conflict_risks', []))}**",
            "",
            "### Recommandation",
            report.get("recommendation", "") or "_Aucune recommandation fournie._",
        ]

        files = report.get("files", [])
        if files:
            lines += ["", "### Fichiers principaux"]
            for file in files[:8]:
                lines.append(
                    f"- `{file.get('path')}` — "
                    f"severity `{file.get('max_severity')}` "
                    f"score `{file.get('score')}`"
                )

        conflicts = report.get("conflict_risks", [])
        if conflicts:
            lines += ["", "### Conflits potentiels"]
            for file in conflicts[:6]:
                lines.append(f"- `{file}`")

        return "\n".join(lines)

    # ── PR readiness ────────────────────────────────────────────────────────

    def _pr_readiness(self, state: Dict[str, Any]) -> str:
        report = state.get("readiness_report") or {}

        if not report.get("success"):
            return (
                "## PR Readiness\n\n"
                f"Erreur : `{report.get('error', 'unknown error')}`"
            )

        ready = report.get("ready")
        verdict = (
            "✅ PR prête à merger."
            if ready
            else "⚠️ PR pas encore prête à merger."
        )

        return (
            "## PR Readiness\n\n"
            f"{verdict}\n\n"
            f"- Ready : **{report.get('ready')}**\n"
            f"- Mergeable : **{report.get('mergeable')}**\n"
            f"- CI pass : **{report.get('ci_pass')}**\n"
            f"- Reviews approved : **{report.get('reviews_approved')}**\n"
        )

    # ── PR review ───────────────────────────────────────────────────────────

    def _pr_review(self, state: Dict[str, Any]) -> str:
        report = state.get("pr_report") or {}

        if not report.get("success"):
            return (
                "## PR Review\n\n"
                f"Erreur : `{report.get('error', 'unknown error')}`"
            )

        score = report.get("score", report.get("total_score", "N/A"))
        critical = report.get("critical", report.get("total_critical", 0))
        high = report.get("high", report.get("total_high", 0))

        return (
            "## PR Review\n\n"
            f"- Score : **{score}**\n"
            f"- Critical : **{critical}**\n"
            f"- High : **{high}**\n"
            f"- Résultat posté sur GitHub si la configuration MCP est active.\n"
        )

    # ── Conflicts ───────────────────────────────────────────────────────────

    def _conflicts(self, state: Dict[str, Any]) -> str:
        report = state.get("conflict_report") or {}

        if not report.get("success"):
            return (
                "## Conflict Resolution\n\n"
                f"Erreur : `{report.get('error', 'unknown error')}`"
            )

        if not report.get("has_conflicts"):
            return "## Conflict Resolution\n\n✅ Aucun fichier en conflit détecté."

        lines = [
            "## Conflict Resolution — Dry Run",
            "",
            "⚠️ Des conflits sont détectés.",
            "",
            "Je reste en mode safe : **aucune écriture automatique**.",
            "",
            "### Fichiers en conflit",
        ]

        for file in report.get("conflict_files", []):
            lines.append(f"- `{file}`")

        lines += [
            "",
            "### Prochaine étape",
            (
                report.get("next_step")
                or "Générer un aperçu de résolution puis demander confirmation."
            ),
        ]

        return "\n".join(lines)


git_synthesis_agent = LCGitSynthesisAgent()