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

        if intent == "pr_fix_guidance":
            return self._pr_fix_guidance(state)

        if intent == "pr_review":
            return self._pr_review(state)

        if intent == "conflict_resolution_dry_run":
            return self._conflicts(state)

        if intent == "secret_scan":
            return self._secret_scan(state)

        if intent == "commit_lint":
            return self._commit_lint(state)

        if intent == "test_impact":
            return self._test_impact(state)

        if intent == "cross_pr_conflicts":
            return self._cross_pr(state)

        if intent == "pr_description":
            return self._pr_description(state)

        if intent == "repo_overview":
            return self._repo_overview(state)

        return self._status(state)

    # ── Repo overview (repo_overview) ────────────────────────────────────────

    def _repo_overview(self, state: Dict[str, Any]) -> str:
        ov = state.get("repo_overview_report") or {}
        if not ov.get("success"):
            return (
                "## Aperçu du dépôt\n\n"
                "Impossible de récupérer les informations du dépôt "
                "(ni via GitHub, ni via le git local)."
            )

        if ov.get("source") == "github":
            lines = [
                f"## Dépôt {ov.get('full_name', '')}",
                "",
            ]
            if ov.get("description"):
                lines.append(f"{ov['description']}")
                lines.append("")
            lines += [
                f"- **Visibilité** : {ov.get('visibility', '—')}",
                f"- **Branche par défaut** : `{ov.get('default_branch', '—')}`",
                f"- **Langage principal** : {ov.get('language') or '—'}",
            ]
            if ov.get("open_prs") is not None:
                lines.append(f"- **PR ouvertes** : {ov['open_prs']}")
            lines += [
                f"- **Étoiles** : {ov.get('stars', 0)}",
                f"- **Dernière activité** : {ov.get('pushed_at', '—')}",
            ]
            if ov.get("html_url"):
                lines += ["", f"🔗 {ov['html_url']}"]
            return "\n".join(lines)

        # Source locale (repli)
        return "\n".join([
            f"## Dépôt local — {ov.get('full_name', '')}",
            "",
            "_Aucun dépôt GitHub connecté — informations issues du git local._",
            "",
            f"- **Branche courante** : `{ov.get('default_branch', '—')}`",
            f"- **Remote** : {ov.get('remote') or '—'}",
            f"- **Commits** : {ov.get('commits') or '—'}",
            f"- **Dernier commit** : {ov.get('last_commit') or '—'}",
        ])

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

        # Use the pre-built markdown body if available (includes CI details, review counts)
        if report.get("body"):
            return report["body"]

        ready   = report.get("ready")
        verdict = "✅ PR prête à merger." if ready else "⚠️ PR pas encore prête à merger."

        return (
            "## PR Readiness\n\n"
            f"{verdict}\n\n"
            f"| Check | Statut |\n"
            f"|---|---|\n"
            f"| Mergeable | {'✅' if report.get('mergeable') else '❌'} |\n"
            f"| CI/CD | {'✅' if report.get('ci_pass') else '❌'} |\n"
            f"| Reviews | {'✅' if report.get('reviews_approved') else 'ℹ️ En attente'} |\n"
        )

    # ── PR fix guidance ──────────────────────────────────────────────────────

    def _pr_fix_guidance(self, state: Dict[str, Any]) -> str:
        """
        "What's the solution to merge it?" — unlike _pr_readiness, this is a
        SOLUTION question, not a status question. Reuses the exact same
        readiness data (mergeable/ci_pass/reviews_approved/security_context)
        but renders it as concrete steps instead of re-stating the same report.
        """
        report = state.get("readiness_report") or {}

        if not report.get("success"):
            return f"## Comment merger cette PR\n\nErreur : `{report.get('error', 'unknown error')}`"

        if report.get("ready"):
            return (
                "## Comment merger cette PR\n\n"
                "✅ Elle est déjà prête — mergez-la directement, aucune action requise."
            )

        steps: list[str] = []
        security = report.get("security_context") or {}
        critical = security.get("critical", 0)
        high     = security.get("high", 0)

        if critical > 0:
            steps.append(
                f"**Corriger {critical} vulnérabilité(s) critique(s)** détectée(s) par la revue "
                f"de code — voir l'onglet **Review** pour le détail de chaque fichier."
            )
        elif high > 0:
            steps.append(
                f"**Examiner {high} vulnérabilité(s) high** signalée(s) par la revue de code "
                f"— voir l'onglet **Review**."
            )
        if not report.get("mergeable"):
            steps.append(
                "**Résoudre les conflits** — ouvrez l'onglet **Resolve** du dashboard "
                "(ou lancez la résolution automatique) avant toute autre étape."
            )
        if not report.get("ci_pass"):
            steps.append(
                "**Corriger les checks CI/CD en échec** — consultez les logs du check "
                "concerné et poussez un commit corrigé."
            )
        if not report.get("reviews_approved"):
            steps.append(
                "**Obtenir une approbation** — demandez une review à un(e) collègue, "
                "ou via l'onglet **Review** du dashboard."
            )

        if not steps:
            steps.append(
                "Aucun blocage clair identifié automatiquement — consultez le détail "
                "du rapport de readiness pour plus de contexte."
            )

        lines = ["## Comment merger cette PR", "", "**Plan d'action, dans l'ordre :**", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        return "\n".join(lines)

    # ── PR review ───────────────────────────────────────────────────────────

    def _pr_review(self, state: Dict[str, Any]) -> str:
        report = state.get("pr_report") or {}

        if not report.get("success"):
            return (
                "## PR Review\n\n"
                f"Erreur : `{report.get('error', 'unknown error')}`"
            )

        # If the review agent returned the full body, use it directly
        if report.get("body"):
            return report["body"]

        score    = report.get("score", report.get("total_score", "N/A"))
        critical = report.get("critical", report.get("total_critical", 0))
        high     = report.get("high", report.get("total_high", 0))
        medium   = report.get("medium", report.get("total_medium", 0))
        verdict  = report.get("verdict", "COMMENT")

        verdict_line = {
            "APPROVE":         "✅ **MERGE AUTORISÉ** — aucun problème critique détecté.",
            "COMMENT":         "⚠️ **MERGE AVEC PRÉCAUTION** — des points méritent attention.",
            "REQUEST_CHANGES": "❌ **MERGE BLOQUÉ** — corrections requises avant fusion.",
        }.get(verdict, f"Verdict : {verdict}")

        return (
            "## PR Review\n\n"
            f"{verdict_line}\n\n"
            f"| Métrique | Valeur |\n"
            f"|---|---|\n"
            f"| Score global | **{score:.1f}** |\n"
            f"| Critical | **{critical}** |\n"
            f"| High | **{high}** |\n"
            f"| Medium | **{medium}** |\n"
            f"| Fichiers analysés | **{report.get('files_analyzed', 0)}** |\n"
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


    # ── F1: Secret scan ─────────────────────────────────────────────────────

    def _secret_scan(self, state: Dict[str, Any]) -> str:
        report = state.get("secret_scan_report") or {}

        if report.get("error"):
            return f"## Secret Scan\n\nErreur : `{report['error']}`"

        count  = len(report.get("findings", []))
        scanned = report.get("scanned_files", 0)

        if not report.get("has_secrets"):
            return (
                f"## Secret Scan\n\n"
                f"✅ Aucun secret détecté dans {scanned} fichier(s) stagé(s)."
            )

        lines = [
            "## Secret Scan — SECRETS DÉTECTÉS",
            "",
            f"🔐 **{count} secret(s)** détecté(s) dans {scanned} fichier(s) stagé(s).",
            f"**⛔ Commit bloqué.** Supprimez les secrets avant de committer.",
            "",
            "### Détails",
        ]

        by_file: dict = {}
        for f in report.get("findings", []):
            by_file.setdefault(f["file_path"], []).append(f)

        for file_path, findings in by_file.items():
            lines.append(f"\n**`{file_path}`**")
            for f in findings:
                lines.append(
                    f"- Ligne {f['line_number']} : `{f['secret_type']}` — `{f['masked_text']}`"
                )

        lines += [
            "",
            "### Actions",
            "1. Supprimez le secret du code",
            "2. Utilisez `os.environ['KEY']` ou un fichier `.env` (ajouté dans `.gitignore`)",
            "3. Si déjà pushé : **révoquez la clé immédiatement**",
        ]
        return "\n".join(lines)

    # ── F3: Commit lint ──────────────────────────────────────────────────────

    def _commit_lint(self, state: Dict[str, Any]) -> str:
        report = state.get("commit_lint_report") or {}

        if report.get("error"):
            return f"## Commit Lint\n\nErreur : `{report['error']}`"

        msg     = report.get("original_message", "")
        score   = report.get("score", 100)
        is_valid = report.get("is_valid", True)
        violations = report.get("violations", [])

        if is_valid and not report.get("has_warnings"):
            return (
                f"## Commit Lint\n\n"
                f"✅ Message valide (score {score}/100).\n\n"
                f"```text\n{msg}\n```"
            )

        icon = "✅" if is_valid else "❌"
        lines = [
            f"## Commit Lint — {icon} {'Valide' if is_valid else 'Invalide'} (score {score}/100)",
            "",
            f"```text\n{msg}\n```",
            "",
        ]

        errors   = [v for v in violations if v["severity"] == "ERROR"]
        warnings = [v for v in violations if v["severity"] == "WARN"]

        if errors:
            lines.append("### Erreurs")
            for v in errors:
                lines.append(f"- ❌ **{v['rule']}** : {v['message']}")
                if v.get("suggestion"):
                    lines.append(f"  → Suggestion : `{v['suggestion']}`")

        if warnings:
            lines.append("\n### Avertissements")
            for v in warnings:
                lines.append(f"- ⚠️ **{v['rule']}** : {v['message']}")
                if v.get("suggestion"):
                    lines.append(f"  → Suggestion : `{v['suggestion']}`")

        if report.get("suggested_message"):
            lines += [
                "",
                "### Message corrigé suggéré",
                f"```text\n{report['suggested_message']}\n```",
            ]

        return "\n".join(lines)

    # ── F4: Test impact ──────────────────────────────────────────────────────

    def _test_impact(self, state: Dict[str, Any]) -> str:
        report = state.get("test_impact_report") or {}

        if report.get("error"):
            return f"## Test Impact\n\nErreur : `{report['error']}`"

        total   = report.get("total_files", 0)
        covered = report.get("covered_files", 0)
        gaps    = report.get("uncovered_files", 0)
        ratio   = round(report.get("coverage_ratio", 1.0) * 100)

        if total == 0:
            return "## Test Impact\n\nAucun fichier source stagé détecté."

        lines = [
            "## Test Impact Analysis",
            "",
            f"- Fichiers source modifiés : **{total}**",
            f"- Couverts par des tests   : **{covered}** ({ratio}%)",
            f"- Sans tests (gaps)        : **{gaps}**",
        ]

        all_tests = report.get("all_test_files", [])
        if all_tests:
            lines += ["", "### Tests à exécuter"]
            for t in all_tests[:20]:
                lines.append(f"- `{t}`")

        impacts = report.get("impacts", [])
        gap_files = [i for i in impacts if i.get("missing_tests")]
        if gap_files:
            lines += ["", "### Fichiers sans tests (coverage gaps)"]
            for i in gap_files:
                lines.append(f"- `{i['source_file']}` — aucun test trouvé")

        return "\n".join(lines)

    # ── F6: Cross-PR conflicts ───────────────────────────────────────────────

    def _cross_pr(self, state: Dict[str, Any]) -> str:
        report = state.get("cross_pr_report") or {}

        if report.get("error"):
            return f"## Cross-PR Conflicts\n\nErreur : `{report['error']}`"

        total_prs = report.get("total_open_prs", 0)
        conflicts = report.get("conflicts", [])

        if not report.get("has_conflicts"):
            return (
                f"## Cross-PR Conflicts\n\n"
                f"✅ Aucun conflit cross-PR détecté parmi {total_prs} PR(s) ouvertes."
            )

        high   = report.get("high_risk_count", 0)
        medium = report.get("medium_risk_count", 0)

        lines = [
            "## Cross-PR Conflicts",
            "",
            f"⚠️ **{len(conflicts)} fichier(s)** partagés entre plusieurs PRs ouvertes.",
            f"- PRs analysées : **{total_prs}**",
            f"- Risque HIGH : **{high}** | Risque MEDIUM : **{medium}**",
            "",
            "### Conflits détectés",
        ]

        for c in conflicts[:15]:
            risk_icon = "🔴" if c["risk_level"] == "HIGH" else "🟡"
            pr_refs = ", ".join(f"#{n}" for n in c["pr_numbers"])
            lines.append(f"\n{risk_icon} **`{c['file_path']}`** — PRs : {pr_refs}")
            lines.append(f"  {c['note']}")

        lines += [
            "",
            "### Recommandation",
            "Merges ces PRs dans un ordre séquentiel et résolvez les conflits au fur et à mesure.",
        ]

        return "\n".join(lines)

    # ── F7: PR description ───────────────────────────────────────────────────

    def _pr_description(self, state: Dict[str, Any]) -> str:
        report = state.get("pr_description") or {}

        if report.get("error"):
            return f"## PR Description\n\nErreur : `{report['error']}`"

        description = report.get("description", "")
        if not description:
            return "## PR Description\n\nImpossible de générer la description."

        from_cache = report.get("from_cache", False)
        commits    = report.get("commits_used", 0)
        files      = report.get("files_changed", 0)
        adds       = report.get("additions", 0)
        dels       = report.get("deletions", 0)

        header = (
            f"## PR Description Générée {'(cache)' if from_cache else '(LLM)'}\n\n"
            f"_Basée sur {commits} commit(s), {files} fichier(s) — "
            f"+{adds}/-{dels} lignes_\n\n"
            "---\n\n"
        )

        return header + description


git_synthesis_agent = LCGitSynthesisAgent()