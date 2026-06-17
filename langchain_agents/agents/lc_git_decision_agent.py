"""
lc_git_decision_agent.py — Git Decision Agent.

Role:
  Understand the developer's Git-related request and decide which Smart Git
  specialist agent should handle it.

This agent does NOT execute Git logic.
It only returns a routing plan.

Examples:
  "est-ce que je peux commit ?"       → can_commit
  "résume mes changements"            → summarize_changes
  "ma branche est prête ?"            → branch_readiness
  "review la PR 17"                   → pr_review
  "est-ce que la PR 17 peut merger ?" → pr_readiness
  "résous les conflits"               → conflict_resolution_dry_run
  "génère un message de commit"       → commit_message
"""

from __future__ import annotations

import re
from typing import Any, Dict


class LCGitDecisionAgent:
    """
    Lightweight decision agent for Smart Git.

    Current strategy:
      - deterministic rules for speed and reliability
      - safe mode by default for sensitive operations

    Later improvement:
      - add small LLM classifier fallback when confidence is low.
    """

    def decide(
        self,
        message: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        msg = (message or "").lower()
        context = context or {}

        plan: Dict[str, Any] = {
            "intent": "git_status",
            "confidence": 0.5,
            "selected_agents": ["session_agent"],
            "safe_mode": True,
            "needs_confirmation": False,
            "reason": "default git status",
        }

        pr_number = self._extract_pr_number(msg)
        if pr_number:
            plan["pr_number"] = pr_number

        # ── Can commit? ─────────────────────────────────────────────────────
        if self._has_any(msg, ["commit", "commiter", "commit?"]) and self._has_any(
            msg,
            ["peux", "can", "safe", "autorisé", "autorise", "allowed", "prêt"],
        ):
            plan.update(
                {
                    "intent": "can_commit",
                    "confidence": 0.88,
                    "selected_agents": ["session_agent", "diff_agent"],
                    "reason": "developer asks if current changes are safe to commit",
                }
            )
            return plan

        # ── Commit message ──────────────────────────────────────────────────
        if self._has_any(
            msg,
            [
                "message de commit",
                "commit message",
                "generate commit",
                "génère un message",
                "generer un message",
                "propose commit",
            ],
        ):
            plan.update(
                {
                    "intent": "commit_message",
                    "confidence": 0.9,
                    "selected_agents": ["diff_agent"],
                    "reason": "developer asks for commit message generation",
                }
            )
            return plan

        # ── Summarize changes ───────────────────────────────────────────────
        if self._has_any(
            msg,
            [
                "résume",
                "resume",
                "summarize",
                "summary",
                "changements",
                "changes",
                "diff",
                "modifications",
            ],
        ):
            plan.update(
                {
                    "intent": "summarize_changes",
                    "confidence": 0.85,
                    "selected_agents": ["diff_agent"],
                    "reason": "developer asks for diff/change summary",
                }
            )
            return plan

        # ── Branch readiness ────────────────────────────────────────────────
        if self._has_any(msg, ["branche", "branch"]) and self._has_any(
            msg,
            ["merge", "prête", "pret", "ready", "risque", "risk"],
        ):
            plan.update(
                {
                    "intent": "branch_readiness",
                    "confidence": 0.9,
                    "selected_agents": ["branch_agent"],
                    "reason": "developer asks branch readiness",
                }
            )
            return plan

        # ── PR review ───────────────────────────────────────────────────────
        if pr_number and self._has_any(
            msg,
            ["review", "revue", "analyse", "analyze", "check", "vérifie", "verifie"],
        ):
            plan.update(
                {
                    "intent": "pr_review",
                    "confidence": 0.9,
                    "selected_agents": ["pr_agent"],
                    "reason": "developer asks PR review",
                }
            )
            return plan

        # ── PR merge readiness ──────────────────────────────────────────────
        if pr_number and self._has_any(
            msg,
            ["merge", "ready", "prête", "pret", "readiness", "peut merger"],
        ):
            plan.update(
                {
                    "intent": "pr_readiness",
                    "confidence": 0.9,
                    "selected_agents": ["pr_agent"],
                    "reason": "developer asks PR merge readiness",
                }
            )
            return plan

        # ── F7: PR description — AVANT conflict/test_impact ─────────────────
        # Les noms de branche / titres de PR contiennent souvent "conflict" ou
        # "test" (ex: branche "test-analyzer-conflict"). On teste donc cette
        # intention spécifique AVANT les règles génériques par mot-clé, sinon
        # "generate pr description for branch test-analyzer-conflict" est
        # mal routé vers la résolution de conflits.
        if self._has_any(
            msg,
            ["description pr", "pr description", "génère description", "genere description",
             "describe pr", "generate description", "rédige pr", "redige pr",
             "description pull request", "generate pr description", "generate pr desc",
             "pr desc"],
        ):
            plan.update(
                {
                    "intent": "pr_description",
                    "confidence": 0.92,
                    "selected_agents": ["pr_description_agent"],
                    "reason": "generate PR description from commits and diff",
                }
            )
            return plan

        # ── Conflict resolution ─────────────────────────────────────────────
        if self._has_any(
            msg,
            [
                "conflit",
                "conflits",
                "conflict",
                "conflicts",
                "resolve",
                "résous",
                "resous",
                "résoudre",
                "resoudre",
            ],
        ):
            plan.update(
                {
                    "intent": "conflict_resolution_dry_run",
                    "confidence": 0.9,
                    "selected_agents": ["conflict_agent"],
                    "safe_mode": True,
                    "needs_confirmation": True,
                    "reason": "conflict resolution is sensitive; dry-run first",
                }
            )
            return plan

        # ── F1: Secret scan ──────────────────────────────────────────────────
        if self._has_any(
            msg,
            ["secret", "credential", "token", "api key", "clé", "mot de passe",
             "password", "scan secret", "détecte secret", "detecte secret"],
        ):
            plan.update(
                {
                    "intent": "secret_scan",
                    "confidence": 0.92,
                    "selected_agents": ["secret_agent"],
                    "reason": "scan staged files for secrets/credentials",
                }
            )
            return plan

        # ── F3: Commit lint ──────────────────────────────────────────────────
        if self._has_any(
            msg,
            ["lint", "valide", "valider", "validate", "conventional commit",
             "format commit", "message valide", "commit valide"],
        ):
            plan.update(
                {
                    "intent": "commit_lint",
                    "confidence": 0.9,
                    "selected_agents": ["commit_linter_agent"],
                    "reason": "validate commit message against Conventional Commits",
                }
            )
            return plan


        # ── F4: Test impact ──────────────────────────────────────────
        if self._has_any(
            msg,
            ["impact test", "test impact", "fichier test",
             "test file", "couverture", "coverage", "quel test", "which test",
             "tests impactés", "impacted tests"],
        ):
            plan.update(
                {
                    "intent": "test_impact",
                    "confidence": 0.88,
                    "selected_agents": ["test_impact_agent"],
                    "reason": "find test files impacted by staged changes",
                }
            )
            return plan

        # ── F6: Cross-PR conflicts ───────────────────────────────────────────
        if self._has_any(
            msg,
            ["cross pr", "cross-pr", "entre pr", "between pr", "plusieurs pr",
             "multiple pr", "pr parallèle", "conflit pr", "pr conflict"],
        ):
            plan.update(
                {
                    "intent": "cross_pr_conflicts",
                    "confidence": 0.9,
                    "selected_agents": ["cross_pr_agent"],
                    "reason": "detect cross-PR file conflicts",
                }
            )
            return plan
        return plan

    @staticmethod
    def _has_any(text: str, words: list[str]) -> bool:
        return any(w in text for w in words)

    @staticmethod
    def _extract_pr_number(text: str) -> int:
        patterns = [
            r"(?:pr|pull request)\s*#?\s*(\d+)",
            r"#\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return 0

        return 0


git_decision_agent = LCGitDecisionAgent()