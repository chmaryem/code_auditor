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

Architecture v2 (Axe a — multi-agent refactor):
  - Primary  : LLM call (~50 tokens, <0.5s) — understands free-text phrasing
               that fools keyword matching (e.g. "is my commit message
               conventional" vs "generate a commit message").
  - Fallback : the original deterministic regex logic (kept verbatim as
               _decide_regex), used when the LLM call fails, returns an
               unknown intent, or returns low confidence (<0.5).
  Same pattern already proven in lc_chat_decision_agent.py for the Chat's
  top-level intent router.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

_VALID_INTENTS = {
    "git_status", "can_commit", "commit_message", "summarize_changes",
    "branch_readiness", "pr_review", "pr_readiness", "pr_description",
    "pr_fix_guidance",
    "conflict_resolution_dry_run", "secret_scan", "commit_lint",
    "test_impact", "cross_pr_conflicts",
    "repo_overview",
}

_AGENTS_FOR_INTENT = {
    "git_status":                  ["session_agent"],
    "can_commit":                  ["session_agent", "diff_agent"],
    "commit_message":              ["diff_agent"],
    "summarize_changes":           ["diff_agent"],
    "branch_readiness":            ["branch_agent"],
    "pr_review":                   ["pr_agent"],
    "pr_readiness":                ["pr_agent"],
    "pr_description":              ["pr_description_agent"],
    "pr_fix_guidance":             ["pr_agent"],
    "conflict_resolution_dry_run": ["conflict_agent"],
    "secret_scan":                 ["secret_agent"],
    "commit_lint":                 ["commit_linter_agent"],
    "test_impact":                 ["test_impact_agent"],
    "cross_pr_conflicts":          ["cross_pr_agent"],
    "repo_overview":               ["repo_agent"],
}

_GIT_DECISION_PROMPT = """\
You are a routing agent for Code Auditor's Smart Git assistant.
Classify the developer's message and return a JSON routing plan.

Developer message: {message}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "intent": "<one of: git_status|can_commit|commit_message|summarize_changes|branch_readiness|pr_review|pr_readiness|pr_description|pr_fix_guidance|conflict_resolution_dry_run|secret_scan|commit_lint|test_impact|cross_pr_conflicts|repo_overview>",
  "confidence": <0.0 to 1.0>,
  "reason": "<one sentence>",
  "branch_name": "<exact branch name ONLY if the developer names a specific branch, else empty string>"
}}

branch_name guide:
- Fill it ONLY when a specific branch is named, e.g. "is my branch
  'test-analyzer-conflict2' ready to merge?" -> "test-analyzer-conflict2".
- Leave it "" for generic references with no name ("my branch", "the current branch").

Intent guide:
- git_status                  : current session status, accumulated risk/bugs — "what's my status"
- can_commit                   : is it safe to commit right now
- commit_message               : generate a commit message from the current diff
- summarize_changes            : summarize/list current changes or diff
- branch_readiness             : is MY current branch ready to merge into its base (asking for a STATUS)
- pr_review                    : review a specific Pull Request's code (mentions review/analyze + a PR)
- pr_readiness                 : is a specific PR ready/mergeable (mentions merge/ready + a PR) — asking IF it's ready, a STATUS question
- pr_description                : generate a description for a Pull Request
- pr_fix_guidance               : asking HOW TO FIX / what to DO to make a PR or branch mergeable — a SOLUTION/action-plan question,
  not a status question. Trigger words: "solution", "fix", "how do I", "what should I do", "comment corriger",
  "que faire", "comment la merger" (asking for STEPS, not a yes/no readiness check).
  e.g. "what's the solution to merge it", "how do I fix this PR", "what should I do to get it merged",
  "c'est quoi la solution pour qu'elle soit mergée". Contrast: "is it ready to merge" -> pr_readiness (status);
  "how do I make it ready to merge" -> pr_fix_guidance (solution).
- conflict_resolution_dry_run   : check/resolve LOCAL merge conflicts (not between PRs)
- secret_scan                   : scan staged files for secrets/credentials/API keys
- commit_lint                   : validate/lint the commit MESSAGE FORMAT against Conventional Commits
  (NOT the same as commit_message — "is my commit message conventional/valid" is commit_lint,
   "generate/write a commit message" is commit_message)
- test_impact                   : which tests are impacted by the current changes
- cross_pr_conflicts            : conflicts BETWEEN MULTIPLE open PRs (not a single branch's conflicts)
- repo_overview                 : DESCRIBE the repository itself — "décris/describe my repo/dépôt/repository",
  "c'est quoi mon repo", "présente le dépôt", "overview of the repo", general info about the project's GitHub
  repository (name, description, default branch, visibility, language, open PRs). NOT the local working-copy
  status (that's git_status), NOT a specific PR (that's pr_review/pr_readiness).
"""


def _call_git_decision_llm(message: str) -> Dict[str, Any] | None:
    """Calls the LLM cascade for routing JSON. Returns None on any failure —
    the caller falls back to the regex classifier."""
    try:
        from services.llm_factory import invoke_with_fallback
        raw = invoke_with_fallback(
            _GIT_DECISION_PROMPT.format(message=(message or "")[:500]),
            label="git_decision", max_tokens=150,
        )
        if not raw:
            return None
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        return json.loads(raw)
    except Exception as e:
        logger.debug("Git decision LLM failed: %s", e)
        return None


class LCGitDecisionAgent:
    """
    Decision agent for Smart Git.

    Primary path  : LLM call → structured JSON plan (understands free-text).
    Fallback path : deterministic regex rules (v1, kept as safety net when
                    the LLM is unavailable, returns an unknown intent, or
                    returns confidence < 0.5).
    """

    def decide(
        self,
        message: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        msg = (message or "").lower()
        context = context or {}

        llm_plan = _call_git_decision_llm(message)
        if llm_plan and llm_plan.get("intent") in _VALID_INTENTS \
                and float(llm_plan.get("confidence", 0.0)) >= 0.5:
            intent = llm_plan["intent"]
            plan: Dict[str, Any] = {
                "intent": intent,
                "confidence": float(llm_plan.get("confidence", 0.8)),
                "selected_agents": _AGENTS_FOR_INTENT.get(intent, ["session_agent"]),
                "safe_mode": True,
                "needs_confirmation": intent == "conflict_resolution_dry_run",
                "reason": llm_plan.get("reason", "LLM routing"),
                "_routing": "llm",
            }
            # pr_number extraction stays with the proven regex (requires an
            # explicit "pr"/"pull request" mention) — the LLM only classifies
            # intent, avoiding false positives like "bug #123" -> pr_number=123.
            pr_number = self._extract_pr_number(msg)
            if pr_number:
                plan["pr_number"] = pr_number
            branch_name = (llm_plan.get("branch_name") or "").strip()
            if branch_name:
                plan["branch_name"] = branch_name
            return plan

        logger.debug("Git decision LLM skipped/low-confidence — using regex fallback")
        plan = self._decide_regex(msg, context)
        plan["_routing"] = "regex"
        return plan

    # ── Regex fallback (original v1 logic — kept verbatim as a safety net) ───

    def _decide_regex(
        self,
        msg: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
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

        # ── Repo overview? (décris/describe mon repo/dépôt) ─────────────────
        # Placé avant les règles génériques : "décris mon repo" ne doit pas tomber
        # sur le git_status local par défaut.
        if (self._has_any(msg, ["repo", "dépôt", "depot", "repository", "référentiel"])
                and self._has_any(msg, ["décri", "decri", "describe", "présente", "presente",
                                        "overview", "aperçu", "apercu", "c'est quoi", "info"])):
            plan.update({
                "intent": "repo_overview",
                "confidence": 0.8,
                "selected_agents": ["repo_agent"],
                "reason": "developer asks for an overview/description of the repository",
            })
            return plan

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

        # ── F8: PR/branch fix guidance — AVANT readiness (solution vs statut) ─
        # "c'est quoi la solution pour qu'elle soit mergée" / "how do I fix
        # this" demande un PLAN D'ACTION, pas un statut déjà connu — testé
        # avant les règles branch_readiness/pr_readiness par mot-clé "merge".
        # Deux groupes (comme branch_readiness plus bas) car les phrases
        # réelles ne collent jamais à une phrase figée ("la solution pour
        # qu'elle soit mergée" ne contient pas "solution pour merger").
        if self._has_any(
            msg,
            [
                "solution", "comment corriger", "comment faire", "que faire",
                "quoi faire", "comment merger", "comment la merger",
                "comment le merger", "how to fix", "how do i fix",
                "what should i do", "what's the fix", "rendre mergeable",
            ],
        ) and self._has_any(
            msg, ["merger", "merge", "mergeable", "fusionner", "fusion"],
        ):
            plan.update(
                {
                    "intent": "pr_fix_guidance",
                    "confidence": 0.85,
                    "selected_agents": ["pr_agent"],
                    "reason": "developer asks how to fix/make a PR mergeable",
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