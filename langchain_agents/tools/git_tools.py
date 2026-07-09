"""
git_tools.py — LangChain @tool layer for Smart Git (pilier TOOLS).

Each tool is the single public capability of the Smart Git domain layer
(smart_git/*.py). Tools delegate to the pure domain functions and own the
dataclass -> dict serialization (previously duplicated in the thin
lc_git_*_agent.py wrappers). The role-based agents (WorkingCopyAgent,
PRAgent, ConflictAgent) declare a subset of these as their `tools` and
invoke them — that is what makes them genuine agents with a toolbox.

Convention mirrors code_tools.py / git_session_tools.py: `@tool`, typed
args, docstring (LLM-readable), dict return. Sync tools for local
deterministic operations; async tools for GitHub-backed PR operations.

IMPORTANT — output parity: each tool returns the EXACT same dict shape the
corresponding lc_git_*_agent produced, so switching the dispatcher/graph to
these tools is byte-identical (verified before deleting any wrapper).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


# ══════════════════════════════════════════════════════════════════════════════
# WorkingCopyAgent tools — local, deterministic (no LLM, no network)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def tool_session_status(project_path: str) -> Dict[str, Any]:
    """Current git session risk snapshot: level, score, files at risk, unanalyzed files.

    Args:
        project_path: Absolute or relative path to the project git root.

    Returns:
        Dict with success, score, level, minutes_since_commit, time_multiplier,
        total_critical, total_high, total_bugs, files_at_risk[], files_unanalyzed[], stats.
    """
    try:
        from smart_git.git_session_tracker import GitSessionTracker
    except Exception as e:
        return {"success": False, "error": f"Cannot import GitSessionTracker: {e}"}

    project = Path(project_path).resolve()
    cache_db = project.parent / "code_auditor" / "data" / "cache" / "analysis_cache.db"

    try:
        tracker = GitSessionTracker(project_path=project, cache_db=cache_db)
        snapshot = tracker.force_check()
        if not snapshot:
            return {"success": False, "error": "Unable to calculate git session status"}

        return {
            "success": True,
            "score": snapshot.score,
            "level": snapshot.level,
            "minutes_since_commit": snapshot.minutes_since_commit,
            "time_multiplier": snapshot.time_multiplier,
            "total_critical": snapshot.total_critical,
            "total_high": snapshot.total_high,
            "total_bugs": snapshot.total_bugs,
            "files_at_risk": [
                {
                    "path": fr.path, "status": fr.status, "staged": fr.staged,
                    "critical": fr.bugs_critical, "high": fr.bugs_high,
                    "medium": fr.bugs_medium, "low": fr.bugs_low, "score": fr.score,
                    "has_analysis": fr.has_analysis, "max_severity": fr.max_severity,
                }
                for fr in snapshot.files_at_risk
            ],
            "files_unanalyzed": snapshot.files_unanalyzed,
            "stats": snapshot.stats,
        }
    except Exception as e:
        return {"success": False, "error": f"Git session status failed: {e}"}


@tool
def tool_changes(project_path: str) -> Dict[str, Any]:
    """Summarize current working-copy changes: staged/uncommitted files and session stats.

    Args:
        project_path: Path to the project git root.

    Returns:
        Dict with success, uncommitted_files[], staged_files[], session_stats.
    """
    try:
        from smart_git.git_diff_parser import (
            get_uncommitted_files, get_staged_files, get_session_stats, is_git_repo,
        )
    except Exception as e:
        return {"success": False, "error": f"Cannot import git diff parser: {e}"}

    project = Path(project_path).resolve()
    try:
        if not is_git_repo(project):
            return {"success": False, "error": f"Not a git repository: {project}"}
        return {
            "success": True,
            "uncommitted_files": get_uncommitted_files(project),
            "staged_files": get_staged_files(project),
            "session_stats": get_session_stats(project),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to read git changes: {e}"}


@tool
def tool_commit_message(project_path: str) -> Dict[str, Any]:
    """Generate a commit message from the current staged diff (LLM).

    Args:
        project_path: Path to the project git root.

    Returns:
        Dict with success, commit_message, error.
    """
    try:
        from smart_git.git_commit_msg import generate_commit_message
    except Exception as e:
        return {"success": False, "commit_message": "",
                "error": f"Cannot import commit message generator: {e}"}

    project = Path(project_path).resolve()
    try:
        message = generate_commit_message(project)
        return {
            "success": bool(message),
            "commit_message": message or "",
            "error": "" if message else "No staged diff found",
        }
    except Exception as e:
        return {"success": False, "commit_message": "",
                "error": f"Commit message generation failed: {e}"}


@tool
def tool_branch_readiness(project_path: str, branch: str = "HEAD", base: str = "main") -> Dict[str, Any]:
    """Analyze whether a LOCAL branch is ready to merge into its base (risk score + verdict).

    Args:
        project_path: Path to the project git root.
        branch: Feature branch to analyze (default HEAD).
        base: Base branch to compare against (default main).

    Returns:
        Dict with success, branch, base, merge_base_hash, commits, files[],
        conflict_risks[], total_score, verdict, recommendation, total_critical,
        total_high, files_clean_count, files_with_issues_count.
    """
    try:
        from smart_git.git_branch_analyzer import GitBranchAnalyzer
    except Exception as e:
        return {"success": False, "error": f"Cannot import GitBranchAnalyzer: {e}"}

    from dataclasses import asdict
    project = Path(project_path).resolve()
    cache_db = project.parent / "code_auditor" / "data" / "cache" / "analysis_cache.db"
    try:
        analyzer = GitBranchAnalyzer(project_path=project, cache_db=cache_db, orchestrator=None)
        report = analyzer.analyze(branch=branch or "HEAD", base=base or "main")
        return {
            "success": True,
            "branch": report.branch, "base": report.base,
            "merge_base_hash": report.merge_base_hash,
            "commits": report.commits,
            "files": [asdict(f) for f in report.files],
            "conflict_risks": report.conflict_risks,
            "total_score": report.total_score,
            "verdict": report.verdict,
            "recommendation": report.recommendation,
            "total_critical": report.total_critical,
            "total_high": report.total_high,
            "files_clean_count": len(report.files_clean),
            "files_with_issues_count": len(report.files_with_issues),
        }
    except Exception as e:
        return {"success": False, "error": f"Branch analysis failed: {e}"}


@tool
def tool_secret_scan(project_path: str) -> Dict[str, Any]:
    """Scan staged files for secrets/credentials/API keys (regex, no LLM).

    Args:
        project_path: Path to the project git root.

    Returns:
        Dict with findings[], scanned_files, blocked, has_secrets, critical_count, error.
    """
    from smart_git.git_secret_scanner import scan_staged_secrets
    try:
        report = scan_staged_secrets(Path(project_path))
        return {
            "findings": [
                {
                    "file_path": f.file_path, "line_number": f.line_number,
                    "secret_type": f.secret_type, "masked_text": f.masked_text,
                    "severity": f.severity, "context": f.context,
                }
                for f in report.findings
            ],
            "scanned_files": report.scanned_files,
            "blocked": report.blocked,
            "has_secrets": report.has_secrets,
            "critical_count": report.critical_count,
            "error": report.error,
        }
    except Exception as e:
        return {"error": str(e), "findings": [], "scanned_files": 0, "blocked": False}


@tool
def tool_commit_lint(project_path: str, commit_message: str = "") -> Dict[str, Any]:
    """Validate a commit message against Conventional Commits (no LLM).

    Args:
        project_path: Path to the project git root (falls back to COMMIT_EDITMSG).
        commit_message: Message to validate; if empty, reads the staged commit message.

    Returns:
        Dict with original_message, is_valid, has_warnings, score, suggested_message,
        violations[], error.
    """
    from smart_git.git_commit_linter import lint_commit_message, lint_staged_commit
    try:
        msg = (commit_message or "").strip()
        report = lint_commit_message(msg) if msg else lint_staged_commit(Path(project_path))
        return {
            "original_message": report.original_message,
            "is_valid": report.is_valid,
            "has_warnings": report.has_warnings,
            "score": report.score,
            "suggested_message": report.suggested_message,
            "violations": [
                {"rule": v.rule, "severity": v.severity, "message": v.message,
                 "suggestion": v.suggestion}
                for v in report.violations
            ],
            "error": report.error,
        }
    except Exception as e:
        return {"error": str(e), "is_valid": False, "score": 0, "violations": []}


@tool
def tool_test_impact(project_path: str, changed_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """Find test files impacted by the current changes (naming conventions + grep, no LLM).

    Args:
        project_path: Path to the project git root.
        changed_files: Optional list of changed source files; else uses staged diff.

    Returns:
        Dict with total_files, covered_files, uncovered_files, coverage_ratio,
        has_gaps, all_test_files[], impacts[], error.
    """
    from smart_git.git_test_impact import analyze_test_impact_staged, analyze_test_impact_files
    project = Path(project_path)
    try:
        report = (analyze_test_impact_files(changed_files, project)
                  if changed_files else analyze_test_impact_staged(project))
        return {
            "total_files": report.total_files,
            "covered_files": report.covered_files,
            "uncovered_files": report.uncovered_files,
            "coverage_ratio": round(report.coverage_ratio, 2),
            "has_gaps": report.has_gaps,
            "all_test_files": report.all_test_files,
            "impacts": [
                {"source_file": i.source_file, "test_files": i.test_files,
                 "missing_tests": i.missing_tests, "discovery_method": i.discovery_method}
                for i in report.impacts
            ],
            "error": report.error,
        }
    except Exception as e:
        return {"error": str(e), "impacts": [], "total_files": 0, "covered_files": 0,
                "uncovered_files": 0, "all_test_files": []}


# ══════════════════════════════════════════════════════════════════════════════
# ConflictAgent tool — local conflict detection (safe dry-run)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def tool_conflict_scan(project_path: str) -> Dict[str, Any]:
    """Detect files with unresolved merge conflicts (safe: no write, no push).

    Args:
        project_path: Path to the project git root.

    Returns:
        Dict with success, has_conflicts, conflict_files[], message, dry_run, next_step.
    """
    try:
        from smart_git.git_diff_parser import detect_conflict_files
    except Exception as e:
        return {"success": False, "error": f"Cannot import conflict detection: {e}",
                "has_conflicts": False, "conflict_files": []}

    try:
        files = detect_conflict_files(Path(project_path).resolve())
    except Exception as e:
        return {"success": False, "error": f"Conflict detection failed: {e}",
                "has_conflicts": False, "conflict_files": []}

    if not files:
        return {"success": True, "has_conflicts": False, "conflict_files": [],
                "message": "No conflict files detected", "dry_run": True}
    return {
        "success": True, "has_conflicts": True, "conflict_files": files,
        "message": "Conflicts detected. Safe dry-run mode is active. No file was modified.",
        "dry_run": True,
        "next_step": ("Use conflict_resolution_dry_run's follow-up (Smart Git PR resolve) "
                      "to preview a resolution before applying changes."),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRAgent tools — GitHub-backed (async)
# ══════════════════════════════════════════════════════════════════════════════

@tool
async def tool_pr_review(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """Review a Pull Request's code and post structured findings (critical/high/medium).

    Args:
        owner: GitHub repository owner.
        repo: GitHub repository name.
        pr_number: Pull request number.

    Returns:
        The review result dict (success, body, score, critical/high/medium counts, verdict...).
    """
    if not owner or not repo or not pr_number:
        return {"success": False, "error": "Missing owner/repo/pr_number"}
    try:
        from smart_git.pr_review_agent import review_pr
    except Exception as e:
        return {"success": False, "error": f"Cannot import PR review agent: {e}"}
    try:
        result = await review_pr(owner, repo, pr_number)
        return result if isinstance(result, dict) else {
            "success": False, "error": "PR review returned non-dict result"}
    except Exception as e:
        return {"success": False, "error": f"PR review failed: {e}"}


@tool
async def tool_pr_readiness(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """Check whether a PR is ready to merge (mergeable, CI/CD checks, reviews, security).

    Args:
        owner: GitHub repository owner.
        repo: GitHub repository name.
        pr_number: Pull request number.

    Returns:
        The readiness result dict (success, ready, mergeable, ci_pass, reviews_approved,
        score, body, security_context...).
    """
    if not owner or not repo or not pr_number:
        return {"success": False, "error": "Missing owner/repo/pr_number"}
    try:
        from smart_git.merge_automation_agent import check_merge_readiness
    except Exception as e:
        return {"success": False, "error": f"Cannot import merge readiness agent: {e}"}
    try:
        result = await check_merge_readiness(owner, repo, pr_number)
        return result if isinstance(result, dict) else {
            "success": False, "error": "PR readiness returned non-dict result"}
    except Exception as e:
        return {"success": False, "error": f"PR readiness failed: {e}"}


@tool
def tool_pr_description(project_path: str, owner: str = "", repo: str = "",
                       pr_number: int = 0, branch: str = "HEAD", base: str = "main",
                       pr_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate a professional PR description (GitHub mode if owner/repo/pr, else local git log).

    Args:
        project_path: Path to the project git root.
        owner: GitHub owner (optional — enables GitHub mode).
        repo: GitHub repo (optional).
        pr_number: PR number (optional).
        branch: Feature branch (local mode).
        base: Base branch (local mode).
        pr_report: Optional prior PR data to enrich the description.

    Returns:
        Dict with title, description, commits_used, files_changed, additions,
        deletions, from_cache, error.
    """
    from smart_git.git_pr_description import (
        generate_pr_description_local, generate_pr_description_github,
    )
    try:
        if owner and repo and pr_number:
            from services.code_mode_client import github
            report = generate_pr_description_github(
                owner, repo, pr_number, github, pr_report or {})
        else:
            report = generate_pr_description_local(Path(project_path), branch, base)
        return {
            "title": report.title, "description": report.description,
            "commits_used": report.commits_used, "files_changed": report.files_changed,
            "additions": report.additions, "deletions": report.deletions,
            "from_cache": report.from_cache, "error": report.error,
        }
    except Exception as e:
        return {"error": str(e), "title": "", "description": "", "from_cache": False}


@tool
def tool_cross_pr(owner: str, repo: str, pr_number: int = 0, base: str = "main") -> Dict[str, Any]:
    """Detect files modified simultaneously across multiple open PRs (conflict risk).

    Args:
        owner: GitHub repository owner.
        repo: GitHub repository name.
        pr_number: Current PR to exclude (optional).
        base: Base branch (default main).

    Returns:
        Dict with repo, total_open_prs, has_conflicts, high_risk_count,
        medium_risk_count, conflicts[], pr_summaries[], error.
    """
    if not owner or not repo:
        return {"error": "owner et repo sont requis pour la détection cross-PR.",
                "conflicts": [], "total_open_prs": 0}
    try:
        from services.code_mode_client import github
        from smart_git.git_cross_pr import detect_cross_pr_conflicts
        report = detect_cross_pr_conflicts(owner, repo, github, base, pr_number or None)
        return {
            "repo": report.repo, "total_open_prs": report.total_open_prs,
            "has_conflicts": report.has_conflicts,
            "high_risk_count": report.high_risk_count,
            "medium_risk_count": report.medium_risk_count,
            "conflicts": [
                {"file_path": c.file_path, "pr_numbers": c.pr_numbers,
                 "pr_titles": c.pr_titles, "risk_level": c.risk_level, "note": c.note}
                for c in report.conflicts
            ],
            "pr_summaries": report.pr_summaries,
            "error": report.error,
        }
    except Exception as e:
        return {"error": str(e), "conflicts": [], "total_open_prs": 0}


# Convenience registry (inspectable overview of the Smart Git toolbox)
GIT_TOOLS = [
    tool_session_status, tool_changes, tool_commit_message, tool_branch_readiness,
    tool_secret_scan, tool_commit_lint, tool_test_impact, tool_conflict_scan,
    tool_pr_review, tool_pr_readiness, tool_pr_description, tool_cross_pr,
]
