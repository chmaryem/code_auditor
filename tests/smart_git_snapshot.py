"""
smart_git_snapshot.py — Golden-snapshot safety net for the Smart Git refactor.

WHY THIS EXISTS
    The Smart Git module is being refactored towards a genuine multi-agent
    architecture (Phase 0: separate deterministic *tools* from real *agents*,
    drop the fake regex `decide` router and the 10 wrapper "agents", and wire
    the FastAPI endpoints straight to the underlying smart_git functions).

    The hard constraint is: **the behaviour visible to the two front-ends
    (VS Code extension + web dashboard) must not change.** What those
    front-ends actually consume are the STRUCTURED report objects
    (`session_snapshot`, `branch_report`, `conflict_report`,
    `secret_scan_report`, `readiness_report`, ...), returned inside a
    `GitResponse`. The markdown `response` string is secondary (chat only).

    This script captures the CURRENT output of the Smart Git graph for a set
    of deterministic cases, normalises the volatile bits (timings, git SHAs,
    minutes-since-commit, absolute temp paths) and stores them as golden JSON.
    After each refactor step, re-run WITHOUT --update: any drift in the
    structured contract fails loudly.

USAGE
    # from the repo root, with the project venv:
    .venv/Scripts/python tests/smart_git_snapshot.py --update   # (re)generate goldens
    .venv/Scripts/python tests/smart_git_snapshot.py            # verify (exit 1 on drift)

    The "system under test" is the produce() function below. Today it drives
    the LangGraph path (invoke_smart_git -> _to_response). During Phase 0,
    repoint produce() at the new direct endpoint functions and re-run to prove
    the goldens still match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# ── make the project importable regardless of CWD ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

SNAP_DIR = PROJECT_ROOT / "tests" / "snapshots" / "smart_git"

# ── deterministic git fixture ─────────────────────────────────────────────────
_FIXED_ENV = {
    **os.environ,
    "GIT_AUTHOR_DATE": "2020-01-01T00:00:00 +0000",
    "GIT_COMMITTER_DATE": "2020-01-01T00:00:00 +0000",
    "GIT_AUTHOR_NAME": "T",
    "GIT_AUTHOR_EMAIL": "t@t.com",
    "GIT_COMMITTER_NAME": "T",
    "GIT_COMMITTER_EMAIL": "t@t.com",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, env=_FIXED_ENV)


def build_fixture_repo() -> Path:
    """A reproducible repo: 1 base commit on main, a feature branch with a
    non-conventional commit ('go'), and a staged file carrying a fake secret."""
    repo = Path(tempfile.mkdtemp(prefix="sg_fix_"))
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "T")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "checkout", "-b", "feature/x")
    (repo / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "go")
    (repo / "config.py").write_text(
        'API_KEY = "sk-liveAAAABBBBCCCCDDDDEEEEFFFF11112222"\n'
    )
    _git(repo, "add", "config.py")
    return repo


# ── the cases (what the 8 graph-backed endpoints send today) ──────────────────
def build_cases(repo: Path) -> Dict[str, Dict[str, Any]]:
    p = str(repo)
    # `_repo` is consumed by endpoint runners; stripped before the graph path.
    cases = {
        # local, deterministic — the 4 endpoints that stay on-device
        "git_status": dict(message="git status", project_path=p, session_id="snap"),
        "branch_readiness": dict(
            message="is branch feature/x ready to merge into main",
            project_path=p, branch="feature/x", base="main", session_id="snap"),
        "conflicts_dry_run": dict(
            message="resolve conflicts dry run", project_path=p, session_id="snap"),
        "secret_scan": dict(
            message="scan staged for secrets", project_path=p, session_id="snap"),
        # GitHub endpoints — deterministic validation-error path (no network).
        # Produced entirely by their endpoint runners (see ENDPOINT_RUNNERS).
        # pr/description is intentionally omitted: its local success path calls
        # an LLM, so it is verified manually rather than snapshotted.
        "pr_review_missing": dict(session_id="snap"),
        "pr_readiness_missing": dict(session_id="snap"),
        "cross_pr_missing": dict(session_id="snap"),
    }
    for c in cases.values():
        c["_repo"] = repo
    return cases


# ── normalisation of volatile fields ──────────────────────────────────────────
_HEX40 = re.compile(r"\b[0-9a-f]{7,40}\b")
_VOLATILE_ZERO = {"minutes_since_commit", "time_multiplier",
                  "elapsed", "elapsed_seconds"}
_SHA_KEYS = {"hash", "merge_base_hash"}


def normalize(obj: Any, repo: str) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _VOLATILE_ZERO:
                out[k] = 0
            elif k in _SHA_KEYS and isinstance(v, str):
                out[k] = "<SHA>"
            elif k == "date" and isinstance(v, str):
                out[k] = "<DATE>"
            else:
                out[k] = normalize(v, repo)
        return out
    if isinstance(obj, list):
        return [normalize(v, repo) for v in obj]
    if isinstance(obj, str):
        s = obj.replace(repo, "<REPO>").replace(repo.replace("\\", "/"), "<REPO>")
        s = _HEX40.sub("<SHA>", s)
        return s
    return obj


# ── system under test ─────────────────────────────────────────────────────────
# As each endpoint is migrated off the LangGraph (Phase 0), we flip its case
# from the default graph path to a runner that calls the REAL endpoint function
# — so the safety net tests exactly what ships to the front-ends, not the graph.

def _local_principal():
    # email == local@localhost → the _pg_save_git_report side-effect is skipped,
    # so no DB is required to exercise the auth'd endpoints.
    from auth.security import Principal
    return Principal(id="local", email="local@localhost", role="Developer")


async def _run_status_endpoint(repo: Path) -> Dict[str, Any]:
    from api.git_router import git_status, GitSessionRequest
    resp = await git_status(GitSessionRequest(project_path=str(repo), session_id="snap"))
    return resp.model_dump()


async def _run_branch_endpoint(repo: Path) -> Dict[str, Any]:
    from api.git_router import git_branch, GitBranchRequest
    resp = await git_branch(
        GitBranchRequest(project_path=str(repo), branch="feature/x",
                         base="main", session_id="snap"),
        user=_local_principal(),
    )
    return resp.model_dump()


async def _run_conflicts_endpoint(repo: Path) -> Dict[str, Any]:
    from api.git_router import git_conflicts, GitConflictRequest
    resp = await git_conflicts(GitConflictRequest(project_path=str(repo), session_id="snap"))
    return resp.model_dump()


async def _run_secret_scan_endpoint(repo: Path) -> Dict[str, Any]:
    from api.git_router import git_secret_scan, GitSecretScanRequest
    resp = await git_secret_scan(
        GitSecretScanRequest(project_path=str(repo), session_id="snap"),
        user=_local_principal(),
    )
    return resp.model_dump()


# GitHub endpoints: the success path needs a real token/network, so the safety
# net covers their DETERMINISTIC validation-error path (empty owner/repo → no
# network call). The success path is validated manually in the dashboard.

async def _run_pr_review_missing(_repo: Path) -> Dict[str, Any]:
    from api.git_router import git_pr_review, GitPRRequest
    resp = await git_pr_review(
        GitPRRequest(owner="", repo="", pr_number=0, session_id="snap"),
        user=_local_principal(),
    )
    return resp.model_dump()


async def _run_pr_readiness_missing(_repo: Path) -> Dict[str, Any]:
    from api.git_router import git_pr_readiness, GitPRRequest
    resp = await git_pr_readiness(
        GitPRRequest(owner="", repo="", pr_number=0, session_id="snap"),
        user=_local_principal(),
    )
    return resp.model_dump()


async def _run_cross_pr_missing(_repo: Path) -> Dict[str, Any]:
    from api.git_router import git_cross_pr_conflicts, GitCrossPRRequest
    resp = await git_cross_pr_conflicts(
        GitCrossPRRequest(owner="", repo="", base="main", pr_number=0, session_id="snap"),
        user=_local_principal(),
    )
    return resp.model_dump()


ENDPOINT_RUNNERS = {
    # migrated in Phase 0 — verified against the live endpoint
    "git_status": _run_status_endpoint,
    "branch_readiness": _run_branch_endpoint,
    "conflicts_dry_run": _run_conflicts_endpoint,
    "secret_scan": _run_secret_scan_endpoint,
    # GitHub endpoints — deterministic validation-error path
    "pr_review_missing": _run_pr_review_missing,
    "pr_readiness_missing": _run_pr_readiness_missing,
    "cross_pr_missing": _run_cross_pr_missing,
}


def produce(name: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the system under test: every Smart Git endpoint is exercised through
    its real FastAPI handler (all 8 endpoints are migrated off the LangGraph)."""
    runner = ENDPOINT_RUNNERS.get(name)
    if runner is None:
        raise RuntimeError(f"no endpoint runner registered for case '{name}'")
    return asyncio.run(runner(case.get("_repo")))


# ── driver ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="(re)generate golden snapshots instead of verifying")
    args = ap.parse_args()

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    repo = build_fixture_repo()
    cases = build_cases(repo)

    failures = []
    for name, case in cases.items():
        try:
            captured = normalize(produce(name, case), str(repo))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append(f"{name}: RAISED {type(e).__name__}: {e}")
            continue

        golden_path = SNAP_DIR / f"{name}.json"
        rendered = json.dumps(captured, indent=2, ensure_ascii=False, sort_keys=True)

        if args.update:
            golden_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"[updated] {name}")
            continue

        if not golden_path.exists():
            failures.append(f"{name}: no golden (run with --update first)")
            continue

        expected = golden_path.read_text(encoding="utf-8").strip()
        if rendered.strip() != expected:
            failures.append(f"{name}: DRIFT vs golden")
            _print_first_diff(name, expected, rendered)
        else:
            print(f"[ok] {name}")

    if args.update:
        print(f"\nGoldens written to {SNAP_DIR}")
        return 0

    if failures:
        print("\n=== SNAPSHOT FAILURES ===")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll Smart Git snapshots match. Contract preserved.")
    return 0


def _print_first_diff(name: str, expected: str, got: str) -> None:
    exp_lines = expected.splitlines()
    got_lines = got.splitlines()
    for i, (a, b) in enumerate(zip(exp_lines, got_lines)):
        if a != b:
            print(f"   {name} first diff @ line {i + 1}:")
            print(f"     - golden: {a[:160]}")
            print(f"     + now   : {b[:160]}")
            return
    if len(exp_lines) != len(got_lines):
        print(f"   {name}: length differs "
              f"(golden={len(exp_lines)} now={len(got_lines)})")


if __name__ == "__main__":
    raise SystemExit(main())
