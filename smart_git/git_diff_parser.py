import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

def _run_git(args: list, cwd: Path = None) -> Optional[str]:
   
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        return result.stdout if result.returncode == 0 else None
    except FileNotFoundError:
        print("  Git n'est pas installé ou introuvable dans le PATH.")
        return None
    except Exception:
        return None


def _extract_int(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0

def get_changed_files(commit: str = "HEAD", project_path: Path = None) -> List[Dict[str, str]]:
    
    output = _run_git(["diff", "--name-status", f"{commit}~1", commit], cwd=project_path)
    if output is None:
        output = _run_git(["show", "--name-status", "--format=", commit], cwd=project_path)
    if not output:
        return []
    files = []
    for line in output.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            files.append({"path": parts[-1], "status": parts[0][0]})
    return files


def get_staged_files(project_path: Path = None) -> List[Dict[str, str]]:

    output = _run_git(["diff", "--cached", "--name-status"], cwd=project_path)
    if not output:
        return []
    files = []
    for line in output.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            files.append({"path": parts[-1], "status": parts[0][0]})
    return files


def get_diff_content(file_path: str, commit: str = "HEAD", project_path: Path = None) -> str:
    
    output = _run_git(["diff", f"{commit}~1", commit, "--", file_path], cwd=project_path)
    return output or ""


def get_current_commit_hash(project_path: Path = None) -> str:
   
    output = _run_git(["rev-parse", "--short", "HEAD"], cwd=project_path)
    return output.strip() if output else "unknown"


def get_commit_message(commit: str = "HEAD", project_path: Path = None) -> str:
   
    output = _run_git(["log", "-1", "--pretty=%s", commit], cwd=project_path)
    return output.strip() if output else ""


def is_git_repo(project_path: Path) -> bool:
  
    return _run_git(["rev-parse", "--git-dir"], cwd=project_path) is not None

def get_uncommitted_files(project_path: Path = None) -> List[Dict[str, str]]:
   
    unstaged = _run_git(["diff", "HEAD", "--name-status"], cwd=project_path) or ""
    staged   = _run_git(["diff", "--cached", "--name-status"], cwd=project_path) or ""

    seen: Dict[str, Dict] = {}

    def _parse(raw: str, is_staged: bool):
        for line in raw.strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                path   = parts[-1]
                status = parts[0][0]
                if path not in seen:
                    seen[path] = {"path": path, "status": status, "staged": is_staged}
                else:
                    seen[path]["staged"] = seen[path]["staged"] or is_staged

    _parse(unstaged, False)
    _parse(staged, True)
    return list(seen.values())


def get_session_stats(project_path: Path = None) -> Dict:
  
    stat_out = _run_git(["diff", "HEAD", "--shortstat"], cwd=project_path) or ""
    # Format typique : " 3 files changed, 45 insertions(+), 12 deletions(-)"
    files_changed = _extract_int(stat_out, r"(\d+) file")
    lines_added   = _extract_int(stat_out, r"(\d+) insertion")
    lines_removed = _extract_int(stat_out, r"(\d+) deletion")

    last_commit_ts  = get_last_commit_time(project_path)
    minutes_elapsed = int((time.time() - last_commit_ts) / 60) if last_commit_ts else 0
    last_msg        = get_commit_message("HEAD", project_path)

    return {
        "files_changed":        files_changed,
        "lines_added":          lines_added,
        "lines_removed":        lines_removed,
        "minutes_since_commit": minutes_elapsed,
        "last_commit_msg":      last_msg,
    }


def get_last_commit_time(project_path: Path = None) -> Optional[float]:
   
    output = _run_git(["log", "-1", "--format=%ct"], cwd=project_path)
    if output and output.strip().isdigit():
        return float(output.strip())
    return None


def get_file_at_commit(file_path: str, commit: str = "HEAD",
                       project_path: Path = None) -> Optional[str]:
   
    return _run_git(["show", f"{commit}:{file_path}"], cwd=project_path)


def get_merge_base(branch: str, base: str = "main",
                   project_path: Path = None) -> Optional[str]:
    
    output = _run_git(["merge-base", branch, base], cwd=project_path)
    if output is None and base == "main":
        output = _run_git(["merge-base", branch, "master"], cwd=project_path)
    return output.strip() if output else None


def get_branch_commits(branch: str = "HEAD", base: str = "main",
                       project_path: Path = None) -> List[Dict]:
   
    merge_base = get_merge_base(branch, base, project_path)
    if not merge_base:
        return []

    fmt    = "%h\t%s\t%an\t%ci"
    output = _run_git(
        ["log", f"{merge_base}..{branch}", f"--format={fmt}"],
        cwd=project_path,
    )
    if not output:
        return []

    commits = []
    for line in output.strip().splitlines():
        parts = line.split("\t", 3)
        if len(parts) >= 1 and parts[0]:
            commits.append({
                "hash":    parts[0],
                "message": parts[1] if len(parts) > 1 else "",
                "author":  parts[2] if len(parts) > 2 else "",
                "date":    parts[3] if len(parts) > 3 else "",
            })
    return commits


def get_branch_diff_files(branch: str = "HEAD", base: str = "main",
                          project_path: Path = None) -> List[Dict[str, str]]:
   
    merge_base = get_merge_base(branch, base, project_path)
    if merge_base:
        output = _run_git(["diff", f"{merge_base}..{branch}", "--name-status"],
                          cwd=project_path)
    else:
       
        output = _run_git(["diff", f"{base}...{branch}", "--name-status"],
                          cwd=project_path)

    if not output:
        return []

    files = []
    for line in output.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            files.append({"path": parts[-1], "status": parts[0][0]})
    return files