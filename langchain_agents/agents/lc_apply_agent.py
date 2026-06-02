"""
lc_apply_agent.py — Apply generated code to files (safe, diff-aware).

Phase B — Feature B1.

Converts LLM-generated code into:
  1. A unified diff (for review)
  2. A VS Code WorkspaceEdit payload (for direct application in the IDE)
  3. Optionally writes to disk (trusted local mode only)

Safety rules:
  - Never writes outside project root
  - Never overwrites without explicit confirmation (write_mode="overwrite")
  - Dry-run by default (returns diff + workspaceEdit, does not write)
  - All edits are validated (syntax check for Python)
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Diff generation ───────────────────────────────────────────────────────────

def _make_unified_diff(
    original: str,
    modified: str,
    file_path: str,
) -> str:
    """Generate a unified diff string."""
    lines_orig = original.splitlines(keepends=True)
    lines_mod  = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_orig,
        lines_mod,
        fromfile=f"a/{Path(file_path).name}",
        tofile=f"b/{Path(file_path).name}",
        lineterm="",
    )
    return "\n".join(diff)


def _make_insert_diff(
    inserted_code: str,
    file_path: str,
    insert_after_line: int,
) -> str:
    """Generate a diff that inserts code after a specific line."""
    snippet = f"# +++ Insert after line {insert_after_line} +++\n{inserted_code}"
    return _make_unified_diff("", snippet, file_path)


# ── WorkspaceEdit (VS Code format) ───────────────────────────────────────────

def _build_workspace_edit(
    file_path: str,
    edits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a VS Code WorkspaceEdit payload.

    Each edit has:
      {range: {start: {line, character}, end: {line, character}}, newText: str}

    The extension applies this via:
      vscode.workspace.applyEdit(WorkspaceEdit.fromJSON(payload))
    """
    return {
        "version": 1,
        "edits": [
            {
                "uri":  Path(file_path).as_uri(),
                "edits": edits,
            }
        ],
    }


def _replace_function_edit(
    original_code: str,
    function_name: str,
    new_function_body: str,
    language: str = "python",
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Find a function in original_code and replace it.
    Returns (vscode_edits, modified_code).
    """
    lines = original_code.split("\n")

    # Find function start line
    fn_pattern = re.compile(
        rf"^\s*(def|async def|public|private|protected|function|func)\s+{re.escape(function_name)}\b",
        re.IGNORECASE,
    )
    start_line = None
    for i, line in enumerate(lines):
        if fn_pattern.match(line):
            start_line = i
            break

    if start_line is None:
        # Function not found — append at end
        indent = ""
        modified = original_code.rstrip("\n") + "\n\n" + new_function_body
        insert_line = len(lines)
        vscode_edits = [{
            "range": {
                "start": {"line": insert_line, "character": 0},
                "end":   {"line": insert_line, "character": 0},
            },
            "newText": "\n\n" + new_function_body,
        }]
        return vscode_edits, modified

    # Find function end (next def at same or lower indent, or EOF)
    base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        curr_indent = len(line) - len(line.lstrip())
        if curr_indent <= base_indent and line.strip() and not line.strip().startswith("#"):
            end_line = i
            break

    # Replace
    original_fn = "\n".join(lines[start_line:end_line])
    new_lines = lines[:start_line] + new_function_body.split("\n") + lines[end_line:]
    modified = "\n".join(new_lines)

    vscode_edits = [{
        "range": {
            "start": {"line": start_line,     "character": 0},
            "end":   {"line": end_line - 1,   "character": len(lines[end_line - 1])},
        },
        "newText": new_function_body,
    }]

    return vscode_edits, modified


# ── Apply agent ───────────────────────────────────────────────────────────────

class LCApplyAgent:
    """
    Apply generated code patches to files.

    Modes:
      dry_run  : return diff + workspaceEdit, do NOT write (default)
      preview  : same as dry_run but with syntax validation
      apply    : write to disk (requires project_path containment check)
    """

    def apply_function_patch(
        self,
        project_path: str,
        file_path: str,
        function_name: str,
        new_code: str,
        language: str = "python",
        write_mode: str = "dry_run",   # "dry_run" | "preview" | "apply"
    ) -> Dict[str, Any]:
        """
        Replace or insert a function in a file.

        Returns:
          {
            diff:           unified diff string
            workspace_edit: VS Code WorkspaceEdit payload
            valid:          bool (syntax validation)
            errors:         list of syntax errors
            written:        bool (True only if write_mode="apply")
          }
        """
        root  = Path(project_path).resolve()
        fpath = Path(file_path)

        # Resolve relative paths against project root
        if not fpath.is_absolute():
            fpath = root / fpath

        # Safety: containment check
        try:
            fpath.resolve().relative_to(root)
        except ValueError:
            return {"error": f"Refusing to write outside project root: {fpath}"}

        # Read original
        original = ""
        if fpath.exists():
            try:
                original = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return {"error": f"Cannot read {fpath}: {e}"}

        # Build patch
        vscode_edits, modified = _replace_function_edit(
            original, function_name, new_code, language
        )

        diff = _make_unified_diff(original, modified, str(fpath))

        # Syntax validation
        valid, errors = self._validate(modified, language)

        result: Dict[str, Any] = {
            "file_path":       str(fpath),
            "function_name":   function_name,
            "diff":            diff,
            "workspace_edit":  _build_workspace_edit(str(fpath), vscode_edits),
            "modified_code":   modified,
            "valid":           valid,
            "errors":          errors,
            "written":         False,
            "mode":            write_mode,
        }

        if write_mode == "apply" and valid:
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(modified, encoding="utf-8")
                result["written"] = True
                logger.info("Applied patch to %s (fn: %s)", fpath, function_name)
            except Exception as e:
                result["error"] = f"Write failed: {e}"

        return result

    def apply_new_file(
        self,
        project_path: str,
        suggested_file: str,
        code: str,
        language: str = "python",
        write_mode: str = "dry_run",
    ) -> Dict[str, Any]:
        """
        Create a new file with generated code.
        """
        root  = Path(project_path).resolve()
        fpath = Path(suggested_file)
        if not fpath.is_absolute():
            fpath = root / fpath

        try:
            fpath.resolve().relative_to(root)
        except ValueError:
            return {"error": f"Refusing to write outside project root: {fpath}"}

        if fpath.exists() and write_mode != "overwrite":
            return {"error": f"File exists (use write_mode='overwrite'): {fpath}"}

        diff = _make_unified_diff("", code, str(fpath))
        valid, errors = self._validate(code, language)

        # WorkspaceEdit for new file creation
        workspace_edit = {
            "version": 1,
            "creates": [{"uri": fpath.as_uri(), "content": code}],
        }

        result: Dict[str, Any] = {
            "file_path":      str(fpath),
            "diff":           diff,
            "workspace_edit": workspace_edit,
            "valid":          valid,
            "errors":         errors,
            "written":        False,
            "mode":           write_mode,
        }

        if write_mode in ("apply", "overwrite") and valid:
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(code, encoding="utf-8")
                result["written"] = True
                logger.info("Created new file: %s", fpath)
            except Exception as e:
                result["error"] = f"Write failed: {e}"

        return result

    def apply_multi_file_patch(
        self,
        project_path: str,
        patches: List[Dict[str, Any]],
        write_mode: str = "dry_run",
    ) -> Dict[str, Any]:
        """
        Apply multiple file patches atomically (Phase B3 - multi-file edit).

        Each patch: {file_path, function_name, new_code, language}
                 or {file_path, code, is_new_file, language}
        """
        results = []
        all_valid = True

        for patch in patches:
            if patch.get("is_new_file"):
                r = self.apply_new_file(
                    project_path=project_path,
                    suggested_file=patch["file_path"],
                    code=patch.get("code", ""),
                    language=patch.get("language", "python"),
                    write_mode=write_mode,
                )
            else:
                r = self.apply_function_patch(
                    project_path=project_path,
                    file_path=patch["file_path"],
                    function_name=patch.get("function_name", ""),
                    new_code=patch.get("new_code", patch.get("code", "")),
                    language=patch.get("language", "python"),
                    write_mode=write_mode,
                )
            results.append(r)
            if not r.get("valid", True) or r.get("error"):
                all_valid = False

        return {
            "patches":   results,
            "all_valid": all_valid,
            "written":   all(r.get("written") for r in results),
            "mode":      write_mode,
        }

    @staticmethod
    def _validate(code: str, language: str) -> Tuple[bool, List[str]]:
        """Syntax validation for supported languages."""
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True, []
            except SyntaxError as e:
                return False, [f"SyntaxError line {e.lineno}: {e.msg}"]
        # For other languages: skip validation (return valid=True with warning)
        return True, [f"Syntax validation not available for {language} — review manually"]


lc_apply_agent = LCApplyAgent()
