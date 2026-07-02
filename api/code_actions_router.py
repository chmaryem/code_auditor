from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

code_actions_router = APIRouter(prefix="/code-actions", tags=["CodeActions"])

# Limit concurrent LLM calls to avoid provider rate limits (Groq: 30 RPM)
_llm_semaphore = asyncio.Semaphore(3)


# ── Pydantic models ──────────────────────────────────────────────────────────

class TextRange(BaseModel):
    start_line: int = 0
    start_char: int = 0
    end_line:   int = 0
    end_char:   int = 0


class DiagnosticContext(BaseModel):
    message:  str = ""
    severity: int = 1
    code:     str = ""
    line:     Optional[int] = None


class CodeActionsRequest(BaseModel):
    """VS Code CodeActionContext — mirrors vscode.CodeActionContext."""
    file_path:    str  = Field(...,      description="Absolute file path")
    project_path: str  = Field(".",      description="Project root")
    language:     str  = Field("",       description="Language ID (python, typescript, ...)")
    selected_code: str = Field("",       description="Currently selected text")
    range:        Optional[TextRange] = Field(None, description="Selection range")
    diagnostics:  List[DiagnosticContext] = Field(
        default_factory=list,
        description="Active diagnostics in the selection (for quick fix)"
    )
    trigger_kind: int = Field(
        1,
        description="1=Automatic (cursor), 2=Invoke (explicit right-click)"
    )
    surrounding_code: str = Field("", description="~10 lines of context around selection")


class CodeActionItem(BaseModel):
    """Maps to vscode.CodeAction."""
    title:       str  = ""
    kind:        str  = ""   # "quickfix" | "refactor" | "source" | "refactor.extract"
    is_preferred: bool = False
    # The edit to apply
    edit_type:   str  = ""   # "replace" | "insert" | "workspace_edit"
    edit_range:  Optional[TextRange] = None
    new_text:    str  = ""
    # For multi-file edits
    workspace_edit: Dict[str, Any] = Field(default_factory=dict)
    # Diagnostic this fixes (if quickfix)
    fixes_diagnostic: str = ""
    # Command to run after applying
    command:     str  = ""
    command_args: List[Any] = Field(default_factory=list)


class CodeActionsResponse(BaseModel):
    actions:     List[CodeActionItem] = Field(default_factory=list)
    total:       int   = 0
    elapsed_ms:  int   = 0
    from_cache:  bool  = False


class QuickFixRequest(BaseModel):
    file_path:    str = Field(..., description="Absolute file path")
    project_path: str = Field(".", description="Project root")
    language:     str = Field("",  description="Language ID")
    diagnostic_message: str = Field(..., description="The diagnostic message to fix")
    diagnostic_code:    str = Field("",  description="Diagnostic code/rule")
    line:         int = Field(0,   description="Line number of the diagnostic (1-indexed)")
    surrounding_code: str = Field("", description="Code context around the diagnostic")


class RefactorRequest(BaseModel):
    file_path:    str = Field(..., description="Absolute file path")
    project_path: str = Field(".", description="Project root")
    language:     str = Field("",  description="Language ID")
    selected_code: str = Field(..., description="Code to refactor")
    refactor_kind: str = Field(
        "general",
        description="'extract_function'|'extract_variable'|'rename'|'add_types'|'add_docstring'|'general'"
    )
    surrounding_code: str = Field("", description="Code context")
    range: Optional[TextRange] = None


# ── Prompt templates ─────────────────────────────────────────────────────────

_QUICK_FIX_PROMPT = """\
You are an expert {language} developer. Generate a minimal, correct fix for this diagnostic.

File: {file_path}
Language: {language}
Diagnostic: {message} (rule: {code}, line: {line})

Code context:
```{language}
{code_context}
```

Instructions:
1. Output ONLY the fixed version of the relevant code line(s)
2. Keep changes minimal — fix only what's broken
3. No explanation, no markdown fences, just the corrected code
4. If the fix requires multiple lines, output all of them

Fixed code:"""

_CODE_ACTIONS_PROMPT = """\
You are an expert {language} code reviewer. Analyze this code selection and suggest actionable improvements.

File: {file_path}
Language: {language}
Selected code:
```{language}
{selected_code}
```

Surrounding context:
```{language}
{surrounding_code}
```

Active diagnostics: {diagnostics_summary}

Generate 2-4 specific, actionable code actions. For each action, output:
ACTION: <title>
KIND: quickfix|refactor|source
PREFERRED: true|false
NEW_CODE:
```
<the replacement code>
```
END_ACTION

Focus on: type annotations, null checks, security issues, naming, simplification."""

_REFACTOR_PROMPT = """\
You are an expert {language} refactoring assistant.

File: {file_path}
Refactor kind: {refactor_kind}
Language: {language}

Code to refactor:
```{language}
{selected_code}
```

Context:
```{language}
{surrounding_code}
```

Generate the refactored code. Output ONLY:
1. The refactored replacement for the selected code
2. If extraction creates a new function/variable, include it just before the selection site
3. No explanation, no markdown fences

Refactored code:"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_llm():
    try:
        from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
        return _build_llm_with_fallback()
    except Exception:
        return None


def _invoke_llm(prompt: str, max_tokens: int = 512) -> str:
    """Synchronous LLM call with fallback."""
    try:
        llm = _get_llm()
        if llm:
            from langchain_core.messages import HumanMessage
            result = llm.invoke(
                [HumanMessage(content=prompt)],
                config={"max_tokens": max_tokens},
            )
            return getattr(result, "content", str(result)).strip()
    except Exception as e:
        logger.warning("code_actions LLM failed: %s", e)
    try:
        from services.llm_factory import invoke_with_fallback
        return invoke_with_fallback(prompt, label="code_actions", max_tokens=max_tokens)
    except Exception as e:
        logger.error("code_actions fallback LLM failed: %s", e)
    return ""


def _parse_code_actions_from_llm(text: str, file_path: str, sel_range: Optional[TextRange]) -> List[CodeActionItem]:
    """Parse the structured ACTION blocks from the LLM output."""
    import re
    actions = []

    # Split on ACTION: markers
    blocks = re.split(r"\nACTION:", "\n" + text)
    for block in blocks[1:]:  # skip first empty split
        lines = block.strip().split("\n")
        if not lines:
            continue

        title = lines[0].strip()
        kind  = "refactor"
        preferred = False
        new_code_lines = []
        in_code = False

        for line in lines[1:]:
            if line.startswith("KIND:"):
                kind = line.split(":", 1)[1].strip().lower()
            elif line.startswith("PREFERRED:"):
                preferred = "true" in line.lower()
            elif line.strip() in ("```", "```python", "```typescript", "```javascript") or "NEW_CODE:" in line:
                in_code = not in_code if line.strip().startswith("```") else True
            elif line.strip() == "END_ACTION":
                break
            elif in_code:
                new_code_lines.append(line)

        new_text = "\n".join(new_code_lines).strip()
        if title and new_text:
            actions.append(CodeActionItem(
                title        = title,
                kind         = kind,
                is_preferred = preferred,
                edit_type    = "replace",
                edit_range   = sel_range,
                new_text     = new_text,
                fixes_diagnostic = "",
            ))

    return actions[:4]  # cap at 4 actions


def _heuristic_actions(file_path: str, language: str, selected_code: str) -> List[CodeActionItem]:
    """
    Fast heuristic actions (no LLM) triggered automatically when cursor moves.
    Only activates when trigger_kind=1 (Automatic).
    """
    actions = []
    fp = Path(file_path)

    # Python-specific heuristics
    if language == "python" and selected_code:
        # Missing type hints on function
        import re
        if re.search(r"def \w+\(", selected_code) and "->" not in selected_code:
            actions.append(CodeActionItem(
                title        = "Add type annotations",
                kind         = "refactor",
                is_preferred = False,
                edit_type    = "",  # requires LLM — defer
                command      = "codeAuditor.addTypeAnnotations",
                command_args = [file_path],
            ))

        # Missing docstring
        if re.search(r"def \w+\(", selected_code) and '"""' not in selected_code:
            actions.append(CodeActionItem(
                title    = "Generate docstring",
                kind     = "source",
                edit_type = "",
                command  = "codeAuditor.generateDocstring",
                command_args = [file_path],
            ))

    # Generic: extract selection as function/variable
    if selected_code and len(selected_code.split("\n")) >= 3:
        actions.append(CodeActionItem(
            title    = "Extract as function",
            kind     = "refactor.extract",
            edit_type = "",
            command  = "codeAuditor.extractFunction",
            command_args = [file_path],
        ))

    return actions


# ── Endpoints ────────────────────────────────────────────────────────────────

@code_actions_router.post(
    "",
    response_model=CodeActionsResponse,
    summary="Generate Code Actions for a selection (lightbulb menu)",
)
async def get_code_actions(req: CodeActionsRequest):
    """
    Generate VS Code CodeActions for the currently selected code.

    Two modes:
    - trigger_kind=1 (Automatic): returns fast heuristic actions only (no LLM, < 50ms)
    - trigger_kind=2 (Invoke):    calls LLM for rich, context-aware actions (< 2s)

    The VS Code extension calls this:
    - On cursor move (automatic) → fast heuristics
    - When user clicks the lightbulb / right-click (invoke) → LLM actions
    """
    t0 = time.time()

    file_path = req.file_path
    if not Path(file_path).exists():
        raise HTTPException(404, f"File not found: {file_path}")

    language = req.language or "python"
    actions: List[CodeActionItem] = []

    # ── Automatic mode: heuristic only (no LLM) ───────────────────────────────
    if req.trigger_kind == 1:
        actions = _heuristic_actions(file_path, language, req.selected_code)
        return CodeActionsResponse(
            actions    = actions,
            total      = len(actions),
            elapsed_ms = round((time.time() - t0) * 1000),
            from_cache = True,
        )

    # ── Invoke mode: LLM-powered ───────────────────────────────────────────────
    diag_summary = "; ".join(d.message for d in req.diagnostics[:3]) or "none"

    prompt = _CODE_ACTIONS_PROMPT.format(
        file_path         = Path(file_path).name,
        language          = language,
        selected_code     = req.selected_code[:1200],
        surrounding_code  = req.surrounding_code[:600],
        diagnostics_summary = diag_summary,
    )

    raw = await asyncio.to_thread(_invoke_llm, prompt, 600)

    if raw:
        actions = _parse_code_actions_from_llm(raw, file_path, req.range)

    # Prepend heuristic actions (they're fast and always useful)
    heuristic = _heuristic_actions(file_path, language, req.selected_code)
    # Avoid duplicates
    existing_titles = {a.title for a in actions}
    actions = [h for h in heuristic if h.title not in existing_titles] + actions

    return CodeActionsResponse(
        actions    = actions[:5],
        total      = len(actions),
        elapsed_ms = round((time.time() - t0) * 1000),
    )


@code_actions_router.post(
    "/quick-fix",
    response_model=CodeActionsResponse,
    summary="Generate a quick fix for a specific diagnostic",
)
async def quick_fix(req: QuickFixRequest):
    """
    Generate a targeted quick fix for one diagnostic.

    Called when the developer clicks the lightbulb on a squiggly line.
    The fix is applied inline via VS Code WorkspaceEdit.
    """
    t0 = time.time()

    file_path = req.file_path
    if not Path(file_path).exists():
        raise HTTPException(404, f"File not found: {file_path}")

    language = req.language or "python"

    # Build a focused code context (lines around the diagnostic)
    context_code = req.surrounding_code
    if not context_code and Path(file_path).exists():
        try:
            lines = Path(file_path).read_text(encoding="utf-8").splitlines()
            line_idx = max(0, req.line - 1)
            start = max(0, line_idx - 5)
            end   = min(len(lines), line_idx + 6)
            context_code = "\n".join(
                f"{i+1}: {lines[i]}" for i in range(start, end)
            )
        except Exception:
            context_code = ""

    prompt = _QUICK_FIX_PROMPT.format(
        language     = language,
        file_path    = Path(file_path).name,
        message      = req.diagnostic_message,
        code         = req.diagnostic_code or "code-auditor",
        line         = req.line,
        code_context = context_code[:800],
    )

    raw = await asyncio.to_thread(_invoke_llm, prompt, 256)

    actions = []
    if raw:
        # Clean up any accidental fences
        import re
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

        line_range = TextRange(
            start_line = max(0, req.line - 1),
            start_char = 0,
            end_line   = max(0, req.line - 1),
            end_char   = 999,
        ) if req.line else None

        actions.append(CodeActionItem(
            title             = f"Fix: {req.diagnostic_message[:60]}",
            kind              = "quickfix",
            is_preferred      = True,
            edit_type         = "replace",
            edit_range        = line_range,
            new_text          = raw,
            fixes_diagnostic  = req.diagnostic_message,
        ))

    return CodeActionsResponse(
        actions    = actions,
        total      = len(actions),
        elapsed_ms = round((time.time() - t0) * 1000),
    )


@code_actions_router.post(
    "/refactor",
    response_model=CodeActionsResponse,
    summary="Generate refactoring suggestion for selected code",
)
async def refactor_code(req: RefactorRequest):
    """
    Targeted refactoring for the selected code.

    Refactor kinds:
    - extract_function  : Extract selection into a new function
    - extract_variable  : Extract expression into a named variable
    - add_types         : Add type annotations to function/variable
    - add_docstring     : Generate docstring for function/class
    - general           : Let the LLM decide the best refactoring

    Returns a WorkspaceEdit the plugin applies directly.
    """
    t0 = time.time()

    file_path = req.file_path
    if not Path(file_path).exists():
        raise HTTPException(404, f"File not found: {file_path}")

    language = req.language or "python"

    prompt = _REFACTOR_PROMPT.format(
        file_path      = Path(file_path).name,
        refactor_kind  = req.refactor_kind,
        language       = language,
        selected_code  = req.selected_code[:1500],
        surrounding_code = req.surrounding_code[:600],
    )

    raw = await asyncio.to_thread(_invoke_llm, prompt, 512)

    actions = []
    if raw:
        import re
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

        kind_map = {
            "extract_function": "refactor.extract",
            "extract_variable": "refactor.extract",
            "add_types":        "refactor.rewrite",
            "add_docstring":    "source",
            "general":          "refactor",
        }

        actions.append(CodeActionItem(
            title        = f"Refactor: {req.refactor_kind.replace('_', ' ').title()}",
            kind         = kind_map.get(req.refactor_kind, "refactor"),
            is_preferred = True,
            edit_type    = "replace",
            edit_range   = req.range,
            new_text     = raw,
        ))

    return CodeActionsResponse(
        actions    = actions,
        total      = len(actions),
        elapsed_ms = round((time.time() - t0) * 1000),
    )
