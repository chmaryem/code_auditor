"""
conflict_resolution_graph.py — Axe 2 of the multi-agent refactor: formalizes
the per-file conflict resolution cascade (previously nested if/for control
flow in resolve_file_smart()) as an explicit LangGraph subgraph.

Real escalation pattern already present in the code, now made explicit:
    classify -> [identical | full_auto_merge | resolve_blocks]
    resolve_blocks -> validate -> [fallback | sandbox_check -> finalize | finalize]

Reuses the EXACT same helper functions as before (no reimplemented logic):
_classify_block_type, _auto_merge_imports, _merge_3way, _resolve_block_with_llm,
_validate_resolution — all imported from conflict_resolution_agent.py.

Scope: only the interactive=False (API) path of resolve_file_smart(). The
interactive=True terminal menu ([1] OURS [2] THEIRS [3] LLM) has zero callers
in the codebase and is left untouched in conflict_resolution_agent.py.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from smart_git.conflict_resolution_agent import (
    _classify_block_type,
    _auto_merge_imports,
    _merge_3way,
    _resolve_block_with_llm,
    _validate_resolution,
    _R, _GR, _YL, _RD,
)


class ConflictResolveState(TypedDict, total=False):
    filename: str
    base_content: str
    ours_content: str
    theirs_content: str
    rag_context: list

    identical: bool
    ours_lines: List[str]
    theirs_lines: List[str]
    all_opcodes: list
    diff_blocks: list
    all_auto: bool

    details: list
    used_llm: bool

    result: Optional[str]
    method: str
    valid: bool
    validation_msg: str


# ══════════════════════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════════════════════

def node_classify(state: ConflictResolveState) -> Dict[str, Any]:
    ours_content = state["ours_content"]
    theirs_content = state["theirs_content"]

    if ours_content == theirs_content:
        return {"identical": True}

    ours_lines = ours_content.splitlines(keepends=True)
    theirs_lines = theirs_content.splitlines(keepends=True)
    opcodes = list(difflib.SequenceMatcher(None, ours_lines, theirs_lines).get_opcodes())
    diff_blocks = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in opcodes if tag != "equal"]

    if not diff_blocks:
        return {"identical": True}

    filename = state["filename"]
    all_auto = True
    for tag, i1, i2, j1, j2 in diff_blocks:
        ours_block = "".join(ours_lines[i1:i2])
        theirs_block = "".join(theirs_lines[j1:j2])
        btype = _classify_block_type(ours_block, theirs_block, filename)
        if btype not in ("import", "simple"):
            all_auto = False
            break

    return {
        "identical": False,
        "ours_lines": ours_lines,
        "theirs_lines": theirs_lines,
        "all_opcodes": opcodes,
        "diff_blocks": diff_blocks,
        "all_auto": all_auto,
    }


def route_after_classify(state: ConflictResolveState) -> str:
    if state.get("identical"):
        return "identical"
    if state.get("all_auto"):
        return "full_auto_merge"
    return "resolve_blocks"


def node_identical(state: ConflictResolveState) -> Dict[str, Any]:
    return {"result": state["ours_content"], "method": "identical", "details": []}


def node_full_auto_merge(state: ConflictResolveState) -> Dict[str, Any]:
    result = _merge_3way(state["base_content"], state["ours_content"], state["theirs_content"])
    if result:
        return {"result": result, "method": "3way",
                "details": [{"type": "import", "choice": "auto"}]}
    return {"result": state["ours_content"], "method": "conservative",
            "details": [{"type": "import", "choice": "auto"}]}


def node_resolve_blocks(state: ConflictResolveState) -> Dict[str, Any]:
    ours_lines = state["ours_lines"]
    theirs_lines = state["theirs_lines"]
    filename = state["filename"]
    rag_context = state.get("rag_context")

    resolved_lines: List[str] = []
    details: List[dict] = []
    used_llm = False

    for tag, i1, i2, j1, j2 in state["all_opcodes"]:
        if tag == "equal":
            resolved_lines.extend(ours_lines[i1:i2])
            continue

        ours_block = "".join(ours_lines[i1:i2])
        theirs_block = "".join(theirs_lines[j1:j2])
        btype = _classify_block_type(ours_block, theirs_block, filename)

        if btype == "import":
            merged = _auto_merge_imports(ours_block, theirs_block)
            resolved_lines.extend(merged.splitlines(keepends=True))
            details.append({"type": "import", "choice": "auto",
                            "ours": ours_block[:100], "theirs": theirs_block[:100]})
        elif btype == "simple":
            resolved_lines.extend(ours_lines[i1:i2])
            details.append({"type": "simple", "choice": "auto"})
        else:
            # interactive=False (API) path always escalates to the LLM here.
            used_llm = True
            merged = _resolve_block_with_llm(filename, ours_block, theirs_block, rag_context)
            resolved_lines.extend(merged.splitlines(keepends=True))
            if not merged.endswith("\n"):
                resolved_lines.append("\n")
            details.append({"type": btype, "choice": "llm",
                            "ours": ours_block[:200], "theirs": theirs_block[:200]})

    return {"result": "".join(resolved_lines), "details": details, "used_llm": used_llm}


def node_validate(state: ConflictResolveState) -> Dict[str, Any]:
    valid, msg = _validate_resolution(state["result"])
    return {"valid": valid, "validation_msg": msg}


def route_after_validate(state: ConflictResolveState) -> str:
    if not state.get("valid"):
        return "fallback"
    if state.get("used_llm"):
        return "sandbox_check"
    return "finalize"


def node_fallback(state: ConflictResolveState) -> Dict[str, Any]:
    print(f"    {_RD}Validation échouée : {state.get('validation_msg')} → fallback OURS{_R}")
    return {"result": state["ours_content"], "method": "fallback"}


def node_sandbox_check(state: ConflictResolveState) -> Dict[str, Any]:
    """Best-effort syntax check for LLM-touched results — diagnostic only,
    never alters the resolved content (mirrors the original try/except)."""
    try:
        from services.sandbox_executor import SandboxExecutor
        result = state["result"]
        script = (
            f"code = '''{result[:3000]}'''\n"
            f"braces = code.count('{{') == code.count('}}')\n"
            f"no_markers = '<<<<<<' not in code\n"
            f"print('SANDBOX_OK' if braces and no_markers else 'SANDBOX_FAIL')\n"
        )
        sb_result = SandboxExecutor(timeout=15).execute(script)
        if sb_result.success and "SANDBOX_OK" in (sb_result.stdout or ""):
            print(f"    {_GR}✓ Sandbox validé{_R}")
        else:
            print(f"    {_YL}⚠ Sandbox warning (résultat conservé){_R}")
    except Exception:
        pass
    return {}


def node_finalize(state: ConflictResolveState) -> Dict[str, Any]:
    used_llm = state.get("used_llm", False)
    return {"method": "interactive" + ("_llm" if used_llm else "")}


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_conflict_resolve_graph():
    g = StateGraph(ConflictResolveState)

    g.add_node("classify", node_classify)
    g.add_node("identical", node_identical)
    g.add_node("full_auto_merge", node_full_auto_merge)
    g.add_node("resolve_blocks", node_resolve_blocks)
    g.add_node("validate", node_validate)
    g.add_node("fallback", node_fallback)
    g.add_node("sandbox_check", node_sandbox_check)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify, {
        "identical": "identical",
        "full_auto_merge": "full_auto_merge",
        "resolve_blocks": "resolve_blocks",
    })
    g.add_edge("identical", END)
    g.add_edge("full_auto_merge", END)
    g.add_edge("resolve_blocks", "validate")
    g.add_conditional_edges("validate", route_after_validate, {
        "fallback": "fallback",
        "sandbox_check": "sandbox_check",
        "finalize": "finalize",
    })
    g.add_edge("fallback", END)
    g.add_edge("sandbox_check", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


_conflict_resolve_graph = None


def run_resolve_file_smart_graph(
    filename: str,
    base_content: str,
    ours_content: str,
    theirs_content: str,
    rag_context: Optional[list] = None,
) -> Tuple[Optional[str], str, list]:
    """Drop-in replacement for resolve_file_smart(..., interactive=False)."""
    global _conflict_resolve_graph
    if _conflict_resolve_graph is None:
        _conflict_resolve_graph = build_conflict_resolve_graph()

    initial: ConflictResolveState = {
        "filename": filename,
        "base_content": base_content,
        "ours_content": ours_content,
        "theirs_content": theirs_content,
        "rag_context": rag_context or [],
    }
    result = _conflict_resolve_graph.invoke(initial)
    return result.get("result"), result.get("method", "identical"), result.get("details", [])
