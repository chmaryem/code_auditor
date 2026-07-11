"""
chat_graph.py — LangGraph ChatGraph.

Goal:
  Provide a conversational entry point over the existing Code Auditor systems.

Architecture:
  load_memory → intent_router → decision_agent
    → route decision:
        - CI/Git questions can skip file context
        - code questions load file context
    → route file context:
        - fast explain
        - contextual Q&A
        - code generation
        - test generation placeholder
    → memory_save → format_response → END

Streaming:
  stream_chat() uses LangGraph astream_events() and yields SSE events:
    status | token | code | done | error
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.state import ChatState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Initial state helper
# ══════════════════════════════════════════════════════════════════════════════

def _initial_state(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    user_id: str = "",
    target_file: str = "",
    cursor_line: int = 0,
    active_function: str = "",
    selected_text: str = "",
    visible_range: list | None = None,
    active_module: str = "",
    branch: str = "",
    active_repository: str = "",
    scope: str = "extension",
    attached_files: list | None = None,
    **services: Any,
) -> ChatState:
    return {
        "session_id": session_id,
        "user_id": user_id,
        "user_message": message,
        "project_path": str(Path(project_path).resolve()),
        "target_file": target_file or "",
        "target_lang": "unknown",

        # Dashboard context
        "active_module":     active_module or "",
        "branch":            branch or "",
        "active_repository": active_repository or "",

        # Scope / périmètre
        "scope":                      scope or "extension",
        "attached_files":             attached_files or [],
        "dashboard_needs_attachment": False,

        # IDE cursor context
        "cursor_line":       cursor_line,
        "active_function":   active_function,
        "selected_text":     selected_text,
        "visible_range":     visible_range or [0, 0],

        # Git / CI context (populated lazily)
        "git_context":  {},
        "ci_context":   {},

        # Intent / decision
        "intent": "question",
        "intent_params": {},
        "decision_plan": {},
        "context_level": "context",
        "selected_agents": [],
        "needs_rag": True,
        "needs_git": False,
        "needs_ci": False,
        "needs_generation": False,
        "needs_tests": False,

        # Context
        "history": [],
        "memory_key": "",
        "file_code": "",
        "file_analysis": {},
        "dependencies": [],
        "dependents": [],
        "project_summary": {},
        "rag_docs": [],
        "rag_scores": [],
        "file_cache": {},

        # Generation
        "generation_target": "",
        "generation_language": "",
        "generated_code": "",
        "generation_valid": True,
        "generation_errors": [],
        "project_patterns": {},
        "apply_to_disk": False,

        # Output
        "response": "",
        "formatted_response": "",
        "code_blocks": [],
        "suggested_files": [],

        **services,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Memory + Intent + Decision
# ══════════════════════════════════════════════════════════════════════════════

async def node_load_ai_settings(state: ChatState) -> Dict[str, Any]:
    """Load per-user AI settings from PostgreSQL. Falls back to defaults silently."""
    user_id = state.get("user_id", "")
    if not user_id:
        return {}
    try:
        from database.connection import AsyncSessionLocal
        from database.repositories.settings_repo import SettingsRepo
        async with AsyncSessionLocal() as db:
            async with db.begin():
                ai = await SettingsRepo(db).get_ai_settings(user_id)
        if ai:
            return {
                "ai_temperature":   ai.temperature,
                "ai_mode":          ai.ai_mode,
                "ai_response_style":ai.response_style,
                "ai_use_rag":       ai.use_rag,
                "ai_use_memory":    ai.use_conversation_memory,
                "ai_max_context":   ai.max_context_size,
                "ai_streaming":     ai.streaming_enabled,
            }
    except Exception as exc:
        logger.debug("node_load_ai_settings: could not load settings: %s", exc)
    return {}


def node_load_memory(state: ChatState) -> Dict[str, Any]:
    """Load conversation history from Redis (skipped when ai_use_memory=False)."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent
    from services.chat_memory_service import chat_memory_service

    if not state.get("ai_use_memory", True):
        session_id = state.get("session_id") or chat_memory_service.new_session_id()
        return {"session_id": session_id, "history": [], "memory_key": ""}

    session_id = state.get("session_id") or chat_memory_service.new_session_id()
    history = lc_chat_agent.load_history(session_id)

    return {
        "session_id": session_id,
        "history": history,
        "memory_key": chat_memory_service.history_key(session_id),
    }


def node_intent_router(state: ChatState) -> Dict[str, Any]:
    """Detect base chat intent using LCChatAgent deterministic router."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.detect_intent(
        state.get("user_message", ""),
        state.get("target_file", "") or "",
    )

    return {
        "intent": result.get("intent", "question"),
        "intent_params": result.get("intent_params", {}),
    }


def node_decision_agent(state: ChatState) -> Dict[str, Any]:
    """
    LLM-powered decision agent (v2).
    Passes cursor context for IDE-aware routing.
    Falls back to regex if LLM unavailable or low confidence.
    """
    from langchain_agents.agents.lc_chat_decision_agent import chat_decision_agent

    plan = chat_decision_agent.decide(
        user_message         = state.get("user_message", ""),
        target_file          = state.get("target_file", "") or "",
        base_intent          = state.get("intent", "question"),
        intent_params        = state.get("intent_params", {}) or {},
        conversation_history = state.get("history", []),
        # cursor context
        cursor_line          = state.get("cursor_line", 0),
        active_function      = state.get("active_function", ""),
        selected_text        = state.get("selected_text", ""),
    )

    old_params = state.get("intent_params", {}) or {}
    merged_params = {
        **old_params,
        "file_hint":        old_params.get("file_hint")        or plan.get("target_file", ""),
        "method_hint":      old_params.get("method_hint")      or plan.get("target_symbol", ""),
        "generation_target":old_params.get("generation_target") or plan.get("generation_target", ""),
    }

    resolved_intent = plan.get("intent", state.get("intent", "question"))

    # ── Périmètre dashboard : les intents « code » exigent un fichier attaché ──
    # Le chat dashboard ne scanne pas le projet. Si une question code arrive sans
    # attachement, on redirige vers une réponse courte invitant à joindre le fichier
    # (le vrai chat d'analyse projet est côté extension VS Code).
    dashboard_needs_attachment = False
    _CODE_INTENTS = {"explain", "complete_fn", "new_class", "code_generation"}
    if (state.get("scope") == "dashboard"
            and resolved_intent in _CODE_INTENTS
            and not (state.get("attached_files") or [])):
        resolved_intent = "question"
        dashboard_needs_attachment = True

    return {
        "decision_plan":    plan,
        "intent":           resolved_intent,
        "intent_params":    merged_params,
        "context_level":    "fast" if dashboard_needs_attachment else plan.get("context_level", "context"),
        "selected_agents":  plan.get("agents", []),
        "needs_rag":        False if dashboard_needs_attachment else bool(plan.get("needs_rag", True)),
        "needs_git":        bool(plan.get("needs_git", False)),
        "needs_ci":         bool(plan.get("needs_ci", False)),
        "needs_generation": False if dashboard_needs_attachment else bool(plan.get("needs_generation", False)),
        "needs_tests":      bool(plan.get("needs_tests", False)),
        "dashboard_needs_attachment": dashboard_needs_attachment,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Context
# ══════════════════════════════════════════════════════════════════════════════

_EXT_LANG = {
    ".py": "python", ".java": "java", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rb": "ruby",
    ".php": "php", ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".rs": "rust",
    ".sql": "sql", ".sh": "bash", ".yml": "yaml", ".yaml": "yaml", ".json": "json",
}


def _lang_from_path(path: str) -> str:
    """Détection légère du langage à partir de l'extension (pour fichiers attachés)."""
    for ext, lang in _EXT_LANG.items():
        if path.lower().endswith(ext):
            return lang
    return "unknown"


def node_load_file_context(state: ChatState) -> Dict[str, Any]:
    """Load target file, cached analysis and dependency context.

    Mode dashboard (scope="dashboard") : AUCUN scan du projet local.
    Le seul code fourni au LLM = les fichiers explicitement attachés par le dev.
    """
    # ── Périmètre dashboard : uniquement les fichiers attachés ────────────────
    if state.get("scope") == "dashboard":
        attached = state.get("attached_files") or []
        if not attached:
            # Aucun fichier attaché : rien à charger (le nudge est géré en amont).
            return {"file_code": "", "target_file": "", "target_lang": "unknown",
                    "file_analysis": {}, "dependencies": [], "dependents": []}
        code = "\n\n".join(
            f"# ── {f.get('path', 'fichier')} ──\n{f.get('content', '')}" for f in attached
        )
        first_path = attached[0].get("path", "")
        return {
            "file_code":     code,
            "target_file":   first_path,
            "target_lang":   _lang_from_path(first_path),
            "file_analysis": {},
            "dependencies":  [],   # pas de graphe de dépendances en mode dashboard
            "dependents":    [],
        }

    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    params = state.get("intent_params", {}) or {}
    target_file = state.get("target_file") or params.get("file_hint", "")

    # Per-invocation cache. A Redis-backed session cache can be added later.
    file_cache = state.get("file_cache", {}) or {}
    if target_file and target_file in file_cache:
        cached = file_cache[target_file]
        return {
            "target_file": target_file,
            "target_lang": cached["target_lang"],
            "file_code": cached["file_code"],
            "file_analysis": cached["file_analysis"],
            "dependencies": cached["dependencies"],
            "dependents": cached["dependents"],
            "intent_params": {
                **(state.get("intent_params") or {}),
                "method_hint": cached.get("method_hint", ""),
            },
            "file_cache": file_cache,
        }

    ctx = lc_chat_agent.load_file_context(
        project_path=state.get("project_path", "."),
        target_file=target_file,
        user_message=state.get("user_message", ""),
    )

    found_target_file = ctx.get("target_file", target_file or "")
    if found_target_file and ctx.get("file_code"):
        file_cache[found_target_file] = {
            "target_lang": ctx.get("target_lang", "unknown"),
            "file_code": ctx.get("file_code", ""),
            "file_analysis": ctx.get("file_analysis", {}),
            "dependencies": ctx.get("dependencies", []),
            "dependents": ctx.get("dependents", []),
            "method_hint": ctx.get("method_hint", ""),
        }

    return {
        "target_file": found_target_file,
        "target_lang": ctx.get("target_lang", "unknown"),
        "file_code": ctx.get("file_code", ""),
        "file_analysis": ctx.get("file_analysis", {}),
        "dependencies": ctx.get("dependencies", []),
        "dependents": ctx.get("dependents", []),
        "intent_params": {
            **(state.get("intent_params") or {}),
            "method_hint": ctx.get("method_hint", ""),
        },
        "file_cache": file_cache,
    }


async def node_parallel_context(state: ChatState) -> Dict[str, Any]:
    """
    Parallel context loader — replaces sequential project_summary → rag_retrieve.

    Runs concurrently:
      - project_summary  (lightweight metadata)
      - rag_retrieve     (ChromaDB vector search)
      - neighborhood     (Knowledge Graph dep lookup)
      - git_context      (GitSessionTracker snapshot — only if needs_git)

    ~60% latency reduction vs sequential execution.

    Mode dashboard : project_summary et RAG local sont désactivés (pas de scan projet) ;
    seul le contexte git (métadonnées, pas le code source) reste chargé si nécessaire.
    """
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    project_path = state.get("project_path", ".")
    user_message = state.get("user_message", "")
    target_file  = state.get("target_file", "") or ""
    file_code    = state.get("file_code", "") or ""
    language     = state.get("target_lang", "unknown")
    needs_git    = state.get("needs_git", False)
    needs_rag    = state.get("needs_rag", True)

    # Périmètre dashboard : aucun scan du projet local (summary + RAG code coupés).
    is_dashboard = state.get("scope") == "dashboard"
    if is_dashboard:
        needs_rag = False

    async def _summary():
        if is_dashboard:
            return {}   # pas de comptage/scan projet en mode dashboard
        try:
            return lc_chat_agent.project_summary(project_path)
        except Exception as e:
            logger.warning("project_summary failed: %s", e)
            return {}

    async def _rag():
        if not needs_rag:
            return {"rag_docs": [], "rag_scores": []}
        try:
            result = await asyncio.to_thread(
                lc_chat_agent.retrieve,
                project_path=project_path,
                query=user_message,
                target_file=target_file,
                file_code=file_code,
                language=language,
            )
            return {"rag_docs": result.get("rag_docs", []), "rag_scores": result.get("rag_scores", [])}
        except Exception as e:
            logger.warning("rag_retrieve failed: %s", e)
            return {"rag_docs": [], "rag_scores": []}

    async def _git():
        if not needs_git:
            return {}
        try:
            from smart_git.git_session_tracker import GitSessionTracker
            tracker = GitSessionTracker(project_path)
            snapshot = await asyncio.to_thread(tracker.get_session_status)
            return snapshot or {}
        except Exception as e:
            logger.debug("git_context failed: %s", e)
            return {}

    summary, rag_result, git_snap = await asyncio.gather(
        _summary(), _rag(), _git()
    )

    return {
        "project_summary": summary,
        "rag_docs":        rag_result["rag_docs"],
        "rag_scores":      rag_result["rag_scores"],
        "git_context":     git_snap,
    }


def node_rag_retrieve(state: ChatState) -> Dict[str, Any]:
    """Standalone RAG retrieve — used only by generation sub-path.
    Skipped when ai_use_rag is disabled in user settings.
    """
    if not state.get("ai_use_rag", True):
        return {"rag_docs": [], "rag_scores": []}

    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.retrieve(
        project_path=state.get("project_path", "."),
        query=state.get("user_message", ""),
        target_file=state.get("target_file", ""),
        file_code=state.get("file_code", ""),
        language=state.get("target_lang", "unknown"),
    )

    return {
        "rag_docs":   result.get("rag_docs", []),
        "rag_scores": result.get("rag_scores", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Answer / Specialized paths
# ══════════════════════════════════════════════════════════════════════════════

_DASHBOARD_ATTACH_NUDGE = (
    "Pour analyser ou expliquer du code, ajoutez le(s) fichier(s) concerné(s) "
    "via le bouton 📎 de la barre de saisie.\n\n"
    "Le chat du dashboard répond sur **Git, les PR, le dépôt, le CI/CD et les questions générales**, "
    "et sur les fichiers que vous attachez — il ne parcourt pas l'ensemble de votre projet. "
    "Pour une analyse complète du code de votre projet, utilisez le chat de l'extension VS Code."
)


async def node_fast_answer(state: ChatState, config: Any = None) -> Dict[str, Any]:
    """Fast streamed response for simple explain/summarize questions."""
    if state.get("dashboard_needs_attachment"):
        return {"response": _DASHBOARD_ATTACH_NUDGE}

    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    response = await lc_chat_agent.afast_answer(dict(state), config=config)
    return {"response": response}


async def node_answer_question(state: ChatState, config: Any = None) -> Dict[str, Any]:
    """Generate final project-aware answer."""
    if state.get("dashboard_needs_attachment"):
        return {"response": _DASHBOARD_ATTACH_NUDGE}

    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    response = await lc_chat_agent.aanswer(dict(state), config=config)
    return {"response": response}


async def node_git_question(state: ChatState) -> Dict[str, Any]:
    """
    Route Git-related questions to SmartGitGraph.

    Phase B enhancement — "Explain what changed":
      If the message is about understanding recent changes (diff, what did I modify,
      résume mes changements…), we inject the actual git diff into the prompt so the
      ChatAgent can answer conversationally without routing to SmartGitGraph at all.

    For heavier operations (PR review, conflict resolution…), SmartGitGraph is used.
    """
    import re

    message = state.get("user_message", "") or ""
    project_path = state.get("project_path", ".") or "."

    # ── "Explain what changed" shortcut ──────────────────────────────────────
    EXPLAIN_DIFF_PATTERNS = [
        r"what (did|have) i (changed|modified|done)",
        r"(explain|describe|résume|résumé|summarize|summary)\s+(what|my|les|mes)?\s*(changed|changes|modifications|diff|recent)",
        r"(qu'est-ce que|qu'ai-je)\s+(j'ai\s+)?(changé|modifié|fait)",
        r"résume\s+mes\s+changements",
        r"what.*(different|changed).*(since|from)",
        r"(show|explain).*(diff|patch)",
    ]
    is_explain_diff = any(
        re.search(p, message, re.IGNORECASE) for p in EXPLAIN_DIFF_PATTERNS
    )

    if is_explain_diff:
        diff_text = ""
        try:
            import subprocess
            result_proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "-C", project_path, "diff", "--stat", "--unified=3"],
                capture_output=True, text=True, timeout=10,
            )
            diff_text = result_proc.stdout or ""
            if not diff_text.strip():
                # Try staged
                result_proc = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "-C", project_path, "diff", "--cached", "--stat", "--unified=3"],
                    capture_output=True, text=True, timeout=10,
                )
                diff_text = result_proc.stdout or ""
        except Exception as e:
            logger.debug("git diff failed: %s", e)

        if diff_text.strip():
            from langchain_agents.agents.lc_chat_agent import lc_chat_agent
            enriched_state = dict(state)
            enriched_state["user_message"] = (
                f"{message}\n\n"
                f"[Git diff context — current working tree]\n"
                f"```diff\n{diff_text[:3000]}\n```"
            )
            response = await lc_chat_agent.aanswer(enriched_state)
            return {"response": response, "git_diff_used": True}
        else:
            return {
                "response": (
                    "## Git Changes\n\n"
                    "✅ Aucune modification non commitée détectée dans le working tree.\n\n"
                    "Si tu viens de committer, utilise :\n"
                    "```bash\ngit show --stat\n```"
                ),
                "git_diff_used": False,
            }

    # ── Direct Smart Git dispatch for heavier operations ──────────────────────
    # Phase 0 (multi-agent refactor): classify + dispatch without the LangGraph.
    # Axe a (2026-07-05): the Chat previously never passed owner/repo/branch,
    # so PR intents (pr_review/pr_readiness) always hit "Missing owner/repo/
    # pr_number" and branch_readiness always analyzed HEAD vs main regardless
    # of what the user asked about.
    #
    # owner/repo resolution (2026-07-05, follow-up): the local git remote of
    # `project_path` is NOT necessarily the repo the user means — the PR
    # Cockpit lets a developer pick ANY GitHub repo independently of the
    # local checkout (users.active_github_repo, already piped into
    # ChatState.active_repository). Prefer that; fall back to the local git
    # remote (previous behavior) only when no active repo is selected.
    from langchain_agents.agents.smart_git_dispatch import adispatch_smart_git_message

    active_repository = (state.get("active_repository") or "").strip()
    if "/" in active_repository:
        owner, repo = active_repository.split("/", 1)
    else:
        owner, repo = _resolve_owner_repo(project_path)
    branch = state.get("branch", "") or "HEAD"

    result = await adispatch_smart_git_message(
        message=message,
        project_path=project_path,
        owner=owner,
        repo=repo,
        branch=branch,
        session_id=state.get("session_id", ""),
    )

    return {
        "response":   result.get("response", ""),
        "git_result": result,
    }


def _resolve_owner_repo(project_path: str) -> tuple[str, str]:
    """Best-effort owner/repo from the git remote — no network call."""
    try:
        import re as _re
        import subprocess
        url = subprocess.run(
            ["git", "-C", project_path, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        m = _re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
        if m:
            return m.group(1), m.group(2)
    except Exception as e:
        logger.debug("owner/repo resolution failed: %s", e)
    return "", ""


async def node_ci_question(state: ChatState) -> Dict[str, Any]:
    """CI/CD readiness via the same scorer used by the CI/CD dashboard."""
    from langchain_agents.tools.project_state_tools import tool_chat_ci_readiness
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    project_path = state.get("project_path", ".") or "."
    owner, repo = _resolve_owner_repo(project_path)
    ci = await asyncio.to_thread(tool_chat_ci_readiness, project_path, repo, owner)

    if not ci.get("available"):
        return {
            "response": (
                "## CI/CD Readiness\n\n"
                f"Je n'ai pas pu calculer le score de déploiement ({ci.get('reason', 'raison inconnue')}).\n\n"
                "Lance une analyse depuis la page **CI/CD** du dashboard pour lier ce projet à son repo GitHub."
            ),
            "project_state_context": {"ci_readiness": ci},
        }

    enriched_state = dict(state)
    enriched_state["user_message"] = (
        f"{state.get('user_message', '')}\n\n"
        f"[CI/CD readiness data]\n"
        f"score={ci['score']} verdict={ci['verdict']}\n"
        f"blocking_reasons={ci['blocking_reasons']}\n"
        f"warnings={ci['warnings']}\n"
        f"component_scores={ci['component_scores']}\n"
    )
    response = await lc_chat_agent.aanswer(enriched_state)
    return {"response": response, "project_state_context": {"ci_readiness": ci}}


async def node_project_state(state: ChatState) -> Dict[str, Any]:
    """Holistic project-state answer — git risk + secrets + test gaps +
    security/quality + CI readiness, combined into one developer-facing summary."""
    from langchain_agents.tools.project_state_tools import tool_chat_project_state_summary
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    project_path = state.get("project_path", ".") or "."
    target_file = state.get("target_file")
    owner, repo = _resolve_owner_repo(project_path)

    summary = await asyncio.to_thread(
        tool_chat_project_state_summary, project_path, target_file, repo, owner
    )

    lines = ["[Project state snapshot]"]
    for key, section in summary.items():
        if section.get("available"):
            lines.append(f"- {key}: {json.dumps({k: v for k, v in section.items() if k != 'available'})[:600]}")
        else:
            lines.append(f"- {key}: unavailable ({section.get('reason', '')})")

    enriched_state = dict(state)
    enriched_state["user_message"] = (
        f"{state.get('user_message', '')}\n\n" + "\n".join(lines) +
        "\n\nUsing ONLY the data above, answer the developer's question. "
        "If a section is unavailable, say so briefly instead of guessing. "
        "Prioritize what's blocking or risky, then suggest concrete next actions."
    )
    response = await lc_chat_agent.aanswer(enriched_state)
    return {"response": response, "project_state_context": summary}


def node_test_generation(state: ChatState) -> Dict[str, Any]:
    """Placeholder for TestAgent integration."""
    return {
        "response": (
            "## Test Generation\n\n"
            "Cette demande doit être traitée par le **Test System**.\n\n"
            "L’intégration directe `ChatAgent → TestAgent` est prévue dans la prochaine étape.\n\n"
            "Pour l’instant, utilise la commande existante du plugin :\n\n"
            "`Code Auditor: Generate Tests for Current File`"
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Phase 2 generation
# ══════════════════════════════════════════════════════════════════════════════

def node_load_project_patterns(state: ChatState) -> Dict[str, Any]:
    """Detect project conventions for code generation."""
    from services.code_generator_service import detect_conventions

    lang = state.get("generation_language") or state.get("target_lang") or "python"
    if lang == "unknown":
        lang = "python"

    try:
        patterns = detect_conventions(state.get("project_path", "."), lang)
    except Exception as e:
        logger.warning("detect_conventions failed: %s", e)
        patterns = {}

    return {"project_patterns": patterns}


def node_generate_completion(state: ChatState) -> Dict[str, Any]:
    """Generate function body completion."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.complete_function(dict(state))
    return result


def node_generate_class(state: ChatState) -> Dict[str, Any]:
    """Generate a new class from scratch."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.generate_class(dict(state))
    return result


def node_validate_generated(state: ChatState) -> Dict[str, Any]:
    """Validate syntax of generated code."""
    from services.code_generator_service import validate_syntax

    code = state.get("generated_code", "")
    lang = state.get("generation_language") or state.get("target_lang") or "python"

    if lang == "unknown":
        lang = "python"

    if not code:
        return {
            "generation_valid": False,
            "generation_errors": ["No code was generated."],
        }

    valid, errors = validate_syntax(code, lang)
    response = state.get("response", "")

    if valid:
        response += "\n\n---\n**Validation:** OK"
    else:
        response += (
            "\n\n---\n"
            "**Validation:** FAILED\n\n"
            + "\n".join(f"- {e}" for e in errors)
        )

    return {
        "generation_valid": valid,
        "generation_errors": errors,
        "response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Memory + Formatting
# ══════════════════════════════════════════════════════════════════════════════

def node_memory_save(state: ChatState) -> Dict[str, Any]:
    """
    Persiste l'échange user↔assistant.

    Architecture dual-layer :
      - Redis (cache chaud, TTL 1 h)  → via lc_chat_agent.save_exchange()
      - PostgreSQL (source de vérité) → via persistent_chat_memory.save_exchange_sync()
                                         dans un thread daemon non-bloquant

    Les deux écritures sont indépendantes : un échec Redis n'affecte pas PG
    et vice-versa.
    """
    import threading
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    plan         = state.get("decision_plan") or {}
    session_id   = state.get("session_id", "")
    user_id      = state.get("user_id", "")
    user_message = state.get("user_message", "")
    response     = state.get("response", "")
    project_path = state.get("project_path", "")
    intent       = state.get("intent", "")
    scope        = state.get("scope", "extension")

    metadata = {
        "intent":          intent,
        "target_file":     state.get("target_file"),
        "language":        state.get("target_lang"),
        "context_level":   state.get("context_level"),
        "selected_agents": state.get("selected_agents", []),
        "decision_reason": plan.get("reason", ""),
        "generated_code":  state.get("generated_code", ""),
        "cursor_line":     state.get("cursor_line", 0),
        "active_function": state.get("active_function", ""),
    }

    # ── 1. Redis (cache chaud — synchrone, fast path) ─────────────────────────
    lc_chat_agent.save_exchange(
        session_id=session_id,
        user_message=user_message,
        response=response,
        metadata=metadata,
        project_path=project_path,
    )

    # ── 2. PostgreSQL (source de vérité — thread daemon non-bloquant) ─────────
    # Périmètre : SEUL le chat DASHBOARD est persisté dans la table `conversations`.
    # Le chat de l'EXTENSION VS Code possède ses propres tables dédiées
    # (extension_chat_sessions / extension_chat_messages, via /api/extension/chat/*)
    # qu'il persiste lui-même côté frontend (ChatSidebarProvider._saveHistory).
    # Sans ce garde-fou, chaque conversation de l'extension fuitait aussi dans
    # `conversations` → /api/chat/sessions (qui lit cette table sans filtre de
    # scope) affichait les chats de l'extension dans l'historique du dashboard.
    if scope == "dashboard" and user_id and session_id:
        def _persist_to_pg() -> None:
            from services.persistent_chat_memory_service import persistent_chat_memory
            persistent_chat_memory.save_exchange_sync(
                session_id=session_id,
                user_message=user_message,
                assistant_response=response,
                user_id=user_id,
                metadata=metadata,
                project_path=project_path,
                intent=intent or None,
            )

        threading.Thread(target=_persist_to_pg, daemon=True).start()

    # ── 3. Semantic memory (Phase C1 — thread daemon non-bloquant) ───────────
    def _write_semantic() -> None:
        try:
            from langchain_agents.memory.lc_semantic_memory import semantic_memory
            semantic_memory.write_memory(
                session_id=session_id,
                user_message=user_message,
                assistant_message=response,
                metadata={"intent": intent, "file": state.get("target_file", "")},
            )
        except Exception as exc:
            logger.debug("semantic memory write failed: %s", exc)

    threading.Thread(target=_write_semantic, daemon=True).start()

    # ── 4. LearningAgent feedback (code generation uniquement) ────────────────
    if intent in ("complete_fn", "new_class", "code_generation"):
        generated = state.get("generated_code", "")
        lang = state.get("generation_language") or state.get("target_lang") or "unknown"
        if generated:
            try:
                from langchain_agents.agents.lc_learning_agent import learning_agent
                learning_agent.submit_feedback(
                    block={
                        "problem":    f"code_generation:{intent}",
                        "fixed_code": generated,
                        "severity":   "LOW",
                    },
                    action="suggested",
                    language=lang,
                )
            except Exception as exc:
                logger.warning("Could not notify LearningAgent: %s", exc)

    return {}


def node_format_response(state: ChatState) -> Dict[str, Any]:
    """Final formatting — clean response ready for the developer."""
    response  = state.get("response", "")
    proactive = state.get("proactive_suggestions", {}) or {}

    formatted = response.strip()

    # Append proactive summary only for critical blocking issues
    if proactive.get("has_critical"):
        n = proactive.get("total", 0)
        formatted += (
            f"\n\n> ⚠️ **{n} issue(s) detected** — check the CI/CD panel for details."
        )

    sources = sorted(
        k for k, v in (state.get("project_state_context") or {}).items()
        if v.get("available")
    )

    return {"formatted_response": formatted, "context_sources": sources}


# ══════════════════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════════════════

def _route_after_decision(state: ChatState) -> str:
    intent        = state.get("intent", "question")
    context_level = state.get("context_level", "context")

    if intent == "ci_question":
        return "ci_question"
    if intent == "git_question":
        return "git_question"
    if intent == "project_state":
        return "project_state"
    # Phase C3: deep context_level → tool-calling pair programmer
    if context_level == "deep" and intent == "question":
        return "semantic_recall"
    return "load_file_context"


def _route_after_file_context(state: ChatState) -> str:
    intent        = state.get("intent", "question")
    context_level = state.get("context_level", "context")

    if intent == "test_generation":
        return "test_generation"

    if context_level == "fast" and state.get("file_code"):
        return "fast_answer"

    # All other paths go through parallel context (RAG + summary + git)
    return "parallel_context"


def _route_after_parallel_context(state: ChatState) -> str:
    intent        = state.get("intent", "question")
    context_level = state.get("context_level", "context")

    if intent in ("complete_fn", "new_class", "code_generation"):
        return "load_project_patterns"

    # Phase C3: very complex questions → tool-calling
    if context_level == "deep" and intent in ("question", "explain"):
        return "tool_calling_answer"

    return "answer_question"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_chat_graph():
    graph = StateGraph(ChatState)

    # Core orchestration nodes
    graph.add_node("load_ai_settings",   node_load_ai_settings)
    graph.add_node("load_memory",        node_load_memory)
    graph.add_node("intent_router",      node_intent_router)
    graph.add_node("decision_agent",     node_decision_agent)
    graph.add_node("load_file_context",  node_load_file_context)
    graph.add_node("parallel_context",   node_parallel_context)
    graph.add_node("rag_retrieve",       node_rag_retrieve)

    # Phase C: semantic recall + tool-calling
    graph.add_node("semantic_recall",    node_semantic_recall)
    graph.add_node("tool_calling_answer",node_tool_calling_answer)

    # Answer nodes
    graph.add_node("fast_answer",        node_fast_answer)
    graph.add_node("answer_question",    node_answer_question)
    graph.add_node("git_question",       node_git_question)
    graph.add_node("ci_question",        node_ci_question)
    graph.add_node("project_state",      node_project_state)
    graph.add_node("test_generation",    node_test_generation)

    # Generation nodes
    graph.add_node("load_project_patterns", node_load_project_patterns)
    graph.add_node("generate_completion",   node_generate_completion)
    graph.add_node("generate_class",        node_generate_class)
    graph.add_node("validate_generated",    node_validate_generated)

    # Tail nodes
    graph.add_node("memory_save",        node_memory_save)
    graph.add_node("format_response",    node_format_response)

    # ── Entry ────────────────────────────────────────────────────────────────
    graph.set_entry_point("load_ai_settings")
    graph.add_edge("load_ai_settings", "load_memory")
    graph.add_edge("load_memory",   "intent_router")
    graph.add_edge("intent_router", "decision_agent")

    graph.add_conditional_edges(
        "decision_agent",
        _route_after_decision,
        {
            "load_file_context": "load_file_context",
            "git_question":      "git_question",
            "ci_question":       "ci_question",
            "project_state":     "project_state",
            "semantic_recall":   "semantic_recall",   # Phase C3 fast track
        },
    )

    # semantic_recall → tool_calling_answer (always)
    graph.add_edge("semantic_recall", "tool_calling_answer")

    graph.add_conditional_edges(
        "load_file_context",
        _route_after_file_context,
        {
            "fast_answer":      "fast_answer",
            "parallel_context": "parallel_context",
            "test_generation":  "test_generation",
        },
    )

    graph.add_conditional_edges(
        "parallel_context",
        _route_after_parallel_context,
        {
            "answer_question":      "answer_question",
            "load_project_patterns":"load_project_patterns",
            "tool_calling_answer":  "tool_calling_answer",  # Phase C3 deep path
        },
    )

    # Generation sub-path
    graph.add_edge("load_project_patterns", "rag_retrieve")
    graph.add_conditional_edges(
        "rag_retrieve",
        lambda s: (
            "generate_completion" if s.get("intent") in ("complete_fn", "code_generation")
            else "generate_class" if s.get("intent") == "new_class"
            else "answer_question"
        ),
        {
            "answer_question":    "answer_question",
            "generate_completion":"generate_completion",
            "generate_class":     "generate_class",
        },
    )

    graph.add_edge("generate_completion", "validate_generated")
    graph.add_edge("generate_class",      "validate_generated")

    # All answer paths → memory_save → format → END
    for node in ("fast_answer", "answer_question", "git_question",
                 "ci_question", "project_state", "test_generation", "validate_generated",
                 "tool_calling_answer"):
        graph.add_edge(node, "memory_save")

    graph.add_edge("memory_save",    "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()

# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Phase C3 : Semantic recall + Tool-calling pair programmer
# ══════════════════════════════════════════════════════════════════════════════

def node_semantic_recall(state: ChatState) -> Dict[str, Any]:
    """
    Phase C1+C3: Recall relevant facts from the semantic memory store.
    Injects past context into the state for use by node_tool_calling_answer.
    """
    session_id = state.get("session_id", "")
    query      = state.get("user_message", "")
    if not session_id or not query:
        return {"semantic_context": []}
    try:
        from langchain_agents.memory.lc_semantic_memory import semantic_memory
        facts = semantic_memory.recall_memory(session_id=session_id, query=query)
        return {"semantic_context": facts}
    except Exception as e:
        logger.debug("node_semantic_recall: %s", e)
        return {"semantic_context": []}


async def node_tool_calling_answer(state: ChatState) -> Dict[str, Any]:
    """
    Phase C3: LLM-driven pair programmer with real tool calls.

    The LLM autonomously decides which tools to call
    (search_codebase, get_file, get_git_diff, analyze_file, get_dependencies, get_ci_status)
    and synthesizes a final answer from the tool results.
    """
    from langchain_agents.agents.lc_tool_calling_agent import lc_tool_calling_agent

    result = await lc_tool_calling_agent.arun(
        message      = state.get("user_message", ""),
        project_path = state.get("project_path", "."),
        session_id   = state.get("session_id", ""),
        history      = state.get("history", []),
        semantic_context = state.get("semantic_context", []),
        target_file  = state.get("target_file", "") or "",
    )

    return {
        "response":      result.get("response", ""),
        "tools_called":  result.get("tools_called", []),
    }



_chat_graph = None


def _get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph


async def ainvoke_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    user_id: str = "",
    target_file: str = "",
    cursor_line: int = 0,
    active_function: str = "",
    selected_text: str = "",
    visible_range: list | None = None,
    active_module: str = "",
    branch: str = "",
    active_repository: str = "",
    **services: Any,
) -> Dict[str, Any]:
    """Async chat invocation for FastAPI and async LangGraph nodes."""
    graph = _get_chat_graph()
    initial = _initial_state(
        message=message,
        project_path=project_path,
        session_id=session_id,
        user_id=user_id,
        target_file=target_file,
        cursor_line=cursor_line,
        active_function=active_function,
        selected_text=selected_text,
        visible_range=visible_range,
        active_module=active_module,
        branch=branch,
        active_repository=active_repository,
        **services,
    )

    start = time.time()
    result = await graph.ainvoke(initial)
    result.setdefault("stats", {})
    result["stats"]["elapsed"] = round(time.time() - start, 2)
    return result


def invoke_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    target_file: str = "",
    **services: Any,
) -> Dict[str, Any]:
    """Synchronous wrapper for CLI usage."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            ainvoke_chat(
                message=message,
                project_path=project_path,
                session_id=session_id,
                target_file=target_file,
                **services,
            )
        )

    raise RuntimeError(
        "invoke_chat() cannot be called from an active event loop. Use await ainvoke_chat()."
    )


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _strip_code_fences(text: str) -> str:
    """Retire les blocs de code ```...``` en gardant la prose autour.
    Utilisé pour la génération : le code passe par l'événement `code`, pas la réponse."""
    import re
    stripped = re.sub(r"```[\w+.-]*\n.*?```", "", text, flags=re.DOTALL)
    # Nettoyage : séparateurs '---' orphelins et lignes vides multiples.
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip().lstrip("-").strip()


async def stream_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    user_id: str = "",
    target_file: str = "",
    cursor_line: int = 0,
    active_function: str = "",
    selected_text: str = "",
    visible_range: list | None = None,
    active_module: str = "",
    branch: str = "",
    active_repository: str = "",
    scope: str = "extension",
    attached_files: list | None = None,
    **services: Any,
):
    """
    Streaming equivalent of ainvoke_chat().

    Yields SSE data events:
      - {type: status, content: str}
      - {type: token, content: str}
      - {type: code, content: str}
      - {type: done, ...metadata}
      - {type: error, content: str}
    """
    graph = _get_chat_graph()
    initial = _initial_state(
        message=message,
        project_path=project_path,
        session_id=session_id,
        user_id=user_id,
        target_file=target_file,
        cursor_line=cursor_line,
        active_function=active_function,
        selected_text=selected_text,
        visible_range=visible_range,
        active_module=active_module,
        branch=branch,
        active_repository=active_repository,
        scope=scope,
        attached_files=attached_files,
        **services,
    )

    started_at = time.time()
    final_session_id = session_id or ""
    final_intent = "question"
    final_target_file = target_file or ""
    final_context_level = "context"
    final_response = ""
    final_context_sources: list = []

    # Génération de code : le code est livré UNE seule fois, via l'événement `code`.
    # On supprime donc (1) le streaming de tokens pendant la génération et (2) le code
    # dans la réponse finale — sinon il apparaît 3 fois dans l'UI de l'extension.
    _GEN_NODES = ("generate_completion", "generate_class")
    _GEN_INTENTS = ("complete_fn", "new_class", "code_generation")
    in_generation = False

    yield _sse({"type": "status", "content": "Starting...", "elapsed_ms": 0})

    node_status = {
        "load_memory":          "Loading conversation memory...",
        "intent_router":        "Detecting intent...",
        "decision_agent":       "Planning multi-agent strategy...",
        "load_file_context":    "Reading file context...",
        "project_summary":      "Summarizing project...",
        "rag_retrieve":         "Searching codebase (RAG)...",
        "parallel_context":     "Loading context (parallel)...",
        "fast_answer":          "Generating response...",
        "answer_question":      "Generating response...",
        "git_question":         "Analyzing git history...",
        "ci_question":          "Analyzing CI/CD pipeline...",
        "semantic_recall":      "Recalling semantic memory...",
        "tool_calling_answer":  "Running tools...",
        "load_project_patterns":"Detecting project conventions...",
        "generate_completion":  "Generating code...",
        "generate_class":       "Generating class...",
        "validate_generated":   "Validating generated code...",
        "memory_save":          "Saving conversation...",
    }

    # Human-readable label sent after decision_agent reveals the chosen path
    _intent_status = {
        "explain":         "Explaining code...",
        "question":        "Searching context & reasoning...",
        "security_review": "Running security analysis...",
        "test_generation": "Preparing test generation...",
        "git_question":    "Analyzing git...",
        "ci_question":     "Analyzing CI/CD...",
        "complete_fn":     "Analyzing code to complete...",
        "new_class":       "Analyzing project conventions...",
        "code_generation": "Preparing code generation...",
    }

    try:
        async for event in graph.astream_events(initial, version="v1"):
            kind = event.get("event", "")
            node_name = event.get("name", "")
            data = event.get("data", {}) or {}

            if kind == "on_chain_start":
                if node_name in _GEN_NODES:
                    in_generation = True   # début génération → on coupe le streaming de tokens
                if node_name in node_status:
                    yield _sse({
                        "type":       "status",
                        "node":       node_name,
                        "content":    node_status[node_name],
                        "elapsed_ms": round((time.time() - started_at) * 1000),
                    })

            elif kind == "on_chat_model_stream":
                # Pendant la génération, le code est livré via l'événement `code` uniquement.
                if in_generation:
                    continue
                chunk = data.get("chunk")
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(str(x) for x in content)
                if content:
                    yield _sse({"type": "token", "content": content})

            elif kind == "on_chain_end":
                output = data.get("output", {})

                if node_name == "load_memory" and isinstance(output, dict):
                    final_session_id = output.get("session_id", final_session_id)

                elif node_name == "decision_agent" and isinstance(output, dict):
                    final_intent = output.get("intent", final_intent)
                    final_context_level = output.get("context_level", final_context_level)
                    elapsed = round((time.time() - started_at) * 1000)
                    yield _sse({
                        "type":            "plan",
                        "intent":          final_intent,
                        "context_level":   final_context_level,
                        "selected_agents": output.get("selected_agents", []),
                        "elapsed_ms":      elapsed,
                    })
                    # Immediately tell the client what path was chosen
                    desc = _intent_status.get(final_intent, "Processing...")
                    yield _sse({"type": "status", "content": desc, "elapsed_ms": elapsed})

                elif node_name == "load_file_context" and isinstance(output, dict):
                    final_target_file = output.get("target_file", final_target_file)

                elif node_name in _GEN_NODES and isinstance(output, dict):
                    in_generation = False   # fin génération → on ré-autorise le streaming
                    if output.get("generated_code"):
                        yield _sse({"type": "code", "content": output["generated_code"]})

                elif node_name == "format_response" and isinstance(output, dict):
                    final_response = output.get("formatted_response", final_response)
                    final_context_sources = output.get("context_sources") or final_context_sources

        # Génération : le code est déjà livré via l'événement `code` — on le retire de la
        # réponse finale pour ne pas l'afficher une seconde fois (on garde la prose/validation).
        if final_intent in _GEN_INTENTS and final_response:
            final_response = _strip_code_fences(final_response)

        yield _sse({
            "type": "done",
            "session_id": final_session_id,
            "intent": final_intent,
            "target_file": final_target_file,
            "context_level": final_context_level,
            "elapsed_seconds": round(time.time() - started_at, 2),
            "response": final_response,
            "context_sources": final_context_sources,
        })

    except Exception as e:
        logger.exception("stream_chat failed: %s", e)
        yield _sse({"type": "error", "content": str(e)})
