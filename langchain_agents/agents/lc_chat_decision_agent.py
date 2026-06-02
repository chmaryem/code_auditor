"""
lc_chat_decision_agent.py — LLM-powered Decision Agent for ChatGraph.

Architecture v2 :
  - Primary : Gemini Flash / GPT-4o-mini LLM call (~50 tokens, <0.5s)
  - Fallback : deterministic regex (original v1 logic, kept as safety net)

The LLM call returns a structured JSON plan with:
  - intent       : explain | complete_fn | new_class | git_question |
                   ci_question | test_generation | question | code_generation
  - target_file  : file hint from message
  - target_symbol: function/class name
  - context_level: fast | context | deep
  - needs_git    : bool
  - needs_ci     : bool
  - needs_generation: bool
  - confidence   : 0.0-1.0
  - reason       : one-line explanation

Improvements over v1:
  - Understands ambiguous messages that fool keyword matching
  - Uses cursor_line + active_function + selected_text for precision
  - Uses last 3 history turns for follow-up detection
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── LLM Decision Prompt ───────────────────────────────────────────────────────

_DECISION_PROMPT = """\
You are a routing agent for a code assistant embedded in an IDE.
Your ONLY job is to classify the developer's message and return a JSON routing plan.

Developer message: {message}

Active file: {target_file}
Active function (cursor): {active_function}
Selected text: {selected_text}
Cursor line: {cursor_line}

Last 3 conversation turns:
{history_snippet}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "intent": "<one of: explain|complete_fn|new_class|code_generation|git_question|ci_question|test_generation|question>",
  "target_file": "<filename or empty string>",
  "target_symbol": "<function or class name, or empty string>",
  "generation_target": "<name to generate, or empty string>",
  "context_level": "<fast|context|deep>",
  "needs_git": <true|false>,
  "needs_ci": <true|false>,
  "needs_rag": <true|false>,
  "needs_generation": <true|false>,
  "needs_tests": <true|false>,
  "confidence": <0.0 to 1.0>,
  "reason": "<one sentence>"
}}

Intent guide:
- explain       : user asks what code does, how it works, summarize
- complete_fn   : user asks to complete/implement an existing function stub
- new_class     : user asks to create a new class from scratch
- code_generation: user asks to write/generate code (not a specific class or fn)
- git_question  : about commits, branches, merges, PRs, conflicts, diffs
- ci_question   : about CI/CD pipeline, builds, deployments, GitHub Actions, SonarCloud
- test_generation: generate or suggest unit tests
- question      : general project Q&A, architecture, dependencies, risks

context_level guide:
- fast    : simple explain, 1 file, no RAG needed
- context : needs file + deps + RAG
- deep    : multi-file, git + CI context needed
"""


def _build_history_snippet(history: List[Dict[str, Any]]) -> str:
    """Last 3 turns formatted for the LLM prompt."""
    if not history:
        return "(no previous conversation)"
    lines = []
    for turn in history[-3:]:
        role = turn.get("role", "?")
        content = (turn.get("content", "") or "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _call_decision_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """Call a fast LLM (Gemini Flash preferred) to get routing JSON."""
    try:
        from services.llm_factory import invoke_with_fallback
        raw = invoke_with_fallback(prompt, label="chat_decision", max_tokens=256)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        return json.loads(raw)
    except Exception as e:
        logger.debug("LLM decision failed: %s", e)
        return None


class LCChatDecisionAgent:
    """
    LLM-powered Decision Agent (v2).

    Primary path  : LLM call → structured JSON plan
    Fallback path : regex/keyword deterministic routing (v1)
    """

    SUPPORTED_LANGS = ["python", "java", "javascript", "typescript"]

    def decide(
        self,
        user_message: str,
        target_file: str = "",
        base_intent: str = "question",
        intent_params: Dict[str, Any] | None = None,
        conversation_history: List[Dict[str, Any]] | None = None,
        # ── New cursor context ──────────────────────────────────
        cursor_line: int = 0,
        active_function: str = "",
        selected_text: str = "",
    ) -> Dict[str, Any]:
        msg_raw = user_message or ""
        params  = intent_params or {}
        history = conversation_history or []

        # ── Attempt LLM routing ──────────────────────────────────────────────
        plan = self._decide_llm(
            msg_raw, target_file, base_intent, params, history,
            cursor_line, active_function, selected_text,
        )

        # ── Fallback to regex if LLM failed or low confidence ───────────────
        if plan is None or plan.get("confidence", 0.0) < 0.5:
            logger.debug("Decision LLM skipped — using regex fallback")
            plan = self._decide_regex(msg_raw, target_file, base_intent, params, history)
            plan["_routing"] = "regex"
        else:
            plan["_routing"] = "llm"

        # ── Enrich with cursor context ───────────────────────────────────────
        if active_function and not plan.get("target_symbol"):
            plan["target_symbol"] = active_function
        if selected_text and not plan.get("target_file"):
            # If user selected code, work with current file
            plan["target_file"] = target_file or plan.get("target_file", "")

        return plan

    # ── LLM routing ──────────────────────────────────────────────────────────

    def _decide_llm(
        self,
        message: str,
        target_file: str,
        base_intent: str,
        params: Dict[str, Any],
        history: List[Dict[str, Any]],
        cursor_line: int,
        active_function: str,
        selected_text: str,
    ) -> Optional[Dict[str, Any]]:
        # Skip LLM for pure Phase 2 intents — already classified upstream
        if base_intent in ("complete_fn", "new_class"):
            return None

        prompt = _DECISION_PROMPT.format(
            message         = message[:800],
            target_file     = target_file or "(none)",
            active_function = active_function or "(none)",
            selected_text   = (selected_text[:200] + "...") if len(selected_text) > 200 else selected_text or "(none)",
            cursor_line     = cursor_line or "(unknown)",
            history_snippet = _build_history_snippet(history),
        )

        raw_plan = _call_decision_llm(prompt)
        if not raw_plan or "intent" not in raw_plan:
            return None

        # Normalize to internal plan format
        intent = raw_plan.get("intent", "question")
        return {
            "intent":             intent,
            "target_file":        raw_plan.get("target_file", target_file) or target_file,
            "target_symbol":      raw_plan.get("target_symbol", "") or "",
            "generation_target":  raw_plan.get("generation_target", "") or "",
            "agents":             self._agents_for_intent(intent),
            "context_level":      raw_plan.get("context_level", "context"),
            "needs_file":         intent not in ("git_question", "ci_question"),
            "needs_project_summary": True,
            "needs_rag":          bool(raw_plan.get("needs_rag", True)),
            "needs_git":          bool(raw_plan.get("needs_git", False)),
            "needs_ci":           bool(raw_plan.get("needs_ci", False)),
            "needs_generation":   bool(raw_plan.get("needs_generation", False)),
            "needs_tests":        bool(raw_plan.get("needs_tests", False)),
            "needs_validation":   bool(raw_plan.get("needs_generation", False)),
            "safe_mode":          True,
            "confidence":         float(raw_plan.get("confidence", 0.8)),
            "reason":             raw_plan.get("reason", "LLM routing"),
        }

    @staticmethod
    def _agents_for_intent(intent: str) -> List[str]:
        return {
            "explain":          ["code_agent", "chat_agent"],
            "complete_fn":      ["code_generation_agent", "validator_agent"],
            "new_class":        ["code_generation_agent", "validator_agent"],
            "code_generation":  ["code_generation_agent", "validator_agent"],
            "git_question":     ["git_agent", "analysis_agent", "chat_agent"],
            "ci_question":      ["ci_agent", "retriever_agent", "chat_agent"],
            "test_generation":  ["test_agent", "retriever_agent", "validator_agent"],
            "question":         ["retriever_agent", "chat_agent"],
        }.get(intent, ["retriever_agent", "chat_agent"])

    # ── Regex fallback (original v1 logic — kept as safety net) ─────────────

    def _decide_regex(
        self,
        msg_raw: str,
        target_file: str,
        base_intent: str,
        params: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        msg = msg_raw.lower()

        recent_files  = self._extract_recent_files(history)
        recent_intent = self._last_intent(history)

        resolved_target = target_file or (recent_files[-1] if recent_files else "")

        plan: Dict[str, Any] = {
            "intent":             base_intent or "question",
            "target_file":        resolved_target or params.get("file_hint", ""),
            "target_symbol":      params.get("method_hint", ""),
            "generation_target":  params.get("generation_target", ""),
            "agents":             ["retriever_agent", "chat_agent"],
            "context_level":      "context",
            "needs_file":         True,
            "needs_project_summary": True,
            "needs_rag":          True,
            "needs_git":          False,
            "needs_ci":           False,
            "needs_generation":   False,
            "needs_tests":        False,
            "needs_validation":   False,
            "safe_mode":          True,
            "confidence":         0.6,
            "reason":             "regex fallback",
        }

        if self._contains_word(msg, [
            "ci/cd", "pipeline", "github actions", "github action", "workflow",
            "build failed", "test failed", "sonar", "quality gate",
            "deploy", "deployment", "rollback", "release", "staging", "production",
        ]):
            plan.update({"intent": "ci_question", "agents": ["ci_agent", "retriever_agent", "chat_agent"],
                         "context_level": "deep", "needs_file": False, "needs_ci": True,
                         "reason": "CI/CD keyword"})
            return plan

        if self._contains_word(msg, [
            "commit", "merge", "branch", "pull request", "pr", "conflict",
            "rebase", "stash", "diff", "safe to merge", "can i merge",
            "est-ce que je peux merge", "est-ce que je peux commit", "résume mes changements",
        ]):
            plan.update({"intent": "git_question", "agents": ["git_agent", "analysis_agent", "chat_agent"],
                         "context_level": "deep", "needs_file": False, "needs_git": True,
                         "reason": "Git keyword"})
            return plan

        if self._contains_word(msg, [
            "generate test", "generate tests", "génère test", "générer test",
            "tests manquants", "missing tests", "pytest", "junit", "jest",
            "unit test", "test coverage", "coverage",
        ]):
            plan.update({"intent": "test_generation", "agents": ["test_agent", "retriever_agent", "validator_agent"],
                         "needs_generation": True, "needs_tests": True, "needs_validation": True,
                         "reason": "test generation keyword"})
            return plan

        if base_intent in ("complete_fn", "new_class"):
            target = (params.get("generation_target") or params.get("method_hint")
                      or self._extract_generation_target(msg_raw))
            plan.update({"intent": base_intent, "target_symbol": target, "generation_target": target,
                         "agents": ["code_generation_agent", "validator_agent"],
                         "needs_file": base_intent == "complete_fn", "needs_generation": True,
                         "needs_validation": True, "reason": f"Phase 2: {base_intent}"})
            return plan

        if self._contains_word(msg, [
            "complete", "complète", "implement", "implémente", "write the body",
            "fill in", "create class", "generate class", "crée une classe",
            "génère une classe", "new class",
        ]):
            target = (params.get("generation_target") or params.get("method_hint")
                      or self._extract_generation_target(msg_raw))
            plan.update({"intent": "code_generation", "target_symbol": target, "generation_target": target,
                         "agents": ["code_generation_agent", "validator_agent"],
                         "needs_generation": True, "needs_validation": True,
                         "reason": "code generation keyword"})
            return plan

        if (recent_intent in ("explain_code", "contextual_code_question", "explain")
                and self._contains_word(msg, ["dependencies", "dépendances", "risks", "risques",
                                              "impact", "details", "more", "plus", "pourquoi"])):
            plan.update({"intent": "contextual_code_question",
                         "agents": ["code_agent", "retriever_agent", "chat_agent"],
                         "reason": "follow-up from history"})
            return plan

        if base_intent == "explain" or self._contains_word(msg, [
            "explain", "explique", "what does", "que fait", "résume", "resume",
            "describe", "décrire", "comment fonctionne", "understand",
        ]):
            plan.update({"intent": "explain_code", "agents": ["code_agent", "chat_agent"],
                         "context_level": "fast", "needs_project_summary": False, "needs_rag": False,
                         "reason": "explain keyword — fast path"})
            return plan

        if self._contains_word(msg, [
            "risk", "risque", "impact", "impacted", "depend", "dépend",
            "used by", "where is used", "refactor", "architecture", "coupling",
        ]):
            plan.update({"intent": "contextual_code_question",
                         "agents": ["code_agent", "retriever_agent", "chat_agent"],
                         "reason": "risk/impact keyword"})
            return plan

        return plan

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _contains_word(text: str, keywords: list[str]) -> bool:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _extract_generation_target(text: str) -> str:
        raw = text or ""
        patterns = [
            r"(?:complete|complète|finish|implement|implémente|remplis|fill in|écris|develop|développe)\s+(?:the\s+)?(?:function\s+|method\s+|méthode\s+)?`?([\w]+)`?",
            r"(?:create|generate|crée|génère|générer|build|make)\s+(?:a\s+|une\s+)?(?:class\s+|classe\s+)?`?([A-Z][\w]*)`?",
            r"(?:create|generate|crée|génère|générer|build|make)\s+(?:a\s+|une\s+)?`?([A-Z][\w]*)`?\s+(?:class|classe)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _extract_recent_files(history: list[Dict[str, Any]]) -> list[str]:
        files = []
        for turn in history[-8:]:
            meta = turn.get("metadata", {}) or {}
            if meta.get("target_file"):
                files.append(meta["target_file"])
        return files

    @staticmethod
    def _last_intent(history: list[Dict[str, Any]]) -> str:
        for turn in reversed(history):
            if turn.get("role") == "assistant":
                meta = turn.get("metadata", {}) or {}
                if meta.get("intent"):
                    return meta["intent"]
        return ""


chat_decision_agent = LCChatDecisionAgent()
