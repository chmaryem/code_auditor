"""
lc_chat_agent.py — LangChain ChatAgent.

Role:
  Thin conversational layer over existing Code Auditor systems.

Supports:
  - project-aware Q&A
  - explain file/class/function questions
  - fast explain mode for better VS Code UX
  - Redis conversation memory
  - RAG + cached analysis + dependency context
  - async answering for SSE streaming
  - Phase 2 code generation:
      * complete function
      * generate class
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_agents.tools.chat_tools import (
    tool_chat_detect_intent,
    tool_chat_load_file_context,
    tool_chat_project_summary,
    tool_chat_rag_retrieve,
)
from services.chat_memory_service import chat_memory_service

logger = logging.getLogger(__name__)


def _build_chat_llm():
    """Reuse the existing LLM cascade/fallback style."""
    try:
        from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
        return _build_llm_with_fallback()
    except Exception:
        return None


class LCChatAgent:
    """
    ChatAgent.

    Anatomy:
      - LLM: OpenRouter/Gemini fallback when available
      - Tools: intent routing, file context, RAG retrieval, project summary
      - Memory: ChatMemoryService Redis
      - Decision: handled by LCChatDecisionAgent inside ChatGraph
    """

    def __init__(self):
        self.memory = chat_memory_service
        self.tools = [
            tool_chat_detect_intent,
            tool_chat_load_file_context,
            tool_chat_project_summary,
            tool_chat_rag_retrieve,
        ]
        self._llm = None
        self._llm_by_level: Dict[str, Any] = {}   # P2 · LLM par context_level

    @property
    def llm(self):
        # Rétro-compat : modèle de niveau "context" (cas normal).
        return self._llm_for_level("context")

    def _llm_for_level(self, level: str = "context"):
        """P2 · Retourne le LLM correspondant au context_level décidé par le
        Decision Agent (fast | context | deep).

        - Si ROUTE_BY_COMPLEXITY=false → renvoie l'unique LLM historique.
        - Sinon → construit (paresseusement, puis cache) un LLM dont le modèle
          primaire dépend du niveau, le mapping étant centralisé dans
          config.api.model_for_level().
        Tombe sur le LLM historique si la construction par niveau échoue.
        """
        from config import config

        if not getattr(config.api, "route_by_complexity", False):
            if self._llm is None:
                self._llm = _build_chat_llm()
            return self._llm

        level = level if level in ("fast", "context", "deep") else "context"
        if level not in self._llm_by_level:
            llm = None
            try:
                from services.llm_factory import build_llm_for_level
                llm = build_llm_for_level(level)
            except Exception as e:
                logger.warning("ChatAgent: build_llm_for_level(%s) failed: %s", level, e)
            self._llm_by_level[level] = llm if llm is not None else _build_chat_llm()
        return self._llm_by_level[level]



    def detect_intent(self, user_message: str, target_file: str = "") -> Dict[str, Any]:
        return tool_chat_detect_intent.invoke(
            {
                "user_message": user_message,
                "target_file": target_file or "",
            }
        )



    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.memory.load_history(session_id)

    def save_exchange(
        self,
        session_id: str,
        user_message: str,
        response: str,
        metadata: Dict[str, Any] | None = None,
        project_path: str = "",
    ) -> None:
        self.memory.save_exchange(session_id, user_message, response, metadata, project_path)

  

    def load_file_context(
        self,
        project_path: str,
        target_file: str,
        user_message: str,
    ) -> Dict[str, Any]:
        return tool_chat_load_file_context.invoke(
            {
                "project_path": project_path,
                "target_file": target_file or "",
                "user_message": user_message or "",
            }
        )

    def project_summary(self, project_path: str) -> Dict[str, Any]:
        return tool_chat_project_summary.invoke({"project_path": project_path})

    def retrieve(
        self,
        project_path: str,
        query: str,
        target_file: str = "",
        file_code: str = "",
        language: str = "unknown",
    ) -> Dict[str, Any]:
        return tool_chat_rag_retrieve.invoke(
            {
                "project_path": project_path,
                "query": query,
                "target_file": target_file or "",
                "file_code": file_code or "",
                "language": language or "unknown",
            }
        )

  

    # ── Prompt builders ────────────────────────────────────────────────────

    @staticmethod
    def _context_header(state: Dict[str, Any]) -> str:
        """Build the shared context header injected into every prompt."""
        parts = []
        repo = state.get("active_repository", "")
        branch = state.get("branch", "")
        module = state.get("active_module", "")
        target = state.get("target_file", "")
        lang = state.get("target_lang", "")

        if repo:
            parts.append(f"Repository: {repo}")
        if branch:
            parts.append(f"Branch: {branch}")
        if module and module not in ("chat", ""):
            parts.append(f"Active module: {module}")
        if target:
            parts.append(f"File: {target}" + (f" ({lang})" if lang and lang != "unknown" else ""))

        return "\n".join(parts) if parts else ""

    def _build_fast_prompt(self, state: Dict[str, Any]) -> str:
        file_code = state.get("file_code", "")
        target_file = state.get("target_file", "")
        language = state.get("target_lang", "unknown")
        deps = state.get("dependencies", [])[:8]
        dependents = state.get("dependents", [])[:8]
        question = state.get("user_message", "")
        intent_params = state.get("intent_params") or {}
        method_hint = intent_params.get("method_hint", "")

        # Security gate (Phase A): redact before sending to external LLM
        try:
            from services.secret_redactor import redact_secrets
            file_code, _ = redact_secrets(file_code)
        except Exception:
            pass

        file_excerpt = file_code[:4500] if file_code else ""
        ctx_header = self._context_header(state)

        method_focus = (
            f"\nFocus on `{method_hint}` — explain its role, parameters, return value, "
            f"side effects, and risks.\n"
            if method_hint else ""
        )

        intent = state.get("intent", "question")
        structure_guide = {
            "explain":    "Purpose → Key logic → Risks → Concrete suggestions",
            "code_analysis": "What it does → How it works → Risks → Improvements",
            "bug_fix":    "Problem identified → Root cause → Fix → Verification",
        }.get(intent, "Direct answer → Key points → Concrete next step")

        return f"""You are Code Auditor AI, an expert developer assistant embedded in a live coding environment.

{f"## Context{chr(10)}{ctx_header}{chr(10)}" if ctx_header else ""}
## Instructions
- Use ONLY the file code and dependency context provided below.
- Never invent imports, classes, methods, CI status, Git status, or files not shown.
- Be concrete and direct — you are talking to a senior developer.
- Cite the exact file path and function name when you reference code.
- Respond in the same language as the developer.
- Structure: {structure_guide}
- Stop as soon as the answer is complete. No sign-offs, no repetition.{method_focus}

## File context
**File:** `{target_file}` | **Language:** {language}
**Dependencies used:** {deps}
**Used by:** {dependents}

```{language}
{file_excerpt}
```

## Developer question
{question}"""

    def _build_context_prompt(self, state: Dict[str, Any]) -> tuple[ChatPromptTemplate, Dict[str, Any]]:
        history = state.get("history", [])[-8:]
        # AI settings from PostgreSQL (injected by node_load_ai_settings)
        max_ctx       = state.get("ai_max_context", 8000)
        response_style = state.get("ai_response_style", "detailed")
        ai_mode       = state.get("ai_mode", "balanced")
        use_rag       = state.get("ai_use_rag", True)

        rag_docs = state.get("rag_docs", [])[:6] if use_rag else []
        project_summary = state.get("project_summary", {})
        file_code = state.get("file_code", "")
        file_analysis = state.get("file_analysis", {})
        deps = state.get("dependencies", [])
        dependents = state.get("dependents", [])
        intent = state.get("intent", "question")

        # Truncate history entries at 1500 chars (was 800) for better follow-up context
        history_text = "\n".join(
            f"{h.get('role', '?')}: {h.get('content', '')[:1500]}"
            for h in history
        )

        docs_text = "\n\n".join(
            f"[DOC {i + 1}] {d.get('content', '')[:1200]}"
            for i, d in enumerate(rag_docs)
        )

        if isinstance(file_analysis, dict):
            analysis_text = str(file_analysis.get("analysis", ""))[:1800]
        else:
            analysis_text = str(file_analysis)[:1800]

        # Security gate (Phase A): redact before sending to external LLM
        try:
            from services.secret_redactor import redact_secrets
            file_code, _ = redact_secrets(file_code)
        except Exception:
            pass

        file_excerpt = file_code[:max_ctx] if file_code else ""

        intent_params = state.get("intent_params") or {}
        method_hint = intent_params.get("method_hint", "")
        method_focus = (
            f"\nFOCUS: Locate `{method_hint}` in the file and explain it specifically: "
            f"role, parameters, return value, side effects, risks.\n"
            if method_hint else ""
        )

        # Dynamic context header
        ctx_header = self._context_header(state)

        # Style directive from user settings
        _style_hints = {
            "concise":       "Be extremely brief. Only essential points, no elaboration.",
            "detailed":      "Be thorough. Cover all relevant aspects with examples.",
            "professional":  "Use formal technical language. Structured sections.",
            "step_by_step":  "Break down into numbered steps. One action per step.",
        }
        _mode_hints = {
            "fast":    "Prioritise speed. Short, direct answers.",
            "strict":  "Be conservative. Flag any uncertainty. No assumptions.",
            "deep":    "Perform deep analysis. Consider edge cases and performance.",
            "balanced":"Balance depth and brevity.",
        }
        style_directive = _style_hints.get(response_style, "")
        mode_directive  = _mode_hints.get(ai_mode, "")

        # Intent-specific response structure
        response_structure = {
            "explain":          "Purpose → Key logic → Dependencies → Risks → Suggestions",
            "code_analysis":    "What it does → How it works → Risks → Improvements",
            "bug_fix":          "Problem → Root cause (cite file:line) → Fix → Verification steps",
            "question":         "Direct answer → Evidence from codebase → Next action",
            "complete_fn":      "Implementation → Explanation → Usage example",
            "new_class":        "Class structure → Key methods → Integration points",
        }.get(intent, "Direct answer → Key points → Concrete next step")

        # Build a rich project summary string
        proj_files = project_summary.get("files", {})
        proj_summary_text = (
            f"Languages: {list(proj_files.keys())}, "
            f"Files: {sum(proj_files.values())} total"
            if proj_files else str(project_summary)[:400]
        )

        system_msg = (
            "You are Code Auditor AI, an expert developer assistant embedded in a live coding environment.\n\n"
            "## Identity\n"
            "You are a senior developer with deep knowledge of this specific project.\n"
            "You reason from real code and real data — never from assumptions.\n\n"
            + (f"## AI Mode: {ai_mode}\n{mode_directive}\n\n" if mode_directive else "")
            + "## Developer context\n"
            f"{ctx_header}\n"
            f"Project: {proj_summary_text}\n\n"
            "## Response rules\n"
            f"- Structure for this question type ({intent}): {response_structure}\n"
            + (f"- Style: {style_directive}\n" if style_directive else "")
            + "- Cite exact file paths and function names when referencing code.\n"
            "- Distinguish CERTAIN (data provided) | PROBABLE (inferred) | TO VERIFY (assumption).\n"
            "- Format code with language-tagged fences.\n"
            "- Warn about risks proactively.\n"
            "- Respond in the same language as the developer.\n\n"
            "## Strict output rules\n"
            "- ONLY use information from the context below — never invent files, imports, CI status, or Git state.\n"
            "- If context is insufficient, say so explicitly and ask for the missing info.\n"
            "- Stop as soon as the answer is complete. No sign-offs, no repetition, no filler.\n"
            f"{method_focus}"
        )

        # Escape literal { } in system_msg so LangChain doesn't treat them as
        # template variables (proj_summary_text can contain dict repr with braces).
        system_msg_safe = system_msg.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg_safe),
                (
                    "human",
                    "## Conversation history\n"
                    "{history}\n\n"
                    "## File context\n"
                    "**File:** `{target_file}` | **Language:** {language}\n"
                    "**Dependencies used:** {deps}\n"
                    "**Used by:** {dependents}\n\n"
                    "**Cached analysis:**\n{analysis}\n\n"
                    "```{language}\n{file_excerpt}\n```\n\n"
                    "## Knowledge base (RAG)\n"
                    "{docs}\n\n"
                    "## Developer question\n"
                    "{question}",
                ),
            ]
        )

        inputs = {
            "history":      history_text or "(no prior conversation)",
            "target_file":  state.get("target_file", ""),
            "language":     state.get("target_lang", "unknown"),
            "deps":         deps[:10],
            "dependents":   dependents[:10],
            "analysis":     analysis_text or "(no cached analysis)",
            "file_excerpt": file_excerpt,
            "docs":         docs_text or "(no RAG documents retrieved)",
            "question":     state.get("user_message", ""),
        }
        return prompt, inputs

    # ── Answer generation ──────────────────────────────────────────────────

    def fast_answer(self, state: Dict[str, Any]) -> str:
        """Fast synchronous answer path. Prefer afast_answer() for API/streaming."""
        if not state.get("file_code", ""):
            return (
                "Je n’ai pas trouvé le contenu du fichier cible.\n\n"
                "Essaie par exemple :\n"
                "- `explique logic.py`\n"
                "- ouvre un fichier supporté puis demande `Explain current file`\n"
            )

        prompt_text = self._build_fast_prompt(state)
        return self._call_llm_raw(prompt_text, label="chat_fast_answer", level="fast")

    async def afast_answer(self, state: Dict[str, Any], config: Any = None) -> str:
        """Async fast answer path for SSE token streaming."""
        if not state.get("file_code", ""):
            return (
                "Je n’ai pas trouvé le contenu du fichier cible.\n\n"
                "Essaie par exemple :\n"
                "- `explique logic.py`\n"
                "- ouvre un fichier supporté puis demande `Explain current file`\n"
            )

        prompt_text = self._build_fast_prompt(state)
        return await self._acall_llm_raw(prompt_text, label="chat_fast_answer", config=config, level="fast")

    def answer(self, state: Dict[str, Any]) -> str:
        """Contextual/deep synchronous answer path."""
        prompt, inputs = self._build_context_prompt(state)
        llm = self._llm_for_level(state.get("context_level", "context"))

        try:
            if llm is not None:
                chain = prompt | llm | StrOutputParser()
                return chain.invoke(inputs)
        except Exception as e:
            logger.error("ChatAgent LLM answer failed: %s", e)

        try:
            from services.llm_factory import invoke_with_fallback

            rendered = prompt.format_messages(**inputs)
            text = "\n".join(getattr(m, "content", str(m)) for m in rendered)
            return invoke_with_fallback(text, label="chat_agent")
        except Exception as e:
            logger.error("ChatAgent fallback answer failed: %s", e)

        return (
            "Je n'ai pas pu appeler le LLM pour répondre. "
            "Le contexte a été chargé, mais vérifie tes clés OpenRouter/Gemini."
        )

    async def aanswer(self, state: Dict[str, Any], config: Any = None) -> str:
        """Async contextual/deep answer path for LangGraph streaming."""
        prompt, inputs = self._build_context_prompt(state)
        llm = self._llm_for_level(state.get("context_level", "context"))

        try:
            if llm is not None:
                chain = prompt | llm | StrOutputParser()
                if config:
                    return await chain.ainvoke(inputs, config=config)
                return await chain.ainvoke(inputs)
        except Exception as e:
            logger.error("ChatAgent LLM aanswer failed: %s", e)

        try:
            from services.llm_factory import invoke_with_fallback

            rendered = prompt.format_messages(**inputs)
            text = "\n".join(getattr(m, "content", str(m)) for m in rendered)
            return await asyncio.to_thread(invoke_with_fallback, text, label="chat_agent")
        except Exception as e:
            logger.error("ChatAgent fallback aanswer failed: %s", e)

        return (
            "Je n'ai pas pu appeler le LLM pour répondre. "
            "Le contexte a été chargé, mais vérifie tes clés OpenRouter/Gemini."
        )

    # ── Phase 2 — Code generation ──────────────────────────────────────────

    def complete_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Generate the body of an incomplete function."""
        from services.code_generator_service import (
            build_completion_prompt,
            extract_code_blocks,
        )

        intent_params = state.get("intent_params") or {}
        fn_name = (
            intent_params.get("generation_target")
            or intent_params.get("method_hint")
            or ""
        )

        file_path = state.get("target_file", "")
        file_code = state.get("file_code", "")

        target_lang = state.get("target_lang") or ""
        generation_language = state.get("generation_language") or ""

        language = (
            generation_language
            or (target_lang if target_lang != "unknown" else "")
            or self._infer_language_from_message(state.get("user_message", ""))
            or "python"
        )

        conventions = state.get("project_patterns") or {}
        rag_docs = state.get("rag_docs") or []
        dependencies = state.get("dependencies") or []
        history = state.get("history", [])[-6:]

        history_text = "\n".join(
            f"{h.get('role', '?')}: {h.get('content', '')[:500]}"
            for h in history
        )

        if not fn_name:
            return {
                "generated_code": "",
                "response": (
                    "Je n'ai pas pu identifier le nom de la fonction à compléter. "
                    "Précise le nom, ex : `complete findByEmail in UserService`."
                ),
                "generation_language": language,
            }

        if not file_code:
            return {
                "generated_code": "",
                "response": (
                    f"Je n’ai pas trouvé le contenu du fichier cible pour compléter `{fn_name}`.\n\n"
                    "Essaie avec un fichier explicite, par exemple :\n"
                    f"`complete {fn_name} in logic.py`"
                ),
                "generation_language": language,
                "generation_target": fn_name,
            }

        prompt_text = build_completion_prompt(
            fn_name=fn_name,
            file_path=file_path,
            file_code=file_code,
            language=language,
            conventions=conventions,
            rag_docs=rag_docs,
            dependencies=dependencies,
            history_text=history_text,
        )

        generated = self._call_llm_raw(prompt_text, label="complete_fn")
        blocks = extract_code_blocks(generated)
        code = blocks[0]["code"] if blocks else generated

        validation_note = (
            f"\n\n> ⚠️ Validate the generated code manually — syntax checking was skipped "
            f"for {language}."
            if language not in ("python",)
            else ""
        )

        response = (
            f"## ✅ Function `{fn_name}` — Generated implementation\n\n"
            f"**File:** `{file_path}`  \n"
            f"**Language:** {language}\n\n"
            f"```{language}\n{code}\n```"
            f"{validation_note}\n\n"
            f"Paste this into `{file_path}` replacing the existing stub."
        )

        return {
            "generated_code": code,
            "generation_language": language,
            "generation_target": fn_name,
            "response": response,
            "code_blocks": blocks,
        }

    def generate_class(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Generate a new class from scratch following project conventions."""
        from services.code_generator_service import (
            build_class_prompt,
            extract_code_blocks,
            extract_suggested_file,
            suggest_file_path,
        )

        intent_params = state.get("intent_params") or {}
        class_name = intent_params.get("generation_target") or ""

        target_lang = state.get("target_lang") or ""
        generation_language = state.get("generation_language") or ""

        language = (
            generation_language
            or (target_lang if target_lang != "unknown" else "")
            or self._infer_language_from_message(state.get("user_message", ""))
            or "python"
        )

        conventions = state.get("project_patterns") or {}
        rag_docs = state.get("rag_docs") or []
        project_path = state.get("project_path", ".")
        history = state.get("history", [])[-6:]
        description = state.get("user_message", "")

        history_text = "\n".join(
            f"{h.get('role', '?')}: {h.get('content', '')[:500]}"
            for h in history
        )

        if not class_name:
            return {
                "generated_code": "",
                "response": (
                    "Je n'ai pas pu identifier le nom de la classe à générer. "
                    "Précise le nom, ex : `create a ProductService class`."
                ),
                "generation_language": language,
            }

        prompt_text = build_class_prompt(
            class_name=class_name,
            language=language,
            description=description,
            project_path=project_path,
            conventions=conventions,
            rag_docs=rag_docs,
            history_text=history_text,
        )

        generated = self._call_llm_raw(prompt_text, label="new_class")
        blocks = extract_code_blocks(generated)
        code = blocks[0]["code"] if blocks else generated
        suggested = (
            extract_suggested_file(generated)
            or suggest_file_path(class_name, language, project_path)
        )

        response = (
            f"## ✅ Class `{class_name}` — Generated\n\n"
            f"**Language:** {language}  \n"
            f"**Suggested file:** `{suggested}`\n\n"
            f"```{language}\n{code}\n```\n\n"
            f"To write to disk safely, use preview/apply flow in the plugin, "
            f"or pass `write_to_disk=true` only from trusted local tooling."
        )

        return {
            "generated_code": code,
            "generation_language": language,
            "generation_target": class_name,
            "response": response,
            "code_blocks": blocks,
            "suggested_files": [suggested],
        }

    # ── Utility methods ────────────────────────────────────────────────────

    def _infer_language_from_message(self, message: str) -> str:
        """Infer generation language from natural user message."""
        msg = (message or "").lower()

        if "java" in msg or "spring" in msg:
            return "java"

        if "typescript" in msg or " ts " in f" {msg} " or ".ts" in msg:
            return "typescript"

        if "javascript" in msg or " js " in f" {msg} " or ".js" in msg:
            return "javascript"

        if "python" in msg or " py " in f" {msg} " or ".py" in msg:
            return "python"

        return ""

    def _call_llm_raw(self, prompt_text: str, label: str = "chat", level: str = "context") -> str:
        """Call LLM with raw text prompt — used by sync fast mode and generation methods."""
        llm = self._llm_for_level(level)
        try:
            if llm is not None:
                from langchain_core.messages import HumanMessage

                result = llm.invoke([HumanMessage(content=prompt_text)])
                return getattr(result, "content", str(result))
        except Exception as e:
            logger.error("%s LLM call failed: %s", label, e)

        try:
            from services.llm_factory import invoke_with_fallback

            return invoke_with_fallback(prompt_text, label=label)
        except Exception as e:
            logger.error("%s fallback failed: %s", label, e)

        return f"[{label}] LLM unavailable — check API keys."

    async def _acall_llm_raw(self, prompt_text: str, label: str = "chat", config: Any = None, level: str = "context") -> str:
        """Async raw LLM call — used by streaming fast mode."""
        llm = self._llm_for_level(level)
        try:
            if llm is not None:
                from langchain_core.messages import HumanMessage

                if config:
                    result = await llm.ainvoke([HumanMessage(content=prompt_text)], config=config)
                else:
                    result = await llm.ainvoke([HumanMessage(content=prompt_text)])
                return getattr(result, "content", str(result))
        except Exception as e:
            logger.error("%s async LLM call failed: %s", label, e)

        try:
            from services.llm_factory import invoke_with_fallback

            return await asyncio.to_thread(invoke_with_fallback, prompt_text, label=label)
        except Exception as e:
            logger.error("%s async fallback failed: %s", label, e)

        return f"[{label}] LLM unavailable — check API keys."


lc_chat_agent = LCChatAgent()
