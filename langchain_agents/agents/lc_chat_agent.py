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
    ) -> None:
        self.memory.save_exchange(session_id, user_message, response, metadata)

  

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

  

    def _build_fast_prompt(self, state: Dict[str, Any]) -> str:
        file_code = state.get("file_code", "")
        target_file = state.get("target_file", "")
        language = state.get("target_lang", "unknown")
        deps = state.get("dependencies", [])[:8]
        dependents = state.get("dependents", [])[:8]
        question = state.get("user_message", "")

        intent_params = state.get("intent_params") or {}
        method_hint = intent_params.get("method_hint", "")

        file_excerpt = file_code[:4500] if file_code else ""

        method_focus = ""
        if method_hint:
            method_focus = (
                f"The developer asks specifically about `{method_hint}`. "
                f"Focus on that method if it exists in the file.\n"
            )

        return f"""
You are Code Auditor ChatAgent.

Fast mode:
- Answer quickly.
- Use ONLY the file code and lightweight dependency context below.
- Do not invent RAG sources, CI status, Git status, or hidden files.
- Be practical for a developer.
- Answer in the same language as the developer.

Required structure:
1. Role
2. Main flow
3. Dependencies
4. Risks
5. Suggestions

Target file: {target_file}
Language: {language}
{method_focus}

Dependencies used by target:
{deps}

Files depending on target:
{dependents}

File code:
```{language}
{file_excerpt}
```

Developer question:
{question}
"""

    def _build_context_prompt(self, state: Dict[str, Any]) -> tuple[ChatPromptTemplate, Dict[str, Any]]:
        history = state.get("history", [])[-8:]
        rag_docs = state.get("rag_docs", [])[:6]
        project_summary = state.get("project_summary", {})
        file_code = state.get("file_code", "")
        file_analysis = state.get("file_analysis", {})
        deps = state.get("dependencies", [])
        dependents = state.get("dependents", [])

        history_text = "\n".join(
            f"{h.get('role', '?')}: {h.get('content', '')[:800]}"
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

        file_excerpt = file_code[:3500] if file_code else ""

        intent_params = state.get("intent_params") or {}
        method_hint = intent_params.get("method_hint", "")
        method_focus = (
            f"\nFOCUS: The developer is asking specifically about the method "
            f"`{method_hint}`. Locate it in the file excerpt and explain ONLY that "
            f"method: role, parameters, return value, side effects, risks.\n"
            if method_hint
            else ""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Code Auditor Assistant, embedded in a live coding environment.\n\n"
                    "PERSONA:\n"
                    "- You are a senior developer who knows this specific project deeply\n"
                    "- You are direct, concrete, and never repeat information the developer already knows\n"
                    "- You format code properly with language-tagged code fences\n"
                    "- You warn about risks proactively if you see them\n\n"
                    "CONTEXT RULES:\n"
                    "- ONLY use information from the project context provided below\n"
                    "- If you don't have enough context, say so explicitly\n"
                    "- Never invent imports, classes, methods, files, CI status, or Git status\n"
                    "- When explaining code, structure: purpose → key logic → risks → suggestions\n\n"
                    "CURRENT PROJECT CONTEXT:\n"
                    "{project_summary}\n\n"
                    "DEPENDENCY MAP for {target_file}:\n"
                    "- Uses: {deps}\n"
                    "- Used by: {dependents}\n\n"
                    "If {target_file} is mentioned, you know its exact content — don't guess.\n"
                    "{method_focus}",
                ),
                (
                    "human",
                    "Conversation history:\n"
                    "{history}\n\n"
                    "Target file: {target_file} ({language})\n\n"
                    "Cached analysis:\n"
                    "{analysis}\n\n"
                    "File content:\n"
                    "```{language}\n"
                    "{file_excerpt}\n"
                    "```\n\n"
                    "RAG knowledge:\n"
                    "{docs}\n\n"
                    "Developer question: {question}\n\n"
                    "Answer in the same language as the developer. Be specific to this codebase.",
                ),
            ]
        )

        inputs = {
            "project_summary": project_summary,
            "history": history_text or "(none)",
            "target_file": state.get("target_file", ""),
            "language": state.get("target_lang", "unknown"),
            "deps": deps[:10],
            "dependents": dependents[:10],
            "analysis": analysis_text or "(none)",
            "file_excerpt": file_excerpt,
            "docs": docs_text or "(no RAG docs found)",
            "question": state.get("user_message", ""),
            "method_focus": method_focus,
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
