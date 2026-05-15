"""
lc_chat_agent.py — LangChain ChatAgent Phase 1.

Role:
  Thin conversational layer over existing Code Auditor systems.

Phase 1 supports:
  - project-aware Q&A
  - explain file/class/function questions
  - Redis conversation memory
  - RAG + cached analysis + dependency context
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        # Fallback to existing non-LangChain factory
        return None


class LCChatAgent:
    """
    ChatAgent Phase 1.

    Anatomy:
      - LLM:      OpenRouter → Gemini fallback when available
      - Tools:    intent routing, file context, RAG retrieval, project summary
      - Memory:   ChatMemoryService Redis
      - Planning: simple deterministic routing: question/explain
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

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _build_chat_llm()
        return self._llm

    # ── Routing ─────────────────────────────────────────────────────────────

    def detect_intent(self, user_message: str, target_file: str = "") -> Dict[str, Any]:
        return tool_chat_detect_intent.invoke({
            "user_message": user_message,
            "target_file": target_file or "",
        })

    # ── Memory ──────────────────────────────────────────────────────────────

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

    # ── Context ─────────────────────────────────────────────────────────────

    def load_file_context(
        self,
        project_path: str,
        target_file: str,
        user_message: str,
    ) -> Dict[str, Any]:
        return tool_chat_load_file_context.invoke({
            "project_path": project_path,
            "target_file": target_file or "",
            "user_message": user_message or "",
        })

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
        return tool_chat_rag_retrieve.invoke({
            "project_path": project_path,
            "query": query,
            "target_file": target_file or "",
            "file_code": file_code or "",
            "language": language or "unknown",
        })

    # ── Answer generation ──────────────────────────────────────────────────

    def answer(self, state: Dict[str, Any]) -> str:
        """
        Generate final conversational answer from already-loaded state.
        """
        history = state.get("history", [])[-8:]
        rag_docs = state.get("rag_docs", [])[:6]
        project_summary = state.get("project_summary", {})
        file_code = state.get("file_code", "")
        file_analysis = state.get("file_analysis", {})
        deps = state.get("dependencies", [])
        dependents = state.get("dependents", [])

        history_text = "\n".join(
            f"{h.get('role','?')}: {h.get('content','')[:800]}"
            for h in history
        )

        docs_text = "\n\n".join(
            f"[DOC {i+1}] {d.get('content','')[:1200]}"
            for i, d in enumerate(rag_docs)
        )

        analysis_text = ""
        if isinstance(file_analysis, dict):
            analysis_text = str(file_analysis.get("analysis", ""))[:1800]
        else:
            analysis_text = str(file_analysis)[:1800]

        file_excerpt = file_code[:3500] if file_code else ""

        # method_hint: extracted from "ClassName.methodName" in the user message
        intent_params = state.get("intent_params") or {}
        method_hint   = intent_params.get("method_hint", "")
        method_focus  = (
            f"\nFOCUS: The developer is asking specifically about the method "
            f"`{method_hint}`. Locate it in the file excerpt and explain ONLY that "
            f"method (role, parameters, return value, side-effects, risks).\n"
            if method_hint else ""
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are Code Auditor ChatAgent, a project-aware developer assistant.\n"
             "You answer ONLY using the project context, RAG documents, cached analysis, "
             "dependency context and conversation history provided.\n"
             "Supported project languages: Java, Python, JavaScript, TypeScript.\n"
             "Be precise, practical, and mention uncertainty when context is missing.\n"
             "Do not invent files, imports, frameworks or CI status.\n"
             "When explaining code, structure the answer as: role, flow, dependencies, risks, suggestions.\n"
             "{method_focus}"),
            ("human",
             "Project summary:\n{project_summary}\n\n"
             "Conversation history:\n{history}\n\n"
             "Target file: {target_file}\n"
             "Language: {language}\n\n"
             "Dependencies used by target:\n{deps}\n\n"
             "Dependents impacted by target:\n{dependents}\n\n"
             "Cached analysis:\n{analysis}\n\n"
             "File excerpt:\n```{language}\n{file_excerpt}\n```\n\n"
             "RAG documents:\n{docs}\n\n"
             "Developer question:\n{question}\n\n"
             "Answer in the same language as the developer when possible.")
        ])

        inputs = {
            "project_summary": project_summary,
            "history":         history_text or "(none)",
            "target_file":     state.get("target_file", ""),
            "language":        state.get("target_lang", "unknown"),
            "deps":            deps[:10],
            "dependents":      dependents[:10],
            "analysis":        analysis_text or "(none)",
            "file_excerpt":    file_excerpt,
            "docs":            docs_text or "(no RAG docs found)",
            "question":        state.get("user_message", ""),
            "method_focus":    method_focus,
        }

        try:
            if self.llm is not None:
                chain = prompt | self.llm | StrOutputParser()
                return chain.invoke(inputs)
        except Exception as e:
            logger.error("ChatAgent LLM answer failed: %s", e)

        # Non-LangChain fallback using existing llm_factory
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


    def complete_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Generate the body of an incomplete function.

        Reads `intent_params.generation_target` as the function name,
        builds a focused prompt with the full file + conventions + RAG docs,
        and returns the generated code in state.
        """
        from services.code_generator_service import (
            build_completion_prompt, extract_code_blocks, suggest_file_path,
        )

        intent_params = state.get("intent_params") or {}
        fn_name   = (
            intent_params.get("generation_target")
            or intent_params.get("method_hint")
            or ""
        )
        file_path    = state.get("target_file", "")
        file_code    = state.get("file_code", "")
        language     = state.get("target_lang") or state.get("generation_language") or "python"
        conventions  = state.get("project_patterns") or {}
        rag_docs     = state.get("rag_docs") or []
        dependencies = state.get("dependencies") or []
        history      = state.get("history", [])[-6:]

        history_text = "\n".join(
            f"{h.get('role','?')}: {h.get('content','')[:500]}" for h in history
        )

        if not fn_name:
            return {
                "generated_code": "",
                "response": (
                    "Je n'ai pas pu identifier le nom de la fonction à compléter. "
                    "Précise le nom, ex : 'complete findByEmail in UserService'."
                ),
                "generation_language": language,
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
        blocks    = extract_code_blocks(generated)
        code      = blocks[0]["code"] if blocks else generated

        # Build a human-readable response with the code block
        validation_note = (
            f"\n\n> ⚠️ Validate the generated code manually — syntax checking was skipped "
            f"for {language}."
            if language not in ("python",) else ""
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
            "generated_code":     code,
            "generation_language": language,
            "generation_target":  fn_name,
            "response":           response,
            "code_blocks":        blocks,
        }

    def generate_class(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Generate a new class from scratch following project conventions.

        Reads `intent_params.generation_target` as the class name.
        """
        from services.code_generator_service import (
            build_class_prompt, extract_code_blocks, extract_suggested_file, suggest_file_path,
        )

        intent_params = state.get("intent_params") or {}
        class_name   = intent_params.get("generation_target") or ""
        language     = state.get("target_lang") or state.get("generation_language") or "python"
        conventions  = state.get("project_patterns") or {}
        rag_docs     = state.get("rag_docs") or []
        project_path = state.get("project_path", ".")
        history      = state.get("history", [])[-6:]
        description  = state.get("user_message", "")

        history_text = "\n".join(
            f"{h.get('role','?')}: {h.get('content','')[:500]}" for h in history
        )

        if not class_name:
            return {
                "generated_code": "",
                "response": (
                    "Je n'ai pas pu identifier le nom de la classe à générer. "
                    "Précise le nom, ex : 'create a ProductService class'."
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

        generated    = self._call_llm_raw(prompt_text, label="new_class")
        blocks       = extract_code_blocks(generated)
        code         = blocks[0]["code"] if blocks else generated
        suggested    = extract_suggested_file(generated) or suggest_file_path(class_name, language, project_path)

        response = (
            f"## ✅ Class `{class_name}` — Generated\n\n"
            f"**Language:** {language}  \n"
            f"**Suggested file:** `{suggested}`\n\n"
            f"```{language}\n{code}\n```\n\n"
            f"To write to disk:\n"
            f"```bash\npython main.py chat --new-class '{class_name}' "
            f"--lang {language} --project . --write\n```"
        )

        return {
            "generated_code":      code,
            "generation_language": language,
            "generation_target":   class_name,
            "response":            response,
            "code_blocks":         blocks,
            "suggested_files":     [suggested],
        }

    def _call_llm_raw(self, prompt_text: str, label: str = "chat") -> str:
        """Call LLM with raw text prompt — used by Phase 2 generation methods."""
        try:
            if self.llm is not None:
                from langchain_core.messages import HumanMessage
                return self.llm.invoke([HumanMessage(content=prompt_text)]).content
        except Exception as e:
            logger.error("%s LLM call failed: %s", label, e)

        try:
            from services.llm_factory import invoke_with_fallback
            return invoke_with_fallback(prompt_text, label=label)
        except Exception as e:
            logger.error("%s fallback failed: %s", label, e)

        return f"[{label}] LLM unavailable — check API keys."


lc_chat_agent = LCChatAgent()
