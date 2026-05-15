"""
chat_tools.py — LangChain tools for ChatAgent Phase 1.

These tools are thin wrappers over the existing Code Auditor systems:
  - file context loading
  - cached analysis lookup
  - project-aware RAG retrieval
  - simple intent routing

No duplicated intelligence: the chat layer calls existing services/agents.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


_LANG_BY_EXT = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

# Regex: explicit filename with known extension  (UserService.java, auth_service.py)
_FILE_RE = re.compile(r"([\w./\\-]+\.(?:java|py|js|jsx|ts|tsx))", re.IGNORECASE)

# Regex: ClassName.methodName reference  (UserService.findByEmail, authService.login)
# Captures the class name part only (before the dot + lowercase word)
_CLASS_METHOD_RE = re.compile(r"\b([A-Z][\w]*)\.([a-z][\w]*)\b")


def _detect_lang(path: str) -> str:
    return _LANG_BY_EXT.get(Path(path).suffix.lower(), "unknown")


def _extract_file_hint(text: str, fallback: str = "") -> dict:
    """Extract a file/class/method reference from free text.

    Returns a dict with three keys:
      file_hint   : best string to pass to _find_file() (filename or class name)
      class_hint  : PascalCase class name if detected (e.g. "CacheService")
      method_hint : method name if detected (e.g. "get_cached_analysis")

    Priority for file_hint:
      1. fallback argument (explicit --file from CLI)
      2. Explicit filename with known extension  (cache_service.py)
      3. ClassName.method pattern                (CacheService.get_cached_analysis)
    """
    if fallback:
        return {"file_hint": fallback, "class_hint": "", "method_hint": ""}

    # 1. Explicit filename with extension
    m = _FILE_RE.search(text or "")
    if m:
        return {"file_hint": m.group(1), "class_hint": "", "method_hint": ""}

    # 2. ClassName.methodName  (CacheService.get_cached_analysis)
    m2 = _CLASS_METHOD_RE.search(text or "")
    if m2:
        return {
            "file_hint":   m2.group(1),   # "CacheService"  → _find_file will resolve
            "class_hint":  m2.group(1),   # "CacheService"
            "method_hint": m2.group(2),   # "get_cached_analysis"
        }

    return {"file_hint": "", "class_hint": "", "method_hint": ""}


def _find_file(project_path: str, hint: str) -> Optional[Path]:
    """Find a source file by path, filename, or class name hint.

    Search order:
      1. Absolute path
      2. Relative to project root
      3. Exact filename match (rglob)
      4. Substring match in filename          (cache → cache_service.py)
      5. PascalCase → extensions              (UserService → UserService.java)
      6. Content scan: look for 'class {hint}' inside every file
         This is the only reliable way to map CacheService → cache_service.py
    """
    if not hint:
        return None

    p = Path(hint)
    if p.exists():
        return p.resolve()

    root = Path(project_path).resolve()
    candidate = root / hint
    if candidate.exists():
        return candidate.resolve()

    # Collect all supported source files (capped for large mono-repos)
    matches: list[Path] = []
    for ext in _LANG_BY_EXT:
        matches.extend(root.rglob(f"*{ext}"))
        if len(matches) > 20_000:
            break

    lower_hint = hint.lower()

    # Level 3 — exact filename match
    for m in matches:
        if m.name.lower() == lower_hint:
            return m.resolve()

    # Level 4 — substring in filename  ("cache" matches cache_service.py)
    for m in matches:
        if lower_hint in m.name.lower():
            return m.resolve()

    # Level 5 — PascalCase class name + every known extension
    # UserService → UserService.java / UserService.py …
    if "." not in hint:
        for ext in _LANG_BY_EXT:
            target_name = f"{hint}{ext}"
            for m in matches:
                if m.name.lower() == target_name.lower():
                    return m.resolve()

    # Level 6 — content scan: search for "class {hint}" inside files
    # Handles PascalCase → snake_case mismatch (CacheService → cache_service.py)
    class_name = hint.split(".")[0]           # "CacheService" (strip method if any)
    class_decl_py   = f"class {class_name}"   # Python / TS/JS
    class_decl_java = f"class {class_name} "  # Java (space before {)
    for m in matches:
        try:
            text = m.read_text(encoding="utf-8", errors="replace")
            if class_decl_py in text or class_decl_java in text:
                return m.resolve()
        except Exception:
            continue

    return None


@tool
def tool_chat_detect_intent(user_message: str, target_file: str = "") -> Dict[str, Any]:
    """Detect the user's chat intent using fast deterministic rules.

    Phase 1: question | explain
    Phase 2: complete_fn | new_class

    Args:
        user_message: raw developer message
        target_file: optional target file

    Returns:
        {"intent": str, "intent_params": dict}
    """
    msg = (user_message or "").lower()

    explain_keywords = [
        "explain", "explique", "شرح", "describe", "what does",
        "que fait", "comment fonctionne", "understand",
    ]
    complete_keywords = [
        "complete", "complète", "finish", "implement", "remplis",
        "fill in", "write the body", "écris", "développe",
    ]
    newclass_keywords = [
        "create a", "generate a", "crée", "génère", "générer",
        "new class", "nouvelle classe", "build a class", "make a class",
    ]

    # Priority: complete_fn > new_class > explain > question
    intent = "question"
    if any(k in msg for k in complete_keywords):
        intent = "complete_fn"
    elif any(k in msg for k in newclass_keywords):
        intent = "new_class"
    elif any(k in msg for k in explain_keywords):
        intent = "explain"

    # Extract file/class/method hints
    extracted = _extract_file_hint(user_message or "", fallback=target_file or "")

    # Generation target extraction
    generation_target = ""
    if intent == "complete_fn":
        # "complete findByEmail in UserService" → target = findByEmail
        m = re.search(
            r"(?:complete|complète|finish|implement|remplis|fill in|écris)\s+(?:the\s+)?(?:method\s+)?`?([\w]+)`?",
            msg,
        )
        generation_target = m.group(1) if m else extracted["method_hint"]

    elif intent == "new_class":
        # "create a ProductService class" → target = ProductService
        m = re.search(
            r"(?:create|generate|crée|génère|générer|build|make)\s+(?:a\s+|une\s+)?`?([A-Z][\w]*)`?",
            user_message or "",
        )
        if m:
            generation_target = m.group(1)
        else:
            # Fallback: any PascalCase word
            m2 = re.search(r"\b([A-Z][a-z][\w]*(?:[A-Z][\w]*)*)\b", user_message or "")
            generation_target = m2.group(1) if m2 else ""

    return {
        "intent": intent,
        "intent_params": {
            "file_hint":         extracted["file_hint"],
            "class_hint":        extracted["class_hint"],
            "method_hint":       extracted["method_hint"],
            "generation_target": generation_target,
        },
    }


@tool
def tool_chat_load_file_context(
    project_path: str,
    target_file: str = "",
    user_message: str = "",
) -> Dict[str, Any]:
    """Load file code + cached analysis + basic dependency context.

    Args:
        project_path: project root
        target_file: optional file path/name
        user_message: raw message, used to extract filename if target_file missing

    Returns:
        context dict with file_code, file_analysis, dependencies, dependents, language.
    """
    extracted  = _extract_file_hint(user_message or "", fallback=target_file or "")
    file_hint  = extracted["file_hint"]
    method_hint = extracted["method_hint"]   # e.g. "get_cached_analysis" — passed to caller
    fp = _find_file(project_path, file_hint) if file_hint else None
    if not fp:
        return {
            "target_file":  "",
            "target_lang":  "unknown",
            "file_code":    "",
            "file_analysis": {},
            "dependencies": [],
            "dependents":   [],
            "method_hint":  method_hint,
            "found": False,
        }

    code = ""
    try:
        code = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        code = f"[READ_ERROR] {e}"

    lang = _detect_lang(str(fp))
    analysis = {}

    # Existing CacheService integration — use the module singleton, not a new instance
    try:
        from services.cache_service import cache_service
        cached = cache_service.get_cached_analysis(fp)
        if cached:
            analysis = cached
    except Exception:
        pass

    # Existing RetrieverAgent neighborhood integration
    dependencies, dependents = [], []
    try:
        from langchain_agents.agents.lc_retriever_agent import lc_retriever_agent
        n = lc_retriever_agent.get_neighborhood(str(fp))
        dependencies = n.get("successors", []) or []
        dependents = n.get("predecessors", []) or []
    except Exception:
        pass

    return {
        "target_file":   str(fp),
        "target_lang":   lang,
        "file_code":     code,
        "file_analysis": analysis,
        "dependencies":  dependencies,
        "dependents":    dependents,
        "method_hint":   method_hint,   # forwarded to answer prompt
        "found": True,
    }


@tool
def tool_chat_rag_retrieve(
    project_path: str,
    query: str,
    target_file: str = "",
    file_code: str = "",
    language: str = "unknown",
) -> Dict[str, Any]:
    """Retrieve project-aware RAG docs for a developer question.

    Args:
        project_path: project root
        query: user question
        target_file: optional target file
        file_code: optional file content
        language: detected language

    Returns:
        {"rag_docs": [...], "rag_scores": [...]}
    """
    try:
        from langchain_agents.agents.lc_retriever_agent import lc_retriever_agent

        if target_file and file_code:
            neighborhood = lc_retriever_agent.get_neighborhood(target_file)
            result = lc_retriever_agent.retrieve_with_context(
                code=file_code[:8000],
                file_path=target_file,
                language=language or "unknown",
                neighborhood=neighborhood,
            )
        else:
            # Use .invoke() — get_relevant_documents() is deprecated in LangChain 0.2+
            docs = lc_retriever_agent.invoke(query)
            result = {"docs": docs, "scores": []}

        docs = result.get("docs", [])
        serialized = []
        for d in docs:
            if hasattr(d, "page_content"):
                serialized.append({
                    "content": d.page_content,
                    "metadata": getattr(d, "metadata", {}),
                })
            elif isinstance(d, dict):
                serialized.append(d)

        return {
            "rag_docs": serialized[:8],
            "rag_scores": result.get("scores", [])[:8],
        }
    except Exception as e:
        return {"rag_docs": [], "rag_scores": [], "error": str(e)}


@tool
def tool_chat_project_summary(project_path: str) -> Dict[str, Any]:
    """Build a lightweight project summary for the chat prompt."""
    root = Path(project_path).resolve()
    counts = {lang: 0 for lang in ["java", "python", "javascript", "typescript"]}
    examples = []

    if not root.exists():
        return {"project_path": str(root), "exists": False, "counts": counts, "examples": []}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        lang = _detect_lang(str(p))
        if lang != "unknown":
            counts[lang] += 1
            if len(examples) < 12:
                examples.append(str(p.relative_to(root)))

    return {
        "project_path": str(root),
        "exists": True,
        "counts": counts,
        "examples": examples,
    }
