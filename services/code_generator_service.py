"""
code_generator_service.py — Code generation utilities for ChatAgent Phase 2.

Responsibilities:
  - Detect project conventions (naming, patterns, imports)
  - Validate generated code syntax (per language)
  - Extract function signatures from source files
  - Enumerate existing classes for context injection
  - Build generation prompts (complete_fn / new_class)

No LLM calls here — this service only prepares context and validates output.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LANG_BY_EXT = {
    ".java": "java",
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
}

_EXT_BY_LANG = {
    "java":       ".java",
    "python":     ".py",
    "javascript": ".js",
    "typescript": ".ts",
}


def _detect_lang(path: str) -> str:
    return _LANG_BY_EXT.get(Path(path).suffix.lower(), "unknown")


# ── Convention detection ──────────────────────────────────────────────────────

def detect_conventions(project_path: str, language: str) -> Dict[str, Any]:
    """Scan a sample of project files to infer naming and structural conventions.

    Returns:
        {
          "naming":         "camelCase" | "snake_case" | "PascalCase",
          "indent":         "4" | "2" | "tab",
          "existing_classes": [list of class names],
          "common_imports": [list of frequent import statements],
          "base_classes":   [list of common base/parent class names],
          "has_tests":      bool,
        }
    """
    root = Path(project_path).resolve()
    ext  = _EXT_BY_LANG.get(language, ".py")

    files: List[Path] = []
    for p in root.rglob(f"*{ext}"):
        # Skip test / build / venv directories
        parts = set(p.parts)
        if parts & {"test", "tests", "__pycache__", "build", "target",
                    "node_modules", ".venv", "venv", ".git"}:
            continue
        files.append(p)
        if len(files) >= 40:
            break

    existing_classes: List[str] = []
    common_imports: Dict[str, int] = {}
    base_classes: List[str] = []
    indent_votes: Dict[str, int] = {"4": 0, "2": 0, "tab": 0}
    fn_names: List[str] = []

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Class names
        for m in re.finditer(r"\bclass\s+(\w+)", text):
            cls = m.group(1)
            if cls not in existing_classes:
                existing_classes.append(cls)

        # Base classes (Java: extends X, Python: class X(Y))
        for m in re.finditer(r"extends\s+(\w+)|class\s+\w+\(([^)]+)\)", text):
            base = m.group(1) or m.group(2)
            if base and base not in base_classes:
                base_classes.append(base.strip())

        # Imports
        for m in re.finditer(r"^(?:import|from)\s+.+", text, re.MULTILINE):
            imp = m.group(0).strip()
            common_imports[imp] = common_imports.get(imp, 0) + 1

        # Indent
        for line in text.splitlines():
            if line.startswith("    "):
                indent_votes["4"] += 1
            elif line.startswith("  ") and not line.startswith("    "):
                indent_votes["2"] += 1
            elif line.startswith("\t"):
                indent_votes["tab"] += 1

        # Function names (to detect camelCase vs snake_case)
        if language == "python":
            fn_names += re.findall(r"def\s+([a-z]\w+)", text)
        else:
            fn_names += re.findall(r"\b([a-z][a-zA-Z0-9]+)\s*\(", text)

    # Naming convention
    camel  = sum(1 for n in fn_names if re.search(r"[a-z][A-Z]", n))
    snake  = sum(1 for n in fn_names if "_" in n)
    naming = "camelCase" if camel > snake else "snake_case"

    # Indent
    indent = max(indent_votes, key=lambda k: indent_votes[k])

    # Top imports
    top_imports = sorted(common_imports, key=lambda k: -common_imports[k])[:10]

    # Has tests
    has_tests = any(
        p for p in root.rglob(f"*{ext}")
        if "test" in p.name.lower() or "spec" in p.name.lower()
    )

    return {
        "naming":           naming,
        "indent":           indent,
        "existing_classes": existing_classes[:30],
        "common_imports":   top_imports,
        "base_classes":     list(set(base_classes))[:10],
        "has_tests":        has_tests,
    }


# ── Function signature extraction ─────────────────────────────────────────────

def extract_function_signature(code: str, fn_name: str, language: str) -> str:
    """Find the signature (and stub body) of a function in the source.

    Returns the function definition block (up to 30 lines) or empty string.
    """
    lang = language.lower()

    if lang == "python":
        pattern = re.compile(
            rf"((?:@\w+(?:\([^)]*\))?\s*\n)*[ \t]*(?:async\s+)?def\s+{re.escape(fn_name)}\s*\([^)]*\)[^:]*:)",
            re.MULTILINE,
        )
    elif lang == "java":
        pattern = re.compile(
            rf"((?:public|private|protected|static|final|synchronized|\s)+[\w<>\[\]]+\s+{re.escape(fn_name)}\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{{)",
            re.MULTILINE,
        )
    else:
        pattern = re.compile(
            rf"((?:export\s+)?(?:async\s+)?(?:function\s+{re.escape(fn_name)}\s*\([^)]*\)|(?:const|let|var)\s+{re.escape(fn_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*(?:=>\s*)?)\{{)",
            re.MULTILINE,
        )

    m = pattern.search(code)
    if not m:
        return ""

    start = m.start()
    lines = code[start:].splitlines()[:30]
    return "\n".join(lines)


# ── Syntax validation ─────────────────────────────────────────────────────────

def validate_syntax(code: str, language: str) -> Tuple[bool, List[str]]:
    """Basic syntax check for generated code.

    Returns (is_valid, [error_messages]).
    Python: uses ast.parse for real syntax checking.
    Java/JS/TS: brace balance + basic heuristics (no compiler available).
    """
    lang = language.lower()
    errors: List[str] = []

    if not code or not code.strip():
        return False, ["Generated code is empty."]

    if lang == "python":
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"SyntaxError line {e.lineno}: {e.msg}"]

    # Java / JS / TS — brace balance check
    open_braces  = code.count("{")
    close_braces = code.count("}")
    open_parens  = code.count("(")
    close_parens = code.count(")")

    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close.")
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close.")

    # Check class/function declaration present
    if lang == "java" and "class " not in code and "interface " not in code:
        if re.search(r"(public|private|protected|static)\s+\w+\s+\w+\s*\(", code) is None:
            errors.append("No class/interface/method declaration found in generated Java code.")

    if lang in ("javascript", "typescript"):
        if not re.search(r"(class |function |const |let |var |export )", code):
            errors.append("No recognizable JS/TS declaration found in generated code.")

    return len(errors) == 0, errors


# ── Generation prompt builders ────────────────────────────────────────────────

def build_completion_prompt(
    fn_name:      str,
    file_path:    str,
    file_code:    str,
    language:     str,
    conventions:  Dict[str, Any],
    rag_docs:     List[Dict[str, Any]],
    dependencies: List[str],
    history_text: str,
) -> str:
    """Build the LLM prompt to complete a specific function."""

    fn_signature = extract_function_signature(file_code, fn_name, language)
    fn_block = (
        f"TARGET FUNCTION (signature found):\n```{language}\n{fn_signature}\n```\n"
        if fn_signature
        else f"TARGET FUNCTION: `{fn_name}` — signature not found, infer from context.\n"
    )

    rag_text = "\n\n".join(
        f"[KB {i+1}] {d.get('content','')[:800]}"
        for i, d in enumerate(rag_docs[:5])
    )

    dep_text = "\n".join(f"  - {d}" for d in dependencies[:10]) or "  (none)"

    conv_text = (
        f"\nCONVERSATION HISTORY:\n{history_text}\n"
        if history_text.strip() else ""
    )

    return f"""You are a senior {language} developer completing an existing function.

TASK: Complete ONLY the function `{fn_name}` in `{file_path}`.

HARD CONSTRAINTS:
  1. Keep the EXACT same function signature (name, parameters, return type).
  2. Use ONLY imports already present in the file — never add new imports.
  3. Use ONLY fields/methods visible in the file — never invent new ones.
  4. Follow the project coding style: naming={conventions.get('naming','?')}, indent={conventions.get('indent','4')} spaces.
  5. Output ONLY the complete function body — no explanation, no markdown, no extra classes.
  6. The output must be directly pasteable into the file.
{conv_text}
FILE: {file_path}
LANGUAGE: {language}

CURRENT FILE CONTENT:
```{language}
{file_code[:6000]}
```

{fn_block}

AVAILABLE DEPENDENCIES:
{dep_text}

KNOWLEDGE BASE (best practices for {language}):
{rag_text or '(none)'}

PROJECT CONVENTIONS:
  - naming:           {conventions.get('naming', '?')}
  - indent:           {conventions.get('indent', '4')} spaces
  - base classes:     {', '.join(conventions.get('base_classes', [])) or 'none'}
  - common imports:   {', '.join(conventions.get('common_imports', [])[:5]) or 'none'}

OUTPUT FORMAT — output EXACTLY this structure, nothing else:
```{language}
[complete implementation of {fn_name} here]
```
"""


def build_class_prompt(
    class_name:        str,
    language:          str,
    description:       str,
    project_path:      str,
    conventions:       Dict[str, Any],
    rag_docs:          List[Dict[str, Any]],
    history_text:      str,
) -> str:
    """Build the LLM prompt to generate a new class."""

    existing = conventions.get("existing_classes", [])[:20]
    existing_text = ", ".join(existing) if existing else "(none found)"

    rag_text = "\n\n".join(
        f"[KB {i+1}] {d.get('content','')[:800]}"
        for i, d in enumerate(rag_docs[:5])
    )

    conv_text = (
        f"\nCONVERSATION HISTORY:\n{history_text}\n"
        if history_text.strip() else ""
    )

    test_note = (
        "\nAlso generate a companion unit test skeleton at the end (clearly separated by a comment)."
        if conventions.get("has_tests") else ""
    )

    return f"""You are a senior {language} developer generating a new class.

TASK: Generate a complete, production-ready class `{class_name}` for the project at `{project_path}`.

DESCRIPTION: {description or f'A {class_name} class following project conventions.'}
{conv_text}
HARD CONSTRAINTS:
  1. Follow the EXACT same conventions as existing classes in the project.
  2. Use ONLY imports that exist in the project (see common imports below).
  3. Do NOT invent external libraries not already used in the project.
  4. naming={conventions.get('naming','?')}, indent={conventions.get('indent','4')} spaces.
  5. The class must be complete and directly compilable/runnable.
  6. One public class per file.{test_note}

EXISTING CLASSES IN PROJECT (for context):
  {existing_text}

BASE/PARENT CLASSES AVAILABLE:
  {', '.join(conventions.get('base_classes', [])) or 'none'}

COMMON IMPORTS IN PROJECT:
  {chr(10).join('  ' + i for i in conventions.get('common_imports', [])[:8]) or '  none'}

KNOWLEDGE BASE (best practices for {language}):
{rag_text or '(none)'}

OUTPUT FORMAT — output EXACTLY this structure:
```{language}
[complete class {class_name} here]
```
SUGGESTED_FILE: [suggested file path relative to project root, e.g. services/{class_name.lower()}.py]
"""


# ── Suggested file path ───────────────────────────────────────────────────────

def suggest_file_path(class_name: str, language: str, project_path: str) -> str:
    """Suggest where to create a new class file based on project structure."""
    root  = Path(project_path).resolve()
    ext   = _EXT_BY_LANG.get(language, ".py")

    # Detect standard directories
    for candidate_dir in ("services", "src", "lib", "app", "core", "domain"):
        d = root / candidate_dir
        if d.is_dir():
            # Convert PascalCase to snake_case for Python files
            if language == "python":
                snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
                return str(d / f"{snake}{ext}")
            return str(d / f"{class_name}{ext}")

    # Fallback: project root
    if language == "python":
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        return str(root / f"{snake}{ext}")
    return str(root / f"{class_name}{ext}")


# ── Code block extractor ──────────────────────────────────────────────────────

def extract_code_blocks(llm_output: str) -> List[Dict[str, str]]:
    """Extract fenced code blocks from LLM output.

    Returns list of {lang, code} dicts.
    """
    blocks = []
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(llm_output):
        lang = m.group(1).strip() or "unknown"
        code = m.group(2).strip()
        if code:
            blocks.append({"lang": lang, "code": code})
    return blocks


def extract_suggested_file(llm_output: str) -> str:
    """Extract SUGGESTED_FILE: path from LLM output."""
    m = re.search(r"SUGGESTED_FILE:\s*(.+)", llm_output)
    return m.group(1).strip() if m else ""


# ── Singleton ─────────────────────────────────────────────────────────────────

code_generator_service = type("CodeGeneratorService", (), {
    "detect_conventions":       staticmethod(detect_conventions),
    "validate_syntax":          staticmethod(validate_syntax),
    "extract_function_signature": staticmethod(extract_function_signature),
    "build_completion_prompt":  staticmethod(build_completion_prompt),
    "build_class_prompt":       staticmethod(build_class_prompt),
    "suggest_file_path":        staticmethod(suggest_file_path),
    "extract_code_blocks":      staticmethod(extract_code_blocks),
    "extract_suggested_file":   staticmethod(extract_suggested_file),
})()
