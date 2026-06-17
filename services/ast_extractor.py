"""
ast_extractor.py — Extraction de signatures de code via AST tree-sitter.

Remplace l'extraction regex dans test_generator_agent._extract_signatures().
Avantages vs regex :
  - Gère les multi-lignes, décorateurs imbriqués, generics Java
  - Visibilité Java extraite des modificateurs réels (pas pattern matching)
  - Robuste aux commentaires et chaînes multilignes
  - Supporte async/await, annotations Java, accessibility_modifier TS

Fail-silent : si tree-sitter est indisponible ou si le parse échoue,
retourne [] → l'appelant bascule sur le fallback regex.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Singleton parsers ────────────────────────────────────────────────────────

_UNINITIALIZED = object()
_parsers: Dict[str, Any] = {}


def _get_parser(lang: str):
    """Retourne le Parser tree-sitter pour le langage (singleton, lazy init)."""
    cached = _parsers.get(lang, _UNINITIALIZED)
    if cached is not _UNINITIALIZED:
        return cached

    try:
        from tree_sitter import Language, Parser

        if lang == "python":
            import tree_sitter_python as ts_lang
            language_fn = ts_lang.language
        elif lang == "java":
            import tree_sitter_java as ts_lang
            language_fn = ts_lang.language
        elif lang == "javascript":
            import tree_sitter_javascript as ts_lang
            language_fn = ts_lang.language
        elif lang == "typescript":
            import tree_sitter_typescript as ts_lang
            language_fn = ts_lang.language_typescript
        elif lang == "tsx":
            import tree_sitter_typescript as ts_lang
            language_fn = ts_lang.language_tsx
        else:
            _parsers[lang] = None
            return None

        parser = Parser(Language(language_fn()))
        _parsers[lang] = parser
        logger.debug("tree-sitter parser initialized for %s", lang)
        return parser
    except Exception as e:
        logger.debug("tree-sitter parser unavailable for %s: %s", lang, e)
        _parsers[lang] = None
        return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _node_text(node, code_bytes: bytes) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_by_type(node, *types) -> Optional[Any]:
    for child in node.children:
        if child.type in types:
            return child
    return None


def _children_by_type(node, *types) -> List[Any]:
    return [c for c in node.children if c.type in types]


# ── API publique ─────────────────────────────────────────────────────────────

def extract_signatures_ast(code: str, file_path: Path) -> List[Dict[str, Any]]:
    """
    Extrait les signatures depuis le code source via AST tree-sitter.

    Retourne une liste de dicts au même format que TestGeneratorAgent._extract_signatures() :
      {"type": str, "name": str, "signature": str, "visibility": str, ...}
    Retourne [] si tree-sitter est indisponible ou si le parse échoue.
    """
    ext = file_path.suffix.lower()
    lang_map = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
    }
    lang = lang_map.get(ext)
    if not lang:
        return []

    parser = _get_parser(lang)
    if parser is None:
        return []

    try:
        code_bytes = code.encode("utf-8", errors="replace")
        tree = parser.parse(code_bytes)

        if lang == "python":
            return _extract_python(tree.root_node, code_bytes)
        elif lang == "java":
            return _extract_java(tree.root_node, code_bytes)
        elif lang in ("javascript", "typescript", "tsx"):
            return _extract_js_ts(tree.root_node, code_bytes)
        return []
    except Exception as e:
        logger.warning("tree-sitter parse error for %s: %s", file_path.name, e)
        return []


# ── Python ───────────────────────────────────────────────────────────────────

def _extract_python(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []
    seen: set = set()

    def _visit(node):
        for child in node.children:
            if child.type == "class_definition":
                name_node = _child_by_type(child, "identifier")
                cls_name = _node_text(name_node, code_bytes) if name_node else "?"
                if cls_name not in seen:
                    seen.add(cls_name)
                    sigs.append({"type": "class", "name": cls_name, "visibility": "public"})
                body = _child_by_type(child, "block")
                if body:
                    _visit(body)

            elif child.type == "decorated_definition":
                decorators = [
                    _node_text(c, code_bytes)
                    for c in child.children
                    if c.type == "decorator"
                ]
                func_node = _child_by_type(child, "function_definition")
                if func_node:
                    _add_python_func(sigs, seen, func_node, code_bytes, decorators)

            elif child.type == "function_definition":
                _add_python_func(sigs, seen, child, code_bytes, [])

            else:
                _visit(child)

    _visit(root)
    return sigs


def _add_python_func(
    sigs: list, seen: set, node, code_bytes: bytes, decorators: List[str]
) -> None:
    name_node = _child_by_type(node, "identifier")
    if name_node is None:
        return
    name = _node_text(name_node, code_bytes)
    if name in seen:
        return
    seen.add(name)

    is_async = any(c.type == "async" for c in node.children)

    params_node = _child_by_type(node, "parameters")
    params_text = _node_text(params_node, code_bytes) if params_node else "()"

    visibility = "private" if name.startswith("_") else "public"

    method_type = "function"
    dec_lower = " ".join(d.lower() for d in decorators)
    if "@property" in dec_lower:
        method_type = "property"
    elif "@classmethod" in dec_lower or "@staticmethod" in dec_lower:
        method_type = "classmethod"

    sig = f"{'async ' if is_async else ''}def {name}{params_text}"
    sigs.append({
        "type": method_type,
        "name": name,
        "signature": sig,
        "visibility": visibility,
        "async": is_async,
    })


# ── Java ──────────────────────────────────────────────────────────────────────

_JAVA_KEYWORDS = frozenset({
    "if", "while", "for", "switch", "catch", "return", "typeof",
    "instanceof", "new", "this", "super", "class", "interface",
    "enum", "try", "else", "finally", "throw",
})


def _extract_java(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []

    def _visit(node):
        for child in node.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                _add_java_method(sigs, child, code_bytes)
            elif child.type in ("class_declaration", "interface_declaration",
                                "enum_declaration", "record_declaration"):
                body = _child_by_type(child, "class_body", "interface_body", "enum_body")
                if body:
                    _visit(body)
            else:
                _visit(child)

    _visit(root)
    return sigs


def _add_java_method(sigs: list, node, code_bytes: bytes) -> None:
    name_node = _child_by_type(node, "identifier")
    if name_node is None:
        return
    name = _node_text(name_node, code_bytes)
    if name in _JAVA_KEYWORDS:
        return

    mods_node = _child_by_type(node, "modifiers")
    mods_text = _node_text(mods_node, code_bytes) if mods_node else ""
    mods = set(mods_text.split())

    if "public" in mods:
        visibility = "public"
    elif "private" in mods:
        visibility = "private"
    elif "protected" in mods:
        visibility = "protected"
    else:
        visibility = "package-private"

    is_static = "static" in mods

    ret_node = _child_by_type(
        node,
        "type_identifier", "void_type", "generic_type",
        "array_type", "integral_type", "floating_point_type", "boolean_type",
        "scoped_type_identifier",
    )
    ret_type = _node_text(ret_node, code_bytes).strip() if ret_node else ""

    params_node = _child_by_type(node, "formal_parameters")
    params_text = _node_text(params_node, code_bytes) if params_node else "()"

    node_type = "constructor" if node.type == "constructor_declaration" else "method"
    signature = " ".join(filter(None, [mods_text, ret_type, f"{name}{params_text}"])).strip()

    sigs.append({
        "type": node_type,
        "name": name,
        "signature": signature,
        "visibility": visibility,
        "static": is_static,
        "return_type": ret_type,
    })


# ── JavaScript / TypeScript ───────────────────────────────────────────────────

_JS_KEYWORDS = frozenset({
    "if", "while", "for", "switch", "catch", "return", "typeof",
    "instanceof", "new", "this", "super", "class", "constructor",
    "async", "await", "import", "export", "default", "get", "set",
})


def _extract_js_ts(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []
    seen: set = set()

    def _visit(node):
        for child in node.children:
            t = child.type

            if t == "function_declaration":
                _add_js_named_function(sigs, seen, child, code_bytes)

            elif t in ("lexical_declaration", "variable_declaration"):
                for decl in _children_by_type(child, "variable_declarator"):
                    name_node = _child_by_type(decl, "identifier")
                    has_arrow = any(
                        c.type in ("arrow_function", "function")
                        for c in decl.children
                    )
                    if name_node and has_arrow:
                        name = _node_text(name_node, code_bytes)
                        if name not in _JS_KEYWORDS and name not in seen:
                            seen.add(name)
                            sigs.append({
                                "type": "arrow_function",
                                "name": name,
                                "visibility": "public",
                            })

            elif t in ("method_definition", "method_signature"):
                _add_js_method(sigs, seen, child, code_bytes)

            elif t in (
                "class_declaration", "class_body", "class",
                "export_statement", "program", "statement_block",
                "abstract_class_declaration",
            ):
                _visit(child)

            else:
                _visit(child)

    _visit(root)
    return sigs


def _add_js_named_function(sigs: list, seen: set, node, code_bytes: bytes) -> None:
    name_node = _child_by_type(node, "identifier")
    if name_node is None:
        return
    name = _node_text(name_node, code_bytes)
    if name in _JS_KEYWORDS or name in seen:
        return
    seen.add(name)
    sigs.append({"type": "function", "name": name, "visibility": "public"})


def _add_js_method(sigs: list, seen: set, node, code_bytes: bytes) -> None:
    name_node = _child_by_type(
        node, "property_identifier", "private_property_identifier"
    )
    if name_node is None:
        return
    name = _node_text(name_node, code_bytes)
    if name in _JS_KEYWORDS or name in seen or name == "constructor":
        return
    seen.add(name)

    # TypeScript : accessibility_modifier (public | private | protected)
    accessibility = _child_by_type(node, "accessibility_modifier")
    if accessibility:
        vis_text = _node_text(accessibility, code_bytes)
        visibility = vis_text if vis_text in ("public", "private", "protected") else "public"
    elif name.startswith("_") or name.startswith("#"):
        visibility = "private"
    else:
        visibility = "public"

    sigs.append({"type": "method", "name": name, "visibility": visibility})
