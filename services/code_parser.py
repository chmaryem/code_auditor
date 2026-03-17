"""
code_parser.py — Parser universel multi-langage
================================================
Architecture hybride en 3 couches :

  Couche 1 — Tree-sitter   (si installé)  : précision maximale, récupération sur erreur
  Couche 2 — AST natif     (Python only)  : robuste, toujours disponible
  Couche 3 — Regex enrichi (Java/JS/TS)   : fallback garanti, amélioré vs l'original

Installation tree-sitter (optionnelle mais recommandée) :
  pip install tree-sitter
  pip install tree-sitter-python tree-sitter-javascript
  pip install tree-sitter-typescript tree-sitter-java

Nouveautés vs l'original :
  - Extraction des décorateurs Python (@staticmethod, @property, @app.route…)
  - Extraction des type hints Python (paramètres + retour)
  - Support TypeScript réel (interfaces, types, enums, generics)
  - Support JS/TS arrow functions, const fn = () => {}, méthodes de classe
  - Support CommonJS require() en plus des ES imports
  - Parser Java amélioré : annotations, interfaces, enums, constructeurs
  - is_async, is_exported, is_static sur les entités
  - parse_source() pour analyser du code en mémoire (mode watch)
  - Parser stats : compteur d'utilisation par backend
"""

from __future__ import annotations

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# DATA CLASSES  (rétro-compatibles avec l'original + champs enrichis)
@dataclass
class CodeEntity:
    """
    Représente une entité de code (fonction, classe, méthode, interface…).

    Champs nouveaux (optionnels, valeur par défaut neutre) :
      decorators   — liste des décorateurs/annotations
      return_type  — type de retour déclaré (str)
      is_async     — True si async def / async function
      is_exported  — True si export / public
      is_static    — True si @staticmethod / static
      docstring    — première docstring détectée
    """
    name: str
    type: str               # function | class | method | interface | enum | constructor
    start_line: int
    end_line: int
    file_path: str
    language: str
    body: Optional[str] = None
    parameters: List[str] = None
    dependencies: List[str] = None

    # champs enrichis
    decorators: List[str] = None
    return_type: Optional[str] = None
    is_async: bool = False
    is_exported: bool = False
    is_static: bool = False
    docstring: Optional[str] = None

    def __post_init__(self):
        if self.parameters  is None: self.parameters  = []
        if self.dependencies is None: self.dependencies = []
        if self.decorators   is None: self.decorators   = []


@dataclass
class ImportStatement:
    """
    Représente un import/require.

    Champs nouveaux :
      is_relative — True pour les imports relatifs (./foo, ../bar, from .module)
      raw         — texte brut de la ligne d'import
      import_type — 'es_import' | 'commonjs_require' | 'python_import' | 'java_import'
    """
    module: str
    items: List[str]
    alias: Optional[str]
    file_path: str
    line: int

    # champs enrichis
    is_relative: bool = False
    raw: Optional[str] = None
    import_type: str = "unknown"



# TREE-SITTER AVAILABILITY CHECK


def _check_treesitter() -> Tuple[bool, Optional[Any]]:
    """
    Vérifie si tree-sitter est installé ET si au moins une grammaire est dispo.
    Retourne (available: bool, module: optional).
    """
    try:
        import tree_sitter                               # noqa: F401
        return True, tree_sitter
    except ImportError:
        return False, None

TREESITTER_AVAILABLE, _ts_module = _check_treesitter()


def _load_ts_language(lang_name: str) -> Optional[Any]:
    """
    Charge une grammaire tree-sitter.
    Supporte l'API v0.20 (Language(path, name)) et v0.21+ (Language(module)).
    Retourne None silencieusement si la grammaire n'est pas installée.
    """
    if not TREESITTER_AVAILABLE:
        return None

    #  API v0.21+  (tree-sitter-python, tree-sitter-javascript, …)
    #    pip install tree-sitter-python  →  from tree_sitter_languages import get_language
    try:
        from tree_sitter_languages import get_language   # bundle all-in-one
        return get_language(lang_name)
    except (ImportError, Exception):
        pass

    #  Tentative package individuel 
    pkg_map = {
        "python":     "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "java":       "tree_sitter_java",
    }
    pkg = pkg_map.get(lang_name)
    if pkg:
        try:
            mod = __import__(pkg)
            from tree_sitter import Language
            return Language(mod.language())
        except (ImportError, Exception):
            pass

    return None



# UNIVERSAL PARSER


class UniversalCodeParser:
    """
    Parser universel pour Python, JavaScript, TypeScript, Java.

    Stratégie par langage :
      Python      → tree-sitter  sinon  ast natif  (toujours disponible)
      JavaScript  → tree-sitter  sinon  regex enrichi
      TypeScript  → tree-sitter  sinon  regex enrichi (sous-ensemble TS)
      Java        → tree-sitter  sinon  regex enrichi (amélioré vs original)
    """

    # Extensions reconnues
    EXTENSION_MAP = {
        ".py":   "python",
        ".js":   "javascript",
        ".jsx":  "javascript",
        ".ts":   "typescript",
        ".tsx":  "typescript",
        ".java": "java",
    }

    def __init__(self):
        self._ts_parsers: Dict[str, Any] = {}   # cache parsers tree-sitter
        self._ts_langs:   Dict[str, Any] = {}   # cache langues tree-sitter
        self.stats = {                           # compteurs d'utilisation
            "treesitter": 0,
            "ast_native": 0,
            "regex":      0,
            "errors":     0,
        }
        self._init_treesitter()

    # ── initialisation ───────────────────────────────────────────────────────

    def _init_treesitter(self):
        """Pré-charge les grammaires tree-sitter disponibles."""
        if not TREESITTER_AVAILABLE:
            logger.info("tree-sitter non installé — utilisation des fallbacks (ast/regex)")
            return

        for lang in ("python", "javascript", "typescript", "java"):
            lang_obj = _load_ts_language(lang)
            if lang_obj:
                try:
                    from tree_sitter import Parser as TSParser
                    p = TSParser()
                    p.set_language(lang_obj)
                    self._ts_parsers[lang] = p
                    self._ts_langs[lang]   = lang_obj
                    logger.debug(f"tree-sitter chargé : {lang}")
                except Exception as e:
                    logger.debug(f"tree-sitter {lang} non disponible : {e}")

        if self._ts_parsers:
            available = list(self._ts_parsers.keys())
            logger.info(f"tree-sitter actif pour : {', '.join(available)}")
        else:
            logger.info("tree-sitter installé mais aucune grammaire trouvée — utilisation des fallbacks")

    # ── API publique ─────────────────────────────────────────────────────────

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse un fichier et retourne ses entités + imports.

        Retour (identique à l'original + champs enrichis) :
          {
            "language": str,
            "file_path": str,
            "entities": List[CodeEntity],
            "imports":  List[ImportStatement],
            "backend":  str,          # "treesitter" | "ast" | "regex"
            "raw_ast":  Any | None,   # Python uniquement
          }
        """
        language = self._detect_language(file_path)
        if language == "unknown":
            return {"error": f"Unsupported extension: {file_path.suffix}",
                    "file_path": str(file_path)}

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError as e:
            self.stats["errors"] += 1
            return {"error": str(e), "file_path": str(file_path)}

        return self.parse_source(source, language, str(file_path))

    def parse_source(self, source: str, language: str, file_path: str = "<memory>") -> Dict[str, Any]:
        """
        Parse du code source directement depuis une chaîne.
        Utile pour le mode watch (diff en mémoire) et les tests unitaires.
        """
        if language == "python":
            return self._parse_python(source, file_path)
        elif language in ("javascript", "typescript"):
            return self._parse_js_ts(source, language, file_path)
        elif language == "java":
            return self._parse_java(source, file_path)
        else:
            return {"error": f"Unsupported language: {language}", "file_path": file_path}

    def _detect_language(self, file_path: Path) -> str:
        return self.EXTENSION_MAP.get(file_path.suffix.lower(), "unknown")

    # ─────────────────────────────────────────────────────────────────────────
    # PYTHON PARSER
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_python(self, source: str, file_path: str) -> Dict[str, Any]:
        """
        Python : tree-sitter si dispo, sinon ast natif.
        Les deux backends remplissent les mêmes champs.
        """
        if "python" in self._ts_parsers:
            result = self._ts_parse_python(source, file_path)
            if "error" not in result:
                self.stats["treesitter"] += 1
                return result
            logger.debug(f"tree-sitter Python échoué sur {file_path}, fallback ast")

        return self._ast_parse_python(source, file_path)

    # ── tree-sitter Python ───────────────────────────────────────────────────

    def _ts_parse_python(self, source: str, file_path: str) -> Dict[str, Any]:
        """Parse Python avec tree-sitter."""
        try:
            ts_parser = self._ts_parsers["python"]
            tree = ts_parser.parse(source.encode("utf-8"))
            root = tree.root_node

            entities: List[CodeEntity] = []
            imports:  List[ImportStatement] = []

            self._ts_walk_python(root, source, file_path, entities, imports)

            return {
                "language": "python",
                "file_path": file_path,
                "entities": entities,
                "imports":  imports,
                "backend":  "treesitter",
                "raw_ast":  None,
            }
        except Exception as e:
            return {"error": str(e), "file_path": file_path}

    def _ts_walk_python(self, node, source: str, file_path: str,
                         entities: list, imports: list, depth: int = 0):
        """Parcourt l'AST tree-sitter Python de façon récursive."""
        ntype = node.type

        # ── imports ──────────────────────────────────────────────────────────
        if ntype == "import_statement":
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import"):
                    module_name = self._ts_node_text(child, source)
                    alias = None
                    if child.type == "aliased_import":
                        parts = [c for c in child.children if c.type not in (",", "as")]
                        module_name = self._ts_node_text(parts[0], source) if parts else ""
                        alias = self._ts_node_text(parts[-1], source) if len(parts) > 1 else None
                    imports.append(ImportStatement(
                        module=module_name, items=[], alias=alias,
                        file_path=file_path, line=node.start_point[0] + 1,
                        is_relative=False, import_type="python_import",
                        raw=self._ts_node_text(node, source),
                    ))

        elif ntype == "import_from_statement":
            module_node = next((c for c in node.children
                                if c.type in ("dotted_name", "relative_import")), None)
            module_raw  = self._ts_node_text(module_node, source) if module_node else ""
            is_relative = module_raw.startswith(".")
            items = [
                self._ts_node_text(c, source)
                for c in node.children
                if c.type in ("dotted_name", "aliased_import")
                and c != module_node
            ]
            imports.append(ImportStatement(
                module=module_raw, items=items, alias=None,
                file_path=file_path, line=node.start_point[0] + 1,
                is_relative=is_relative, import_type="python_import",
                raw=self._ts_node_text(node, source),
            ))

        # ── fonctions ─────────────────────────────────────────────────────────
        elif ntype in ("function_definition", "async_function_definition"):
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            if name_node:
                name = self._ts_node_text(name_node, source)
                params = self._ts_extract_python_params(node, source)
                decorators = self._ts_extract_decorators(node, source)
                ret_type = self._ts_extract_return_type_python(node, source)
                docstring = self._ts_extract_docstring(node, source)
                entities.append(CodeEntity(
                    name=name, type="function",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language="python",
                    parameters=params,
                    decorators=decorators,
                    return_type=ret_type,
                    is_async=ntype.startswith("async_"),
                    is_static="staticmethod" in decorators,
                    docstring=docstring,
                ))

        # ── classes ───────────────────────────────────────────────────────────
        elif ntype == "class_definition":
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            if name_node:
                decorators = self._ts_extract_decorators(node, source)
                docstring  = self._ts_extract_docstring(node, source)
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language="python",
                    decorators=decorators,
                    docstring=docstring,
                ))

        # ── récursion ─────────────────────────────────────────────────────────
        for child in node.children:
            self._ts_walk_python(child, source, file_path, entities, imports, depth + 1)

    # ── ast natif Python ─────────────────────────────────────────────────────

    def _ast_parse_python(self, source: str, file_path: str) -> Dict[str, Any]:
        """Parse Python avec ast natif (toujours disponible)."""
        try:
            tree = ast.parse(source, filename=file_path)
            entities: List[CodeEntity] = []
            imports:  List[ImportStatement] = []

            for node in ast.walk(tree):
                # ── fonctions / méthodes async ────────────────────────────────
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorators = [
                        ast.unparse(d) if hasattr(ast, "unparse") else d.id
                        if hasattr(d, "id") else ""
                        for d in node.decorator_list
                    ]
                    ret_type = (
                        ast.unparse(node.returns)
                        if hasattr(ast, "unparse") and node.returns
                        else None
                    )
                    params = self._ast_extract_params(node)
                    docstring = ast.get_docstring(node)
                    entities.append(CodeEntity(
                        name=node.name, type="function",
                        start_line=node.lineno, end_line=node.end_lineno,
                        file_path=file_path, language="python",
                        parameters=params,
                        decorators=decorators,
                        return_type=ret_type,
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        is_static="staticmethod" in decorators,
                        docstring=docstring,
                    ))

                # ── classes ───────────────────────────────────────────────────
                elif isinstance(node, ast.ClassDef):
                    decorators = [
                        ast.unparse(d) if hasattr(ast, "unparse") else (d.id if hasattr(d, "id") else "")
                        for d in node.decorator_list
                    ]
                    entities.append(CodeEntity(
                        name=node.name, type="class",
                        start_line=node.lineno, end_line=node.end_lineno,
                        file_path=file_path, language="python",
                        decorators=decorators,
                        docstring=ast.get_docstring(node),
                    ))

                # ── import ────────────────────────────────────────────────────
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportStatement(
                            module=alias.name, items=[], alias=alias.asname,
                            file_path=file_path, line=node.lineno,
                            is_relative=False, import_type="python_import",
                        ))

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.append(ImportStatement(
                        module=module,
                        items=[a.name for a in node.names],
                        alias=None,
                        file_path=file_path, line=node.lineno,
                        is_relative=(node.level or 0) > 0,
                        import_type="python_import",
                    ))

            self.stats["ast_native"] += 1
            return {
                "language": "python",
                "file_path": file_path,
                "entities": entities,
                "imports":  imports,
                "backend":  "ast",
                "raw_ast":  tree,
            }

        except SyntaxError as e:
            # Essayer malgré tout avec un mode tolérant
            self.stats["errors"] += 1
            return {"error": f"SyntaxError: {e}", "file_path": file_path}
        except Exception as e:
            self.stats["errors"] += 1
            return {"error": str(e), "file_path": file_path}

    # ─────────────────────────────────────────────────────────────────────────
    # JAVASCRIPT / TYPESCRIPT PARSER
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_js_ts(self, source: str, language: str, file_path: str) -> Dict[str, Any]:
        """JS/TS : tree-sitter si dispo, sinon regex enrichi."""
        ts_lang = language  # "javascript" ou "typescript"

        if ts_lang in self._ts_parsers:
            result = self._ts_parse_js_ts(source, ts_lang, file_path)
            if "error" not in result:
                self.stats["treesitter"] += 1
                return result
            logger.debug(f"tree-sitter {ts_lang} échoué sur {file_path}, fallback regex")

        return self._regex_parse_js_ts(source, language, file_path)

    # ── tree-sitter JS/TS ────────────────────────────────────────────────────

    def _ts_parse_js_ts(self, source: str, language: str, file_path: str) -> Dict[str, Any]:
        try:
            ts_parser = self._ts_parsers[language]
            tree = ts_parser.parse(source.encode("utf-8"))
            entities: List[CodeEntity] = []
            imports:  List[ImportStatement] = []
            self._ts_walk_js(tree.root_node, source, language, file_path, entities, imports)
            return {
                "language": language, "file_path": file_path,
                "entities": entities, "imports": imports,
                "backend": "treesitter", "raw_ast": None,
            }
        except Exception as e:
            return {"error": str(e), "file_path": file_path}

    def _ts_walk_js(self, node, source: str, language: str, file_path: str,
                     entities: list, imports: list):
        ntype = node.type

        # ── ES imports ───────────────────────────────────────────────────────
        if ntype == "import_statement":
            src_node = next((c for c in node.children if c.type == "string"), None)
            module_raw = self._ts_node_text(src_node, source).strip("'\"`") if src_node else ""
            items = [
                self._ts_node_text(c, source)
                for c in node.named_children
                if c.type in ("identifier", "namespace_import", "named_imports")
            ]
            imports.append(ImportStatement(
                module=module_raw, items=items, alias=None,
                file_path=file_path, line=node.start_point[0] + 1,
                is_relative=module_raw.startswith(("./", "../")),
                import_type="es_import",
                raw=self._ts_node_text(node, source),
            ))

        # ── require() ────────────────────────────────────────────────────────
        elif ntype == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node and self._ts_node_text(fn_node, source) == "require":
                args = node.child_by_field_name("arguments")
                if args:
                    str_nodes = [c for c in args.children if c.type == "string"]
                    if str_nodes:
                        module_raw = self._ts_node_text(str_nodes[0], source).strip("'\"`")
                        imports.append(ImportStatement(
                            module=module_raw, items=[], alias=None,
                            file_path=file_path, line=node.start_point[0] + 1,
                            is_relative=module_raw.startswith(("./", "../")),
                            import_type="commonjs_require",
                        ))

        # ── function declaration ──────────────────────────────────────────────
        elif ntype in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                params = self._ts_extract_js_params(node, source)
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type="function",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language=language,
                    parameters=params,
                    is_async=any(c.type == "async" for c in node.children),
                    is_exported=self._ts_is_exported(node),
                ))

        # ── const/let fn = () => {} ──────────────────────────────────────────
        elif ntype == "lexical_declaration":
            for decl in node.named_children:
                if decl.type == "variable_declarator":
                    val = decl.child_by_field_name("value")
                    if val and val.type in ("arrow_function", "function",
                                            "generator_function"):
                        name_node = decl.child_by_field_name("name")
                        if name_node:
                            params = self._ts_extract_js_params(val, source)
                            entities.append(CodeEntity(
                                name=self._ts_node_text(name_node, source),
                                type="function",
                                start_line=node.start_point[0] + 1,
                                end_line=val.end_point[0] + 1,
                                file_path=file_path, language=language,
                                parameters=params,
                                is_async=any(c.type == "async" for c in val.children),
                                is_exported=self._ts_is_exported(node),
                            ))

        # ── class ─────────────────────────────────────────────────────────────
        elif ntype == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language=language,
                    is_exported=self._ts_is_exported(node),
                ))

        # ── TS interface ──────────────────────────────────────────────────────
        elif ntype == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type="interface",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language=language,
                    is_exported=self._ts_is_exported(node),
                ))

        # ── TS enum ───────────────────────────────────────────────────────────
        elif ntype == "enum_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type="enum",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language=language,
                    is_exported=self._ts_is_exported(node),
                ))

        # ── récursion ─────────────────────────────────────────────────────────
        for child in node.children:
            self._ts_walk_js(child, source, language, file_path, entities, imports)

    # ── regex JS/TS (fallback) ───────────────────────────────────────────────

    def _regex_parse_js_ts(self, source: str, language: str, file_path: str) -> Dict[str, Any]:
        """
        Fallback regex enrichi pour JS/TypeScript.
        Couvre : ES imports, require(), function declarations, arrow fns,
                 classes, TS interfaces, TS enums, exports.
        """
        entities: List[CodeEntity] = []
        imports:  List[ImportStatement] = []
        lines = source.splitlines()

        # ── ES imports : import X from '...' | import { A, B } from '...' ────
        es_import_re = re.compile(
            r"""^(?:export\s+)?import\s+
                (?:type\s+)?                          # TS import type
                (?:
                    (\*\s+as\s+\w+)                   # import * as ns
                  | \{([^}]*)\}                       # import { A, B }
                  | (\w+)(?:\s*,\s*\{([^}]*)\})?      # import Def, { A }
                )?
                \s*(?:from\s+)?
                ['"`]([^'"`]+)['"`]""",
            re.VERBOSE | re.MULTILINE,
        )
        for m in es_import_re.finditer(source):
            module_path = m.group(5) or ""
            named_raw   = (m.group(2) or "") + ("," + m.group(4) if m.group(4) else "")
            items = [s.strip().split(" as ")[0].strip()
                     for s in named_raw.split(",") if s.strip()]
            if m.group(3):
                items.insert(0, m.group(3).strip())
            line_num = source[: m.start()].count("\n") + 1
            imports.append(ImportStatement(
                module=module_path, items=items, alias=None,
                file_path=file_path, line=line_num,
                is_relative=module_path.startswith(("./", "../")),
                import_type="es_import",
                raw=m.group(0),
            ))

        # ── CommonJS require ─────────────────────────────────────────────────
        req_re = re.compile(r"""require\s*\(\s*['"`]([^'"`]+)['"`]\s*\)""")
        for m in req_re.finditer(source):
            module_path = m.group(1)
            line_num = source[: m.start()].count("\n") + 1
            imports.append(ImportStatement(
                module=module_path, items=[], alias=None,
                file_path=file_path, line=line_num,
                is_relative=module_path.startswith(("./", "../")),
                import_type="commonjs_require",
                raw=m.group(0),
            ))

        # ── function declarations ─────────────────────────────────────────────
        fn_re = re.compile(
            r"^(?P<export>export\s+(?:default\s+)?)?(?P<async>async\s+)?"
            r"function\s*\*?\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
            re.MULTILINE,
        )
        for m in fn_re.finditer(source):
            line_num = source[: m.start()].count("\n") + 1
            params = [p.strip().split(":")[0].strip().lstrip("...")
                      for p in m.group("params").split(",") if p.strip()]
            entities.append(CodeEntity(
                name=m.group("name"), type="function",
                start_line=line_num, end_line=line_num,
                file_path=file_path, language=language,
                parameters=params,
                is_async=bool(m.group("async")),
                is_exported=bool(m.group("export")),
            ))

        # ── arrow / const fn = () => {} ──────────────────────────────────────
        arrow_re = re.compile(
            r"^(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*"
            r"(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>",
            re.MULTILINE,
        )
        for m in arrow_re.finditer(source):
            line_num = source[: m.start()].count("\n") + 1
            entities.append(CodeEntity(
                name=m.group("name"), type="function",
                start_line=line_num, end_line=line_num,
                file_path=file_path, language=language,
            ))

        # ── classes ───────────────────────────────────────────────────────────
        class_re = re.compile(
            r"^(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+(?P<name>\w+)",
            re.MULTILINE,
        )
        for m in class_re.finditer(source):
            line_num = source[: m.start()].count("\n") + 1
            entities.append(CodeEntity(
                name=m.group("name"), type="class",
                start_line=line_num, end_line=line_num,
                file_path=file_path, language=language,
                is_exported="export" in m.group(0),
            ))

        # ── TS interfaces ─────────────────────────────────────────────────────
        if language == "typescript":
            iface_re = re.compile(
                r"^(?:export\s+)?interface\s+(?P<name>\w+)", re.MULTILINE)
            for m in iface_re.finditer(source):
                line_num = source[: m.start()].count("\n") + 1
                entities.append(CodeEntity(
                    name=m.group("name"), type="interface",
                    start_line=line_num, end_line=line_num,
                    file_path=file_path, language="typescript",
                    is_exported="export" in m.group(0),
                ))

            # TS enums
            enum_re = re.compile(
                r"^(?:export\s+)?(?:const\s+)?enum\s+(?P<name>\w+)", re.MULTILINE)
            for m in enum_re.finditer(source):
                line_num = source[: m.start()].count("\n") + 1
                entities.append(CodeEntity(
                    name=m.group("name"), type="enum",
                    start_line=line_num, end_line=line_num,
                    file_path=file_path, language="typescript",
                    is_exported="export" in m.group(0),
                ))

        self.stats["regex"] += 1
        return {
            "language": language, "file_path": file_path,
            "entities": entities, "imports": imports,
            "backend": "regex", "raw_ast": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # JAVA PARSER
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_java(self, source: str, file_path: str) -> Dict[str, Any]:
        """Java : tree-sitter si dispo, sinon regex enrichi."""
        if "java" in self._ts_parsers:
            result = self._ts_parse_java(source, file_path)
            if "error" not in result:
                self.stats["treesitter"] += 1
                return result
            logger.debug(f"tree-sitter Java échoué sur {file_path}, fallback regex")

        return self._regex_parse_java(source, file_path)

    # ── tree-sitter Java ─────────────────────────────────────────────────────

    def _ts_parse_java(self, source: str, file_path: str) -> Dict[str, Any]:
        try:
            ts_parser = self._ts_parsers["java"]
            tree = ts_parser.parse(source.encode("utf-8"))
            entities: List[CodeEntity] = []
            imports:  List[ImportStatement] = []
            self._ts_walk_java(tree.root_node, source, file_path, entities, imports)
            return {
                "language": "java", "file_path": file_path,
                "entities": entities, "imports": imports,
                "backend": "treesitter", "raw_ast": None,
            }
        except Exception as e:
            return {"error": str(e), "file_path": file_path}

    def _ts_walk_java(self, node, source: str, file_path: str,
                       entities: list, imports: list):
        ntype = node.type

        if ntype == "import_declaration":
            static = any(c.type == "static" for c in node.children)
            name_node = next((c for c in node.children
                               if c.type in ("scoped_identifier", "identifier",
                                             "asterisk")), None)
            if name_node:
                module_raw = self._ts_node_text(name_node, source)
                imports.append(ImportStatement(
                    module=module_raw, items=[], alias=None,
                    file_path=file_path, line=node.start_point[0] + 1,
                    is_relative=False, import_type="java_import",
                    raw=self._ts_node_text(node, source),
                ))

        elif ntype in ("class_declaration", "interface_declaration",
                        "enum_declaration", "annotation_type_declaration",
                        "record_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                mods = self._ts_java_modifiers(node, source)
                etype = {
                    "class_declaration": "class",
                    "interface_declaration": "interface",
                    "enum_declaration": "enum",
                    "record_declaration": "class",
                    "annotation_type_declaration": "interface",
                }.get(ntype, "class")
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type=etype,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language="java",
                    is_exported="public" in mods,
                    decorators=mods,
                ))

        elif ntype in ("method_declaration", "constructor_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                mods = self._ts_java_modifiers(node, source)
                params = self._ts_extract_java_params(node, source)
                ret_node = node.child_by_field_name("type")
                etype = "constructor" if ntype == "constructor_declaration" else "method"
                entities.append(CodeEntity(
                    name=self._ts_node_text(name_node, source),
                    type=etype,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path, language="java",
                    parameters=params,
                    return_type=self._ts_node_text(ret_node, source) if ret_node else None,
                    is_static="static" in mods,
                    is_exported="public" in mods,
                ))

        for child in node.children:
            self._ts_walk_java(child, source, file_path, entities, imports)

    # ── regex Java (fallback amélioré) ───────────────────────────────────────

    def _regex_parse_java(self, source: str, file_path: str) -> Dict[str, Any]:
        """
        Fallback regex Java amélioré.
        Couvre : imports, classes, interfaces, enums, constructeurs,
                 méthodes (dont génériques), annotations.
        Évite les faux positifs dans les commentaires.
        """
        entities: List[CodeEntity] = []
        imports:  List[ImportStatement] = []

        # Supprimer les commentaires pour réduire les faux positifs
        source_clean = re.sub(r'//[^\n]*', '', source)
        source_clean = re.sub(r'/\*.*?\*/', '', source_clean, flags=re.DOTALL)

        # ── imports ───────────────────────────────────────────────────────────
        imp_re = re.compile(
            r"^\s*import\s+(?:static\s+)?([a-zA-Z_$][\w$.]*(?:\.\*)?)\s*;",
            re.MULTILINE,
        )
        for m in imp_re.finditer(source_clean):
            line_num = source[: m.start()].count("\n") + 1
            imports.append(ImportStatement(
                module=m.group(1), items=[], alias=None,
                file_path=file_path, line=line_num,
                is_relative=False, import_type="java_import",
            ))

        # ── classes / interfaces / enums / records ────────────────────────────
        type_re = re.compile(
            r"(?:@\w+\s*(?:\([^)]*\))?\s*)*"            # annotations
            r"(?:(?:public|protected|private)\s+)?"
            r"(?:(?:abstract|final|static|sealed|non-sealed)\s+)*"
            r"(?P<kw>class|interface|enum|record|@interface)"
            r"\s+(?P<name>[A-Z][\w$]*)",
            re.MULTILINE,
        )
        for m in type_re.finditer(source_clean):
            line_num = source_clean[: m.start()].count("\n") + 1
            kw_to_type = {
                "class": "class", "interface": "interface",
                "enum": "enum", "record": "class", "@interface": "interface",
            }
            entities.append(CodeEntity(
                name=m.group("name"),
                type=kw_to_type.get(m.group("kw"), "class"),
                start_line=line_num, end_line=line_num,
                file_path=file_path, language="java",
            ))

        # ── méthodes et constructeurs ─────────────────────────────────────────
        # ── constructeurs (nom majuscule, visibilite explicite, pas de type retour) ──
        ctor_re = re.compile(
            r"(?:(?:@\w+\s*(?:\([^)]*\))?\s*)+)?"
            r"(?:(?:public|protected|private)\s+)"
            r"(?P<n>[A-Z][\w$]*)\s*\((?P<params>[^)]*)\)"
            r"\s*(?:throws\s+[\w,\s]+)?\s*\{",
            re.MULTILINE,
        )
        for m in ctor_re.finditer(source_clean):
            name = m.group("n")
            line_num = source_clean[: m.start()].count("\n") + 1
            params = [p.strip().rsplit(" ", 1)[-1].lstrip("@")
                      for p in (m.group("params") or "").split(",") if p.strip()]
            entities.append(CodeEntity(
                name=name, type="constructor",
                start_line=line_num, end_line=line_num,
                file_path=file_path, language="java",
                parameters=params,
            ))

        # ── méthodes (nom minuscule, type retour obligatoire) ──────────────────
        method_re = re.compile(
            r"(?:(?:@\w+\s*(?:\([^)]*\))?\s*)+)?"
            r"(?:(?:public|protected|private)\s+)?"
            r"(?:(?:static|final|abstract|synchronized|native|default)\s+)*"
            r"(?:<[^>]+>\s+)?"
            r"(?P<ret>(?:void|[A-Z][\w$<>\[\],\s?]+?)\s+)"
            r"(?P<n>[a-z_$][\w$]*)\s*\((?P<params>[^)]*)\)"
            r"\s*(?:throws\s+[\w,\s]+)?\s*\{",
            re.MULTILINE,
        )
        JAVA_KW = {"if","while","for","switch","catch","try","new","return","throw","else","finally"}
        for m in method_re.finditer(source_clean):
            name = m.group("n")
            if name in JAVA_KW:
                continue
            line_num = source_clean[: m.start()].count("\n") + 1
            params = [p.strip().rsplit(" ", 1)[-1].lstrip("@")
                      for p in (m.group("params") or "").split(",") if p.strip()]
            ret = (m.group("ret") or "").strip()
            entities.append(CodeEntity(
                name=name, type="method",
                start_line=line_num, end_line=line_num,
                file_path=file_path, language="java",
                parameters=params,
                return_type=ret or None,
            ))

        self.stats["regex"] += 1
        return {
            "language": "java", "file_path": file_path,
            "entities": entities, "imports": imports,
            "backend": "regex", "raw_ast": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS TREE-SITTER
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ts_node_text(node, source: str) -> str:
        if node is None:
            return ""
        return source[node.start_byte: node.end_byte]

    @staticmethod
    def _ts_is_exported(node) -> bool:
        """Vérifie si un nœud est précédé d'un nœud 'export'."""
        parent = node.parent
        if parent is None:
            return False
        for child in parent.children:
            if child.type in ("export_statement", "export"):
                return True
            if child == node:
                break
        return parent.type == "export_statement"

    def _ts_extract_decorators(self, node, source: str) -> List[str]:
        """Extrait les décorateurs d'un nœud fonction/classe Python."""
        decorators = []
        parent = node.parent
        if parent is None:
            return decorators
        found = False
        for child in parent.children:
            if child == node:
                found = True
                break
            if child.type == "decorator":
                decorators.append(self._ts_node_text(child, source).lstrip("@"))
        return decorators if found else []

    def _ts_extract_python_params(self, node, source: str) -> List[str]:
        params_node = next((c for c in node.children if c.type == "parameters"), None)
        if not params_node:
            return []
        result = []
        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter",
                               "default_parameter", "typed_default_parameter",
                               "list_splat_pattern", "dictionary_splat_pattern"):
                raw = self._ts_node_text(child, source).split(":")[0].split("=")[0]
                raw = raw.lstrip("*").strip()
                if raw and raw not in ("self", "cls"):
                    result.append(raw)
        return result

    def _ts_extract_return_type_python(self, node, source: str) -> Optional[str]:
        ret_node = next((c for c in node.children if c.type == "type"), None)
        return self._ts_node_text(ret_node, source) if ret_node else None

    def _ts_extract_docstring(self, node, source: str) -> Optional[str]:
        """Extrait la première string du corps comme docstring."""
        body = next((c for c in node.children if c.type == "block"), None)
        if not body:
            return None
        for child in body.children:
            if child.type == "expression_statement":
                str_node = next((c for c in child.children
                                  if c.type in ("string", "concatenated_string")), None)
                if str_node:
                    raw = self._ts_node_text(str_node, source)
                    return raw.strip("\"'").strip()
        return None

    def _ts_extract_js_params(self, node, source: str) -> List[str]:
        params_node = next((c for c in node.children if c.type == "formal_parameters"), None)
        if not params_node:
            return []
        result = []
        for child in params_node.children:
            if child.type in ("identifier", "assignment_pattern",
                               "rest_pattern", "object_pattern",
                               "array_pattern", "required_parameter",
                               "optional_parameter"):
                raw = self._ts_node_text(child, source).split("=")[0].split(":")[0]
                raw = raw.lstrip("...").strip()
                if raw:
                    result.append(raw)
        return result

    @staticmethod
    def _ts_java_modifiers(node, source: str) -> List[str]:
        mods = []
        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    if mod.type in ("public", "private", "protected",
                                     "static", "final", "abstract"):
                        mods.append(mod.type)
        return mods

    def _ts_extract_java_params(self, node, source: str) -> List[str]:
        params_node = next((c for c in node.children
                             if c.type == "formal_parameters"), None)
        if not params_node:
            return []
        result = []
        for child in params_node.children:
            if child.type in ("formal_parameter", "spread_parameter",
                               "receiver_parameter"):
                name_node = next(
                    (c for c in child.children if c.type == "identifier"), None)
                if name_node:
                    result.append(self._ts_node_text(name_node, source))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS AST Python
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ast_extract_params(node: ast.FunctionDef) -> List[str]:
        """Extrait les paramètres (sans self/cls) depuis un nœud ast.FunctionDef."""
        args = node.args
        params = []
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.arg not in ("self", "cls"):
                params.append(arg.arg)
        if args.vararg:
            params.append("*" + args.vararg.arg)
        if args.kwarg:
            params.append("**" + args.kwarg.arg)
        return params

    # ─────────────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────────────

    def print_stats(self):
        total = sum(self.stats.values())
        print(f"\n{'─'*50}")
        print(" Parser stats :")
        for backend, count in self.stats.items():
            if count:
                pct = count / max(total, 1) * 100
                print(f"   • {backend:<14} {count:>5}  ({pct:.0f}%)")
        ts_langs = list(self._ts_parsers.keys()) or ["aucune"]
        print(f"   tree-sitter grammaires : {', '.join(ts_langs)}")
        print(f"{'─'*50}\n")


#instance globale (rétro-compatible)
parser = UniversalCodeParser()