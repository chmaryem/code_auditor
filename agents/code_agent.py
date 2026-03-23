"""
CodeAgent — Parse le code ET analyse l'importance des changements.

Responsabilités :
  1. parse()          : lit un fichier → entités, imports, langage (existant)
  2. analyze_change() : compare ancien/nouveau code → décide si analyse LLM nécessaire
     (ChangeAnalyzer migré depuis incremental_analyzer.py lignes 695-783)
  3. detect_language(): détecte le langage depuis l'extension
"""
import difflib
from pathlib import Path
from typing import Any, Dict

from services.code_parser import parser as _parser


# ── ChangeAnalyzer ────────────────────────────────────────────────────────────

class ChangeAnalyzer:
    """
    Analyse l'importance d'un changement pour décider s'il faut appeler le LLM.
    Changements mineurs (whitespace, commentaires, imports seuls) → RAS immédiat.
    Migré depuis incremental_analyzer.py
    """

    MINOR_TYPES = frozenset({
        "whitespace_only", "comment_only", "import_only", "docstring_only", "no_change"
    })

    @staticmethod
    def analyze_change(old_content: str, new_content: str) -> Dict[str, Any]:
        if not old_content and not new_content:
            return {"significant": False, "score": 0, "lines_changed": 0,
                    "change_type": "no_change", "reason": "Aucun contenu"}

        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines()
        diff      = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
        added     = [l[1:].strip() for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed   = [l[1:].strip() for l in diff if l.startswith("-") and not l.startswith("---")]

        lines_changed = len(added) + len(removed)
        change_type   = ChangeAnalyzer._classify_change(added, removed)
        score         = ChangeAnalyzer._calculate_score(change_type, lines_changed, added, removed)
        significant   = score >= 20
        reason        = ChangeAnalyzer._get_reason(change_type, lines_changed, significant)

        return {"significant": significant, "score": score,
                "lines_changed": lines_changed, "change_type": change_type, "reason": reason}

    @staticmethod
    def _classify_change(added, removed):
        all_lines = added + removed
        if not all_lines: return "no_change"
        non_empty = [l for l in all_lines if l]
        if not non_empty: return "whitespace_only"
        if all(l.startswith(("import ", "from ")) for l in non_empty): return "import_only"
        if all(l.startswith(("#", "//", "/*", "*", "*/")) for l in non_empty): return "comment_only"
        if all('"""' in l or "'''" in l for l in non_empty): return "docstring_only"
        if any("def " in l or "class " in l or "function " in l for l in added):
            if not any("def " in l or "class " in l or "function " in l for l in removed):
                return "new_function"
        if (any("def " in l or "class " in l for l in added) and
                any("def " in l or "class " in l for l in removed)):
            return "function_signature"
        return "logic_change"

    @staticmethod
    def _calculate_score(change_type, lines_changed, added, removed):
        base_score  = lines_changed * 10
        type_scores = {
            "import_only": 5, "comment_only": 0, "whitespace_only": 0,
            "docstring_only": 10, "new_function": max(base_score, 50),
            "function_signature": max(base_score, 70), "logic_change": max(base_score, 30),
        }
        return min(type_scores.get(change_type, base_score), 100)

    @staticmethod
    def _get_reason(change_type, lines_changed, significant):
        if not significant:
            return {
                "import_only":     f"Import seulement ({lines_changed} ligne(s))",
                "comment_only":    "Commentaires seulement",
                "whitespace_only": "Formatage seulement",
                "docstring_only":  "Documentation seulement",
                "no_change":       "Aucun changement",
            }.get(change_type, f"Changement mineur ({lines_changed} ligne(s))")
        return {
            "logic_change":       f"Logique modifiée ({lines_changed} ligne(s))",
            "new_function":       f"Nouvelle fonction ({lines_changed} ligne(s))",
            "function_signature": f"Signature modifiée ({lines_changed} ligne(s)) — Impact possible",
        }.get(change_type, f"Changement important ({lines_changed} ligne(s))")


# ── CodeAgent ─────────────────────────────────────────────────────────────────

class CodeAgent:
    """
    Parse un fichier de code et analyse l'importance des changements.

    Usage :
        parsed      = code_agent.parse(Path("fichier.py"))
        change_info = code_agent.analyze_change(old_content, new_content)
    """

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Lit et parse un fichier → entités, imports, langage, code brut."""
        if not file_path.exists():
            return self._error(file_path, "Fichier introuvable")
        try:
            code = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return self._error(file_path, str(e))
        try:
            parsed = _parser.parse_file(file_path)
        except Exception:
            parsed = {"entities": [], "imports": []}
        return {
            "file_path":  str(file_path),
            "language":   self.detect_language(file_path),
            "entities":   parsed.get("entities", []),
            "imports":    parsed.get("imports", []),
            "code":       code,
            "line_count": len(code.splitlines()),
            "error":      None,
        }

    def analyze_change(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """Compare ancien/nouveau contenu → décide si analyse LLM nécessaire."""
        return ChangeAnalyzer.analyze_change(old_content, new_content)

    @staticmethod
    def detect_language(file_path: Path) -> str:
        return {".py": "python", ".js": "javascript", ".jsx": "javascript",
                ".ts": "typescript", ".tsx": "typescript", ".java": "java"
                }.get(file_path.suffix.lower(), "unknown")

    def _error(self, p: Path, msg: str) -> Dict[str, Any]:
        return {"file_path": str(p), "language": "unknown", "entities": [],
                "imports": [], "code": "", "line_count": 0, "error": msg}


code_agent = CodeAgent()