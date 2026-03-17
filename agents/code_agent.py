"""
CodeAgent — Lit et parse le code source.
Rôle UNIQUE : transformer un fichier brut en données structurées.
"""
from pathlib import Path
from typing import Any, Dict

from services.code_parser import parser as _parser


class CodeAgent:
    def parse(self, file_path: Path) -> Dict[str, Any]:
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
            "language":   self._detect_language(file_path),
            "entities":   parsed.get("entities", []),
            "imports":    parsed.get("imports", []),
            "code":       code,
            "line_count": len(code.splitlines()),
            "error":      None,
        }

    def _detect_language(self, p: Path) -> str:
        return {".py":"python",".js":"javascript",".jsx":"javascript",
                ".ts":"typescript",".tsx":"typescript",".java":"java"
                }.get(p.suffix.lower(), "unknown")

    def _error(self, p: Path, msg: str) -> Dict[str, Any]:
        return {"file_path":str(p),"language":"unknown","entities":[],"imports":[],
                "code":"","line_count":0,"error":msg}


code_agent = CodeAgent()