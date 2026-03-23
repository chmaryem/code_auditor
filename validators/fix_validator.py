"""
FixValidator — Vérifie chaque correction proposée par l'IA avant affichage.

Problème résolu : l'IA peut inventer des numéros de ligne inexistants,
proposer du code non compilable, ou utiliser des imports absents du projet.
Ce validateur rejette ces cas avant qu'ils arrivent à l'utilisateur.
"""
import ast
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FixBlock:
    problem:      str
    severity:     str
    location:     str
    line_number:  Optional[int]
    current_code: str
    fixed_code:   str
    why:          str


class FixValidator:
    """
    Valide un bloc de correction proposé par le LLM.

    Usage :
        ok, raison = fix_validator.validate(bloc, source_code, "python")
        if not ok:
            print(f"[NON VÉRIFIÉ] {raison}")
    """

    def validate(self, block: FixBlock, source_code: str, language: str) -> tuple[bool, str]:
        checks = [
            self._check_line_exists(block, source_code),
            self._check_current_code_present(block, source_code),
            self._check_fixed_code_parseable(block, language),
            self._check_no_phantom_imports(block, source_code, language),
        ]
        failures = [msg for ok, msg in checks if not ok]
        return (len(failures) == 0), " | ".join(failures)

    def _check_line_exists(self, block: FixBlock, source: str) -> tuple[bool, str]:
        if block.line_number is None:
            return True, ""
        total = len(source.splitlines())
        if block.line_number > total:
            return False, f"ligne {block.line_number} inventée (fichier = {total} lignes)"
        return True, ""

    def _check_current_code_present(self, block: FixBlock, source: str) -> tuple[bool, str]:
        if not block.current_code:
            return True, ""
        first_line = block.current_code.strip().splitlines()[0].strip()
        if first_line and len(first_line) > 10 and first_line not in source:
            return False, f"code actuel introuvable : '{first_line[:50]}'"
        return True, ""

    def _check_fixed_code_parseable(self, block: FixBlock, language: str) -> tuple[bool, str]:
        if language != "python" or not block.fixed_code:
            return True, ""
        try:
            ast.parse(block.fixed_code)
            return True, ""
        except SyntaxError as e:
            return False, f"code corrigé invalide syntaxiquement : {e}"

    def _check_no_phantom_imports(self, block: FixBlock, source: str, language: str) -> tuple[bool, str]:
        if language != "python" or not block.fixed_code:
            return True, ""
        new_imports = set(re.findall(r'^(?:import|from)\s+(\S+)', block.fixed_code, re.MULTILINE))
        existing    = set(re.findall(r'^(?:import|from)\s+(\S+)', source, re.MULTILINE))
        stdlib_ok   = {"os","sys","re","json","logging","pathlib","typing",
                       "datetime","collections","abc","__future__"}
        phantom = new_imports - existing - stdlib_ok
        if phantom:
            return False, f"imports inexistants introduits : {phantom}"
        return True, ""


fix_validator = FixValidator()