"""
api/diff_utils.py — Utilitaire de calcul de diff pour les fixes WS.

Calcule des hunks de diff entre le code original et le code corrigé
en utilisant uniquement difflib (stdlib Python — zéro dépendance externe).

Ces hunks permettent au plugin VS Code de savoir EXACTEMENT quelles lignes
ont changé sans recevoir le fichier entier en payload WebSocket.
"""
from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def compute_diff_hunks(
    original: str,
    fixed: str,
    context_lines: int = 0,
) -> List[Dict[str, Any]]:
    """
    Calcule des hunks de diff entre ``original`` et ``fixed``.

    Utilise ``difflib.SequenceMatcher`` pour produire des opérations de
    modification minimales (replace/insert/delete).

    Args:
        original:       Code source original (avant correction).
        fixed:          Code source corrigé (après correction).
        context_lines:  Nombre de lignes de contexte autour de chaque hunk
                        (0 = seulement les lignes modifiées).

    Returns:
        Liste de dicts conformes au schéma ``WSDiffHunk`` :
        {
            "start_line":      int,   # 1-indexed, ligne de début dans l'original
            "end_line":        int,   # 1-indexed, ligne de fin dans l'original (inclusive)
            "original_lines":  list,  # lignes supprimées / remplacées
            "new_lines":       list,  # lignes insérées / remplaçantes
            "context":         str,   # description de l'opération
        }
    """
    if not original and not fixed:
        return []

    if original == fixed:
        return []

    orig_lines = original.splitlines()
    fixed_lines = fixed.splitlines()

    hunks: List[Dict[str, Any]] = []

    try:
        matcher = difflib.SequenceMatcher(None, orig_lines, fixed_lines, autojunk=False)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            # Expand context lines (clamped to array bounds)
            ctx_i1 = max(0, i1 - context_lines)
            ctx_i2 = min(len(orig_lines), i2 + context_lines)
            ctx_j1 = max(0, j1 - context_lines)
            ctx_j2 = min(len(fixed_lines), j2 + context_lines)

            hunks.append({
                "start_line":     i1 + 1,          # 1-indexed
                "end_line":       i2 if i2 > i1 else i1 + 1,
                "original_lines": orig_lines[ctx_i1:ctx_i2],
                "new_lines":      fixed_lines[ctx_j1:ctx_j2],
                "context":        _describe_hunk(tag, i1, i2, j1, j2),
            })
    except Exception as exc:
        logger.debug("compute_diff_hunks failed: %s", exc)
        return []

    return hunks


def _describe_hunk(tag: str, i1: int, i2: int, j1: int, j2: int) -> str:
    """Retourne une description lisible de l'opération de diff."""
    if tag == "replace":
        n_old = i2 - i1
        n_new = j2 - j1
        return f"Replace {n_old} line(s) with {n_new} line(s) at line {i1 + 1}"
    if tag == "insert":
        return f"Insert {j2 - j1} line(s) at line {i1 + 1}"
    if tag == "delete":
        return f"Delete {i2 - i1} line(s) at line {i1 + 1}"
    return f"{tag} at line {i1 + 1}"


def truncate_code_for_ws(code: str, max_chars: int = 300) -> str:
    """
    Tronque le code pour les payloads WebSocket de stratégie ``full_file``.

    Le plugin utilise les ``diff_hunks`` pour l'affichage ; ``current_code``
    et ``fixed_code`` ne sont gardés que comme fallback de compatibilité.
    """
    if not code or len(code) <= max_chars:
        return code
    lines = code.splitlines()
    truncated = "\n".join(lines[:5])
    return f"{truncated}\n... [{len(lines)} lines total, use diff_hunks for details]"
