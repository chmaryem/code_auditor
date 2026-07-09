"""
test_gap_scanner.py — Balayage complet d'un projet pour l'état de couverture de tests.

Rôle :
  Contrairement à TestGapAgent (utilisé par le WatchGraph, un seul fichier à la
  fois, ne remonte que les gaps significatifs), ce module balaie TOUT le projet
  et retourne le statut de CHAQUE fichier source (testé ou non) — utilisé pour
  peupler le panel Tests de l'extension à l'activation, indépendamment de Watch.

Réutilise :
  - CodeChangeHandler (watchers/file_watcher.py) pour le filtrage (extensions
    supportées, dossiers exclus, exclusion des fichiers de test eux-mêmes) —
    mêmes règles que Watch, pas de logique de filtrage dupliquée.
  - code_agent.parse() pour les entités par fichier (0 token, déterministe).
  - TestGapAgent.check() pour le statut de couverture par fichier.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def scan_project_test_gaps(project_path: Path, max_files: int = 500) -> List[Dict[str, Any]]:
    """
    Balaie `project_path` et retourne le statut de couverture de tests pour
    chaque fichier source (testé ET en gap — l'appelant construit l'image
    complète du panel Tests à partir de cette liste).
    """
    from agents.code_agent import code_agent
    from agents.test_gap_agent import get_test_gap_agent
    from watchers.file_watcher import CodeChangeHandler

    handler = CodeChangeHandler(callback=lambda *_a, **_k: None)
    gap_agent = get_test_gap_agent(project_path)

    results: List[Dict[str, Any]] = []
    for fp in project_path.rglob("*"):
        if len(results) >= max_files:
            logger.info("scan_project_test_gaps: max_files=%d atteint, arrêt anticipé", max_files)
            break
        if not fp.is_file() or not handler._should_process_file(fp):
            continue

        parsed = code_agent.parse(fp)
        if parsed.get("error"):
            continue

        try:
            status = gap_agent.check(source_file=fp, parsed_entities=parsed.get("entities", []))
        except Exception as e:
            logger.debug("TestGapAgent.check failed for %s: %s", fp.name, e)
            continue

        results.append({
            "file_path":         str(fp),
            "language":          parsed.get("language", ""),
            "has_test":          not status.missing,
            "test_file":         str(status.test_file) if status.test_file else None,
            "coverage_ratio":    status.coverage_ratio,
            "untested_entities": status.untested_entities,
            "tested_entities":   status.tested_entities,
            "framework":         status.framework,
            "reason":            status.reason,
        })

    return results
