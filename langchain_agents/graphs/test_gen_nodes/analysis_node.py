"""
analysis_node.py — AnalysisAgent.

Wraps steps 3–4 + the incremental-mode detection of generate_for_file():
  - extract_signatures (AST tree-sitter → regex fallback), ALL visibilities
  - extract_imports
  - extract_dependency_classes (mock guidance)
  - incremental mode: read the existing test file, filter out already-tested
    signatures. If nothing is untested, emit the existing file as final_output
    (fully-tested early return → END).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_nodes._helpers import (
    extract_dependency_classes,
    extract_imports,
    extract_signatures,
    filter_untested_signatures,
)

logger = logging.getLogger(__name__)


def node_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]
    source_code = state["source_code"]
    target_path = state["target_path"]
    framework = state["framework"]
    incremental = state.get("incremental", False)

    # 3. Parsing — toutes les signatures (public + private pour la validation)
    all_signatures = extract_signatures(source_code, source_path)

    # 4. Extraction des imports du fichier source
    source_imports = extract_imports(source_code, source_path)

    # Dépendances à mocker (guidance)
    dependency_classes = extract_dependency_classes(source_code, file_path=source_path)

    # ── Mode incrémentiel : détecter les entités déjà testées ────────────
    is_incremental = False
    existing_test_code = ""
    untested_signatures = all_signatures

    if incremental and target_path.exists():
        try:
            existing_test_code = target_path.read_text(encoding="utf-8", errors="replace")
            untested_signatures = filter_untested_signatures(
                all_signatures, existing_test_code
            )
            if not untested_signatures:
                logger.info("Mode incrémentiel : toutes les entités sont déjà testées dans %s", target_path.name)
                return {
                    "signatures": all_signatures,
                    "imports": source_imports,
                    "dependency_classes": dependency_classes,
                    "final_output": {
                        "test_file": target_path,
                        "test_code": existing_test_code,
                        "framework": framework,
                        "error": None,
                        "rag_docs_used": 0,
                        "validated": True,
                        "incremental": True,
                    },
                }
            is_incremental = True
            logger.info(
                "Mode incrémentiel : %d entité(s) sans test dans %s",
                len(untested_signatures), source_path.name,
            )
        except Exception as e:
            logger.warning("Mode incrémentiel : lecture du fichier de test existant échouée (%s), génération complète.", e)
            is_incremental = False
            untested_signatures = all_signatures

    return {
        "signatures":          all_signatures,
        "imports":             source_imports,
        "dependency_classes":  dependency_classes,
        "untested_signatures": untested_signatures,
        "is_incremental":      is_incremental,
        "existing_test_code":  existing_test_code,
    }
