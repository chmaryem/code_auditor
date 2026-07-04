"""
discovery_node.py — DiscoveryAgent.

Wraps steps 0–2 + 5 of the original generate_for_file():
  - Redis cache-in (test_generation_cache.get) — Memory read at graph entry
  - TestDiscoveryService.find_test_for → framework + convention
  - extension-based framework fallback
  - read the source file (error-out on failure)
  - build the target test path (Maven src/main → src/test mapping preserved)

On a cache hit, sets cache_hit=True and final_output=<cached dict> — the graph
routes straight to END, exactly like the original early return.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_nodes._helpers import (
    EXT_FRAMEWORK,
    build_target_path,
    detect_language,
)

logger = logging.getLogger(__name__)


def node_discovery(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]
    project_path = state["project_path"]
    incremental = state.get("incremental", False)
    discovery = state["_discovery"]

    # 0. Cache Redis — éviter un appel LLM si le fichier source n'a pas changé
    from services.test_generation_cache import test_generation_cache
    cached = test_generation_cache.get(source_path, incremental)
    if cached:
        return {"cache_hit": True, "final_output": cached}

    # 1. Discovery — conventions
    test_info = discovery.find_test_for(source_path)
    # Fallback: if discovery returns unknown, infer from file extension
    framework = (
        test_info.test_framework
        if test_info.test_framework and test_info.test_framework != "unknown"
        else EXT_FRAMEWORK.get(source_path.suffix.lower(), "pytest")
    )
    convention = test_info.convention_used or "{name}_test.py"

    # 2. Lecture du code source
    try:
        source_code = source_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"Lecture impossible : {e}",
                "final_output": {"error": f"Lecture impossible : {e}", "test_file": None}}

    # 5. Déterminer le chemin cible tôt pour le mode incrémentiel
    target_path = build_target_path(source_path, convention, project_path)
    language = detect_language(source_path)

    return {
        "cache_hit":   False,
        "framework":   framework,
        "convention":  convention,
        "source_code": source_code,
        "target_path": target_path,
        "language":    language,
    }
