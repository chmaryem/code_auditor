"""
compare_chat_orchestrators.py — outil d'aide à la validation MANUELLE.

Fait tourner le MÊME message sur les deux orchestrateurs (legacy et blackboard)
et affiche côte à côte : intent, agents/routing, nombre d'appels LLM, latence, et
un aperçu de la réponse — pour juger toi-même de la parité avant de basculer
CHAT_ORCHESTRATOR en production. Aucune assertion automatique : c'est un outil
d'inspection, pas un test (conformément à la validation manuelle uniquement).

Usage :
    python -m scripts.compare_chat_orchestrators "What changed since my last commit?"
    python -m scripts.compare_chat_orchestrators "Is my CI/CD pipeline ready to merge?" --project .
    python -m scripts.compare_chat_orchestrators "explique storage.py" --scope dashboard \\
        --attach demo.py:"def add(a,b): return a+b"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any, Dict


def _run(message: str, project_path: str, scope: str, attached_files: list) -> Dict[str, Any]:
    """Exécute le message sur les deux graphes (import direct des builders — ne
    dépend pas du flag CHAT_ORCHESTRATOR courant, donc compare toujours les deux)."""
    from langchain_agents.graphs.chat_graph import (
        build_chat_graph, build_blackboard_chat_graph, _initial_state,
    )

    results = {}
    for label, builder in (("legacy", build_chat_graph), ("blackboard", build_blackboard_chat_graph)):
        graph = builder()
        state = _initial_state(
            message=message, project_path=project_path,
            scope=scope, attached_files=attached_files,
        )
        t0 = time.time()
        r = asyncio.run(graph.ainvoke(state))
        elapsed = round(time.time() - t0, 2)
        results[label] = {
            "intent":        r.get("intent"),
            "routing":       (r.get("decision_plan") or {}).get("_routing"),
            "context_level": r.get("context_level"),
            "active_agents": r.get("active_agents", []),
            "llm_calls":     r.get("llm_calls", 0),
            "elapsed_s":     elapsed,
            "response":      (r.get("formatted_response") or r.get("response") or "").strip(),
        }
    return results


def _parse_attach(raw: list[str]) -> list[Dict[str, str]]:
    out = []
    for item in raw or []:
        if ":" in item:
            path, content = item.split(":", 1)
        else:
            path, content = item, ""
        out.append({"path": path, "content": content})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", help="Message développeur à tester")
    parser.add_argument("--project", default=".", help="project_path (défaut: .)")
    parser.add_argument("--scope", default="extension", choices=["extension", "dashboard"])
    parser.add_argument("--attach", action="append", default=[],
                         help='Fichier attaché "path:contenu" (répétable, mode dashboard)')
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    attached = _parse_attach(args.attach)
    results = _run(args.message, args.project, args.scope, attached)

    print(f"\nMessage : {args.message!r}  (scope={args.scope})\n")
    print(f"{'':14}{'legacy':<30}{'blackboard':<30}")
    for key in ("intent", "routing", "context_level", "llm_calls", "elapsed_s"):
        a = str(results["legacy"].get(key, ""))
        b = str(results["blackboard"].get(key, ""))
        flag = "  <-- DIFF" if a != b else ""
        print(f"{key:14}{a:<30}{b:<30}{flag}")
    print(f"{'active_agents':14}{str(results['legacy'].get('active_agents')):<30}"
          f"{str(results['blackboard'].get('active_agents')):<30}")

    print("\n--- legacy response ---")
    print(results["legacy"]["response"][:600])
    print("\n--- blackboard response ---")
    print(results["blackboard"]["response"][:600])

    diff_llm = results["legacy"]["llm_calls"] - results["blackboard"]["llm_calls"]
    print(f"\nÉcart d'appels LLM (legacy - blackboard) : {diff_llm:+d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
