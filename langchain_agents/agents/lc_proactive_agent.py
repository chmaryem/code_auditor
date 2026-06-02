"""
lc_proactive_agent.py — Proactive suggestions engine (Phase C2).

Instead of waiting for the developer to ask, this agent:
  - Detects uncommitted bugs (from GitSessionTracker + WatchGraph results)
  - Detects test gaps (files modified but not tested)
  - Detects CI risk patterns (file changed = previously failed CI stage)
  - Detects architectural coupling changes (dep graph impact)

Output format:
  {
    "suggestions": [
      {
        "type": "test_gap" | "uncommitted_bug" | "ci_risk" | "coupling",
        "severity": "info" | "warning" | "critical",
        "title": "short title",
        "message": "markdown message",
        "file":  "affected file",
        "action": "generate_tests" | "analyze_file" | "review_pr" | null
      }
    ],
    "total": N,
    "has_critical": bool
  }

Usage:
  - Called automatically by ChatGraph after node_memory_save
  - Also exposed via POST /api/chat/proactive for VS Code extension
  - VS Code extension shows suggestions in a side panel (CodeLens or notification)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProactiveAgent:
    """
    Proactive suggestion engine.

    Runs lightweight heuristics (no LLM on the critical path) to generate
    actionable suggestions shown to the developer without them asking.
    """

    SEVERITY_PRIORITY = {"critical": 0, "warning": 1, "info": 2}

    # ── Main entry ─────────────────────────────────────────────────────────────

    def scan(
        self,
        project_path: str,
        session_id: str = "",
        target_file: str = "",
        git_context: Optional[Dict[str, Any]] = None,
        ci_context: Optional[Dict[str, Any]] = None,
        max_suggestions: int = 5,
    ) -> Dict[str, Any]:
        """
        Run all proactive checks and return sorted suggestions.

        Args:
            project_path:  Project root
            session_id:    Current chat session (for semantic memory)
            target_file:   Currently focused file (for targeted checks)
            git_context:   From GitSessionTracker (optional)
            ci_context:    From CIGraph last result (optional)
            max_suggestions: Cap on returned suggestions

        Returns:
            {suggestions, total, has_critical, elapsed_ms}
        """
        t0 = time.time()
        suggestions: List[Dict[str, Any]] = []

        # Run all detectors
        suggestions.extend(self._check_uncommitted_bugs(project_path, git_context))
        suggestions.extend(self._check_test_gaps(project_path, target_file))
        suggestions.extend(self._check_ci_risk(project_path, ci_context, git_context))
        suggestions.extend(self._check_coupling_impact(project_path, target_file))

        # Sort by severity, deduplicate by title
        seen_titles: set = set()
        unique: List[Dict[str, Any]] = []
        for s in sorted(suggestions, key=lambda x: self.SEVERITY_PRIORITY.get(x["severity"], 99)):
            if s["title"] not in seen_titles:
                seen_titles.add(s["title"])
                unique.append(s)

        result_suggestions = unique[:max_suggestions]

        return {
            "suggestions":   result_suggestions,
            "total":         len(unique),
            "has_critical":  any(s["severity"] == "critical" for s in unique),
            "elapsed_ms":    round((time.time() - t0) * 1000),
        }

    # ── Detector 1: Uncommitted bugs from git session ──────────────────────────

    def _check_uncommitted_bugs(
        self,
        project_path: str,
        git_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        suggestions = []

        # Try to get fresh data if not provided
        if not git_context:
            try:
                from smart_git.git_session_tracker import GitSessionTracker
                tracker = GitSessionTracker(project_path)
                git_context = tracker.get_session_status() or {}
            except Exception:
                return []

        bugs = git_context.get("uncommitted_critical_bugs", [])
        risk_score = git_context.get("risk_score", 0)
        session_age_h = git_context.get("session_age_hours", 0)
        modified_files = git_context.get("modified_files", [])

        # Uncommitted critical bugs
        for bug in bugs[:3]:
            suggestions.append({
                "type":     "uncommitted_bug",
                "severity": "critical",
                "title":    f"Uncommitted bug in {Path(bug.get('file', '?')).name}",
                "message":  (
                    f"## ⚠️ Bug CRITICAL non commité\n\n"
                    f"**Fichier** : `{bug.get('file', '?')}`\n"
                    f"**Problème** : {bug.get('description', '?')[:150]}\n\n"
                    f"> Ce bug existe dans ton working tree depuis la dernière analyse. "
                    f"Commit ou fixe avant de continuer."
                ),
                "file":   bug.get("file", ""),
                "action": "analyze_file",
            })

        # High risk score warning
        if risk_score >= 70 and not bugs:
            files_str = ", ".join(Path(f).name for f in modified_files[:3])
            suggestions.append({
                "type":     "uncommitted_bug",
                "severity": "warning",
                "title":    f"Session à risque élevé (score {risk_score}/100)",
                "message":  (
                    f"## ⚠️ Session Git à Risque\n\n"
                    f"**Risk score** : {risk_score}/100\n"
                    f"**Fichiers modifiés** : {files_str or '?'}\n"
                    f"**Durée session** : {session_age_h:.1f}h\n\n"
                    f"> Beaucoup de changements non commités. "
                    f"Pense à commit pour sécuriser ton travail."
                ),
                "file":   "",
                "action": None,
            })

        return suggestions

    # ── Detector 2: Test gaps (modified files without tests) ──────────────────

    def _check_test_gaps(
        self,
        project_path: str,
        target_file: str,
    ) -> List[Dict[str, Any]]:
        suggestions = []

        try:
            from agents.test_gap_agent import TestGapAgent
            agent = TestGapAgent()
            gaps = agent.detect_gaps_for_file(target_file or project_path) if target_file else []
        except Exception:
            # Fallback: check if target file has a corresponding test file
            gaps = []
            if target_file:
                fp = Path(target_file)
                project_root = Path(project_path)
                test_candidates = [
                    project_root / "tests" / f"test_{fp.name}",
                    project_root / "tests" / f"test_{fp.stem}.py",
                    fp.parent / f"test_{fp.name}",
                ]
                has_test = any(t.exists() for t in test_candidates)
                if not has_test and fp.suffix == ".py":
                    gaps = [{"file": str(fp), "missing_tests": ["(no test file found)"]}]

        for gap in gaps[:2]:
            file_name = Path(gap.get("file", target_file or "?")).name
            missing = gap.get("missing_tests", [])
            suggestions.append({
                "type":     "test_gap",
                "severity": "warning",
                "title":    f"Tests manquants pour {file_name}",
                "message":  (
                    f"## 🧪 Test Gap Détecté — `{file_name}`\n\n"
                    f"Ce fichier n'a pas de couverture de tests suffisante.\n\n"
                    + (f"**Tests manquants** :\n" + "\n".join(f"- `{m}`" for m in missing[:5]) if missing else "")
                    + f"\n\n> Veux-tu que je génère les tests unitaires ?"
                ),
                "file":   gap.get("file", target_file or ""),
                "action": "generate_tests",
            })

        return suggestions

    # ── Detector 3: CI risk (file changed = historically failed CI) ───────────

    def _check_ci_risk(
        self,
        project_path: str,
        ci_context: Optional[Dict[str, Any]],
        git_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        suggestions = []

        if not ci_context:
            return []

        outcome       = ci_context.get("outcome", "")
        failure_type  = ci_context.get("failure_type", "")
        stage_failed  = ci_context.get("stage_failed", "")
        modified_files = (git_context or {}).get("modified_files", [])

        if outcome == "failure":
            severity = "critical" if failure_type == "security" else "warning"
            suggestions.append({
                "type":     "ci_risk",
                "severity": severity,
                "title":    f"CI échoué — {stage_failed or failure_type or 'unknown'}",
                "message":  (
                    f"## ❌ Pipeline CI en Échec\n\n"
                    f"**Stage** : `{stage_failed or '?'}`  \n"
                    f"**Type** : `{failure_type or '?'}`\n\n"
                    f"Tu travailles sur des fichiers qui ont causé un échec CI récent.\n\n"
                    f"> Veux-tu que j'analyse la cause racine ?"
                ),
                "file":   "",
                "action": "ci_analyze",
            })
        elif outcome == "success" and modified_files:
            # Check if current changes touch files that historically fail
            risky = [f for f in modified_files
                     if any(k in Path(f).name.lower() for k in ("deploy", "config", "security", "auth", "docker"))]
            if risky:
                suggestions.append({
                    "type":     "ci_risk",
                    "severity": "info",
                    "title":    f"Fichiers à risque CI modifiés",
                    "message":  (
                        f"## ℹ️ Fichiers Sensibles Modifiés\n\n"
                        f"Tu as modifié : {', '.join(Path(f).name for f in risky[:3])}\n\n"
                        f"Ces fichiers sont historiquement sensibles pour le pipeline CI.\n\n"
                        f"> Le dernier CI était ✅. Reste vigilant."
                    ),
                    "file":   risky[0] if risky else "",
                    "action": None,
                })

        return suggestions

    # ── Detector 4: Architectural coupling impact ──────────────────────────────

    def _check_coupling_impact(
        self,
        project_path: str,
        target_file: str,
    ) -> List[Dict[str, Any]]:
        if not target_file:
            return []

        try:
            from services.graph_service import dependency_builder
            dep_graph = dependency_builder.build_from_project(Path(project_path))

            # Count dependents
            node_key = str(Path(target_file).relative_to(project_path))
            dependents = list(dep_graph.predecessors(node_key))

            if len(dependents) >= 5:
                return [{
                    "type":     "coupling",
                    "severity": "warning",
                    "title":    f"{Path(target_file).name} impacte {len(dependents)} modules",
                    "message":  (
                        f"## 🔗 Impact Architectural Élevé\n\n"
                        f"`{Path(target_file).name}` est utilisé par **{len(dependents)} autres modules**.\n\n"
                        f"**Dépendants** : {', '.join(Path(d).name for d in dependents[:5])}\n\n"
                        f"> Toute modification peut avoir un effet de cascade. "
                        f"Pense à lancer les tests de tous les modules dépendants."
                    ),
                    "file":   target_file,
                    "action": "analyze_file",
                }]
        except Exception:
            pass

        return []


proactive_agent = ProactiveAgent()
