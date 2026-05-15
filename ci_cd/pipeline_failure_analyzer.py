"""
pipeline_failure_analyzer.py — Agent LangChain pour l'analyse root-cause CI.

4 Pillars :
  LLM     : LLM Factory (Gemini → Groq cascade) via invoke_with_fallback()
  Tools   : tool_sonar_quality_gate, tool_sonar_get_issues, tool_search_similar_failure
  Memory  : Redis via CILogsIndexer (failures passées + fixes)
  Planning : Si fix similaire Redis (confidence > 0.8) → poster directement
              Sinon → LLM analyse logs + SonarCloud data → suggère fix

Usage :
    analyzer = PipelineFailureAnalyzer()
    result = analyzer.analyze(
        run_id="123456789",
        repo="chmaryem/myapp",
        owner="chmaryem",
        project_key="chmaryem_myapp",
        logs="build failed: ...",
        stage_failed="build-test",
        pr_number=42,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """\
You are an expert CI/CD DevOps engineer analyzing a pipeline failure.

## Pipeline Information
- Repository: {repo}
- Failed Stage: {stage_failed}
- Failure Type: {failure_type}
- PR Number: {pr_number}

## Raw Logs (last 3000 chars)
```
{logs}
```

## SonarCloud Quality Gate
Status: {gate_status}
Conditions failing:
{gate_conditions}

## SonarCloud Issues (Critical/Blocker)
{sonar_issues}

## Task
1. Identify the ROOT CAUSE of this pipeline failure in 2-3 sentences max.
2. Provide a CONCRETE FIX (commands, code snippet, or configuration change).
3. Keep your response concise and actionable.

## Format your response EXACTLY as:
ROOT CAUSE:
<2-3 sentences explaining the root cause>

SUGGESTED FIX:
<specific commands or code to fix the issue>
"""


class PipelineFailureAnalyzer:
    """
    Analyse les failures CI/CD en croisant :
      1. Logs GitHub Actions (bruts)
      2. Quality Gate SonarCloud (via MCP)
      3. Issues critiques SonarCloud (via MCP)
      4. Failures similaires déjà vues (Redis)

    Stratégie :
      - Si fix similaire trouvé (confidence > 0.8) → utiliser directement
      - Sinon → LLM Factory (Gemini → Groq) pour analyse complète
    """

    CONFIDENCE_THRESHOLD = 0.8  # Au-delà → pas besoin du LLM

    def __init__(self):
        self._indexer = None
        self._llm = None

    def _get_indexer(self):
        if self._indexer is None:
            from ci_cd.ci_logs_indexer import CILogsIndexer
            self._indexer = CILogsIndexer()
        return self._indexer

    def _get_llm(self):
        if self._llm is None:
            try:
                from services.llm_factory import invoke_with_fallback
                self._llm = invoke_with_fallback
            except Exception as e:
                logger.warning("LLM Factory unavailable: %s", e)
        return self._llm

    # ── Core Analysis ─────────────────────────────────────────────────────────

    def analyze(
        self,
        run_id: str,
        repo: str,
        owner: str,
        logs: str,
        stage_failed: str = "",
        failure_type: str = "unknown",
        project_key: str = "",
        pr_number: Optional[int] = None,
        head_sha: str = "",
    ) -> Dict[str, Any]:
        """
        Analyse complète d'une failure CI.

        Returns:
            {
                "root_cause": str,
                "suggested_fix": str,
                "confidence": float,
                "source": "redis_cache" | "llm_analysis",
                "sonar_gate": dict,
                "sonar_metrics": dict,
                "sonar_issues": list,
                "similar_fixes": list,
            }
        """
        result = {
            "root_cause": None,
            "suggested_fix": None,
            "confidence": 0.0,
            "source": "none",
            "sonar_gate": {},
            "sonar_metrics": {},
            "sonar_issues": [],
            "similar_fixes": [],
        }

        # ── Step 1: Redis — Chercher des fixes similaires ────────────────────
        indexer = self._get_indexer()
        similar = indexer.search_similar(logs, max_results=5)
        result["similar_fixes"] = similar

        # Fix avec fix_applied déjà documenté
        best_fix = None
        best_confidence = 0.0
        for match in similar:
            if match.get("fix_applied") and match["similarity"] >= best_confidence:
                best_fix = match
                best_confidence = match["similarity"]

        result["confidence"] = best_confidence

        # ── Step 2: SonarCloud MCP — Quality Gate + Issues ───────────────────
        if project_key:
            try:
                from services.mcp_sonarqube_service import get_mcp_sonarqube
                sonar = get_mcp_sonarqube()
                result["sonar_gate"] = sonar.get_quality_gate_status(project_key)
                result["sonar_metrics"] = sonar.get_project_metrics(project_key)
                result["sonar_issues"] = sonar.get_issues(
                    project_key, severity="CRITICAL", issue_type="VULNERABILITY"
                )
                # Ajouter aussi les bugs bloquants
                blockers = sonar.get_issues(
                    project_key, severity="BLOCKER", issue_type="BUG"
                )
                result["sonar_issues"].extend(blockers)
            except Exception as e:
                logger.debug("SonarCloud MCP unavailable: %s", e)

        # ── Step 3: Décision — Cache Redis ou LLM ? ──────────────────────────
        if best_fix and best_confidence >= self.CONFIDENCE_THRESHOLD:
            # Haute confiance → utiliser le fix Redis directement
            result["root_cause"] = (
                f"Erreur similaire vue {best_fix['count']}x "
                f"(stage: {best_fix.get('stage', stage_failed) or stage_failed}). "
                f"Correspondance Redis : {int(best_confidence * 100)}%."
            )
            result["suggested_fix"] = best_fix["fix_applied"]
            result["source"] = "redis_cache"
            logger.info(
                "[Analyzer] Fix Redis utilisé (conf=%.2f) pour %s/%s",
                best_confidence, repo, run_id
            )
        else:
            # Basse confiance → LLM analysis
            result.update(self._llm_analyze(
                run_id=run_id,
                repo=repo,
                logs=logs,
                stage_failed=stage_failed,
                failure_type=failure_type,
                pr_number=pr_number,
                sonar_gate=result["sonar_gate"],
                sonar_issues=result["sonar_issues"],
            ))
            result["source"] = "llm_analysis"

        return result

    def _llm_analyze(
        self,
        run_id: str,
        repo: str,
        logs: str,
        stage_failed: str,
        failure_type: str,
        pr_number: Optional[int],
        sonar_gate: Dict[str, Any],
        sonar_issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyse LLM complète (Gemini → Groq fallback)."""

        invoke = self._get_llm()
        if invoke is None:
            return {
                "root_cause": "LLM non disponible pour l'analyse.",
                "suggested_fix": "Vérifier les logs manuellement.",
            }

        # Préparer le contexte
        gate_status = sonar_gate.get("status", "N/A")
        gate_conditions = ""
        for cond in sonar_gate.get("conditions", []):
            if cond.get("status") != "OK":
                gate_conditions += (
                    f"  - {cond.get('metric', '?')}: "
                    f"{cond.get('actualValue', '?')} "
                    f"(seuil: {cond.get('errorThreshold', '?')})\n"
                )
        if not gate_conditions:
            gate_conditions = "  Aucune condition en échec.\n"

        issues_text = ""
        for i in sonar_issues[:10]:
            issues_text += (
                f"  - [{i.get('severity', '?')}] {i.get('rule', '?')}: "
                f"{i.get('message', '')[:100]}\n"
            )
        if not issues_text:
            issues_text = "  Aucune issue critique.\n"

        prompt = _ANALYSIS_PROMPT.format(
            repo=repo,
            stage_failed=stage_failed or "non spécifié",
            failure_type=failure_type,
            pr_number=f"#{pr_number}" if pr_number else "N/A (push direct)",
            logs=logs[-3000:] if logs else "(logs non disponibles)",
            gate_status=gate_status,
            gate_conditions=gate_conditions,
            sonar_issues=issues_text,
        )

        try:
            logger.info("[Analyzer] Invocation LLM pour %s/%s...", repo, run_id)
            response_text = invoke(prompt)

            # Parsing du format ROOT CAUSE / SUGGESTED FIX
            root_cause, suggested_fix = self._parse_llm_response(response_text)
            return {
                "root_cause": root_cause,
                "suggested_fix": suggested_fix,
            }

        except Exception as e:
            logger.error("[Analyzer] LLM invoke failed: %s", e)
            return {
                "root_cause": f"Analyse LLM échouée : {e}",
                "suggested_fix": "Vérifier les logs du run manuellement.",
            }

    @staticmethod
    def _parse_llm_response(text: str) -> tuple[str, str]:
        """Parse la réponse LLM en (root_cause, suggested_fix)."""
        if not text:
            return "Analyse non disponible.", "Fix non disponible."

        root_cause = ""
        suggested_fix = ""
        current_section = None

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("ROOT CAUSE:"):
                current_section = "root"
                rest = stripped[len("ROOT CAUSE:"):].strip()
                if rest:
                    root_cause += rest + "\n"
            elif stripped.startswith("SUGGESTED FIX:"):
                current_section = "fix"
                rest = stripped[len("SUGGESTED FIX:"):].strip()
                if rest:
                    suggested_fix += rest + "\n"
            elif current_section == "root":
                root_cause += line + "\n"
            elif current_section == "fix":
                suggested_fix += line + "\n"

        # Fallback si le LLM n'a pas respecté le format
        if not root_cause:
            root_cause = text[:500]
        if not suggested_fix:
            suggested_fix = "Voir l'analyse complète ci-dessus."

        return root_cause.strip(), suggested_fix.strip()

    # ── Generate PR Comment ───────────────────────────────────────────────────

    def format_pr_comment(
        self,
        analysis: Dict[str, Any],
        run_id: str,
        repo: str,
        stage_failed: str = "",
        failure_type: str = "unknown",
        codeql_alerts: Optional[List[Dict[str, Any]]] = None,
        trivy_report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Génère un commentaire Markdown structuré pour la PR.

        Args:
            analysis: Résultat de analyze()
            run_id: GitHub Actions run ID
            repo: "{owner}/{repo}"
            stage_failed: Stage qui a échoué
            failure_type: Type de failure
            codeql_alerts: Alertes CodeQL [{rule_id, severity, description, location_path}]
            trivy_report: CVEs Trivy {critical:[...], high:[...], total:N}

        Returns:
            Commentaire Markdown prêt à poster
        """
        gate = analysis.get("sonar_gate", {})
        metrics = analysis.get("sonar_metrics", {})
        issues = analysis.get("sonar_issues", [])
        root_cause = analysis.get("root_cause", "")
        suggested_fix = analysis.get("suggested_fix", "")
        source = analysis.get("source", "")
        confidence = analysis.get("confidence", 0.0)
        gate_status = gate.get("status", "N/A")
        codeql_alerts = codeql_alerts or []
        trivy_report = trivy_report or {}

        gate_badge = (
            "🟢 OK" if gate_status == "OK"
            else "🔴 FAILED" if gate_status == "ERROR"
            else "🟡 WARN"
        )

        comment = f"""## 🚨 Code Auditor — CI/CD Failure Analysis

**Run:** [`{run_id}`](https://github.com/{repo}/actions/runs/{run_id})
**Stage échoué:** `{stage_failed or 'N/A'}` | **Type:** `{failure_type}`

---

### 📊 SonarCloud Quality Gate : {gate_badge}

"""
        if metrics:
            cov = metrics.get("coverage", "N/A")
            bugs = metrics.get("bugs", "N/A")
            vulns = metrics.get("vulnerabilities", "N/A")
            comment += f"| Métrique | Valeur |\n|---|---|\n"
            comment += f"| Coverage | {cov}% |\n"
            comment += f"| Bugs | {bugs} |\n"
            comment += f"| Vulnerabilities | {vulns} |\n\n"

        if issues:
            comment += f"**Issues critiques ({len(issues)}) :**\n"
            for i in issues[:5]:
                comment += f"- `[{i.get('severity', '?')}]` **{i.get('rule', '')}** — {i.get('message', '')[:100]}\n"
            comment += "\n"

        # ── Section Sécurité (T3) — CodeQL + Trivy ───────────────────────────
        trivy_total = trivy_report.get("total", 0)
        trivy_crit  = trivy_report.get("critical", [])
        trivy_high  = trivy_report.get("high", [])

        if codeql_alerts or trivy_total > 0:
            comment += "---\n\n### 🔒 Sécurité\n\n"

            if codeql_alerts:
                comment += f"#### CodeQL — {len(codeql_alerts)} vulnérabilité(s) détectée(s)\n\n"
                comment += "| Règle | Sévérité | Fichier | Ligne |\n|---|---|---|---|\n"
                for a in codeql_alerts[:8]:
                    sev_icon = "🔴" if a.get("severity") in ("critical", "error") else "🟠"
                    path = a.get("location_path", "")
                    line = a.get("location_line", "")
                    loc  = f"`{path}:{line}`" if path else "—"
                    comment += f"| `{a.get('rule_id', '')}` | {sev_icon} {a.get('severity', '')} | {loc} | |\n"
                comment += "\n"
                if codeql_alerts:
                    comment += f"> 🔗 [Voir toutes les alertes CodeQL](https://github.com/{repo}/security/code-scanning)\n\n"

            if trivy_total > 0:
                comment += f"#### Trivy — {trivy_total} CVE(s) (🔴 CRITICAL: {len(trivy_crit)}, 🟠 HIGH: {len(trivy_high)})\n\n"
                if trivy_crit:
                    comment += "**CVEs CRITIQUES :**\n"
                    for cve in trivy_crit[:5]:
                        comment += (
                            f"- **`{cve.get('id', '?')}`** — `{cve.get('pkg', '')}:{cve.get('installed', '')}`"
                            f" → Fix : `{cve.get('fixed_version', 'N/A')}` — {cve.get('title', '')[:80]}\n"
                        )
                    comment += "\n"
                if trivy_high:
                    comment += "**CVEs HIGH :**\n"
                    for cve in trivy_high[:3]:
                        comment += (
                            f"- **`{cve.get('id', '?')}`** — `{cve.get('pkg', '')}:{cve.get('installed', '')}`"
                            f" → Fix : `{cve.get('fixed_version', 'N/A')}`\n"
                        )
                    comment += "\n"

        # ── Root cause + Fix ─────────────────────────────────────────────────
        if root_cause:
            comment += f"---\n\n### 🔍 Cause Racine\n\n{root_cause}\n\n"

        if suggested_fix:
            comment += f"---\n\n### 🔧 Fix Suggéré\n\n"
            if any(c in suggested_fix for c in ("```", "pip ", "mvn ", "npm ", "$", "def ", "import ")):
                comment += f"```\n{suggested_fix}\n```\n\n"
            else:
                comment += f"{suggested_fix}\n\n"

        if source == "redis_cache":
            comment += f"> 💡 *Fix basé sur {len(analysis.get('similar_fixes', []))} failure(s) similaire(s) — confiance : {int(confidence * 100)}%*\n"
        else:
            comment += f"> 🤖 *Analyse générée par Code Auditor AI (LLM Factory)*\n"

        return comment
