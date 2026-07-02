"""
ci_graph.py — CIGraph LangGraph StateGraph pour le pipeline CI/CD Intelligence.

Orchestration complète :
  fetch_run → classify_failure → [success→index_result | failure→sonar_mcp_query]
  → search_similar → [found_fix→generate_comment | needs_llm→analyze_root_cause]
  → generate_fix → post_pr_comment → index_result → notify → END

Déclenché en mode polling via :
  from langchain_agents.graphs.ci_graph import invoke_ci_run
  result = invoke_ci_run(run_id, repo, owner, project_key, pr_number)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from langchain_agents.graphs.state import CIState

logger = logging.getLogger(__name__)


# ── Node implementations ───────────────────────────────────────────────────────

def node_fetch_run(state: CIState) -> CIState:
    """Récupère les logs du run GitHub Actions."""
    from langchain_agents.tools.ci_tools import (
        tool_fetch_run_logs,
        tool_fetch_job_logs,
        tool_fetch_workflow_runs,
        tool_classify_run,
    )

    run_id = state.get("run_id", "")
    owner = state.get("owner", "")
    repo_name = state.get("repo", "").split("/")[-1] if "/" in state.get("repo", "") else state.get("repo", "")
    job_id = state.get("job_id", "")

    # Récupérer les logs (retourne maintenant un dict avec stage_failed)
    logs = ""
    detected_stage = ""
    if job_id and owner and repo_name:
        # Analyse ciblée : le dashboard a demandé l'analyse d'une issue
        # précise → on récupère UNIQUEMENT les logs de ce job, pas ceux de
        # tout le run. stage_failed vient de l'override explicite du state
        # (nom réel du job, transmis par le frontend), pas d'une détection.
        try:
            result = tool_fetch_job_logs.invoke({
                "owner": owner,
                "repo": repo_name,
                "job_id": str(job_id),
            })
            logs = result.get("logs", "")
        except Exception as e:
            logger.debug("fetch_job_logs failed: %s", e)
    elif run_id and owner and repo_name:
        try:
            result = tool_fetch_run_logs.invoke({
                "owner": owner,
                "repo": repo_name,
                "run_id": str(run_id),
            })
            # Compat : peut retourner str (ancienne version) ou dict (nouvelle)
            if isinstance(result, dict):
                logs = result.get("logs", "")
                detected_stage = result.get("stage_failed", "")
            else:
                logs = str(result)
        except Exception as e:
            logger.debug("fetch_run_logs failed: %s", e)

    # Si stage_failed non fourni explicitement, utiliser celui détecté depuis les logs
    stage_failed = state.get("stage_failed", "") or detected_stage

    # Si on n'a pas encore la conclusion → chercher dans l'API
    conclusion = state.get("run_conclusion", "")
    duration = state.get("run_duration_seconds", 0)

    if not conclusion and owner and repo_name:
        try:
            runs = tool_fetch_workflow_runs.invoke({
                "owner": owner,
                "repo": repo_name,
                "status": "completed",
                "limit": 50,
            })
            for r in runs:
                if str(r.get("id", "")) == str(run_id):
                    conclusion = r.get("conclusion", "")
                    prs = r.get("pull_requests", [])
                    if prs and not state.get("pr_number"):
                        state = dict(state)
                        # prs[0] est un dict {number:N, head:{sha:...}}, pas un int
                        pr_obj = prs[0]
                        state["pr_number"] = pr_obj.get("number") if isinstance(pr_obj, dict) else int(pr_obj)
                    break
        except Exception as e:
            logger.debug("fetch_workflow_runs failed: %s", e)

    # Fallback : si pr_number toujours vide, chercher la PR par head_sha
    if not state.get("pr_number") and (state.get("head_sha") or state.get("pr_branch")) and owner and repo_name:
        try:
            import urllib.request, urllib.parse
            token = (
                __import__("os").environ.get("GITHUB_TOKEN")
                or __import__("os").environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            )
            if token:
                head_sha  = state.get("head_sha", "")
                pr_branch = state.get("pr_branch", "")
                # Chercher PR ouverte correspondant au head_sha ou branch
                params = {"state": "open", "per_page": "10"}
                if pr_branch:
                    params["head"] = f"{owner}:{pr_branch}"
                url = (
                    f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
                    f"?{urllib.parse.urlencode(params)}"
                )
                req = urllib.request.Request(url)
                req.add_header("Authorization", f"token {token}")
                req.add_header("Accept", "application/vnd.github.v3+json")
                with urllib.request.urlopen(req, timeout=10) as r:
                    import json as _json
                    open_prs = _json.loads(r.read().decode())
                for pr in open_prs:
                    if head_sha and pr.get("head", {}).get("sha", "") == head_sha:
                        state = dict(state)
                        state["pr_number"] = pr["number"]
                        logger.info("[FetchRun] PR #%s trouvée via head_sha fallback", pr["number"])
                        break
                    elif not head_sha and open_prs:
                        # Prendre la plus récente
                        state = dict(state)
                        state["pr_number"] = open_prs[0]["number"]
                        logger.info("[FetchRun] PR #%s trouvée via branch fallback", open_prs[0]["number"])
                        break
        except Exception as e:
            logger.debug("PR fallback lookup failed: %s", e)

    return {**state, "logs": logs, "run_conclusion": conclusion or "unknown", "stage_failed": stage_failed}


def node_classify_failure(state: CIState) -> CIState:
    """Classifie le run (success/failure) et détermine failure_type."""
    from langchain_agents.tools.ci_tools import tool_classify_run

    conclusion = state.get("run_conclusion", "")
    stage = state.get("stage_failed", "")
    logs = state.get("logs", "")

    result = tool_classify_run.invoke({
        "conclusion": conclusion,
        "stage_name": stage,
        "logs_snippet": logs[:500],
    })

    return {
        **state,
        "outcome": result.get("outcome", "unknown"),
        "failure_type": result.get("failure_type", "unknown"),
        "severity": result.get("severity", "INFO"),
    }


def node_sonar_mcp_query(state: CIState) -> CIState:
    """Interroge SonarCloud MCP : Quality Gate + métriques + issues critiques."""
    from langchain_agents.tools.ci_tools import (
        tool_sonar_quality_gate,
        tool_sonar_get_metrics,
        tool_sonar_get_issues,
    )

    project_key = state.get("project_key", "")
    if not project_key:
        logger.debug("node_sonar_mcp_query: no project_key, skipping")
        return {**state, "sonar_gate": {}, "sonar_metrics": {}, "sonar_issues": []}

    gate = {}
    metrics = {}
    issues = []

    try:
        gate = tool_sonar_quality_gate.invoke({"project_key": project_key})
    except Exception as e:
        logger.debug("sonar quality gate: %s", e)

    try:
        metrics = tool_sonar_get_metrics.invoke({"project_key": project_key})
    except Exception as e:
        logger.debug("sonar metrics: %s", e)

    try:
        critical_issues = tool_sonar_get_issues.invoke({
            "project_key": project_key,
            "severity": "CRITICAL",
            "issue_type": "VULNERABILITY",
        })
        blocker_issues = tool_sonar_get_issues.invoke({
            "project_key": project_key,
            "severity": "BLOCKER",
            "issue_type": "BUG",
        })
        issues = critical_issues + blocker_issues
    except Exception as e:
        logger.debug("sonar issues: %s", e)

    return {**state, "sonar_gate": gate, "sonar_metrics": metrics, "sonar_issues": issues}


def node_security_intel(state: CIState) -> CIState:
    """
    T3 — Récupère les alertes CodeQL + CVEs Trivy pour enrichir le commentaire PR.
    Inséré entre sonar_mcp_query et search_similar dans le graph.
    Graceful : échoue silencieusement si GitHub Advanced Security n'est pas activé.
    """
    from langchain_agents.tools.ci_tools import (
        tool_fetch_code_scanning_alerts,
        tool_fetch_trivy_artifact,
    )

    owner    = state.get("owner", "")
    repo_raw = state.get("repo", "")
    repo     = repo_raw.split("/")[-1] if "/" in repo_raw else repo_raw
    run_id   = state.get("run_id", "")
    head_sha = state.get("head_sha", "")

    codeql_alerts: list = []
    trivy_report:  dict = {"critical": [], "high": [], "total": 0, "source": "skipped"}

    if owner and repo:
        # CodeQL — GitHub Code Scanning API
        try:
            codeql_alerts = tool_fetch_code_scanning_alerts.invoke({
                "owner":    owner,
                "repo":     repo,
                "ref":      head_sha,
                "tool_name": "CodeQL",
            })
            logger.info("[SecurityIntel] CodeQL: %d alertes", len(codeql_alerts))
        except Exception as e:
            logger.debug("node_security_intel CodeQL: %s", e)

        # Trivy — artifact JSON du run GitHub Actions
        if run_id:
            try:
                trivy_report = tool_fetch_trivy_artifact.invoke({
                    "owner":  owner,
                    "repo":   repo,
                    "run_id": str(run_id),
                })
                logger.info("[SecurityIntel] Trivy: %d CVEs total", trivy_report.get("total", 0))
            except Exception as e:
                logger.debug("node_security_intel Trivy: %s", e)

    return {**state, "codeql_alerts": codeql_alerts, "trivy_report": trivy_report}


def node_search_similar(state: CIState) -> CIState:
    """Cherche des failures similaires dans Redis."""
    from langchain_agents.tools.ci_tools import tool_search_similar_failure

    logs = state.get("logs", "")
    if not logs:
        return {**state, "similar_fixes": [], "confidence": 0.0}

    try:
        similar = tool_search_similar_failure.invoke({"error_text": logs})
        best_confidence = 0.0
        for m in similar:
            if m.get("fix_applied") and m.get("similarity", 0) > best_confidence:
                best_confidence = m["similarity"]

        return {**state, "similar_fixes": similar, "confidence": best_confidence}
    except Exception as e:
        logger.debug("search_similar: %s", e)
        return {**state, "similar_fixes": [], "confidence": 0.0}


def node_analyze_root_cause(state: CIState) -> CIState:
    """LLM Factory : analyse root-cause en croisant logs + SonarCloud."""
    from ci_cd.pipeline_failure_analyzer import PipelineFailureAnalyzer

    analyzer = PipelineFailureAnalyzer()
    result = analyzer.analyze(
        run_id=state.get("run_id", ""),
        repo=state.get("repo", ""),
        owner=state.get("owner", ""),
        logs=state.get("logs", ""),
        stage_failed=state.get("stage_failed", ""),
        failure_type=state.get("failure_type", "unknown"),
        project_key=state.get("project_key", ""),
        pr_number=state.get("pr_number"),
        head_sha=state.get("head_sha", ""),
    )

    return {
        **state,
        "root_cause": result.get("root_cause"),
        "suggested_fix": result.get("suggested_fix"),
        # Mettre à jour les données SonarCloud si elles n'étaient pas encore là
        "sonar_gate": result.get("sonar_gate") or state.get("sonar_gate", {}),
        "sonar_metrics": result.get("sonar_metrics") or state.get("sonar_metrics", {}),
        "sonar_issues": result.get("sonar_issues") or state.get("sonar_issues", []),
        "similar_fixes": result.get("similar_fixes") or state.get("similar_fixes", []),
        "confidence": result.get("confidence", state.get("confidence", 0.0)),
    }


def node_generate_comment(state: CIState) -> CIState:
    """
    Génère le commentaire PR à partir du fix Redis (haute confiance).
    Pas de LLM — utilise le fix stocké directement.
    """
    similar = state.get("similar_fixes", [])
    best = next((m for m in similar if m.get("fix_applied")), None)

    if best:
        root_cause = (
            f"Erreur similaire vue {best['count']}x "
            f"(stage: {best.get('stage', '') or state.get('stage_failed', '')}). "
            f"Correspondance Redis : {int(best['similarity'] * 100)}%."
        )
        suggested_fix = best["fix_applied"]
    else:
        root_cause = "Failure similaire détectée — fix Redis appliqué."
        suggested_fix = ""

    return {**state, "root_cause": root_cause, "suggested_fix": suggested_fix}


def node_post_pr_comment(state: CIState) -> CIState:
    """Poste le commentaire d'analyse sur la PR GitHub."""
    pr_number = state.get("pr_number")
    repo = state.get("repo", "")
    owner = state.get("owner", "")

    if not pr_number or not repo or not owner:
        logger.debug("node_post_pr_comment: no PR to comment on")
        return {**state, "comment_posted": False}

    # Générer le commentaire via PipelineFailureAnalyzer
    try:
        from ci_cd.pipeline_failure_analyzer import PipelineFailureAnalyzer
        analyzer = PipelineFailureAnalyzer()
        comment_body = analyzer.format_pr_comment(
            analysis={
                "root_cause": state.get("root_cause", ""),
                "suggested_fix": state.get("suggested_fix", ""),
                "sonar_gate": state.get("sonar_gate", {}),
                "sonar_metrics": state.get("sonar_metrics", {}),
                "sonar_issues": state.get("sonar_issues", []),
                "similar_fixes": state.get("similar_fixes", []),
                "confidence": state.get("confidence", 0.0),
                "source": "redis_cache" if state.get("confidence", 0) >= 0.8 else "llm_analysis",
            },
            run_id=state.get("run_id", ""),
            repo=repo,
            stage_failed=state.get("stage_failed", ""),
            failure_type=state.get("failure_type", "unknown"),
            codeql_alerts=state.get("codeql_alerts", []),   # T1
            trivy_report=state.get("trivy_report", {}),     # T2
        )
    except Exception as e:
        logger.error("format_pr_comment failed: %s", e)
        comment_body = (
            f"## 🚨 CI Pipeline Failure\n\n"
            f"**Stage:** `{state.get('stage_failed', 'N/A')}`\n\n"
            f"**Root cause:** {state.get('root_cause', 'Non disponible')}\n\n"
            f"**Fix suggéré:** {state.get('suggested_fix', 'Non disponible')}"
        )

    # Poster
    from langchain_agents.tools.ci_tools import tool_post_pr_comment
    repo_name = repo.split("/")[-1] if "/" in repo else repo

    try:
        posted = tool_post_pr_comment.invoke({
            "owner": owner,
            "repo": repo_name,
            "pr_number": pr_number,
            "body": comment_body,
        })
        return {**state, "comment_posted": bool(posted)}
    except Exception as e:
        logger.error("post_pr_comment failed: %s", e)
        return {**state, "comment_posted": False}


def node_index_result(state: CIState) -> CIState:
    """Indexe le run dans Redis (pour mémoire future) + T5 : coverage delta tracking."""
    from langchain_agents.tools.ci_tools import tool_index_ci_run

    try:
        tool_index_ci_run.invoke({
            "run_id": str(state.get("run_id", "")),
            "repo": state.get("repo", ""),
            "logs": state.get("logs", ""),
            "status": state.get("outcome", "unknown"),
            "stage_failed": state.get("stage_failed", ""),
            "duration_seconds": state.get("run_duration_seconds", 0),
        })

        # Si on a un fix suggéré et qu'il vient du LLM → le stocker
        fix = state.get("suggested_fix", "")
        if fix and state.get("confidence", 0) < 0.8:
            from langchain_agents.tools.ci_tools import tool_store_ci_fix
            tool_store_ci_fix.invoke({
                "error_text": state.get("logs", ""),
                "fix": fix,
                "run_id": str(state.get("run_id", "")),
            })

        # T5 — Coverage Delta Tracking
        # Stocker l'évolution du coverage dans un Sorted Set Redis
        coverage = state.get("sonar_metrics", {}).get("coverage")
        if coverage is not None and state.get("outcome") == "success":
            try:
                from services.mcp_redis_service import get_mcp_redis
                redis = get_mcp_redis()
                repo  = state.get("repo", "unknown")
                run_id = str(state.get("run_id", ""))
                ts    = time.time()
                key   = f"ci:coverage:{repo}"
                # Score = timestamp, membre = "coverage:run_id"
                redis.zadd(key, {f"{coverage}:{run_id}": ts})
                redis.zremrangebyrank(key, 0, -51)   # Garder les 50 derniers
                redis.expire(key, 90 * 86400)        # TTL 90 jours

                # Détecter une régression (≥5% de baisse vs moyenne des 10 derniers)
                history = redis.zrange(key, -11, -1)  # 10 précédents
                if history and len(history) >= 3:
                    past_vals = []
                    for entry in history[:-1]:  # Exclure l'entrée qu'on vient d'ajouter
                        try:
                            past_vals.append(float(str(entry).split(":")[0]))
                        except Exception:
                            pass
                    if past_vals:
                        avg = sum(past_vals) / len(past_vals)
                        drop = avg - float(coverage)
                        if drop >= 5.0:
                            logger.warning(
                                "[T5-Coverage] REGRESSION detected on %s: %.1f%% -> %.1f%% (delta: -%.1f%%)",
                                repo, avg, float(coverage), drop
                            )
            except Exception as cov_err:
                logger.debug("Coverage delta tracking: %s", cov_err)

        return {**state, "indexed": True}
    except Exception as e:
        logger.error("node_index_result: %s", e)
        return {**state, "indexed": False}


def node_notify(state: CIState) -> CIState:
    """Notification console (INFO/WARN/URGENT)."""
    from langchain_agents.agents.lc_ci_notifier import LCCINotifier

    notifier = LCCINotifier()
    level = notifier.notify(state)
    return {**state, "notification_level": level}


def node_retry_check(state: CIState) -> CIState:
    """
    T6 — Détecte les flaky tests et re-déclenche le run automatiquement.
    S'exécute après classify_failure, avant sonar_mcp_query.

    Un test est 'flaky' si dans les 8 derniers runs du même stage :
      - Au moins 2 ont réussi ET au moins 2 ont échoué (alternance)
      - Le failure_type est 'test'
    """
    # Seulement pour les failures de type test, et pas si déjà relancé
    if state.get("failure_type") != "test" or state.get("auto_retried"):
        return state

    owner     = state.get("owner", "")
    repo_full = state.get("repo", "")
    stage     = state.get("stage_failed", "")
    run_id    = state.get("run_id", "")

    if not (owner and repo_full and run_id):
        return state

    try:
        from ci_cd.ci_logs_indexer import CILogsIndexer
        indexer = CILogsIndexer()
        recent = indexer.get_recent_runs(repo_full, limit=10)
        if not recent or len(recent) < 4:
            return state

        # Filtrer les runs du même stage défaillant
        pattern = [
            r["status"] for r in recent[:8]
            if not stage or r.get("stage_failed") == stage
        ]

        # Flaky : ≥2 successes ET ≥2 failures sur les derniers runs
        successes = pattern.count("success")
        failures  = pattern.count("failure")
        is_flaky  = successes >= 2 and failures >= 2

        if is_flaky:
            logger.warning(
                "[T6-Retry] Flaky test sur %s stage=%s (succ=%d fail=%d) — auto re-run",
                repo_full, stage or "?", successes, failures
            )
            # Déclencher le re-run via GitHub REST API
            import os, urllib.request
            token = (
                os.environ.get("GITHUB_TOKEN")
                or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            )
            if token:
                repo_name = repo_full.split("/")[-1] if "/" in repo_full else repo_full
                url = (
                    f"https://api.github.com/repos/{owner}/{repo_name}"
                    f"/actions/runs/{run_id}/rerun-failed-jobs"
                )
                req = urllib.request.Request(url, data=b"{}", method="POST")
                req.add_header("Authorization", f"token {token}")
                req.add_header("Accept", "application/vnd.github.v3+json")
                req.add_header("Content-Type", "application/json")
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        logger.info("[T6-Retry] Re-run déclenché HTTP %s", r.status)
                    return {**state, "auto_retried": True, "outcome": "retrying"}
                except Exception as re_err:
                    logger.warning("[T6-Retry] Re-run API échoué: %s", re_err)
    except Exception as e:
        logger.debug("node_retry_check: %s", e)

    return state


# ── Conditional edges ──────────────────────────────────────────────────────────

def route_after_classify(state: CIState) -> str:
    """
    Après classification :
      success/cancelled/retrying → index_result (fast path, 0 LLM)
      failure                   → retry_check (T6) puis sonar si pas flaky
    """
    outcome = state.get("outcome", "failure")
    if outcome in ("success", "cancelled", "retrying"):
        return "index_result"
    return "retry_check"


def route_after_search(state: CIState) -> str:
    """
    Après recherche de similarité Redis :
      confidence > 0.8 ET fix_applied → generate_comment (sans LLM)
      sinon                           → analyze_root_cause (LLM)
    """
    confidence = state.get("confidence", 0.0)
    similar = state.get("similar_fixes", [])
    has_fix = any(m.get("fix_applied") for m in similar)

    if confidence >= 0.8 and has_fix:
        return "generate_comment"
    return "analyze_root_cause"


def route_after_comment(state: CIState) -> str:
    """Après generate_comment → post si PR disponible, sinon index directement."""
    if state.get("pr_number"):
        return "post_pr_comment"
    return "index_result"


def route_after_root_cause(state: CIState) -> str:
    """Après analyze_root_cause → post si PR disponible, sinon index."""
    if state.get("pr_number") and state.get("root_cause"):
        return "post_pr_comment"
    return "index_result"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_ci_graph() -> StateGraph:
    """
    Construit et compile le CIGraph LangGraph.

    Returns:
        Compiled StateGraph prêt à être invoqué avec invoke_ci_run()
    """
    g = StateGraph(CIState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    g.add_node("fetch_run",          node_fetch_run)
    g.add_node("classify_failure",   node_classify_failure)
    g.add_node("retry_check",        node_retry_check)       
    g.add_node("sonar_mcp_query",    node_sonar_mcp_query)
    g.add_node("security_intel",     node_security_intel)   
    g.add_node("search_similar",     node_search_similar)
    g.add_node("generate_comment",   node_generate_comment)
    g.add_node("analyze_root_cause", node_analyze_root_cause)
    g.add_node("post_pr_comment",    node_post_pr_comment)
    g.add_node("index_result",       node_index_result)
    g.add_node("notify",             node_notify)

    # ── Entry ──────────────────────────────────────────────────────────────
    g.set_entry_point("fetch_run")

    # ── Edges ──────────────────────────────────────────────────────────────
    g.add_edge("fetch_run", "classify_failure")

    g.add_conditional_edges(
        "classify_failure",
        route_after_classify,
        {
            "index_result": "index_result",
            "retry_check":  "retry_check",   # T6 : flaky test check
        },
    )

    g.add_edge("retry_check", "sonar_mcp_query")       # T6 : retry → sonar
    g.add_edge("sonar_mcp_query", "security_intel")    # T3 : sonar → CodeQL+Trivy
    g.add_edge("security_intel",  "search_similar")    # T3 : security → search

    g.add_conditional_edges(
        "search_similar",
        route_after_search,
        {
            "generate_comment": "generate_comment",
            "analyze_root_cause": "analyze_root_cause",
        },
    )

    g.add_conditional_edges(
        "generate_comment",
        route_after_comment,
        {
            "post_pr_comment": "post_pr_comment",
            "index_result": "index_result",
        },
    )

    g.add_conditional_edges(
        "analyze_root_cause",
        route_after_root_cause,
        {
            "post_pr_comment": "post_pr_comment",
            "index_result": "index_result",
        },
    )

    g.add_edge("post_pr_comment", "index_result")
    g.add_edge("index_result",    "notify")
    g.add_edge("notify",          END)

    return g.compile()


# ── Public invoke function ──────────────────────────────────────────────────────

_ci_graph = None


def invoke_ci_run(
    run_id: str,
    repo: str,
    owner: str,
    project_key: str = "",
    pr_number: int = None,
    head_sha: str = "",
    pr_branch: str = "",
    stage_failed: str = "",
    job_id: str = "",
    run_conclusion: str = "",
    run_duration_seconds: int = 0,
) -> Dict[str, Any]:
    """
    Invoque le CIGraph pour analyser un run GitHub Actions.

    Args:
        run_id: GitHub Actions workflow run ID
        repo: "{owner}/{repo}"
        owner: Repository owner
        project_key: SonarCloud project key (ex: "chmaryem_myapp")
        pr_number: PR number associée (optionnel)
        head_sha: SHA du commit HEAD
        pr_branch: Branche source de la PR (fallback pour détecter pr_number)
        stage_failed: Nom du stage qui a échoué (optionnel)
        job_id: ID numérique du job GitHub Actions à analyser spécifiquement
                (optionnel — analyse ciblée sur une issue précise du dashboard,
                au lieu de deviner le stage depuis tout le run)
        run_conclusion: "success" | "failure" | "cancelled" (optionnel)
        run_duration_seconds: Durée du run

    Returns:
        Final CIState dict
    """
    global _ci_graph
    if _ci_graph is None:
        _ci_graph = build_ci_graph()

    initial_state: CIState = {
        "run_id": str(run_id),
        "repo": repo,
        "owner": owner,
        "project_key": project_key or "",
        "pr_number": pr_number,
        "head_sha": head_sha or "",
        "pr_branch": pr_branch or "",   # fallback lookup
        "stage_failed": stage_failed or "",
        "job_id": job_id or "",
        "run_conclusion": run_conclusion or "",
        "run_duration_seconds": run_duration_seconds,
        # Defaults
        "logs": "",
        "outcome": "",
        "failure_type": "unknown",
        "severity": "INFO",
        "sonar_gate": {},
        "sonar_metrics": {},
        "sonar_issues": [],
        "similar_fixes": [],
        "confidence": 0.0,
        "root_cause": None,
        "suggested_fix": None,
        "comment_posted": False,
        "indexed": False,
        "notification_level": "INFO",
        "codeql_alerts": [],          # T1 — CodeQL Security Intel
        "trivy_report": {},            # T2 — Trivy CVEs
        "auto_retried": False,         # T6 — Flaky test retry
    }

    try:
        start = time.time()
        result = _ci_graph.invoke(initial_state)
        elapsed = round(time.time() - start, 2)
        logger.info(
            "[CIGraph] run=%s outcome=%s notification=%s (%.1fs)",
            run_id,
            result.get("outcome", "?"),
            result.get("notification_level", "?"),
            elapsed,
        )
        return result
    except Exception as e:
        logger.error("[CIGraph] invoke failed for run=%s: %s", run_id, e)
        return {**initial_state, "root_cause": f"CIGraph error: {e}"}
