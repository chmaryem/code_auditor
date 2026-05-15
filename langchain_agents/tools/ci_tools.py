"""
ci_tools.py — LangChain @tool wrappers pour le CIGraph.

4 groupes de tools :
  A. SonarCloud MCP — Quality Gate, métriques, issues
  B. GitHub Actions  — Fetch logs, post commentaire PR
  C. Redis CI Memory — Indexer, rechercher des failures similaires
  D. Analyse        — Classifier un run (success/failure)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── A. SonarCloud MCP Tools ───────────────────────────────────────────────────

@tool
def tool_sonar_quality_gate(project_key: str) -> Dict[str, Any]:
    """
    Vérifie le Quality Gate SonarCloud d'un projet via MCP.

    Args:
        project_key: Clé SonarCloud (ex: "chmaryem_myapp")

    Returns:
        {"status": "OK"|"ERROR"|"WARN", "conditions": [...], "source": "mcp"|"rest"}
    """
    try:
        from services.mcp_sonarqube_service import get_mcp_sonarqube
        sonar = get_mcp_sonarqube()
        result = sonar.get_quality_gate_status(project_key)
        result["source"] = "mcp_sonar"
        return result
    except Exception as e:
        logger.error("tool_sonar_quality_gate failed: %s", e)
        return {"status": "NONE", "conditions": [], "error": str(e)}


@tool
def tool_sonar_get_metrics(project_key: str) -> Dict[str, Any]:
    """
    Récupère les métriques clés d'un projet SonarCloud via MCP.

    Args:
        project_key: Clé SonarCloud

    Returns:
        Dict metric_key → value (coverage, bugs, vulnerabilities, etc.)
    """
    try:
        from services.mcp_sonarqube_service import get_mcp_sonarqube
        sonar = get_mcp_sonarqube()
        return sonar.get_project_metrics(project_key)
    except Exception as e:
        logger.error("tool_sonar_get_metrics failed: %s", e)
        return {}


@tool
def tool_sonar_get_issues(
    project_key: str,
    severity: str = "CRITICAL",
    issue_type: str = "VULNERABILITY",
) -> List[Dict[str, Any]]:
    """
    Récupère les issues SonarCloud filtrées par sévérité et type.

    Args:
        project_key: Clé SonarCloud
        severity: BLOCKER | CRITICAL | MAJOR | MINOR | INFO
        issue_type: BUG | VULNERABILITY | CODE_SMELL | SECURITY_HOTSPOT

    Returns:
        Liste d'issues {message, severity, rule, component, line}
    """
    try:
        from services.mcp_sonarqube_service import get_mcp_sonarqube
        sonar = get_mcp_sonarqube()
        return sonar.get_issues(project_key, severity=severity, issue_type=issue_type)
    except Exception as e:
        logger.error("tool_sonar_get_issues failed: %s", e)
        return []


@tool
def tool_sonar_pr_issues(project_key: str, pr_number: str) -> List[Dict[str, Any]]:
    """
    Récupère les nouvelles issues introduites par une PR spécifique.

    Args:
        project_key: Clé SonarCloud
        pr_number: Numéro de la PR (en string, ex: "42")

    Returns:
        Liste d'issues nouvelles sur cette PR
    """
    try:
        from services.mcp_sonarqube_service import get_mcp_sonarqube
        sonar = get_mcp_sonarqube()
        return sonar.get_new_issues_on_pr(project_key, pr_number)
    except Exception as e:
        logger.error("tool_sonar_pr_issues failed: %s", e)
        return []


# ── B. GitHub Actions Tools ───────────────────────────────────────────────────

@tool
def tool_fetch_run_logs(owner: str, repo: str, run_id: str) -> Dict[str, Any]:
    """
    Récupère les logs d'un run GitHub Actions via la REST API.

    Args:
        owner: Propriétaire du repo (ex: "chmaryem")
        repo: Nom du repo (ex: "myapp")
        run_id: ID du workflow run

    Returns:
        {"logs": str, "stage_failed": str, "total_chars": int}
        stage_failed est détecté automatiquement depuis les noms de fichiers ZIP.
        Format GitHub Actions : "{job-name}/{N}_{step-name}.txt"
    """
    def _parse_zip(zf) -> Dict[str, Any]:
        """Extrait logs + détecte le job qui a échoué."""
        _FAIL_KEYWORDS = (
            "error:", "error :", "failed",
            "exit code 1", "process completed with exit code",
            "build failure", "compilation error", "assert",
            "tests run:", "failures:", "quality gate has failed",
        )
        logs_parts = []
        stage_failed = ""
        # Trier pour avoir les fichiers du job le plus récent en premier
        names = sorted(zf.namelist())
        for name in names:
            try:
                content = zf.open(name).read().decode("utf-8", errors="replace")
                logs_parts.append(f"=== {name} ===\n{content}")
                # Détecter le job en échec (format: "job-name/N_step-name.txt")
                if not stage_failed and "/" in name:
                    job_name = name.split("/")[0]
                    content_lower = content.lower()
                    if any(k in content_lower for k in _FAIL_KEYWORDS):
                        stage_failed = job_name
            except Exception:
                pass
        logs_text = "\n".join(logs_parts)
        return {
            "logs": logs_text[:10000],
            "stage_failed": stage_failed,
            "total_chars": len(logs_text),
        }

    try:
        import io
        import zipfile
        import urllib.request
        import urllib.error

        token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        )
        if not token:
            return {"logs": "[ERROR] GITHUB_TOKEN manquant", "stage_failed": "", "total_chars": 0}

        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return _parse_zip(zipfile.ZipFile(io.BytesIO(resp.read())))
        except urllib.error.HTTPError as e:
            if e.code == 302:
                location = e.headers.get("Location", "")
                if location:
                    with urllib.request.urlopen(location, timeout=30) as resp2:
                        return _parse_zip(zipfile.ZipFile(io.BytesIO(resp2.read())))
            return {"logs": f"[HTTP {e.code}] Impossible de récupérer les logs", "stage_failed": "", "total_chars": 0}

    except Exception as e:
        logger.error("tool_fetch_run_logs %s/%s#%s: %s", owner, repo, run_id, e)
        return {"logs": f"[ERROR] {e}", "stage_failed": "", "total_chars": 0}


@tool
def tool_fetch_workflow_runs(
    owner: str,
    repo: str,
    status: str = "completed",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Liste les derniers workflow runs d'un repo GitHub Actions (pour le polling).

    Args:
        owner: Propriétaire du repo
        repo: Nom du repo
        status: completed | in_progress | queued
        limit: Nombre max de runs à retourner

    Returns:
        Liste de runs {id, status, conclusion, head_commit, created_at, html_url}
    """
    try:
        import urllib.request
        import urllib.parse

        token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        )
        if not token:
            return []

        params = urllib.parse.urlencode({"status": status, "per_page": str(limit)})
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs?{params}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            runs = data.get("workflow_runs", [])
            return [
                {
                    "id": r["id"],
                    "status": r["status"],
                    "conclusion": r.get("conclusion"),
                    "name": r.get("name", ""),
                    "head_branch": r.get("head_branch", ""),
                    "head_sha": r.get("head_sha", ""),
                    "pull_requests": [
                        pr["number"] for pr in r.get("pull_requests", [])
                    ],
                    "created_at": r.get("created_at", ""),
                    "updated_at": r.get("updated_at", ""),
                    "html_url": r.get("html_url", ""),
                    "run_attempt": r.get("run_attempt", 1),
                }
                for r in runs
            ]
    except Exception as e:
        logger.error("tool_fetch_workflow_runs %s/%s: %s", owner, repo, e)
        return []


@tool
def tool_post_pr_comment(
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
) -> bool:
    """
    Poste un commentaire sur une PR GitHub via MCP (fallback REST).

    Args:
        owner: Propriétaire du repo
        repo: Nom du repo
        pr_number: Numéro de la PR
        body: Corps du commentaire (markdown)

    Returns:
        True si posté avec succès
    """
    # Essai via MCPGitHubService (non-async wrapper)
    try:
        from services.mcp_github_service import MCPGitHubService
        from services.mcp_github_service import _loop_manager as gh_loop

        async def _post():
            svc = MCPGitHubService()
            await svc.connect()
            try:
                result = await svc.post_pr_comment(owner, repo, pr_number, body)
                return bool(result)
            finally:
                await svc.disconnect()

        return gh_loop.run(_post(), timeout=30)
    except Exception:
        pass

    # Fallback REST direct
    try:
        import urllib.request
        token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        )
        if not token:
            return False

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        data = json.dumps({"body": body}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"token {token}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/vnd.github.v3+json")

        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 201)
    except Exception as e:
        logger.error("tool_post_pr_comment %s/%s#%s: %s", owner, repo, pr_number, e)
        return False


# ── C. Redis CI Memory Tools ──────────────────────────────────────────────────

def _error_hash(error_text: str) -> str:
    """Hash court d'un message d'erreur pour la clé Redis."""
    return hashlib.md5(error_text.strip().lower()[:500].encode()).hexdigest()[:12]


@tool
def tool_index_ci_run(
    run_id: str,
    repo: str,
    logs: str,
    status: str,
    stage_failed: str = "",
    duration_seconds: int = 0,
) -> bool:
    """
    Indexe un CI run dans Redis pour la recherche future de failures similaires.

    Structures Redis :
      ci:runs:{repo}         → Sorted Set (score = timestamp)
      ci:run:{run_id}        → Hash (status, stage, duration, logs_summary)
      ci:errors:{error_hash} → Hash (count, last_seen, repo, stage)

    Args:
        run_id: ID du workflow run GitHub
        repo: "{owner}/{repo}"
        logs: Logs du run (utilisés pour extraire le message d'erreur)
        status: "success" | "failure" | "cancelled"
        stage_failed: Nom du stage qui a échoué (optionnel)
        duration_seconds: Durée du run en secondes

    Returns:
        True si indexé avec succès
    """
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()

        ts = int(time.time())
        logs_summary = logs[:500] if logs else ""

        # 1. Sorted Set : historique des runs par repo
        redis.zadd(f"ci:runs:{repo}", float(ts), run_id)

        # 2. Hash : détails du run
        redis.hset_dict(f"ci:run:{run_id}", {
            "run_id": run_id,
            "repo": repo,
            "status": status,
            "stage_failed": stage_failed or "",
            "duration": str(duration_seconds),
            "logs_summary": logs_summary,
            "timestamp": str(ts),
        }, expire_seconds=86400 * 30)  # 30 jours

        # 3. Si failure → indexer l'erreur pour similarité
        if status == "failure" and logs:
            err_hash = _error_hash(logs_summary)
            key = f"ci:errors:{err_hash}"

            # Incrémenter le compteur
            existing = redis.hgetall(key) or {}
            count = int(existing.get("count", 0)) + 1

            redis.hset_dict(key, {
                "count": str(count),
                "last_seen": str(ts),
                "repo": repo,
                "stage": stage_failed or "",
                "sample": logs_summary,
                "fix_applied": existing.get("fix_applied", ""),
            }, expire_seconds=86400 * 90)  # 90 jours

        return True
    except Exception as e:
        logger.error("tool_index_ci_run %s: %s", run_id, e)
        return False


@tool
def tool_search_similar_failure(error_text: str) -> List[Dict[str, Any]]:
    """
    Cherche dans Redis si une erreur similaire a déjà été vue et corrigée.

    Args:
        error_text: Texte de l'erreur (logs, message d'échec)

    Returns:
        Liste de matches {error_hash, count, last_seen, stage, fix_applied, similarity}
    """
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()

        err_hash = _error_hash(error_text)
        key = f"ci:errors:{err_hash}"
        data = redis.hgetall(key)

        if data and int(data.get("count", 0)) > 0:
            return [{
                "error_hash": err_hash,
                "count": int(data.get("count", 0)),
                "last_seen": data.get("last_seen", ""),
                "stage": data.get("stage", ""),
                "fix_applied": data.get("fix_applied", ""),
                "sample": data.get("sample", ""),
                "similarity": 1.0,  # Hash exact = similarité parfaite
            }]

        # Pas de match exact — chercher les patterns voisins (fuzzy)
        results = []
        try:
            # Scanner les clés ci:errors:* (limité à 100)
            all_keys = redis.scan_keys("ci:errors:*") or []
            for k in all_keys[:100]:
                d = redis.hgetall(k) or {}
                sample = d.get("sample", "")
                if not sample:
                    continue
                # Similarité simple : mots communs
                words_query = set(error_text.lower().split())
                words_sample = set(sample.lower().split())
                if not words_query or not words_sample:
                    continue
                sim = len(words_query & words_sample) / max(len(words_query), len(words_sample))
                if sim > 0.4:
                    results.append({
                        "error_hash": k.replace("ci:errors:", ""),
                        "count": int(d.get("count", 0)),
                        "last_seen": d.get("last_seen", ""),
                        "stage": d.get("stage", ""),
                        "fix_applied": d.get("fix_applied", ""),
                        "sample": sample[:200],
                        "similarity": round(sim, 2),
                    })
        except Exception:
            pass

        results.sort(key=lambda x: (-x["similarity"], -x["count"]))
        return results[:5]

    except Exception as e:
        logger.error("tool_search_similar_failure: %s", e)
        return []


@tool
def tool_store_ci_fix(error_text: str, fix: str, run_id: str = "") -> bool:
    """
    Stocke un fix qui a fonctionné pour un type d'erreur dans Redis.

    Args:
        error_text: Texte de l'erreur (pour calculer le hash)
        fix: Description du fix appliqué
        run_id: ID du run où le fix a été appliqué

    Returns:
        True si stocké
    """
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()

        err_hash = _error_hash(error_text)

        # Ajouter à la liste des fixes pour ce hash
        redis.json_set(
            f"ci:fixes:{err_hash}",
            "$",
            json.dumps([{"fix": fix, "run_id": run_id, "timestamp": int(time.time())}])
        )

        # Marquer l'erreur comme "fix appliqué"
        redis.hset(f"ci:errors:{err_hash}", "fix_applied", fix[:200])
        return True
    except Exception as e:
        logger.error("tool_store_ci_fix: %s", e)
        return False


# ── D. Run Classification Tool ────────────────────────────────────────────────

@tool
def tool_classify_run(
    conclusion: str,
    stage_name: str = "",
    logs_snippet: str = "",
) -> Dict[str, Any]:
    """
    Classifie un run GitHub Actions pour router le CIGraph.

    Args:
        conclusion: "success" | "failure" | "cancelled" | "skipped" | None
        stage_name: Nom du job/stage qui a échoué
        logs_snippet: Extrait des logs (pour détecter le type d'erreur)

    Returns:
        {
            "outcome": "success" | "failure" | "cancelled",
            "failure_type": "build" | "test" | "security" | "docker" | "unknown",
            "severity": "URGENT" | "WARN" | "INFO"
        }
    """
    outcome = "success" if conclusion == "success" else (
        "cancelled" if conclusion in ("cancelled", "skipped") else "failure"
    )

    failure_type = "unknown"
    severity = "INFO"

    if outcome == "failure":
        severity = "URGENT"
        logs_lower = logs_snippet.lower()
        stage_lower = stage_name.lower()

        if any(k in stage_lower or k in logs_lower for k in ["build", "compile", "mvn", "gradle", "npm run build"]):
            failure_type = "build"
        elif any(k in stage_lower or k in logs_lower for k in ["test", "pytest", "jest", "junit", "coverage"]):
            failure_type = "test"
        elif any(k in stage_lower or k in logs_lower for k in ["sonar", "codeql", "trivy", "security", "cve", "vulnerability"]):
            failure_type = "security"
        elif any(k in stage_lower or k in logs_lower for k in ["docker", "container", "push", "build docker"]):
            failure_type = "docker"
        else:
            failure_type = "unknown"
    elif outcome == "cancelled":
        severity = "WARN"

    return {
        "outcome": outcome,
        "failure_type": failure_type,
        "severity": severity,
    }


# ── E. Security Intelligence Tools (T1 + T2) ─────────────────────────────────

@tool
def tool_fetch_code_scanning_alerts(
    owner: str,
    repo: str,
    ref: str = "",
    tool_name: str = "CodeQL",
    severity_filter: str = "critical,high,error",
) -> List[Dict[str, Any]]:
    """
    Récupère les alertes CodeQL via GitHub Code Scanning API (T1).
    Retourne les vulnérabilités CRITICAL/HIGH pour enrichir les commentaires PR.

    Args:
        owner: Propriétaire du repo
        repo: Nom du repo
        ref: Branche ou SHA du commit (optionnel)
        tool_name: "CodeQL" | "Trivy" | "" (tous)
        severity_filter: Sévérités à inclure (csv, ex: "critical,high,error")

    Returns:
        [{rule_id, severity, description, location_path, location_line, html_url}]
    """
    import urllib.request, urllib.parse, urllib.error

    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    )
    if not token:
        return []

    severities = {s.strip().lower() for s in severity_filter.split(",")}
    alerts = []

    try:
        params: Dict[str, str] = {"state": "open", "per_page": "20"}
        if ref:
            params["ref"] = ref
        if tool_name:
            params["tool_name"] = tool_name

        url = (
            f"https://api.github.com/repos/{owner}/{repo}/code-scanning/alerts"
            f"?{urllib.parse.urlencode(params)}"
        )
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        for alert in data:
            rule = alert.get("rule", {})
            sev  = (rule.get("security_severity_level") or rule.get("severity") or "").lower()
            if sev not in severities:
                continue
            loc = (alert.get("most_recent_instance") or {}).get("location", {})
            alerts.append({
                "rule_id":       rule.get("id", ""),
                "severity":      sev,
                "description":   rule.get("description") or rule.get("full_description", ""),
                "location_path": loc.get("path", ""),
                "location_line": loc.get("start_line", 0),
                "state":         alert.get("state", "open"),
                "html_url":      alert.get("html_url", ""),
            })

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("Code Scanning API: 404 — GitHub Advanced Security inactive ou pas d'alertes")
        elif e.code == 403:
            logger.debug(
                "tool_fetch_code_scanning_alerts HTTP 403 — token sans permission 'security_events:read'. "
                "Activer GitHub Advanced Security sur ce repo."
            )
        else:
            logger.error("tool_fetch_code_scanning_alerts HTTP %s: %s", e.code, e)
    except Exception as e:
        logger.error("tool_fetch_code_scanning_alerts: %s", e)

    return alerts


@tool
def tool_fetch_trivy_artifact(
    owner: str,
    repo: str,
    run_id: str,
    artifact_name: str = "trivy-security-report",
) -> Dict[str, Any]:
    """
    Télécharge l'artifact Trivy JSON depuis GitHub Actions et extrait les CVEs (T2).
    L'artifact 'trivy-security-report' est uploadé par le job docker-trivy.

    Args:
        owner: Propriétaire du repo
        repo: Nom du repo
        run_id: ID du workflow run
        artifact_name: Nom de l'artifact (défaut: 'trivy-security-report')

    Returns:
        {critical: [{pkg, id, title, fixed_version, target}],
         high: [...], total: N, source: str}
    """
    import io, zipfile, urllib.request, urllib.error

    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    )
    if not token:
        return {"critical": [], "high": [], "total": 0, "source": "no_token"}

    def _gh_get(url: str) -> Any:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    try:
        # 1. Lister les artifacts du run
        artifacts_data = _gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
        )
        artifact = next(
            (a for a in artifacts_data.get("artifacts", [])
             if artifact_name.lower() in a["name"].lower()),
            None,
        )
        if not artifact:
            logger.debug("Artifact '%s' introuvable pour run=%s", artifact_name, run_id)
            return {"critical": [], "high": [], "total": 0, "source": "artifact_not_found"}

        # 2. Télécharger le ZIP de l'artifact (suit la redirection)
        dl_url = artifact["archive_download_url"]
        req = urllib.request.Request(dl_url)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                zip_bytes = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 302:
                location = e.headers.get("Location", "")
                with urllib.request.urlopen(location, timeout=30) as resp2:
                    zip_bytes = resp2.read()
            else:
                raise

        # 3. Extraire le JSON Trivy
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        trivy_json = None
        for name in zf.namelist():
            if name.endswith(".json"):
                trivy_json = json.loads(
                    zf.open(name).read().decode("utf-8", errors="replace")
                )
                break

        if not trivy_json:
            return {"critical": [], "high": [], "total": 0, "source": "json_not_found"}

        # 4. Parser le format Trivy JSON (Results[].Vulnerabilities[])
        critical, high = [], []
        for result in trivy_json.get("Results", []):
            for vuln in result.get("Vulnerabilities") or []:
                sev = (vuln.get("Severity") or "").upper()
                entry = {
                    "pkg":           vuln.get("PkgName", ""),
                    "id":            vuln.get("VulnerabilityID", ""),
                    "title":         (vuln.get("Title") or vuln.get("Description", ""))[:120],
                    "fixed_version": vuln.get("FixedVersion", "N/A"),
                    "installed":     vuln.get("InstalledVersion", ""),
                    "target":        result.get("Target", ""),
                }
                if sev == "CRITICAL":
                    critical.append(entry)
                elif sev == "HIGH":
                    high.append(entry)

        return {
            "critical": critical[:10],   # top 10
            "high":     high[:10],
            "total":    len(critical) + len(high),
            "source":   "trivy_artifact",
        }

    except Exception as e:
        # 403 = token sans permission actions:read → log debug, pas erreur
        err_str = str(e)
        if "403" in err_str or "401" in err_str:
            logger.debug(
                "tool_fetch_trivy_artifact %s/%s#%s: auth 403/401 — "
                "token doit avoir la permission 'actions:read'",
                owner, repo, run_id
            )
        else:
            logger.error("tool_fetch_trivy_artifact %s/%s#%s: %s", owner, repo, run_id, e)
        return {"critical": [], "high": [], "total": 0, "source": f"error: {e}"}
