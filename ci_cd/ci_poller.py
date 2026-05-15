"""
ci_poller.py — T7+T8 : Polling loop GitHub Actions → CIGraph.

Surveille les runs GitHub Actions d'un repo en temps réel et déclenche
automatiquement le CIGraph (analyse IA) sur chaque run terminé.

Features :
  T7 : Connecte les jobs publish/deploy au CIGraph (notifie sur échec)
  T8 : Boucle polling configurable (interval, branches, limites)

Usage CLI :
    python -m ci_cd.ci_poller --repo owner/repo --interval 60
    python -m ci_cd.ci_poller --repo owner/repo --branch main --watch-jobs publish deploy

Usage code :
    from ci_cd.ci_poller import CIPoller
    poller = CIPoller(repo="owner/repo", interval=60)
    poller.run()  # Blocking loop
"""
from __future__ import annotations

import logging
import os
import time
import json
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── T7 : Jobs à surveiller pour le CIGraph ────────────────────────────────────
# Inclut les stages "publish" et "deploy" en plus des jobs classiques
WATCHED_FAILURE_JOBS: Set[str] = {
    "build-test",
    "sonar-scan",
    "dep-scan",
    "codeql-scan",
    "docker-trivy",
    "publish",     # T7 : connecté au CIGraph
    "deploy",      # T7 : connecté au CIGraph
}


class CIPoller:
    """
    Polling loop GitHub Actions → CIGraph.

    Interroge l'API GitHub Actions toutes les `interval` secondes.
    Pour chaque run terminé non encore traité :
      - Détecte le job qui a échoué (depuis les jobs du run)
      - Invoque invoke_ci_run() du CIGraph
      - Marque le run comme traité (Set en mémoire + Redis)

    T7 : Les échecs des jobs 'publish' et 'deploy' déclenchent le CIGraph
         avec failure_type='docker' pour le publish, 'deploy' pour le deploy.
    """

    def __init__(
        self,
        repo: str,
        interval: int = 60,
        branch: str = "",
        project_key: str = "",
        watch_jobs: Optional[List[str]] = None,
        max_runs_per_poll: int = 5,
    ):
        """
        Args:
            repo: "{owner}/{repo}"
            interval: Secondes entre chaque poll (défaut : 60s)
            branch: Surveiller uniquement cette branche (vide = toutes)
            project_key: Clé SonarCloud (ex: "chmaryem_myapp")
            watch_jobs: Jobs spécifiques à surveiller (défaut : WATCHED_FAILURE_JOBS)
            max_runs_per_poll: Nombre max de runs à analyser par cycle
        """
        parts = repo.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"repo doit être 'owner/repo', reçu: {repo!r}")

        # Charger le .env si pas encore chargé (GITHUB_TOKEN, etc.)
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass

        self.owner, self.repo_name = parts
        self.repo       = repo
        self.interval   = interval
        self.branch     = branch
        self.project_key = project_key
        self.watch_jobs = set(watch_jobs or WATCHED_FAILURE_JOBS)
        self.max_runs   = max_runs_per_poll

        self._token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        )
        self._processed: Set[str] = set()   # run_ids déjà traités (mémoire locale)
        self._load_processed_from_redis()

    # ── Redis persistence des run_ids traités ─────────────────────────────────

    def _load_processed_from_redis(self):
        """Charge les run_ids déjà traités depuis Redis (évite les doublons au restart)."""
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            key = f"ci:poller:processed:{self.repo}"
            members = redis.smembers(key)
            if members:
                self._processed = {str(m) for m in members}
                logger.debug("[Poller] %d runs déjà traités chargés depuis Redis", len(self._processed))
        except Exception as e:
            logger.debug("[Poller] Redis load failed: %s", e)

    def _mark_processed(self, run_id: str):
        """Marque un run comme traité (local + Redis)."""
        self._processed.add(str(run_id))
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            key = f"ci:poller:processed:{self.repo}"
            redis.sadd(key, str(run_id))
            redis.expire(key, 7 * 86400)    # TTL 7 jours
        except Exception:
            pass

    # ── GitHub API helpers ────────────────────────────────────────────────────

    def _gh_get(self, url: str) -> Any:
        """HTTP GET vers GitHub API."""
        if not self._token:
            raise RuntimeError("GITHUB_TOKEN manquant")
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {self._token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def _fetch_completed_runs(self) -> List[Dict[str, Any]]:
        """Récupère les derniers runs terminés du repo."""
        params = {"status": "completed", "per_page": str(self.max_runs * 2)}
        if self.branch:
            params["branch"] = self.branch
        url = (
            f"https://api.github.com/repos/{self.owner}/{self.repo_name}/actions/runs"
            f"?{urllib.parse.urlencode(params)}"
        )
        try:
            data = self._gh_get(url)
            return data.get("workflow_runs", [])
        except Exception as e:
            logger.error("[Poller] fetch_completed_runs: %s", e)
            return []

    def _fetch_failed_job(self, run_id: str) -> str:
        """
        Récupère le nom du job qui a échoué dans un run via /actions/runs/{id}/jobs.
        T7 : Inclut 'publish' et 'deploy' dans la détection.
        """
        url = (
            f"https://api.github.com/repos/{self.owner}/{self.repo_name}"
            f"/actions/runs/{run_id}/jobs"
        )
        try:
            data = self._gh_get(url)
            jobs = data.get("jobs", [])
            for job in jobs:
                if job.get("conclusion") in ("failure", "timed_out"):
                    return job.get("name", "")
        except Exception as e:
            logger.debug("[Poller] fetch_failed_job %s: %s", run_id, e)
        return ""

    # ── T7 : Classifier le failure_type depuis le nom du job ─────────────────

    @staticmethod
    def _classify_job_failure(job_name: str) -> str:
        """
        T7 — Mappe le nom du job GitHub Actions vers un failure_type CIGraph.
        Étend la classification pour couvrir publish et deploy.
        """
        j = job_name.lower()
        # Sécurité en premier (SonarCloud, CodeQL, Trivy, OWASP)
        if any(k in j for k in ("sonar", "codeql", "trivy", "dep-scan", "security", "audit", "owasp")):
            return "security"
        # Docker / publish
        if any(k in j for k in ("publish", "docker", "container", "push", "buildx")):
            return "docker"     # T7 : publish → docker failure type
        # Déploiement
        if any(k in j for k in ("deploy", "ssh", "compose", "release")):
            return "deploy"     # T7 : deploy → nouveau failure type
        # Tests
        if any(k in j for k in ("test", "pytest", "jest", "junit")):
            return "test"
        # Build en dernier
        if any(k in j for k in ("build", "compile", "maven", "gradle", "npm")):
            return "build"
        return "unknown"

    # ── Main polling loop ─────────────────────────────────────────────────────

    # ── CD job classifier ────────────────────────────────────────────────────

    @staticmethod
    def _is_deploy_job(job_name: str) -> bool:
        """Returns True if this job should be handled by CDGraph, not CIGraph."""
        j = job_name.lower()
        return any(k in j for k in ("deploy", "publish", "ssh", "compose", "release", "docker"))

    # ── Main polling loop ─────────────────────────────────────────────────────

    def poll_once(self) -> int:
        """
        Un cycle de polling. Analyse les runs non encore traités.
        CD jobs (deploy/publish) → CDGraph.
        All other CI jobs        → CIGraph.
        Returns: nombre de runs analysés
        """
        from langchain_agents.graphs.ci_graph import invoke_ci_run
        from langchain_agents.graphs.cd_graph import invoke_cd_run

        runs = self._fetch_completed_runs()
        analyzed = 0

        for run in runs[:self.max_runs]:
            run_id_str = str(run["id"])

            # Skip si déjà traité
            if run_id_str in self._processed:
                continue

            conclusion  = run.get("conclusion", "")
            head_sha    = run.get("head_sha", "")
            head_branch = run.get("head_branch", "")
            prs         = run.get("pull_requests", [])
            pr_number   = prs[0]["number"] if prs else None
            duration    = 0

            # Calculer la durée
            try:
                from datetime import datetime
                created = run.get("created_at", "")
                updated = run.get("updated_at", "")
                if created and updated:
                    fmt = "%Y-%m-%dT%H:%M:%SZ"
                    duration = int(
                        (datetime.strptime(updated, fmt) - datetime.strptime(created, fmt)).total_seconds()
                    )
            except Exception:
                pass

            # Détecter le job échoué (T7 : inclut publish/deploy)
            stage_failed = ""
            if conclusion == "failure":
                stage_failed = self._fetch_failed_job(run_id_str)

            failure_type = self._classify_job_failure(stage_failed)

            logger.info(
                "[Poller] Run %s | %s | branch=%s | stage=%s | type=%s",
                run_id_str, conclusion, head_branch, stage_failed or "?", failure_type
            )

            # Route to CDGraph (deploy/publish) or CIGraph (all other jobs)
            use_cd_graph = self._is_deploy_job(stage_failed or "")
            graph_name   = "CDGraph" if use_cd_graph else "CIGraph"

            try:
                logger.info(
                    "[Poller] Invoque %s — run=%s pr_number=%s branch=%s",
                    graph_name, run_id_str, pr_number, head_branch
                )
                if use_cd_graph:
                    result = invoke_cd_run(
                        run_id=run_id_str,
                        repo=self.repo,
                        owner=self.owner,
                        project_key=self.project_key,
                        pr_number=pr_number,
                        head_sha=head_sha,
                        pr_branch=head_branch,
                        stage_failed=stage_failed,
                        run_conclusion=conclusion,
                        run_duration_seconds=duration,
                    )
                else:
                    result = invoke_ci_run(
                        run_id=run_id_str,
                        repo=self.repo,
                        owner=self.owner,
                        project_key=self.project_key,
                        pr_number=pr_number,
                        head_sha=head_sha,
                        pr_branch=head_branch,
                        stage_failed=stage_failed,
                        run_conclusion=conclusion,
                        run_duration_seconds=duration,
                    )

                notif_level = result.get("notification_level", "INFO")
                logger.info(
                    "[Poller] Run %s (%s) → notification=%s comment=%s",
                    run_id_str, graph_name, notif_level, result.get("comment_posted")
                )
            except Exception as e:
                logger.error("[Poller] %s failed for run %s: %s", graph_name, run_id_str, e)

            self._mark_processed(run_id_str)
            analyzed += 1

        return analyzed

    def run(self, max_cycles: int = 0):
        """
        Boucle principale de polling (bloquante).

        Args:
            max_cycles: 0 = infini, N = s'arrête après N cycles (tests)
        """
        cycle = 0
        logger.info(
            "[Poller] Démarrage — repo=%s interval=%ds branch=%s",
            self.repo, self.interval, self.branch or "all"
        )
        print(f"\n🔄 CI Poller démarré — {self.repo} (interval: {self.interval}s)")
        print(f"   Jobs surveillés: {', '.join(sorted(self.watch_jobs))}")
        print(f"   T7: publish + deploy connectés au CIGraph\n")

        try:
            while True:
                cycle += 1
                try:
                    n = self.poll_once()
                    if n:
                        print(f"   [{cycle}] {n} run(s) analysé(s)")
                    else:
                        print(f"   [{cycle}] Aucun nouveau run", end="\r")
                except Exception as e:
                    logger.error("[Poller] poll_once error: %s", e)

                if max_cycles and cycle >= max_cycles:
                    break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n⏹ Poller arrêté (Ctrl+C)")
            logger.info("[Poller] Arrêt manuel")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CI Poller — Surveille GitHub Actions et déclenche le CIGraph IA"
    )
    parser.add_argument("--repo",       required=True, help="owner/repo")
    parser.add_argument("--interval",   type=int, default=60, help="Secondes entre polls (défaut: 60)")
    parser.add_argument("--branch",     default="", help="Surveiller uniquement cette branche")
    parser.add_argument("--project-key", default="", dest="project_key",
                        help="SonarCloud project key")
    parser.add_argument("--watch-jobs", nargs="*", dest="watch_jobs",
                        help="Jobs à surveiller (défaut: tous)")
    parser.add_argument("--max-cycles", type=int, default=0, dest="max_cycles",
                        help="Nombre max de cycles (0=infini)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    poller = CIPoller(
        repo=args.repo,
        interval=args.interval,
        branch=args.branch,
        project_key=args.project_key,
        watch_jobs=args.watch_jobs,
    )
    poller.run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
