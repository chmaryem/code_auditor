"""
cd_deploy_tracker.py — Tracks every deployment lifecycle in Redis.

Redis keys:
  cd:deploy:{deploy_id}   → Hash  (full deployment record)
  cd:deploys:{repo}       → Sorted Set (score=timestamp, member=deploy_id)
  cd:env:{repo}:{env}     → Hash  (current state of an environment)
  cd:rollback:{repo}:{env}→ String (last successful deploy_id for rollback)

Usage:
    tracker = CDDeployTracker()
    deploy_id = tracker.record_deploy(
        repo="owner/repo", environment="production",
        commit_sha="abc123", version="sha-abc123", branch="main",
        deploy_url="https://myapp.com", run_id="123456789"
    )
    tracker.mark_deploy_success(deploy_id, health_ok=True)
    tracker.mark_deploy_failure(deploy_id, reason="health check failed")
    last_ok = tracker.get_last_successful_deploy("owner/repo", "production")
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Redis TTLs
_DEPLOY_TTL   = 86400 * 90   # 90 days — deployment records
_ENV_TTL      = 86400 * 365  # 1 year  — environment state
_HISTORY_MAX  = 200           # Keep last 200 deployments per repo


@dataclass
class DeployRecord:
    """In-memory representation of a deployment record."""
    deploy_id:   str
    repo:        str
    environment: str
    commit_sha:  str
    version:     str
    branch:      str
    run_id:      str
    deploy_url:  str
    status:      str           # "pending" | "deploying" | "success" | "failed" | "rolledback"
    started_at:  int           # Unix timestamp
    finished_at: int = 0
    duration_s:  int = 0
    health_ok:   bool = False
    failure_reason: str = ""


class CDDeployTracker:
    """
    Tracks deployment lifecycle for every environment.

    Key behaviors:
      - record_deploy()         → creates a new pending record
      - mark_deploy_success()   → updates status, stores as last-good for rollback
      - mark_deploy_failure()   → updates status, preserves last-good pointer
      - get_last_successful_deploy() → returns the record to roll back to
      - get_env_state()         → current production state
      - get_recent_deploys()    → history for a repo
    """

    def __init__(self) -> None:
        self._redis = None

    # ── Redis connection ──────────────────────────────────────────────────────

    def _get_redis(self):
        if self._redis is None:
            from services.mcp_redis_service import get_mcp_redis
            self._redis = get_mcp_redis()
        return self._redis

    # ── Core: record a new deployment ─────────────────────────────────────────

    def record_deploy(
        self,
        repo:        str,
        environment: str,
        commit_sha:  str,
        version:     str,
        branch:      str       = "main",
        deploy_url:  str       = "",
        run_id:      str       = "",
        deploy_id:   str       = "",
    ) -> str:
        """
        Creates a new deployment record in Redis with status "deploying".

        Args:
            repo:        "{owner}/{repo}"
            environment: "production" | "staging" | "dev"
            commit_sha:  Full commit SHA being deployed
            version:     Docker tag or semver (ex: "sha-abc1234" or "v1.2.3")
            branch:      Source branch (ex: "main")
            deploy_url:  URL to check health (ex: "https://myapp.com")
            run_id:      GitHub Actions run ID that triggered the deploy
            deploy_id:   Optional — auto-generated if empty

        Returns:
            deploy_id (string UUID)
        """
        if not deploy_id:
            deploy_id = f"deploy-{int(time.time())}-{uuid.uuid4().hex[:8]}"

        ts = int(time.time())
        record = {
            "deploy_id":      deploy_id,
            "repo":           repo,
            "environment":    environment,
            "commit_sha":     commit_sha,
            "version":        version,
            "branch":         branch,
            "run_id":         run_id,
            "deploy_url":     deploy_url,
            "status":         "deploying",
            "started_at":     str(ts),
            "finished_at":    "0",
            "duration_s":     "0",
            "health_ok":      "false",
            "failure_reason": "",
        }

        try:
            redis = self._get_redis()
            # 1. Full record Hash
            redis.hset_dict(f"cd:deploy:{deploy_id}", record,
                            expire_seconds=_DEPLOY_TTL)

            # 2. Timeline Sorted Set per repo
            redis.zadd(f"cd:deploys:{repo}", float(ts), deploy_id)
            # Trim to last _HISTORY_MAX
            all_ids = redis.zrange(f"cd:deploys:{repo}", 0, -1)
            if len(all_ids) > _HISTORY_MAX:
                for old in all_ids[: len(all_ids) - _HISTORY_MAX]:
                    redis.zrem(f"cd:deploys:{repo}", str(old))

            # 3. Update env state (optimistic — mark as "deploying")
            env_key = f"cd:env:{repo}:{environment}"
            redis.hset_dict(env_key, {
                "current_deploy_id": deploy_id,
                "status":            "deploying",
                "version":           version,
                "commit_sha":        commit_sha,
                "branch":            branch,
                "deploy_url":        deploy_url,
                "last_updated":      str(ts),
            }, expire_seconds=_ENV_TTL)

            logger.info(
                "[CDTracker] Deploy started — id=%s repo=%s env=%s ver=%s",
                deploy_id, repo, environment, version
            )
        except Exception as e:
            logger.error("[CDTracker] record_deploy failed: %s", e)

        return deploy_id

    # ── Lifecycle transitions ──────────────────────────────────────────────────

    def mark_deploy_success(
        self,
        deploy_id:   str,
        health_ok:   bool = True,
    ) -> None:
        """
        Marks a deployment as successful and stores it as the rollback anchor.

        Args:
            deploy_id: The deploy_id returned by record_deploy()
            health_ok: Whether the post-deploy health check passed
        """
        ts = int(time.time())
        try:
            redis = self._get_redis()
            key = f"cd:deploy:{deploy_id}"

            # Fetch original record to compute duration and update env
            rec = redis.hgetall(key) or {}
            started = int(rec.get("started_at", ts))
            duration = ts - started
            repo        = rec.get("repo", "")
            environment = rec.get("environment", "")

            # Update the record
            redis.hset_dict(key, {
                "status":      "success",
                "finished_at": str(ts),
                "duration_s":  str(duration),
                "health_ok":   "true" if health_ok else "false",
            }, expire_seconds=_DEPLOY_TTL)

            if repo and environment:
                # Update env state
                redis.hset_dict(f"cd:env:{repo}:{environment}", {
                    "status":            "success",
                    "last_ok_deploy_id": deploy_id,
                    "last_ok_version":   rec.get("version", ""),
                    "last_ok_sha":       rec.get("commit_sha", ""),
                    "last_ok_at":        str(ts),
                    "last_updated":      str(ts),
                }, expire_seconds=_ENV_TTL)

                # Store as rollback anchor
                redis.hset_dict(f"cd:rollback:{repo}:{environment}", {
                    "deploy_id":  deploy_id,
                    "version":    rec.get("version", ""),
                    "commit_sha": rec.get("commit_sha", ""),
                    "branch":     rec.get("branch", ""),
                    "deploy_url": rec.get("deploy_url", ""),
                    "success_at": str(ts),
                }, expire_seconds=_ENV_TTL)

            logger.info(
                "[CDTracker] Deploy SUCCESS — id=%s repo=%s env=%s dur=%ds",
                deploy_id, repo, environment, duration
            )
        except Exception as e:
            logger.error("[CDTracker] mark_deploy_success failed: %s", e)

    def mark_deploy_failure(
        self,
        deploy_id:      str,
        failure_reason: str = "",
    ) -> None:
        """
        Marks a deployment as failed.
        The rollback anchor is NOT updated — it still points to the last success.

        Args:
            deploy_id:      The deploy_id returned by record_deploy()
            failure_reason: Human-readable description of what went wrong
        """
        ts = int(time.time())
        try:
            redis = self._get_redis()
            key = f"cd:deploy:{deploy_id}"
            rec = redis.hgetall(key) or {}
            started  = int(rec.get("started_at", ts))
            duration = ts - started
            repo        = rec.get("repo", "")
            environment = rec.get("environment", "")

            redis.hset_dict(key, {
                "status":         "failed",
                "finished_at":    str(ts),
                "duration_s":     str(duration),
                "health_ok":      "false",
                "failure_reason": failure_reason[:400],
            }, expire_seconds=_DEPLOY_TTL)

            if repo and environment:
                redis.hset_dict(f"cd:env:{repo}:{environment}", {
                    "status":       "failed",
                    "last_updated": str(ts),
                    "last_failure": failure_reason[:200],
                }, expire_seconds=_ENV_TTL)

            logger.warning(
                "[CDTracker] Deploy FAILED — id=%s repo=%s env=%s reason=%s",
                deploy_id, repo, environment, failure_reason[:100]
            )
        except Exception as e:
            logger.error("[CDTracker] mark_deploy_failure failed: %s", e)

    # ── Query methods ──────────────────────────────────────────────────────────

    def get_deploy(self, deploy_id: str) -> Dict[str, Any]:
        """Returns the full record for a deploy_id."""
        try:
            return self._get_redis().hgetall(f"cd:deploy:{deploy_id}") or {}
        except Exception as e:
            logger.error("[CDTracker] get_deploy failed: %s", e)
            return {}

    def get_env_state(self, repo: str, environment: str) -> Dict[str, Any]:
        """
        Returns the current state of an environment.

        Returns:
            {current_deploy_id, status, version, commit_sha, branch, deploy_url,
             last_ok_deploy_id, last_ok_version, last_updated, last_failure}
        """
        try:
            return self._get_redis().hgetall(
                f"cd:env:{repo}:{environment}"
            ) or {}
        except Exception as e:
            logger.error("[CDTracker] get_env_state failed: %s", e)
            return {}

    def get_last_successful_deploy(
        self,
        repo:        str,
        environment: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the last successful deployment record for rollback.

        Returns:
            {deploy_id, version, commit_sha, branch, deploy_url, success_at}
            or None if no successful deploy exists.
        """
        try:
            redis  = self._get_redis()
            anchor = redis.hgetall(f"cd:rollback:{repo}:{environment}") or {}
            if not anchor or not anchor.get("deploy_id"):
                return None
            return {
                "deploy_id":  anchor.get("deploy_id", ""),
                "version":    anchor.get("version", ""),
                "commit_sha": anchor.get("commit_sha", ""),
                "branch":     anchor.get("branch", ""),
                "deploy_url": anchor.get("deploy_url", ""),
                "success_at": int(anchor.get("success_at", 0)),
            }
        except Exception as e:
            logger.error("[CDTracker] get_last_successful_deploy failed: %s", e)
            return None

    def get_recent_deploys(
        self,
        repo:   str,
        limit:  int = 20,
        env:    Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns the N most recent deployments for a repo.

        Args:
            repo:  "{owner}/{repo}"
            limit: Max results
            env:   Filter by environment (optional)

        Returns:
            List of deploy records sorted by date descending
        """
        try:
            redis = self._get_redis()
            ids   = redis.zrevrange(f"cd:deploys:{repo}", 0, limit * 2 - 1)
            out   = []
            for did in ids:
                if len(out) >= limit:
                    break
                rec = redis.hgetall(f"cd:deploy:{did}") or {}
                if not rec:
                    continue
                if env and rec.get("environment") != env:
                    continue
                out.append({
                    "deploy_id":      rec.get("deploy_id", str(did)),
                    "environment":    rec.get("environment", ""),
                    "status":         rec.get("status", ""),
                    "version":        rec.get("version", ""),
                    "commit_sha":     rec.get("commit_sha", "")[:8],
                    "branch":         rec.get("branch", ""),
                    "started_at":     int(rec.get("started_at", 0)),
                    "duration_s":     int(rec.get("duration_s", 0)),
                    "health_ok":      rec.get("health_ok") == "true",
                    "failure_reason": rec.get("failure_reason", ""),
                })
            return out
        except Exception as e:
            logger.error("[CDTracker] get_recent_deploys failed: %s", e)
            return []

    def get_deploy_stats(self, repo: str, environment: str = "") -> Dict[str, Any]:
        """
        Computes deployment success rate and average duration.

        Returns:
            {total, successes, failures, success_rate, avg_duration_s,
             last_deploy_at, last_deploy_status}
        """
        try:
            deploys = self.get_recent_deploys(repo, limit=50, env=environment or None)
            total     = len(deploys)
            successes = sum(1 for d in deploys if d["status"] == "success")
            failures  = sum(1 for d in deploys if d["status"] == "failed")
            durations = [d["duration_s"] for d in deploys if d["duration_s"] > 0]
            avg_dur   = round(sum(durations) / len(durations)) if durations else 0
            last      = deploys[0] if deploys else {}
            return {
                "repo":               repo,
                "environment":        environment,
                "total":              total,
                "successes":          successes,
                "failures":           failures,
                "success_rate":       round(successes / total * 100, 1) if total else 100.0,
                "avg_duration_s":     avg_dur,
                "last_deploy_at":     last.get("started_at", 0),
                "last_deploy_status": last.get("status", ""),
            }
        except Exception as e:
            logger.error("[CDTracker] get_deploy_stats failed: %s", e)
            return {"repo": repo, "total": 0, "successes": 0, "failures": 0, "success_rate": 100.0}
