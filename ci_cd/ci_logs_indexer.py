"""
ci_logs_indexer.py — Agent LangChain pour indexer les runs CI dans Redis.

4 Pillars :
  LLM     : None (indexation pure — 0 token consommé)
  Tools   : tool_index_ci_run, tool_search_similar_failure
  Memory  : Redis (ci:runs:{repo}, ci:run:{run_id}, ci:errors:{hash})
  Planning: Hash des logs → détecter si erreur similaire déjà vue

Usage :
    indexer = CILogsIndexer()
    summary = indexer.index_run(run_id, repo, logs, status, stage_failed)
    similar = indexer.search_similar(logs)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CILogsIndexer:
    """
    Indexe les runs GitHub Actions dans Redis pour :
      1. Historique des runs (Sorted Set par timestamp)
      2. Détails par run (Hash)
      3. Clustering d'erreurs similaires (Hash par hash d'erreur)
      4. Fixes historiques (List par hash d'erreur)

    Réutilise les structures Redis définies dans ci_tools.py :
      ci:runs:{repo}         → Sorted Set
      ci:run:{run_id}        → Hash
      ci:errors:{error_hash} → Hash
      ci:fixes:{error_hash}  → List
    """

    # TTL (seconds)
    RUN_TTL = 86400 * 30    # 30 jours
    ERROR_TTL = 86400 * 90  # 90 jours
    FIX_TTL = 86400 * 365   # 1 an

    def __init__(self):
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            from services.mcp_redis_service import get_mcp_redis
            self._redis = get_mcp_redis()
        return self._redis

    @staticmethod
    def _error_hash(text: str) -> str:
        """Hash court (12 chars) pour regrouper les erreurs similaires."""
        return hashlib.md5(text.strip().lower()[:500].encode()).hexdigest()[:12]

    @staticmethod
    def _extract_error_signature(logs: str) -> str:
        """
        Extrait la signature de l'erreur depuis les logs.
        Ignore les lignes de timestamp et les préfixes variables.
        """
        if not logs:
            return ""
        lines = logs.splitlines()
        error_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Garder les lignes qui semblent être des erreurs
            lower = line_stripped.lower()
            if any(k in lower for k in [
                "error", "fail", "exception", "traceback", "fatal",
                "cannot", "not found", "permission denied", "exit code",
                "build failed", "test failed", "compilation failed",
            ]):
                error_lines.append(line_stripped[:200])
            if len(error_lines) >= 10:
                break
        return "\n".join(error_lines) if error_lines else logs[:300]

    # ── Core: Index a run ─────────────────────────────────────────────────────

    def index_run(
        self,
        run_id: str,
        repo: str,
        logs: str,
        status: str,
        stage_failed: str = "",
        duration_seconds: int = 0,
        pr_number: Optional[int] = None,
        head_sha: str = "",
    ) -> Dict[str, Any]:
        """
        Indexe un CI run dans Redis.

        Args:
            run_id: GitHub Actions run ID
            repo: "{owner}/{repo}"
            logs: Logs du run (bruts, tronqués à 8000 chars)
            status: "success" | "failure" | "cancelled"
            stage_failed: Nom du job/stage qui a échoué
            duration_seconds: Durée du run
            pr_number: Numéro de PR associé (optionnel)
            head_sha: SHA du commit HEAD

        Returns:
            {
                "indexed": True,
                "run_id": ...,
                "error_hash": ... (si failure),
                "similar_seen": N (nombre fois cette erreur vue avant),
            }
        """
        try:
            redis = self._get_redis()
            ts = int(time.time())
            error_hash = None
            similar_seen = 0

            # 1. Sorted Set : historique des runs du repo (score = timestamp)
            redis.zadd(f"ci:runs:{repo}", float(ts), run_id)

            # 2. Nettoyage : garder seulement les 500 derniers runs par repo
            try:
                all_run_ids = redis.zrange(f"ci:runs:{repo}", 0, -1)
                if len(all_run_ids) > 500:
                    for old_id in all_run_ids[: len(all_run_ids) - 500]:
                        try:
                            redis.zrem(f"ci:runs:{repo}", str(old_id))
                        except Exception:
                            pass
            except Exception:
                pass

            # 3. Hash : détails complets du run
            logs_summary = logs[:500] if logs else ""
            run_data = {
                "run_id": run_id,
                "repo": repo,
                "status": status,
                "stage_failed": stage_failed or "",
                "duration": str(duration_seconds),
                "logs_summary": logs_summary,
                "timestamp": str(ts),
                "head_sha": head_sha or "",
                "pr_number": str(pr_number) if pr_number else "",
            }
            redis.hset_dict(f"ci:run:{run_id}", run_data, expire_seconds=self.RUN_TTL)

            # 4. Si failure → indexer l'erreur pour similarité future
            if status == "failure":
                error_sig = self._extract_error_signature(logs)
                error_hash = self._error_hash(error_sig)
                error_key = f"ci:errors:{error_hash}"

                existing = redis.hgetall(error_key) or {}
                similar_seen = int(existing.get("count", 0))
                new_count = similar_seen + 1

                redis.hset_dict(error_key, {
                    "count": str(new_count),
                    "last_seen": str(ts),
                    "last_run_id": run_id,
                    "repo": repo,
                    "stage": stage_failed or "",
                    "sample": error_sig[:400],
                    "fix_applied": existing.get("fix_applied", ""),
                }, expire_seconds=self.ERROR_TTL)

                logger.info(
                    "[CILogsIndexer] %s/%s failure indexed — error_hash=%s (seen %dx)",
                    repo, run_id, error_hash, new_count
                )
            else:
                logger.info(
                    "[CILogsIndexer] %s/%s %s indexed", repo, run_id, status
                )

            return {
                "indexed": True,
                "run_id": run_id,
                "repo": repo,
                "status": status,
                "error_hash": error_hash,
                "similar_seen": similar_seen,
            }

        except Exception as e:
            logger.error("[CILogsIndexer] index_run failed: %s", e)
            return {"indexed": False, "error": str(e)}

    # ── Search: Similar failures ──────────────────────────────────────────────

    def search_similar(
        self,
        logs_or_error: str,
        max_results: int = 5,
        min_similarity: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Recherche des failures similaires dans Redis.

        Stratégie :
          1. Hash exact (signature d'erreur identique) → similarité 1.0
          2. Fuzzy match sur les mots (tokens) des samples stockés

        Args:
            logs_or_error: Logs ou message d'erreur à comparer
            max_results: Nombre max de résultats
            min_similarity: Seuil minimum de similarité (0–1)

        Returns:
            Liste de {error_hash, count, last_seen, stage, fix_applied, similarity}
        """
        try:
            redis = self._get_redis()
            error_sig = self._extract_error_signature(logs_or_error)
            error_hash = self._error_hash(error_sig)

            results = []

            # 1. Hash exact
            exact_key = f"ci:errors:{error_hash}"
            exact_data = redis.hgetall(exact_key)
            if exact_data and int(exact_data.get("count", 0)) > 0:
                results.append({
                    "error_hash": error_hash,
                    "count": int(exact_data.get("count", 0)),
                    "last_seen": exact_data.get("last_seen", ""),
                    "last_run_id": exact_data.get("last_run_id", ""),
                    "stage": exact_data.get("stage", ""),
                    "fix_applied": exact_data.get("fix_applied", ""),
                    "sample": exact_data.get("sample", "")[:200],
                    "similarity": 1.0,
                    "match_type": "exact",
                })

            # 2. Fuzzy search sur les autres clés ci:errors:*
            try:
                all_keys = redis.keys("ci:errors:*") or []
                words_query = set(error_sig.lower().split())

                for k in all_keys[:200]:
                    if k == exact_key:
                        continue  # déjà traité
                    d = redis.hgetall(k) or {}
                    sample = d.get("sample", "")
                    if not sample:
                        continue

                    words_sample = set(sample.lower().split())
                    if not words_query or not words_sample:
                        continue

                    sim = len(words_query & words_sample) / max(
                        len(words_query), len(words_sample)
                    )
                    if sim >= min_similarity:
                        results.append({
                            "error_hash": k.replace("ci:errors:", ""),
                            "count": int(d.get("count", 0)),
                            "last_seen": d.get("last_seen", ""),
                            "last_run_id": d.get("last_run_id", ""),
                            "stage": d.get("stage", ""),
                            "fix_applied": d.get("fix_applied", ""),
                            "sample": sample[:200],
                            "similarity": round(sim, 2),
                            "match_type": "fuzzy",
                        })
            except Exception as e:
                logger.debug("Fuzzy search failed: %s", e)

            # Trier par similarité décroissante puis count décroissant
            results.sort(key=lambda x: (-x["similarity"], -x["count"]))
            return results[:max_results]

        except Exception as e:
            logger.error("[CILogsIndexer] search_similar failed: %s", e)
            return []

    # ── Store fix ─────────────────────────────────────────────────────────────

    def store_fix(
        self,
        logs_or_error: str,
        fix_description: str,
        run_id: str = "",
    ) -> bool:
        """
        Stocke un fix qui a fonctionnné pour un type d'erreur.
        Utilise json_set (list JSON) à la place de rpush (non disponible dans redis-mcp-server).
        """
        try:
            redis = self._get_redis()
            error_sig = self._extract_error_signature(logs_or_error)
            error_hash = self._error_hash(error_sig)
            fixes_key = f"ci:fixes:{error_hash}"

            # Lire la liste actuelle (JSON stocke en json_set)
            existing = redis.json_get(fixes_key, "$") or []
            if not isinstance(existing, list):
                existing = []

            existing.append({
                "fix": fix_description,
                "run_id": run_id,
                "timestamp": int(time.time()),
            })
            # Garder les 20 derniers fixes
            existing = existing[-20:]

            redis.json_set(fixes_key, "$", existing)

            # Mettre à jour le champ fix_applied dans l'index d'erreurs
            redis.hset(f"ci:errors:{error_hash}", "fix_applied", fix_description[:300])

            logger.info("[CILogsIndexer] Fix stocké pour error_hash=%s", error_hash)
            return True

        except Exception as e:
            logger.error("[CILogsIndexer] store_fix failed: %s", e)
            return False

    # ── Get recent runs ───────────────────────────────────────────────────────

    def get_recent_runs(
        self,
        repo: str,
        limit: int = 20,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les N derniers runs indexés pour un repo.

        Args:
            repo: "{owner}/{repo}"
            limit: Nombre de runs à retourner
            status_filter: "success" | "failure" | None (tous)

        Returns:
            Liste de runs triés par date décroissante
        """
        try:
            redis = self._get_redis()
            run_ids = redis.zrevrange(f"ci:runs:{repo}", 0, limit * 2 - 1)

            runs = []
            for rid in run_ids:
                if len(runs) >= limit:
                    break
                data = redis.hgetall(f"ci:run:{rid}") or {}
                if not data:
                    continue
                if status_filter and data.get("status") != status_filter:
                    continue
                runs.append({
                    "run_id": rid,
                    "status": data.get("status", ""),
                    "stage_failed": data.get("stage_failed", ""),
                    "duration": int(data.get("duration", 0)),
                    "timestamp": int(data.get("timestamp", 0)),
                    "head_sha": data.get("head_sha", ""),
                    "pr_number": data.get("pr_number", ""),
                })

            return runs

        except Exception as e:
            logger.error("[CILogsIndexer] get_recent_runs failed: %s", e)
            return []

    def get_failure_stats(self, repo: str) -> Dict[str, Any]:
        """
        Statistiques de failure rate pour un repo.

        Returns:
            {total, failures, success_rate, top_failed_stages}
        """
        try:
            redis = self._get_redis()
            all_ids = redis.zrevrange(f"ci:runs:{repo}", 0, 99)

            total = len(all_ids)
            failures = 0
            stage_counts: Dict[str, int] = {}

            for rid in all_ids:
                data = redis.hgetall(f"ci:run:{rid}") or {}
                if data.get("status") == "failure":
                    failures += 1
                    stage = data.get("stage_failed", "unknown")
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1

            top_stages = sorted(
                [{"stage": s, "count": c} for s, c in stage_counts.items()],
                key=lambda x: -x["count"]
            )[:5]

            success_rate = round((total - failures) / total * 100, 1) if total > 0 else 100.0

            return {
                "repo": repo,
                "total": total,
                "failures": failures,
                "success_rate": success_rate,
                "top_failed_stages": top_stages,
            }

        except Exception as e:
            logger.error("[CILogsIndexer] get_failure_stats failed: %s", e)
            return {"repo": repo, "total": 0, "failures": 0, "success_rate": 100.0}
