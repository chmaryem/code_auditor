

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

NPM_TOOL_NAMES = {
    "get_pr":           "get_pull_request",
    "list_pr_files":    "list_pull_request_files",
    "get_pr_files_go":  "get_pull_request_files",
    "get_file":         "get_file_contents",
    "create_file":      "create_or_update_file",
    "push_files":       "push_files",
    "create_branch":    "create_branch",
    "post_comment":     "add_issue_comment",
    "post_comment_npm": "create_issue_comment",
    "list_comments":    "list_issue_comments",
    "update_comment":   "update_issue_comment",
    "create_review":    "create_pull_request_review",
    "list_reviews":     "get_pull_request_reviews",
    "list_reviews_npm": "list_pull_request_reviews",
    "list_checks":      "list_check_runs_for_ref",
    "search_code":      "search_code",
    "list_commits":     "list_commits",
    "pr_status":        "get_pull_request_status",
}


def _get_github_token() -> str:
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GITHUB_PERSONAL_ACCESS_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    return token


class _LoopManager:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or (self._thread and not self._thread.is_alive()):
                self._ready.clear()
                self._thread = threading.Thread(
                    target=self._run_loop, daemon=True, name="mcp-event-loop"
                )
                self._thread.start()
                self._ready.wait(timeout=5)
        return self._loop

    def run(self, coro, timeout: int = 120):
        loop = self.get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def shutdown(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


_loop_manager = _LoopManager()


class MCPGitHubService:

    def __init__(self, token: Optional[str] = None):
        self._token = token or _get_github_token()
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._cm = None
        self._available_tools: Set[str] = set()
        self._tool_map: Dict[str, str] = {}

    async def connect(self) -> None:
        if not self._token:
            raise ValueError("GitHub token manquant. Définissez GITHUB_PERSONAL_ACCESS_TOKEN.")
        import platform
        npx_cmd = "npx.cmd" if platform.system() == "Windows" else "npx"
        server_params = StdioServerParameters(
            command=npx_cmd,
            args=["-y", "@modelcontextprotocol/server-github"],
            env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": self._token},
        )
        self._cm = stdio_client(server_params)
        self._read, self._write = await self._cm.__aenter__()
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        await self._session.initialize()
        await self._discover_tools()

    async def _discover_tools(self) -> None:
        tools_result = await self._session.list_tools()
        self._available_tools = {t.name for t in tools_result.tools}
        sys.stderr.write(f"[MCP] {len(self._available_tools)} tools connectés\n")
        sys.stderr.flush()
        self._tool_map = {}
        for alias, default_name in NPM_TOOL_NAMES.items():
            if default_name in self._available_tools:
                self._tool_map[alias] = default_name
            else:
                fallback = self._find_closest_tool(alias)
                if fallback:
                    self._tool_map[alias] = fallback

    def _find_closest_tool(self, alias: str) -> Optional[str]:
        keywords = {
            "get_pr":           ["pull_request", "get_pull"],
            "list_pr_files":    ["pull_request_files", "pr_files"],
            "get_pr_files_go":  ["pull_request_files", "pr_files"],
            "get_file":         ["file_contents", "get_file"],
            "create_file":      ["create_or_update", "update_file"],
            "push_files":       ["push_files"],
            "create_branch":    ["create_branch"],
            "post_comment":     ["issue_comment", "add_issue"],
            "post_comment_npm": ["issue_comment", "create_issue"],
            "create_review":    ["pull_request_review", "create_review"],
            "list_reviews":     ["pull_request_reviews", "get_pull_request_reviews"],
            "list_reviews_npm": ["pull_request_reviews"],
            "list_checks":      ["check_runs", "checks"],
            "pr_status":        ["pull_request_status", "pr_status"],
        }
        for kw in keywords.get(alias, []):
            for tool_name in self._available_tools:
                if kw in tool_name.lower():
                    return tool_name
        return None

    def _resolve_tool(self, alias: str) -> str:
        if alias in self._tool_map:
            return self._tool_map[alias]
        fallback = self._find_closest_tool(alias)
        if fallback:
            self._tool_map[alias] = fallback
            return fallback
        raise ValueError(f"Aucun tool pour '{alias}'. Dispo: {sorted(self._available_tools)}")

    def has_tool(self, alias: str) -> bool:
        return alias in self._tool_map or bool(self._find_closest_tool(alias))

    async def disconnect(self) -> None:
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._cm:
            await self._cm.__aexit__(None, None, None)
            self._cm = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError("MCP non connecté.")
        result = await self._session.call_tool(name, arguments)
        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        return json.loads(block.text)
                    except json.JSONDecodeError:
                        return block.text
        return None

    async def _call_alias(self, alias: str, arguments: Dict[str, Any]) -> Any:
        return await self._call_tool(self._resolve_tool(alias), arguments)

    # ── REST helper (writes) ──────────────────────────────────────────────────
    # Le serveur MCP npm échoue silencieusement sur les écritures (create_branch,
    # push_file, create_pr) — mêmes limitations que les lectures. On passe donc en
    # REST-first pour les écritures aussi. MCP reste en fallback.
    def _rest_api(self, method: str, url: str, payload: Optional[dict] = None) -> Any:
        """Appel REST GitHub synchrone. Lève une exception claire en cas d'échec
        (contrairement à MCP qui renvoie {} en silence). HTTPError propagée pour
        que l'appelant gère les cas 422 (déjà existant)."""
        import urllib.request as _ur
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if not token:
            raise RuntimeError("GITHUB token manquant pour l'appel REST")
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "code-auditor/1.0",
            "Content-Type": "application/json",
        }
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = _ur.Request(url, data=data, headers=headers, method=method)
        with _ur.urlopen(req, timeout=20) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}

    # ── Pull Requests ─────────────────────────────────────────────────────────

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        # FIX v6.1 : pull_number en premier (le serveur MCP npm l'attend)
        for key in ["pull_number", "pullNumber"]:
            try:
                result = await self._call_alias("get_pr", {
                    "owner": owner, "repo": repo, key: pr_number,
                })
                if result and isinstance(result, dict) and result.get("number"):
                    return result
            except Exception:
                continue

        # REST fallback — MCP retourne {} quand le tool échoue
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if token:
            try:
                import urllib.request as _ur
                api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
                req = _ur.Request(api_url, headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "code-auditor/1.0",
                })
                with _ur.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                if isinstance(data, dict) and data.get("number"):
                    sys.stderr.write(f"[REST] get_pull_request fallback OK: PR #{pr_number}\n")
                    sys.stderr.flush()
                    return data
            except Exception as e:
                logger.debug("get_pull_request REST fallback: %s", e)

        return {}

    async def get_pr_mergeable_status(
        self, owner: str, repo: str, pr_number: int, max_polls: int = 4
    ) -> Dict[str, Any]:
        """
        Détecte les conflits de manière fiable.

        STRATÉGIE (par ordre de priorité) :

        1. get_pull_request_status → mergeableState
           "dirty"   = conflits CONFIRMÉS (has_conflicts=True)
           "clean"   = pas de conflits (has_conflicts=False)
           "unknown" = calcul en cours → continuer

        2. Poll get_pull_request → champ 'mergeable'
           true  → has_conflicts=False
           false → has_conflicts=True
           null  → calcul en cours → réessayer (max 4 fois, 3s entre)

        3. Fallback : lire les fichiers de la PR et regarder si le
           champ 'status' contient 'conflicted' ou si 'patch' contient
           des marqueurs de conflit.

        Returns dict avec has_conflicts (bool fiable) + métadonnées PR.
        """
        pr_data = {}
        mergeable = None
        has_conflicts = False

        # ── Stratégie 0 : Appel REST API direct (contourne la limitation MCP) ──
        # Le serveur MCP npm ne transmet pas le champ "mergeable" de l'API REST.
        # On appelle directement l'API GitHub avec urllib pour avoir ce champ.
        # C'est la source la plus fiable (même données que l'UI GitHub).
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if token:
            import urllib.request
            api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
            headers_rest = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }

            def _result_from(rest_pr: dict, has_conflicts: bool, mergeable) -> dict:
                return {
                    "has_conflicts": has_conflicts,
                    "mergeable": mergeable,
                    "conflict_files": [],
                    "base_ref": rest_pr.get("base", {}).get("ref", "main"),
                    "head_ref": rest_pr.get("head", {}).get("ref", ""),
                    "head_sha": rest_pr.get("head", {}).get("sha", ""),
                    "pr_data": rest_pr,
                }

            # mergeable_state plus stable que mergeable :
            #   "dirty"  = conflits confirmés
            #   ces états = pas de conflit (mergeable même si CI/review bloquent)
            _CLEAN_STATES = {"clean", "blocked", "behind", "unstable", "has_hooks"}

            # GitHub calcule mergeable en asynchrone → backoff progressif (5 essais).
            delays = [0, 1, 2, 3, 3]
            last_pr: dict = {}
            for attempt, delay in enumerate(delays):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    req = urllib.request.Request(api_url, headers=headers_rest)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        rest_pr = json.loads(resp.read().decode())
                    last_pr = rest_pr

                    rest_mergeable = rest_pr.get("mergeable")
                    rest_state = rest_pr.get("mergeable_state", "")
                    sys.stderr.write(
                        f"[REST] Stratégie 0 : mergeable={rest_mergeable!r} "
                        f"state={rest_state!r} (attempt {attempt+1}/{len(delays)})\n"
                    )
                    sys.stderr.flush()

                    # 1) Signal le plus fiable : mergeable_state == "dirty" = conflit
                    if rest_state == "dirty":
                        return _result_from(rest_pr, True, False)
                    # 2) mergeable explicite (true/false)
                    if rest_mergeable is not None:
                        return _result_from(rest_pr, not rest_mergeable, rest_mergeable)
                    # 3) mergeable null mais state concluant → pas de conflit
                    if rest_state in _CLEAN_STATES:
                        return _result_from(rest_pr, False, True)
                    # 4) null + state "unknown"/"" → GitHub calcule encore, on retente
                except Exception as e:
                    sys.stderr.write(f"[REST] Stratégie 0 erreur: {e}\n")
                    sys.stderr.flush()
                    break  # REST inaccessible, passer aux stratégies MCP

            # Tous les essais épuisés : si on a un mergeable_state, le respecter
            # plutôt que de tomber sur le défaut peu fiable des stratégies MCP.
            if last_pr:
                final_state = last_pr.get("mergeable_state", "")
                if final_state == "dirty":
                    return _result_from(last_pr, True, False)
                if final_state in _CLEAN_STATES:
                    return _result_from(last_pr, False, True)

        # ── Stratégie 1 : get_pull_request_status ────────────────────────────
        # NOTE : Ce tool MCP retourne le statut CI/CD (checks), pas mergeableState.
        # On le tente quand même pour extraire mergeableState si présent.
        if "get_pull_request_status" in self._available_tools:
            # FIX v6.1 : pull_number EN PREMIER (le serveur MCP npm l'attend)
            for key in ["pull_number", "pullNumber"]:
                try:
                    status_result = await self._call_tool("get_pull_request_status", {
                        "owner": owner, "repo": repo, key: pr_number,
                    })
                    sys.stderr.write(f"[MCP] pr_status raw: {str(status_result)[:200]}\n")
                    sys.stderr.flush()
                    if isinstance(status_result, dict):
                        # Chercher mergeableState dans la réponse (peut être imbriqué)
                        mergeable_state = (
                            status_result.get("mergeableState")
                            or status_result.get("mergeable_state")
                            or status_result.get("mergeability")
                            or ""
                        )
                        # Chercher aussi dans les sous-objets
                        if not mergeable_state:
                            for v in status_result.values():
                                if isinstance(v, str) and v in ("dirty", "clean", "unknown", "blocked"):
                                    mergeable_state = v
                                    break
                                if isinstance(v, dict):
                                    ms = v.get("mergeableState", v.get("mergeable_state", ""))
                                    if ms:
                                        mergeable_state = ms
                                        break

                        sys.stderr.write(f"[MCP] mergeableState={mergeable_state!r}\n")
                        sys.stderr.flush()

                        if mergeable_state == "dirty":
                            has_conflicts = True
                            pr_data = await self.get_pull_request(owner, repo, pr_number)
                            base_ref = pr_data.get("base", {}).get("ref", "main") if pr_data else "main"
                            head_ref = pr_data.get("head", {}).get("ref", "") if pr_data else ""
                            head_sha = pr_data.get("head", {}).get("sha", "") if pr_data else ""
                            return {
                                "has_conflicts": True,
                                "mergeable": False,
                                "conflict_files": [],
                                "base_ref": base_ref,
                                "head_ref": head_ref,
                                "head_sha": head_sha,
                                "pr_data": pr_data,
                            }
                        elif mergeable_state in ("clean", "blocked"):
                            has_conflicts = False
                            pr_data = await self.get_pull_request(owner, repo, pr_number)
                            base_ref = pr_data.get("base", {}).get("ref", "main") if pr_data else "main"
                            head_ref = pr_data.get("head", {}).get("ref", "") if pr_data else ""
                            head_sha = pr_data.get("head", {}).get("sha", "") if pr_data else ""
                            return {
                                "has_conflicts": False,
                                "mergeable": True,
                                "conflict_files": [],
                                "base_ref": base_ref,
                                "head_ref": head_ref,
                                "head_sha": head_sha,
                                "pr_data": pr_data,
                            }
                        # unknown / "" → continuer avec stratégie 2
                        break  # Si on a un résultat (même vide), pas besoin d'essayer l'autre clé
                except Exception as e:
                    sys.stderr.write(f"[MCP] pr_status erreur: {e}\n")
                    sys.stderr.flush()
                    continue

        # ── Stratégie 2 : Poll get_pull_request → champ mergeable ────────────
        # Réduit à 4 polls (au lieu de 6) : la Stratégie 0 REST a déjà attendu.
        # Délai maintenu à 3s (GitHub nécessite ce temps pour calculer mergeable).
        max_polls = 4
        try:
            # Kick initial : déclencher le calcul de mergeable côté GitHub
            await self.get_pull_request(owner, repo, pr_number)
            await asyncio.sleep(2)
        except Exception:
            pass

        for attempt in range(max_polls):
            try:
                pr_data = await self.get_pull_request(owner, repo, pr_number)
                if not pr_data:
                    await asyncio.sleep(3)
                    continue

                raw_mergeable = pr_data.get("mergeable")
                sys.stderr.write(f"[MCP] poll {attempt+1}/{max_polls} mergeable={raw_mergeable!r}\n")
                sys.stderr.flush()

                if raw_mergeable is None:
                    if attempt < max_polls - 1:
                        await asyncio.sleep(5)  # FIX v6.1 : 5s au lieu de 3s
                        continue
                    # Dernier essai et toujours null → stratégie 3
                else:
                    mergeable = bool(raw_mergeable)
                    break
            except Exception as e:
                logger.debug("get_pr_mergeable_status poll %d: %s", attempt, e)
                await asyncio.sleep(3)

        base_ref = pr_data.get("base", {}).get("ref", "main") if pr_data else "main"
        head_ref = pr_data.get("head", {}).get("ref", "") if pr_data else ""
        head_sha = pr_data.get("head", {}).get("sha", "") if pr_data else ""

        if mergeable is not None:
            has_conflicts = not mergeable
            sys.stderr.write(f"[MCP] Stratégie 2 résultat: has_conflicts={has_conflicts}\n")
            sys.stderr.flush()
            return {
                "has_conflicts": has_conflicts,
                "mergeable": mergeable,
                "conflict_files": [],
                "base_ref": base_ref,
                "head_ref": head_ref,
                "head_sha": head_sha,
                "pr_data": pr_data,
            }

        # ── Stratégie 3 : Inspecter le patch des fichiers de la PR ───────────
        sys.stderr.write("[MCP] Stratégie 3 : inspection des patches PR\n")
        sys.stderr.flush()
        conflict_files = []
        try:
            files = await self.get_pr_files(owner, repo, pr_number)
            for f in files:
                filename = f.get("filename", f.get("path", ""))
                patch = f.get("patch", "")
                file_status = f.get("status", "")
                if file_status == "conflicted":
                    conflict_files.append(filename)
                elif patch and any(m in patch for m in ["<<<<<<<", "======="]):
                    conflict_files.append(filename)
        except Exception as e:
            logger.debug("Stratégie 3: %s", e)

        if conflict_files:
            has_conflicts = True
            sys.stderr.write(f"[MCP] Stratégie 3 résultat: has_conflicts=True, files={conflict_files}\n")
            sys.stderr.flush()
            return {
                "has_conflicts": True,
                "mergeable": False,
                "conflict_files": conflict_files,
                "base_ref": base_ref,
                "head_ref": head_ref,
                "head_sha": head_sha,
                "pr_data": pr_data,
            }

        # ── Stratégie 4 : Comparaison contenu base vs patch ─────────────────
        # Le serveur MCP ne donne pas "mergeable". On détecte les conflits
        # en vérifiant si les lignes supprimées ("-") du patch existent encore
        # dans le fichier actuel sur main. Si elles n'existent plus, main a
        # aussi modifié ces lignes → conflit.
        sys.stderr.write("[MCP] Stratégie 4 : comparaison contenu base vs patch\n")
        sys.stderr.flush()
        conflict_files = []
        try:
            if not files:
                files = await self.get_pr_files(owner, repo, pr_number)
            for f in files:
                filename = f.get("filename", f.get("path", ""))
                patch = f.get("patch", "")
                if not filename or not patch:
                    continue

                # Extraire les lignes supprimées du patch (lignes "-")
                # et les lignes de contexte (lignes " ")
                removed_lines = []
                for line in patch.splitlines():
                    if line.startswith("-") and not line.startswith("---"):
                        removed_lines.append(line[1:].strip())

                if not removed_lines:
                    continue

                # Récupérer le contenu actuel du fichier sur main
                base_content = await self.get_file_content(
                    owner, repo, filename, base_ref
                )
                if not base_content:
                    continue

                # Vérifier : les lignes supprimées par le patch existent-elles
                # encore sur main ? Si NON → main a aussi été modifié → conflit
                base_content_stripped = base_content.replace(" ", "").replace("\t", "")
                missing_count = 0
                for removed in removed_lines:
                    removed_stripped = removed.replace(" ", "").replace("\t", "")
                    if removed_stripped and removed_stripped not in base_content_stripped:
                        missing_count += 1

                # Si >30% des lignes supprimées ne sont plus dans main,
                # main a divergé significativement → conflit probable
                if removed_lines and missing_count / len(removed_lines) > 0.3:
                    conflict_files.append(filename)
                    sys.stderr.write(
                        f"[MCP] Stratégie 4 : {filename} — "
                        f"{missing_count}/{len(removed_lines)} lignes divergent\n"
                    )
                    sys.stderr.flush()
        except Exception as e:
            logger.debug("Stratégie 4: %s", e)

        has_conflicts = len(conflict_files) > 0
        sys.stderr.write(
            f"[MCP] Stratégie 4 résultat: has_conflicts={has_conflicts}, files={conflict_files}\n"
        )
        sys.stderr.flush()

        return {
            "has_conflicts": has_conflicts,
            "mergeable": not has_conflicts if conflict_files else None,
            "conflict_files": conflict_files,
            "base_ref": base_ref,
            "head_ref": head_ref,
            "head_sha": head_sha,
            "pr_data": pr_data,
        }

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """Liste les fichiers d'une PR via MCP (outil `get_pull_request_files`,
        paramètre `pull_number` en snake_case). REST direct en fallback
        uniquement si MCP échoue vraiment (serveur down, outil absent)."""
        # ── Stratégie MCP (prioritaire) ──────────────────────────────────────
        if "get_pull_request_files" in self._available_tools:
            # pull_number (snake_case) est le nom attendu par le serveur MCP
            # officiel ; pullNumber est tenté en second pour compat anciens forks.
            for key in ["pull_number", "pullNumber"]:
                try:
                    result = await self._call_tool("get_pull_request_files", {
                        "owner": owner, "repo": repo, key: pr_number,
                    })
                    if isinstance(result, list) and result:
                        return result
                except Exception:
                    continue
        if self.has_tool("list_pr_files"):
            for key in ["pull_number", "pullNumber"]:
                try:
                    result = await self._call_alias("list_pr_files", {
                        "owner": owner, "repo": repo, key: pr_number,
                    })
                    if isinstance(result, list) and result:
                        return result
                except Exception:
                    continue

        # ── Fallback REST : seulement si MCP n'a rien renvoyé ────────────────
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if token:
            import urllib.request
            api_url = (
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
                f"?per_page=100"
            )
            headers_rest = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "code-auditor/1.0",
            }
            try:
                req = urllib.request.Request(api_url, headers=headers_rest)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    rest_files = json.loads(resp.read().decode())
                if isinstance(rest_files, list) and rest_files:
                    sys.stderr.write(
                        f"[REST] get_pr_files fallback : {len(rest_files)} fichier(s)\n"
                    )
                    sys.stderr.flush()
                    return rest_files
            except Exception as e:
                sys.stderr.write(f"[REST] get_pr_files erreur: {e}\n")
                sys.stderr.flush()

        try:
            pr = await self.get_pull_request(owner, repo, pr_number)
            if isinstance(pr, dict):
                files = pr.get("files", pr.get("changed_files_list", []))
                if isinstance(files, list) and files:
                    return files
        except Exception:
            pass
        return []

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str = "main",
        max_chars: int = 8000
    ) -> str:
        """
        Lit le contenu d'un fichier GitHub via MCP (outil `get_file_contents`).
        REST direct en fallback uniquement si MCP échoue vraiment.

        max_chars : limite de caractères (LLM budget). 0 ou négatif = illimité
        (utilisé par la résolution de conflits qui a besoin du fichier ENTIER —
        une troncature corromprait le merge).

        FIX v7.2 — Détection contenu déjà décodé :
          Le serveur MCP npm retourne parfois encoding='base64' MAIS le contenu
          est déjà du texte clair (pas du vrai base64). Si on tente b64decode()
          sur du texte Java, on obtient du garbage binaire.
        """
        def _cap(text: str) -> str:
            return text[:max_chars] if (max_chars and max_chars > 0) else text

        # ── Stratégie MCP (prioritaire) ──────────────────────────────────────
        for args in [
            {"owner": owner, "repo": repo, "path": path, "branch": ref},
            {"owner": owner, "repo": repo, "path": path, "ref": ref},
        ]:
            try:
                result = await self._call_alias("get_file", args)
                if isinstance(result, dict):
                    content = result.get("content", "")
                    encoding = result.get("encoding", "")
                    if encoding == "base64" and content:
                        # FIX v7.2 : Le MCP retourne parfois du texte clair
                        # avec encoding='base64'. Détecter ce cas AVANT de décoder.
                        if self._looks_like_source_code(content):
                            # Le contenu est déjà du texte — ne PAS décoder
                            logger.debug(
                                "get_file_content: contenu déjà en texte clair pour %s@%s (skip b64decode)",
                                path, ref,
                            )
                        else:
                            # Vrai base64 — décoder normalement
                            import base64
                            try:
                                raw_bytes = base64.b64decode(content.replace("\n", ""))
                                decoded = None
                                for enc in ("utf-8", "latin-1", "cp1252"):
                                    try:
                                        decoded = raw_bytes.decode(enc)
                                        break
                                    except (UnicodeDecodeError, ValueError):
                                        continue
                                if decoded is None:
                                    decoded = raw_bytes.decode("utf-8", errors="replace")
                                content = decoded
                            except Exception:
                                # base64 decode échoué — garder le contenu tel quel
                                pass
                    content = _cap(content or "")
                    # Validation : rejeter le contenu binaire (NUL bytes)
                    if content and "\x00" in content:
                        logger.warning(
                            "get_file_content: contenu binaire (NUL bytes) pour %s@%s — ignoré",
                            path, ref,
                        )
                        return ""
                    return content
                if isinstance(result, str) and result:
                    return _cap(result)
            except Exception:
                continue

        # ── Fallback REST : seulement si MCP n'a rien renvoyé ────────────────
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        if token:
            import urllib.request
            import urllib.parse
            import base64 as _b64
            api_url = (
                f"https://api.github.com/repos/{owner}/{repo}/contents/"
                f"{urllib.parse.quote(path, safe='/')}"
                f"?ref={urllib.parse.quote(ref, safe='')}"
            )
            headers_rest = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "code-auditor/1.0",
            }
            try:
                req = urllib.request.Request(api_url, headers=headers_rest)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                raw_b64 = data.get("content", "") if isinstance(data, dict) else ""
                if raw_b64:
                    raw_bytes = _b64.b64decode(raw_b64.replace("\n", ""))
                    text = None
                    for enc in ("utf-8", "latin-1", "cp1252"):
                        try:
                            text = raw_bytes.decode(enc)
                            break
                        except (UnicodeDecodeError, ValueError):
                            continue
                    if text is None:
                        text = raw_bytes.decode("utf-8", errors="replace")
                    if "\x00" in text[:1000]:
                        sys.stderr.write(f"[REST] get_file_content: binaire ignoré {path}@{ref}\n")
                        sys.stderr.flush()
                        return ""
                    return _cap(text)
            except Exception as e:
                sys.stderr.write(f"[REST] get_file_content erreur {path}@{ref}: {e}\n")
                sys.stderr.flush()
        return ""

    @staticmethod
    def _looks_like_source_code(content: str) -> bool:
        """
        Heuristique rapide : le contenu ressemble-t-il à du code source ?
        Utilisé pour détecter quand le MCP retourne du texte clair
        alors qu'il annonce encoding='base64'.

        Vérifie les 500 premiers caractères pour des indicateurs courants :
        - Mots-clés de langage (import, package, class, def, function, const)
        - Caractères non-base64 (espaces, points-virgules, accolades)
        - Ratio de caractères imprimables > 90%
        """
        if not content or len(content) < 10:
            return False
        sample = content[:500]
        # Base64 ne contient PAS ces caractères
        code_indicators = (";", "{", "}", "(", ")", "import ", "package ", "class ",
                           "def ", "function ", "const ", "public ", "private ")
        if any(ind in sample for ind in code_indicators):
            return True
        # Vérifier le ratio imprimable + whitespace
        printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
        ratio = printable / len(sample)
        # Du vrai base64 a un ratio printable ~100% aussi, mais sans espaces/newlines
        # Du code source a toujours des newlines et des espaces
        has_newlines = "\n" in sample
        has_spaces = " " in sample
        return ratio > 0.90 and has_newlines and has_spaces

    async def post_pr_comment(self, owner: str, repo: str, pr_number: int, body: str) -> Dict[str, Any]:
        for alias in ["post_comment", "post_comment_npm"]:
            if self.has_tool(alias):
                try:
                    result = await self._call_alias(alias, {
                        "owner": owner, "repo": repo,
                        "issue_number": pr_number, "body": body,
                    })
                    if result:
                        return result if isinstance(result, dict) else {}
                except Exception:
                    continue
        return {}

    async def create_or_update_file(
        self, owner: str, repo: str, path: str,
        content: str, message: str, branch: str,
        sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ── Stratégie 0 : REST PUT /contents (fiable) ────────────────────────
        try:
            import base64 as _b64
            import urllib.parse
            url = (
                f"https://api.github.com/repos/{owner}/{repo}/contents/"
                f"{urllib.parse.quote(path, safe='/')}"
            )
            # Récupérer le SHA existant sur la branche (requis pour un update)
            if not sha:
                try:
                    existing = self._rest_api(
                        "GET", f"{url}?ref={urllib.parse.quote(branch, safe='')}"
                    )
                    if isinstance(existing, dict):
                        sha = existing.get("sha")
                except Exception:
                    sha = None  # le fichier n'existe pas encore → création
            payload = {
                "message": message,
                "content": _b64.b64encode(content.encode("utf-8")).decode("ascii"),
                "branch":  branch,
            }
            if sha:
                payload["sha"] = sha
            result = self._rest_api("PUT", url, payload)
            sys.stderr.write(f"[REST] push_file OK: {path}@{branch}\n")
            sys.stderr.flush()
            return result if isinstance(result, dict) else {}
        except Exception as e:
            sys.stderr.write(f"[REST] push_file erreur {path}@{branch}: {e}\n")
            sys.stderr.flush()

        # ── Fallback MCP ─────────────────────────────────────────────────────
        if "push_files" in self._available_tools:
            try:
                result = await self._call_tool("push_files", {
                    "owner": owner, "repo": repo, "branch": branch,
                    "message": message,
                    "files": [{"path": path, "content": content}],
                })
                if result:
                    return result if isinstance(result, dict) else {}
            except Exception:
                pass
        args = {
            "owner": owner, "repo": repo, "path": path,
            "content": content, "message": message, "branch": branch,
        }
        if sha:
            args["sha"] = sha
        try:
            return await self._call_alias("create_file", args) or {}
        except Exception as e:
            sys.stderr.write(f"[MCP] create_file erreur {path}: {e}\n")
            sys.stderr.flush()
            return {}

    async def create_branch(self, owner: str, repo: str, branch_name: str, from_ref: str = "main") -> Dict[str, Any]:
        # ── Stratégie 0 : REST (fiable) ──────────────────────────────────────
        try:
            import urllib.parse
            import urllib.error as _ue
            # from_ref peut être un nom de branche OU un SHA. Résoudre le SHA de base.
            is_sha = len(from_ref) == 40 and all(c in "0123456789abcdef" for c in from_ref.lower())
            base_sha = from_ref
            if not is_sha:
                ref_data = self._rest_api(
                    "GET",
                    f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/"
                    f"{urllib.parse.quote(from_ref, safe='')}",
                )
                base_sha = ref_data.get("object", {}).get("sha", from_ref)
            try:
                result = self._rest_api(
                    "POST",
                    f"https://api.github.com/repos/{owner}/{repo}/git/refs",
                    {"ref": f"refs/heads/{branch_name}", "sha": base_sha},
                )
                sys.stderr.write(f"[REST] create_branch OK: {branch_name}\n")
                sys.stderr.flush()
                return result if isinstance(result, dict) else {}
            except _ue.HTTPError as he:
                if he.code == 422:  # la branche existe déjà → réutilisable
                    sys.stderr.write(f"[REST] create_branch: {branch_name} existe déjà (OK)\n")
                    sys.stderr.flush()
                    return {"ref": f"refs/heads/{branch_name}", "exists": True}
                raise
        except Exception as e:
            sys.stderr.write(f"[REST] create_branch erreur: {e}\n")
            sys.stderr.flush()

        # ── Fallback MCP ─────────────────────────────────────────────────────
        for args in [
            {"owner": owner, "repo": repo, "branch": branch_name, "from_branch": from_ref},
            {"owner": owner, "repo": repo, "branch": branch_name, "sha": from_ref},
        ]:
            try:
                result = await self._call_alias("create_branch", args)
                if result:
                    return result if isinstance(result, dict) else {}
            except Exception:
                continue
        return {}

    async def create_pull_request(
        self, owner: str, repo: str,
        title: str, body: str, head: str, base: str
    ) -> Dict[str, Any]:
        # ── Stratégie 0 : REST POST /pulls (fiable) ──────────────────────────
        try:
            import urllib.error as _ue
            try:
                result = self._rest_api(
                    "POST",
                    f"https://api.github.com/repos/{owner}/{repo}/pulls",
                    {"title": title, "body": body, "head": head, "base": base},
                )
                if isinstance(result, dict) and result.get("html_url"):
                    sys.stderr.write(f"[REST] create_pull_request OK: {result.get('html_url')}\n")
                    sys.stderr.flush()
                    return result
            except _ue.HTTPError as he:
                # 422 : une PR existe déjà pour ce head/base → la retrouver
                if he.code == 422:
                    sys.stderr.write(f"[REST] create_pull_request: PR existe déjà pour {head}→{base}\n")
                    sys.stderr.flush()
                    try:
                        existing = self._rest_api(
                            "GET",
                            f"https://api.github.com/repos/{owner}/{repo}/pulls"
                            f"?head={owner}:{head}&base={base}&state=open",
                        )
                        if isinstance(existing, list) and existing:
                            return existing[0]
                    except Exception:
                        pass
                else:
                    raise
        except Exception as e:
            sys.stderr.write(f"[REST] create_pull_request erreur: {e}\n")
            sys.stderr.flush()

        # ── Fallback MCP ─────────────────────────────────────────────────────
        if "create_pull_request" not in self._available_tools:
            return {}
        try:
            result = await self._call_tool("create_pull_request", {
                "owner": owner, "repo": repo,
                "title": title, "body": body,
                "head": head, "base": base,
            })
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.debug("create_pull_request: %s", e)
            return {}

    def _rest_create_review(
        self, owner: str, repo: str, pr_number: int,
        body: str, event: str, comments: List[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Soumet un review via l'API REST GitHub directe — contrôle total du format
        des commentaires inline ({path, line, side, start_line, start_side, body}).

        Gère le cas 422 : GitHub interdit APPROVE/REQUEST_CHANGES sur sa PROPRE PR.
        On replie alors sur l'event COMMENT, ce qui poste quand même les
        commentaires inline + le corps du review.

        Retourne le dict review en cas de succès, None sinon (→ repli MCP).
        """
        token = (os.getenv("GITHUB_TOKEN")
                 or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
                 or _get_github_token())
        if not token:
            return None

        import urllib.request as _ur
        import urllib.error as _ue

        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "code-auditor/1.0",
            "Content-Type": "application/json",
        }

        def _post(evt: str) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"body": body or "", "event": evt}
            if comments:
                payload["comments"] = comments
            req = _ur.Request(
                api_url, data=json.dumps(payload).encode("utf-8"),
                headers=headers, method="POST",
            )
            with _ur.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())

        try:
            return _post(event)
        except _ue.HTTPError as e:
            # 422 : impossible d'APPROVE/REQUEST_CHANGES sa propre PR → repli COMMENT
            if e.code == 422 and event in ("APPROVE", "REQUEST_CHANGES"):
                try:
                    logger.debug("create_review 422 (own PR) → retry as COMMENT")
                    return _post("COMMENT")
                except Exception as e2:
                    logger.debug("REST create_review COMMENT retry: %s", e2)
                    return None
            logger.debug("REST create_review HTTPError %s", e.code)
            return None
        except Exception as e:
            logger.debug("REST create_review: %s", e)
            return None

    async def create_pr_review(
        self, owner: str, repo: str, pr_number: int,
        body: str, event: str, comments: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Avec commentaires inline : REST d'abord (format fiable + gestion du 422
        # own-PR). Le serveur MCP gère mal le schéma inline {path,line,side}.
        if comments:
            rest = self._rest_create_review(owner, repo, pr_number, body, event, comments)
            if isinstance(rest, dict) and rest.get("id"):
                return rest

        for key in ["pullNumber", "pull_number"]:
            args = {"owner": owner, "repo": repo, key: pr_number, "body": body, "event": event}
            if comments:
                args["comments"] = comments
            try:
                result = await self._call_alias("create_review", args)
                if result:
                    return result if isinstance(result, dict) else {}
            except Exception:
                continue

        # Repli REST sans inline (gère aussi le 422 own-PR pour un review simple)
        rest2 = self._rest_create_review(owner, repo, pr_number, body, event, None)
        if isinstance(rest2, dict) and rest2.get("id"):
            return rest2

        return await self.post_pr_comment(owner, repo, pr_number, f"[{event}]\n\n{body[:2000]}")

    async def get_pr_reviews(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        for alias in ["list_reviews", "list_reviews_npm"]:
            if self.has_tool(alias):
                for key in ["pullNumber", "pull_number"]:
                    try:
                        result = await self._call_alias(alias, {
                            "owner": owner, "repo": repo, key: pr_number,
                        })
                        if isinstance(result, list):
                            return result
                    except Exception:
                        continue
        return []

    async def get_check_runs(self, owner: str, repo: str, ref: str) -> List[Dict[str, Any]]:
        if not self.has_tool("list_checks"):
            return []
        try:
            result = await self._call_alias("list_checks", {
                "owner": owner, "repo": repo, "ref": ref,
            })
            if isinstance(result, dict):
                return result.get("check_runs", [])
            return result if isinstance(result, list) else []
        except Exception:
            return []

    async def search_code(self, query: str) -> List[Dict[str, Any]]:
        if not self.has_tool("search_code"):
            return []
        result = await self._call_alias("search_code", {"q": query})
        if isinstance(result, dict):
            return result.get("items", [])
        return []

    async def list_open_prs(
        self, owner: str, repo: str, base: str = "", per_page: int = 30
    ) -> List[Dict[str, Any]]:
        """Liste les PRs ouvertes sur un dépôt (optionnellement filtrées par base branch)."""
        params: Dict[str, Any] = {"owner": owner, "repo": repo, "state": "open"}
        if base:
            params["base"] = base

        for alias in ["list_pull_requests", "list_pulls", "list_prs"]:
            if self.has_tool(alias):
                for key in ["pullNumber", "pull_number", "number"]:
                    try:
                        result = await self._call_alias(alias, params)
                        if isinstance(result, list):
                            return result
                        if isinstance(result, dict):
                            return result.get("pull_requests", result.get("items", []))
                    except Exception:
                        break

        # Fallback : search PRs via search_code or REST
        try:
            result = await self._call_tool("search_issues", {
                "owner": owner, "repo": repo,
                "q": f"repo:{owner}/{repo} is:pr is:open",
            })
            if isinstance(result, dict):
                return result.get("items", [])
        except Exception:
            pass

        return []

    async def get_pr_commits(
        self, owner: str, repo: str, pr_number: int
    ) -> List[Dict[str, Any]]:
        """Liste les commits d'une PR (REST-first, MCP en fallback)."""
        # ── Stratégie 0 : REST direct ────────────────────────────────────────
        try:
            data = self._rest_api(
                "GET",
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits?per_page=100",
            )
            if isinstance(data, list) and data:
                sys.stderr.write(f"[REST] get_pr_commits : {len(data)} commit(s) via REST\n")
                sys.stderr.flush()
                return data
        except Exception as e:
            sys.stderr.write(f"[REST] get_pr_commits erreur: {e}\n")
            sys.stderr.flush()

        # ── Fallback MCP ─────────────────────────────────────────────────────
        for alias in ["list_pr_commits", "get_pull_request_commits", "list_commits"]:
            if self.has_tool(alias):
                for key in ["pullNumber", "pull_number"]:
                    try:
                        result = await self._call_alias(alias, {
                            "owner": owner, "repo": repo, key: pr_number,
                        })
                        if isinstance(result, list) and result:
                            return result
                    except Exception:
                        continue
        return []

    async def list_available_tools(self) -> List[str]:
        return sorted(self._available_tools)

    def get_tool_mapping(self) -> Dict[str, str]:
        return dict(self._tool_map)