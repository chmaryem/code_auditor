"""
mcp_github_service.py — Client MCP GitHub.

FIX v5 — Détection fiable des conflits :

  PROBLÈME RACINE : get_pull_request() via le serveur MCP npm retourne le champ
  'mergeable' dans un objet imbriqué parfois, ou None systématiquement car le
  serveur normalise la réponse GitHub différemment.

  SOLUTION :
    1. Utiliser get_pull_request_status (outil dédié présent dans les 26 tools)
       qui retourne mergeableState: "dirty" | "clean" | "unknown"
       "dirty" = conflits confirmés, sans ambiguïté.
    2. Si get_pull_request_status indisponible → poll get_pull_request jusqu'à
       4 fois pour obtenir mergeable != null.
    3. Si toujours null → comparer les listes de fichiers modifiés entre base et
       head via l'API (pas les contenus — ça coûte trop de tokens).

  FIX quota LLM :
    - Toutes les fonctions get_file_content() limitent le retour à 8000 chars
      max pour éviter l'explosion du contexte Gemini.
"""

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

    # ── Pull Requests ─────────────────────────────────────────────────────────

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        # FIX v6.1 : pull_number en premier (le serveur MCP npm l'attend)
        for key in ["pull_number", "pullNumber"]:
            try:
                result = await self._call_alias("get_pr", {
                    "owner": owner, "repo": repo, key: pr_number,
                })
                if result and isinstance(result, dict):
                    return result
            except Exception:
                continue
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
            for attempt in range(3):
                try:
                    req = urllib.request.Request(api_url, headers=headers_rest)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        rest_pr = json.loads(resp.read().decode())

                    rest_mergeable = rest_pr.get("mergeable")
                    rest_state = rest_pr.get("mergeable_state", "")
                    sys.stderr.write(
                        f"[REST] Stratégie 0 : mergeable={rest_mergeable!r} "
                        f"state={rest_state!r} (attempt {attempt+1})\n"
                    )
                    sys.stderr.flush()

                    if rest_mergeable is not None:
                        pr_data = rest_pr
                        base_ref = rest_pr.get("base", {}).get("ref", "main")
                        head_ref = rest_pr.get("head", {}).get("ref", "")
                        head_sha = rest_pr.get("head", {}).get("sha", "")
                        return {
                            "has_conflicts": not rest_mergeable,
                            "mergeable": rest_mergeable,
                            "conflict_files": [],
                            "base_ref": base_ref,
                            "head_ref": head_ref,
                            "head_sha": head_sha,
                            "pr_data": rest_pr,
                        }
                    # mergeable=None → GitHub calcule encore, attendre
                    await asyncio.sleep(3)
                except Exception as e:
                    sys.stderr.write(f"[REST] Stratégie 0 erreur: {e}\n")
                    sys.stderr.flush()
                    break  # REST inaccessible, passer aux stratégies MCP

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
        # FIX v6.1 : 6 tentatives (au lieu de 4) avec 5s de délai (au lieu de 3s).
        # GitHub met parfois 15-20s pour calculer le champ mergeable après un push.
        # Premier appel = "kick" pour déclencher le calcul côté GitHub.
        max_polls = 6
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
        """Multi-strategy fallback pour lister les fichiers d'une PR."""
        if "get_pull_request_files" in self._available_tools:
            for key in ["pullNumber", "pull_number"]:
                try:
                    result = await self._call_tool("get_pull_request_files", {
                        "owner": owner, "repo": repo, key: pr_number,
                    })
                    if isinstance(result, list) and result:
                        return result
                except Exception:
                    continue
        if self.has_tool("list_pr_files"):
            for key in ["pullNumber", "pull_number"]:
                try:
                    result = await self._call_alias("list_pr_files", {
                        "owner": owner, "repo": repo, key: pr_number,
                    })
                    if isinstance(result, list) and result:
                        return result
                except Exception:
                    continue
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
        Lit le contenu d'un fichier GitHub.
        max_chars=8000 par défaut pour limiter la consommation de tokens Gemini.
        """
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
                        import base64
                        try:
                            content = base64.b64decode(
                                content.replace("\n", "")
                            ).decode("utf-8", errors="replace")
                        except Exception:
                            pass
                    return (content or "")[:max_chars]
                if isinstance(result, str) and result:
                    return result[:max_chars]
            except Exception:
                continue
        return ""

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
        return await self._call_alias("create_file", args) or {}

    async def create_branch(self, owner: str, repo: str, branch_name: str, from_ref: str = "main") -> Dict[str, Any]:
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

    async def create_pr_review(
        self, owner: str, repo: str, pr_number: int,
        body: str, event: str, comments: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

    async def list_available_tools(self) -> List[str]:
        return sorted(self._available_tools)

    def get_tool_mapping(self) -> Dict[str, str]:
        return dict(self._tool_map)