"""
mcp_sonarqube_service.py — Client MCP SonarCloud.

Connecte le Code Auditor au SonarSource/sonarqube-mcp-server (officiel)
via Docker (mcp/sonarqube) en utilisant le même patron que MCPGitHubService.

Configuration SonarCloud (env vars) :
  SONARQUBE_TOKEN : User token SonarCloud (requis)
  SONARQUBE_ORG   : Clé de l'organisation SonarCloud (requis pour Cloud)

Usage :
    sonar = get_mcp_sonarqube()
    gate = sonar.get_quality_gate_status("chmaryem_myapp")
    # → {"status": "OK"|"ERROR"|"WARN", "conditions": [...]}

Fallback :
    Si Docker n'est pas disponible, utilise la REST API SonarCloud directement
    via urllib (même pattern que _push_via_rest dans ci_deploy_agent.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SONARCLOUD_URL = "https://sonarcloud.io"
SONARCLOUD_API = "https://sonarcloud.io/api"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_sonar_token() -> str:
    token = os.environ.get("SONARQUBE_TOKEN", "")
    if not token:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("SONARQUBE_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    return token


def _get_sonar_org() -> str:
    org = os.environ.get("SONARQUBE_ORG", "")
    if not org:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("SONARQUBE_ORG="):
                    org = line.split("=", 1)[1].strip()
                    break
    return org


# ── Loop Manager (identique à MCPGitHubService) ───────────────────────────────

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
                    target=self._run_loop, daemon=True, name="mcp-sonar-loop"
                )
                self._thread.start()
                self._ready.wait(timeout=5)
        return self._loop

    def run(self, coro, timeout: int = 60):
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


# ── MCP SonarQube Service ─────────────────────────────────────────────────────

class MCPSonarQubeService:
    """
    Client MCP pour SonarCloud.

    Stratégie :
      1. Essaie de se connecter via Docker mcp/sonarqube (MCP officiel)
      2. Fallback : REST API SonarCloud directe (urllib, sans Docker)

    Usage synchrone depuis du code non-async :
        sonar = MCPSonarQubeService()
        gate = sonar.get_quality_gate_status("my_project")
    """

    def __init__(
        self,
        token: Optional[str] = None,
        org: Optional[str] = None,
    ):
        self._token = token or _get_sonar_token()
        self._org = org or _get_sonar_org()
        self._session = None
        self._read = None
        self._write = None
        self._cm = None
        self._available_tools: set = set()
        self._mcp_available = False

    # ── MCP Connection ────────────────────────────────────────────────────────

    async def _connect_mcp(self) -> bool:
        """
        Tente de se connecter via Docker mcp/sonarqube.
        Retourne True si succès, False si Docker non disponible.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            env = dict(os.environ)
            env["SONARQUBE_TOKEN"] = self._token
            if self._org:
                env["SONARQUBE_ORG"] = self._org

            server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run", "--init", "--pull=always", "-i", "--rm",
                    "-e", "SONARQUBE_TOKEN",
                    "-e", "SONARQUBE_ORG",
                    "mcp/sonarqube",
                ],
                env=env,
            )
            self._cm = stdio_client(server_params)
            self._read, self._write = await self._cm.__aenter__()
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
            await self._session.initialize()

            # Découvrir les tools disponibles
            tools_result = await self._session.list_tools()
            self._available_tools = {t.name for t in tools_result.tools}
            sys.stderr.write(
                f"[MCP Sonar] {len(self._available_tools)} tools connectés\n"
            )
            sys.stderr.flush()
            self._mcp_available = True
            return True

        except Exception as e:
            logger.debug("MCPSonarQube Docker non disponible: %s — fallback REST", e)
            self._mcp_available = False
            return False

    async def _disconnect_mcp(self) -> None:
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Appelle un tool MCP et retourne le résultat parsé."""
        if not self._session:
            raise RuntimeError("MCP Sonar non connecté.")
        result = await self._session.call_tool(tool_name, arguments)
        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        return json.loads(block.text)
                    except json.JSONDecodeError:
                        return block.text
        return None

    # ── REST Fallback ─────────────────────────────────────────────────────────

    def _rest_call(self, endpoint: str, params: Dict[str, str] = None) -> Any:
        """
        Appelle l'API REST SonarCloud directement.
        Utilisé comme fallback si Docker non disponible.
        """
        import urllib.request
        import urllib.parse

        if not self._token:
            logger.error("SONARQUBE_TOKEN manquant — impossible d'appeler SonarCloud API")
            return None

        url = f"{SONARCLOUD_API}/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url)
        # SonarCloud auth : Bearer token
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error("SonarCloud REST %s: %s", endpoint, e)
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_quality_gate_status(self, project_key: str) -> Dict[str, Any]:
        """
        Retourne le statut du Quality Gate pour un projet.

        Returns:
            {
                "status": "OK" | "ERROR" | "WARN" | "NONE",
                "conditions": [
                    {"metric": "coverage", "status": "ERROR",
                     "actualValue": "65.3", "errorThreshold": "70"}
                ]
            }
        """
        # Essai MCP (si connecté)
        if self._mcp_available and self._session:
            try:
                # Tool MCP : get_quality_gate_status ou équivalent
                for tool_name in ["get_quality_gate_status", "qualityGates_project_status"]:
                    if tool_name in self._available_tools:
                        result = _loop_manager.run(
                            self._call_tool(tool_name, {"projectKey": project_key})
                        )
                        if result:
                            return self._normalize_gate(result)
            except Exception as e:
                logger.debug("MCP quality gate failed: %s", e)

        # Fallback REST
        data = self._rest_call(
            "qualitygates/project_status",
            {"projectKey": project_key}
        )
        if data and "projectStatus" in data:
            ps = data["projectStatus"]
            return {
                "status": ps.get("status", "NONE"),
                "conditions": ps.get("conditions", []),
            }
        return {"status": "NONE", "conditions": []}

    def get_project_metrics(
        self,
        project_key: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retourne les métriques clés d'un projet SonarCloud.

        Args:
            project_key: Clé du projet (ex: "chmaryem_myapp")
            metrics: Liste de métriques. Par défaut : coverage, bugs, vulnerabilities, etc.

        Returns:
            Dict metric_key → value
        """
        default_metrics = [
            "coverage",
            "bugs",
            "vulnerabilities",
            "code_smells",
            "duplicated_lines_density",
            "reliability_rating",
            "security_rating",
            "sqale_rating",
            "ncloc",
        ]
        metric_keys = metrics or default_metrics

        # Fallback REST (le plus fiable)
        import urllib.parse
        data = self._rest_call(
            "measures/component",
            {
                "component": project_key,
                "metricKeys": ",".join(metric_keys),
            }
        )
        if data and "component" in data:
            result = {}
            for measure in data["component"].get("measures", []):
                result[measure["metric"]] = measure.get("value", "N/A")
            return result
        return {}

    def get_issues(
        self,
        project_key: str,
        severity: str = "CRITICAL",
        issue_type: str = "VULNERABILITY",
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les issues filtrées par sévérité et type.

        Args:
            project_key: Clé du projet SonarCloud
            severity: BLOCKER | CRITICAL | MAJOR | MINOR | INFO
            issue_type: BUG | VULNERABILITY | CODE_SMELL | SECURITY_HOTSPOT
            max_results: Nombre max d'issues à retourner

        Returns:
            Liste d'issues avec message, component, severity, rule
        """
        data = self._rest_call(
            "issues/search",
            {
                "componentKeys": project_key,
                "severities": severity,
                "types": issue_type,
                "ps": str(max_results),
                "p": "1",
            }
        )
        if data and "issues" in data:
            return [
                {
                    "key": i.get("key", ""),
                    "message": i.get("message", ""),
                    "severity": i.get("severity", ""),
                    "rule": i.get("rule", ""),
                    "component": i.get("component", ""),
                    "line": i.get("line"),
                    "effort": i.get("effort", ""),
                }
                for i in data["issues"]
            ]
        return []

    def analyze_code_snippet(
        self,
        code: str,
        language: str,
        project_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyse un snippet de code directement via le MCP Server.
        Uniquement disponible si Docker mcp/sonarqube est connecté.

        Returns:
            {"issues": [...], "metrics": {...}} ou {} si MCP non disponible
        """
        if not self._mcp_available or not self._session:
            logger.debug("analyze_code_snippet: MCP non disponible, skip")
            return {}

        try:
            for tool_name in ["analyze_code", "analyzeCode", "analyze_code_snippet"]:
                if tool_name in self._available_tools:
                    result = _loop_manager.run(
                        self._call_tool(tool_name, {
                            "code": code,
                            "language": language,
                            **({"projectKey": project_key} if project_key else {}),
                        })
                    )
                    if result:
                        return result if isinstance(result, dict) else {"raw": result}
        except Exception as e:
            logger.debug("analyze_code_snippet MCP failed: %s", e)

        return {}

    def get_new_issues_on_pr(
        self,
        project_key: str,
        pull_request: str,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les nouvelles issues introduites par une PR spécifique.

        Args:
            project_key: Clé du projet
            pull_request: Numéro de PR (ex: "42")
        """
        data = self._rest_call(
            "issues/search",
            {
                "componentKeys": project_key,
                "pullRequest": pull_request,
                "resolved": "false",
                "ps": "50",
            }
        )
        if data and "issues" in data:
            return data["issues"]
        return []

    def disconnect(self) -> None:
        """Déconnecte le MCP server (libère le processus Docker)."""
        if self._mcp_available:
            try:
                _loop_manager.run(self._disconnect_mcp(), timeout=10)
            except Exception:
                pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_gate(raw: Any) -> Dict[str, Any]:
        """Normalise la réponse brute du MCP en format standard."""
        if isinstance(raw, dict):
            # Format direct
            if "status" in raw:
                return {
                    "status": raw["status"],
                    "conditions": raw.get("conditions", []),
                }
            # Format imbriqué projectStatus
            if "projectStatus" in raw:
                ps = raw["projectStatus"]
                return {
                    "status": ps.get("status", "NONE"),
                    "conditions": ps.get("conditions", []),
                }
        return {"status": "NONE", "conditions": []}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()


# ── Singleton ─────────────────────────────────────────────────────────────────

_sonar_instance: Optional[MCPSonarQubeService] = None
_sonar_lock = threading.Lock()


def get_mcp_sonarqube() -> MCPSonarQubeService:
    """
    Retourne le singleton MCPSonarQubeService.

    Lit SONARQUBE_TOKEN et SONARQUBE_ORG depuis les variables d'environnement
    (ou .env). Tente d'abord Docker MCP, fallback REST si Docker absent.
    """
    global _sonar_instance
    with _sonar_lock:
        if _sonar_instance is None:
            _sonar_instance = MCPSonarQubeService()
        return _sonar_instance
