"""
mcp_redis_service.py — Client MCP Redis (séparé de MCP GitHub).

Architecture :
  - _RedisLoopManager : event loop async dédié (séparé de celui de MCP GitHub)
  - MCPRedisService   : client MCP avec méthodes typées pour chaque opération Redis
  - mcp_redis          : singleton global importable depuis n'importe quel module

Le serveur utilisé est le MCP officiel redis/mcp-redis (Python, via pip).
Il expose 47 tools : hset/hget/hgetall, zadd/zrange, json_set/json_get,
set/get, scan_all_keys, etc.

Séparation MCP :
  - MCPGitHubService  → npx @modelcontextprotocol/server-github
  - MCPRedisService   → redis-mcp-server --url redis://localhost:6379/0
  Deux processus distincts, deux event loops, zéro couplage.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# ── Namespace prefix ─────────────────────────────────────────────────────────
# Toutes les clés Code Auditor sont préfixées pour éviter les collisions.
KEY_PREFIX = "ca:"


# ── Utility ──────────────────────────────────────────────────────────────────

def key_hash(value: str) -> str:
    """Hash SHA-256 tronqué (16 hex chars) pour clés Redis sûres."""
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


# ── Event Loop dédié ─────────────────────────────────────────────────────────

class _RedisLoopManager:
    """
    Event loop asyncio dédié au transport MCP Redis.
    Séparé de _LoopManager de mcp_github_service.py pour éviter
    toute interférence entre les deux connexions MCP.
    """

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
                    target=self._run_loop, daemon=True, name="mcp-redis-loop"
                )
                self._thread.start()
                self._ready.wait(timeout=5)
        return self._loop

    def run(self, coro, timeout: int = 30):
        """Exécute une coroutine depuis un thread synchrone."""
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


_redis_loop = _RedisLoopManager()


# ── Localisation du serveur MCP Redis ────────────────────────────────────────

def _find_redis_mcp_command() -> str:
    """
    Trouve le chemin de redis-mcp-server sur le système.
    Ordre de recherche :
      1. PATH (shutil.which)
      2. APPDATA/Python/Python3XX/Scripts/ (pip install --user sur Windows)
      3. Scripts/ du même répertoire que sys.executable (venv)
    """
    cmd = shutil.which("redis-mcp-server")
    if cmd:
        return cmd

    # Pip user-site sur Windows
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        py_ver = f"Python{sys.version_info.major}{sys.version_info.minor}"
        user_script = Path(appdata) / "Python" / py_ver / "Scripts" / "redis-mcp-server.exe"
        if user_script.exists():
            return str(user_script)

    # Venv scripts
    scripts_dir = Path(sys.executable).parent
    for name in ("redis-mcp-server.exe", "redis-mcp-server"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)
        candidate = scripts_dir / "Scripts" / name
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "redis-mcp-server introuvable. "
        "Installez-le : pip install redis-mcp-server"
    )


# ── MCPRedisService ──────────────────────────────────────────────────────────

class MCPRedisService:
    """
    Client MCP pour le serveur officiel redis/mcp-redis.

    Usage :
        from services.mcp_redis_service import mcp_redis
        mcp_redis.set("my:key", "value")
        val = mcp_redis.get("my:key")
        mcp_redis.hset("my:hash", "field1", "val1")
        data = mcp_redis.hgetall("my:hash")

    Toutes les méthodes publiques sont synchrones (thread-safe).
    L'async est géré en interne via _RedisLoopManager.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._cm = None
        self._available_tools: Set[str] = set()
        self._lock = threading.Lock()
        self._connected = False

    # ── Connexion MCP ────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Connecte au serveur redis-mcp-server via stdio."""
        if self._connected and self._session:
            return

        cmd = _find_redis_mcp_command()
        server_params = StdioServerParameters(
            command=cmd,
            args=["--url", self._redis_url],
        )

        sys.stderr.write(f"[MCP-Redis] Connexion : {cmd} --url {self._redis_url}\n")
        sys.stderr.flush()

        self._cm = stdio_client(server_params)
        self._read, self._write = await self._cm.__aenter__()
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        await self._session.initialize()

        # Découvrir les tools disponibles
        tools_result = await self._session.list_tools()
        self._available_tools = {t.name for t in tools_result.tools}
        sys.stderr.write(f"[MCP-Redis] {len(self._available_tools)} tools connectés\n")
        sys.stderr.flush()

        self._connected = True

    def _ensure_connected(self) -> None:
        """Connexion lazy — appelée automatiquement avant chaque opération."""
        if not self._connected:
            with self._lock:
                if not self._connected:
                    _redis_loop.run(self._connect())

    async def _disconnect(self) -> None:
        """Ferme la connexion MCP proprement."""
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
        self._connected = False

    def disconnect(self) -> None:
        """Fermeture synchrone."""
        if self._connected:
            try:
                _redis_loop.run(self._disconnect())
            except Exception:
                pass

    # ── Appel de tool MCP ────────────────────────────────────────────────────

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Appelle un tool MCP Redis et retourne le résultat parsé.
        Gère automatiquement la reconnexion si la session est perdue.
        """
        if not self._session:
            await self._connect()

        if tool_name not in self._available_tools:
            raise ValueError(
                f"[MCP-Redis] Tool '{tool_name}' non disponible. "
                f"Outils disponibles : {sorted(self._available_tools)}"
            )

        try:
            result = await self._session.call_tool(tool_name, arguments)
        except Exception as e:
            # Tentative de reconnexion une fois
            logger.warning("MCP-Redis call_tool erreur, reconnexion : %s", e)
            self._connected = False
            await self._connect()
            result = await self._session.call_tool(tool_name, arguments)

        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    text = block.text
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return text
        return None

    def _run(self, coro) -> Any:
        """Wrapper synchrone pour les appels async."""
        self._ensure_connected()
        return _redis_loop.run(coro)

    # ── String operations ────────────────────────────────────────────────────

    def set(self, key: str, value: str, expire_seconds: Optional[int] = None) -> Any:
        """SET key value [EX seconds]"""
        args: Dict[str, Any] = {"key": key, "value": value}
        if expire_seconds is not None:
            args["expiration"] = expire_seconds
        return self._run(self._call_tool("set", args))

    def get(self, key: str) -> Optional[str]:
        """GET key → valeur ou None"""
        result = self._run(self._call_tool("get", {"key": key}))
        if result is None or result == "nil" or result == "(nil)":
            return None
        return str(result) if result is not None else None

    # ── Hash operations ──────────────────────────────────────────────────────

    # Workaround: redis-mcp-server silently drops hset values that start with
    # '[' or '{' (interpreted as JSON structures). We prefix them with 'J:'.
    _JSON_PREFIX = "J:"

    @classmethod
    def _escape_hval(cls, val: str) -> str:
        """Escape values that would be silently dropped by the MCP server."""
        if val and val[0] in ("[", "{"):
            return cls._JSON_PREFIX + val
        return val

    @classmethod
    def _unescape_hval(cls, val: str) -> str:
        """Reverse _escape_hval."""
        if val and val.startswith(cls._JSON_PREFIX):
            return val[len(cls._JSON_PREFIX):]
        return val

    def hset(self, name: str, key: str, value: Any) -> Any:
        """HSET name field value — un champ à la fois."""
        val = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        val = self._escape_hval(val)
        return self._run(self._call_tool("hset", {"name": name, "key": key, "value": val}))

    def hset_dict(self, name: str, mapping: Dict[str, Any],
                  expire_seconds: Optional[int] = None) -> None:
        """
        Écrit un dictionnaire complet dans un Hash Redis.
        Appelle hset() pour chaque champ (le serveur MCP n'accepte qu'un champ par appel).
        """
        for field, value in mapping.items():
            val = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            val = self._escape_hval(val)
            args: Dict[str, Any] = {"name": name, "key": field, "value": val}
            if expire_seconds is not None:
                args["expire_seconds"] = expire_seconds
            self._run(self._call_tool("hset", args))

    def hget(self, name: str, key: str) -> Optional[str]:
        """HGET name field → valeur ou None"""
        result = self._run(self._call_tool("hget", {"name": name, "key": key}))
        if result is None or result == "nil" or result == "(nil)":
            return None
        val = str(result) if result is not None else None
        return self._unescape_hval(val) if val else val

    def hgetall(self, name: str) -> Dict[str, str]:
        """HGETALL name → {field: value, ...}"""
        result = self._run(self._call_tool("hgetall", {"name": name}))
        if isinstance(result, dict):
            out = {}
            for k, v in result.items():
                if isinstance(v, str):
                    out[str(k)] = self._unescape_hval(v)
                else:
                    out[str(k)] = json.dumps(v, ensure_ascii=False)
            return out
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    out = {}
                    for k, v in parsed.items():
                        if isinstance(v, str):
                            out[str(k)] = self._unescape_hval(v)
                        else:
                            out[str(k)] = json.dumps(v, ensure_ascii=False)
                    return out
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def hdel(self, name: str, key: str) -> Any:
        """HDEL name field"""
        return self._run(self._call_tool("hdel", {"name": name, "key": key}))

    def hexists(self, name: str, key: str) -> bool:
        """HEXISTS name field → True/False"""
        result = self._run(self._call_tool("hexists", {"name": name, "key": key}))
        return bool(result) and result not in (0, "0", False, "false", "nil")

    # ── Sorted Set operations ────────────────────────────────────────────────

    def zadd(self, key: str, score: float, member: str) -> Any:
        """ZADD key score member — un membre à la fois."""
        return self._run(self._call_tool("zadd", {
            "key": key, "score": score, "member": member,
        }))

    @staticmethod
    def _parse_zrange_result(result: Any, with_scores: bool) -> List[Any]:
        """
        Parse le résultat de zrange du serveur redis-mcp-server.
        Le serveur retourne un format Python repr : "[('member', score), ...]"
        ou un JSON array, ou directement une list Python.
        """
        if result is None:
            return []
        if isinstance(result, list):
            # Déjà une liste — peut contenir des sous-listes ou tuples
            flat = []
            for item in result:
                if isinstance(item, str):
                    # Tenter de parser un repr Python (ex: "[('a',1.0), ('b',2.0)]")
                    try:
                        import ast
                        parsed = ast.literal_eval(item)
                        if isinstance(parsed, list):
                            flat.extend(parsed)
                            continue
                    except (ValueError, SyntaxError):
                        pass
                flat.append(item)
            if with_scores:
                # Retourne des tuples (member, score)
                return [(str(m), float(s)) for m, s in flat] if flat and isinstance(flat[0], (list, tuple)) else flat
            else:
                # Retourne uniquement les members
                if flat and isinstance(flat[0], (list, tuple)):
                    return [str(m) for m, _ in flat]
                return [str(m) for m in flat]
        if isinstance(result, str):
            try:
                import ast
                parsed = ast.literal_eval(result)
                if isinstance(parsed, list):
                    if with_scores:
                        return [(str(m), float(s)) for m, s in parsed]
                    return [str(m) for m, *_ in parsed]
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
        return []

    def zrange(self, key: str, start: int, end: int,
               with_scores: bool = False) -> List[Any]:
        """ZRANGE key start end [WITHSCORES] — ordre croissant."""
        result = self._run(self._call_tool("zrange", {
            "key": key, "start": start, "end": end, "with_scores": with_scores,
        }))
        return self._parse_zrange_result(result, with_scores)

    def zrevrange(self, key: str, start: int, end: int,
                  with_scores: bool = False) -> List[Any]:
        """
        ZREVRANGE key start end — ordre décroissant.
        On utilise zrange complet puis on inverse et pagine.
        """
        all_items = self.zrange(key, 0, -1, with_scores=with_scores)
        reversed_items = list(reversed(all_items))
        if end == -1:
            return reversed_items[start:]
        return reversed_items[start:end + 1]

    def zrem(self, key: str, member: str) -> Any:
        """ZREM key member"""
        return self._run(self._call_tool("zrem", {"key": key, "member": member}))

    # ── JSON operations ──────────────────────────────────────────────────────

    def json_set(self, name: str, path: str, value: Any,
                 expire_seconds: Optional[int] = None) -> Any:
        """JSON.SET name path value"""
        val_str = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        args: Dict[str, Any] = {"name": name, "path": path, "value": val_str}
        if expire_seconds is not None:
            args["expire_seconds"] = expire_seconds
        return self._run(self._call_tool("json_set", args))

    def json_get(self, name: str, path: str = "$") -> Any:
        """JSON.GET name [path] → objet Python parsé"""
        result = self._run(self._call_tool("json_get", {"name": name, "path": path}))
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return result
        return result

    # ── Key operations ───────────────────────────────────────────────────────

    def delete(self, key: str) -> Any:
        """DEL key"""
        return self._run(self._call_tool("delete", {"key": key}))

    def delete_many(self, keys: List[str]) -> int:
        """Supprime plusieurs clés. Retourne le nombre de clés supprimées."""
        count = 0
        for k in keys:
            try:
                self._run(self._call_tool("delete", {"key": k}))
                count += 1
            except Exception:
                pass
        return count

    def scan_keys(self, pattern: str = "*") -> List[str]:
        """
        SCAN avec pattern — retourne toutes les clés correspondantes.
        Utilise scan_all_keys du serveur MCP (itération complète).
        """
        result = self._run(self._call_tool("scan_all_keys", {"pattern": pattern}))
        if isinstance(result, list):
            return [str(k) for k in result]
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return [str(k) for k in parsed]
            except (json.JSONDecodeError, TypeError):
                return [result] if result else []
        return []

    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe via hgetall ou get."""
        # Le serveur n'a pas "exists" — on utilise get ou type
        try:
            result = self._run(self._call_tool("type", {"key": key}))
            return result is not None and result != "none"
        except Exception:
            return False

    # ── Counter ──────────────────────────────────────────────────────────────

    def incr(self, key: str) -> int:
        """
        Incrémente un compteur. Le serveur MCP n'a pas INCR natif.
        On simule avec get+set (atomicité non garantie — acceptable pour nos IDs).
        """
        current = self.get(key)
        new_val = int(current) + 1 if current and current.isdigit() else 1
        self.set(key, str(new_val))
        return new_val

    # ── Server info ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Vérifie la connectivité Redis via info()."""
        try:
            self._ensure_connected()
            result = self._run(self._call_tool("dbsize", {}))
            return result is not None
        except Exception as e:
            logger.error("MCP-Redis ping failed : %s", e)
            return False

    def dbsize(self) -> int:
        """Retourne le nombre de clés dans la base."""
        result = self._run(self._call_tool("dbsize", {}))
        if isinstance(result, int):
            return result
        if isinstance(result, str) and result.isdigit():
            return int(result)
        return 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass


# ── Singleton global ─────────────────────────────────────────────────────────
# Importable partout : from services.mcp_redis_service import mcp_redis

mcp_redis: Optional[MCPRedisService] = None


def get_mcp_redis() -> MCPRedisService:
    """Retourne le singleton MCPRedisService (lazy init)."""
    global mcp_redis
    if mcp_redis is None:
        mcp_redis = MCPRedisService()
    return mcp_redis
