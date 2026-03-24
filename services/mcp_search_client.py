# services/mcp_search_client.py
"""
  Mode 2 — DuckDuckGo      (fallback gratuit, sans clé)

Utilisé par FeedbackProcessor pour enrichir les règles KB.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional
from ddgs import DDGS

logger = logging.getLogger(__name__)


class MCPSearchClient:
    """
    Façade unifiée : Brave MCP si clé dispo, DuckDuckGo sinon.
    Toujours non-bloquant — retourne '' en cas d'erreur.
    """

    def __init__(self):
        self._api_key = os.getenv("BRAVE_API_KEY", "").strip()

        # Paramètres serveur MCP (utilisés seulement si clé présente)
        self._server_params = None
        if self._api_key:
            try:
                from mcp import StdioServerParameters
                self._server_params = StdioServerParameters(
                    command = "npx",
                    args    = ["-y", "@modelcontextprotocol/server-brave-search"],
                    env     = {"BRAVE_API_KEY": self._api_key},
                )
            except ImportError:
                logger.debug("Package 'mcp' absent — fallback DuckDuckGo activé")

        # Vérifier disponibilité DuckDuckGo
        self._ddg_available: Optional[bool] = None

    # ── API publique ──────────────────────────────────────────────────────────

    async def search(self, query: str, count: int = 3) -> str:
        """Recherche async. Retourne '' si erreur."""
        if not query.strip():
            return ""
        try:
            return await asyncio.wait_for(
                self._search_internal(query, count),
                timeout=12,
            )
        except asyncio.TimeoutError:
            logger.debug("Search timeout pour : %s", query[:60])
            return ""
        except Exception as e:
            logger.debug("Search erreur : %s", e)
            return ""

    def search_sync(self, query: str, count: int = 3) -> str:
        """
        Version synchrone — compatible avec FeedbackProcessor (non-async).
        Gère correctement le cas où une event loop tourne déjà (thread worker).
        """
        if not query.strip():
            return ""
        try:
            # Si une loop tourne déjà (ex: thread asyncio), on lance dans un thread séparé
            try:
                loop = asyncio.get_running_loop()
                # On est dans une loop → exécuter dans un thread dédié
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        lambda: asyncio.run(self.search(query, count))
                    )
                    return future.result(timeout=15)
            except RuntimeError:
                # Pas de loop running → on peut appeler run_until_complete
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.search(query, count))
                finally:
                    loop.close()
        except Exception as e:
            logger.debug("search_sync erreur : %s", e)
            return ""

    # ── Routage interne ───────────────────────────────────────────────────────

    async def _search_internal(self, query: str, count: int) -> str:
        """Choisit le backend selon disponibilité."""
        if self._api_key and self._server_params:
            result = await self._search_brave_mcp(query, count)
            if result:
                logger.debug("MCP Brave Search : %d chars", len(result))
                return result
            # MCP a échoué → fallback DDG
            logger.debug("MCP Brave échoué → fallback DuckDuckGo")

        return await self._search_duckduckgo(query, count)

    # ── Backend 1 : MCP Brave Search ─────────────────────────────────────────

    async def _search_brave_mcp(self, query: str, count: int) -> str:
        try:
            from mcp.client.stdio import stdio_client
            from mcp import ClientSession

            async with stdio_client(self._server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "brave_web_search",
                        {"query": query, "count": count},
                    )
                    parts = [
                        c.text for c in result.content
                        if hasattr(c, "text") and c.text
                    ]
                    return "\n".join(parts)
        except Exception as e:
            logger.debug("Brave MCP erreur : %s", e)
            return ""

    # ── Backend 2 : DuckDuckGo (gratuit, sans clé) ───────────────────────────

    async def _search_duckduckgo(self, query: str, count: int) -> str:
        """
        Utilise duckduckgo-search (pip install duckduckgo-search).
        Retourne '' si la lib est absente — jamais d'exception.
        """
        if self._ddg_available is False:
            return ""

        try:
           
            self._ddg_available = True
        except ImportError:
            self._ddg_available = False
            logger.debug("duckduckgo-search absent — pip install duckduckgo-search")
            return ""

        try:
            # DDGS est synchrone — on l'exécute dans un executor pour ne pas bloquer
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=count))
            )
            if not results:
                return ""

            parts = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body",  "")
                href  = r.get("href",  "")
                if title or body:
                    parts.append(f"Titre : {title}\nURL : {href}\nRésumé : {body}")

            combined = "\n---\n".join(parts)
            logger.debug("DuckDuckGo : %d résultat(s), %d chars", len(results), len(combined))
            return combined

        except Exception as e:
            logger.debug("DuckDuckGo erreur : %s", e)
            return ""


# Instance globale — importée par feedback_processor.py
mcp_search_client = MCPSearchClient()