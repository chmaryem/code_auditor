"""
api/websocket_manager.py — Gestionnaire de connexions WebSocket.

Permet à plusieurs clients VS Code de recevoir les résultats d'analyse
en temps réel (watch mode). Chaque client reçoit le broadcast.

Usage:
    manager = ConnectionManager()
    await manager.connect(websocket)
    await manager.broadcast({"type": "analysis_result", "data": {...}})
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Gère les connexions WebSocket actives.

    Thread-safe : les méthodes connect/disconnect peuvent être appelées
    depuis n'importe quel thread grâce au lock asyncio.
    """

    def __init__(self):
        self._connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepte et enregistre une nouvelle connexion WebSocket."""
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info("WebSocket client connecté (%d actifs)", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Retire une connexion fermée."""
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info("WebSocket client déconnecté (%d actifs)", len(self._connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Envoie un message JSON à tous les clients connectés.
        Les connexions mortes sont automatiquement nettoyées.
        """
        dead: List[WebSocket] = []

        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        # Nettoyer les connexions mortes
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)
            logger.debug("Nettoyé %d connexion(s) morte(s)", len(dead))

    async def send_to(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Envoie un message à un client spécifique."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.debug("Envoi échoué vers client: %s", e)

    @property
    def active_count(self) -> int:
        return len(self._connections)
