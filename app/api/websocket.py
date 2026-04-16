"""
WebSocket manager for real-time dashboard updates.
"""

import logging
from datetime import datetime
from typing import List

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages active WebSocket connections and broadcasts updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients, dropping stale connections."""
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_update(self, update_type: str, data: dict) -> None:
        await self.broadcast({
            "type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })


# Module-level singleton
manager = WebSocketManager()


# ─── Endpoint ─────────────────────────────────────────────────────────────────

async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket handler — keeps the connection alive and handles pings."""
    await manager.connect(websocket)
    try:
        while True:
            # Accept incoming text (e.g. ping / subscription requests)
            text = await websocket.receive_text()
            if text == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
        manager.disconnect(websocket)


# ─── Broadcast helpers ────────────────────────────────────────────────────────

async def broadcast_trade(trade: dict) -> None:
    await manager.send_update("trade_executed", trade)


async def broadcast_position_update(symbol: str, position: dict) -> None:
    await manager.send_update("position_update", {"symbol": symbol, "position": position})


async def broadcast_alert(alert_type: str, message: str) -> None:
    await manager.send_update("alert", {"type": alert_type, "message": message})
