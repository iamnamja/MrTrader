from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from app.database.session import get_session
from app.database.models import AgentDecision


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    def __init__(self, name: str):
        self.agent_name = name
        self.logger = logging.getLogger(f"agents.{name}")
        self.status = "initialized"

    @abstractmethod
    async def run(self):
        """Main agent loop - implement in subclasses."""
        pass

    async def log_decision(
        self,
        decision_type: str,
        trade_id: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an agent decision to the database for auditing."""
        db = get_session()
        try:
            decision = AgentDecision(
                agent_name=self.agent_name,
                decision_type=decision_type,
                trade_id=trade_id,
                reasoning=reasoning,
                timestamp=datetime.utcnow(),
            )
            db.add(decision)
            db.commit()
            self.logger.info(f"Logged decision: {decision_type}")
        except Exception as e:
            self.logger.error(f"Failed to log decision {decision_type}: {e}")
            db.rollback()
        finally:
            db.close()

        # Broadcast to dashboard WebSocket (best-effort, non-blocking)
        try:
            import asyncio
            from app.api.websocket import broadcast_agent_decision
            coro = broadcast_agent_decision(self.agent_name, decision_type, reasoning or {})
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
        except Exception:
            pass

    def send_message(self, queue_name: str, message: Dict[str, Any]) -> bool:
        """Send a message to a Redis queue."""
        from app.integrations import get_redis_queue
        return get_redis_queue().push(queue_name, message)

    def get_message(self, queue_name: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """Receive a message from a Redis queue (blocking)."""
        from app.integrations import get_redis_queue
        return get_redis_queue().pop(queue_name, timeout=timeout)
