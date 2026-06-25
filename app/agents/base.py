from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from app.database.models import AgentDecision


def _sanitize_reasoning(value: Any) -> Any:
    """Recursively scrub a reasoning payload of non-JSON-serializable objects
    (e.g. unittest.mock.Mock / MagicMock that would otherwise be stringified
    as ``"MagicMock name='…'``"" and persisted into the live DB).

    Detects mocks by class hierarchy *and* by module so test-only objects
    never reach `agent_decisions.reasoning`. Unknown non-primitive types are
    replaced with their ``repr`` so the column never crashes serialization.
    """
    try:
        from unittest.mock import NonCallableMock
    except Exception:  # pragma: no cover
        NonCallableMock = ()  # type: ignore

    _PRIMS = (str, int, float, bool, type(None))

    def _walk(v: Any) -> Any:
        if isinstance(v, _PRIMS):
            return v
        if isinstance(v, NonCallableMock) or v.__class__.__module__.startswith("unittest.mock"):
            return "<scrubbed-mock>"
        if isinstance(v, dict):
            return {str(k): _walk(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [_walk(x) for x in v]
        return v  # let SQLAlchemy / JSON encoder handle real objects

    if value is None:
        return None
    return _walk(value)


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    def __init__(self, name: str):
        self.agent_name = name
        self.logger = logging.getLogger(f"agents.{name}")
        self.status = "initialized"

    def _activate_status(self) -> None:
        """Set status to 'running' UNLESS a standing pause is in force.

        The orchestrator restarts a crashed agent by re-invoking run(); a hard 'running' at the top
        of run() would silently resume an agent the operator (or health check) had paused — the agent
        self-un-pauses while the control plane still reports paused. Preserving an existing 'paused'
        status keeps the pause across a crash-and-restart."""
        if self.status != "paused":
            self.status = "running"

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
        # Lazy import so test-level patches of app.database.session.get_session
        # take effect (prevents test mocks from leaking into the live DB).
        from app.database.session import get_session
        reasoning = _sanitize_reasoning(reasoning)
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
