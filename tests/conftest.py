"""
Shared pytest fixtures for MrTrader tests.

Fixture hierarchy:
  - db_engine / db_session  : in-memory SQLite DB (no PostgreSQL needed)
  - mock_alpaca             : MagicMock replacing AlpacaClient
  - mock_redis              : MagicMock replacing RedisQueue
  - test_client             : FastAPI TestClient with all external deps patched
"""
from __future__ import annotations

from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.database.models import Base


# ── In-memory SQLite DB ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_engine():
    """One SQLite engine for the whole test session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    """Fresh transactional DB session per test — rolls back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ── Mock Alpaca ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_alpaca():
    client = MagicMock()
    client.health_check.return_value = True
    client.get_account.return_value = {
        "cash": 10000.0,
        "buying_power": 10000.0,
        "portfolio_value": 10000.0,
        "equity": 10000.0,
        "account_blocked": False,
        "status": "ACTIVE",
    }
    client.get_positions.return_value = []
    client.get_position.return_value = None
    client.place_market_order.return_value = {
        "order_id": "test-order-123",
        "symbol": "AAPL",
        "qty": 10,
        "side": "buy",
        "status": "filled",
        "created_at": datetime.utcnow().isoformat(),
    }
    client.get_latest_price.return_value = 150.0
    client.get_bars.return_value = MagicMock(empty=True)
    return client


# ── Mock Redis ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.health_check.return_value = True
    redis.enqueue.return_value = True
    redis.dequeue.return_value = None
    return redis


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest.fixture
def test_client(mock_alpaca, mock_redis, db_session):
    """
    TestClient with all external services mocked and DB session injected.
    Import app lazily so env-var validation doesn't fire at import time.
    """
    from unittest.mock import AsyncMock
    with (
        patch("app.integrations.get_alpaca_client", return_value=mock_alpaca),
        patch("app.integrations.get_redis_queue", return_value=mock_redis),
        patch("app.database.check_db_connection", return_value=True),
        patch("app.database.session.get_session", return_value=db_session),
        patch("app.orchestrator.AgentOrchestrator.start", new_callable=AsyncMock),
        patch("app.orchestrator.AgentOrchestrator.stop", new_callable=AsyncMock),
        patch("app.main.init_db"),
        patch("app.startup_reconciler.reconcile", return_value={"ghost_positions": [], "orphaned_orders": []}),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ── Sample model factories ─────────────────────────────────────────────────────

def make_trade(db_session, symbol="AAPL", direction="BUY", status="ACTIVE",
               entry_price=150.0, quantity=10, pnl=None, signal_type="EMA_CROSSOVER"):
    from app.database.models import Trade
    t = Trade(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        status=status,
        pnl=pnl,
        signal_type=signal_type,
        stop_price=entry_price - 5.0,
        target_price=entry_price + 8.0,
        highest_price=entry_price,
        bars_held=0,
        created_at=datetime.utcnow(),
    )
    db_session.add(t)
    db_session.flush()
    return t


def make_audit_log(db_session, action="TEST_ACTION", details=None):
    from app.database.models import AuditLog
    log = AuditLog(
        action=action,
        details=details or {},
        timestamp=datetime.utcnow(),
    )
    db_session.add(log)
    db_session.flush()
    return log
