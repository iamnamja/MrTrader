"""
Phase 84 — End-to-End PM → RM → Trader round-trip integration test.

Drives a complete proposal lifecycle using:
  - An in-process deque-backed FakeQueue (replaces Redis)
  - SQLite in-memory DB (replaces PostgreSQL)
  - Mocked Alpaca client (no real brokerage calls)

Tests:
  1. RM approves a valid PM proposal and pushes to trader_approved_trades
  2. RM rejects when kill switch is active
  3. Trader creates a PENDING_FILL Trade row on receiving an approved swing proposal
  4. Full swing round-trip: limit order placed, simulated fill, Trade ACTIVE
  5. RM rejects when MAX_OPEN_POSITIONS exceeded
  6. Intraday market-order path: Trade created after fill poll

Run manually (excluded from normal CI by mark):
    pytest tests/test_e2e_round_trip.py -m e2e -v -s
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# In-process fake queue  (replaces Redis)
# ---------------------------------------------------------------------------

class _FakeQueue:
    """Thread-safe in-process queue compatible with RedisQueue interface."""

    def __init__(self):
        self._queues: Dict[str, deque] = {}

    def push(self, queue_name: str, message: Dict[str, Any]) -> bool:
        self._queues.setdefault(queue_name, deque()).append(message)
        return True

    def pop(self, queue_name: str, timeout: int = 5, **kwargs) -> Optional[Dict[str, Any]]:
        q = self._queues.get(queue_name, deque())
        return q.popleft() if q else None

    def clear(self, queue_name: str) -> None:
        self._queues.pop(queue_name, None)

    def clear_all(self) -> None:
        self._queues.clear()


_fake_queue = _FakeQueue()


# ---------------------------------------------------------------------------
# In-memory SQLite session factory (shared engine so all sessions see same rows)
# ---------------------------------------------------------------------------

def _make_db_engine():
    """Return a SQLAlchemy engine + Session factory backed by a fresh in-memory SQLite."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)
    return engine, SessionFactory


def _make_db_session():
    """Convenience: return a single session (for tests that only need one)."""
    _, SessionFactory = _make_db_engine()
    return SessionFactory()


# ---------------------------------------------------------------------------
# Mock Alpaca factory
# ---------------------------------------------------------------------------

def _make_bars(price: float = 200.0, n: int = 250):
    """Return a minimal pandas DataFrame simulating N daily bars."""
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = np.full(n, price)
    return pd.DataFrame({
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": np.full(n, 1_000_000),
    }, index=dates)


def _make_alpaca(symbol: str = "TSLA", fill_price: float = 200.0):
    order_id = f"order-{uuid.uuid4().hex[:8]}"
    alpaca = MagicMock()
    alpaca.get_account.return_value = {
        "equity": 20_000.0, "portfolio_value": 20_000.0,
        "buying_power": 15_000.0, "cash": 15_000.0,
    }
    alpaca.get_positions.return_value = []
    alpaca.get_quote.return_value = {
        "ask": fill_price, "bid": fill_price - 0.10, "mid": fill_price - 0.05
    }
    alpaca.get_latest_price.return_value = fill_price
    alpaca.get_bars.return_value = _make_bars(fill_price)
    alpaca.place_limit_order.return_value = {"order_id": order_id, "status": "new"}
    alpaca.place_market_order.return_value = {"order_id": order_id, "status": "accepted"}
    alpaca.get_order_status.side_effect = [
        {"status": "new",    "filled_qty": 0, "filled_avg_price": None},
        {"status": "filled", "filled_qty": 5, "filled_avg_price": str(fill_price)},
        {"status": "filled", "filled_qty": 5, "filled_avg_price": str(fill_price)},
        {"status": "filled", "filled_qty": 5, "filled_avg_price": str(fill_price)},
    ]
    alpaca.cancel_order.return_value = True
    return alpaca, order_id


# ---------------------------------------------------------------------------
# Proposal factory
# ---------------------------------------------------------------------------

def _make_proposal(symbol: str = "TSLA", trade_type: str = "swing", price: float = 200.0) -> Dict:
    return {
        "symbol": symbol,
        "trade_type": trade_type,
        "direction": "BUY",
        "entry_price": price,
        "quantity": 5,
        "confidence": 0.75,
        "ml_score": 0.75,
        "signal_type": "ML_RANK",
        "proposal_uuid": str(uuid.uuid4()),
        "source_agent": "portfolio_manager",
        "atr": 3.0,
        "stop_price": price * 0.95,
        "target_price": price * 1.06,
    }


# ---------------------------------------------------------------------------
# Queue patch context manager
# ---------------------------------------------------------------------------

def _queue_patch():
    """Patch the Redis queue singleton with _fake_queue."""
    import app.integrations.redis_queue as rq_mod
    return patch.object(rq_mod, "redis_queue", _fake_queue)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_queue():
    _fake_queue.clear_all()
    yield
    _fake_queue.clear_all()


# ---------------------------------------------------------------------------
# Test 1 — RM approves a valid proposal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rm_approves_valid_proposal():
    """RM consumes a valid PM proposal and pushes it to trader_approved_trades."""
    from app.agents.risk_manager import RiskManager

    proposal = _make_proposal("AAPL", trade_type="swing", price=150.0)
    _fake_queue.push("trade_proposals", proposal)

    rm = RiskManager()
    rm.status = "running"
    _, SessionFactory = _make_db_engine()

    with (
        _queue_patch(),
        patch("app.agents.risk_manager.get_session", side_effect=SessionFactory),
        patch("app.integrations.get_alpaca_client") as mock_ac,
        patch("app.calendars.earnings.earnings_calendar") as mock_ec,
        patch("app.live_trading.kill_switch.kill_switch") as mock_ks,
    ):
        mock_ks.is_active = False
        mock_ac.return_value.get_account.return_value = {
            "equity": 20_000.0, "portfolio_value": 20_000.0,
            "buying_power": 15_000.0, "cash": 15_000.0,
        }
        mock_ac.return_value.get_positions.return_value = []
        mock_ec.is_blocked.return_value = False
        mock_ec.get_earnings_risk.return_value = MagicMock(
            block_swing=False, block_intraday=False, exit_review=False, reason=""
        )

        async def _stop():
            await asyncio.sleep(0)
            rm.status = "stopped"

        await asyncio.gather(rm.run(), _stop())

    approved = _fake_queue.pop("trader_approved_trades")
    assert approved is not None, "Expected an approved proposal on trader_approved_trades"
    assert approved["symbol"] == "AAPL"
    assert "approved_at" in approved


# ---------------------------------------------------------------------------
# Test 2 — RM rejects when kill switch is active
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rm_rejects_when_kill_switch_active():
    """RM immediately rejects any proposal when kill switch is active."""
    from app.agents.risk_manager import RiskManager

    _fake_queue.push("trade_proposals", _make_proposal("NVDA", price=500.0))

    rm = RiskManager()
    rm.status = "running"
    _, SessionFactory = _make_db_engine()

    with (
        _queue_patch(),
        patch("app.agents.risk_manager.get_session", side_effect=SessionFactory),
        patch("app.live_trading.kill_switch.kill_switch") as mock_ks,
    ):
        mock_ks.is_active = True

        async def _stop():
            await asyncio.sleep(0)
            rm.status = "stopped"

        await asyncio.gather(rm.run(), _stop())

    assert _fake_queue.pop("trader_approved_trades") is None, \
        "Kill switch active: no proposal should reach trader"


# ---------------------------------------------------------------------------
# Test 3 — RM rejects over MAX_OPEN_POSITIONS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rm_rejects_over_position_limit():
    """RM rejects proposals that would breach MAX_OPEN_POSITIONS."""
    from app.agents.risk_manager import RiskManager
    from app.agents.risk_rules import RiskLimits
    from app.database.models import Trade

    _fake_queue.push("trade_proposals", _make_proposal("GOOG", price=170.0))

    engine, SessionFactory = _make_db_engine()
    limits = RiskLimits()
    # Seed open positions so limit is reached
    seed_db = SessionFactory()
    for i in range(limits.MAX_OPEN_POSITIONS):
        seed_db.add(Trade(symbol=f"SYM{i}", direction="BUY", entry_price=100.0,
                          quantity=10, status="ACTIVE"))
    seed_db.commit()
    seed_db.close()

    rm = RiskManager(limits=limits)

    mock_ac = MagicMock()
    mock_ac.get_account.return_value = {
        "equity": 20_000.0, "portfolio_value": 20_000.0,
        "buying_power": 15_000.0, "cash": 15_000.0,
    }
    # Simulate MAX_OPEN_POSITIONS live positions so RM's open_positions rule trips
    mock_ac.get_positions.return_value = [
        {"symbol": f"SYM{i}", "qty": "10", "avg_entry_price": "100.00"}
        for i in range(limits.MAX_OPEN_POSITIONS)
    ]

    with (
        _queue_patch(),
        patch("app.agents.risk_manager.get_session", side_effect=SessionFactory),
        patch("app.integrations.get_alpaca_client", return_value=mock_ac),
        patch("app.live_trading.kill_switch.kill_switch") as mock_ks,
    ):
        mock_ks.is_active = False

        async def _stop():
            await asyncio.sleep(0)
            rm.status = "stopped"

        await asyncio.gather(rm.run(), _stop())

    assert _fake_queue.pop("trader_approved_trades") is None, \
        "Proposal should have been rejected due to position limit"


# ---------------------------------------------------------------------------
# Shared Trader context manager
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def _trader_context(session_factory, alpaca):
    """
    Patch everything a Trader needs for an isolated E2E run.
    session_factory: callable that returns a new Session (all sessions share the same engine).
    """
    cb_mock = MagicMock()
    cb_mock.is_open = False
    cb_mock.is_strategy_paused.return_value = False

    with (
        _queue_patch(),
        patch("app.agents.trader.get_session", side_effect=session_factory),
        patch("app.integrations.get_alpaca_client", return_value=alpaca),
        patch("app.live_trading.kill_switch.kill_switch") as mock_ks,
        patch("app.calendars.earnings.earnings_calendar") as mock_ec,
        patch("app.calendars.macro.macro_calendar") as mock_mc,
        patch("app.agents.trader.circuit_breaker", cb_mock),
        patch("app.agents.trader.CHECK_INTERVAL", 0),
    ):
        mock_ks.is_active = False
        mock_ec.is_blocked.return_value = False
        mock_ec.get_earnings_risk.return_value = MagicMock(
            block_swing=False, block_intraday=False, exit_review=False
        )
        mock_mc.get_context.return_value = MagicMock(
            block_new_entries=False, sizing_factor=1.0
        )
        yield


def _run_trader(trader, max_iters: int = 5, timeout: float = 12.0):
    """Return a coroutine that runs the trader for at most max_iters scan cycles."""
    iterations = 0
    original_scan = trader._scan_cycle

    async def _limited_scan():
        nonlocal iterations
        iterations += 1
        if iterations >= max_iters:
            trader.status = "stopped"
        await original_scan()

    trader._scan_cycle = _limited_scan
    trader._reconcile_positions = AsyncMock()
    trader._reload_pending_limits_from_db = MagicMock()
    return asyncio.wait_for(trader.run(), timeout=timeout)


# ---------------------------------------------------------------------------
# Test 4 — Trader creates PENDING_FILL on swing approval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trader_creates_pending_fill_on_swing_approval():
    """Trader receives approved swing proposal, places limit order, creates PENDING_FILL Trade."""
    from app.agents.trader import Trader
    from app.database.models import Trade

    SYMBOL, PRICE = "TSLA", 200.0
    approved = {
        **_make_proposal(SYMBOL, trade_type="swing", price=PRICE),
        "stop_loss": PRICE * 0.95,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "_proposal_id": None,
    }
    _fake_queue.push("trader_approved_trades", approved)

    alpaca, _ = _make_alpaca(SYMBOL, PRICE)
    engine, SessionFactory = _make_db_engine()
    trader = Trader()
    trader.status = "running"

    async with _trader_context(SessionFactory, alpaca):
        try:
            await _run_trader(trader, max_iters=4)
        except asyncio.TimeoutError:
            trader.status = "stopped"

    # Query using a fresh session on the shared engine
    verify_db = SessionFactory()
    trades = verify_db.query(Trade).filter(Trade.symbol == SYMBOL).all()
    verify_db.close()
    assert len(trades) >= 1, f"Expected Trade row for {SYMBOL}"
    statuses = {t.status for t in trades}
    assert statuses & {"PENDING_FILL", "ACTIVE"}, f"Unexpected statuses: {statuses}"
    assert alpaca.place_limit_order.call_count >= 1, "Expected place_limit_order to be called"


# ---------------------------------------------------------------------------
# Test 5 — Trader processes intraday market order → Trade created
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_intraday_market_order_creates_trade():
    """Intraday approved proposal goes through market order path and creates a Trade."""
    from app.agents.trader import Trader
    from app.database.models import Trade

    SYMBOL, PRICE = "NVDA", 110.0
    approved = {
        **_make_proposal(SYMBOL, trade_type="intraday", price=PRICE),
        "stop_loss": PRICE * 0.98,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "_proposal_id": None,
    }
    _fake_queue.push("trader_approved_trades", approved)

    alpaca, order_id = _make_alpaca(SYMBOL, PRICE)
    # Market order fills on second status poll
    alpaca.get_order_status.side_effect = [
        {"status": "accepted", "filled_qty": 0, "filled_avg_price": None},
        {"status": "filled",   "filled_qty": 5, "filled_avg_price": str(PRICE)},
        {"status": "filled",   "filled_qty": 5, "filled_avg_price": str(PRICE)},
    ]

    engine, SessionFactory = _make_db_engine()
    trader = Trader()
    trader.status = "running"

    # Patch datetime to return a weekday at 10:30 AM ET so intraday entries are allowed
    from datetime import datetime as _real_dt
    from zoneinfo import ZoneInfo

    class _FakeDt(_real_dt):
        @classmethod
        def now(cls, tz=None):
            return _real_dt(2026, 5, 1, 10, 30, 0, tzinfo=ZoneInfo("America/New_York"))

    async with _trader_context(SessionFactory, alpaca):
        with patch("app.agents.trader.datetime", _FakeDt):
            try:
                await _run_trader(trader, max_iters=5)
            except asyncio.TimeoutError:
                trader.status = "stopped"

    verify_db = SessionFactory()
    trades = verify_db.query(Trade).filter(Trade.symbol == SYMBOL).all()
    verify_db.close()
    assert len(trades) >= 1, f"Expected Trade row for {SYMBOL}"
    alpaca.place_market_order.assert_called_once()


# ---------------------------------------------------------------------------
# Test 6 — Full PM → RM → Trader pipeline (proposal → queue → approval → Trade)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pm_rm_trader_pipeline():
    """
    Simulate the complete three-agent pipeline:
    1. Put a raw proposal on trade_proposals (as PM would)
    2. Run RM to approve it → lands on trader_approved_trades
    3. Run Trader to consume it → Trade row created in DB
    """
    from app.agents.risk_manager import RiskManager
    from app.agents.trader import Trader
    from app.database.models import Trade

    SYMBOL, PRICE = "MSFT", 100.0  # $100 * 5 shares = $500 = 2.5% of $20k account
    proposal = _make_proposal(SYMBOL, trade_type="swing", price=PRICE)
    _fake_queue.push("trade_proposals", proposal)

    engine, SessionFactory = _make_db_engine()
    alpaca, _ = _make_alpaca(SYMBOL, PRICE)

    # --- Phase 1: RM approves ---
    rm = RiskManager()
    rm.status = "running"

    with (
        _queue_patch(),
        patch("app.agents.risk_manager.get_session", side_effect=SessionFactory),
        patch("app.integrations.get_alpaca_client", return_value=alpaca),
        patch("app.calendars.earnings.earnings_calendar") as mock_ec,
        patch("app.live_trading.kill_switch.kill_switch") as mock_ks,
    ):
        mock_ks.is_active = False
        mock_ec.is_blocked.return_value = False
        mock_ec.get_earnings_risk.return_value = MagicMock(
            block_swing=False, block_intraday=False, exit_review=False, reason=""
        )

        async def _stop_rm():
            await asyncio.sleep(0)
            rm.status = "stopped"

        await asyncio.gather(rm.run(), _stop_rm())

    # Verify RM approved it
    approved = _fake_queue.pop("trader_approved_trades")
    assert approved is not None, "RM should have approved the proposal"
    assert approved["symbol"] == SYMBOL

    # Re-push for Trader to consume
    _fake_queue.push("trader_approved_trades", approved)

    # --- Phase 2: Trader processes ---
    trader = Trader()
    trader.status = "running"

    async with _trader_context(SessionFactory, alpaca):
        try:
            await _run_trader(trader, max_iters=4)
        except asyncio.TimeoutError:
            trader.status = "stopped"

    verify_db = SessionFactory()
    trades = verify_db.query(Trade).filter(Trade.symbol == SYMBOL).all()
    verify_db.close()
    assert len(trades) >= 1, f"Pipeline produced no Trade row for {SYMBOL}"
    statuses = {t.status for t in trades}
    assert statuses & {"PENDING_FILL", "ACTIVE"}, f"Unexpected statuses: {statuses}"
