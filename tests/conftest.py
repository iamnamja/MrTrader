"""
Shared pytest fixtures for MrTrader tests.

Fixture hierarchy:
  - db_engine / db_session  : in-memory SQLite DB (no PostgreSQL needed)
  - mock_alpaca             : MagicMock replacing AlpacaClient
  - mock_redis              : MagicMock replacing RedisQueue
  - test_client             : FastAPI TestClient with all external deps patched
"""
from __future__ import annotations

import os as _os
# Test-mode marker for the WHOLE session, set at conftest import (before any test
# imports app.main). Unlike `pytest in sys.modules` / PYTEST_CURRENT_TEST, an env var
# is INHERITED by spawned child processes, so the _DailyFileHandler in a pytest-spawned
# subprocess still routes logs to test_mrtrader_<date>.log instead of leaking into the
# live ops log. force-set (not setdefault) so the test session always wins.
_os.environ["MRTRADER_TEST_MODE"] = "1"

# Cap thread counts for all pytest workers — prevents loky/OpenMP spawning
# cpu_count() threads per worker when running under pytest-xdist.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    _os.environ.setdefault(_var, "2")

# pytest-xdist: give each worker its OWN SQLite files so the ~4 workers within a
# shard can't contend on a shared file ("database is locked" — the recurring CI
# flake). The env-overridable path constants in app.ml.feature_store /
# live_trading.pead_tracker / notifications.notifier read these. Also keeps tests
# off the real (multi-GB) feature_store.db. Cleaned implicitly (OS temp dir).
#
# Worker id: PYTEST_XDIST_WORKER is NOT yet set when conftest is *imported*, so a
# module-level read returns the same fallback for every worker (no isolation). The
# reliable source is xdist's `config.workerinput["workerid"]`, available in
# `pytest_configure` — which runs per-worker BEFORE collection (i.e. before the app
# path-constants are imported). So we set a shared safe default at import (covers
# any pre-configure import + non-xdist) and FORCE the per-worker path in
# pytest_configure.
import tempfile as _tempfile
from pathlib import Path as _Path


def _set_test_db_env(worker: str = "gw_main", *, force: bool = False) -> None:
    db_dir = _Path(_tempfile.gettempdir()) / "mrtrader_test_sqlite" / worker
    db_dir.mkdir(parents=True, exist_ok=True)
    setter = _os.environ.__setitem__ if force else _os.environ.setdefault
    setter("MRTRADER_FEATURE_STORE_DB", str(db_dir / "feature_store.db"))
    setter("MRTRADER_PEAD_TRACKING_DB", str(db_dir / "pead_tracking.db"))
    setter("MRTRADER_TREND_TRACKING_DB", str(db_dir / "trend_tracking.db"))
    setter("MRTRADER_ALLOCATOR_TRACKING_DB", str(db_dir / "allocator_tracking.db"))
    setter("MRTRADER_NOTIFICATIONS_DB", str(db_dir / "notifications.db"))
    setter("MRTRADER_RESEARCH_REGISTRY_DB", str(db_dir / "research_registry.db"))
    # Main-DB fail-closed backstop: session._resolve_database_url redirects a test-mode Postgres URL
    # here. Its value matters for a spawned app SUBPROCESS (which inherits this env but not conftest's
    # in-process patches) — per-worker so subprocesses of different xdist workers don't collide. The
    # in-process module `engine` is built once at conftest import and is never written to (every test
    # patches SessionLocal), so its exact guard path is immaterial in-process.
    setter("MRTRADER_TEST_DATABASE_URL", f"sqlite:///{db_dir / 'session_guard.db'}")


_set_test_db_env()  # import-time safety default (shared 'gw_main'); overridden below


def pytest_configure(config):
    # Per-worker isolation: workerinput is set by xdist in each worker before
    # collection; force per-worker paths so no two workers share a SQLite file.
    worker = getattr(config, "workerinput", {}).get("workerid", "gw_main")
    _set_test_db_env(worker, force=True)


from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.database.models import Base

# Pre-import modules that bind `get_session` via `from app.database.session import
# get_session`, BEFORE any test patches app.database.session.get_session. The
# test_client fixture patches session.get_session AND these modules' get_session in one
# `with` block. If such a module is imported FRESH while session.get_session is already
# patched, its `from ... import get_session` binds to the MOCK; unittest.mock then
# captures that mock as the patch's "original" and "restores" to it on exit — leaking the
# mock into every later test in the worker (proven: test_client left
# signal_attribution.get_session a MagicMock). Importing them here binds their name to the
# REAL function once, so the patch's capture/restore is symmetric and the leak is closed.
import app.analytics.signal_attribution  # noqa: E402,F401
import app.analytics.drawdown_analyzer  # noqa: E402,F401


# ── Phase-4 gate-mode default for the legacy test corpus ──────────────────────
# The production default is GATE_MODE="ruler_v2" (Alpha-v7, live 2026-06-13; was
# "significance", retained as legacy). The
# large pre-Phase-4 gate test corpus, however, asserts the LEGACY mean-Sharpe gate
# semantics (avg_sharpe>=0.80, t-stat WARN-only, paper_gate relaxed thresholds,
# WF promotion allowed). Those tests are exactly the "faithful legacy reproduction"
# proof — so this autouse fixture forces GATE_MODE="mean_sharpe" for every test by
# default. Tests that want the NEW significance behavior opt in explicitly with the
# `significance_gate_mode` fixture (see test_significance_gate.py). The code reads
# GATE_MODE via `from app.ml.retrain_config import GATE_MODE` inside each gate
# function, so patching the module attribute is sufficient.
@pytest.fixture(autouse=True)
def _ensure_current_event_loop():
    """Guarantee a current event loop on the main thread for EVERY test.

    eventkit (ib_insync's dependency) calls asyncio.get_event_loop() at IMPORT
    time; on Python 3.12 that RAISES "no current event loop" when none is set on
    the thread. Under pytest-asyncio strict mode an async test can leave the
    main-thread loop reset to None, so whichever test first imports ib_insync on
    a worker (directly, or via a fixture like `fake_ib`) crashes at import —
    purely as a function of shard/loadscope ordering. This makes that import
    order-independent. Autouse + function-scoped so it runs before the requested
    fixtures that do the import; a no-op when a loop already exists.
    """
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield


@pytest.fixture(autouse=True)
def _legacy_gate_mode_default(request):
    if "significance_gate_mode" in getattr(request, "fixturenames", ()):  # opted in elsewhere
        yield
        return
    with patch("app.ml.retrain_config.GATE_MODE", "mean_sharpe"):
        yield


@pytest.fixture
def significance_gate_mode():
    """Opt-in: run the test under the production GATE_MODE='significance'."""
    with patch("app.ml.retrain_config.GATE_MODE", "significance"):
        yield


# ── Kill switch safety net ────────────────────────────────────────────────────
# Defense-in-depth: ensure no test can ever persist kill_switch.active=True to
# the production DB, even if a test forgets to patch the singleton.
@pytest.fixture(autouse=True)
def _isolate_kill_switch():
    """Force the real kill_switch singleton to a clean, non-persisting state
    around every test. _persist_state and _audit are also no-ops under pytest
    (see app/live_trading/kill_switch.py::_running_under_pytest), but this
    guarantees the in-memory flag is also reset.
    """
    try:
        from app.live_trading.kill_switch import kill_switch as _ks
        _ks._active = False
    except Exception:
        _ks = None
    yield
    if _ks is not None:
        _ks._active = False


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

@pytest.fixture(scope="session")
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

@pytest.fixture(scope="session")
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
        # Startup code calls get_session() then db.close() — give it a throwaway mock
        # so it doesn't close the test's db_session and invalidate the transaction.
        # Route handlers use db_session exclusively via the override_get_db dependency.
        patch("app.database.session.get_session", return_value=MagicMock()),
        patch("app.analytics.signal_attribution.get_session", return_value=MagicMock()),
        patch("app.analytics.drawdown_analyzer.get_session", return_value=MagicMock()),
        patch("app.orchestrator.AgentOrchestrator.start", new_callable=AsyncMock),
        patch("app.orchestrator.AgentOrchestrator.stop", new_callable=AsyncMock),
        patch("app.main.init_db"),
        patch("app.startup_reconciler.reconcile", return_value={"ghost_positions": [], "orphaned_orders": []}),
    ):
        from app.main import app
        from app.database.session import get_db

        def override_get_db():
            yield db_session

        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
        app.dependency_overrides.pop(get_db, None)


# ── Sample model factories ─────────────────────────────────────────────────────

def make_trade(db_session, symbol="AAPL", direction="BUY", status="ACTIVE",
               entry_price=150.0, quantity=10, pnl=None, signal_type="EMA_CROSSOVER"):
    from app.database.models import Trade
    now = datetime.utcnow()
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
        created_at=now,
        closed_at=now if status == "CLOSED" else None,
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
