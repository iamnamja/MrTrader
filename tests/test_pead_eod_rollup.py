"""
Tests for the PEAD EOD P&L update + Friday weekly rollup wiring
(feat/pead-eod-pnl-weekly-rollup).

Covers:
  PART 1 — _compute_pead_eod_stats: sources the PEAD-tagged book's
           gross/realized/unrealized + n_entered/n_filled, FILTERING OUT
           non-PEAD trades; mark-to-market via mocked Alpaca.
  PART 1 — EOD update path: _run_eod_jobs upserts today's pead_daily row with
           nonzero gross (helper mocked) while PRESERVING the signals-stage
           n_signals/vix fields (real upsert via temp DB).
  PART 2 — weekly_rollup: real Sharpe with seeded nonzero rows (matches hand
           computation); "n/a" when gross==0; skipped when < min_days deployed.
  PART 2 — Friday-only firing: rollup called on Friday, NOT Mon-Thu.
  Trade.selector — Trader populates it from proposal["selector"].

All external calls (Alpaca, DB, clock) are mocked — no network/live I/O.
"""
import asyncio
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — weekly_rollup Sharpe correctness + guards
# ════════════════════════════════════════════════════════════════════════════════

class TestWeeklyRollupSharpe:
    def test_real_sharpe_matches_hand_computation(self, tmp_path, monkeypatch):
        """Seeded nonzero gross/pnl rows → numeric Sharpe == mean/std*sqrt(252)."""
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        # daily returns = pnl / gross : 0.05, -0.02, 0.03
        pt.record_daily(date(2026, 5, 26), realized_pnl=50.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 27), realized_pnl=-20.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 28), realized_pnl=30.0, gross_deployed=1000.0)

        payload = pt.weekly_rollup(date(2026, 5, 28), send=False)

        rets = np.array([0.05, -0.02, 0.03])
        expected = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
        assert payload["realized_sharpe"] == pytest.approx(round(expected, 3))
        assert "skipped" not in payload

    def test_gross_zero_rows_give_na(self, tmp_path, monkeypatch):
        """All gross==0 (signals-stage only) → Sharpe 'n/a' (and skipped)."""
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        pt.record_daily(date(2026, 5, 26), n_signals=3, gross_deployed=0.0)
        pt.record_daily(date(2026, 5, 27), n_signals=2, gross_deployed=0.0)
        pt.record_daily(date(2026, 5, 28), n_signals=4, gross_deployed=0.0)

        payload = pt.weekly_rollup(date(2026, 5, 28), send=False)
        assert payload["realized_sharpe"] == "n/a"

    def test_fewer_than_min_days_is_skipped_not_sent(self, tmp_path, monkeypatch):
        """< 3 deployed days → payload marked skipped, notifier NOT called."""
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        pt.record_daily(date(2026, 5, 27), realized_pnl=10.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 28), realized_pnl=5.0, gross_deployed=1000.0)
        # only 2 deployed days < min_days=3

        called = {"n": 0}

        def _fake_enqueue(*a, **k):
            called["n"] += 1
            return 1

        with patch("app.notifications.notifier.enqueue", _fake_enqueue):
            payload = pt.weekly_rollup(date(2026, 5, 28), send=True)

        assert payload["skipped"] == "insufficient data"
        assert called["n"] == 0  # not sent

    def test_min_days_satisfied_sends(self, tmp_path, monkeypatch):
        """>= 3 deployed days → notifier enqueued with the dedup key."""
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        pt.record_daily(date(2026, 5, 26), realized_pnl=50.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 27), realized_pnl=-20.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 28), realized_pnl=30.0, gross_deployed=1000.0)

        captured = {}

        def _fake_enqueue(event_type, payload, dedup_key=None):
            captured["event_type"] = event_type
            captured["dedup_key"] = dedup_key
            return 1

        with patch("app.notifications.notifier.enqueue", _fake_enqueue):
            payload = pt.weekly_rollup(date(2026, 5, 28), send=True)

        assert "skipped" not in payload
        assert captured["event_type"] == "pead_weekly"
        assert captured["dedup_key"] == "pead_weekly_2026-05-28"


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — _compute_pead_eod_stats
# ════════════════════════════════════════════════════════════════════════════════

def _bare_pm(monkeypatch=None):
    """Bare PM. _alpaca is a read-only property returning get_alpaca_client(),
    so callers that need Alpaca must patch get_alpaca_client (see _patch_alpaca)."""
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = MagicMock()
    return pm


def _patch_alpaca(monkeypatch, client):
    monkeypatch.setattr(
        "app.integrations.get_alpaca_client", lambda: client, raising=True
    )


class _FakeTrade:
    def __init__(self, symbol, selector, status, entry_price=0.0, quantity=0,
                 pnl=None, closed_today=False, created_today=True):
        self.symbol = symbol
        self.selector = selector
        self.status = status
        self.entry_price = entry_price
        self.quantity = quantity
        self.pnl = pnl
        # closed_at / created_at set by the fake query filter logic below
        self._closed_today = closed_today
        self._created_today = created_today


class _FakeQuery:
    """Minimal SQLAlchemy-query stand-in that filters an in-memory trade list
    according to the (status / closed_today / created_today) intent of each call.
    We distinguish the three queries in _compute_pead_eod_stats by inspecting the
    filter arguments passed (string repr), which is brittle to mimic, so instead
    we route on a call-order counter.
    """
    def __init__(self, trades, counter):
        self._trades = trades
        self._counter = counter

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        idx = self._counter["n"]
        self._counter["n"] += 1
        if idx == 0:  # open positions (ACTIVE / PENDING_FILL, PEAD)
            return [t for t in self._trades
                    if t.selector == "pead" and t.status in ("ACTIVE", "PENDING_FILL")]
        if idx == 1:  # closed today (PEAD)
            return [t for t in self._trades
                    if t.selector == "pead" and t.status == "CLOSED" and t._closed_today]
        # created today (PEAD)
        return [t for t in self._trades
                if t.selector == "pead" and t._created_today]


def _patch_session(monkeypatch, trades):
    counter = {"n": 0}
    fake_db = MagicMock()
    fake_db.query.side_effect = lambda model: _FakeQuery(trades, counter)
    monkeypatch.setattr(
        "app.database.session.get_session", lambda: fake_db, raising=True
    )
    return fake_db


class TestComputePeadEodStats:
    def test_filters_out_non_pead_and_computes_fields(self, monkeypatch):
        pm = _bare_pm()
        trades = [
            # open PEAD positions → gross = 100*10 + 50*20 = 2000
            _FakeTrade("AAPL", "pead", "ACTIVE", entry_price=100.0, quantity=10),
            _FakeTrade("MSFT", "pead", "PENDING_FILL", entry_price=50.0, quantity=20),
            # closed PEAD today → realized = 30 + (-10) = 20
            _FakeTrade("NVDA", "pead", "CLOSED", pnl=30.0, closed_today=True),
            _FakeTrade("AMD", "pead", "CLOSED", pnl=-10.0, closed_today=True),
            # NON-PEAD: must be filtered OUT of every aggregate
            _FakeTrade("SPY", "", "ACTIVE", entry_price=400.0, quantity=100),
            _FakeTrade("TSLA", "quality_short", "CLOSED", pnl=999.0, closed_today=True),
        ]
        _patch_session(monkeypatch, trades)

        # Alpaca positions: AAPL/MSFT held (PEAD) + SPY (non-PEAD, ignored)
        alpaca = MagicMock()
        alpaca.get_positions.return_value = [
            {"symbol": "AAPL", "unrealized_pl": 15.0},
            {"symbol": "MSFT", "unrealized_pl": -5.0},
            {"symbol": "SPY", "unrealized_pl": 1000.0},  # non-PEAD → excluded
        ]
        _patch_alpaca(monkeypatch, alpaca)

        stats = pm._compute_pead_eod_stats()

        assert stats["gross_deployed"] == pytest.approx(2000.0)
        assert stats["realized_pnl"] == pytest.approx(20.0)
        # unrealized only counts PEAD-held symbols AAPL+MSFT: 15 - 5 = 10
        assert stats["unrealized_pnl"] == pytest.approx(10.0)
        # n_entered = all PEAD created today (6 PEAD rows minus 0 non-pead) = 4 PEAD
        assert stats["n_entered"] == 4
        # n_filled = PEAD rows in ACTIVE/CLOSED = AAPL, NVDA, AMD = 3 (MSFT PENDING)
        assert stats["n_filled"] == 3

    def test_no_open_positions_skips_alpaca(self, monkeypatch):
        pm = _bare_pm()
        trades = [
            _FakeTrade("NVDA", "pead", "CLOSED", pnl=42.0, closed_today=True),
        ]
        _patch_session(monkeypatch, trades)
        alpaca = MagicMock()
        _patch_alpaca(monkeypatch, alpaca)

        stats = pm._compute_pead_eod_stats()
        assert stats["gross_deployed"] == pytest.approx(0.0)
        assert stats["realized_pnl"] == pytest.approx(42.0)
        assert stats["unrealized_pnl"] == pytest.approx(0.0)
        alpaca.get_positions.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 — EOD update path preserves signals-stage fields
# ════════════════════════════════════════════════════════════════════════════════

class TestEodUpdatePreservesSignalsStage:
    def test_upsert_preserves_n_signals_and_vix(self, tmp_path, monkeypatch):
        """The EOD upsert overwrites P&L/fill fields but PRESERVES n_signals/vix
        written at the signals stage (ON CONFLICT DO UPDATE)."""
        import app.live_trading.pead_tracker as pt
        import app.agents.portfolio_manager as pm_mod
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        # _run_eod_jobs computes today via `datetime.date.today()` (real clock), so
        # seed the signals-stage row at the SAME real today to exercise the upsert.
        today = date.today()
        pt.record_daily(today, n_signals=7, vix_level=18.5, vix_block_fired=False)

        pm = _bare_pm()
        # Stub the helper to return real EOD numbers.
        pm._compute_pead_eod_stats = MagicMock(return_value={
            "gross_deployed": 5000.0,
            "realized_pnl": 120.0,
            "unrealized_pnl": 30.0,
            "n_entered": 4,
            "n_filled": 3,
        })
        # Stub the unrelated EOD sub-jobs imported lazily inside _run_eod_jobs
        monkeypatch.setattr("app.database.decision_audit.backfill_outcomes",
                            lambda *a, **k: 0, raising=True)
        monkeypatch.setattr("app.database.decision_audit.backfill_gate_outcomes",
                            lambda *a, **k: 0, raising=True)
        monkeypatch.setattr("app.database.decision_audit.backfill_scan_abstention_outcomes",
                            lambda *a, **k: 0, raising=True)
        monkeypatch.setattr("app.database.daily_summary.write_daily_summary",
                            lambda *a, **k: None, raising=True)
        pm._log_regime_divergence_today = MagicMock()

        # Force a Monday clock so the Friday rollup branch does NOT fire.
        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 6, 1, 16, 35)  # Monday

        monkeypatch.setattr(pm_mod, "datetime", _FrozenDateTime, raising=True)

        rollup_spy = MagicMock()
        monkeypatch.setattr(pt, "weekly_rollup", rollup_spy, raising=True)

        asyncio.run(pm._run_eod_jobs())

        rows = pt.read_daily()
        assert len(rows) == 1
        r = rows[0]
        # Preserved from signals stage:
        assert r["n_signals"] == 7
        assert r["vix_level"] == pytest.approx(18.5)
        # Overwritten by EOD update:
        assert r["gross_deployed"] == pytest.approx(5000.0)
        assert r["realized_pnl"] == pytest.approx(120.0)
        assert r["unrealized_pnl"] == pytest.approx(30.0)
        assert r["n_entered"] == 4
        assert r["n_filled"] == 3
        # Monday → rollup not called
        rollup_spy.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — Friday-only firing
# ════════════════════════════════════════════════════════════════════════════════

class TestFridayOnlyRollup:
    def _run_with_weekday(self, monkeypatch, tmp_path, dt_value):
        import app.live_trading.pead_tracker as pt
        import app.agents.portfolio_manager as pm_mod
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        pm = _bare_pm()
        pm._compute_pead_eod_stats = MagicMock(return_value=None)  # skip EOD upsert
        monkeypatch.setattr("app.database.decision_audit.backfill_outcomes",
                            lambda *a, **k: 0, raising=True)
        monkeypatch.setattr("app.database.decision_audit.backfill_gate_outcomes",
                            lambda *a, **k: 0, raising=True)
        monkeypatch.setattr("app.database.decision_audit.backfill_scan_abstention_outcomes",
                            lambda *a, **k: 0, raising=True)
        monkeypatch.setattr("app.database.daily_summary.write_daily_summary",
                            lambda *a, **k: None, raising=True)
        pm._log_regime_divergence_today = MagicMock()

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return dt_value

        monkeypatch.setattr(pm_mod, "datetime", _FrozenDateTime, raising=True)

        rollup_spy = MagicMock()
        monkeypatch.setattr(pt, "weekly_rollup", rollup_spy, raising=True)

        asyncio.run(pm._run_eod_jobs())
        return rollup_spy

    def test_rollup_fires_on_friday(self, monkeypatch, tmp_path):
        # 2026-05-29 is a Friday (weekday()==4)
        spy = self._run_with_weekday(monkeypatch, tmp_path,
                                     datetime(2026, 5, 29, 16, 35))
        spy.assert_called_once()

    def test_rollup_skipped_mon_thru_thu(self, monkeypatch, tmp_path):
        # Mon=2026-05-25 ... Thu=2026-05-28
        for d in (datetime(2026, 5, 25, 16, 35),
                  datetime(2026, 5, 26, 16, 35),
                  datetime(2026, 5, 27, 16, 35),
                  datetime(2026, 5, 28, 16, 35)):
            spy = self._run_with_weekday(monkeypatch, tmp_path, d)
            spy.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════════
# Trade.selector attribution (model column + Trader population)
# ════════════════════════════════════════════════════════════════════════════════

class TestTradeSelectorColumn:
    def test_trade_model_has_selector_column(self):
        from app.database.models import Trade
        assert hasattr(Trade, "selector")
        # default empty for non-PEAD
        col = Trade.__table__.columns["selector"]
        assert col.default.arg == ""

    def test_write_pending_fill_sets_selector_from_proposal(self, monkeypatch):
        """Trader._write_pending_fill threads proposal['selector'] onto the Trade."""
        from app.agents.trader import Trader
        import app.agents.trader as trader_mod

        captured = {}

        class _FakeTrade:
            def __init__(self, **kw):
                captured.update(kw)
                self.id = 123

        fake_db = MagicMock()
        monkeypatch.setattr(trader_mod, "Trade", _FakeTrade, raising=True)
        monkeypatch.setattr(trader_mod, "get_session", lambda: fake_db, raising=True)

        trader = Trader.__new__(Trader)
        trader.logger = MagicMock()

        result = SimpleNamespace(stop_price=95.0, target_price=110.0)
        proposal = {
            "trade_type": "swing",
            "proposal_uuid": "uuid-1",
            "direction": "BUY",
            "stop_price": 95.0,
            "target_price": 110.0,
            "selector": "pead",
        }
        trader._write_pending_fill(
            "AAPL", 10, 100.0, result, proposal, "ML_RANK"
        )
        assert captured["selector"] == "pead"

    def test_write_pending_fill_defaults_empty_selector(self, monkeypatch):
        """Missing proposal['selector'] → '' (non-PEAD)."""
        from app.agents.trader import Trader
        import app.agents.trader as trader_mod

        captured = {}

        class _FakeTrade:
            def __init__(self, **kw):
                captured.update(kw)
                self.id = 1

        monkeypatch.setattr(trader_mod, "Trade", _FakeTrade, raising=True)
        monkeypatch.setattr(trader_mod, "get_session", lambda: MagicMock(), raising=True)

        trader = Trader.__new__(Trader)
        trader.logger = MagicMock()
        result = SimpleNamespace(stop_price=95.0, target_price=110.0)
        proposal = {"trade_type": "swing", "proposal_uuid": "u", "direction": "BUY",
                    "stop_price": 95.0, "target_price": 110.0}
        trader._write_pending_fill("AAPL", 10, 100.0, result, proposal, "ML_RANK")
        assert captured["selector"] == ""
