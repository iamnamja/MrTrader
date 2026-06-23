"""Alpha-v10 audit Wave 3b — runtime/data state-corruption fixes.

Pins: the capital ramp stage persists across a restart; FRED fetches the MOST RECENT observations
(not the oldest); the earnings blackout counts TRADING days (not calendar days); single-row model
inference doesn't collapse to 0.0; and the PEAD daily-P&L uses the CHANGE in unrealized (not the
level).
"""
from __future__ import annotations

import datetime as _dt
import json
import types

import numpy as np


# ── capital ramp persistence ─────────────────────────────────────────────────────
def test_capital_ramp_persists_and_restores(monkeypatch):
    from app.live_trading.capital_manager import CapitalManager
    store: dict = {}
    monkeypatch.setattr("app.database.config_store.set_config",
                        lambda db, k, v, desc=None: store.__setitem__(k, v))
    monkeypatch.setattr("app.database.config_store.get_config", lambda db, k: store.get(k))
    monkeypatch.setattr("app.database.session.get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))
    cm = CapitalManager()
    monkeypatch.setattr(cm, "_running_under_pytest", lambda: False)
    cm.start()
    cm.advance()
    cm.advance()
    assert cm.current_stage.stage == 3
    # a fresh instance (simulating a daemon restart) restores Stage 3, NOT Stage 1
    cm2 = CapitalManager()
    monkeypatch.setattr(cm2, "_running_under_pytest", lambda: False)
    cm2.load_state()
    assert cm2.current_stage.stage == 3


# ── FRED most-recent ─────────────────────────────────────────────────────────────
def test_fred_fetches_most_recent_not_oldest(monkeypatch):
    import importlib
    fc = importlib.import_module("app.macro.fred_client")  # the module (package rebinds the name)
    monkeypatch.setattr(fc, "settings", types.SimpleNamespace(fred_api_key="k"))
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            # FRED returns desc -> newest first
            return json.dumps({"observations": [{"v": "newest"}, {"v": "older"}]}).encode()

    monkeypatch.setattr(fc.urllib.request, "urlopen",
                        lambda url, timeout=10: captured.__setitem__("url", url) or _Resp())
    out = fc.FredClient()._fetch_api("CPIAUCSL")
    assert "sort_order=desc" in captured["url"]      # request the most recent, not the oldest
    assert out[-1]["v"] == "newest"                  # reversed back to ascending (newest last)


# ── earnings blackout counts trading days ────────────────────────────────────────
def test_earnings_blackout_uses_trading_days(monkeypatch):
    from app.calendars import earnings as em

    class _D(_dt.date):
        @classmethod
        def today(cls):
            return _dt.date(2026, 6, 19)   # a Friday
    monkeypatch.setattr(em, "date", _D)
    ec = em.EarningsCalendar.__new__(em.EarningsCalendar)
    ec._cache = {}
    # earnings the following Tuesday: 4 CALENDAR days but only 2 TRADING days -> must block swing
    monkeypatch.setattr(ec, "_get_next_earnings", lambda s: (_dt.date(2026, 6, 23), True))
    risk = ec.get_earnings_risk("AAPL", "swing")
    assert risk.block_swing is True


# ── single-row model inference doesn't collapse to 0.0 ───────────────────────────
def test_regression_single_row_not_collapsed_to_zero():
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel.__new__(PortfolioSelectorModel)
    m._is_regression = True
    m.is_trained = True
    m._feature_weights = None
    m.scaler = types.SimpleNamespace(transform=lambda X: np.asarray(X, dtype=float))
    m.model = types.SimpleNamespace(predict=lambda X: np.array([0.8]))   # single row
    _, probs = m.predict([[1.0, 2.0]])
    assert probs[0] > 0.0                             # sigmoid(0.8) ~ 0.69, not a collapsed 0.0


# ── PEAD daily P&L uses the unrealized DELTA, not the level ───────────────────────
def test_pead_daily_pnl_uses_unrealized_delta(monkeypatch, tmp_path):
    from app.live_trading import pead_tracker as pt
    monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead.db")
    # Day 1: unrealized level 100 (prior 0) -> daily_pnl 100
    pt.record_daily(realized_pnl=0.0, unrealized_pnl=100.0, gross_deployed=1000.0,
                    trade_date=_dt.date(2026, 6, 20))
    # Day 2: unrealized STILL 100 (no change) -> daily_pnl must be 0, not 100 (the level)
    pt.record_daily(realized_pnl=0.0, unrealized_pnl=100.0, gross_deployed=1000.0,
                    trade_date=_dt.date(2026, 6, 21))
    rows = {r["trade_date"]: r for r in pt.read_daily()}
    assert abs(rows["2026-06-21"]["daily_pnl"]) < 1e-9          # delta = 0, not re-added 100
    assert abs(rows["2026-06-21"]["cumulative_pnl"] - 100.0) < 1e-9   # cumulative stays 100
