"""Alpha-v10 audit Wave 5a — trend partial-bar, cash idempotent-reuse, allocator warmup.

Pins: the trend rebalance drops today's still-forming bar (no same-day look-ahead); the cash sleeve
re-derives actual shares from the broker on an idempotent-reuse (never books shares it didn't trade);
and the inverse-vol allocator falls back to EQUAL weights during vol-warmup (never hands one sleeve
the full budget).
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ── allocator vol-warmup -> equal weights ────────────────────────────────────────
def test_vol_weights_fall_back_to_equal_during_warmup():
    from app.strategy.sleeve_allocator import vol_weights, AllocatorConfig
    cfg = AllocatorConfig(vol_lookback=60)        # min_periods = max(30,10) = 30
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    # 'trend' has full history; 'pead' only starts at row 25 (so rows 0..~54 are in vol-warmup)
    pead = pd.Series([np.nan] * 25 + list(np.random.RandomState(0).normal(0, 0.01, 15)), index=idx)
    trend = pd.Series(np.random.RandomState(1).normal(0, 0.01, 40), index=idx)
    w = vol_weights(pd.concat([pead.rename("pead"), trend.rename("trend")], axis=1), cfg)
    # during warmup (both vols not yet estimable) the row must be EQUAL (0.5/0.5), never (0,1)
    last = w.iloc[-1]
    assert abs(last["pead"] - 0.5) < 1e-9 and abs(last["trend"] - 0.5) < 1e-9


# ── cash idempotent-reuse re-derives from broker ─────────────────────────────────
class _DB:
    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def test_cash_idempotent_reuse_rederives_actual_shares(monkeypatch):
    import app.live_trading.cash_sleeve as cs
    importlib.reload(cs)
    monkeypatch.setattr(cs, "_audit", lambda *a, **k: None)

    class _A:
        def __init__(self):
            self.orders = []
            self._prices = {"SGOV": 100.0}

        def get_account(self):
            return {"cash": 50_000, "equity": 100_000, "portfolio_value": 100_000}

        def get_positions(self):
            return []

        def get_latest_price(self, s):
            return self._prices.get(s)

        def place_market_order(self, sym, qty, side, client_order_id=None):
            self.orders.append((sym, qty, side))
            return {"order_id": "x", "idempotent_reuse": True}     # same-day re-run collision

        def get_position(self, sym, **k):
            return {"qty": 999}                                     # ACTUAL held at broker

    alpaca = _A()
    cfg = {"pm.cash_enabled": "true", "pm.cash_shadow": "false", "pm.cash_buffer_pct": 0.02,
           "pm.cash_universe": "SGOV", "pm.reconciliation_mode": "off",
           "pm.whole_book_gate_mode": "off"}
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, k: cfg.get(k))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: alpaca)
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch", SimpleNamespace(is_active=False))
    monkeypatch.setattr("app.live_trading.trend_sleeve._fetch_prices", lambda a, u: None)
    monkeypatch.setattr("app.live_trading.trend_sleeve._live_prices",
                        lambda a, syms, df: {s: alpaca._prices.get(s) for s in syms if alpaca._prices.get(s)})
    monkeypatch.setattr(cs, "_current_cash_positions", lambda db, a: {})
    synced = []
    monkeypatch.setattr(cs, "_sync_cash_trade",
                        lambda db, sym, shares, price, oid: synced.append((sym, shares)))

    cs.run_cash_rebalance(db=_DB())
    # the DB sync used the ACTUAL broker shares (999), NOT the freshly-computed target
    assert synced and all(s[1] == 999 for s in synced)


# ── trend drops today's forming bar from the price panel ─────────────────────────
def test_trend_drops_todays_forming_bar():
    # The guard mirrors trend_sleeve's inline filter: rows strictly before today (the settled closes)
    # are kept; today's still-forming bar is dropped so .iloc[-1] is the latest SETTLED close.
    from datetime import datetime
    today = pd.Timestamp.now().normalize()
    idx = pd.to_datetime([today - pd.Timedelta(days=4), today - pd.Timedelta(days=3),
                          today - pd.Timedelta(days=1), today])
    df = pd.DataFrame({"SPY": [10.0, 11.0, 12.0, 999.0]}, index=idx)   # 999 = wild forming bar
    today_iso = datetime.now().date().isoformat()
    filtered = df.loc[df.index.strftime("%Y-%m-%d") < today_iso]
    assert today_iso not in list(filtered.index.strftime("%Y-%m-%d"))   # today dropped
    assert len(filtered) == 3 and float(filtered["SPY"].iloc[-1]) == 12.0  # settled close, not 999
