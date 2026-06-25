"""Alpha-v10 audit Wave 6b — intraday sim must populate portfolio.positions (caps + deployment).

Previously _process_day stored open positions only in a LOCAL window_entered dict; portfolio.positions
stayed empty. So validate_open_positions (MAX_OPEN) and validate_portfolio_heat read an empty dict
(dead caps), and the deployment metric (sampled EOD from position_market_value) was always 0. Fix:
register positions during the entry pass (caps bind; equity = cash + pmv), release them on exit, and
sample PEAK concurrent deployment during the window. portfolio.positions is empty at day-end (same-day
in/out) so the equity curve is unaffected.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from app.backtesting.intraday_agent_simulator import (
    IntradayAgentSimulator, _PortfolioState)
from app.agents.risk_rules import RiskLimits


def _day_bars(day: date, base: float = 100.0, n: int = 78) -> pd.DataFrame:
    idx = [datetime(day.year, day.month, day.day, 9, 30) + timedelta(minutes=5 * i)
           for i in range(n)]
    rows = [{"open": base, "high": base * 1.001, "low": base * 0.999,
             "close": base, "volume": 10000} for _ in range(n)]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _sim(max_open: int):
    s = IntradayAgentSimulator.__new__(IntradayAgentSimulator)
    s.limits = RiskLimits(MAX_OPEN_POSITIONS=max_open, MAX_POSITION_SIZE_PCT=0.20)
    s.transaction_cost_pct = 0.0
    s.scan_offsets = [12]
    s.meta_model = None
    s.earnings_blackout = None
    s.intraday_blackout_days_after = 0
    s.intraday_blackout_days_before = 0
    # stub the feature + scoring layers so we exercise the entry/exit/cap bookkeeping directly
    s._pm_score = lambda feats: [(sym, 0.9) for sym in feats]
    s._simulate_exit = lambda pos, fut: (pos.entry_price, "TARGET", 1)  # flat exit
    return s


def _run_one_day(sim, symbols, day):
    import app.backtesting.intraday_agent_simulator as mod
    symbols_data = {s: _day_bars(day) for s in symbols}
    sector_map = {s: "TECH" for s in symbols}
    portfolio = _PortfolioState(cash=100_000.0, peak_equity=100_000.0)
    # bypass the real feature computation (returns a non-None dict for every symbol/window)
    orig = mod.compute_intraday_features
    mod.compute_intraday_features = lambda *a, **k: {"feat1": 0.5}
    try:
        trades, tx, peak = sim._process_day(day, symbols_data, None, portfolio, sector_map)
    finally:
        mod.compute_intraday_features = orig
    return trades, peak, portfolio


def test_max_open_cap_now_binds():
    # 5 high-confidence symbols, MAX_OPEN=2 -> the concurrent cap must limit entries to 2
    sim = _sim(max_open=2)
    trades, _peak, portfolio = _run_one_day(sim, [f"S{i}" for i in range(5)], date(2026, 1, 5))
    assert len(trades) == 2                          # was unbounded before (cap read an empty dict)
    assert portfolio.positions == {}                 # same-day in/out -> flat at day-end


def test_deployment_metric_nonzero():
    sim = _sim(max_open=5)
    trades, peak, _portfolio = _run_one_day(sim, [f"S{i}" for i in range(3)], date(2026, 1, 5))
    assert len(trades) == 3
    assert peak > 0.0                                # was always 0 (EOD sample on an empty book)


def test_positions_cleared_at_day_end():
    sim = _sim(max_open=10)
    _trades, _peak, portfolio = _run_one_day(sim, [f"S{i}" for i in range(4)], date(2026, 1, 5))
    # equity curve sample (= cash, positions empty) is preserved: no leaked open positions
    assert portfolio.positions == {}
    assert portfolio.equity == portfolio.cash
