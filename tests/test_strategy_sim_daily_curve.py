"""Phase 3 (KL-8) — StrategySimulator build_daily_equity_curve flag tests.

Default (False) = legacy entry-date-keyed equity curve. True = forward-fill onto
every calendar day in [start_date, end_date] so flat days between sparse entries
are counted in the Sharpe/Calmar denominator.

NOTE: StrategySimulator is TIER-2 only — never used in WF/CPCV.
"""
from datetime import date, timedelta

from app.backtesting.metrics import BacktestResult, Trade
from app.backtesting.strategy_simulator import StrategySimulator


def _trade(symbol, entry, hold_days, pnl_pct):
    exit_d = entry + timedelta(days=hold_days)
    return Trade(
        symbol=symbol, entry_date=entry, exit_date=exit_d,
        entry_price=100.0, exit_price=100.0 * (1 + pnl_pct), quantity=10,
        pnl=1000.0 * pnl_pct, pnl_pct=pnl_pct, hold_bars=hold_days,
        exit_reason="TARGET",
    )


def _sparse_result():
    """A few entries spread far apart in time → sparse entry dates."""
    trades = [
        _trade("AAA", date(2022, 1, 3), 3, 0.05),
        _trade("BBB", date(2022, 3, 1), 3, -0.02),
        _trade("CCC", date(2022, 6, 1), 3, 0.04),
        _trade("DDD", date(2022, 9, 1), 3, 0.03),
    ]
    return BacktestResult.from_trades(trades, model_type="swing")


def test_default_curve_is_entry_date_keyed():
    br = _sparse_result()
    sim = StrategySimulator()
    res = sim.run(br, start_date=date(2022, 1, 1), end_date=date(2022, 12, 31),
                  build_daily_equity_curve=False)
    distinct_entries = len({t.entry_date for t in br.trades})
    assert len(res.equity_curve) == distinct_entries


def test_daily_curve_spans_every_calendar_day():
    br = _sparse_result()
    sim = StrategySimulator()
    start, end = date(2022, 1, 1), date(2022, 12, 31)
    res = sim.run(br, start_date=start, end_date=end, build_daily_equity_curve=True)
    expected_days = (end - start).days + 1
    assert len(res.equity_curve) == expected_days
    # Curve is contiguous daily and forward-filled (monotone dates, no gaps).
    dates = [d for d, _ in res.equity_curve]
    assert dates[0] == start
    assert dates[-1] == end
    for i in range(1, len(dates)):
        assert dates[i] == dates[i - 1] + timedelta(days=1)
    # Flat days between entries carry the prior equity value (forward fill).
    vals = [v for _, v in res.equity_curve]
    # The first stretch before any entry settles is the starting capital.
    assert vals[0] == sim.starting_capital


def test_daily_curve_sharpe_differs_and_is_lower():
    br = _sparse_result()
    sim = StrategySimulator()
    start, end = date(2022, 1, 1), date(2022, 12, 31)
    res_entry = sim.run(br, start_date=start, end_date=end,
                        build_daily_equity_curve=False)
    res_daily = sim.run(br, start_date=start, end_date=end,
                        build_daily_equity_curve=True)
    # Different granularity → different Sharpe.
    assert res_daily.sharpe_ratio != res_entry.sharpe_ratio
    # Flat days dilute volatility-per-observation: daily curve Sharpe magnitude is
    # generally lower for sparse-entry strategies.
    assert abs(res_daily.sharpe_ratio) < abs(res_entry.sharpe_ratio)
