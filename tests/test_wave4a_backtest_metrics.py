"""Alpha-v10 audit Wave 4a — backtest metrics correctness (research/readiness gating).

Pins: break-even trades are neither winners nor losers (don't depress win-rate/profit-factor);
Sharpe is annualised by the ACTUAL trade frequency (not the sample size); max-drawdown is computed
in exit-date order.
"""
from __future__ import annotations

import math
from datetime import date

from app.backtesting.metrics import BacktestResult, Trade, _sharpe, _trades_per_year


def _t(pnl, pnl_pct, entry, exit_, sym="X"):
    return Trade(symbol=sym, entry_date=entry, exit_date=exit_, entry_price=100.0,
                 exit_price=100.0 * (1 + pnl_pct), quantity=1, pnl=pnl, pnl_pct=pnl_pct,
                 hold_bars=1, exit_reason="TARGET")


def test_break_even_is_neither_winner_nor_loser():
    trades = [_t(10.0, 0.10, date(2026, 1, 2), date(2026, 1, 3)),     # winner
              _t(0.0, 0.0, date(2026, 1, 4), date(2026, 1, 5)),       # break-even
              _t(-5.0, -0.05, date(2026, 1, 6), date(2026, 1, 7))]    # loser
    r = BacktestResult.from_trades(trades)
    assert r.winning_trades == 1 and r.losing_trades == 1            # break-even excluded from both
    # profit factor = gross_win / gross_loss = 10 / 5 = 2.0 (break-even not added to gross_loss)
    assert abs(r.profit_factor - 2.0) < 1e-6


def test_sharpe_annualises_by_frequency_not_sample_size():
    # same per-trade mean/std, but 200 trades over ~2y vs 50 trades over ~6mo: the OLD min(n,252)
    # made Sharpe grow with n; the frequency-based version is ~stable for the same trades/year.
    rets = [0.02, -0.01] * 100        # 200 returns
    # 200 trades over 365 days -> ~200/yr ; 200 trades over 730 days -> ~100/yr
    s_dense = _sharpe(rets, trades_per_year=200.0)
    s_sparse = _sharpe(rets, trades_per_year=100.0)
    assert s_dense > s_sparse                                  # frequency drives it, as designed
    # explicit formula check
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    std = math.sqrt(var)
    assert abs(s_dense - (mean / std) * math.sqrt(200.0)) < 1e-9


def test_trades_per_year_uses_span():
    # 10 trades spanning ~365 days -> ~10/yr
    trades = [_t(1.0, 0.01, date(2026, 1, 1), date(2026, 1, 1)) for _ in range(9)]
    trades.append(_t(1.0, 0.01, date(2026, 1, 1), date(2026, 12, 31)))
    tpy = _trades_per_year(trades)
    assert 9.0 < tpy < 11.0


def test_max_drawdown_uses_exit_order():
    # Two trades appended OUT of exit order: a +10% (exits later) then a -20% (exits earlier).
    # In exit order the -20% comes first (peak then trough), giving a real ~20% drawdown.
    trades = [_t(10.0, 0.10, date(2026, 1, 1), date(2026, 3, 1)),    # exits LATER
              _t(-20.0, -0.20, date(2026, 1, 1), date(2026, 2, 1))]  # exits EARLIER
    r = BacktestResult.from_trades(trades)
    # exit-ordered curve: 1.0 -> 0.8 (dd .20) -> 0.88 ; max_dd ~ 0.20
    assert abs(r.max_drawdown_pct - 0.20) < 1e-6
