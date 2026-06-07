"""
Trend (TSMOM) sleeve beta-isolation — does its Sharpe survive hedging out SPY?

Mirrors scripts/pead_phase1_attribution.py for the ETF trend sleeve: run the
validated TSMOM backtest, regress its net daily returns on SPY (CAPM + Newey-West
HAC), and stress the "uncorrelated crisis-diversifier" claim with crisis-window
returns, full-sample SPY correlation, and per-year Sharpe stability.

SPY is taken from the SAME price DataFrame the sleeve trades (no separate fetch) so
the market proxy and the book share one data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
from scripts.run_tsmom import _fetch_etf_prices
from scripts.walkforward.attribution import capm_alpha

ANN = 252
# (label, start, end) — classic equity stress windows to test crisis-positivity.
CRISES = [
    ("GFC 2008",        "2008-09-01", "2009-03-31"),
    ("Euro/US 2011",    "2011-07-01", "2011-10-31"),
    ("Q4-2018",         "2018-10-01", "2018-12-31"),
    ("COVID crash",     "2020-02-19", "2020-03-23"),
    ("2022 bear",       "2022-01-01", "2022-10-31"),
]


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def _window_ret(r: pd.Series, a: str, b: str) -> tuple[float, int]:
    w = r.loc[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    return (float((1.0 + w).prod() - 1.0), len(w))


def main() -> int:
    cfg = TSMOMConfig()
    prices = _fetch_etf_prices(cfg.universe)
    res = tsmom_backtest(prices, cfg)
    r_trend = res.returns.copy()
    r_trend.index = pd.to_datetime(r_trend.index)

    spy = prices["SPY"].copy()
    spy.index = pd.to_datetime(spy.index)
    r_spy = spy.pct_change().dropna()

    m = capm_alpha(r_trend, r_spy)

    print("\n" + "=" * 78)
    print("  TREND (TSMOM) SLEEVE — BETA ISOLATION (CAPM + Newey-West HAC)")
    print(f"  Universe: {','.join(cfg.universe)}")
    print(f"  Window: {r_trend.index[0].date()} -> {r_trend.index[-1].date()}  "
          f"({len(r_trend)} days)")
    print("=" * 78)
    print(f"  raw Sharpe (net):     {m['raw_sharpe']:+.3f}")
    print(f"  SPY beta:             {m['beta']:+.3f}")
    print(f"  annualized alpha:     {m['alpha_ann']*100:+.2f}%")
    print(f"  alpha t (OLS):        {m['t_alpha_ols']:+.2f}")
    print(f"  alpha t (HAC, lag5):  {m['t_alpha_hac']:+.2f}")
    print(f"  beta-hedged Sharpe:   {m['resid_sharpe']:+.3f}   "
          f"(PEAD's was -0.37 — the failing comparison)")
    print(f"  R^2 vs SPY:           {m['r2']:.3f}")
    print(f"  corr(trend, SPY):     {float(r_trend.corr(r_spy)):+.3f}")
    print("=" * 78)

    # Crisis-window total returns: trend vs SPY (the diversification claim).
    print("  CRISIS WINDOWS (total return: trend vs SPY) — the crisis-positive test")
    print(f"  {'window':<16}{'trend':>10}{'SPY':>10}{'days':>7}")
    for label, a, b in CRISES:
        tr, n = _window_ret(r_trend, a, b)
        sp, _ = _window_ret(r_spy, a, b)
        print(f"  {label:<16}{tr*100:>+9.1f}%{sp*100:>+9.1f}%{n:>7}")
    print("=" * 78)

    # Per-year Sharpe stability (is +0.71 a few years or broad?).
    print("  PER-YEAR net Sharpe (stability)")
    by_year = r_trend.groupby(r_trend.index.year).apply(_sharpe)
    pos_years = int((by_year > 0).sum())
    cells = "  ".join(f"{y}:{s:+.2f}" for y, s in by_year.items())
    print(f"  {cells}")
    print(f"  positive years: {pos_years}/{len(by_year)}")
    print("=" * 78)

    # Verdict (mirrors pead_phase1 wording, oriented to the diversifier thesis).
    if m["resid_sharpe"] > 0.2 and m["beta"] < 0.6:
        verdict = ("DIVERSIFIER CONFIRMED — positive hedged Sharpe + modest beta; the "
                   "edge is NOT just market exposure (contrast PEAD residSR -0.37).")
    elif m["resid_sharpe"] > 0.0:
        verdict = ("WEAK-POSITIVE hedged Sharpe — some standalone content, but check beta "
                   "and crisis windows before leaning on it.")
    else:
        verdict = ("BETA-DRIVEN — hedged Sharpe <= 0; the raw Sharpe is largely market "
                   "exposure. Re-examine the diversification claim.")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
