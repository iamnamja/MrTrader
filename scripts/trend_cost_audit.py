"""
Trend (TSMOM) sleeve — per-instrument transaction-cost audit.

The flat 2bps in the standalone backtest is optimistic for the thinner ETFs
(DBC/UUP/EEM/EFA) at ~14x/yr turnover. This reconstructs the sleeve's net returns
with a PER-SYMBOL one-way cost vector (reusing the exact tsmom rebalance + PIT cost
convention) and re-runs the CAPM/HAC beta-isolation, so we see whether the edge
survives realistic, name-specific costs rather than a single blended assumption.

Self-check: with a uniform 2bps vector this reproduces tsmom_backtest(cost_bps=2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.tsmom import TSMOMConfig, tsmom_backtest, tsmom_weights
from scripts.run_tsmom import _fetch_etf_prices
from scripts.walkforward.attribution import capm_alpha

ANN = 252

# Realistic retail-size one-way cost (bps): half bid-ask spread + ~0 commission +
# small impact. Tight majors ~1bp; thin commodity/USD ETFs materially wider.
REALISTIC = {
    "SPY": 1.0, "QQQ": 1.0, "IWM": 1.5, "EFA": 2.0, "EEM": 2.5,
    "TLT": 1.5, "IEF": 1.5, "GLD": 1.5, "DBC": 4.0, "UUP": 4.0,
}
# Pessimistic (wide-spread / larger size / stressed liquidity) — ~2x realistic.
PESSIMISTIC = {
    "SPY": 2.0, "QQQ": 2.0, "IWM": 3.0, "EFA": 4.0, "EEM": 5.0,
    "TLT": 3.0, "IEF": 3.0, "GLD": 3.0, "DBC": 8.0, "UUP": 8.0,
}


def _net_returns(prices, cfg, cost_vec_bps: dict) -> tuple[pd.Series, float]:
    """Reconstruct net daily returns with per-symbol one-way costs, matching the
    tsmom_backtest rebalance + cost-timing convention. Returns (net_returns,
    turnover-weighted blended bps)."""
    rets = prices.pct_change()
    w = tsmom_weights(prices, cfg)
    n = len(w)
    is_rebal = (np.arange(n) % cfg.rebalance_days == 0)
    held = w.where(pd.Series(is_rebal, index=w.index), other=np.nan).ffill().fillna(0.0)
    dw = held.diff().abs()
    dw.iloc[0] = held.abs().iloc[0]          # initial entry (matches tsmom dw.fillna)
    cost_vec = pd.Series({s: cost_vec_bps.get(s, 0.0) / 1e4 for s in held.columns})
    cost_per_day = (dw * cost_vec).sum(axis=1)
    gross_ret = (held.shift(1) * rets).sum(axis=1)
    net = (gross_ret - cost_per_day.shift(1)).dropna()
    # Turnover-weighted blended one-way bps (what flat-cost would have to be).
    tot_turn = float(dw.sum(axis=1).sum())
    blended = float((dw * cost_vec).sum(axis=1).sum() / tot_turn * 1e4) if tot_turn else 0.0
    return net, blended


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def main() -> int:
    cfg = TSMOMConfig()
    prices = _fetch_etf_prices(cfg.universe)
    spy = prices["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    r_spy = spy.pct_change().dropna()

    # Self-check: uniform 2bps must reproduce tsmom_backtest(cost_bps=2).
    flat2 = {s: 2.0 for s in cfg.universe}
    net_flat, _ = _net_returns(prices, cfg, flat2)
    net_flat.index = pd.to_datetime(net_flat.index)
    ref = tsmom_backtest(prices, TSMOMConfig(cost_bps=2.0)).returns
    ref.index = pd.to_datetime(ref.index)
    aligned = net_flat.reindex(ref.index)
    max_err = float((aligned - ref).abs().max())
    print(f"  [self-check] per-symbol@2bps vs tsmom_backtest@2bps: max daily-return "
          f"diff = {max_err:.2e}  ({'OK' if max_err < 1e-9 else 'MISMATCH'})")

    print("\n" + "=" * 78)
    print("  TREND SLEEVE — PER-INSTRUMENT COST AUDIT (CAPM/HAC, lag=21)")
    print("=" * 78)
    print(f"  {'cost model':<14}{'blendBps':>9}{'rawSR':>8}{'beta':>7}"
          f"{'alphaAnn':>9}{'t_HAC':>7}{'hedgedSR':>9}")
    for label, vec in [("flat 2bps", flat2), ("realistic", REALISTIC),
                       ("pessimistic", PESSIMISTIC)]:
        net, blended = _net_returns(prices, cfg, vec)
        net.index = pd.to_datetime(net.index)
        m = capm_alpha(net, r_spy, hac_lag=21)
        print(f"  {label:<14}{blended:>9.1f}{m['raw_sharpe']:>+8.3f}{m['beta']:>+7.2f}"
              f"{m['alpha_ann']*100:>+8.2f}%{m['t_alpha_hac']:>+7.2f}{m['resid_sharpe']:>+9.3f}")
    print("=" * 78)

    # Per-symbol turnover share — where the cost actually bites.
    w = tsmom_weights(prices, cfg)
    n = len(w); is_rebal = (np.arange(n) % cfg.rebalance_days == 0)
    held = w.where(pd.Series(is_rebal, index=w.index), other=np.nan).ffill().fillna(0.0)
    dw = held.diff().abs(); dw.iloc[0] = held.abs().iloc[0]
    turn_share = (dw.sum() / dw.sum().sum()).sort_values(ascending=False)
    print("  Turnover share by symbol (where cost concentrates):")
    print("  " + "  ".join(f"{s}:{v*100:.0f}%" for s, v in turn_share.items()))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
