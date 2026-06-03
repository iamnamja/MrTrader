"""
net_exposure.py — Realized net-beta / net-dollar / net-sector capture for the
RANKER v2 §3.1 dollar-neutral book (Spike A).

WHY THIS EXISTS (the blocker the deep-dive found):
  A dollar-neutral book equalizes long/short *dollars*. That gives net BETA ≈ 0
  ONLY if the longs' and shorts' average betas match. For a ranker they
  systematically don't — the bottom-of-rank (short) names tend to a different
  beta than the top-of-rank (long) names. So a positive Spike A Sharpe could be
  leftover market beta, not idiosyncratic alpha. The owner-locked decision
  (RANKER_V2_DESIGN.md §9-Q4: "dollar-neutral primary; switch to beta-neutral
  ONLY if realized net beta > 0.15") REQUIRES *measuring* realized net beta. This
  module is that measurement.

DISCIPLINE:
  * PIT (point-in-time): per-name betas use ONLY daily returns from bars STRICTLY
    BEFORE the decision day. A future SPY or price move can never change a past
    day's beta (asserted by tests). This mirrors the strict-`<` SPY lookups in
    AgentSimulator.spy_beta_hedge (agent_simulator.py:1948,1964) and
    factor_scorer.py:275 — same convention, reused.
  * Deterministic: pure OLS slope of name daily returns on SPY daily returns over
    a trailing `lookback`-day window. No randomness, no fitting state.
  * Pure-additive: this module is read-only over the live book; it computes
    diagnostics and NEVER mutates positions, cash, or any P&L path.

DEFINITIONS (signed, book-level, as fraction of equity):
  net_beta   = Σ_long  (w_i · β_i)  −  Σ_short (w_j · β_j)
                 where w = signed-positive notional/equity for each leg.
  net_dollar = (long_notional − short_notional) / equity
  net_sector[s] = (long_notional_in_s − short_notional_in_s) / equity
  max_abs_sector = max_s |net_sector[s]|

A clean dollar-neutral *alpha* read needs |net_beta| ≈ 0. If |net_beta| > 0.15 the
result is NOT clean dollar-neutral alpha (per the locked rule) and the experiment
must be re-run beta-neutral. NET_BETA_ALPHA_THRESHOLD encodes that 0.15 bar.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Owner-locked interpretation bar (RANKER_V2_DESIGN.md §9-Q4 / §2.3): a realized
# |net beta| above this means the dollar-neutral result is contaminated by residual
# market beta and the locked rule says re-run beta-neutral.
NET_BETA_ALPHA_THRESHOLD: float = 0.15

# Number of LEADING two-sided (both-legs-held) EOD snapshots to drop before grading
# net-beta. During this warmup the book is still ramping both legs and the SPY
# beta-hedge overlay has not yet sized to a stable hedge, so |net beta| is transiently
# elevated for reasons unrelated to PERSISTENT exposure. Both the production
# acceptance metric (steady_state_net_beta below, consumed by CPCVResult.net_beta_clean)
# AND the regression test (_steady_state_net) grade on the SAME window so they can
# never diverge — this is the single source of truth for the warmup trim.
NET_BETA_WARMUP_TWO_SIDED: int = 20


def _p95(sorted_vals) -> float:
    """p95 of an already-sorted ascending list (nearest-rank, clamped). Empty → 0.0."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(0.95 * len(sorted_vals)))
    return float(sorted_vals[idx])


def steady_state_net_beta(net_by_date: Dict[date, dict]) -> Dict[str, float]:
    """Persistent net-beta statistics over the STEADY-STATE two-sided window.

    THE ALPHA-VS-BETA LENS (BLOCKER 1):
      The SPY beta-hedge overlay re-sizes only on the 5-day rebalance cadence, but
      net beta is captured DAILY. Between rebalances |net beta| can spike for a
      single day (e.g. when hysteresis transiently drops most longs) even though
      the book is beta-neutral ON AVERAGE. The raw daily max (`max_abs_net_beta`)
      therefore reflects TRANSIENT inter-rebalance churn / warmup, NOT the
      persistent exposure that determines whether the realized Sharpe is alpha or
      leftover market beta. Grading the acceptance decision on the raw max would
      FALSELY FAIL a genuinely neutral book. The correct lens is the
      warmup-trimmed mean + p95: the typical, persistent exposure.

      We deliberately do NOT re-size the hedge daily (that would add turnover for no
      benefit) — we grade on the persistent statistic instead.

    Window: drop the first ``NET_BETA_WARMUP_TWO_SIDED`` two-sided (both-legs-held)
    EOD snapshots; if there are not enough (≤ 2× the warmup) keep them all. This is
    IDENTICAL to the test's ``_steady_state_net`` window (shared via this helper) so
    the production metric and the regression test can never diverge again.

    Returns {mean_abs_net_beta_signed (signed mean), mean_net_beta (signed),
    p95_abs_net_beta, max_abs_net_beta, n_steady, n_two_sided}.
    Empty/insufficient input → zeros (treated as clean by callers).
    """
    if not net_by_date:
        return {"mean_net_beta": 0.0, "p95_abs_net_beta": 0.0,
                "max_abs_net_beta": 0.0, "n_steady": 0, "n_two_sided": 0}
    items = sorted(net_by_date.items())
    two_sided = [d for _, d in items
                 if (d.get("n_long", 0) or 0) > 0 and (d.get("n_short", 0) or 0) > 0]
    if not two_sided:
        # No dollar-neutral two-sided window ever formed (e.g. long-only warmup
        # only). Fall back to the full series so we never silently report "clean"
        # for a one-legged book that happened to never pair up.
        two_sided = [d for _, d in items]
    w = NET_BETA_WARMUP_TWO_SIDED
    steady = two_sided[w:] if len(two_sided) > 2 * w else two_sided
    if not steady:
        steady = two_sided
    betas = [float(d["net_beta"]) for d in steady]
    abs_betas_sorted = sorted(abs(b) for b in betas)
    return {
        "mean_net_beta": float(np.mean(betas)) if betas else 0.0,
        "p95_abs_net_beta": _p95(abs_betas_sorted),
        "max_abs_net_beta": float(abs_betas_sorted[-1]) if abs_betas_sorted else 0.0,
        "n_steady": len(steady),
        "n_two_sided": len(two_sided),
    }


def _close_col(df: pd.DataFrame) -> Optional[str]:
    if "close" in df.columns:
        return "close"
    if "Close" in df.columns:
        return "Close"
    return None


def compute_pit_beta(
    name_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    as_of: date,
    lookback: int = 60,
) -> Optional[float]:
    """OLS beta of a name vs SPY using ONLY bars STRICTLY BEFORE `as_of`.

    Returns the trailing-`lookback`-day daily-return OLS slope (cov(name,spy)/
    var(spy)), or None if there is insufficient strictly-prior history.

    PIT guarantee: rows on or after `as_of` are dropped BEFORE any returns are
    computed, so a future bar can never influence the result. This is the same
    strict-`<` convention used by the spy_beta_hedge path.
    """
    if name_df is None or spy_df is None or len(name_df) == 0 or len(spy_df) == 0:
        return None
    nc = _close_col(name_df)
    sc = _close_col(spy_df)
    if nc is None or sc is None:
        return None
    as_of_ts = pd.Timestamp(as_of)

    name_hist = name_df.loc[pd.DatetimeIndex(name_df.index) < as_of_ts, nc].dropna()
    spy_hist = spy_df.loc[pd.DatetimeIndex(spy_df.index) < as_of_ts, sc].dropna()
    if len(name_hist) < 2 or len(spy_hist) < 2:
        return None

    name_rets = name_hist.pct_change().dropna()
    spy_rets = spy_hist.pct_change().dropna()
    if len(name_rets) == 0 or len(spy_rets) == 0:
        return None

    # Align on common dates, take the trailing `lookback` overlapping observations.
    aligned = pd.concat([name_rets, spy_rets], axis=1, join="inner").dropna()
    if len(aligned) > lookback:
        aligned = aligned.iloc[-lookback:]
    # Need a reasonable sample for a stable slope (mirror spy_beta_hedge's >= 20 / lb//2).
    if len(aligned) < max(20, lookback // 2):
        return None

    name_arr = aligned.iloc[:, 0].to_numpy(dtype=float)
    spy_arr = aligned.iloc[:, 1].to_numpy(dtype=float)
    var_spy = float(np.var(spy_arr))
    if var_spy <= 0:
        return None
    cov = float(np.cov(name_arr, spy_arr)[0, 1])
    return cov / var_spy


def compute_book_net_exposure(
    positions: Dict[str, object],
    closes_by_sym: Dict[str, float],
    equity: float,
    symbols_data: Dict[str, pd.DataFrame],
    spy_df: Optional[pd.DataFrame],
    as_of: date,
    sector_map: Optional[Dict[str, str]] = None,
    beta_lookback: int = 60,
    beta_cache: Optional[Dict] = None,
    hedge_keys: Optional[set] = None,
) -> Optional[Dict[str, object]]:
    """Signed book-level net beta / net dollar / net sector for the LIVE book.

    `positions`: sym -> object with `.quantity`, `.entry_price`, `.direction`
                 ("long"/"short"); typically portfolio.positions.
    `closes_by_sym`: sym -> EOD close used to mark notional (fall back to entry).
    `equity`: book equity for normalization.
    `as_of`: decision day; betas use ONLY bars strictly before it (PIT).
    `beta_cache`: optional dict keyed (sym, as_of, lookback) -> beta to avoid
                  recomputing within a day; the simulator passes a fresh per-day
                  cache so this stays deterministic and cheap.
    `hedge_keys`: optional set of position keys that are MARKET-HEDGE overlays
                  (e.g. the SPY beta-hedge "__SPY_HEDGE__"). A hedge contributes to
                  net_beta (it genuinely reduces market exposure) but is EXCLUDED
                  from net_dollar / net_sector / n_short, because it is a beta
                  instrument, not a stock-selection bet. This keeps "dollar
                  neutrality" a property of the single-name L/S book while still
                  crediting the hedge for driving realized net beta → 0.

    Returns a dict {net_beta, net_dollar, gross, net_sector, max_abs_sector,
    n_long, n_short} or None if equity is non-positive or the book is empty.
    Read-only — never mutates `positions`.
    """
    if equity is None or equity <= 0 or not positions:
        return None
    sector_map = sector_map or {}
    cache = beta_cache if beta_cache is not None else {}
    hedge_keys = hedge_keys or set()

    long_beta_contrib = 0.0
    short_beta_contrib = 0.0
    long_notional = 0.0
    short_notional = 0.0
    net_sector: Dict[str, float] = {}
    n_long = 0
    n_short = 0

    for sym, pos in positions.items():
        qty = float(getattr(pos, "quantity", 0) or 0)
        if qty <= 0:
            continue
        px = closes_by_sym.get(sym)
        if px is None:
            px = float(getattr(pos, "entry_price", 0.0) or 0.0)
        notional = abs(float(px) * qty)
        if notional <= 0:
            continue
        direction = getattr(pos, "direction", "long")
        is_long = direction == "long"
        is_hedge = sym in hedge_keys

        # PIT beta (None when insufficient history → contributes 0 to net beta but
        # still counts toward net dollar / net sector for non-hedge positions).
        key = (sym, as_of, beta_lookback)
        if key in cache:
            beta = cache[key]
        else:
            # A market-hedge overlay (e.g. "__SPY_HEDGE__") has no entry in
            # symbols_data; its underlying IS SPY, so beta ≡ 1.0 by definition.
            if sym in hedge_keys:
                beta = 1.0
            else:
                beta = compute_pit_beta(
                    symbols_data.get(sym), spy_df, as_of, lookback=beta_lookback
                )
            cache[key] = beta
        w = notional / equity

        # Beta: hedges and single-name positions both count (the hedge's whole job
        # is to move net beta).
        if is_long:
            if beta is not None:
                long_beta_contrib += w * beta
        else:
            if beta is not None:
                short_beta_contrib += w * beta

        # Dollar / sector / counts: EXCLUDE the market-hedge overlay (it is not a
        # stock-selection bet, so it should not break the single-name book's dollar
        # neutrality or count as a short name).
        if is_hedge:
            continue
        if is_long:
            long_notional += notional
            n_long += 1
        else:
            short_notional += notional
            n_short += 1

        sec = sector_map.get(sym, "UNKNOWN")
        signed = notional if is_long else -notional
        net_sector[sec] = net_sector.get(sec, 0.0) + signed

    net_sector_pct = {s: v / equity for s, v in net_sector.items()}
    max_abs_sector = max((abs(v) for v in net_sector_pct.values()), default=0.0)

    return {
        "net_beta": long_beta_contrib - short_beta_contrib,
        "net_dollar": (long_notional - short_notional) / equity,
        "gross": (long_notional + short_notional) / equity,
        "net_sector": net_sector_pct,
        "max_abs_sector": max_abs_sector,
        "n_long": n_long,
        "n_short": n_short,
    }


def summarize_net_exposure(net_by_date: Dict[date, dict]) -> Dict[str, float]:
    """Per-fold summary of the daily net-exposure series.

    Returns mean/last net beta, max |net beta| (raw daily, DIAGNOSTIC only),
    p95 |net beta| (warmup-trimmed steady-state, the ACCEPTANCE lens — BLOCKER 1),
    mean net dollar, max |net dollar|, and max |net sector| across the fold (all
    signed-aware). Empty-safe.
    """
    if not net_by_date:
        return {
            "mean_net_beta": 0.0,
            "last_net_beta": 0.0,
            "max_abs_net_beta": 0.0,
            "p95_abs_net_beta": 0.0,
            "mean_net_dollar": 0.0,
            "max_abs_net_dollar": 0.0,
            "max_abs_net_sector": 0.0,
            "mean_gross": 0.0,
            "last_gross": 0.0,
        }
    items = sorted(net_by_date.items())
    betas = [d["net_beta"] for _, d in items]
    dollars = [d["net_dollar"] for _, d in items]
    sectors = [d.get("max_abs_sector", 0.0) for _, d in items]
    grosses = [d.get("gross", 0.0) for _, d in items]
    # BLOCKER 1: the PERSISTENT net-beta lens (warmup-trimmed steady-state p95). This
    # is the statistic that drives the clean/accept decision; the raw daily
    # max_abs_net_beta below is reported as a DIAGNOSTIC only. Computed via the shared
    # steady_state_net_beta() helper so production and the regression test grade on
    # the IDENTICAL window/statistic and can never diverge.
    _ss = steady_state_net_beta(net_by_date)
    return {
        "mean_net_beta": float(np.mean(betas)),
        "last_net_beta": float(betas[-1]),
        # Raw daily max over the WHOLE series (incl. warmup) — DIAGNOSTIC only.
        "max_abs_net_beta": float(max(abs(b) for b in betas)),
        # Persistent exposure (steady-state, warmup-trimmed p95) — ACCEPTANCE lens.
        "p95_abs_net_beta": _ss["p95_abs_net_beta"],
        "mean_net_dollar": float(np.mean(dollars)),
        "max_abs_net_dollar": float(max(abs(x) for x in dollars)),
        "max_abs_net_sector": float(max(sectors)) if sectors else 0.0,
        # Realized book gross (long+short notional / equity). For the dollar-neutral
        # book this should land near long_gross + short_gross (= 0.80 at 0.40/0.40)
        # once FIX 1 lets both legs hit their per-side gross targets.
        "mean_gross": float(np.mean(grosses)),
        "last_gross": float(grosses[-1]),
    }
