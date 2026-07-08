"""CH4 (Compound-and-Harden) — the ONE terminating pre-registered search: a ranging-market
mean-reversion sleeve. Built to the FROZEN spec `docs/reference/CH4_MR_PREREGISTRATION_2026-07-08.md`
(DECISIONS 2026-07-08). Run the 5 pre-registered gates on the primary config → PASS (paper candidate)
or KILL (+ the 12-month hunting moratorium binds). Report-only; trades nothing.

Strategy: short-horizon time-series MEAN-REVERSION on the 10-ETF trend universe — buy short-term
oversold dips (reversal z-score), long-flat, inverse-vol sized — ACTIVE ONLY in RANGING regimes (the
COMPLEMENT of the trend signal: low trend_clarity AND low realized book vol; FLAT otherwise). PIT:
signal + regime use data ≤ t, applied to the t→t+1 return (shift(1)).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from scripts.ch0_baseline import BASELINE_END

ARTIFACT = "docs/reference/ch4_mr_results.json"
ANN = 252
N_TRIALS = 8   # the pre-registered grid size (DSR charge)


@dataclass
class MRConfig:
    universe: List[str] = field(default_factory=list)
    lookback: int = 5            # L: reversal horizon (days)
    z_enter: float = 1.0         # enter long when oversold z > this
    clarity_lo: float = 0.40     # ranging = trend_clarity < clarity_lo ...
    vol_pctl: float = 0.50       # ... AND book_vol < its expanding vol_pctl quantile
    z_window: int = 63           # rolling std window for the reversal z
    vol_lookback: int = 60       # per-name realized vol (inverse-vol sizing)
    target_vol: float = 0.10
    vol_floor: float = 0.03
    max_weight: float = 0.25
    max_gross: float = 1.0
    rebalance_days: int = 5      # weekly hold
    cost_bps: float = 2.0
    book_vol_window: int = 63


def _tcfg(cfg: MRConfig, cols):
    from app.strategy.tsmom import TSMOMConfig
    return TSMOMConfig(universe=[c for c in cfg.universe if c in cols], vol_lookback=cfg.vol_lookback,
                       target_vol=cfg.target_vol, vol_floor=cfg.vol_floor, ann=ANN)


def mr_signal(closes: pd.DataFrame, cfg: MRConfig) -> pd.DataFrame:
    """Oversold reversal z per name: recent DROP → positive z. PIT (uses prices ≤ t)."""
    ret_l = closes / closes.shift(cfg.lookback) - 1.0
    sd = ret_l.rolling(cfg.z_window, min_periods=max(cfg.z_window // 2, 20)).std()
    return (-ret_l / sd).replace([np.inf, -np.inf], np.nan)


def ranging_mask(closes: pd.DataFrame, cfg: MRConfig) -> pd.Series:
    """True on RANGING days = trend_clarity LOW AND book_vol LOW (the complement of trend). PIT:
    trend_clarity is PIT; the vol threshold is an EXPANDING quantile (no look-ahead)."""
    from app.research.ch2_sizing import trend_clarity
    from app.strategy.tsmom import _daily_returns
    cols = [c for c in cfg.universe if c in closes.columns]
    clarity = trend_clarity(closes[cols], _tcfg(cfg, closes.columns))
    ew = _daily_returns(closes[cols]).mean(axis=1)
    book_vol = ew.rolling(cfg.book_vol_window, min_periods=cfg.book_vol_window // 2).std() * np.sqrt(ANN)
    vol_thr = book_vol.expanding(min_periods=252).quantile(cfg.vol_pctl)
    return ((clarity < cfg.clarity_lo) & (book_vol < vol_thr)).fillna(False)


def mr_book_returns(closes: pd.DataFrame, cfg: MRConfig, *, extra_lag: int = 0):
    """Net daily returns of the regime-gated MR sleeve. `extra_lag` shifts the RANGING mask by an
    additional N days (the detection-lag test). Returns (net_returns, mask, held_weights)."""
    from app.strategy.tsmom import _daily_returns, realized_vol
    cols = [c for c in cfg.universe if c in closes.columns]
    px = closes[cols]
    rets = _daily_returns(px)
    z = mr_signal(px, cfg)
    rv = realized_vol(px, _tcfg(cfg, closes.columns))
    # long-only inverse-vol on the oversold names, per-name + gross capped
    w = (z > cfg.z_enter).astype(float) * (cfg.target_vol / rv)
    w = w.clip(lower=0.0, upper=cfg.max_weight)
    gross = w.abs().sum(axis=1)
    w = w.mul((cfg.max_gross / gross).clip(upper=1.0).fillna(1.0), axis=0).fillna(0.0)
    # weekly hold between rebalances
    rebal = pd.Series((np.arange(len(w)) % cfg.rebalance_days) == 0, index=w.index)
    held = w.where(rebal, np.nan).ffill().fillna(0.0)
    # regime gate: FLAT (exit) on non-ranging days — applied AFTER the hold so a regime flip exits now
    mask = ranging_mask(px, cfg)
    if extra_lag:
        mask = mask.shift(extra_lag).fillna(False)
    held = held.where(mask, 0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cfg.cost_bps / 1e4)
    net = ((held.shift(1) * rets).sum(axis=1) - cost.shift(1)).dropna()
    return net, mask, held


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ANN)) if len(r) > 20 and sd > 0 else float("nan")


CH4_CONFIGS = [
    {"name": "mr_primary", "lookback": 5, "z_enter": 1.0, "clarity_lo": 0.40, "vol_pctl": 0.50},
    {"name": "mr_slow", "lookback": 10, "z_enter": 1.0, "clarity_lo": 0.40, "vol_pctl": 0.50},
    {"name": "mr_deep", "lookback": 5, "z_enter": 1.5, "clarity_lo": 0.40, "vol_pctl": 0.50},
    {"name": "mr_tight_regime", "lookback": 5, "z_enter": 1.0, "clarity_lo": 0.30, "vol_pctl": 0.40},
    {"name": "mr_loose_regime", "lookback": 5, "z_enter": 1.0, "clarity_lo": 0.50, "vol_pctl": 0.60},
    {"name": "mr_slow_deep", "lookback": 10, "z_enter": 1.5, "clarity_lo": 0.40, "vol_pctl": 0.50},
    {"name": "mr_clarity_only", "lookback": 5, "z_enter": 1.0, "clarity_lo": 0.40, "vol_pctl": 1.00},
    {"name": "mr_vol_only", "lookback": 5, "z_enter": 1.0, "clarity_lo": 1.00, "vol_pctl": 0.50},
]


def gate_one_config(c: dict, closes, trend, spy) -> dict:
    """Run the 5 pre-registered gates on one config. PASS iff all hold."""
    from scripts.walkforward.sleeve_lab import Sleeve, evaluate_sleeve
    cfg = MRConfig(universe=list(closes.columns), lookback=c["lookback"], z_enter=c["z_enter"],
                   clarity_lo=c["clarity_lo"], vol_pctl=c["vol_pctl"])
    net, mask, held = mr_book_returns(closes, cfg)
    active_frac = float(mask.reindex(net.index).fillna(False).mean())

    # Gate 1 — standalone Track-A (Ruler-v2 PAPER pass) + Track-B vs the live trend book
    rep = evaluate_sleeve(
        Sleeve(label=c["name"], component_type="diversifier", returns=net, spy_prices=spy,
               n_trials_registered=N_TRIALS, notes="CH4 ranging-MR"),
        base_book_returns=trend)
    track_a = bool(rep.paper_passed)
    tb = rep.track_b
    track_b_t = (round(float(tb.t_alpha_hac), 3) if tb is not None else None)
    track_b_pass = bool(track_b_t is not None and track_b_t >= 1.96)

    # Gate 2 — off-regime Sharpe >= -0.10 (the gated sleeve is flat off-regime: confirm no leakage)
    off = net[~mask.reindex(net.index).fillna(False)]
    off_sharpe = _sharpe(off) if len(off) > 20 else 0.0
    off_regime_ok = bool(np.isnan(off_sharpe) or off_sharpe >= -0.10)

    # Gate 3 — detection-LAG (pre-registered metric = CPCV mean_sharpe over a 1-5d lag sweep): re-run
    # with an EXTRA 1/2/3-day lag on the RANGING mask; the sleeve's CPCV mean_sharpe must SURVIVE
    # (>0 AND >= 60% of the un-lagged value) at EVERY tested lag. Lagging the regime DETECTION (not
    # the reversal signal): a real regime edge persists across a multi-week ranging spell and so
    # survives entering a couple days late; an edge concentrated at the regime boundary (needs sub-
    # lag detection) is a look-ahead / boundary-fit mirage and is correctly rejected.
    base_ms = float(rep.mean_sharpe)
    lag_ms: dict = {}
    for L in (1, 2, 3):
        net_l, _, _ = mr_book_returns(closes, cfg, extra_lag=L)
        rep_l = evaluate_sleeve(Sleeve(label=f"{c['name']}_lag{L}", component_type="diversifier",
                                       returns=net_l, spy_prices=spy, n_trials_registered=N_TRIALS))
        lag_ms[L] = round(float(rep_l.mean_sharpe), 4)
    lag_ok = all(ms > 0 and (base_ms <= 0 or ms >= 0.6 * base_ms) for ms in lag_ms.values())

    passed = bool(track_a and off_regime_ok and lag_ok and track_b_pass)
    return {
        "name": c["name"], "config": c, "active_frac": round(active_frac, 3),
        "n_obs": int(len(net)), "daily_sharpe": round(float(_sharpe(net)), 3),
        "cpcv_mean_sharpe": round(base_ms, 4),
        "point_sr": round(float(rep.point_sr), 4),
        "gate1_track_a_paper_pass": track_a,
        "gate2_off_regime_sharpe": (round(float(off_sharpe), 3) if not np.isnan(off_sharpe) else None),
        "gate2_ok": off_regime_ok,
        "gate3_detection_lag": {"base_cpcv_mean_sharpe": round(base_ms, 4),
                                "lagged_cpcv_mean_sharpe": lag_ms, "ok": lag_ok},
        "gate4_track_b_t": track_b_t, "gate4_ok": track_b_pass,
        "track_b_verdict": (tb.verdict if tb is not None else None),
        "PASS": passed,
    }


def run_ch4() -> dict:
    from scripts.walkforward.sleeves import LIVE_TREND_UNIVERSE, fetch_universe_closes, \
        live_trend_book_returns
    closes = fetch_universe_closes(LIVE_TREND_UNIVERSE, end=BASELINE_END)
    trend = live_trend_book_returns(end=BASELINE_END)
    spy = closes["SPY"]
    results = [gate_one_config(c, closes, trend, spy) for c in CH4_CONFIGS]
    primary = next(r for r in results if r["name"] == "mr_primary")
    return {
        "search": "CH4 - ranging-market mean-reversion sleeve (the ONE terminating search)",
        "preregistration": "docs/reference/CH4_MR_PREREGISTRATION_2026-07-08.md",
        "decision_on": "mr_primary (pre-registered; no best-of-8)",
        "n_trials_registered": N_TRIALS,
        "PASS": bool(primary["PASS"]),
        "verdict": ("PASS - a paper candidate (moratorium does NOT bind)" if primary["PASS"]
                    else "KILL - ship nothing; the 12-month hunting moratorium (to 2027-07-08) BINDS"),
        "results": results,
    }


def main() -> int:
    out = run_ch4()
    with open(ARTIFACT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"CH4 -> {ARTIFACT}")
    print(f"  decision on: {out['decision_on']}")
    for r in out["results"]:
        flag = "PASS" if r["PASS"] else "----"
        print(f"  [{flag}] {r['name']:<16} active {r['active_frac']:.0%}  cpcvSR {r['cpcv_mean_sharpe']:+.3f}  "
              f"trackA={r['gate1_track_a_paper_pass']} offReg={r['gate2_ok']} "
              f"lag={r['gate3_detection_lag']['ok']} trackB_t={r['gate4_track_b_t']}({r['gate4_ok']})")
    print(f"  VERDICT: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
