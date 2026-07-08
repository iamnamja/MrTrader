"""CH2 (Compound-and-Harden) — antifragile gross-sizing multipliers on the ETF trend book.

Each candidate is a CONTINUOUS per-date gross multiplier m[t] applied to the constant-gross trend
book. The DUAL GATE (from the CH0a baseline + its Opus review): a multiplier SHIPS only if the
governed book (base × m[t]) BEATS the frozen CH0a constant-gross baseline out-of-sample on the
SAME CPCV gate (mean_sharpe > 0.7009) AND does NOT regress the BEAR regime-conditional Sharpe
(>= -0.77), with the multiplier's parameters charged to the DSR trial count. Else ship nothing —
you've only added a knob.

This module is the shared GATE HARNESS + the individual multiplier SIGNALS. It is RESEARCH /
report-only: it produces a gate verdict + a versioned artifact. Wiring a PASSING multiplier into
the live path (shadow-first) is a separate, gated step.

PIT discipline: every signal function returns an ALREADY-SHIFTED multiplier — m[t] uses only data
through t-1, so that scaling day-t's return (which the base book already earns PIT via held.shift(1)
inside tsmom_backtest) introduces no look-ahead. The gate harness just multiplies base × m.

CH2b — correlation-regime gross scaling: cut gross when the realized cross-sectional correlation of
the universe rises toward 1 (every position becomes one bet; "diversification" is illusory). A
continuous de-risk multiplier in [floor, 1].
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Single source of the pinned end (import CH0a's constant, don't redefine) so the base book stays
# byte-identical to the frozen baseline (apples-to-apples). MINOR-3 from the CH2b review.
from scripts.ch0_baseline import BASELINE_END  # noqa: E402  (pinned baseline end date)

BASELINE_ARTIFACT = "docs/reference/ch0_trend_baseline.json"
RESULTS_ARTIFACT = "docs/reference/ch2_sizing_results.json"
ANN = 252
MULT_TURNOVER_COST_BPS = 2.0   # one-way cost charged on |Δm| (re-sizing gross) = TSMOM cost_bps


# ──────────────────────────────────────────────────────────────────────────────────
# Base book + the frozen gate bar
# ──────────────────────────────────────────────────────────────────────────────────
def build_base() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """The constant-gross trend book returns (identical to CH0a) + SPY prices + the close panel
    the multiplier signals are computed from — all pinned to BASELINE_END."""
    from scripts.walkforward.sleeves import (LIVE_TREND_UNIVERSE, fetch_universe_closes,
                                             live_trend_book_returns)
    base = live_trend_book_returns(end=BASELINE_END)
    closes = fetch_universe_closes(LIVE_TREND_UNIVERSE, end=BASELINE_END)
    return base, closes["SPY"], closes


def load_baseline_bar() -> Dict[str, float]:
    """The frozen CH0a gate bar the governed book must clear: CPCV mean_sharpe + BEAR regime SR."""
    with open(BASELINE_ARTIFACT) as f:
        d = json.load(f)
    bear = d.get("regime_conditional_sharpe", {}).get("BEAR", {}).get("sharpe")
    return {"mean_sharpe": float(d["cpcv_gate"]["mean_sharpe"]),
            "bear_sharpe": (float(bear) if bear is not None else None)}


# ──────────────────────────────────────────────────────────────────────────────────
# The DUAL gate harness (same CPCV path CH0a was frozen on)
# ──────────────────────────────────────────────────────────────────────────────────
def _daily_sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1) if len(r) > 2 else 0.0
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")


def paired_delta_sharpe_pvalue(governed: pd.Series, base: pd.Series, *,
                               n_boot: int = 1000, block: int = 20, seed: int = 0) -> float:
    """Stationary-block-bootstrap p-value for H0: governed Sharpe ≤ base Sharpe, PAIRED on the same
    dates (the two books differ only by the multiplier, so a paired resample isolates the sizing
    effect from shared market noise). Returns P(ΔSharpe ≤ 0) — small = the improvement is unlikely
    to be sampling noise. Guards MAJOR-1: a point 'beat' that is within the noise band won't ship."""
    df = pd.concat([governed.rename("g"), base.rename("b")], axis=1).dropna()
    g, b = df["g"].to_numpy(), df["b"].to_numpy()
    n = len(g)
    if n < block * 3:
        return 1.0
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    le = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        if _daily_sharpe(g[idx]) - _daily_sharpe(b[idx]) <= 0:
            le += 1
    return le / n_boot


def gate_decision(mean_sharpe: float, bear: Optional[float], bar: Dict[str, float],
                  bear_tol: float = 1e-6) -> tuple[bool, bool, bool]:
    """Pure DUAL-gate decision → (beats_baseline, bear_no_regression, both). PASS ⇔ governed
    mean_sharpe strictly beats the baseline AND the governed BEAR Sharpe does not regress below the
    baseline BEAR (within tol). None BEAR → treated as non-regressing (the mean prong still binds).
    NOTE: the full harness also requires the improvement to be significant (see gate_multiplier)."""
    beats = float(mean_sharpe) > float(bar["mean_sharpe"])
    bear_ok = (bear is None or bar["bear_sharpe"] is None
               or float(bear) >= bar["bear_sharpe"] - bear_tol)
    return beats, bear_ok, (beats and bear_ok)


def _evaluate(returns: pd.Series, spy: pd.Series, label: str, n_trials: int,
              component_type: str = "risk_premium"):
    from scripts.walkforward.sleeve_lab import Sleeve, evaluate_sleeve
    return evaluate_sleeve(Sleeve(label=label, component_type=component_type, returns=returns,
                                  spy_prices=spy, n_trials_registered=max(1, int(n_trials)),
                                  notes=f"CH2 {label}"))


def governed_returns(raw_mult: pd.Series, base_returns: pd.Series, *, apply_lag: bool = True,
                     cost_bps: float = MULT_TURNOVER_COST_BPS) -> tuple[pd.Series, pd.Series]:
    """Apply a candidate signal to the base book with the harness-owned PIT lag + re-sizing cost.
    Returns (governed_returns, effective_multiplier). The HARNESS shifts (apply_lag): raw_mult is
    the signal through t → m[t] uses ≤ t-1. m≡1 reproduces base exactly. Pure."""
    aligned = raw_mult.reindex(base_returns.index)
    m = (aligned.shift(1) if apply_lag else aligned).fillna(1.0)
    turnover = m.diff().abs().fillna(0.0) * (float(cost_bps) / 1e4)
    return (base_returns * m - turnover).dropna(), m


def gate_multiplier(raw_mult: pd.Series, base_returns: pd.Series, spy: pd.Series, *,
                    label: str, n_trials: int, base_mean_sharpe: float,
                    apply_lag: bool = True, cost_bps: float = MULT_TURNOVER_COST_BPS,
                    bear_tol: float = 1e-6, sig_alpha: float = 0.05, n_boot: int = 1000,
                    component_type: str = "risk_premium") -> dict:
    """Score a candidate multiplier against the constant-gross baseline through the SAME CPCV path
    CH0a was frozen on. The HARNESS owns the PIT lag (MAJOR-2): `raw_mult` is the signal computed
    with info THROUGH t; the harness shifts it by 1 day so m[t] uses only ≤ t-1 — a signal author
    cannot leak by forgetting to shift. governed[t] = m[t]·base[t] − |Δm[t]|·cost (the re-sizing
    turnover, MINOR-1).

    TRIPLE gate (ship only if all hold):
      (a) governed mean_sharpe > base_mean_sharpe (FULL PRECISION, evaluated in-run — MAJOR-1),
      (b) the improvement is SIGNIFICANT (paired block-bootstrap ΔSharpe p < sig_alpha), and
      (c) BEAR regime-conditional Sharpe does not regress below the CH0a baseline.
    Also reports a shift-sensitivity diagnostic (daily Sharpe with one MORE day of lag): a large
    drop flags a signal whose edge is suspiciously timing-precise (possible residual leak)."""
    from scripts.ch0_baseline import regime_conditional_sharpe

    governed, m = governed_returns(raw_mult, base_returns, apply_lag=apply_lag, cost_bps=cost_bps)
    rep = _evaluate(governed, spy, label, n_trials, component_type)
    bear = regime_conditional_sharpe(governed).get("BEAR", {}).get("sharpe")
    bar = load_baseline_bar()
    p_improve = paired_delta_sharpe_pvalue(governed, base_returns, n_boot=n_boot)

    # shift-sensitivity: one EXTRA day of lag (cheap daily-Sharpe proxy, no extra CPCV).
    gov_xlag, _ = governed_returns(raw_mult.reindex(base_returns.index).shift(1), base_returns,
                                   apply_lag=apply_lag, cost_bps=cost_bps)
    lag_sensitivity = _daily_sharpe(governed) - _daily_sharpe(gov_xlag)

    beats, bear_ok, _ = gate_decision(rep.mean_sharpe, bear,
                                      {"mean_sharpe": base_mean_sharpe,
                                       "bear_sharpe": bar["bear_sharpe"]}, bear_tol)
    significant = p_improve < sig_alpha
    passed = beats and significant and bear_ok
    return {
        "label": label,
        "n_trials_registered": max(1, int(n_trials)),
        "mean_sharpe": round(float(rep.mean_sharpe), 4),
        "mean_sharpe_full": float(rep.mean_sharpe),   # unrounded (for full-precision deltas)
        "baseline_mean_sharpe": round(float(base_mean_sharpe), 4),
        "d_mean_sharpe": round(float(rep.mean_sharpe) - float(base_mean_sharpe), 4),
        "improve_pvalue": round(float(p_improve), 4),
        "bear_sharpe": (round(float(bear), 4) if bear is not None else None),
        "baseline_bear_sharpe": bar["bear_sharpe"],
        "path_sharpe_tstat": round(float(rep.path_sharpe_tstat), 4),
        "point_sr": round(float(rep.point_sr), 4),
        "worst_regime_sharpe": (round(float(rep.worst_regime_sharpe), 4)
                                if rep.worst_regime_sharpe is not None else None),
        "avg_mult": round(float(m.reindex(governed.index).mean()), 4),
        "min_mult": round(float(m.reindex(governed.index).min()), 4),
        "lag_sensitivity": round(float(lag_sensitivity), 4),
        "beats_baseline": bool(beats),
        "improvement_significant": bool(significant),
        "bear_no_regression": bool(bear_ok),
        "PASS": bool(passed),
    }


# ──────────────────────────────────────────────────────────────────────────────────
# CH2b — correlation-regime gross scaling
# ──────────────────────────────────────────────────────────────────────────────────
def _daily_returns_safe(prices: pd.DataFrame) -> pd.DataFrame:
    from app.strategy.tsmom import _daily_returns
    return _daily_returns(prices)


def held_book_corr(prices: pd.DataFrame, weights: pd.DataFrame, window: int) -> pd.Series:
    """Per-date NAV-weighted average SIGNED pairwise correlation of the HELD book (the backtest
    twin of the live per_name_gate metric). For each date t: the trailing-`window` correlation
    matrix of daily returns (window ENDS at t → PIT before the caller shifts) weighted by the
    book's own weights at t:  Σ_{i<j} w_i·w_j·ρ_ij / Σ_{i<j} w_i·w_j.

    High ⇒ the held book has collapsed to one bet (diversifiers dropped / crisis correlation);
    low/negative ⇒ genuinely diversified. Only names that BOTH have a full corr window AND a
    non-zero weight contribute (so the raw universe's idle diversifiers don't wash it out — that
    was the bug: unweighted full-universe corr stays < 0.4 and never fires). NaN until warmup."""
    rets = _daily_returns_safe(prices)
    cols = list(rets.columns)
    arr = rets[cols].to_numpy(dtype=float)
    W = weights.reindex(index=rets.index, columns=cols).to_numpy(dtype=float)
    idx = rets.index
    out = np.full(len(idx), np.nan)
    for e in range(window, len(idx) + 1):
        t = e - 1
        win = arr[e - window:e]
        keep = ~np.isnan(win).any(axis=0)
        wt = np.where(np.isnan(W[t]), 0.0, W[t])
        wt = np.where(keep, np.abs(wt), 0.0)          # long-only book; guard sign anyway
        nz = int(np.count_nonzero(wt))
        if nz == 0:
            continue                                  # no book held → no signal (m stays 1.0)
        if nz == 1:
            out[t] = 1.0                              # single-name book = maximal concentration
            continue
        c = np.corrcoef(win[:, keep], rowvar=False)
        wk = wt[keep]
        num = den = 0.0
        k = wk.size
        for i in range(k):
            if wk[i] == 0.0:
                continue
            for j in range(i + 1, k):
                cij = c[i, j]
                if wk[j] != 0.0 and cij == cij:       # both weighted + not NaN
                    num += wk[i] * wk[j] * cij
                    den += wk[i] * wk[j]
        if den > 0:
            out[t] = num / den
    return pd.Series(out, index=idx)


def correlation_gross_multiplier(prices: pd.DataFrame, weights: pd.DataFrame, *, window: int = 63,
                                 corr_lo: float = 0.60, corr_hi: float = 0.90,
                                 floor: float = 0.50) -> pd.Series:
    """CH2b multiplier: as the HELD-book weighted correlation rises from corr_lo→corr_hi, scale
    gross linearly 1.0→floor (cut when the book collapses to one bet); m=1 below corr_lo, m=floor
    above corr_hi. Returns the signal computed with info THROUGH t (UN-shifted, NaN in warmup) —
    the gate harness owns the PIT 1-day lag (MAJOR-2), so a caller can't leak by forgetting it."""
    corr = held_book_corr(prices, weights, window)
    frac = ((corr - corr_lo) / max(corr_hi - corr_lo, 1e-9)).clip(lower=0.0, upper=1.0)
    return 1.0 - (1.0 - floor) * frac


# One PRIMARY pre-registered config + 2 sensitivity variants (OPT-5 discipline: pre-register, no
# post-hoc flipping). Thresholds are on HELD-BOOK weighted correlation (structurally ~0.85 in
# equity-led weeks), so the cut band sits high (0.60→0.90). n_trials charged = configs tried (3).
CH2B_CONFIGS = [
    {"name": "ch2b_primary", "window": 63, "corr_lo": 0.60, "corr_hi": 0.90, "floor": 0.50},
    {"name": "ch2b_shorter_window", "window": 42, "corr_lo": 0.60, "corr_hi": 0.90, "floor": 0.50},
    {"name": "ch2b_gentler_band", "window": 63, "corr_lo": 0.70, "corr_hi": 0.95, "floor": 0.60},
]


def run_ch2b() -> dict:
    """Build the base book, gate every pre-registered CH2b config, and return the verdict block.
    PASS is decided on the PRIMARY config (the others are sensitivity, but all count toward DSR)."""
    from app.strategy.tsmom import TSMOMConfig, tsmom_weights
    base, spy, closes = build_base()
    cfg = TSMOMConfig(universe=[c for c in closes.columns])
    weights = tsmom_weights(closes[[c for c in cfg.universe if c in closes.columns]], cfg)
    n_trials = len(CH2B_CONFIGS)
    # Full-precision baseline through the SAME CPCV path (MAJOR-1): gate against THIS, not the
    # rounded 0.7009 in the artifact. m≡1 reproduces it exactly, so this equals the frozen bar.
    base_rep = _evaluate(base, spy, "ch2b_base", n_trials)
    base_mean_sharpe = float(base_rep.mean_sharpe)
    results = []
    for c in CH2B_CONFIGS:
        m = correlation_gross_multiplier(closes, weights, window=c["window"], corr_lo=c["corr_lo"],
                                         corr_hi=c["corr_hi"], floor=c["floor"])
        r = gate_multiplier(m, base, spy, label=c["name"], n_trials=n_trials,
                            base_mean_sharpe=base_mean_sharpe)
        r["config"] = c
        results.append(r)
    primary = next(r for r in results if r["label"] == "ch2b_primary")
    return {
        "multiplier": "CH2b — correlation-regime gross scaling",
        "gate": "governed must beat the base CPCV mean_sharpe (SIGNIFICANTLY, paired bootstrap) "
                "AND not regress BEAR",
        "base_mean_sharpe": round(base_mean_sharpe, 4),
        "n_trials_registered": n_trials,
        "primary_config": "ch2b_primary",
        "PASS": bool(primary["PASS"]),
        "verdict": ("SHIP (shadow-first)" if primary["PASS"]
                    else "KILL - does not beat constant-gross; ship nothing"),
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────────
# CH2c — trending-vs-whipsaw-aware crash governor
# ──────────────────────────────────────────────────────────────────────────────────
def vix_ratio_series(index: pd.Index) -> pd.Series:
    """VIX/VIX3M term-structure ratio aligned to `index` (ffill onto trading days). >1 =
    backwardation = acute stress. ^VIX3M starts ~2008, so 2007 is NaN (→ no stress → m=1)."""
    from scripts.walkforward.sleeves import fetch_vix_term
    vix, vix3m = fetch_vix_term(end=BASELINE_END)
    df = pd.concat([pd.Series(vix).rename("vix"), pd.Series(vix3m).rename("vix3m")],
                   axis=1, join="inner").dropna().sort_index()
    ratio = (df["vix"] / df["vix3m"])
    ratio.index = pd.to_datetime(ratio.index)
    return ratio.reindex(pd.DatetimeIndex(index), method="ffill")


def trend_clarity(prices: pd.DataFrame, cfg) -> pd.Series:
    """Book-wide trend CLARITY per date = mean over names of |ensemble-sign| ∈ [0,1]. Per name,
    |ens|=1 iff ALL lookbacks agree (a clean strong trend); <1 iff lookbacks conflict (choppy/
    reversing = whipsaw). High = broad clean trends; low = whipsaw. PIT (info through t).

    CAVEAT: tsmom_signals fills insufficient-history to 0, so a name in its OWN lookback warmup
    reads as clarity 0 (max whipsaw). Harmless for this universe (all 10 ETFs listed by 2007, and
    stress/VIX3M only exists from 2008 → the book is fully warmed before any cut can fire, and the
    warmup book is flat anyway). But a LATE-LISTING name added to the universe would spuriously
    read as whipsaw during its warmup — recheck this if the trend universe changes."""
    from app.strategy.tsmom import tsmom_signals
    return tsmom_signals(prices, cfg).abs().mean(axis=1)


def whipsaw_governor_multiplier(prices: pd.DataFrame, cfg, vix_ratio: pd.Series, *,
                                r_lo: float = 1.00, r_hi: float = 1.15,
                                derisk_to: float = 0.50) -> pd.Series:
    """CH2c multiplier: de-risk ONLY when the market is stressed AND the book's trends are choppy.
      stress[t]   = clip((ratio[t]-r_lo)/(r_hi-r_lo), 0, 1)      (backwardation depth)
      whipsaw[t]  = 1 - trend_clarity[t]                          (0 = clean trend, 1 = chop)
      m[t]        = 1 - (1-derisk_to)·stress[t]·whipsaw[t]
    So a broad TRENDING crash (stress high, whipsaw≈0) is NOT cut (trend is paying), while choppy
    stress (stress high, whipsaw high) cuts to ~derisk_to. Un-shifted (harness owns the lag)."""
    ratio = vix_ratio.reindex(prices.index, method="ffill")
    stress = ((ratio - r_lo) / max(r_hi - r_lo, 1e-9)).clip(lower=0.0, upper=1.0).fillna(0.0)
    whipsaw = (1.0 - trend_clarity(prices, cfg)).clip(lower=0.0, upper=1.0)
    return 1.0 - (1.0 - derisk_to) * stress * whipsaw


def plain_vix_governor_multiplier(vix_ratio: pd.Series, index: pd.Index, *,
                                  ratio_threshold: float = 1.00,
                                  derisk_to: float = 0.50) -> pd.Series:
    """The EXISTING live VIX-term crash governor (binary: cut to derisk_to whenever backwardated),
    as a raw un-shifted signal for the same harness — the CH2c reference ('does whipsaw-awareness
    beat the plain governor?'). Not double-shifted: we build the raw at-close signal and let the
    harness apply the 1-day lag (matching crash_governor.vix_term_multiplier's own shift)."""
    ratio = vix_ratio.reindex(pd.DatetimeIndex(index), method="ffill")
    return pd.Series(np.where(ratio > ratio_threshold, derisk_to, 1.0), index=ratio.index)


CH2C_CONFIGS = [
    {"name": "ch2c_primary", "r_lo": 1.00, "r_hi": 1.15, "derisk_to": 0.50},
    {"name": "ch2c_deeper_cut", "r_lo": 1.00, "r_hi": 1.15, "derisk_to": 0.25},
    {"name": "ch2c_wider_band", "r_lo": 1.00, "r_hi": 1.25, "derisk_to": 0.50},
]


def run_ch2c() -> dict:
    """Gate the whipsaw-aware governor vs constant-gross (the program gate) + report the PLAIN VIX
    governor as a reference (is whipsaw-awareness worth it?). PASS decided on ch2c_primary."""
    from app.strategy.tsmom import TSMOMConfig
    base, spy, closes = build_base()
    cfg = TSMOMConfig(universe=[c for c in closes.columns])
    ratio = vix_ratio_series(closes.index)
    n_trials = len(CH2C_CONFIGS)
    base_rep = _evaluate(base, spy, "ch2c_base", n_trials)
    base_mean_sharpe = float(base_rep.mean_sharpe)

    # reference: the plain (binary, stress-only) VIX governor already live
    plain_m = plain_vix_governor_multiplier(ratio, closes.index, ratio_threshold=1.00, derisk_to=0.50)
    plain = gate_multiplier(plain_m, base, spy, label="plain_vix_governor", n_trials=1,
                            base_mean_sharpe=base_mean_sharpe)

    results = []
    for c in CH2C_CONFIGS:
        m = whipsaw_governor_multiplier(closes, cfg, ratio, r_lo=c["r_lo"], r_hi=c["r_hi"],
                                        derisk_to=c["derisk_to"])
        r = gate_multiplier(m, base, spy, label=c["name"], n_trials=n_trials,
                            base_mean_sharpe=base_mean_sharpe)
        r["config"] = c
        r["d_vs_plain_governor"] = round(r["mean_sharpe_full"] - plain["mean_sharpe_full"], 4)
        results.append(r)
    primary = next(r for r in results if r["label"] == "ch2c_primary")
    return {
        "multiplier": "CH2c — trending-vs-whipsaw-aware crash governor",
        "gate": "governed must beat the base CPCV mean_sharpe (SIGNIFICANTLY) AND not regress BEAR",
        "base_mean_sharpe": round(base_mean_sharpe, 4),
        "plain_vix_governor_reference": plain,
        "n_trials_registered": n_trials,
        "primary_config": "ch2c_primary",
        "PASS": bool(primary["PASS"]),
        "verdict": ("SHIP (shadow-first)" if primary["PASS"]
                    else "KILL - does not beat constant-gross; ship nothing"),
        "results": results,
    }


def _print_block(out: dict) -> None:
    print(f"  {out['multiplier']}")
    print(f"    base_mean_sharpe (full precision): {out['base_mean_sharpe']}")
    if "plain_vix_governor_reference" in out:
        p = out["plain_vix_governor_reference"]
        print(f"    [ref ] plain_vix_governor     meanSR {p['mean_sharpe']:+.4f} "
              f"(d {p['d_mean_sharpe']:+.4f}, p_impr {p['improve_pvalue']})  BEAR {p['bear_sharpe']}")
    for r in out["results"]:
        flag = "PASS" if r["PASS"] else "----"
        extra = (f" vsPlain {r['d_vs_plain_governor']:+.4f}" if "d_vs_plain_governor" in r else "")
        print(f"    [{flag}] {r['label']:<22} meanSR {r['mean_sharpe']:+.4f} "
              f"(d {r['d_mean_sharpe']:+.4f}, p_impr {r['improve_pvalue']})  BEAR {r['bear_sharpe']}  "
              f"avg_m {r['avg_mult']}  beats={r['beats_baseline']} sig={r['improvement_significant']} "
              f"bear_ok={r['bear_no_regression']}{extra}")
    print(f"    VERDICT: {out['verdict']}")


def main() -> int:
    out = {"candidates": [run_ch2b(), run_ch2c()]}
    with open(RESULTS_ARTIFACT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"CH2 sizing -> {RESULTS_ARTIFACT}")
    for block in out["candidates"]:
        _print_block(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
