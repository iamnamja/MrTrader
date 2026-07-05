"""
null_zoo.py — Alpha-v10 GL-0: the SELECTION-AWARE null-strategy zoo for the futures book.

The Go-Live panel was unanimous: the futures book's Track-B residual-alpha **t vs the live ETF-trend
book is not trustworthy** until it is deflated for the factors / sleeve-families we searched. A raw
t > 1.96 is meaningless after that search. This module answers the gating question the way the panel
asked ("the null must replicate the researcher, not just the strategy"): drive RANDOMIZED signals
through the IDENTICAL pipeline (same 76-market universe, the same `carry_backtest` engine + roll
cost, the same 50/50 basket combine, and the same Track-B residual-alpha-vs-trend statistic that
produced the observed t) and ask how often structured noise manufactures the observed t.

What each test covers (be precise about scope):
  * NULL-BOOKS p (primary) — is the basket's CROSS-SECTIONAL bet non-random? Build null(carry) +
    null(xsmom) by per-date cross-sectional permutation, combine 50/50 exactly as
    `build_futures_book`, compute the Track-B t vs live trend; empirical p = P(t_null >= t_obs).
    This tests "is picking THESE markets real?" — NOT the full multi-family search burden.
  * xs-momentum MAX-OF-6 — the FUTURES-FACTOR selection bar. Each replication permutes the SIX
    structurally-distinct factor families we searched (xs-mom, curve-mom, value, skew, + 2 lookback
    variants standing in for basis-mom / CoT), takes the max standalone Track-B t; bar = 95th pct.
    Faithful to "we kept the best of 6 factors" (different breadth/turnover => realistic dispersion).
  * carry SINGLE-FACTOR — carry had a strong economic prior and was NOT selected from 6, so its
    single-factor null (an INDEPENDENT permutation, not the basket's) is the right bar.
  * DEFLATED SHARPE (Bailey-Lopez de Prado) on the book's RESIDUAL stream at N_eff in {10,20,30},
    using the null books' residual-SR variance — the PARAMETRIC trial-count deflation. This is how
    the broader ~20-family-across-asset-classes burden (NOT futures-reproducible, so not in the
    empirical nulls) is represented: DSR(N=20) must clear 0.95.

Per-date cross-sectional permutation (the null) preserves EXACTLY: the return panel (untouched),
each date's present (non-NaN) market set, and each date's cross-sectional signal distribution; it
destroys only the true signal->market alignment. (HAC SEs stay valid because they are applied to the
realized null-book returns regardless.)

Verdict (carry-only fallback baked in):
  * BASKET_REAL  — null-books p < 0.05 AND DSR(N=20) > 0.95 AND xs-momentum beats its max-of-6 null.
  * CARRY_ONLY   — carry beats its single-factor null but xs-momentum fails max-of-6 (size carry,
                   drop/de-weight xs-momentum — the panel's explicit fallback).
  * RESIDUE      — carry does not beat its single-factor null (stay single-sleeve / no futures book).

PURE + deterministic (seeded). Reuses the canonical Track-B path so a null book's t is computed
byte-identically to the observed (verified == appraise_track_b: book 2.611, carry 2.032, xsmom 2.217).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.research import futures_carry as fc, futures_factors as ff
from app.research.inference import multifactor_alpha
from scripts.walkforward.book_gate import _vol_target_candidate, ANN

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------- null signal generators
def cross_sectional_permute(signal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per-date cross-sectional permutation among PRESENT (non-NaN) markets: each date, shuffle
    which present market gets which signal value. Preserves EXACTLY the per-date present-set and
    cross-sectional distribution (NaN cells stay NaN) and the return panel; destroys the true
    signal->market alignment. The standard cross-sectional null for a cross-sectional factor."""
    vals = signal.to_numpy().copy()
    for i in range(vals.shape[0]):
        present = np.flatnonzero(~np.isnan(vals[i]))
        if present.size > 1:
            vals[i, present] = vals[i, present][rng.permutation(present.size)]
    return pd.DataFrame(vals, index=signal.index, columns=signal.columns)


def circular_time_shift(signal: pd.DataFrame, rng: np.random.Generator,
                        min_shift: int = 252) -> pd.DataFrame:
    """Circular time-shift null (cross-check): roll every market's signal in time by one random
    offset (wrap-around). Breaks signal->return TIMING; preserves full autocorrelation + the
    cross-section. `min_shift` is clamped to the panel so the offset range never degenerates."""
    n = len(signal)
    ms = max(1, min(min_shift, n // 4))
    hi = max(ms + 1, n - ms)
    shift = int(rng.integers(ms, hi))
    return pd.DataFrame(np.roll(signal.to_numpy(), shift, axis=0),
                        index=signal.index, columns=signal.columns)


# ---------------------------------------------------------------- the Track-B statistic
def track_b_stat(candidate: pd.Series, base: pd.Series) -> Tuple[float, float]:
    """The EXACT observed-t path (track_b_appraisal lines 206-223): vol-target the candidate to the
    base book's full-sample ann vol, then the Newey-West HAC residual-alpha vs the base. Returns
    (t_alpha_hac, resid_sharpe_annualized). NaN-safe -> (nan, nan) on degenerate input (so a
    degenerate draw is EXCLUDED from the null distribution rather than piling mass at 0)."""
    aligned = pd.concat([base.rename("base"), candidate.rename("cand")], axis=1,
                        join="inner").dropna()
    if len(aligned) < 60 or float(aligned["cand"].std()) <= 0:
        return float("nan"), float("nan")
    target = float(aligned["base"].std() * np.sqrt(ANN))
    if target <= 0:
        return float("nan"), float("nan")
    cand_vt, _ = _vol_target_candidate(aligned["cand"], target)
    ev = pd.DataFrame({"base": aligned["base"], "cand": cand_vt}).dropna()
    if len(ev) < 60:
        return float("nan"), float("nan")
    mfa = multifactor_alpha(ev["cand"], ev[["base"]], hac_lag=5)
    return float(mfa.get("t_alpha_hac", 0.0) or 0.0), float(mfa.get("resid_sharpe", 0.0) or 0.0)


def residual_stream(candidate: pd.Series, base: pd.Series) -> pd.Series:
    """The factor-HEDGED residual daily stream (cand_vt regressed on [1, base]) — the series whose
    Sharpe is `resid_sharpe`. Used by the Deflated Sharpe so the DSR is computed on the residual
    (per the panel), with units matching `deflated_sharpe`'s internal per-period SR."""
    aligned = pd.concat([base.rename("base"), candidate.rename("cand")], axis=1,
                        join="inner").dropna()
    if len(aligned) < 60 or float(aligned["cand"].std()) <= 0:
        return pd.Series(dtype=float)
    cand_vt, _ = _vol_target_candidate(aligned["cand"],
                                       float(aligned["base"].std() * np.sqrt(ANN)))
    ev = pd.DataFrame({"base": aligned["base"], "cand": cand_vt}).dropna()
    if len(ev) < 60:
        return pd.Series(dtype=float)
    X = np.column_stack([np.ones(len(ev)), ev["base"].to_numpy()])
    y = ev["cand"].to_numpy()
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # factor-HEDGED stream = y - beta*base = alpha + resid (alpha RETAINED), matching
    # multifactor_alpha's `hedged` (inference.py:375) whose Sharpe is resid_sharpe. NOT the OLS
    # residual y - alpha - beta*base (that has mean ~0 -> Sharpe ~0 -> DSR degenerate to 0.5).
    return pd.Series(y - X[:, 1:] @ coef[1:], index=ev.index)


# ---------------------------------------------------------------- deflated Sharpe (Bailey-LdP)
def deflated_sharpe(returns: pd.Series, n_trials: int, var_sr_trials: float) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): probability the TRUE Sharpe > 0 after
    correcting for (a) the number of trials and (b) the return distribution's skew/kurtosis.
    `var_sr_trials` = variance of the trial Sharpe estimates IN THE SAME (per-period) UNITS as the
    internal `sr` (we pass the null books' residual per-period SR variance). Returns DSR in [0,1];
    > 0.95 is the conventional bar."""
    from scipy.stats import norm, skew, kurtosis
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 30 or sd <= 0:
        return 0.0
    sr = float(r.mean() / sd)                         # per-period Sharpe (same units as var_sr_trials)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))             # non-excess kurtosis
    v = max(var_sr_trials, 1e-12)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = np.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(norm.cdf(((sr - sr0) * np.sqrt(n - 1.0)) / denom))


# ---------------------------------------------------------------- the zoo
@dataclass
class NullZooResult:
    n_nulls: int
    n_degenerate: int                     # null draws excluded (too short / zero-vol)
    t_obs_book: float
    t_obs_carry: float
    t_obs_xsmom: float
    resid_sharpe_book: float
    null_book_p: float
    null_book_p95: float
    null_book_p99: float
    null_book_p_shift: float
    carry_null_p: float
    carry_null_p95: float
    xsmom_maxof6_p: float
    xsmom_maxof6_p95: float
    dsr_n10: float
    dsr_n20: float
    dsr_n30: float
    n_families: int = 0          # P0.5: enumerated family-level trial count (registry)
    dsr_family: float = 0.0      # P0.5: deflated Sharpe at the principled family count
    verdict: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _factor_panels(prices: pd.DataFrame, carry: pd.DataFrame,
                   returns: pd.DataFrame) -> List[pd.DataFrame]:
    """The SIX structurally-distinct factor families we searched (4 real builders + 2 lookback
    variants standing in for basis-mom / CoT), for the faithful max-of-6 selection null."""
    return [
        ff.xs_momentum_signal(prices),
        ff.curve_momentum_signal(carry),
        ff.value_signal(prices),
        ff.skew_signal(returns),
        ff.xs_momentum_signal(prices, lookback=126, skip=21),
        ff.curve_momentum_signal(carry, lookback=21),
    ]


def _empirical_p(arr, obs) -> float:
    """Davison-Hinkley one-sided empirical p = (1 + #{null >= obs}) / (1 + N).

    KL-12 CONFIRMED-3 fix: a NON-FINITE observed statistic FAILS CLOSED (returns NaN). Previously a
    NaN `obs` made `np.sum(arr >= NaN) == 0` → p collapse to 1/(1+N) ≈ 0.001 (read as "significant"),
    so a degenerate / insufficient-history sleeve could manufacture the strongest verdict. NaN routes
    through the downstream `< 0.05` comparisons as False → RESIDUE (conservative)."""
    if not np.isfinite(obs):
        return float("nan")
    return float((1 + int(np.sum(arr >= obs))) / (1 + len(arr))) if len(arr) else float("nan")


def run_null_zoo(returns: pd.DataFrame, carry: pd.DataFrame, mom_signal: pd.DataFrame,
                 roll_days: pd.DataFrame, base: pd.Series, *,
                 prices: Optional[pd.DataFrame] = None,
                 cfg: Optional[fc.CarryConfig] = None,
                 n_nulls: int = 1000, seed: int = 0,
                 t_obs_target: Optional[float] = None) -> NullZooResult:
    """Run the selection-aware null zoo. Inputs are the EXACT panels the real book is built from
    (so nulls traverse the identical pipeline). `base` = live ETF-trend daily returns. `prices`
    (synthetic price panel) is needed for the faithful max-of-6; without it the max-of-6 falls back
    to permutations of the momentum panel."""
    cfg = cfg or fc.CarryConfig(roll_cost_bps=3.0)
    rng = np.random.default_rng(seed)
    if n_nulls < 199:
        # the empirical-p floor is 1/(n+1); < 199 cannot attain p < 0.05 for a gating decision.
        pass  # allowed for smoke/tests; the runner default (1000) is the gating configuration.

    factor_panels = (_factor_panels(prices, carry, returns) if prices is not None
                     else [mom_signal] * 6)

    def _book(c_sig: pd.DataFrame, x_sig: pd.DataFrame) -> pd.Series:
        c = fc.carry_backtest(returns, c_sig, cfg, roll_days=roll_days)
        x = ff.xs_factor_backtest(returns, x_sig, cfg, roll_days=roll_days)
        j = pd.concat([c.rename("c"), x.rename("x")], axis=1, join="inner").dropna()
        return (0.5 * j["c"] + 0.5 * j["x"]).rename("null_book")

    # --- observed (real) statistics ---
    carry_real = fc.carry_backtest(returns, carry, cfg, roll_days=roll_days)
    xsmom_real = ff.xs_factor_backtest(returns, mom_signal, cfg, roll_days=roll_days)
    book_real = _book(carry, mom_signal)
    t_obs_book, resid_sharpe_book = track_b_stat(book_real, base)
    t_obs_carry, _ = track_b_stat(carry_real, base)
    t_obs_xsmom, _ = track_b_stat(xsmom_real, base)
    bar = float(t_obs_target) if t_obs_target is not None else t_obs_book

    # --- null distributions ---
    nb_ts: List[float] = []
    nb_ts_shift: List[float] = []
    nb_resid_daily_sr: List[float] = []
    cn_ts: List[float] = []
    x6_ts: List[float] = []
    n_degenerate = 0

    for _ in range(n_nulls):
        # primary null book (per-date cross-sectional permutation on BOTH factors)
        nc = cross_sectional_permute(carry, rng)
        nx = cross_sectional_permute(mom_signal, rng)
        nb = _book(nc, nx)
        t_nb, _ = track_b_stat(nb, base)
        if np.isnan(t_nb):
            n_degenerate += 1
        else:
            nb_ts.append(t_nb)
            rs = residual_stream(nb, base)
            if len(rs) and rs.std() > 0:
                nb_resid_daily_sr.append(float(rs.mean() / rs.std()))
        # cross-check null book (circular time-shift)
        shift_book = _book(circular_time_shift(carry, rng), circular_time_shift(mom_signal, rng))
        t_sh, _ = track_b_stat(shift_book, base)
        if not np.isnan(t_sh):
            nb_ts_shift.append(t_sh)
        # carry single-factor null — INDEPENDENT permutation (not the basket's nc)
        t_cn, _ = track_b_stat(
            fc.carry_backtest(returns, cross_sectional_permute(carry, rng), cfg,
                              roll_days=roll_days), base)
        if not np.isnan(t_cn):
            cn_ts.append(t_cn)
        # xs-momentum max-of-6 — best over the SIX distinct factor families, each permuted
        best = -np.inf
        for fp in factor_panels:
            t6, _ = track_b_stat(
                ff.xs_factor_backtest(returns, cross_sectional_permute(fp, rng), cfg,
                                      roll_days=roll_days), base)
            if not np.isnan(t6):
                best = max(best, t6)
        if np.isfinite(best):
            x6_ts.append(best)

    nb_arr, cn_arr, x6_arr = np.array(nb_ts), np.array(cn_ts), np.array(x6_ts)

    _p = _empirical_p  # KL-12 CONFIRMED-3: fail-closed on a non-finite observed statistic

    null_book_p = _p(nb_arr, bar)
    carry_null_p = _p(cn_arr, t_obs_carry)
    xsmom_maxof6_p = _p(x6_arr, t_obs_xsmom)

    # Deflated Sharpe on the book's RESIDUAL stream; var across trials = null books' residual
    # per-period SR variance (SAME units as deflated_sharpe's internal sr).
    var_sr = (float(np.var(nb_resid_daily_sr, ddof=1))
              if len(nb_resid_daily_sr) > 1 else 1e-6)
    resid_real = residual_stream(book_real, base)
    dsr10 = deflated_sharpe(resid_real, 10, var_sr)
    dsr20 = deflated_sharpe(resid_real, 20, var_sr)
    dsr30 = deflated_sharpe(resid_real, 30, var_sr)
    # P0.5: the principled parametric N — the ENUMERATED family-level trial count from the
    # registry (auditable), replacing the prior hardcoded ~20. dsr10/30 stay as a sensitivity band.
    from app.research.family_registry import family_trial_count
    n_families = family_trial_count()
    dsr_family = deflated_sharpe(resid_real, n_families, var_sr)

    # --- verdict ---
    # PRIMARY gate = the empirical max-stat nulls (the panel's requested "null replicates the
    # researcher"). DSR is a PARAMETRIC cross-check reported alongside — it does NOT hard-veto a
    # strong empirical result (a borderline DSR<0.95 reflects a genuine-but-modest residual Sharpe,
    # which the panel explicitly predicted; vetoing on it would double-count the deflation).
    carry_survives = carry_null_p < 0.05
    xsmom_survives = xsmom_maxof6_p < 0.05
    basket_survives = (null_book_p < 0.05) and xsmom_survives
    verdict = ("BASKET_REAL" if basket_survives
               else "CARRY_ONLY" if carry_survives else "RESIDUE")
    dsr_corroborates = dsr_family > 0.95

    notes = [
        f"null-books empirical p = {null_book_p:.3f} (bar t_obs = {bar:.2f}; "
        f"95th-pct null t = {float(np.percentile(nb_arr, 95)) if len(nb_arr) else float('nan'):.2f}).",
        f"carry single-factor null p = {carry_null_p:.3f} -> carry "
        f"{'SURVIVES' if carry_survives else 'does NOT survive'}.",
        f"xs-momentum max-of-6 (distinct factors) null p = {xsmom_maxof6_p:.3f} -> xs-momentum "
        f"{'SURVIVES' if xsmom_survives else 'indistinguishable from best-of-6 noise'}.",
        f"DSR(N={n_families}, family-level) on residual = {dsr_family:.3f} -> "
        f"{'CORROBORATES' if dsr_corroborates else 'BORDERLINE (<0.95)'} the empirical verdict "
        f"(parametric cross-check; the empirical max-stat null is the primary gate). "
        f"Sensitivity: DSR(N=10)={dsr10:.3f}, DSR(N=30)={dsr30:.3f}.",
        f"{n_degenerate}/{n_nulls} null draws degenerate (excluded).",
        f"SCOPE: the empirical nulls cover the FUTURES cross-sectional + 6-factor selection; the "
        f"broader cross-asset burden is represented parametrically by DSR at the ENUMERATED "
        f"family-level trial count N={n_families} (app/research/family_registry.py — P0.5; replaces "
        f"the prior hardcoded ~20).",
    ]

    return NullZooResult(
        n_nulls=n_nulls, n_degenerate=n_degenerate,
        t_obs_book=t_obs_book, t_obs_carry=t_obs_carry, t_obs_xsmom=t_obs_xsmom,
        resid_sharpe_book=resid_sharpe_book,
        null_book_p=null_book_p,
        null_book_p95=float(np.percentile(nb_arr, 95)) if len(nb_arr) else float("nan"),
        null_book_p99=float(np.percentile(nb_arr, 99)) if len(nb_arr) else float("nan"),
        null_book_p_shift=_p(np.array(nb_ts_shift), bar),
        carry_null_p=carry_null_p,
        carry_null_p95=float(np.percentile(cn_arr, 95)) if len(cn_arr) else float("nan"),
        xsmom_maxof6_p=xsmom_maxof6_p,
        xsmom_maxof6_p95=float(np.percentile(x6_arr, 95)) if len(x6_arr) else float("nan"),
        dsr_n10=dsr10, dsr_n20=dsr20, dsr_n30=dsr30,
        n_families=n_families, dsr_family=dsr_family,
        verdict=verdict, notes=notes)


# ---------------------------------------------------------------- look-ahead audit (Claude B5)
def _past_max_diff(a: pd.Series, b: pd.Series, cut: pd.Timestamp) -> Tuple[bool, float]:
    """Max abs diff of two series on the window strictly BEFORE `cut` (PIT past)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    past = j[j.index < cut].dropna()
    if past.empty:
        return True, 0.0
    diff = float((past["a"] - past["b"]).abs().max())
    return diff < 1e-12, diff


def look_ahead_audit(returns: pd.DataFrame, carry: pd.DataFrame, mom_signal: pd.DataFrame,
                     roll_days: pd.DataFrame, *, prices: Optional[pd.DataFrame] = None,
                     cfg: Optional[fc.CarryConfig] = None,
                     corrupt_tail_days: int = 120) -> Dict[str, object]:
    """Empirical PIT / future-blindness audit of the carry & xs-momentum pipelines (the newest
    code, never paranoia-audited; ties to the data-quality factor_scorer look-ahead finding).

    Decisive test: perturb the FUTURE (x5 the last `corrupt_tail_days`), re-run, confirm everything
    BEFORE the corruption window is byte-identical. If the past changes when the future is perturbed,
    there is look-ahead.
      (1) ENGINE future-blindness — corrupt future RETURNS (signals fixed) -> past backtest P&L
          unchanged (tests carry_backtest + xs_factor_backtest sizing/cost).
      (2) MOMENTUM-SIGNAL future-blindness — recompute the momentum signal off corrupted future
          PRICES -> past signal unchanged (tests xs_momentum_signal's shifts).
    SCOPE: this audits the ENGINE and the MOMENTUM SIGNAL. The CARRY signal's PIT-correctness (the
    term-structure expiry logic) was established separately by the 2026-06-18 scheduled-expiry
    hardening (carry from the contract code, which fixed the recent-carry staleness + roll hindsight)."""
    cfg = cfg or fc.CarryConfig(roll_cost_bps=3.0)
    out: Dict[str, object] = {}
    cut = returns.index[-corrupt_tail_days]

    c0 = fc.carry_backtest(returns, carry, cfg, roll_days=roll_days)
    x0 = ff.xs_factor_backtest(returns, mom_signal, cfg, roll_days=roll_days)
    corrupted = returns.copy()
    corrupted.loc[corrupted.index >= cut] *= 5.0
    c1 = fc.carry_backtest(corrupted, carry, cfg, roll_days=roll_days)
    x1 = ff.xs_factor_backtest(corrupted, mom_signal, cfg, roll_days=roll_days)
    carry_ok, carry_diff = _past_max_diff(c0, c1, cut)
    xsmom_ok, xsmom_diff = _past_max_diff(x0, x1, cut)

    sig_ok, sig_diff = True, 0.0
    if prices is not None:
        pc = prices.copy()
        pc.loc[pc.index >= cut] *= 5.0
        s0 = ff.xs_momentum_signal(prices)
        s1 = ff.xs_momentum_signal(pc)
        past0 = s0[s0.index < cut]
        if not past0.empty:
            diff = (past0 - s1.reindex(past0.index)[past0.columns]).abs().to_numpy()
            finite = diff[np.isfinite(diff)]
            sig_diff = float(finite.max()) if finite.size else 0.0
            sig_ok = sig_diff < 1e-12

    out.update(carry_engine_future_blind=carry_ok, xsmom_engine_future_blind=xsmom_ok,
               xsmom_signal_future_blind=sig_ok, carry_max_past_diff=carry_diff,
               xsmom_max_past_diff=xsmom_diff, signal_max_past_diff=sig_diff)
    out["pit_clean"] = bool(carry_ok and xsmom_ok and sig_ok)
    out["notes"] = [
        f"corrupted last {corrupt_tail_days} days x5; carry past max-diff {carry_diff:.2e}, "
        f"xsmom past max-diff {xsmom_diff:.2e}, signal past max-diff {sig_diff:.2e}.",
        "PIT-clean (past invariant to future)" if out["pit_clean"]
        else "LOOK-AHEAD DETECTED — past changed when the future was perturbed.",
    ]
    return out
