"""
gates.py — FoldResult, WalkForwardReport, and all gate / metric logic.

Extracted from walkforward_tier3.py as part of WF-2 pluggable engine refactor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm


class GateOutcome(str, Enum):
    """Tri-state promotion outcome for a gate evaluation.

    Phase-4 FIX-1: distinguishes "gate failed → retire/rollback" from "gate could
    not be evaluated for promotion → keep current model status, do nothing."

    Under GATE_MODE='significance', a bare WalkForwardReport carries only a single
    point estimate — no path-Sharpe distribution and therefore NO t-stat — so it
    cannot EARN promotion (the significance gate is a distribution test). But
    "can't evaluate for promotion" is NOT "failed → retire." Conflating the two
    auto-retires every scheduled WF retrain and rolls back the fresh model. The
    cron caller must treat INCONCLUSIVE as report-only: do not retire, do not roll
    back, log that CPCV is required for a promotion decision.

      PROMOTE      — gate evaluated and PASSED. Promote / keep ACTIVE.
      RETIRE       — gate evaluated and FAILED. Retire new, restore previous.
      INCONCLUSIVE — gate could not be evaluated for promotion (e.g. significance
                     mode + WF-only). Report-only: keep current status untouched.
    """
    PROMOTE = "promote"
    RETIRE = "retire"
    INCONCLUSIVE = "inconclusive"

# ── Gate thresholds ───────────────────────────────────────────────────────────
SHARPE_GATE = 0.8
MIN_FOLD_SHARPE = -0.3
# N_TRIALS_TESTED is the single source of truth in retrain_config.py.
# Imported here for backward compatibility — do not re-define or shadow it.
from app.ml.retrain_config import N_TRIALS_TESTED  # noqa: E402
MIN_PROFIT_FACTOR = 1.10
MIN_CALMAR = 0.30
MIN_WORST_REGIME_SHARPE = -0.5
# CR-1/C8-9: minimum number of "active" folds required before PF/Calmar gates are binding.
# =1 only closed the zero-fold bypass; =2 closes the single-lucky-fold bypass where one
# favorable fold satisfies PF≥MIN and the rest abstain.  Requires majority participation.
MIN_ACTIVE_FOLDS_FOR_GATE = 2


class DataSpanError(RuntimeError):
    """Raised when strategy data covers fewer than MIN_DATA_SPAN_TRADING_DAYS trading days."""


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).
    Returns (dsr_z, p_value). p_value > 0.95 = significant after selection bias correction.

    Bug fix (WF deep-review pass 2): E[SR_max] must be multiplied by sqrt(V[SR]); the
    prior implementation omitted that scaling factor. `n_obs` is the number of return
    observations (trading days), not the number of trades.
    """
    if n_trials <= 1 or n_obs <= 1:
        return sharpe, 0.5
    euler_mascheroni = 0.5772156649
    sr_var = (1 + 0.5 * sharpe ** 2) / max(n_obs - 1, 1)
    sr_var_sqrt = math.sqrt(sr_var)
    sr_star = sr_var_sqrt * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )
    dsr_z = (sharpe - sr_star) / sr_var_sqrt
    return dsr_z, float(norm.cdf(dsr_z))


PF_NO_LOSS_SENTINEL = 5.0   # all-wins fold → use cap value (not 0 or 999)
MAX_PF_FOR_AVG = 5.0        # cap before averaging so one lucky fold can't dominate
CAL_TOTAL_LOSS_SENTINEL = -5.0  # total loss >100% → negative sentinel so avg_calmar reflects disaster
CAL_NO_DD_SENTINEL = 5.0    # max_drawdown=0 + profitable fold → cap value
MIN_TRADES_FOR_CAL_SENTINEL = 5  # guard against abstaining models gaming no-DD calmar sentinel


def compute_profit_factor(trade_returns: list) -> float:
    """Profit factor = sum(positive returns) / sum(abs(negative returns)).

    Returns 0.0 if no trades. Returns PF_NO_LOSS_SENTINEL (5.0) when there are
    winners but zero losers — matching the cap used during averaging so that
    all-wins folds are included (not dropped) with a bounded contribution.
    The prior implementation returned 0.0 for no-loss folds, which silently
    excluded the best-looking folds from avg_profit_factor.
    """
    if not trade_returns:
        return 0.0
    wins = sum(r for r in trade_returns if r > 0)
    losses = sum(abs(r) for r in trade_returns if r < 0)
    if losses > 0:
        return float(wins / losses)
    return PF_NO_LOSS_SENTINEL if wins > 0 else 0.0


def compute_calmar(total_return_pct: float, max_drawdown_pct: float, years: float,
                   daily_returns: Optional[list] = None) -> float:
    """Calmar ratio = CAGR / max drawdown (inputs as fractions, e.g. 0.20 = 20%).

    Geometric annualisation: (1 + total_return)^(1/years) - 1. Arithmetic
    division (total/years) overstates multi-year returns and understates sub-year
    ones. This matches tier3._compute_calmar — intraday was already correct,
    now swing is too.

    MEDIUM-1 fix: when max_drawdown == 0, use a vol-based floor drawdown
    (0.5 × monthly annualised vol from daily_returns) instead of CAL_NO_DD_SENTINEL.
    Falls back to MIN_CALMAR_FLOOR_DD (1%) when daily_returns unavailable.
    Controlled by USE_CALMAR_VOL_FLOOR flag; set False for legacy behaviour.

    Returns 0.0 if years <= 0 or (legacy, no vol-floor) max_drawdown_pct <= 0.
    Returns CAL_TOTAL_LOSS_SENTINEL (-5.0) when 1+total_return <= 0 (total wipeout)
    so catastrophic folds pull down avg_calmar rather than disappearing silently.
    """
    from app.ml.retrain_config import USE_CALMAR_VOL_FLOOR, MIN_CALMAR_FLOOR_DD
    if years <= 0:
        return 0.0
    base = 1.0 + total_return_pct
    if base <= 0:
        return CAL_TOTAL_LOSS_SENTINEL
    cagr = base ** (1.0 / years) - 1.0
    dd = max_drawdown_pct
    if dd <= 0:
        if USE_CALMAR_VOL_FLOOR:
            if daily_returns and len(daily_returns) >= 2:
                ann_vol = float(np.std(daily_returns, ddof=1)) * math.sqrt(252.0)
                monthly_vol = ann_vol / 12.0
                dd = max(0.5 * monthly_vol, MIN_CALMAR_FLOOR_DD)
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "compute_calmar: max_dd=0 and no daily_returns — "
                    "using MIN_CALMAR_FLOOR_DD=%.3f floor", MIN_CALMAR_FLOOR_DD
                )
                dd = MIN_CALMAR_FLOOR_DD
        else:
            return 0.0  # legacy: caller applies sentinel in avg_calmar property
    return float(cagr / dd)


def compute_k_ratio(equity_curve) -> float:
    """K-ratio = annualised slope of log-equity / std of daily log returns.

    Two bugs fixed:
    1. AgentSimulator.equity_curve is a list of (date, value) tuples. The prior
       np.array(..., dtype=float) raised on the date column; the bare except
       swallowed it, so every fold silently reported k_ratio=0.0.
    2. The prior implementation regressed raw dollar equity and divided by std
       of dollar diffs — scale-dependent and not annualised. Now uses log-equity
       and sqrt(252), matching tier3._compute_k_ratio.

    Accepts flat list of equity values OR list of (date, value) tuples.
    Returns 0.0 if insufficient data, non-positive equity, or zero volatility.
    """
    if equity_curve is None or len(equity_curve) < 4:
        return 0.0
    try:
        first = equity_curve[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            values = [v for _, v in equity_curve]
        else:
            values = list(equity_curve)
        y = np.asarray(values, dtype=float)
        if y.size < 4 or not np.all(np.isfinite(y)) or not np.all(y > 0):
            return 0.0
        log_y = np.log(y)
        x = np.arange(log_y.size, dtype=float)
        slope = float(np.polyfit(x, log_y, 1)[0])
        log_rets = np.diff(log_y)
        vol = float(np.std(log_rets, ddof=1)) if log_rets.size > 1 else 0.0
        if vol <= 0:
            return 0.0
        return (slope / vol) * math.sqrt(252.0)
    except Exception:
        return 0.0


def fold_years(test_start: date, test_end: date) -> float:
    return max((test_end - test_start).days / 365.0, 1 / 365.0)


@dataclass
class FoldResult:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float
    stop_exit_rate: float
    model_version: int = 0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    k_ratio: float = 0.0
    regime_sharpes: Dict[str, float] = field(default_factory=dict)
    regime_diversity: int = 0
    # Phase-4 FIX-2: RAW per-regime observation counts BEFORE the REGIME_MIN_OBS
    # filter (populated by compute_regime_sharpes via its obs_counts out-param).
    # Used to distinguish EVENT-SPARSITY (counts present but all < REGIME_MIN_OBS →
    # regime_sharpes empty) from a DATA-BUG (no counts at all). See cpcv.py
    # run_cpcv regime aggregation and CPCVResult.regime_insufficient_obs.
    regime_obs_counts: Dict[str, int] = field(default_factory=dict)
    # WF-5a: abstention tracking
    opp_score_abstain_days: int = 0
    earnings_blackout_days: int = 0
    macro_gate_days: int = 0
    # Number of return observations (trading days) for DSR (0 = unknown).
    n_obs: int = 0
    # CRITICAL-2: capital deployment tracking (diagnostic — not a gate).
    avg_capital_deployed_pct: float = 0.0
    deployment_adjusted_sharpe: float = 0.0
    low_deployment_warning: bool = False
    # Phase A diagnostics: per-feature mean IC over the test window (optional).
    # Populated by the walk-forward engine when --compute-fold-ic is passed.
    # Not a gate — used to monitor feature decay between train and test.
    # Key = feature name, value = mean cross-sectional Spearman IC (h=10d).
    feature_ic: Optional[Dict[str, float]] = None

    def passed_gate(self) -> bool:
        return self.sharpe >= MIN_FOLD_SHARPE

    def summary_line(self) -> str:
        gate = "OK" if self.passed_gate() else "FAIL"
        pf_str = f"  PF={self.profit_factor:.2f}" if self.profit_factor > 0 else ""
        cal_str = f"  Cal={self.calmar_ratio:.2f}" if self.calmar_ratio != 0 else ""
        return (
            f"  Fold {self.fold} [{gate}] "
            f"test={self.test_start}->{self.test_end}  "
            f"trades={self.trades}  win={self.win_rate:.1%}  "
            f"Sharpe={self.sharpe:.2f}  DD={self.max_drawdown:.1%}"
            f"{pf_str}{cal_str}"
        )


@dataclass
class WalkForwardReport:
    model_type: str
    folds: List[FoldResult] = field(default_factory=list)
    # True when the OOS guard was bypassed with allow_in_sample=True.
    # In-sample runs can never promote past gates.
    in_sample_override: bool = False
    # True only when per-fold retraining was performed (not just re-scoring one
    # frozen model). Affects DSR interpretation — set by FoldEngine when used.
    is_true_walkforward: bool = False
    # MEDIUM-3: (n_days, start_date, end_date, n_symbols, source_hint) — set by engine.
    data_span: Optional[tuple] = None

    # Paper-trade gate thresholds (less strict; for deploy-to-paper decisions)
    PAPER_SHARPE_GATE: float = 0.50
    PAPER_MIN_FOLD_SHARPE: float = -0.40

    @property
    def avg_sharpe(self) -> float:
        """n_obs-weighted mean Sharpe across folds.

        C8-4: unweighted mean lets a short, lucky fold lift avg_sharpe cheaply while
        total_obs (DSR denominator) is correctly summed. Weighting by n_obs makes
        avg_sharpe and total_obs inputs to DSR consistent. Falls back to equal
        weighting when no fold reports n_obs (all zeros).
        """
        if not self.folds:
            return 0.0
        weights = [getattr(f, "n_obs", 0) or 0 for f in self.folds]
        total_w = sum(weights)
        if total_w > 0:
            return float(sum(f.sharpe * w for f, w in zip(self.folds, weights)) / total_w)
        return float(np.mean([f.sharpe for f in self.folds]))

    @property
    def min_sharpe(self) -> float:
        return float(np.min([f.sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def avg_win_rate(self) -> float:
        return float(np.mean([f.win_rate for f in self.folds])) if self.folds else 0.0

    @property
    def total_trades(self) -> int:
        return sum(f.trades for f in self.folds)

    @property
    def total_obs(self) -> int:
        """Total return observations (trading days) across folds.

        Does NOT fall back to total_trades: trade count ≠ observation count for
        DSR which requires daily-return observations. If no folds report n_obs,
        returns 0 and gate_passed() will use the n_trials fallback in
        deflated_sharpe_ratio() rather than a semantically incorrect trade count.
        """
        return sum(getattr(f, "n_obs", 0) or 0 for f in self.folds)

    @property
    def avg_profit_factor(self) -> float:
        # Cap at MAX_PF_FOR_AVG before averaging: PF=999 sentinel (all-wins fold)
        # must not inflate the mean above the gate threshold by itself. Also
        # includes folds with PF=PF_NO_LOSS_SENTINEL (5.0) — all-wins folds are
        # now properly included with a bounded contribution.
        pfs = [min(f.profit_factor, MAX_PF_FOR_AVG)
               for f in self.folds if f.profit_factor > 0]
        return float(np.mean(pfs)) if pfs else 0.0

    @property
    def avg_calmar(self) -> float:
        # A zero calmar_ratio can mean either (a) max_drawdown=0 (a good fold,
        # silently dropped by the old != 0 filter) or (b) a degenerate fold with
        # negative return and zero drawdown. Distinguish by checking total_return.
        # CAL_NO_DD_SENTINEL guard: require MIN_TRADES_FOR_CAL_SENTINEL trades so
        # an abstaining model (0 trades, 0 DD, 0 return) can't stack sentinel values
        # and trivially pass the Calmar gate.
        #
        # MEDIUM-1: under USE_CALMAR_VOL_FLOOR (default), compute_calmar already
        # returns a vol-floored value for profitable no-DD folds (calmar_ratio != 0),
        # so the legacy sentinel branch is gated off. Set the flag False for legacy.
        from app.ml.retrain_config import USE_CALMAR_VOL_FLOOR
        cals: List[float] = []
        for f in self.folds:
            # M-2 fix: calmar_ratio is stored rounded to 3 decimal places. A genuine
            # small positive Calmar (e.g. 0.0004) rounds to exactly 0.0, making the
            # != 0 filter falsely drop it. Use the (max_drawdown, total_return) pair
            # as the primary activeness signal instead: a fold with no DD and positive
            # return IS active regardless of the stored calmar float value.
            if USE_CALMAR_VOL_FLOOR:
                if f.calmar_ratio != 0:
                    cals.append(f.calmar_ratio)
                elif (f.max_drawdown == 0 and f.total_return > 0
                      and f.trades >= MIN_TRADES_FOR_CAL_SENTINEL):
                    # calmar_ratio rounded to 0.0 but fold IS active — re-compute
                    # from stored values so the fold contributes its real (floored) Calmar.
                    from scripts.walkforward.gates import fold_years as _fy
                    _years = _fy(f.test_start, f.test_end)
                    _recomputed = compute_calmar(f.total_return, f.max_drawdown, _years)
                    if _recomputed != 0:
                        cals.append(_recomputed)
                # else: degenerate fold (no trades, negative return, or tiny trade count) — skip
            else:
                # Legacy sentinel path: no-DD + profitable + enough trades → sentinel
                if f.calmar_ratio != 0:
                    cals.append(f.calmar_ratio)
                elif (f.max_drawdown == 0 and f.total_return > 0
                        and f.trades >= MIN_TRADES_FOR_CAL_SENTINEL):
                    cals.append(CAL_NO_DD_SENTINEL)
        return float(np.mean(cals)) if cals else 0.0

    @property
    def avg_k_ratio(self) -> float:
        ks = [f.k_ratio for f in self.folds if f.k_ratio != 0]
        return float(np.mean(ks)) if ks else 0.0

    @property
    def worst_regime_sharpe(self) -> Optional[float]:
        """Worst per-regime mean Sharpe across folds. None if no regime data collected.

        IM-3 fix: aggregates per-regime (mean across folds) THEN takes the min over
        regimes, rather than min over all (fold × regime) pairs. The raw min over all
        pairs is dominated by single-fold noise — one bad fold in one regime can gate-
        fail an otherwise solid model, and the minimum gets noisier as more paths are
        added. Per-regime mean first gives a stable, interpretable worst-regime signal.
        """
        regime_to_sharpes: Dict[str, List[float]] = {}
        for f in self.folds:
            for regime, sh in f.regime_sharpes.items():
                regime_to_sharpes.setdefault(regime, []).append(sh)
        if not regime_to_sharpes:
            return None
        per_regime_means = [float(np.mean(v)) for v in regime_to_sharpes.values()]
        return float(min(per_regime_means))

    # ── CRITICAL-1: DSR ceiling / human-review flags ──────────────────────────
    def dsr_saturated(self, dsr_n: int = N_TRIALS_TESTED) -> bool:
        """True when DSR p > DSR_SATURATION_P — gate is non-binding at this Sharpe."""
        from app.ml.retrain_config import DSR_SATURATION_P
        _, p = deflated_sharpe_ratio(self.avg_sharpe, dsr_n, self.total_obs)
        return p > DSR_SATURATION_P

    def requires_human_review(self) -> bool:
        """True when avg_sharpe > SHARPE_IMPLAUSIBILITY_CEILING.
        Does NOT affect gate_passed() — must be checked by the promotion runner."""
        from app.ml.retrain_config import SHARPE_IMPLAUSIBILITY_CEILING
        return self.avg_sharpe > SHARPE_IMPLAUSIBILITY_CEILING

    # ── CRITICAL-2: deployment diagnostics ────────────────────────────────────
    @property
    def avg_deployment_pct(self) -> float:
        """n_obs-weighted mean capital deployment across folds."""
        weights = [getattr(f, "n_obs", 0) or 0 for f in self.folds]
        total_w = sum(weights)
        if total_w > 0:
            return float(sum(f.avg_capital_deployed_pct * w for f, w in zip(self.folds, weights)) / total_w)
        return float(np.mean([f.avg_capital_deployed_pct for f in self.folds])) if self.folds else 0.0

    @property
    def avg_deployment_adjusted_sharpe(self) -> float:
        weights = [getattr(f, "n_obs", 0) or 0 for f in self.folds]
        total_w = sum(weights)
        if total_w > 0:
            return float(sum(f.deployment_adjusted_sharpe * w for f, w in zip(self.folds, weights)) / total_w)
        return float(np.mean([f.deployment_adjusted_sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def low_deployment(self) -> bool:
        from app.ml.retrain_config import MIN_DEPLOYMENT_PCT_WARN
        return self.avg_deployment_pct < MIN_DEPLOYMENT_PCT_WARN

    # Phase-4: human-readable reason a WF-only run cannot promote under significance.
    SIGNIFICANCE_WF_BLOCK_REASON = (
        "CPCV required for significance gate (WF has no path t-stat)"
    )

    def gate_passed(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> bool:
        import logging as _logging
        # Phase-4: a standard (non-CPCV) WalkForwardReport has only a single point
        # estimate — no path-Sharpe distribution and therefore no t-stat. The
        # significance gate is fundamentally a distribution test, so a WF-only
        # promotion HARD-FAILS under significance mode (it must NOT fabricate a
        # t-stat). Legacy mean_sharpe mode keeps the original WF behavior.
        from app.ml.retrain_config import GATE_MODE
        if GATE_MODE == "significance":
            _logging.getLogger(__name__).error(
                "WalkForwardReport.gate_passed under GATE_MODE='significance': %s. "
                "Run CPCV (CPCVResult.gate_passed) to obtain a path-Sharpe t-stat.",
                self.SIGNIFICANCE_WF_BLOCK_REASON,
            )
            return False
        if self.in_sample_override:
            return False
        # Frozen-mode (not true per-fold) runs cannot promote when the project-wide
        # flag is set. Default False during Phase 1 rollout — flip to True in Phase 3.
        from app.ml.retrain_config import REQUIRE_TRUE_WF_FOR_PROMOTION
        if REQUIRE_TRUE_WF_FOR_PROMOTION and not self.is_true_walkforward:
            return False
        _, dsr_p = deflated_sharpe_ratio(self.avg_sharpe, dsr_n, self.total_obs)
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        # CR-1: "avg == 0 → pass" bypass closed: require at least MIN_ACTIVE_FOLDS_FOR_GATE
        # folds contributing to each metric before treating the gate as optional.
        from app.ml.retrain_config import USE_CALMAR_VOL_FLOOR
        n_pf_active = sum(1 for f in self.folds if f.profit_factor > 0)
        # M-2 fix: count a fold as "active" when it has a meaningful Calmar OR when it
        # is a no-DD profitable fold with enough trades (calmar_ratio may round to 0.0
        # but the fold still contributed to avg_calmar via the re-computation branch).
        n_cal_active = sum(
            1 for f in self.folds
            if f.calmar_ratio != 0
            or (f.max_drawdown == 0 and f.total_return > 0
                and f.trades >= MIN_TRADES_FOR_CAL_SENTINEL)
        )
        pf_ok = (paper_gate
                 or n_pf_active < MIN_ACTIVE_FOLDS_FOR_GATE
                 or self.avg_profit_factor >= MIN_PROFIT_FACTOR)
        cal_ok = (paper_gate
                  or n_cal_active < MIN_ACTIVE_FOLDS_FOR_GATE
                  or self.avg_calmar >= MIN_CALMAR)
        wrs = self.worst_regime_sharpe
        from app.ml.retrain_config import ALLOW_NO_REGIME_GATE
        if wrs is None:
            if ALLOW_NO_REGIME_GATE:
                _logging.getLogger(__name__).warning(
                    "WalkForwardReport: worst_regime_sharpe=None, gate bypassed "
                    "(ALLOW_NO_REGIME_GATE=True). Regime gate NOT enforced."
                )
                regime_ok = True
            else:
                _logging.getLogger(__name__).error(
                    "WalkForwardReport: worst_regime_sharpe=None — regime data insufficient. "
                    "GATE FAILING (set ALLOW_NO_REGIME_GATE=True to bypass). "
                    "Ensure fetch_data populated _global_regime_map."
                )
                regime_ok = False
        else:
            regime_ok = wrs >= MIN_WORST_REGIME_SHARPE
        return (
            self.avg_sharpe >= sharpe_gate
            and self.min_sharpe >= min_fold_gate
            and dsr_p > 0.95
            and pf_ok
            and cal_ok
            and regime_ok
        )

    def gate_outcome(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> GateOutcome:
        """Tri-state promotion outcome (Phase-4 FIX-1).

        Under GATE_MODE='significance', a WF-only report cannot be evaluated for
        promotion (no path t-stat) → INCONCLUSIVE. This is REPORT-ONLY: the cron
        caller must NOT retire/roll back a freshly trained model on this outcome.
        Under GATE_MODE='mean_sharpe' (legacy), the outcome maps directly from the
        boolean gate: PROMOTE on pass, RETIRE on a real legacy fail (unchanged).
        """
        from app.ml.retrain_config import GATE_MODE
        if GATE_MODE == "significance":
            # WF carries no path-Sharpe distribution → cannot earn promotion, but
            # this is "not evaluable", NOT a failure. CPCV is required for a verdict.
            return GateOutcome.INCONCLUSIVE
        return (GateOutcome.PROMOTE
                if self.gate_passed(dsr_n=dsr_n, paper_gate=paper_gate)
                else GateOutcome.RETIRE)

    def gate_detail(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> dict:
        # Phase-4: under significance mode a WF-only run cannot promote (no path
        # t-stat). Surface that as a single failed criterion so the reporter prints
        # the "CPCV required" reason rather than a misleading per-metric breakdown.
        from app.ml.retrain_config import GATE_MODE
        if GATE_MODE == "significance":
            return {
                "cpcv_required_for_significance": (self.avg_sharpe, False),
            }
        _, dsr_p = deflated_sharpe_ratio(self.avg_sharpe, dsr_n, self.total_obs)
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        n_pf_active = sum(1 for f in self.folds if f.profit_factor > 0)
        # Must match gate_passed() exactly — always count no-DD profitable folds as active
        # (calmar_ratio may round to 0.0 for small values; use max_drawdown+total_return
        # as the activeness signal, not the rounded float). See M-2 fix.
        n_cal_active = sum(
            1 for f in self.folds
            if f.calmar_ratio != 0
            or (f.max_drawdown == 0 and f.total_return > 0
                and f.trades >= MIN_TRADES_FOR_CAL_SENTINEL)
        )
        pf_ok = (paper_gate
                 or n_pf_active < MIN_ACTIVE_FOLDS_FOR_GATE
                 or self.avg_profit_factor >= MIN_PROFIT_FACTOR)
        cal_ok = (paper_gate
                  or n_cal_active < MIN_ACTIVE_FOLDS_FOR_GATE
                  or self.avg_calmar >= MIN_CALMAR)
        wrs = self.worst_regime_sharpe
        from app.ml.retrain_config import ALLOW_NO_REGIME_GATE as _ALLOW_NRG
        _wrs_ok = (wrs is not None and wrs >= MIN_WORST_REGIME_SHARPE) or (wrs is None and _ALLOW_NRG)
        return {
            "avg_sharpe": (self.avg_sharpe, self.avg_sharpe >= sharpe_gate),
            "min_sharpe": (self.min_sharpe, self.min_sharpe >= min_fold_gate),
            "dsr_p": (dsr_p, dsr_p > 0.95),
            "avg_profit_factor": (self.avg_profit_factor, pf_ok),
            "avg_calmar": (self.avg_calmar, cal_ok),
            "worst_regime_sharpe": (wrs, _wrs_ok),
            # CRITICAL-1/2: informational keys — NOT part of gate_passed() boolean AND.
            # ok=True when NOT triggered, so they only appear in the failed-keys list
            # when the condition fires.
            "human_review_required": (self.avg_sharpe, not self.requires_human_review()),
            "low_deployment_warning": (self.avg_deployment_pct, not self.low_deployment),
        }

    def print(self, dsr_n: int | None = None, paper_gate: bool = False) -> None:
        from scripts.walkforward.reports import print_report
        print_report(self, dsr_n=dsr_n, paper_gate=paper_gate)
