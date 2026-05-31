"""
gates.py — FoldResult, WalkForwardReport, and all gate / metric logic.

Extracted from walkforward_tier3.py as part of WF-2 pluggable engine refactor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

# ── Gate thresholds ───────────────────────────────────────────────────────────
SHARPE_GATE = 0.8
MIN_FOLD_SHARPE = -0.3
# N_TRIALS_TESTED is the single source of truth in retrain_config.py.
# Imported here for backward compatibility — do not re-define or shadow it.
from app.ml.retrain_config import N_TRIALS_TESTED  # noqa: E402
MIN_PROFIT_FACTOR = 1.10
MIN_CALMAR = 0.30
MIN_WORST_REGIME_SHARPE = -0.5


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


def compute_profit_factor(trade_returns: list) -> float:
    """Profit factor = sum(positive returns) / sum(abs(negative returns)).
    Returns 0.0 if no trades or no losses."""
    if not trade_returns:
        return 0.0
    wins = sum(r for r in trade_returns if r > 0)
    losses = sum(abs(r) for r in trade_returns if r < 0)
    return float(wins / losses) if losses > 0 else 0.0


def compute_calmar(total_return_pct: float, max_drawdown_pct: float, years: float) -> float:
    """Calmar ratio = annualised return / max drawdown.
    Returns 0.0 if max_drawdown or years is zero."""
    if max_drawdown_pct <= 0 or years <= 0:
        return 0.0
    annualised = total_return_pct / years
    return float(annualised / max_drawdown_pct)


def compute_k_ratio(equity_curve: list) -> float:
    """K-ratio = slope of cumulative return / std of periodic returns.
    Positive = equity curve trends up consistently. Returns 0.0 if insufficient data."""
    if len(equity_curve) < 4:
        return 0.0
    try:
        y = np.array(equity_curve, dtype=float)
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        diffs = np.diff(y)
        vol = float(np.std(diffs)) if len(diffs) > 1 else 1.0
        return slope / vol if vol > 0 else 0.0
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
    # WF-5a: abstention tracking
    opp_score_abstain_days: int = 0
    earnings_blackout_days: int = 0
    macro_gate_days: int = 0
    # Number of return observations (trading days) for DSR (0 = unknown).
    n_obs: int = 0
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

    @property
    def avg_sharpe(self) -> float:
        return float(np.mean([f.sharpe for f in self.folds])) if self.folds else 0.0

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

        BUG-6 fix: previously fell back to total_trades when n_obs=0.
        Trade count ≠ observation count — DSR requires daily-return observations.
        Folds without n_obs data are excluded from the sum; if no folds have
        n_obs, returns 0 and gate_passed() will use the n_trials fallback in
        deflated_sharpe_ratio() rather than a semantically incorrect trade count.
        """
        return sum(getattr(f, "n_obs", 0) or 0 for f in self.folds)

    @property
    def avg_profit_factor(self) -> float:
        pfs = [f.profit_factor for f in self.folds if f.profit_factor > 0]
        return float(np.mean(pfs)) if pfs else 0.0

    @property
    def avg_calmar(self) -> float:
        cals = [f.calmar_ratio for f in self.folds if f.calmar_ratio != 0]
        return float(np.mean(cals)) if cals else 0.0

    @property
    def avg_k_ratio(self) -> float:
        ks = [f.k_ratio for f in self.folds if f.k_ratio != 0]
        return float(np.mean(ks)) if ks else 0.0

    @property
    def worst_regime_sharpe(self) -> Optional[float]:
        """Minimum per-regime Sharpe across all folds. None if no regime data collected."""
        all_regime_sharpes: List[float] = []
        for f in self.folds:
            all_regime_sharpes.extend(f.regime_sharpes.values())
        return float(min(all_regime_sharpes)) if all_regime_sharpes else None

    def gate_passed(self) -> bool:
        if self.in_sample_override:
            return False
        _, dsr_p = deflated_sharpe_ratio(self.avg_sharpe, N_TRIALS_TESTED, self.total_obs)
        pf_ok = self.avg_profit_factor == 0 or self.avg_profit_factor >= MIN_PROFIT_FACTOR
        cal_ok = self.avg_calmar == 0 or self.avg_calmar >= MIN_CALMAR
        wrs = self.worst_regime_sharpe
        regime_ok = wrs is None or wrs >= MIN_WORST_REGIME_SHARPE
        return (
            self.avg_sharpe >= SHARPE_GATE
            and self.min_sharpe >= MIN_FOLD_SHARPE
            and dsr_p > 0.95
            and pf_ok
            and cal_ok
            and regime_ok
        )

    def gate_detail(self) -> dict:
        _, dsr_p = deflated_sharpe_ratio(self.avg_sharpe, N_TRIALS_TESTED, self.total_obs)
        wrs = self.worst_regime_sharpe
        return {
            "avg_sharpe": (self.avg_sharpe, self.avg_sharpe >= SHARPE_GATE),
            "min_sharpe": (self.min_sharpe, self.min_sharpe >= MIN_FOLD_SHARPE),
            "dsr_p": (dsr_p, dsr_p > 0.95),
            "avg_profit_factor": (self.avg_profit_factor,
                                  self.avg_profit_factor == 0 or self.avg_profit_factor >= MIN_PROFIT_FACTOR),
            "avg_calmar": (self.avg_calmar,
                           self.avg_calmar == 0 or self.avg_calmar >= MIN_CALMAR),
            "worst_regime_sharpe": (wrs, wrs is None or wrs >= MIN_WORST_REGIME_SHARPE),
        }

    def print(self) -> None:
        from scripts.walkforward.reports import print_report
        print_report(self)
