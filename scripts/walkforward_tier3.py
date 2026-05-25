"""
walkforward_tier3.py — Rolling walk-forward validation using Tier 3 agent simulation.

Design (per PHASES_18_23_SPEC.md):
  Fold 1: train on Y1-Y2, test Tier 3 on Y3
  Fold 2: train on Y1-Y3, test Tier 3 on Y4
  Fold 3: train on Y1-Y4, test Tier 3 on Y5 (most recent)

Gate: avg OOS Tier 3 Sharpe > 0.8, no fold below -0.3

WF-1 additions (2026-05-07):
  - embargo_days parameter: clean gap on BOTH sides of every test window
      train | purge_days | TEST | embargo_days | next_train
  - New fold metrics: profit_factor, calmar_ratio, k_ratio
  - Extended gate: avg profit_factor > 1.10, avg calmar > 0.30

Usage:
    python scripts/walkforward_tier3.py --model swing --folds 3 --years 5
    python scripts/walkforward_tier3.py --model intraday --folds 3 --days 730
    python scripts/walkforward_tier3.py --model both --folds 3

Exit codes:
    0 — gate passed (avg Sharpe > 0.8, no fold < -0.3)
    1 — gate not passed or error
"""
from __future__ import annotations

import argparse
import atexit
import logging
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import math  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from app.ml.retrain_config import MAX_WORKERS, MAX_FOLD_WORKERS, N_TRIALS_TESTED  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Gate thresholds ───────────────────────────────────────────────────────────
SHARPE_GATE = 0.8          # avg OOS Sharpe required to pass
MIN_FOLD_SHARPE = -0.3     # no individual fold may be below this
# N_TRIALS_TESTED imported from app.ml.retrain_config — single source of truth.
# WF-1: multi-metric gates
MIN_PROFIT_FACTOR = 1.10   # avg profit factor across folds (sum wins / sum |losses|)
MIN_CALMAR = 0.30          # avg Calmar ratio (annualised return / max drawdown)


def _deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).
    Returns (dsr_z, p_value). p_value > 0.95 = significant after selection bias correction.

    Formula (per B&LP 2014, eq. 8-10):
      V[SR]      = (1 + 0.5·SR²) / (T - 1)              # IID-normal variance of SR estimator
      E[SR_max]  = sqrt(V[SR]) · [ (1-γ)·Φ⁻¹(1-1/N)
                                  + γ·Φ⁻¹(1-1/(N·e)) ]   # selection-bias correction
      DSR_z      = (SR_obs - E[SR_max]) / sqrt(V[SR])

    Bug fix (WF deep-review pass 2): previous implementation omitted the sqrt(V[SR])
    scaling factor on E[SR_max], so the deflation term carried wrong units and
    massively overstated DSR for high-T runs. The numerator and denominator of
    DSR_z must use the same variance scaling.

    Args:
      sharpe:   The observed (annualized) Sharpe ratio.
      n_trials: Number of model variants tried historically (selection-bias N).
      n_obs:    Number of return OBSERVATIONS used to compute SR (i.e. number of
                daily returns / trading days in the test window). NOT trade count —
                using trade count here under-states T and produces an inflated DSR.
                Callers must pass total trading days, not total_trades.
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


# ── WF-1: Additional metric helpers ──────────────────────────────────────────

def _compute_profit_factor(trade_returns: list) -> float:
    """Profit factor = sum(positive returns) / sum(abs(negative returns)).
    Returns 0.0 if no trades (cannot compute).
    Returns 999.0 if wins > 0 but no losses (infinite edge, small fold).
    """
    if not trade_returns:
        return 0.0
    wins = sum(r for r in trade_returns if r > 0)
    losses = sum(abs(r) for r in trade_returns if r < 0)
    if losses > 0:
        return float(wins / losses)
    return 999.0 if wins > 0 else 0.0


def _compute_calmar(total_return_pct: float, max_drawdown_pct: float, years: float) -> float:
    """Calmar ratio = annualised return / max drawdown.
    Returns 0.0 if max_drawdown is zero (avoids division by zero).
    """
    if max_drawdown_pct <= 0 or years <= 0:
        return 0.0
    annualised = total_return_pct / years
    return float(annualised / max_drawdown_pct)


def _compute_k_ratio(equity_curve: list) -> float:
    """K-ratio = slope of cumulative return / std of annual returns.
    Uses a simple linear regression on the equity curve index.
    Returns 0.0 if insufficient data.

    A positive K-ratio means the equity curve trends up consistently.
    """
    if len(equity_curve) < 4:
        return 0.0
    try:
        y = np.array(equity_curve, dtype=float)
        x = np.arange(len(y), dtype=float)
        # slope via lstsq
        slope = float(np.polyfit(x, y, 1)[0])
        # std of differences as proxy for volatility of returns
        diffs = np.diff(y)
        vol = float(np.std(diffs)) if len(diffs) > 1 else 1.0
        return slope / vol if vol > 0 else 0.0
    except Exception:
        return 0.0


def _fold_years(test_start: date, test_end: date) -> float:
    """Return the length of a test fold in years."""
    return max((test_end - test_start).days / 365.0, 1 / 365.0)


def _make_regime_gate_fn(
    symbols_data: dict,
    spy_ma_days: int = 200,
    vix_bull: float = 20.0,
    vix_bear: float = 30.0,
    bull_mult: float = 1.0,
    neutral_mult: float = 0.7,
    bear_mult: float = 0.3,
):
    """Return a callable(day: date) -> float that gives gross exposure multiplier.

    Regime rules (PIT-safe — uses data strictly before *day*):
      BULL:    SPY > SPY_MA(spy_ma_days) AND VIX < vix_bull  -> bull_mult
      BEAR:    SPY < SPY_MA(spy_ma_days) OR  VIX >= vix_bear -> bear_mult
      NEUTRAL: otherwise                                      -> neutral_mult
    """
    import pandas as _pd

    spy_df = symbols_data.get("SPY") if "SPY" in symbols_data else symbols_data.get("spy")
    vix_df = symbols_data.get("^VIX") if "^VIX" in symbols_data else symbols_data.get("VIX")

    spy_close: "_pd.Series | None" = spy_df["close"] if spy_df is not None else None
    vix_close: "_pd.Series | None" = vix_df["close"] if vix_df is not None else None

    spy_ma: "_pd.Series | None" = (
        spy_close.rolling(spy_ma_days, min_periods=max(spy_ma_days // 2, 1)).mean()
        if spy_close is not None else None
    )

    def _regime_fn(day: date) -> float:
        # PIT: use data strictly before the rebalance day
        _day_ts = _pd.Timestamp(day)

        spy_val = vix_val = spy_ma_val = None
        if spy_close is not None:
            _spy_hist = spy_close[spy_close.index < _day_ts]
            if len(_spy_hist) > 0:
                spy_val = float(_spy_hist.iloc[-1])
        if spy_ma is not None:
            _ma_hist = spy_ma[spy_ma.index < _day_ts]
            if len(_ma_hist) > 0:
                spy_ma_val = float(_ma_hist.iloc[-1])
        if vix_close is not None:
            _vix_hist = vix_close[vix_close.index < _day_ts]
            if len(_vix_hist) > 0:
                vix_val = float(_vix_hist.iloc[-1])

        # Determine regime
        above_ma = (spy_val is not None and spy_ma_val is not None
                    and spy_val > spy_ma_val)
        vix_low = vix_val is not None and vix_val < vix_bull
        vix_high = vix_val is not None and vix_val >= vix_bear
        spy_below = not above_ma if (spy_val is not None and spy_ma_val is not None) else False

        if above_ma and vix_low:
            return bull_mult      # BULL
        if spy_below or vix_high:
            return bear_mult      # BEAR
        return neutral_mult       # NEUTRAL

    return _regime_fn


# ── Console helpers ───────────────────────────────────────────────────────────

def _ok(msg):
    print(f"  \033[32mOK\033[0m  {msg}")


def _warn(msg):
    print(f"  \033[33mWARN\033[0m  {msg}")


def _err(msg):
    print(f"  \033[31mFAIL\033[0m  {msg}")


def _header(msg):
    print(f"\n{'='*62}\n  {msg}\n{'='*62}")


def _subheader(msg):
    print(f"\n{'-'*62}\n  {msg}\n{'-'*62}")


# ── Fold result ───────────────────────────────────────────────────────────────

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
    # WF-1: additional metrics
    profit_factor: float = 0.0   # sum(wins) / sum(|losses|); 0 = not computed
    calmar_ratio: float = 0.0    # annualised_return / max_drawdown; 0 = not computed
    k_ratio: float = 0.0         # slope(cum_ret) / std(annual_ret); 0 = not computed
    # WF-4: regime stratification
    regime_sharpes: dict = field(default_factory=dict)
    regime_diversity: int = 0
    # WF-5a: abstention tracking
    opp_score_abstain_days: int = 0
    earnings_blackout_days: int = 0
    macro_gate_days: int = 0
    # Number of return observations (trading days) in this fold's equity curve.
    # Used to compute T for the Deflated Sharpe Ratio. 0 means "unknown" — callers
    # that aggregate folds should sum this across folds for the DSR n_obs argument.
    n_obs: int = 0
    # Phase A diagnostics: per-feature mean IC over the test window (optional).
    # Populated when --compute-fold-ic is passed. Not a gate — used to monitor
    # feature decay between train and test.
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
    # WF design note (deep-review pass 3): the harness loads ONE pre-trained
    # model via _load_model() and re-scores it across every fold's test window.
    # This is a GENERALIZATION TEST — "does this single model hold up across
    # multiple out-of-sample regimes?" — not a true expanding-window walk-forward
    # (which would retrain on [train_start, fold_train_end] inside each fold).
    # The flag below makes that distinction explicit for downstream consumers
    # (reporting, ML_EXPERIMENT_LOG, DSR interpretation). Set to True only when
    # a per-fold retrain has actually been performed.
    is_true_walkforward: bool = False

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
        """Total number of return observations (trading days) across all folds.
        Falls back to total_trades only when no fold reported n_obs (legacy)."""
        obs = sum(getattr(f, "n_obs", 0) or 0 for f in self.folds)
        return obs if obs > 0 else self.total_trades

    @property
    def avg_profit_factor(self) -> float:
        # Cap at 5.0 before averaging: PF=999 sentinel (all-wins fold) would otherwise
        # inflate the mean far above the gate threshold, giving a spurious pass.
        MAX_PF = 5.0
        pfs = [min(f.profit_factor, MAX_PF) for f in self.folds if f.profit_factor > 0]
        return float(np.mean(pfs)) if pfs else 0.0

    @property
    def avg_calmar(self) -> float:
        cals = [f.calmar_ratio for f in self.folds if f.calmar_ratio != 0]
        return float(np.mean(cals)) if cals else 0.0

    @property
    def avg_k_ratio(self) -> float:
        ks = [f.k_ratio for f in self.folds if f.k_ratio != 0]
        return float(np.mean(ks)) if ks else 0.0

    # P3: paper-gate thresholds (less strict, for deploy-to-paper decisions)
    PAPER_SHARPE_GATE = 0.50
    PAPER_MIN_FOLD_SHARPE = -0.40

    def gate_passed(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> bool:
        _, dsr_p = _deflated_sharpe_ratio(self.avg_sharpe, dsr_n, self.total_obs)
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        pf_ok = paper_gate or self.avg_profit_factor == 0 or self.avg_profit_factor >= MIN_PROFIT_FACTOR
        cal_ok = paper_gate or self.avg_calmar == 0 or self.avg_calmar >= MIN_CALMAR
        return (
            self.avg_sharpe >= sharpe_gate
            and self.min_sharpe >= min_fold_gate
            and dsr_p > 0.95
            and pf_ok
            and cal_ok
        )

    def gate_detail(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> dict:
        """Return per-gate pass/fail dict for logging and tests."""
        _, dsr_p = _deflated_sharpe_ratio(self.avg_sharpe, dsr_n, self.total_obs)
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        pf_ok = paper_gate or self.avg_profit_factor == 0 or self.avg_profit_factor >= MIN_PROFIT_FACTOR
        cal_ok = paper_gate or self.avg_calmar == 0 or self.avg_calmar >= MIN_CALMAR
        return {
            "avg_sharpe": (self.avg_sharpe, self.avg_sharpe >= sharpe_gate),
            "min_sharpe": (self.min_sharpe, self.min_sharpe >= min_fold_gate),
            "dsr_p": (dsr_p, dsr_p > 0.95),
            "avg_profit_factor": (self.avg_profit_factor, pf_ok),
            "avg_calmar": (self.avg_calmar, cal_ok),
        }

    def print(self, dsr_n: int = N_TRIALS_TESTED, paper_gate: bool = False) -> None:
        _header(f"Walk-Forward Report — {self.model_type.upper()} (Tier 3)"
                + (" [PAPER-GATE MODE]" if paper_gate else ""))
        for f in self.folds:
            print(f.summary_line())
        print()
        detail = self.gate_detail(dsr_n=dsr_n, paper_gate=paper_gate)
        sharpe_gate = self.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
        min_fold_gate = self.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE
        print(f"  Avg Sharpe:      {self.avg_sharpe:+.3f}  (gate: > {sharpe_gate})  "
              f"{'OK' if detail['avg_sharpe'][1] else 'FAIL'}")
        print(f"  Min fold Sharpe: {self.min_sharpe:+.3f}  (gate: > {min_fold_gate})  "
              f"{'OK' if detail['min_sharpe'][1] else 'FAIL'}")
        print(f"  Avg win rate:    {self.avg_win_rate:.1%}")
        print(f"  Total trades:    {self.total_trades}")
        dsr_z, dsr_p = _deflated_sharpe_ratio(self.avg_sharpe, dsr_n, self.total_obs)
        print(f"  DSR (N={dsr_n} trials): z={dsr_z:+.3f}  p={dsr_p:.3f}  "
              f"(gate: p > 0.95)  {'OK' if dsr_p > 0.95 else 'FAIL'}")
        if self.avg_profit_factor > 0 and not paper_gate:
            print(f"  Avg profit factor: {self.avg_profit_factor:.3f}  "
                  f"(gate: > {MIN_PROFIT_FACTOR})  "
                  f"{'OK' if detail['avg_profit_factor'][1] else 'FAIL'}")
        if self.avg_calmar != 0 and not paper_gate:
            print(f"  Avg Calmar ratio:  {self.avg_calmar:.3f}  "
                  f"(gate: > {MIN_CALMAR})  "
                  f"{'OK' if detail['avg_calmar'][1] else 'FAIL'}")
        if self.avg_k_ratio != 0:
            print(f"  Avg K-ratio:       {self.avg_k_ratio:.3f}  (directional; > 0 = improving)")
        print()
        if self.gate_passed(dsr_n=dsr_n, paper_gate=paper_gate):
            mode = "PAPER GATE" if paper_gate else "GATE"
            _ok(f"{mode} PASSED — avg Sharpe {self.avg_sharpe:.3f}, DSR p={dsr_p:.3f}, "
                f"PF={self.avg_profit_factor:.2f}, Calmar={self.avg_calmar:.2f}")
        else:
            failed = [k for k, (v, ok) in detail.items() if not ok]
            _err(f"GATE NOT MET — failed: {', '.join(failed)}")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_name: str, version: Optional[int] = None):
    """Load a model by name. If version is given, load that specific version
    regardless of status — used for re-testing historical models."""
    import pickle
    try:
        from app.database.models import ModelVersion as MV
        from app.database.session import get_session
        db = get_session()
        try:
            if version is not None:
                mv = db.query(MV).filter_by(
                    model_name=model_name, version=version
                ).first()
            else:
                mv = (
                    db.query(MV)
                    .filter_by(model_name=model_name, status="ACTIVE")
                    .order_by(MV.version.desc())
                    .first()
                )
            if mv and mv.model_path:
                path = Path(mv.model_path)
                if path.exists():
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    if hasattr(obj, "is_trained"):
                        logger.info("Loaded %s model v%d (status=%s)",
                                    model_name, mv.version, mv.status)
                        return obj, mv.version
                from app.ml.model import PortfolioSelectorModel
                m = PortfolioSelectorModel(model_type="xgboost")
                m.load(str(path.parent), mv.version, model_name=model_name)
                logger.info("Loaded %s model v%d (status=%s)",
                            model_name, mv.version, mv.status)
                return m, mv.version
        finally:
            db.close()
    except Exception as exc:
        logger.warning("DB model load failed: %s", exc)
    # Fallback: load from pkl files when DB is unreachable.
    # Only load a specific version if explicitly requested, OR if a .gate_passed
    # sentinel exists — prevents auto-promoting models that never passed the gate.
    model_dir = Path("app/ml/models")
    if version is not None:
        path = model_dir / f"{model_name}_v{version}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "is_trained"):
                logger.info("Loaded %s model v%d from file", model_name, version)
                return obj, version
            from app.ml.model import PortfolioSelectorModel
            m = PortfolioSelectorModel(model_type="xgboost")
            m.load(str(model_dir), version, model_name=model_name)
            return m, version

    # When no version specified: only consider pkls that have a .gate_passed sentinel.
    # This prevents a silently-retrained model from displacing the known-good version
    # just because its version number is higher.
    gated_files = sorted(
        [p for p in model_dir.glob(f"{model_name}_v*.pkl")
         if (p.parent / (p.stem + ".gate_passed")).exists()],
        key=lambda p: int(p.stem.split("_v")[-1]),
    )
    if not gated_files:
        # Warn loudly — no gated model means the system is misconfigured
        logger.error(
            "No gate_passed sentinel found for any %s model in %s — "
            "cannot safely load a fallback model. Run the walk-forward gate "
            "and touch %s_v<N>.gate_passed to certify a version.",
            model_name, model_dir, model_name,
        )
        return None, 0

    # Load the highest-versioned gated model
    latest_gated = gated_files[-1]
    ver = int(latest_gated.stem.split("_v")[-1])
    with open(latest_gated, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "is_trained"):
        logger.info("Loaded %s model v%d from file (gate_passed sentinel present)",
                    model_name, ver)
        return obj, ver
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.load(str(latest_gated.parent), ver, model_name=model_name)
    logger.info("Loaded %s model v%d from file (gate_passed sentinel present)",
                model_name, ver)
    return m, ver


# ── Earnings calendar pre-fetch (Phase 2b) ────────────────────────────────────

def _fetch_earnings_calendar(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> Dict[str, set]:
    """Pre-fetch historical earnings dates for all symbols using yfinance.

    Returns dict of symbol → set of datetime.date when earnings were reported.
    Earnings within the simulation window are included. Symbols with no data
    get an empty set (no blackout applied — fail-open).

    Takes ~2-3 min for 700 symbols. Call once before the sim, not per-fold.
    """
    import yfinance as yf
    cal: Dict[str, set] = {}
    logger.info("Pre-fetching earnings calendar for %d symbols (yfinance)...", len(symbols))
    ok, fail = 0, 0
    for sym in symbols:
        try:
            ed = yf.Ticker(sym).earnings_dates
            if ed is not None and len(ed) > 0:
                dates: set = set()
                for idx in ed.index:
                    d = idx.date() if hasattr(idx, "date") else idx
                    if start_date <= d <= end_date:
                        dates.add(d)
                cal[sym] = dates
                ok += 1
            else:
                cal[sym] = set()
                ok += 1
        except Exception:
            cal[sym] = set()
            fail += 1
    logger.info("Earnings calendar: %d ok, %d failed (fail-open for failures)", ok, fail)
    return cal


# ── Bootstrap walk-forward (Phase 1d) ─────────────────────────────────────────

def _bootstrap_folds(
    run_fn,
    n_bootstrap: int = 100,
    perturb_days: int = 30,
    **kwargs,
) -> Dict:
    """Run the walk-forward n_bootstrap times with ±perturb_days randomised fold offsets.

    Returns dict with Sharpe distribution statistics for reporting.
    """
    sharpes: List[float] = []
    rng = np.random.default_rng(42)
    _subheader(f"Bootstrap walk-forward: {n_bootstrap} iterations × ±{perturb_days}d perturbation")
    for i in range(n_bootstrap):
        jitter = int(rng.integers(-perturb_days, perturb_days + 1))
        try:
            if "total_years" in kwargs:
                kw = dict(kwargs, total_years=kwargs["total_years"] + jitter / 365)
            elif "total_days" in kwargs:
                kw = dict(kwargs, total_days=max(180, kwargs["total_days"] + jitter))
            else:
                kw = kwargs
            report = run_fn(**kw)
            if report.folds:
                sharpes.append(report.avg_sharpe)
        except Exception as exc:
            logger.debug("Bootstrap iter %d failed: %s", i, exc)
    if not sharpes:
        return {}
    arr = np.array(sharpes)
    result = {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "pct_positive": float(np.mean(arr > 0)),
    }
    print(f"\n  Bootstrap Sharpe distribution ({len(arr)} iters):")
    print(f"    Mean:   {result['mean']:+.3f}   Median: {result['median']:+.3f}")
    print(f"    Std:    {result['std']:.3f}")
    print(f"    P5/P95: {result['p5']:+.3f} / {result['p95']:+.3f}")
    print(f"    % positive: {result['pct_positive']:.1%}")
    sig = result["pct_positive"] >= 0.75
    if sig:
        _ok(f"Bootstrap passes: {result['pct_positive']:.1%} of iterations positive")
    else:
        _warn(f"Bootstrap low confidence: only {result['pct_positive']:.1%} positive")
    return result


# ── Swing walk-forward ─────────────────────────────────────────────────────────

def run_swing_walkforward(
    n_folds: int = 3,
    total_years: int = 5,
    symbols: Optional[List[str]] = None,
    atr_stop_mult: float = 0.5,
    atr_target_mult: float = 1.5,
    meta_model=None,
    pm_abstention_vix: float = 0.0,
    pm_abstention_spy_ma_days: int = 0,
    pm_abstention_spy_5d: bool = False,
    model_version: Optional[int] = None,
    transaction_cost_pct: float = 0.0005,
    purge_days: int = 10,
    embargo_days: Optional[int] = None,  # WF-1: post-test gap before next fold trains (None = same as purge_days)
    use_opportunity_score: bool = False,
    no_prefilters: bool = False,
    train_years: Optional[int] = None,  # Phase 3b: rolling window (None = expanding)
    earnings_blackout: Optional[Dict[str, set]] = None,  # Phase 2b: pre-built calendar
    macro_blocked_dates: Optional[set] = None,  # WF-5a: FOMC/NFP/CPI/GDP blocked dates
    benign_blocked_dates: Optional[set] = None,  # P1: adverse-regime dates
    wf_max_symbols: Optional[int] = None,  # cap universe to top-N by avg dollar volume
    feature_cache_workers: int = 0,
    feature_cache_executor: str = "process",
    feature_cache_disable: bool = False,
    sim_scan_interval_days: int = 1,
    use_factor_portfolio: bool = False,  # Phase D: bypass ML model, use factor composite scorer
    scorer_instance=None,  # Phase G: inject any callable scorer directly (overrides use_factor_portfolio)
    max_hold_bars_override: Optional[int] = None,  # Phase H+: force per-position hold cap
    no_atr_stops: bool = False,  # Phase 4: disable ATR stops entirely
    # Phase RA — REBALANCE mode
    rebalance_mode: bool = False,
    rebalance_days: int = 20,
    rebalance_target_n: int = 30,
    rebalance_sector_cap: float = 0.30,
    rebalance_add_threshold: int = 15,
    rebalance_drop_threshold: int = 30,
    rebalance_min_adv: float = 20_000_000.0,
    rebalance_regime_gate: bool = False,
    rebalance_regime_vix_bull: float = 20.0,
    rebalance_regime_vix_bear: float = 30.0,
    rebalance_regime_spy_ma_days: int = 200,
    rebalance_inv_vol: bool = False,
    rebalance_inv_vol_lookback: int = 20,
    rebalance_inv_vol_min_mult: float = 0.5,
    rebalance_inv_vol_max_mult: float = 2.0,
) -> WalkForwardReport:
    import yfinance as yf
    from app.backtesting.agent_simulator import AgentSimulator

    report = WalkForwardReport(model_type="swing")
    # Design note (deep-review pass 3): one pre-trained model is loaded here and
    # reused across all n_folds test windows. This is a generalization test, not
    # a true expanding-window WF with per-fold retrain. `is_true_walkforward`
    # stays False; downstream reporting/DSR consumers should treat fold Sharpes
    # as OOS evaluations of the SAME model, not of independently-fit models.
    model, version = _load_model("swing", version=model_version)
    if model is None:
        _err("No swing model found — retrain first.")
        return report
    _warn(
        f"NOTE: Using pre-trained swing model v{version} for all {n_folds} folds — "
        "this is a generalization test, not a true expanding-window walk-forward "
        "(no per-fold retrain). See WalkForwardReport.is_true_walkforward."
    )

    from app.utils.constants import RUSSELL_1000_TICKERS
    from app.data.universe_history import pit_union as _pit_union, historical_trade_symbols as _hist_syms
    symbols = symbols or list(RUSSELL_1000_TICKERS)

    # Download full history
    end_all = datetime.now()
    start_all = end_all - timedelta(days=total_years * 365 + 30)
    _subheader(f"Swing walk-forward: {n_folds} folds | {total_years}yr | "
               f"{len(symbols)} symbols | model v{version}")

    # WF-A2: augment download seed with symbols that appeared in the feature store
    # across the full backtest window (combats survivorship bias for delisted names).
    extra_seed = _hist_syms(start_all.date(), end_all.date(), trade_type="swing")
    if extra_seed:
        pre_len = len(symbols)
        symbols = sorted(set(symbols) | set(extra_seed))
        logger.info(
            "WF-A2: augmented download seed %d → %d (+%d historical)",
            pre_len, len(symbols), len(symbols) - pre_len,
        )

    logger.info("Downloading daily bars %s -> %s", start_all.date(), end_all.date())
    t0 = time.time()
    symbols_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start_all.date().isoformat(),
                             end=end_all.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 210:
                symbols_data[sym] = df
        except Exception:
            pass

    spy_raw = yf.download("SPY", start=start_all.date().isoformat(),
                          end=end_all.date().isoformat(), progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    spy_prices = spy_raw["close"]
    symbols_data["SPY"] = spy_raw  # make SPY available to factor_scorer regime gate

    # Download VIX for opportunity score (Phase 2a) and for external scorer regime gates.
    if use_opportunity_score or scorer_instance is not None:
        try:
            vix_raw = yf.download("^VIX", start=start_all.date().isoformat(),
                                  end=end_all.date().isoformat(), progress=False, auto_adjust=True)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
            vix_raw.columns = [c.lower() for c in vix_raw.columns]
            if len(vix_raw) >= 5:
                symbols_data["^VIX"] = vix_raw
                logger.info("VIX history loaded: %d rows (for opportunity score)", len(vix_raw))
        except Exception as exc:
            logger.warning("VIX download failed — opportunity score will use defaults: %s", exc)

    logger.info("Data loaded: %d symbols in %.1fs", len(symbols_data), time.time() - t0)

    # Speed-up: cap to top-N symbols by average dollar volume (price × volume).
    if wf_max_symbols and wf_max_symbols < len(symbols_data):
        _synthetic = {"^VIX", "SPY"}
        _ranked = []
        for sym, df in symbols_data.items():
            if sym in _synthetic:
                continue
            if "close" in df.columns and "volume" in df.columns:
                avg_dv = (df["close"] * df["volume"]).mean()
            else:
                avg_dv = 0.0
            _ranked.append((avg_dv, sym))
        _ranked.sort(reverse=True)
        _keep = {sym for _, sym in _ranked[:wf_max_symbols]} | _synthetic
        _before = len(symbols_data)
        symbols_data = {s: d for s, d in symbols_data.items() if s in _keep}
        logger.info("--wf-max-symbols %d: trimmed %d → %d symbols by avg dollar volume",
                    wf_max_symbols, _before, len(symbols_data))

    # WF-1: embargo_days defaults to purge_days when not specified.
    # Both sides of every test window now have a clean gap:
    #   train | purge_days | TEST | embargo_days | next_fold_train
    _embargo = embargo_days if embargo_days is not None else purge_days

    # Build fold boundaries.
    # train_years=None: expanding window (train always from start_all).
    # train_years=N: rolling window (train starts at train_end - N years per fold).
    # purge_days gap between train_end and test_start prevents 5-day label leakage.
    # embargo_days gap after test_end prevents test rows appearing in next train.
    segment_days = int(total_years * 365 / (n_folds + 1))
    fold_boundaries = []
    for fold_idx in range(n_folds):
        train_end_dt = end_all - timedelta(days=segment_days * (n_folds - fold_idx))
        test_start_dt = train_end_dt + timedelta(days=purge_days + 1)
        # test_end must leave embargo_days gap before the next fold's train_end,
        # otherwise fold N's test rows (with their lookback features) bleed into
        # fold N+1's training set.  Without this offset, adjacent folds abut and
        # embargo_days has no effect.
        raw_test_end_dt = train_end_dt + timedelta(days=segment_days - _embargo)
        # For all but the last fold, ensure next train starts at least embargo_days after test_end
        # (enforced implicitly: next fold's train_end is the next segment boundary,
        #  so expanding window naturally excludes the test window of the current fold)
        if train_years is not None:
            # Rolling window: shift train start to exclude rows within embargo of prior test
            fold_train_start = max(
                start_all.date(),
                (train_end_dt - timedelta(days=train_years * 365)).date(),
            )
        else:
            fold_train_start = start_all.date()
        fold_boundaries.append((
            fold_train_start,
            train_end_dt.date(),
            test_start_dt.date(),
            min(raw_test_end_dt.date(), end_all.date()),
            _embargo,  # carry embargo into the fold runner for logging
        ))

    # Load sector ETF bars once for all folds (PIT sector-neutral features)
    _sector_etf_bars_wf: Optional[dict] = None
    _etf_hist_path_wf = Path("data/sector_etf/sector_etf_history.parquet")
    if _etf_hist_path_wf.exists():
        try:
            import pandas as _pd_etf
            _ef_wf = _pd_etf.read_parquet(_etf_hist_path_wf)
            _sector_etf_bars_wf = {}
            for _etf, _grp in _ef_wf.groupby("etf"):
                _grp_s = _grp.sort_values("date")
                _sector_etf_bars_wf[_etf] = [
                    (row["date"], float(row["close"]))
                    for _, row in _grp_s.iterrows()
                ]
            logger.info("Walk-forward: loaded sector ETF bars for %d ETFs", len(_sector_etf_bars_wf))
        except Exception as _exc_etf:
            logger.warning("Walk-forward: could not load sector_etf_history.parquet — %s", _exc_etf)

    def _run_swing_fold(args):
        fold_idx, tr_start, tr_end, te_start, te_end, emb = args
        _subheader(f"Fold {fold_idx}/{n_folds}  train:{tr_start}->{tr_end}  "
                   f"test:{te_start}->{te_end}  purge={purge_days}d  embargo={emb}d")
        t_fold = time.time()
        # Point-in-time filter: only use symbols that were in the index at fold train start.
        # Synthetic symbols (^VIX, VIX, SPY) bypass the filter — they're needed for
        # regime gates and opportunity score regardless of index membership.
        # WF-A2/A3: PIT filter using Russell 1000 (matches training universe).
        # pit_union() captures members at both fold endpoints (catches mid-fold adds/removes)
        # plus DB-sourced historical names for survivorship-bias correction.
        extra = _hist_syms(tr_start, te_end, trade_type="swing")
        pit_members = set(_pit_union("russell1000", tr_start, te_end, extra_symbols=extra))
        _synthetic = {"^VIX", "VIX", "SPY"}
        fold_symbols_data = {
            s: d for s, d in symbols_data.items()
            if s in pit_members or s in _synthetic
        }
        # Build per-fold feature cache (pre-computes all (sym, day) features in parallel)
        _feature_cache = None
        if not feature_cache_disable:
            try:
                from app.backtesting.feature_cache import build_feature_cache as _build_fc
                import os as _os
                _test_days = sorted({
                    d.date() if hasattr(d, "date") else d
                    for df in fold_symbols_data.values()
                    for d in df.index
                    if te_start <= (d.date() if hasattr(d, "date") else d) <= te_end
                })
                _vix_df = fold_symbols_data.get("^VIX")
                if _vix_df is None:
                    _vix_df = fold_symbols_data.get("VIX")
                _vix_s = _vix_df["close"] if _vix_df is not None and "close" in _vix_df.columns else None
                _feat_names = getattr(model, "feature_names", None) or []
                _workers = feature_cache_workers or max(2, min(_os.cpu_count() or 4, MAX_WORKERS))
                logger.info("Fold %d: building feature cache (%d syms × %d days, %d %s workers)",
                            fold_idx, len(fold_symbols_data), len(_test_days), _workers, feature_cache_executor)
                _feature_cache = _build_fc(
                    symbols_data=fold_symbols_data,
                    trading_days=_test_days,
                    feature_names=_feat_names,
                    vix_history=_vix_s,
                    sector_etf_bars=_sector_etf_bars_wf,
                    workers=_workers,
                    executor=feature_cache_executor,
                )
            except Exception as _exc:
                logger.warning("Feature cache build failed, falling back to live compute: %s", _exc)
                _feature_cache = None

            # Lockstep diagnostic: verify cache breadth before sim runs.
            # If symbols_with(mid_day) << len(fold_symbols_data), cache is sparse
            # (date-key mismatch or worker crash) and WF results will be invalid.
            if _feature_cache is not None and _feature_cache.n_symbols > 0:
                _mid = _test_days[len(_test_days) // 2]
                _n_on_mid = len(_feature_cache.symbols_with(_mid))
                logger.info(
                    "Fold %d cache breadth: %d/%d symbols populated; mid-fold day %s has %d symbols scored",
                    fold_idx, _feature_cache.n_symbols, len(fold_symbols_data), _mid, _n_on_mid,
                )
                if _n_on_mid < len(fold_symbols_data) * 0.5:
                    logger.warning(
                        "Fold %d: only %d/%d symbols available on %s — "
                        "lockstep scoring will be degraded. Check for date-key mismatch or worker crashes.",
                        fold_idx, _n_on_mid, len(fold_symbols_data), _mid,
                    )

        # Phase RB.1: build regime gate fn if requested (PIT-safe, uses fold's SPY+VIX)
        _regime_gate_fn = None
        if rebalance_mode and rebalance_regime_gate:
            _regime_gate_fn = _make_regime_gate_fn(
                fold_symbols_data,
                spy_ma_days=rebalance_regime_spy_ma_days,
                vix_bull=rebalance_regime_vix_bull,
                vix_bear=rebalance_regime_vix_bear,
            )

        _factor_scorer_inst = scorer_instance  # Phase G: externally injected scorer takes priority
        if _factor_scorer_inst is None and use_factor_portfolio:
            from app.ml.factor_scorer import FactorPortfolioScorer
            _factor_scorer_inst = FactorPortfolioScorer(
                top_n=20,
                top_n_short=15,
                long_short=True,  # Phase F: directional L/S, 40% net long
            )

        # Phase F: factor portfolio needs wider limits than the ML-signal defaults
        # (top-20 longs + top-15 shorts = 35 positions, 15% drawdown tolerance)
        if _factor_scorer_inst is not None:
            from app.agents.risk_rules import RiskLimits
            _sim_limits = RiskLimits(
                MAX_OPEN_POSITIONS=40,          # headroom for 20L + 15S + slack
                MAX_POSITION_SIZE_PCT=0.05,     # 5% per position (same)
                MAX_ACCOUNT_DRAWDOWN_PCT=0.15,  # 15% drawdown gate (vs 5% for single trades)
                MAX_DAILY_LOSS_PCT=0.05,        # 5% daily loss (vs 2%)
                MAX_PORTFOLIO_HEAT_PCT=0.30,    # 30% heat (diversified portfolio)
                MAX_SECTOR_CONCENTRATION_PCT=0.30,  # 30% sector cap
            )
        else:
            _sim_limits = None  # use AgentSimulator defaults

        sim = AgentSimulator(
            model=model,
            atr_stop_mult=atr_stop_mult,
            atr_target_mult=atr_target_mult,
            meta_model=meta_model,
            pm_abstention_vix=pm_abstention_vix,
            pm_abstention_spy_ma_days=pm_abstention_spy_ma_days,
            pm_abstention_spy_5d=pm_abstention_spy_5d,
            transaction_cost_pct=transaction_cost_pct,
            use_opportunity_score=use_opportunity_score,
            no_prefilters=no_prefilters,
            earnings_blackout=earnings_blackout,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            feature_cache=_feature_cache,
            sim_scan_interval_days=sim_scan_interval_days,
            factor_scorer=_factor_scorer_inst,
            limits=_sim_limits,
            max_hold_bars_override=max_hold_bars_override,
            no_atr_stops=no_atr_stops,
            rebalance_mode=rebalance_mode,
            rebalance_days=rebalance_days,
            rebalance_target_n=rebalance_target_n,
            rebalance_sector_cap=rebalance_sector_cap,
            rebalance_add_threshold=rebalance_add_threshold,
            rebalance_drop_threshold=rebalance_drop_threshold,
            rebalance_min_adv=rebalance_min_adv,
            rebalance_regime_fn=_regime_gate_fn,
            rebalance_inv_vol=rebalance_inv_vol,
            rebalance_inv_vol_lookback=rebalance_inv_vol_lookback,
            rebalance_inv_vol_min_mult=rebalance_inv_vol_min_mult,
            rebalance_inv_vol_max_mult=rebalance_inv_vol_max_mult,
        )
        result = sim.run(
            fold_symbols_data,
            start_date=te_start,
            end_date=te_end,
            spy_prices=spy_prices,
        )
        elapsed = time.time() - t_fold
        stop_exits = result.exit_breakdown.get("STOP", 0)
        stop_rate = stop_exits / max(result.total_trades, 1)
        # WF-1: compute additional metrics
        trade_rets = (getattr(result, "trade_returns", None)
                      or [t.pnl_pct for t in (getattr(result, "trades", None) or [])
                          if t.exit_reason != "OPEN"])
        pf = _compute_profit_factor(trade_rets)
        fold_yrs = _fold_years(te_start, te_end)
        calmar = _compute_calmar(result.total_return_pct, result.max_drawdown_pct, fold_yrs)
        equity = getattr(result, "equity_curve", []) or []
        kr = _compute_k_ratio(equity)
        _ok(f"Fold {fold_idx} done in {elapsed:.1f}s — {result.total_trades} trades, "
            f"Sharpe {result.sharpe_ratio:.2f}  PF={pf:.2f}  Calmar={calmar:.2f}")
        return FoldResult(
            fold=fold_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe=result.sharpe_ratio,
            max_drawdown=result.max_drawdown_pct,
            total_return=result.total_return_pct,
            stop_exit_rate=stop_rate,
            model_version=version,
            profit_factor=pf,
            calmar_ratio=calmar,
            k_ratio=kr,
            n_obs=max(len(equity) - 1, 0),  # daily returns count for DSR
        )

    from concurrent.futures import ThreadPoolExecutor
    fold_args = [
        (i + 1, tr_start, tr_end, te_start, te_end, emb)
        for i, (tr_start, tr_end, te_start, te_end, emb) in enumerate(fold_boundaries)
    ]
    _fold_workers = min(n_folds, MAX_FOLD_WORKERS)
    with ThreadPoolExecutor(max_workers=_fold_workers) as pool:
        results = list(pool.map(_run_swing_fold, fold_args))
    report.folds = sorted(results, key=lambda f: f.fold)

    return report


# ── Intraday walk-forward ──────────────────────────────────────────────────────

def run_intraday_walkforward(
    n_folds: int = 3,
    total_days: int = 730,
    symbols: Optional[List[str]] = None,
    meta_model=None,
    pm_abstention_vix: float = 0.0,
    pm_abstention_spy_ma_days: int = 0,
    scan_offsets: Optional[List[int]] = None,
    model_version: Optional[int] = None,
    transaction_cost_pct: float = 0.0015,
    purge_days: int = 2,
    embargo_days: Optional[int] = None,  # WF-1: post-test gap (None = same as purge_days)
    use_opportunity_score: bool = False,
    use_dispersion_gate: bool = False,
    earnings_blackout: Optional[Dict[str, set]] = None,  # Phase 2b: pre-built calendar
    macro_blocked_dates: Optional[set] = None,  # WF-5a: FOMC/NFP/CPI/GDP blocked dates
    benign_blocked_dates: Optional[set] = None,  # P1: adverse-regime dates
    use_regime_gate: bool = False,  # Phase R5: regime gates (R5-A/B/C)
    regime_map: Optional[Dict] = None,  # Phase R5: {date: label} from WF-4 regime.py
) -> WalkForwardReport:
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
    from app.data.intraday_cache import load_many, available_symbols as poly_syms

    report = WalkForwardReport(model_type="intraday")
    # Design note (deep-review pass 3): same single-model-across-folds convention
    # as the swing harness. See run_swing_walkforward for rationale.
    model, version = _load_model("intraday", version=model_version)
    if model is None:
        _err("No intraday model found — retrain first.")
        return report

    _warn(
        f"NOTE: Using pre-trained intraday model v{version} for all {n_folds} folds — "
        "this is a generalization test, not a true expanding-window walk-forward "
        "(no per-fold retrain). See WalkForwardReport.is_true_walkforward."
    )

    from app.utils.constants import RUSSELL_1000_TICKERS
    from app.data.universe_history import members_at as _members_at
    symbols = symbols or list(RUSSELL_1000_TICKERS)

    end_all = datetime.now().date()
    start_all = end_all - timedelta(days=total_days + 10)

    _subheader(f"Intraday walk-forward: {n_folds} folds | {total_days}d | "
               f"{len(symbols)} symbols | model v{version}")

    # Load 5-min data from Polygon cache
    cache_syms = set(poly_syms())
    if cache_syms:
        logger.info("Loading from Polygon cache (%d symbols available)", len(cache_syms))
        symbols_data = load_many(
            [s for s in symbols if s in cache_syms],
            start=start_all, end=end_all,
        )
    else:
        _warn("Polygon cache empty — falling back to yfinance (≤55 days)")
        import yfinance as yf
        symbols_data = {}
        for sym in symbols[:100]:  # cap at 100 for yfinance fallback
            try:
                df = yf.download(sym, period="55d", interval="5m",
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 12:
                    symbols_data[sym] = df
            except Exception:
                pass

    spy_data: Optional[pd.DataFrame] = None
    if "SPY" in symbols_data:
        spy_data = symbols_data["SPY"]

    # Phase 86: fetch SPY daily bars for market-condition features
    spy_daily_data: Optional[pd.DataFrame] = None
    try:
        import yfinance as yf
        _spy_daily = yf.download("SPY", period="3y", progress=False, auto_adjust=True)
        if isinstance(_spy_daily.columns, pd.MultiIndex):
            _spy_daily.columns = _spy_daily.columns.get_level_values(0)
        _spy_daily.columns = [c.lower() for c in _spy_daily.columns]
        if len(_spy_daily) >= 6:
            spy_daily_data = _spy_daily
            logger.info("SPY daily bars loaded: %d rows", len(spy_daily_data))
    except Exception as exc:
        logger.warning("SPY daily fetch failed (Phase 86 features will use defaults): %s", exc)

    # Download VIX daily bars for opportunity score (Phase 2a)
    if use_opportunity_score:
        try:
            import yfinance as yf
            _vix_daily = yf.download("^VIX", period="3y", progress=False, auto_adjust=True)
            if isinstance(_vix_daily.columns, pd.MultiIndex):
                _vix_daily.columns = _vix_daily.columns.get_level_values(0)
            _vix_daily.columns = [c.lower() for c in _vix_daily.columns]
            if len(_vix_daily) >= 5:
                symbols_data["^VIX"] = _vix_daily
                logger.info("VIX daily bars loaded: %d rows (for opportunity score)", len(_vix_daily))
        except Exception as exc:
            logger.warning("VIX download failed — opportunity score will use SPY-only: %s", exc)

    logger.info("Intraday data loaded: %d symbols", len(symbols_data))

    # Fold boundaries — split the test period into n_folds equal segments.
    # Exclude daily-bar overlay symbols (^VIX, SPY) which may span 3y and would
    # inflate all_days_sorted far beyond the intraday data window.
    _daily_overlay_keys = {"^VIX", "SPY"}
    all_days_sorted = sorted({
        d for sym, df in symbols_data.items()
        if sym not in _daily_overlay_keys
        for d in pd.to_datetime(df.index).date
    })
    if not all_days_sorted:
        _err("No trading days found in data.")
        return report

    # WF-1: embargo defaults to purge_days when not specified.
    _embargo_intra = embargo_days if embargo_days is not None else purge_days

    # purge_days trading-day gap between train_end and test_start prevents intraday
    # label leakage (2-day hold means labels at train boundary use test-window bars).
    # embargo_days post-test gap prevents test rows entering next fold's training set.
    segment_size = max(1, len(all_days_sorted) // (n_folds + 1))
    fold_boundaries = []
    for fi in range(n_folds):
        tr_end_idx = segment_size * (fi + 1) - 1
        te_start_idx = min(tr_end_idx + purge_days + 1, len(all_days_sorted) - 1)
        te_end_idx = min(segment_size * (fi + 1) + segment_size - 1, len(all_days_sorted) - 1)
        fold_boundaries.append((
            all_days_sorted[0],
            all_days_sorted[tr_end_idx],
            all_days_sorted[te_start_idx],
            all_days_sorted[te_end_idx],
            _embargo_intra,
        ))

    def _run_intraday_fold(args):
        fold_idx, tr_start, tr_end, te_start, te_end, emb = args
        _subheader(f"Fold {fold_idx}/{n_folds}  train:{tr_start}->{tr_end}  "
                   f"test:{te_start}->{te_end}  purge={purge_days}d  embargo={emb}d")
        t_fold = time.time()
        # Point-in-time filter: only use symbols that were in the index at fold train start
        pit_members = set(_members_at("russell1000", tr_start))
        fold_symbols_data = {s: d for s, d in symbols_data.items() if s in pit_members}
        sim = IntradayAgentSimulator(
            model=model,
            meta_model=meta_model,
            pm_abstention_vix=pm_abstention_vix,
            pm_abstention_spy_ma_days=pm_abstention_spy_ma_days,
            scan_offsets=scan_offsets,
            transaction_cost_pct=transaction_cost_pct,
            use_opportunity_score=use_opportunity_score,
            use_dispersion_gate=use_dispersion_gate,
            earnings_blackout=earnings_blackout,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            use_regime_gate=use_regime_gate,
            regime_map=regime_map,
        )
        result = sim.run(
            fold_symbols_data,
            spy_data=spy_data,
            start_date=te_start,
            end_date=te_end,
            spy_daily_data=spy_daily_data,
        )
        elapsed = time.time() - t_fold
        stop_exits = result.exit_breakdown.get("STOP", 0)
        stop_rate = stop_exits / max(result.total_trades, 1)
        # WF-1: additional metrics
        trade_rets = (getattr(result, "trade_returns", None)
                      or [t.pnl_pct for t in (getattr(result, "trades", None) or [])
                          if t.exit_reason != "OPEN"])
        pf = _compute_profit_factor(trade_rets)
        fold_yrs = _fold_years(te_start, te_end)
        calmar = _compute_calmar(result.total_return_pct, result.max_drawdown_pct, fold_yrs)
        equity = getattr(result, "equity_curve", []) or []
        kr = _compute_k_ratio(equity)
        _ok(f"Fold {fold_idx} done in {elapsed:.1f}s — {result.total_trades} trades, "
            f"Sharpe {result.sharpe_ratio:.2f}  PF={pf:.2f}  Calmar={calmar:.2f}")
        return FoldResult(
            fold=fold_idx,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe=result.sharpe_ratio,
            max_drawdown=result.max_drawdown_pct,
            total_return=result.total_return_pct,
            stop_exit_rate=stop_rate,
            model_version=version,
            profit_factor=pf,
            calmar_ratio=calmar,
            k_ratio=kr,
            n_obs=max(len(equity) - 1, 0),  # daily returns count for DSR
        )

    fold_args = [
        (i + 1, tr_start, tr_end, te_start, te_end, emb)
        for i, (tr_start, tr_end, te_start, te_end, emb) in enumerate(fold_boundaries)
    ]
    report.folds = [_run_intraday_fold(args) for args in fold_args]

    return report


# ── WF-3: CPCV helpers ────────────────────────────────────────────────────────

def _momentum_baseline_scorer(lookback_days: int = 60):
    """Return a factor_scorer callable that ranks symbols by trailing return (PIT-safe)."""
    import pandas as _pd

    def _scorer(day, symbols_data, vix_history=None):
        ts = _pd.Timestamp(day)
        scores = []
        for sym, df in symbols_data.items():
            if sym in ("SPY", "^VIX", "VIX"):
                continue
            if df is None or df.empty:
                continue
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                continue
            hist = df[df.index < ts][close_col]
            if len(hist) < lookback_days + 1:
                continue
            ret = float(hist.iloc[-1] / hist.iloc[-(lookback_days + 1)] - 1)
            scores.append((sym, ret))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    return _scorer


def _run_cpcv_swing(args, symbols, swing_ver, meta_model, earnings_cal, passed):
    """Run CPCV for swing model and print the result."""
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.strategies.swing import SwingStrategy
    from app.utils.constants import RUSSELL_1000_TICKERS
    model, version = _load_model("swing", version=swing_ver)
    if model is None:
        _warn("CPCV: no swing model found — skipping")
        return
    syms = symbols or list(RUSSELL_1000_TICKERS)

    _factor_scorer = None
    if getattr(args, "rebalance_momentum_baseline", False):
        _factor_scorer = _momentum_baseline_scorer(lookback_days=60)
        print("  CPCV: momentum baseline mode — bypassing ML model, using 60d trailing return ranker")
    elif getattr(args, "rebalance_ic_composite", False):
        from app.ml.factor_scorer import IcCompositeScorer
        _factor_scorer = IcCompositeScorer()
        print("  CPCV: IC composite mode — Phase 88 deterministic IC-weighted scorer (v219 weights)")

    strategy = SwingStrategy(
        model=model, version=version, symbols=syms,
        atr_stop_mult=args.stop_mult, atr_target_mult=args.target_mult,
        meta_model=meta_model,
        pm_abstention_vix=args.pm_abstention_vix,
        pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
        pm_abstention_spy_5d=args.pm_abstention_spy_5d,
        transaction_cost_pct=args.swing_cost_bps / 10_000 / 2,
        use_opportunity_score=args.pm_opportunity_score,
        no_prefilters=args.no_prefilters,
        earnings_blackout=earnings_cal,
        feature_cache_workers=args.feature_cache_workers,
        feature_cache_executor=args.feature_cache_executor,
        feature_cache_disable=args.feature_cache_disable,
        sim_scan_interval_days=args.sim_scan_interval_days,
        rebalance_mode=getattr(args, "rebalance_mode", False),
        rebalance_days=getattr(args, "rebalance_days", 20),
        rebalance_target_n=getattr(args, "rebalance_target_n", 30),
        rebalance_sector_cap=getattr(args, "rebalance_sector_cap", 0.30),
        rebalance_add_threshold=getattr(args, "rebalance_add_threshold", 15),
        rebalance_drop_threshold=getattr(args, "rebalance_drop_threshold", 30),
        rebalance_min_adv=getattr(args, "rebalance_min_adv", 0.0),
        rebalance_regime_gate=getattr(args, "rebalance_regime_gate", False),
        rebalance_regime_spy_ma_days=getattr(args, "rebalance_regime_spy_ma_days", 200),
        rebalance_regime_vix_bull=getattr(args, "rebalance_regime_vix_bull", 20.0),
        rebalance_regime_vix_bear=getattr(args, "rebalance_regime_vix_bear", 30.0),
        rebalance_inv_vol=getattr(args, "rebalance_inv_vol", False),
        rebalance_inv_vol_lookback=getattr(args, "rebalance_inv_vol_lookback", 20),
        rebalance_inv_vol_min_mult=getattr(args, "rebalance_inv_vol_min_mult", 0.5),
        rebalance_inv_vol_max_mult=getattr(args, "rebalance_inv_vol_max_mult", 2.0),
        factor_scorer=_factor_scorer,
        no_atr_stops=getattr(args, "no_atr_stops", False),
    )
    strategy.model_type = "swing"
    from datetime import datetime, timedelta
    end_all = datetime.now()
    start_all = end_all - timedelta(days=args.years * 365 + 30)
    strategy.fetch_data(start_all, end_all)
    cpcv_result = run_cpcv(
        strategy=strategy,
        purge_days=args.swing_purge_days,
        embargo_days=args.swing_embargo_days,
        n_folds=args.cpcv_k,
        n_paths=args.cpcv_paths,
        total_years=args.years,
        train_years=args.swing_train_years,
        allow_sacred_holdout=args.allow_sacred_holdout,
    )
    cpcv_result.print()
    return cpcv_result


def _run_cpcv_intraday(args, symbols, intraday_ver, intraday_meta_model, earnings_cal, passed):
    """Run CPCV for intraday model and print the result."""
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.strategies.intraday import IntradayStrategy
    from app.utils.constants import RUSSELL_1000_TICKERS
    from datetime import date as _date, timedelta
    model, version = _load_model("intraday", version=intraday_ver)
    if model is None:
        _warn("CPCV: no intraday model found — skipping")
        return
    syms = symbols or list(RUSSELL_1000_TICKERS)
    intraday_scan_offsets = [12, 18, 24] if args.intraday_multi_scan else None
    strategy = IntradayStrategy(
        model=model, version=version, symbols=syms,
        meta_model=intraday_meta_model,
        pm_abstention_vix=args.pm_abstention_vix,
        pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
        scan_offsets=intraday_scan_offsets,
        transaction_cost_pct=args.intraday_cost_bps / 10_000 / 2,
        use_opportunity_score=args.pm_opportunity_score,
        use_dispersion_gate=args.dispersion_gate,
        earnings_blackout=earnings_cal,
    )
    strategy.model_type = "intraday"
    end_date = _date.today()
    start_date = end_date - timedelta(days=args.days + 10)
    strategy.fetch_data(start_date, end_date)
    cpcv_result = run_cpcv(
        strategy=strategy,
        purge_days=args.intraday_purge_days,
        embargo_days=args.intraday_embargo_days,
        n_folds=args.cpcv_k,
        n_paths=args.cpcv_paths,
        total_days=args.days,
        allow_sacred_holdout=args.allow_sacred_holdout,
    )
    cpcv_result.print()
    return cpcv_result


# ── Entry point ───────────────────────────────────────────────────────────────

def _cleanup_workers(signum=None, frame=None) -> None:
    """Kill any surviving multiprocessing worker processes on exit/interrupt."""
    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    for child in multiprocessing.active_children():
        try:
            child.join(timeout=1.0)
            if child.is_alive():
                child.kill()
        except Exception:
            pass
    if signum is not None:
        raise SystemExit(130)


# Register cleanup on normal exit and on Ctrl-C / SIGTERM so WF workers never
# survive as orphaned processes when a run is interrupted.
atexit.register(_cleanup_workers)
signal.signal(signal.SIGINT, _cleanup_workers)
try:
    signal.signal(signal.SIGTERM, _cleanup_workers)
except (AttributeError, ValueError):
    pass  # SIGTERM unavailable in some Windows console contexts


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward Tier 3 validation")
    parser.add_argument("--model", choices=["swing", "intraday", "both"], default="both")
    parser.add_argument("--folds", type=int, default=3, help="Number of OOS folds")
    parser.add_argument("--years", type=int, default=5,
                        help="Total years of swing history (default: 5)")
    parser.add_argument("--days", type=int, default=730,
                        help="Total days of intraday history (default: 730)")
    parser.add_argument("--symbols", nargs="+", default=None, metavar="TICKER")
    parser.add_argument("--stop-mult", type=float, default=0.5,
                        help="ATR stop multiplier (default: 0.5)")
    parser.add_argument("--target-mult", type=float, default=1.5,
                        help="ATR target multiplier (default: 1.5)")
    parser.add_argument("--no-atr-stops", action="store_true", default=False,
                        help="Phase 4: disable ATR stops entirely. Positions hold to max_hold_bars."
                             " Overrides --stop-mult. Use with --no-pm-opportunity-score for clean signal isolation.")
    parser.add_argument("--rebalance-mode", action="store_true", default=False,
                        help="Phase RA: use top-N portfolio rebalance instead of signal entries")
    parser.add_argument("--rebalance-days", type=int, default=20,
                        help="Calendar days between rebalances (default: 20)")
    parser.add_argument("--rebalance-target-n", type=int, default=30,
                        help="Target number of positions in rebalance mode (default: 30)")
    parser.add_argument("--rebalance-sector-cap", type=float, default=0.30,
                        help="Max sector weight in rebalance portfolio (default: 0.30)")
    parser.add_argument("--rebalance-add-threshold", type=int, default=15,
                        help="Add a symbol if its rank <= this (default: 15)")
    parser.add_argument("--rebalance-drop-threshold", type=int, default=30,
                        help="Drop a symbol if its rank > this (default: 30)")
    parser.add_argument("--rebalance-min-adv", type=float, default=20_000_000.0,
                        help="Min avg daily dollar volume for liquidity filter (default: 20M)")
    parser.add_argument("--rebalance-regime-gate", action="store_true", default=False,
                        help="Phase RB.1: scale gross exposure by regime "
                             "(Bull=100%%, Neutral=70%%, Bear=30%%)")
    parser.add_argument("--rebalance-regime-vix-bull", type=float, default=20.0,
                        help="VIX threshold below which regime is BULL (default: 20)")
    parser.add_argument("--rebalance-regime-vix-bear", type=float, default=30.0,
                        help="VIX threshold at/above which regime is BEAR (default: 30)")
    parser.add_argument("--rebalance-regime-spy-ma-days", type=int, default=200,
                        help="SPY MA lookback for regime detection (default: 200)")
    parser.add_argument("--rebalance-inv-vol", action="store_true", default=False,
                        help="Phase RB.2: use inverse-volatility position sizing")
    parser.add_argument("--rebalance-inv-vol-lookback", type=int, default=20,
                        help="Lookback days for realized vol estimate (default: 20)")
    parser.add_argument("--rebalance-inv-vol-min-mult", type=float, default=0.5,
                        help="Min weight vs equal weight (default: 0.5x)")
    parser.add_argument("--rebalance-inv-vol-max-mult", type=float, default=2.0,
                        help="Max weight vs equal weight (default: 2.0x)")
    parser.add_argument("--rebalance-momentum-baseline", action="store_true", default=False,
                        help="CPCV diagnostic: replace v216 with 60d trailing-return momentum ranker "
                             "(same REBALANCE harness, same regime gate + inv-vol sizing)")
    parser.add_argument("--rebalance-ic-composite", action="store_true", default=False,
                        help="Phase 88: use IC-weighted deterministic factor composite (v219) "
                             "instead of ML model. Weights from 2026-05-24 IC audit h20 IC IR.")
    parser.add_argument("--meta-model-version", type=int, default=0,
                        help="Swing MetaLabelModel version to load (0 = none)")
    parser.add_argument("--intraday-meta-model-version", type=int, default=0,
                        help="Intraday MetaLabelModel version to load (0 = none)")
    parser.add_argument("--pm-abstention-vix", type=float, default=0.0,
                        help="PM abstention gate: skip entries when VIX >= this level (0 = off)")
    parser.add_argument("--pm-abstention-spy-ma-days", type=int, default=0,
                        help="PM abstention gate: skip entries when SPY < N-day SMA (0 = off)")
    parser.add_argument("--pm-abstention-spy-5d", action="store_true", default=False,
                        help="PM abstention gate: skip entries when SPY 5-day return <= 0 (Phase 55)")
    parser.add_argument("--intraday-multi-scan", action="store_true", default=False,
                        help="Phase 51: scan at offsets [12, 18, 24] instead of single scan at 12")
    parser.add_argument("--intraday-model-version", type=int, default=0,
                        help="Load a specific intraday model version (0 = active). "
                             "Use this to re-test historical/retired models.")
    parser.add_argument("--swing-model-version", type=int, default=0,
                        help="Load a specific swing model version (0 = active). "
                             "Use this to re-test historical/retired models.")
    parser.add_argument("--record-results", action="store_true", default=False,
                        help="Write tier3 Sharpe + gate result back to ModelVersion DB record")
    parser.add_argument("--swing-cost-bps", type=float, default=5.0,
                        help="Round-trip transaction cost in bps for swing (default: 5bps)")
    parser.add_argument("--intraday-cost-bps", type=float, default=15.0,
                        help="Round-trip transaction cost in bps for intraday (default: 15bps)")
    parser.add_argument("--swing-purge-days", type=int, default=85,
                        help="Calendar days to skip between train_end and test_start for swing. "
                             "Must be >= 60d feature lookback + 20d label horizon + 5d buffer = 85d. "
                             "Old default was 15 (too short, caused train/test overlap). Default: 85")
    parser.add_argument("--swing-embargo-days", type=int, default=None,
                        help="WF-1: Calendar days to skip after test_end before next fold trains "
                             "(defaults to --swing-purge-days if not set)")
    parser.add_argument("--intraday-purge-days", type=int, default=2,
                        help="Trading days to skip between train_end and test_start for intraday "
                             "(prevents 2-day hold label leakage; default: 2)")
    parser.add_argument("--intraday-embargo-days", type=int, default=None,
                        help="WF-1: Trading days to skip after test_end before next fold trains "
                             "(defaults to --intraday-purge-days if not set)")
    parser.add_argument("--pm-opportunity-score", action="store_true", default=True,
                        help="WF-5a: apply PM continuous opportunity score gate in simulation "
                             "(score<0.35=skip, 0.35-0.65=cap at 2 candidates). On by default. "
                             "Use --no-pm-opportunity-score to disable.")
    parser.add_argument("--no-pm-opportunity-score", dest="pm_opportunity_score",
                        action="store_false",
                        help="WF-5a: disable the PM opportunity score gate.")
    parser.add_argument("--dispersion-gate", action="store_true", default=True,
                        help="WF-5a: skip intraday entries on days where cross-sectional return "
                             "dispersion < 0.5x rolling 60-day median. On by default. "
                             "Use --no-dispersion-gate to disable.")
    parser.add_argument("--no-dispersion-gate", dest="dispersion_gate",
                        action="store_false",
                        help="WF-5a: disable the dispersion gate.")
    parser.add_argument("--no-prefilters", action="store_true", default=False,
                        help="Phase 3a: bypass RSI 40-70 and EMA20/50 trader pre-filters in swing. "
                             "Lets ML model score the full universe without rule-based entry gates.")
    parser.add_argument("--swing-train-years", type=int, default=None,
                        help="Phase 3b: rolling training window per fold — limit each fold's "
                             "training data to the N most recent years before train_end "
                             "(None = expanding window). Use 2 to exclude 2021-2022 bull regime.")
    parser.add_argument("--earnings-blackout", action="store_true", default=True,
                        help="WF-5a: skip entries within earnings blackout window "
                             "(swing: 3d before; intraday: 1d before / 3d after). On by default. "
                             "Use --no-earnings-blackout to disable.")
    parser.add_argument("--no-earnings-blackout", dest="earnings_blackout",
                        action="store_false",
                        help="WF-5a: disable the earnings blackout gate.")
    parser.add_argument("--regime-gate", action="store_true", default=False,
                        help="Phase R5: enable intraday regime gates (R5-A/B/C). Downloads SPY+VIX "
                             "history to build regime map. Off by default (requires WF-4 regime tagger).")
    parser.add_argument("--macro-gate", action="store_true", default=True,
                        help="WF-5a: block entries on FOMC/NFP/CPI/GDP days. On by default. "
                             "Use --no-macro-gate to disable.")
    parser.add_argument("--no-macro-gate", dest="macro_gate",
                        action="store_false",
                        help="WF-5a: disable the macro event gate.")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="Phase 1d: run N bootstrap iterations with ±30d fold perturbation "
                             "to quantify selection bias. 0 = disabled. 100 recommended.")
    # WF-3: CPCV flags
    parser.add_argument("--cpcv", action="store_true", default=False,
                        help="WF-3: run Combinatorial Purged Cross-Validation instead of "
                             "standard expanding walk-forward. Requires --cpcv-k and --cpcv-paths.")
    parser.add_argument("--cpcv-k", type=int, default=6,
                        help="WF-3: number of CPCV groups k (default: 6). "
                             "C(k, paths) combinations will be tested.")
    parser.add_argument("--cpcv-paths", type=int, default=2,
                        help="WF-3: number of test paths per combination (default: 2). "
                             "Larger = more combinations, slower but higher statistical power.")
    # P1: BenignGate — block test-fold entries on adverse-regime days
    parser.add_argument("--benign-gate", action="store_true", default=False,
                        help="P1: block new entries on days where PIT composite regime score < "
                             "BENIGN_REGIME_THRESHOLD (from macro_history.parquet). "
                             "Off by default. Adds ~1s to setup time.")
    parser.add_argument("--wf-max-symbols", type=int, default=0, metavar="N",
                        help="Speed-up: cap the swing WF universe to the top-N symbols by "
                             "average dollar volume. 0 = use all (default). "
                             "300 gives ~2-3x speedup with negligible Sharpe impact.")
    parser.add_argument("--feature-cache-workers", type=int, default=0,
                        help="Workers for per-fold feature pre-computation (0 = auto: "
                             "min(cpu_count, 12)). Ignored when --feature-cache-disable is set.")
    parser.add_argument("--feature-cache-executor", choices=["process", "thread"],
                        default="process",
                        help="Executor for feature cache build: 'process' (default, avoids GIL) "
                             "or 'thread' (lower overhead, useful on Windows with small universes).")
    parser.add_argument("--feature-cache-disable", action="store_true", default=False,
                        help="Disable the feature cache and fall back to per-day per-symbol "
                             "engineer_features() calls (original behavior). Use for debugging.")
    parser.add_argument("--sim-scan-interval-days", type=int, default=1,
                        help="Score symbols every N trading days instead of daily (default: 1). "
                             "N=5 gives ~5x sim speedup with minor strategy-behavior change. "
                             "Exits still run daily regardless of this setting.")
    # P3: DSR n-trials override and paper-gate mode
    parser.add_argument("--dsr-n", type=int, default=N_TRIALS_TESTED, metavar="N",
                        help=f"P3: number of model variants tried historically (for DSR correction). "
                             f"Default={N_TRIALS_TESTED}. Set to actual count (e.g. 200) for honest "
                             f"selection-bias correction. Warn if < 100.")
    parser.add_argument("--paper-gate", action="store_true", default=False,
                        help="P3: use paper-trading readiness gate instead of production gate. "
                             "Criteria: avg_sharpe > 0.50 and min_fold_sharpe > -0.40 (less strict). "
                             "DSR check still applies. Useful for deploy-to-paper decisions.")
    # P0: sacred holdout bypass (one-shot promotion run only)
    parser.add_argument("--allow-sacred-holdout", action="store_true", default=False,
                        help="P0: bypass the SACRED_HOLDOUT_START guard. Use ONLY for the "
                             "single, final promotion-candidate evaluation. Logs a banner "
                             "warning. See app/ml/retrain_config.py.")
    args = parser.parse_args()

    # P0: hard guard against using sacred holdout data in development WF runs.
    from app.ml.retrain_config import assert_no_sacred_holdout as _assert_holdout_wf
    _wf_end_today = date.today()
    _assert_holdout_wf(
        _wf_end_today,
        allow_sacred_holdout=args.allow_sacred_holdout,
        context="walkforward_tier3.main",
    )

    # P3: warn if dsr_n is too low to give honest selection-bias correction
    if args.dsr_n < 100 and not args.paper_gate:
        _warn(
            f"--dsr-n={args.dsr_n} is likely understated. "
            f"Set to the actual number of model variants tried (typically 100-300) "
            f"for an honest DSR correction. Use --paper-gate to suppress this warning."
        )

    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    passed = True

    meta_model = None
    if args.meta_model_version > 0:
        from app.ml.meta_model import MetaLabelModel
        try:
            meta_model = MetaLabelModel.load("app/ml/models", args.meta_model_version, model_type="swing")
            _ok(f"Swing MetaLabelModel v{args.meta_model_version} loaded")
        except Exception as e:
            _warn(f"Could not load swing MetaLabelModel v{args.meta_model_version}: {e}")

    intraday_meta_model = None
    if args.intraday_meta_model_version > 0:
        from app.ml.meta_model import MetaLabelModel
        try:
            intraday_meta_model = MetaLabelModel.load(
                "app/ml/models", args.intraday_meta_model_version, model_type="intraday"
            )
            _ok(f"Intraday MetaLabelModel v{args.intraday_meta_model_version} loaded")
        except Exception as e:
            _warn(f"Could not load intraday MetaLabelModel v{args.intraday_meta_model_version}: {e}")

    _header("MrTrader — Walk-Forward Tier 3 Validation")

    swing_ver = args.swing_model_version if args.swing_model_version > 0 else None
    intraday_ver = args.intraday_model_version if args.intraday_model_version > 0 else None

    # WF-5a: pre-fetch macro event blocked dates once for the full WF range
    macro_blocked_dates: Optional[set] = None
    if args.macro_gate:
        from datetime import datetime as _dt2
        _wf_end = _dt2.now().date()
        _wf_start = _wf_end - timedelta(days=max(args.years * 365, args.days) + 30)
        try:
            from scripts.walkforward.macro_calendar import load_macro_blocked_dates
            macro_blocked_dates = load_macro_blocked_dates(_wf_start, _wf_end)
            _ok(f"Macro gate: {len(macro_blocked_dates)} blocked dates loaded "
                f"({_wf_start} → {_wf_end})")
        except Exception as _me:
            _warn(f"Macro gate calendar fetch failed: {_me} — macro gate disabled for this run")

    # P1: pre-compute benign-blocked dates from macro_history.parquet
    benign_blocked_dates: Optional[set] = None
    if getattr(args, "benign_gate", False):
        try:
            from app.ml.regime_score_pit import build_regime_score_map
            from app.ml.retrain_config import BENIGN_REGIME_THRESHOLD
            _score_map = build_regime_score_map()
            benign_blocked_dates = {
                d for d, score in _score_map.items() if score < BENIGN_REGIME_THRESHOLD
            }
            _ok(f"BenignGate: {len(benign_blocked_dates)} adverse-regime dates "
                f"(threshold={BENIGN_REGIME_THRESHOLD})")
        except Exception as _be:
            _warn(f"BenignGate setup failed: {_be} — benign gate disabled for this run")

    # Phase 2b: pre-fetch earnings calendar once (used by both swing and intraday)
    earnings_cal: Optional[Dict[str, set]] = None
    if args.earnings_blackout:
        from datetime import datetime as _dt
        _end = _dt.now().date()
        _start = _end - timedelta(days=max(args.years * 365, args.days) + 30)
        _syms = symbols or []
        if not _syms:
            # Default to SP100 + Russell1000 union if no explicit symbols
            try:
                from app.utils.constants import RUSSELL_1000_TICKERS
                _syms = list(RUSSELL_1000_TICKERS)
            except Exception:
                _syms = []
        if _syms:
            earnings_cal = _fetch_earnings_calendar(_syms, _start, _end)

    if args.model in ("swing", "both"):
        t0 = time.time()
        _swing_kwargs = dict(
            n_folds=args.folds,
            total_years=args.years,
            symbols=symbols,
            atr_stop_mult=args.stop_mult,
            atr_target_mult=args.target_mult,
            meta_model=meta_model,
            pm_abstention_vix=args.pm_abstention_vix,
            pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
            pm_abstention_spy_5d=args.pm_abstention_spy_5d,
            model_version=swing_ver,
            transaction_cost_pct=args.swing_cost_bps / 10_000 / 2,
            purge_days=args.swing_purge_days,
            embargo_days=args.swing_embargo_days,
            use_opportunity_score=args.pm_opportunity_score,
            no_prefilters=args.no_prefilters,
            train_years=args.swing_train_years,
            earnings_blackout=earnings_cal,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            wf_max_symbols=args.wf_max_symbols if args.wf_max_symbols > 0 else None,
            feature_cache_workers=args.feature_cache_workers,
            feature_cache_executor=args.feature_cache_executor,
            feature_cache_disable=args.feature_cache_disable,
            sim_scan_interval_days=args.sim_scan_interval_days,
            no_atr_stops=args.no_atr_stops,
            rebalance_mode=args.rebalance_mode,
            rebalance_days=args.rebalance_days,
            rebalance_target_n=args.rebalance_target_n,
            rebalance_sector_cap=args.rebalance_sector_cap,
            rebalance_add_threshold=args.rebalance_add_threshold,
            rebalance_drop_threshold=args.rebalance_drop_threshold,
            rebalance_min_adv=args.rebalance_min_adv,
            rebalance_regime_gate=args.rebalance_regime_gate,
            rebalance_regime_vix_bull=args.rebalance_regime_vix_bull,
            rebalance_regime_vix_bear=args.rebalance_regime_vix_bear,
            rebalance_regime_spy_ma_days=args.rebalance_regime_spy_ma_days,
            rebalance_inv_vol=args.rebalance_inv_vol,
            rebalance_inv_vol_lookback=args.rebalance_inv_vol_lookback,
            rebalance_inv_vol_min_mult=args.rebalance_inv_vol_min_mult,
            rebalance_inv_vol_max_mult=args.rebalance_inv_vol_max_mult,
        )
        if getattr(args, "rebalance_momentum_baseline", False):
            _swing_kwargs["scorer_instance"] = _momentum_baseline_scorer(lookback_days=60)
            print("  WF: momentum baseline mode — 60d trailing return ranker")
        elif getattr(args, "rebalance_ic_composite", False):
            from app.ml.factor_scorer import IcCompositeScorer
            _swing_kwargs["scorer_instance"] = IcCompositeScorer()
            print("  WF: IC composite mode — Phase 88 deterministic IC-weighted scorer (v219 weights)")
        swing_report = run_swing_walkforward(**_swing_kwargs)
        swing_report.print(dsr_n=args.dsr_n, paper_gate=args.paper_gate)
        print(f"  Swing walk-forward elapsed: {time.time()-t0:.0f}s")
        if args.bootstrap > 0:
            _bootstrap_folds(run_swing_walkforward, n_bootstrap=args.bootstrap, **_swing_kwargs)
        if args.cpcv and args.model in ("swing", "both"):
            _run_cpcv_swing(args, symbols, swing_ver, meta_model, earnings_cal, passed)
        if not swing_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate):
            passed = False
        if args.record_results and swing_report.folds:
            from app.ml.training import ModelTrainer
            loaded_ver = swing_report.folds[0].model_version if swing_report.folds else 0
            ModelTrainer.record_tier3_result(
                version=loaded_ver,
                avg_sharpe=swing_report.avg_sharpe,
                fold_sharpes=[f.sharpe for f in swing_report.folds],
                gate_passed=swing_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate),
            )

    if args.model in ("intraday", "both"):
        t0 = time.time()
        intraday_scan_offsets = [12, 18, 24] if args.intraday_multi_scan else None
        # Phase R5: pre-fetch regime map if --regime-gate enabled
        _regime_map: Optional[Dict] = None
        if getattr(args, "regime_gate", False):
            from datetime import datetime as _dt3
            _rg_end = _dt3.now().date()
            _rg_start = _rg_end - timedelta(days=args.days + 80)
            try:
                from scripts.walkforward.regime import load_regime_map
                _regime_map = load_regime_map(_rg_start, _rg_end)
                _ok(f"R5 regime map: {len(_regime_map)} dates tagged ({_rg_start} → {_rg_end})")
            except Exception as _re:
                _warn(f"R5 regime map fetch failed: {_re} — regime gate disabled for this run")
        _intraday_kwargs = dict(
            n_folds=args.folds,
            total_days=args.days,
            symbols=symbols,
            meta_model=intraday_meta_model,
            pm_abstention_vix=args.pm_abstention_vix,
            pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
            scan_offsets=intraday_scan_offsets,
            model_version=intraday_ver,
            transaction_cost_pct=args.intraday_cost_bps / 10_000 / 2,
            purge_days=args.intraday_purge_days,
            use_opportunity_score=args.pm_opportunity_score,
            use_dispersion_gate=args.dispersion_gate,
            earnings_blackout=earnings_cal,
            embargo_days=args.intraday_embargo_days,
            macro_blocked_dates=macro_blocked_dates,
            benign_blocked_dates=benign_blocked_dates,
            use_regime_gate=getattr(args, "regime_gate", False),
            regime_map=_regime_map,
        )
        intraday_report = run_intraday_walkforward(**_intraday_kwargs)
        intraday_report.print(dsr_n=args.dsr_n, paper_gate=args.paper_gate)
        print(f"  Intraday walk-forward elapsed: {time.time()-t0:.0f}s")
        if args.bootstrap > 0:
            _bootstrap_folds(run_intraday_walkforward, n_bootstrap=args.bootstrap, **_intraday_kwargs)
        if args.cpcv and args.model in ("intraday", "both"):
            _run_cpcv_intraday(args, symbols, intraday_ver, intraday_meta_model, earnings_cal, passed)
        if not intraday_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate):
            passed = False
        if args.record_results and intraday_report.folds:
            from app.ml.intraday_training import IntradayModelTrainer
            loaded_ver = intraday_report.folds[0].model_version if intraday_report.folds else 0
            IntradayModelTrainer.record_tier3_result(
                version=loaded_ver,
                avg_sharpe=intraday_report.avg_sharpe,
                fold_sharpes=[f.sharpe for f in intraday_report.folds],
                gate_passed=intraday_report.gate_passed(dsr_n=args.dsr_n, paper_gate=args.paper_gate),
            )

    print()
    if passed:
        _ok("ALL GATES PASSED — proceed to Phase 23 (paper trading)")
        return 0
    else:
        _warn("Gates not yet met — further model improvement needed before paper trading")
        return 1


if __name__ == "__main__":
    sys.exit(main())
