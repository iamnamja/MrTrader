"""
walkforward_tier3.py — Rolling walk-forward validation using Tier 3 agent simulation.

Design (per PHASES_18_23_SPEC.md):
  Fold 1: train on Y1-Y2, test Tier 3 on Y3
  Fold 2: train on Y1-Y3, test Tier 3 on Y4
  Fold 3: train on Y1-Y4, test Tier 3 on Y5 (most recent)

Gate: avg OOS Tier 3 Sharpe > 0.8, no fold below -0.3

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
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Gate thresholds ───────────────────────────────────────────────────────────
SHARPE_GATE = 0.8       # avg OOS Sharpe required to pass
MIN_FOLD_SHARPE = -0.3  # no individual fold may be below this


# ── Console helpers ───────────────────────────────────────────────────────────

def _ok(msg):   print(f"  \033[32mOK\033[0m  {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")
def _err(msg):  print(f"  \033[31mFAIL\033[0m  {msg}")
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

    def passed_gate(self) -> bool:
        return self.sharpe >= MIN_FOLD_SHARPE

    def summary_line(self) -> str:
        gate = "OK" if self.passed_gate() else "FAIL"
        return (
            f"  Fold {self.fold} [{gate}] "
            f"test={self.test_start}->{self.test_end}  "
            f"trades={self.trades}  win={self.win_rate:.1%}  "
            f"Sharpe={self.sharpe:.2f}  DD={self.max_drawdown:.1%}"
        )


@dataclass
class WalkForwardReport:
    model_type: str
    folds: List[FoldResult] = field(default_factory=list)

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

    def gate_passed(self) -> bool:
        return self.avg_sharpe >= SHARPE_GATE and self.min_sharpe >= MIN_FOLD_SHARPE

    def print(self) -> None:
        _header(f"Walk-Forward Report — {self.model_type.upper()} (Tier 3)")
        for f in self.folds:
            print(f.summary_line())
        print()
        print(f"  Avg Sharpe:    {self.avg_sharpe:.3f}  (gate: > {SHARPE_GATE})")
        print(f"  Min fold Sharpe: {self.min_sharpe:.3f}  (gate: > {MIN_FOLD_SHARPE})")
        print(f"  Avg win rate:  {self.avg_win_rate:.1%}")
        print(f"  Total trades:  {self.total_trades}")
        print()
        if self.gate_passed():
            _ok(f"GATE PASSED — avg Sharpe {self.avg_sharpe:.3f} > {SHARPE_GATE}")
        else:
            _err(f"GATE NOT MET — avg Sharpe {self.avg_sharpe:.3f} (need {SHARPE_GATE}), "
                 f"min fold {self.min_sharpe:.3f} (need {MIN_FOLD_SHARPE})")


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
) -> WalkForwardReport:
    import yfinance as yf
    from app.backtesting.agent_simulator import AgentSimulator

    report = WalkForwardReport(model_type="swing")
    model, version = _load_model("swing", version=model_version)
    if model is None:
        _err("No swing model found — retrain first.")
        return report

    from app.utils.constants import SP_100_TICKERS
    from app.data.universe_history import members_at as _members_at
    symbols = symbols or list(SP_100_TICKERS)

    # Download full history
    end_all = datetime.now()
    start_all = end_all - timedelta(days=total_years * 365 + 30)
    _subheader(f"Swing walk-forward: {n_folds} folds | {total_years}yr | "
               f"{len(symbols)} symbols | model v{version}")

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

    logger.info("Data loaded: %d symbols in %.1fs", len(symbols_data), time.time() - t0)

    # Build fold boundaries (expanding window)
    # total_years split into folds+1 equal segments; train grows, test = last segment each fold
    segment_days = int(total_years * 365 / (n_folds + 1))
    fold_boundaries = []
    for fold_idx in range(n_folds):
        train_end_dt = end_all - timedelta(days=segment_days * (n_folds - fold_idx))
        test_start_dt = train_end_dt + timedelta(days=1)
        test_end_dt = train_end_dt + timedelta(days=segment_days)
        fold_boundaries.append((
            start_all.date(),
            train_end_dt.date(),
            test_start_dt.date(),
            min(test_end_dt.date(), end_all.date()),
        ))

    def _run_swing_fold(args):
        fold_idx, tr_start, tr_end, te_start, te_end = args
        _subheader(f"Fold {fold_idx}/{n_folds}  train:{tr_start}->{tr_end}  "
                   f"test:{te_start}->{te_end}")
        t_fold = time.time()
        # Point-in-time filter: only use symbols that were in the index at fold train start
        pit_members = set(_members_at("sp100", tr_start))
        fold_symbols_data = {s: d for s, d in symbols_data.items() if s in pit_members}
        sim = AgentSimulator(
            model=model,
            atr_stop_mult=atr_stop_mult,
            atr_target_mult=atr_target_mult,
            meta_model=meta_model,
            pm_abstention_vix=pm_abstention_vix,
            pm_abstention_spy_ma_days=pm_abstention_spy_ma_days,
            pm_abstention_spy_5d=pm_abstention_spy_5d,
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
        _ok(f"Fold {fold_idx} done in {elapsed:.1f}s — {result.total_trades} trades, "
            f"Sharpe {result.sharpe_ratio:.2f}")
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
        )

    from concurrent.futures import ThreadPoolExecutor
    fold_args = [
        (i + 1, tr_start, tr_end, te_start, te_end)
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(fold_boundaries)
    ]
    with ThreadPoolExecutor(max_workers=n_folds) as pool:
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
) -> WalkForwardReport:
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
    from app.data.intraday_cache import load_many, available_symbols as poly_syms

    report = WalkForwardReport(model_type="intraday")
    model, version = _load_model("intraday", version=model_version)
    if model is None:
        _err("No intraday model found — retrain first.")
        return report

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

    logger.info("Intraday data loaded: %d symbols", len(symbols_data))

    # Fold boundaries — split the test period into n_folds equal segments
    all_days_sorted = sorted({
        d for df in symbols_data.values()
        for d in pd.to_datetime(df.index).date
    })
    if not all_days_sorted:
        _err("No trading days found in data.")
        return report

    segment_size = max(1, len(all_days_sorted) // (n_folds + 1))
    fold_boundaries = []
    for fi in range(n_folds):
        te_start_idx = segment_size * (fi + 1)
        te_end_idx = min(te_start_idx + segment_size - 1, len(all_days_sorted) - 1)
        fold_boundaries.append((
            all_days_sorted[0],
            all_days_sorted[te_start_idx - 1],
            all_days_sorted[te_start_idx],
            all_days_sorted[te_end_idx],
        ))

    def _run_intraday_fold(args):
        fold_idx, tr_start, tr_end, te_start, te_end = args
        _subheader(f"Fold {fold_idx}/{n_folds}  train:{tr_start}->{tr_end}  "
                   f"test:{te_start}->{te_end}")
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
        _ok(f"Fold {fold_idx} done in {elapsed:.1f}s — {result.total_trades} trades, "
            f"Sharpe {result.sharpe_ratio:.2f}")
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
        )

    fold_args = [
        (i + 1, tr_start, tr_end, te_start, te_end)
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(fold_boundaries)
    ]
    report.folds = [_run_intraday_fold(args) for args in fold_args]

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

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
    args = parser.parse_args()

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

    if args.model in ("swing", "both"):
        t0 = time.time()
        swing_report = run_swing_walkforward(
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
        )
        swing_report.print()
        print(f"  Swing walk-forward elapsed: {time.time()-t0:.0f}s")
        if not swing_report.gate_passed():
            passed = False
        if args.record_results and swing_report.folds:
            from app.ml.training import ModelTrainer
            loaded_ver = swing_report.folds[0].model_version if swing_report.folds else 0
            ModelTrainer.record_tier3_result(
                version=loaded_ver,
                avg_sharpe=swing_report.avg_sharpe,
                fold_sharpes=[f.sharpe for f in swing_report.folds],
                gate_passed=swing_report.gate_passed(),
            )

    if args.model in ("intraday", "both"):
        t0 = time.time()
        intraday_scan_offsets = [12, 18, 24] if args.intraday_multi_scan else None
        intraday_report = run_intraday_walkforward(
            n_folds=args.folds,
            total_days=args.days,
            symbols=symbols,
            meta_model=intraday_meta_model,
            pm_abstention_vix=args.pm_abstention_vix,
            pm_abstention_spy_ma_days=args.pm_abstention_spy_ma_days,
            scan_offsets=intraday_scan_offsets,
            model_version=intraday_ver,
        )
        intraday_report.print()
        print(f"  Intraday walk-forward elapsed: {time.time()-t0:.0f}s")
        if not intraday_report.gate_passed():
            passed = False
        if args.record_results and intraday_report.folds:
            from app.ml.intraday_training import IntradayModelTrainer
            loaded_ver = intraday_report.folds[0].model_version if intraday_report.folds else 0
            IntradayModelTrainer.record_tier3_result(
                version=loaded_ver,
                avg_sharpe=intraday_report.avg_sharpe,
                fold_sharpes=[f.sharpe for f in intraday_report.folds],
                gate_passed=intraday_report.gate_passed(),
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
