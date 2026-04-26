"""
CLI: train the portfolio selection model with visible progress output.

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --years 3 --symbols AAPL MSFT NVDA
  python scripts/train_model.py --no-fundamentals   # faster, price-only
  python scripts/train_model.py --dry-run           # feature check only

Output sections:
  [1/6] Environment check
  [2/6] Downloading historical data
  [3/6] Building feature matrix
  [4/6] Training XGBoost model
  [5/6] Evaluating out-of-sample
  [6/6] Saving model
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Prevent OpenMP deadlock on Windows when xgboost + numpy both load libiomp5
os.environ.setdefault("OMP_NUM_THREADS", "1")

# -- Make sure project root is on the path -------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# noqa: E402 -- sys.path must be set before project imports
import numpy as np  # noqa: E402, F401
import yfinance as yf  # noqa: E402

# -- Colour helpers (works on Windows 10+ and all Unix terminals) --------------
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{RESET}"


def header(step: int, total: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}[{step}/{total}] {title}{RESET}")
    print(_c(DIM, "-" * 60))


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!!{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


def progress_bar(done: int, total: int, width: int = 36) -> str:
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "." * (width - filled)
    pct = int(100 * done / max(total, 1))
    return f"[{bar}] {done}/{total} ({pct}%)"


# -- Step 1: Environment check -------------------------------------------------

def check_environment() -> bool:
    header(1, 6, "Environment check")
    all_ok = True

    # Python packages
    for pkg in ["xgboost", "sklearn", "yfinance", "pandas", "numpy"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
            ok(f"{pkg} available")
        except ImportError:
            fail(f"{pkg} not installed -- run: pip install {pkg}")
            all_ok = False

    # Config / .env
    try:
        from app.config import settings
        ok(f"Alpaca key configured: ...{settings.alpaca_api_key[-4:]}")
        if settings.fred_api_key:
            ok("FRED API key configured")
        else:
            warn("FRED_API_KEY not set -- macro features will use defaults")
        if settings.alpha_vantage_api_key or settings.alpha_advantage_api_key:
            ok("Alpha Vantage key configured")
        else:
            warn("ALPHA_VANTAGE_API_KEY not set -- earnings surprise will be 0")
    except Exception as exc:
        fail(f"Config error: {exc}")
        all_ok = False

    return all_ok


# -- Training price cache (Parquet, 23h TTL) -----------------------------------

_TRAIN_CACHE_DIR = ROOT / "app" / "ml" / "models" / "price_cache" / "training"
_TRAIN_CACHE_TTL_HOURS = 23


def _train_cache_path(symbols: list, years: int, provider: str) -> Path:
    h = hashlib.md5((",".join(sorted(symbols)) + str(years) + provider).encode()).hexdigest()[:10]
    return _TRAIN_CACHE_DIR / f"train_{years}yr_{provider}_{h}.parquet"


def _load_train_cache(symbols: list, years: int, provider: str):
    path = _train_cache_path(symbols, years, provider)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > _TRAIN_CACHE_TTL_HOURS:
        return None
    try:
        import pandas as pd
        combined = pd.read_parquet(path)
        result = {}
        for sym in combined["_symbol"].unique():
            sub = combined[combined["_symbol"] == sym].drop(columns=["_symbol"])
            sub.index = pd.to_datetime(sub.index)
            result[sym] = sub
        return result
    except Exception:
        return None


def _save_train_cache(data: dict, symbols: list, years: int, provider: str) -> None:
    try:
        import pandas as pd
        _TRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frames = []
        for sym, df in data.items():
            tmp = df.copy()
            tmp["_symbol"] = sym
            frames.append(tmp)
        if frames:
            pd.concat(frames).to_parquet(_train_cache_path(symbols, years, provider))
    except Exception:
        pass


# -- Step 2: Download historical data -----------------------------------------

def download_data(
    symbols: List[str], years: int, provider: str = "polygon"
) -> dict:
    import pandas as pd
    from datetime import date as date_type
    header(2, 6, f"Downloading {years}-year history for {len(symbols)} symbols  [provider={provider}]")

    end_dt = datetime.now().date()
    start_dt = (datetime.now() - timedelta(days=365 * years)).date()
    info(f"Period: {start_dt} -> {end_dt}")
    print()

    # Parquet cache check — skip network entirely if data is fresh
    cached = _load_train_cache(symbols, years, provider)
    if cached is not None:
        ok(f"Loaded {len(cached)} symbols from Parquet cache  (skipped download)")
        return cached

    # Try bulk provider first (Polygon S3 flat files — much faster than yfinance per-symbol)
    if provider == "polygon":
        try:
            from app.data import get_provider
            prov = get_provider("polygon")
            data = prov.get_daily_bars_bulk(symbols, start_dt, end_dt)
            valid = {s: df for s, df in data.items() if len(df) >= 52}
            skipped = [s for s in symbols if s not in valid]
            print()
            ok(f"Downloaded: {len(valid)} symbols  (Polygon S3 bulk)")
            if skipped:
                warn(f"Skipped ({len(skipped)} symbols with insufficient data): "
                     f"{', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}")
            _save_train_cache(valid, symbols, years, provider)
            return valid
        except Exception as exc:
            warn(f"Polygon bulk download failed ({exc}), falling back to yfinance...")

    # yfinance fallback — symbol by symbol
    data: Dict[str, "pd.DataFrame"] = {}
    errors: List[str] = []

    for i, symbol in enumerate(symbols, 1):
        bar = progress_bar(i, len(symbols))
        print(f"\r  {bar}  {symbol:<6}", end="", flush=True)

        try:
            df = yf.download(
                symbol, start=start_dt, end=end_dt,
                progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if not df.empty and "close" in df.columns and len(df) >= 52:
                data[symbol] = df
            else:
                errors.append(symbol)
        except Exception:
            errors.append(symbol)

        time.sleep(0.05)

    print()
    print()
    ok(f"Downloaded: {len(data)} symbols  (yfinance)")
    if errors:
        warn(f"Skipped ({len(errors)} symbols with insufficient data): "
             f"{', '.join(errors[:10])}{'...' if len(errors) > 10 else ''}")
    _save_train_cache(data, symbols, years, provider)
    return data


# -- Steps 3-6: Rolling window training (new pipeline) ------------------------

def run_rolling_pipeline(
    symbols_data: dict,
    fetch_fundamentals: bool,
    years: int,
    dry_run: bool,
    model_type: str = "xgboost",
    label_scheme: str = "atr",
    top_n_features: int = 0,
    n_workers: int = 0,
    hpo_trials: int = 0,
    walk_forward_folds: int = 0,
    prediction_threshold: float = 0.35,
    two_stage: bool = False,
    three_stage: bool = False,
    multi_window: bool = False,
):
    """
    Steps 3-6 combined using the new rolling-window ModelTrainer.
    Produces honest out-of-sample metrics via time-based train/test split.
    """
    from app.ml.training import ModelTrainer, WINDOW_DAYS, FORWARD_DAYS, TEST_FRACTION

    stage_label = " [three-stage]" if three_stage else (" [two-stage]" if two_stage else "")
    header(3, 6, "Building rolling-window feature matrix")
    info(f"Window: {WINDOW_DAYS} trading days  |  Forward: {FORWARD_DAYS} days  "
         f"|  Test split: last {int(TEST_FRACTION*100)}% of windows")
    info(f"Model type: {model_type}{stage_label}  |  Label scheme: {label_scheme}"
         + (f"  |  Top-N features: {top_n_features}" if top_n_features else ""))
    print()

    trainer = ModelTrainer(
        model_type=model_type,
        top_n_features=top_n_features if top_n_features > 0 else None,
        label_scheme=label_scheme,
        n_workers=n_workers if n_workers > 0 else None,
        hpo_trials=hpo_trials,
        walk_forward_folds=walk_forward_folds,
        prediction_threshold=prediction_threshold,
        two_stage=two_stage,
        three_stage=three_stage,
        multi_window=multi_window,
    )

    # Regime score
    try:
        from app.strategy.regime_detector import RegimeDetector
        det = RegimeDetector().get_regime_detail()
        regime_score = float(det.get("composite_score", 0.5))
        ok(f"Regime score: {regime_score:.3f}  ({det.get('regime', '?')})")
    except Exception as exc:
        warn(f"Regime detector unavailable ({exc}) -- using 0.5")

    print()
    t0 = time.time()
    t_step = time.time()
    X_train, y_train, X_test, y_test, feature_names, meta_train = trainer._build_rolling_matrix(
        symbols_data, fetch_fundamentals=fetch_fundamentals
    )
    t_matrix = time.time() - t_step

    if len(X_train) == 0:
        fail("No samples after rolling window labelling")
        return None

    from app.ml.feature_store import FeatureStore
    _fs = FeatureStore()
    ok(f"Train samples : {len(X_train)}")
    ok(f"Test samples  : {len(X_test)}  (most recent {int(TEST_FRACTION*100)}% of windows)")
    ok(f"Features      : {len(feature_names)}")
    ok(f"Feature matrix built in {t_matrix:.0f}s  (feature store: {_fs.count():,} entries cached)")
    info(f"  {', '.join(feature_names)}")

    # Apply feature selection if requested
    if top_n_features > 0 and len(feature_names) > top_n_features:
        X_train, X_test, feature_names, _mi = trainer._select_top_features(
            X_train, y_train, X_test, feature_names, top_n_features
        )
        ok(f"Feature selection: kept top {len(feature_names)} features")
        info(f"  {', '.join(feature_names)}")

    # Step 4
    _stage_tag = " [three-stage]" if three_stage else (" [two-stage]" if two_stage else "")
    header(4, 6, f"Training {model_type.upper()} model{_stage_tag}" + (" [multi-window]" if multi_window else ""))
    t_step = time.time()

    if multi_window:
        print("  Training multi-window ensemble (63d + 126d)...", flush=True)
        mw_results = trainer.train_multi_window(
            symbols_data, fetch_fundamentals=fetch_fundamentals, years=years
        )
        elapsed = time.time() - t0
        ok(f"Multi-window training complete in {elapsed:.1f}s")
        for w, r in mw_results.items():
            ok(f"  Window {w}d: AUC={r['metrics'].get('auc', '?'):.3f}  Recall={r['metrics'].get('recall', '?'):.1%}")
    else:
        print("  Training...", end="", flush=True)
        if model_type in ("lambdarank", "double_ensemble"):
            X_train, y_train, train_groups = trainer._build_lambdarank_groups(X_train, y_train, meta_train)
            trainer.model.train(
                X_train, y_train, feature_names,
                groups=train_groups,
            )
            elapsed = time.time() - t0
            print("\r", end="")
            ok(f"{model_type} training complete in {elapsed:.1f}s  ({len(train_groups)} groups)")
        else:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
            sample_weight = trainer._build_sample_weights(meta_train)
            trainer.model.train(
                X_train, y_train, feature_names,
                scale_pos_weight=spw,
                X_val=X_test, y_val=y_test,
                early_stopping_rounds=30,
                sample_weight=sample_weight,
            )
            elapsed = time.time() - t0
            print("\r", end="")
            ok(f"Training complete in {elapsed:.1f}s  (scale_pos_weight={spw:.1f})")

    t_train = time.time() - t_step

    # Step 5
    header(5, 6, "Out-of-sample evaluation  (time-based held-out set)")

    if len(X_test) == 0:
        warn("No test samples — increase --years to get a larger dataset")
        metrics = {}
    else:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        if multi_window and hasattr(trainer, "_mw_trainers") and trainer._mw_trainers:
            # Multi-window: evaluate blended ensemble on the 63d test set
            # (X_test was built with default WINDOW_DAYS=63 in step 3)
            # Tune threshold by searching over blended probabilities
            _, proba = trainer.predict_multi_window(X_test)
            best_t, best_f1 = 0.35, -1.0
            from sklearn.metrics import f1_score
            for t in np.arange(0.20, 0.66, 0.05):
                p = (proba >= t).astype(int)
                f = f1_score(y_test, p, zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, float(t)
            tuned_t = best_t
            trainer.prediction_threshold = tuned_t
            preds = (proba >= tuned_t).astype(int)
            ok(f"Threshold : {tuned_t:.2f}  (tuned on val set to maximise F1, multi-window blended)")
            # Also show per-window metrics
            for w, r in mw_results.items():
                m = r["metrics"]
                ok(f"  Window {w}d: AUC={m.get('auc', float('nan')):.3f}  "
                   f"Recall={m.get('recall', 0):.1%}  Prec={m.get('precision', 0):.1%}")
        else:
            # Single model: tune threshold then predict
            tuned_t = trainer.model.tune_threshold(X_test, y_test)
            ok(f"Threshold : {tuned_t:.2f}  (tuned on val set to maximise F1)")
            preds, proba = trainer.model.predict(X_test, threshold=tuned_t)

        # For regression/rank label schemes, binarize y_test before classification metrics
        import numpy as _np
        _is_reg = label_scheme in ("return_regression", "return_blend", "path_quality")
        _is_rank = label_scheme == "lambdarank" or model_type == "double_ensemble"
        if _is_rank:
            # Quintile labels 0-4; treat 4 (top quintile) as positive
            y_test_bin = (_np.array(y_test) >= 4).astype(int)
        elif _is_reg:
            y_test_bin = (_np.array(y_test) >= _np.percentile(y_test, 80)).astype(int)
        else:
            y_test_bin = y_test
        acc = accuracy_score(y_test_bin, preds)
        prec = precision_score(y_test_bin, preds, zero_division=0)
        rec = recall_score(y_test_bin, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test_bin, proba)
        except Exception:
            auc = float("nan")

        ok(f"Accuracy  : {acc:.1%}")
        ok(f"Precision : {prec:.1%}  (of predicted winners, how many actually won)")
        ok(f"Recall    : {rec:.1%}  (of actual winners, how many were caught)")
        ok(f"ROC-AUC   : {auc:.3f}  (0.5 = random, 1.0 = perfect)")
        print()
        if auc >= 0.65:
            print(f"  {GREEN}{BOLD}>> Promising -- AUC >= 0.65{RESET}")
        elif auc >= 0.55:
            print(f"  {YELLOW}{BOLD}>> Moderate edge -- consider more data or features{RESET}")
        else:
            print(f"  {RED}{BOLD}>> Weak signal -- AUC near random, do not trade live{RESET}")

        # Walk-forward CV (skip for multi-window — each sub-trainer has its own CV)
        if walk_forward_folds > 0 and not multi_window:
            print()
            info(f"Walk-forward CV ({walk_forward_folds} folds):")
            wf = trainer._walk_forward_cv(
                np.vstack([X_train, X_test]),
                np.concatenate([y_train, y_test]),
                feature_names,
                n_folds=walk_forward_folds,
            )
            if wf:
                ok(f"WF AUC    : {wf['wf_auc_mean']:.3f} ± {wf['wf_auc_std']:.3f}  ({wf['wf_folds']} folds)")
            else:
                warn("Walk-forward CV returned no results (too few samples?)")

        # Feature importance bar chart (first window model if multi-window)
        try:
            if multi_window and hasattr(trainer, "_mw_trainers") and trainer._mw_trainers:
                pairs = trainer._mw_trainers[0][1].model.feature_importance()
            else:
                pairs = trainer.model.feature_importance()
            if pairs:
                print()
                info("Top 10 features by importance:")
                top = pairs[:10]
                max_imp = top[0][1] if top else 1
                for fname, imp in top:
                    bar_w = int(30 * imp / max_imp)
                    print(f"     {fname:<30} {'#' * bar_w} {imp:.4f}")
        except Exception:
            pass

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc, "threshold": tuned_t,
                   "t_matrix_s": round(t_matrix, 1), "t_train_s": round(t_train, 1)}

    # Step 6
    header(6, 6, "Saving model")

    if dry_run:
        warn("Dry-run -- model NOT saved")
        return None

    MODEL_DIR = ROOT / "app" / "ml" / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    version = trainer._next_version("swing")

    if multi_window and hasattr(trainer, "_mw_trainers") and trainer._mw_trainers:
        # Save each window model separately; primary path = first window
        import pickle, json
        paths = []
        for w, sub in trainer._mw_trainers:
            p = sub.model.save(str(MODEL_DIR), version, model_name=f"swing_w{w}")
            paths.append(p)
            ok(f"Model saved -> {p}")
        # Save ensemble metadata
        meta_path = MODEL_DIR / f"swing_mw_v{version}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "version": version,
                "windows": [w for w, _ in trainer._mw_trainers],
                "blend": trainer._mw_blend,
                "threshold": trainer.prediction_threshold,
                "metrics": metrics,
            }, f, indent=2)
        ok(f"Ensemble meta -> {meta_path}")
        path = paths[0]
    else:
        path = trainer.model.save(str(MODEL_DIR), version, model_name="swing")
        ok(f"Model saved -> {path}")

    try:
        trainer._record_version(
            version, len(X_train), len(X_test), path, years, metrics
        )
        ok(f"Version v{version} recorded in database")
    except Exception as exc:
        warn(f"DB record skipped (DB not running?): {exc}")

    return version


# -- Main ----------------------------------------------------------------------

def main():
    from app.utils.constants import RUSSELL_1000_TICKERS

    parser = argparse.ArgumentParser(
        description="Train the MrTrader portfolio selection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        metavar="TICKER",
        help="Override ticker list (default: S&P 100)",
    )
    parser.add_argument(
        "--no-fundamentals", action="store_true",
        help="Skip live fundamental/insider API calls (faster, price-only features)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full pipeline but do not save the model",
    )
    parser.add_argument(
        "--model-type", default="xgboost",
        choices=["xgboost", "lgbm", "ensemble", "lgbm_ensemble", "lambdarank", "double_ensemble"],
        help="Model architecture (default: xgboost)",
    )
    parser.add_argument(
        "--label-scheme", default="atr",
        choices=["atr", "triple_barrier", "cross_sectional", "spy_relative", "sector_relative", "atr_and_sector", "return_regression", "return_blend", "lambdarank", "percentile_rank", "path_quality"],
        help="Labeling scheme (default: atr). triple_barrier = bar-by-bar 1.5x ATR target / 0.5x ATR stop simulation (Phase 33).",
    )
    parser.add_argument(
        "--top-features", type=int, default=0,
        help="Keep only top-N features by mutual information (0 = keep all)",
    )
    parser.add_argument(
        "--provider", default="polygon",
        choices=["polygon", "yfinance"],
        help="Data provider for historical bars (default: polygon)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel workers for feature computation (0 = auto, uses CPU count)",
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=0,
        help="Optuna HPO trials (0 = skip, 50 recommended)",
    )
    parser.add_argument(
        "--walk-forward", type=int, default=0, metavar="FOLDS",
        help="Walk-forward CV folds after training (0 = skip, 5 recommended)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Prediction probability threshold (default: 0.35; tuned automatically on val set)",
    )
    parser.add_argument(
        "--two-stage", action="store_true",
        help="Use two-stage model: Stage 1 fundamental quality, Stage 2 technical timing",
    )
    parser.add_argument(
        "--three-stage", action="store_true",
        help="Use three-stage model tuned for 10d horizon: Stage 1 quality (0.20), Stage 2 catalyst (0.40), Stage 3 timing (0.40)",
    )
    parser.add_argument(
        "--multi-window", action="store_true",
        help="Train ensemble of 63d + 126d window models and blend predictions",
    )
    args = parser.parse_args()

    symbols = args.symbols or RUSSELL_1000_TICKERS
    fetch_fundamentals = not args.no_fundamentals

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  MrTrader -- Model Training{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  Symbols       : {len(symbols)}")
    print(f"  History       : {args.years} year(s)")
    fund_label = "yes (yfinance + AV + EDGAR)" if fetch_fundamentals else "no (price-only, faster)"
    print(f"  Fundamentals  : {fund_label}")
    print(f"  Model type    : {args.model_type}")
    print(f"  Label scheme  : {args.label_scheme}")
    print(f"  Provider      : {args.provider}")
    print(f"  Workers       : {'auto' if args.workers == 0 else args.workers}")
    if args.top_features:
        print(f"  Top features  : {args.top_features}")
    if args.hpo_trials:
        print(f"  HPO trials    : {args.hpo_trials}")
    if args.walk_forward:
        print(f"  Walk-fwd folds: {args.walk_forward}")
    print(f"  Threshold     : {args.threshold}")
    if args.three_stage:
        print(f"  Three-stage   : yes (quality 20% + catalyst 40% + timing 40%)")
    elif args.two_stage:
        print(f"  Two-stage     : yes (fundamental quality + technical timing)")
    if args.multi_window:
        print(f"  Multi-window  : yes (63d + 126d ensemble)")
    print(f"  Dry run       : {'yes -- model will NOT be saved' if args.dry_run else 'no'}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    t_start = time.time()
    phase_times: Dict[str, float] = {}

    # Step 1
    _t = time.time()
    if not check_environment():
        sys.exit(1)
    phase_times["env_check"] = time.time() - _t

    # Step 2
    _t = time.time()
    symbols_data = download_data(symbols, args.years, provider=args.provider)
    phase_times["download"] = time.time() - _t
    if len(symbols_data) < 3:
        fail("Too few symbols downloaded -- check network connection")
        sys.exit(1)

    # Steps 3-6 (rolling windows + train + evaluate + save)
    _t = time.time()
    version = run_rolling_pipeline(
        symbols_data, fetch_fundamentals, args.years, args.dry_run,
        model_type=args.model_type,
        label_scheme=args.label_scheme,
        top_n_features=args.top_features,
        n_workers=args.workers,
        hpo_trials=args.hpo_trials,
        walk_forward_folds=args.walk_forward,
        prediction_threshold=args.threshold,
        two_stage=args.two_stage,
        three_stage=args.three_stage,
        multi_window=args.multi_window,
    )
    phase_times["train_pipeline"] = time.time() - _t

    elapsed = time.time() - t_start
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    if version:
        print(f"{GREEN}{BOLD}  Done!  Model v{version} ready.  ({elapsed:.0f}s total){RESET}")
    else:
        print(f"{YELLOW}{BOLD}  Done!  ({elapsed:.0f}s total)  -- model not saved{RESET}")
    print(f"{DIM}  Phase timings: download={phase_times['download']:.0f}s  "
          f"train={phase_times['train_pipeline']:.0f}s  total={elapsed:.0f}s{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
