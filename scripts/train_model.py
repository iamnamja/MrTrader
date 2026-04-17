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
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

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


# -- Step 2: Download historical data -----------------------------------------

def download_data(
    symbols: List[str], years: int
) -> dict:
    import pandas as pd
    header(2, 6, f"Downloading {years}-year history for {len(symbols)} symbols")

    end = datetime.now()
    start = end - timedelta(days=365 * years)
    info(f"Period: {start.date()} -> {end.date()}")
    print()

    data: Dict[str, "pd.DataFrame"] = {}
    errors: List[str] = []

    for i, symbol in enumerate(symbols, 1):
        bar = progress_bar(i, len(symbols))
        # Overwrite current line
        print(f"\r  {bar}  {symbol:<6}", end="", flush=True)

        try:
            df = yf.download(
                symbol, start=start, end=end,
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

        time.sleep(0.05)  # be polite to yfinance

    print()  # newline after progress bar
    print()
    ok(f"Downloaded: {len(data)} symbols")
    if errors:
        warn(f"Skipped ({len(errors)} symbols with insufficient data): "
             f"{', '.join(errors[:10])}{'...' if len(errors) > 10 else ''}")
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
):
    """
    Steps 3-6 combined using the new rolling-window ModelTrainer.
    Produces honest out-of-sample metrics via time-based train/test split.
    """
    from app.ml.training import ModelTrainer, WINDOW_DAYS, FORWARD_DAYS, TEST_FRACTION

    header(3, 6, "Building rolling-window feature matrix")
    info(f"Window: {WINDOW_DAYS} trading days  |  Forward: {FORWARD_DAYS} days  "
         f"|  Test split: last {int(TEST_FRACTION*100)}% of windows")
    info(f"Model type: {model_type}  |  Label scheme: {label_scheme}"
         + (f"  |  Top-N features: {top_n_features}" if top_n_features else ""))
    print()

    trainer = ModelTrainer(
        model_type=model_type,
        label_scheme=label_scheme,
        top_n_features=top_n_features if top_n_features > 0 else None,
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
    X_train, y_train, X_test, y_test, feature_names, _meta = trainer._build_rolling_matrix(
        symbols_data, fetch_fundamentals=fetch_fundamentals
    )

    if len(X_train) == 0:
        fail("No samples after rolling window labelling")
        return None

    ok(f"Train samples : {len(X_train)}")
    ok(f"Test samples  : {len(X_test)}  (most recent {int(TEST_FRACTION*100)}% of windows)")
    ok(f"Features      : {len(feature_names)}")
    info(f"  {', '.join(feature_names)}")

    # Apply feature selection if requested (after matrix build)
    if top_n_features > 0 and len(feature_names) > top_n_features:
        X_train, X_test, feature_names = trainer._select_top_features(
            X_train, y_train, X_test, feature_names, top_n_features
        )
        ok(f"Feature selection: kept top {len(feature_names)} features")
        info(f"  {', '.join(feature_names)}")

    # Step 4
    header(4, 6, f"Training {model_type.upper()} model")
    print("  Training...", end="", flush=True)
    trainer.model.train(X_train, y_train, feature_names)
    elapsed = time.time() - t0
    print("\r", end="")
    ok(f"Training complete in {elapsed:.1f}s")

    # Step 5
    header(5, 6, "Out-of-sample evaluation  (time-based held-out set)")

    if len(X_test) == 0:
        warn("No test samples — increase --years to get a larger dataset")
        metrics = {}
    else:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        preds, proba = trainer.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, proba)
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

        # Feature importance bar chart
        try:
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

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}

    # Step 6
    header(6, 6, "Saving model")

    if dry_run:
        warn("Dry-run -- model NOT saved")
        return None

    MODEL_DIR = ROOT / "app" / "ml" / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    version = trainer._next_version("swing")
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
    from app.utils.constants import SP_100_TICKERS

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
        choices=["xgboost", "lgbm", "ensemble"],
        help="Model architecture (default: xgboost)",
    )
    parser.add_argument(
        "--label-scheme", default="atr",
        choices=["atr", "cross_sectional", "spy_relative"],
        help="Labeling scheme: atr (ATR-hit), cross_sectional (rank by return), spy_relative (rank by SPY-adjusted return)",
    )
    parser.add_argument(
        "--top-features", type=int, default=0,
        help="Keep only top-N features by mutual information (0 = keep all)",
    )
    args = parser.parse_args()

    symbols = args.symbols or SP_100_TICKERS
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
    if args.top_features:
        print(f"  Top features  : {args.top_features}")
    print(f"  Dry run       : {'yes -- model will NOT be saved' if args.dry_run else 'no'}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    t_start = time.time()

    # Step 1
    if not check_environment():
        sys.exit(1)

    # Step 2
    symbols_data = download_data(symbols, args.years)
    if len(symbols_data) < 3:
        fail("Too few symbols downloaded -- check network connection")
        sys.exit(1)

    # Steps 3-6 (rolling windows + train + evaluate + save)
    version = run_rolling_pipeline(
        symbols_data, fetch_fundamentals, args.years, args.dry_run,
        model_type=args.model_type,
        label_scheme=args.label_scheme,
        top_n_features=args.top_features,
    )

    elapsed = time.time() - t_start
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    if version:
        print(f"{GREEN}{BOLD}  Done!  Model v{version} ready.  ({elapsed:.0f}s total){RESET}")
    else:
        print(f"{YELLOW}{BOLD}  Done!  ({elapsed:.0f}s total)  -- model not saved{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
