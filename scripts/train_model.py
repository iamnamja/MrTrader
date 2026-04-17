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
from typing import Dict, List, Optional

# -- Make sure project root is on the path -------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# noqa: E402 -- sys.path must be set before project imports
import numpy as np  # noqa: E402
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


# -- Step 3: Build feature matrix ----------------------------------------------

def build_features(
    symbols_data: dict,
    fetch_fundamentals: bool,
) -> tuple:
    from app.ml.features import FeatureEngineer
    from app.ml.training import ModelTrainer
    from app.utils.constants import SECTOR_MAP

    header(3, 6, "Building feature matrix")

    # Labels (top 30% = 1, bottom 30% = 0)
    trainer = ModelTrainer()
    labels = trainer._create_labels(symbols_data)
    good = sum(1 for v in labels.values() if v == 1)
    bad = sum(1 for v in labels.values() if v == 0)
    skipped = sum(1 for v in labels.values() if v is None)
    info(f"Labels -> {good} top performers, {bad} poor performers, {skipped} middle (skipped)")
    print()

    # Regime score (single fetch for all symbols)
    regime_score: Optional[float] = None
    try:
        from app.strategy.regime_detector import RegimeDetector
        det = RegimeDetector().get_regime_detail()
        regime_score = float(det.get("composite_score", 0.5))
        ok(f"Regime score: {regime_score:.3f}  ({det.get('regime', '?')})")
    except Exception as exc:
        warn(f"Regime detector unavailable ({exc}) -- using neutral 0.5")
        regime_score = 0.5

    print()
    fe = FeatureEngineer()
    X_rows, y_vals, feature_names = [], [], None
    symbols = [s for s, lbl in labels.items() if lbl is not None]

    for i, symbol in enumerate(symbols, 1):
        bar = progress_bar(i, len(symbols))
        print(f"\r  {bar}  {symbol:<6}", end="", flush=True)

        label = labels[symbol]
        df = symbols_data[symbol]
        sector = SECTOR_MAP.get(symbol)

        features = fe.engineer_features(
            symbol, df,
            sector=sector,
            regime_score=regime_score,
            fetch_fundamentals=fetch_fundamentals,
        )
        if features is None:
            continue

        if feature_names is None:
            feature_names = list(features.keys())
        X_rows.append(list(features.values()))
        y_vals.append(label)

    print()
    print()
    X = np.array(X_rows)
    y = np.array(y_vals)
    feature_names = feature_names or []

    ok(f"Feature matrix: {X.shape[0]} samples ? {X.shape[1]} features")
    info(f"Feature names: {', '.join(feature_names)}")
    return X, y, feature_names


# -- Step 4: Train model -------------------------------------------------------

def train_model(X: "np.ndarray", y: "np.ndarray", feature_names: List[str]):
    from sklearn.model_selection import train_test_split
    from app.ml.model import PortfolioSelectorModel

    header(4, 6, "Training XGBoost model")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    info(f"Train: {len(X_train)} samples | Test (held-out): {len(X_test)} samples")
    print()

    model = PortfolioSelectorModel(model_type="xgboost")
    t0 = time.time()
    print("  Training...", end="", flush=True)
    model.train(X_train, y_train, feature_names)
    elapsed = time.time() - t0
    print("\r", end="")
    ok(f"Training complete in {elapsed:.1f}s")

    return model, X_test, y_test


# -- Step 5: Evaluate ----------------------------------------------------------

def evaluate(model, X_test: "np.ndarray", y_test: "np.ndarray", feature_names: List[str]):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    header(5, 6, "Out-of-sample evaluation")

    preds, proba = model.predict(X_test)

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

    # Qualitative verdict
    print()
    if auc >= 0.65:
        print(f"  {GREEN}{BOLD}>> Model looks promising -- AUC >= 0.65{RESET}")
    elif auc >= 0.55:
        print(f"  {YELLOW}{BOLD}>> Moderate edge -- consider more data or features{RESET}")
    else:
        print(f"  {RED}{BOLD}>> Weak signal -- AUC near random, do not trade live yet{RESET}")

    # Feature importances (top 10)
    try:
        pairs = model.feature_importance()
        if pairs:
            print()
            info("Top 10 features by importance:")
            top = pairs[:10]
            max_imp = top[0][1] if top else 1
            for fname, imp in top:
                bar_w = int(30 * imp / max_imp)
                bar = "#" * bar_w
                print(f"     {fname:<30} {bar} {imp:.4f}")
    except Exception:
        pass

    return {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}


# -- Step 6: Save model --------------------------------------------------------

def save_model(model, metrics: dict, n_samples: int, years: int, dry_run: bool):
    header(6, 6, "Saving model")

    if dry_run:
        warn("Dry-run mode -- model NOT saved to disk or DB")
        return None

    try:
        from app.database.session import get_session
        from app.database.models import ModelVersion

        MODEL_DIR = ROOT / "app" / "ml" / "models"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        db = get_session()
        try:
            latest = (
                db.query(ModelVersion)
                .filter_by(model_name="portfolio_selector")
                .order_by(ModelVersion.version.desc())
                .first()
            )
            version = (latest.version + 1) if latest else 1
        finally:
            db.close()

        path = model.save(str(MODEL_DIR), version)
        ok(f"Model saved -> {path}")

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years)
        db = get_session()
        try:
            db.add(ModelVersion(
                model_name="portfolio_selector",
                version=version,
                training_date=datetime.utcnow(),
                data_range_start=start_dt.strftime("%Y-%m-%d"),
                data_range_end=end_dt.strftime("%Y-%m-%d"),
                performance={**metrics, "n_samples": n_samples},
                status="ACTIVE",
                model_path=path,
            ))
            db.commit()
            ok(f"Version v{version} recorded in database")
        except Exception as exc:
            warn(f"DB record skipped (DB not running?): {exc}")
        finally:
            db.close()

        return version

    except Exception as exc:
        warn(f"Could not save model: {exc}")
        return None


# -- Main ----------------------------------------------------------------------

def main():
    from app.utils.constants import SP_100_TICKERS

    parser = argparse.ArgumentParser(
        description="Train the MrTrader portfolio selection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--years", type=int, default=3, help="Years of history (default: 3)")
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
    print(f"  Dry run       : {'yes -- model will NOT be saved' if args.dry_run else 'no'}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    t_start = time.time()

    # Step 1
    if not check_environment():
        sys.exit(1)

    # Step 2
    symbols_data = download_data(symbols, args.years)
    if len(symbols_data) < 10:
        fail("Too few symbols downloaded -- check network connection")
        sys.exit(1)

    # Step 3
    X, y, feature_names = build_features(symbols_data, fetch_fundamentals)
    if len(X) == 0:
        fail("No samples after feature engineering")
        sys.exit(1)

    if args.dry_run and len(X) > 0:
        ok("Dry-run complete -- features look good, exiting without training")
        print()
        sys.exit(0)

    # Step 4
    model, X_test, y_test = train_model(X, y, feature_names)

    # Step 5
    metrics = evaluate(model, X_test, y_test, feature_names)

    # Step 6
    version = save_model(model, metrics, len(X), args.years, args.dry_run)

    elapsed = time.time() - t_start
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    if version:
        print(f"{GREEN}{BOLD}  Done!  Model v{version} ready.  ({elapsed:.0f}s total){RESET}")
    else:
        print(f"{YELLOW}{BOLD}  Done!  ({elapsed:.0f}s total)  -- model not saved{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
