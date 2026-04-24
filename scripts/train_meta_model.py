"""
Phase 37 — Train the Meta-Label Model (Expected R gate)

Runs a fresh Tier 3 AgentSimulator backtest on the active swing model,
collects (features_at_entry, pnl_pct) pairs from all completed trades,
and trains a secondary XGBoost regressor (MetaLabelModel).

The trained model is saved to app/ml/models/swing_meta_label_v{N}.pkl.
In AgentSimulator, entries where predicted E[R] < min_expected_r are skipped.

Usage:
    python scripts/train_meta_model.py [--years N] [--min-er FLOAT] [--sample N]

Gate: meta-model corr(predicted, actual) > 0.10 on held-out val set
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def ok(msg):   print(f"  {GREEN}OK{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!!{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")


def main():
    parser = argparse.ArgumentParser(description="Phase 37 — Meta-label model training")
    parser.add_argument("--years", type=int, default=2, help="Years of backtest history")
    parser.add_argument("--min-er", type=float, default=0.002,
                        help="Min expected R gate threshold (default 0.002 = 0.2%%)")
    parser.add_argument("--sample", type=int, default=None, help="Sample N symbols (faster)")
    args = parser.parse_args()

    import yfinance as yf
    import random
    from app.utils.constants import SP_500_TICKERS
    from app.backtesting.agent_simulator import AgentSimulator
    from app.ml.meta_model import MetaLabelModel, collect_trade_features

    symbols = SP_500_TICKERS
    if args.sample:
        symbols = random.sample(symbols, min(args.sample, len(symbols)))

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Phase 37 — Meta-Label Model Training{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  Symbols: {len(symbols)}  |  History: {args.years}yr  |  min_er: {args.min_er:.3f}")

    # Load active swing model
    sys.path.insert(0, str(ROOT / "scripts"))
    from backtest_ml_models import _load_model, _load_price_cache, _save_price_cache

    model = _load_model("swing")
    if model is None:
        fail("No active swing model. Train one first.")
        sys.exit(1)
    ok(f"Loaded swing model (feature count: {len(getattr(model, 'feature_names', []))})")

    # Load or download price data
    end = datetime.now()
    start = end - timedelta(days=365 * args.years + 35)

    cached = _load_price_cache(symbols, args.years)
    if cached is not None:
        symbols_data, spy_prices = cached
        ok(f"Loaded {len(symbols_data)} symbols from cache")
    else:
        ok("Downloading price data...")
        import pandas as pd
        raw = yf.download(
            symbols, start=start.date().isoformat(), end=end.date().isoformat(),
            progress=False, auto_adjust=True, group_by="ticker",
        )
        symbols_data = {}
        for sym in symbols:
            try:
                df = raw[sym].dropna()
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    symbols_data[sym] = df
            except Exception:
                pass
        spy_raw = yf.download("SPY", start=start.date().isoformat(),
                              end=end.date().isoformat(), progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_raw.columns = [c.lower() for c in spy_raw.columns]
        spy_prices = spy_raw["close"]
        _save_price_cache(symbols_data, spy_prices, symbols, args.years)
        ok(f"Downloaded {len(symbols_data)} symbols")

    # Run Tier 3 simulation (training data generation pass)
    from datetime import timedelta as td, date as date_type
    agent_start = start.date() + td(days=420)
    ok(f"Running Tier 3 simulation {agent_start} → {end.date()} to collect trade outcomes...")

    import time
    t0 = time.time()
    sim = AgentSimulator(model=model)
    result = sim.run(
        symbols_data, spy_prices=spy_prices,
        start_date=agent_start, end_date=end.date(),
    )
    ok(f"Simulation done in {time.time()-t0:.1f}s — {result.total_trades} trades")

    if result.total_trades < 50:
        fail(f"Too few trades ({result.total_trades}) for meta-label training. Run with more years/symbols.")
        sys.exit(1)

    # Build (features, pnl_pct) training data
    feature_names = getattr(model, "feature_names", [])
    if not feature_names:
        fail("Model has no feature_names — cannot build meta-label training data.")
        sys.exit(1)

    ok(f"Collecting features for {result.total_trades} trades...")
    t0 = time.time()
    X, y = collect_trade_features(result.trades, symbols_data, feature_names)
    ok(f"Built meta-label dataset: {len(X)} samples in {time.time()-t0:.1f}s")

    if len(X) < 30:
        fail(f"Too few enrichable trades ({len(X)}). Need more data.")
        sys.exit(1)

    import numpy as np
    ok(f"Target distribution: mean={np.mean(y):.3f} std={np.std(y):.3f} "
       f"pos={np.mean(y>0):.1%}")

    # Train MetaLabelModel
    ok("Training meta-label regressor...")
    meta = MetaLabelModel(min_expected_r=args.min_er)
    metrics = meta.train(X, y, feature_names)
    ok(f"Trained: R²={metrics['r2']:.3f}  MAE={metrics['mae']:.4f}  "
       f"corr={metrics['corr']:.3f}  ({metrics['n_train']} train / {metrics['n_val']} val)")

    if metrics["corr"] < 0.05:
        warn(f"Low correlation ({metrics['corr']:.3f}) — meta-model may not add value")
    elif metrics["corr"] >= 0.10:
        ok(f"Gate passed: corr {metrics['corr']:.3f} >= 0.10")
    else:
        warn(f"Marginal correlation ({metrics['corr']:.3f}) — proceed with caution")

    # Save model
    model_dir = ROOT / "app" / "ml" / "models"
    from pathlib import Path as _P
    existing = sorted(_P(model_dir).glob("swing_meta_label_v*.pkl"))
    next_v = int(existing[-1].stem.split("_v")[-1]) + 1 if existing else 1

    path = meta.save(str(model_dir), next_v)
    ok(f"Saved meta-label model v{next_v} -> {path}")

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  Meta-label model ready. Set use_meta_model=True in")
    print(f"  AgentSimulator to gate entries on E[R] >= {args.min_er:.3f}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
