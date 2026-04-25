"""
Phase 45 Phase 3: Train MetaLabelModel (v120 meta) on v119 trade outcomes.

Steps:
  1. Run AgentSimulator with v119 on in-sample data (years 1-4) to collect trades
  2. For each trade: extract features at entry, label = pnl_pct
  3. Train MetaLabelModel (XGBRegressor) to predict expected trade return
  4. Save as swing_meta_label_v1.pkl
  5. Run walk-forward with meta_model enabled (--meta-model-version 1)
  6. Write results to docs/phase45/v120_meta_results.md
"""
import sys
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pickle
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "docs" / "phase45"
MODEL_DIR = ROOT / "app" / "ml" / "models"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

ATR_STOP_MULT = 0.5
ATR_TARGET_MULT = 1.0
META_VERSION = 1
MIN_EXPECTED_R = 0.0   # threshold: only enter if predicted pnl > 0


def _ok(msg):   print(f"  \033[32mOK\033[0m  {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")
def _err(msg):  print(f"  \033[31mFAIL\033[0m  {msg}")


def load_swing_model():
    """Load latest swing model (v119)."""
    files = sorted(MODEL_DIR.glob("swing_v*.pkl"),
                   key=lambda p: int(p.stem.split("_v")[-1]))
    if not files:
        raise RuntimeError("No swing model found")
    path = files[-1]
    version = int(path.stem.split("_v")[-1])
    with open(path, "rb") as f:
        obj = pickle.load(f)
    _ok(f"Loaded swing model v{version} from {path.name}")

    if hasattr(obj, "is_trained"):
        return obj, version
    from app.ml.model import PortfolioSelectorModel
    m = PortfolioSelectorModel(model_type="xgboost")
    m.load(str(path.parent), version, model_name="swing")
    return m, version


def collect_training_trades(model, symbols_data: Dict, spy_prices, start_date: date, end_date: date):
    """Run AgentSimulator on [start_date, end_date] and return completed trades list."""
    from app.backtesting.agent_simulator import AgentSimulator

    sim = AgentSimulator(
        model=model,
        atr_stop_mult=ATR_STOP_MULT,
        atr_target_mult=ATR_TARGET_MULT,
    )
    result = sim.run(
        symbols_data,
        start_date=start_date,
        end_date=end_date,
        spy_prices=spy_prices,
    )
    _ok(f"Collected {len(result.trades)} trades ({start_date} -> {end_date}), "
        f"Sharpe {result.sharpe_ratio:.2f}")
    return result.trades


def build_meta_dataset(trades, symbols_data: Dict, feature_names: List[str]):
    """Build (X, y) from completed trades using features at entry date."""
    from app.ml.meta_model import collect_trade_features

    X, y = collect_trade_features(trades, symbols_data, feature_names)
    _ok(f"Meta dataset: {len(X)} samples, {len(feature_names)} features")
    return X, y


def train_meta_model(X, y):
    """Train MetaLabelModel and save."""
    from app.ml.meta_model import MetaLabelModel

    if len(X) < 50:
        raise RuntimeError(f"Too few training samples for meta-model: {len(X)}")

    meta = MetaLabelModel(min_expected_r=MIN_EXPECTED_R)
    metrics = meta.train(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])
    _ok(f"MetaLabel trained: R2={metrics['r2']:.3f} MAE={metrics['mae']:.4f} "
        f"corr={metrics['corr']:.3f} ({metrics['n_train']} train, {metrics['n_val']} val)")
    path = meta.save(str(MODEL_DIR), META_VERSION)
    _ok(f"Saved -> {path}")
    return meta, metrics


def run_walkforward_with_meta(meta_model):
    """Run Tier 3 walk-forward swing only with meta_model enabled."""
    import subprocess
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "walkforward_tier3.py"),
        "--model", "swing",
        "--stop-mult", str(ATR_STOP_MULT),
        "--target-mult", str(ATR_TARGET_MULT),
        "--meta-model-version", str(META_VERSION),
    ]
    log_path = ROOT / "results" / "phase45" / f"p3_walkforward_{date.today().isoformat()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ok(f"Running walk-forward with meta_model v{META_VERSION}...")
    with open(log_path, "w", encoding="utf-8") as f:
        import subprocess
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        output = f.read()
    if proc.returncode not in (0, 1):
        _err(f"Walk-forward crashed (exit {proc.returncode})")
    return output, log_path


def parse_walkforward_output(output: str):
    """Extract fold results from walk-forward stdout."""
    import re
    folds = []
    for line in output.splitlines():
        m = re.search(r"Fold\s+(\d+)\s+\[(OK|FAIL)\].*trades=(\d+)\s+win=([\d.]+)%\s+Sharpe=([-\d.]+)", line)
        if m:
            folds.append({
                "fold": int(m.group(1)),
                "gate": m.group(2),
                "trades": int(m.group(3)),
                "win_rate": float(m.group(4)) / 100,
                "sharpe": float(m.group(5)),
            })
    avg_sharpe = sum(f["sharpe"] for f in folds) / len(folds) if folds else 0
    min_sharpe = min(f["sharpe"] for f in folds) if folds else 0
    return folds, avg_sharpe, min_sharpe


def write_results_doc(folds, avg_sharpe, min_sharpe, meta_metrics):
    gate = avg_sharpe > 0.8 and min_sharpe > -0.30
    gate_str = "PASSED" if gate else "FAILED"

    lines = [
        "# Phase 45 - Phase 3 (v120 meta): MetaLabelModel Walk-Forward Results",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Primary model:** v119 (path_quality regression)",
        f"**Meta model:** MetaLabelModel v{META_VERSION} (XGBRegressor, pnl_pct target)",
        f"**Config:** stop={ATR_STOP_MULT}x ATR, target={ATR_TARGET_MULT}x ATR",
        f"**Min expected R threshold:** {MIN_EXPECTED_R}",
        "",
        "## Meta Model Training Metrics",
        "",
        f"- R2: {meta_metrics.get('r2', 0):.3f}",
        f"- MAE: {meta_metrics.get('mae', 0):.4f}",
        f"- Corr: {meta_metrics.get('corr', 0):.3f}",
        f"- Train samples: {meta_metrics.get('n_train', 0)}",
        f"- Val samples: {meta_metrics.get('n_val', 0)}",
        "",
        "## Walk-Forward Results (v119 + MetaLabelModel)",
        "",
        "| Fold | Gate | Trades | Win% | Sharpe |",
        "|---|---|---|---|---|",
    ]
    for fd in folds:
        lines.append(f"| {fd['fold']} | {fd['gate']} | {fd['trades']} | "
                     f"{fd['win_rate']*100:.1f}% | {fd['sharpe']:+.3f} |")
    lines += [
        f"| **Avg** | | | | **{avg_sharpe:+.3f}** |",
        "",
        f"## Gate Assessment: {gate_str}",
        "",
        f"- Avg Sharpe: {avg_sharpe:+.3f} (gate: > +0.80)",
        f"- Min Fold: {min_sharpe:+.3f} (gate: > -0.30)",
        "",
        "## vs Baselines",
        "",
        "| Model | Label | Meta | Avg Sharpe | Min Fold | Gate |",
        "|---|---|---|---|---|---|",
        "| v110 | cross_sectional binary | None | +0.34 | -0.73 | FAIL |",
        "| v119 (Phase 2) | path_quality regression | None | +0.476 | -0.806 | FAIL |",
        f"| v119 + meta (Phase 3) | path_quality regression | v{META_VERSION} | "
        f"{avg_sharpe:+.3f} | {min_sharpe:+.3f} | {gate_str} |",
    ]

    doc_path = DOCS_DIR / "v120_meta_results.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    _ok(f"Results written -> {doc_path}")
    print(f"\n  Gate: {gate_str}  Avg Sharpe: {avg_sharpe:+.3f}  Min Fold: {min_sharpe:+.3f}")


def main():
    print("\n" + "="*62)
    print("  Phase 45 Phase 3: Meta-Label Training + Walk-Forward")
    print("="*62)

    # 1. Load swing model
    model, version = load_swing_model()

    # 2. Download 5yr data
    _ok("Downloading 5yr data for in-sample simulation...")
    import yfinance as yf
    from app.utils.constants import SP_100_TICKERS
    symbols = list(SP_100_TICKERS)
    end_all = datetime.now()
    start_all = end_all - timedelta(days=5 * 365 + 30)

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
    _ok(f"Loaded {len(symbols_data)} symbols")

    spy_raw = yf.download("SPY", start=start_all.date().isoformat(),
                          end=end_all.date().isoformat(), progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    spy_prices = spy_raw["close"]

    # 3. Run in-sample sim on first 80% of data to collect trades
    cutoff = start_all + timedelta(days=int(5 * 365 * 0.8))
    _ok(f"Running in-sample sim {start_all.date()} -> {cutoff.date()} to collect trades...")
    t0 = time.time()
    trades = collect_training_trades(model, symbols_data, spy_prices,
                                     start_date=start_all.date(), end_date=cutoff.date())
    _ok(f"In-sample sim done in {time.time()-t0:.0f}s, {len(trades)} trades collected")

    if len(trades) < 50:
        _err(f"Only {len(trades)} trades — not enough to train meta-model. Abort.")
        sys.exit(1)

    # 4. Get feature names from model
    feature_names = []
    if hasattr(model, "_last_feature_names") and model._last_feature_names:
        feature_names = model._last_feature_names
    elif hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback: use first trade to derive names
        from app.ml.features import FeatureEngineer
        fe = FeatureEngineer()
        t = trades[0]
        sym = t.symbol
        df = symbols_data.get(sym)
        if df is not None:
            idx = pd.DatetimeIndex(df.index).date
            entry_dt = t.entry_date if isinstance(t.entry_date, date) else t.entry_date.date()
            window = df.loc[idx <= entry_dt]
            feats = fe.engineer_features(sym, window, fetch_fundamentals=False,
                                         as_of_date=entry_dt, regime_score=0.5)
            if feats:
                feature_names = list(feats.keys())
    _ok(f"Feature names: {len(feature_names)} features")

    # 5. Build meta dataset
    X, y = build_meta_dataset(trades, symbols_data, feature_names)

    # 6. Train and save MetaLabelModel
    meta_model, meta_metrics = train_meta_model(X, y)
    # Override feature_names on saved model
    meta_model.feature_names = feature_names
    meta_model.save(str(MODEL_DIR), META_VERSION)

    # 7. Run walk-forward with meta_model
    output, log_path = run_walkforward_with_meta(meta_model)
    _ok(f"Walk-forward log: {log_path}")

    # 8. Parse and write results
    folds, avg_sharpe, min_sharpe = parse_walkforward_output(output)
    write_results_doc(folds, avg_sharpe, min_sharpe, meta_metrics)

    print("\n  Phase 3 done.")


if __name__ == "__main__":
    main()
