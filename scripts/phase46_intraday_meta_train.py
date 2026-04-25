"""
Phase 46-B: Train IntraDay MetaLabelModel on v19 trade outcomes.

Steps:
  1. Load latest intraday model
  2. Run IntradayAgentSimulator on in-sample data (first 80% of available days)
     to collect completed trades
  3. For each trade: extract features at entry, label = pnl_pct
  4. Train MetaLabelModel (XGBRegressor) to predict expected trade return
  5. Save as intraday_meta_label_v1.pkl
  6. Run walk-forward with meta_model + abstention gate enabled
  7. Write results to docs/phase46/v19_meta_results.md
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "docs" / "phase46"
MODEL_DIR = ROOT / "app" / "ml" / "models"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

META_VERSION = 1
MIN_EXPECTED_R = 0.0
PM_ABSTENTION_VIX = 25.0
PM_ABSTENTION_SPY_MA = 20


def _ok(msg):   print(f"  \033[32mOK\033[0m  {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")
def _err(msg):  print(f"  \033[31mFAIL\033[0m  {msg}")


def load_intraday_model():
    files = sorted(MODEL_DIR.glob("intraday_v*.pkl"),
                   key=lambda p: int(p.stem.split("_v")[-1]))
    if not files:
        raise RuntimeError("No intraday model found")
    path = files[-1]
    version = int(path.stem.split("_v")[-1])
    with open(path, "rb") as f:
        obj = pickle.load(f)
    _ok(f"Loaded intraday model v{version} from {path.name}")
    return obj, version


def collect_training_trades(model, symbols_data: Dict, spy_data, all_days):
    """Run IntradayAgentSimulator on in-sample days to collect trades."""
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator

    cutoff_idx = int(len(all_days) * 0.8)
    te_start = all_days[0]
    te_end = all_days[cutoff_idx]

    sim = IntradayAgentSimulator(model=model)
    result = sim.run(
        symbols_data,
        spy_data=spy_data,
        start_date=te_start,
        end_date=te_end,
    )
    _ok(f"Collected {len(result.trades)} trades ({te_start} -> {te_end}), "
        f"Sharpe {result.sharpe_ratio:.2f}")
    return result.trades


def build_meta_dataset(trades, symbols_data: Dict, spy_data):
    """Build (X, y) from completed trades using features at entry."""
    from app.ml.intraday_features import compute_intraday_features
    from app.backtesting.intraday_agent_simulator import _index_by_day, FEATURE_BARS

    spy_by_day = _index_by_day(spy_data) if spy_data is not None else {}

    rows_X = []
    rows_y = []

    for t in trades:
        sym = t.symbol
        df = symbols_data.get(sym)
        if df is None:
            continue

        entry_date = t.entry_date if isinstance(t.entry_date, date) else t.entry_date.date()
        df_idx = pd.DatetimeIndex(df.index)
        day_mask = df_idx.normalize().date == entry_date
        day_bars = df.loc[day_mask]

        if len(day_bars) < FEATURE_BARS:
            continue

        feat_bars = day_bars.iloc[:FEATURE_BARS]
        prior_close = prior_high = prior_low = None
        all_dates = sorted(set(df_idx.normalize().date))
        prior_days = [d for d in all_dates if d < entry_date]
        if prior_days:
            prev_day = prior_days[-1]
            prev_mask = df_idx.normalize().date == prev_day
            prev_bars = df.loc[prev_mask]
            if len(prev_bars) > 0:
                prior_close = float(prev_bars["close"].iloc[-1])
                prior_high = float(prev_bars["high"].max())
                prior_low = float(prev_bars["low"].min())

        try:
            feats = compute_intraday_features(
                feat_bars,
                spy_by_day.get(entry_date),
                prior_close,
                prior_day_high=prior_high,
                prior_day_low=prior_low,
            )
        except Exception:
            feats = None
        if feats is None:
            continue

        rows_X.append(list(feats.values()))
        rows_y.append(float(t.pnl_pct))

    if not rows_X:
        raise RuntimeError("No features collected from trades")

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)
    _ok(f"Meta dataset: {len(X)} samples, {X.shape[1]} features")
    return X, y


def train_meta_model(X, y):
    from app.ml.meta_model import MetaLabelModel

    if len(X) < 30:
        raise RuntimeError(f"Too few training samples: {len(X)}")

    meta = MetaLabelModel(min_expected_r=MIN_EXPECTED_R)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    metrics = meta.train(X, y, feature_names=feature_names)
    _ok(f"Meta trained: R2={metrics['r2']:.3f} MAE={metrics['mae']:.4f} "
        f"corr={metrics['corr']:.3f} ({metrics['n_train']} train, {metrics['n_val']} val)")
    path = meta.save(str(MODEL_DIR), META_VERSION, model_type="intraday")
    _ok(f"Saved -> {path}")
    return meta, metrics


def run_walkforward_with_meta():
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "walkforward_tier3.py"),
        "--model", "intraday",
        "--intraday-meta-model-version", str(META_VERSION),
        "--pm-abstention-vix", str(PM_ABSTENTION_VIX),
        "--pm-abstention-spy-ma-days", str(PM_ABSTENTION_SPY_MA),
    ]
    log_path = ROOT / "results" / "phase46" / f"p46b_walkforward_{date.today().isoformat()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ok(f"Running intraday walk-forward with meta v{META_VERSION} + abstention gate...")
    import subprocess
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        output = f.read()
    if proc.returncode not in (0, 1):
        _err(f"Walk-forward crashed (exit {proc.returncode})")
    return output, log_path


def parse_walkforward_output(output: str):
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


def write_results_doc(folds, avg_sharpe, min_sharpe, meta_metrics, baseline_folds=None):
    gate = avg_sharpe > 0.8 and min_sharpe > -0.30
    gate_str = "PASSED" if gate else "FAILED"

    lines = [
        "# Phase 46 - 46-B+C: Intraday MetaLabelModel + Abstention Gate Results",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Model:** Intraday v19 (ATR-adaptive labels, 38 features)",
        f"**Meta model:** MetaLabelModel v{META_VERSION} (XGBRegressor, pnl_pct target)",
        f"**PM abstention gate:** VIX >= {PM_ABSTENTION_VIX} OR SPY < {PM_ABSTENTION_SPY_MA}-day SMA",
        "",
        "## Meta Model Training Metrics",
        "",
        f"- R2: {meta_metrics.get('r2', 0):.3f}",
        f"- MAE: {meta_metrics.get('mae', 0):.4f}",
        f"- Corr: {meta_metrics.get('corr', 0):.3f}",
        f"- Train samples: {meta_metrics.get('n_train', 0)}",
        f"- Val samples: {meta_metrics.get('n_val', 0)}",
        "",
        "## Walk-Forward Results (v19 + MetaLabelModel + PM Abstention)",
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
    ]

    doc_path = DOCS_DIR / "v19_meta_results.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    _ok(f"Results written -> {doc_path}")
    print(f"\n  Gate: {gate_str}  Avg Sharpe: {avg_sharpe:+.3f}  Min Fold: {min_sharpe:+.3f}")


def main():
    print("\n" + "="*62)
    print("  Phase 46-B+C: Intraday Meta-Label + Abstention Gate")
    print("="*62)

    model, version = load_intraday_model()

    # Load 5-min data from Polygon cache
    from app.data.intraday_cache import load_many, available_symbols
    from app.utils.constants import RUSSELL_1000_TICKERS

    symbols = list(RUSSELL_1000_TICKERS)
    cache_syms = set(available_symbols())
    _ok(f"Polygon cache: {len(cache_syms)} symbols available")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=740)

    symbols_data = load_many(
        [s for s in symbols if s in cache_syms],
        start=start_date, end=end_date,
    )
    _ok(f"Loaded {len(symbols_data)} symbols")

    spy_data = symbols_data.get("SPY")

    all_days = sorted({
        d for df in symbols_data.values()
        for d in pd.to_datetime(df.index).date
    })
    _ok(f"Trading days available: {len(all_days)} ({all_days[0]} -> {all_days[-1]})")

    if len(all_days) < 50:
        _err("Not enough trading days. Abort.")
        sys.exit(1)

    t0 = time.time()
    trades = collect_training_trades(model, symbols_data, spy_data, all_days)
    _ok(f"Simulation done in {time.time()-t0:.0f}s, {len(trades)} trades collected")

    if len(trades) < 30:
        _err(f"Only {len(trades)} trades — not enough for meta-model. Abort.")
        sys.exit(1)

    X, y = build_meta_dataset(trades, symbols_data, spy_data)
    meta_model, meta_metrics = train_meta_model(X, y)

    output, log_path = run_walkforward_with_meta()
    _ok(f"Walk-forward log: {log_path}")

    folds, avg_sharpe, min_sharpe = parse_walkforward_output(output)
    write_results_doc(folds, avg_sharpe, min_sharpe, meta_metrics)

    print("\n  Phase 46-B+C done.")


if __name__ == "__main__":
    main()
