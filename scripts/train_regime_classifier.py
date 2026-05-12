"""Train the MVP logistic regime classifier (R5).

Downloads SPY / ^VIX / HYG daily bars via yfinance, fits RegimeClassifier
on 2015-2023, evaluates on 2024 (hold-out: 2025+), and saves to
app/ml/models/regime_v1.pkl.

Usage:
    python scripts/train_regime_classifier.py
    python scripts/train_regime_classifier.py --start 2010-01-01 --vix-threshold 20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.regime_classifier import (
    RegimeClassifier,
    build_regime_features,
    build_regime_labels,
    FEATURE_NAMES,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("app/ml/models/regime_v1.pkl")
TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"


def _download(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    dfs = {}
    for sym in symbols:
        logger.info("Downloading %s (%s to %s)...", sym, start, end)
        df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"yfinance returned empty DataFrame for {sym}")
        # yfinance >= 0.2 returns MultiIndex columns (field, ticker) - flatten to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        dfs[sym] = df
    return dfs


def _evaluate(clf: RegimeClassifier, features: pd.DataFrame, labels: pd.Series,
              split_start: str, split_end: str, split_name: str) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    idx = features.index.normalize()
    mask = (idx >= pd.Timestamp(split_start)) & (idx <= pd.Timestamp(split_end))
    X = features.loc[mask]
    y = labels.reindex(X.index).dropna()
    X = X.loc[y.index]
    if len(y) < 10:
        logger.warning("%s: too few samples (%d), skipping eval", split_name, len(y))
        return {}
    probs = clf.predict_proba_series(X)
    y_arr = y.values.astype(int)
    p_arr = probs.values
    auc = roc_auc_score(y_arr, p_arr)
    brier = brier_score_loss(y_arr, p_arr)
    baseline_brier = float(y_arr.mean() * (1 - y_arr.mean()))
    label_mean = float(y_arr.mean())
    print(f"\n  {split_name} ({split_start} -> {split_end}):")
    print(f"    Samples:       {len(y)}")
    print(f"    Label mean:    {label_mean:.3f}  {'WARN: IMBALANCED (>80% or <20%)' if label_mean > 0.8 or label_mean < 0.2 else 'OK'}")
    print(f"    AUC:           {auc:.3f}  {'PASS (>=0.75)' if auc >= 0.75 else 'FAIL (<0.75)'}")
    print(f"    Brier:         {brier:.4f}  (baseline: {baseline_brier:.4f})  {'PASS' if brier < baseline_brier else 'FAIL: WORSE THAN BASELINE'}")
    return {"auc": auc, "brier": brier, "label_mean": label_mean, "n_samples": len(y)}


def main():
    parser = argparse.ArgumentParser(description="Train regime classifier (R5)")
    parser.add_argument("--start", default="2015-01-01", help="Download start date")
    parser.add_argument("--vix-threshold", type=float, default=25.0,
                        help="VIX threshold for label (default: 25)")
    parser.add_argument("--output", default=str(MODEL_PATH),
                        help="Output pkl path")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MrTrader - Regime Classifier Training (R5)")
    print("=" * 60)
    print(f"  Download: {args.start} -> {VAL_END}")
    print(f"  Train:    {args.start} -> {TRAIN_END}")
    print(f"  Validate: {TRAIN_END[:4]+'-01-01'} (2024) -> {VAL_END}")
    print(f"  VIX threshold for label: {args.vix_threshold}")

    # Download
    raw = _download(["SPY", "^VIX", "HYG"], start=args.start, end=VAL_END)
    spy_df, vix_df, hyg_df = raw["SPY"], raw["^VIX"], raw["HYG"]

    # Build features and labels
    logger.info("Building features...")
    features = build_regime_features(spy_df, vix_df, hyg_df)
    labels = build_regime_labels(spy_df, vix_df, vix_threshold=args.vix_threshold)

    # Align
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    logger.info("Aligned dataset: %d rows (%s to %s)", len(features),
                str(features.index[0].date()), str(features.index[-1].date()))

    label_mean_full = labels.mean()
    print(f"\n  Full label mean: {label_mean_full:.3f}", end="")
    if label_mean_full > 0.8 or label_mean_full < 0.2:
        print(" WARN: IMBALANCED - consider adjusting --vix-threshold")
    else:
        print(" (OK)")

    # Train
    clf = RegimeClassifier(vix_threshold=args.vix_threshold)
    clf.fit(features, labels, train_end=TRAIN_END)

    # Evaluate
    train_metrics = _evaluate(clf, features, labels, args.start, TRAIN_END, "Train")
    val_metrics = _evaluate(clf, features, labels, "2024-01-01", VAL_END, "Validation (2024)")

    # Store val metrics on classifier
    clf._val_auc = val_metrics.get("auc")
    clf._val_brier = val_metrics.get("brier")

    # Save
    out_path = Path(args.output)
    clf.save(out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Meta:  {out_path.parent / (out_path.stem + '_meta.json')}")

    # Summary gate
    val_auc = val_metrics.get("auc", 0.0)
    val_brier = val_metrics.get("brier", 1.0)
    val_baseline = val_metrics.get("label_mean", 0.5) * (1 - val_metrics.get("label_mean", 0.5))
    gate_passed = val_auc >= 0.75 and val_brier < val_baseline
    print(f"\n  Gate: {'PASS' if gate_passed else 'FAIL'} (AUC>=0.75 and Brier<baseline)")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
