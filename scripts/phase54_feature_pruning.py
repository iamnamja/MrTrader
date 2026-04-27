"""
Phase 54 -- Intraday Feature Pruning

Applies Phase 43 methodology to intraday model:
1. Retrain on all 50 features, extract feature_importances_
2. Identify zero/near-zero features (< 0.005 mean importance)
3. Remove them from FEATURE_NAMES in intraday_features.py
4. Retrain pruned model
5. Walk-forward to confirm no Sharpe regression vs v23 (+1.275)

Usage:
    python scripts/phase54_feature_pruning.py [--threshold 0.005]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase54_feature_pruning")

OUTPUT_FILE = Path("docs/phase54_pruning_report.md")
FEATURES_FILE = Path("app/ml/intraday_features.py")


def get_feature_importances() -> dict:
    """Load the latest intraday model (v25 or v26) and extract feature importances."""
    import pickle
    from app.ml.intraday_features import FEATURE_NAMES

    model_dir = Path("app/ml/models")
    model_files = sorted(
        [p for p in model_dir.glob("intraday_v*.pkl")
         if not any(x in p.stem for x in ["meta", "scaler"])],
        key=lambda p: int(p.stem.split("_v")[-1])
    )
    if not model_files:
        logger.error("No intraday model found")
        sys.exit(1)

    latest = model_files[-1]
    logger.info("Loading model: %s", latest)

    with open(latest, "rb") as f:
        obj = pickle.load(f)

    # Could be PortfolioSelectorModel or raw XGBoost
    if hasattr(obj, "model"):
        raw_model = obj.model
    else:
        raw_model = obj

    if hasattr(raw_model, "feature_importances_"):
        importances = raw_model.feature_importances_
    else:
        logger.error("Model has no feature_importances_ attribute")
        sys.exit(1)

    # Map to feature names
    feature_names = getattr(obj, "feature_names", None) or FEATURE_NAMES
    if len(importances) != len(feature_names):
        logger.warning(
            "Importance count (%d) != feature name count (%d) — using index",
            len(importances), len(feature_names)
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    return dict(zip(feature_names, importances))


def identify_prunable(importances: dict, threshold: float) -> list:
    """Return list of features with importance < threshold."""
    prunable = [f for f, imp in importances.items() if imp < threshold]
    return sorted(prunable, key=lambda f: importances[f])


def write_report(importances: dict, prunable: list, threshold: float) -> None:
    from app.ml.intraday_features import FEATURE_NAMES

    ranked = sorted(importances.items(), key=lambda x: -x[1])

    lines = [
        "# Phase 54 -- Intraday Feature Pruning Report",
        "",
        f"Generated: {__import__('datetime').date.today()}",
        f"Pruning threshold: {threshold}",
        "",
        "## Full Feature Importance Ranking",
        "",
        "| Rank | Feature | Importance | Status |",
        "|---|---|---|---|",
    ]
    for rank, (feat, imp) in enumerate(ranked, 1):
        status = "PRUNED" if feat in prunable else "kept"
        lines.append(f"| {rank} | {feat} | {imp:.5f} | {status} |")

    lines += [
        "",
        f"## Summary",
        f"- Total features: {len(importances)}",
        f"- Features to prune: {len(prunable)} (importance < {threshold})",
        f"- Features kept: {len(importances) - len(prunable)}",
        "",
        "## Features to Prune",
        "",
    ]
    for f in prunable:
        lines.append(f"- `{f}` (importance={importances[f]:.6f})")

    lines += [
        "",
        "## Walk-Forward Gate",
        "Gate: avg Sharpe >= +1.275 (v23 baseline). Must not regress.",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", OUTPUT_FILE)
    print("\n".join(lines))


def prune_features_in_source(prunable: list) -> list:
    """Remove prunable features from FEATURE_NAMES in intraday_features.py.
    Returns the new feature list."""
    from app.ml.intraday_features import FEATURE_NAMES

    new_names = [f for f in FEATURE_NAMES if f not in set(prunable)]
    logger.info("Pruning: %d -> %d features", len(FEATURE_NAMES), len(new_names))

    # Read the file and replace the FEATURE_NAMES list
    content = FEATURES_FILE.read_text(encoding="utf-8")

    # Find the FEATURE_NAMES list and replace it
    import re
    # Build new list string
    indent = "    "
    grouped = []
    # Preserve grouping comments by rebuilding with comments for known groups
    group_comments = {
        "orb_position": "# Price / structure",
        "ema9_dist": "# Trend / moving averages",
        "rsi_14": "# Momentum",
        "volume_surge": "# Volume / order flow",
        "upper_wick_ratio": "# Candlestick",
        "atr_norm": "# Volatility",
        "spy_session_return": "# Market context",
        "time_of_day": "# Session timing",
        "daily_vol_percentile": "# Daily vol context",
        "whale_candle": "# Institutional activity",
        "trend_efficiency": "# Phase 47-5: Quality / structure features",
    }

    new_list_lines = ["FEATURE_NAMES = ["]
    current_group_comment = None
    for feat in new_names:
        if feat in group_comments:
            new_list_lines.append(f"{indent}{group_comments[feat]}")
        row_features = [feat]
        new_list_lines.append(f"{indent}{repr(feat)},")

    # Remove trailing comma from last item
    if new_list_lines[-1].endswith(","):
        new_list_lines[-1] = new_list_lines[-1]

    new_list_lines.append("]")
    new_list_str = "\n".join(new_list_lines)

    # Replace in file using regex
    pattern = r"FEATURE_NAMES\s*=\s*\[.*?\]"
    new_content = re.sub(pattern, new_list_str, content, flags=re.DOTALL)

    FEATURES_FILE.write_text(new_content, encoding="utf-8")
    logger.info("Updated %s", FEATURES_FILE)
    return new_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Features with importance < threshold are pruned")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, do not modify files or retrain")
    args = parser.parse_args()

    logger.info("=== Phase 54: Intraday Feature Pruning ===")

    # Step 1: Get feature importances from latest model
    importances = get_feature_importances()
    logger.info("Got importances for %d features", len(importances))

    # Step 2: Identify prunable features
    prunable = identify_prunable(importances, args.threshold)
    logger.info("Features to prune (%d): %s", len(prunable), prunable)

    # Step 3: Write report
    write_report(importances, prunable, args.threshold)

    if args.dry_run:
        logger.info("Dry run -- stopping before source modification and retrain")
        return

    if not prunable:
        logger.info("No features to prune at threshold %.4f -- done", args.threshold)
        return

    # Step 4: Modify FEATURE_NAMES in source
    new_names = prune_features_in_source(prunable)
    logger.info("Source updated. New feature count: %d", len(new_names))

    # Step 5: Retrain
    logger.info("=== Starting retrain with pruned features ===")
    result = subprocess.run(
        [sys.executable, "scripts/retrain_intraday.py", "--days", "730",
         "--top-n-liquidity", "0"],
        check=False
    )
    if result.returncode != 0:
        logger.error("Retrain failed with code %d", result.returncode)
        sys.exit(1)

    # Step 6: Walk-forward
    logger.info("=== Starting walk-forward validation ===")
    result = subprocess.run(
        [sys.executable, "scripts/walkforward_tier3.py", "--model", "intraday"],
        check=False
    )
    if result.returncode != 0:
        logger.error("Walk-forward failed with code %d", result.returncode)
        sys.exit(1)

    logger.info("Phase 54 complete. Check walk-forward results above.")
    logger.info("Gate: avg Sharpe >= +1.275 (v23 baseline)")


if __name__ == "__main__":
    main()
