"""
Phase 4a — Feature Correlation Audit

Analyzes saved model files to identify:
1. Zero-importance features (XGBoost never uses them)
2. Semantically related feature groups (candidates for pruning)
3. Permutation importance ranking from the saved model

Works from saved model pkl files — no data reloading required.

Usage:
    python scripts/feature_correlation_audit.py [--model swing|intraday|both]
    python scripts/feature_correlation_audit.py --output logs/feature_audit.txt
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Semantic feature groups for swing — features in the same group are semantically related
# (multiple measurements of the same underlying concept)
SWING_SEMANTIC_GROUPS = {
    "RSI variants": ["rsi_14", "rsi_7"],
    "MACD variants": ["macd", "macd_signal", "macd_histogram"],
    "EMA trend": ["ema_20", "ema_50", "price_above_ema20", "price_above_ema50"],
    "Stochastic variants": ["stoch_k", "stoch_d", "stochrsi_k", "stochrsi_d", "stochrsi_signal"],
    "52-week range": ["near_52w_high", "near_52w_low", "price_to_52w_high", "price_to_52w_low"],
    "Momentum horizons": [
        "momentum_20d", "momentum_63d", "momentum_252d_ex1m",
        "momentum_20d_sector_neutral", "sector_momentum", "sector_momentum_5d",
    ],
    "Volatility family": [
        "volatility", "vol_of_vol", "atr_norm", "parkinson_vol",
        "realized_vol_20d", "vrp",
    ],
    "Volume family": ["volume_ratio", "volume_trend", "volume_surge_3d", "pressure_index"],
    "Trend family": ["uptrend", "downtrend", "trend_consistency_63d", "trend_efficiency"],
    "WQ alphas": ["wq_alpha43", "wq_alpha44", "wq_alpha98", "wq_alpha101"],
    "DEMA/Keltner/CCI": ["dema_20_dist", "keltner_position", "cci_20", "cmf_20"],
}

INTRADAY_SEMANTIC_GROUPS = {
    "ORB variants": ["orb_position", "orb_breakout", "orb_direction_strength"],
    "VWAP variants": ["vwap_distance", "vwap_cross_count", "above_vwap_ratio"],
    "Gap variants": ["gap_pct", "gap_fill_pct", "gap_followthrough", "gap_vs_spy_gap"],
    "Return windows": ["session_return", "ret_15m", "ret_30m"],
    "EMA trend": ["ema9_dist", "ema20_dist", "ema_cross"],
    "Session timing": ["time_of_day", "minutes_since_open", "is_open_session", "is_close_session", "session_segment"],
    "Volatility": ["atr_norm", "range_compression", "daily_vol_percentile", "daily_vol_regime", "daily_parkinson_vol"],
    "Wick / candle": ["upper_wick_ratio", "lower_wick_ratio", "body_ratio", "consecutive_bars"],
    "Volume flow": ["volume_surge", "cum_delta", "vol_trend", "obv_slope"],
    "Seg features": ["seg_x_high_dist", "seg_x_atr_norm"],
    "SPY relative": ["spy_session_return", "spy_rsi_14", "rel_vol_spy",
                     "stock_vs_spy_5d_return", "stock_vs_spy_mom_ratio"],
    "Stochastic/Williams": ["stoch_k", "williams_r"],
    "Branch B (global)": ["vix_regime_level", "spy_5d_return_daily", "day_of_week"],
}


def _load_model(pkl_path: str):
    import pickle
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_latest_version(model_name: str) -> Tuple[int, str, str]:
    """Returns (version, model_path, meta_path) for the latest saved model."""
    import glob, os
    pattern = f"app/ml/models/{model_name}_v*.pkl"
    paths = [p for p in glob.glob(pattern) if "meta" not in p and "scaler" not in p]
    if not paths:
        raise FileNotFoundError(f"No {model_name} model pkl found")
    latest = max(paths, key=os.path.getmtime)
    version = int(latest.split("_v")[-1].replace(".pkl", ""))
    meta_path = f"app/ml/models/{model_name}_meta_v{version}.pkl"
    return version, latest, meta_path


def audit_model(model_name: str, semantic_groups: Dict[str, List[str]]):
    print(f"\n{'=' * 70}")
    print(f"Feature Correlation Audit — {model_name.upper()}")
    print(f"{'=' * 70}")

    version, model_path, meta_path = _get_latest_version(model_name)
    print(f"Model: v{version}  ({model_path})")

    model = _load_model(model_path)
    meta = _load_model(meta_path)
    feature_names: List[str] = meta.get("feature_names", [])
    importances = np.array(model.feature_importances_)

    if len(feature_names) != len(importances):
        print(f"  [WARN] feature_names len {len(feature_names)} != importances len {len(importances)}")
        n = min(len(feature_names), len(importances))
        feature_names = feature_names[:n]
        importances = importances[:n]

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print(f"Total features: {len(feature_names)}")

    # ── Zero-importance features ──────────────────────────────────────────────
    zero_imp = feat_imp[feat_imp == 0.0]
    low_imp = feat_imp[(feat_imp > 0) & (feat_imp < 0.002)]
    print(f"\n[1] ZERO importance ({len(zero_imp)} features — safe to remove):")
    if len(zero_imp):
        for f in zero_imp.index:
            print(f"      - {f}")
    else:
        print("      none")

    print(f"\n[2] LOW importance <0.2% ({len(low_imp)} features — review):")
    if len(low_imp):
        for f, imp in low_imp.items():
            print(f"      {imp:.4f}  {f}")
    else:
        print("      none")

    # ── Semantic group analysis ───────────────────────────────────────────────
    print(f"\n[3] SEMANTIC GROUPS (within-group redundancy):")
    drop_candidates = set(zero_imp.index)
    group_recs = []

    for group_name, members in semantic_groups.items():
        present = [f for f in members if f in feat_imp.index]
        if len(present) < 2:
            continue
        group_imp = feat_imp[present].sort_values(ascending=False)
        winner = group_imp.index[0]
        losers = list(group_imp.index[1:])
        useful_losers = [f for f in losers if feat_imp[f] > 0.002]
        dead_losers = [f for f in losers if feat_imp[f] <= 0.002]
        print(f"\n  {group_name}: {present}")
        print(f"    Keep:  {winner} ({feat_imp[winner]:.4f})")
        for f in useful_losers:
            print(f"    Low:   {f} ({feat_imp[f]:.4f}) — consider keeping if adds diversity")
        for f in dead_losers:
            drop_candidates.add(f)
            print(f"    Drop:  {f} ({feat_imp[f]:.4f}) — zero/near-zero, redundant")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    print(f"PRUNING SUMMARY for {model_name.upper()} v{version}:")
    print(f"  Current features:  {len(feature_names)}")
    print(f"  Recommended drops: {len(drop_candidates)}")
    print(f"  After pruning:     {len(feature_names) - len(drop_candidates)}")
    print(f"\n  Drop list:")
    for f in sorted(drop_candidates):
        print(f"    - {f}  (importance={feat_imp.get(f, 0.0):.4f})")

    print(f"\n  Top 10 by importance (must keep):")
    for f, imp in feat_imp.head(10).items():
        print(f"    {imp:.4f}  {f}")

    return {
        "version": version,
        "total_features": len(feature_names),
        "zero_importance": list(zero_imp.index),
        "low_importance": list(low_imp.index),
        "drop_candidates": sorted(drop_candidates),
        "recommended_feature_count": len(feature_names) - len(drop_candidates),
        "top_features": list(feat_imp.head(10).index),
    }


def run_audit(model: str = "both", output: str = None):
    results = {}
    if model in ("swing", "both"):
        results["swing"] = audit_model("swing", SWING_SEMANTIC_GROUPS)
    if model in ("intraday", "both"):
        results["intraday"] = audit_model("intraday", INTRADAY_SEMANTIC_GROUPS)

    print(f"\n{'=' * 70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'=' * 70}")
    for name, r in results.items():
        print(f"  {name.upper()} v{r['version']}: "
              f"{r['total_features']} -> {r['recommended_feature_count']} features "
              f"(drop {len(r['drop_candidates'])})")

    if output:
        import json
        Path(output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output}")

    print("\nAudit complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["swing", "intraday", "both"], default="both")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_audit(args.model, args.output)
