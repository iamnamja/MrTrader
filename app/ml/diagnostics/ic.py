"""
app/ml/diagnostics/ic.py — Cross-sectional Information Coefficient (IC) library.

Computes Spearman rank correlation between feature values at time t and forward
returns over horizons h, grouped by day (cross-sectional). Aggregates to per-
feature IC mean, IC IR (annualised), hit rate, and regime-conditional breakdown.

Used by:
    scripts/diag_feature_ic.py   — batch 5-year IC audit (Phase A1)
    scripts/walkforward/engine.py — optional per-fold IC monitoring (future)

All functions are pure (no I/O, no DB). Callers supply the data panels.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ── Thresholds (documented, not enforced here — callers decide) ───────────────
# Industry benchmarks for cross-sectional daily equity factors:
IC_MEAN_MIN = 0.02      # minimum mean IC to consider a feature informative
IC_IR_MIN = 0.5         # minimum annualised IC IR (IC_mean / IC_std * sqrt(252))
HIT_RATE_MIN = 0.53     # minimum fraction of days where IC has correct sign


def compute_daily_ic(
    feature_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20),
    min_symbols_per_day: int = 30,
) -> pd.DataFrame:
    """Compute daily cross-sectional Spearman IC for each (feature, horizon).

    Args:
        feature_panel:   MultiIndex (date, symbol) DataFrame, columns = feature names.
                         Features should be the raw (un-normalised) values as seen
                         by the model at that date (point-in-time correct).
        forward_returns: MultiIndex (date, symbol) DataFrame with columns named by
                         horizon, e.g. {5: 'fwd_5d', 10: 'fwd_10d', ...}.
                         Values are simple forward returns (not log).
        horizons:        Forecast horizons in trading days.
        min_symbols_per_day: Drop a (date, horizon) pair if fewer than this many
                         symbols have valid (feature, return) pairs. Prevents
                         spurious ICs on thin trading days (e.g. 24 Dec).

    Returns:
        Long-format DataFrame with columns:
            date, feature, horizon, ic, n_symbols
        One row per (date, feature, horizon) triple where n_symbols >= threshold.
    """
    if feature_panel.empty or forward_returns.empty:
        return pd.DataFrame(columns=["date", "feature", "horizon", "ic", "n_symbols"])

    feature_names = list(feature_panel.columns)
    records: List[Dict] = []

    dates = feature_panel.index.get_level_values("date").unique()
    for day in dates:
        try:
            X = feature_panel.loc[day]   # (symbols, features) for this day
        except KeyError:
            continue
        if not isinstance(X, pd.DataFrame):
            X = X.to_frame().T

        for h in horizons:
            col = f"fwd_{h}d"
            if col not in forward_returns.columns:
                continue
            try:
                R = forward_returns.loc[day, col]
            except KeyError:
                continue
            if isinstance(R, float):
                continue  # only one symbol — skip

            # Align on symbol
            common = X.index.intersection(R.index)
            if len(common) < min_symbols_per_day:
                continue

            X_day = X.loc[common]
            R_day = R.loc[common]

            for feat in feature_names:
                x_vals = X_day[feat].values.astype(float)
                r_vals = R_day.values.astype(float)
                # Drop NaN pairs
                mask = np.isfinite(x_vals) & np.isfinite(r_vals)
                if mask.sum() < min_symbols_per_day:
                    continue
                try:
                    corr, _ = spearmanr(x_vals[mask], r_vals[mask])
                    if np.isfinite(corr):
                        records.append({
                            "date": day,
                            "feature": feat,
                            "horizon": h,
                            "ic": float(corr),
                            "n_symbols": int(mask.sum()),
                        })
                except Exception:
                    continue

    return pd.DataFrame(records)


def aggregate_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily IC to per-feature summary statistics.

    Args:
        daily_ic: Output of compute_daily_ic().

    Returns:
        DataFrame indexed by feature with columns per horizon:
            ic_mean_h{N}, ic_std_h{N}, ic_ir_h{N}, ic_t_h{N},
            hit_rate_h{N}, n_days_h{N}
        Plus cross-horizon:
            decay_5_10  (ic_mean_h10 / ic_mean_h5, if both available)
            decay_10_20 (ic_mean_h20 / ic_mean_h10)
            best_horizon (horizon with highest |ic_ir|)
    """
    if daily_ic.empty:
        return pd.DataFrame()

    features = daily_ic["feature"].unique()
    horizons = sorted(daily_ic["horizon"].unique())
    rows = []

    for feat in features:
        row: Dict = {"feature": feat}
        ic_means = {}
        for h in horizons:
            sub = daily_ic.loc[
                (daily_ic["feature"] == feat) & (daily_ic["horizon"] == h), "ic"
            ]
            n = len(sub)
            if n == 0:
                continue
            mu = float(sub.mean())
            sigma = float(sub.std()) if n > 1 else 1.0
            ir = mu / sigma * np.sqrt(252) if sigma > 0 else 0.0
            t_stat = mu / (sigma / np.sqrt(n)) if sigma > 0 else 0.0
            hit = float((sub * np.sign(mu) > 0).mean()) if mu != 0 else 0.5
            row[f"ic_mean_h{h}"] = round(mu, 6)
            row[f"ic_std_h{h}"] = round(sigma, 6)
            row[f"ic_ir_h{h}"] = round(ir, 4)
            row[f"ic_t_h{h}"] = round(t_stat, 4)
            row[f"hit_rate_h{h}"] = round(hit, 4)
            row[f"n_days_h{h}"] = n
            ic_means[h] = mu

        # Decay ratios
        if 5 in ic_means and 10 in ic_means and ic_means[5] != 0:
            row["decay_5_10"] = round(ic_means[10] / ic_means[5], 4)
        if 10 in ic_means and 20 in ic_means and ic_means[10] != 0:
            row["decay_10_20"] = round(ic_means[20] / ic_means[10], 4)

        # Best horizon by |IC IR|
        best_h = max(
            [(h, abs(row.get(f"ic_ir_h{h}", 0.0))) for h in horizons],
            key=lambda x: x[1],
            default=(horizons[0] if horizons else 0, 0.0),
        )
        row["best_horizon"] = best_h[0]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("feature")
    # Sort by |IC IR| at shortest available horizon
    sort_col = f"ic_ir_h{horizons[0]}" if horizons else "feature"
    if sort_col in df.columns:
        df = df.reindex(df[sort_col].abs().sort_values(ascending=False).index)
    return df


def summarize_by_regime(
    daily_ic: pd.DataFrame,
    regime_labels: Dict[date, str],
) -> pd.DataFrame:
    """Break down IC statistics per regime label.

    Args:
        daily_ic:      Output of compute_daily_ic().
        regime_labels: {date: label_str} mapping, e.g. {date(2025,2,1): "RISK_OFF"}.
                       Days not in the dict get label "UNKNOWN".

    Returns:
        DataFrame with MultiIndex (feature, regime) and IC summary columns
        for each horizon found in daily_ic.
    """
    if daily_ic.empty:
        return pd.DataFrame()

    daily_ic = daily_ic.copy()
    daily_ic["regime"] = daily_ic["date"].map(
        lambda d: regime_labels.get(d, "UNKNOWN")
    )

    horizons = sorted(daily_ic["horizon"].unique())
    records = []
    for (feat, regime), grp in daily_ic.groupby(["feature", "regime"]):
        row: Dict = {"feature": feat, "regime": regime}
        for h in horizons:
            sub = grp.loc[grp["horizon"] == h, "ic"]
            n = len(sub)
            if n == 0:
                continue
            mu = float(sub.mean())
            sigma = float(sub.std()) if n > 1 else 1.0
            row[f"ic_mean_h{h}"] = round(mu, 6)
            row[f"ic_ir_h{h}"] = round(mu / sigma * np.sqrt(252) if sigma > 0 else 0.0, 4)
            row[f"n_days_h{h}"] = n
        records.append(row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index(["feature", "regime"])


def summarize_by_year(daily_ic: pd.DataFrame) -> pd.DataFrame:
    """Break down IC by calendar year — detects temporal drift / regime shift."""
    if daily_ic.empty:
        return pd.DataFrame()

    daily_ic = daily_ic.copy()
    daily_ic["year"] = pd.to_datetime(daily_ic["date"]).dt.year

    horizons = sorted(daily_ic["horizon"].unique())
    records = []
    for (feat, year), grp in daily_ic.groupby(["feature", "year"]):
        row: Dict = {"feature": feat, "year": int(year)}
        for h in horizons:
            sub = grp.loc[grp["horizon"] == h, "ic"]
            n = len(sub)
            if n == 0:
                continue
            mu = float(sub.mean())
            sigma = float(sub.std()) if n > 1 else 1.0
            row[f"ic_mean_h{h}"] = round(mu, 6)
            row[f"ic_ir_h{h}"] = round(mu / sigma * np.sqrt(252) if sigma > 0 else 0.0, 4)
            row[f"n_days_h{h}"] = n
        records.append(row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index(["feature", "year"])


def passes_ic_threshold(
    summary: pd.DataFrame,
    min_ic_mean: float = IC_MEAN_MIN,
    min_ic_ir: float = IC_IR_MIN,
    min_hit_rate: float = HIT_RATE_MIN,
    horizon: int = 5,
) -> Tuple[List[str], List[str]]:
    """Classify features into passing and failing based on IC thresholds.

    Returns:
        (passing_features, failing_features) — feature name lists.
    """
    passing, failing = [], []
    for feat, row in summary.iterrows():
        mu = row.get(f"ic_mean_h{horizon}", 0.0)
        ir = row.get(f"ic_ir_h{horizon}", 0.0)
        hr = row.get(f"hit_rate_h{horizon}", 0.0)
        if (abs(mu) >= min_ic_mean
                and abs(ir) >= min_ic_ir
                and hr >= min_hit_rate):
            passing.append(str(feat))
        else:
            failing.append(str(feat))
    return passing, failing


def format_ic_markdown(
    summary: pd.DataFrame,
    top_n: int = 20,
    horizon: int = 5,
) -> str:
    """Format top/bottom N features as a markdown table for experiment log."""
    sort_col = f"ic_ir_h{horizon}"
    if sort_col not in summary.columns:
        return "_No IC data available._"

    top = summary.nlargest(top_n, sort_col)
    bottom = summary.nsmallest(top_n, sort_col)

    def _table(df: pd.DataFrame, title: str) -> str:
        lines = [f"### {title}", ""]
        lines.append(
            f"| Feature | IC Mean (h={horizon}d) | IC IR | Hit Rate | N Days |"
        )
        lines.append("|---|---|---|---|---|")
        for feat, row in df.iterrows():
            lines.append(
                f"| {feat} "
                f"| {row.get(f'ic_mean_h{horizon}', float('nan')):.4f} "
                f"| {row.get(f'ic_ir_h{horizon}', float('nan')):.3f} "
                f"| {row.get(f'hit_rate_h{horizon}', float('nan')):.3f} "
                f"| {int(row.get(f'n_days_h{horizon}', 0))} |"
            )
        return "\n".join(lines)

    return _table(top, f"Top {top_n} Features by IC IR (h={horizon}d)") + "\n\n" + \
           _table(bottom, f"Bottom {top_n} Features by IC IR (h={horizon}d)")
