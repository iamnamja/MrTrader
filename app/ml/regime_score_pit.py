"""
P1 — Point-in-time regime score computation.

Single source of truth for computing a reproducible, PIT-safe composite
regime score from macro_history.parquet. Used for:
  1. Training row filter: only train on windows where regime >= threshold
  2. BenignGate: block inference when current regime < threshold
  3. DB persistence: daily_regime_scores table

Design principles:
- No look-ahead: all rolling windows use only past data (closed prices through window_end_date)
- No FRED API calls: uses only what's in macro_history.parquet (already persisted)
- Deterministic: same input always produces same score
- Five equal-weight binary components → composite in [0.0, 1.0]
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MACRO_PARQUET = Path("data/macro/macro_history.parquet")
MACRO_STALENESS_DAYS = 3  # fail-closed if parquet older than this


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_pit_regime_series(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a daily PIT regime score from macro history.

    Input columns required: date (or DatetimeIndex), vix, vix3m, hyg, ief, rsp, spy

    Returns DataFrame with columns:
        date, spy_above_ma50, spy_above_ma200, vix_term_ratio,
        breadth_20d_change, credit_20d_change, composite_score

    Each component is binary [0,1]; composite is their equal-weighted mean.
    """
    df = macro_df.copy()

    # Normalise index
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = {"vix", "vix3m", "hyg", "ief", "rsp", "spy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"macro_df missing columns: {missing}")

    spy = df["spy"].astype(float)
    vix = df["vix"].astype(float)
    vix3m = df["vix3m"].astype(float)
    rsp = df["rsp"].astype(float)
    hyg = df["hyg"].astype(float)
    ief = df["ief"].astype(float)

    # Component 1 & 2: SPY above 50-day and 200-day MA
    spy_ma50 = spy.rolling(50, min_periods=40).mean()
    spy_ma200 = spy.rolling(200, min_periods=150).mean()
    above_ma50 = (spy > spy_ma50).astype(float)
    above_ma200 = (spy > spy_ma200).astype(float)

    # Component 3: VIX term structure — vix3m/vix >= 1.0 = contango = calm
    vix_term_ratio = vix3m / vix.replace(0, np.nan)
    vix_term_ok = (vix_term_ratio >= 1.0).astype(float)

    # Component 4: Breadth — RSP outperforming SPY over 20d (equal-weight beats cap-weight)
    breadth_20d = rsp.pct_change(20) - spy.pct_change(20)
    breadth_ok = (breadth_20d > 0).astype(float)

    # Component 5: Credit — HYG outperforming IEF over 20d (risk-on credit signal)
    credit_20d = hyg.pct_change(20) - ief.pct_change(20)
    credit_ok = (credit_20d > 0).astype(float)

    # Equal-weighted composite
    components = pd.concat(
        [above_ma50, above_ma200, vix_term_ok, breadth_ok, credit_ok], axis=1
    )
    composite = components.mean(axis=1)

    result = pd.DataFrame({
        "spy_above_ma50": above_ma50,
        "spy_above_ma200": above_ma200,
        "vix_term_ratio": vix_term_ratio.round(4),
        "breadth_20d_change": breadth_20d.round(6),
        "credit_20d_change": credit_20d.round(6),
        "composite_score": composite.round(4),
    }, index=df.index)

    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# Score map for training filter
# ---------------------------------------------------------------------------

def build_regime_score_map(
    macro_parquet: Path = MACRO_PARQUET,
) -> dict[date, float]:
    """
    Load macro_history.parquet and return {date: composite_score} dict.
    Used by training workers to look up regime score per window-end date.
    Returns empty dict (fail-open for training) if parquet is missing.
    """
    if not macro_parquet.exists():
        logger.warning("regime_score_pit: %s not found — benign filter disabled", macro_parquet)
        return {}

    try:
        df = pd.read_parquet(macro_parquet)
        series = compute_pit_regime_series(df)
        return {ts.date(): float(score) for ts, score in series["composite_score"].items()
                if not np.isnan(score)}
    except Exception as exc:
        logger.error("regime_score_pit: failed to build score map — %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Current-day score for inference gate
# ---------------------------------------------------------------------------

def get_current_regime_score(
    macro_parquet: Path = MACRO_PARQUET,
    staleness_days: int = MACRO_STALENESS_DAYS,
) -> tuple[float, dict]:
    """
    Return (composite_score, components_dict) for the most recent trading day.

    Fails closed (returns 0.0) if:
    - Parquet not found
    - Most recent row is older than staleness_days business days
    - Any computation error

    This is the inference-time entry point for BenignGate.
    """
    if not macro_parquet.exists():
        logger.error("regime_score_pit: macro parquet not found — failing closed")
        return 0.0, {"error": "parquet_not_found"}

    try:
        df = pd.read_parquet(macro_parquet)

        # Staleness check
        if "date" in df.columns:
            max_date = pd.to_datetime(df["date"]).max().date()
        else:
            max_date = pd.to_datetime(df.index).max().date()

        today = date.today()
        # Allow for weekends: count business days
        cal_days_old = (today - max_date).days
        if cal_days_old > staleness_days * 2:  # generous: 2x to account for weekends
            logger.error(
                "regime_score_pit: macro data is %d days old (max=%s) — failing closed",
                cal_days_old, max_date,
            )
            return 0.0, {"error": f"stale_data_age_{cal_days_old}d", "max_date": str(max_date)}

        series = compute_pit_regime_series(df)
        row = series.iloc[-1]

        components = {
            "spy_above_ma50": float(row["spy_above_ma50"]),
            "spy_above_ma200": float(row["spy_above_ma200"]),
            "vix_term_ratio": float(row["vix_term_ratio"]),
            "breadth_20d_change": float(row["breadth_20d_change"]),
            "credit_20d_change": float(row["credit_20d_change"]),
            "data_date": str(series.index[-1].date()),
        }
        score = float(row["composite_score"])
        return score, components

    except Exception as exc:
        logger.error("regime_score_pit: computation error — %s", exc)
        return 0.0, {"error": str(exc)}
