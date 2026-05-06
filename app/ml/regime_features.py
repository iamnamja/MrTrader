"""Phase R1 — Regime feature builder.

Computes market-wide (non-cross-sectional) features for any historical date.
These features are intentionally NOT passed through cs_normalize — they describe
the macro environment, not individual stocks.

Usage:
    builder = RegimeFeatureBuilder()
    feats = builder.build(date(2025, 4, 7))  # returns dict of floats
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_FEATURE_NAMES = [
    # VIX / realized vol
    "vix_level",
    "vix_pct_1y",
    "vix_pct_60d",
    "spy_rvol_5d",
    "spy_rvol_20d",
    # SPY price / trend
    "spy_1d_return",
    "spy_5d_return",
    "spy_20d_return",
    "spy_ma20_dist",
    "spy_ma50_dist",
    "spy_ma200_dist",
    # Macro calendar
    "days_to_fomc",
    "days_to_cpi",
    "days_to_nfp",
    "is_fomc_day",
    "is_cpi_day",
    "is_nfp_day",
    # NIS Tier 1 macro (NULL before May 2025 — XGBoost handles missing values)
    "nis_risk_numeric",
    "nis_sizing_factor",
    # Market breadth — Phase R2 (NULL until computed)
    "breadth_pct_ma50",
]

# How many calendar days of SPY/VIX history to fetch
_SPY_LOOKBACK_DAYS = 320   # 252 trading days + buffer
_VIX_LOOKBACK_DAYS = 320


class RegimeFeatureBuilder:
    """Builds the regime feature vector for a given date.

    Designed for:
    - Backfill: `build(as_of_date=date(2023, 1, 5))`
    - Live scoring: `build()` → uses today's data
    - Historical training: loop over dates calling `build(as_of_date=d)`

    All data fetches are PIT-correct: only data available on/before `as_of_date`
    is used. Caches SPY and VIX data frames across calls when building many dates
    in sequence (pass `_spy_df` / `_vix_df` pre-fetched for speed).
    """

    def __init__(self):
        self._macro_cal = None  # lazy init

    def _get_macro_cal(self):
        if self._macro_cal is None:
            from app.calendars.macro import MacroCalendar
            self._macro_cal = MacroCalendar()
        return self._macro_cal

    # ── Public ────────────────────────────────────────────────────────────────

    def build(
        self,
        as_of_date: Optional[date] = None,
        _spy_df: Optional[pd.DataFrame] = None,
        _vix_df: Optional[pd.Series] = None,
    ) -> dict:
        """Return regime feature dict for `as_of_date` (defaults to today).

        Missing values are represented as np.nan so they can be stored as NULL
        in SQLite and handled by XGBoost missing-value logic.
        """
        if as_of_date is None:
            as_of_date = date.today()

        feats: dict = {k: np.nan for k in REGIME_FEATURE_NAMES}

        spy_df = _spy_df if _spy_df is not None else self._fetch_spy(as_of_date)
        vix_s = _vix_df if _vix_df is not None else self._fetch_vix(as_of_date)

        self._add_spy_features(feats, spy_df, as_of_date)
        self._add_vix_features(feats, vix_s, as_of_date)
        self._add_macro_calendar_features(feats, as_of_date)
        self._add_nis_features(feats, as_of_date)
        # breadth_pct_ma50 stays NaN until Phase R2

        return feats

    # ── SPY features ──────────────────────────────────────────────────────────

    def _fetch_spy(self, as_of_date: date) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            end = as_of_date + timedelta(days=1)
            start = as_of_date - timedelta(days=_SPY_LOOKBACK_DAYS)
            df = yf.download(
                "SPY",
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            # Drop any rows after as_of_date (PIT safety)
            df = df[df.index.date <= as_of_date]
            return df if not df.empty else None
        except Exception as exc:
            logger.warning("RegimeFeatureBuilder: SPY fetch failed: %s", exc)
            return None

    def _add_spy_features(
        self, feats: dict, df: Optional[pd.DataFrame], as_of_date: date
    ) -> None:
        if df is None or len(df) < 5:
            return
        close = df["close"]
        last = float(close.iloc[-1])

        # Daily returns
        if len(close) >= 2:
            feats["spy_1d_return"] = float(close.iloc[-1] / close.iloc[-2] - 1.0)
        if len(close) >= 6:
            feats["spy_5d_return"] = float(close.iloc[-1] / close.iloc[-6] - 1.0)
        if len(close) >= 21:
            feats["spy_20d_return"] = float(close.iloc[-1] / close.iloc[-21] - 1.0)

        # MA distances
        if len(close) >= 20:
            ma20 = float(close.tail(20).mean())
            feats["spy_ma20_dist"] = (last - ma20) / ma20
        if len(close) >= 50:
            ma50 = float(close.tail(50).mean())
            feats["spy_ma50_dist"] = (last - ma50) / ma50
        if len(close) >= 200:
            ma200 = float(close.tail(200).mean())
            feats["spy_ma200_dist"] = (last - ma200) / ma200

        # Realized vol (annualized)
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) >= 5:
            feats["spy_rvol_5d"] = float(log_ret.tail(5).std() * np.sqrt(252) * 100)
        if len(log_ret) >= 20:
            feats["spy_rvol_20d"] = float(log_ret.tail(20).std() * np.sqrt(252) * 100)

    # ── VIX features ──────────────────────────────────────────────────────────

    def _fetch_vix(self, as_of_date: date) -> Optional[pd.Series]:
        try:
            import yfinance as yf
            end = as_of_date + timedelta(days=1)
            start = as_of_date - timedelta(days=_VIX_LOOKBACK_DAYS)
            df = yf.download(
                "^VIX",
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            s = df["close"]
            s = s[s.index.date <= as_of_date]
            return s if not s.empty else None
        except Exception as exc:
            logger.warning("RegimeFeatureBuilder: VIX fetch failed: %s", exc)
            return None

    def _add_vix_features(
        self, feats: dict, vix_s: Optional[pd.Series], as_of_date: date
    ) -> None:
        if vix_s is None or vix_s.empty:
            return
        vix = float(np.clip(vix_s.iloc[-1], 5.0, 80.0))
        feats["vix_level"] = vix

        if len(vix_s) >= 10:
            # 60-day percentile
            hist60 = vix_s.tail(60)
            feats["vix_pct_60d"] = float((hist60 <= vix).mean())

        if len(vix_s) >= 50:
            # 1-year percentile (252 trading days)
            hist1y = vix_s.tail(252)
            feats["vix_pct_1y"] = float((hist1y <= vix).mean())

    # ── Macro calendar ────────────────────────────────────────────────────────

    def _add_macro_calendar_features(self, feats: dict, as_of: date) -> None:
        try:
            cal = self._get_macro_cal()
            # Get all events sorted by date
            events = sorted(cal._events, key=lambda e: e.date_str)  # list[MacroEvent]

            as_of_ts = pd.Timestamp(as_of)

            for event_type, feat_days, feat_is in [
                ("FOMC", "days_to_fomc", "is_fomc_day"),
                ("CPI",  "days_to_cpi",  "is_cpi_day"),
                ("NFP",  "days_to_nfp",  "is_nfp_day"),
            ]:
                typed = [e for e in events if e.event_type == event_type]
                # Find next event on or after as_of
                future = [
                    e for e in typed
                    if pd.Timestamp(e.date_str) >= as_of_ts
                ]
                if future:
                    next_evt = future[0]
                    days = (pd.Timestamp(next_evt.date_str) - as_of_ts).days
                    feats[feat_days] = float(min(days, 30))
                    feats[feat_is] = 1.0 if days == 0 else 0.0
                else:
                    feats[feat_days] = 30.0
                    feats[feat_is] = 0.0
        except Exception as exc:
            logger.warning("RegimeFeatureBuilder: macro calendar failed: %s", exc)

    # ── NIS features ──────────────────────────────────────────────────────────

    def _add_nis_features(self, feats: dict, as_of: date) -> None:
        try:
            from app.database.session import get_session
            from app.database.models import NisMacroSnapshot
            db = get_session()
            try:
                row = (
                    db.query(NisMacroSnapshot)
                    .filter(NisMacroSnapshot.snapshot_date <= as_of)
                    .order_by(NisMacroSnapshot.snapshot_date.desc())
                    .first()
                )
                if row is not None:
                    risk_map = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
                    feats["nis_risk_numeric"] = risk_map.get(
                        (row.overall_risk or "MEDIUM").upper(), 0.5
                    )
                    feats["nis_sizing_factor"] = float(row.global_sizing_factor or 1.0)
                # else: stays NaN — expected for dates before May 2025
            finally:
                db.close()
        except Exception as exc:
            logger.warning("RegimeFeatureBuilder: NIS fetch failed: %s", exc)
