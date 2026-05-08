"""Phase R7 — Regime V2 feature builder.

V2 adds: VIX term structure, credit spread (HYG/IEF), sector dispersion,
equal-weight breadth proxy (RSP/SPY), and extended SPY trend features.
All fetches accept pre-loaded price dicts for backfill efficiency.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ordered list — must match XGBoost feature order exactly
REGIME_FEATURE_NAMES = [
    # VIX / realized vol
    "vix_level",
    "vix_pct_1y",
    "vix_pct_60d",
    "vix_5d_change",
    "vix_term_ratio",
    "spy_rvol_5d",
    "spy_rvol_20d",
    # SPY price / trend
    "spy_1d_return",
    "spy_5d_return",
    "spy_20d_return",
    "spy_50d_return",
    "spy_ma20_dist",
    "spy_ma50_dist",
    "spy_ma200_dist",
    "spy_above_ma50",
    "spy_above_ma200",
    # Breadth & credit
    "breadth_rsp_spy_ratio_20d",
    "credit_hyg_ief_5d",
    "credit_hyg_ief_20d",
    # Sector dispersion
    "sector_dispersion_20d",
    "sector_leader_lag_20d",
    # Macro calendar
    "days_to_fomc",
    "days_to_cpi",
    "days_to_nfp",
    "is_fomc_day",
    "is_cpi_day",
    "is_nfp_day",
    # NIS (NULL before May 2025 — XGBoost handles missing)
    "nis_risk_numeric",
    "nis_sizing_factor",
]

# Tickers fetched for V2 features
MULTI_TICKERS = [
    "SPY", "RSP",
    "^VIX", "^VIX3M",
    "HYG", "IEF",
    "XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLC", "XLP", "XLU",
]

SECTOR_TICKERS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLC", "XLP", "XLU"]

_SPY_LOOKBACK_DAYS = 320
_VIX_LOOKBACK_DAYS = 320
_MULTI_LOOKBACK_DAYS = 320


def label_regime_day(row: dict) -> int:
    """
    V2 rule-based 3-class label.

    Returns:
      0 = RISK_OFF     (target ~15-20% of days)
      1 = RISK_CAUTION (target ~25-35%)
      2 = RISK_ON      (target ~45-60%)

    Multi-factor: no single signal dominates. Uses VIX percentile + term
    structure + trend (not single-day return) + credit + breadth.
    """
    vix = row.get("vix_level") or 20.0
    vix_pct1y = row.get("vix_pct_1y") or 0.5
    vix_term = row.get("vix_term_ratio") or 1.0
    ma50_dist = row.get("spy_ma50_dist") or 0.0
    ma200_dist = row.get("spy_ma200_dist") or 0.0
    credit_20d = row.get("credit_hyg_ief_20d") or 0.0
    breadth = row.get("breadth_rsp_spy_ratio_20d") or 0.0
    spy_20d = row.get("spy_20d_return") or 0.0

    # RISK_OFF: any strong hostile signal
    risk_off = (
        vix > 28.0
        or (vix_pct1y > 0.85 and vix_term > 1.05)       # elevated + backwardation
        or (ma50_dist < -0.04 and ma200_dist < 0.0)      # broken below key MAs
        or credit_20d < -0.03                            # credit stress
        or (breadth < -0.03 and spy_20d < -0.05)         # breadth collapse + momentum
    )
    if risk_off:
        return 0

    # RISK_ON: clean favorable tape across all dimensions
    risk_on = (
        vix < 20.0
        and vix_pct1y < 0.70
        and vix_term <= 1.0                              # contango
        and ma50_dist > 0.0
        and ma200_dist > 0.01
        and credit_20d > -0.005
    )
    if risk_on:
        return 2

    return 1   # RISK_CAUTION by elimination


_LABEL_NAMES = {0: "RISK_OFF", 1: "RISK_CAUTION", 2: "RISK_ON"}


def label_name(label_int: int) -> str:
    return _LABEL_NAMES.get(label_int, "UNKNOWN")


class RegimeFeatureBuilder:
    """Builds the regime feature vector for a given date.

    Designed for:
    - Live scoring: build(as_of_date=None) → uses today
    - Backfill: build(as_of_date=d, _prefetched=prices_dict) for speed

    All computations are PIT-correct: only data on/before as_of_date used.
    """

    def __init__(self):
        self._macro_cal = None

    def _get_macro_cal(self):
        if self._macro_cal is None:
            from app.calendars.macro import MacroCalendar
            self._macro_cal = MacroCalendar()
        return self._macro_cal

    def build(
        self,
        as_of_date: Optional[date] = None,
        _spy_df: Optional[pd.DataFrame] = None,
        _vix_df: Optional[pd.Series] = None,
        _prefetched: Optional[dict] = None,
    ) -> dict:
        """Return regime feature dict for as_of_date.

        Pass _prefetched={ticker: DataFrame} for backfill speed (avoids one
        yfinance call per ticker per day).
        """
        if as_of_date is None:
            as_of_date = date.today()

        feats: dict = {k: np.nan for k in REGIME_FEATURE_NAMES}

        if _prefetched is not None:
            spy_df = _slice_to_date(_prefetched.get("SPY"), as_of_date)
            vix_s = _close_series(_slice_to_date(_prefetched.get("^VIX"), as_of_date))
            vix3m_s = _close_series(_slice_to_date(_prefetched.get("^VIX3M"), as_of_date))
            rsp_s = _close_series(_slice_to_date(_prefetched.get("RSP"), as_of_date))
            hyg_s = _close_series(_slice_to_date(_prefetched.get("HYG"), as_of_date))
            ief_s = _close_series(_slice_to_date(_prefetched.get("IEF"), as_of_date))
            sector_map = {
                t: _close_series(_slice_to_date(_prefetched.get(t), as_of_date))
                for t in SECTOR_TICKERS
            }
        else:
            spy_df = _spy_df if _spy_df is not None else self._fetch_spy(as_of_date)
            vix_s = _vix_df if _vix_df is not None else self._fetch_vix(as_of_date)
            vix3m_s = self._fetch_single("^VIX3M", as_of_date)
            rsp_s = self._fetch_single("RSP", as_of_date)
            hyg_s = self._fetch_single("HYG", as_of_date)
            ief_s = self._fetch_single("IEF", as_of_date)
            sector_map = {t: self._fetch_single(t, as_of_date) for t in SECTOR_TICKERS}

        self._add_spy_features(feats, spy_df, as_of_date)
        self._add_vix_features(feats, vix_s, vix3m_s, as_of_date)
        self._add_breadth_features(feats, rsp_s, spy_df)
        self._add_credit_features(feats, hyg_s, ief_s)
        self._add_sector_features(feats, sector_map)
        self._add_macro_calendar_features(feats, as_of_date)
        self._add_nis_features(feats, as_of_date)

        return feats

    @staticmethod
    def fetch_all_prefetched(
        start: date,
        end: date,
        lookback_days: int = _MULTI_LOOKBACK_DAYS,
    ) -> dict:
        """Batch-fetch all V2 tickers. Returns {ticker: DataFrame}.

        Pass the result as _prefetched= to build() for fast per-day iteration.
        """
        import yfinance as yf
        fetch_start = start - timedelta(days=lookback_days)
        fetch_end = end + timedelta(days=1)
        logger.info(
            "Batch-fetching %d regime tickers %s → %s",
            len(MULTI_TICKERS), fetch_start, fetch_end,
        )
        try:
            raw = yf.download(
                MULTI_TICKERS,
                start=fetch_start.isoformat(),
                end=fetch_end.isoformat(),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
        except Exception as exc:
            logger.error("Batch yfinance download failed: %s", exc)
            return {}

        result: dict = {}
        for ticker in MULTI_TICKERS:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw[ticker].copy()
                else:
                    df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if not df.empty and "close" in df.columns:
                    result[ticker] = df
            except Exception:
                pass
        logger.info("Prefetched %d/%d tickers", len(result), len(MULTI_TICKERS))
        return result

    # ── Private fetch helpers ──────────────────────────────────────────────────

    def _fetch_spy(self, as_of_date: date) -> Optional[pd.DataFrame]:
        return _fetch_df("SPY", as_of_date, _SPY_LOOKBACK_DAYS)

    def _fetch_vix(self, as_of_date: date) -> Optional[pd.Series]:
        return _close_series(_fetch_df("^VIX", as_of_date, _VIX_LOOKBACK_DAYS))

    def _fetch_single(self, ticker: str, as_of_date: date) -> Optional[pd.Series]:
        return _close_series(_fetch_df(ticker, as_of_date, _MULTI_LOOKBACK_DAYS))

    # ── Feature builders ──────────────────────────────────────────────────────

    def _add_spy_features(self, feats: dict, df: Optional[pd.DataFrame], as_of_date: date) -> None:
        if df is None or len(df) < 5:
            return
        close = df["close"] if "close" in df.columns else df.iloc[:, 0]
        last = float(close.iloc[-1])

        if len(close) >= 2:
            feats["spy_1d_return"] = float(close.iloc[-1] / close.iloc[-2] - 1.0)
        if len(close) >= 6:
            feats["spy_5d_return"] = float(close.iloc[-1] / close.iloc[-6] - 1.0)
        if len(close) >= 21:
            feats["spy_20d_return"] = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        if len(close) >= 51:
            feats["spy_50d_return"] = float(close.iloc[-1] / close.iloc[-51] - 1.0)

        if len(close) >= 20:
            ma20 = float(close.tail(20).mean())
            feats["spy_ma20_dist"] = (last - ma20) / ma20
        if len(close) >= 50:
            ma50 = float(close.tail(50).mean())
            feats["spy_ma50_dist"] = (last - ma50) / ma50
            feats["spy_above_ma50"] = 1.0 if last >= ma50 else 0.0
        if len(close) >= 200:
            ma200 = float(close.tail(200).mean())
            feats["spy_ma200_dist"] = (last - ma200) / ma200
            feats["spy_above_ma200"] = 1.0 if last >= ma200 else 0.0

        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) >= 5:
            feats["spy_rvol_5d"] = float(log_ret.tail(5).std() * np.sqrt(252) * 100)
        if len(log_ret) >= 20:
            feats["spy_rvol_20d"] = float(log_ret.tail(20).std() * np.sqrt(252) * 100)

    def _add_vix_features(
        self,
        feats: dict,
        vix_s: Optional[pd.Series],
        vix3m_s: Optional[pd.Series],
        as_of_date: date,
    ) -> None:
        if vix_s is None or vix_s.empty:
            return
        vix = float(np.clip(vix_s.iloc[-1], 5.0, 80.0))
        feats["vix_level"] = vix

        if len(vix_s) >= 10:
            feats["vix_pct_60d"] = float((vix_s.tail(60) <= vix).mean())
        if len(vix_s) >= 50:
            feats["vix_pct_1y"] = float((vix_s.tail(252) <= vix).mean())
        if len(vix_s) >= 6:
            feats["vix_5d_change"] = float(vix_s.iloc[-1] / vix_s.iloc[-6] - 1.0)

        if vix3m_s is not None and not vix3m_s.empty:
            vix3m = float(np.clip(vix3m_s.iloc[-1], 5.0, 80.0))
            if vix3m > 0:
                feats["vix_term_ratio"] = round(vix / vix3m, 4)

    def _add_breadth_features(
        self,
        feats: dict,
        rsp_s: Optional[pd.Series],
        spy_df: Optional[pd.DataFrame],
    ) -> None:
        spy_s = _close_series(spy_df)
        if rsp_s is None or rsp_s.empty or spy_s is None or spy_s.empty:
            return
        if len(rsp_s) >= 21 and len(spy_s) >= 21:
            rsp_ret = float(rsp_s.iloc[-1] / rsp_s.iloc[-21] - 1.0)
            spy_ret = float(spy_s.iloc[-1] / spy_s.iloc[-21] - 1.0)
            feats["breadth_rsp_spy_ratio_20d"] = round(rsp_ret - spy_ret, 5)

    def _add_credit_features(
        self,
        feats: dict,
        hyg_s: Optional[pd.Series],
        ief_s: Optional[pd.Series],
    ) -> None:
        if hyg_s is None or hyg_s.empty or ief_s is None or ief_s.empty:
            return
        if len(hyg_s) >= 6 and len(ief_s) >= 6:
            feats["credit_hyg_ief_5d"] = round(
                float(hyg_s.iloc[-1] / hyg_s.iloc[-6] - 1.0)
                - float(ief_s.iloc[-1] / ief_s.iloc[-6] - 1.0), 5
            )
        if len(hyg_s) >= 21 and len(ief_s) >= 21:
            feats["credit_hyg_ief_20d"] = round(
                float(hyg_s.iloc[-1] / hyg_s.iloc[-21] - 1.0)
                - float(ief_s.iloc[-1] / ief_s.iloc[-21] - 1.0), 5
            )

    def _add_sector_features(self, feats: dict, sector_map: dict) -> None:
        rets = []
        for t in SECTOR_TICKERS:
            s = sector_map.get(t)
            if s is not None and not s.empty and len(s) >= 21:
                rets.append(float(s.iloc[-1] / s.iloc[-21] - 1.0))
        if len(rets) >= 4:
            feats["sector_dispersion_20d"] = round(float(np.std(rets)), 5)
            feats["sector_leader_lag_20d"] = round(max(rets) - min(rets), 5)

    def _add_macro_calendar_features(self, feats: dict, as_of: date) -> None:
        try:
            cal = self._get_macro_cal()
            events = sorted(cal._events, key=lambda e: e.date_str)
            as_of_ts = pd.Timestamp(as_of)
            for event_type, feat_days, feat_is in [
                ("FOMC", "days_to_fomc", "is_fomc_day"),
                ("CPI",  "days_to_cpi",  "is_cpi_day"),
                ("NFP",  "days_to_nfp",  "is_nfp_day"),
            ]:
                typed = [e for e in events if e.event_type == event_type]
                future = [e for e in typed if pd.Timestamp(e.date_str) >= as_of_ts]
                if future:
                    days = (pd.Timestamp(future[0].date_str) - as_of_ts).days
                    feats[feat_days] = float(min(days, 30))
                    feats[feat_is] = 1.0 if days == 0 else 0.0
                else:
                    feats[feat_days] = 30.0
                    feats[feat_is] = 0.0
        except Exception as exc:
            logger.warning("Macro calendar features failed: %s", exc)

    def _add_nis_features(self, feats: dict, as_of: date) -> None:
        try:
            from app.database.session import get_session
            from app.database.models import NisMacroSnapshot
            with get_session() as db:
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
        except Exception as exc:
            logger.warning("NIS features failed: %s", exc)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _fetch_df(ticker: str, as_of_date: date, lookback_days: int) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        end = as_of_date + timedelta(days=1)
        start = as_of_date - timedelta(days=lookback_days)
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[df.index.date <= as_of_date]
        return df if not df.empty else None
    except Exception as exc:
        logger.debug("_fetch_df %s failed: %s", ticker, exc)
        return None


def _close_series(df) -> Optional[pd.Series]:
    if df is None:
        return None
    if isinstance(df, pd.Series):
        return df
    if df.empty:
        return None
    return df["close"] if "close" in df.columns else df.iloc[:, 0]


def _slice_to_date(df: Optional[pd.DataFrame], as_of: date) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    mask = pd.to_datetime(df.index).date <= as_of
    sliced = df[mask]
    return sliced if not sliced.empty else None
