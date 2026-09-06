"""Phase R7 — Regime V2 feature builder.

V2 adds: VIX term structure, credit spread (HYG/IEF), sector dispersion,
equal-weight breadth proxy (RSP/SPY), and extended SPY trend features.
All fetches accept pre-loaded price dicts for backfill efficiency.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache
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

# How old a VIX3M close may be and still describe `as_of_date`. 5 calendar days covers a
# three-day weekend plus a holiday; anything older is a data outage, not a settled close.
#
# WHY THIS EXISTS (2026-09-06). `_slice_to_date(...)` then `.iloc[-1]` returns the last
# available row REGARDLESS of its age, so a series that stops updating silently pins the
# feature to its final value forever. yfinance's ^VIX3M is doing exactly that — its history
# rots backwards over time (macro_history.py documents the decay: 100% coverage through
# Apr 2026 -> 5% by Aug). Measured on 2026-09-06, `build(as_of_date=2026-09-04)` returned
# vix_term_ratio=0.7074, computed from the 2026-07-17 VIX3M close (20.54) against a
# 2026-09-04 VIX — a 7-week-stale denominator. The FRED-backed value is 14.53/17.61 = 0.825.
# Silent because nothing errored: the number was present, plausible, and wrong.
#
# macro_history carries the same two series with a FRED backstop (DECISIONS 2026-08-xx / #674,
# which fixed this same outage for the crash governor but not for this second consumer).
_MAX_VIX3M_STALENESS_DAYS = 5


def _coalesce(row, key: str, default: float) -> float:
    """`row[key]`, treating BOTH None and NaN as missing.

    `row.get(k) or default` — the idiom this replaces — is NaN-blind, because NaN is
    truthy: `float('nan') or 1.0` is NaN, not 1.0. Every subsequent comparison against
    that NaN is then False, which silently rewrites the rule. Concretely, a NULL
    `vix_term_ratio` made the RISK_ON contango test (`vix_term <= 1.0`) unsatisfiable
    and the RISK_OFF backwardation test (`vix_term > 1.05`) unreachable, so an
    outage day was force-labelled RISK_CAUTION instead of taking the 1.0 default.
    Harmless while the feed was always populated; a live mislabeller the moment
    `_vix3m_as_of` started (correctly) returning None on a dead feed.

    Also coerces to float so a Decimal/str from the DB cannot compare oddly.
    """
    val = row.get(key)
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return default if f != f else f      # f != f  <=>  f is NaN


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
    vix = _coalesce(row, "vix_level", 20.0)
    vix_pct1y = _coalesce(row, "vix_pct_1y", 0.5)
    vix_term = _coalesce(row, "vix_term_ratio", 1.0)
    ma50_dist = _coalesce(row, "spy_ma50_dist", 0.0)
    ma200_dist = _coalesce(row, "spy_ma200_dist", 0.0)
    credit_20d = _coalesce(row, "credit_hyg_ief_20d", 0.0)
    breadth = _coalesce(row, "breadth_rsp_spy_ratio_20d", 0.0)
    spy_20d = _coalesce(row, "spy_20d_return", 0.0)

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

        # SPY needs the same staleness discipline as VIX, for the same reason: `.iloc[-1]`
        # on a slice pins to the last available bar however old it is, and
        # spy_ma200_dist / spy_20d_return are BOTH in load_dataset's core set AND drivers
        # of label_regime_day. A frozen SPY feed would quietly hold the trend features at
        # their last value — the identical failure this change fixed for VIX3M, and the
        # one the VIX3M fix would otherwise have left one ticker away.
        close = _freshest_vol_series(close, "spy", as_of_date)
        if close is None or len(close) == 0:
            return
        close = close.dropna()
        if len(close) < 5:
            return
        if not _is_fresh(close, as_of_date):
            logger.warning(
                "No SPY close within %d days of %s (last %s) — SPY features left NULL",
                _MAX_VIX3M_STALENESS_DAYS, as_of_date,
                pd.to_datetime(close.index[-1]).date(),
            )
            return

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
        # The NUMERATOR needs the same freshness guarantee as the denominator: pairing a
        # weeks-old VIX with a current VIX3M is the same silent-wrong-number failure,
        # just relocated. Fall back to the FRED-backed macro series before giving up.
        vix_s = _freshest_vol_series(vix_s, "vix", as_of_date)

        if vix_s is None or len(vix_s) == 0:
            return
        vix_s = vix_s.dropna()
        if vix_s.empty:
            return
        if not _is_fresh(vix_s, as_of_date):
            logger.warning(
                "No VIX close within %d days of %s (last %s) — VIX features left NULL",
                _MAX_VIX3M_STALENESS_DAYS, as_of_date,
                pd.to_datetime(vix_s.index[-1]).date(),
            )
            return

        vix = float(np.clip(vix_s.iloc[-1], 5.0, 80.0))
        feats["vix_level"] = vix

        if len(vix_s) >= 10:
            feats["vix_pct_60d"] = float((vix_s.tail(60) <= vix).mean())
        if len(vix_s) >= 50:
            feats["vix_pct_1y"] = float((vix_s.tail(252) <= vix).mean())
        if len(vix_s) >= 6:
            feats["vix_5d_change"] = float(vix_s.iloc[-1] / vix_s.iloc[-6] - 1.0)

        # Paired by DATE — not two independent "within 5 days" lookups. See
        # _vix_term_ratio_as_of: the denominator is normally FRED (a business day behind)
        # while the numerator is today's yfinance close, so unpaired resolution divides
        # closes from different days on exactly the volatile days that matter.
        ratio = _vix_term_ratio_as_of(vix_s, vix3m_s, as_of_date)
        if ratio is not None:
            feats["vix_term_ratio"] = ratio

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


@lru_cache(maxsize=4)
def _macro_series_cached(field: str, _mtime: float) -> Optional[pd.Series]:
    """Date-indexed close series for `field` from the macro-history parquet.

    Keyed on the file's mtime rather than plain-cached so a long-running process picks
    up the startup/daily macro refresh — a permanently-cached series would re-create the
    very staleness bug this fallback exists to fix.
    """
    try:
        from app.data.macro_history import load_macro_history

        df = load_macro_history()
        if df is None or df.empty or field not in df.columns:
            return None
        sub = df.dropna(subset=[field])
        if sub.empty:
            return None
        return pd.Series(
            sub[field].astype(float).to_numpy(),
            index=pd.to_datetime(sub["date"]),
            name="close",
        )
    except Exception as exc:  # never let a fallback take down feature building
        logger.warning("macro_history '%s' fallback unavailable: %s", field, exc)
        return None


def _macro_series(field: str) -> Optional[pd.Series]:
    try:
        from app.data.macro_history import MACRO_PATH

        mtime = MACRO_PATH.stat().st_mtime if MACRO_PATH.exists() else 0.0
    except Exception:
        mtime = 0.0
    return _macro_series_cached(field, mtime)


def _macro_vix3m_map() -> dict:
    """{'YYYY-MM-DD': vix3m}. Thin view over _macro_series for point lookups."""
    s = _macro_series("vix3m")
    if s is None:
        return {}
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items()}


def _is_fresh(series: Optional[pd.Series], as_of_date: date) -> bool:
    """True when `series` carries a non-NaN value dated (as_of - bound, as_of]."""
    if series is None or len(series) == 0:
        return False
    fresh = series.dropna()
    if fresh.empty:
        return False
    last_dt = pd.to_datetime(fresh.index[-1]).date()
    return (as_of_date - timedelta(days=_MAX_VIX3M_STALENESS_DAYS)) <= last_dt <= as_of_date


def _freshest_vol_series(
    series: Optional[pd.Series], field: str, as_of_date: date
) -> Optional[pd.Series]:
    """`series` if it is fresh for as_of_date, else the FRED-backed macro_history series.

    Applies to the VIX NUMERATOR as well as the VIX3M denominator. Guarding only the
    denominator would still let the ratio pair a weeks-old VIX against a current VIX3M —
    the same silent-wrong-number failure, merely relocated. Returns a series (not a
    scalar) because the percentile and 5-day-change features need the whole window.
    """
    if _is_fresh(series, as_of_date):
        return series

    # The fallback fires for None/empty too, not only for stale-but-present. An earlier
    # cut restricted it to sources that had returned rows, reasoning that a caller passing
    # nothing should not have global data substituted behind it. That was wrong on the
    # facts: `_fetch_single`/`_fetch_spy` return None on ANY yfinance failure, so the
    # restriction disabled the backstop in exactly the outage it exists for — blanking the
    # whole VIX block while a good FRED value sat in the parquet. There is no look-ahead
    # risk to trade off, because the fallback is sliced by as_of_date below.
    fallback = _macro_series(field)
    if fallback is None:
        return series      # nothing better available; caller's staleness checks apply

    sliced = fallback[pd.to_datetime(fallback.index).date <= as_of_date]
    if sliced.empty:
        return series
    if series is not None and len(series.dropna()) and not _is_fresh(series, as_of_date):
        logger.debug("%s stale at %s — falling back to macro_history", field, as_of_date)
    return sliced


def _vix3m_as_of(vix3m_s: Optional[pd.Series], as_of_date: date) -> Optional[float]:
    """VIX3M close describing `as_of_date`, or None when no fresh value exists.

    Order: yfinance (intraday-fresh when it works) -> macro_history (FRED-backed).
    Both are subject to the same staleness bound, so a dead feed yields None — and a
    NULL feature, which XGBoost handles natively — rather than a stale number that
    looks real. Returning None here is the point: the prior code could not tell the
    difference between "VIX3M is 20.54 today" and "VIX3M was 20.54 seven weeks ago".
    """
    oldest_ok = as_of_date - timedelta(days=_MAX_VIX3M_STALENESS_DAYS)

    if vix3m_s is not None and not vix3m_s.empty:
        fresh = vix3m_s.dropna()
        if not fresh.empty:
            last_dt = pd.to_datetime(fresh.index[-1]).date()
            if oldest_ok <= last_dt <= as_of_date:
                return float(fresh.iloc[-1])

    macro = _macro_vix3m_map()
    if macro:
        probe = as_of_date
        while probe >= oldest_ok:
            hit = macro.get(probe.isoformat())
            if hit is not None:
                return float(hit)
            probe -= timedelta(days=1)

    logger.debug("No VIX3M close within %d days of %s — vix_term_ratio left NULL",
                 _MAX_VIX3M_STALENESS_DAYS, as_of_date)
    return None


def _series_as_of_map(series: Optional[pd.Series]) -> dict:
    """{'YYYY-MM-DD': value} for a date-indexed close series; {} for None/empty."""
    if series is None or len(series) == 0:
        return {}
    fresh = series.dropna()
    if fresh.empty:
        return {}
    return {pd.to_datetime(d).strftime("%Y-%m-%d"): float(v) for d, v in fresh.items()}


def _vix_term_ratio_as_of(
    vix_s: Optional[pd.Series],
    vix3m_s: Optional[pd.Series],
    as_of_date: date,
) -> Optional[float]:
    """VIX / VIX3M for the most recent date where BOTH are available, or None.

    THE TWO LEGS MUST COME FROM THE SAME DATE. Resolving them through independent
    lookups — each merely "within 5 days of as_of" — routinely divides closes from
    different days AND different sources, and that is the DEFAULT live path, not an edge
    case: yfinance's ^VIX3M is ~95% NaN, so the denominator comes from FRED, which
    publishes with roughly a one-business-day lag, while the numerator is today's
    yfinance close. Today's VIX over yesterday's VIX3M.

    That is not a rounding difference on a vol spike. VIX 15 -> 25 against a prior-day
    VIX3M of 16 reads 1.56 rather than the true ~1.32 — across the 1.05 backwardation
    threshold, flipping the rule label to RISK_OFF on exactly the days the feature is
    supposed to be trusted. macro_history's crash governor already requires both series
    on the SAME settled date (#674); this consumer now does too.

    Sources are still tried yfinance-first per leg, but the PAIRING is by date: we walk
    back from as_of_date and take the first day that has both.
    """
    oldest_ok = as_of_date - timedelta(days=_MAX_VIX3M_STALENESS_DAYS)

    vix_map = dict(_macro_series_map("vix"))
    vix_map.update(_series_as_of_map(vix_s))          # yfinance wins per date
    v3_map = dict(_macro_series_map("vix3m"))
    v3_map.update(_series_as_of_map(vix3m_s))

    probe = as_of_date
    while probe >= oldest_ok:
        key = probe.isoformat()
        v, v3 = vix_map.get(key), v3_map.get(key)
        if v is not None and v3 is not None:
            v = float(np.clip(v, 5.0, 80.0))
            v3 = float(np.clip(v3, 5.0, 80.0))
            if v3 > 0:
                return round(v / v3, 4)
        probe -= timedelta(days=1)

    logger.debug("No date within %d days of %s carries BOTH VIX and VIX3M — "
                 "vix_term_ratio left NULL", _MAX_VIX3M_STALENESS_DAYS, as_of_date)
    return None


def _macro_series_map(field: str) -> dict:
    s = _macro_series(field)
    return {} if s is None else _series_as_of_map(s)


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
