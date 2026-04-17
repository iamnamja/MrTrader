"""
Polygon S3 flat-file downloader.

Polygon stores complete market history as gzipped CSV files on S3:
  us_stocks_sip/day_aggs_v1/{year}/{month}/{YYYY-MM-DD}.csv.gz   — daily bars
  us_stocks_sip/minute_aggs_v1/{year}/{month}/{YYYY-MM-DD}.csv.gz — 1-min bars

Each file contains ALL traded symbols for that day, so one download gives
data for every SP100 stock at once — far faster than per-symbol REST calls.
"""

import gzip
import io
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Polygon S3 column names → our standard names
_DAY_COLS = {
    "ticker": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "window_start": "timestamp",
}

_MIN_COLS = {
    "ticker": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "window_start": "timestamp",
}


class PolygonS3:
    """
    Thin wrapper around boto3 for Polygon flat-file access.
    Downloads, decompresses, and parses CSV.gz files into DataFrames.
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        endpoint_url: str,
        bucket: str,
    ):
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_daily_bulk(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download daily OHLCV for all *symbols* between *start* and *end*.
        Returns {symbol: DataFrame} with DatetimeIndex and lowercase columns.
        """
        symbol_set = set(s.upper() for s in symbols)
        frames: Dict[str, List[pd.DataFrame]] = {s: [] for s in symbol_set}

        for day_date in _business_days(start, end):
            df = self._fetch_day_file("day_aggs_v1", day_date)
            if df is None:
                continue
            df_filtered = df[df["symbol"].isin(symbol_set)]
            for sym, grp in df_filtered.groupby("symbol"):
                frames[sym].append(grp)

        result = {}
        for sym, parts in frames.items():
            if not parts:
                continue
            combined = pd.concat(parts, ignore_index=True)
            combined = _to_daily_df(combined)
            if combined is not None and len(combined) > 0:
                result[sym] = combined

        logger.info(
            "S3 daily bulk: %d/%d symbols returned for %s→%s",
            len(result), len(symbols), start, end,
        )
        return result

    def get_minute_bulk(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download 1-minute OHLCV for all *symbols* between *start* and *end*.
        Returns {symbol: DataFrame} with UTC DatetimeIndex.
        """
        symbol_set = set(s.upper() for s in symbols)
        frames: Dict[str, List[pd.DataFrame]] = {s: [] for s in symbol_set}

        for day_date in _business_days(start, end):
            df = self._fetch_day_file("minute_aggs_v1", day_date)
            if df is None:
                continue
            df_filtered = df[df["symbol"].isin(symbol_set)]
            for sym, grp in df_filtered.groupby("symbol"):
                frames[sym].append(grp)

        result = {}
        for sym, parts in frames.items():
            if not parts:
                continue
            combined = pd.concat(parts, ignore_index=True)
            combined = _to_intraday_df(combined)
            if combined is not None and len(combined) > 0:
                result[sym] = combined

        logger.info(
            "S3 minute bulk: %d/%d symbols returned for %s→%s",
            len(result), len(symbols), start, end,
        )
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_day_file(self, prefix: str, day: date) -> Optional[pd.DataFrame]:
        """Download and parse one CSV.gz file for *day*."""
        key = f"us_stocks_sip/{prefix}/{day.year}/{day.month:02d}/{day}.csv.gz"
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            raw = resp["Body"].read()
            with gzip.open(io.BytesIO(raw), "rt") as f:
                df = pd.read_csv(f)
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            rename = {}
            if "ticker" in df.columns:
                rename["ticker"] = "symbol"
            if "window_start" in df.columns:
                rename["window_start"] = "timestamp"
            if rename:
                df = df.rename(columns=rename)
            return df
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("NoSuchKey", "404"):
                logger.debug("S3 file not found: %s", key)
            else:
                logger.warning("S3 error fetching %s: %s", key, exc)
            return None
        except Exception as exc:
            logger.warning("Failed to parse S3 file %s: %s", key, exc)
            return None

    def health_check(self) -> bool:
        """Verify S3 connectivity."""
        try:
            self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix="us_stocks_sip/day_aggs_v1/",
                MaxKeys=1,
            )
            return True
        except Exception:
            return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _business_days(start: date, end: date) -> List[date]:
    """Return list of Mon–Fri dates between start and end inclusive."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _to_daily_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Convert raw S3 daily rows to a standard DatetimeIndex OHLCV DataFrame."""
    try:
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            return None
        if "timestamp" in df.columns:
            df = df.copy()
            df.index = pd.to_datetime(df["timestamp"], unit="ns", utc=True).dt.normalize()
            df.index = df.index.tz_localize(None)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as exc:
        logger.debug("_to_daily_df failed: %s", exc)
        return None


def _to_intraday_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Convert raw S3 minute rows to a UTC DatetimeIndex OHLCV DataFrame."""
    try:
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            return None
        df = df.copy()
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as exc:
        logger.debug("_to_intraday_df failed: %s", exc)
        return None
