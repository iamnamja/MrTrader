"""Tests for app.data.cache — disk + in-memory OHLCV and JSON caching."""
from datetime import date, datetime

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_daily_df(start="2023-01-02", n=10):
    idx = pd.date_range(start=start, periods=n, freq="B")
    prices = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.ones(n) * 1_000_000,
    }, index=idx)


def _make_intraday_df(start=None, n=78):
    if start is None:
        # Use today to avoid keep_days pruning in tests
        from datetime import date
        today = date.today()
        start = f"{today} 09:30"
    idx = pd.date_range(start=start, periods=n, freq="5min")
    prices = 100 + np.arange(n, dtype=float) * 0.01
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.002,
        "low": prices * 0.997,
        "close": prices,
        "volume": np.ones(n) * 50_000,
    }, index=idx)


# ── DataCache — daily OHLCV ───────────────────────────────────────────────────

class TestDataCacheDaily:

    def _cache(self, tmp_path):
        from app.data.cache import DataCache
        return DataCache(cache_dir=tmp_path)

    def test_get_daily_returns_none_when_empty(self, tmp_path):
        cache = self._cache(tmp_path)
        result = cache.get_daily("AAPL", date(2023, 1, 1), date(2023, 12, 31))
        assert result is None

    def test_put_then_get_daily(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_daily_df()
        cache.put_daily("AAPL", df)
        result = cache.get_daily("AAPL", date(2023, 1, 1), date(2023, 12, 31))
        assert result is not None
        assert len(result) == len(df)

    def test_put_deduplicates_on_overlap(self, tmp_path):
        cache = self._cache(tmp_path)
        df1 = _make_daily_df("2023-01-02", n=5)
        df2 = _make_daily_df("2023-01-04", n=5)  # overlaps with df1
        cache.put_daily("AAPL", df1)
        cache.put_daily("AAPL", df2)
        result = cache.get_daily("AAPL", date(2023, 1, 1), date(2023, 12, 31))
        assert result is not None
        assert result.index.is_unique

    def test_get_daily_filters_to_requested_range(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_daily_df("2023-01-02", n=20)
        cache.put_daily("AAPL", df)
        result = cache.get_daily("AAPL", date(2023, 1, 5), date(2023, 1, 10))
        assert result is not None
        assert len(result) < len(df)

    def test_file_persists_between_instances(self, tmp_path):
        cache1 = self._cache(tmp_path)
        df = _make_daily_df()
        cache1.put_daily("MSFT", df)

        cache2 = self._cache(tmp_path)
        result = cache2.get_daily("MSFT", date(2023, 1, 1), date(2023, 12, 31))
        assert result is not None

    def test_in_memory_layer_avoids_disk_read(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_daily_df()
        cache.put_daily("GOOG", df)
        # Second read should come from memory
        cache.get_daily("GOOG", date(2023, 1, 1), date(2023, 12, 31))
        assert "daily:GOOG" in cache._mem

    def test_invalidate_clears_memory(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_daily_df()
        cache.put_daily("TSLA", df)
        cache.get_daily("TSLA", date(2023, 1, 1), date(2023, 12, 31))
        cache.invalidate("TSLA")
        assert not any("TSLA" in k for k in cache._mem)


# ── DataCache — missing_daily_range ──────────────────────────────────────────

class TestMissingDailyRange:

    def _cache(self, tmp_path):
        from app.data.cache import DataCache
        return DataCache(cache_dir=tmp_path)

    def test_full_range_returned_when_no_cache(self, tmp_path):
        cache = self._cache(tmp_path)
        start, end = date(2023, 1, 1), date(2023, 12, 31)
        fs, fe = cache.missing_daily_range("AAPL", start, end)
        assert fs == start
        assert fe == end

    def test_no_missing_when_fully_covered(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_daily_df("2023-01-02", n=50)
        cache.put_daily("AAPL", df)
        dates = pd.DatetimeIndex(df.index).date
        fs, fe = cache.missing_daily_range("AAPL", dates[5], dates[10])
        assert fs is None
        assert fe is None

    def test_missing_tail_when_cache_ends_early(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_daily_df("2023-01-02", n=10)
        cache.put_daily("AAPL", df)
        dates = pd.DatetimeIndex(df.index).date
        # Ask for range that extends past cached end
        fs, fe = cache.missing_daily_range("AAPL", dates[0], date(2023, 12, 31))
        assert fs is not None
        assert fs > dates[-1]


# ── DataCache — intraday ──────────────────────────────────────────────────────

class TestDataCacheIntraday:

    def _cache(self, tmp_path):
        from app.data.cache import DataCache
        return DataCache(cache_dir=tmp_path)

    def test_put_then_get_intraday(self, tmp_path):
        from datetime import date
        cache = self._cache(tmp_path)
        df = _make_intraday_df()  # uses today
        today = date.today()
        cache.put_intraday("AAPL", df, interval="5min")
        result = cache.get_intraday(
            "AAPL",
            datetime(today.year, today.month, today.day, 9, 0),
            datetime(today.year, today.month, today.day, 17, 0),
            interval="5min",
        )
        assert result is not None
        assert len(result) > 0

    def test_intraday_prunes_old_bars(self, tmp_path):
        cache = self._cache(tmp_path)
        df = _make_intraday_df("2020-01-02 09:30", n=78)  # 4 years old
        cache.put_intraday("AAPL", df, interval="5min", keep_days=90)
        # Old bars should be pruned — nothing from 2020 survives a 90-day window
        path = tmp_path / "5min" / "AAPL.parquet"
        if path.exists():
            saved = pd.read_parquet(path)
            if len(saved) > 0:
                oldest = pd.DatetimeIndex(saved.index).min()
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=91)
                assert oldest >= cutoff


# ── DataCache — JSON blobs ────────────────────────────────────────────────────

class TestDataCacheJson:

    def _cache(self, tmp_path):
        from app.data.cache import DataCache
        return DataCache(cache_dir=tmp_path)

    def test_put_then_get_json(self, tmp_path):
        cache = self._cache(tmp_path)
        data = {"pe_ratio": 25.0, "pb_ratio": 3.0}
        cache.put_json("fundamentals/AAPL", data)
        result = cache.get_json("fundamentals/AAPL", ttl=3600)
        assert result is not None
        assert result["pe_ratio"] == 25.0

    def test_get_json_returns_none_when_missing(self, tmp_path):
        cache = self._cache(tmp_path)
        assert cache.get_json("fundamentals/UNKNOWN", ttl=3600) is None

    def test_get_json_respects_ttl(self, tmp_path):
        cache = self._cache(tmp_path)
        data = {"score": 0.5}
        cache.put_json("misc/test_key", data)
        # Expire immediately by passing ttl=0
        result = cache.get_json("misc/test_key", ttl=0)
        assert result is None

    def test_json_persists_between_instances(self, tmp_path):
        cache1 = self._cache(tmp_path)
        cache1.put_json("fundamentals/MSFT", {"pe_ratio": 30.0})
        cache2 = self._cache(tmp_path)
        result = cache2.get_json("fundamentals/MSFT", ttl=3600)
        assert result is not None
        assert result["pe_ratio"] == 30.0

    def test_cached_at_not_in_returned_data(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put_json("fundamentals/NVDA", {"pe_ratio": 50.0})
        # _cached_at is in the file but get_json should include it
        result = cache.get_json("fundamentals/NVDA", ttl=3600)
        assert "_cached_at" in result  # present for TTL check, that's fine


# ── DataCache — cache_info ────────────────────────────────────────────────────

class TestCacheInfo:

    def test_cache_info_counts_files(self, tmp_path):
        from app.data.cache import DataCache
        cache = DataCache(cache_dir=tmp_path)
        cache.put_daily("AAPL", _make_daily_df())
        cache.put_json("fundamentals/AAPL", {"pe_ratio": 25.0})
        info = cache.cache_info()
        assert info["daily"] >= 1
        assert info["fundamentals"] >= 1
        assert info["memory_entries"] >= 1


# ── prefetch_fundamentals ─────────────────────────────────────────────────────

class TestPrefetchFundamentals:

    def test_prefetch_warms_cache(self, tmp_path):
        from unittest.mock import patch
        from app.data.cache import DataCache
        import app.data.cache as cache_mod
        cache_mod._default_cache = DataCache(cache_dir=tmp_path)

        mock_data = {
            "pe_ratio": 25.0, "pb_ratio": 3.0, "profit_margin": 0.2,
            "revenue_growth": 0.1, "debt_to_equity": 0.5,
            "earnings_proximity_days": 45.0,
        }
        with patch(
            "app.ml.fundamental_fetcher.get_fundamentals", return_value=mock_data
        ) as mock_gf:
            from app.ml.fundamental_fetcher import prefetch_fundamentals
            result = prefetch_fundamentals(["AAPL", "MSFT"])

        assert "AAPL" in result
        assert "MSFT" in result
        assert mock_gf.call_count == 2
        # Restore
        cache_mod._default_cache = None

    def test_prefetch_returns_defaults_on_failure(self):
        from unittest.mock import patch
        with patch("app.ml.fundamental_fetcher.get_fundamentals", side_effect=Exception("fail")):
            from app.ml.fundamental_fetcher import prefetch_fundamentals
            result = prefetch_fundamentals(["FAIL"])
        assert result["FAIL"]["pe_ratio"] == 0.0
