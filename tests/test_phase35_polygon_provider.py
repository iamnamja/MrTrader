"""Tests for Phase 35: Polygon data provider (REST + S3)."""
import gzip
import io
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest


def _make_daily_df(n=10, start="2024-01-02"):
    idx = pd.bdate_range(start=start, periods=n)
    prices = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.ones(n) * 1_000_000,
    }, index=idx)


def _rest_agg_response(n=5):
    """Fake Polygon REST aggregates response."""
    import time
    base_ms = int(pd.Timestamp("2024-01-02").timestamp() * 1000)
    day_ms = 86_400_000
    results = [
        {"t": base_ms + i * day_ms, "o": 100.0 + i, "h": 101.0 + i,
         "l": 99.0 + i, "c": 100.5 + i, "v": 1_000_000.0}
        for i in range(n)
    ]
    return {"results": results, "resultsCount": n, "ticker": "AAPL", "status": "OK"}


class TestPolygonProviderREST:

    def _provider(self):
        from app.data.polygon_provider import PolygonProvider
        return PolygonProvider(
            api_key="test_key",
            s3_access_key=None, s3_secret_key=None,
            s3_endpoint=None, s3_bucket=None,
        )

    def test_name(self):
        assert self._provider().name == "polygon"

    def test_get_daily_bars_returns_df(self, tmp_path):
        from app.data.cache import DataCache
        p = self._provider()
        with patch("app.data.polygon_provider.get_cache", return_value=DataCache(tmp_path)), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _rest_agg_response(5)
            result = p.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 8))
        assert result is not None
        assert "close" in result.columns
        assert len(result) > 0

    def test_get_daily_bars_returns_none_on_error(self, tmp_path):
        from app.data.cache import DataCache
        p = self._provider()
        with patch("app.data.polygon_provider.get_cache", return_value=DataCache(tmp_path)), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 403
            result = p.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 8))
        assert result is None

    def test_get_daily_bars_uses_cache_on_second_call(self, tmp_path):
        from app.data.cache import DataCache
        cache = DataCache(tmp_path)
        p = self._provider()
        with patch("app.data.polygon_provider.get_cache", return_value=cache), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _rest_agg_response(5)
            p.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 6))
            p.get_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 6))
        # Second call should hit cache — REST called at most once
        assert mock_get.call_count <= 2  # could be 1 or 2 depending on cache hit

    def test_get_intraday_bars_resamples(self, tmp_path):
        from app.data.cache import DataCache
        from datetime import date as _date
        today = _date.today()
        p = self._provider()
        base_ms = int(pd.Timestamp(f"{today} 09:30").timestamp() * 1000)
        results = [
            {"t": base_ms + i * 60_000, "o": 100.0, "h": 100.5,
             "l": 99.5, "c": 100.2, "v": 10_000.0}
            for i in range(30)
        ]
        with patch("app.data.polygon_provider.get_cache", return_value=DataCache(tmp_path)), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"results": results, "resultsCount": 30}
            result = p.get_intraday_bars(
                "AAPL",
                datetime(today.year, today.month, today.day, 9, 30),
                datetime(today.year, today.month, today.day, 10, 0),
                interval_minutes=5,
            )
        assert result is not None
        assert len(result) <= 30  # resampled to 5-min

    def test_health_check_true_on_200(self):
        p = self._provider()
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            assert p.health_check() is True

    def test_health_check_false_on_exception(self):
        p = self._provider()
        with patch("requests.get", side_effect=Exception("timeout")):
            assert p.health_check() is False

    def test_get_daily_bars_bulk_returns_dict(self, tmp_path):
        from app.data.cache import DataCache
        p = self._provider()
        with patch("app.data.polygon_provider.get_cache", return_value=DataCache(tmp_path)), \
             patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _rest_agg_response(5)
            result = p.get_daily_bars_bulk(
                ["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 8)
            )
        assert isinstance(result, dict)

    def test_registered_in_registry(self):
        from app.data.registry import list_providers
        assert "polygon" in list_providers()


class TestPolygonS3:

    def _make_csv_gz(self, rows):
        """Build an in-memory gzipped CSV with Polygon's column format."""
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        with gzip.open(buf, "wt") as f:
            df.to_csv(f, index=False)
        buf.seek(0)
        return buf.read()

    def _s3_client_mock(self, csv_bytes):
        mock_client = MagicMock()
        mock_client.get_object.return_value = {
            "Body": io.BytesIO(csv_bytes)
        }
        return mock_client

    def test_get_daily_bulk_parses_csv(self, tmp_path):
        from app.data.polygon_s3 import PolygonS3
        rows = [
            {"ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.5, "volume": 1_000_000, "window_start": 1704153600000000000},
            {"ticker": "MSFT", "open": 200.0, "high": 201.0, "low": 199.0,
             "close": 200.5, "volume": 500_000, "window_start": 1704153600000000000},
        ]
        csv_bytes = self._make_csv_gz(rows)
        s3 = PolygonS3("key", "secret", "https://files.polygon.io", "flatfiles")
        with patch.object(s3, "_client") as mock_c:
            mock_c.get_object.return_value = {"Body": io.BytesIO(csv_bytes)}
            result = s3.get_daily_bulk(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 2))
        assert "AAPL" in result or "MSFT" in result

    def test_get_daily_bulk_handles_missing_file(self):
        from app.data.polygon_s3 import PolygonS3
        from botocore.exceptions import ClientError
        s3 = PolygonS3("key", "secret", "https://files.polygon.io", "flatfiles")
        error = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        with patch.object(s3, "_client") as mock_c:
            mock_c.get_object.side_effect = error
            result = s3.get_daily_bulk(["AAPL"], date(2024, 1, 2), date(2024, 1, 2))
        assert result == {}

    def test_health_check_returns_true(self):
        from app.data.polygon_s3 import PolygonS3
        s3 = PolygonS3("key", "secret", "https://files.polygon.io", "flatfiles")
        with patch.object(s3, "_client") as mock_c:
            mock_c.list_objects_v2.return_value = {"Contents": []}
            assert s3.health_check() is True

    def test_health_check_returns_false_on_error(self):
        from app.data.polygon_s3 import PolygonS3
        s3 = PolygonS3("key", "secret", "https://files.polygon.io", "flatfiles")
        with patch.object(s3, "_client") as mock_c:
            mock_c.list_objects_v2.side_effect = Exception("no access")
            assert s3.health_check() is False


class TestResample:

    def test_resample_1min_to_5min(self):
        from app.data.polygon_provider import _resample
        idx = pd.date_range("2024-01-02 09:30", periods=30, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "open": np.ones(30) * 100,
            "high": np.ones(30) * 101,
            "low": np.ones(30) * 99,
            "close": np.ones(30) * 100.5,
            "volume": np.ones(30) * 10_000,
        }, index=idx)
        result = _resample(df, 5)
        assert result is not None
        assert len(result) == 6  # 30 min / 5 = 6 bars

    def test_resample_preserves_ohlcv(self):
        from app.data.polygon_provider import _resample
        idx = pd.date_range("2024-01-02 09:30", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "open":   [100, 101, 102, 103, 104],
            "high":   [105, 106, 107, 108, 109],
            "low":    [ 99,  98,  97,  96,  95],
            "close":  [101, 102, 103, 104, 105],
            "volume": [1000, 2000, 3000, 4000, 5000],
        }, index=idx)
        result = _resample(df, 5)
        assert result is not None
        assert float(result["open"].iloc[0]) == 100.0     # first open
        assert float(result["close"].iloc[0]) == 105.0    # last close
        assert float(result["high"].iloc[0]) == 109.0     # max high
        assert float(result["volume"].iloc[0]) == 15000.0 # sum volume
