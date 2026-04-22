"""
Tests for Phase 21 — Polygon Intraday Data Infrastructure.

Verifies:
  1. intraday_cache.load() reads Parquet correctly
  2. intraday_cache.save() writes Parquet and can be reloaded
  3. intraday_cache.load_many() works across multiple symbols
  4. intraday_cache.cache_is_fresh() respects TTL and date coverage
  5. intraday_cache.available_symbols() enumerates Parquet files
  6. intraday_cache.cache_stats() returns correct counts
  7. IntradayModelTrainer._fetch_data() prefers Polygon cache over Alpaca
  8. fetch_intraday_history script is importable and main() requires API key
"""
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _5min_bars(n: int = 200, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": [base] * n,
        "high": [base * 1.002] * n,
        "low": [base * 0.998] * n,
        "close": [base] * n,
        "volume": [10_000.0] * n,
    }, index=idx)


# ── Cache Read/Write ───────────────────────────────────────────────────────────

class TestIntradayCache:

    @pytest.fixture(autouse=True)
    def temp_cache(self, tmp_path, monkeypatch):
        """Redirect INTRADAY_CACHE_DIR to a temp directory for all tests."""
        monkeypatch.setattr("app.data.intraday_cache.INTRADAY_CACHE_DIR", tmp_path)
        self.cache_dir = tmp_path

    def test_save_and_load_roundtrip(self):
        from app.data.intraday_cache import save, load
        df = _5min_bars(200)
        save("AAPL", df)
        loaded = load("AAPL")
        assert loaded is not None
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)

    def test_load_returns_none_for_missing(self):
        from app.data.intraday_cache import load
        assert load("NONEXISTENT") is None

    def test_load_date_filter(self):
        from app.data.intraday_cache import save, load
        df = _5min_bars(400)
        save("TSLA", df)
        cutoff = df.index[200].date()
        loaded = load("TSLA", start=cutoff)
        assert loaded is not None
        assert loaded.index.min().date() >= cutoff

    def test_load_many_returns_found_symbols(self):
        from app.data.intraday_cache import save, load_many
        save("AAPL", _5min_bars(100))
        save("MSFT", _5min_bars(100))
        result = load_many(["AAPL", "MSFT", "MISSING"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert "MISSING" not in result

    def test_available_symbols(self):
        from app.data.intraday_cache import save, available_symbols
        save("GOOG", _5min_bars(100))
        save("AMZN", _5min_bars(100))
        syms = available_symbols()
        assert "GOOG" in syms
        assert "AMZN" in syms

    def test_cache_is_fresh_missing(self):
        from app.data.intraday_cache import cache_is_fresh
        assert cache_is_fresh("NOTHERE") is False

    def test_cache_is_fresh_new_file(self):
        from app.data.intraday_cache import save, cache_is_fresh
        save("FRESH", _5min_bars(100))
        assert cache_is_fresh("FRESH") is True

    def test_cache_stats(self):
        from app.data.intraday_cache import save, cache_stats
        save("A", _5min_bars(100))
        save("B", _5min_bars(50))
        stats = cache_stats()
        assert stats["symbols"] == 2
        assert stats["total_bars"] == 150

    def test_save_overwrites_existing(self):
        from app.data.intraday_cache import save, load
        df1 = _5min_bars(100)
        df2 = _5min_bars(200)
        save("REWRITE", df1)
        save("REWRITE", df2)
        loaded = load("REWRITE")
        assert len(loaded) == 200

    def test_load_empty_df_returns_none(self):
        """A Parquet file containing an empty DataFrame should return None."""
        from app.data.intraday_cache import save, load
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        save("EMPTY", empty)
        assert load("EMPTY") is None


# ── Trainer prefers Polygon cache ─────────────────────────────────────────────

class TestTrainerPrefersPolygon:

    def test_fetch_data_uses_polygon_when_available(self, tmp_path, monkeypatch):
        """When Polygon cache has the symbol, trainer should not call Alpaca."""
        monkeypatch.setattr("app.data.intraday_cache.INTRADAY_CACHE_DIR", tmp_path)

        from app.data.intraday_cache import save
        save("AAPL", _5min_bars(200))

        from app.ml.intraday_training import IntradayModelTrainer
        trainer = IntradayModelTrainer.__new__(IntradayModelTrainer)
        trainer._force_refresh = False

        alpaca_called = []

        def fake_fetch_all(symbols, start, end, force_refresh):
            alpaca_called.extend(symbols)
            return {}

        monkeypatch.setattr(trainer, "_fetch_all", fake_fetch_all)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        result = trainer._fetch_data(["AAPL"], start, end)

        assert "AAPL" in result
        assert "AAPL" not in alpaca_called

    def test_fetch_data_falls_back_to_alpaca_for_missing(self, tmp_path, monkeypatch):
        """Symbols not in Polygon cache fall back to Alpaca."""
        monkeypatch.setattr("app.data.intraday_cache.INTRADAY_CACHE_DIR", tmp_path)

        from app.data.intraday_cache import save
        save("AAPL", _5min_bars(200))  # only AAPL in cache

        from app.ml.intraday_training import IntradayModelTrainer
        trainer = IntradayModelTrainer.__new__(IntradayModelTrainer)
        trainer._force_refresh = False

        alpaca_called = []

        def fake_fetch_all(symbols, start, end, force_refresh):
            alpaca_called.extend(symbols)
            return {s: _5min_bars(50) for s in symbols}

        monkeypatch.setattr(trainer, "_fetch_all", fake_fetch_all)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        result = trainer._fetch_data(["AAPL", "MSFT"], start, end)

        assert "AAPL" in result  # from cache
        assert "MSFT" in result  # from alpaca fallback
        assert "MSFT" in alpaca_called
        assert "AAPL" not in alpaca_called


# ── Fetch script importability ─────────────────────────────────────────────────

class TestFetchScript:

    def test_script_is_importable(self):
        import scripts.fetch_intraday_history as fih
        assert hasattr(fih, "main")
        assert hasattr(fih, "fetch_symbol")
        assert hasattr(fih, "cache_is_fresh")

    def test_main_returns_1_without_api_key(self, monkeypatch):
        """main() should exit with code 1 when polygon_api_key is missing."""
        import scripts.fetch_intraday_history as fih
        mock_settings = MagicMock()
        mock_settings.polygon_api_key = ""
        monkeypatch.setattr(
            "scripts.fetch_intraday_history.argparse.ArgumentParser.parse_args",
            lambda *a, **kw: MagicMock(symbols=["AAPL"], days=30,
                                       workers=1, force_refresh=False),
        )
        with patch("app.config.settings", mock_settings):
            result = fih.main()
        assert result == 1
