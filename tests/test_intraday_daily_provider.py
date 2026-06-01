"""Tests for the intraday DAILY-feature provider fix.

The intraday model's per-symbol DAILY bars (52-week-position / vol-percentile
features) were historically fetched via the trainer's 5-min provider
(self._provider="alpaca"), which caps at ~100 recent daily bars on this
deployment — silently degrading those features to 0.5 defaults across most of
the training window. The fix routes the daily fetch through a configurable
full-history provider (INTRADAY_DAILY_FEATURE_PROVIDER, default "yfinance").

These tests use stubs/monkeypatch only — they never hit the network.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

pytest.importorskip("numpy")

from app.ml.intraday_training import IntradayModelTrainer


def _make_daily_df(n_bars: int, end: datetime) -> pd.DataFrame:
    """Build a synthetic daily OHLCV frame with `n_bars` rows ending at `end`."""
    idx = pd.date_range(end=end, periods=n_bars, freq="B")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        },
        index=idx,
    )


class _RecordingProvider:
    """Stub provider that records the (start, end) it was asked for and returns
    a fixed set of frames with `bars_per_symbol` rows each."""

    def __init__(self, bars_per_symbol: int, end: datetime):
        self.bars_per_symbol = bars_per_symbol
        self._end = end
        self.calls = []  # list of (symbols, start, end)

    def get_daily_bars_bulk(self, symbols, start, end):
        self.calls.append((list(symbols), start, end))
        return {s: _make_daily_df(self.bars_per_symbol, self._end) for s in symbols}


def _patch_get_provider(monkeypatch, recorder):
    """Patch get_provider in both the registry and the intraday_training import
    site to record the requested provider name and return `recorder`'s provider."""
    def _fake_get_provider(name=None):
        recorder["requested"].append(name)
        return recorder["provider"]

    # _fetch_daily_all does `from app.data import get_provider`
    monkeypatch.setattr("app.data.get_provider", _fake_get_provider)
    monkeypatch.setattr("app.data.registry.get_provider", _fake_get_provider)


def test_fetch_daily_all_uses_configured_provider(monkeypatch):
    """_fetch_daily_all must request INTRADAY_DAILY_FEATURE_PROVIDER ('yfinance'),
    NOT the trainer's 5-min self._provider ('alpaca')."""
    end = datetime(2024, 1, 1)
    stub = _RecordingProvider(bars_per_symbol=500, end=end)
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    # Trainer constructed with the legacy 5-min provider name.
    trainer = IntradayModelTrainer(provider="alpaca")
    start = datetime(2022, 1, 1)
    out = trainer._fetch_daily_all(["AAPL", "MSFT"], start, end)

    assert out  # got frames back
    # The provider requested for DAILY bars must be 'yfinance' (the config default),
    # never 'alpaca' (the 5-min provider).
    assert "yfinance" in recorder["requested"]
    assert "alpaca" not in recorder["requested"]


def test_fetch_daily_all_applies_1yr_buffer(monkeypatch):
    """The start passed to the daily provider must be requested_start - 365d."""
    end = datetime(2024, 1, 1)
    stub = _RecordingProvider(bars_per_symbol=500, end=end)
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    start = datetime(2022, 6, 1)
    trainer._fetch_daily_all(["AAPL"], start, end)

    assert stub.calls, "provider was never called"
    _syms, called_start, called_end = stub.calls[0]
    assert called_start == start - timedelta(days=365)
    assert called_end == end


def test_shallow_coverage_warns(monkeypatch, caplog):
    """A provider returning ~100-bar frames over a 2yr request must trip the
    shallow-coverage warning (the Alpaca-cap symptom)."""
    end = datetime(2024, 1, 1)
    stub = _RecordingProvider(bars_per_symbol=100, end=end)  # the ~100-bar cap
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    start = datetime(2022, 1, 1)  # ~2yr request → ~500 trading days expected
    with caplog.at_level("WARNING"):
        trainer._fetch_daily_all(["AAPL", "MSFT"], start, end)

    assert any("shallow" in rec.message.lower() for rec in caplog.records), \
        f"expected a shallow-coverage warning, got: {[r.message for r in caplog.records]}"


def test_no_warning_on_full_coverage(monkeypatch, caplog):
    """Full multi-year coverage must NOT trip the shallow-coverage warning."""
    end = datetime(2024, 1, 1)
    stub = _RecordingProvider(bars_per_symbol=600, end=end)  # ~2.4yr of trading days
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    start = datetime(2022, 1, 1)
    with caplog.at_level("WARNING"):
        trainer._fetch_daily_all(["AAPL", "MSFT"], start, end)

    assert not any("shallow" in rec.message.lower() for rec in caplog.records)


def test_config_flag_override(monkeypatch):
    """Patching INTRADAY_DAILY_FEATURE_PROVIDER='alpaca' must select the alpaca
    provider (legacy path remains selectable)."""
    import app.ml.retrain_config as rc
    monkeypatch.setattr(rc, "INTRADAY_DAILY_FEATURE_PROVIDER", "alpaca", raising=True)

    end = datetime(2024, 1, 1)
    stub = _RecordingProvider(bars_per_symbol=100, end=end)
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    start = datetime(2022, 1, 1)
    trainer._fetch_daily_all(["AAPL"], start, end)

    assert "alpaca" in recorder["requested"]
    assert "yfinance" not in recorder["requested"]
