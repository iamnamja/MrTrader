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


def _force_yfinance(monkeypatch):
    """Pin the provider flag to 'yfinance' so the network-provider branch (not the
    default 'aggregate_5min' branch) is exercised."""
    import app.ml.retrain_config as rc
    monkeypatch.setattr(rc, "INTRADAY_DAILY_FEATURE_PROVIDER", "yfinance", raising=True)


def test_fetch_daily_all_uses_configured_provider(monkeypatch):
    """_fetch_daily_all must request INTRADAY_DAILY_FEATURE_PROVIDER ('yfinance'),
    NOT the trainer's 5-min self._provider ('alpaca')."""
    _force_yfinance(monkeypatch)
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
    _force_yfinance(monkeypatch)
    end = datetime(2024, 1, 1)
    stub = _RecordingProvider(bars_per_symbol=500, end=end)
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    start = datetime(2022, 6, 1)
    trainer._fetch_daily_all(["AAPL"], start, end)

    assert stub.calls, "provider was never called"
    _syms, called_start, called_end = stub.calls[0]
    # PR #343 follow-up: _fetch_daily_all now coerces the range to datetime.date
    # before handing it to the provider (the yfinance cache compares against date
    # and raised TypeError on datetime). The 365d buffer is preserved.
    assert called_start == (start - timedelta(days=365)).date()
    assert called_end == end.date()


def test_shallow_coverage_warns(monkeypatch, caplog):
    """A provider returning ~100-bar frames over a 2yr request must trip the
    shallow-coverage warning (the Alpaca-cap symptom)."""
    _force_yfinance(monkeypatch)
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
    _force_yfinance(monkeypatch)
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


# ─────────────────────────────────────────────────────────────────────────────
# PR #343 follow-up: datetime→date coercion + aggregate_5min source.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import date  # noqa: E402

from app.ml.intraday_training import aggregate_5min_to_daily  # noqa: E402


class _DateAssertingProvider:
    """Provider that asserts get_daily_bars_bulk receives datetime.date (not
    datetime.datetime) — regression lock for the PR #343 'can't compare
    datetime.datetime to datetime.date' bug."""

    def __init__(self, end: datetime):
        self._end = end
        self.calls = []

    def get_daily_bars_bulk(self, symbols, start, end):
        assert isinstance(start, date) and not isinstance(start, datetime), (
            f"start must be a datetime.date, got {type(start).__name__}: {start!r}"
        )
        assert isinstance(end, date) and not isinstance(end, datetime), (
            f"end must be a datetime.date, got {type(end).__name__}: {end!r}"
        )
        self.calls.append((list(symbols), start, end))
        return {s: _make_daily_df(500, self._end) for s in symbols}


def test_fetch_daily_all_coerces_datetime(monkeypatch):
    """Regression lock: passing datetime start/end must reach the provider as
    datetime.date objects (the bug returned 0/703 symbols on the live run)."""
    _force_yfinance(monkeypatch)
    end = datetime(2024, 1, 1)
    stub = _DateAssertingProvider(end=end)
    recorder = {"requested": [], "provider": stub}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    # Pass DATETIME objects (the per-fold path does) — must not raise.
    out = trainer._fetch_daily_all(
        ["AAPL", "MSFT"], datetime(2022, 1, 1, 9, 30), datetime(2024, 1, 1, 16, 0)
    )
    assert out
    assert stub.calls, "provider was never called"
    _syms, called_start, called_end = stub.calls[0]
    assert called_start == date(2021, 1, 1)  # 2022-01-01 - 365d, coerced to date
    assert called_end == date(2024, 1, 1)


def _make_5min_df(days, bars_per_day=78, start_day=date(2023, 1, 2)):
    """Synthetic 5-min OHLCV spanning `days` business days, `bars_per_day` bars
    each. Values are deterministic so the daily aggregation is checkable."""
    rows = []
    idx = []
    d = pd.Timestamp(start_day)
    placed = 0
    while placed < days:
        if d.weekday() < 5:  # business day
            for b in range(bars_per_day):
                ts = d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * b)
                idx.append(ts)
                # open of first bar = 100 + placed; ramp within the day.
                o = 100.0 + placed + b * 0.1
                rows.append({
                    "open": o,
                    "high": o + 1.0,
                    "low": o - 1.0,
                    "close": o + 0.5,
                    "volume": 1000 + b,
                })
            placed += 1
        d += pd.Timedelta(days=1)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_aggregate_5min_daily():
    """aggregate_5min_to_daily must reduce 5-min bars to correct daily OHLCV:
    open=first, high=max, low=min, close=last, volume=sum, N daily rows."""
    n_days = 5
    bpd = 78
    df = _make_5min_df(n_days, bars_per_day=bpd)
    daily = aggregate_5min_to_daily({"AAPL": df})
    assert "AAPL" in daily
    agg = daily["AAPL"]
    assert len(agg) == n_days, f"expected {n_days} daily rows, got {len(agg)}"

    # Verify the first day's reduction against the raw 5-min bars.
    first_key = agg.index[0]
    day_bars = df[pd.DatetimeIndex(df.index).normalize() == first_key]
    assert agg.iloc[0]["open"] == day_bars.iloc[0]["open"]
    assert agg.iloc[0]["high"] == day_bars["high"].max()
    assert agg.iloc[0]["low"] == day_bars["low"].min()
    assert agg.iloc[0]["close"] == day_bars.iloc[-1]["close"]
    assert agg.iloc[0]["volume"] == day_bars["volume"].sum()


def test_aggregate_5min_excludes_vix_and_malformed():
    """VIX overlays are excluded; frames missing OHLCV columns are skipped."""
    df = _make_5min_df(3)
    bad = pd.DataFrame({"close": [1, 2, 3]},
                       index=pd.date_range("2023-01-02", periods=3, freq="D"))
    daily = aggregate_5min_to_daily({"AAPL": df, "^VIX": df, "BAD": bad})
    assert "AAPL" in daily
    assert "^VIX" not in daily
    assert "BAD" not in daily


def test_aggregate_5min_coverage_matches_5min():
    """The aggregated daily span must equal the 5-min span (no gap that would
    empty the per-fold matrix)."""
    n_days = 10
    df = _make_5min_df(n_days)
    daily = aggregate_5min_to_daily({"AAPL": df, "MSFT": df})
    five_min_days = set(pd.DatetimeIndex(df.index).normalize().date)
    for sym in ("AAPL", "MSFT"):
        agg_days = set(pd.DatetimeIndex(daily[sym].index).date)
        assert agg_days == five_min_days, (
            f"{sym}: aggregated daily span {sorted(agg_days)} != 5-min span "
            f"{sorted(five_min_days)}"
        )


def test_fetch_daily_all_aggregate_5min_uses_in_memory(monkeypatch):
    """With the flag set to 'aggregate_5min' and symbols_data passed, the daily
    bars come from the in-memory 5-min aggregation — the network provider must
    NEVER be called."""
    import app.ml.retrain_config as rc
    monkeypatch.setattr(rc, "INTRADAY_DAILY_FEATURE_PROVIDER", "aggregate_5min",
                        raising=True)
    # Any get_provider call here would be a bug → make it explode.

    def _boom(name=None):
        raise AssertionError(f"get_provider({name!r}) called for aggregate_5min")
    monkeypatch.setattr("app.data.get_provider", _boom)
    monkeypatch.setattr("app.data.registry.get_provider", _boom)

    df = _make_5min_df(8)
    trainer = IntradayModelTrainer(provider="alpaca")
    out = trainer._fetch_daily_all(
        ["AAPL"], datetime(2023, 1, 2), datetime(2023, 1, 20),
        symbols_data={"AAPL": df},
    )
    assert "AAPL" in out
    assert len(out["AAPL"]) == 8


def test_daily_fetch_logs_error_not_silent(monkeypatch, caplog):
    """Forcing the provider to raise must log the exception TYPE + message (not
    silently swallow it to {} like the original bare except)."""
    _force_yfinance(monkeypatch)

    class _Boom:
        def get_daily_bars_bulk(self, symbols, start, end):
            raise TypeError("can't compare datetime.datetime to datetime.date")

    recorder = {"requested": [], "provider": _Boom()}
    _patch_get_provider(monkeypatch, recorder)

    trainer = IntradayModelTrainer(provider="alpaca")
    with caplog.at_level("WARNING"):
        out = trainer._fetch_daily_all(["AAPL"], datetime(2022, 1, 1),
                                       datetime(2024, 1, 1))
    assert out == {}
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "TypeError" in msgs, f"exception type not logged: {msgs}"
    assert "FAILED" in msgs
    # exc_info must be attached (traceback), not just a bare message.
    assert any(r.exc_info for r in caplog.records), "exc_info/traceback not logged"
