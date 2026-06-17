"""P3-5 — FINRA daily short-volume provider tests (network-free)."""
from __future__ import annotations

import importlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

fsv = importlib.import_module("app.data.finra_short_volume")


_SAMPLE = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20260612|SPY|600000|10|1000000|B,Q,N\n"
    "20260612|AA|300000|0|1000000|B,Q,N\n"
    "20260612|QQQ|100000|0|500000|Q\n"
    "0612|RECORD COUNT|3\n"          # footer-ish junk row -> must be dropped
)


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status


class _Session:
    def __init__(self, resp):
        self._resp = resp

    def get(self, url, timeout=None):
        return self._resp


def test_fetch_daily_parses_and_drops_footer():
    df = fsv.fetch_daily(date(2026, 6, 12), session=_Session(_Resp(_SAMPLE)))
    assert df is not None
    assert set(df["symbol"]) == {"SPY", "AA", "QQQ"}     # footer 'RECORD COUNT' dropped
    spy = df.set_index("symbol").loc["SPY"]
    assert spy["short_volume"] == 600000 and spy["total_volume"] == 1000000


def test_fetch_daily_none_on_accessdenied():
    xml = "<?xml version='1.0'?><Error><Code>AccessDenied</Code></Error>"
    assert fsv.fetch_daily(date(2015, 1, 2), session=_Session(_Resp(xml))) is None


def test_fetch_daily_none_on_non200():
    assert fsv.fetch_daily(date(2026, 6, 13), session=_Session(_Resp("", 404))) is None


def test_distil_aggregate_and_per_symbol():
    df = fsv.fetch_daily(date(2026, 6, 12), session=_Session(_Resp(_SAMPLE)))
    row = fsv._distil(date(2026, 6, 12), df, ["SPY", "QQQ", "MISSING"])
    # aggregate ratio = (600000+300000+100000) / (1000000+1000000+500000) = 1.0M/2.5M = 0.40
    assert row["agg_short_ratio"] == pytest.approx(0.40)
    assert row["short_ratio_SPY"] == pytest.approx(0.60)
    assert row["short_ratio_QQQ"] == pytest.approx(0.20)
    assert pd.isna(row["short_ratio_MISSING"])           # absent symbol -> NaN


def test_build_panel_incremental_and_cached(monkeypatch, tmp_path):
    cache = str(tmp_path / "fsv.parquet")
    calls = {"n": 0}

    def _fake_fetch(d, session=None):
        calls["n"] += 1
        # constant synthetic day: agg ratio 0.5, SPY 0.6
        return pd.DataFrame({"symbol": ["SPY", "AA"],
                             "short_volume": [600.0, 400.0],
                             "total_volume": [1000.0, 1000.0]})

    monkeypatch.setattr(fsv, "fetch_daily", _fake_fetch)
    # 1 business week
    p1 = fsv.build_panel(start=date(2026, 6, 8), end=date(2026, 6, 12),
                         symbols=["SPY"], cache_path=cache, log_every=0)
    assert len(p1) == 5 and calls["n"] == 5
    assert p1["agg_short_ratio"].iloc[0] == pytest.approx(0.5)
    assert p1["short_ratio_SPY"].iloc[0] == pytest.approx(0.6)

    # re-run same range -> everything cached, no new fetches
    calls["n"] = 0
    p2 = fsv.build_panel(start=date(2026, 6, 8), end=date(2026, 6, 12),
                         symbols=["SPY"], cache_path=cache, log_every=0)
    assert calls["n"] == 0 and len(p2) == 5


def test_load_aggregate_ratio(monkeypatch, tmp_path):
    cache = str(tmp_path / "fsv.parquet")
    monkeypatch.setattr(fsv, "fetch_daily", lambda d, session=None: pd.DataFrame(
        {"symbol": ["SPY"], "short_volume": [500.0], "total_volume": [1000.0]}))
    fsv.build_panel(start=date(2026, 6, 8), end=date(2026, 6, 10),
                    symbols=["SPY"], cache_path=cache, log_every=0)
    s = fsv.load_aggregate_ratio(cache_path=cache)
    assert not s.empty and np.allclose(s.to_numpy(), 0.5)
    assert s.index.is_monotonic_increasing
