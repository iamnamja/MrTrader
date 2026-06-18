"""P4 — Norgate futures parquet-mirror provider tests (no NDU / network).

A fake `norgatedata` module is injected into sys.modules so the resume-safe extract/load
logic is exercised without a live subscription.
"""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

ng = importlib.import_module("app.data.norgate_provider")


def _price_df(n=10, start="2020-01-02"):
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Open": base, "High": base + 1, "Low": base - 1, "Close": base,
        "Volume": rng.integers(1000, 5000, n).astype(float),
        "Delivery Month": 202003, "Open Interest": rng.integers(100, 900, n).astype(float),
    }, index=idx)


@pytest.fixture()
def fake_nd(monkeypatch, tmp_path):
    importlib.reload(ng)
    monkeypatch.setattr(ng, "NORGATE_DIR", str(tmp_path))
    monkeypatch.setattr(ng, "CONTINUOUS_DIR", str(tmp_path / "continuous"))
    monkeypatch.setattr(ng, "CONTRACTS_DIR", str(tmp_path / "contracts"))
    monkeypatch.setattr(ng, "MARKETS_META", str(tmp_path / "_markets.parquet"))

    prices = {
        "&ES_CCB": _price_df(), "&ES": _price_df(),
        "ES-2020H": _price_df(8), "ES-2020M": _price_df(8),
        "&CL_CCB": _price_df(), "&CL": None,           # one missing series
        "CL-2020F": _price_df(8),
    }
    contracts_map = {"ES": ["ES-2020H", "ES-2020M"], "CL": ["CL-2020F"]}

    m = types.SimpleNamespace(
        price_timeseries=lambda sym, format=None: prices.get(sym),
        futures_market_symbols=lambda: list(contracts_map.keys()),
        futures_market_session_contracts=lambda mk: contracts_map[mk],
        futures_market_name=lambda mk: f"{mk} Market",
    )
    monkeypatch.setitem(sys.modules, "norgatedata", m)
    return ng


def test_extract_continuous_writes_both_price_types(fake_nd):
    s = fake_nd.extract_continuous(["ES"])
    assert s["saved"] == 1
    df = pd.read_parquet(f"{fake_nd.CONTINUOUS_DIR}/ES.parquet")
    assert set(df["price_type"].unique()) == {"backadjusted", "unadjusted"}
    assert {"open", "high", "low", "close", "volume", "open_interest",
            "delivery_month"}.issubset(df.columns)


def test_extract_continuous_handles_missing_series(fake_nd):
    # &CL exists (backadj) but &CL unadjusted is None -> still saved with one price_type
    fake_nd.extract_continuous(["CL"])
    df = pd.read_parquet(f"{fake_nd.CONTINUOUS_DIR}/CL.parquet")
    assert set(df["price_type"].unique()) == {"backadjusted"}


def test_extract_continuous_resume_safe(fake_nd):
    s1 = fake_nd.extract_continuous(["ES"])
    s2 = fake_nd.extract_continuous(["ES"])          # already on disk
    assert s1["saved"] == 1 and s2["skipped"] == 1 and s2["saved"] == 0
    s3 = fake_nd.extract_continuous(["ES"], force=True)
    assert s3["saved"] == 1


def test_extract_contracts_long_format(fake_nd):
    s = fake_nd.extract_contracts(["ES"])
    assert s["markets_saved"] == 1 and s["contracts"] == 2
    df = pd.read_parquet(f"{fake_nd.CONTRACTS_DIR}/ES.parquet")
    assert set(df["contract"].unique()) == {"ES-2020H", "ES-2020M"}


def test_load_round_trip(fake_nd):
    fake_nd.extract_continuous(["ES"])
    fake_nd.extract_contracts(["ES"])
    cont = fake_nd.load_continuous("ES", price_type="backadjusted")
    assert cont.index.name == "date" and "close" in cont.columns
    assert "price_type" not in cont.columns
    contr = fake_nd.load_contracts("ES")
    assert contr["contract"].nunique() == 2


def test_load_missing_raises(fake_nd):
    with pytest.raises(FileNotFoundError):
        fake_nd.load_continuous("ZZZ")
    with pytest.raises(FileNotFoundError):
        fake_nd.load_contracts("ZZZ")


def test_metadata_and_status(fake_nd):
    md = fake_nd.extract_market_metadata()
    assert set(md["market"]) == {"ES", "CL"}
    fake_nd.extract_continuous(["ES", "CL"])
    fake_nd.extract_contracts(["ES"])
    st = fake_nd.mirror_status()
    assert st["continuous_markets"] == 2 and st["contract_markets"] == 1
    assert st["total_size_mb"] >= 0


def test_norm_maps_columns():
    raw = _price_df(5)
    out = ng._norm(raw)
    assert "close" in out.columns and "open_interest" in out.columns
    assert out.index.name == "date"
