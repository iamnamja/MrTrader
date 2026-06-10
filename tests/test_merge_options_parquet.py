"""Tests for the bounded-memory options parquet stream-merge (scripts/merge_options_parquet.py).

Guards: disjoint date windows concatenate losslessly with schema preserved; OVERLAPPING windows
abort (rather than emit duplicate (contract,date) rows); contracts dedup keeps the earliest
first_date/knowable_date per contract (survivorship + PIT correct).
"""
import pandas as pd
import pytest

import scripts.merge_options_parquet as m


def _bars(tmp_path, name, dates):
    df = pd.DataFrame({
        "underlying": ["AAPL"] * len(dates),
        "contract": [f"O:AAPL{i}" for i in range(len(dates))],
        "date": pd.to_datetime(dates),
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10,
        "knowable_date": pd.to_datetime(dates),
    })
    p = tmp_path / name
    df.to_parquet(p, index=False)
    return str(p)


def test_merge_bars_disjoint_lossless(tmp_path):
    a = _bars(tmp_path, "early.parquet", ["2022-06-16", "2022-06-17"])
    b = _bars(tmp_path, "late.parquet", ["2024-06-10", "2024-06-11", "2024-06-12"])
    out = tmp_path / "merged.parquet"
    m.merge_bars([a, b], str(out))
    res = pd.read_parquet(out)
    assert len(res) == 5
    assert list(res.columns) == ["underlying", "contract", "date", "open", "high",
                                 "low", "close", "volume", "knowable_date"]


def test_merge_bars_overlap_aborts(tmp_path):
    a = _bars(tmp_path, "a.parquet", ["2023-01-01", "2024-07-01"])  # spans into b
    b = _bars(tmp_path, "b.parquet", ["2024-06-10", "2024-06-11"])
    out = tmp_path / "merged.parquet"
    with pytest.raises(SystemExit):
        m.merge_bars([a, b], str(out))
    assert not out.exists()  # nothing written on abort


def test_merge_contracts_keeps_earliest(tmp_path):
    def _c(name, first, knowable):
        df = pd.DataFrame({
            "underlying": ["AAPL"], "contract": ["O:AAPLX"],
            "contract_type": ["call"], "strike": [100.0],
            "expiration": pd.to_datetime(["2024-12-20"]),
            "first_date": pd.to_datetime([first]), "knowable_date": pd.to_datetime([knowable]),
        })
        p = tmp_path / name
        df.to_parquet(p, index=False)
        return str(p)

    early = _c("c_early.parquet", "2022-06-16", "2022-06-17")
    late = _c("c_late.parquet", "2024-06-10", "2024-06-11")
    out = tmp_path / "contracts.parquet"
    m.merge_contracts([early, late], str(out))
    res = pd.read_parquet(out)
    assert len(res) == 1  # same contract deduped
    assert res.iloc[0]["first_date"] == pd.Timestamp("2022-06-16")  # earliest kept
    assert res.iloc[0]["knowable_date"] == pd.Timestamp("2022-06-17")
