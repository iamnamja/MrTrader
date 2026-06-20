"""P0.2 — futures roll-schedule tests (the contract-switch detector for roll cost)."""
from __future__ import annotations

import importlib
import os

import pandas as pd
import pytest

from app.data import norgate_provider as ng

fr = importlib.import_module("app.research.futures_roll")


def _write_3_contract_market(dirpath, market):
    """Three contracts delivering H(Mar)/M(Jun)/U(Sep) 2010, all quoting Jan->Aug 2010, so the
    SCHEDULED-expiry front rolls H->M (~Mar-10) then M->U (~Jun-10) as each nears expiry."""
    dates = pd.bdate_range("2010-01-04", "2010-08-31")
    rows = []
    for code in ("H", "M", "U"):
        rows.append(pd.DataFrame({"date": dates, "contract": f"{market}-2010{code}",
                                  "close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0,
                                  "volume": 1000.0, "open_interest": 1000.0}))
    pd.concat(rows, ignore_index=True).to_parquet(
        os.path.join(dirpath, f"{market}.parquet"), index=False)


@pytest.fixture()
def cmirror(monkeypatch, tmp_path):
    contr = tmp_path / "contracts"
    contr.mkdir()
    monkeypatch.setattr(ng, "CONTRACTS_DIR", str(contr))
    importlib.reload(fr)
    monkeypatch.setattr(fr.ng, "CONTRACTS_DIR", str(contr))
    return str(contr)


def test_roll_schedule_detects_two_rolls(cmirror):
    _write_3_contract_market(cmirror, "XX")
    s = fr.roll_schedule("XX")
    rolls = s[s]
    assert len(rolls) == 2                       # H->M and M->U
    # rolls land ~5 trading days before the 15th of Mar and Jun
    months = sorted(rolls.index.month.tolist())
    assert months == [3, 6]


def test_roll_days_panel_shape_and_reindex(cmirror):
    _write_3_contract_market(cmirror, "XX")
    _write_3_contract_market(cmirror, "YY")
    idx = pd.bdate_range("2010-01-04", "2010-08-31")
    panel = fr.roll_days_panel(["XX", "YY"], index=idx)
    assert list(panel.columns) == ["XX", "YY"]
    assert panel.index.equals(pd.DatetimeIndex(idx))
    assert panel.values.sum() == 4               # 2 rolls per market
    assert panel.dtypes.eq(bool).all()


def test_roll_days_panel_missing_market_skipped(cmirror):
    _write_3_contract_market(cmirror, "XX")
    panel = fr.roll_days_panel(["XX", "ZZ"])      # ZZ has no file
    assert "XX" in panel.columns and "ZZ" not in panel.columns
