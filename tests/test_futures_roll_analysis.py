"""R1.3 — futures_roll_analysis: OI-crossover roll detection + fixed/FND/liquidity gap table.
Uses a synthetic Norgate panel (monkeypatched) so it runs without the mirror."""
import pandas as pd
import pytest

from app.research import futures_roll_analysis as ra


def _panel():
    # Two ZS contracts: N (Jul, nearer) and X (Nov, later). OI sits in N for the first 5 days, then
    # migrates to X on day 6 → the liquidity roll out of ZS-2026N is 2026-02-06.
    rows = []
    dates = pd.date_range("2026-02-01", periods=10, freq="D")
    for i, d in enumerate(dates):
        n_oi, x_oi = (100, 10) if i < 5 else (10, 100)
        rows.append({"date": d, "contract": "ZS-2026N", "open_interest": n_oi, "volume": n_oi})
        rows.append({"date": d, "contract": "ZS-2026X", "open_interest": x_oi, "volume": x_oi})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_norgate(monkeypatch):
    monkeypatch.setattr(ra.ng, "load_contracts", lambda market: _panel())


def test_contract_ym_parses_month_code():
    assert ra._contract_ym("ZS-2026N") == (2026, 7)      # N = July
    assert ra._contract_ym("ES-2026U") == (2026, 9)      # U = September
    assert ra._contract_ym("garbage") is None


def test_liquidity_front_flips_on_oi_crossover(fake_norgate):
    front = ra.liquidity_front("ZS")
    assert front.loc["2026-02-01"] == "ZS-2026N"         # OI in the near contract
    assert front.loc["2026-02-06"] == "ZS-2026X"         # OI migrated → front flipped


def test_roll_timing_comparison_gaps(fake_norgate):
    t = ra.roll_timing_comparison("ZS", last_n=5)
    assert len(t) == 1                                   # one roll (out of the N contract)
    row = t.iloc[0]
    assert row["cycle"] == "ZS-2026N" and str(row["liquidity_roll"]) == "2026-02-06"
    assert str(row["fixed_roll"]) == "2026-07-10"        # scheduled (Jul-15) minus 5
    assert str(row["fnd"]) == "2026-06-30"               # grain FND = last biz day of June
    # fixed rolls WAY after liquidity migrated — the headline finding for physical markets.
    assert row["fixed_minus_liq"] > 100 and row["fnd_minus_liq"] > 100


def test_summarize_shape(fake_norgate):
    s = ra.summarize(["ZS"], last_n=5)
    assert s.iloc[0]["market"] == "ZS" and s.iloc[0]["settlement"] == "physical"
    assert s.iloc[0]["median_fixed_minus_liq"] > 100
