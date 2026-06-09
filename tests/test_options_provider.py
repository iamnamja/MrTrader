"""OPT-1b: options data provider — OCC parsing + PIT/survivorship correctness.

These tests use injected synthetic frames (no network / no S3). They lock the two
leak-killers: knowable_date filtering (no look-ahead) and survivorship (expired contracts
remain visible as-of a past date).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.options.contracts import OptionsDataProvider
from app.data.options_provider import (
    parse_occ, occ_root_pattern, contracts_from_bars, PolygonOptionsProvider,
    BARS_COLS, knowable_date, OPT_BAR_LAG_BDAYS, load_options_bars,
)
from scripts.backfill_options import filter_day_panel, _merge


# ── OCC parsing ────────────────────────────────────────────────────────────────

def test_parse_occ_call_and_put():
    c = parse_occ("O:SPY260116C00500000")
    assert c == {"underlying": "SPY", "contract_type": "call",
                 "strike": 500.0, "expiration": date(2026, 1, 16)}
    p = parse_occ("O:AAPL241220P00185500")
    assert p["underlying"] == "AAPL" and p["contract_type"] == "put"
    assert p["strike"] == 185.5 and p["expiration"] == date(2024, 12, 20)


def test_parse_occ_rejects_malformed():
    for bad in ("AAPL", "O:SPY260116X00500000", "O:SPY26011C00500000", "", "O:SPY"):
        assert parse_occ(bad) is None


def test_occ_root_pattern_disambiguates_prefix_roots():
    # SPY must not swallow SPYG (the trailing 6-digit date is required after the root).
    pat = occ_root_pattern(["SPY"])
    assert pat.match("O:SPY260116C00500000") is not None
    assert pat.match("O:SPYG260116C00500000") is None
    # group 1 captures the root
    assert pat.match("O:SPY260116C00500000").group(1) == "SPY"


# ── Synthetic store ────────────────────────────────────────────────────────────

def _bars():
    rows = [
        # (contract, underlying, trade_date)
        ("O:SPY260116C00500000", "SPY", date(2025, 12, 1)),
        ("O:SPY260116C00500000", "SPY", date(2025, 12, 2)),
        ("O:SPY251219P00480000", "SPY", date(2025, 11, 20)),  # expires 2025-12-19
        ("O:AAPL260116C00200000", "AAPL", date(2025, 12, 1)),
    ]
    recs = []
    for contract, und, d in rows:
        recs.append({
            "underlying": und, "contract": contract, "date": pd.Timestamp(d),
            "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 100.0,
            "knowable_date": pd.Timestamp(knowable_date(d, OPT_BAR_LAG_BDAYS)),
        })
    return pd.DataFrame(recs, columns=BARS_COLS)


def test_contracts_from_bars_derives_metadata():
    meta = contracts_from_bars(_bars())
    assert set(meta["contract"]) == {
        "O:SPY260116C00500000", "O:SPY251219P00480000", "O:AAPL260116C00200000"}
    spy_call = meta[meta["contract"] == "O:SPY260116C00500000"].iloc[0]
    assert spy_call["strike"] == 500.0 and spy_call["contract_type"] == "call"
    # knowable_date is the first traded bar's knowable date (2025-12-01 + 1 bday)
    assert spy_call["knowable_date"] == pd.Timestamp(knowable_date(date(2025, 12, 1), 1))


def test_get_contract_bars_is_point_in_time():
    prov = PolygonOptionsProvider(bars=_bars())
    # As of 2025-12-01, the 12-01 bar (knowable 12-02) is NOT yet knowable.
    early = prov.get_contract_bars("SPY", date(2025, 12, 1))
    assert (early["knowable_date"] <= pd.Timestamp(date(2025, 12, 1))).all()
    assert pd.Timestamp(date(2025, 12, 2)) not in set(early["date"])
    # As of 2025-12-03, both SPY-call bars are knowable.
    later = prov.get_contract_bars("SPY", date(2025, 12, 3))
    assert len(later[later["contract"] == "O:SPY260116C00500000"]) == 2
    # underlying filter holds (no AAPL rows)
    assert (later["underlying"] == "SPY").all()


def test_get_universe_survivorship_and_pit():
    prov = PolygonOptionsProvider(bars=_bars())
    # As of 2025-12-22 (after the Dec put expired): include_expired keeps it visible...
    u_all = prov.get_universe("SPY", date(2025, 12, 22), include_expired=True)
    assert "O:SPY251219P00480000" in u_all
    # ...but a live-only universe drops the already-expired contract.
    u_live = prov.get_universe("SPY", date(2025, 12, 22), include_expired=False)
    assert "O:SPY251219P00480000" not in u_live
    assert "O:SPY260116C00500000" in u_live


def test_get_universe_excludes_not_yet_knowable():
    prov = PolygonOptionsProvider(bars=_bars())
    # Before the SPY put ever traded (2025-11-20), it isn't in the universe.
    u = prov.get_universe("SPY", date(2025, 11, 1))
    assert u == []


def test_provider_satisfies_frozen_contract():
    prov = PolygonOptionsProvider(bars=_bars())
    assert isinstance(prov, OptionsDataProvider)


# ── Review-driven regressions ──────────────────────────────────────────────────

def test_knowable_date_is_holiday_aware():
    # Wed before Thanksgiving 2025 (Nov 26): the next calendar weekday (Thu Nov 27) is a
    # market holiday, so the bar is only knowable Fri Nov 28 — never on the closed session.
    assert knowable_date(date(2025, 11, 26), 1) == date(2025, 11, 28)
    # Tue Dec 24 2024 -> Christmas Wed holiday -> knowable Thu Dec 26.
    assert knowable_date(date(2024, 12, 24), 1) == date(2024, 12, 26)
    # plain weekday with no holiday: simple +1
    assert knowable_date(date(2025, 12, 1), 1) == date(2025, 12, 2)


def test_parse_occ_drops_adjusted_root_and_bad_calendar():
    # Post-split / adjusted roots carry a numeric suffix -> deliberately unparseable.
    assert parse_occ("O:AAPL1241220C00100000") is None
    # Structurally valid but impossible calendar date -> None (not a crash).
    assert parse_occ("O:SPY261301C00500000") is None


def test_get_universe_expiration_boundary_inclusive():
    # A contract expiring exactly on as_of is still 'live' (options trade on expiry day).
    prov = PolygonOptionsProvider(bars=_bars())
    u = prov.get_universe("SPY", date(2025, 12, 19), include_expired=False)
    assert "O:SPY251219P00480000" in u


def test_contracts_from_bars_knowable_is_min_not_first():
    # Bars arriving out of order: knowable_date must be the MIN (earliest), not first-seen.
    df = _bars()
    # prepend a later-but-listed-first row scenario: shuffle so a later date is first
    shuffled = df.sort_values("date", ascending=False).reset_index(drop=True)
    meta = contracts_from_bars(shuffled)
    spy = meta[meta["contract"] == "O:SPY260116C00500000"].iloc[0]
    assert spy["knowable_date"] == pd.Timestamp(knowable_date(date(2025, 12, 1), 1))


def test_filter_day_panel_multi_underlying_alignment():
    # The str.extract->underlying assignment must align (no cross-contamination), and
    # non-target / junk rows are dropped.
    panel = pd.DataFrame({
        "contract": ["O:SPY260116C00500000", "O:AAPL260116P00200000",
                     "O:MSFT260116C00400000", "GARBAGE", "O:SPYG260116C00500000"],
        "open": [1.0, 2.0, 3.0, 9.0, 5.0], "high": [1, 2, 3, 9, 5],
        "low": [1, 2, 3, 9, 5], "close": [1.1, 2.1, 3.1, 9.1, 5.1],
        "volume": [10, 20, 30, 40, 50],
    })
    pat = occ_root_pattern(["SPY", "AAPL"])  # MSFT/SPYG/GARBAGE excluded
    out = filter_day_panel(panel, date(2025, 12, 1), pat)
    got = dict(zip(out["contract"], out["underlying"]))
    assert got == {"O:SPY260116C00500000": "SPY", "O:AAPL260116P00200000": "AAPL"}
    # Producer stamps datetime64[ns] (not the [s] pandas infers from a date scalar), so
    # concat with the ns parquet store never hits a resolution mismatch.
    assert out["date"].dtype == "datetime64[ns]"
    assert out["knowable_date"].dtype == "datetime64[ns]"


def test_merge_keeps_revised_bar_on_rerun():
    cols = BARS_COLS
    key = ["contract", "date"]
    base = pd.DataFrame([{
        "underlying": "SPY", "contract": "O:SPY260116C00500000",
        "date": pd.Timestamp("2025-12-01"), "open": 1, "high": 1, "low": 1,
        "close": 1.10, "volume": 100, "knowable_date": pd.Timestamp("2025-12-02")}],
        columns=cols)
    revised = base.copy()
    revised.loc[0, "close"] = 1.25  # Polygon revision on re-run
    merged = _merge(base, [revised], cols, key)
    assert len(merged) == 1 and merged.iloc[0]["close"] == 1.25


def test_load_coerces_datetime_dtypes(tmp_path, monkeypatch):
    # A parquet whose date columns round-tripped as strings must still PIT-filter
    # correctly (dtype coercion on load), not crash or string-compare.
    import app.data.options_provider as opm
    df = _bars().copy()
    df["date"] = df["date"].astype(str)
    df["knowable_date"] = df["knowable_date"].astype(str)
    pq = tmp_path / "options_bars.parquet"
    df.to_parquet(pq, index=False)
    monkeypatch.setattr(opm, "OPTIONS_BARS_PARQUET", pq)
    loaded = load_options_bars(refresh=True)
    assert pd.api.types.is_datetime64_any_dtype(loaded["knowable_date"])
    prov = PolygonOptionsProvider()  # reads the (string-origin) parquet via load
    bars = prov.get_contract_bars("SPY", date(2025, 12, 3))
    assert len(bars) >= 1
    monkeypatch.undo()
    load_options_bars(refresh=True)  # reset module cache to the real store for other tests
