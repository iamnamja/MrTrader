"""Unit tests for the dollar-neutral short-interest factor scorer (Alpha-v3 A2).

Mock the SI store; verify the DTC ranking (long lowest / short highest), the
dollar-neutral balance, the min-DTC short floor, long/short disjointness, the VIX
gate, and PIT correctness (knowable_date filtering via latest_known_si).
"""

from datetime import date

import pandas as pd
import pytest

from app.data import short_interest_provider as sip
from app.ml.short_interest_factor_scorer import ShortInterestFactorScorer

AS_OF = date(2024, 6, 14)


def _store(rows):
    """rows = list of (ticker, settlement_iso, days_to_cover)."""
    recs = []
    for tkr, sd, dtc in rows:
        sdt = pd.Timestamp(sd)
        recs.append({
            "ticker": tkr, "settlement_date": sdt,
            "knowable_date": pd.Timestamp(sip.knowable_date(sdt.date(), sip.SI_PUBLICATION_LAG_BDAYS)),
            "short_interest": 1e6, "avg_daily_volume": 1e5, "days_to_cover": dtc,
        })
    return pd.DataFrame(recs, columns=sip._SI_COLS)


def _bars():
    idx = pd.date_range(end=pd.Timestamp(AS_OF), periods=10, freq="B")
    return pd.DataFrame({"close": 100.0}, index=idx)


def _symbols_data(tickers, vix=None):
    sd = {t: _bars() for t in tickers}
    if vix is not None:
        sd["^VIX"] = pd.DataFrame({"close": vix}, index=pd.date_range(end=pd.Timestamp(AS_OF), periods=5, freq="B"))
    return sd


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(sip, "load_short_interest", lambda refresh=False: store)


def test_long_low_short_high_dtc(monkeypatch):
    # 6 names, DTC 0.5..6; n_per_leg=2 -> short top-2 (6,5), long bottom-2 (0.5,1)
    rows = [("A", "2024-05-15", 0.5), ("B", "2024-05-15", 1.0), ("C", "2024-05-15", 2.5),
            ("D", "2024-05-15", 3.5), ("E", "2024-05-15", 5.0), ("F", "2024-05-15", 6.0)]
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=2.0)
    out = sc(AS_OF, _symbols_data(list("ABCDEF")))
    longs = {s for s, c, d in out if d == "long"}
    shorts = {s for s, c, d in out if d == "short"}
    assert shorts == {"E", "F"}      # highest DTC
    assert longs == {"A", "B"}       # lowest DTC
    # dollar-neutral: equal counts, signed confidence
    assert len(longs) == len(shorts) == 2
    assert all(c == -1.0 for s, c, d in out if d == "short")
    assert all(c == 1.0 for s, c, d in out if d == "long")


def test_min_dtc_short_floor(monkeypatch):
    # all names have low DTC < min_dtc_short -> no shorts qualify -> empty book
    rows = [(t, "2024-05-15", 0.5) for t in "ABCDEF"]
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=2.0)
    assert sc(AS_OF, _symbols_data(list("ABCDEF"))) == []


def test_long_short_disjoint(monkeypatch):
    rows = [("A", "2024-05-15", 0.5), ("B", "2024-05-15", 1.0), ("C", "2024-05-15", 5.0),
            ("D", "2024-05-15", 6.0)]
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=2.0)
    out = sc(AS_OF, _symbols_data(list("ABCD")))
    longs = {s for s, c, d in out if d == "long"}
    shorts = {s for s, c, d in out if d == "short"}
    assert longs.isdisjoint(shorts)


def test_short_only_mode(monkeypatch):
    rows = [("A", "2024-05-15", 0.5), ("B", "2024-05-15", 1.0), ("C", "2024-05-15", 5.0),
            ("D", "2024-05-15", 6.0)]
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=2.0, long_short=False)
    out = sc(AS_OF, _symbols_data(list("ABCD")))
    assert all(d == "short" for s, c, d in out)
    assert {s for s, c, d in out} == {"C", "D"}


def test_too_few_names_returns_empty(monkeypatch):
    rows = [("A", "2024-05-15", 0.5), ("B", "2024-05-15", 6.0)]  # < 2*n_per_leg
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=2.0)
    assert sc(AS_OF, _symbols_data(list("AB"))) == []


def test_vix_crisis_blocks(monkeypatch):
    rows = [(t, "2024-05-15", float(i + 1)) for i, t in enumerate("ABCDEF")]
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=1.0, vix_block_all=30.0)
    assert sc(AS_OF, _symbols_data(list("ABCDEF"), vix=45.0)) == []


def test_pit_excludes_undisseminated(monkeypatch):
    # 2024-06-14 settlement is knowable ~2024-06-28 (> as_of) -> excluded; only the
    # 2024-05-15 print is knowable, so the cross-section uses that one.
    rows = [("A", "2024-05-15", 0.5), ("B", "2024-05-15", 6.0),
            ("C", "2024-05-15", 1.0), ("D", "2024-05-15", 5.0),
            ("A", "2024-06-14", 9.9), ("B", "2024-06-14", 0.1)]  # not yet knowable on 06-14
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=2.0)
    out = sc(AS_OF, _symbols_data(list("ABCD")))
    shorts = {s for s, c, d in out if d == "short"}
    # If the future 06-14 print leaked, A (9.9) would be a short. It must NOT.
    assert "A" not in shorts
    assert shorts == {"B", "D"}  # from the knowable 05-15 print (6.0, 5.0)


def test_restricts_to_tradable_universe(monkeypatch):
    rows = [(t, "2024-05-15", float(i + 1)) for i, t in enumerate("ABCDEF")]
    _patch_store(monkeypatch, _store(rows))
    sc = ShortInterestFactorScorer(n_per_leg=2, min_dtc_short=1.0)
    # only A..D are tradable (in symbols_data); E,F absent -> excluded from ranking
    out = sc(AS_OF, _symbols_data(list("ABCD")))
    syms = {s for s, c, d in out}
    assert syms <= set("ABCD")
