"""Alpha-v10 H10 — cash-ETF mapping is a SINGLE SOURCE OF TRUTH, so the whole-book gate cannot
fail-close on a legal cash config.

The landmine this closes: the gate builds the proposed book from ALL Alpaca positions and treats any
symbol that is NOT a registered cash-equivalent as risk gross AND `unmapped` -> a fail-closed breach.
`cash_sleeve.CASH_ETFS` (the tradeable cash universe) had 8 tickers; `instrument_master` knew only 3
(SGOV/BIL/SHV). If `pm.cash_universe` were ever set to one of the other 5 (e.g. USFR), ENFORCE mode
would fail-close the entire trend rebalance every week on a perfectly legal config.

These tests pin: (1) the two lists can never drift; (2) every cash ETF is registered, cash-equivalent,
and factor-mapped; (3) the behavioral regression — a held previously-missing cash ETF is excluded from
risk gross, never flagged unmapped, and does NOT produce an enforce-mode breach.
"""
from __future__ import annotations

from app.live_trading import instrument_master as im
from app.live_trading import book_state as bs
from app.live_trading import whole_book_gate as wbg
from app.live_trading.cash_sleeve import CASH_ETFS

# the 5 that instrument_master used to be missing (the actual landmine)
_PREVIOUSLY_MISSING = ["BILS", "VGSH", "GBIL", "USFR", "TBIL"]


def test_cash_universe_is_single_source_of_truth():
    # cash_sleeve.CASH_ETFS imports instrument_master.CASH_EQUIVALENT_ETFS — they CANNOT drift.
    assert set(CASH_ETFS) == set(im.CASH_EQUIVALENT_ETFS)
    # and the registry seeded exactly that set as cash-equivalents
    registered_cash = {iid for iid, inst in im.all_instruments().items() if inst.is_cash_equivalent}
    assert registered_cash == set(im.CASH_EQUIVALENT_ETFS)


def test_every_cash_etf_is_registered_cash_equivalent_and_lookup_able():
    for sym in im.CASH_EQUIVALENT_ETFS:
        assert im.is_cash_equivalent(sym), f"{sym} not flagged cash-equivalent"
        inst = im.get(sym)
        assert inst is not None and inst.asset_class == im.CASH_ETF
        assert im.lookup(im.ALPACA, sym) == sym, f"{sym} not lookup-able on Alpaca (would be unmapped)"


def test_every_cash_etf_has_a_factor_map_entry():
    # cash-equivalents are excluded from gross BEFORE the factor-map lookup, but a {} entry is
    # defense-in-depth so a cash ETF can never be flagged `unmapped` even if that order changed.
    for sym in im.CASH_EQUIVALENT_ETFS:
        assert bs.factor_loadings(sym) == {}, f"{sym} missing its zero-exposure factor-map entry"


def test_previously_missing_cash_etfs_are_now_covered():
    for sym in _PREVIOUSLY_MISSING:
        assert im.is_cash_equivalent(sym)
        assert bs.factor_loadings(sym) == {}


def test_held_cash_etf_excluded_from_gross_and_not_unmapped():
    # Hold USFR (a formerly-unknown cash ETF) — it must be cash-equivalent, not risk gross, not unmapped.
    book = wbg.build_proposed_book(
        current_positions_raw=[{"symbol": "USFR", "qty": 1000, "current_price": 50.0}],
        intents=[], prices={}, nav=100_000.0)
    assert book.cash_equiv_value == 50_000.0          # counted as cash
    assert book.gross_notional == 0.0                 # NOT risk gross
    assert "USFR" not in book.unmapped_factor_instruments


def test_enforce_mode_does_not_false_block_on_legal_cash_position():
    # The regression H10 fixes: before the fix, USFR -> unmapped -> breach -> ENFORCE blocks the
    # whole rebalance. After: the cash leg produces NO breach. A tiny trend leg keeps it a realistic
    # mixed book while staying far inside any plausible cap (decoupled from exact policy thresholds).
    v = wbg.shadow_gate_from_intents(
        current_positions_raw=[
            {"symbol": "USFR", "qty": 1000, "current_price": 50.0},   # $50k cash
            {"symbol": "SPY", "qty": 1, "current_price": 100.0},      # $100 trend — trivially in-cap
        ],
        intents=[], prices={}, nav=100_000.0, mode=wbg.ENFORCE, label="h10-test")
    # The H10-specific guarantee: the cash ETF never causes an unmapped/USFR breach (independent of
    # any unrelated risk-cap threshold). Before the fix this list contained an `unmapped [...USFR...]`.
    assert not any("unmapped" in b for b in v.breaches), v.breaches
    assert not any("USFR" in b for b in v.breaches), v.breaches
    assert v.allow is True, f"enforce false-blocked a legal cash+trend book: {v.breaches}"
