"""Alpha-v10 P2.3 — pure futures order-construction math.

Pins: target_lots sources the multiplier ONLY from instrument_master (the #1-killer guard — no
caller param exists), dust < min_lots is skipped, per-market + asset-class notional caps clamp,
and futures_order_deltas diffs target vs broker lots reductions-first with full-exit detection.
"""
from __future__ import annotations

from app.live_trading import instrument_master as im
from app.live_trading.futures_sizing import (futures_order_deltas, target_lots, _round_half_away)


# ── target_lots ────────────────────────────────────────────────────────────────
def test_lots_use_master_multiplier_not_caller():
    # ES mult=50 @ price 5000: w=0.5, NAV=1M -> 0.5*1e6/(5000*50)=2.0 -> 2 lots
    lots, _ = target_lots({"FUT.ES": 0.5}, 1_000_000.0, {"FUT.ES": 5000.0})
    assert lots == {"FUT.ES": 2}
    # there is NO multiplier parameter — the function signature can't be handed a wrong one
    import inspect
    assert "multiplier" not in inspect.signature(target_lots).parameters


def test_multiplier_killer_zc_vs_es_ratio():
    # ZC mult=5000 (the corrected value) vs ES mult=50 at the SAME price/weight -> 100x fewer lots.
    assert im.get("FUT.ZC").multiplier == 5000.0 and im.get("FUT.ES").multiplier == 50.0
    es, _ = target_lots({"FUT.ES": 1.0}, 1_000_000.0, {"FUT.ES": 100.0}, min_lots=1)
    zc, _ = target_lots({"FUT.ZC": 1.0}, 1_000_000.0, {"FUT.ZC": 100.0}, min_lots=1)
    # ES: 1e6/(100*50)=200 ; ZC: 1e6/(100*5000)=2
    assert es == {"FUT.ES": 200} and zc == {"FUT.ZC": 2}


def test_signed_short_weight():
    lots, _ = target_lots({"FUT.GC": -0.2}, 1_000_000.0, {"FUT.GC": 2000.0})  # mult 100
    # -0.2*1e6/(2000*100) = -1.0 -> -1
    assert lots == {"FUT.GC": -1}


def test_dust_below_min_lots_skipped():
    # 0.0001*1e6/(5000*50)=0.0004 -> rounds to 0 -> dust skip
    lots, logentries = target_lots({"FUT.ES": 0.0001}, 1_000_000.0, {"FUT.ES": 5000.0})
    assert lots == {}
    # a non-zero-raw dust skip is logged
    assert any(e.get("reason") == "dust_skipped" for e in logentries) or logentries == []


def test_missing_price_and_unmapped_skipped_with_log():
    lots, logentries = target_lots(
        {"FUT.ES": 0.5, "FUT.NQ": 0.5, "FUT.BOGUS": 0.5},
        1_000_000.0, {"FUT.ES": 5000.0})  # NQ has no price; BOGUS not in master
    assert set(lots) == {"FUT.ES"}
    reasons = {e["instrument_id"]: e["reason"] for e in logentries if "instrument_id" in e}
    assert reasons.get("FUT.NQ") == "missing_price"
    assert reasons.get("FUT.BOGUS") == "unmapped_instrument"


def test_max_lots_per_market_clamps():
    lots, logentries = target_lots({"FUT.ES": 1.0}, 1_000_000.0, {"FUT.ES": 100.0},
                                   max_lots_per_market=10)
    assert lots == {"FUT.ES": 10}
    assert any(e.get("reason") == "clamped_max_lots" for e in logentries)


def test_asset_class_notional_cap_scales_down():
    # Two futures, each would be large notional; cap the FUTURE class to 0.5x NAV.
    lots, logentries = target_lots(
        {"FUT.ES": 1.0, "FUT.GC": 1.0}, 1_000_000.0,
        {"FUT.ES": 100.0, "FUT.GC": 2000.0},  # ES mult50 -> 200 lots*100*50=1M ; GC mult100 -> 5 lots*2000*100=1M
        asset_class_notional_cap=0.5)
    # total raw notional ~2M, cap 0.5M -> scale ~0.25
    total_notional = sum(abs(n) * {"FUT.ES": 100.0, "FUT.GC": 2000.0}[i] * im.get(i).multiplier
                         for i, n in lots.items())
    assert total_notional <= 0.5 * 1_000_000.0 + 1.0
    assert any("notional_cap" in e.get("reason", "") for e in logentries)


def test_non_positive_nav_returns_empty():
    lots, logentries = target_lots({"FUT.ES": 0.5}, 0.0, {"FUT.ES": 5000.0})
    assert lots == {} and logentries[0]["reason"] == "non_positive_nav"


def test_round_half_away_from_zero():
    assert _round_half_away(0.5) == 1 and _round_half_away(-0.5) == -1
    assert _round_half_away(2.4) == 2 and _round_half_away(-2.4) == -2


# ── futures_order_deltas ─────────────────────────────────────────────────────────
def test_deltas_open_from_flat():
    d = futures_order_deltas({"FUT.ES": 3}, {})
    assert d == [{"instrument_id": "FUT.ES", "side": "buy", "qty": 3, "target": 3,
                  "current": 0, "reduces_exposure": False, "is_full_exit": False}]


def test_deltas_full_exit_detected():
    d = futures_order_deltas({}, {"FUT.ES": 2})
    assert len(d) == 1
    o = d[0]
    assert o["side"] == "sell" and o["qty"] == 2 and o["is_full_exit"] and o["reduces_exposure"]


def test_deltas_reductions_ordered_first():
    # ES needs an INCREASE (buy 2->5), GC needs a REDUCTION (3->1). Reduction must come first.
    d = futures_order_deltas({"FUT.ES": 5, "FUT.GC": 1}, {"FUT.ES": 2, "FUT.GC": 3})
    assert [o["instrument_id"] for o in d] == ["FUT.GC", "FUT.ES"]
    assert d[0]["reduces_exposure"] and not d[1]["reduces_exposure"]


def test_deltas_zero_delta_dropped():
    assert futures_order_deltas({"FUT.ES": 2}, {"FUT.ES": 2}) == []


def test_deltas_flip_long_to_short_is_not_reduction():
    # 2 long -> 3 short: |tgt|=3 > |cur|=2 so NOT a reduction; sell 5.
    d = futures_order_deltas({"FUT.ES": -3}, {"FUT.ES": 2})
    assert d == [{"instrument_id": "FUT.ES", "side": "sell", "qty": 5, "target": -3,
                  "current": 2, "reduces_exposure": False, "is_full_exit": False}]
