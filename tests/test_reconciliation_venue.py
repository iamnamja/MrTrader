"""R1.2 Phase 2 — reconciliation is venue-aware (expected + actual tagged by the ACTIVE venue).

The reconciler keys breaks by (venue, instrument_id). Before Phase 2 both sides were hardcoded to
im.ALPACA. Now a sleeve passes its active venue so the two sides stay consistent after a cutover.
These pin: (1) the default is byte-identical (im.ALPACA); (2) a lower-case router venue ('ibkr', as
resolve_venue returns it) normalizes to the im constant so im.lookup still resolves; (3) expected and
actual keyed with the SAME venue reconcile MATCH (no phantom breaks).
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from app.live_trading import reconciliation as rec
from app.live_trading import instrument_master as im


class _Q:
    def __init__(self, rows):
        self._rows = rows

    def filter_by(self, **kw):
        status = kw.get("status")
        return _Q([t for t in self._rows if t.status == status] if status else self._rows)

    def all(self):
        return list(self._rows)


class FakeDB:
    def __init__(self, trades):
        self._trades = trades

    def query(self, _model):
        return _Q(self._trades)


def _trade(symbol, direction, quantity, status="ACTIVE"):
    return NS(symbol=symbol, direction=direction, quantity=quantity, status=status)


def _pos(symbol, qty, price=100.0):
    return {"symbol": symbol, "qty": qty, "current_price": price, "market_value": qty * price}


# ── default is byte-identical (im.ALPACA) ─────────────────────────────────────────────
def test_default_venue_is_alpaca():
    db = FakeDB([_trade("SPY", "BUY", 100)])
    exp = rec.db_expected_positions(db)
    assert (im.ALPACA, "SPY") in exp
    out = rec.alpaca_actual_positions([_pos("SPY", 100)])
    assert out[0].venue == im.ALPACA


# ── a lower-case router venue normalizes to the im constant ───────────────────────────
def test_lowercase_router_venue_normalizes_to_im_constant():
    db = FakeDB([_trade("SPY", "BUY", 100)])
    exp = rec.db_expected_positions(db, "ibkr")          # as resolve_venue returns it
    assert (im.IBKR, "SPY") in exp                        # keyed under the IBKR constant, not 'ibkr'
    out = rec.alpaca_actual_positions([_pos("SPY", 100)], "ibkr")
    assert out[0].venue == im.IBKR
    assert out[0].mapped is True                          # im.lookup(IBKR, SPY) resolved (M1)


# ── expected + actual on the SAME venue reconcile MATCH (no phantom breaks) ────────────
def test_ibkr_expected_and_actual_match_no_phantom_breaks():
    db = FakeDB([_trade("SPY", "BUY", 100), _trade("QQQ", "BUY", 10)])
    result = rec.shadow_reconcile_before_trade(
        db, [_pos("SPY", 100), _pos("QQQ", 10)], nav=100000.0, venue="ibkr", mode="enforce")
    assert result.ok_to_trade is True
    assert not result.position_breaks
    # and the breaks that WOULD surface carry the IBKR venue tag
    bad = rec.shadow_reconcile_before_trade(
        db, [_pos("SPY", 999), _pos("QQQ", 10)], nav=100000.0, venue="ibkr", mode="enforce")
    assert bad.ok_to_trade is False
    assert all(b.venue == im.IBKR for b in bad.position_breaks)


# ── the futures cross-venue path (expected={}, raw=[]) is unaffected by the venue param ─
def test_explicit_expected_path_ignores_default_venue():
    db = FakeDB([_trade("SPY", "BUY", 100)])   # would be a phantom if db_expected were consulted
    result = rec.shadow_reconcile_before_trade(
        db, [], nav=100000.0, expected={}, mode="shadow")
    # expected={} means the DB book is NOT pulled -> no SPY phantom break
    assert not result.position_breaks
