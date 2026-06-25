"""Alpha-v10 H1 — reconciliation-before-trade live-path assembly + the shadow-first fail-safe gate.

The pure reconcile() is covered by test_book_state_reconciliation_killswitch.py; here we pin the new
H1 wiring: DB-expected from active Trades (signed by direction), broker-actual from raw Alpaca dicts,
and shadow_reconcile_before_trade — MATCH vs phantom/orphan breaks, the cash note, the fail-CLOSED-
on-error guarantee (never raises), and the notifier emission on a break.
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
    return {"symbol": symbol, "qty": qty, "current_price": price,
            "market_value": qty * price}


class _Notifier:
    def __init__(self):
        self.events = []

    def enqueue(self, event_type, payload, dedup_key=None):
        self.events.append((event_type, payload))


# ── DB-expected (active Trades -> signed qty per (venue, iid)) ─────────────────────
def test_db_expected_signs_and_filters_status():
    db = FakeDB([
        _trade("SPY", "BUY", 100),
        _trade("QQQ", "SELL_SHORT", 10),
        _trade("IWM", "BUY", 5, status="CLOSED"),     # not ACTIVE -> excluded
        _trade("SPY", "BUY", 50),                      # same symbol -> summed
    ])
    exp = rec.db_expected_positions(db)
    assert exp[(im.ALPACA, "SPY")] == 150.0
    assert exp[(im.ALPACA, "QQQ")] == -10.0           # SELL_SHORT is negative
    assert (im.ALPACA, "IWM") not in exp              # CLOSED filtered out


# ── broker-actual canonicalization ────────────────────────────────────────────────
def test_alpaca_actual_canonicalizes():
    out = rec.alpaca_actual_positions([_pos("SPY", 150), _pos("ZZZZ", 3)])
    by = {p.instrument_id: p for p in out}
    assert by["SPY"].venue == im.ALPACA and by["SPY"].quantity == 150.0 and by["SPY"].mapped is True
    assert by["ZZZZ"].mapped is False                 # unknown symbol still surfaced (as a break)


# ── the shadow-first gate: MATCH / phantom / orphan ───────────────────────────────
def test_reconcile_match_ok_to_trade():
    db = FakeDB([_trade("SPY", "BUY", 150)])
    r = rec.shadow_reconcile_before_trade(db, [_pos("SPY", 150)], nav=100_000.0, label="trend")
    assert r.status == rec.MATCH and r.ok_to_trade is True


def test_reconcile_phantom_position_fails_closed():
    # DB thinks we hold SPY; broker shows nothing -> phantom -> FAIL_CLOSED
    db = FakeDB([_trade("SPY", "BUY", 150)])
    r = rec.shadow_reconcile_before_trade(db, [], nav=100_000.0, label="trend")
    assert r.status == rec.FAIL_CLOSED and r.ok_to_trade is False
    brk = r.position_breaks[0]
    assert brk.instrument_id == "SPY" and brk.expected_qty == 150.0 and brk.actual_qty == 0.0


def test_reconcile_orphan_broker_position_fails_closed():
    # broker holds QQQ the DB doesn't know -> orphan -> FAIL_CLOSED
    db = FakeDB([])
    r = rec.shadow_reconcile_before_trade(db, [_pos("QQQ", 20)], nav=100_000.0, label="cash")
    assert r.status == rec.FAIL_CLOSED and r.ok_to_trade is False
    assert r.position_breaks[0].instrument_id == "QQQ" and r.position_breaks[0].actual_qty == 20.0


def test_reconcile_short_position_matches_signed():
    db = FakeDB([_trade("QQQ", "SELL_SHORT", 10)])
    r = rec.shadow_reconcile_before_trade(db, [_pos("QQQ", -10)], nav=100_000.0)
    assert r.status == rec.MATCH and r.ok_to_trade is True


# ── held (ACTIVE) vs pending (PENDING_FILL) are split; pending widens the tolerance band ──
def test_db_held_is_active_only_pending_is_separate():
    db = FakeDB([_trade("SPY", "BUY", 100, status="PENDING_FILL"),
                 _trade("QQQ", "BUY", 5, status="ACTIVE"),
                 _trade("IWM", "BUY", 9, status="CLOSED")])     # CLOSED excluded from both
    held = rec.db_expected_positions(db)
    pending = rec.db_pending_positions(db)
    assert held == {(im.ALPACA, "QQQ"): 5.0}                    # ACTIVE only
    assert pending == {(im.ALPACA, "SPY"): 100.0}              # PENDING_FILL only
    assert (im.ALPACA, "IWM") not in held and (im.ALPACA, "IWM") not in pending


def test_pending_buy_unfilled_is_not_a_break():
    # a just-placed BUY (PENDING_FILL) the broker hasn't filled yet -> actual 0 is in [0, 100] -> OK
    db = FakeDB([_trade("SPY", "BUY", 100, status="PENDING_FILL")])
    r = rec.shadow_reconcile_before_trade(db, [], nav=100_000.0, label="trend")
    assert r.status == rec.MATCH and r.ok_to_trade is True


def test_pending_buy_filled_not_yet_recorded_is_not_a_break():
    # the PENDING_FILL order has filled at the broker (actual 100) but the DB row is still
    # PENDING_FILL -> 100 is the top of [0, 100] band -> OK (no false break in the post-fill window)
    db = FakeDB([_trade("SPY", "BUY", 100, status="PENDING_FILL")])
    r = rec.shadow_reconcile_before_trade(db, [_pos("SPY", 100)], nav=100_000.0, label="trend")
    assert r.status == rec.MATCH and r.ok_to_trade is True


def test_pending_partial_fill_within_band_is_not_a_break():
    db = FakeDB([_trade("SPY", "BUY", 100, status="PENDING_FILL")])
    r = rec.shadow_reconcile_before_trade(db, [_pos("SPY", 40)], nav=100_000.0, label="trend")
    assert r.status == rec.MATCH and r.ok_to_trade is True     # 40 in [0,100]


def test_pending_overfill_beyond_band_still_breaks():
    # broker shows MORE than held+pending -> outside the band -> real break (caught)
    db = FakeDB([_trade("SPY", "BUY", 100, status="PENDING_FILL")])
    r = rec.shadow_reconcile_before_trade(db, [_pos("SPY", 130)], nav=100_000.0, label="trend")
    assert r.status == rec.FAIL_CLOSED and r.ok_to_trade is False
    assert r.position_breaks[0].actual_qty == 130.0


def test_genuine_phantom_with_no_pending_still_breaks():
    # held ACTIVE but broker flat and NO working order -> band collapses to {150} -> break
    db = FakeDB([_trade("SPY", "BUY", 150, status="ACTIVE")])
    r = rec.shadow_reconcile_before_trade(db, [], nav=100_000.0, label="trend")
    assert r.status == rec.FAIL_CLOSED and r.ok_to_trade is False


def test_held_plus_pending_band_for_add_to_existing():
    # hold 100 ACTIVE + a working +50 add (PENDING_FILL): broker anywhere in [100,150] is OK
    db = FakeDB([_trade("SPY", "BUY", 100, status="ACTIVE"),
                 _trade("SPY", "BUY", 50, status="PENDING_FILL")])
    assert rec.shadow_reconcile_before_trade(db, [_pos("SPY", 100)], nav=1e5).status == rec.MATCH  # unfilled
    assert rec.shadow_reconcile_before_trade(db, [_pos("SPY", 150)], nav=1e5).status == rec.MATCH  # filled
    assert rec.shadow_reconcile_before_trade(db, [_pos("SPY", 90)], nav=1e5).status == rec.FAIL_CLOSED  # below band


def test_reducing_pending_sell_band(monkeypatch):
    # hold 100 ACTIVE + a working SELL-100 (PENDING_FILL, signed -100): broker anywhere in [0,100] OK
    # (unfilled=100 … fully-sold=0); a flat broker is the ACCEPTED mask (we did place the sell);
    # but the broker going SHORT (-1, beyond the band) is a real break.
    db = FakeDB([_trade("SPY", "BUY", 100, status="ACTIVE"),
                 _trade("SPY", "SELL_SHORT", 100, status="PENDING_FILL")])  # signed pending -100
    assert rec.shadow_reconcile_before_trade(db, [_pos("SPY", 100)], nav=1e5).status == rec.MATCH   # unfilled
    assert rec.shadow_reconcile_before_trade(db, [], nav=1e5).status == rec.MATCH                    # fully sold (accepted)
    assert rec.shadow_reconcile_before_trade(db, [_pos("SPY", -1)], nav=1e5).status == rec.FAIL_CLOSED  # flipped short


def test_cross_sleeve_same_symbol_sums():
    # same symbol held by two sleeves (two ACTIVE rows) nets to one (venue,iid) expectation
    db = FakeDB([_trade("SGOV", "BUY", 100, status="ACTIVE"),
                 _trade("SGOV", "BUY", 60, status="ACTIVE")])
    r = rec.shadow_reconcile_before_trade(db, [_pos("SGOV", 160)], nav=1e5, label="cash")
    assert r.status == rec.MATCH and r.ok_to_trade is True


def test_pending_fill_order_is_not_a_false_break():
    # order just placed (PENDING_FILL) + its near-instant broker fill -> MATCH, not orphan
    db = FakeDB([_trade("SPY", "BUY", 100, status="PENDING_FILL")])
    r = rec.shadow_reconcile_before_trade(db, [_pos("SPY", 100)], nav=100_000.0)
    assert r.status == rec.MATCH and r.ok_to_trade is True


# ── fail-safe: never raises; an internal error -> FAIL_CLOSED (holds in enforce) ──
def test_reconcile_internal_error_fails_closed_not_raises():
    class Boom:
        def query(self, _m):
            raise RuntimeError("db down")
    r = rec.shadow_reconcile_before_trade(Boom(), [_pos("SPY", 1)], nav=1.0, label="trend")
    assert r.status == rec.FAIL_CLOSED and r.ok_to_trade is False
    assert any("error" in n for n in r.notes)


# ── notifier fires on a break, not on a match ─────────────────────────────────────
def test_notifier_emits_on_break_only():
    n = _Notifier()
    db_break = FakeDB([_trade("SPY", "BUY", 1)])
    rec.shadow_reconcile_before_trade(db_break, [], nav=100_000.0, label="trend", notifier=n)
    assert len(n.events) == 1 and n.events[0][0] == "reconciliation_break"
    n2 = _Notifier()
    db_ok = FakeDB([_trade("SPY", "BUY", 1)])
    rec.shadow_reconcile_before_trade(db_ok, [_pos("SPY", 1)], nav=100_000.0, notifier=n2)
    assert n2.events == []                            # MATCH -> no alert


# ── extra_actual (cross-venue, e.g. IBKR) is included ─────────────────────────────
def test_extra_actual_positions_are_reconciled():
    from app.live_trading.broker_adapter import CanonicalPosition
    ibkr = CanonicalPosition(instrument_id="FUT.ES", venue=im.IBKR, broker_symbol="ES",
                             asset_class=im.FUTURE, quantity=1, price=5000.0, multiplier=50.0,
                             currency="USD", market_value=0.0, notional=250000.0, mapped=True)
    db = FakeDB([])      # DB knows nothing about the IBKR ES position -> orphan break
    r = rec.shadow_reconcile_before_trade(db, [], nav=100_000.0, extra_actual=[ibkr], label="trend")
    assert r.status == rec.FAIL_CLOSED
    assert any(b.venue == im.IBKR and b.instrument_id == "FUT.ES" for b in r.position_breaks)
