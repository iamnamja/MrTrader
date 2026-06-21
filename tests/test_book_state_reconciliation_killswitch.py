"""R0.4 tests — consolidated book-state + factor view, reconciliation-before-trade, kill-switch.

These are safety-critical even in shadow (they will later gate live money), so the factor
aggregation across venues, the fail-closed reconciliation, and the kill-switch transition rules are
pinned tightly.
"""
from __future__ import annotations

import pytest

from app.live_trading import book_state as bs
from app.live_trading import instrument_master as im
from app.live_trading import reconciliation as rec
from app.live_trading import kill_switch_state as ks
from app.live_trading.broker_adapter import AccountState, CanonicalPosition


def _pos(iid, venue, qty, price, mult=1.0, cash=False):
    return CanonicalPosition(instrument_id=iid, venue=venue, broker_symbol=iid.replace("FUT.", ""),
                             asset_class=("CASH_ETF" if cash else "ETF"), quantity=qty, price=price,
                             multiplier=mult, currency="USD", market_value=qty * price * mult,
                             notional=abs(qty) * price * mult, mapped=True)


class _FakeAdapter:
    def __init__(self, venue, nav, positions):
        self.venue = venue
        self._nav = nav
        self._positions = positions

    def health(self):
        from app.live_trading.broker_adapter import BrokerHealth
        return BrokerHealth(self.venue, True, True)

    def get_account(self):
        return AccountState(venue=self.venue, nav=self._nav, cash=self._nav * 0.5,
                            buying_power=self._nav)

    def get_positions(self):
        return self._positions

    def normalize_instrument(self, s):
        return s


# ---- book_state ----
def test_factor_view_aggregates_equity_beta_across_venues():
    # SPY on Alpaca + ES on IBKR -> stacked equity beta the per-trade RM could never see
    alpaca = _FakeAdapter(im.ALPACA, 100_000, [_pos("SPY", im.ALPACA, 100, 500.0)])     # $50k beta1
    ibkr = _FakeAdapter(im.IBKR, 100_000, [_pos("FUT.ES", im.IBKR, 1, 5000.0, mult=50.0)])  # $250k beta1
    book = bs.build_book_state([alpaca, ibkr])
    assert book.total_nav == 200_000
    assert abs(book.factor_exposures[bs.EQUITY_BETA] - (50_000 + 250_000)) < 1e-6   # netted across venues
    assert book.gross_notional == 50_000 + 250_000


def test_cash_equivalents_excluded_from_gross():
    a = _FakeAdapter(im.ALPACA, 100_000, [
        _pos("SPY", im.ALPACA, 100, 500.0),
        _pos("SGOV", im.ALPACA, 400, 100.0, cash=True),
    ])
    book = bs.build_book_state([a])
    assert book.cash_equiv_value == 40_000
    assert book.gross_notional == 50_000                # SGOV excluded
    assert bs.EQUITY_BETA in book.factor_exposures


def test_unmapped_factor_instrument_flagged():
    a = _FakeAdapter(im.ALPACA, 100_000, [_pos("WEIRD", im.ALPACA, 10, 50.0)])
    book = bs.build_book_state([a])
    assert "WEIRD" in book.unmapped_factor_instruments  # fail-closed candidate at the risk gate


def test_offsetting_equity_beta_nets():
    a = _FakeAdapter(im.ALPACA, 100_000, [_pos("SPY", im.ALPACA, 100, 500.0)])          # +50k
    b = _FakeAdapter(im.IBKR, 100_000, [_pos("FUT.ES", im.IBKR, -1, 1000.0, mult=50.0)])  # -50k
    book = bs.build_book_state([a, b])
    assert abs(book.factor_exposures[bs.EQUITY_BETA]) < 1e-6   # nets to ~0


# ---- reconciliation ----
def test_reconcile_exact_match_ok():
    actual = [_pos("SPY", im.ALPACA, 100, 500.0)]
    r = rec.reconcile({(im.ALPACA, "SPY"): 100}, actual,
                      expected_cash=50_000, actual_cash=50_000, nav=100_000)
    assert r.status == rec.MATCH and r.ok_to_trade


def test_reconcile_position_break_fails_closed():
    actual = [_pos("SPY", im.ALPACA, 90, 500.0)]        # broker holds 90, DB expects 100
    r = rec.reconcile({(im.ALPACA, "SPY"): 100}, actual)
    assert r.status == rec.FAIL_CLOSED and not r.ok_to_trade
    assert r.position_breaks[0].delta == -10


def test_reconcile_pending_order_accounts_for_diff():
    actual = [_pos("SPY", im.ALPACA, 100, 500.0)]       # DB expected 80 + a pending +20 buy
    r = rec.reconcile({(im.ALPACA, "SPY"): 80}, actual, pending_qty={(im.ALPACA, "SPY"): 20})
    assert r.status == rec.MATCH


def test_reconcile_same_id_two_venues_does_not_collide():
    # the BLOCKER fix: SPY on Alpaca + FUT.ES on IBKR must both reconcile independently; an id-only
    # key would drop one. Here Alpaca SPY matches but IBKR ES is a break -> FAIL_CLOSED (caught).
    actual = [_pos("SPY", im.ALPACA, 100, 500.0), _pos("FUT.ES", im.IBKR, 1, 5000.0, mult=50.0)]
    r = rec.reconcile({(im.ALPACA, "SPY"): 100, (im.IBKR, "FUT.ES"): 2}, actual)
    assert r.status == rec.FAIL_CLOSED
    assert any(b.venue == im.IBKR and b.delta == -1 for b in r.position_breaks)


def test_reconcile_untracked_broker_position_fails_closed():
    actual = [_pos("AAPL", im.ALPACA, 5, 200.0)]        # broker has it, DB doesn't
    r = rec.reconcile({}, actual)
    assert r.status == rec.FAIL_CLOSED


def test_reconcile_cash_within_then_over_tolerance():
    actual = [_pos("SPY", im.ALPACA, 100, 500.0)]
    ok = rec.reconcile({(im.ALPACA, "SPY"): 100}, actual,
                       expected_cash=50_000, actual_cash=50_030, nav=100_000)
    assert ok.status == rec.MATCH                        # $30 < max($100, 5bps*100k=$50)
    bad = rec.reconcile({(im.ALPACA, "SPY"): 100}, actual,
                        expected_cash=50_000, actual_cash=49_000, nav=100_000)
    assert bad.status == rec.FAIL_CLOSED and bad.cash_break == 1000.0


# ---- kill switch ----
def test_killswitch_risk_gates_by_state():
    k = ks.KillSwitch()
    assert k.state == ks.NORMAL and k.can_increase_risk() and k.allows_reduce_only()
    k.set_state(ks.HALT_NEW_RISK, reason="x", actor="t")
    assert not k.can_increase_risk() and k.allows_reduce_only()
    k.set_state(ks.FLATTEN_ALL, reason="x", actor="op", manual=True)   # flatten needs a human
    assert k.requires_flatten() and not k.can_increase_risk()


def test_killswitch_auto_cannot_reach_flatten():
    # a flaky watchdog / auto trigger must NOT be able to liquidate the book
    k = ks.KillSwitch()
    assert k.set_state(ks.FLATTEN_ALL, reason="auto", actor="bot", manual=False) is False
    assert k.set_state(ks.FLATTEN_NON_CORE, reason="auto", actor="bot", manual=False) is False
    assert k.state == ks.NORMAL
    assert k.set_state(ks.CANCEL_ONLY, reason="auto", actor="bot", manual=False) is True  # ok up to here
    assert k.set_state(ks.FLATTEN_ALL, reason="human", actor="op", manual=True) is True   # human can


def test_killswitch_auto_cannot_deescalate_manual_can():
    k = ks.KillSwitch(ks.HALT_NEW_RISK)
    assert k.set_state(ks.NORMAL, reason="auto", actor="t", manual=False) is False   # auto can't relax
    assert k.state == ks.HALT_NEW_RISK
    assert k.set_state(ks.NORMAL, reason="re-arm", actor="human", manual=True) is True
    assert k.state == ks.NORMAL


def test_killswitch_manual_lock_needs_human():
    k = ks.KillSwitch(ks.MANUAL_LOCK)
    assert k.set_state(ks.NORMAL, reason="auto", actor="t") is False
    assert k.set_state(ks.NORMAL, reason="human", actor="op", manual=True) is True


def test_dead_man_escalates_to_halt_not_flatten():
    t = {"now": 1000.0}
    k = ks.KillSwitch(clock=lambda: t["now"])
    k.heartbeat()
    t["now"] = 1000.0 + 30
    assert k.dead_man_check(max_stale_sec=60) is False   # fresh
    t["now"] = 1000.0 + 120
    assert k.dead_man_check(max_stale_sec=60) is True     # stale -> escalate
    assert k.state == ks.HALT_NEW_RISK                    # HALT, never auto-flatten


def test_reconciliation_fail_escalates_killswitch():
    k = ks.KillSwitch()
    assert k.on_reconciliation_fail() is True
    assert k.state == ks.HALT_NEW_RISK
