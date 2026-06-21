"""R0.5 whole-book risk-gate tests (app/live_trading/whole_book_gate.py).

The gate is the holistic risk check the live sleeves lack. These pin the cap evaluation, the
proposed-book delta logic, and — critically — the FAIL-SAFE shadow contract (a gate error must never
raise / must never block a live rebalance).
"""
from __future__ import annotations

from app.live_trading import whole_book_gate as wbg
from app.live_trading import instrument_master as im
from app.live_trading.risk_policy import RISK_POLICY_V1 as P


# ---- evaluate (caps) ----
def test_clean_book_allows():
    book = wbg.build_proposed_book([], [{"symbol": "SPY", "side": "buy", "qty": 5}],
                                   {"SPY": 746.0}, nav=100_000)
    v = wbg.evaluate(book)
    assert v.allow and not v.breaches


def test_gross_and_single_notional_breach_blocks():
    # SPY 200 @ 746 = $149.2k notional on a $100k book -> gross 149%, single 149%, beta 1.49x
    book = wbg.build_proposed_book([], [{"symbol": "SPY", "side": "buy", "qty": 200}],
                                   {"SPY": 746.0}, nav=100_000)
    v = wbg.evaluate(book)
    assert not v.allow
    assert any("gross_ex_cash" in b for b in v.breaches)
    assert any("single_notional" in b for b in v.breaches)
    assert any("net_equity_beta" in b for b in v.breaches)


def test_unmapped_instrument_blocks():
    book = wbg.build_proposed_book([], [{"symbol": "WEIRD", "side": "buy", "qty": 1}],
                                   {"WEIRD": 100.0}, nav=100_000)
    v = wbg.evaluate(book)
    assert not v.allow and any("unmapped" in b for b in v.breaches)


def test_missing_price_position_fails_closed():
    # a held position with a zero/missing price would report notional 0 and HIDE a breach -> it must
    # be a breach (fail-closed), not a silent allow (the enforce-mode blind-spot fix)
    current = [{"symbol": "SPY", "qty": 100, "current_price": 0.0}]
    book = wbg.build_proposed_book(current, [], {}, nav=100_000)
    v = wbg.evaluate(book)
    assert not v.allow and any("missing_price" in b for b in v.breaches)


def test_cash_equivalent_excluded_from_caps():
    # a huge T-bill position must NOT trip the gross/notional caps (cash-equivalent)
    book = wbg.build_proposed_book([], [{"symbol": "SGOV", "side": "buy", "qty": 900}],
                                   {"SGOV": 100.0}, nav=100_000)
    v = wbg.evaluate(book)
    assert v.allow and not v.breaches


# ---- build_proposed_book delta logic ----
def test_proposed_book_applies_buy_sell_deltas():
    current = [{"symbol": "SPY", "qty": 100, "current_price": 700.0}]
    # sell 100 SPY (-> closed) + buy 10 IWM
    book = wbg.build_proposed_book(
        current, [{"symbol": "SPY", "side": "sell", "qty": 100},
                  {"symbol": "IWM", "side": "buy", "qty": 10}],
        {"IWM": 295.0}, nav=100_000)
    syms = {p.broker_symbol: p.quantity for p in book.positions}
    assert "SPY" not in syms                            # netted to 0 -> dropped
    assert syms.get("IWM") == 10


# ---- shadow / enforce / fail-safe ----
def test_shadow_mode_returns_verdict_without_raising():
    book_intents = [{"symbol": "SPY", "side": "buy", "qty": 200}]   # breaches caps
    v = wbg.shadow_gate_from_intents([], book_intents, {"SPY": 746.0}, 100_000,
                                     mode=wbg.SHADOW, label="trend")
    assert v.mode == wbg.SHADOW
    assert v.breaches                                  # it SEES the breach
    # shadow just reports; the caller does not block on it (that's the sleeve's job in enforce only)


def test_enforce_mode_verdict_reflects_caps():
    clean = wbg.shadow_gate_from_intents([], [{"symbol": "SPY", "side": "buy", "qty": 5}],
                                         {"SPY": 746.0}, 100_000, mode=wbg.ENFORCE, label="t")
    assert clean.allow and clean.mode == wbg.ENFORCE
    breach = wbg.shadow_gate_from_intents([], [{"symbol": "SPY", "side": "buy", "qty": 200}],
                                          {"SPY": 746.0}, 100_000, mode=wbg.ENFORCE, label="t")
    assert not breach.allow


def test_gate_is_fail_safe_never_raises():
    # garbage inputs (None nav, malformed intents) must NOT raise and must fail-safe to allow=True
    v = wbg.shadow_gate_from_intents(None, [{"bad": "intent"}], None, None,
                                     mode=wbg.SHADOW, label="trend")
    assert v.allow is True and v.error is not None      # fail-safe: a gate bug can't block trading
