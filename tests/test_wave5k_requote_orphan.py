"""Alpha-v10 audit Wave 5k — re-quote/escalation limit orders must carry an idempotency key.

The entry limit placements pass a client_order_id, but the re-quote and escalation REPLACEMENTS did
not. After cancelling the old order they placed a fresh limit with no key, so a lost-response retry
left an untracked live order resting at the broker (orphan). Fix: a deterministic requote_order_id
keyed on (trade_id, generation) + a recovery retry that reuses the same key to recover a resting
order instead of orphaning it.
"""
from __future__ import annotations

import logging
import types

from app.live_trading.order_ids import requote_order_id, exit_order_id, idempotency_key


# ── deterministic, distinct keys ─────────────────────────────────────────────────
def test_requote_key_stable_per_generation():
    a = requote_order_id(42, "AAPL", 1)
    b = requote_order_id(42, "AAPL", 1)
    assert a == b                                  # same generation retry -> same key (dedups)


def test_requote_key_distinct_across_generations_and_esc():
    k0 = requote_order_id(42, "AAPL", 0)
    k1 = requote_order_id(42, "AAPL", 1)
    kesc = requote_order_id(42, "AAPL", "esc")
    assert len({k0, k1, kesc}) == 3                # each generation/escalation is a distinct order


def test_requote_key_distinct_per_trade_and_namespace():
    assert requote_order_id(1, "AAPL", 1) != requote_order_id(2, "AAPL", 1)
    # distinct namespace from market/exit keys -> no cross-collision
    assert not requote_order_id(1, "AAPL", 1).startswith(("x", "trend", "cash"))
    assert requote_order_id(1, "AAPL", 1) != exit_order_id(1, "AAPL", "full")
    assert requote_order_id(1, "AAPL", 1) != idempotency_key("trend", "AAPL")


# ── _place_replacement_limit recovers a resting order on a lost response ──────────
def _trader():
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    return t


def test_replacement_passes_client_order_id():
    t = _trader()
    seen = {}
    alpaca = types.SimpleNamespace(
        place_limit_order=lambda *a, **k: seen.update(k) or {"order_id": "o1"})
    out = t._place_replacement_limit(alpaca, "AAPL", 10, "buy", 100.0, "rq1-abc")
    assert out["order_id"] == "o1"
    assert seen.get("client_order_id") == "rq1-abc"   # key IS passed (was missing -> orphan)


def test_replacement_recovers_on_lost_response():
    t = _trader()
    calls = {"n": 0}

    def _place(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("timeout — response lost")   # first place: response lost
        # retry with the SAME client_order_id hits idempotent-reuse -> returns the resting order
        return {"order_id": "resting", "idempotent_reuse": True,
                "client_order_id": k.get("client_order_id")}

    alpaca = types.SimpleNamespace(place_limit_order=_place)
    out = t._place_replacement_limit(alpaca, "AAPL", 10, "buy", 100.0, "rq1-abc")
    assert out["order_id"] == "resting"               # recovered, not orphaned
    assert calls["n"] == 2                            # retried once with the same key


def test_replacement_reraises_when_truly_unplaceable():
    t = _trader()

    def _place(*a, **k):
        raise RuntimeError("hard reject")

    alpaca = types.SimpleNamespace(place_limit_order=_place)
    try:
        t._place_replacement_limit(alpaca, "AAPL", 10, "buy", 100.0, "rq1-abc")
        assert False, "expected re-raise when the order genuinely cannot be placed"
    except RuntimeError:
        pass
