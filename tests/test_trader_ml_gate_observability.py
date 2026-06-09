"""Regression: the Trader's ML-score entry gate must reject VISIBLY.

Before the fix, a proposal with confidence below ML_SCORE_THRESHOLD was skipped at DEBUG
with no trader_status written, so RM-approved-but-sub-threshold proposals sat looking
"Queued" forever — the root cause of "why aren't trades firing?". This locks the visible
behavior: INFO log, ENTRY_REJECTED_ML_SCORE decision, and the symbol dropped from the queue.
"""
from __future__ import annotations

import asyncio
from unittest import mock

import pandas as pd

from app.agents.trader import Trader, MIN_BARS, ET
from app.strategy.signals import ML_SCORE_THRESHOLD


def _trader_stub(approved):
    t = object.__new__(Trader)            # bypass heavy __init__; set only what _check_entry uses
    t.active_positions = {}
    t._pending_limit_orders = {}
    t._daily_discarded_symbols = set()
    t._daily_quality_rejections = {}
    t.approved_symbols = dict(approved)
    t.logger = mock.MagicMock()
    t.log_decision = mock.AsyncMock()
    t._release_intraday_slot = mock.MagicMock()
    return t


def test_below_threshold_entry_rejected_visibly():
    sym = "AAPL"
    proposal = {"confidence": ML_SCORE_THRESHOLD - 0.04, "trade_type": "swing",
                "proposal_uuid": None}  # None uuid -> skip DB write, exercise the rest
    t = _trader_stub({sym: proposal})

    alpaca = mock.MagicMock()
    bars = pd.DataFrame({"open": [1.0] * (MIN_BARS + 5), "high": [1.0] * (MIN_BARS + 5),
                         "low": [1.0] * (MIN_BARS + 5), "close": [1.0] * (MIN_BARS + 5),
                         "volume": [1000] * (MIN_BARS + 5)})
    alpaca.get_bars.return_value = bars

    with mock.patch("app.agents.trader.circuit_breaker.is_strategy_paused", return_value=False), \
         mock.patch("app.agents.premarket.premarket_intel.is_swing_blocked", return_value=False):
        asyncio.run(t._check_entry(sym, proposal, alpaca))

    # Dropped from the queue (ml_score is fixed -> can never clear; must not churn).
    assert sym not in t.approved_symbols
    # Logged a visible decision with the right type.
    assert t.log_decision.await_count == 1
    assert t.log_decision.await_args.args[0] == "ENTRY_REJECTED_ML_SCORE"
    # INFO log (not the old silent DEBUG).
    assert t.logger.info.called and not t.logger.debug.called


def test_at_or_above_threshold_passes_ml_gate():
    # A proposal AT the threshold must NOT be rejected by the ML gate (it proceeds past it;
    # we stop it right after by making generate_signal unavailable so it raises -> caught).
    sym = "MSFT"
    proposal = {"confidence": ML_SCORE_THRESHOLD, "trade_type": "swing", "proposal_uuid": None}
    t = _trader_stub({sym: proposal})
    alpaca = mock.MagicMock()
    bars = pd.DataFrame({"open": [1.0] * (MIN_BARS + 5), "high": [1.0] * (MIN_BARS + 5),
                         "low": [1.0] * (MIN_BARS + 5), "close": [1.0] * (MIN_BARS + 5),
                         "volume": [1000] * (MIN_BARS + 5)})
    alpaca.get_bars.return_value = bars
    with mock.patch("app.agents.trader.circuit_breaker.is_strategy_paused", return_value=False), \
         mock.patch("app.agents.premarket.premarket_intel.is_swing_blocked", return_value=False), \
         mock.patch("app.agents.trader.generate_signal", side_effect=RuntimeError("stop here")):
        try:
            asyncio.run(t._check_entry(sym, proposal, alpaca))
        except RuntimeError:
            pass
    # The ML gate did NOT fire (no ENTRY_REJECTED_ML_SCORE decision logged).
    rejected = [c for c in t.log_decision.await_args_list
                if c.args and c.args[0] == "ENTRY_REJECTED_ML_SCORE"]
    assert not rejected
    assert ET is not None  # sanity: module imported cleanly
