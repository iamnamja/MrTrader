"""Lock-in tests for the reusable EventEdgeStrategy harness (Alpha-v3 A0).

Two jobs:
  1. Guard that PEADStrategy (refactored to subclass EventEdgeStrategy) still
     builds the EXACT AgentSimulator kwargs of the committed +0.546 config.
  2. Verify the generic base implements the run_cpcv strategy interface and that
     a new edge needs only a scorer.
"""

import os
from datetime import date

import pytest

from scripts.walkforward.event_edge import EventEdgeStrategy


def _noop_scorer(day, symbols_data, vix_history=None):
    return []


# ── generic base: run_cpcv interface contract ─────────────────────────────────

def test_base_implements_run_cpcv_interface():
    s = EventEdgeStrategy(_noop_scorer, ["AAPL", "MSFT"], model_type="demo")
    assert s.model_type == "demo"
    assert hasattr(s, "fetch_data") and hasattr(s, "run_fold")
    # rules-based -> every test fold is OOS
    assert s.allow_in_sample is False
    assert s.model.trained_through == date.min
    assert s.symbols_data == {} and s.all_days_sorted == []


def test_base_default_sim_kwargs_minimal():
    s = EventEdgeStrategy(_noop_scorer, ["AAPL"])
    assert s._fold_sim_kwargs(None, None) == {"no_prefilters": True}


def test_base_forwards_explicit_overrides_only():
    s = EventEdgeStrategy(_noop_scorer, ["AAPL"], entry_slippage_pct=0.001,
                          max_hold_bars_override=20, no_prefilters=False)
    kw = s._fold_sim_kwargs(None, None)
    assert kw["no_prefilters"] is False
    assert kw["max_hold_bars_override"] == 20
    assert kw["entry_slippage_pct"] == 0.001
    assert "stop_slippage_pct" not in kw  # not set -> falls through to sim default


def test_fold_universe_uses_configured_index(monkeypatch):
    captured = {}

    def _fake_pit_union(index, start, end, extra_symbols=None):
        captured["index"] = index
        return ["AAPL", "MSFT"]

    def _fake_hist(start, end, trade_type=None):
        captured["trade_type"] = trade_type
        return []

    import app.data.universe_history as uh
    monkeypatch.setattr(uh, "pit_union", _fake_pit_union)
    monkeypatch.setattr(uh, "historical_trade_symbols", _fake_hist)

    s = EventEdgeStrategy(_noop_scorer, ["AAPL"])
    s.pit_index = "russell2000"
    s.pit_trade_type = "intraday"
    universe = s._fold_universe(date(2022, 1, 1), date(2023, 1, 1))
    assert universe == {"AAPL", "MSFT"}
    assert captured == {"index": "russell2000", "trade_type": "intraday"}


# ── PEADStrategy byte-identity guard ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_pead_env():
    for k in ("PEAD_MAX_HOLD_BARS", "PEAD_CONVICTION_SIZE"):
        os.environ.pop(k, None)
    yield
    for k in ("PEAD_MAX_HOLD_BARS", "PEAD_CONVICTION_SIZE"):
        os.environ.pop(k, None)


def _pead():
    import scripts.run_pead_cpcv as rp
    return rp.PEADStrategy(scorer=_noop_scorer, symbols=["AAPL"], transaction_cost_pct=0.0005)


def test_pead_default_kwargs_are_committed_config():
    # Exactly the kwargs the pre-refactor run_fold passed to AgentSimulator.
    assert _pead()._fold_sim_kwargs(None, None) == {
        "no_prefilters": True,
        "max_hold_bars_override": None,
        "pead_conviction_size": False,
    }


def test_pead_respects_env_levers():
    os.environ["PEAD_MAX_HOLD_BARS"] = "15"
    os.environ["PEAD_CONVICTION_SIZE"] = "1"
    kw = _pead()._fold_sim_kwargs(None, None)
    assert kw["max_hold_bars_override"] == 15
    assert kw["pead_conviction_size"] is True


def test_pead_slippage_only_when_set():
    import scripts.run_pead_cpcv as rp
    s = rp.PEADStrategy(scorer=_noop_scorer, symbols=["AAPL"],
                        entry_slippage_pct=0.0003, stop_slippage_pct=0.0005)
    kw = s._fold_sim_kwargs(None, None)
    assert kw["entry_slippage_pct"] == 0.0003
    assert kw["stop_slippage_pct"] == 0.0005


def test_pead_is_event_edge_subclass():
    import scripts.run_pead_cpcv as rp
    assert issubclass(rp.PEADStrategy, EventEdgeStrategy)
    assert _pead().model_type == "pead"
    assert _pead().pit_index == "russell1000"
