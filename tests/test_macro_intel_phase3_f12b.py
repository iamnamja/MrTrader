"""Macro Intel Phase 3 F12b — macro-driven exit tightening (tighten_exits).

Two parts:
  1. NIS sets MacroContext.tighten_exits ONLY on a digested adverse read (all high-impact events
     released AND net BEARISH); benign / unreleased days leave it False.
  2. Trader._apply_macro_exit_tightening (gated by MACRO_TIGHTEN_EXITS, default OFF) reuses the
     regime stop-tightening core: tightens open swing stops, never widens, never liquidates,
     fires at most once per ET day.
"""
from __future__ import annotations

import asyncio
import logging


# ── Part 1: MacroContext.tighten_exits derivation ──────────────────────────────
def _evt(et, importance="high", actual=None, estimate=None):
    return {"event_type": et, "importance": importance, "actual": actual, "estimate": estimate,
            "event_time": "12:30 UTC", "event_name": et, "prior": None}


def _svc():
    from app.news.intelligence_service import NewsIntelligenceService
    return NewsIntelligenceService()


def _stub_llm(monkeypatch, risk="LOW", sizing=1.0, block=False):
    monkeypatch.setattr("app.news.llm_scorer.macro_classify", lambda evts: {
        "risk_level": risk, "sizing_factor": sizing, "block_new_entries": block,
        "direction": "NEUTRAL", "rationale": "stub"})


def test_tighten_exits_set_on_digested_adverse(monkeypatch):
    svc = _svc()
    # hotter CPI (risk_off) AND released → digested adverse → tighten_exits True
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("CPI", actual=0.6, estimate=0.4)])
    _stub_llm(monkeypatch, risk="MEDIUM", sizing=0.85)
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    ctx = svc._build_macro_context()
    assert ctx.tighten_exits is True


def test_tighten_exits_false_on_benign(monkeypatch):
    svc = _svc()
    # cooler CPI (risk_on), released → benign → no tighten
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("CPI", actual=0.3, estimate=0.4)])
    _stub_llm(monkeypatch, risk="LOW", sizing=1.0)
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    ctx = svc._build_macro_context()
    assert ctx.tighten_exits is False


def test_tighten_exits_false_when_unreleased(monkeypatch):
    svc = _svc()
    # high-impact event NOT yet released → uncertainty, not a digested read → no tighten
    monkeypatch.setattr("app.news.sources.economic_calendar.fetch_economic_calendar",
                        lambda **k: [_evt("CPI", actual=None, estimate=0.4)])
    _stub_llm(monkeypatch, risk="HIGH", sizing=0.75, block=True)
    monkeypatch.setattr(svc, "_persist_macro", lambda *a, **k: None)
    ctx = svc._build_macro_context()
    assert ctx.tighten_exits is False


# ── Part 2: Trader._apply_macro_exit_tightening ────────────────────────────────
class _FakeAlpaca:
    def __init__(self, price, atr_bars):
        self._price = price
        self._bars = atr_bars

    def get_quote(self, sym):
        return {"mid": self._price}

    def get_latest_price(self, sym):
        return self._price

    def get_bars(self, sym, timeframe="1Day", limit=20):
        return self._bars


def _bars(high, low, close, n=20):
    import pandas as pd
    return pd.DataFrame({"high": [high] * n, "low": [low] * n, "close": [close] * n})


def _trader():
    from app.agents.trader import Trader
    t = Trader.__new__(Trader)
    t.logger = logging.getLogger("t")
    t.active_positions = {}
    t._macro_exits_tightened_date = None
    return t


class _Ctx:
    def __init__(self, tighten):
        self.tighten_exits = tighten
        self.rationale = "digested adverse"


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_flag_off_is_noop(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.MACRO_TIGHTEN_EXITS", False)
    t = _trader()
    t.active_positions = {"AAPL": {"trade_type": "swing", "direction": "BUY", "stop_price": 90.0}}
    _run(t._apply_macro_exit_tightening(_FakeAlpaca(100.0, _bars(101, 99, 100))))
    assert t.active_positions["AAPL"]["stop_price"] == 90.0   # untouched


def test_flag_on_tightens_long_stop(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.MACRO_TIGHTEN_EXITS", True)
    monkeypatch.setattr("app.news.intelligence_service.nis.get_macro_context",
                        lambda *a, **k: _Ctx(True))
    monkeypatch.setattr("app.agents.trader.Trader.log_decision",
                        lambda self, *a, **k: asyncio.sleep(0))
    t = _trader()
    # long stop at 90; current 100, ATR≈2 → tight_stop=98 > 90 → tightens up
    t.active_positions = {"AAPL": {"trade_type": "swing", "direction": "BUY", "stop_price": 90.0}}
    _run(t._apply_macro_exit_tightening(_FakeAlpaca(100.0, _bars(101, 99, 100))))
    assert t.active_positions["AAPL"]["stop_price"] > 90.0


def test_never_widens_stop(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.MACRO_TIGHTEN_EXITS", True)
    monkeypatch.setattr("app.news.intelligence_service.nis.get_macro_context",
                        lambda *a, **k: _Ctx(True))
    monkeypatch.setattr("app.agents.trader.Trader.log_decision",
                        lambda self, *a, **k: asyncio.sleep(0))
    t = _trader()
    # existing stop 99 already tighter than 1×ATR stop (98) → must NOT widen to 98
    t.active_positions = {"AAPL": {"trade_type": "swing", "direction": "BUY", "stop_price": 99.0}}
    _run(t._apply_macro_exit_tightening(_FakeAlpaca(100.0, _bars(101, 99, 100))))
    assert t.active_positions["AAPL"]["stop_price"] == 99.0   # unchanged (never loosened)


def test_no_tighten_when_ctx_false(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.MACRO_TIGHTEN_EXITS", True)
    monkeypatch.setattr("app.news.intelligence_service.nis.get_macro_context",
                        lambda *a, **k: _Ctx(False))   # not a digested-adverse day
    t = _trader()
    t.active_positions = {"AAPL": {"trade_type": "swing", "direction": "BUY", "stop_price": 90.0}}
    _run(t._apply_macro_exit_tightening(_FakeAlpaca(100.0, _bars(101, 99, 100))))
    assert t.active_positions["AAPL"]["stop_price"] == 90.0


def test_regime_path_still_tightens_via_shared_core(monkeypatch):
    # pin the shared _tighten_one_stop refactor on the REGIME side too (byte-equivalent behavior)
    monkeypatch.setattr("app.strategy.regime_detector.regime_detector.get_regime", lambda: "HIGH")
    monkeypatch.setattr("app.agents.trader.Trader.log_decision",
                        lambda self, *a, **k: asyncio.sleep(0))
    t = _trader()
    t._last_regime = "LOW"   # transition LOW → HIGH fires the tightening
    t.active_positions = {"AAPL": {"trade_type": "swing", "direction": "BUY", "stop_price": 90.0}}
    _run(t._apply_regime_stop_tightening(_FakeAlpaca(100.0, _bars(101, 99, 100))))
    assert t.active_positions["AAPL"]["stop_price"] > 90.0
    assert t._last_regime == "HIGH"


def test_once_per_day_guard(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.MACRO_TIGHTEN_EXITS", True)
    calls = {"n": 0}

    def _ctx(*a, **k):
        calls["n"] += 1
        return _Ctx(True)
    monkeypatch.setattr("app.news.intelligence_service.nis.get_macro_context", _ctx)
    monkeypatch.setattr("app.agents.trader.Trader.log_decision",
                        lambda self, *a, **k: asyncio.sleep(0))
    t = _trader()
    t.active_positions = {"AAPL": {"trade_type": "swing", "direction": "BUY", "stop_price": 90.0}}
    alp = _FakeAlpaca(100.0, _bars(101, 99, 100))
    _run(t._apply_macro_exit_tightening(alp))
    _run(t._apply_macro_exit_tightening(alp))   # second call same day → guarded
    assert calls["n"] == 1                       # macro context fetched only once today
