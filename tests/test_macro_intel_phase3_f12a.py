"""Macro Intel Phase 3 F12a — unified macro sizing on entries (PortfolioManager._apply_macro_sizing).

Behind UNIFIED_MACRO_SIZING (default OFF). When ON, the graded NIS macro global_sizing_factor folds
into per-symbol order quantity (after the news multiplier). Invariants: OFF → no quantity change (but
the factor is still recorded for audit); the factor only ever SHRINKS exposure (clamped [0.5, 1.0]);
never crashes on a missing context / zero quantity.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace


def _pm():
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = logging.getLogger("t")
    return pm


def _ctx(factor):
    return SimpleNamespace(global_sizing_factor=factor)


def _on(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.UNIFIED_MACRO_SIZING", True)


def _off(monkeypatch):
    monkeypatch.setattr("app.ml.retrain_config.UNIFIED_MACRO_SIZING", False)


def test_flag_off_records_but_does_not_apply(monkeypatch):
    _off(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(0.8), "AAPL")
    assert p["quantity"] == 100               # unchanged — no live behavior change
    assert p["nis_macro_sizing_factor"] == 0.8    # but recorded for audit/UI


def test_flag_on_scales_quantity(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(0.8), "AAPL")
    assert p["quantity"] == 80 and p["nis_macro_sizing_factor"] == 0.8


def test_factor_one_is_noop(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(1.0), "AAPL")
    assert p["quantity"] == 100 and p["nis_macro_sizing_factor"] == 1.0


def test_factor_clamped_to_floor(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(0.2), "AAPL")   # 0.2 clamps up to the 0.5 floor
    assert p["nis_macro_sizing_factor"] == 0.5 and p["quantity"] == 50


def test_factor_never_raises_exposure(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(1.5), "AAPL")   # >1 clamps to 1.0 → noop, never grows size
    assert p["quantity"] == 100 and p["nis_macro_sizing_factor"] == 1.0


def test_none_context_is_safe(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, None, "AAPL")        # getattr default 1.0 → noop, no crash
    assert p["quantity"] == 100 and p["nis_macro_sizing_factor"] == 1.0


def test_keeps_at_least_one_share(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 1}
    pm._apply_macro_sizing(p, _ctx(0.5), "AAPL")   # int(1*0.5)=0 → floored to 1
    assert p["quantity"] == 1


def test_zero_quantity_safe(monkeypatch):
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 0}
    pm._apply_macro_sizing(p, _ctx(0.8), "AAPL")   # nothing to scale
    assert p["quantity"] == 0 and p["nis_macro_sizing_factor"] == 0.8


# ── build + send contract: NIS (build) and calendar (send) never double-shrink / collide ─────
def test_flag_off_calendar_owns_sizing_no_collision(monkeypatch):
    # Flag OFF (today's behavior): build records NIS factor but does NOT change qty; the send-time
    # calendar path shrinks qty and records the calendar factor under its OWN key.
    _off(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(0.8), "AAPL")        # build (no qty change, flag off)
    pm._apply_calendar_sizing([p], 0.85)                # send (calendar shrink)
    assert p["quantity"] == 85                          # 100 × 0.85 (calendar only — no NIS shrink)
    assert p["nis_macro_sizing_factor"] == 0.8          # build audit preserved (distinct key)
    assert p["macro_sizing_factor"] == 0.85             # calendar audit, not clobbered


def test_flag_on_nis_owns_sizing_no_double_shrink(monkeypatch):
    # Flag ON: build applies the GRADED NIS factor; send-time calendar path must NOT shrink again
    # (no 0.8 × 0.85 double-count), only record the calendar factor for audit.
    _on(monkeypatch)
    pm = _pm()
    p = {"quantity": 100}
    pm._apply_macro_sizing(p, _ctx(0.8), "AAPL")        # build → 80 (NIS owns sizing)
    pm._apply_calendar_sizing([p], 0.85)                # send → records only, no further shrink
    assert p["quantity"] == 80                          # NOT 68 — no double-shrink
    assert p["nis_macro_sizing_factor"] == 0.8
    assert p["macro_sizing_factor"] == 0.85             # calendar factor still recorded for audit
