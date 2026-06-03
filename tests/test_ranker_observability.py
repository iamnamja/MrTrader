"""Phase 1 (Layer 1 — Observability) regression tests for the ranker validity fix.

Verifies that realized net-exposure AND realized gross are plumbed end-to-end
(FoldResult -> CPCVResult aggregation -> print panel -> JSON), so the dollar-neutral
L/S arm can be validated (was it actually neutral AND funded to target gross?).
The §3.1 treatment run was uninterpretable precisely because gross was dropped and
nothing was surfaced.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os

from scripts.walkforward.cpcv import CPCVResult
from scripts.walkforward.gates import FoldResult
from datetime import date


def _captured_result(gross_paths=(0.80, 0.78)) -> CPCVResult:
    """A CPCVResult standing in for a captured L/S arm (neutral-ish book)."""
    r = CPCVResult(model_type="swing", n_folds=6, n_paths=2)
    r.path_sharpes = [0.9, -0.4, 1.2, 0.3]
    r.path_deployments = [0.62, 0.61, 0.63, 0.62]
    r.path_deployment_adj_sharpes = [0.4, -0.2, 0.5, 0.1]
    r.net_exposure_captured = True
    r.path_mean_net_betas = [-0.06, -0.04]
    r.path_p95_abs_net_betas = [0.11, 0.09]
    r.path_max_abs_net_betas = [0.34, 0.31]
    r.path_mean_net_dollars = [0.03, 0.02]
    r.path_max_abs_net_dollars = [0.07, 0.06]
    r.path_max_abs_net_sectors = [0.08, 0.07]
    r.path_mean_grosses = list(gross_paths)
    return r


# ── FoldResult carries gross ────────────────────────────────────────────────────

def test_foldresult_has_gross_fields_defaulting_zero():
    fr = FoldResult(
        fold=1, train_start=date(2020, 1, 1), train_end=date(2021, 1, 1),
        test_start=date(2021, 1, 2), test_end=date(2021, 4, 1),
        trades=10, win_rate=0.5, sharpe=0.5, max_drawdown=-0.1,
        total_return=0.05, stop_exit_rate=0.2,
    )
    assert fr.mean_gross == 0.0  # additive, default zero
    fr2 = FoldResult(
        fold=1, train_start=date(2020, 1, 1), train_end=date(2021, 1, 1),
        test_start=date(2021, 1, 2), test_end=date(2021, 4, 1),
        trades=10, win_rate=0.5, sharpe=0.5, max_drawdown=-0.1,
        total_return=0.05, stop_exit_rate=0.2, mean_gross=0.79,
    )
    assert fr2.mean_gross == 0.79


# ── CPCVResult.mean_gross aggregation ────────────────────────────────────────────

def test_cpcv_mean_gross_property_aggregates_paths():
    r = _captured_result(gross_paths=(0.80, 0.76))
    assert abs(r.mean_gross - 0.78) < 1e-9


def test_cpcv_mean_gross_zero_when_no_capture():
    r = CPCVResult(model_type="swing", n_folds=6, n_paths=2)
    assert r.mean_gross == 0.0  # long-only / non-capture arm unaffected


# ── print() net-exposure panel ──────────────────────────────────────────────────

def test_print_emits_net_exposure_panel_when_captured():
    r = _captured_result()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r.print()
    out = buf.getvalue()
    assert "Realized net beta:" in out
    assert "Realized net dollar:" in out
    assert "Realized gross NAV:" in out
    assert "0.790" in out  # mean of (0.80, 0.78)


def test_print_omits_panel_for_long_only_arm():
    r = CPCVResult(model_type="swing", n_folds=6, n_paths=2)
    r.path_sharpes = [0.5, 0.6]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r.print()
    out = buf.getvalue()
    assert "Realized gross NAV:" not in out  # gated on net_exposure_captured


def test_print_panel_is_ascii_safe():
    # Windows console is cp1252 — the panel must not emit non-ASCII (the documented
    # crash mode). Capture and assert the net-exposure lines encode cleanly.
    r = _captured_result()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r.print()
    panel = "\n".join(l for l in buf.getvalue().splitlines() if "Realized" in l)
    panel.encode("cp1252")  # raises UnicodeEncodeError if a non-ASCII char slipped in


# ── JSON persistence ─────────────────────────────────────────────────────────────

def test_dump_cpcv_result_json_includes_validity_fields(tmp_path, monkeypatch):
    import scripts.walkforward_tier3 as t3
    monkeypatch.chdir(tmp_path)  # helper writes to ./logs/
    r = _captured_result(gross_paths=(0.80, 0.76))
    t3._dump_cpcv_result_json(r, "swing")
    files = glob.glob(str(tmp_path / "logs" / "cpcv_swing_*.json"))
    assert len(files) == 1
    d = json.load(open(files[0]))
    assert d["net_exposure_captured"] is True
    assert d["net_beta_clean"] is True
    assert abs(d["mean_gross"] - 0.78) < 1e-9
    assert d["path_mean_grosses"] == [0.80, 0.76]
    assert "gate_failed" in d
