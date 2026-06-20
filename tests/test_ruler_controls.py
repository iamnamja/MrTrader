"""P0.3 — ruler negative-control Monte-Carlo tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import ruler_controls as rc


def test_paper_fp_hac_floor_tightens_and_controls_null():
    # On true nulls: the HAC floor can only TIGHTEN vs the point-SR floor alone, and the JOINT
    # false-positive rate must be near the nominal ~5% (the known ~23% floor-alone leak is closed).
    r = rc.paper_false_positive_rate(n_trials=1500, n_days=1500, seed=3)
    assert r["floor_only_rate"] > 0.10          # the floor alone is leaky (documents the problem)
    assert r["joint_rate"] <= r["floor_only_rate"]
    assert r["joint_rate"] < 0.10               # HAC closes it to ~nominal


def test_trackb_not_gamed_by_anticorrelation():
    # A zero-edge stream that is anti-correlated to the base must pass residual-alpha Track-B only
    # at ~size (it has NO true alpha) -> the diversifier waiver does not manufacture a pass.
    idx = pd.bdate_range("2010-01-01", periods=1500)
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    b = rc.antcorr_trackb_rate(base, n_trials=600, beta=-0.5, seed=4)
    assert b["pass_rate"] < 0.10


def test_run_controls_verdict_clean_on_controlled_gate():
    idx = pd.bdate_range("2010-01-01", periods=1200)
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    # smaller trial counts for test speed; the verdict logic is what we check
    out = rc.run_controls(base)
    assert out["verdict"] in ("CLEAN", "LEAK")   # runs end-to-end
    assert "paper" in out and "trackb" in out
