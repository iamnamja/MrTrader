"""
ruler_controls.py — Alpha-v10 P0.3: empirical negative controls for the Ruler-v2 gate.

The panel asked us to PROVE the ruler isn't leaky (we've been wrong in both directions before):
  (A) TRUE-NULL PAPER false-positive rate. The PAPER point-SR floor (>=0.30) ALONE admits ~23%
      of zero-edge nulls at n~1500 (a known Type-I leak); the HAC-significance floor (p<0.05) was
      ADDED to close it. Control: the JOINT pass-rate (SR>=0.30 AND HAC p<0.05) on true nulls must
      be <= ~5%.
  (B) ANTI-CORRELATED ZERO-EDGE null through Track-B. The worry: the "diversifier waiver" /
      vol-matched appraisal could manufacture a PASS from pure anti-correlation. Control: a stream
      with ZERO true alpha (pure exposure to the base book + zero-mean noise) must NOT pass the
      residual-alpha Track-B more than ~size (~5%).

Pure + deterministic (seeded). Uses the canonical inference (hac_sharpe, multifactor_alpha).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

ANN = 252


def paper_false_positive_rate(*, n_trials: int = 5000, n_days: int = 1500,
                              daily_vol: float = 0.01, sr_floor: float = 0.30,
                              hac_p_max: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo the PAPER-tier core gates on TRUE nulls (mean-zero daily returns).
    Reports the floor-alone pass-rate (the known leak) vs the JOINT (floor AND HAC) rate."""
    from app.research.inference import hac_sharpe
    rng = np.random.default_rng(seed)
    floor_only = 0
    joint = 0
    for _ in range(n_trials):
        r = rng.normal(0.0, daily_vol, n_days)
        sd = r.std()
        sr = float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0
        if sr >= sr_floor:
            floor_only += 1
            if hac_sharpe(r, annualize=ANN).p_one_sided < hac_p_max:
                joint += 1
    return {"n_trials": n_trials, "n_days": n_days,
            "floor_only_rate": floor_only / n_trials, "joint_rate": joint / n_trials,
            "sr_floor": sr_floor, "hac_p_max": hac_p_max}


def antcorr_trackb_rate(base_returns: pd.Series, *, n_trials: int = 2000,
                        daily_vol: float = 0.01, beta: float = -0.5,
                        alpha_t_min: float = 1.65, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo Track-B (residual-alpha) on ZERO-EDGE streams that are (anti-)correlated to the
    base book: r = beta*base + zero-mean noise (true alpha = 0). The residual-alpha t should be
    ~N(0,1) -> pass-rate ~size. A materially higher rate = a Track-B leak."""
    from app.research.inference import multifactor_alpha
    base = base_returns.dropna()
    base_arr = base.to_numpy()
    n = len(base_arr)
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame({"base": base})
    passes = 0
    for _ in range(n_trials):
        r = pd.Series(beta * base_arr + rng.normal(0.0, daily_vol, n), index=base.index)
        t = float(multifactor_alpha(r, factors).get("t_alpha_hac", 0.0) or 0.0)
        if t >= alpha_t_min:
            passes += 1
    return {"n_trials": n_trials, "beta": beta, "alpha_t_min": alpha_t_min,
            "pass_rate": passes / n_trials}


def run_controls(base_returns: Optional[pd.Series] = None) -> Dict[str, object]:
    """Both controls + a PASS/FAIL on the frozen criterion (both rates <= ~6%)."""
    paper = paper_false_positive_rate()
    out: Dict[str, object] = {"paper": paper}
    if base_returns is not None:
        out["trackb"] = antcorr_trackb_rate(base_returns)
    paper_ok = paper["joint_rate"] <= 0.06
    trackb_ok = (out.get("trackb", {}).get("pass_rate", 0.0) <= 0.06) if base_returns is not None else True
    out["verdict"] = "CLEAN" if (paper_ok and trackb_ok) else "LEAK"
    return out
