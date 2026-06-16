"""P1-3 — tests for the credit-overlay shadow monitor (network-free parts)."""
import numpy as np
import pandas as pd
import pytest

from scripts.shadow_credit_governor import _stats, OVERLAY_FLOOR


def test_overlay_floor_matches_live_trend_sleeve():
    """Drift guard: the shadow's combined-multiplier floor must equal the live floor,
    or the shadow P&L would diverge from what the live overlay would actually apply."""
    from app.live_trading.trend_sleeve import _OVERLAY_DERISK_FLOOR
    assert OVERLAY_FLOOR == _OVERLAY_DERISK_FLOOR


def test_stats_known_answer():
    # constant positive daily return -> zero vol -> Sharpe NaN, cum compounds
    r = pd.Series([0.01, 0.01, 0.01])
    s = _stats(r)
    assert s["n"] == 3
    assert s["cum"] == pytest.approx((1.01 ** 3) - 1)
    assert np.isnan(s["sharpe"])  # zero variance


def test_stats_drawdown_and_sharpe_sign():
    r = pd.Series([0.02, -0.05, 0.01, 0.03, 0.02])  # clearly positive mean (+0.006/day)
    s = _stats(r)
    assert s["mdd"] <= 0.0
    # positive mean -> positive Sharpe
    assert s["sharpe"] > 0
