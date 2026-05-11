"""
P2 tests: RegimeGate for swing entry blocking.

Tests cover:
- is_blocked() correctly identifies adverse-regime days
- fail_open behavior when data unavailable
- build_blocked_dates returns subset for date range
- threshold sensitivity (lower threshold = fewer blocked days)
- from_df factory works correctly
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _make_macro_df(n_days: int = 300, vix_trend: str = "flat") -> pd.DataFrame:
    """Create synthetic macro_history DataFrame for testing."""
    dates = pd.date_range(end=date.today(), periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    spy = 400 + np.cumsum(rng.normal(0.5, 2.0, n_days))
    # Alternate bull/bear to create varied regime scores
    vix = 18 + 8 * np.sin(np.linspace(0, 4 * np.pi, n_days))
    vix3m = vix * (1.05 - 0.1 * np.sin(np.linspace(0, 6 * np.pi, n_days)))
    rsp = 60 + np.cumsum(rng.normal(0.3, 1.5, n_days))
    hyg = 80 + np.cumsum(rng.normal(0.02, 0.4, n_days))
    ief = 95 + np.cumsum(rng.normal(0.01, 0.3, n_days))

    return pd.DataFrame({
        "date": dates,
        "spy": spy,
        "vix": np.clip(vix, 12, 60),
        "vix3m": np.clip(vix3m, 12, 70),
        "rsp": rsp,
        "hyg": hyg,
        "ief": ief,
    })


# ── RegimeGate unit tests ─────────────────────────────────────────────────────

def test_from_df_loads_without_error():
    from app.risk.regime_gate import RegimeGate
    macro_df = _make_macro_df()
    gate = RegimeGate.from_df(macro_df)
    assert gate.n_loaded > 0
    assert gate._loaded is True


def test_is_blocked_returns_bool():
    from app.risk.regime_gate import RegimeGate
    macro_df = _make_macro_df(300)
    gate = RegimeGate.from_df(macro_df)
    today = date.today()
    result = gate.is_blocked(today)
    assert isinstance(result, bool)


def test_missing_date_fails_open():
    from app.risk.regime_gate import RegimeGate, RegimeGateConfig
    macro_df = _make_macro_df(300)
    config = RegimeGateConfig(fail_open=True)
    gate = RegimeGate.from_df(macro_df, config=config)
    # A date far in the future won't be in the data
    far_future = date.today() + timedelta(days=365 * 5)
    assert gate.is_blocked(far_future) is False  # fail-open → not blocked


def test_missing_date_fails_closed():
    from app.risk.regime_gate import RegimeGate, RegimeGateConfig
    macro_df = _make_macro_df(300)
    config = RegimeGateConfig(fail_open=False)
    gate = RegimeGate.from_df(macro_df, config=config)
    far_future = date.today() + timedelta(days=365 * 5)
    assert gate.is_blocked(far_future) is True  # fail-closed → blocked


def test_lower_threshold_fewer_blocked():
    from app.risk.regime_gate import RegimeGate, RegimeGateConfig
    macro_df = _make_macro_df(300)
    gate_strict = RegimeGate.from_df(macro_df, RegimeGateConfig(threshold=0.7))
    gate_lenient = RegimeGate.from_df(macro_df, RegimeGateConfig(threshold=0.2))
    # stricter threshold blocks more days
    assert gate_strict.n_blocked >= gate_lenient.n_blocked


def test_build_blocked_dates_respects_range():
    from app.risk.regime_gate import RegimeGate, RegimeGateConfig
    macro_df = _make_macro_df(300)
    gate = RegimeGate.from_df(macro_df, RegimeGateConfig(threshold=0.4))
    end = date.today()
    start = end - timedelta(days=60)
    blocked_in_range = gate.build_blocked_dates(start, end)
    # all returned dates must be in range
    assert all(start <= d <= end for d in blocked_in_range)


def test_score_returns_float_for_known_date():
    from app.risk.regime_gate import RegimeGate
    macro_df = _make_macro_df(300)
    gate = RegimeGate.from_df(macro_df)
    known_date = sorted(gate._scores.keys())[100]
    s = gate.score(known_date)
    assert s is not None
    assert 0.0 <= s <= 1.0


def test_score_returns_none_for_unknown_date():
    from app.risk.regime_gate import RegimeGate
    macro_df = _make_macro_df(300)
    gate = RegimeGate.from_df(macro_df)
    far_future = date.today() + timedelta(days=3650)
    assert gate.score(far_future) is None


def test_build_regime_gate_factory():
    from app.risk.regime_gate import build_regime_gate
    macro_df = _make_macro_df(300)
    gate = build_regime_gate(macro_df=macro_df, threshold=0.4)
    assert gate.n_loaded > 0


def test_unloaded_gate_fails_open():
    from app.risk.regime_gate import RegimeGate
    gate = RegimeGate()
    # Not loaded → fail_open=True → is_blocked() = False
    assert gate.is_blocked(date.today()) is False
