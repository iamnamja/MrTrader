"""Scan-resilience hardening tests.

#1 — `_fetch_swing_features` degrades GRACEFULLY when the feature-engineering pool
     blows its 120s budget under CPU pressure: it returns the symbols that DID
     complete instead of letting the as_completed TimeoutError abort the whole scan.
#2 — the 20-day VWAP-distance feature uses a SAFE divide: no div-by-zero
     RuntimeWarning and correct values on zero-volume bars.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ── #2: safe VWAP divide ─────────────────────────────────────────────────────────

def _bars_with_zero_volume_prefix(n=60):
    rng = np.random.default_rng(0)
    px = 100 + np.cumsum(rng.normal(0, 1, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    vol[:5] = 0.0  # leading zero-volume bars -> cumulative volume starts at 0
    return pd.DataFrame(
        {"open": px, "high": px + 1, "low": px - 1, "close": px, "volume": vol}
    )


def test_vwap_distance_no_divide_warning_on_zero_volume():
    from app.ml.features import FeatureEngineer
    fe = FeatureEngineer()
    bars = _bars_with_zero_volume_prefix()
    with warnings.catch_warnings():
        # Any div-by-zero / invalid RuntimeWarning now fails the test.
        warnings.simplefilter("error", RuntimeWarning)
        result = fe.engineer_features("AAPL", bars, fetch_fundamentals=False)
    assert result is not None
    v = result.get("vwap_distance_20d")
    assert v is not None and np.isfinite(v)


def test_safe_divide_matches_legacy_np_where_semantics():
    """The np.divide(out=px.copy(), where=cum_v>0) form must equal the old
    np.where(cum_v>0, cum_tv/cum_v, px) — without the eager div-by-zero."""
    cum_tv = np.array([0.0, 200.0, 600.0, 1200.0])
    cum_v = np.array([0.0, 2.0, 4.0, 6.0])
    px = np.array([10.0, 11.0, 12.0, 13.0])
    safe = np.divide(cum_tv, cum_v, out=np.asarray(px, dtype=float).copy(), where=cum_v > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        legacy = np.where(cum_v > 0, cum_tv / cum_v, px)
    assert np.allclose(safe, legacy)
    assert safe[0] == 10.0  # zero-volume bar -> price fallback


# ── #1: feature-pool graceful degradation ────────────────────────────────────────

def _make_pm():
    from app.agents.portfolio_manager import PortfolioManager
    with patch("app.integrations.get_alpaca_client", return_value=MagicMock()):
        pm = PortfolioManager()
    pm._redis = MagicMock()
    pm.send_message = MagicMock()
    return pm


def _fetch_with_fake_as_completed(pm, n_yield: int):
    """Run _fetch_swing_features with as_completed patched to yield `n_yield`
    completed futures then raise TimeoutError (simulating the 120s budget blowing
    with only some symbols done). Returns the features dict (must NOT raise)."""
    bars = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}
    )
    mock_alpaca = MagicMock()  # _alpaca is a property -> get_alpaca_client(); patch that
    mock_alpaca.get_bars_batch.return_value = {"AAA": bars, "BBB": bars, "CCC": bars}
    pm.feature_engineer = MagicMock()
    pm.feature_engineer.engineer_features.side_effect = lambda sym, b, **k: {"f": 1.0}

    import concurrent.futures as _cf
    real_as_completed = _cf.as_completed

    def _fake_as_completed(futs, timeout=None):
        # Let the real pool finish the futures, then yield only the first n_yield
        # and raise the same TimeoutError the real as_completed would.
        done = list(real_as_completed(futs, timeout=30))
        for f in done[:n_yield]:
            yield f
        raise TimeoutError(f"{len(done) - n_yield} (of {len(done)}) futures unfinished")

    with patch("app.integrations.get_alpaca_client", return_value=mock_alpaca), \
         patch("concurrent.futures.as_completed", _fake_as_completed):
        return pm._fetch_swing_features()


def test_fetch_swing_features_partial_timeout_returns_completed_subset():
    pm = _make_pm()
    feats = _fetch_with_fake_as_completed(pm, n_yield=2)
    # Did NOT raise; kept the 2 symbols that completed before the timeout.
    assert isinstance(feats, dict)
    assert len(feats) == 2


def test_fetch_swing_features_full_timeout_returns_empty_not_raise():
    pm = _make_pm()
    feats = _fetch_with_fake_as_completed(pm, n_yield=0)
    # Worst case (nothing finished): empty dict, scan proceeds, no exception.
    assert feats == {}
