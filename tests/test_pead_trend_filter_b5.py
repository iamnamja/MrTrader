"""B5 SPY-trend-filter tests: scorer regime control + config registration.

Validated 2026-06-04: trend filter +0.661 vs +0.546 VIX-block (same window).
"""

from datetime import date

import numpy as np
import pandas as pd

from app.ml.pead_scorer import PEADScorer
from app.database.agent_config import _DEFAULTS, CONFIG_SCHEMA

AS_OF = date(2024, 6, 14)


def _spy(trend: str, n: int = 260):
    """Build a 260-bar SPY close series that is clearly above or below its 200d SMA."""
    idx = pd.date_range(end=pd.Timestamp(AS_OF), periods=n, freq="B")
    if trend == "up":
        closes = np.linspace(100, 200, n)          # rising → last >> SMA
    else:
        closes = np.linspace(200, 100, n)          # falling → last << SMA
    return pd.DataFrame({"close": closes}, index=idx)


def _scorer():
    return PEADScorer(long_short=False, vix_block_all=float("inf"),
                      regime_control="trend", regime_control_trend_ma=200,
                      regime_control_floor=0.5)


# ── regime scalar (the trend gate) ────────────────────────────────────────────

def test_scalar_1_when_spy_above_200d():
    s = _scorer()
    assert s._regime_exposure_scalar(AS_OF, {"SPY": _spy("up")}) == 1.0


def test_scalar_0_when_spy_below_200d():
    s = _scorer()
    assert s._regime_exposure_scalar(AS_OF, {"SPY": _spy("down")}) == 0.0


def test_scalar_fail_open_when_spy_missing():
    # No SPY -> fail OPEN (1.0). (Live wiring fails CLOSED to the VIX block instead.)
    s = _scorer()
    assert s._regime_exposure_scalar(AS_OF, {}) == 1.0


def test_scalar_fail_open_when_history_too_short():
    s = _scorer()
    short = _spy("down", n=120)  # < 200 bars
    assert s._regime_exposure_scalar(AS_OF, {"SPY": short}) == 1.0


def test_pit_future_drop_does_not_change_today():
    # Append a future crash AFTER as_of; the .loc[:as_of] slice must ignore it.
    s = _scorer()
    up = _spy("up")
    future = pd.DataFrame(
        {"close": [50.0, 40.0]},
        index=pd.date_range(start=pd.Timestamp(AS_OF) + pd.Timedelta(days=1), periods=2, freq="B"),
    )
    combined = pd.concat([up, future])
    assert s._regime_exposure_scalar(AS_OF, {"SPY": combined}) == 1.0


def test_call_blocks_all_entries_in_downtrend():
    # scalar 0 < floor 0.5 -> __call__ returns [] before any earnings fetch.
    s = _scorer()
    sd = {"SPY": _spy("down"), "AAPL": _spy("down")}  # AAPL df content irrelevant; blocked early
    assert s(AS_OF, sd) == []


def test_disabled_regime_is_byte_identical_default():
    # regime_control=None -> scalar always 1.0 (committed +0.546 behaviour)
    s = PEADScorer(regime_control=None)
    assert s._regime_exposure_scalar(AS_OF, {"SPY": _spy("down")}) == 1.0


# ── config registration ───────────────────────────────────────────────────────

def test_b5_config_keys_registered():
    assert _DEFAULTS["pm.pead_regime_control"] == "trend"
    assert _DEFAULTS["pm.pead_trend_ma"] == 200
    assert _DEFAULTS["pm.pead_regime_floor"] == 0.5


def test_b5_config_keys_typed():
    by_key = {s["key"]: s for s in CONFIG_SCHEMA}
    assert by_key["pm.pead_regime_control"]["type"] == "str"
    assert by_key["pm.pead_trend_ma"]["type"] == "int"
    assert by_key["pm.pead_regime_floor"]["type"] == "float"
