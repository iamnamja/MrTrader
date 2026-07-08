"""CH4 ranging-MR sleeve — pure-piece tests (app/research/ch4_mr_sleeve.py).

Pins the reversal signal, the ranging-regime mask, and the PIT (no-look-ahead) property of the
gated book returns. The heavy CPCV/Track-B gates are exercised by running the module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research import ch4_mr_sleeve as ch4


def _cfg(**kw):
    base = dict(universe=["A", "B"], lookback=5, z_enter=1.0, clarity_lo=0.40, vol_pctl=0.50,
                z_window=20, book_vol_window=20)
    base.update(kw)
    return ch4.MRConfig(**base)


def test_mr_signal_positive_when_oversold():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    # A: steady then a sharp recent DROP -> oversold -> z > 0 at the end
    a = np.concatenate([np.full(70, 100.0) + np.linspace(0, 1, 70), np.linspace(101, 90, 10)])
    b = pd.Series(100.0 + np.sin(np.arange(80) / 3.0), index=idx)   # choppy
    closes = pd.DataFrame({"A": pd.Series(a, index=idx), "B": b})
    z = ch4.mr_signal(closes, _cfg())
    assert z["A"].dropna().iloc[-1] > 0     # recent drop -> oversold -> buy signal


def test_ranging_mask_true_only_when_low_clarity_and_low_vol(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    # calm, low-vol prices -> low book_vol; monkeypatch clarity to LOW so both conditions hold
    calm = pd.DataFrame({"A": 100 + np.cumsum(np.full(300, 0.001)),
                         "B": 100 + np.cumsum(np.full(300, 0.0005))}, index=idx)
    monkeypatch.setattr("app.research.ch2_sizing.trend_clarity",
                        lambda p, c: pd.Series(0.1, index=p.index))   # LOW clarity everywhere
    mask = ch4.ranging_mask(calm, _cfg())
    assert mask.dtype == bool
    assert mask.iloc[-1]                    # low clarity + low vol -> ranging

    monkeypatch.setattr("app.research.ch2_sizing.trend_clarity",
                        lambda p, c: pd.Series(0.9, index=p.index))   # HIGH clarity (trending)
    assert not ch4.ranging_mask(calm, _cfg()).iloc[-1]   # high clarity -> NOT ranging


def test_mr_book_returns_is_pit_no_future_leak(monkeypatch):
    idx = pd.date_range("2019-01-01", periods=340, freq="B")   # > 252 so the regime mask is active
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 0.5, 340)),
                       "B": 100 + np.cumsum(rng.normal(0, 0.5, 340))}, index=idx)
    monkeypatch.setattr("app.research.ch2_sizing.trend_clarity",
                        lambda p, c: pd.Series(0.1, index=p.index))
    net0, _, _ = ch4.mr_book_returns(px, _cfg())
    px2 = px.copy()
    px2.iloc[300:] *= 1.10                  # perturb ONLY the future (rows >= 300)
    net1, _, _ = ch4.mr_book_returns(px2, _cfg())
    common = net0.index[net0.index < idx[298]]
    assert len(common) > 40
    assert np.allclose(net0.loc[common].to_numpy(), net1.loc[common].to_numpy())


def test_extra_lag_shifts_the_regime_mask(monkeypatch):
    idx = pd.date_range("2019-01-01", periods=340, freq="B")   # > 252 so the mask has True days
    px = pd.DataFrame({"A": 100 + np.cumsum(np.random.default_rng(4).normal(0, 0.5, 340)),
                       "B": 100 + np.cumsum(np.random.default_rng(5).normal(0, 0.5, 340))}, index=idx)
    monkeypatch.setattr("app.research.ch2_sizing.trend_clarity",
                        lambda p, c: pd.Series(0.1, index=p.index))
    _, m0, _ = ch4.mr_book_returns(px, _cfg(), extra_lag=0)
    _, m2, _ = ch4.mr_book_returns(px, _cfg(), extra_lag=2)
    assert m0.any() and not m0.equals(m2)   # mask has active days AND the +2 lag shifts it
