"""P3-1 — crypto trend sleeve: provider PIT/normalization, sleeve config/shape, and the
Ruler-v2 periods_per_year (365 annualization) plumbing."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ── crypto data provider ──────────────────────────────────────────────────────
def _fake_provider(df):
    from app.data.alpaca_crypto_provider import AlpacaCryptoProvider
    prov = AlpacaCryptoProvider.__new__(AlpacaCryptoProvider)  # skip __init__/network
    prov._client = SimpleNamespace(get_crypto_bars=lambda req: SimpleNamespace(df=df))
    return prov


def _bars_df(symbols, days, price=100.0):
    rows, closes = [], []
    for s in symbols:
        for d in days:
            rows.append((s, pd.Timestamp(d, tz="UTC")))
            closes.append(price)
    idx = pd.MultiIndex.from_tuples(rows, names=["symbol", "timestamp"])
    return pd.DataFrame({"close": closes}, index=idx)


def test_provider_drops_partial_today_bar_and_normalizes():
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=2), today - timedelta(days=1), today]
    prov = _fake_provider(_bars_df(["BTC/USD", "ETH/USD"], days))
    closes = prov.get_daily_closes(["BTC/USD", "ETH/USD"])
    # today's in-progress UTC bar is dropped (PIT); only the 2 fully-closed days remain
    assert list(closes.columns) == ["BTC/USD", "ETH/USD"]
    assert len(closes) == 2
    assert pd.Timestamp(today) not in closes.index
    assert closes.index.tz is None  # naive, normalized


def test_provider_keeps_pinned_end_bar():
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=1), today]
    prov = _fake_provider(_bars_df(["BTC/USD"], days))
    closes = prov.get_daily_closes(["BTC/USD"], end=today)  # pinned end -> keep today
    assert len(closes) == 2


def test_provider_drops_all_nan_symbol():
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=2), today - timedelta(days=1)]
    df = _bars_df(["BTC/USD", "DEAD/USD"], days)
    df.loc[(df.index.get_level_values("symbol") == "DEAD/USD"), "close"] = float("nan")
    prov = _fake_provider(df)
    closes = prov.get_daily_closes(["BTC/USD", "DEAD/USD"])
    assert "BTC/USD" in closes.columns and "DEAD/USD" not in closes.columns


# ── sleeve config + builder ───────────────────────────────────────────────────
def test_crypto_trend_config_pre_registered():
    from scripts.walkforward.sleeves import crypto_trend_config
    cfg = crypto_trend_config()
    assert cfg.ann == 365                       # 365-day crypto calendar
    assert cfg.lookbacks == (30, 90, 180, 365)  # calendar-day ~1/3/6/12mo
    assert cfg.allow_short is False             # Alpaca spot long-flat
    assert cfg.book_vol_target == 0.20          # hard book vol target
    assert cfg.cost_bps == 25.0                 # conservative crypto cost
    assert "BTC/USD" in cfg.universe and "ETH/USD" in cfg.universe


def test_build_crypto_trend_sleeve_shape():
    from scripts.walkforward.sleeves import build_crypto_trend, crypto_trend_config
    cfg = crypto_trend_config()
    # synthetic trending closes for each symbol (network-free via injected prices)
    idx = pd.date_range("2021-01-01", periods=500, freq="D")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {s: 100 * np.cumprod(1 + 0.001 + rng.normal(0, 0.03, len(idx))) for s in cfg.universe},
        index=idx)
    s = build_crypto_trend(prices=prices)
    assert s.component_type == "diversifier"
    assert s.periods_per_year == 365
    assert s.registration_id == "P3-1-CRYPTO-TREND"
    assert s.n_trials_registered == 1
    assert s.spy_prices is None
    assert len(s.returns) > 0


def test_plain_sleeve_defaults_to_252():
    from scripts.walkforward.sleeve_lab import Sleeve
    s = Sleeve(label="x", component_type="alpha", returns=pd.Series([0.01, -0.01]))
    assert s.periods_per_year == 252


# ── C1: Ruler-v2 point_SR respects periods_per_year ───────────────────────────
def test_ruler_v2_point_sr_scales_with_periods_per_year():
    from scripts.walkforward.cpcv import CPCVResult
    from app.research import ruler_v2
    idx = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(0)
    rets = 0.0008 + rng.normal(0, 0.01, len(idx))   # positive drift -> hac gating True
    res = CPCVResult(model_type="test", n_folds=12, n_paths=20)
    res.path_sharpes = [1.0] * 20
    res.is_true_walkforward = True
    res.oos_returns_dated = [(d.strftime("%Y-%m-%d"), float(r)) for d, r in zip(idx, rets)]

    kw = dict(tier="paper", component_type="diversifier", regime_waiver_approved=True)
    sr252 = ruler_v2.evaluate(res, periods_per_year=252, **kw)["point_sr_floor"][0]
    sr365 = ruler_v2.evaluate(res, periods_per_year=365, **kw)["point_sr_floor"][0]
    assert sr252 > 0 and sr365 > 0
    # annualized SR scales as sqrt(periods) -> 365/252 ratio
    assert sr365 / sr252 == pytest.approx(math.sqrt(365 / 252), rel=1e-3)
