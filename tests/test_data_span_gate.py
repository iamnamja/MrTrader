"""MEDIUM-3: data span gate tests."""
from datetime import date, timedelta

import pandas as pd
import pytest
from unittest.mock import patch

from scripts.walkforward.engine import FoldEngine
from scripts.walkforward.gates import DataSpanError, FoldResult
from scripts.walkforward.cpcv import run_cpcv


class _StubModel:
    trained_through = date(2000, 1, 1)


class _StubStrategy:
    """Minimal strategy: fetch_data is a no-op (data pre-seeded by the test)."""

    def __init__(self, all_days=None, symbols_data=None, data_source="test"):
        self.model_type = "intraday"
        self.version = 1
        self.model = _StubModel()
        self.all_days_sorted = all_days if all_days is not None else []
        self.symbols_data = symbols_data or {}
        self.data_source = data_source
        self.allow_in_sample = True

    def fetch_data(self, start, end):
        pass  # data pre-seeded

    def run_fold(self, idx, n_folds, tr_start, tr_end, te_start, te_end):
        return FoldResult(
            fold=idx, train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            trades=10, win_rate=0.5, sharpe=1.0,
            max_drawdown=0.05, total_return=0.1, stop_exit_rate=0.1, n_obs=50,
        )


def _trading_days(n: int):
    base = date(2022, 1, 3)
    return [base + timedelta(days=i) for i in range(n)]


def _run(strategy):
    engine = FoldEngine(strategy=strategy, purge_days=2, embargo_days=2, parallel=False)
    return engine.run(n_folds=5, total_days=300, allow_in_sample=True)


def test_short_span_raises():
    s = _StubStrategy(all_days=_trading_days(55), data_source="yfinance-fallback (<=55d)")
    with pytest.raises(DataSpanError):
        _run(s)


def test_exactly_250_no_raise():
    s = _StubStrategy(all_days=_trading_days(250))
    # Boundary: 250 == MIN, < is False → no DataSpanError.
    _run(s)  # must not raise DataSpanError


def test_249_raises():
    s = _StubStrategy(all_days=_trading_days(249))
    with pytest.raises(DataSpanError):
        _run(s)


def test_enforce_false_warns_only():
    s = _StubStrategy(all_days=_trading_days(55))
    with patch("app.ml.retrain_config.ENFORCE_MIN_DATA_SPAN", False):
        _run(s)  # must not raise


def test_run_cpcv_short_span_raises():
    s = _StubStrategy(all_days=_trading_days(55))
    with pytest.raises(DataSpanError):
        run_cpcv(s, purge_days=2, embargo_days=2, n_folds=6, n_paths=2, total_days=300)


def test_swing_path_span():
    # Swing: span derived from symbols_data distinct dates (no all_days_sorted).
    idx_300 = pd.DatetimeIndex([date(2021, 1, 1) + timedelta(days=i) for i in range(300)])
    df_300 = pd.DataFrame({"close": range(300)}, index=idx_300)
    s_pass = _StubStrategy(all_days=[], symbols_data={"AAA": df_300})
    s_pass.model_type = "swing"
    _run(s_pass)  # 300 distinct dates → no raise

    idx_100 = pd.DatetimeIndex([date(2021, 1, 1) + timedelta(days=i) for i in range(100)])
    df_100 = pd.DataFrame({"close": range(100)}, index=idx_100)
    s_fail = _StubStrategy(all_days=[], symbols_data={"AAA": df_100})
    s_fail.model_type = "swing"
    with pytest.raises(DataSpanError):
        _run(s_fail)


def test_intraday_fetch_data_sets_data_source(monkeypatch):
    from scripts.walkforward.strategies.intraday import IntradayStrategy

    # Cache populated → polygon-cache
    import app.data.intraday_cache as ic
    monkeypatch.setattr(ic, "available_symbols", lambda: ["AAA"])
    monkeypatch.setattr(ic, "load_many", lambda syms, start, end: {})
    strat = IntradayStrategy(model=_StubModel(), version=1, symbols=["AAA"])
    # SPY/VIX downloads will be attempted; stub yfinance to avoid network.
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
    strat.fetch_data(date(2022, 1, 1), date(2022, 6, 1))
    assert strat.data_source == "polygon-cache"

    # Cache empty → yfinance fallback
    monkeypatch.setattr(ic, "available_symbols", lambda: [])
    strat2 = IntradayStrategy(model=_StubModel(), version=1, symbols=["AAA"])
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
    strat2.fetch_data(date(2022, 1, 1), date(2022, 6, 1))
    assert "yfinance-fallback" in strat2.data_source
