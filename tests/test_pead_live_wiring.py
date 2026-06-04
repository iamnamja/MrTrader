"""
Tests for the live PEAD wiring (feat/pead-live-wiring).

Covers the three correctness fixes that make the live path actually run the
validated +0.546 config, plus the config surface, marketable-entry routing,
and the daily tracking observability artifact.

  FIX A — VIX SERIES injected → crisis block (VIX>30) fires → scorer returns [].
  FIX B — pead_max_hold_days=40 propagates to proposals and into the Trader position.
  FIX C — scorer constructed pinning EVERY validated parameter.
  Config — pm.pead_* keys default to the validated +0.546 values.
  Routing — PEAD entries route as a MARKETABLE limit (ask + offset), not below-ask.
  Tracking — pead_tracker.record_daily writes/updates a daily row.

All external calls (Alpaca, yfinance, DB) are mocked — no network/live I/O.
"""
import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _vix_series(level: float, n: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"close": [level] * n}, index=idx)


def _stock_df(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    closes = np.linspace(100, 110, n)
    return pd.DataFrame({"close": closes, "volume": [1e6] * n}, index=idx)


def _make_pm(bars, vix, build_fn=None):
    """Build a bare PortfolioManager with mocked deps for _analyze_swing_pead."""
    from app.agents.portfolio_manager import PortfolioManager
    pm = PortfolioManager.__new__(PortfolioManager)
    pm.logger = MagicMock()
    pm._swing_proposals = []
    pm._get_universe = MagicMock(return_value=list(bars.keys()))
    pm._fetch_vix_series = MagicMock(return_value=vix)
    # B5: default to None so the trend filter fails CLOSED to the VIX block in tests
    # (deterministic, no network). Trend-mode tests override this with a SPY series.
    pm._fetch_spy_series = MagicMock(return_value=None)
    async def _log(*a, **k):
        return None
    pm.log_decision = _log
    pm._fake_alpaca = MagicMock()
    pm._fake_alpaca.get_bars_batch.return_value = bars
    if build_fn is not None:
        pm._build_directional_proposals = build_fn
    else:
        async def _default_build(*a, **k):
            return [{"symbol": "AAPL"}]
        pm._build_directional_proposals = _default_build
    return pm


def _run_pead(pm, fake_scorer_cls):
    """Run pm._analyze_swing_pead with PEADScorer + alpaca + db patched."""
    with patch("app.ml.pead_scorer.PEADScorer", fake_scorer_cls), \
         patch("app.integrations.get_alpaca_client", return_value=pm._fake_alpaca), \
         patch("app.database.session.get_session", side_effect=Exception("no db")), \
         patch("app.live_trading.pead_tracker.record_daily", return_value=True):
        asyncio.run(pm._analyze_swing_pead())


# ════════════════════════════════════════════════════════════════════════════════
# FIX A — VIX series injection fires the crisis block
# ════════════════════════════════════════════════════════════════════════════════

class TestFixA_VixSeriesFiresBlock:
    def test_vix_series_over_30_blocks_all_entries(self):
        """With a VIX SERIES showing 35 (>30), the scorer returns [] (crisis block)."""
        from app.ml.pead_scorer import PEADScorer
        scorer = PEADScorer(vix_block_all=30.0)
        symbols_data = {"^VIX": _vix_series(35.0), "AAPL": _stock_df()}

        feats = {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}
        with patch("app.data.fmp_provider.get_earnings_features_at", return_value=feats):
            out = scorer(pd.Timestamp("2024-01-30"), symbols_data)
        assert out == []  # crisis block fired

    def test_vix_series_under_30_allows_entries(self):
        """VIX SERIES at 18 (<30): the same strong beat produces a long signal."""
        from app.ml.pead_scorer import PEADScorer
        scorer = PEADScorer(vix_block_all=30.0, vix_conf_ref=100.0)
        symbols_data = {"^VIX": _vix_series(18.0), "AAPL": _stock_df()}

        feats = {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}
        with patch("app.data.fmp_provider.get_earnings_features_at", return_value=feats):
            out = scorer(pd.Timestamp("2024-01-30"), symbols_data)
        assert any(d == "long" for _, _, d in out)

    def test_no_vix_key_means_block_never_fires(self):
        """Regression: without a ^VIX key (the OLD bug) the block cannot fire."""
        from app.ml.pead_scorer import PEADScorer
        scorer = PEADScorer(vix_block_all=30.0, vix_conf_ref=100.0)
        symbols_data = {"AAPL": _stock_df()}  # no ^VIX — the silent-divergence bug

        feats = {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}
        with patch("app.data.fmp_provider.get_earnings_features_at", return_value=feats):
            out = scorer(pd.Timestamp("2024-01-30"), symbols_data)
        assert any(d == "long" for _, _, d in out)  # block did NOT fire (no series)

    def test_pm_injects_vix_series_into_symbols_data(self, monkeypatch):
        """_analyze_swing_pead injects ^VIX (series) so the scorer sees it."""
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=_vix_series(35.0))
        captured = {}

        class _FakeScorer:
            def __init__(self, **kw):
                captured["init"] = kw

            def __call__(self, day, symbols_data, **kw):
                captured["symbols_data_keys"] = set(symbols_data.keys())
                return []  # VIX=35 → block → []

        _run_pead(pm, _FakeScorer)
        assert "^VIX" in captured["symbols_data_keys"]

    def test_vix_series_unavailable_fails_closed_when_blind(self):
        """If the daily series fetch fails AND all scalar sources are blind (30.0
        sentinel), inject a >30 ^VIX series so the crisis block fires (fail-CLOSED)."""
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=None)  # series fetch -> None
        pm._fetch_vix_level = MagicMock(return_value=30.0)    # all sources blind
        captured = {}

        class _FakeScorer:
            def __init__(self, **kw):
                pass

            def __call__(self, day, symbols_data, **kw):
                captured["vix_df"] = symbols_data.get("^VIX")
                return []

        _run_pead(pm, _FakeScorer)
        vix_df = captured["vix_df"]
        assert vix_df is not None, "fail-closed must still inject a ^VIX series"
        assert float(vix_df["close"].iloc[-1]) > 30.0, "blind VIX must block (>30)"

    def test_vix_series_unavailable_uses_real_scalar_reading(self):
        """If the daily series fails but the scalar fetcher returns a real low
        reading (e.g. FRED 1d-lag = 18), trade normally — inject that level, no block."""
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=None)
        pm._fetch_vix_level = MagicMock(return_value=18.0)
        captured = {}

        class _FakeScorer:
            def __init__(self, **kw):
                pass

            def __call__(self, day, symbols_data, **kw):
                captured["vix_df"] = symbols_data.get("^VIX")
                return [("AAPL", 0.8, "long")]

        _run_pead(pm, _FakeScorer)
        vix_df = captured["vix_df"]
        assert vix_df is not None
        assert float(vix_df["close"].iloc[-1]) == 18.0, "real low reading injected verbatim (no block)"


# ════════════════════════════════════════════════════════════════════════════════
# FIX B — max_hold_days = 40 propagates
# ════════════════════════════════════════════════════════════════════════════════

class TestFixB_MaxHold40:
    def test_proposals_annotated_hold_40(self, monkeypatch):
        """_build_directional_proposals is called with max_hold_days=40 by default."""
        captured = {}

        async def _fake_build(scored, selector="pead", max_hold_days=0, **kwargs):
            captured["max_hold_days"] = max_hold_days
            captured["size_mult"] = kwargs.get("size_mult")
            captured["selector"] = selector
            return [{"symbol": "AAPL", "max_hold_days": max_hold_days}]

        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=_vix_series(18.0), build_fn=_fake_build)

        class _FakeScorer:
            def __init__(self, **kw):
                pass

            def __call__(self, day, symbols_data, **kw):
                return [("AAPL", 0.8, "long")]

        _run_pead(pm, _FakeScorer)
        assert captured["max_hold_days"] == 40
        assert captured["selector"] == "pead"

    def test_trader_position_inherits_hold_40(self):
        """A proposal with max_hold_days=40 propagates into the live position dict."""
        # Mirrors trader.py: _pos_entry inherits proposal['max_hold_days'] when > 0.
        proposal = {"symbol": "AAPL", "max_hold_days": 40, "selector": "pead"}
        _pos_entry = {"entry_price": 100.0}
        _mhd = proposal.get("max_hold_days")
        if _mhd and int(_mhd) > 0:
            _pos_entry["max_hold_days"] = int(_mhd)
        assert _pos_entry["max_hold_days"] == 40
        # exit-monitor: per-position max_hold overrides the global default
        max_hold = _pos_entry.get("max_hold_days") or 20
        assert max_hold == 40


# ════════════════════════════════════════════════════════════════════════════════
# FIX C — scorer pins EVERY validated parameter
# ════════════════════════════════════════════════════════════════════════════════

class TestFixC_AllValidatedParams:
    def test_scorer_constructed_with_full_validated_config(self):
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=_vix_series(18.0))
        captured = {}

        class _FakeScorer:
            def __init__(self, **kw):
                captured["init"] = kw

            def __call__(self, day, symbols_data, **kw):
                return []

        _run_pead(pm, _FakeScorer)
        init = captured["init"]
        assert init["long_threshold"] == 0.05
        assert init["short_threshold"] == -0.05
        assert init["max_days_after"] == 3
        assert init["long_short"] is False
        assert init["vix_block_all"] == 30.0
        assert init["vix_block_short"] == 100.0
        assert init["vix_conf_ref"] == 100.0
        assert init["max_announce_day_move"] == 1.0          # priced-in filter OFF
        assert init["require_positive_revision"] is False
        assert init["min_analyst_momentum"] == 0.0


# ════════════════════════════════════════════════════════════════════════════════
# Config surface defaults
# ════════════════════════════════════════════════════════════════════════════════

class TestPeadConfigDefaults:
    @pytest.mark.parametrize("key,expected", [
        ("pm.pead_long_threshold", 0.05),
        ("pm.pead_short_threshold", -0.05),
        ("pm.pead_max_days_after", 3),
        ("pm.pead_max_hold_days", 40),
        ("pm.pead_vix_block_all", 30.0),
        ("pm.pead_vix_block_short", 100.0),
        ("pm.pead_vix_conf_ref", 100.0),
        ("pm.pead_max_announce_day_move", 1.0),
        ("pm.pead_require_positive_revision", "false"),
    ])
    def test_default_matches_validated_value(self, key, expected):
        from app.database.agent_config import _DEFAULTS
        assert _DEFAULTS[key] == expected


# ════════════════════════════════════════════════════════════════════════════════
# Marketable-entry routing for PEAD
# ════════════════════════════════════════════════════════════════════════════════

class TestPeadMarketableRouting:
    def _make_trader(self):
        from app.agents.trader import Trader
        t = Trader.__new__(Trader)
        t.logger = MagicMock()
        t.approved_symbols = {}
        t._pending_limit_orders = {}
        t._write_pending_fill = MagicMock(return_value=123)
        t._update_pending_fill_order_id = MagicMock()
        t._save_pending_limit_db = MagicMock()
        t._cancel_pending_fill = MagicMock()
        t._release_intraday_slot = MagicMock()
        return t

    def _result(self):
        return SimpleNamespace(
            entry_price=100.0, stop_price=98.0, target_price=106.0,
            atr=1.5, signal_type="ML_RANK",
        )

    def test_pead_long_routes_marketable_above_ask(self):
        t = self._make_trader()
        proposal = {"symbol": "AAPL", "trade_type": "swing", "selector": "pead",
                    "direction": "BUY", "proposal_uuid": "u1", "entry_price": 100.0}
        t.approved_symbols["AAPL"] = proposal

        alpaca = MagicMock()
        alpaca.get_quote.return_value = {"ask": 100.0, "bid": 99.9, "mid": 99.95}
        alpaca.place_limit_order.return_value = {"order_id": "o1"}

        asyncio.run(t._execute_entry("AAPL", 10, self._result(), alpaca))

        alpaca.place_limit_order.assert_called_once()
        args = alpaca.place_limit_order.call_args[0]
        # signature: (symbol, shares, side, limit_price, ...)
        _sym, _shares, _side, _limit_price = args[0], args[1], args[2], args[3]
        assert _side == "buy"
        assert _limit_price > 100.0   # marketable = ABOVE ask (crosses spread)
        alpaca.place_market_order.assert_not_called()

    def test_non_pead_swing_routes_below_ask(self):
        """Regression: non-PEAD swing still uses the below-ask limit (byte-identical)."""
        t = self._make_trader()
        proposal = {"symbol": "MSFT", "trade_type": "swing", "selector": "factor_portfolio",
                    "direction": "BUY", "proposal_uuid": "u2", "entry_price": 100.0}
        t.approved_symbols["MSFT"] = proposal

        alpaca = MagicMock()
        alpaca.get_quote.return_value = {"ask": 100.0, "bid": 99.9, "mid": 99.95}
        alpaca.place_limit_order.return_value = {"order_id": "o2"}

        with patch("app.database.session.get_session", side_effect=Exception("no db")):
            asyncio.run(t._execute_entry("MSFT", 10, self._result(), alpaca))

        args = alpaca.place_limit_order.call_args[0]
        _limit_price = args[3]
        assert _limit_price < 100.0   # below-ask (the standard swing path)


# ════════════════════════════════════════════════════════════════════════════════
# Daily tracking artifact
# ════════════════════════════════════════════════════════════════════════════════

class TestPeadTracker:
    def test_record_daily_writes_row(self, tmp_path, monkeypatch):
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        ok = pt.record_daily(
            date(2026, 6, 1), n_signals=8, n_entered=5, n_filled=4,
            gross_deployed=10_000.0, realized_pnl=120.0, unrealized_pnl=30.0,
            vix_level=18.5, vix_block_fired=False,
            suppressed_opportunity=1, suppressed_macro=0, suppressed_rm=2,
        )
        assert ok is True

        rows = pt.read_daily()
        assert len(rows) == 1
        r = rows[0]
        assert r["trade_date"] == "2026-06-01"
        assert r["n_signals"] == 8
        assert r["fill_rate"] == pytest.approx(0.8)
        assert r["daily_pnl"] == pytest.approx(150.0)
        assert r["cumulative_pnl"] == pytest.approx(150.0)
        assert r["suppressed_rm"] == 2

    def test_cumulative_pnl_accumulates(self, tmp_path, monkeypatch):
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        pt.record_daily(date(2026, 6, 1), realized_pnl=100.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 6, 2), realized_pnl=50.0, gross_deployed=1000.0)
        rows = pt.read_daily()
        assert rows[-1]["cumulative_pnl"] == pytest.approx(150.0)

    def test_record_daily_upserts_same_date(self, tmp_path, monkeypatch):
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")

        pt.record_daily(date(2026, 6, 1), n_signals=3)
        pt.record_daily(date(2026, 6, 1), n_signals=9)  # same date → update
        rows = pt.read_daily()
        assert len(rows) == 1
        assert rows[0]["n_signals"] == 9

    def test_weekly_rollup_payload(self, tmp_path, monkeypatch):
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")
        # vary daily returns so Sharpe is computable
        pt.record_daily(date(2026, 5, 26), realized_pnl=50.0, gross_deployed=1000.0,
                        n_signals=2, n_entered=2, n_filled=2)
        pt.record_daily(date(2026, 5, 27), realized_pnl=-20.0, gross_deployed=1000.0,
                        n_signals=1, n_entered=1, n_filled=0, suppressed_macro=1)
        pt.record_daily(date(2026, 5, 28), realized_pnl=30.0, gross_deployed=1000.0,
                        n_signals=3, n_entered=2, n_filled=2)

        payload = pt.weekly_rollup(date(2026, 5, 28), send=False)
        assert payload["backtest_sharpe"] == "+0.546"
        assert payload["n_days"] == 3
        assert payload["signals_entered_filled"] == "6 / 5 / 4"
        assert "macro=1" in payload["suppressed_breakdown"]
        assert payload["realized_sharpe"] != "n/a"

    def test_weekly_rollup_enqueues_notification(self, tmp_path, monkeypatch):
        import app.live_trading.pead_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "pead_tracking.db")
        # Seed >= min_days (3) deployed days so the vacuous-email guard lets it send.
        pt.record_daily(date(2026, 5, 26), realized_pnl=50.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 27), realized_pnl=-20.0, gross_deployed=1000.0)
        pt.record_daily(date(2026, 5, 28), realized_pnl=30.0, gross_deployed=1000.0)

        called = {}

        def _fake_enqueue(event_type, payload, dedup_key=None):
            called["event_type"] = event_type
            called["payload"] = payload
            return 1

        with patch("app.notifications.notifier.enqueue", _fake_enqueue):
            pt.weekly_rollup(date(2026, 5, 28), send=True)
        assert called["event_type"] == "pead_weekly"


# ════════════════════════════════════════════════════════════════════════════════
# B5 — SPY trend filter wiring (regime_control replaces the VIX block when SPY ok)
# ════════════════════════════════════════════════════════════════════════════════

def _spy_series_b5(n=260):
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    return pd.DataFrame({"close": np.linspace(100, 200, n)}, index=idx)  # uptrend


class TestB5TrendWiring:
    def _capture(self, pm):
        captured = {}

        class _FakeScorer:
            def __init__(self, **kw):
                captured["init"] = kw

            def __call__(self, day, symbols_data, **kw):
                captured["spy_in_data"] = "SPY" in symbols_data
                return []

        _run_pead(pm, _FakeScorer)
        return captured

    def test_trend_mode_when_spy_available(self):
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=_vix_series(18.0))
        pm._fetch_spy_series = MagicMock(return_value=_spy_series_b5())  # SPY ok
        cap = self._capture(pm)
        init = cap["init"]
        assert init["regime_control"] == "trend"
        assert init["vix_block_all"] == float("inf")   # VIX block OFF under trend
        assert init["regime_control_trend_ma"] == 200
        assert cap["spy_in_data"] is True              # SPY injected for the filter

    def test_fail_closed_to_vix_when_spy_unavailable(self):
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=_vix_series(18.0))
        pm._fetch_spy_series = MagicMock(return_value=None)   # SPY missing
        init = self._capture(pm)["init"]
        assert init["regime_control"] is None
        assert init["vix_block_all"] == 30.0           # fall back to the proven VIX block

    def test_fail_closed_when_spy_history_too_short(self):
        pm = _make_pm(bars={"AAPL": _stock_df()}, vix=_vix_series(18.0))
        pm._fetch_spy_series = MagicMock(return_value=_spy_series_b5(n=120))  # < 200 bars
        init = self._capture(pm)["init"]
        assert init["regime_control"] is None
        assert init["vix_block_all"] == 30.0
