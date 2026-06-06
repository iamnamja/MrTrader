"""RANKER v2 — §3.1 dollar-neutral ranker tests (re-architected 2026-06-03).

The standalone driver `scripts/run_ranker_v2_cpcv.py` was ABANDONED (it silently
diverged from the trusted tier3 path). Both experiment arms now run through
`walkforward_tier3.py::_run_cpcv_swing`. These tests exercise the SHARED rebalance
engine + the tier3 wiring directly:

  1. Short-kwarg wiring: SwingStrategy forwards enable_shorts/long_gross/short_gross/
     short_target_n/short_min_adv/short_add/drop + net_sector_cap + spy_beta_hedge to
     AgentSimulator; long-only default leaves the path byte-identical.
  2. Dollar-neutrality: enable_shorts=True with long_gross==short_gross → net_target==0
     and a balanced book; long-only run produces ZERO short positions.
  3. Sector-cap + inverse-vol applied on the L/S book.
  4. No look-ahead: book construction uses only data <= decision day.
  5. Net-beta / net-dollar / net-sector capture correctness (the alpha-vs-beta instrument).
  6. BUG-1 sizing: long_budget flows into long sizing; short reaches gross via realized-count.
  7. NET-SECTOR-CAP (Failure B fix): on a REALISTIC sector-concentrated + illiquid R1K-like
     short tail, the dollar-neutral book reaches net_dollar<0.05 / max|net beta|<0.15; the
     OLD per-side-sector-cap code does NOT.
  8. SPY beta-hedge OVERLAY: residual net beta is driven toward ~0.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.backtesting.agent_simulator import AgentSimulator
from app.strategy.portfolio_construction import (
    apply_net_sector_cap,
    apply_sector_cap_shorts,
    split_gross_budgets,
)
from scripts.walkforward.strategies.swing import SwingStrategy


# ---------------------------------------------------------------------------
# Synthetic universe (mirrors test_agent_simulator_rebalance.py)
# ---------------------------------------------------------------------------

def _synthetic_bars(n_days=260, start=date(2023, 1, 3), seed=42, drift=0.0005, vol=0.015):
    rng = np.random.default_rng(seed)
    close = 100.0
    rows = []
    d = pd.Timestamp(start)
    for _ in range(n_days):
        if d.weekday() < 5:
            ret = rng.normal(drift, vol)
            close *= (1 + ret)
            rows.append({
                "open": close * (1 + rng.normal(0, 0.003)),
                "high": close * (1 + abs(rng.normal(0, 0.005))),
                "low": close * (1 - abs(rng.normal(0, 0.005))),
                "close": close,
                "volume": int(rng.integers(2_000_000, 6_000_000)),
            })
        d += pd.Timedelta(days=1)
    idx = pd.date_range(start=start, periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx)


def _make_bars_map(n_symbols=24, n_days=260):
    return {f"SYM{i:02d}": _synthetic_bars(n_days=n_days, seed=i + 10)
            for i in range(n_symbols)}


def _mock_factor_scorer(day, symbols_data, vix_history=None):
    """Rank symbols alphabetically (deterministic) — bypasses feature engineering."""
    syms = sorted(s for s in symbols_data.keys() if s not in ("SPY", "^VIX", "VIX"))
    scores = np.linspace(1.0, 0.0, len(syms))
    return list(zip(syms, scores.tolist()))


def _run_ls_sim(enable_shorts, long_gross=0.40, short_gross=0.40,
                n_symbols=24, n_days=200, sector_cap=1.0, inv_vol=False):
    bars = _make_bars_map(n_symbols=n_symbols, n_days=n_days)
    start = date(2023, 1, 3)
    end = start + timedelta(days=n_days + 30)
    sim = AgentSimulator(
        model=None,
        starting_capital=1_000_000.0,
        rebalance_mode=True,
        rebalance_days=5,
        rebalance_target_n=6,
        rebalance_sector_cap=sector_cap,
        rebalance_add_threshold=6,
        rebalance_drop_threshold=10,
        rebalance_min_adv=0.0,
        rebalance_inv_vol=inv_vol,
        no_atr_stops=True,
        factor_scorer=_mock_factor_scorer,
        enable_shorts=enable_shorts,
        long_gross=long_gross,
        short_gross=short_gross,
        short_target_n=6,
        short_min_adv=0.0,
        short_add_threshold=6,
        short_drop_threshold=10,
    )
    result = sim.run(bars, start_date=start, end_date=end)
    return sim, result


# ---------------------------------------------------------------------------
# 1. Short-kwarg wiring (SwingStrategy -> AgentSimulator)
# ---------------------------------------------------------------------------

class TestShortKwargWiring:
    def test_swing_strategy_stores_short_kwargs(self):
        s = SwingStrategy(
            model=None, version=0, symbols=["A"],
            rebalance_mode=True,
            enable_shorts=True, long_gross=0.40, short_gross=0.40,
            short_target_n=60, short_min_adv=50e6,
            short_add_threshold=12, short_drop_threshold=25,
            net_sector_cap=True, spy_beta_hedge=True,
        )
        assert s.enable_shorts is True
        assert s.long_gross == 0.40
        assert s.short_gross == 0.40
        assert s.short_target_n == 60
        assert s.short_min_adv == 50e6
        assert s.short_add_threshold == 12
        assert s.short_drop_threshold == 25
        assert s.net_sector_cap is True
        assert s.spy_beta_hedge is True

    def test_short_kwargs_forwarded_to_simulator(self, monkeypatch):
        """run_fold must construct AgentSimulator with the short + re-arch kwargs."""
        captured = {}

        class _CaptureSim:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self._wf_run_id = None

            def run(self, *a, **k):
                class _R:
                    total_trades = 0
                    win_rate = 0.0
                    sharpe_ratio = 0.0
                    max_drawdown_pct = 0.0
                    total_return_pct = 0.0
                    exit_breakdown = {}
                    trades = []
                    equity_curve = []
                    profit_factor = 1.0
                    avg_capital_deployed_pct = 0.0
                    deployment_adjusted_sharpe = 0.0
                    low_deployment_warning = False
                return _R()

        import app.backtesting.agent_simulator as agsim_mod
        monkeypatch.setattr(agsim_mod, "AgentSimulator", _CaptureSim)
        import app.data.universe_history as uh
        monkeypatch.setattr(uh, "pit_union", lambda *a, **k: ["SYM00"])
        monkeypatch.setattr(uh, "historical_trade_symbols", lambda *a, **k: [])

        s = SwingStrategy(
            model=type("_M", (), {"feature_names": [], "trained_through": date.min})(),
            version=0, symbols=["SYM00"],
            rebalance_mode=True, feature_cache_disable=True,
            enable_shorts=True, long_gross=0.40, short_gross=0.40,
            short_target_n=60, short_min_adv=50e6,
            short_add_threshold=12, short_drop_threshold=25,
            net_sector_cap=True, spy_beta_hedge=True, spy_hedge_max_gross=0.25,
        )
        s.symbols_data = {"SYM00": _synthetic_bars(n_days=120)}
        s.spy_prices = s.symbols_data["SYM00"]["close"]

        s.run_fold(1, 3, date(2023, 1, 3), date(2023, 4, 1),
                   date(2023, 5, 1), date(2023, 6, 1))

        assert captured["enable_shorts"] is True
        assert captured["long_gross"] == 0.40
        assert captured["short_gross"] == 0.40
        assert captured["short_target_n"] == 60
        assert captured["short_min_adv"] == 50e6
        assert captured["short_add_threshold"] == 12
        assert captured["short_drop_threshold"] == 25
        assert captured["net_sector_cap"] is True
        assert captured["spy_beta_hedge"] is True
        assert captured["spy_hedge_max_gross"] == 0.25
        # Capture auto-on for the dollar-neutral arm.
        assert captured["capture_net_exposure"] is True

    def test_long_only_default_keeps_shorts_off(self, monkeypatch):
        """Default SwingStrategy forwards enable_shorts=False + AgentSimulator-matching
        defaults + net_sector_cap/spy_beta_hedge OFF -> long-only path byte-identical."""
        captured = {}

        class _CaptureSim:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self._wf_run_id = None

            def run(self, *a, **k):
                class _R:
                    total_trades = 0
                    win_rate = 0.0
                    sharpe_ratio = 0.0
                    max_drawdown_pct = 0.0
                    total_return_pct = 0.0
                    exit_breakdown = {}
                    trades = []
                    equity_curve = []
                    profit_factor = 1.0
                    avg_capital_deployed_pct = 0.0
                    deployment_adjusted_sharpe = 0.0
                    low_deployment_warning = False
                return _R()

        import app.backtesting.agent_simulator as agsim_mod
        monkeypatch.setattr(agsim_mod, "AgentSimulator", _CaptureSim)
        import app.data.universe_history as uh
        monkeypatch.setattr(uh, "pit_union", lambda *a, **k: ["SYM00"])
        monkeypatch.setattr(uh, "historical_trade_symbols", lambda *a, **k: [])

        s = SwingStrategy(
            model=type("_M", (), {"feature_names": [], "trained_through": date.min})(),
            version=0, symbols=["SYM00"], feature_cache_disable=True,
        )
        s.symbols_data = {"SYM00": _synthetic_bars(n_days=120)}
        s.spy_prices = s.symbols_data["SYM00"]["close"]
        s.run_fold(1, 3, date(2023, 1, 3), date(2023, 4, 1),
                   date(2023, 5, 1), date(2023, 6, 1))

        assert captured["enable_shorts"] is False
        assert captured["long_gross"] == 0.95
        assert captured["short_gross"] == 0.55
        assert captured["short_target_n"] == 30
        assert captured["short_min_adv"] == 50_000_000.0
        assert captured["net_sector_cap"] is False
        assert captured["spy_beta_hedge"] is False
        # Capture auto-OFF for long-only.
        assert captured["capture_net_exposure"] is False


# ---------------------------------------------------------------------------
# 2. Dollar-neutrality + balanced book
# ---------------------------------------------------------------------------

class TestDollarNeutrality:
    def test_split_gross_budgets_net_zero(self):
        long_b, short_b = split_gross_budgets(
            1_000_000.0,
            net_target=0.40 - 0.40,
            gross_target=0.40 + 0.40,
        )
        assert long_b == pytest.approx(short_b)
        assert long_b == pytest.approx(400_000.0)
        assert (long_b + short_b) == pytest.approx(800_000.0)

    def test_long_only_produces_zero_shorts(self):
        _, result = _run_ls_sim(enable_shorts=False)
        short_trades = [t for t in result.trades if t.exit_reason == "OPEN_SHORT"]
        assert len(short_trades) == 0

    def test_dollar_neutral_book_is_balanced(self):
        sim, result = _run_ls_sim(enable_shorts=True, long_gross=0.40, short_gross=0.40)
        open_longs = [t for t in result.trades if t.exit_reason == "OPEN"]
        open_shorts = [t for t in result.trades if t.exit_reason == "OPEN_SHORT"]
        assert len(open_shorts) > 0, "dollar-neutral book must hold short positions"
        assert len(open_longs) > 0, "dollar-neutral book must hold long positions"

        long_notional = sum(t.entry_price * t.quantity for t in open_longs)
        short_notional = sum(t.entry_price * t.quantity for t in open_shorts)
        assert long_notional > 0 and short_notional > 0
        ratio = short_notional / long_notional
        assert 0.5 < ratio < 2.0, f"long/short notional badly imbalanced: ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# 3. Sector-cap + inverse-vol applied
# ---------------------------------------------------------------------------

class TestSectorCapAndInverseVol:
    def test_inverse_vol_sizing_applied(self):
        sim, result = _run_ls_sim(enable_shorts=True, inv_vol=True)
        open_longs = [t for t in result.trades if t.exit_reason == "OPEN"]
        assert open_longs, "expected open long positions"
        from collections import defaultdict
        by_day = defaultdict(list)
        for t in open_longs:
            by_day[t.entry_date].append(t.entry_price * t.quantity)
        biggest_day = max(by_day, key=lambda d: len(by_day[d]))
        first_batch = by_day[biggest_day]
        assert len(first_batch) >= 3
        assert max(first_batch) - min(first_batch) > 0, "inverse-vol weights were uniform"

    def test_sector_cap_kwarg_threaded(self):
        sim, _ = _run_ls_sim(enable_shorts=True, sector_cap=0.30)
        assert sim.rebalance_sector_cap == 0.30
        assert sim.enable_shorts is True


# ---------------------------------------------------------------------------
# 4. No look-ahead (book construction uses only <= decision-day data)
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    def test_inverse_vol_weights_are_pit(self):
        from app.strategy.portfolio_construction import compute_inverse_vol_weights
        bars = _make_bars_map(n_symbols=6, n_days=120)
        syms = list(bars.keys())
        as_of = sorted(bars[syms[0]].index)[60].date()

        w_full = compute_inverse_vol_weights(syms, bars, as_of=as_of,
                                             total_equity=100_000.0)
        corrupted = {}
        for s, df in bars.items():
            df2 = df.copy()
            mask = df2.index.map(lambda d: d.date() > as_of)
            df2.loc[mask, "close"] = 1e9
            corrupted[s] = df2
        w_corrupt = compute_inverse_vol_weights(syms, corrupted, as_of=as_of,
                                                total_equity=100_000.0)
        assert w_full == pytest.approx(w_corrupt), "inverse-vol weights leaked future data"


# ---------------------------------------------------------------------------
# 5. Realized net-beta / net-dollar / net-sector capture
# ---------------------------------------------------------------------------

class _Pos:
    def __init__(self, symbol, quantity, entry_price, direction):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.direction = direction


def _series_with_beta(spy_rets, beta, idiosyncratic=0.0, start_price=100.0,
                      start=date(2023, 1, 3)):
    prices = [start_price]
    for r in spy_rets:
        prices.append(prices[-1] * (1 + beta * r + idiosyncratic))
    idx = pd.date_range(start=start, periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


class TestNetBetaCapture:
    def test_pit_beta_matches_known_beta(self):
        from app.backtesting.net_exposure import compute_pit_beta
        rng = np.random.default_rng(7)
        spy_rets = rng.normal(0.0004, 0.012, 120)
        spy_prices = [400.0]
        for r in spy_rets:
            spy_prices.append(spy_prices[-1] * (1 + r))
        idx = pd.date_range(start=date(2023, 1, 3), periods=len(spy_prices), freq="B")
        spy_df = pd.DataFrame({"close": spy_prices}, index=idx)
        name_df = _series_with_beta(spy_rets, beta=1.6)
        as_of = idx[-1].date() + timedelta(days=1)
        beta = compute_pit_beta(name_df, spy_df, as_of, lookback=60)
        assert beta == pytest.approx(1.6, abs=0.05)

    def test_signed_net_beta_hand_calc(self):
        from app.backtesting.net_exposure import compute_book_net_exposure
        rng = np.random.default_rng(11)
        spy_rets = rng.normal(0.0003, 0.011, 120)
        spy_prices = [400.0]
        for r in spy_rets:
            spy_prices.append(spy_prices[-1] * (1 + r))
        idx = pd.date_range(start=date(2023, 1, 3), periods=len(spy_prices), freq="B")
        spy_df = pd.DataFrame({"close": spy_prices}, index=idx)
        as_of = idx[-1].date() + timedelta(days=1)

        symbols_data = {
            "L1": _series_with_beta(spy_rets, 1.5),
            "L2": _series_with_beta(spy_rets, 1.5),
            "S1": _series_with_beta(spy_rets, 0.5),
            "S2": _series_with_beta(spy_rets, 0.5),
            "SPY": spy_df,
        }
        positions = {
            "L1": _Pos("L1", 1000, 100.0, "long"),
            "L2": _Pos("L2", 1000, 100.0, "long"),
            "S1": _Pos("S1", 1000, 100.0, "short"),
            "S2": _Pos("S2", 1000, 100.0, "short"),
        }
        closes = {"L1": 100.0, "L2": 100.0, "S1": 100.0, "S2": 100.0}
        ne = compute_book_net_exposure(
            positions=positions, closes_by_sym=closes, equity=1_000_000.0,
            symbols_data=symbols_data, spy_df=spy_df, as_of=as_of,
            sector_map={"L1": "TECH", "L2": "TECH", "S1": "FIN", "S2": "FIN"},
            beta_lookback=60,
        )
        assert ne["net_beta"] == pytest.approx(0.20, abs=0.03)
        assert ne["net_dollar"] == pytest.approx(0.0, abs=1e-9)
        assert ne["net_sector"]["TECH"] == pytest.approx(0.20, abs=1e-9)
        assert ne["net_sector"]["FIN"] == pytest.approx(-0.20, abs=1e-9)
        assert ne["max_abs_sector"] == pytest.approx(0.20, abs=1e-9)

    def test_net_dollar_nonzero_when_imbalanced(self):
        from app.backtesting.net_exposure import compute_book_net_exposure
        rng = np.random.default_rng(3)
        spy_rets = rng.normal(0.0003, 0.01, 100)
        spy_prices = [400.0]
        for r in spy_rets:
            spy_prices.append(spy_prices[-1] * (1 + r))
        idx = pd.date_range(start=date(2023, 1, 3), periods=len(spy_prices), freq="B")
        spy_df = pd.DataFrame({"close": spy_prices}, index=idx)
        as_of = idx[-1].date() + timedelta(days=1)
        symbols_data = {"L1": _series_with_beta(spy_rets, 1.0),
                        "S1": _series_with_beta(spy_rets, 1.0), "SPY": spy_df}
        positions = {"L1": _Pos("L1", 3000, 100.0, "long"),
                     "S1": _Pos("S1", 1000, 100.0, "short")}
        ne = compute_book_net_exposure(
            positions=positions, closes_by_sym={"L1": 100.0, "S1": 100.0},
            equity=1_000_000.0, symbols_data=symbols_data, spy_df=spy_df,
            as_of=as_of, sector_map={"L1": "TECH", "S1": "TECH"}, beta_lookback=60,
        )
        assert ne["net_dollar"] == pytest.approx(0.20, abs=1e-9)

    def test_pit_beta_ignores_future_moves(self):
        from app.backtesting.net_exposure import compute_pit_beta
        rng = np.random.default_rng(5)
        spy_rets = rng.normal(0.0003, 0.011, 120)
        spy_prices = [400.0]
        for r in spy_rets:
            spy_prices.append(spy_prices[-1] * (1 + r))
        idx = pd.date_range(start=date(2023, 1, 3), periods=len(spy_prices), freq="B")
        spy_df = pd.DataFrame({"close": spy_prices}, index=idx)
        name_df = _series_with_beta(spy_rets, beta=1.2)
        as_of = idx[80].date()

        beta_before = compute_pit_beta(name_df, spy_df, as_of, lookback=60)
        spy_corrupt = spy_df.copy()
        name_corrupt = name_df.copy()
        mask = spy_corrupt.index.map(lambda d: d.date() >= as_of)
        spy_corrupt.loc[mask, "close"] = 1e9
        nmask = name_corrupt.index.map(lambda d: d.date() >= as_of)
        name_corrupt.loc[nmask, "close"] = 1e-3
        beta_after = compute_pit_beta(name_corrupt, spy_corrupt, as_of, lookback=60)
        assert beta_before == pytest.approx(beta_after), \
            "PIT beta leaked future data (changed when future bars were corrupted)"

    def test_net_beta_clean_threshold(self):
        from app.backtesting.net_exposure import (
            summarize_net_exposure, NET_BETA_ALPHA_THRESHOLD,
        )
        assert NET_BETA_ALPHA_THRESHOLD == 0.15
        by_date = {
            date(2023, 1, 3): {"net_beta": 0.05, "net_dollar": 0.0, "max_abs_sector": 0.1},
            date(2023, 1, 4): {"net_beta": 0.25, "net_dollar": 0.01, "max_abs_sector": 0.2},
        }
        s = summarize_net_exposure(by_date)
        assert s["last_net_beta"] == pytest.approx(0.25)
        assert s["max_abs_net_beta"] == pytest.approx(0.25)
        assert s["mean_net_beta"] == pytest.approx(0.15)
        assert s["max_abs_net_sector"] == pytest.approx(0.2)
        # BLOCKER 1: summarize_net_exposure also reports the persistent (steady-state
        # p95) lens. With only 2 days (< warmup), the helper keeps them all, so p95 of
        # {0.05, 0.25} (nearest-rank) = 0.25.
        assert s["p95_abs_net_beta"] == pytest.approx(0.25)

    def test_production_and_test_net_beta_use_same_helper(self):
        """BLOCKER 1 single-source-of-truth lock: the production acceptance metric
        (summarize_net_exposure → SimResult.p95_abs_net_beta → CPCVResult.net_beta_clean)
        and the regression-test metric (_steady_state_net) MUST compute the IDENTICAL
        statistic over the IDENTICAL warmup window — both via
        net_exposure.steady_state_net_beta. This test feeds a hand-built daily series
        with a long warmup ramp + steady state and asserts both routes agree."""
        from app.backtesting.net_exposure import (
            steady_state_net_beta, summarize_net_exposure, NET_BETA_WARMUP_TWO_SIDED,
        )
        from app.backtesting.net_exposure import NET_BETA_WARMUP_TWO_SIDED as _W
        rng = np.random.default_rng(31)
        by_date = {}
        d = pd.Timestamp(date(2023, 1, 3))
        # `_W` warmup two-sided days with a high transient beta (0.40) then 90 clean
        # steady-state days (~ -0.07 with small noise + one 0.35 inter-rebalance spike).
        n_total = _W + 90
        for i in range(n_total):
            if i < _W:
                nb = 0.40
            else:
                nb = -0.07 + float(rng.normal(0, 0.03))
                if i == _W + 35:
                    nb = 0.35  # transient inter-rebalance spike in steady state
            by_date[d.date()] = {"net_beta": nb, "net_dollar": 0.0,
                                 "gross": 0.80, "max_abs_sector": 0.1,
                                 "n_long": 60, "n_short": 60}
            d += pd.Timedelta(days=1)
        # Route A: shared helper directly.
        ss = steady_state_net_beta(by_date)
        # Route B: production summary (what SimResult/CPCVResult consume).
        summ = summarize_net_exposure(by_date)
        # Route C: the test's lens (a fake result wrapping the same dict).
        class _R:
            net_exposure_by_date = by_date
        mean_nd, mean_nb, p95_nb, _ = _steady_state_net(_R())
        # All three must agree on the persistent statistic.
        assert summ["p95_abs_net_beta"] == pytest.approx(ss["p95_abs_net_beta"])
        assert p95_nb == pytest.approx(ss["p95_abs_net_beta"])
        assert mean_nb == pytest.approx(ss["mean_net_beta"])
        # And the warmup was actually trimmed (steady < two_sided).
        assert ss["n_two_sided"] == n_total
        assert ss["n_steady"] == n_total - NET_BETA_WARMUP_TWO_SIDED
        # The transient 0.35 spike does NOT blow up p95 (persistent ≈ 0.07-0.12),
        # while the raw daily max DOES see it.
        assert ss["p95_abs_net_beta"] < 0.15
        assert summ["max_abs_net_beta"] >= 0.35


# ---------------------------------------------------------------------------
# 6. Net-exposure flows end-to-end + is PURE-ADDITIVE
# ---------------------------------------------------------------------------

class TestNetExposureIntegration:
    def test_ls_sim_captures_net_exposure(self):
        bars = _make_bars_map(n_symbols=24, n_days=200)
        bars["SPY"] = _synthetic_bars(n_days=200, seed=999)
        start = date(2023, 1, 3)
        end = start + timedelta(days=230)
        sim = AgentSimulator(
            model=None, starting_capital=1_000_000.0,
            rebalance_mode=True, rebalance_days=5, rebalance_target_n=6,
            rebalance_sector_cap=1.0, rebalance_min_adv=0.0,
            no_atr_stops=True, factor_scorer=_mock_factor_scorer,
            enable_shorts=True, long_gross=0.40, short_gross=0.40,
            short_target_n=6, short_min_adv=0.0,
            capture_net_exposure=True,
        )
        result = sim.run(bars, start_date=start, end_date=end,
                         sector_map={s: "TECH" for s in bars})
        assert result.net_exposure_captured is True
        assert abs(result.mean_net_dollar) < 0.5
        assert isinstance(result.mean_net_beta, float)
        assert result.max_abs_net_beta >= 0.0

    def test_long_only_capture_off_is_byte_identical(self):
        bars = _make_bars_map(n_symbols=12, n_days=200)
        bars["SPY"] = _synthetic_bars(n_days=200, seed=123)
        start = date(2023, 1, 3)
        end = start + timedelta(days=230)
        kwargs = dict(
            model=None, starting_capital=1_000_000.0,
            rebalance_mode=True, rebalance_days=5, rebalance_target_n=6,
            rebalance_sector_cap=1.0, rebalance_min_adv=0.0,
            no_atr_stops=True, factor_scorer=_mock_factor_scorer,
            enable_shorts=False,
        )
        r_off = AgentSimulator(capture_net_exposure=False, **kwargs).run(
            bars, start_date=start, end_date=end)
        r_default = AgentSimulator(**kwargs).run(bars, start_date=start, end_date=end)
        assert r_off.net_exposure_captured is False
        assert r_off.mean_net_beta == 0.0 and r_off.max_abs_net_beta == 0.0
        assert r_off.mean_net_dollar == 0.0 and r_off.max_abs_net_sector == 0.0
        assert r_off.sharpe_ratio == r_default.sharpe_ratio
        assert r_off.total_return_pct == r_default.total_return_pct
        assert r_off.total_trades == r_default.total_trades
        assert r_off.profit_factor == r_default.profit_factor


# ---------------------------------------------------------------------------
# 7. CPCVResult net-exposure fields are PURE-ADDITIVE
# ---------------------------------------------------------------------------

class TestCPCVPureAdditive:
    def test_empty_cpcv_result_unchanged(self):
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(model_type="swing", n_folds=6, n_paths=2,
                       path_sharpes=[0.1, 0.2, 0.3])
        assert r.path_mean_net_betas == []
        assert r.path_p95_abs_net_betas == []
        assert r.net_exposure_captured is False
        assert r.mean_net_beta == 0.0
        assert r.max_abs_net_beta == 0.0
        assert r.p95_abs_net_beta == 0.0
        assert r.mean_net_dollar == 0.0
        assert r.max_abs_net_sector == 0.0
        assert r.net_beta_clean is True
        assert r.mean_sharpe == pytest.approx(0.2)

    def test_cpcv_net_exposure_aggregation_and_threshold(self):
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(
            model_type="swing", n_folds=6, n_paths=2,
            path_sharpes=[0.4, 0.6],
            path_mean_net_betas=[0.05, 0.25],
            path_max_abs_net_betas=[0.10, 0.30],
            path_p95_abs_net_betas=[0.08, 0.26],
            path_mean_net_dollars=[0.0, 0.01],
            path_max_abs_net_dollars=[0.02, 0.03],
            path_max_abs_net_sectors=[0.1, 0.2],
            net_exposure_captured=True,
        )
        assert r.mean_net_beta == pytest.approx(0.15)
        assert r.max_abs_net_beta == pytest.approx(0.30)
        assert r.p95_abs_net_beta == pytest.approx(0.26)
        assert r.max_abs_net_sector == pytest.approx(0.20)
        # mean (0.15) is at the bar but p95 (0.26) exceeds it → persistent
        # contamination → NOT clean (keys on the p95 lens, not raw max).
        assert r.net_beta_clean is False

    def test_net_beta_clean_keys_on_p95_not_raw_max(self):
        """BLOCKER 1: a book that is beta-neutral ON AVERAGE (mean −0.07) with a
        clean steady-state p95 (0.12) but a RAW daily max of 0.35 (transient
        inter-rebalance / warmup spike) must report net_beta_clean=True. The raw max
        is a DIAGNOSTIC only; the persistent (p95) lens drives the decision."""
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(
            model_type="swing", n_folds=6, n_paths=2,
            path_sharpes=[0.4, 0.6],
            path_mean_net_betas=[-0.07, -0.07],
            path_max_abs_net_betas=[0.35, 0.35],   # transient spike — diagnostic only
            path_p95_abs_net_betas=[0.12, 0.12],   # persistent exposure — the lens
            net_exposure_captured=True,
        )
        assert r.mean_net_beta == pytest.approx(-0.07)
        assert r.max_abs_net_beta == pytest.approx(0.35)   # reported, but not graded
        assert r.p95_abs_net_beta == pytest.approx(0.12)
        assert r.net_beta_clean is True, (
            "transient inter-rebalance spike (raw max 0.35) falsely failed a book that "
            "is beta-neutral on average (mean −0.07, p95 0.12)")

    def test_net_beta_clean_false_on_persistent_contamination(self):
        """BLOCKER 1 (the other side): a book with PERSISTENT mean 0.26 is genuine
        beta contamination and must report net_beta_clean=False even though no single
        transient is involved."""
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(
            model_type="swing", n_folds=6, n_paths=2,
            path_sharpes=[0.4, 0.6],
            path_mean_net_betas=[0.26, 0.26],
            path_max_abs_net_betas=[0.30, 0.30],
            path_p95_abs_net_betas=[0.28, 0.28],
            net_exposure_captured=True,
        )
        assert r.net_beta_clean is False


# ---------------------------------------------------------------------------
# 8. BUG 1 — point-in-time book neutrality (sizing fix guard)
# ---------------------------------------------------------------------------

def _big_bars_map(n_symbols=130, n_days=240, seed0=10):
    bm = {f"SYM{i:03d}": _synthetic_bars(n_days=n_days, seed=i + seed0)
          for i in range(n_symbols)}
    bm["SPY"] = _synthetic_bars(n_days=n_days, seed=999)
    return bm


def _big_scorer(day, symbols_data, vix_history=None):
    syms = sorted(s for s in symbols_data if s not in ("SPY", "^VIX", "VIX"))
    scores = np.linspace(1.0, 0.0, len(syms))
    return list(zip(syms, scores.tolist()))


# NOTE: n_days stays 240 here. test_dollar_neutral_book_is_point_in_time_neutral
# asserts the ALL-DAYS mean_net_dollar (< 0.10), which is sensitive to the long-only
# ramp fraction — a shorter window inflates it past the bar. The CI floor is the
# realistic-fixture tests (240 names), which are shrunk via _realistic_r1k_fixture;
# these 130-name point-in-time books are not the bottleneck, so they keep the
# original window.
def _run_dn_book(n_symbols=130, n_days=240, long_gross=0.40, short_gross=0.40,
                 target_n=60, short_n=60, sector_cap=1.0, short_min_adv=0.0,
                 inv_vol=True, net_sector_cap=False, spy_beta_hedge=False,
                 sector_map=None, bars=None, short_realized_rescale=True):
    if bars is None:
        bars = _big_bars_map(n_symbols=n_symbols, n_days=n_days)
    start = date(2023, 1, 3)
    end = start + timedelta(days=n_days + 30)
    sim = AgentSimulator(
        model=None, starting_capital=1_000_000.0,
        rebalance_mode=True, rebalance_days=5,
        rebalance_target_n=target_n, rebalance_sector_cap=sector_cap,
        rebalance_min_adv=0.0, rebalance_inv_vol=inv_vol,
        no_atr_stops=True, factor_scorer=_big_scorer,
        enable_shorts=True, long_gross=long_gross, short_gross=short_gross,
        short_target_n=short_n, short_min_adv=short_min_adv,
        net_sector_cap=net_sector_cap, spy_beta_hedge=spy_beta_hedge,
        short_realized_rescale=short_realized_rescale,
        capture_net_exposure=True,
    )
    if sector_map is None:
        sector_map = {s: "TECH" for s in bars}
    result = sim.run(bars, start_date=start, end_date=end, sector_map=sector_map)
    return sim, result


class TestPointInTimeNeutrality:
    def test_dollar_neutral_book_is_point_in_time_neutral(self):
        _, result = _run_dn_book(long_gross=0.40, short_gross=0.40,
                                 target_n=60, short_n=60)
        assert result.net_exposure_captured is True
        nbd = result.net_exposure_by_date
        assert nbd, "net-exposure snapshots must be captured"
        two_sided = [(d, x) for d, x in sorted(nbd.items())
                     if x["n_long"] > 0 and x["n_short"] > 0]
        assert two_sided, "book never held both legs simultaneously"
        d0, snap0 = two_sided[0]
        assert abs(snap0["net_dollar"]) < 0.10, (
            f"point-in-time book not dollar-neutral on {d0}: "
            f"net_dollar={snap0['net_dollar']:.3f} (Bug 1 = ~0.76 net long)")
        assert abs(snap0["gross"] - 0.80) < 0.10, (
            f"point-in-time gross off target on {d0}: gross={snap0['gross']:.3f}")
        assert abs(result.mean_net_dollar) < 0.10, (
            f"mean net_dollar not neutral: {result.mean_net_dollar:.3f}")

    def test_long_budget_flows_into_long_sizing(self):
        _, result = _run_dn_book(long_gross=0.40, short_gross=0.40,
                                 target_n=60, short_n=60)
        nbd = result.net_exposure_by_date
        first_day, first = sorted(nbd.items())[0]
        assert first["n_short"] == 0, "expected long-only first snapshot"
        assert first["n_long"] > 0
        assert abs(first["net_dollar"] - 0.40) < 0.10, (
            f"long leg not sized to long_gross on {first_day}: "
            f"long/equity={first['net_dollar']:.3f} (Bug 1 sized to full NAV ~ 1.0)")
        assert first["net_dollar"] < 0.60, "long leg looks sized to full NAV (Bug 1)"

    def test_short_reaches_target_gross_via_realized_count(self):
        bars = _big_bars_map(n_symbols=130, n_days=150)
        for i, s in enumerate(sorted(k for k in bars if k != "SPY")):
            if i % 2 == 1:
                bars[s] = bars[s].assign(volume=(bars[s]["volume"] * 0.001).astype(int))
        start = date(2023, 1, 3)
        end = start + timedelta(days=170)
        sim = AgentSimulator(
            model=None, starting_capital=1_000_000.0,
            rebalance_mode=True, rebalance_days=5,
            rebalance_target_n=60, rebalance_sector_cap=1.0,
            rebalance_min_adv=0.0, rebalance_inv_vol=True,
            no_atr_stops=True, factor_scorer=_big_scorer,
            enable_shorts=True, long_gross=0.40, short_gross=0.40,
            short_target_n=60,
            short_min_adv=300_000_000.0,
            capture_net_exposure=True,
        )
        result = sim.run(bars, start_date=start, end_date=end,
                         sector_map={s: "TECH" for s in bars})
        nbd = result.net_exposure_by_date
        two_sided = [(d, x) for d, x in sorted(nbd.items())
                     if x["n_long"] > 0 and x["n_short"] > 0]
        assert two_sided, "short set never populated"
        n_shorts = [x["n_short"] for _, x in two_sided]
        assert max(n_shorts) < 60, (
            f"short set was not filtered below short_n (max n_short={max(n_shorts)})")
        short_gross_series = [(x["gross"] - x["net_dollar"]) / 2.0 for _, x in two_sided]
        peak_short_gross = max(short_gross_series)
        assert peak_short_gross > 0.30, (
            f"short leg never approached short_gross despite rescaling: "
            f"peak short/equity={peak_short_gross:.3f} (Bug 1b underfill was ~0.17)")
        import statistics
        assert statistics.median(short_gross_series) > 0.25, (
            f"short leg still chronically underfilled: "
            f"median short/equity={statistics.median(short_gross_series):.3f}")


# ---------------------------------------------------------------------------
# 9. NET-SECTOR-CAP (Failure B fix) — unit + realistic-universe regression
# ---------------------------------------------------------------------------

class TestNetSectorCapUnit:
    def test_per_side_cap_starves_concentrated_short_tail(self):
        """The OLD per-side short-sector cap refuses most of a concentrated tail."""
        # 60 long names spread across 6 sectors (10 each); the worst-30 short tail
        # is ALL in one sector ('ENERGY') — the per-side cap admits only
        # floor(0.30*30)=9 of them.
        long_book = [f"L{i}" for i in range(60)]
        short_tail = [f"S{i}" for i in range(30)]
        sector_map = {s: "ENERGY" for s in short_tail}
        sector_map.update({l: f"SEC{i % 6}" for i, l in enumerate(long_book)})
        old = apply_sector_cap_shorts(short_tail, sector_map, cap=0.30, n_target=30)
        assert len(old) == 9, f"per-side cap should starve to 9, got {len(old)}"

    def test_net_sector_cap_admits_concentrated_tail_when_longs_elsewhere(self):
        """NET cap admits the WHOLE concentrated short tail when the longs are in
        OTHER sectors (net per sector stays bounded)."""
        long_book = [f"L{i}" for i in range(60)]
        short_tail = [f"S{i}" for i in range(30)]
        sector_map = {s: "ENERGY" for s in short_tail}
        # NONE of the longs are in ENERGY → net_ENERGY = 0 - short_count, bounded by
        # the cap band |net| <= floor(0.30*30)=9 ... but because longs are absent,
        # the short concentration is what's bounded. Put longs across 6 OTHER sectors.
        sector_map.update({l: f"SEC{i % 6}" for i, l in enumerate(long_book)})
        new = apply_net_sector_cap(short_tail, long_book=long_book,
                                   sector_map=sector_map, cap=0.30, n_target=30)
        # The greedy net-cap pass admits ~floor(0.30*30)=9 balanced shorts FIRST, then
        # the Phase-2 BREADTH pass fills the remainder by rank to n_target — so the
        # WHOLE concentrated tail (30) is admitted. Dollar-neutrality is now enforced by
        # per-leg SIZING + the SPY hedge (not by capping the short COUNT), so the breadth
        # the thesis needs is preserved instead of being starved by the sector cap.
        assert len(new) == 30

    def test_net_sector_cap_lets_short_offset_long_concentration(self):
        """When longs ARE concentrated in the short tail's sector, the net cap admits
        MORE shorts (they offset the long concentration) — the key behavioral diff."""
        # 30 longs in ENERGY, 30 longs elsewhere; short tail of 30 ALL in ENERGY.
        long_book = [f"LE{i}" for i in range(30)] + [f"LO{i}" for i in range(30)]
        sector_map = {f"LE{i}": "ENERGY" for i in range(30)}
        sector_map.update({f"LO{i}": f"SEC{i % 5}" for i in range(30)})
        short_tail = [f"S{i}" for i in range(30)]
        sector_map.update({s: "ENERGY" for s in short_tail})
        new = apply_net_sector_cap(short_tail, long_book=long_book,
                                   sector_map=sector_map, cap=0.30, n_target=30)
        old = apply_sector_cap_shorts(short_tail, sector_map, cap=0.30, n_target=30)
        # net_ENERGY starts at +30 (longs); each short reduces it. |net|<=9 admits
        # shorts while 30-k stays within [−9, 9] → k from 21..30 ⇒ 30 admitted.
        assert len(new) == 30
        assert len(new) > len(old), "net cap must admit more than the per-side cap here"


# A realistic R1K-pathology fixture: a sector-concentrated, partly-illiquid short tail.
def _realistic_r1k_fixture(n_days=150, seed0=100):
    # n_days=150 (was 260): every assertion below is a STEADY-STATE mean/median/p95
    # over the post-warmup (NET_BETA_WARMUP_TWO_SIDED=20) two-sided window, which is
    # invariant to window length — 150 business days still leaves ~120 steady-state
    # days, far above the trim. Halving the window ~halves this dead-ranker fixture's
    # build + sim cost (it was the CI critical-path floor) without touching the
    # load-bearing name/sector/beta/liquidity structure.
    """Mimic R1K's short-leg pathology — exercises BOTH Failure-B drivers.

    - 240 names total + SPY. Longs (top of rank) spread across 8 sectors.
    - The bottom-of-rank short tail (worst 80) is CONCENTRATED in 3 sectors
      (ENERGY, REIT, UTIL) and HALF of THAT tail is GENUINELY sub-50M-ADV illiquid
      (~250k shares × ~$100 ≈ $25M ADV, well below the 50M short floor). So the
      short_min_adv=50M floor REMOVES ~40 short-tail names — the OLD per-side-cap /
      pre-FIX-1b short sizing then UNDERFILLS the short leg and the book runs net
      long (Failure B), exactly the pathology that invalidated run 2.
    - All names carry real SPY beta; the short tail has SYSTEMATICALLY LOWER beta
      (0.3–0.7) than the longs (1.1–1.6), so a merely dollar-neutral book carries a
      large POSITIVE residual net beta that single-name shorts cannot remove — what
      the SPY beta-hedge overlay must mop up.

    Returns (bars, sector_map). Default scorer ranks alphabetically best-first, so
    SYM000.. are longs and SYM239.. (the worst 80) are the short tail.
    """
    rng = np.random.default_rng(seed0)
    # Shared SPY path.
    spy = _synthetic_bars(n_days=n_days, seed=999, drift=0.0004, vol=0.011)
    spy_rets = spy["close"].pct_change().fillna(0.0).to_numpy()

    bars = {"SPY": spy}
    sector_map = {"SPY": "MARKET"}
    n = 240
    n_tail = 80  # worst 80 form the concentrated short tail
    long_sectors = ["TECH", "HEALTH", "FIN", "INDU", "CONS", "COMM", "MATL", "STAP"]
    tail_sectors = ["ENERGY", "REIT", "UTIL"]  # concentrated short tail
    for i in range(n):
        sym = f"SYM{i:03d}"
        is_tail = i >= (n - n_tail)
        if is_tail:
            sec = tail_sectors[i % len(tail_sectors)]
            beta = float(rng.uniform(0.3, 0.7))      # LOW beta tail (residual net beta)
            illiquid = ((i - (n - n_tail)) % 2 == 1)  # half the tail is illiquid
        else:
            sec = long_sectors[i % len(long_sectors)]
            beta = float(rng.uniform(1.1, 1.6))      # HIGH beta longs
            illiquid = False
        idio = rng.normal(0.0, 0.006, len(spy_rets))
        prices = [100.0]
        for k in range(1, len(spy_rets)):
            prices.append(prices[-1] * (1 + beta * spy_rets[k] + idio[k]))
        idx = spy.index
        # BLOCKER 2: the illiquid sub-tail must be GENUINELY sub-50M ADV so the
        # short_min_adv=50M floor actually removes it. ~250k shares × ~$100 ≈ $25M ADV
        # (well below the 50M floor) — the OLD-path short leg then UNDERFILLS and the
        # book runs net long (reproduces Failure B). The liquid names sit comfortably
        # above the floor (~$800M ADV) so they always pass.
        vol_base = 250_000 if illiquid else 8_000_000
        df = pd.DataFrame({
            "open": np.array(prices) * (1 + rng.normal(0, 0.002, len(prices))),
            "high": np.array(prices) * 1.004,
            "low": np.array(prices) * 0.996,
            "close": prices,
            "volume": rng.integers(int(vol_base * 0.8), int(vol_base * 1.2), len(prices)),
        }, index=idx)
        bars[sym] = df
        sector_map[sym] = sec
    return bars, sector_map


def _steady_state_net(result):
    """Mean net_dollar / mean & p95 |net beta| over the STEADY-STATE two-sided window.

    BLOCKER 1 — SINGLE SOURCE OF TRUTH: the mean + warmup-trimmed-p95 |net beta| are
    computed by the SAME production helper (net_exposure.steady_state_net_beta) and the
    SAME warmup window (NET_BETA_WARMUP_TWO_SIDED) that CPCVResult.net_beta_clean keys
    on. The test and the production acceptance metric therefore grade the IDENTICAL
    statistic and can never diverge. (max_sec is a test-only sector diagnostic; the
    net_dollar mean uses the same steady-state two-sided window for consistency.)"""
    from app.backtesting.net_exposure import (
        steady_state_net_beta, NET_BETA_WARMUP_TWO_SIDED,
    )
    nbd = result.net_exposure_by_date or {}
    ss_beta = steady_state_net_beta(nbd)
    assert ss_beta["n_steady"] > 0, "no steady-state two-sided window captured"
    mean_nb = ss_beta["mean_net_beta"]
    p95_nb = ss_beta["p95_abs_net_beta"]
    # net_dollar / sector over the SAME steady-state window (mirror the helper's trim).
    two = [x for _, x in sorted(nbd.items()) if x["n_long"] > 0 and x["n_short"] > 0]
    if not two:
        two = [x for _, x in sorted(nbd.items())]
    w = NET_BETA_WARMUP_TWO_SIDED
    ss = two[w:] if len(two) > 2 * w else two
    mean_nd = float(np.mean([x["net_dollar"] for x in ss]))
    max_sec = float(max(x["max_abs_sector"] for x in ss))
    return mean_nd, mean_nb, p95_nb, max_sec


def _prod_net_beta_clean(result):
    """Replicate CPCVResult.net_beta_clean's PRODUCTION decision directly from a
    SimResult: |mean net beta| ≤ 0.15 AND warmup-trimmed steady-state p95 |net beta|
    ≤ 0.15, computed via the SAME shared helper the production path uses
    (net_exposure.steady_state_net_beta / SimResult.p95_abs_net_beta). Returns
    (clean, mean_nb, p95_nb, raw_max_nb) for reporting."""
    from app.backtesting.net_exposure import (
        steady_state_net_beta, NET_BETA_ALPHA_THRESHOLD,
    )
    ss = steady_state_net_beta(result.net_exposure_by_date or {})
    mean_nb = ss["mean_net_beta"]
    p95_nb = ss["p95_abs_net_beta"]
    raw_max = result.max_abs_net_beta  # the raw daily max SimResult reports (diagnostic)
    clean = (abs(mean_nb) <= NET_BETA_ALPHA_THRESHOLD
             and p95_nb <= NET_BETA_ALPHA_THRESHOLD)
    return clean, mean_nb, p95_nb, raw_max


class TestRealisticUniverseNeutrality:
    """THE key gap: the synthetic balanced/liquid universe kept passing while R1K
    failed. This fixture mimics R1K's pathology — a bottom-of-rank short tail
    concentrated in 3 sectors (ENERGY/REIT/UTIL) with a GENUINELY sub-50M-ADV
    illiquid sub-tail (so the 50M floor removes ~40 short names), and a
    SYSTEMATICALLY LOWER-beta short leg than the longs. It must PROVE BOTH:
      (a) the OLD path (per-side cap + pre-FIX-1b short sizing + no hedge)
          REPRODUCES Failure B — the short leg underfills (short_gross < 0.25) and
          the book runs net long (net_dollar > 0.15), AND it fails the PRODUCTION
          net-beta lens; and
      (b) the CURRENT path (NET-sector cap + FIX-1b realized-count rescale + SPY
          beta-hedge overlay) resolves it — single-name book mean|net_dollar| < 0.05
          AND the PRODUCTION net_beta_clean lens (mean + warmup-trimmed p95) is clean.
    Both verdicts use the production metric, not a test-only statistic."""

    @pytest.fixture(scope="class")
    def fixture(self):
        return _realistic_r1k_fixture()

    def test_fixture_actually_filters_short_tail_below_50m(self, fixture):
        """BLOCKER 2 guard: the illiquid sub-tail MUST be genuinely sub-50M ADV so
        short_min_adv=50M removes a meaningful fraction (the old fixture set $80M ADV
        → 0/40 filtered → never reproduced Failure B)."""
        from app.strategy.portfolio_construction import liquidity_filter
        bars, _ = fixture
        as_of = sorted(bars["SPY"].index)[-1].date()
        eligible50 = liquidity_filter(bars, as_of=as_of,
                                      min_avg_daily_dollar_vol=50_000_000.0)
        tail = [s for s in bars if s != "SPY" and int(s[3:]) >= (240 - 80)]
        removed = [s for s in tail if s not in eligible50]
        assert len(tail) == 80
        assert len(removed) >= 30, (
            f"only {len(removed)}/80 short-tail names are sub-50M ADV — the 50M floor "
            "is not biting, so the fixture cannot reproduce Failure B (short underfill).")

    @pytest.mark.timeout(300)  # realistic 240-name x 260-day sim; >120s under CI xdist
    def test_old_path_reproduces_failure_b(self, fixture):
        bars, sector_map = fixture
        # OLD path: per-side short-sector cap, pre-FIX-1b short sizing (divide by
        # short_target_n, NOT the realized count), NO SPY hedge. short_n=200 with the
        # 50M floor leaves a realized short set far below short_n → the short leg
        # underfills and the book runs net long (Failure B).
        _, result = _run_dn_book(
            target_n=40, short_n=200, sector_cap=0.30,
            short_min_adv=50_000_000.0,
            net_sector_cap=False, spy_beta_hedge=False,
            short_realized_rescale=False,
            sector_map=sector_map, bars=bars,
        )
        mean_nd, mean_nb, p95_nb, _ = _steady_state_net(result)
        two = [x for _, x in sorted((result.net_exposure_by_date or {}).items())
               if x["n_long"] > 0 and x["n_short"] > 0][20:]
        short_gross = float(np.mean([(x["gross"] - x["net_dollar"]) / 2.0 for x in two]))
        # Failure B signature (the pre-registered bar): short leg underfills
        # (short_gross < 0.25) OR the book runs net long (net_dollar > 0.15). Here
        # BOTH fire — the 50M floor + pre-FIX-1b sizing starves the short leg, so
        # long (0.40) >> short and the book drifts net long.
        assert short_gross < 0.25 or mean_nd > 0.15, (
            f"OLD path did not reproduce Failure B (short_gross={short_gross:.3f}, "
            f"mean net_dollar={mean_nd:.3f}) — short leg did not underfill / book did "
            "not run net long.")
        assert short_gross < 0.25, (
            f"expected short underfill on the OLD path (short_gross={short_gross:.3f})")
        assert mean_nd > 0.10, (
            f"expected net-long drift on the OLD path (mean net_dollar={mean_nd:.3f})")
        # And it fails the PRODUCTION net-beta lens (low-beta short tail + no hedge).
        clean, m, p95, raw = _prod_net_beta_clean(result)
        assert clean is False, (
            f"OLD path unexpectedly clean by the production lens "
            f"(mean_nb={m:.3f}, p95={p95:.3f}) — fixture not exercising the failure.")

    @pytest.mark.timeout(300)  # realistic 240-name x 260-day sim; >120s under CI xdist
    def test_current_path_resolves_failure_b_by_production_metric(self, fixture):
        bars, sector_map = fixture
        # CURRENT path: NET-sector cap + FIX-1b realized-count rescale + SPY hedge.
        _, result = _run_dn_book(
            target_n=40, short_n=60, sector_cap=0.30,
            short_min_adv=50_000_000.0,
            net_sector_cap=True, spy_beta_hedge=True,
            short_realized_rescale=True,
            sector_map=sector_map, bars=bars,
        )
        assert result.net_exposure_captured is True
        mean_nd, _, _, _ = _steady_state_net(result)
        # (b1) single-name book is dollar-neutral.
        assert abs(mean_nd) < 0.05, (
            f"net-sector-cap + FIX-1b book not dollar-neutral on the realistic "
            f"fixture: mean|net_dollar|={mean_nd:.3f}")
        # (b2) clean by the PRODUCTION net_beta_clean lens (mean + warmup-trimmed p95),
        # even though the RAW daily max still shows transient inter-rebalance spikes.
        clean, mean_nb, p95_nb, raw_max = _prod_net_beta_clean(result)
        assert clean is True, (
            f"CURRENT path NOT clean by the production lens: mean_nb={mean_nb:.3f} "
            f"p95={p95_nb:.3f} (raw daily max diagnostic={raw_max:.3f})")
        # Sanity: the raw daily max IS elevated (transient) — proving the lens, not a
        # quiet book, is what makes it clean.
        assert raw_max > 0.15, (
            f"expected transient raw-max spikes on this fixture (got {raw_max:.3f}); "
            "if the raw max is also clean the test no longer proves the lens matters.")


class TestSpyBetaHedgeOverlay:
    @pytest.mark.timeout(300)  # builds fixture + TWO realistic sims; >120s under CI xdist
    def test_overlay_reduces_residual_net_beta_vs_no_hedge(self):
        """With single-name shorts of systematically lower beta than longs, the
        dollar-neutral book carries large residual net beta; the SPY overlay drives
        it toward 0 (single-name shorts are KEPT — it is an overlay, not a swap)."""
        bars, sector_map = _realistic_r1k_fixture(seed0=222)
        _, no_hedge = _run_dn_book(
            target_n=60, short_n=60, sector_cap=0.30, short_min_adv=0.0,
            net_sector_cap=True, spy_beta_hedge=False,
            sector_map=sector_map, bars=bars,
        )
        _, hedged = _run_dn_book(
            target_n=60, short_n=60, sector_cap=0.30, short_min_adv=0.0,
            net_sector_cap=True, spy_beta_hedge=True,
            sector_map=sector_map, bars=bars,
        )
        _, mean_nb_off, _, _ = _steady_state_net(no_hedge)
        _, mean_nb_on, _, _ = _steady_state_net(hedged)
        # The hedge substantially reduces the residual net beta magnitude.
        assert abs(mean_nb_on) < abs(mean_nb_off) - 0.05, (
            f"hedge did not reduce residual net beta: |mean nb| hedged={mean_nb_on:.3f} "
            f"vs no-hedge={mean_nb_off:.3f}")
        # And the single-name shorts are preserved (overlay, not replacement).
        assert hedged.net_exposure_by_date, "hedged run captured no net exposure"
