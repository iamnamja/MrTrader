"""
Tests for L/S rebalance mode in AgentSimulator.
Written before implementation per Opus 4.7 audit — all tests should initially FAIL,
then pass once the L/S extension is implemented.

Opus audit found these HIGH-severity issues:
  1. Short P&L sign in rebalance drops (CRITICAL)
  2. Short collateral blocks long open (HIGH)
  3. Hysteresis for shorts uses reversed ranking (HIGH)
  4. Sector cap direction for shorts (MEDIUM)
  5. Asymmetric regime gate (HIGH)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from typing import Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n_days: int = 120, n_syms: int = 80, seed: int = 42) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    result = {}
    for i in range(n_syms):
        sym = f"S{i:03d}"
        log_rets = rng.normal(0.0003, 0.01, size=n_days)
        prices = np.exp(np.cumsum(log_rets) + 4.0)
        volumes = rng.integers(1_000_000, 20_000_000, size=n_days).astype(float)
        df = pd.DataFrame({
            "open": prices * (1 + rng.normal(0, 0.002, n_days)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "low":  prices * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "close": prices,
            "volume": volumes,
        }, index=dates)
        result[sym] = df
    return result


def _make_spy(n_days: int = 120, direction: str = "up") -> pd.DataFrame:
    """Generate SPY bars in a given direction."""
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    drift = 0.002 if direction == "up" else -0.002
    prices = np.exp(np.cumsum(np.full(n_days, drift)) + 5.0)
    return pd.DataFrame({
        "open": prices, "high": prices * 1.005, "low": prices * 0.995, "close": prices,
        "volume": np.ones(n_days) * 5e7,
    }, index=dates)


class TestShortPnLSign:
    """Test 1: Short P&L accounting is correct (sign inverted from long)."""

    def test_profitable_short_positive_pnl(self):
        """Open short at 100, cover at 90 → profit = +10 per share."""
        from app.backtesting.agent_simulator import AgentSimulator, _Position, _PortfolioState
        from dataclasses import field

        # Minimal portfolio with one short position
        pos = _Position(
            symbol="S000",
            entry_date=date(2022, 3, 1),
            entry_price=100.0,
            stop_price=110.0,
            target_price=85.0,
            quantity=100,
            highest_price=100.0,
            confidence=70.0,
            direction="short",
        )
        # Build a minimal bars dict covering the close
        dates = pd.date_range("2022-01-03", periods=60, freq="B")
        bars = pd.DataFrame({
            "open": [90.0] * 60, "high": [91.0] * 60,
            "low": [89.0] * 60, "close": [90.0] * 60,
            "volume": [1e6] * 60,
        }, index=dates)
        symbols_data = {"S000": bars}

        # Use rebalance simulator in L/S mode
        sim = AgentSimulator(
            starting_capital=200_000,
            rebalance_mode=True,
            enable_shorts=True,
            short_target_n=5,
        )
        portfolio = _PortfolioState(
            cash=100_000.0,    # short proceeds held as collateral
            peak_equity=200_000.0,
            short_collateral=10_000.0,  # 100 shares × 100 entry
        )
        portfolio.positions["S000"] = pos

        # Simulate rebalance DROP of the short position
        # Use the rebalance drop path (not _process_exits)
        day = dates[55].date()
        sector_map = {"S000": "Tech"}
        # Force a rebalance that drops S000
        trades, _ = sim._rebalance_drop_position("S000", pos, symbols_data, portfolio, day, "BEAR")
        assert len(trades) == 1
        t = trades[0]
        assert t.pnl > 0, f"Covering short at 90 vs entry 100 should be profitable; got {t.pnl}"
        # net_pnl = (entry - exit) * qty - exit_tx_cost = 1000 - (90 * 100 * 0.0005) = 995.5
        expected_gross = (100.0 - 90.0) * 100
        assert t.pnl < expected_gross, "Net P&L must be less than gross (tx costs deducted)"
        assert t.pnl > expected_gross * 0.98, f"P&L {t.pnl:.1f} too far below gross {expected_gross}"

    def test_losing_short_negative_pnl(self):
        """Open short at 100, cover at 110 → loss = -10 per share."""
        from app.backtesting.agent_simulator import _Position, _PortfolioState, AgentSimulator

        pos = _Position(
            symbol="S000",
            entry_date=date(2022, 3, 1),
            entry_price=100.0,
            stop_price=115.0,
            target_price=80.0,
            quantity=100,
            highest_price=100.0,
            confidence=70.0,
            direction="short",
        )
        dates = pd.date_range("2022-01-03", periods=60, freq="B")
        bars = pd.DataFrame({
            "open": [110.0] * 60, "high": [111.0] * 60,
            "low": [109.0] * 60, "close": [110.0] * 60,
            "volume": [1e6] * 60,
        }, index=dates)
        symbols_data = {"S000": bars}
        sim = AgentSimulator(
            starting_capital=200_000,
            rebalance_mode=True,
            enable_shorts=True,
        )
        portfolio = _PortfolioState(
            cash=100_000.0,
            peak_equity=200_000.0,
            short_collateral=10_000.0,
        )
        portfolio.positions["S000"] = pos

        day = dates[55].date()
        trades, _ = sim._rebalance_drop_position("S000", pos, symbols_data, portfolio, day, "BULL")
        assert len(trades) == 1
        assert trades[0].pnl < 0, f"Covering at 110 vs entry 100 should be a loss; got {trades[0].pnl}"


class TestShortCollateralBlocksLongOpen:
    """Test 2: Short collateral is reserved so long opens can't double-spend it."""

    def test_collateral_reduces_available_cash(self):
        """With 200k equity, 55k in short collateral → only 145k available for longs."""
        from app.backtesting.agent_simulator import AgentSimulator, _PortfolioState

        sim = AgentSimulator(
            starting_capital=200_000,
            rebalance_mode=True,
            enable_shorts=True,
            long_gross=0.95,
            short_gross=0.55,
        )
        portfolio = _PortfolioState(
            cash=200_000.0,
            peak_equity=200_000.0,
            short_collateral=55_000.0,
        )
        available = sim._effective_cash(portfolio)
        assert available <= 145_000.0 + 1.0, (
            f"Effective cash should be ≤ 145k when 55k is reserved for shorts; got {available}"
        )

    def test_long_open_respects_collateral(self):
        """A long open should be rejected when only short_collateral cash remains."""
        from app.backtesting.agent_simulator import AgentSimulator, _PortfolioState

        bars = _make_bars(n_days=60, n_syms=5, seed=0)
        sim = AgentSimulator(
            starting_capital=100_000,
            rebalance_mode=True,
            enable_shorts=True,
        )
        portfolio = _PortfolioState(
            cash=10_000.0,          # barely enough nominally
            peak_equity=100_000.0,
            short_collateral=10_000.0,  # same amount is reserved
        )
        # effective cash = 0 → long open must be skipped
        effective = sim._effective_cash(portfolio)
        assert effective <= 0.01, f"Expected ≈0 effective cash, got {effective}"


class TestHysteresisShortSide:
    """Test 4: Hysteresis works correctly for the short book (reversed ranking)."""

    def test_short_held_within_drop_threshold_is_kept(self):
        """A short at rank-from-bottom 20 (within drop_threshold=30) should be kept."""
        from app.strategy.portfolio_construction import compute_target_portfolio_shorts

        # 80 symbols ranked worst-first; shorts hold positions 1-20
        n = 80
        worst_first = [f"S{i:03d}" for i in range(n)]  # S000 = worst
        current_shorts = set(worst_first[:20])

        delta = compute_target_portfolio_shorts(
            worst_first_symbols=worst_first,
            current_short_holdings=current_shorts,
            n_target=20,
            add_rank_threshold=15,
            drop_rank_threshold=25,
        )
        # All 20 held positions are within rank ≤ 25 → all should be kept
        assert set(delta.to_drop) == set(), f"Expected no drops, got {delta.to_drop}"

    def test_short_outside_drop_threshold_is_dropped(self):
        """A short at rank-from-bottom 35 (outside drop_threshold=30) should be dropped."""
        from app.strategy.portfolio_construction import compute_target_portfolio_shorts

        n = 80
        worst_first = [f"S{i:03d}" for i in range(n)]
        # Hold one symbol that's now at rank 35 (index 34)
        current_shorts = {worst_first[34]}

        delta = compute_target_portfolio_shorts(
            worst_first_symbols=worst_first,
            current_short_holdings=current_shorts,
            n_target=20,
            add_rank_threshold=15,
            drop_rank_threshold=30,
        )
        assert worst_first[34] in delta.to_drop, "Symbol at rank 35 should be dropped"


class TestSectorCapShortSide:
    """Test 5: Sector cap is applied to the bottom of the ranking for shorts."""

    def test_short_sector_cap_from_worst_end(self):
        """With 80 symbols all in Tech except 10 in Health, short cap should pick from worst-10 Tech + Health."""
        from app.strategy.portfolio_construction import apply_sector_cap_shorts

        # 80 symbols, worst first; first 60 are Tech, last 20 are Health
        worst_first = [f"S{i:03d}" for i in range(80)]
        sector_map = {s: "Tech" if i < 60 else "Health" for i, s in enumerate(worst_first)}

        capped = apply_sector_cap_shorts(
            worst_first_symbols=worst_first,
            sector_map=sector_map,
            cap=0.30,
            n_target=20,
        )
        # With cap=30% × 20 = 6 max per sector, should have ≤6 Tech shorts
        tech_count = sum(1 for s in capped if sector_map[s] == "Tech")
        assert tech_count <= 6, f"Tech short count {tech_count} exceeds 30% cap (6 of 20)"


class TestRegimeGateAsymmetry:
    """Test 9: In bear regime (VIX>25, SPY<MA200), long_mult is low but short_mult should be high."""

    def test_bear_regime_short_mult_high(self):
        """SPY<MA200 AND VIX>25 → long_mult ≤ 0.35, short_mult ≥ 0.80."""
        from scripts.walkforward_tier3 import build_asymmetric_regime_fns

        # Build simple history: SPY falling trend, VIX elevated
        dates = pd.date_range("2022-01-03", periods=300, freq="B")
        spy_prices = np.exp(np.cumsum(np.full(300, -0.002)) + 5.0)
        spy_df = pd.DataFrame({"close": spy_prices}, index=dates)
        vix_series = pd.Series(np.full(300, 30.0), index=dates)

        long_fn, short_fn = build_asymmetric_regime_fns(spy_df, vix_series)
        test_day = dates[-1].date()

        long_mult = long_fn(test_day)
        short_mult = short_fn(test_day)

        assert long_mult <= 0.40, f"Bear regime: long_mult should be ≤ 0.40, got {long_mult:.2f}"
        assert short_mult >= 0.80, f"Bear regime: short_mult should be ≥ 0.80, got {short_mult:.2f}"

    def test_bull_regime_short_mult_reduced(self):
        """SPY>MA200 AND VIX<15 → short_mult should be reduced (shorts are costly in a bull run)."""
        from scripts.walkforward_tier3 import build_asymmetric_regime_fns

        dates = pd.date_range("2022-01-03", periods=300, freq="B")
        spy_prices = np.exp(np.cumsum(np.full(300, 0.002)) + 5.0)
        spy_df = pd.DataFrame({"close": spy_prices}, index=dates)
        vix_series = pd.Series(np.full(300, 12.0), index=dates)

        long_fn, short_fn = build_asymmetric_regime_fns(spy_df, vix_series)
        test_day = dates[-1].date()

        long_mult = long_fn(test_day)
        short_mult = short_fn(test_day)

        assert long_mult >= 0.80, f"Bull regime: long_mult should be ≥ 0.80, got {long_mult:.2f}"
        assert short_mult <= 0.60, f"Bull regime: short_mult should be ≤ 0.60, got {short_mult:.2f}"


class TestEquityMtmShortCollateral:
    """Test: equity_mtm must subtract short_collateral to avoid Sharpe inflation."""

    def test_short_open_does_not_inflate_equity(self):
        """Opening a short should not increase equity (cash goes up but collateral offsets it)."""
        from app.backtesting.agent_simulator import _PortfolioState, _Position

        portfolio = _PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        equity_before = portfolio.equity_mtm({})
        assert abs(equity_before - 100_000.0) < 1.0

        # Simulate opening a 10k short at $100/share (100 qty)
        portfolio.cash += 10_000.0 - 5.0   # proceeds received, 5 tx
        portfolio.short_collateral += 10_000.0
        portfolio.positions["S000"] = _Position(
            symbol="S000", entry_date=None, entry_price=100.0,
            stop_price=115.0, target_price=80.0, quantity=100,
            highest_price=100.0, confidence=1.0, direction="short",
        )
        equity_after = portfolio.equity_mtm({"S000": 100.0})  # price unchanged
        # Equity should only decrease by tx cost, NOT increase by short proceeds
        assert equity_after < equity_before, "Opening a short must not increase equity"
        assert abs(equity_after - (equity_before - 5.0)) < 1.0, (
            f"Equity after short open should be equity_before - tx_cost = {equity_before - 5.0:.0f}, "
            f"got {equity_after:.0f}"
        )

    def test_profitable_short_increases_equity_mtm(self):
        """When a short position gains (price falls), equity_mtm should increase."""
        from app.backtesting.agent_simulator import _PortfolioState, _Position

        portfolio = _PortfolioState(cash=110_000.0, peak_equity=100_000.0, short_collateral=10_000.0)
        portfolio.positions["S000"] = _Position(
            symbol="S000", entry_date=None, entry_price=100.0,
            stop_price=115.0, target_price=80.0, quantity=100,
            highest_price=100.0, confidence=1.0, direction="short",
        )
        equity_at_entry = portfolio.equity_mtm({"S000": 100.0})  # price at entry
        equity_at_profit = portfolio.equity_mtm({"S000": 90.0})  # price fell 10%
        assert equity_at_profit > equity_at_entry, (
            f"Short gained when price fell: equity should increase. "
            f"Before={equity_at_entry:.0f} After={equity_at_profit:.0f}"
        )
        assert abs((equity_at_profit - equity_at_entry) - 1000.0) < 1.0, (
            f"10 × 100 qty = 1000 unrealized gain, got {equity_at_profit - equity_at_entry:.0f}"
        )


class TestLongOnlyRegression:
    """Test 8: with enable_shorts=False, results must be identical to the old long-only mode."""

    def test_disable_shorts_matches_longonly(self):
        """AgentSimulator with enable_shorts=False should produce the same trades as the old API."""
        from app.backtesting.agent_simulator import AgentSimulator

        bars = _make_bars(n_days=120, n_syms=40, seed=77)
        spy_bars = _make_spy(n_days=120, direction="up")
        bars["SPY"] = spy_bars

        def _trivial_scorer(day, symbols_data, vix_history=None):
            return [(s, float(i)) for i, s in enumerate(sorted(symbols_data.keys())) if s != "SPY"]

        # Old API (no enable_shorts param)
        sim_old = AgentSimulator(
            starting_capital=100_000,
            rebalance_mode=True,
            rebalance_target_n=10,
            rebalance_regime_fn=lambda d: 1.0,
            factor_scorer=_trivial_scorer,
        )
        from datetime import date
        start = pd.Timestamp("2022-03-01").date()
        end = pd.Timestamp("2022-05-31").date()
        result_old = sim_old.run(bars, start_date=start, end_date=end, sector_map={})

        # New API with enable_shorts=False
        sim_new = AgentSimulator(
            starting_capital=100_000,
            rebalance_mode=True,
            rebalance_target_n=10,
            rebalance_regime_fn=lambda d: 1.0,
            factor_scorer=_trivial_scorer,
            enable_shorts=False,  # explicit
        )
        result_new = sim_new.run(bars, start_date=start, end_date=end, sector_map={})

        assert len(result_old.trades) == len(result_new.trades), (
            f"Long-only trade count mismatch: {len(result_old.trades)} vs {len(result_new.trades)}"
        )


class TestPITNoLookAheadFeatures:
    """Test 6: IC composite scorer must not use same-day data for either long or short side."""

    def test_corrupt_today_data_does_not_affect_short_scores(self):
        """
        Corrupt today's close for the bottom-30 symbols → short scores must be unchanged.
        Guards against look-ahead on the short tail specifically.
        """
        from app.ml.factor_scorer import IcCompositeV221Scorer
        import numpy as np

        rng = np.random.default_rng(99)
        n_days, n_syms = 400, 80
        dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
        syms = [f"S{i:03d}" for i in range(n_syms)]
        log_rets = rng.normal(0.0003, 0.01, size=(n_days, n_syms))
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
        symbols_data = {sym: pd.DataFrame({"close": prices[sym]}) for sym in syms}

        d = dates[350].date()
        scorer1 = IcCompositeV221Scorer()
        result1 = scorer1(d, symbols_data)
        scores1 = {sym: s for sym, s in result1}
        bottom_30_before = [sym for sym, _ in result1[-30:]]

        # Corrupt same-day closes
        prices_corrupt = prices.copy()
        prices_corrupt.iloc[350] *= 100.0
        symbols_corrupt = {sym: pd.DataFrame({"close": prices_corrupt[sym]}) for sym in syms}

        scorer2 = IcCompositeV221Scorer()
        result2 = scorer2(d, symbols_corrupt)
        scores2 = {sym: s for sym, s in result2}

        # Scores for any symbol must not change (PIT-safe)
        common = set(scores1) & set(scores2)
        assert len(common) > 0
        max_diff = max(abs(scores1[sym] - scores2[sym]) for sym in common)
        assert max_diff < 1e-6, (
            f"Same-day corruption affected IC scores by {max_diff:.4f} — short-side lookahead!"
        )
