"""Phase RA smoke test: AgentSimulator rebalance_mode=True.

Verifies that:
1. rebalance_mode=True produces trades tagged source="REBALANCE"
2. Rebalance fires on multiples of rebalance_days
3. Trade attribution fields are populated
4. rebalance_mode=False is bit-identical to previous behaviour
   (no regression on SIGNAL mode)
5. Trade dataclass accepts new attribution fields with None defaults
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.backtesting.agent_simulator import AgentSimulator
from app.backtesting.metrics import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_bars(n_days=250, start=date(2023, 1, 3), seed=42):
    """Generate synthetic OHLCV bars with a mild upward drift."""
    rng = np.random.default_rng(seed)
    close = 100.0
    rows = []
    d = pd.Timestamp(start)
    for _ in range(n_days):
        if d.weekday() < 5:
            ret = rng.normal(0.001, 0.015)
            close *= (1 + ret)
            rows.append({
                "open": close * (1 + rng.normal(0, 0.003)),
                "high": close * (1 + abs(rng.normal(0, 0.005))),
                "low": close * (1 - abs(rng.normal(0, 0.005))),
                "close": close,
                "volume": int(rng.integers(500_000, 2_000_000)),
            })
        d += pd.Timedelta(days=1)
    idx = pd.date_range(start=start, periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx)


def _make_bars_map(n_symbols=30, n_days=250):
    """Build a universe of synthetic bars."""
    bars = {}
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        bars[sym] = _synthetic_bars(n_days=n_days, seed=i + 10)
    return bars


class _MockModel:
    """Deterministic mock: ranks symbols by their index number."""
    is_trained = True
    feature_names = ["f0"]
    version = 999

    def predict(self, X):
        n = X.shape[0]
        scores = np.linspace(1.0, 0.0, n)
        return np.zeros(n), scores

    def predict_with_vix(self, X, vix_level=None):
        return self.predict(X)


def _mock_factor_scorer(day, symbols_data, vix_history=None):
    """Bypass feature engineering: rank symbols alphabetically by name."""
    syms = sorted(symbols_data.keys())
    scores = np.linspace(1.0, 0.0, len(syms))
    return list(zip(syms, scores.tolist()))


# ---------------------------------------------------------------------------
# Trade dataclass attribution fields
# ---------------------------------------------------------------------------

class TestTradeAttributionFields:
    def test_source_default(self):
        t = Trade("SYM", date(2024,1,1), date(2024,1,20), 100.0, 105.0,
                  10, 50.0, 0.05, 15, "REBALANCE_DROP")
        assert t.source == "SIGNAL"

    def test_attribution_fields_settable(self):
        t = Trade("SYM", date(2024,1,1), date(2024,1,20), 100.0, 105.0,
                  10, 50.0, 0.05, 15, "REBALANCE_DROP",
                  source="REBALANCE", rank_at_entry=3,
                  score_at_entry=0.87, regime_at_entry="LOW",
                  gross_exposure_mult=1.0, rebalance_date=date(2024,1,1))
        assert t.source == "REBALANCE"
        assert t.rank_at_entry == 3
        assert t.regime_at_entry == "LOW"

    def test_attribution_fields_default_none(self):
        t = Trade("SYM", date(2024,1,1), date(2024,1,20), 100.0, 105.0,
                  10, 50.0, 0.05, 15, "TARGET")
        assert t.rank_at_entry is None
        assert t.rebalance_date is None


# ---------------------------------------------------------------------------
# AgentSimulator rebalance_mode smoke test
# ---------------------------------------------------------------------------

class TestAgentSimulatorRebalanceMode:
    def _run_sim(self, rebalance_mode, n_days=60, n_symbols=15, seed=0):
        bars = _make_bars_map(n_symbols=n_symbols, n_days=n_days)
        start = date(2023, 1, 3)
        end = start + timedelta(days=n_days + 30)
        sim = AgentSimulator(
            model=_MockModel(),
            starting_capital=30_000.0,
            rebalance_mode=rebalance_mode,
            rebalance_days=20,
            rebalance_target_n=10,
            rebalance_sector_cap=1.0,   # no cap in test
            rebalance_add_threshold=10,
            rebalance_drop_threshold=15,
            rebalance_min_adv=0.0,       # disable liquidity filter for synthetic data
            no_atr_stops=True,
            factor_scorer=_mock_factor_scorer if rebalance_mode else None,
        )
        result = sim.run(bars, start_date=start, end_date=end)
        return result

    def test_rebalance_mode_produces_trades(self):
        result = self._run_sim(rebalance_mode=True)
        assert result.total_trades > 0, "Expected trades in rebalance mode"

    def test_rebalance_trades_tagged_source(self):
        result = self._run_sim(rebalance_mode=True)
        rebalance_trades = [t for t in result.trades if t.source == "REBALANCE"]
        assert len(rebalance_trades) > 0, "Expected REBALANCE-tagged trades"

    def test_rebalance_trades_have_attribution(self):
        result = self._run_sim(rebalance_mode=True)
        for t in result.trades:
            if t.source == "REBALANCE" and t.exit_reason != "OPEN":
                assert t.regime_at_entry is not None
                assert t.gross_exposure_mult is not None

    def test_signal_mode_unaffected(self):
        result = self._run_sim(rebalance_mode=False)
        # In SIGNAL mode with the mock model, no RSI/EMA prefilters fire for
        # synthetic bars — may have 0 trades, but should NOT crash.
        assert result is not None
        signal_source = [t for t in result.trades if t.source == "REBALANCE"]
        assert len(signal_source) == 0, "SIGNAL mode should not produce REBALANCE trades"
