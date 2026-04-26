"""
Tests for Tier 3 IntradayAgentSimulator — agent-driven intraday backtest.

Uses mock model + synthetic 5-min bars. Validates PM scoring, RM rules,
ORB gate, exit logic, and portfolio accounting.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, time, datetime
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _5min_bars(
    n_per_day: int = 78,
    n_days: int = 5,
    base: float = 100.0,
    drift_per_bar: float = 0.0001,
    start_date: str = "2024-01-02",
) -> pd.DataFrame:
    """n_days of 5-min OHLCV bars (n_per_day bars per day)."""
    rows = []
    d = pd.Timestamp(start_date)
    bar_num = 0
    for _ in range(n_days):
        session_open = d.replace(hour=9, minute=30)
        for i in range(n_per_day):
            ts = session_open + pd.Timedelta(minutes=5 * i)
            price = base * (1 + drift_per_bar) ** bar_num
            rows.append({
                "open":   price * 0.999,
                "high":   price * 1.003,
                "low":    price * 0.997,
                "close":  price,
                "volume": 10_000,
            })
            bar_num += 1
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)

    idx = []
    d = pd.Timestamp(start_date)
    for _ in range(n_days):
        session_open = d.replace(hour=9, minute=30)
        for i in range(n_per_day):
            idx.append(session_open + pd.Timedelta(minutes=5 * i))
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _mock_model(proba: float = 0.7) -> MagicMock:
    m = MagicMock()
    m.is_trained = True
    m.feature_names = None
    def _predict(X):
        n = len(X)
        return np.ones(n, dtype=int), np.full(n, proba)
    m.predict.side_effect = _predict
    return m


def _make_sim(model=None, max_positions: int = 5):
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
    from app.agents.risk_rules import RiskLimits
    limits = RiskLimits(MAX_OPEN_POSITIONS=max_positions, MAX_POSITION_SIZE_PCT=0.20)
    return IntradayAgentSimulator(
        model=model or _mock_model(),
        limits=limits,
        min_confidence=0.50,
        top_n=5,
    )


# ── Tests: Initialization ─────────────────────────────────────────────────────

class TestIntradayAgentSimulatorInit:
    def test_imports(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator()
        assert sim.model is None

    def test_no_model_returns_empty(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.backtesting.strategy_simulator import SimResult
        sim = IntradayAgentSimulator(model=None)
        res = sim.run({"AAPL": _5min_bars()})
        assert isinstance(res, SimResult)
        assert res.total_trades == 0
        assert res.ending_capital == sim.starting_capital

    def test_empty_data_returns_empty(self):
        sim = _make_sim()
        res = sim.run({})
        assert res.total_trades == 0

    def test_untrained_model_returns_empty(self):
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        m = MagicMock()
        m.is_trained = False
        sim = IntradayAgentSimulator(model=m)
        res = sim.run({"AAPL": _5min_bars()})
        assert res.total_trades == 0


# ── Tests: PM Scoring ─────────────────────────────────────────────────────────

class TestIntradayPMScoring:
    def test_min_confidence_filters_low_scores(self):
        """Symbols below min_confidence should not be proposed."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits
        model = _mock_model(0.3)
        sim = IntradayAgentSimulator(model=model, min_confidence=0.50,
                                     limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))
        mock_feats = {"feat1": 0.5, "orb_breakout": 1.0}
        proposals = sim._pm_score({"AAPL": mock_feats})
        assert len(proposals) == 0

    def test_top_n_limits_proposals(self):
        """PM should not propose more than top_n symbols."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits
        model = _mock_model(0.9)
        sim = IntradayAgentSimulator(model=model, min_confidence=0.50, top_n=2,
                                     limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))
        sym_feats = {f"S{i}": {"feat1": 0.5, "orb_breakout": 1.0} for i in range(10)}
        proposals = sim._pm_score(sym_feats)
        assert len(proposals) <= 2

    def test_model_called_once_with_batch(self):
        """PM must batch all symbols in a single model.predict() call."""
        model = _mock_model(0.8)
        sim = _make_sim(model=model)
        sym_feats = {f"S{i}": {"feat1": float(i)} for i in range(6)}
        sim._pm_score(sym_feats)
        assert model.predict.call_count == 1
        X = model.predict.call_args[0][0]
        assert X.shape[0] == 6


# ── Tests: ORB features (no gate — Phase 46 removed hard ORB gate) ───────────
# The hard ORB breakout gate was removed in Phase 46 because it starved trades
# in range-bound markets. orb_breakout is now a model input feature only.

class TestORBGate:
    def test_inside_orb_does_not_block_entry(self):
        """Phase 46: hard ORB gate removed — inside-ORB entries are allowed."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits

        model = _mock_model(0.9)
        sim = IntradayAgentSimulator(model=model, min_confidence=0.5,
                                     limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))

        bars = _5min_bars(n_days=3)
        syms = {"AAPL": bars}

        # orb_breakout=0 (inside ORB) — should NOT block entry anymore
        mock_feats = {"orb_breakout": 0.0, "feat1": 0.5}
        with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
            with patch("app.backtesting.intraday_agent_simulator.compute_intraday_features",
                       return_value=mock_feats):
                res = sim.run(syms)
        assert res.total_trades >= 1, "Inside-ORB entries should be allowed (soft ORB gate)"

    def test_high_confidence_score_allows_entry(self):
        """High model confidence (above min_confidence) should allow entry."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits

        model = _mock_model(0.9)
        sim = IntradayAgentSimulator(model=model, min_confidence=0.5,
                                     limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))

        bars = _5min_bars(n_days=3)
        syms = {"AAPL": bars}

        mock_feats = {"orb_breakout": 1.0, "volume_surge": 1.5, "feat1": 0.5}
        with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
            with patch("app.backtesting.intraday_agent_simulator.compute_intraday_features",
                       return_value=mock_feats):
                res = sim.run(syms)
        assert res.total_trades >= 1


# ── Tests: RM Rules ───────────────────────────────────────────────────────────

class TestIntradayRMRules:
    def test_daily_loss_limit_blocks_entries(self):
        """validate_daily_loss should reject new entries after limit is hit."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits
        from app.backtesting.agent_simulator import validate_daily_loss

        limits = RiskLimits(MAX_DAILY_LOSS_PCT=0.02)
        ok, _ = validate_daily_loss(-3000.0, 100_000.0, limits)
        assert not ok

        ok, _ = validate_daily_loss(-1000.0, 100_000.0, limits)
        assert ok

    def test_max_positions_enforced(self):
        """Concurrent intraday positions should not exceed MAX_OPEN_POSITIONS."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits

        model = _mock_model(0.9)
        sim = IntradayAgentSimulator(
            model=model,
            starting_capital=500_000.0,
            limits=RiskLimits(MAX_OPEN_POSITIONS=2, MAX_POSITION_SIZE_PCT=0.20),
            min_confidence=0.5, top_n=5,
        )
        bars = _5min_bars(n_days=3)
        syms = {f"S{i}": bars.copy() for i in range(5)}

        mock_feats = {"orb_breakout": 1.0, "volume_surge": 1.5, "feat1": 0.5}
        proposals = [(f"S{i}", 0.9) for i in range(5)]
        with patch.object(sim, "_pm_score", return_value=proposals):
            with patch("app.backtesting.intraday_agent_simulator.compute_intraday_features",
                       return_value=mock_feats):
                res = sim.run(syms)
        # At most 2 positions per day × n_days
        assert res.total_trades <= 2 * 3


# ── Tests: Exit Logic ─────────────────────────────────────────────────────────

class TestIntradayExitLogic:
    def test_stop_hit(self):
        """Position should close on STOP when low pierces stop price."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator, _IntradayPosition
        sim = _make_sim()
        pos = _IntradayPosition(
            symbol="AAPL", entry_date=date(2024, 1, 2), entry_bar_idx=12,
            entry_price=100.0, stop_price=99.7, target_price=100.5,
            quantity=10,
        )
        bars = _5min_bars(n_days=1).iloc[:24].copy()
        bars.iloc[3, bars.columns.get_loc("low")] = 99.5  # pierce stop
        exit_price, reason, _ = sim._simulate_exit(pos, bars)
        assert reason == "STOP"
        assert abs(exit_price - 99.7) < 0.01

    def test_target_hit(self):
        """Position should close on TARGET when high exceeds target price."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator, _IntradayPosition
        sim = _make_sim()
        pos = _IntradayPosition(
            symbol="AAPL", entry_date=date(2024, 1, 2), entry_bar_idx=12,
            entry_price=100.0, stop_price=80.0,  # far below synthetic lows (~99.7)
            target_price=100.5,
            quantity=10,
        )
        bars = _5min_bars(n_days=1).iloc[:24].copy()
        bars.iloc[5, bars.columns.get_loc("high")] = 101.0  # pierce target
        exit_price, reason, _ = sim._simulate_exit(pos, bars)
        assert reason == "TARGET"
        assert abs(exit_price - 100.5) < 0.01

    def test_time_exit(self):
        """Position should close TIME_EXIT when neither stop nor target is hit."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator, _IntradayPosition
        sim = _make_sim()
        pos = _IntradayPosition(
            symbol="AAPL", entry_date=date(2024, 1, 2), entry_bar_idx=12,
            entry_price=100.0, stop_price=90.0, target_price=200.0,
            quantity=10,
        )
        bars = _5min_bars(n_days=1).iloc[:24]
        _, reason, hold = sim._simulate_exit(pos, bars)
        assert reason == "TIME_EXIT"
        assert hold == len(bars)


# ── Tests: Portfolio Accounting ────────────────────────────────────────────────

class TestIntradayPortfolioAccounting:
    def test_result_has_required_fields(self):
        """SimResult should have all expected fields."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.backtesting.strategy_simulator import SimResult

        sim = IntradayAgentSimulator(model=None)
        res = sim.run({})
        assert isinstance(res, SimResult)
        assert res.model_type == "intraday_agent"
        assert isinstance(res.equity_curve, list)
        assert isinstance(res.monthly_pnl, dict)
        assert isinstance(res.exit_breakdown, dict)

    def test_print_report_does_not_raise(self):
        """print_report() should not raise on empty result."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        sim = IntradayAgentSimulator(model=None)
        sim.run({}).print_report()

    def test_transaction_costs_applied(self):
        """Transaction costs should be non-zero when trades are executed."""
        from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
        from app.agents.risk_rules import RiskLimits

        model = _mock_model(0.9)
        sim = IntradayAgentSimulator(model=model, min_confidence=0.5,
                                     limits=RiskLimits(MAX_POSITION_SIZE_PCT=0.20))
        bars = _5min_bars(n_days=3)
        mock_feats = {"orb_breakout": 1.0, "volume_surge": 1.5, "feat1": 0.5}
        with patch.object(sim, "_pm_score", return_value=[("AAPL", 0.9)]):
            with patch("app.backtesting.intraday_agent_simulator.compute_intraday_features",
                       return_value=mock_feats):
                res = sim.run({"AAPL": bars})
        if res.total_trades > 0:
            assert res.transaction_costs_total > 0
