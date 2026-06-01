"""
Tests for PEAD CPCV instrumentation fixes (run_pead_cpcv.PEADStrategy).

Three instrumentation bugs made honest promotion decisions impossible:
  1. n_obs never set → DSR fell back to ~path-count instead of ~250 OOS days.
  2. regime_sharpes never set (no global regime map) → worst_regime_sharpe=None,
     regime gate failed closed.
  3. trade_returns always [] → profit_factor computed as 0.0 instead of real value.

These tests verify the fixes (mirroring scripts/walkforward/strategies/swing.py)
at the unit level — no network, no full CPCV run.
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_pead_cpcv import PEADStrategy  # noqa: E402


class _FakeTrade:
    def __init__(self, pnl_pct):
        self.pnl_pct = pnl_pct


def _make_sim_result(equity_curve, profit_factor=1.0, trades=None, total_trades=0):
    """Build a SimResult-like stub (only the attrs run_fold reads)."""
    return SimpleNamespace(
        exit_breakdown={"STOP": 0},
        total_trades=total_trades,
        sharpe_ratio=0.5,
        total_return_pct=0.1,
        max_drawdown_pct=0.05,
        win_rate=0.55,
        profit_factor=profit_factor,
        equity_curve=equity_curve,
        trades=trades or [],
    )


def _daily_equity_curve(n_points, start=date(2022, 1, 3), step_pct=0.0):
    """A daily (date, equity) curve with one point per trading day."""
    curve = []
    eq = 100_000.0
    d = start
    for _ in range(n_points):
        curve.append((d, eq))
        eq *= (1.0 + step_pct)
        d += timedelta(days=1)
    return curve


def _patch_run_fold_deps(monkeypatch, sim_result, regime_map=None):
    """Stub AgentSimulator + PIT-universe helpers used inside run_fold."""
    import app.backtesting.agent_simulator as agent_mod
    import app.data.universe_history as uni_mod

    class _FakeSim:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            return sim_result

    monkeypatch.setattr(agent_mod, "AgentSimulator", _FakeSim, raising=True)
    # PIT helpers: return everything so the symbols_data passes through.
    monkeypatch.setattr(uni_mod, "pit_union",
                        lambda *a, **k: ["AAPL"], raising=True)
    monkeypatch.setattr(uni_mod, "historical_trade_symbols",
                        lambda *a, **k: [], raising=True)


def _make_strategy(regime_map=None):
    scorer = SimpleNamespace()
    strat = PEADStrategy(scorer=scorer, symbols=["AAPL"])
    strat.symbols_data = {"AAPL": SimpleNamespace()}
    strat.spy_prices = None
    if regime_map is not None:
        strat._global_regime_map = regime_map
    return strat


def test_pead_fold_sets_n_obs(monkeypatch):
    """250-point equity curve → n_obs == 249 (daily-return observations)."""
    eq = _daily_equity_curve(250)
    sim_result = _make_sim_result(eq, profit_factor=1.0, total_trades=5)
    _patch_run_fold_deps(monkeypatch, sim_result)
    strat = _make_strategy(regime_map={})

    fr = strat.run_fold(
        0, 1,
        date(2021, 1, 1), date(2021, 12, 31),
        date(2022, 1, 1), date(2022, 12, 31),
    )
    assert fr.n_obs == 249


def test_pead_fold_profit_factor_from_result(monkeypatch):
    """profit_factor reflects result.profit_factor (1.4), not 0.0 from empty list."""
    eq = _daily_equity_curve(50)
    trades = [_FakeTrade(0.02), _FakeTrade(-0.01)]
    sim_result = _make_sim_result(eq, profit_factor=1.4, trades=trades, total_trades=2)
    _patch_run_fold_deps(monkeypatch, sim_result)
    strat = _make_strategy(regime_map={})

    fr = strat.run_fold(
        0, 1,
        date(2021, 1, 1), date(2021, 12, 31),
        date(2022, 1, 1), date(2022, 12, 31),
    )
    assert fr.profit_factor == pytest.approx(1.4)
    assert fr.profit_factor != 0.0


def test_pead_fetch_builds_regime_map(monkeypatch):
    """fetch_data populates _global_regime_map from load_regime_map."""
    known_map = {date(2022, 1, 3): "BULL", date(2022, 1, 4): "BEAR"}

    # Stub the regime loader (imported lazily inside fetch_data).
    import scripts.walkforward.regime as regime_mod
    monkeypatch.setattr(regime_mod, "load_regime_map",
                        lambda *a, **k: dict(known_map), raising=True)

    # Stub yfinance so fetch_data does no network I/O.
    import pandas as pd
    import yfinance as yf

    def _fake_download(sym, *a, **k):
        idx = pd.date_range("2022-01-03", periods=220, freq="D")
        return pd.DataFrame({"Close": range(220)}, index=idx)

    monkeypatch.setattr(yf, "download", _fake_download, raising=True)

    strat = PEADStrategy(scorer=SimpleNamespace(), symbols=["AAPL"])
    strat.fetch_data(datetime(2022, 1, 1), datetime(2022, 12, 31))

    assert strat._global_regime_map == known_map


def test_pead_fold_populates_regime_sharpes(monkeypatch):
    """Multi-regime curve with >=20 obs/regime → regime_sharpes non-empty."""
    # 30 BULL days then 30 BEAR days (>= REGIME_MIN_OBS=20 each).
    start = date(2022, 1, 3)
    curve = _daily_equity_curve(60, start=start, step_pct=0.001)
    regime_map = {}
    for i in range(60):
        d = start + timedelta(days=i)
        regime_map[d] = "BULL" if i < 30 else "BEAR"

    sim_result = _make_sim_result(curve, profit_factor=1.2, total_trades=10)
    _patch_run_fold_deps(monkeypatch, sim_result)
    strat = _make_strategy(regime_map=regime_map)

    fr = strat.run_fold(
        0, 1,
        date(2021, 1, 1), date(2021, 12, 31),
        start, start + timedelta(days=60),
    )
    assert fr.regime_sharpes  # non-empty
    assert set(fr.regime_sharpes.keys()) <= {"BULL", "BEAR"}
