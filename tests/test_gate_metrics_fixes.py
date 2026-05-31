"""
Tests for gate metric fixes:
  BUG-4: profit_factor 999-sentinel cap in WalkForwardReport and CPCVResult
  BUG-5: compute_calmar geometric annualisation (was arithmetic)
  BUG-7: compute_k_ratio log-equity (was raw dollar equity — scale-dependent)
  BUG-19: retrain_cron passes purge_days=85 (not the function default of 10)
"""
from __future__ import annotations

import math
import pytest
from datetime import date


# ---------------------------------------------------------------------------
# BUG-5: compute_calmar — geometric vs arithmetic annualisation
# ---------------------------------------------------------------------------

class TestComputeCalmar:
    def test_geometric_vs_arithmetic_differs_multi_year(self):
        from scripts.walkforward.gates import compute_calmar
        # 40% total return over 2 years, 20% max drawdown
        # Geometric CAGR: (1.40)^(1/2) - 1 = 0.1832 → Calmar = 0.916
        # Arithmetic:  0.40 / 2 = 0.20 → Calmar = 1.00
        result = compute_calmar(total_return_pct=0.40, max_drawdown_pct=0.20, years=2.0)
        expected_geometric = ((1.40 ** 0.5) - 1) / 0.20
        assert abs(result - expected_geometric) < 1e-6

    def test_matches_tier3_implementation(self):
        """gates.compute_calmar and walkforward_tier3._compute_calmar must agree."""
        from scripts.walkforward.gates import compute_calmar
        from scripts.walkforward_tier3 import _compute_calmar as tier3_calmar
        for ret, dd, yrs in [(0.20, 0.10, 1.0), (0.50, 0.25, 3.0), (-0.10, 0.15, 0.5)]:
            assert abs(compute_calmar(ret, dd, yrs) - tier3_calmar(ret, dd, yrs)) < 1e-9

    def test_zero_drawdown_returns_zero(self):
        from scripts.walkforward.gates import compute_calmar
        assert compute_calmar(0.30, 0.0, 1.0) == 0.0

    def test_zero_years_returns_zero(self):
        from scripts.walkforward.gates import compute_calmar
        assert compute_calmar(0.30, 0.10, 0.0) == 0.0

    def test_negative_return_calmar(self):
        from scripts.walkforward.gates import compute_calmar
        result = compute_calmar(-0.10, 0.15, 1.0)
        assert result < 0


# ---------------------------------------------------------------------------
# BUG-7: compute_k_ratio — log-equity, scale-invariant
# ---------------------------------------------------------------------------

class TestComputeKRatio:
    def test_scale_invariant(self):
        """K-ratio must be identical for identical-return curves at different capital levels."""
        import numpy as np
        from scripts.walkforward.gates import compute_k_ratio
        rng = np.random.default_rng(42)
        # Construct realistic equity curve with drift + noise
        rets = 0.001 + rng.normal(0, 0.01, 60)
        base = list(100.0 * np.cumprod(1 + rets))
        scaled = [x * 10 for x in base]
        k1 = compute_k_ratio(base)
        k2 = compute_k_ratio(scaled)
        assert abs(k1 - k2) < 1e-6, f"Scale dependence: {k1} vs {k2}"

    def test_returns_zero_for_non_positive_equity(self):
        from scripts.walkforward.gates import compute_k_ratio
        assert compute_k_ratio([100, 110, 0, 105]) == 0.0

    def test_returns_zero_for_insufficient_data(self):
        from scripts.walkforward.gates import compute_k_ratio
        assert compute_k_ratio([100, 110, 105]) == 0.0

    def test_matches_tier3_implementation(self):
        """gates.compute_k_ratio and walkforward_tier3._compute_k_ratio must agree."""
        from scripts.walkforward.gates import compute_k_ratio
        from scripts.walkforward_tier3 import _compute_k_ratio as tier3_k
        curve = [100.0 * (1.001 ** i) for i in range(100)]
        assert abs(compute_k_ratio(curve) - tier3_k(curve)) < 1e-9

    def test_flat_equity_returns_zero(self):
        from scripts.walkforward.gates import compute_k_ratio
        assert compute_k_ratio([100.0] * 20) == 0.0


# ---------------------------------------------------------------------------
# BUG-4: profit_factor 999-sentinel cap
# ---------------------------------------------------------------------------

class TestProfitFactorCap:
    def _make_report(self, pf_values):
        from scripts.walkforward.gates import WalkForwardReport, FoldResult
        r = WalkForwardReport(model_type="swing")
        for i, pf in enumerate(pf_values):
            r.folds.append(FoldResult(
                fold=i + 1,
                train_start=date(2020, 1, 1), train_end=date(2021, 1, 1),
                test_start=date(2021, 1, 1), test_end=date(2022, 1, 1),
                trades=50, win_rate=0.6, sharpe=1.0, max_drawdown=0.05,
                total_return=0.10, stop_exit_rate=0.2, profit_factor=pf,
            ))
        return r

    def test_999_sentinel_capped_at_5(self):
        r = self._make_report([999.0, 1.5, 1.3])
        # Without cap: (999 + 1.5 + 1.3) / 3 ≈ 333.9; with cap: (5 + 1.5 + 1.3) / 3 ≈ 2.6
        assert r.avg_profit_factor < 10.0, f"PF not capped: {r.avg_profit_factor}"
        assert abs(r.avg_profit_factor - (5.0 + 1.5 + 1.3) / 3) < 1e-6

    def test_normal_pf_unchanged(self):
        r = self._make_report([1.8, 1.4, 2.0])
        assert abs(r.avg_profit_factor - (1.8 + 1.4 + 2.0) / 3) < 1e-6

    def test_cpcv_pf_cap(self):
        from scripts.walkforward.cpcv import CPCVResult
        r = CPCVResult(model_type="swing", n_folds=6, n_paths=2)
        r.path_profit_factors = [999.0, 1.5, 1.3]
        assert r.avg_profit_factor < 10.0
        assert abs(r.avg_profit_factor - (5.0 + 1.5 + 1.3) / 3) < 1e-6


# ---------------------------------------------------------------------------
# BUG-19: retrain_cron passes purge_days=85
# ---------------------------------------------------------------------------

class TestRetrainCronPurgeDays:
    def test_swing_walkforward_called_with_purge_85(self):
        """retrain_cron.run_swing must call run_swing_walkforward with purge_days=85."""
        import inspect
        import ast
        import textwrap
        from pathlib import Path

        src = Path("scripts/retrain_cron.py").read_text(encoding="utf-8")
        # Check that purge_days=85 appears in the run_swing function
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_swing":
                fn_src = ast.get_source_segment(src, node)
                assert "purge_days=85" in fn_src, (
                    "retrain_cron.run_swing does not pass purge_days=85 to "
                    "run_swing_walkforward. BUG-19: default is 10, CLI uses 85."
                )
                assert "embargo_days=85" in fn_src, (
                    "retrain_cron.run_swing does not pass embargo_days=85."
                )
                return
        pytest.fail("run_swing function not found in retrain_cron.py")
