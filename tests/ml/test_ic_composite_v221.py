"""
Tests for Phase 91 v221 fundamentals-downweighted scorer.

Key properties:
- Stateless (no regime tracking, no mutable state across calls)
- V221 weights = V219 with profit_margin/operating_margin/pe_ratio * 0.30, renormalized
- Fundamentals load must not crash even if module absent (fail-open)
- Returns valid (sym, score) tuples for eligible symbols
"""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from app.ml.factor_scorer import IcCompositeV221Scorer, V221_IC_WEIGHTS, V219_IC_WEIGHTS


def _make_symbols_data(n_days: int = 400, n_syms: int = 80, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    syms = [f"S{i:03d}" for i in range(n_syms)]
    log_rets = rng.normal(0.0003, 0.01, size=(n_days, n_syms))
    prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)
    return {sym: pd.DataFrame({"close": prices[sym]}) for sym in syms}


class TestV221Weights:
    def test_weights_sum_to_one(self):
        assert abs(sum(V221_IC_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_positive(self):
        for feat, w in V221_IC_WEIGHTS.items():
            assert w > 0, f"Weight for {feat} is non-positive"

    def test_fundamentals_downweighted_vs_v219(self):
        """Quality features should be proportionally lower in v221 than v219."""
        for feat in ("profit_margin", "operating_margin", "pe_ratio"):
            assert V221_IC_WEIGHTS[feat] < V219_IC_WEIGHTS[feat], (
                f"{feat}: v221 weight {V221_IC_WEIGHTS[feat]:.4f} should be < "
                f"v219 weight {V219_IC_WEIGHTS[feat]:.4f}"
            )

    def test_momentum_relatively_higher_vs_v219(self):
        """After renormalization, momentum features should be proportionally higher in v221."""
        v221_mom = V221_IC_WEIGHTS.get("ix_momentum_vol", 0) + V221_IC_WEIGHTS.get("momentum_252d_ex1m", 0)
        v219_mom = V219_IC_WEIGHTS.get("ix_momentum_vol", 0) + V219_IC_WEIGHTS.get("momentum_252d_ex1m", 0)
        assert v221_mom > v219_mom, "v221 should have higher relative momentum weight than v219"

    def test_fundamentals_approx_30pct_of_v219(self):
        """Each fundamental feature should be ~30% of v219 raw weight (before renorm)."""
        from app.ml.factor_scorer import _V221_IC_WEIGHTS_RAW, _V219_IC_WEIGHTS_RAW
        for feat in ("profit_margin", "operating_margin", "pe_ratio"):
            ratio = _V221_IC_WEIGHTS_RAW[feat] / _V219_IC_WEIGHTS_RAW[feat]
            assert abs(ratio - 0.30) < 0.01, f"{feat} raw ratio {ratio:.3f} should be ~0.30"


class TestIcCompositeV221Scorer:
    def test_instantiation_does_not_crash(self):
        """Scorer must instantiate without any imports or network calls."""
        scorer = IcCompositeV221Scorer()
        assert scorer is not None

    def test_returns_list_of_tuples(self):
        symbols_data = _make_symbols_data()
        scorer = IcCompositeV221Scorer()
        result = scorer(date(2021, 6, 1), symbols_data)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_scores_in_reasonable_range(self):
        symbols_data = _make_symbols_data()
        scorer = IcCompositeV221Scorer()
        result = scorer(date(2021, 6, 1), symbols_data)
        assert len(result) > 10
        scores = [s for _, s in result]
        assert max(abs(s) for s in scores) < 20.0

    def test_sorted_descending(self):
        """Results must be sorted highest-score first (for top-N selection)."""
        symbols_data = _make_symbols_data()
        scorer = IcCompositeV221Scorer()
        result = scorer(date(2021, 6, 1), symbols_data)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_stateless_across_calls(self):
        """Calling scorer on day A then day B must give same result as day B alone."""
        symbols_data = _make_symbols_data()
        scorer1 = IcCompositeV221Scorer()
        # Call day A first
        scorer1(date(2021, 5, 1), symbols_data)
        result_after_a = scorer1(date(2021, 6, 1), symbols_data)

        scorer2 = IcCompositeV221Scorer()
        result_cold = scorer2(date(2021, 6, 1), symbols_data)

        # Results must be identical (stateless)
        r1 = {sym: s for sym, s in result_after_a}
        r2 = {sym: s for sym, s in result_cold}
        common = set(r1) & set(r2)
        assert len(common) > 0
        max_diff = max(abs(r1[sym] - r2[sym]) for sym in common)
        assert max_diff < 1e-9, f"v221 is not stateless — diff {max_diff:.6f}"

    def test_fundamentals_failure_does_not_crash(self):
        """If fundamentals loading fails, scorer must still return valid scores."""
        symbols_data = _make_symbols_data()
        scorer = IcCompositeV221Scorer()
        # Force fundamentals to fail
        scorer._fmp_loaded = True
        scorer._fmp_fundamentals = None
        result = scorer(date(2021, 6, 1), symbols_data)
        # Should still score on price/vol features
        assert isinstance(result, list)
        assert len(result) > 0

    def test_pit_safe_no_lookahead(self):
        """Corrupting same-day close must not affect scores."""
        n_days, n_syms = 400, 80
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
        syms = [f"S{i:03d}" for i in range(n_syms)]
        log_rets = rng.normal(0.0003, 0.01, size=(n_days, n_syms))
        prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0) + 4.0), index=dates, columns=syms)

        d = dates[350].date()
        symbols_clean = {sym: pd.DataFrame({"close": prices[sym]}) for sym in syms}
        symbols_corrupt = {sym: pd.DataFrame({"close": prices[sym].copy()}) for sym in syms}
        for sym in syms:
            symbols_corrupt[sym].loc[dates[350], "close"] *= 10.0

        scorer1 = IcCompositeV221Scorer()
        scorer1._fmp_loaded = True
        result_clean = {sym: s for sym, s in scorer1(d, symbols_clean)}

        scorer2 = IcCompositeV221Scorer()
        scorer2._fmp_loaded = True
        result_corrupt = {sym: s for sym, s in scorer2(d, symbols_corrupt)}

        common = set(result_clean) & set(result_corrupt)
        assert len(common) > 0
        diff = max(abs(result_clean[sym] - result_corrupt[sym]) for sym in common)
        assert diff < 1e-6, f"Same-day corruption affected scores by {diff:.4f} — lookahead!"

    def test_cli_flag_in_help(self):
        """--rebalance-ic-composite-v221 must appear in CLI help."""
        import sys, subprocess
        from pathlib import Path
        repo_root = Path(__file__).parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True, cwd=str(repo_root)
        )
        assert "rebalance-ic-composite-v221" in result.stdout

    def test_vol_damper_in_help(self):
        """--rebalance-spy-vol-damper must appear in CLI help."""
        import sys, subprocess
        from pathlib import Path
        repo_root = Path(__file__).parents[2]
        result = subprocess.run(
            [sys.executable, "scripts/walkforward_tier3.py", "--help"],
            capture_output=True, text=True, cwd=str(repo_root)
        )
        assert "rebalance-spy-vol-damper" in result.stdout
