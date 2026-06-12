"""
Tests for the Ruler-v2 gate (Alpha-v7 Phase B, Phase 2) — app/research/bayes_sr.py
+ app/research/ruler_v2.py + the GATE_MODE="ruler_v2" dispatch in CPCVResult.

Covers: the Bayesian posterior (uninformative→0.5, strong→>0.95, trial-shrinkage,
live-paper fold-in, degenerate prior); the tier inversion (PAPER plausibility passes
without a significance t; CAPITAL needs posterior+residual-α+bootstrap+power floor);
fail-closed on an empty OOS series; the implausibility ceiling; the event-sparsity
regime waiver; PBO gating only when M>1; and BACKWARD-COMPAT — the dark branch is
inert at the default GATE_MODE (R5: zero diffs).
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.research import bayes_sr
from app.research import ruler_v2
from scripts.walkforward.cpcv import CPCVResult


# ───────────────────────────── helpers ──────────────────────────────────────

def _dated(returns: np.ndarray) -> list:
    idx = pd.bdate_range("2018-01-02", periods=len(returns))
    return [(d.strftime("%Y-%m-%d"), float(r)) for d, r in zip(idx, returns)]


def _result(returns, *, n_folds=12, n_paths=20, worst_regime=0.10,
            residual_t=3.0, regime_insufficient=False, true_wf=True) -> CPCVResult:
    res = CPCVResult(model_type="test", n_folds=n_folds, n_paths=n_paths)
    res.path_sharpes = [1.0] * n_paths
    res.is_true_walkforward = true_wf
    res.worst_regime_sharpe = worst_regime
    res.regime_insufficient_obs = regime_insufficient
    res.residual_alpha_t_hac = residual_t
    res.oos_returns_dated = _dated(np.asarray(returns, dtype=float))
    return res


def _strong(n=560, seed=0):
    rng = np.random.default_rng(seed)
    return 0.0009 + rng.normal(0, 0.010, n)     # ann SR ~1.4: plausible, significant


# A concordant live-paper observation + the registry's TRUE (small) trial count.
# CAPITAL is DELIBERATELY unreachable on backtest alone — the posterior is P(SR>0 |
# backtest AND live paper); these are what a genuine capital candidate carries.
_LIVE_PAPER = {"sr": 1.2, "se": 0.30}
_REAL_TRIALS = 12


# ───────────────────────────── bayes_sr ─────────────────────────────────────

def test_posterior_half_when_backtest_uninformative():
    # se=inf (HAC t≈0) ⇒ backtest contributes no precision ⇒ posterior = prior(0) ⇒ 0.5
    res = bayes_sr.posterior_sr(0.0, float("inf"), n_trials=10, prior_sd=0.30)
    assert res.gating is False
    assert res.p_sr_gt_0 == pytest.approx(0.5, abs=1e-9)


def test_posterior_high_for_strong_low_se_backtest():
    res = bayes_sr.posterior_sr(1.2, 0.30, n_trials=10, prior_sd=0.30)
    assert res.gating is True
    assert res.p_sr_gt_0 > 0.95
    assert 0.0 < res.posterior_mean < 1.2          # shrunk toward the prior mean


def test_posterior_trial_count_shrinks_toward_half():
    few = bayes_sr.posterior_sr(0.8, 0.40, n_trials=2, prior_sd=0.30).p_sr_gt_0
    many = bayes_sr.posterior_sr(0.8, 0.40, n_trials=5000, prior_sd=0.30).p_sr_gt_0
    assert few > many                              # more trials → tighter prior → lower P
    assert many < few


def test_posterior_live_paper_folds_in():
    bt = bayes_sr.posterior_sr(0.5, 0.40, n_trials=20, prior_sd=0.30)
    both = bayes_sr.posterior_sr(0.5, 0.40, n_trials=20, prior_sd=0.30,
                                 sr_live=0.6, se_live=0.30)
    assert both.used_live is True and bt.used_live is False
    assert both.p_sr_gt_0 > bt.p_sr_gt_0           # concordant live evidence raises P
    assert both.posterior_sd < bt.posterior_sd     # more precision → tighter


def test_posterior_degenerate_prior_fails_closed():
    res = bayes_sr.posterior_sr(1.0, 0.10, n_trials=10, prior_sd=0.0)
    assert res.gating is False and res.p_sr_gt_0 == pytest.approx(0.5)


# ───────────────────────── ruler_v2 — tiers ─────────────────────────────────

def test_capital_passes_with_live_paper_confirmation():
    # A long, strong, significant backtest + concordant live-paper + the registry's
    # true (small) trial count → the posterior clears 0.95 and CAPITAL passes.
    res = _result(_strong(n=1500))
    assert ruler_v2.gate_passed(res, tier="paper") is True
    assert ruler_v2.gate_passed(res, tier="capital",
                                n_trials=_REAL_TRIALS, live_paper=_LIVE_PAPER) is True
    d = ruler_v2.evaluate(res, tier="capital",
                          n_trials=_REAL_TRIALS, live_paper=_LIVE_PAPER)
    assert d["live_paper_present"][1] is True
    assert d["posterior_p_sr_gt_0"][1] is True
    assert d["bootstrap_p_sr_gt_0"][1] is True
    assert d["power_floor"][1] is True


def test_capital_structurally_unreachable_on_backtest_alone():
    # The core safety property, enforced STRUCTURALLY (not by threshold luck): with
    # NO live-paper observation, CAPITAL fails on the `live_paper_present` criterion
    # for EVERY draw — the posterior is P(SR>0 | backtest AND live paper). This is the
    # deliberate fix for "promoted to capital on a backtest alone".
    for seed in range(25):
        res = _result(_strong(n=1500, seed=seed))
        d = ruler_v2.evaluate(res, tier="capital", n_trials=_REAL_TRIALS)  # no live
        assert d["live_paper_present"] == (False, False)
        assert ruler_v2.gate_passed(res, tier="capital", n_trials=_REAL_TRIALS) is False


def test_live_paper_is_load_bearing_even_when_posterior_would_clear():
    # seed=3's backtest-alone posterior clears 0.95 on its own — proving the
    # live-paper requirement is STRUCTURAL: capital still fails without live paper
    # (gated by live_paper_present), and passes once it is supplied.
    res = _result(_strong(n=1500, seed=3))
    bt_only = ruler_v2.evaluate(res, tier="capital", n_trials=_REAL_TRIALS)
    assert bt_only["posterior_p_sr_gt_0"][1] is True        # threshold alone WOULD pass
    assert bt_only["live_paper_present"][1] is False         # but the structural gate blocks
    assert ruler_v2.gate_passed(res, tier="capital", n_trials=_REAL_TRIALS) is False
    assert ruler_v2.gate_passed(res, tier="capital",
                                n_trials=_REAL_TRIALS, live_paper=_LIVE_PAPER) is True


def test_paper_passes_without_significance_capital_fails():
    # A short, plausible-but-underpowered series: PAPER (plausibility) passes,
    # CAPITAL fails the power floor (n_obs < RULERV2_MIN_DAILY_OBS=504).
    res = _result(_strong(n=200))
    assert ruler_v2.gate_passed(res, tier="paper") is True
    assert ruler_v2.gate_passed(res, tier="capital") is False
    assert ruler_v2.evaluate(res, tier="capital")["power_floor"][1] is False


def test_empty_oos_series_fails_closed_both_tiers():
    res = _result(_strong())
    res.oos_returns_dated = []                     # legacy result / not populated
    assert ruler_v2.gate_passed(res, tier="paper") is False
    assert ruler_v2.gate_passed(res, tier="capital") is False
    assert ruler_v2.evaluate(res, tier="paper")["point_sr_floor"][1] is False


def test_implausibility_ceiling_rejects_absurd_sr():
    rng = np.random.default_rng(1)
    absurd = 0.010 + rng.normal(0, 0.0008, 560)    # ann SR ~ 150 — overfit signature
    res = _result(absurd)
    assert ruler_v2.gate_passed(res, tier="paper") is False
    assert ruler_v2.evaluate(res, tier="paper")["implausibility_ceiling"][1] is False


def test_catastrophic_regime_fails_paper():
    res = _result(_strong(), worst_regime=-1.5)    # below RULERV2_CATASTROPHIC_REGIME_SR
    assert ruler_v2.gate_passed(res, tier="paper") is False


def test_event_sparsity_regime_waiver_passes_paper_with_flag():
    res = _result(_strong(), worst_regime=None, regime_insufficient=True)
    assert ruler_v2.gate_passed(res, tier="paper") is True
    d = ruler_v2.evaluate(res, tier="paper")
    assert "requires_human_review" in d
    assert res.requires_human_review_flag is True


def test_capital_regime_none_databug_fails_closed():
    # worst_regime None WITHOUT event-sparsity = data bug → fail closed (no waiver).
    res = _result(_strong(), worst_regime=None, regime_insufficient=False)
    assert ruler_v2.gate_passed(res, tier="capital",
                                regime_waiver_approved=True) is False


def test_capital_needs_residual_alpha():
    res = _result(_strong(), residual_t=None)      # diagnostic not computed
    assert ruler_v2.gate_passed(res, tier="capital") is False
    res2 = _result(_strong(), residual_t=1.0)      # below RULERV2_RESIDUAL_ALPHA_MIN_T
    assert ruler_v2.gate_passed(res2, tier="capital") is False


def test_in_sample_and_non_truewf_cannot_promote():
    res = _result(_strong())
    res.in_sample_override = True
    assert ruler_v2.gate_passed(res, tier="paper") is False


def test_pbo_gates_only_when_multi_config():
    res = _result(_strong(n=1500))
    kw = dict(n_trials=_REAL_TRIALS, live_paper=_LIVE_PAPER)
    # No matrix → PBO is non-gating (report-only); an otherwise-strong sleeve passes.
    assert ruler_v2.gate_passed(res, tier="capital", **kw) is True
    # An overfit per-block matrix (M>1) flips the PBO criterion → CAPITAL fails.
    overfit = np.eye(12) * 10.0
    d = ruler_v2.evaluate(res, tier="capital", pbo_perf=overfit, **kw)
    assert "pbo" in d and d["pbo"][1] is False
    assert ruler_v2.gate_passed(res, tier="capital", pbo_perf=overfit, **kw) is False


def test_concurrent_paper_sleeve_cap():
    res = _result(_strong())
    assert ruler_v2.gate_passed(res, tier="paper", concurrent_paper_sleeves=99) is False


# ───────────────────────── backward-compat (R5) ─────────────────────────────

def test_dark_branch_inert_unless_flag_set(monkeypatch):
    # R5 backward-compat: the ruler_v2 branch must NOT activate under the legacy
    # modes — assert its signature keys are absent for BOTH non-dark modes (robust
    # to whatever ambient GATE_MODE a sibling test left in the xdist worker).
    import app.ml.retrain_config as rc
    res = _result(_strong())
    for mode in ("significance", "mean_sharpe"):
        monkeypatch.setattr(rc, "GATE_MODE", mode)
        d = res.gate_detail(tier="paper")
        assert "point_sr_floor" not in d and "posterior_p_sr_gt_0" not in d


def test_run_cpcv_populates_oos_returns_dated_sorted_deduped():
    # M2 integration: the REAL producer (run_cpcv) must assemble oos_returns_dated
    # sorted-by-date and deduped-keep-first, and _oos_return_array must round-trip it.
    from scripts.walkforward.cpcv import run_cpcv

    base = date(2024, 9, 30)
    all_days = [base - timedelta(days=i) for i in range(300, 0, -1)]
    all_days += [base + timedelta(days=i) for i in range(1, 600)]
    strategy = SimpleNamespace(
        model=SimpleNamespace(trained_through=base), model_type="intraday",
        version=63, all_days_sorted=all_days, allow_in_sample=False)
    boundaries = [
        (date(2022, 1, 1), date(2024, 10, 15), date(2024, 10, 15), date(2025, 1, 15)),
        (date(2022, 1, 1), date(2025, 1, 15), date(2025, 1, 15), date(2025, 4, 15)),
        (date(2022, 1, 1), date(2025, 4, 15), date(2025, 4, 15), date(2025, 7, 15)),
        (date(2022, 1, 1), date(2025, 7, 15), date(2025, 7, 15), date(2025, 10, 15)),
        (date(2022, 1, 1), date(2025, 10, 15), date(2025, 10, 15), date(2026, 1, 15)),
        (date(2022, 1, 1), date(2026, 1, 15), date(2026, 1, 15), date(2026, 4, 15)),
    ]
    # Deliberately UNSORTED dated returns, identical for every fold → exercises both
    # the sort and the keep-first dedup (6 folds → only the 3 unique dates survive).
    fold_result = SimpleNamespace(
        sharpe=2.5, profit_factor=2.0, calmar_ratio=1.0, n_obs=80,
        daily_returns_dated=[("2025-03-01", 0.010),
                             ("2025-01-01", 0.020),
                             ("2025-02-01", -0.010)])
    strategy.run_fold = MagicMock(return_value=fold_result)

    with patch("scripts.walkforward.engine.FoldEngine") as mock_engine_cls, \
         patch("app.ml.retrain_config.assert_no_sacred_holdout"):
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine._build_trading_day_folds.return_value = boundaries
        result = run_cpcv(strategy, purge_days=5, embargo_days=5, n_folds=6,
                          n_paths=2, allow_sacred_holdout=True)

    assert result.oos_returns_dated == [("2025-01-01", 0.020),
                                        ("2025-02-01", -0.010),
                                        ("2025-03-01", 0.010)]
    arr = ruler_v2._oos_return_array(result)
    assert np.allclose(arr, [0.020, -0.010, 0.010])


def test_dispatch_routes_to_ruler_v2_when_flag_set(monkeypatch):
    import app.ml.retrain_config as rc
    monkeypatch.setattr(rc, "GATE_MODE", "ruler_v2")
    res = _result(_strong(n=1500))
    # The dispatch returns the ruler_v2 detail shape, not the significance shape.
    d = res.gate_detail(tier="capital")
    assert "point_sr_floor" in d and "posterior_p_sr_gt_0" in d
    assert "tstat" not in d
    # PAPER (plausibility) is reachable through the dispatch (no live-paper needed).
    assert res.gate_passed(tier="paper") is True
    # CAPITAL through the bare dispatch is backtest-only (no live paper, default
    # trials) → fails closed, as designed.
    assert res.gate_passed(tier="capital") is False
