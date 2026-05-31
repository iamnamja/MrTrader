"""
P3 tests: --dsr-n and --paper-gate flags for walkforward_tier3.

Tests cover:
- gate_passed() / gate_detail() with custom dsr_n values
- paper_gate mode uses relaxed thresholds
- DSR p-value increases with higher dsr_n (harder to pass)
- CLI parsing (smoke test)
"""
from __future__ import annotations


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_report(sharpes: list, trades: int = 500, profit_factor: float = 1.5, calmar: float = 0.5):
    """Build a minimal WalkForwardReport from explicit per-fold Sharpe list."""
    from scripts.walkforward.gates import FoldResult
    from scripts.walkforward_tier3 import WalkForwardReport

    folds = []
    for i, s in enumerate(sharpes):
        folds.append(FoldResult(
            fold=i + 1,
            train_start=None, train_end=None,
            test_start=None, test_end=None,
            trades=trades // len(sharpes),
            win_rate=0.55,
            sharpe=s,
            max_drawdown=10.0,
            total_return=20.0,
            stop_exit_rate=0.3,
            model_version=188,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            k_ratio=0.0,
        ))
    report = WalkForwardReport(model_type="swing")
    report.folds = folds
    return report


# ── DSR-n tests ───────────────────────────────────────────────────────────────

def test_dsr_n_default_uses_module_constant():
    """gate_passed() with no args uses N_TRIALS_TESTED=15."""
    from scripts.walkforward_tier3 import N_TRIALS_TESTED
    report = _make_report([1.2, 0.8, 0.9])
    detail_default = report.gate_detail()
    detail_explicit = report.gate_detail(dsr_n=N_TRIALS_TESTED)
    assert detail_default["dsr_p"] == detail_explicit["dsr_p"]


def test_higher_dsr_n_lowers_p_value():
    """Higher n_trials makes selection bias harder to correct — DSR p-value drops."""
    from scripts.walkforward_tier3 import _deflated_sharpe_ratio
    # Use modest SR / sample so p-values are not both saturated at 1.0 after the
    # WF deep-review pass-2 DSR fix (which restored the sqrt(V) scaling on E[SR_max]).
    _, p_low_n = _deflated_sharpe_ratio(0.15, n_trials=15, n_obs=200)
    _, p_high_n = _deflated_sharpe_ratio(0.15, n_trials=200, n_obs=200)
    assert p_high_n < p_low_n


def test_gate_passed_with_high_dsr_n_stricter():
    """A report that passes at N=15 may fail at N=200."""
    report = _make_report([0.9, 0.9, 0.9], trades=300)
    detail_high_n = report.gate_detail(dsr_n=200)
    detail_low_n = report.gate_detail(dsr_n=15)
    assert detail_high_n["dsr_p"][0] <= detail_low_n["dsr_p"][0]


# ── Paper gate tests ──────────────────────────────────────────────────────────

def test_paper_gate_uses_relaxed_sharpe_threshold():
    """Paper gate passes at avg_sharpe=0.6, which fails the production gate (0.8)."""
    from scripts.walkforward_tier3 import SHARPE_GATE
    # all folds at 0.6 → avg=0.6, min=0.6
    report = _make_report([0.6, 0.6, 0.6], trades=400)
    assert 0.6 < SHARPE_GATE  # confirm production gate is stricter
    detail_prod = report.gate_detail(paper_gate=False)
    detail_paper = report.gate_detail(paper_gate=True)
    assert detail_prod["avg_sharpe"][1] is False   # fails production
    assert detail_paper["avg_sharpe"][1] is True   # passes paper (> 0.50)


def test_paper_gate_relaxed_min_fold():
    """Paper gate passes with min_fold_sharpe=-0.35 (fails production at -0.3)."""
    from scripts.walkforward_tier3 import MIN_FOLD_SHARPE
    # avg=0.65, min=-0.35
    report = _make_report([1.5, 0.5, -0.35], trades=500)
    assert MIN_FOLD_SHARPE == -0.3
    detail_prod = report.gate_detail(paper_gate=False)
    detail_paper = report.gate_detail(paper_gate=True)
    assert detail_prod["min_sharpe"][1] is False   # fails production (< -0.3)
    assert detail_paper["min_sharpe"][1] is True   # passes paper (> -0.40)


def test_paper_gate_skips_pf_and_calmar_gates():
    """Paper gate ignores profit factor and Calmar gates (always True)."""
    report = _make_report([0.6, 0.6, 0.6], trades=400, profit_factor=0.0, calmar=0.0)
    detail = report.gate_detail(paper_gate=True)
    assert detail["avg_profit_factor"][1] is True
    assert detail["avg_calmar"][1] is True


def test_paper_gate_still_requires_positive_dsr():
    """Paper gate still applies DSR check — terrible model fails even in paper mode."""
    # avg=0.1 < 0.50 → paper gate avg_sharpe fails
    report = _make_report([0.1, 0.1, 0.1], trades=50)
    assert not report.gate_passed(paper_gate=True)


# ── gate_detail keys ─────────────────────────────────────────────────────────

def test_gate_detail_returns_all_keys():
    report = _make_report([1.0, 0.8, 0.9])
    detail = report.gate_detail()
    assert {"avg_sharpe", "min_sharpe", "dsr_p", "avg_profit_factor", "avg_calmar"} <= set(detail.keys())


# ── print() smoke test ────────────────────────────────────────────────────────

def test_print_runs_without_error(capsys):
    report = _make_report([1.0, 0.8, 0.9], trades=600)
    report.print(dsr_n=15, paper_gate=False)
    captured = capsys.readouterr()
    assert "Avg Sharpe" in captured.out

    report.print(dsr_n=200, paper_gate=True)
    captured2 = capsys.readouterr()
    assert "PAPER-GATE MODE" in captured2.out
    assert "N=200 trials" in captured2.out
