"""
reports.py — Console print and CSV export for WalkForwardReport.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.walkforward.gates import WalkForwardReport


def _ok(msg): print(f"  \033[32mOK\033[0m  {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")
def _err(msg): print(f"  \033[31mFAIL\033[0m  {msg}")
def _header(msg): print(f"\n{'='*62}\n  {msg}\n{'='*62}")


def print_report(
    report: "WalkForwardReport",
    dsr_n: int | None = None,
    paper_gate: bool = False,
) -> None:
    from scripts.walkforward.gates import (
        SHARPE_GATE, MIN_FOLD_SHARPE, N_TRIALS_TESTED,
        MIN_PROFIT_FACTOR, MIN_CALMAR, deflated_sharpe_ratio,
    )
    if dsr_n is None:
        dsr_n = N_TRIALS_TESTED

    sharpe_gate = report.PAPER_SHARPE_GATE if paper_gate else SHARPE_GATE
    min_fold_gate = report.PAPER_MIN_FOLD_SHARPE if paper_gate else MIN_FOLD_SHARPE

    title = f"Walk-Forward Report — {report.model_type.upper()} (Tier 3)"
    if paper_gate:
        title += " [PAPER-GATE MODE]"
    _header(title)
    for f in report.folds:
        print(f.summary_line())
    print()
    detail = report.gate_detail(dsr_n=dsr_n, paper_gate=paper_gate)
    print(f"  Avg Sharpe:      {report.avg_sharpe:+.3f}  (gate: > {sharpe_gate})  "
          f"{'OK' if detail['avg_sharpe'][1] else 'FAIL'}")
    print(f"  Min fold Sharpe: {report.min_sharpe:+.3f}  (gate: > {min_fold_gate})  "
          f"{'OK' if detail['min_sharpe'][1] else 'FAIL'}")
    print(f"  Avg win rate:    {report.avg_win_rate:.1%}")
    print(f"  Total trades:    {report.total_trades}")
    dsr_z, dsr_p = deflated_sharpe_ratio(report.avg_sharpe, dsr_n, report.total_obs)
    print(f"  DSR (N={dsr_n} trials): z={dsr_z:+.3f}  p={dsr_p:.3f}  "
          f"(gate: p > 0.95)  {'OK' if dsr_p > 0.95 else 'FAIL'}")
    if report.avg_profit_factor > 0 and not paper_gate:
        print(f"  Avg profit factor: {report.avg_profit_factor:.3f}  "
              f"(gate: > {MIN_PROFIT_FACTOR})  "
              f"{'OK' if detail['avg_profit_factor'][1] else 'FAIL'}")
    if report.avg_calmar != 0 and not paper_gate:
        print(f"  Avg Calmar ratio:  {report.avg_calmar:.3f}  "
              f"(gate: > {MIN_CALMAR})  "
              f"{'OK' if detail['avg_calmar'][1] else 'FAIL'}")
    if report.avg_k_ratio != 0:
        print(f"  Avg K-ratio:       {report.avg_k_ratio:.3f}  (directional; > 0 = improving)")
    wrs = report.worst_regime_sharpe
    if wrs is not None:
        print(f"  Worst regime Sharpe: {wrs:+.3f}  (gate: > -0.5)  "
              f"{'OK' if detail.get('worst_regime_sharpe', (None, True))[1] else 'FAIL'}")
    print()
    if report.gate_passed(dsr_n=dsr_n, paper_gate=paper_gate):
        mode = "PAPER GATE" if paper_gate else "GATE"
        _ok(f"{mode} PASSED — avg Sharpe {report.avg_sharpe:.3f}, DSR p={dsr_p:.3f}, "
            f"PF={report.avg_profit_factor:.2f}, Calmar={report.avg_calmar:.2f}")
    else:
        failed = [k for k, (v, ok) in detail.items() if not ok]
        _err(f"GATE NOT MET — failed: {', '.join(failed)}")
