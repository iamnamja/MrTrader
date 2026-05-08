"""
reports.py — Console print and CSV export for WalkForwardReport.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.walkforward.gates import WalkForwardReport


def _ok(msg):   print(f"  \033[32mOK\033[0m  {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")
def _err(msg):  print(f"  \033[31mFAIL\033[0m  {msg}")
def _header(msg):    print(f"\n{'='*62}\n  {msg}\n{'='*62}")


def print_report(report: "WalkForwardReport") -> None:
    from scripts.walkforward.gates import (
        SHARPE_GATE, MIN_FOLD_SHARPE, N_TRIALS_TESTED,
        MIN_PROFIT_FACTOR, MIN_CALMAR, deflated_sharpe_ratio,
    )
    _header(f"Walk-Forward Report - {report.model_type.upper()} (Tier 3)")
    for f in report.folds:
        print(f.summary_line())
    print()
    detail = report.gate_detail()
    print(f"  Avg Sharpe:      {report.avg_sharpe:+.3f}  (gate: > {SHARPE_GATE})  "
          f"{'OK' if detail['avg_sharpe'][1] else 'FAIL'}")
    print(f"  Min fold Sharpe: {report.min_sharpe:+.3f}  (gate: > {MIN_FOLD_SHARPE})  "
          f"{'OK' if detail['min_sharpe'][1] else 'FAIL'}")
    print(f"  Avg win rate:    {report.avg_win_rate:.1%}")
    print(f"  Total trades:    {report.total_trades}")
    dsr_z, dsr_p = deflated_sharpe_ratio(report.avg_sharpe, N_TRIALS_TESTED, report.total_trades)
    print(f"  DSR (N={N_TRIALS_TESTED} trials): z={dsr_z:+.3f}  p={dsr_p:.3f}  "
          f"(gate: p > 0.95)  {'OK' if dsr_p > 0.95 else 'FAIL'}")
    if report.avg_profit_factor > 0:
        print(f"  Avg profit factor: {report.avg_profit_factor:.3f}  "
              f"(gate: > {MIN_PROFIT_FACTOR})  "
              f"{'OK' if detail['avg_profit_factor'][1] else 'FAIL'}")
    if report.avg_calmar != 0:
        print(f"  Avg Calmar ratio:  {report.avg_calmar:.3f}  "
              f"(gate: > {MIN_CALMAR})  "
              f"{'OK' if detail['avg_calmar'][1] else 'FAIL'}")
    if report.avg_k_ratio != 0:
        print(f"  Avg K-ratio:       {report.avg_k_ratio:.3f}  (directional; > 0 = improving)")
    print()
    if report.gate_passed():
        _ok(f"GATE PASSED - avg Sharpe {report.avg_sharpe:.3f}, DSR p={dsr_p:.3f}, "
            f"PF={report.avg_profit_factor:.2f}, Calmar={report.avg_calmar:.2f}")
    else:
        failed = [k for k, (v, ok) in detail.items() if not ok]
        _err(f"GATE NOT MET - failed: {', '.join(failed)}")
