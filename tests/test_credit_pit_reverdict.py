"""Alpha-v10 P0.4 — credit-overlay PIT re-verdict logic (scripts/run_credit_pit_reverdict.py).

Tests the pure verdict function (no backtest): does the credit overlay's marginal benefit
SURVIVE the PIT-vol check (refuting the panel's 'shrinks' hypothesis), or shrink to noise?
"""
from __future__ import annotations

from types import SimpleNamespace

from scripts.run_credit_pit_reverdict import build_verdict_from_report, SHRUNK_DSHARPE


def _report(d_sharpe, d_calmar=0.03, d_max_dd=0.001, crisis=None):
    return SimpleNamespace(d_sharpe=d_sharpe, d_calmar=d_calmar, d_max_dd=d_max_dd,
                           crisis=crisis or {})


_THREE_CRISES_HELPED = {
    "GFC_2008": {"dd_improve": 0.019},
    "COVID_2020": {"dd_improve": 0.022},
    "BEAR_2022": {"dd_improve": 0.006},
}


def test_actual_p0_4_numbers_are_pit_robust():
    # the values measured 2026-06-22 on current data
    v = build_verdict_from_report(_report(0.0639, 0.0298, 0.0010, _THREE_CRISES_HELPED),
                                  h1_dmaxdd=0.0224, h2_dmaxdd=0.0010)
    assert v.pit_robust is True
    assert v.verdict == "PIT_ROBUST_CANDIDATE"
    assert v.crises_helped == 3 and v.crises_total == 3
    assert v.both_halves_tail_positive is True


def test_shrunk_dsharpe_kills():
    # if the marginal dSharpe had collapsed (like carry's did), the verdict must flip to KILL
    v = build_verdict_from_report(_report(0.005, 0.001, 0.0001, _THREE_CRISES_HELPED),
                                  h1_dmaxdd=0.0, h2_dmaxdd=0.0)
    assert v.pit_robust is False
    assert v.verdict == "SHRUNK_KILL"


def test_negative_tail_not_robust_even_if_dsharpe_clears():
    # a Sharpe gain that comes WITH a worse tail (dMaxDD < 0) is not a tail-insurance win
    v = build_verdict_from_report(_report(0.10, 0.05, -0.01, _THREE_CRISES_HELPED),
                                  h1_dmaxdd=0.01, h2_dmaxdd=-0.01)
    assert v.pit_robust is False


def test_shrunk_threshold_boundary():
    assert build_verdict_from_report(_report(SHRUNK_DSHARPE)).pit_robust is True
    assert build_verdict_from_report(_report(SHRUNK_DSHARPE - 1e-6)).pit_robust is False


def test_crises_helped_counts_only_positive():
    crisis = {"A": {"dd_improve": 0.01}, "B": {"dd_improve": -0.01}, "C": {"dd_improve": 0.0}}
    v = build_verdict_from_report(_report(0.0639, crisis=crisis), h1_dmaxdd=0.02, h2_dmaxdd=0.01)
    assert v.crises_helped == 1 and v.crises_total == 3
