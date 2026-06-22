"""Phase A — combined-book multi-strategy evaluator (app/research/multistrat_eval.py).

Pins the holistic-eval mechanics: book assembly (common + fold-in union), the PIT drawdown
governor, per-sleeve leave-one-out Track-B attribution, and the combined-book CPCV wiring.
CPCV tests use a tiny geometry + a trivial regime map so they stay fast and offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.research.multistrat_eval as mse


def _series(n=800, mu=0.0004, sd=0.01, seed=0, start="2018-01-01"):
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=idx)


# ── assembly ──────────────────────────────────────────────────────────────────
def test_assemble_common_book_inner_joins():
    a = _series(seed=1)
    b = _series(seed=2, start="2018-06-01")          # starts later
    book = mse.assemble_common_book({"a": a, "b": b}, scheme="vol")
    assert book.returns.index.min() >= b.index.min()  # inner-join → starts at the later sleeve
    assert set(book.weights.columns) == {"a", "b"}


def test_assemble_common_book_empty_overlap_raises():
    a = _series(n=100, start="2018-01-01")
    b = _series(n=100, start="2025-01-01")            # disjoint windows
    with pytest.raises(ValueError):
        mse.assemble_common_book({"a": a, "b": b})


# ── PIT drawdown governor ─────────────────────────────────────────────────────
def test_apply_drawdown_ladder_degrosses_in_drawdown_and_is_pit():
    # a continuous decline (returns stay NONZERO as the drawdown deepens) so the de-gross is
    # measurable on the same days; the governor must cut gross AFTER the drawdown (PIT), not day 0.
    idx = pd.bdate_range("2020-01-01", periods=15)
    r = pd.Series([-0.05] * 15, index=idx)          # steady decline → drawdown ramps past the rungs
    governed = mse.apply_drawdown_ladder(r)
    assert len(governed) == len(r)
    # day 0 multiplier is 1.0 (no prior drawdown known) → first return unchanged
    assert governed.iloc[0] == pytest.approx(r.iloc[0])
    # once the ladder rungs (-8%/-12%/-16%/-20%) are breached, gross is STRICTLY reduced
    deep = governed.iloc[8:15].abs().sum()
    raw_deep = r.iloc[8:15].abs().sum()
    assert deep < raw_deep                          # strict — the governor actually cut gross


def test_governor_noop_when_no_drawdown():
    r = pd.Series([0.001] * 30, index=pd.bdate_range("2021-01-01", periods=30))  # only up
    governed = mse.apply_drawdown_ladder(r)
    assert np.allclose(governed.values, r.values)     # never in drawdown → multiplier always 1.0


# ── fold-in union book (ragged history) ───────────────────────────────────────
def test_union_book_uses_all_data():
    a = _series(n=800, seed=1, start="2018-01-01")
    b = _series(n=400, seed=2, start="2019-06-03")    # shorter / later
    union = mse.assemble_union_book({"a": a, "b": b})
    common = mse.assemble_common_book({"a": a, "b": b}).returns
    assert len(union) > len(common)                   # fold-in spans more than the inner-join
    assert union.index.min() <= common.index.min()    # starts at the earliest sleeve (after warmup)


def test_union_renormalizes_weights_to_present_sleeves():
    # behavioral proof that absent-sleeve weight redistributes to the present one (sums to 1):
    # in the window where ONLY `a` is present, the union book return must track `a` alone
    # (weight on `a` renormalised to ~1.0, lagged), with no contribution from the absent `b`.
    a = _series(n=600, seed=1, start="2018-01-01")
    b = _series(n=200, seed=2, start="2019-12-02")    # `b` joins late
    union = mse.assemble_union_book({"a": a, "b": b}, cost_bps=0.0)
    solo = mse.assemble_union_book({"a": a}, cost_bps=0.0)   # a-only book = a vol-targeted at w=1
    # on a date well inside the a-only region (post-warmup, pre-b), union == a-only book
    probe = a.index[120]
    assert probe < b.index.min()
    assert union.loc[probe] == pytest.approx(solo.loc[probe], abs=1e-9)


# ── per-sleeve Track-B attribution ────────────────────────────────────────────
def test_per_sleeve_contributions_leave_one_out():
    sleeves = {"a": _series(seed=1), "b": _series(seed=2), "c": _series(seed=3)}
    out = mse.per_sleeve_contributions(sleeves, n_boot=80, seed=0)
    assert {c.label for c in out} == {"a", "b", "c"}
    for c in out:
        assert np.isfinite(c.standalone_sharpe)
        assert c.track_b_passed in (True, False)      # 3 sleeves → each has "others" to hedge vs


def test_per_sleeve_single_sleeve_has_no_trackb():
    out = mse.per_sleeve_contributions({"only": _series()}, n_boot=50)
    assert len(out) == 1 and out[0].track_b_passed is None   # nothing to hedge against


# ── combined-book CPCV (bounded) ──────────────────────────────────────────────
# A short geometry on a ~6y synthetic series — enough purged folds to record pooled OOS obs while
# staying fast/offline (regime_map={} → worst-regime None, fine for the wiring test).
_TINY_CPCV = dict(n_folds=4, n_paths=2, purge_days=5, embargo_days=1, regime_map={})


def test_book_walkforward_returns_metrics():
    book = mse.assemble_common_book(
        {"a": _series(n=1500, seed=1), "b": _series(n=1500, seed=2)}).returns
    wf = mse.book_walkforward(book, n_families=25, **_TINY_CPCV)
    assert wf.n_obs > 0 and wf.n_folds == 4
    assert np.isfinite(wf.mean_sharpe) and 0.0 <= wf.dsr_family_p <= 1.0
    assert isinstance(wf.paper_passed, bool)


# ── end-to-end ────────────────────────────────────────────────────────────────
def test_run_multistrat_eval_end_to_end():
    sleeves = {"a": _series(n=1500, seed=1), "b": _series(n=1500, seed=2)}
    rep = mse.run_multistrat_eval(sleeves, scheme="vol", apply_governor=True,
                                  run_tail=False, n_boot=80, cpcv_kw=_TINY_CPCV)
    assert rep.n_days > 0 and rep.n_families == 25
    assert rep.book_raw is not None and rep.book_governed is not None   # governor ran
    assert len(rep.sleeves) == 2
    assert rep.union is not None and rep.union["n_days"] > 0
    d = rep.to_dict()                                  # serializable summary
    assert d["book_raw"]["mean_sharpe"] == rep.book_raw.mean_sharpe


def test_cpcv_kw_reserved_keys_are_stripped_not_crashed():
    # a caller fat-fingering a reserved key into cpcv_kw must NOT crash (it's geometry-only)
    sleeves = {"a": _series(n=1500, seed=1), "b": _series(n=1500, seed=2)}
    bad_kw = dict(_TINY_CPCV, n_families=999, label="oops")   # reserved keys present
    rep = mse.run_multistrat_eval(sleeves, scheme="vol", apply_governor=False,
                                  run_tail=False, n_boot=50, cpcv_kw=bad_kw)
    assert rep.n_families == 25                          # the registry count, not the injected 999
