"""Alpha-v10 audit Wave 6c — _walk_forward_cv must split folds in TIME order.

The expanding-window WF-CV received a symbol-major-concatenated matrix and split by row index, so a
fold's test rows weren't strictly after its train rows in time (a research-metric look-ahead — note
this `wf_auc` is NOT the live-promotion gate; the separate CPCV pipeline is). Fix: pass a per-row
time key (window_idx) and stable-sort by it before the expanding split; warn loudly when absent so
wf_auc is never trusted as time-OOS.
"""
from __future__ import annotations

import logging

import numpy as np

from app.ml.training import ModelTrainer


def _trainer():
    t = ModelTrainer.__new__(ModelTrainer)
    t.label_scheme = "direction"     # any non-regression scheme -> classification path
    t.n_workers = 1
    return t


def _data(n=360, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4).astype(np.float32)
    y = (rng.rand(n) > 0.5).astype(int)   # balanced 0/1 -> every fold slice has both classes
    return X, y


def test_row_order_sort_is_applied():
    # passing a permuted row_order must give the SAME result as pre-sorting the data by that order
    # (XGBoost is seeded/deterministic) — proving the time-sort is actually applied.
    t = _trainer()
    X, y = _data()
    perm = np.random.RandomState(1).permutation(len(X))   # a scrambled time key
    res_a = t._walk_forward_cv(X, y, ["f0", "f1", "f2", "f3"], n_folds=5, row_order=perm)
    order = np.argsort(perm, kind="stable")
    res_b = t._walk_forward_cv(X[order], y[order], ["f0", "f1", "f2", "f3"],
                               n_folds=5, row_order=None)
    assert res_a.get("wf_auc_mean") == res_b.get("wf_auc_mean")
    assert res_a.get("wf_folds") == res_b.get("wf_folds")


def test_warns_when_no_row_order(caplog):
    t = _trainer()
    X, y = _data()
    with caplog.at_level(logging.WARNING):
        t._walk_forward_cv(X, y, ["f0", "f1", "f2", "f3"], n_folds=5, row_order=None)
    assert any("NOT time-OOS" in r.message for r in caplog.records)


def test_warns_on_length_mismatch_and_falls_back(caplog):
    t = _trainer()
    X, y = _data()
    bad = np.arange(len(X) - 5)     # wrong length -> must not crash, must warn, must fall back
    with caplog.at_level(logging.WARNING):
        res = t._walk_forward_cv(X, y, ["f0", "f1", "f2", "f3"], n_folds=5, row_order=bad)
    assert any("NOT time-OOS" in r.message for r in caplog.records)
    assert "wf_auc_mean" in res or res == {}    # still runs (or skips), never raises
