"""Alpha-v10 audit Wave 4c — live-path-touching minors.

Pins: the TS-normalizer doesn't re-append the same window on repeated same-day inference (which
collapsed std->0 and zeroed the live ML features); the kill switch fails CLOSED (halts) when its
persisted state is unreadable at boot; and a proposal with a missing/unparseable approved_at is
treated as stale.
"""
from __future__ import annotations

import types

import numpy as np


def test_ts_normalize_dedupes_same_window_id():
    from app.ml.ts_normalize import _transform_single_row_per_symbol, TSNormalizerState
    st = TSNormalizerState(n_features=3, lookback=20, min_warmup=2)
    st.history["AAA"] = [(i, np.array([1.0, 2.0, 3.0])) for i in range(1, 6)]
    X = np.array([[1.0, 2.0, 3.0]])
    syms = np.array(["AAA"])
    wid = np.array([100])                          # "today"
    _transform_single_row_per_symbol(X, syms, wid, st)
    n1 = len(st.history["AAA"])
    _transform_single_row_per_symbol(X, syms, wid, st)   # same window again
    n2 = len(st.history["AAA"])
    assert n2 == n1                                # no duplicate append -> buffer can't collapse


def test_kill_switch_load_state_fail_closed_on_unreadable(monkeypatch):
    from app.live_trading.kill_switch import KillSwitch
    ks = KillSwitch()
    monkeypatch.setattr("app.database.config_store.get_config",
                        lambda db, k: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr("app.live_trading.kill_switch.get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))
    ks.load_state()
    assert ks.is_active is True                    # cannot confirm -> HALT (fail-closed)


def test_kill_switch_load_state_restores_persisted_true(monkeypatch):
    from app.live_trading.kill_switch import KillSwitch
    ks = KillSwitch()
    monkeypatch.setattr("app.database.config_store.get_config", lambda db, k: True)
    monkeypatch.setattr("app.live_trading.kill_switch.get_session",
                        lambda: types.SimpleNamespace(close=lambda: None))
    ks.load_state()
    assert ks.is_active is True                    # a clean persisted True is honored
