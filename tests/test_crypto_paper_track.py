"""P3-1 — crypto trend LIVE-PAPER tracker tests (report-only OOS recorder)."""
from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def cpt(monkeypatch, tmp_path):
    mod = importlib.import_module("app.live_trading.crypto_paper_track")
    importlib.reload(mod)
    monkeypatch.setattr(mod, "CRYPTO_PAPER_PATH", str(tmp_path / "crypto_paper_track.parquet"))
    monkeypatch.setattr(mod, "_audit", lambda *a, **k: None)
    return mod


def _cfg(enabled="true"):
    return lambda db, k: {"pm.crypto_paper_enabled": enabled}.get(k)


def _returns(n, *, mu=0.001, sd=0.02, seed=0, end="2026-06-16"):
    idx = pd.date_range(end=end, periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def _patch_returns(monkeypatch, series):
    monkeypatch.setattr("scripts.walkforward.sleeves.crypto_trend_book_returns",
                        lambda *a, **k: series)


# ── gating ──────────────────────────────────────────────────────────────────
def test_dormant_when_disabled(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("false"))
    s = cpt.run_crypto_paper_track(db=object())
    assert s["status"] == "dormant" and s["enabled"] is False


def test_force_runs_even_when_disabled(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("false"))
    _patch_returns(monkeypatch, _returns(400))
    s = cpt.run_crypto_paper_track(db=object(), force=True)
    assert s["status"] == "ok"


# ── inception / OOS slicing ──────────────────────────────────────────────────
def test_first_run_starts_clock_at_latest_date(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("true"))
    r = _returns(500, end="2026-06-16")
    _patch_returns(monkeypatch, r)
    s = cpt.run_crypto_paper_track(db=object())
    # first run: no prior history -> OOS starts NOW (latest date), not back-dated
    assert s["inception"] == "2026-06-16"
    assert s["n_oos_days"] == 1
    assert os.path.exists(cpt.CRYPTO_PAPER_PATH)


def test_inception_preserved_and_oos_grows(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("true"))
    # run 1 ends 2026-06-16 -> inception 2026-06-16, n=1
    _patch_returns(monkeypatch, _returns(500, end="2026-06-16", seed=1))
    s1 = cpt.run_crypto_paper_track(db=object())
    assert s1["inception"] == "2026-06-16" and s1["n_oos_days"] == 1
    # run 2 a week later: same series + 7 new days -> inception unchanged, OOS = 8 days
    _patch_returns(monkeypatch, _returns(507, end="2026-06-23", seed=1))
    s2 = cpt.run_crypto_paper_track(db=object())
    assert s2["inception"] == "2026-06-16"     # NOT re-set to the new latest date
    assert s2["n_oos_days"] == 8               # 2026-06-16 .. 2026-06-23 inclusive


def test_oos_metrics_after_enough_days(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("true"))
    # seed a prior parquet so inception is far back -> full series is OOS
    r = _returns(400, mu=0.002, sd=0.02, end="2026-06-16", seed=3)
    cpt._save(cpt.CRYPTO_PAPER_PATH, r)        # prior OOS history (inception = r.index.min())
    _patch_returns(monkeypatch, r)
    s = cpt.run_crypto_paper_track(db=object())
    assert s["n_oos_days"] == 400
    assert s["oos_sharpe"] is not None and np.isfinite(s["oos_sharpe"])
    # sanity: positive-drift series -> positive Sharpe
    assert s["oos_sharpe"] > 0


# ── failure handling ─────────────────────────────────────────────────────────
def test_failed_when_returns_unavailable(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("true"))
    monkeypatch.setattr("scripts.walkforward.sleeves.crypto_trend_book_returns",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("alpaca down")))
    s = cpt.run_crypto_paper_track(db=object())
    assert s["status"] == "failed" and s["block_reason"] == "returns_unavailable"


def test_failed_on_empty_returns(cpt, monkeypatch):
    monkeypatch.setattr("app.database.agent_config.get_agent_config", _cfg("true"))
    _patch_returns(monkeypatch, pd.Series(dtype=float))
    s = cpt.run_crypto_paper_track(db=object())
    assert s["status"] == "failed" and s["block_reason"] == "no_returns"


# ── metrics helper ───────────────────────────────────────────────────────────
def test_metrics_single_day_has_no_sharpe(cpt):
    m = cpt._metrics(pd.Series([0.01], index=pd.to_datetime(["2026-06-16"])))
    assert m["sharpe"] == 0.0 and m["ann_vol"] == 0.0 and m["n_days"] == 1


def test_metrics_multi_day(cpt):
    m = cpt._metrics(_returns(300, mu=0.001, sd=0.02, seed=5))
    assert m["n_days"] == 300 and np.isfinite(m["sharpe"]) and m["ann_vol"] > 0
