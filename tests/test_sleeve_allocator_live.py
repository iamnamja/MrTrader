"""Alpha-v4 P3 — live regime-aware sleeve allocator.

Pure math/mapping is unit-tested directly; the effective-weight readers and run_allocator
are tested with a dict-backed agent_config fake + monkeypatched trackers/regime, so no DB
or network is touched. The headline guarantee — DISABLED == today's static behavior — is
asserted explicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

import app.live_trading.sleeve_allocator_live as L
from app.strategy.sleeve_allocator import AllocatorConfig


def _df(pead_vol=0.01, trend_vol=0.02, n=80, seed=0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"pead": rng.normal(0, pead_vol, n), "trend": rng.normal(0, trend_vol, n)},
        index=idx,
    )


# ── pure: compute_sleeve_weights ──────────────────────────────────────────────

def test_equal_scheme_is_half_half():
    w = L.compute_sleeve_weights(_df(), "equal", "NEUTRAL", AllocatorConfig())
    assert w == {"pead": 0.5, "trend": 0.5}


def test_vol_scheme_favors_low_vol_sleeve():
    w = L.compute_sleeve_weights(_df(pead_vol=0.01, trend_vol=0.03), "vol", "NEUTRAL", AllocatorConfig())
    assert w["pead"] > w["trend"]                      # lower-vol pead heavier
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_regime_bull_upweights_pead_bear_upweights_trend():
    cfg = AllocatorConfig()
    bull = L.compute_sleeve_weights(_df(), "regime", "BULL", cfg)
    bear = L.compute_sleeve_weights(_df(), "regime", "BEAR", cfg)
    assert bull["pead"] > bear["pead"]                 # PEAD heavier in BULL than BEAR
    assert bear["trend"] > bull["trend"]               # trend heavier in BEAR
    assert abs(sum(bull.values()) - 1.0) < 1e-9 and abs(sum(bear.values()) - 1.0) < 1e-9


def test_degenerate_inputs_fall_back_to_equal():
    assert L.compute_sleeve_weights(None, "vol", "NEUTRAL", AllocatorConfig()) == {}
    empty = pd.DataFrame(columns=["pead", "trend"])
    assert L.compute_sleeve_weights(empty, "vol", "NEUTRAL", AllocatorConfig()) == {"pead": 0.5, "trend": 0.5}


# ── pure: regime map + pead size-mult map ─────────────────────────────────────

def test_regime_label_map():
    assert L.map_regime_label("RISK_ON") == "BULL"
    assert L.map_regime_label("RISK_CAUTION") == "NEUTRAL"
    assert L.map_regime_label("RISK_OFF") == "BEAR"
    assert L.map_regime_label("anything-else") == "NEUTRAL"
    assert L.map_regime_label(None) == "NEUTRAL"


def test_pead_size_mult_map_and_clamp():
    assert L.map_pead_size_mult(1.0, 0.6) == pytest.approx(1.2)   # 1.0 * 0.6/0.5
    assert L.map_pead_size_mult(1.0, 0.3) == 1.0                  # 0.6 -> clamp to floor 1.0
    assert L.map_pead_size_mult(3.0, 0.9) == pytest.approx(5.4)   # 3.0 * 0.9/0.5
    assert L.map_pead_size_mult(1.0, 100.0) == 10.0              # clamp to ceiling


# ── effective readers (dict-backed config fake) ───────────────────────────────

class _Cfg:
    """Dict-backed stand-in for get_agent_config/set_agent_config."""
    def __init__(self, **vals):
        base = {
            "pm.trend_allocation_pct": 0.40, "pm.pead_size_mult": 1.0,
            "pm.allocator_enabled": "false", "pm.allocator_scheme": "equal",
            "pm.allocator_total_budget_pct": 0.80, "pm.allocator_stale_days": 8,
            "pm.allocator_trend_weight": 0.40, "pm.allocator_pead_weight": 0.40,
            "pm.allocator_last_computed": "", "pm.allocator_vol_lookback": 60,
            "pm.allocator_min_deployed_days": 20,
        }
        base.update(vals)
        self.d = base

    def get(self, db, key):
        return self.d.get(key)

    def set(self, db, key, value):
        self.d[key] = value


@pytest.fixture
def patch_cfg(monkeypatch):
    def _apply(cfg: _Cfg):
        monkeypatch.setattr("app.database.agent_config.get_agent_config", cfg.get)
        monkeypatch.setattr("app.database.agent_config.set_agent_config", cfg.set)
        return cfg
    return _apply


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def test_effective_trend_allocation_disabled_is_static(patch_cfg):
    patch_cfg(_Cfg(allocator_enabled="false"))  # default disabled
    assert L.effective_trend_allocation(object()) == 0.40   # == pm.trend_allocation_pct


def test_effective_trend_allocation_stale_is_static(patch_cfg):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    patch_cfg(_Cfg(**{"pm.allocator_enabled": "true", "pm.allocator_trend_weight": 0.7,
                      "pm.allocator_last_computed": old}))
    assert L.effective_trend_allocation(object()) == 0.40   # stale -> static fallback


def test_effective_trend_allocation_enabled_fresh_uses_allocator(patch_cfg):
    patch_cfg(_Cfg(**{"pm.allocator_enabled": "true", "pm.allocator_trend_weight": 0.7,
                      "pm.allocator_total_budget_pct": 0.80, "pm.allocator_last_computed": _now_iso()}))
    assert L.effective_trend_allocation(object()) == pytest.approx(0.56)  # 0.7 * 0.80


def test_effective_pead_size_mult_disabled_is_base(patch_cfg):
    patch_cfg(_Cfg(allocator_enabled="false"))
    assert L.effective_pead_size_mult(object(), 1.0) == 1.0


def test_effective_pead_size_mult_enabled_fresh_maps(patch_cfg):
    patch_cfg(_Cfg(**{"pm.allocator_enabled": "true", "pm.allocator_scheme": "vol",
                      "pm.allocator_pead_weight": 0.6, "pm.allocator_last_computed": _now_iso()}))
    assert L.effective_pead_size_mult(object(), 1.0) == pytest.approx(1.2)


def test_effective_pead_size_mult_regime_scheme_is_noop_no_double_tilt(patch_cfg):
    # Critical guard: under scheme=regime the PM already tilts PEAD by regime per-name,
    # so the allocator must NOT also scale PEAD's size_mult (would double-count regime).
    patch_cfg(_Cfg(**{"pm.allocator_enabled": "true", "pm.allocator_scheme": "regime",
                      "pm.allocator_pead_weight": 0.9, "pm.allocator_last_computed": _now_iso()}))
    assert L.effective_pead_size_mult(object(), 1.0) == 1.0   # base unchanged, no tilt


# ── _load_sleeve_returns warmup ───────────────────────────────────────────────

def _rows(n, pnl=10.0, gross=1000.0, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return [{"trade_date": d.date().isoformat(), "daily_pnl": pnl, "gross_deployed": gross}
            for d in idx]


def test_load_returns_warmup_returns_none(monkeypatch):
    monkeypatch.setattr("app.live_trading.trend_tracker.read_daily", lambda since=None: _rows(5))
    monkeypatch.setattr("app.live_trading.pead_tracker.read_daily", lambda since=None: _rows(40))
    assert L._load_sleeve_returns(min_days=20) is None  # trend has only 5 deployed days


def test_load_returns_builds_frame(monkeypatch):
    monkeypatch.setattr("app.live_trading.trend_tracker.read_daily", lambda since=None: _rows(40, pnl=20, gross=2000))
    monkeypatch.setattr("app.live_trading.pead_tracker.read_daily", lambda since=None: _rows(40, pnl=10, gross=1000))
    df = L._load_sleeve_returns(min_days=20)
    assert df is not None and list(df.columns) == ["pead", "trend"]
    assert df["pead"].dropna().iloc[0] == pytest.approx(0.01)   # 10/1000
    assert df["trend"].dropna().iloc[0] == pytest.approx(0.01)  # 20/2000


# ── run_allocator ─────────────────────────────────────────────────────────────

def test_run_allocator_disabled_persists_nothing(patch_cfg, monkeypatch):
    cfg = patch_cfg(_Cfg(allocator_enabled="false"))
    sets = []
    monkeypatch.setattr("app.database.agent_config.set_agent_config",
                        lambda db, k, v: sets.append((k, v)))
    out = L.run_allocator(db=object())
    assert out["status"] == "disabled"
    assert sets == []   # nothing written


def test_run_allocator_enabled_persists_weights_and_records(patch_cfg, monkeypatch):
    cfg = patch_cfg(_Cfg(**{"pm.allocator_enabled": "true", "pm.allocator_scheme": "equal"}))
    sets = {}
    monkeypatch.setattr("app.database.agent_config.set_agent_config",
                        lambda db, k, v: sets.__setitem__(k, v))
    monkeypatch.setattr(L, "_load_sleeve_returns", lambda min_days: _df())
    monkeypatch.setattr(L, "_live_regime_label", lambda: "NEUTRAL")
    recorded = {}
    monkeypatch.setattr("app.live_trading.allocator_tracker.record",
                        lambda **kw: recorded.update(kw) or True)
    out = L.run_allocator(db=object(), force=True)
    assert out["status"] == "ok" and out["source"] == "allocator"
    assert sets["pm.allocator_trend_weight"] == 0.5 and sets["pm.allocator_pead_weight"] == 0.5
    assert "pm.allocator_last_computed" in sets
    assert recorded["source"] == "allocator" and recorded["scheme"] == "equal"


def test_run_allocator_warmup_persists_no_weights(patch_cfg, monkeypatch):
    cfg = patch_cfg(_Cfg(**{"pm.allocator_enabled": "true"}))
    sets = {}
    monkeypatch.setattr("app.database.agent_config.set_agent_config",
                        lambda db, k, v: sets.__setitem__(k, v))
    monkeypatch.setattr(L, "_load_sleeve_returns", lambda min_days: None)  # warmup
    monkeypatch.setattr(L, "_live_regime_label", lambda: "NEUTRAL")
    monkeypatch.setattr("app.live_trading.allocator_tracker.record", lambda **kw: True)
    out = L.run_allocator(db=object(), force=True)
    assert out["status"] == "warmup"
    assert "pm.allocator_trend_weight" not in sets  # no weights persisted in warmup


def test_allocator_config_keys_registered():
    from app.database.agent_config import _DEFAULTS
    assert _DEFAULTS["pm.allocator_enabled"] == "false"   # ships disabled
    assert _DEFAULTS["pm.allocator_scheme"] == "equal"
    assert _DEFAULTS["pm.allocator_total_budget_pct"] == 0.80
