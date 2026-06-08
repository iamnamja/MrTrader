"""
Live regime-aware sleeve allocator — Alpha-v4 P3.

Turns the backtest-only sleeve allocator (app/strategy/sleeve_allocator.py) into a live
book-construction layer. Once a week (just before the trend rebalance) it:
  1. assembles trailing per-sleeve daily returns from the live trackers,
  2. reads the live market regime,
  3. computes target sleeve weights via the configured scheme (equal / vol / regime),
  4. persists the effective weights to agent_config.

Both sleeve readers then consult those weights — `effective_trend_allocation` (trend) and
`effective_pead_size_mult` (PEAD) — and FALL BACK to their static config whenever the
allocator is disabled, stale, or still in warmup. So with `pm.allocator_enabled=false`
(the default) the live book is byte-identical to today's fixed budgets.

Strictly additive + kill-switchable. The pure functions (compute_sleeve_weights,
effective_pead_size_mult, the regime map) are unit-tested in isolation; the I/O
orchestrator (run_allocator) never raises into the scheduler.

NOTE (gate discipline): on the current 2-sleeve book the validated gate selects EQUAL
(equal +1.082 > vol +0.715 > regime +0.593), so `pm.allocator_scheme` stays 'equal' until
`scripts/run_book_allocator.py` selects vol/regime (expected after a 3rd sleeve).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Live regime label (RISK_*) -> allocator tilt key (BULL/NEUTRAL/BEAR). Unknown/stale
# -> NEUTRAL (the safe no-tilt key, matching sleeve_allocator._persist_regime's default).
_REGIME_MAP = {
    "RISK_ON": "BULL", "RISK_CAUTION": "NEUTRAL", "RISK_OFF": "BEAR",
    "BULL": "BULL", "NEUTRAL": "NEUTRAL", "BEAR": "BEAR",
}
_EQUAL_BASELINE = 0.5   # 2-sleeve equal weight; the PEAD size-mult mapping pivots on this
_PEAD_MULT_MIN, _PEAD_MULT_MAX = 1.0, 10.0   # pm.pead_size_mult schema bounds
_TREND_ALLOC_MAX = 0.80                       # pm.trend_allocation_pct schema max
_TREND_ALLOC_DEFAULT = 0.40                   # last-resort fallback (matches schema default)


def _truthy(val) -> bool:
    return str(val).strip().lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Pure math / mapping (unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def map_regime_label(live_label) -> str:
    """RISK_ON/RISK_CAUTION/RISK_OFF (or BULL/NEUTRAL/BEAR) -> BULL/NEUTRAL/BEAR.
    Anything unknown/empty -> NEUTRAL (no tilt)."""
    return _REGIME_MAP.get(str(live_label or "").strip().upper(), "NEUTRAL")


def compute_sleeve_weights(returns_df, scheme: str, regime_label: str, cfg) -> dict:
    """Target sleeve weights (summing to 1) for the LATEST row. PURE.

    scheme: 'equal' -> 1/N; 'vol' -> inverse-vol risk parity (latest row); 'regime' ->
    inverse-vol x the current regime's tilt multiplier, renormalized (one-shot live tilt —
    hysteresis/EWMA need a label series; the live regime label is already smoothed and the
    recompute is weekly, so the one-shot tilt is the faithful live analog). Reuses
    app/strategy/sleeve_allocator. Falls back to equal on any degeneracy.
    """
    from app.strategy.sleeve_allocator import vol_weights, DEFAULT_REGIME_TILT

    if returns_df is None or len(returns_df.columns) == 0:
        return {}
    cols = list(returns_df.columns)
    n = len(cols)

    def _equal():
        return {c: 1.0 / n for c in cols}

    if scheme == "equal" or len(returns_df) == 0:
        return _equal()

    try:
        volw = vol_weights(returns_df, cfg).iloc[-1]
        base = {c: float(volw.get(c, 0.0)) for c in cols}
    except Exception:
        return _equal()

    if scheme == "vol":
        w = base
    elif scheme == "regime":
        tilt = (getattr(cfg, "regime_tilt", None) or DEFAULT_REGIME_TILT).get(regime_label, {})
        w = {c: base[c] * float(tilt.get(c, 1.0)) for c in cols}
    else:
        return _equal()

    s = sum(w.values())
    if s <= 0 or any(v != v for v in w.values()):  # non-positive or NaN -> equal
        return _equal()
    return {c: w[c] / s for c in cols}


def map_pead_size_mult(base_mult: float, pead_weight: float) -> float:
    """Map the allocator's PEAD fraction onto PEAD's size-mult dial, clamped to schema
    bounds. PURE. `base_mult * (pead_weight / equal_baseline)`, clamped to [1.0, 10.0].
    NOTE: at the telemetry base of 1.0 the floor pins the result at >= 1.0 (the allocator
    can only grow PEAD). For a base > 1.0 (e.g. a re-ramped 3.0) a sub-equal PEAD weight
    CAN reduce the effective mult down toward the 1.0 floor — i.e. it can down-size PEAD."""
    try:
        eff = float(base_mult) * (float(pead_weight) / _EQUAL_BASELINE)
    except Exception:
        return float(base_mult)
    return max(_PEAD_MULT_MIN, min(eff, _PEAD_MULT_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# I/O: trailing returns, live regime, freshness
# ─────────────────────────────────────────────────────────────────────────────

def _load_sleeve_returns(min_days: int):
    """Trailing per-sleeve daily returns (= daily_pnl/gross_deployed) from the live
    trackers, outer-joined into DataFrame[['pead','trend']]. Returns None when EITHER
    sleeve has < min_days deployed (gross>0) rows — the warmup guard. Never raises."""
    try:
        import pandas as pd
        from app.live_trading import trend_tracker, pead_tracker
    except Exception:
        return None

    def _series(rows):
        out = {}
        for r in rows or []:
            g = r.get("gross_deployed")
            p = r.get("daily_pnl")
            if g and float(g) > 0 and p is not None:
                try:
                    out[pd.Timestamp(r["trade_date"])] = float(p) / float(g)
                except Exception:
                    continue
        return pd.Series(out, dtype=float).sort_index()

    try:
        pe = _series(pead_tracker.read_daily())
        tr = _series(trend_tracker.read_daily())
    except Exception:
        return None
    if len(pe) < min_days or len(tr) < min_days:
        return None
    df = pd.concat([pe.rename("pead"), tr.rename("trend")], axis=1).sort_index()
    return df


def _live_regime_label() -> str:
    """Current mapped regime (BULL/NEUTRAL/BEAR) from the persisted live score."""
    try:
        from app.agents.premarket import premarket_intel
        ctx = premarket_intel.get_regime_context()
        return map_regime_label((ctx or {}).get("regime_label"))
    except Exception:
        return "NEUTRAL"


def _weights_fresh(db, stale_days: int) -> bool:
    from app.database.agent_config import get_agent_config
    ts = get_agent_config(db, "pm.allocator_last_computed")
    if not ts:
        return False
    try:
        t = datetime.fromisoformat(str(ts))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).days <= int(stale_days)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Effective-weight readers (the integration seam — always fall back to static)
# ─────────────────────────────────────────────────────────────────────────────

def effective_trend_allocation(db) -> float:
    """Trend sleeve's effective allocation (fraction of NAV). Allocator weight x budget
    when enabled+fresh; otherwise the static pm.trend_allocation_pct. Never raises —
    falls back to static, then to a hardcoded default if even the static read fails."""
    from app.database.agent_config import get_agent_config
    try:
        static = float(get_agent_config(db, "pm.trend_allocation_pct"))
        if not _truthy(get_agent_config(db, "pm.allocator_enabled")):
            return static
        if not _weights_fresh(db, int(get_agent_config(db, "pm.allocator_stale_days"))):
            return static
        tw = float(get_agent_config(db, "pm.allocator_trend_weight"))
        budget = float(get_agent_config(db, "pm.allocator_total_budget_pct"))
        return max(0.0, min(tw * budget, _TREND_ALLOC_MAX))
    except Exception:
        try:
            return float(get_agent_config(db, "pm.trend_allocation_pct"))
        except Exception:
            return _TREND_ALLOC_DEFAULT


def effective_pead_size_mult(db, base_mult: float) -> float:
    """PEAD's effective size multiplier. Allocator PEAD weight mapped onto the size dial
    when enabled+fresh; otherwise base_mult unchanged. Never raises.

    DOUBLE-TILT GUARD: the PM already applies a per-name regime sizing multiplier to PEAD
    (`_regime_sizing_multiplier`). Under `allocator_scheme='regime'` the allocator's regime
    tilt would compound that same regime bet on PEAD, so the allocator does NOT drive
    PEAD's size_mult in regime mode — PEAD's own per-name mult governs its regime response,
    while the allocator's regime tilt still flows to the TREND budget via
    effective_trend_allocation. 'equal'/'vol' have no regime component, so they map normally.
    """
    from app.database.agent_config import get_agent_config
    try:
        if not _truthy(get_agent_config(db, "pm.allocator_enabled")):
            return float(base_mult)
        scheme = str(get_agent_config(db, "pm.allocator_scheme") or "equal").strip().lower()
        if scheme == "regime":
            return float(base_mult)   # avoid double-tilting PEAD by regime
        if not _weights_fresh(db, int(get_agent_config(db, "pm.allocator_stale_days"))):
            return float(base_mult)
        pw = float(get_agent_config(db, "pm.allocator_pead_weight"))
        return map_pead_size_mult(float(base_mult), pw)
    except Exception:
        return float(base_mult)


# ─────────────────────────────────────────────────────────────────────────────
# Weekly recompute (I/O orchestrator — never raises into the scheduler)
# ─────────────────────────────────────────────────────────────────────────────

def run_allocator(db=None, *, force: bool = False) -> dict:
    """Recompute + persist effective sleeve weights. Gated on pm.allocator_enabled
    (unless force). In warmup (insufficient live history) it persists NOTHING — readers
    keep falling back to static. Records an allocator_tracker row. Never raises."""
    summary = {"status": "ok", "source": None, "scheme": None, "weights": None, "regime": None}
    own_db = db is None
    if own_db:
        from app.database.session import get_session
        db = get_session()
    try:
        from app.database.agent_config import get_agent_config, set_agent_config
        from app.strategy.sleeve_allocator import AllocatorConfig
        from app.live_trading import allocator_tracker

        enabled = _truthy(get_agent_config(db, "pm.allocator_enabled"))
        if not enabled and not force:
            log.info("allocator: DISABLED (pm.allocator_enabled=false) — sleeves use static budgets")
            summary["status"] = "disabled"
            return summary

        scheme = str(get_agent_config(db, "pm.allocator_scheme") or "equal").strip().lower()
        min_days = int(get_agent_config(db, "pm.allocator_min_deployed_days"))
        vol_lb = int(get_agent_config(db, "pm.allocator_vol_lookback"))
        cfg = AllocatorConfig(vol_lookback=vol_lb)

        returns = _load_sleeve_returns(min_days)
        regime = _live_regime_label()
        summary["scheme"] = scheme
        summary["regime"] = regime

        if returns is None:
            # Warmup: not enough live history. Persist nothing -> readers fall back to
            # static (today's fixed budgets). Record the decision for observability.
            log.info("allocator: WARMUP (< %d deployed days per sleeve) — fallback to static", min_days)
            summary["status"] = "warmup"
            summary["source"] = "fallback_warmup"
            allocator_tracker.record(scheme=scheme, enabled=enabled,
                                     source="fallback_warmup", regime=regime, n_days=0)
            return summary

        weights = compute_sleeve_weights(returns, scheme, regime, cfg)
        tw = round(float(weights.get("trend", 0.0)), 6)
        pw = round(float(weights.get("pead", 0.0)), 6)
        set_agent_config(db, "pm.allocator_trend_weight", tw)
        set_agent_config(db, "pm.allocator_pead_weight", pw)
        set_agent_config(db, "pm.allocator_last_computed", datetime.now(timezone.utc).isoformat())

        allocator_tracker.record(scheme=scheme, enabled=enabled, source="allocator",
                                 trend_weight=tw, pead_weight=pw, regime=regime,
                                 n_days=int(len(returns)))
        log.info("allocator: scheme=%s regime=%s -> trend=%.3f pead=%.3f (n=%d)",
                 scheme, regime, tw, pw, len(returns))
        summary.update(status="ok", source="allocator", weights={"trend": tw, "pead": pw})
        return summary
    except Exception:
        log.exception("allocator: run_allocator failed (swallowed)")
        summary["status"] = "error"
        return summary
    finally:
        if own_db:
            try:
                db.close()
            except Exception:
                pass
