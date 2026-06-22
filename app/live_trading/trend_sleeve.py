"""
Live TSMOM trend sleeve — Alpha-v4 weekly ETF rebalancer.

The trend sleeve (app/strategy/tsmom.py, validated standalone Sharpe +0.71, the
book's crisis-diversifier) trades a fixed liquid-ETF basket by rebalancing weekly
to inverse-vol, long-flat target weights. Unlike PEAD (a daily event-driven stock
scan routed through the PM -> RM -> Trader proposal queue), the trend sleeve is a
self-contained weekly rebalance-to-target that runs ALONGSIDE PEAD and places
orders DIRECTLY via Alpaca with its own lightweight risk gate.

Why standalone (not a pm.swing_selector value): the selector is mutual-exclusion
(one daily scan); trend must coexist with PEAD as a peer sleeve on its own cadence.

Safety model (deploys dormant + shadow-first):
  * pm.trend_enabled (default false)  — master flag; false = code present, no run.
  * pm.trend_shadow  (default true)   — true = compute + LOG would-be orders to
    decision_audit (block_reason="shadow") WITHOUT sending. Flip to false to arm.
  * Capital: pm.trend_allocation_pct (0.50) of NAV gross — the SOLE live sleeve since
    the H1 DEMOTE (PEAD off); raised 0.25->0.50 in Alpha-v9 P1-2 (2026-06-16) after a
    Kelly/vol-target analysis (~4.7% standalone vol), under the global 80% gross cap.
    Per-ETF cap pm.trend_max_position_pct (0.25).
  * Fail-closed everywhere: kill-switch, data-fetch failure, NAV-fetch failure, or
    a missing core symbol -> NO orders.

Positions are tagged selector="trend", trade_type="trend"; the Trader's per-tick
stop/target exit loop SKIPS these (this rebalancer is their sole manager).

The two pure functions (compute_trend_deltas, apply_risk_gate) hold all the math
and are unit-tested in isolation, mirroring apply_pead_size_ramp. run_trend_rebalance
is the defensive I/O orchestrator — it never raises into the scheduler.
"""
from __future__ import annotations

import logging
from datetime import date as _date, datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Minimum daily-bar history required for the longest TSMOM lookback (252) plus a
# margin so realized-vol warmup and a possibly-missing latest SIP bar don't starve
# the signal. Below this the whole run fails closed.
MIN_HISTORY_ROWS = 253
# Bars to request: get_bars_batch daily backs off start by limit*1.5 calendar days,
# so ~330 rows of request -> ~495 calendar days -> comfortably 253+ trading rows.
BARS_LIMIT = 330
# A buy below this notional is dust — skip to avoid churn/cost on tiny weight drift
# (a full sell-to-zero is always executed regardless, for a clean exit).
MIN_NOTIONAL = 50.0
# If fewer than this fraction of the universe returns enough history, fail closed
# (a partial universe distorts the gross-scaling of the remaining weights).
MIN_UNIVERSE_PRESENT_FRAC = 0.6
# Core symbol that must be present or the run fails closed.
CORE_SYMBOL = "SPY"


# ─────────────────────────────────────────────────────────────────────────────
# Pure math (no I/O) — unit-tested in tests/test_trend_sleeve.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_deltas(
    target_weights: Dict[str, float],
    current_trend_positions: Dict[str, int],
    prices: Dict[str, float],
    nav: float,
    trend_allocation_pct: float,
    max_position_pct: float,
    *,
    min_notional: float = MIN_NOTIONAL,
) -> List[Dict[str, Any]]:
    """Translate TSMOM target weights into whole-share buy/sell order intents.

    target_weights:  symbol -> weight, from ``tsmom_weights(prices).iloc[-1]``
                     (long-flat, sums to <= max_gross). Negative/zero ignored.
    current_trend_positions: symbol -> signed shares currently held & tagged trend.
    prices:          symbol -> live price used for sizing (skip if <= 0).
    nav:             account equity.
    trend_allocation_pct: sleeve budget as a fraction of NAV (e.g. 0.50).
    max_position_pct:     per-ETF ceiling as a fraction of NAV (e.g. 0.25).

    Returns a list of intent dicts {symbol, side, qty, target_shares,
    current_shares, reason} with **all sells before all buys** (free buying power
    and drop names before adding). Whole shares only (Alpaca wrapper is int-only).
    """
    per_name_cap_dollars = max(0.0, max_position_pct) * max(0.0, nav)

    target_shares: Dict[str, int] = {}
    for sym, w in target_weights.items():
        price = prices.get(sym, 0.0)
        if price <= 0 or w is None or w <= 0:
            continue
        target_dollars = float(w) * trend_allocation_pct * nav
        if per_name_cap_dollars > 0:
            target_dollars = min(target_dollars, per_name_cap_dollars)
        shares = int(target_dollars / price)  # floor; never fractional
        if shares > 0:
            target_shares[sym] = shares

    sells: List[Dict[str, Any]] = []
    buys: List[Dict[str, Any]] = []
    for sym in sorted(set(target_shares) | set(current_trend_positions)):
        cur = int(current_trend_positions.get(sym, 0))
        tgt = int(target_shares.get(sym, 0))
        delta = tgt - cur
        if delta == 0:
            continue
        price = prices.get(sym, 0.0)
        is_full_exit = tgt == 0 and cur != 0
        # Skip dust trades (but never skip a full exit — keep the book clean).
        if (not is_full_exit) and price > 0 and abs(delta) * price < min_notional:
            continue
        intent = {
            "symbol": sym,
            "side": "buy" if delta > 0 else "sell",
            "qty": abs(delta),
            "target_shares": tgt,
            "current_shares": cur,
            "reason": "exit_not_in_target" if is_full_exit else "rebalance",
        }
        (buys if delta > 0 else sells).append(intent)

    return sells + buys


def apply_risk_gate(
    intents: List[Dict[str, Any]],
    *,
    total_gross: float,
    nav: float,
    max_position_pct: float,
    prices: Dict[str, float],
    gross_cap: Optional[float] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Lightweight per-order risk gate. Returns (approved, blocked).

    Sells are always approved (they reduce gross). Buys are rejected if they fail a
    fat-finger sanity check or would push total deployed gross over ``gross_cap``
    (the global 80% cap shared with PEAD — existing trend holdings are NOT force-sold
    to make room; only new buys are constrained, so the sleeve simply runs underweight
    when PEAD has overshot). Assumes ``intents`` is sells-first (compute_trend_deltas
    guarantees this) so freed gross is available to subsequent buys.

    ``total_gross`` is Sum|market_value| across ALL Alpaca positions (both sleeves);
    same formula as Risk Manager rule 0b.
    """
    if gross_cap is None:
        # Canonical source: app/agents/risk_manager.GROSS_EXPOSURE_CAP (lazy import
        # to avoid re-declaring the constant and to keep module import light).
        from app.agents.risk_manager import GROSS_EXPOSURE_CAP as gross_cap  # type: ignore

    approved: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    projected = float(total_gross)
    cap_dollars = float(gross_cap) * float(nav)

    for it in intents:
        sym = it["symbol"]
        price = prices.get(sym, 0.0)
        cost = it["qty"] * price

        if it["side"] == "sell":
            projected -= cost
            approved.append(it)
            continue

        # ── buy: fat-finger + gross cap ──
        if it["qty"] <= 0 or price <= 0 or cost > max_position_pct * nav * 1.1 or cost > nav:
            blocked.append({**it, "block_reason": "fat_finger"})
            continue
        if projected + cost > cap_dollars + 1e-6:
            blocked.append({**it, "block_reason": "gross_cap"})
            continue
        projected += cost
        approved.append(it)

    return approved, blocked


# ─────────────────────────────────────────────────────────────────────────────
# I/O orchestrator — defensive; never raises into the scheduler
# ─────────────────────────────────────────────────────────────────────────────

def _truthy(val: Any) -> bool:
    return str(val).strip().lower() == "true"


def _refresh_macro_history_bounded(timeout_s: float = 20.0) -> None:
    """Best-effort macro refresh with a HARD wall-clock timeout, so a yfinance network
    *hang* (not just an exception) can never block the weekly rebalance thread. Runs in a
    daemon thread (won't block interpreter exit) and is joined for at most `timeout_s`; on
    timeout/error we simply fall through to the cached parquet (the staleness guard then
    decides). An orphaned hung fetch finishes or dies on its own and cannot stall the
    rebalance."""
    import threading

    def _run():
        try:
            from app.data.macro_history import update_macro_history
            update_macro_history()
        except Exception as _exc:  # noqa: BLE001
            log.debug("crash governor: macro refresh failed (using cache): %s", _exc)

    t = threading.Thread(target=_run, name="crash-gov-macro-refresh", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        log.warning("crash governor: macro refresh exceeded %.0fs -> using cached macro data",
                    timeout_s)


def _crash_governor_multiplier(db) -> float:
    """Alpha-v7 F1b: VIX term-structure crash-governor exposure multiplier for the trend
    sleeve, in [derisk_to, 1.0]. De-risks the whole sleeve when the VIX term structure is
    inverted (VIX > VIX3M = backwardation = acute stress).

    FAIL-SAFE by design — returns 1.0 (no de-risk = exactly today's behavior) when the flag
    is off, VIX/VIX3M is missing or stale, the signal is insufficient, or ANY error occurs.
    A data outage can therefore NEVER flatten the book; the governor can only ever REDUCE
    exposure (multiplier <= 1.0), and the downstream per-name + 80% gross caps still apply.
    """
    from app.database.agent_config import get_agent_config
    try:
        if not _truthy(get_agent_config(db, "pm.crash_governor_enabled")):
            return 1.0
        import pandas as pd
        from datetime import date as _date
        from app.data.macro_history import load_macro_history
        from app.strategy.crash_governor import (
            VixTermGovernorConfig, live_governor_multiplier,
        )
        cfg = VixTermGovernorConfig(
            derisk_to=float(get_agent_config(db, "pm.crash_governor_derisk_to")),
            ratio_threshold=float(get_agent_config(db, "pm.crash_governor_ratio_threshold")),
            confirm_days=int(get_agent_config(db, "pm.crash_governor_confirm_days")),
        )
        # Best-effort fresh tail, bounded by a hard timeout; falls through to cache on
        # timeout/error (the staleness guard below then decides).
        _refresh_macro_history_bounded()
        mh = load_macro_history()
        if mh is None or mh.empty or "vix" not in mh.columns or "vix3m" not in mh.columns:
            log.warning("crash governor: macro history unavailable -> mult=1.0 (fail-safe)")
            return 1.0
        # PIT: drop any not-yet-settled row dated TODAY. The Mon 09:45 ET rebalance runs
        # DURING market hours and update_macro_history() fetches through today, so yfinance
        # returns an intraday PARTIAL ^VIX/^VIX3M bar for today. Size only off SETTLED closes
        # (strictly before today) — the latest settled close governs the upcoming session,
        # exactly matching the backtest's shift(1). Without this the live signal would use
        # today's unsettled VIX (a same-day look-ahead, and a live-vs-backtest mismatch).
        _today_iso = _date.today().isoformat()
        mh = mh[mh["date"].astype(str).str[:10] < _today_iso]
        if mh.empty:
            log.warning("crash governor: no settled macro rows before today -> 1.0 (fail-safe)")
            return 1.0
        # Staleness guard (on the latest SETTLED close): a wedged feed must not de-risk on
        # stale data.
        last_date = pd.Timestamp(str(mh["date"].iloc[-1])[:10])
        if (pd.Timestamp(_date.today()) - last_date).days > 7:
            log.warning("crash governor: macro history stale (last settled=%s) -> 1.0 (fail-safe)",
                        last_date.date())
            return 1.0
        idx = pd.to_datetime(mh["date"])
        n = max(1, cfg.confirm_days)
        mult = live_governor_multiplier(
            pd.Series(mh["vix"].to_numpy(), index=idx).tail(n + 5),
            pd.Series(mh["vix3m"].to_numpy(), index=idx).tail(n + 5),
            cfg,
        )
        if mult is None or not (0.0 <= mult <= 1.0):
            log.warning("crash governor: insufficient/invalid signal -> mult=1.0 (fail-safe)")
            return 1.0
        if mult < 1.0:
            # Comprehensive audit line (the live record of WHY exposure was cut on this date):
            # the de-risk multiplier, the term-structure ratio, and the settled close used.
            try:
                ratio = float(mh["vix"].iloc[-1]) / float(mh["vix3m"].iloc[-1])
            except Exception:  # noqa: BLE001
                ratio = float("nan")
            log.info("crash governor ACTIVE: de-risking trend exposure x%.2f "
                     "(VIX/VIX3M=%.3f backwardation, settled close %s)",
                     mult, ratio, last_date.date())
        return float(mult)
    except Exception:  # noqa: BLE001
        log.warning("crash governor failed -> mult=1.0 (fail-safe)", exc_info=True)
        return 1.0


# Compose floor for the stacked overlay multipliers (mirrors sleeve_lab.GLOBAL_DERISK_FLOOR;
# duplicated as a local app-layer constant to avoid importing research code into the live path).
_OVERLAY_DERISK_FLOOR = 0.25


def _credit_governor_multiplier(db) -> float:
    """Alpha-v8 G1: credit-spread de-risk multiplier for the trend sleeve, in [derisk_to, 1.0].
    De-risks when the HYG/IEF ratio is >`band` below its `lookback`-day MA (credit spreads
    widening). The validated G1 candidate (L120 / 2%-band / 0.5), evaluated marginal to the VIX
    governor — a small tail-insurance overlay on a SLOWER credit axis than the governor.

    Flag `pm.credit_governor_enabled` defaults OFF (this is an owner-gated CANDIDATE, not yet
    approved like the VIX governor). FAIL-SAFE: flag off / missing / stale / error -> 1.0 (no
    de-risk); can only REDUCE exposure. PIT: settled closes only (matches the backtest)."""
    from app.database.agent_config import get_agent_config
    try:
        if not _truthy(get_agent_config(db, "pm.credit_governor_enabled")):
            return 1.0
        import pandas as pd
        from datetime import date as _date
        from app.data.macro_history import load_macro_history
        from app.strategy.credit_curve_governor import (
            CreditGovernorConfig, live_credit_multiplier,
        )
        cfg = CreditGovernorConfig(
            lookback=int(get_agent_config(db, "pm.credit_governor_lookback")),
            band=float(get_agent_config(db, "pm.credit_governor_band")),
            derisk_to=float(get_agent_config(db, "pm.credit_governor_derisk_to")),
        )
        _refresh_macro_history_bounded()
        mh = load_macro_history()
        if mh is None or mh.empty or "hyg" not in mh.columns or "ief" not in mh.columns:
            log.warning("credit governor: macro history unavailable -> mult=1.0 (fail-safe)")
            return 1.0
        # PIT: settled closes only (drop today's intraday partial bar), like the VIX governor.
        _today_iso = _date.today().isoformat()
        mh = mh[mh["date"].astype(str).str[:10] < _today_iso]
        if mh.empty:
            log.warning("credit governor: no settled macro rows before today -> 1.0 (fail-safe)")
            return 1.0
        last_date = pd.Timestamp(str(mh["date"].iloc[-1])[:10])
        if (pd.Timestamp(_date.today()) - last_date).days > 7:
            log.warning("credit governor: macro stale (last settled=%s) -> 1.0 (fail-safe)",
                        last_date.date())
            return 1.0
        idx = pd.to_datetime(mh["date"])
        need = cfg.lookback + max(0, cfg.confirm_days - 1) + 5
        mult = live_credit_multiplier(
            pd.Series(mh["hyg"].to_numpy(), index=idx).tail(need),
            pd.Series(mh["ief"].to_numpy(), index=idx).tail(need),
            cfg,
        )
        if mult is None or not (0.0 <= mult <= 1.0):
            log.warning("credit governor: insufficient/invalid signal -> mult=1.0 (fail-safe)")
            return 1.0
        if mult < 1.0:
            try:
                ratio = float(mh["hyg"].iloc[-1]) / float(mh["ief"].iloc[-1])
            except Exception:  # noqa: BLE001
                ratio = float("nan")
            log.info("credit governor ACTIVE: de-risking trend exposure x%.2f "
                     "(HYG/IEF=%.3f below trend, settled close %s)", mult, ratio, last_date.date())
        return float(mult)
    except Exception:  # noqa: BLE001
        log.warning("credit governor failed -> mult=1.0 (fail-safe)", exc_info=True)
        return 1.0


def _fetch_prices(alpaca, universe: List[str]):
    """Return a (n_days, n_symbols) close-price DataFrame, or None on failure.

    Drops symbols with < MIN_HISTORY_ROWS rows; fails closed (None) if the core
    symbol is missing or too few of the universe survived.
    """
    import pandas as pd
    try:
        raw = alpaca.get_bars_batch(universe, "1D", BARS_LIMIT)
    except Exception as exc:
        log.warning("trend: get_bars_batch failed: %s", exc)
        return None
    if not raw:
        return None

    series = {}
    for sym, df in raw.items():
        if df is None or "close" not in df.columns:
            continue
        closes = df["close"].dropna()
        if len(closes) >= MIN_HISTORY_ROWS:
            series[sym] = closes

    if CORE_SYMBOL not in series:
        log.warning("trend: core symbol %s missing/short — fail-closed", CORE_SYMBOL)
        return None
    if len(series) < max(1, int(len(universe) * MIN_UNIVERSE_PRESENT_FRAC)):
        log.warning("trend: only %d/%d symbols have history — fail-closed",
                    len(series), len(universe))
        return None

    prices = pd.DataFrame(series).sort_index()
    return prices


def _live_prices(alpaca, symbols: List[str], fallback_df) -> Dict[str, float]:
    """Latest price per symbol for sizing — live quote first, last daily close as
    fallback (the latest SIP daily bar may be delayed on the free tier)."""
    out: Dict[str, float] = {}
    for sym in symbols:
        px = None
        try:
            px = alpaca.get_latest_price(sym)
        except Exception:
            px = None
        if (px is None or px <= 0) and fallback_df is not None and sym in fallback_df.columns:
            try:
                px = float(fallback_df[sym].dropna().iloc[-1])
            except Exception:
                px = None
        if px and px > 0:
            out[sym] = float(px)
    return out


def _current_trend_positions(db, alpaca) -> Dict[str, int]:
    """Symbols tagged selector='trend' in the DB, sized from ACTUAL Alpaca shares
    (trust Alpaca for quantity = handles partial fills; trust DB for attribution)."""
    from app.database.models import Trade
    try:
        rows = (
            db.query(Trade)
            .filter(Trade.selector == "trend",
                    Trade.status.in_(["ACTIVE", "PENDING_FILL"]))
            .all()
        )
        trend_syms = {r.symbol for r in rows}
    except Exception as exc:
        log.warning("trend: DB position query failed: %s", exc)
        trend_syms = set()

    if not trend_syms:
        return {}

    try:
        positions = alpaca.get_positions() or []
    except Exception as exc:
        log.warning("trend: get_positions failed during reconcile: %s", exc)
        return {}

    return {
        p["symbol"]: int(p.get("qty") or 0)
        for p in positions
        if p.get("symbol") in trend_syms
    }


def _sync_trend_trade(db, symbol: str, target_shares: int, price: float,
                      order_id: Optional[str]) -> None:
    """Keep one Trade row per trend symbol reflecting the post-rebalance position.

    The rebalancer is authoritative; the source of truth for share count next cycle
    is Alpaca (via _current_trend_positions), so this row is primarily attribution
    (selector/trade_type='trend') + dashboard state. target_shares==0 closes it.
    """
    from app.database.models import Trade
    existing = (
        db.query(Trade)
        .filter(Trade.symbol == symbol, Trade.selector == "trend",
                Trade.status.in_(["ACTIVE", "PENDING_FILL"]))
        .order_by(Trade.id.desc())
        .first()
    )
    if target_shares <= 0:
        if existing:
            existing.status = "CLOSED"
            existing.closed_at = datetime.utcnow()
            existing.exit_price = price or existing.exit_price
            existing.exit_reason = "trend_rebalance_exit"
        return
    if existing:
        existing.quantity = int(target_shares)
        if order_id:
            existing.alpaca_order_id = order_id
    else:
        db.add(Trade(
            symbol=symbol,
            direction="BUY",            # trend is long-flat — positions are always long
            entry_price=float(price or 0.0),
            quantity=int(target_shares),
            status="PENDING_FILL",
            signal_type="TSMOM",
            trade_type="trend",
            selector="trend",
            alpaca_order_id=order_id,
            created_at=datetime.utcnow(),
        ))


def _audit(symbol: str, side: str, *, price: float, final_decision: str,
           block_reason: Optional[str] = None, size_mult: float = 1.0) -> None:
    """Persist one trend decision row. `size_mult` is the applied exposure multiplier —
    1.0 normally, or the crash-governor multiplier (<1.0) on a de-risk day — so the
    decision_audit trail records WHY a given day's trend orders were sized down (the
    queryable artifact for incident review; the run log carries the VIX/VIX3M detail)."""
    try:
        from app.database.decision_audit import write_decision
        write_decision(
            symbol=symbol,
            strategy="trend",
            final_decision=final_decision,
            direction=("BUY" if side == "buy" else "SELL"),
            size_multiplier=float(size_mult),
            price_at_decision=price,
            block_reason=block_reason,
        )
    except Exception:
        log.debug("trend: decision_audit write failed (swallowed)", exc_info=True)


def run_trend_rebalance(db=None, *, force: bool = False) -> Dict[str, Any]:
    """Execute (or shadow) one weekly trend rebalance. Never raises.

    force=True bypasses the trend_enabled dormancy gate (used for shadow dry-runs);
    the kill-switch and shadow flag are always honored.

    Returns a summary dict (status, mode, counts, intents) for logging/tests.
    """
    summary: Dict[str, Any] = {"status": "ok", "mode": None,
                               "approved": [], "blocked": [], "nav": None}
    own_db = db is None
    if own_db:
        from app.database.session import get_session
        db = get_session()

    try:
        from app.database.agent_config import get_agent_config

        enabled = _truthy(get_agent_config(db, "pm.trend_enabled"))
        shadow = _truthy(get_agent_config(db, "pm.trend_shadow"))
        if not enabled and not force:
            log.info("trend: DORMANT (pm.trend_enabled=false) — no rebalance")
            summary["status"] = "dormant"
            return summary

        summary["mode"] = "shadow" if shadow else "live"
        # Alpha-v4 P3: effective allocation = allocator weight x budget when the live
        # allocator is enabled+fresh, else the static pm.trend_allocation_pct (today's
        # behavior). Fail-closed to the static value on any error.
        from app.live_trading.sleeve_allocator_live import effective_trend_allocation
        alloc = effective_trend_allocation(db)
        # Alpha-v7 F1b: VIX term-structure crash governor — scale the sleeve's exposure by
        # [derisk_to, 1.0] (de-risk in backwardation). Fail-safe to 1.0 (= today's behavior);
        # can only REDUCE exposure, never increase it. Applied to the scalar budget so it
        # uniformly de-risks every target dollar; per-name + 80% gross caps still bind.
        # Overlays COMPOSE multiplicatively, clamped to the floor so a stacked de-risk can't
        # near-flatten the book. VIX governor (F1b, default ON) x credit governor (G1 candidate,
        # default OFF). Each is fail-safe to 1.0; the product can only REDUCE exposure; per-name
        # + 80% gross caps still bind downstream.
        gov_mult = _crash_governor_multiplier(db)
        credit_mult = _credit_governor_multiplier(db)
        overlay_mult = max(_OVERLAY_DERISK_FLOOR, gov_mult * credit_mult)
        alloc = alloc * overlay_mult
        summary["crash_governor_mult"] = gov_mult
        summary["credit_governor_mult"] = credit_mult
        summary["overlay_mult"] = overlay_mult
        max_pos = float(get_agent_config(db, "pm.trend_max_position_pct"))
        universe = [s.strip().upper()
                    for s in str(get_agent_config(db, "pm.trend_universe")).split(",")
                    if s.strip()]
        if not universe:
            log.warning("trend: empty universe — abort")
            summary["status"] = "failed"
            return summary

        # ── Kill switch (fail-closed) ──
        from app.live_trading.kill_switch import kill_switch
        if kill_switch.is_active:
            log.warning("trend: kill switch ACTIVE — no rebalance")
            summary["status"] = "blocked"
            summary["block_reason"] = "kill_switch"
            return summary

        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()

        # ── Data (fail-closed) ──
        prices_df = _fetch_prices(alpaca, universe)
        if prices_df is None or prices_df.empty:
            log.warning("trend: price data unavailable — fail-closed")
            _audit(CORE_SYMBOL, "buy", price=0.0, final_decision="block",
                   block_reason="data_fetch_failed")
            summary["status"] = "failed"
            summary["block_reason"] = "data_fetch_failed"
            return summary

        # ── Target weights ──
        from app.strategy.tsmom import TSMOMConfig, tsmom_weights
        cfg = TSMOMConfig(universe=[c for c in universe if c in prices_df.columns])
        target_weights = tsmom_weights(prices_df, cfg).iloc[-1].to_dict()
        # P1-4: the INTENDED book (fraction of NAV per name) the sleeve aimed to hold this
        # week — inverse-vol target x effective alloc (post-governor), clipped to the per-name
        # cap. Purely additive (a summary field); back_validation replays it on the SAME Alpaca
        # prices/calendar as the actual book to isolate execution friction. Pre-rounding,
        # pre-gross-cap (those deviations ARE the friction we measure).
        summary["intended_weights"] = {
            s: float(min(float(w) * float(alloc), max_pos))
            for s, w in target_weights.items() if w and float(w) > 0
        }

        # ── NAV (fail-closed; never size off a hardcoded fallback) ──
        try:
            nav = float(alpaca.get_account()["equity"])
        except Exception as exc:
            log.warning("trend: NAV fetch failed — fail-closed: %s", exc)
            _audit(CORE_SYMBOL, "buy", price=0.0, final_decision="block",
                   block_reason="nav_unavailable")
            summary["status"] = "failed"
            summary["block_reason"] = "nav_unavailable"
            return summary
        summary["nav"] = nav
        if nav <= 0:
            summary["status"] = "failed"
            summary["block_reason"] = "nav_unavailable"
            return summary

        # ── Current trend positions + live prices ──
        current = _current_trend_positions(db, alpaca)
        size_syms = sorted(set(target_weights) | set(current))
        live = _live_prices(alpaca, size_syms, prices_df)

        intents = compute_trend_deltas(
            target_weights, current, live, nav, alloc, max_pos,
        )

        # ── Gross cap (trend + PEAD <= 80%) ──
        try:
            from app.live_trading.cash_sleeve import CASH_ETFS  # cash-equiv: not risk gross
            all_positions = alpaca.get_positions() or []
            total_gross = sum(abs(float(p.get("market_value") or 0.0))
                              for p in all_positions
                              if p.get("symbol") not in CASH_ETFS)
        except Exception as exc:
            log.warning("trend: gross-cap position fetch failed — fail-closed: %s", exc)
            summary["status"] = "failed"
            summary["block_reason"] = "positions_unavailable"
            return summary

        approved, blocked = apply_risk_gate(
            intents, total_gross=total_gross, nav=nav,
            max_position_pct=max_pos, prices=live,
        )
        summary["approved"] = approved
        summary["blocked"] = blocked

        for b in blocked:
            _audit(b["symbol"], b["side"], price=live.get(b["symbol"], 0.0),
                   final_decision="block", block_reason=b.get("block_reason"))

        # ── H1 reconciliation-before-trade (shadow-first; the broker is reality, the DB is memory) ──
        # Runs BEFORE the risk gate: verify the DB book matches the broker before sizing/placing.
        # Mode read first so an ENFORCE-mode wiring failure fails CLOSED (HOLD); shadow is always inert.
        try:
            recon_mode = str(get_agent_config(db, "pm.reconciliation_mode") or "shadow").strip().lower()
        except Exception:
            recon_mode = "shadow"
        blocked_by_recon = False
        if recon_mode != "off":
            try:
                from app.live_trading import reconciliation as _recon
                from app.notifications import notifier as _rnotifier
                recon_result = _recon.shadow_reconcile_before_trade(
                    db, all_positions, nav=nav, mode=recon_mode, label="trend", notifier=_rnotifier)
                summary["recon_mode"] = recon_mode
                summary["recon_status"] = recon_result.status
                summary["recon_breaks"] = [(b.venue, b.instrument_id, b.expected_qty, b.actual_qty)
                                           for b in recon_result.position_breaks]
                blocked_by_recon = (recon_mode == "enforce" and not recon_result.ok_to_trade)
            except Exception:
                log.debug("trend: reconciliation wiring failed", exc_info=True)
                blocked_by_recon = (recon_mode == "enforce")   # fail-CLOSED in enforce; inert in shadow
        if blocked_by_recon:
            log.warning("trend: reconciliation (mode=%s) -> HOLD rebalance: %s",
                        recon_mode, summary.get("recon_breaks"))
            summary["status"] = "blocked"
            summary["block_reason"] = "reconciliation"
            return summary

        # ── R0.5 whole-book risk gate (shadow-first; observe-only unless mode=enforce) ──
        # Read the mode FIRST (outside the gate try) so an ENFORCE-mode gate failure fails CLOSED
        # (HOLD), while SHADOW mode is always inert — any failure leaves blocked_by_gate False so the
        # rebalance proceeds exactly as before.
        try:
            gate_mode = (get_agent_config(db, "pm.whole_book_gate_mode") or "shadow")
        except Exception:
            gate_mode = "shadow"
        blocked_by_gate = False
        try:
            from app.live_trading import whole_book_gate as wbg
            from app.notifications import notifier as _notifier
            gate_verdict = wbg.shadow_gate_from_intents(
                all_positions, approved, live, nav, mode=gate_mode, label="trend",
                notifier=_notifier)
            summary["gate_mode"] = gate_mode
            summary["gate_allow"] = bool(gate_verdict.allow)
            summary["gate_breaches"] = list(gate_verdict.breaches)
            blocked_by_gate = (gate_mode == "enforce" and not gate_verdict.allow)
        except Exception:
            log.debug("trend: whole-book gate wiring failed", exc_info=True)
            blocked_by_gate = (gate_mode == "enforce")   # fail-CLOSED in enforce; inert in shadow
        if blocked_by_gate:
            log.warning("trend: whole-book gate (mode=%s) -> HOLD rebalance: %s",
                        gate_mode, summary.get("gate_breaches"))
            summary["status"] = "blocked"
            summary["block_reason"] = "whole_book_gate"
            return summary

        # ── Execute or shadow ──
        if shadow:
            for it in approved:
                _audit(it["symbol"], it["side"], price=live.get(it["symbol"], 0.0),
                       final_decision="enter", block_reason="shadow", size_mult=overlay_mult)
            log.info("trend SHADOW: would place %d order(s) (%d blocked) — none sent",
                     len(approved), len(blocked))
        else:
            placed = 0
            stamp = _date.today().strftime("%Y%m%d")
            for it in approved:
                sym, side, qty = it["symbol"], it["side"], it["qty"]
                price = live.get(sym, 0.0)
                try:
                    order = alpaca.place_market_order(
                        sym, int(qty), side,
                        client_order_id=f"trend-{stamp}-{sym}",
                    )
                    oid = order.get("order_id") if isinstance(order, dict) else None
                except Exception as exc:
                    log.error("trend: order failed %s %s x%d: %s", side, sym, qty, exc)
                    _audit(sym, side, price=price, final_decision="block",
                           block_reason="order_error")
                    continue
                # Persist the PENDING_FILL row and COMMIT immediately, per order. A
                # deferred (post-loop) commit leaves every already-placed ETF with an
                # UNCOMMITTED Trade row, so a crash/restart mid-loop orphans real Alpaca
                # positions — which startup reconciliation then adopts as trade_type=
                # "swing" with a synthetic 2%/6% stop/target, letting the Trader liquidate
                # a trend leg mid-week and re-buy it next Monday (double-trade). Committing
                # per order shrinks that window to a single in-flight order.
                try:
                    _sync_trend_trade(db, sym, it["target_shares"], price, oid)
                    db.commit()
                except Exception:
                    db.rollback()
                    log.debug("trend: _sync_trend_trade/commit failed (swallowed)", exc_info=True)
                _audit(sym, side, price=price, final_decision="enter", size_mult=overlay_mult)
                placed += 1
            log.info("trend LIVE: placed %d order(s) (%d blocked)", placed, len(blocked))
            summary["placed"] = placed

        # ── Observability ──
        try:
            from app.live_trading import trend_tracker
            n_target = len([s for s in target_weights if (target_weights.get(s) or 0) > 0])
            trend_tracker.record_daily(
                n_positions=n_target,
                gross_deployed=_estimate_trend_gross(approved, current, live),
                turnover=sum(it["qty"] * live.get(it["symbol"], 0.0) for it in approved),
                extra={"mode": summary["mode"], "nav": nav,
                       "n_approved": len(approved), "n_blocked": len(blocked)},
            )
        except Exception:
            log.debug("trend: tracker record_daily failed (swallowed)", exc_info=True)

        return summary

    except Exception:
        log.exception("trend: run_trend_rebalance failed (swallowed)")
        summary["status"] = "error"
        return summary
    finally:
        if own_db:
            try:
                db.close()
            except Exception:
                pass


def _estimate_trend_gross(approved, current, live) -> float:
    """Approximate post-rebalance trend gross $ from target shares + live prices."""
    target_by_sym: Dict[str, int] = dict(current)
    for it in approved:
        target_by_sym[it["symbol"]] = it.get("target_shares", 0)
    return sum(abs(int(q)) * live.get(s, 0.0) for s, q in target_by_sym.items())
