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
  * Capital: pm.trend_allocation_pct (0.40) of NAV gross, equal-capital 50/50 with
    PEAD under the global 80% gross cap. Per-ETF cap pm.trend_max_position_pct (0.25).
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
    trend_allocation_pct: sleeve budget as a fraction of NAV (e.g. 0.40).
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
           block_reason: Optional[str] = None) -> None:
    try:
        from app.database.decision_audit import write_decision
        write_decision(
            symbol=symbol,
            strategy="trend",
            final_decision=final_decision,
            direction=("BUY" if side == "buy" else "SELL"),
            size_multiplier=1.0,
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
            all_positions = alpaca.get_positions() or []
            total_gross = sum(abs(float(p.get("market_value") or 0.0))
                              for p in all_positions)
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

        # ── Execute or shadow ──
        if shadow:
            for it in approved:
                _audit(it["symbol"], it["side"], price=live.get(it["symbol"], 0.0),
                       final_decision="enter", block_reason="shadow")
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
                try:
                    _sync_trend_trade(db, sym, it["target_shares"], price, oid)
                except Exception:
                    log.debug("trend: _sync_trend_trade failed (swallowed)", exc_info=True)
                _audit(sym, side, price=price, final_decision="enter")
                placed += 1
            db.commit()
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
