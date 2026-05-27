"""
Startup reconciler — detects and heals inconsistencies between the DB and
Alpaca on every application startup.

Three classes of problem:
  1. Ghost positions   — ACTIVE in DB but NOT in Alpaca (position closed externally)
  2. Pending fills     — PENDING_FILL in DB; check Alpaca order status and promote/cancel
  3. Untracked         — in Alpaca but no DB Trade at all (limit filled during restart)

The reconciler NEVER modifies Alpaca state; it only updates DB records and
writes AuditLog entries so humans can review.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── Exit-price provenance ──────────────────────────────────────────────────────
# Canonical set of sources that represent a real broker fill (not a reconciler
# estimate).  Used by Rules A/B/C to decide whether to override a previous close.
_LEGITIMATE_EXIT_SOURCES = frozenset({"alpaca_fill", "agent_exit", "manual"})

# All recognised source values — enforced at write time.
EXIT_PRICE_SOURCES = frozenset({
    "alpaca_fill",
    "fallback_bar_close",
    "manual",
    "agent_exit",
    "legacy_unknown",
})

# ── Ghost state-machine constants ─────────────────────────────────────────────
GHOST_MIN_DETECTIONS = 2                    # must be seen missing this many times
GHOST_MIN_ELAPSED = timedelta(minutes=20)   # and at least this long since first seen
GHOST_MAX_PENDING_HOURS = 24               # auto-escalate to UNRESOLVED after this

RECONCILE_GHOST_PENDING = "RECONCILE_GHOST_PENDING"
RECONCILE_GHOST_UNRESOLVED = "RECONCILE_GHOST_UNRESOLVED"

# ── Untracked-skip constants ───────────────────────────────────────────────────
# Max times we skip reactivation for a "reconciler-exited" trade before we force
# a new RECONCILED record.  Keeps the reconciler from ignoring a genuine new
# position that happens to match a symbol we recently closed.
UNTRACKED_REACTIVATE_AFTER = 2

# Exit reasons written by the reconciler itself — used for Rule A: always
# reactivate when a previous close was a reconciler estimate (not a broker fill).
RECONCILER_EXIT_REASONS: frozenset = frozenset({
    "reconcile_ghost_expired",
    "reconcile_superseded",
    "reconcile_ghost_pending_timeout",
})


def write_exit_price(
    trade,
    price: float,
    source: str,
    order_id: str | None = None,
    written_by: str = "reconciler",
    *,
    force: bool = False,
) -> bool:
    """Write exit_price with provenance.  Returns True if written, False if blocked.

    Immutability guard: if a legitimate exit price is already recorded, refuse
    to overwrite unless ``force=True``.  This prevents the reconciler's bar-close
    fallback from clobbering a real broker fill.
    """
    if source not in EXIT_PRICE_SOURCES:
        logger.error("write_exit_price: unknown source %r for Trade#%d — rejected", source, trade.id)
        return False

    existing_source = getattr(trade, "exit_price_source", None)
    if (
        not force
        and trade.exit_price is not None
        and existing_source in _LEGITIMATE_EXIT_SOURCES
    ):
        logger.warning(
            "write_exit_price: Trade#%d already has exit_price=$%.4f (source=%s) — "
            "refusing to overwrite with source=%s (use force=True to override)",
            trade.id, trade.exit_price, existing_source, source,
        )
        return False

    trade.exit_price = price
    trade.exit_price_source = source
    trade.exit_order_id = order_id
    trade.exit_price_written_at = datetime.utcnow()
    trade.exit_price_written_by = written_by
    return True


# ── Target/Stop sanity bounds ──────────────────────────────────────────────────
# Bounds were chosen to be wide enough to never reject a legitimate trade while
# still catching gross corruption (e.g. AVGO target $1,993 on $413 entry, ~382%
# above entry — caused by self-reinforcing ATR feedback bug in EXTEND_TARGET).
#
# Swing trades use ATR_TARGET_MULT≈1.5 and ATR_STOP_MULT≈0.5 against atr_norm
# clipped to [0.005, 0.08] for stop and [0.01, 0.16] for target → real bounds
# fall comfortably inside [1%, 50%] target and [0.5%, 20%] stop. We add a
# safety margin to avoid edge-case false positives.
#
# Intraday uses tighter targets/stops (down to 0.3%/0.2%) so its lower bounds
# are looser.
TARGET_STOP_BOUNDS = {
    "swing": {
        "target_min_pct": 0.005,   # 0.5% — anything tighter is almost certainly a bug
        "target_max_pct": 0.50,    # 50% — catches runaway-ATR corruption
        "stop_min_pct":   0.003,   # 0.3% — tighter than any realistic swing stop
        "stop_max_pct":   0.20,    # 20% — wider than any sane swing risk
    },
    "intraday": {
        "target_min_pct": 0.0015,  # 0.15% — covers 0.3% intraday floor with margin
        "target_max_pct": 0.15,    # 15% — intraday never legitimately needs more
        "stop_min_pct":   0.001,   # 0.10% — covers 0.2% intraday floor with margin
        "stop_max_pct":   0.10,    # 10%
    },
}


def validate_target_stop(
    entry_price: float,
    target_price: float | None,
    stop_price: float | None,
    direction: str = "BUY",
    trade_type: str = "swing",
) -> tuple[bool, str]:
    """Validate that a target/stop pair is internally consistent and within sane bounds.

    Returns (is_valid, reason_if_invalid).  Either target_price or stop_price may
    be None — only the supplied side is checked.  Bounds are deliberately wide to
    avoid false positives on legitimate trades; the goal is to catch obvious
    corruption (e.g. target 5× entry from runaway-ATR), not enforce strategy rules.
    """
    if entry_price is None or entry_price <= 0:
        return False, f"entry_price={entry_price!r} is non-positive"

    bounds = TARGET_STOP_BOUNDS.get(trade_type or "swing", TARGET_STOP_BOUNDS["swing"])
    is_short = (direction or "BUY") == "SELL_SHORT"

    if target_price is not None:
        if target_price <= 0:
            return False, f"target_price={target_price} is non-positive"
        # For longs: target above entry; for shorts: target below entry
        if is_short:
            if target_price >= entry_price:
                return False, (
                    f"SHORT target ${target_price:.4f} >= entry ${entry_price:.4f} "
                    "(target must be below entry for shorts)"
                )
            pct = (entry_price - target_price) / entry_price
        else:
            if target_price <= entry_price:
                return False, (
                    f"LONG target ${target_price:.4f} <= entry ${entry_price:.4f} "
                    "(target must be above entry for longs)"
                )
            pct = (target_price - entry_price) / entry_price
        if pct < bounds["target_min_pct"]:
            return False, (
                f"target ${target_price:.4f} only {pct*100:.3f}% from entry "
                f"${entry_price:.4f} — below min {bounds['target_min_pct']*100:.2f}% "
                f"for {trade_type}"
            )
        if pct > bounds["target_max_pct"]:
            return False, (
                f"target ${target_price:.4f} is {pct*100:.1f}% from entry "
                f"${entry_price:.4f} — exceeds max {bounds['target_max_pct']*100:.0f}% "
                f"for {trade_type} (likely corruption)"
            )

    if stop_price is not None:
        if stop_price <= 0:
            return False, f"stop_price={stop_price} is non-positive"
        if is_short:
            if stop_price <= entry_price:
                return False, (
                    f"SHORT stop ${stop_price:.4f} <= entry ${entry_price:.4f} "
                    "(stop must be above entry for shorts)"
                )
            pct = (stop_price - entry_price) / entry_price
        else:
            if stop_price >= entry_price:
                return False, (
                    f"LONG stop ${stop_price:.4f} >= entry ${entry_price:.4f} "
                    "(stop must be below entry for longs)"
                )
            pct = (entry_price - stop_price) / entry_price
        if pct < bounds["stop_min_pct"]:
            return False, (
                f"stop ${stop_price:.4f} only {pct*100:.3f}% from entry "
                f"${entry_price:.4f} — below min {bounds['stop_min_pct']*100:.2f}% "
                f"for {trade_type}"
            )
        if pct > bounds["stop_max_pct"]:
            return False, (
                f"stop ${stop_price:.4f} is {pct*100:.1f}% from entry "
                f"${entry_price:.4f} — exceeds max {bounds['stop_max_pct']*100:.0f}% "
                f"for {trade_type} (likely corruption)"
            )

    return True, ""


def write_target_stop(
    trade,
    *,
    target_price: float | None = None,
    stop_price: float | None = None,
    written_by: str = "unknown",
    reason: str = "",
) -> bool:
    """Single chokepoint for writing target/stop on a Trade row.

    Validates against ``validate_target_stop()`` first.  Logs an ERROR and
    returns False if the values are insane — does NOT silently cap or fix
    them; silent fixes hide bugs.

    Pass only the fields you want to update; the other is left untouched.
    Returns True if at least one field was written.
    """
    if target_price is None and stop_price is None:
        return False

    entry = float(getattr(trade, "entry_price", 0) or 0)
    direction = getattr(trade, "direction", "BUY") or "BUY"
    trade_type = getattr(trade, "trade_type", "swing") or "swing"

    ok, why = validate_target_stop(
        entry_price=entry,
        target_price=target_price,
        stop_price=stop_price,
        direction=direction,
        trade_type=trade_type,
    )
    if not ok:
        logger.error(
            "write_target_stop: REJECTED Trade#%s %s (%s, %s) by=%s reason=%s: %s",
            getattr(trade, "id", "?"), getattr(trade, "symbol", "?"),
            direction, trade_type, written_by, reason or "n/a", why,
        )
        return False

    if target_price is not None:
        trade.target_price = float(target_price)
    if stop_price is not None:
        trade.stop_price = float(stop_price)
    logger.debug(
        "write_target_stop: Trade#%s %s target=%s stop=%s by=%s",
        getattr(trade, "id", "?"), getattr(trade, "symbol", "?"),
        target_price, stop_price, written_by,
    )
    return True


def audit_active_target_stops(db_session) -> list[dict]:
    """Scan all ACTIVE trades for corrupt target/stop values.

    Logs WARN for each failure with full details.  Does NOT auto-fix — surfaces
    the problem so a human can investigate.  Returns a list of {trade_id,
    symbol, entry, target, stop, reason} dicts for inclusion in the reconciler
    summary.
    """
    from app.database.models import Trade
    findings: list[dict] = []
    try:
        trades = db_session.query(Trade).filter_by(status="ACTIVE").all()
    except Exception as exc:
        logger.error("audit_active_target_stops: query failed: %s", exc)
        return findings

    for trade in trades:
        entry = float(trade.entry_price or 0)
        if entry <= 0:
            continue
        direction = getattr(trade, "direction", "BUY") or "BUY"
        trade_type = getattr(trade, "trade_type", "swing") or "swing"
        ok, why = validate_target_stop(
            entry_price=entry,
            target_price=trade.target_price,
            stop_price=trade.stop_price,
            direction=direction,
            trade_type=trade_type,
        )
        if not ok:
            logger.warning(
                "TARGET/STOP AUDIT: Trade#%d %s (%s %s) entry=$%.4f "
                "target=%s stop=%s — %s",
                trade.id, trade.symbol, direction, trade_type, entry,
                trade.target_price, trade.stop_price, why,
            )
            findings.append({
                "trade_id": trade.id,
                "symbol": trade.symbol,
                "direction": direction,
                "trade_type": trade_type,
                "entry_price": entry,
                "target_price": trade.target_price,
                "stop_price": trade.stop_price,
                "reason": why,
            })
    if findings:
        logger.warning(
            "TARGET/STOP AUDIT: %d ACTIVE trade(s) have insane target/stop — "
            "review and correct manually", len(findings),
        )
    return findings


def _is_broker_view_trusted(alpaca, alpaca_positions: dict) -> bool:
    """Return True if Alpaca's position snapshot is trustworthy.

    An empty snapshot is usually valid (no positions), but if the account
    has equity well above cash it almost certainly still holds something —
    the API returned a partial/error response.  In that case, skip ghost
    marking for this run.
    """
    if alpaca_positions:
        return True  # non-empty snapshot is always trusted
    try:
        acct = alpaca.trading_client.get_account()
        equity = float(getattr(acct, "equity", 0) or 0)
        cash = float(getattr(acct, "cash", 0) or 0)
        if equity - cash > 1.0:
            logger.warning(
                "_is_broker_view_trusted: Alpaca returned 0 positions but "
                "equity=%.2f cash=%.2f — API may be returning incomplete data; "
                "skipping ghost marking this run",
                equity, cash,
            )
            return False
    except Exception as exc:
        logger.debug("_is_broker_view_trusted account check failed: %s", exc)
    return True


# Minimum number of Alpaca positions required before we trust a "ghost" detection.
# If Alpaca returns fewer positions than this threshold AND we have many ACTIVE DB
# trades, it's likely an API error rather than genuine position closure — skip ghost
# marking to avoid false positives that generate duplicate records on next restart.
_MIN_POSITIONS_TO_TRUST_GHOST = 1


def reconcile(alpaca, db_session) -> Dict[str, Any]:
    """
    Run reconciliation. Returns a summary dict.

    Args:
        alpaca:      AlpacaClient instance
        db_session:  SQLAlchemy session
    """
    from app.database.models import Trade, Order, AuditLog

    result: Dict[str, Any] = {
        "ghost_positions": [],
        "pending_fills_resolved": [],
        "untracked_positions": [],
        "orphaned_orders": [],
    }

    # Fetch Alpaca state once — used by all checks below.
    # NOTE: An exception (or None return) here means the API call FAILED and we
    # cannot trust any state — bail out entirely. A successful call that returns
    # an empty list is a *valid* response meaning "no positions" and MUST be
    # trusted; otherwise we'd never detect a ghost when the user closes the only
    # remaining position outside MrTrader.
    api_call_ok = True
    try:
        raw_positions = alpaca.get_positions()
        if raw_positions is None:
            api_call_ok = False
            logger.error("Could not fetch Alpaca positions — get_positions() returned None; reconciliation skipped")
            return result
        alpaca_positions: Dict[str, Any] = {p["symbol"]: p for p in raw_positions}
    except Exception as exc:
        logger.error("Could not fetch Alpaca positions — reconciliation skipped: %s", exc)
        return result

    active_trades = db_session.query(Trade).filter_by(status="ACTIVE").all()
    n_alpaca_positions = len(alpaca_positions)

    # ── 1. Ghost positions — multi-run state machine ──────────────────────────
    # A ghost = ACTIVE in DB but NOT in Alpaca, AND old enough that we know it
    # isn't a brand-new market order whose fill hasn't propagated yet.
    #
    # Two-pass state machine (prevents single-restart false positives):
    #   Pass A: ACTIVE → RECONCILE_GHOST_PENDING  (first detection)
    #   Pass B: PENDING (≥2 detections, ≥20 min elapsed) → CLOSED
    #           PENDING (>24 h elapsed)             → RECONCILE_GHOST_UNRESOLVED
    #           PENDING (in Alpaca again)            → revert to ACTIVE
    if api_call_ok and _is_broker_view_trusted(alpaca, alpaca_positions):
        try:
            # Read ghost minimum age from agent_config (default 5 minutes).
            try:
                from app.database.agent_config import get_agent_config
                ghost_min_age_minutes = int(get_agent_config(db_session, "reconcile.ghost_min_age_minutes") or 5)
            except Exception:
                ghost_min_age_minutes = 5
            ghost_cutoff = datetime.utcnow() - timedelta(minutes=ghost_min_age_minutes)
            now = datetime.utcnow()

            # Pass A — promote ACTIVE → RECONCILE_GHOST_PENDING
            for trade in active_trades:
                if trade.symbol in alpaca_positions:
                    continue
                trade_age_anchor = trade.created_at or now
                if trade_age_anchor > ghost_cutoff:
                    logger.info(
                        "Skipping ghost check for Trade#%d %s — too recent (%.1f min old, threshold %d min)",
                        trade.id, trade.symbol,
                        (now - trade_age_anchor).total_seconds() / 60.0,
                        ghost_min_age_minutes,
                    )
                    continue
                # First detection — enter PENDING state
                count = (trade.ghost_detection_count or 0) + 1
                trade.ghost_detection_count = count
                trade.ghost_last_detected_at = now
                if trade.ghost_first_detected_at is None:
                    trade.ghost_first_detected_at = now
                trade.status = RECONCILE_GHOST_PENDING
                logger.warning(
                    "GHOST PENDING: Trade#%d %s not in Alpaca (detection #%d, "
                    "Alpaca returned %d position(s))",
                    trade.id, trade.symbol, count, n_alpaca_positions,
                )
                result["ghost_positions"].append({
                    "trade_id": trade.id, "symbol": trade.symbol,
                    "entry_price": trade.entry_price, "quantity": trade.quantity,
                    "ghost_detection_count": count,
                })
                db_session.add(AuditLog(
                    action="RECONCILE_GHOST_PENDING",
                    details={
                        "trade_id": trade.id, "symbol": trade.symbol,
                        "ghost_detection_count": count,
                        "alpaca_position_count": n_alpaca_positions,
                        "detected_at": now.isoformat(),
                    },
                    timestamp=now,
                ))

            # Pass B — resolve RECONCILE_GHOST_PENDING trades
            pending_ghosts = db_session.query(Trade).filter_by(status=RECONCILE_GHOST_PENDING).all()
            for ghost in pending_ghosts:
                if ghost.symbol in alpaca_positions:
                    # Position reappeared — false alarm, revert to ACTIVE
                    ghost.status = "ACTIVE"
                    ghost.ghost_detection_count = 0
                    ghost.ghost_first_detected_at = None
                    ghost.ghost_last_detected_at = None
                    logger.info(
                        "GHOST RESOLVED: Trade#%d %s reappeared in Alpaca — reverting to ACTIVE",
                        ghost.id, ghost.symbol,
                    )
                    result["ghost_positions"].append({
                        "trade_id": ghost.id, "symbol": ghost.symbol, "action": "reverted",
                    })
                    db_session.add(AuditLog(
                        action="RECONCILE_GHOST_REVERTED",
                        details={
                            "trade_id": ghost.id, "symbol": ghost.symbol,
                            "reason": "Position reappeared in Alpaca — false alarm",
                            "reverted_at": now.isoformat(),
                        },
                        timestamp=now,
                    ))
                    continue

                first_seen = ghost.ghost_first_detected_at or now
                elapsed = now - first_seen
                count = ghost.ghost_detection_count or 0

                if elapsed > timedelta(hours=GHOST_MAX_PENDING_HOURS):
                    # Too long in PENDING — escalate to UNRESOLVED for human review
                    ghost.status = RECONCILE_GHOST_UNRESOLVED
                    ghost.exit_reason = "reconcile_ghost_pending_timeout"
                    logger.error(
                        "GHOST UNRESOLVED: Trade#%d %s has been PENDING for %.1fh — "
                        "manual review required (admin_force_close.py)",
                        ghost.id, ghost.symbol, elapsed.total_seconds() / 3600,
                    )
                    db_session.add(AuditLog(
                        action="RECONCILE_GHOST_UNRESOLVED",
                        details={
                            "trade_id": ghost.id, "symbol": ghost.symbol,
                            "elapsed_hours": elapsed.total_seconds() / 3600,
                            "ghost_detection_count": count,
                            "escalated_at": now.isoformat(),
                        },
                        timestamp=now,
                    ))
                    continue

                if count < GHOST_MIN_DETECTIONS or elapsed < GHOST_MIN_ELAPSED:
                    logger.info(
                        "GHOST PENDING: Trade#%d %s — waiting for confirmation "
                        "(detections=%d/%d, elapsed=%.1fmin/%.0fmin)",
                        ghost.id, ghost.symbol, count, GHOST_MIN_DETECTIONS,
                        elapsed.total_seconds() / 60, GHOST_MIN_ELAPSED.total_seconds() / 60,
                    )
                    continue

                # Confirmed ghost — close it
                _ghost_short = (getattr(ghost, "direction", "BUY") or "BUY") == "SELL_SHORT"
                _entry = float(ghost.entry_price or 0)
                exit_price, exit_order_id = _lookup_close_fill(
                    alpaca, ghost.symbol, ghost.quantity, _ghost_short, entry_price=_entry,
                )
                _exit_source = "alpaca_fill" if exit_price is not None else None

                if exit_price is None:
                    try:
                        bars = alpaca.get_bars(ghost.symbol, timeframe="1D", limit=2)
                        if bars is not None and not bars.empty:
                            _fb_price = float(bars["close"].iloc[-1])
                            _fb_round = (_fb_price == int(_fb_price) and _fb_price > 0)
                            _fb_sanity = (_entry <= 0 or abs(_fb_price - _entry) / _entry <= 0.30)
                            if _fb_round:
                                logger.warning(
                                    "Trade#%d %s: fallback bar close $%.4f is a round number — skipping",
                                    ghost.id, ghost.symbol, _fb_price,
                                )
                            elif not _fb_sanity:
                                logger.warning(
                                    "Trade#%d %s: fallback bar close $%.4f is >30%% from entry $%.4f — skipping",
                                    ghost.id, ghost.symbol, _fb_price, _entry,
                                )
                            else:
                                exit_price = _fb_price
                                exit_order_id = "fallback_last_close"
                                _exit_source = "fallback_bar_close"
                                logger.warning(
                                    "Trade#%d %s: no Alpaca fill — using last bar close $%.4f",
                                    ghost.id, ghost.symbol, exit_price,
                                )
                    except Exception as _fb_exc:
                        logger.debug("Bar fallback failed for %s: %s", ghost.symbol, _fb_exc)

                ghost.status = "CLOSED"
                ghost.exit_reason = "reconcile_ghost_expired"
                ghost.closed_at = now
                if exit_price is not None:
                    write_exit_price(ghost, exit_price, source=_exit_source,
                                     order_id=exit_order_id, written_by="reconciler")
                    _qty = ghost.quantity or 0
                    if _ghost_short:
                        ghost.pnl = round((_entry - exit_price) * _qty, 2)
                    else:
                        ghost.pnl = round((exit_price - _entry) * _qty, 2)
                    logger.info(
                        "Closing confirmed ghost Trade#%d %s — exit=$%.4f pnl=$%.2f (source=%s order=%s)",
                        ghost.id, ghost.symbol, exit_price, ghost.pnl or 0, _exit_source, exit_order_id or "?",
                    )
                else:
                    logger.warning(
                        "Closing ghost Trade#%d %s with no exit price — "
                        "check Alpaca manually and backfill exit_price in DB.",
                        ghost.id, ghost.symbol,
                    )
                db_session.add(AuditLog(
                    action="RECONCILE_GHOST_CLOSED",
                    details={
                        "trade_id": ghost.id, "symbol": ghost.symbol,
                        "exit_price": exit_price, "exit_source": _exit_source,
                        "ghost_detection_count": count, "elapsed_minutes": elapsed.total_seconds() / 60,
                        "closed_at": now.isoformat(),
                    },
                    timestamp=now,
                ))
        except Exception as exc:
            logger.error("Ghost position check failed: %s", exc)

    # ── 2. Pending fills — resolve via Alpaca order status ────────────────────
    # PENDING_FILL trades have an alpaca_order_id; check whether they filled,
    # were cancelled, or are still open (leave those — Trader will poll them).
    # Also check CANCELLED trades with real Alpaca order IDs — the order may have
    # filled before the DB was updated (restart race condition).
    try:
        cancelled_with_id = [
            t for t in db_session.query(Trade).filter_by(status="CANCELLED").all()
            if t.alpaca_order_id and not t.alpaca_order_id.startswith("order-")
        ]
        for trade in cancelled_with_id:
            try:
                order_status = alpaca.get_order_status(trade.alpaca_order_id)
            except Exception:
                continue
            if order_status is None:
                continue
            status_str = str(order_status.get("status", "")).lower()
            filled_qty = int(order_status.get("filled_qty") or 0)
            filled_price = order_status.get("filled_avg_price")
            if status_str in ("filled", "partially_filled") and filled_qty > 0 and filled_price:
                # The order actually filled — this is a cancel/fill mismatch
                # Mark it as superseded; the position is tracked elsewhere (RECONCILED row)
                logger.warning(
                    "CANCEL/FILL MISMATCH: Trade#%d %s marked CANCELLED but Alpaca shows FILLED "
                    "@ $%s — marking RECONCILE_SUPERSEDED",
                    trade.id, trade.symbol, filled_price,
                )
                trade.status = "RECONCILE_SUPERSEDED"
                trade.status_reason = "Order filled on Alpaca but DB marked CANCELLED; position tracked separately"
                db_session.add(AuditLog(
                    action="RECONCILE_CANCEL_FILL_MISMATCH",
                    details={
                        "trade_id": trade.id, "symbol": trade.symbol,
                        "alpaca_order_id": trade.alpaca_order_id,
                        "filled_price": float(filled_price), "filled_qty": filled_qty,
                        "reason": "CANCELLED in DB but FILLED on Alpaca — restart race condition",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
        pending_trades = db_session.query(Trade).filter_by(status="PENDING_FILL").all()
        for trade in pending_trades:
            if not trade.alpaca_order_id:
                # No order ID recorded — order placement may have failed; cancel it
                logger.warning(
                    "PENDING_FILL Trade#%d %s has no alpaca_order_id — marking CANCELLED",
                    trade.id, trade.symbol,
                )
                trade.status = "CANCELLED"
                continue

            try:
                order_status = alpaca.get_order_status(trade.alpaca_order_id)
            except Exception as exc:
                logger.warning("Could not fetch order status for Trade#%d %s: %s",
                               trade.id, trade.symbol, exc)
                continue

            if order_status is None:
                continue

            status_str = str(order_status.get("status", "")).lower()
            filled_qty = int(order_status.get("filled_qty") or 0)
            filled_price = order_status.get("filled_avg_price")

            if status_str in ("filled", "partially_filled") and filled_qty > 0 and filled_price:
                filled_price = float(filled_price)
                intended_price = float(trade.entry_price or filled_price)
                trade.status = "ACTIVE"
                trade.entry_price = filled_price
                trade.quantity = filled_qty
                trade.highest_price = filled_price
                # Record the fill in the Order table
                db_order = Order(
                    trade_id=trade.id,
                    order_type="ENTRY",
                    order_id=trade.alpaca_order_id,
                    status="FILLED",
                    filled_price=filled_price,
                    filled_qty=filled_qty,
                    intended_price=intended_price,
                    slippage_bps=(
                        round(
                            (intended_price - filled_price) / intended_price * 10000, 2
                        ) if (trade.direction or "BUY") == "SELL_SHORT"
                        else round(
                            (filled_price - intended_price) / intended_price * 10000, 2
                        )
                    ) if intended_price > 0 else 0.0,
                )
                db_session.add(db_order)
                logger.info(
                    "PENDING_FILL resolved: Trade#%d %s filled x%d @ $%.4f",
                    trade.id, trade.symbol, filled_qty, filled_price,
                )
                result["pending_fills_resolved"].append({
                    "trade_id": trade.id, "symbol": trade.symbol,
                    "filled_qty": filled_qty, "filled_price": filled_price,
                })
                db_session.add(AuditLog(
                    action="RECONCILE_PENDING_FILL_RESOLVED",
                    details={
                        "trade_id": trade.id, "symbol": trade.symbol,
                        "filled_qty": filled_qty, "filled_price": filled_price,
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))

            elif status_str in ("canceled", "expired", "rejected"):
                trade.status = "CANCELLED"
                logger.info(
                    "PENDING_FILL Trade#%d %s order %s — marking CANCELLED",
                    trade.id, trade.symbol, status_str,
                )

            # else: still open — leave as PENDING_FILL; Trader will resume polling

    except Exception as exc:
        logger.error("Pending fill resolution failed: %s", exc)

    # ── 3. Untracked Alpaca positions ─────────────────────────────────────────
    # In Alpaca but no ACTIVE/PENDING_FILL DB record.
    # Fix 3: Before creating a synthetic record, check if a recent trade already
    # exists for this symbol (within 7 days). If so, reuse or skip — prevents
    # duplicate records when the same position gets reconciled across multiple restarts.
    try:
        live_statuses = ("ACTIVE", "PENDING_FILL")
        tracked_symbols = {
            t.symbol for t in db_session.query(Trade)
            .filter(Trade.status.in_(live_statuses)).all()
        }
        lookback_cutoff = datetime.utcnow() - timedelta(days=7)

        for symbol, pos in alpaca_positions.items():
            if symbol in tracked_symbols:
                continue

            qty = int(float(pos.get("qty") or pos.get("quantity") or 0))
            avg = float(pos.get("avg_entry_price") or pos.get("avg_price") or 0)

            # Fix 3: Check for a recent trade record for this symbol.
            # If one exists within the last 7 days that is CLOSED/CANCELLED/RECONCILE_GHOST,
            # it means the position was recently tracked and then closed — the closure may
            # have happened externally (stop hit, EOD cancel) without the DB being updated.
            # Reactivate the most recent record instead of creating a new duplicate.
            recent_trade = (
                db_session.query(Trade)
                .filter(
                    Trade.symbol == symbol,
                    Trade.created_at >= lookback_cutoff,
                    Trade.status.in_(("CLOSED", "CANCELLED", "RECONCILE_GHOST",
                                      RECONCILE_GHOST_PENDING, RECONCILE_GHOST_UNRESOLVED)),
                )
                .order_by(Trade.id.desc())
                .first()
            )

            _alpaca_dir = "SELL_SHORT" if qty < 0 else "BUY"
            _price_match = recent_trade and abs(float(recent_trade.entry_price or 0) - avg) / max(avg, 1) < 0.05
            _dir_match = (getattr(recent_trade, "direction", "BUY") or "BUY") == _alpaca_dir if recent_trade else False
            if _price_match and _dir_match:
                # ── Rule A: reconciler-written exit → always reactivate ──────
                # If the previous close was a reconciler estimate (ghost, superseded,
                # etc.) and the position is back in Alpaca, the reconciler was wrong.
                # Always reactivate — never cap this at UNTRACKED_REACTIVATE_AFTER.
                _prev_source = getattr(recent_trade, "exit_price_source", None)
                _prev_reason = recent_trade.exit_reason or ""
                _is_reconciler_exit = _prev_reason in RECONCILER_EXIT_REASONS

                # ── Rule B: legitimate exit → skip up to N runs, then force new ─
                # If exit was a real broker fill and we've seen the position N times,
                # stop reactivating — the user probably opened a new position in the
                # same symbol and we don't want to corrupt the old record.
                if not _is_reconciler_exit and recent_trade.exit_price is not None:
                    _exit = float(recent_trade.exit_price)
                    _entry_chk = float(recent_trade.entry_price or avg)
                    _exit_diff = abs(_exit - _entry_chk) / max(_entry_chk, 1)
                    if _exit_diff <= 0.30 and _prev_source in _LEGITIMATE_EXIT_SOURCES:
                        _skip_count = (recent_trade.untracked_detection_count or 0) + 1
                        if _skip_count <= UNTRACKED_REACTIVATE_AFTER:
                            # Only write the counter when we're actually skipping (not falling through)
                            recent_trade.untracked_detection_count = _skip_count
                            recent_trade.untracked_last_detected_at = datetime.utcnow()
                            logger.info(
                                "UNTRACKED POSITION: %s x%d @ $%.2f — Trade#%d has legitimate "
                                "exit_price=$%.4f (source=%s, skip %d/%d). Alpaca position settling.",
                                symbol, qty, avg, recent_trade.id, _exit, _prev_source,
                                _skip_count, UNTRACKED_REACTIVATE_AFTER,
                            )
                            continue
                        else:
                            logger.warning(
                                "UNTRACKED POSITION: %s x%d @ $%.2f — Trade#%d seen %d times "
                                "after legitimate close — treating as genuinely new position.",
                                symbol, qty, avg, recent_trade.id, _skip_count,
                            )
                            # Fall through to create a new RECONCILED record; don't pollute old trade
                            _price_match = False

                # ── Rule C: corrupt exit_price (>30% diff) → reactivate ──────
                if _price_match and recent_trade.exit_price is not None:
                    _exit = float(recent_trade.exit_price)
                    _entry_chk = float(recent_trade.entry_price or avg)
                    _exit_diff = abs(_exit - _entry_chk) / max(_entry_chk, 1)
                    if _exit_diff > 0.30:
                        logger.warning(
                            "UNTRACKED POSITION: %s x%d @ $%.2f — Trade#%d exit_price=$%.4f "
                            "is >30%% from entry=$%.4f — treating as corrupt, reactivating.",
                            symbol, qty, avg, recent_trade.id, _exit, _entry_chk,
                        )
                        # Will reactivate below

            if _price_match and _dir_match:
                logger.warning(
                    "UNTRACKED POSITION: %s x%d @ $%.2f — reactivating Trade#%d (was %s)",
                    symbol, qty, avg, recent_trade.id, recent_trade.status,
                )
                from app.database.models import recompute_partial_pnl
                _entry = float(recent_trade.entry_price or avg)
                _dir = getattr(recent_trade, "direction", "BUY") or "BUY"
                _partial = recompute_partial_pnl(db_session, recent_trade.id, _entry, _dir)
                recent_trade.status = "ACTIVE"
                recent_trade.quantity = abs(qty)
                recent_trade.exit_price = None
                recent_trade.exit_price_source = None
                recent_trade.exit_order_id = None
                recent_trade.exit_price_written_at = None
                recent_trade.exit_price_written_by = None
                recent_trade.pnl = _partial or None
                recent_trade.closed_at = None
                recent_trade.exit_reason = None
                recent_trade.highest_price = avg
                result["untracked_positions"].append({
                    "symbol": symbol, "qty": qty, "avg_price": avg,
                    "action": "reactivated", "trade_id": recent_trade.id,
                })
                db_session.add(AuditLog(
                    action="RECONCILE_REACTIVATED_TRADE",
                    details={
                        "trade_id": recent_trade.id, "symbol": symbol,
                        "qty": qty, "avg_price": avg,
                        "rule": "A" if _is_reconciler_exit else "C",
                        "reason": "Reactivated existing trade instead of creating duplicate",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
            else:
                # Genuinely new/untracked — create a fresh synthetic record
                logger.warning(
                    "UNTRACKED POSITION: %s x%d @ $%.2f — creating RECONCILED trade",
                    symbol, qty, avg,
                )
                result["untracked_positions"].append({
                    "symbol": symbol, "qty": qty, "avg_price": avg, "action": "created",
                })
                _syn_dir = "SELL_SHORT" if qty < 0 else "BUY"
                _syn_short = _syn_dir == "SELL_SHORT"
                # Short stop is above entry; short target is below entry
                stop_price = round(avg * (1.02 if _syn_short else 0.98), 2) if avg > 0 else None
                target_price = round(avg * (0.94 if _syn_short else 1.06), 2) if avg > 0 else None
                placeholder = Trade(
                    symbol=symbol, direction=_syn_dir, entry_price=avg,
                    quantity=abs(qty), status="ACTIVE", signal_type="RECONCILED",
                    trade_type="swing", stop_price=stop_price, target_price=target_price,
                    highest_price=avg, bars_held=0,
                    created_at=datetime.utcnow(),
                )
                db_session.add(placeholder)
                db_session.add(AuditLog(
                    action="RECONCILE_UNTRACKED_POSITION",
                    details={
                        "symbol": symbol, "qty": qty, "avg_price": avg,
                        "reason": "Alpaca position with no DB Trade — likely filled during restart",
                        "detected_at": datetime.utcnow().isoformat(),
                    },
                    timestamp=datetime.utcnow(),
                ))
    except Exception as exc:
        logger.error("Untracked position check failed: %s", exc)

    # ── 4. Orphaned open Alpaca orders ────────────────────────────────────────
    # Open orders in Alpaca with no DB record at all — log only (Trader will
    # pick them up via _pending_limit_orders once restarted, or they'll expire).
    try:
        open_alpaca_orders = _get_open_alpaca_orders(alpaca)
        from app.database.models import Order as OrderModel
        db_order_ids = {
            row.order_id for row in db_session.query(OrderModel.order_id).all()
            if row.order_id
        }
        pending_fill_order_ids = {
            t.alpaca_order_id for t in db_session.query(Trade)
            .filter(Trade.status == "PENDING_FILL").all()
            if t.alpaca_order_id
        }
        for ao in open_alpaca_orders:
            oid = ao["order_id"]
            if oid not in db_order_ids and oid not in pending_fill_order_ids:
                logger.warning(
                    "ORPHANED ORDER: Alpaca order %s (%s %s x%s) not in DB",
                    oid, ao["side"], ao["symbol"], ao["qty"],
                )
                result["orphaned_orders"].append(ao)
                db_session.add(AuditLog(
                    action="RECONCILE_ORPHANED_ORDER",
                    details={**ao, "reason": "Open order in Alpaca but not in DB on startup",
                             "detected_at": datetime.utcnow().isoformat()},
                    timestamp=datetime.utcnow(),
                ))
    except Exception as exc:
        logger.error("Orphaned order check failed: %s", exc)

    # ── 5. Target/stop sanity audit (read-only) ───────────────────────────────
    # Surface (don't auto-fix) any ACTIVE trades whose target/stop is corrupt.
    # Prevents recurrences of the runaway-ATR EXTEND_TARGET bug from going
    # unnoticed in production.
    try:
        result["insane_target_stops"] = audit_active_target_stops(db_session)
    except Exception as exc:
        logger.error("Target/stop audit failed: %s", exc)
        result["insane_target_stops"] = []

    try:
        db_session.commit()
    except Exception as exc:
        logger.error("Reconciler commit failed: %s", exc)
        db_session.rollback()

    n_ghosts = len(result["ghost_positions"])
    n_resolved = len(result["pending_fills_resolved"])
    n_untracked = len(result["untracked_positions"])
    n_orphans = len(result["orphaned_orders"])

    if any([n_ghosts, n_resolved, n_untracked, n_orphans]):
        logger.warning(
            "Startup reconciliation: %d ghost(s), %d pending fills resolved, "
            "%d untracked position(s), %d orphaned order(s)",
            n_ghosts, n_resolved, n_untracked, n_orphans,
        )
    else:
        logger.info("Startup reconciliation: clean — no issues found")

    return result


def _lookup_close_fill(
    alpaca, symbol: str, qty, is_short: bool = False,
    max_age_days: int = 30, entry_price: float = 0.0,
) -> tuple:
    """Return (filled_avg_price, order_id) for the most recent closing fill for symbol.

    Longs close via SELL; shorts close via BUY (cover).
    Guards against stale/wrong matches:
    - date filter: only orders within max_age_days
    - qty tolerance: tightened to 5% (was 20%) to avoid cross-lot matches
    - price sanity: rejects fills >30% away from entry_price (if provided)
    - newest-first: DESC direction ensures most recent match wins
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus, QueryOrderDirection
        from datetime import timezone as _tz
        cutoff = datetime.utcnow().replace(tzinfo=_tz.utc) - timedelta(days=max_age_days)
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=100, after=cutoff,
            direction=QueryOrderDirection.DESC,
        )
        orders = alpaca.trading_client.get_orders(req)
        target_qty = abs(int(qty or 0))
        close_side = "BUY" if is_short else "SELL"
        for o in orders:
            if o.symbol != symbol:
                continue
            if close_side not in str(o.side).upper():
                continue
            if str(o.status).upper() != "FILLED":
                continue
            filled_qty = int(o.filled_qty or 0)
            if target_qty > 0 and abs(filled_qty - target_qty) > max(1, target_qty * 0.05):
                continue
            if not o.filled_avg_price:
                continue
            fill_price = float(o.filled_avg_price)
            # Reject fills that differ from entry by >30% — almost certainly a wrong match
            if entry_price > 0 and abs(fill_price - entry_price) / entry_price > 0.30:
                logger.warning(
                    "_lookup_close_fill: rejecting fill $%.4f for %s (entry $%.4f, diff %.1f%%)",
                    fill_price, symbol, entry_price, abs(fill_price - entry_price) / entry_price * 100,
                )
                continue
            # Reject suspiciously round fills (e.g. exactly $110.00) — real market
            # fills always have cents. A zero-cent price is a sentinel/test value.
            if fill_price == int(fill_price) and fill_price > 0:
                logger.warning(
                    "_lookup_close_fill: rejecting round-number fill $%.2f for %s — likely stale test/sentinel",
                    fill_price, symbol,
                )
                continue
            return fill_price, str(o.id)
    except Exception as exc:
        logger.debug("_lookup_close_fill failed for %s: %s", symbol, exc)
    return None, None


# Backward-compat alias
_lookup_sell_fill = _lookup_close_fill


def _get_open_alpaca_orders(alpaca) -> List[Dict[str, Any]]:
    """Return list of open (pending/new) orders from Alpaca."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = alpaca.trading_client.get_orders(req)
        return [
            {
                "order_id": str(o.id),
                "symbol":   o.symbol,
                "qty":      str(o.qty),
                "side":     str(o.side),
                "status":   str(o.status),
            }
            for o in orders
        ]
    except Exception as exc:
        logger.error("Could not fetch open Alpaca orders: %s", exc)
        return []
