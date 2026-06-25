"""P1-1 — explicit cash / T-bill sleeve.

Idle book capital (~50% of NAV, since trend is 50%) sat as ZERO-yield cash. This sleeve
parks the idle remainder in a T-bill ETF (SGOV/BIL) so it earns the risk-free rate, with a
defined liquidity policy: keep a cash BUFFER (pm.cash_buffer_pct of NAV) as settled cash for
other sleeves' intraday needs, deploy the rest into T-bills, and replenish the buffer by
selling T-bills whenever settled cash dips below it (the risk-off path).

Mirrors trend_sleeve.py: config-gated (pm.cash_enabled / pm.cash_shadow), kill-switch honored,
fail-closed, per-order DB commit, decision_audit trail, positions tagged selector/trade_type
='cash' so the Trader's exit loop and the startup reconciler leave them alone (T-bills must
never be stop-lossed or adopted as synthetic swing trades).

CAP INTERACTION: T-bills are cash-equivalents, NOT risk. They are deliberately EXCLUDED from
the 80% GROSS_EXPOSURE_CAP (see trend_sleeve / risk_manager, which skip CASH_ETFS when summing
risk gross) so parking idle cash never starves trend/PEAD. The sleeve sizes off ACTUAL settled
cash (can't buy more than it has), so it cannot over-deploy.

Runs weekly AFTER the trend rebalance (orchestrator 09:50 ET) so trend's deployed gross — and
thus the idle remainder — is already settled.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.live_trading.instrument_master import CASH_EQUIVALENT_ETFS

log = logging.getLogger(__name__)

# Known ultra-short Treasury / T-bill ETFs. The risk-gross exclusion (trend_sleeve +
# risk_manager) keys off this set, so a cash position never counts against the 80% RISK cap.
# Keep pm.cash_universe a subset of these.
# SINGLE SOURCE OF TRUTH: imported from instrument_master so the trading universe and the
# instrument master (which the whole-book gate trusts) can never drift (Alpha-v10 H10). A drift
# would fail-close the gate in ENFORCE mode on a legal cash config — see instrument_master.
CASH_ETFS = frozenset(CASH_EQUIVALENT_ETFS)

_MIN_NOTIONAL = 100.0   # skip dust rebalances below this $ (one T-bill ETF share ~$50-100)


def _truthy(val: Any) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def cash_universe(db) -> List[str]:
    """Configured cash-sleeve T-bill ETFs (first = primary buy target). Symbols are filtered
    to CASH_ETFS: a ticker outside that set would be TRADED but NOT excluded from the risk
    gross cap (the exclusion keys off CASH_ETFS), so it's dropped + logged to avoid the
    inconsistent half-excluded state. Falls back to SGOV."""
    from app.database.agent_config import get_agent_config
    raw = [s.strip().upper() for s in str(get_agent_config(db, "pm.cash_universe")).split(",")
           if s.strip()]
    syms = [s for s in raw if s in CASH_ETFS]
    dropped = [s for s in raw if s not in CASH_ETFS]
    if dropped:
        log.warning("cash: dropping non-T-bill symbol(s) %s from cash_universe (not in "
                    "CASH_ETFS — would not be excluded from the risk gross cap)", dropped)
    return syms or ["SGOV"]


class _PositionsUnavailable(RuntimeError):
    """Raised when current cash positions cannot be determined (read error). The caller must
    fail CLOSED — an empty {} here is indistinguishable from 'truly flat' and would make the
    rebalance re-buy an already-held T-bill (target_shares = 0 + qty) or skip a risk-off sell."""


def _current_cash_positions(db, alpaca) -> Dict[str, int]:
    """Symbols tagged selector='cash' in the DB, sized from ACTUAL Alpaca shares.

    Raises _PositionsUnavailable on an indeterminate read (DB or broker error) so the caller
    fail-closes; returns {} ONLY when there are genuinely no cash positions."""
    from app.database.models import Trade
    try:
        rows = (db.query(Trade)
                .filter(Trade.selector == "cash",
                        Trade.status.in_(["ACTIVE", "PENDING_FILL"])).all())
        cash_syms = {r.symbol for r in rows}
    except Exception as exc:
        raise _PositionsUnavailable(f"DB position query failed: {exc}") from exc
    if not cash_syms:
        return {}
    try:
        positions = alpaca.get_positions() or []
    except Exception as exc:
        raise _PositionsUnavailable(f"get_positions failed: {exc}") from exc
    return {p["symbol"]: int(p.get("qty") or 0)
            for p in positions if p.get("symbol") in cash_syms}


def _sync_cash_trade(db, symbol: str, target_shares: int, price: float,
                     order_id: Optional[str]) -> None:
    """One Trade row per cash symbol reflecting the post-rebalance position (attribution
    + dashboard state; Alpaca is the source of truth for share count). target_shares==0 closes."""
    from app.database.models import Trade
    existing = (db.query(Trade)
                .filter(Trade.symbol == symbol, Trade.selector == "cash",
                        Trade.status.in_(["ACTIVE", "PENDING_FILL"]))
                .order_by(Trade.id.desc()).first())
    if target_shares <= 0:
        if existing:
            existing.status = "CLOSED"
            existing.closed_at = datetime.utcnow()
            existing.exit_price = price or existing.exit_price
            existing.exit_reason = "cash_rebalance_exit"
        return
    if existing:
        existing.quantity = int(target_shares)
        if order_id:
            existing.alpaca_order_id = order_id
    else:
        db.add(Trade(
            symbol=symbol, direction="BUY", entry_price=float(price or 0.0),
            quantity=int(target_shares), status="PENDING_FILL", signal_type="CASH_TBILL",
            trade_type="cash", selector="cash", alpaca_order_id=order_id,
            created_at=datetime.utcnow(),
        ))


def _cash_fallback_prices(alpaca, universe: List[str]):
    """Daily-close panel for the cash universe as a fallback for live quotes. NOT
    trend_sleeve._fetch_prices — that fails closed unless SPY (its CORE_SYMBOL) is present,
    and the cash universe never contains SPY, so it would ALWAYS return None and silently
    strip the fallback. T-bills are stable so a short history suffices. None on total failure."""
    import pandas as pd
    try:
        raw = alpaca.get_bars_batch(universe, "1D", 30)
    except Exception as exc:
        log.warning("cash: get_bars_batch failed: %s", exc)
        return None
    series = {}
    for sym, df in (raw or {}).items():
        if df is None or "close" not in getattr(df, "columns", []):
            continue
        closes = df["close"].dropna()
        if len(closes) >= 1:
            series[sym] = closes
    return pd.DataFrame(series).sort_index() if series else None


def _audit(symbol: str, side: str, *, price: float, final_decision: str,
           block_reason: Optional[str] = None) -> None:
    try:
        from app.database.decision_audit import write_decision
        write_decision(symbol=symbol, strategy="cash", final_decision=final_decision,
                       direction=("BUY" if side == "buy" else "SELL"),
                       price_at_decision=price, block_reason=block_reason)
    except Exception:
        log.debug("cash: decision_audit write failed (swallowed)", exc_info=True)


def run_cash_rebalance(db=None, *, force: bool = False) -> Dict[str, Any]:
    """Park idle settled cash (beyond the buffer) in T-bills; replenish the buffer by selling
    T-bills when cash dips below it. Never raises. Returns a summary dict for logging/tests."""
    summary: Dict[str, Any] = {"status": "ok", "mode": None, "action": None,
                               "approved": [], "blocked": [], "nav": None}
    own_db = db is None
    if own_db:
        from app.database.session import get_session
        db = get_session()
    try:
        from app.database.agent_config import get_agent_config
        enabled = _truthy(get_agent_config(db, "pm.cash_enabled"))
        shadow = _truthy(get_agent_config(db, "pm.cash_shadow"))
        if not enabled and not force:
            log.info("cash: DORMANT (pm.cash_enabled=false) — no rebalance")
            summary["status"] = "dormant"
            return summary
        summary["mode"] = "shadow" if shadow else "live"

        from app.live_trading.kill_switch import kill_switch
        if kill_switch.is_active:
            log.warning("cash: kill switch ACTIVE — no rebalance")
            summary["status"] = "blocked"
            summary["block_reason"] = "kill_switch"
            return summary

        from app.integrations import get_alpaca_client
        alpaca = get_alpaca_client()
        try:
            acct = alpaca.get_account()
            nav = float(acct["equity"])
            cash_on_hand = float(acct.get("cash") or 0.0)
            buying_power = float(acct.get("buying_power") or cash_on_hand)
        except Exception as exc:
            log.warning("cash: account fetch failed — fail-closed: %s", exc)
            summary["status"] = "failed"
            summary["block_reason"] = "account_unavailable"
            return summary
        summary["nav"] = nav
        if nav <= 0:
            summary["status"] = "failed"
            summary["block_reason"] = "nav_unavailable"
            return summary

        # ── H1 reconciliation-before-trade (shadow-first; also closes the cash-bypass gap) ──
        # The cash sleeve trades T-bills directly; reconcile the whole DB book vs broker reality
        # before placing, same as trend. Mode read first -> enforce wiring failure fails CLOSED.
        try:
            recon_mode = str(get_agent_config(db, "pm.reconciliation_mode") or "shadow").strip().lower()
        except Exception:
            recon_mode = "shadow"
        if recon_mode != "off":
            blocked_by_recon = False
            try:
                from app.live_trading import reconciliation as _recon
                from app.notifications import notifier as _rnotifier
                raw_positions = alpaca.get_positions() or []
                recon_result = _recon.shadow_reconcile_before_trade(
                    db, raw_positions, nav=nav, mode=recon_mode, label="cash", notifier=_rnotifier)
                summary["recon_mode"] = recon_mode
                summary["recon_status"] = recon_result.status
                blocked_by_recon = (recon_mode == "enforce" and not recon_result.ok_to_trade)
            except Exception:
                log.debug("cash: reconciliation wiring failed", exc_info=True)
                blocked_by_recon = (recon_mode == "enforce")   # fail-CLOSED in enforce; inert in shadow
            if blocked_by_recon:
                log.warning("cash: reconciliation (mode=%s) -> HOLD rebalance", recon_mode)
                summary["status"] = "blocked"
                summary["block_reason"] = "reconciliation"
                return summary

        buffer_pct = float(get_agent_config(db, "pm.cash_buffer_pct"))
        buffer = max(0.0, buffer_pct) * nav
        # Size off min(settled cash, buying power) so we never over-deploy on the trend
        # SETTLEMENT RACE: cash runs 09:50, 5 min after trend's 09:45 market orders —
        # buying_power drops the instant trend places them (even before `cash` settles), so
        # min() stops the cash sleeve from parking money trend just committed. >0 -> buy
        # T-bills; <0 -> sell to refill the buffer.
        deployable = min(cash_on_hand, buying_power) - buffer
        summary["cash_on_hand"] = cash_on_hand
        summary["buffer"] = buffer
        summary["deployable"] = deployable

        universe = cash_universe(db)
        primary = universe[0]
        from app.live_trading.trend_sleeve import _live_prices
        prices_df = _cash_fallback_prices(alpaca, universe)   # NOT _fetch_prices (SPY-gated)
        live = _live_prices(alpaca, universe, prices_df)
        try:
            current = _current_cash_positions(db, alpaca)
        except _PositionsUnavailable as exc:
            # Fail CLOSED: without a confirmed position map we could re-buy a held T-bill or
            # under-replenish the buffer. Skip this cycle; positions stay as-is, retry next run.
            log.warning("cash: positions unavailable — HOLD rebalance (%s)", exc)
            summary["status"] = "failed"
            summary["block_reason"] = "positions_unavailable"
            return summary

        intents: List[Dict[str, Any]] = []
        if deployable > _MIN_NOTIONAL:
            px = live.get(primary)
            if not px or px <= 0:
                log.warning("cash: no price for primary %s — fail-closed", primary)
                summary["status"] = "failed"
                summary["block_reason"] = "price_unavailable"
                return summary
            qty = int(deployable // px)
            if qty >= 1:
                cur = current.get(primary, 0)
                intents.append({"symbol": primary, "side": "buy", "qty": qty,
                                "target_shares": cur + qty})
                summary["action"] = "deploy"
        elif deployable < -_MIN_NOTIONAL:
            # Risk-off: settled cash below buffer -> sell T-bills to raise (buffer - cash).
            # NOTE: T-bill sales settle T+1, so this restores the buffer for the NEXT session,
            # not same-day; the buffer itself is the same-day cushion (see module docstring).
            need = -deployable
            summary["action"] = "raise"
            # Sell primary first, then any other held cash symbols.
            for sym in [primary] + [s for s in current if s != primary]:
                if need <= _MIN_NOTIONAL:
                    break
                held = current.get(sym, 0)
                if held <= 0:
                    continue
                px = live.get(sym)
                if not px or px <= 0:
                    # SAFETY-CRITICAL direction: a missing price here would silently leave the
                    # buffer un-replenished. Escalate (log + audit) instead of a silent skip.
                    log.error("cash: RISK-OFF cannot price held %s — buffer under-replenished "
                              "by ~$%.0f", sym, need)
                    _audit(sym, "sell", price=0.0, final_decision="block",
                           block_reason="price_unavailable")
                    summary["blocked"].append({"symbol": sym, "side": "sell",
                                               "block_reason": "price_unavailable"})
                    continue
                sell_qty = min(held, int(need // px) + 1)
                if sell_qty <= 0:
                    continue
                intents.append({"symbol": sym, "side": "sell", "qty": sell_qty,
                                "target_shares": held - sell_qty})
                need -= sell_qty * px
            if need > _MIN_NOTIONAL:
                summary["buffer_shortfall"] = round(need, 2)
        else:
            log.info("cash: within buffer band (deployable=$%.0f) — no action", deployable)
            return summary

        # ── Execute or shadow ──
        if shadow:
            for it in intents:
                _audit(it["symbol"], it["side"], price=live.get(it["symbol"], 0.0),
                       final_decision="enter", block_reason="shadow")
            summary["approved"] = intents
            log.info("cash SHADOW: would %s %d order(s) — none sent",
                     summary.get("action"), len(intents))
            return summary

        from app.live_trading.order_ids import idempotency_key
        for it in intents:
            sym, side, qty = it["symbol"], it["side"], it["qty"]
            price = live.get(sym, 0.0)
            try:
                order = alpaca.place_market_order(
                    sym, int(qty), side, client_order_id=idempotency_key("cash", sym, side=side))
                oid = order.get("order_id") if isinstance(order, dict) else None
            except Exception as exc:
                log.error("cash: order failed %s %s x%d: %s", side, sym, qty, exc)
                _audit(sym, side, price=price, final_decision="block", block_reason="order_error")
                summary["blocked"].append({**it, "block_reason": "order_error"})
                continue

            # Idempotent-reuse: a same-day re-run (force=True / operator re-trigger / retry) collided
            # with an already-placed order — the broker did NOT execute a SECOND trade. Do NOT record
            # the freshly-computed target_shares (that would book shares never traded). Re-derive the
            # ACTUAL held shares from the broker so the DB matches reality; on the risk-off SELL path
            # surface the under-replenishment so it's visible, never silently swallowed.
            if isinstance(order, dict) and order.get("idempotent_reuse"):
                log.info("cash: idempotent reuse %s %s — re-deriving actual shares (no new fill)",
                         side, sym)
                _actual = None
                try:
                    _pos = alpaca.get_position(sym)
                    if isinstance(_pos, dict):
                        _actual = abs(int(_pos.get("qty") or 0))
                except Exception:
                    _actual = None
                if _actual is not None:
                    try:
                        _sync_cash_trade(db, sym, _actual, price, oid)
                        db.commit()
                    except Exception:
                        db.rollback()
                if side == "sell":
                    # distinct key (buffer_shortfall is a float set on the risk-off sizing path)
                    summary.setdefault("reuse_unfilled_sells", []).append(sym)
                _audit(sym, side, price=price, final_decision="enter", block_reason="idempotent_reuse")
                summary["approved"].append(it)
                continue

            try:
                _sync_cash_trade(db, sym, it["target_shares"], price, oid)
                db.commit()
            except Exception as exc:
                log.error("cash: trade sync failed %s: %s", sym, exc)
                db.rollback()
            _audit(sym, side, price=price, final_decision="enter")
            summary["approved"].append(it)
        log.info("cash: %s %d T-bill order(s) placed", summary.get("action"),
                 len(summary["approved"]))

        try:
            from app.live_trading import cash_tracker
            tbill_mv = sum(current.get(s, 0) * live.get(s, 0.0) for s in set(current) | {primary})
            cash_tracker.record_daily(n_positions=len(current) or len(intents),
                                      tbill_deployed=tbill_mv, cash_buffer=cash_on_hand,
                                      extra={"action": summary.get("action"), "nav": nav})
        except Exception:
            log.debug("cash: tracker record failed (swallowed)", exc_info=True)

        return summary
    except Exception:
        log.exception("cash: run_cash_rebalance failed (swallowed)")
        summary["status"] = "failed"
        return summary
    finally:
        if own_db:
            try:
                db.close()
            except Exception:
                pass
