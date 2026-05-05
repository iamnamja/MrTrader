"""
Repository functions for the decision_audit table (Phase 61).

PM calls write_decision() at every enter/block choice so we can later
answer: did each gate (news, macro, correlation) block winners or losers?
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def write_decision(
    symbol: str,
    strategy: str,
    final_decision: str,
    *,
    model_score: Optional[float] = None,
    size_multiplier: float = 1.0,
    block_reason: Optional[str] = None,
    news_signal=None,           # NewsSignal dataclass or None
    macro_context=None,         # MacroContext dataclass or None
    top_features: Optional[dict] = None,
    price_at_decision: Optional[float] = None,
    direction: Optional[str] = None,   # 'BUY' | 'SELL'
) -> None:
    """
    Persist one PM decision row.  Fails silently — never blocks trading.

    final_decision: 'enter' | 'block' | 'size_down' | 'exit_review'
    """
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit
        from app.database.gate_categories import classify_gate

        row = DecisionAudit(
            id=str(uuid.uuid4()),
            decided_at=datetime.now(timezone.utc),
            symbol=symbol,
            strategy=strategy,
            model_score=model_score,
            final_decision=final_decision,
            size_multiplier=size_multiplier,
            block_reason=block_reason,
            gate_category=classify_gate(block_reason) if final_decision == "block" else None,
            price_at_decision=price_at_decision,
            direction=direction,
        )

        if news_signal is not None:
            row.news_action_policy = getattr(news_signal, "action_policy", None)
            row.news_direction_score = getattr(news_signal, "direction_score", None)
            row.news_materiality = getattr(news_signal, "materiality_score", None)
            row.news_sizing_multiplier = getattr(news_signal, "sizing_multiplier", None)
            row.news_rationale = getattr(news_signal, "rationale", None)

        if macro_context is not None:
            row.macro_risk_level = getattr(macro_context, "overall_risk", None)
            row.macro_sizing_factor = getattr(macro_context, "global_sizing_factor", None)

        if top_features:
            row.top_features = top_features

        with get_session() as db:
            db.add(row)
            db.commit()

    except Exception as exc:
        logger.debug("decision_audit write failed (non-fatal): %s", exc)


def write_scan_abstention(
    gate_type: str,
    gate_detail: str,
    *,
    proposal_log_batch_id: Optional[str] = None,
    spy_price: Optional[float] = None,
    spy_first_hour_range_pct: Optional[float] = None,
    strategy: str = "intraday",
) -> None:
    """Record a scan-level abstention (gate1a, gate1c) in scan_abstentions.

    Called by PM when an entire intraday scan is skipped due to a market gate.
    Fails silently — never blocks trading.
    """
    try:
        from app.database.session import get_session
        from app.database.models import ScanAbstention

        row = ScanAbstention(
            abstained_at=datetime.now(timezone.utc),
            strategy=strategy,
            gate_type=gate_type,
            gate_detail=gate_detail,
            proposal_log_batch_id=proposal_log_batch_id,
            spy_price_at_abstention=spy_price,
            spy_first_hour_range_pct=spy_first_hour_range_pct,
        )
        with get_session() as db:
            db.add(row)
            db.commit()
    except Exception as exc:
        logger.debug("scan_abstention write failed (non-fatal): %s", exc)


def persist_nis_macro_snapshot(macro_context) -> None:
    """Upsert today's NIS macro context to nis_macro_snapshots.

    Called by premarket routine after NIS macro loads.  Uses snapshot_date as
    the unique key so re-runs on the same day overwrite (UPSERT via delete+insert).
    Fails silently — never blocks trading.
    """
    try:
        from app.database.session import get_session
        from app.database.models import NisMacroSnapshot

        today = datetime.now(timezone.utc).date()
        as_of = getattr(macro_context, "as_of", datetime.now(timezone.utc))
        events = []
        for e in getattr(macro_context, "events_today", []):
            events.append({
                "event_type": getattr(e, "event_type", None),
                "event_time": str(getattr(e, "event_time", None)),
                "risk_level": getattr(e, "risk_level", None),
                "direction": getattr(e, "direction", None),
                "sizing_factor": getattr(e, "sizing_factor", None),
                "block_new_entries": getattr(e, "block_new_entries", False),
                "consensus_summary": getattr(e, "consensus_summary", None),
                "rationale": getattr(e, "rationale", None),
                "already_priced_in": getattr(e, "already_priced_in", False),
            })

        with get_session() as db:
            existing = db.query(NisMacroSnapshot).filter(
                NisMacroSnapshot.snapshot_date == today
            ).first()
            if existing:
                existing.as_of = as_of
                existing.overall_risk = macro_context.overall_risk
                existing.block_new_entries = macro_context.block_new_entries
                existing.global_sizing_factor = macro_context.global_sizing_factor
                existing.rationale = getattr(macro_context, "rationale", None)
                existing.events_json = events
            else:
                db.add(NisMacroSnapshot(
                    snapshot_date=today,
                    as_of=as_of,
                    overall_risk=macro_context.overall_risk,
                    block_new_entries=macro_context.block_new_entries,
                    global_sizing_factor=macro_context.global_sizing_factor,
                    rationale=getattr(macro_context, "rationale", None),
                    events_json=events,
                ))
            db.commit()
    except Exception as exc:
        logger.debug("nis_macro_snapshot write failed (non-fatal): %s", exc)


def backfill_gate_outcomes(lookback_days: int = 7) -> dict:
    """
    EOD job: fetch counterfactual price outcomes for calibratable blocked rows.

    For each decision_audit row where:
      - final_decision = 'block'
      - gate_category IN (alpha, quality, risk)
      - price_at_decision is set
      - outcome_1d_pct is null
      - decided_at is at least 4h ago (intraday) or 24h ago (swing)

    Fetches Alpaca historical bars and computes signed counterfactual P&L.
    Returns counts of rows attempted/updated/failed.
    """
    from app.database.session import get_session
    from app.database.models import DecisionAudit
    from app.database.gate_categories import CALIBRATABLE_CATEGORIES

    attempted = updated = failed = 0
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)

        with get_session() as db:
            rows = (
                db.query(DecisionAudit)
                .filter(
                    DecisionAudit.final_decision == "block",
                    DecisionAudit.gate_category.in_(list(CALIBRATABLE_CATEGORIES)),
                    DecisionAudit.price_at_decision.isnot(None),
                    DecisionAudit.outcome_1d_pct.is_(None),
                    DecisionAudit.decided_at >= cutoff,
                )
                .all()
            )

            for row in rows:
                # Require at least 4h for intraday, 24h for swing
                min_age = timedelta(hours=4) if row.strategy == "intraday" else timedelta(hours=24)
                decided = row.decided_at.replace(tzinfo=timezone.utc) if row.decided_at.tzinfo is None else row.decided_at
                if now - decided < min_age:
                    continue

                attempted += 1
                try:
                    price_4h, price_1d = _fetch_outcome_prices(row.symbol, decided)
                    base = row.price_at_decision
                    direction_mult = -1.0 if (row.direction or "BUY") == "SELL" else 1.0

                    if price_4h is not None and base and base > 0:
                        row.outcome_4h_pct = round((price_4h - base) / base * direction_mult * 100, 3)
                    if price_1d is not None and base and base > 0:
                        row.outcome_1d_pct = round((price_1d - base) / base * direction_mult * 100, 3)
                    row.outcome_fetched_at = now
                    updated += 1
                except Exception as exc:
                    logger.debug("outcome fetch failed for %s %s: %s", row.symbol, row.decided_at, exc)
                    failed += 1

            db.commit()

    except Exception as exc:
        logger.warning("backfill_gate_outcomes failed: %s", exc)

    return {"attempted": attempted, "updated": updated, "failed": failed}


def backfill_scan_abstention_outcomes(lookback_days: int = 7) -> dict:
    """
    EOD job: fetch SPY returns for scan_abstention rows.

    Looks for rows where spy_outcome_1d_pct is null and abstained_at is ≥ 4h ago.
    """
    from app.database.session import get_session
    from app.database.models import ScanAbstention

    attempted = updated = failed = 0
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)

        with get_session() as db:
            rows = (
                db.query(ScanAbstention)
                .filter(
                    ScanAbstention.spy_outcome_1d_pct.is_(None),
                    ScanAbstention.abstained_at >= cutoff,
                    ScanAbstention.spy_price_at_abstention.isnot(None),
                )
                .all()
            )

            for row in rows:
                abstained = row.abstained_at.replace(tzinfo=timezone.utc) if row.abstained_at.tzinfo is None else row.abstained_at
                if now - abstained < timedelta(hours=4):
                    continue
                attempted += 1
                try:
                    spy_4h, spy_1d = _fetch_outcome_prices("SPY", abstained)
                    base = row.spy_price_at_abstention
                    if spy_4h is not None and base and base > 0:
                        row.spy_outcome_4h_pct = round((spy_4h - base) / base * 100, 3)
                    if spy_1d is not None and base and base > 0:
                        row.spy_outcome_1d_pct = round((spy_1d - base) / base * 100, 3)
                        # negative SPY = good abstention (we avoided a down day)
                        row.verdict = "good_abstention" if spy_1d < base else "bad_abstention"
                    row.outcome_fetched_at = now
                    updated += 1
                except Exception as exc:
                    logger.debug("SPY outcome fetch failed for %s: %s", row.abstained_at, exc)
                    failed += 1

            db.commit()

    except Exception as exc:
        logger.warning("backfill_scan_abstention_outcomes failed: %s", exc)

    return {"attempted": attempted, "updated": updated, "failed": failed}


def _fetch_outcome_prices(symbol: str, decided_at: datetime):
    """Fetch price ~4h and ~1d after decided_at from Alpaca historical bars.

    Returns (price_4h, price_1d) — either may be None if data unavailable.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from app.config import settings

    target_4h = decided_at + timedelta(hours=4)
    target_1d = decided_at + timedelta(hours=24)

    data_client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=decided_at,
        end=min(target_1d + timedelta(hours=2), datetime.now(timezone.utc)),
    )
    bars = data_client.get_stock_bars(req)
    bar_list = bars.get(symbol, [])
    if not bar_list:
        return None, None

    # Find bar closest to each target time
    def _closest(target):
        best = min(
            bar_list,
            key=lambda b: abs((b.timestamp.replace(tzinfo=timezone.utc) if b.timestamp.tzinfo is None else b.timestamp) - target),
            default=None,
        )
        return float(best.close) if best else None

    return _closest(target_4h), _closest(target_1d)


def backfill_outcomes(lookback_days: int = 7) -> int:
    """
    EOD script: fill outcome_pnl_pct for decision_audit rows where a trade was entered.
    Returns number of rows updated.
    """
    updated = 0
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit, Trade

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        with get_session() as db:
            rows = (
                db.query(DecisionAudit)
                .filter(
                    DecisionAudit.final_decision == "enter",
                    DecisionAudit.outcome_pnl_pct.is_(None),
                    DecisionAudit.decided_at >= cutoff,
                )
                .all()
            )

            for row in rows:
                trade = (
                    db.query(Trade)
                    .filter(
                        Trade.symbol == row.symbol,
                        Trade.created_at >= row.decided_at - timedelta(minutes=30),
                        Trade.status == "CLOSED",
                    )
                    .order_by(Trade.created_at)
                    .first()
                )
                if trade and trade.pnl is not None and trade.entry_price:
                    row.outcome_pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                    updated += 1

            db.commit()

    except Exception as exc:
        logger.warning("decision_audit backfill failed: %s", exc)

    return updated


def gate_performance_summary() -> list[dict]:
    """
    Return aggregate stats per block_reason — the calibration query.
    Shows whether each gate is blocking winners or losers.
    """
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit
        from sqlalchemy import func

        with get_session() as db:
            rows = (
                db.query(
                    DecisionAudit.block_reason,
                    func.count(DecisionAudit.id).label("count"),
                    func.avg(DecisionAudit.outcome_pnl_pct).label("avg_pnl_pct"),
                )
                .filter(DecisionAudit.final_decision == "block")
                .group_by(DecisionAudit.block_reason)
                .all()
            )
            return [
                {
                    "block_reason": r.block_reason,
                    "count": r.count,
                    "avg_pnl_pct": round(float(r.avg_pnl_pct or 0), 4),
                }
                for r in rows
            ]
    except Exception as exc:
        logger.warning("gate_performance_summary failed: %s", exc)
        return []


def gate_calibration_report() -> dict:
    """
    Rich gate calibration report for the Analytics UI.

    Returns:
      - per_gate: list of {block_reason, gate_category, strategy, count,
                            outcome_count, avg_outcome_4h_pct, avg_outcome_1d_pct, verdict_rate}
      - by_category: aggregated totals per gate_category
      - scan_abstentions: recent ScanAbstention rows with SPY outcomes
    """
    try:
        from app.database.session import get_session
        from app.database.models import DecisionAudit, ScanAbstention
        from sqlalchemy import func

        with get_session() as db:
            # Per-gate breakdown
            rows = (
                db.query(
                    DecisionAudit.block_reason,
                    DecisionAudit.gate_category,
                    DecisionAudit.strategy,
                    func.count(DecisionAudit.id).label("count"),
                    func.count(DecisionAudit.outcome_1d_pct).label("outcome_count"),
                    func.avg(DecisionAudit.outcome_4h_pct).label("avg_4h"),
                    func.avg(DecisionAudit.outcome_1d_pct).label("avg_1d"),
                )
                .filter(DecisionAudit.final_decision == "block")
                .group_by(
                    DecisionAudit.block_reason,
                    DecisionAudit.gate_category,
                    DecisionAudit.strategy,
                )
                .all()
            )

            per_gate = []
            for r in rows:
                avg_1d = float(r.avg_1d) if r.avg_1d is not None else None
                # verdict: negative avg 1d = gate correctly blocked a loser (good)
                verdict = None
                if avg_1d is not None:
                    if avg_1d < -0.5:
                        verdict = "correct"
                    elif avg_1d > 0.5:
                        verdict = "recalibrate"
                    else:
                        verdict = "neutral"
                per_gate.append({
                    "block_reason": r.block_reason,
                    "gate_category": r.gate_category,
                    "strategy": r.strategy,
                    "count": r.count,
                    "outcome_count": r.outcome_count,
                    "avg_outcome_4h_pct": round(float(r.avg_4h), 3) if r.avg_4h is not None else None,
                    "avg_outcome_1d_pct": round(avg_1d, 3) if avg_1d is not None else None,
                    "verdict": verdict,
                })

            # By-category summary
            cat_rows = (
                db.query(
                    DecisionAudit.gate_category,
                    func.count(DecisionAudit.id).label("count"),
                    func.count(DecisionAudit.outcome_1d_pct).label("outcome_count"),
                    func.avg(DecisionAudit.outcome_1d_pct).label("avg_1d"),
                )
                .filter(DecisionAudit.final_decision == "block")
                .group_by(DecisionAudit.gate_category)
                .all()
            )
            by_category = [
                {
                    "gate_category": r.gate_category,
                    "count": r.count,
                    "outcome_count": r.outcome_count,
                    "avg_outcome_1d_pct": round(float(r.avg_1d), 3) if r.avg_1d is not None else None,
                }
                for r in cat_rows
            ]

            # Recent scan abstentions
            abstentions = (
                db.query(ScanAbstention)
                .order_by(ScanAbstention.abstained_at.desc())
                .limit(30)
                .all()
            )
            scan_rows = [
                {
                    "abstained_at": a.abstained_at.isoformat(),
                    "gate_type": a.gate_type,
                    "gate_detail": a.gate_detail,
                    "spy_price_at_abstention": a.spy_price_at_abstention,
                    "spy_first_hour_range_pct": a.spy_first_hour_range_pct,
                    "spy_outcome_4h_pct": a.spy_outcome_4h_pct,
                    "spy_outcome_1d_pct": a.spy_outcome_1d_pct,
                    "verdict": a.verdict,
                }
                for a in abstentions
            ]

            return {
                "per_gate": per_gate,
                "by_category": by_category,
                "scan_abstentions": scan_rows,
            }

    except Exception as exc:
        logger.warning("gate_calibration_report failed: %s", exc)
        return {"per_gate": [], "by_category": [], "scan_abstentions": []}
