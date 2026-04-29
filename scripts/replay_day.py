"""
Day replay — reconstruct what the system saw, decided, and did on any trading day.

Usage:
    python scripts/replay_day.py --date 2026-05-01
    python scripts/replay_day.py --date 2026-05-01 --symbol NVDA
    python scripts/replay_day.py  # defaults to today

Prints a chronological timeline:
  [09:00]  NIS morning digest signals (from news_signal_cache)
  [09:XX]  PM decisions (decision_audit) — enter/block with reasons and top features
  [09:XX]  Trades opened (trades table)
  [10:XX+] Position reviews, exit flags
  [16:XX]  Trades closed, daily P&L
  [audit]  Key agent_decisions and audit_logs events
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


def _c(colour, text):
    return f"{colour}{text}{RESET}"


def _fmt_time(dt) -> str:
    if dt is None:
        return "??:??"
    if hasattr(dt, "strftime"):
        return dt.strftime("%H:%M:%S")
    return str(dt)[:8]


def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "P&L N/A"
    sign = "+" if v >= 0 else ""
    colour = GREEN if v >= 0 else RED
    return _c(colour, f"{sign}${v:.2f}")


def _print_header(date_str: str, symbol_filter: Optional[str]) -> None:
    filt = f"  symbol={symbol_filter}" if symbol_filter else "  all symbols"
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  MrTrader Day Replay — {date_str}{filt}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")


def replay_nis_signals(db, day_start, day_end, symbol_filter):
    from app.database.models import NewsSignalCache
    rows = (
        db.query(NewsSignalCache)
        .filter(
            NewsSignalCache.evaluated_at >= day_start,
            NewsSignalCache.evaluated_at < day_end,
        )
        .order_by(NewsSignalCache.evaluated_at)
        .all()
    )
    if symbol_filter:
        rows = [r for r in rows if r.symbol == symbol_filter.upper()]

    if not rows:
        return

    print(f"\n{BOLD}{CYAN}── NIS Stock Signals scored this day ──{RESET}")
    for r in rows:
        policy_colour = RED if r.action_policy == "block_entry" else (
            YELLOW if r.action_policy and "size_down" in r.action_policy else GREEN
        )
        print(
            f"  {_fmt_time(r.evaluated_at)}  {BOLD}{r.symbol:<6}{RESET}  "
            f"policy={_c(policy_colour, r.action_policy or 'N/A'):<22}  "
            f"dir={r.direction_score:+.2f}  mat={r.materiality_score:.2f}  "
            f"conf={r.confidence or 0:.2f}  size={r.sizing_multiplier or 1:.2f}×"
        )
        if r.rationale:
            print(f"           {DIM}{r.rationale[:100]}{RESET}")


def replay_decisions(db, day_start, day_end, symbol_filter):
    from app.database.models import DecisionAudit
    rows = (
        db.query(DecisionAudit)
        .filter(
            DecisionAudit.decided_at >= day_start,
            DecisionAudit.decided_at < day_end,
        )
        .order_by(DecisionAudit.decided_at)
        .all()
    )
    if symbol_filter:
        rows = [r for r in rows if r.symbol == symbol_filter.upper()]

    if not rows:
        return

    print(f"\n{BOLD}{CYAN}── PM Decisions ──{RESET}")
    for r in rows:
        if r.final_decision == "enter":
            icon = _c(GREEN, "ENTER   ")
        elif r.final_decision == "block":
            icon = _c(RED, "BLOCK   ")
        elif r.final_decision == "size_down":
            icon = _c(YELLOW, "SIZE_DN ")
        else:
            icon = _c(YELLOW, r.final_decision[:8].upper())

        score_str = f"score={r.model_score:.3f}" if r.model_score else "score=N/A"
        size_str = f"  size={r.size_multiplier:.2f}×" if r.size_multiplier != 1.0 else ""
        print(
            f"  {_fmt_time(r.decided_at)}  {BOLD}{r.symbol:<6}{RESET}  "
            f"{icon}  {r.strategy:<9}  {score_str}{size_str}"
        )

        if r.block_reason:
            print(f"           {DIM}block: {r.block_reason[:100]}{RESET}")

        if r.news_action_policy and r.news_action_policy != "ignore":
            print(
                f"           {DIM}NIS: {r.news_action_policy}  "
                f"dir={r.news_direction_score or 0:+.2f}  "
                f"mat={r.news_materiality or 0:.2f}{RESET}"
            )
            if r.news_rationale:
                print(f"           {DIM}  → {r.news_rationale[:90]}{RESET}")

        if r.macro_risk_level and r.macro_risk_level != "LOW":
            print(f"           {DIM}macro: {r.macro_risk_level}  sizing={r.macro_sizing_factor}{RESET}")

        if r.top_features:
            feat_str = "  ".join(f"{k}={v:.3f}" for k, v in list(r.top_features.items())[:5])
            print(f"           {DIM}features: {feat_str}{RESET}")

        if r.outcome_pnl_pct is not None:
            pnl_col = GREEN if r.outcome_pnl_pct >= 0 else RED
            print(f"           outcome: {_c(pnl_col, f'{r.outcome_pnl_pct:+.2%}')}")


def replay_trades(db, day_start, day_end, symbol_filter):
    from app.database.models import Trade, Order
    # Opened today
    opened = (
        db.query(Trade)
        .filter(
            Trade.created_at >= day_start,
            Trade.created_at < day_end,
        )
        .order_by(Trade.created_at)
        .all()
    )
    # Closed today (may have been opened earlier)
    closed = (
        db.query(Trade)
        .filter(
            Trade.closed_at >= day_start,
            Trade.closed_at < day_end,
            Trade.status == "CLOSED",
        )
        .order_by(Trade.closed_at)
        .all()
    )
    if symbol_filter:
        sym = symbol_filter.upper()
        opened = [t for t in opened if t.symbol == sym]
        closed = [t for t in closed if t.symbol == sym]

    if not opened and not closed:
        return

    print(f"\n{BOLD}{CYAN}── Trades ──{RESET}")

    shown_ids = set()
    for t in opened:
        shown_ids.add(t.id)
        strategy = t.trade_type or "swing"
        print(
            f"  {_fmt_time(t.created_at)}  {BOLD}{t.symbol:<6}{RESET}  "
            f"{_c(GREEN, 'OPEN ')}  {strategy:<9}  "
            f"qty={t.quantity}  entry=${t.entry_price:.2f}  "
            f"stop=${t.stop_price:.2f}  target=${t.target_price:.2f}"
            if t.stop_price and t.target_price else
            f"  {_fmt_time(t.created_at)}  {BOLD}{t.symbol:<6}{RESET}  "
            f"{_c(GREEN, 'OPEN ')}  {strategy:<9}  "
            f"qty={t.quantity}  entry=${t.entry_price:.2f}"
        )

    for t in closed:
        if t.id not in shown_ids:
            strategy = t.trade_type or "swing"
            print(
                f"  {_fmt_time(t.created_at)}  {BOLD}{t.symbol:<6}{RESET}  "
                f"{_c(GREEN, 'OPEN ')}  {strategy:<9}  "
                f"qty={t.quantity}  entry=${t.entry_price:.2f}"
            )
        exit_str = f"${t.exit_price:.2f}" if t.exit_price else "N/A"
        bars = f"  bars={t.bars_held}" if t.bars_held else ""
        print(
            f"  {_fmt_time(t.closed_at)}  {BOLD}{t.symbol:<6}{RESET}  "
            f"{_c(RED, 'CLOSE')}  {(t.trade_type or 'swing'):<9}  "
            f"exit={exit_str}  {_fmt_pnl(t.pnl)}{bars}"
        )

    # Fill details (slippage)
    trade_ids = [t.id for t in opened + closed]
    if trade_ids:
        orders = (
            db.query(Order)
            .filter(Order.trade_id.in_(trade_ids), Order.status == "FILLED")
            .order_by(Order.timestamp)
            .all()
        )
        if orders:
            print(f"\n  {DIM}Fill details:{RESET}")
            for o in orders:
                slip_str = (f"  slip={o.slippage_bps:+.1f}bps" if o.slippage_bps is not None else "")
                print(
                    f"  {_fmt_time(o.timestamp)}  {DIM}{o.order_type:<6}  "
                    f"fill=${o.filled_price or 0:.2f}  qty={o.filled_qty}{slip_str}{RESET}"
                )


def replay_audit_events(db, day_start, day_end, symbol_filter):
    from app.database.models import AuditLog, AgentDecision
    # Key audit_log events
    audit_rows = (
        db.query(AuditLog)
        .filter(
            AuditLog.timestamp >= day_start,
            AuditLog.timestamp < day_end,
            AuditLog.action.in_([
                "TRADE_EXECUTED", "POSITION_CLOSED", "KILL_SWITCH_ACTIVATED",
                "KILL_SWITCH_DEACTIVATED", "PAPER_TRADING_ENABLED",
                "MODEL_DRIFT_ALERT", "NIS_MORNING_DIGEST",
            ]),
        )
        .order_by(AuditLog.timestamp)
        .all()
    )

    # Key agent_decisions (task completions, skips, gate fires)
    decision_rows = (
        db.query(AgentDecision)
        .filter(
            AgentDecision.timestamp >= day_start,
            AgentDecision.timestamp < day_end,
            AgentDecision.decision_type.in_([
                "TASK_COMPLETED", "TASK_FAILED", "SELECTION_SKIPPED",
                "MACRO_GATE_BLOCKED", "SWING_PREMARKET_ANALYSIS",
            ]),
        )
        .order_by(AgentDecision.timestamp)
        .all()
    )

    if not audit_rows and not decision_rows:
        return

    print(f"\n{BOLD}{CYAN}── Agent Events ──{RESET}")

    # Merge and sort
    events = []
    for r in audit_rows:
        events.append((r.timestamp, "AUDIT", r.action, r.details or {}))
    for r in decision_rows:
        events.append((r.timestamp, "AGENT", r.decision_type, r.reasoning or {}))
    events.sort(key=lambda x: x[0])

    for ts, source, action, details in events:
        colour = MAGENTA if source == "AUDIT" else BLUE
        detail_str = ""
        if isinstance(details, dict):
            parts = []
            for k in ("task", "reason", "strategy", "window", "error"):
                if k in details:
                    parts.append(f"{k}={details[k]}")
            detail_str = "  " + "  ".join(parts[:3]) if parts else ""
        print(f"  {_fmt_time(ts)}  {_c(colour, f'[{source}]'):<20}  {action}{detail_str}")


def main():
    parser = argparse.ArgumentParser(description="MrTrader day replay")
    parser.add_argument("--date", default=None,
                        help="Date to replay (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--symbol", default=None,
                        help="Optional: filter to a single symbol.")
    args = parser.parse_args()

    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now(ET).strftime("%Y-%m-%d")

    day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ET)
    day_end = day_start + timedelta(days=1)

    _print_header(date_str, args.symbol)

    from app.database.session import get_session
    with get_session() as db:
        # NIS macro context for the day
        try:
            from app.database.models import MacroSignalCache
            macro = (
                db.query(MacroSignalCache)
                .filter(MacroSignalCache.date == date_str)
                .first()
            )
            if macro:
                risk_col = RED if macro.risk_level == "HIGH" else (
                    YELLOW if macro.risk_level == "MEDIUM" else GREEN
                )
                print(
                    f"\n{BOLD}{CYAN}── Macro Context ──{RESET}"
                    f"\n  risk={_c(risk_col, macro.risk_level)}  "
                    f"sizing={macro.sizing_factor}×  "
                    f"block_entries={macro.block_new_entries}"
                )
                if macro.rationale:
                    print(f"  {DIM}{macro.rationale[:120]}{RESET}")
        except Exception:
            pass

        replay_nis_signals(db, day_start, day_end, args.symbol)
        replay_decisions(db, day_start, day_end, args.symbol)
        replay_trades(db, day_start, day_end, args.symbol)
        replay_audit_events(db, day_start, day_end, args.symbol)

    # Day summary from risk_metrics
    try:
        from app.database.session import get_session as _gs
        from app.database.models import RiskMetric
        with _gs() as db2:
            rm = db2.query(RiskMetric).filter_by(date=date_str).first()
            if rm:
                pos = rm.position_concentration or {}
                sec = rm.sector_concentration or {}
                print(f"\n{BOLD}{CYAN}── Day Summary ──{RESET}")
                pnl_col = GREEN if (rm.daily_pnl or 0) >= 0 else RED
                print(
                    f"  Total P&L: {_c(pnl_col, f'{rm.daily_pnl:+.2f}' if rm.daily_pnl is not None else 'N/A')}"
                    f"    swing={pos.get('swing_pnl', 'N/A'):+.2f}"
                    f"    intraday={pos.get('intraday_pnl', 'N/A'):+.2f}"
                )
                print(
                    f"  Trades: {pos.get('swing_trades', 0)} swing  "
                    f"{pos.get('intraday_trades', 0)} intraday  "
                    f"swing_wr={pos.get('swing_win_rate', 'N/A')}"
                )
                print(
                    f"  PM decisions: {sec.get('pm_decisions_total', 0)} total  "
                    f"{sec.get('entries', 0)} entered  "
                    f"{sec.get('blocks', 0)} blocked  "
                    f"({sec.get('block_rate', 0):.0%} block rate)"
                )
    except Exception:
        pass

    print(f"\n{BOLD}{'=' * 70}{RESET}\n")


if __name__ == "__main__":
    main()
