"""
Phase 56: Paper Trading Monitoring Report
Usage: python scripts/paper_trading_report.py [--days N]

Outputs a CLI summary covering:
  - Daily P&L by strategy (swing vs intraday)
  - Fill quality / slippage per trade
  - RM veto breakdown (which rules fire most)
  - Abstention gate firing rate
  - 7-day rolling Sharpe with WARNING if < 0.5
"""
import argparse
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# -- DB bootstrap -------------------------------------------------------------


def _get_db():
    from app.database.session import get_session
    return get_session()

# -- Helpers -------------------------------------------------------------------


def _since(days: int) -> datetime:
    return datetime.now(tz=timezone.utc) - timedelta(days=days)


def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "  N/A   "
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"


def _sharpe(returns: List[float]) -> Optional[float]:
    """Annualized Sharpe from daily returns (252 trading days)."""
    if len(returns) < 2:
        return None
    import statistics
    mu = statistics.mean(returns)
    sd = statistics.stdev(returns)
    if sd == 0:
        return None
    return (mu / sd) * (252 ** 0.5)


# -- Section builders ----------------------------------------------------------

def section_pnl(db, days: int) -> None:
    from app.database.models import Trade
    cutoff = _since(days)
    trades = (
        db.query(Trade)
        .filter(Trade.closed_at >= cutoff, Trade.status == "CLOSED")
        .all()
    )

    swing_pnl: Dict[str, float] = defaultdict(float)
    intra_pnl: Dict[str, float] = defaultdict(float)

    for t in trades:
        if t.pnl is None or t.closed_at is None:
            continue
        date_str = t.closed_at.strftime("%Y-%m-%d")
        # Infer strategy from signal_type or trade metadata
        # Intraday trades closed same day as opened; swing held overnight
        opened = t.created_at.date() if t.created_at else None
        closed = t.closed_at.date()
        is_intraday = (opened == closed) if opened else False
        if is_intraday:
            intra_pnl[date_str] += t.pnl
        else:
            swing_pnl[date_str] += t.pnl

    all_dates = sorted(set(list(swing_pnl) + list(intra_pnl)))

    print(f"\n{'-'*60}")
    print(f"  P&L BY STRATEGY  (last {days} days)")
    print(f"{'-'*60}")
    if not all_dates:
        print("  No closed trades in this period.")
        return

    print(f"  {'Date':<12}  {'Swing':>10}  {'Intraday':>10}  {'Total':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    total_swing = total_intra = 0.0
    for d in all_dates:
        s = swing_pnl.get(d, 0.0)
        i = intra_pnl.get(d, 0.0)
        total_swing += s
        total_intra += i
        print(f"  {d}  {_fmt_pnl(s):>10}  {_fmt_pnl(i):>10}  {_fmt_pnl(s+i):>10}")
    print(f"  {'TOTAL':<12}  {_fmt_pnl(total_swing):>10}  {_fmt_pnl(total_intra):>10}  {_fmt_pnl(total_swing+total_intra):>10}")


def section_slippage(db, days: int) -> None:
    from app.database.models import Order
    cutoff = _since(days)
    orders = (
        db.query(Order)
        .filter(Order.timestamp >= cutoff, Order.status == "FILLED",
                Order.order_type == "ENTRY", Order.slippage_bps.isnot(None))
        .all()
    )

    print(f"\n{'-'*60}")
    print(f"  FILL QUALITY / SLIPPAGE  (last {days} days)")
    print(f"{'-'*60}")
    if not orders:
        print("  No filled entry orders with slippage data.")
        return

    slips = [o.slippage_bps for o in orders if o.slippage_bps is not None]
    import statistics
    avg = statistics.mean(slips)
    worst = max(slips)
    best = min(slips)
    n_positive = sum(1 for s in slips if s > 0)
    print(f"  Fills analyzed : {len(slips)}")
    print(f"  Avg slippage   : {avg:+.1f} bps")
    print(f"  Best fill      : {best:+.1f} bps")
    print(f"  Worst fill     : {worst:+.1f} bps")
    print(f"  Adverse fills  : {n_positive}/{len(slips)} ({100*n_positive/len(slips):.0f}%)")


def section_veto_breakdown(db, days: int) -> None:
    from app.database.models import AgentDecision
    cutoff = _since(days)
    rows = (
        db.query(AgentDecision)
        .filter(AgentDecision.timestamp >= cutoff,
                AgentDecision.decision_type.in_(["TRADE_REJECTED", "TRADE_APPROVED"]))
        .all()
    )

    approved = rejected = 0
    rule_counts: Counter = Counter()
    for r in rows:
        if r.decision_type == "TRADE_APPROVED":
            approved += 1
        else:
            rejected += 1
            rule = (r.reasoning or {}).get("failed_rule", "unknown")
            rule_counts[rule] += 1

    total = approved + rejected
    veto_rate = 100.0 * rejected / total if total else 0.0

    print(f"\n{'-'*60}")
    print(f"  RM VETO BREAKDOWN  (last {days} days)")
    print(f"{'-'*60}")
    print(f"  Total proposals : {total}")
    print(f"  Approved        : {approved}")
    print(f"  Rejected        : {rejected}  ({veto_rate:.1f}% veto rate)")
    if rule_counts:
        print("\n  Top rejection rules:")
        for rule, cnt in rule_counts.most_common(10):
            print(f"    {rule:<35} {cnt:>4}")


def section_abstention(db, days: int) -> None:
    from app.database.models import AgentDecision
    cutoff = _since(days)

    abstentions = (
        db.query(AgentDecision)
        .filter(AgentDecision.timestamp >= cutoff,
                AgentDecision.decision_type == "SWING_ABSTAINED")
        .count()
    )
    swing_analyses = (
        db.query(AgentDecision)
        .filter(AgentDecision.timestamp >= cutoff,
                AgentDecision.decision_type == "SWING_PREMARKET_ANALYSIS")
        .count()
    )
    gate_blocks = (
        db.query(AgentDecision)
        .filter(AgentDecision.timestamp >= cutoff,
                AgentDecision.decision_type == "SELECTION_SKIPPED")
        .count()
    )

    total_opportunities = swing_analyses + abstentions
    abstention_rate = 100.0 * abstentions / total_opportunities if total_opportunities else 0.0

    print(f"\n{'-'*60}")
    print(f"  ABSTENTION GATE ACTIVITY  (last {days} days)")
    print(f"{'-'*60}")
    print(f"  Swing analyses run    : {swing_analyses}")
    print(f"  Swing abstentions     : {abstentions}  ({abstention_rate:.1f}% of opportunities)")
    print(f"  Selection skipped     : {gate_blocks}")

    # Breakdowns for skipped
    if gate_blocks > 0:
        skip_rows = (
            db.query(AgentDecision)
            .filter(AgentDecision.timestamp >= cutoff,
                    AgentDecision.decision_type == "SELECTION_SKIPPED")
            .all()
        )
        reasons: Counter = Counter()
        for r in skip_rows:
            reason = (r.reasoning or {}).get("reason", "unknown")
            reasons[reason] += 1
        print("\n  Skip reasons:")
        for reason, cnt in reasons.most_common():
            print(f"    {reason:<40} {cnt:>4}")


def section_rolling_sharpe(db, days: int) -> None:
    from app.database.models import Trade
    cutoff = _since(days)
    trades = (
        db.query(Trade)
        .filter(Trade.closed_at >= cutoff, Trade.status == "CLOSED",
                Trade.pnl.isnot(None))
        .order_by(Trade.closed_at)
        .all()
    )

    daily_pnl: Dict[str, float] = defaultdict(float)
    for t in trades:
        if t.closed_at:
            daily_pnl[t.closed_at.strftime("%Y-%m-%d")] += (t.pnl or 0.0)

    returns = list(daily_pnl.values())
    sharpe = _sharpe(returns)

    print(f"\n{'-'*60}")
    print(f"  ROLLING SHARPE  (last {days} days)")
    print(f"{'-'*60}")
    if sharpe is None:
        print(f"  Insufficient data ({len(returns)} trading days with closed trades).")
        return

    flag = "  !! WARNING: Sharpe below 0.5 threshold!" if sharpe < 0.5 else "  OK Above threshold"
    print(f"  Trading days    : {len(returns)}")
    print(f"  Rolling Sharpe  : {sharpe:+.3f}")
    print(flag)


def section_decision_audit(db, days: int) -> None:
    from app.database.models import DecisionAudit
    cutoff = _since(days)
    rows = (
        db.query(DecisionAudit)
        .filter(DecisionAudit.decided_at >= cutoff)
        .all()
    )

    print(f"\n{'-'*60}")
    print(f"  PM DECISION AUDIT  (last {days} days)")
    print(f"{'-'*60}")

    if not rows:
        print("  No decision_audit rows yet — PM has not run or audit is empty.")
        return

    n_enter = sum(1 for r in rows if r.final_decision == "enter")
    n_block = sum(1 for r in rows if r.final_decision == "block")
    n_size_down = sum(1 for r in rows if r.final_decision == "size_down")
    n_exit_review = sum(1 for r in rows if r.final_decision == "exit_review")
    total = len(rows)
    block_rate = 100.0 * n_block / total if total else 0.0

    print(f"  Total decisions   : {total}")
    print(f"  Entered           : {n_enter}")
    print(f"  Blocked           : {n_block}  ({block_rate:.1f}% block rate)")
    if n_size_down:
        print(f"  Sized down        : {n_size_down}")
    if n_exit_review:
        print(f"  Exit-review flags : {n_exit_review}")

    # Block reason breakdown
    block_rows = [r for r in rows if r.final_decision == "block" and r.block_reason]
    if block_rows:
        reason_counts: Counter = Counter()
        for r in block_rows:
            # Bucket by prefix (e.g. "nis_block_entry: ..." → "nis_block_entry")
            prefix = r.block_reason.split(":")[0].strip()
            reason_counts[prefix] += 1
        print("\n  Top block reasons:")
        for reason, cnt in reason_counts.most_common(10):
            pct = 100.0 * cnt / n_block if n_block else 0.0
            print(f"    {reason:<40} {cnt:>4}  ({pct:.0f}%)")

    # NIS / macro breakdown
    nis_blocks = sum(1 for r in block_rows if r.block_reason and "nis" in r.block_reason.lower())
    macro_blocks = sum(1 for r in block_rows if r.block_reason and "macro" in r.block_reason.lower())
    corr_blocks = sum(1 for r in block_rows if r.block_reason and "corr" in r.block_reason.lower())
    if n_block:
        print("\n  Block source split:")
        print(f"    NIS (news)        : {nis_blocks}")
        print(f"    Macro calendar    : {macro_blocks}")
        print(f"    Correlation gate  : {corr_blocks}")
        print(f"    Other / RM        : {n_block - nis_blocks - macro_blocks - corr_blocks}")

    # Outcome quality — rows where outcome is known
    with_outcome = [r for r in rows if r.final_decision == "enter" and r.outcome_pnl_pct is not None]
    blocked_with_outcome = [r for r in rows if r.final_decision == "block" and r.outcome_pnl_pct is not None]
    if with_outcome:
        import statistics
        avg_enter_pnl = statistics.mean(r.outcome_pnl_pct for r in with_outcome)
        wins = sum(1 for r in with_outcome if r.outcome_pnl_pct > 0)
        print(f"\n  Entered trade outcomes ({len(with_outcome)} with known P&L):")
        print(f"    Avg realized P&L  : {avg_enter_pnl:+.2%}")
        print(f"    Win rate          : {wins}/{len(with_outcome)} ({100*wins/len(with_outcome):.0f}%)")
    else:
        print("\n  Outcomes: not yet backfilled (run EOD job or wait for 16:30 task)")

    if blocked_with_outcome:
        avg_missed = statistics.mean(r.outcome_pnl_pct for r in blocked_with_outcome)
        print(f"  Blocked trade missed alpha: {avg_missed:+.2%} avg  ({len(blocked_with_outcome)} known)")

    # NIS sizing overlay activity
    sized_down = [r for r in rows if r.final_decision == "enter" and r.news_sizing_multiplier
                  and r.news_sizing_multiplier < 0.99]
    if sized_down:
        avg_mult = sum(r.news_sizing_multiplier for r in sized_down) / len(sized_down)
        print(f"\n  NIS size-down applied  : {len(sized_down)} trades  (avg {avg_mult:.2f}×)")

    # Top symbols blocked
    blocked_symbols: Counter = Counter(r.symbol for r in rows if r.final_decision == "block")
    if blocked_symbols:
        top_blocked = blocked_symbols.most_common(5)
        print(f"  Most-blocked symbols   : {', '.join(f'{s}({n})' for s, n in top_blocked)}")


def section_daily_summary(db, days: int) -> None:
    from app.database.models import RiskMetric
    cutoff = _since(days)
    rows = (
        db.query(RiskMetric)
        .filter(RiskMetric.timestamp >= cutoff)
        .order_by(RiskMetric.date)
        .all()
    )

    print(f"\n{'-'*60}")
    print(f"  DAILY SUMMARY TABLE  (last {days} days)")
    print(f"{'-'*60}")

    if not rows:
        print("  No daily summary rows yet — EOD job has not run.")
        return

    print(f"  {'Date':<12} {'TotalP&L':>10} {'SwingP&L':>10} {'IntraP&L':>10} "
          f"{'Trades':>7} {'BlockRate':>10} {'WinRate':>9}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*10} {'-'*9}")

    for r in rows:
        pos = r.position_concentration or {}
        sec = r.sector_concentration or {}
        swing_pnl = pos.get("swing_pnl", 0.0)
        intra_pnl = pos.get("intraday_pnl", 0.0)
        n_trades = pos.get("swing_trades", 0) + pos.get("intraday_trades", 0)
        block_rate = sec.get("block_rate", None)
        sw_wr = pos.get("swing_win_rate", None)
        block_str = f"{block_rate:.0%}" if block_rate is not None else "  N/A"
        wr_str = f"{sw_wr:.0%}" if sw_wr is not None else " N/A"
        pnl_str = f"{r.daily_pnl:+.2f}" if r.daily_pnl is not None else "  N/A"
        print(
            f"  {r.date:<12} {pnl_str:>10} {swing_pnl:>+10.2f} {intra_pnl:>+10.2f} "
            f"{n_trades:>7} {block_str:>10} {wr_str:>9}"
        )

    total_pnl = sum(r.daily_pnl for r in rows if r.daily_pnl is not None)
    print(f"\n  Period total P&L  : {total_pnl:+.2f}")


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MrTrader paper trading report")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in calendar days")
    args = parser.parse_args()

    db = _get_db()
    try:
        print(f"\n{'='*60}")
        print(f"  MRTRADER PAPER TRADING REPORT  —  last {args.days} days")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        section_pnl(db, args.days)
        section_slippage(db, args.days)
        section_decision_audit(db, args.days)
        section_daily_summary(db, args.days)
        section_veto_breakdown(db, args.days)
        section_abstention(db, args.days)
        section_rolling_sharpe(db, args.days)

        print(f"\n{'='*60}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
