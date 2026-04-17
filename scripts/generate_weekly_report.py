"""
Weekly performance report generator.

Run every Monday pre-market (or manually):
    python scripts/generate_weekly_report.py

Outputs a Markdown file to reports/YYYY-MM-DD.md with:
  - P&L vs SPY benchmark
  - Per-symbol breakdown
  - Win rate trend (rolling 20 trades)
  - Signal attribution (EMA_CROSSOVER vs RSI_DIP)
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure env vars are loaded from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.database.models import RiskMetric, Trade
from app.database.session import get_session
from app.analytics.signal_attribution import get_signal_attribution
from app.analytics.drawdown_analyzer import get_drawdown_summary


def _get_spy_return(start: date, end: date) -> float:
    """Fetch SPY return for the period. Returns 0.0 if yfinance unavailable."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start.isoformat(), end=end.isoformat(),
                          progress=False, auto_adjust=True)
        if len(spy) >= 2:
            start_price = float(spy["Close"].iloc[0])
            end_price = float(spy["Close"].iloc[-1])
            return round((end_price - start_price) / start_price * 100, 2)
    except Exception:
        pass
    return 0.0


def _rolling_win_rate(trades, window: int = 20) -> float:
    """Win rate of the last `window` closed trades."""
    closed = [t for t in trades if t.pnl is not None][-window:]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t.pnl > 0)
    return round(wins / len(closed) * 100, 1)


def generate(days: int = 7) -> str:
    """Generate the report and return the Markdown string."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    start_dt = datetime.combine(start_date, datetime.min.time())

    db = get_session()
    try:
        trades = (
            db.query(Trade)
            .filter(Trade.status == "CLOSED", Trade.closed_at >= start_dt)
            .order_by(Trade.closed_at)
            .all()
        )
        all_trades = db.query(Trade).filter(Trade.status == "CLOSED").order_by(Trade.closed_at).all()
        metrics = (
            db.query(RiskMetric)
            .filter(RiskMetric.date >= start_date.isoformat())
            .order_by(RiskMetric.date)
            .all()
        )
    finally:
        db.close()

    total_pnl = sum(float(t.pnl) for t in trades if t.pnl) or 0.0
    wins = sum(1 for t in trades if t.pnl and t.pnl > 0)
    win_rate = round(wins / len(trades) * 100, 1) if trades else 0.0
    max_dd = max((float(m.max_drawdown or 0) * 100 for m in metrics), default=0.0)
    spy_return = _get_spy_return(start_date, end_date)
    rolling_wr = _rolling_win_rate(all_trades)

    # Per-symbol breakdown
    by_symbol: dict = {}
    for t in trades:
        s = by_symbol.setdefault(t.symbol, {"trades": 0, "pnl": 0.0, "wins": 0})
        s["trades"] += 1
        s["pnl"] = round(s["pnl"] + (float(t.pnl) if t.pnl else 0.0), 2)
        if t.pnl and t.pnl > 0:
            s["wins"] += 1

    # Signal attribution
    attribution = get_signal_attribution(days=days)

    lines = [
        f"# MrTrader Weekly Report — {end_date.isoformat()}",
        f"**Period**: {start_date.isoformat()} → {end_date.isoformat()}",
        "",
        "## Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total P&L | ${total_pnl:+.2f} |",
        f"| SPY return (benchmark) | {spy_return:+.2f}% |",
        f"| Alpha | {total_pnl:.2f} vs SPY |",
        f"| Trades | {len(trades)} |",
        f"| Win rate (week) | {win_rate}% |",
        f"| Win rate (rolling 20) | {rolling_wr}% |",
        f"| Max drawdown | {max_dd:.2f}% |",
        "",
        "## Per-Symbol Breakdown",
        "| Symbol | Trades | Wins | P&L |",
        "|--------|--------|------|-----|",
    ]
    for sym, data in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True):
        lines.append(f"| {sym} | {data['trades']} | {data['wins']} | ${data['pnl']:+.2f} |")

    lines += [
        "",
        "## Signal Attribution",
        "| Signal | Trades | Win % | Avg P&L | Total P&L |",
        "|--------|--------|-------|---------|-----------|",
    ]
    for sig, data in sorted(attribution.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
        lines.append(
            f"| {sig} | {data['trades']} | {data['win_rate']}% "
            f"| ${data['avg_pnl']:+.2f} | ${data['total_pnl']:+.2f} |"
        )

    lines += [
        "",
        f"---",
        f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC by MrTrader*",
    ]
    return "\n".join(lines)


def main():
    report = generate()
    out_dir = Path(__file__).parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{date.today().isoformat()}.md"
    path.write_text(report, encoding="utf-8")
    print(f"Report written to {path}")
    print(report)


if __name__ == "__main__":
    main()
