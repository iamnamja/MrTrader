"""
MrTrader notification system.

Producers call:
    notifier.enqueue("event_type", payload_dict)          # async via SQLite queue
    notifier.send_now("kill_switch", payload_dict)        # immediate SMTP (emergencies only)

Background watcher (scripts/notify_watcher.py) drains the queue every 5 seconds.

Credentials in .env:
    NOTIFY_GMAIL_USER=kimfamrecipebank@gmail.com
    NOTIFY_GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import sqlite3
import ssl
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Dedicated SQLite file — independent of Postgres main DB so it always works.
# Env-overridable so pytest-xdist workers each use an isolated file (no lock).
_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("MRTRADER_NOTIFICATIONS_DB", str(_ROOT / "data" / "notifications.db")))

RECIPIENT = "kimminjae@gmail.com"

# Per-event-type rate limit in seconds. 0 = never throttle (critical events).
RATE_LIMITS: dict[str, int] = {
    "a1_progress":       15 * 60,   # max one per 15 min (noisy progress)
    "diag_complete":     0,
    "paper_eod":         0,
    "kill_switch":       0,
    "training_complete": 0,
    "phase_complete":    0,
    "pead_weekly":       0,
    "trend_weekly":      0,   # trend_tracker.weekly_rollup (was dropped — unregistered)
    "trend_backval_weekly": 0,  # P1-4 live-vs-sim tracking-error report
    "cash_weekly":       0,   # P1-1 cash/T-bill sleeve idle-capital utilization
    "options_spread_mature": 0,  # P2-4 NBBO spread surface crossed maturity (fires once)
}

VALID_EVENTS = set(RATE_LIMITS)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    dedup_key   TEXT,
    created_at  REAL    NOT NULL,
    sent_at     REAL,
    error       TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_nq_unsent ON notification_queue (sent_at, id);
CREATE INDEX IF NOT EXISTS ix_nq_dedup  ON notification_queue (dedup_key);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("PRAGMA journal_mode=WAL;")
    c.executescript(_SCHEMA)
    return c


# ── Producer API ──────────────────────────────────────────────────────────────

def enqueue(event_type: str, payload: dict[str, Any], dedup_key: str | None = None) -> int | None:
    """Enqueue a notification. Returns row id, or None if deduped/invalid.
    Never raises — safe to call from any script."""
    if event_type not in VALID_EVENTS:
        log.warning("notifier.enqueue: unknown event_type=%r — dropped", event_type)
        return None
    try:
        with _conn() as c:
            if dedup_key:
                exists = c.execute(
                    "SELECT id FROM notification_queue WHERE dedup_key=? AND sent_at IS NULL",
                    (dedup_key,),
                ).fetchone()
                if exists:
                    return None
            cur = c.execute(
                "INSERT INTO notification_queue(event_type, payload, dedup_key, created_at) "
                "VALUES (?, ?, ?, ?)",
                (event_type, json.dumps(payload, default=str), dedup_key, time.time()),
            )
            return cur.lastrowid
    except Exception:
        log.exception("notifier.enqueue failed (swallowed)")
        return None


def send_now(event_type: str, payload: dict[str, Any]) -> bool:
    """Bypass queue — synchronous SMTP. Use ONLY for kill_switch / urgent alerts."""
    try:
        subject, html = render(event_type, payload)
        return _smtp_send(subject, html)
    except Exception:
        log.exception("notifier.send_now failed")
        return False


# ── Watcher-side API ──────────────────────────────────────────────────────────

def pending(limit: int = 20) -> list[tuple]:
    with _conn() as c:
        return c.execute(
            "SELECT id, event_type, payload, created_at, attempts "
            "FROM notification_queue "
            "WHERE sent_at IS NULL AND attempts < 5 "
            "ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()


def last_sent_at(event_type: str) -> float | None:
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(sent_at) FROM notification_queue "
            "WHERE event_type=? AND sent_at IS NOT NULL",
            (event_type,),
        ).fetchone()
        return row[0] if row else None


def mark_sent(row_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE notification_queue SET sent_at=? WHERE id=?", (time.time(), row_id))


def mark_failed(row_id: int, err: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE notification_queue SET attempts=attempts+1, error=? WHERE id=?",
            (err[:500], row_id),
        )


def should_throttle(event_type: str) -> bool:
    limit = RATE_LIMITS.get(event_type, 0)
    if not limit:
        return False
    last = last_sent_at(event_type)
    return bool(last and (time.time() - last) < limit)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render(event_type: str, p: dict[str, Any]) -> tuple[str, str]:
    """Return (subject, html_body)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if event_type == "a1_progress":
        pct = int(p.get("processed", 0)) * 100 // max(int(p.get("total", 1)), 1)
        subj = f"[MrTrader] A1 {pct}% — {p.get('processed', '?')}/{p.get('total', '?')} symbols"
        body = _section("A1 Feature IC — progress update", [
            ("Processed", f"{p.get('processed', '?')} / {p.get('total', '?')} ({pct}%)"),
            ("Elapsed", f"{p.get('elapsed_min', '?')} min"),
            ("ETA", f"{p.get('eta_min', '?')} min remaining"),
            ("Log", p.get("log_path", "")),
        ])
        if p.get("ic_table_html"):
            body += "<h3>Top features (interim)</h3>" + p["ic_table_html"]

    elif event_type == "diag_complete":
        subj = f"[MrTrader] DONE: {p.get('script', '?')} ({p.get('duration', '?')})"
        body = _section(f"Diagnostic complete — {p.get('script', '?')}", [
            ("Script", p.get("script")),
            ("Duration", p.get("duration")),
            ("Outcome", p.get("outcome", "")),
            ("Artifacts", "<br>".join(p.get("artifacts", []))),
        ])
        if p.get("summary_html"):
            body += "<h3>Summary</h3>" + p["summary_html"]

    elif event_type == "paper_eod":
        pnl = p.get("pnl", 0)
        pnl_color = "#1a7340" if pnl >= 0 else "#b00020"
        subj = f"[MrTrader] EOD {p.get('date', '')} — P&L ${pnl:+,.2f}"
        body = _section("Paper trading — end of day", [
            ("Date", p.get("date")),
            ("P&L", f"<span style='color:{pnl_color};font-weight:bold'>${pnl:+,.2f}</span>"),
            ("Portfolio eq.", f"${p.get('equity', 0):,.2f}"),
            ("Trades taken", p.get("trades")),
            ("Open positions", p.get("open_positions")),
            ("Regime", p.get("regime", "")),
        ])
        if p.get("trades_html"):
            body += "<h3>Trades</h3>" + p["trades_html"]
        if p.get("positions_html"):
            body += "<h3>Open positions</h3>" + p["positions_html"]

    elif event_type == "kill_switch":
        subj = f"[MrTrader] ⚠ KILL SWITCH: {p.get('reason', '?')}"
        body = (
            "<h2 style='color:#b00020'>⚠ Kill switch activated</h2>"
            + _section("Details", [
                ("Reason", p.get("reason")),
                ("Trigger", p.get("trigger", "")),
                ("Portfolio eq.", f"${p.get('equity', 0):,.2f}"),
                ("Action taken", p.get("action", "All orders cancelled, trading halted")),
            ])
        )

    elif event_type == "training_complete":
        gate = p.get("gate_result", "?")
        gate_color = "#1a7340" if "PASS" in str(gate).upper() else "#b00020"
        subj = f"[MrTrader] Trained {p.get('model', '?')} v{p.get('version', '?')} — {gate}"
        body = _section("Model training complete", [
            ("Model", p.get("model")),
            ("Version", p.get("version")),
            ("Sharpe (WF)", p.get("sharpe")),
            ("AUC", p.get("auc", "")),
            ("Gate result", f"<span style='color:{gate_color};font-weight:bold'>{gate}</span>"),
            ("Log", p.get("log_path", "")),
        ])
        if p.get("fold_table_html"):
            body += "<h3>Fold results</h3>" + p["fold_table_html"]

    elif event_type == "phase_complete":
        subj = f"[MrTrader] Phase {p.get('phase', '?')} complete"
        body = _section(f"Phase {p.get('phase')} done", [
            ("Tasks completed", p.get("tasks_done")),
            ("Outcome", p.get("outcome", "")),
            ("Next phase", p.get("next_phase", "")),
            ("Notes", p.get("notes", "")),
        ])

    elif event_type == "pead_weekly":
        subj = f"[MrTrader] PEAD weekly rollup — {p.get('week_ending', '?')}"
        body = _section("PEAD live-vs-backtest weekly rollup", [
            ("Week ending", p.get("week_ending")),
            ("Realized Sharpe (PEAD book)", p.get("realized_sharpe")),
            ("Backtest expectation", p.get("backtest_sharpe", "+0.546")),
            ("Trading days", p.get("n_days")),
            ("Signals / entered / filled", p.get("signals_entered_filled")),
            ("Cumulative P&L", p.get("cumulative_pnl")),
            ("Avg fill rate", p.get("avg_fill_rate")),
            ("VIX blocks fired", p.get("vix_blocks_fired")),
            ("Suppressed (opp/macro/RM)", p.get("suppressed_breakdown")),
            ("Notes", p.get("notes", "")),
        ])

    elif event_type == "trend_weekly":
        subj = f"[MrTrader] Trend weekly rollup — {p.get('week_ending', '?')}"
        body = _section("Trend live-vs-backtest weekly rollup", [
            ("Week ending", p.get("week_ending")),
            ("Realized Sharpe (trend book)", p.get("realized_sharpe")),
            ("Backtest expectation", p.get("backtest_sharpe", "+0.710")),
            ("Trading days", p.get("n_days")),
            ("Avg positions", p.get("avg_positions")),
            ("Avg gross deployed", p.get("avg_gross_deployed")),
            ("Cumulative P&L", p.get("cumulative_pnl")),
        ])

    elif event_type == "trend_backval_weekly":
        _v = p.get("verdict", "?")
        subj = f"[MrTrader] Trend back-validation (intended vs actual): {_v} — {p.get('window_end', '?')}"
        body = _section(f"Trend back-validation (intended vs actual) — verdict: {_v}", [
            ("Window", f"{p.get('window_start')} .. {p.get('window_end')}"),
            ("Trading days", p.get("n_days")),
            ("Intended-vs-actual correlation", p.get("corr")),
            ("Annualized tracking error", p.get("tracking_error_ann")),
            ("Annualized drift (actual - intended)", p.get("drift_ann")),
            ("Execution drag (bps/day)", p.get("slippage_drag_bps_day")),
            ("Actual cum return (NAV contrib)", p.get("actual_cum_return")),
            ("Intended cum return (NAV contrib)", p.get("intended_cum_return")),
            ("Governor-active days", p.get("governor_days")),
            ("Total rebalance blocks", p.get("total_blocked")),
            ("Note", p.get("note", "")),
        ])

    elif event_type == "cash_weekly":
        subj = f"[MrTrader] Cash/T-bill sleeve weekly — {p.get('week_ending', '?')}"
        body = _section("Cash sleeve (idle-capital utilization)", [
            ("Week ending", p.get("week_ending")),
            ("Trading days", p.get("n_days")),
            ("Latest T-bill deployed", p.get("latest_tbill_deployed")),
            ("Avg T-bill deployed", p.get("avg_tbill_deployed")),
            ("Latest settled cash", p.get("latest_cash_buffer")),
        ])

    elif event_type == "options_spread_mature":
        subj = "[MrTrader] Option spread surface is MATURE — VRP cost model ready to validate"
        body = _section("Option spread cost surface matured (P2-4)", [
            ("Distinct trading days", f"{p.get('n_days')} (>= {p.get('mature_min_days')})"),
            ("Observations", p.get("n_obs")),
            ("Window", p.get("window")),
            ("Next", "Re-assess the Phase-3 VRP go/no-go on the matured surface + live-paper."),
        ])

    else:
        subj = f"[MrTrader] {event_type}"
        body = f"<pre>{json.dumps(p, indent=2, default=str)}</pre>"

    html = f"""<!DOCTYPE html>
<html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  color:#1a1a1a;max-width:700px;margin:0 auto;padding:20px">
  {body}
  <hr style="border:none;border-top:1px solid #ddd;margin-top:32px">
  <p style="color:#888;font-size:12px">{ts} · MrTrader automated notifier</p>
</body></html>"""
    return subj, html


def _section(title: str, rows: list[tuple[str, Any]]) -> str:
    cells = "".join(
        f"<tr>"
        f"<td style='padding:5px 16px 5px 0;color:#666;white-space:nowrap;vertical-align:top'>{k}</td>"
        f"<td style='padding:5px 0'><strong>{v}</strong></td>"
        f"</tr>"
        for k, v in rows
        if v not in (None, "", [])
    )
    return (
        f"<h2 style='border-bottom:2px solid #0066cc;padding-bottom:8px;color:#0066cc'>{title}</h2>"
        f"<table style='border-collapse:collapse;margin-bottom:16px'>{cells}</table>"
    )


# ── SMTP ──────────────────────────────────────────────────────────────────────

def _smtp_send(subject: str, html_body: str) -> bool:
    user = os.environ.get("NOTIFY_GMAIL_USER", "").strip()
    pw = os.environ.get("NOTIFY_GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    if not user or not pw:
        log.error("NOTIFY_GMAIL_USER / NOTIFY_GMAIL_APP_PASSWORD not set in env")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"MrTrader <{user}>"
    msg["To"] = RECIPIENT
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
        s.ehlo()
        s.starttls(context=ctx)
        s.login(user, pw)
        s.sendmail(user, [RECIPIENT], msg.as_string())
    log.info("email sent: %s -> %s", subject, RECIPIENT)
    return True
