# While-Away Runbook (unattended operation)

**Written 2026-06-25 for a ~1-week absence. Account = Alpaca PAPER. Low-turnover book: trend ETFs
(weekly Monday rebalance) + cash/T-bill sleeve (daily). Swing-ML / intraday / PEAD are OFF.**

> **Stakes are low:** it's paper. If the machine dies, positions just **idle at the broker — nothing
> liquidates**. The realistic failure is the brain crashing; the dead-man watchdog catches that.

---

## Before you leave — checklist

1. **Start the dead-man watchdog** (the #1 unattended safety — alerts you if the brain dies/hangs).
   Run in a **persistent terminal** that stays open all week, alert-only (no auto-flatten):
   ```
   PYTHONPATH=. venv/Scripts/python scripts/dead_man_watchdog.py
   ```
   It reads the 1-min heartbeat file and emails `[CRITICAL] dead_man_alert` if it goes stale. It only
   **enqueues** — `notify_watcher` (which has the SMTP creds) does the sending, so the watchdog needs
   no creds of its own. Alerts once per episode, re-arms on recovery.

2. **Phone notifications for the alerts (Option 1 — pure email, no webhook).** Every alert already
   emails `kimminjae@gmail.com`; catastrophic ones are subject-prefixed **`[CRITICAL] `**. Make your
   phone buzz on them:
   - **iPhone:** add the sender to **VIP** + enable **Emergency Bypass** (rings through Do-Not-Disturb).
   - **Android (Gmail):** filter `subject:[CRITICAL]` (or `[MrTrader]`) → label → **High-priority**
     label notifications + custom sound.
   - A **test** `[CRITICAL]` email was sent 2026-06-25 (subject "ALERTING SELF-TEST — PLEASE IGNORE").

3. **(Recommended, optional) one uvicorn restart** to activate H3 (the pre-trade fat-finger order
   cap — a good backstop to have active while unattended) + H2 SM (shadow). Do it with a buffer:
   confirm `GET /health` shows `status: healthy` and a fresh heartbeat for ~15 min **before** you
   leave. If you'd rather not touch a healthy process, skip it — H3 is additive, not a fix.

4. **Confirm `notify_watcher` is running** (it drains the email queue). `notify_watcher.log` should be
   active. Without it, no alert email is sent.

---

## Do NOT do while away
- **Do not flip any shadow gate → enforce.** `pm.reconciliation_mode`, `pm.whole_book_gate_mode`,
  `pm.kill_switch_sm_mode` stay **shadow** (the enforce flip is an owner-present step, ~after a clean
  7-day soak). They log/observe only; flipping unattended could HOLD a legit rebalance.
- No new deploys, no flag flips, no config changes.

---

## What you'll receive (and what it means)
| Email | Meaning | Action |
|---|---|---|
| `[MrTrader] trend weekly` + `cash weekly` (Monday) | The weekly rebalance ran | None — confirms alive + trading |
| `[CRITICAL] dead_man_alert` | Brain died/hung (heartbeat stale) | Halt or restart (below); the book idles safely meanwhile |
| `[CRITICAL] SAFETY GATE ERROR` (`gate_error`) | A safety gate couldn't evaluate (it fails-safe) | Note it; not urgent (gates are shadow) |
| `reconciliation_break` / `whole_book_gate_breach` | A shadow gate WOULD-hold (observation only) | Note it; no live effect in shadow |
- **No daily "all's well" email is wired** — silence is normal/healthy. Your positive signal is the
  Monday rebalance emails; your negative signal is a `[CRITICAL]` alert.
- **Heads-up:** macro events **Jun 30 (ISM) → Jul 2 (NFP/jobless/unemployment)** will exercise the
  now-live F12 macro layer (paper) — the Macro Intel panel risk will move and sizing/tighten may
  fire. Expected; deep-dived.

## Auto-protection with no human
- **Circuit breaker** — auto-pauses on a VIX spike / consecutive losses / errors; **auto-resumes**
  when DB+Redis+Alpaca are healthy again.
- **2% daily-loss hard stop** — load-bearing intraday.
- (The drawdown ladder is still SHADOW — it won't auto-de-gross. Acceptable for a 1-week paper book.)

---

## If you need to act remotely
Requires reachability to the dashboard/API at `http://localhost:8000` (VPN/tunnel from home).
- **Halt trading (kill switch):** `POST /control/kill-switch` or `POST /live/kill-switch` (flattens +
  persists; survives restart). Reversible only by an explicit reset.
- **Pause (reversible):** `POST /control/pause`.
- **Close one symbol:** `POST /control/close-position/{symbol}` (short-aware).
- **Out-of-band flatten (no app dependency)** — from a terminal on the box:
  ```
  PYTHONPATH=. venv/Scripts/python scripts/emergency_flatten.py            # DRY-RUN (safe, shows what it'd do)
  PYTHONPATH=. venv/Scripts/python scripts/emergency_flatten.py --execute  # actually flatten
  ```
- **Restart the brain:** restart uvicorn; on boot it reconciles positions from the broker, restores a
  standing pause / kill-switch, and purges any phantom intraday position.

## Total-machine-death (power/OS)
On-box monitors can't alert if the whole box dies. **It's paper**, so the only effect is a missed
rebalance until you return (positions persist at Alpaca, nothing liquidates). If you want
external detection anyway: a free dead-man's-snitch (e.g. healthchecks.io) that emails you when the
box stops pinging — ask and it's ~5 lines in the heartbeat job (deferred; low value for paper).
