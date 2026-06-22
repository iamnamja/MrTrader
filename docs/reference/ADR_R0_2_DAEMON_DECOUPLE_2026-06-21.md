# ADR R0.2 — Decouple the trading daemon from FastAPI (phased; default-off)

**Status:** Phases 1-2 IMPLEMENTED (default-off); Phases 3-4 PROPOSED (owner-gated). **Date:** 2026-06-21 (P2: 2026-06-22).
**Context:** Portfolio-Brain roadmap R0.2. Today the orchestrator + PM/RM/Trader agents + scheduler +
news monitor all boot **inside the FastAPI `lifespan`** (`app/main.py:239-407`), so a web restart
stops trading and trading load shares the web process. The roadmap + the external critics call this
the prerequisite for live futures capital — but also "the single largest, riskiest item."

## The good news (from the 2026-06-21 boundary inventory)
The codebase is **already highly modular** for this split:
- The orchestrator and all three agents are **module-level singletons**; `orchestrator.start()`
  already spawns each agent as an asyncio task + the scheduler.
- **State that matters already persists to Postgres:** `kill_switch.active`, `capital_ramp.stage_*`
  (config table); agents already write decisions to `agent_decisions`; `trades` already has
  `highest_price`/`bars_held` columns for Trader-cache rebuild.
- A **"run both in one process" fallback needs ZERO code changes** — it is exactly today's behavior.
- Inter-agent messaging is **already Redis queues** (`trade_proposals`, `trader_approved_trades`,
  …); the web layer does **not** push to them.

So the split is primarily: (a) a standalone daemon entry point, (b) a mode flag, (c) an IPC bridge
for the handful of **web→agent direct calls** (manual triggers, pause/resume, kill-switch, capital
advance), and (d) bridging the in-process **WebSocket broadcast** + **news-monitor watchlist**.

## What moves vs stays
- **`mtrader-tradingd` (new process):** orchestrator + PM/RM/Trader + scheduler + news monitor.
- **`mtrader-api` (web, read-only):** dashboard REST + WS; Postgres/Alpaca **reads**; control
  commands become **Postgres/Redis-mediated** writes the daemon consumes.

## The couplings to convert (web→agent direct calls today → mediated)
| Coupling | Today | After |
|---|---|---|
| kill-switch | in-mem `kill_switch._active` (+persists) | already Postgres `kill_switch.active`; daemon polls |
| capital ramp | in-mem stage (+persists) | already Postgres `capital_ramp.*`; daemon owns advance |
| pause/resume | `orchestrator.pause_trading()` | Postgres `orchestrator_state` row; daemon polls |
| manual triggers (swing/intraday/retrain) | `orchestrator.agents[..].method()` | Redis `pm_commands` queue (already exists, unused) |
| WebSocket agent-decision push | in-process `broadcast_*` | agents write `agent_decisions` (already do) → web tails / Redis pub-sub |
| news watchlist | in-mem `_watched` set | Postgres `news_watchlist`; both sync on poll |
| Trader mid-run cache (highest_price/bars_held) | in-mem `active_positions` | persist to `trades` (columns exist); reconciler rebuilds |

## Riskiest seams (+ mitigations)
1. **Double-running agents** — if the daemon boots the orchestrator AND the web process also does, two
   Traders trade. *Mitigation:* the mode flag makes exactly ONE process boot the orchestrator;
   default `in_process` = web boots it (today); `subprocess` = ONLY the daemon does.
2. **Boot-path regression** — a bug in the `lifespan` flag-guard could break startup (no web, no
   trading). *Mitigation:* the guard defaults to today's exact behavior; do the change in a
   **dedicated restart with the operator present**, NOT bundled with the shadow-gate restart.
3. **Web control routes dark in subprocess mode until bridged** — kill-switch/pause/triggers call the
   in-process orchestrator. *Mitigation:* Phase 2 bridges them via Postgres/Redis before flipping to
   subprocess; until then those routes return "managed by daemon".
4. **WebSocket dashboard goes quiet** in subprocess mode (broadcasts are in-process). *Mitigation:*
   Phase 3 (Redis pub-sub or DB-tail); acceptable degradation meanwhile (dashboard still reads DB).

## Phased plan (each phase independently shippable; default-off until the final flip)
- **Phase 1 — capability, inert. ✅ DONE 2026-06-21.** Added `MRTRADER_DAEMON_MODE` (default
  `in_process`) + a standalone daemon (`python -m app.tradingd`) that runs the brain (orchestrator +
  agents + scheduler + news monitor) with NO FastAPI. The brain lifecycle + preamble (state restore /
  reconcile / queue flush) + shutdown-hardening helpers moved to `app/trading_runtime.py` (imports no
  FastAPI) so both boot paths share identical code; `main.py` lifespan boots the brain only when
  `web_boots_brain()` (i.e. `mode != subprocess`). **Default = byte-identical to pre-R0.2** (web boots
  everything; the daemon refuses to start unless `mode=subprocess`). **Mutual-exclusion interlock:**
  the web boots the brain unless subprocess, and the daemon boots ONLY if subprocess → exactly one
  process ever runs the brain (the "double-running agents" risk cannot occur). Files:
  `app/trading_runtime.py`, `app/tradingd.py`, `app/main.py` (guards); `tests/test_trading_runtime.py`
  (10). Full suite green; flake8 clean. **Inert: merging changes nothing until the env var is set.**
- **Phase 2 — the IPC bridge. ✅ DONE 2026-06-22.** `app/control_bridge.py` (new, no FastAPI import):
  a command bus over the existing Redis `pm_commands` queue + a daemon consumer + a Postgres
  state-sync poll. The control routes are **mode-conditional** via `bridge_or_none()`: in_process they
  run their direct path UNCHANGED (byte-identical); in subprocess they EMIT a command and DO NOT touch
  the web's empty orchestrator. Bridged: pause/resume, manual triggers (cycle/swing/retrain/intraday),
  job pause/resume, capital-advance. Kill-switch keeps closing positions from the web (Alpaca-reachable)
  + persists to Postgres; the daemon picks up the halt via the **3 s `state_sync_loop` reload** plus an
  **immediate `reload_state` command** on activate/reset (agents/sleeves read `kill_switch.is_active`
  live, so a reload is honored). The daemon (`tradingd.py`) launches `consume_control_commands` +
  `state_sync_loop`; long triggers run as tracked background tasks (a 7-min retrain never blocks a
  pause/kill); the consumer never dies on a bad command and cancels in-flight triggers on shutdown.
  Three deep-dive passes (1 self + 2 independent), full suite green, flake8 clean. Files:
  `app/control_bridge.py`, `app/api/orchestrator_routes.py`, `app/api/routes.py`, `app/tradingd.py`;
  `tests/test_control_bridge.py` (23). *Exit: every control route works with the brain in a separate
  process.* **Known limitations (subprocess-only, documented):** (a) a bounded kill-propagation window
  (≤ command latency, 3 s backstop) vs zero in in_process; (b) capital-ramp state is in-memory — full
  ramp bridging (go-live `start()` + `/live/status` display) is finished in R1 go-live; (c) the PM
  trigger bodies are duplicated route↔bridge (extract to public PM methods in a later pass).
- **Phase 3 — observability bridge.** Agent-decision push + news watchlist + Trader mid-run cache via
  Postgres/Redis so the dashboard stays live and news-exits fire in subprocess mode.
- **Phase 4 — flip + harden.** Set `MRTRADER_DAEMON_MODE=subprocess` (web = read-only); add the
  external dead-man watchdog (R0.4 `kill_switch_state` persisted) + systemd units (Linux prod host).
  *This is the step that makes "web restart ≠ trading restart" real — the gate to live futures capital.*

## Recommendation
Phase 1 shipped **default-off and inert** — merging it changes nothing (the default `in_process` mode
is byte-identical to today; the daemon refuses to run unless `mode=subprocess`), so no special restart
was needed to land it. The **behavior-changing step is the flip to `subprocess`** (Phase 4): that is
what must be done **operator-present**, after Phases 2-3 bridge the web's control + observability paths
(otherwise the dashboard's kill-switch/pause/trigger routes go dark while the daemon owns the brain).
Phases 2-4 follow incrementally, each behind the default-off flag, so the live book is never at risk
from a half-built phase. The fallback (`in_process`) means we can stop after any phase.

**How to use the daemon (dev / once Phase 4 lands):** run the web read-only with
`MRTRADER_DAEMON_MODE=subprocess` in its environment, and start the brain separately with
`MRTRADER_DAEMON_MODE=subprocess python -m app.tradingd`. In the default deployment neither env var is
set and nothing changes.

> Cross-refs: boundary inventory (this session); `PORTFOLIO_BRAIN_ROADMAP_2026-06-21.md` R0.2;
> `R0_FOUNDATION_2026-06-21.md` (the read-only substrate the daemon will host).
