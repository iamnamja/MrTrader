# ADR R0.2 — Decouple the trading daemon from FastAPI (phased; default-off)

**Status:** PROPOSED (execution owner-gated — touches the live boot path). **Date:** 2026-06-21.
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
- **Phase 1 — capability, inert.** Add `MRTRADER_DAEMON_MODE` (default `in_process`) + a standalone
  `mtrader-tradingd` entry point that runs `orchestrator.start()` + the scheduler with NO FastAPI.
  In `main.py` lifespan, guard the orchestrator/news-monitor boot behind `mode != "subprocess"`.
  **Default = byte-identical to today** (web boots everything; the daemon entry point is never run).
  *Exit: default config unchanged; `python -m app.tradingd` can run the brain standalone in a dev box.*
- **Phase 2 — the IPC bridge.** Convert the web→agent control routes (kill-switch, pause/resume,
  manual triggers, capital advance) to Postgres/Redis-mediated commands the daemon consumes; the
  daemon polls `orchestrator_state` + drains `pm_commands`. *Exit: every control route works with the
  brain in a separate process.*
- **Phase 3 — observability bridge.** Agent-decision push + news watchlist + Trader mid-run cache via
  Postgres/Redis so the dashboard stays live and news-exits fire in subprocess mode.
- **Phase 4 — flip + harden.** Set `MRTRADER_DAEMON_MODE=subprocess` (web = read-only); add the
  external dead-man watchdog (R0.4 `kill_switch_state` persisted) + systemd units (Linux prod host).
  *This is the step that makes "web restart ≠ trading restart" real — the gate to live futures capital.*

## Recommendation
Execute **Phase 1 in a dedicated step with the operator present** (so the boot-path restart is
watched), **separate from the shadow-gate uvicorn restart** (don't compound two boot changes in one
restart). Phases 2-4 follow incrementally, each behind the default-off flag, so the live book is
never at risk from a half-built phase. The fallback (`in_process`) means we can stop after any phase.

> Cross-refs: boundary inventory (this session); `PORTFOLIO_BRAIN_ROADMAP_2026-06-21.md` R0.2;
> `R0_FOUNDATION_2026-06-21.md` (the read-only substrate the daemon will host).
