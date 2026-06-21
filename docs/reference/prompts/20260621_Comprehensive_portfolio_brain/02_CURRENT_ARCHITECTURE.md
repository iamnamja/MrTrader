# 02 — Current architecture (how the live system actually works today)

Precise as of 2026-06-21. This is the *as-built* runtime, not an aspiration.

## Stack
- **In-process FastAPI app** (`app/main.py` lifespan) hosts everything: a React dashboard API +
  the trading orchestrator in the **same process**.
- **Orchestrator** (`app/orchestrator.py`) spawns one asyncio task per agent + an **APScheduler**
  (`app/scheduler.py`, `AsyncIOScheduler`, tz America/New_York).
- **Postgres** (SQLAlchemy) = persistent state. **Redis** = inter-agent queues + transient state
  (kill-switch flag, capital stage).
- **Broker: Alpaca only**, wired **directly** (`app/integrations/alpaca.py`) — no abstraction layer.
- **LLM usage today:** Claude is already called for (a) the **News/Macro Intelligence** sizing
  signals (NIS Tier-1 macro classification, Tier-2 per-symbol news), (b) optional **reject-reason
  explanations** from the Risk Manager, and (c) this design/research process. Every call is logged
  to `llm_call_log`. LLMs are **not** in the deterministic sizing/execution path.

## The three agents (the proposal→risk→execute pipeline)
```
[Portfolio Manager] --proposals(Redis: trade_proposals)--> [Risk Manager] --approved(Redis: trader_approved_trades)--> [Trader] --> [Alpaca]
        ^                                                                                                                  |
        └------------------------------ pm_reeval_requests / trader_exit_requests (Redis) ---------------------------------┘
```
- **Portfolio Manager** (`app/agents/portfolio_manager.py`) — 60s heartbeat. Runs the ML selectors
  (swing/intraday) + PEAD, produces trade **proposals** (symbol, dir, qty, entry, score, uuid),
  pushes to Redis `trade_proposals`. Re-scores open positions; requests exits. **These ML selectors
  are currently OFF** (swing/intraday/PEAD all turned off after kills).
- **Risk Manager** (`app/agents/risk_manager.py`) — the **sole gatekeeper** for the proposal path.
  Pops proposals, runs ~10-14 rules **in order, first-failure-rejects** (buying power, position size
  ≤5%, daily loss ≤2%, max 5 open, sector ≤20%, **beta ≤1.30**, factor ≤60%, **return correlation
  ≤0.75 vs open positions**, spread, ADTV, portfolio heat ≤6%, gross ≤80%, strategy budgets). If
  pass → Redis `trader_approved_trades`; else reject + log.
- **Trader** (`app/agents/trader.py`) — drains approved queue, places **market/limit orders** via
  Alpaca, manages stops/targets/exits, reconciles Alpaca vs DB on startup + every ~15 min.

## The live sleeves — and the critical asymmetry
**The strategies that are actually LIVE bypass the agent pipeline above.** They are independent
scheduled rebalancers that trade **directly via Alpaca** with their own lightweight per-order gate:

| Sleeve | File | Cadence (ET) | Path | Risk handling |
|---|---|---|---|---|
| **ETF trend** (LIVE, ~50% gross) | `app/live_trading/trend_sleeve.py` | Weekly, Mon 09:45 | computes TSMOM weights → deltas → **direct Alpaca orders** | own `apply_risk_gate` (fat-finger + shared 80% gross cap) + VIX-term crash governor; **does NOT go through the RiskManager** |
| **Cash / T-bills** (LIVE) | `app/live_trading/cash_sleeve.py` | Weekly, Mon 09:50 | parks idle settled cash in SGOV/BIL → **direct Alpaca** | T-bills **excluded** from the 80% gross cap; bypasses RM |
| **Crypto-trend** (report-only) | `app/live_trading/crypto_paper_track.py` | Weekly, Mon 09:55 | recomputes book on live closes → **logs only, no orders** | n/a |

**So the RiskManager's correlation/beta/heat/sector checks apply only to the (currently-off)
proposal-driven strategies — NOT to what is actually trading live.** The trend sleeve and cash
sleeve coordinate only through a shared 80% gross cap and a tag (`selector`) that tells the Trader's
exit loop to leave them alone. There is no joint sizing, no cross-sleeve correlation awareness, no
netting.

## State stores
- **Postgres** (`app/database/models.py`): `trades` (source of truth for "open"), `orders`
  (immutable fill ledger), `trade_proposals`, `proposal_log`/`proposal_event` (lineage),
  `risk_metrics` (EOD daily snapshot: gross, sector/position concentration, drawdown),
  `agent_decisions`, `decision_audit`, `calendar_events`, `macro_signal_cache`, `news_signal_cache`,
  `llm_call_log`.
- **Redis**: queues (`trade_proposals`, `trader_approved_trades`, `trader_exit_requests`,
  `pm_reeval_requests`, `pm_commands`) + kill-switch flag + capital stage.
- **In-process dicts**: Trader's `active_positions`, `approved_symbols` (fast-path state).

## "What's open" vs "what's being considered" — today
- **Open:** `trades` WHERE status in (ACTIVE, PENDING_FILL) ∪ live Alpaca positions (Trader keeps an
  in-memory mirror for fast exit checks).
- **Being considered:** scattered — `trade_proposals` (PENDING), the `trader_approved_trades` queue,
  and the Trader's in-memory `approved_symbols`. **There is no single consolidated object** that
  represents the whole book (open + pending + exposures) in real time. Risk is evaluated
  **per-proposal**, atomically, against an account snapshot; the only book-level view is the **EOD**
  `risk_metrics` row.

## Scheduler jobs (weekday, ET)
health_check (5 min) · portfolio_selection_trigger 09:30 · trend_rebalance 09:45 (Mon) ·
cash_rebalance 09:50 (Mon) · crypto_paper_track 09:55 (Mon) · options_nbbo_logger 15:55 ·
daily_session_summary 16:15 · model_retraining 17:00. Weekly jobs have a 1800s misfire grace.

## What's coming (and why this matters now)
**IBKR futures** (carry/xsmom/VRP) is days away — a **second venue** with its own positions, cash,
and margin. There is **no broker abstraction**, so today it would be added as **another silo**
(a parallel `ibkr.py` + per-sleeve routing), compounding the coordination gap. That is the forcing
function for this redesign.
