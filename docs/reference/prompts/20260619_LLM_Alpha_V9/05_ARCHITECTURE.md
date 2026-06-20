# 05 — System Architecture

We want your read on whether this is sound + where it should improve.

## Runtime (live paper)
- **In-process FastAPI app** (`uvicorn`) hosts the orchestrator: an **APScheduler** loop that
  fires the agents on a calendar.
- **Three agents:**
  - **Portfolio Manager (PM)** — proposes target positions per sleeve (trend, [PEAD off], cash),
    reads live config from the DB (no restart to change weights/flags).
  - **Risk Manager (RM)** — enforces caps: 80% gross exposure, per-name / per-sector /
    per-strategy-budget limits, kill-switch, drawdown peak tracking.
  - **Trader** — turns approved targets into Alpaca orders; a reconciler keeps DB ↔ broker in sync.
- **Sleeves are independent, config-gated modules** (`app/live_trading/*`): `trend_sleeve`,
  `cash_sleeve`, `crypto_paper_track`, each with shadow/live flags + per-order DB commit +
  decision-audit trail. Adding a sleeve = a new module + a scheduled trigger + config keys.
- **Storage:** Postgres (state, trades, config) + Redis (queue/cache). React dashboard for
  monitoring (positions, agent event log, PEAD/trend tabs, decision audit).
- **Notifications:** an email queue (`notifier`) drained by a watcher (phase-complete, weekly
  rollups, etc.).

## Research / validation pipeline (offline)
- **`docs/living/PIPELINE_ARCHITECTURE.md`** is our internal SSOT (not attached; ask if needed).
- **Simulators:** `AgentSimulator` (swing WF/CPCV, daily MTM), `IntradayAgentSimulator` (intraday
  WF/CPCV), `StrategySimulator` (tier-2 only).
- **Walk-forward + CPCV** (combinatorial purged cross-validation) with **85-calendar-day purge**
  (swing) / 2-trading-day (intraday) and a **sacred holdout (2026-11-09)** never touched.
- **Sleeve Lab** (`scripts/walkforward/sleeves.py` + `sleeve_lab.py`): a uniform pipeline —
  declare a sleeve (data fetch → vectorized PIT backtest → returns) → run through the Ruler-v2
  two-track gate → Track-B vs the live book. New premia are ~20-line declarations here.
- **Research registry / pre-registration ledger** (`app/research/registry.py`): every hypothesis
  is pre-registered with frozen acceptance criteria before the decisive run (the program's true
  N_TRIALS; we explicitly fight p-hacking / goalpost-moving — there's a history of *reverting*
  a post-hoc carve-out a reviewer flagged as goalpost-moving).

## Engines (the reusable math)
- `app/strategy/tsmom.py` — the TSMOM engine (signals, inverse-vol sizing, gross cap, optional
  book-vol-target overlay, calendar **or** no-trade-band rebalance). Drives ETF trend, crypto,
  futures trend.
- `app/research/futures_data.py` + `futures_carry.py` — correct difference-adjusted futures
  returns + term-structure carry (reads the local Norgate parquet mirror).
- `app/research/inference.py` — the inference keystone (HAC Sharpe significance, stationary
  bootstrap, PBO/CSCV, multi-factor residual-α). Pure, unit-tested vs known answers.

## Engineering posture
- Heavy test suite (~3,600 tests); CI shards + lint + coverage gate; one-branch-per-change,
  squash-merge, no-drift doc rule (a change updates all docs it makes stale in the same PR).
- Every model/strategy change goes through: build (Opus) → independent adversarial deep-dive
  (Opus) → fix-iterate → tests → docs → merge. We recently ran a 4-agent adversarial review of
  the futures pipeline and found/fixed 5 real bugs (sign-flip, look-ahead, staleness, etc.).

## What we'd love your verdict on (architecture)
- Is the **single-process orchestrator + agent** design appropriate, or should research/execution
  be more decoupled? Any resilience risks (single point of failure, state/broker drift)?
- Is the **Sleeve-Lab + two-track-gate + pre-registration** approach the right backbone for
  finding alpha, or is it structurally biasing us (e.g., toward slow, low-turnover sleeves only)?
- What's missing to make this a genuinely *resilient* multi-strategy book (risk overlay,
  regime allocation, position-level risk, capital allocation across sleeves)?
