# MrTrader Portfolio Brain Architecture Review

**Date:** 2026-06-21  
**Prepared for:** MrTrader  
**Role assumed:** buy-side systematic trading systems architect / quant infrastructure lead / distributed-systems engineer  
**Scope:** architecture, data model, risk-budgeting, cross-venue execution, safety, LLM placement, and migration path for a cohesive multi-strategy, multi-venue portfolio brain.

---

## Executive verdict

The current MrTrader architecture should not be extended by adding IBKR futures as another sleeve with its own scheduler, sizing, and broker routing. That would institutionalize the exact failure mode you are trying to remove: independent strategy silos that trade into the same capital pool without a single book-level decision.

My strong recommendation is to **replace the PM/RM/Trader mental model as the primary live architecture** with a **target-book architecture**:

```text
Strategy Intents
    -> Portfolio Constructor / Risk Budgeter
    -> Whole-Book Pre-Trade Risk Gate
    -> Netting + Execution Planner
    -> Broker Adapters
    -> Fills + Reconciliation
    -> Immutable Book Decision Ledger
```

The old PM/RM/Trader path can remain temporarily as a legacy adapter, but the live book should move to a new rule:

> **No strategy places orders. No sleeve owns capital. No venue has a separate book. Every strategy emits an intent; the portfolio brain decides one target book; the execution planner trades the minimum safe set of deltas to get there.**

For your size, cadence, and solo-operator constraint, the right design is not a microservice platform, not an optimizer-heavy institutional stack, and not a clever LLM-driven “agent” architecture. It is a boring, deterministic, auditable control system with a clean book-state model, a robust risk-budgeting layer, broker adapters, and a hard fail-closed posture.

The north-star is simple:

> **MrTrader should become a weekly target-portfolio machine. At each rebalance, it freezes broker reality and market data, collects every strategy’s desired model portfolio, constructs one constrained target book under a single risk budget, nets everything across strategies and venues, runs whole-book risk and margin checks, executes through broker adapters using idempotent orders, reconciles broker reality back to the ledger, and produces a reproducible audit trail.**

The first three things I would build are:

1. **Canonical book-state + event/snapshot ledger.** Without this, every other layer is reasoning over fiction.
2. **A separate trading daemon with broker abstraction and global kill-switch.** FastAPI should not host the trading brain.
3. **Strategy-intent → book-decision → execution-plan pipeline, initially in shadow mode for ETF trend and cash.** Prove the new brain can reproduce today’s book before it controls it.

The single biggest mistake a solo operator makes here is building a sophisticated allocator or “portfolio optimizer” before the state, reconciliation, and idempotency layers are correct. You do not yet need a Nobel Prize optimizer. You need one source of truth, one risk budget, one deterministic decision ledger, and one way to halt the entire book.

---

## 1. What I would tear down, and what I would keep

### 1.1 The PM/RM/Trader model is the wrong abstraction for the live book

The current PM/RM/Trader design is a **proposal pipeline**:

```text
PM proposes individual trades
RM approves/rejects each trade
Trader executes approved trades
```

That abstraction is acceptable for a discretionary-like stream of independent trade proposals. It is not the right abstraction for a weekly, multi-strategy systematic book.

The reason is fundamental: the live problem is not “should I accept this trade?” The live problem is:

> **Given the entire current book, every pending order, every strategy’s target, the current margin/cash state, factor exposures, and the shared drawdown budget, what should the whole book be after this rebalance?**

A per-trade risk manager is structurally late. It sees a proposed trade, not the target book. It can reject bad trades, but it cannot construct the best netted portfolio. It cannot decide that two strategies should both be scaled down because correlations spiked. It cannot know that an ES futures long and SPY ETF long are effectively stacked equity beta unless everything flows through it before orders are emitted.

So I would **not** preserve PM/RM/Trader as the live design. I would demote it.

### 1.2 What replaces it

The replacement is a **Target Book Pipeline**.

```text
                            ┌────────────────────────────┐
                            │  Operator / Dashboard       │
                            │  read-only + commands       │
                            └──────────────┬─────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Trading Daemon                                  │
│                                                                         │
│  ┌───────────────┐   ┌────────────────┐   ┌─────────────────────────┐   │
│  │ Preflight +   │   │ Book State     │   │ Strategy Intent Engine  │   │
│  │ Reconciler    │-->| Builder        │-->| all strategies emit      │   │
│  │ broker truth  │   │ current book   │   │ desired model books      │   │
│  └──────┬────────┘   └───────┬────────┘   └───────────┬─────────────┘   │
│         │                    │                        │                 │
│         ▼                    ▼                        ▼                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Portfolio Constructor / Risk Budgeter                              │  │
│  │ one target book, strategy risk weights, robust correlation model    │  │
│  └─────────────────────────────┬─────────────────────────────────────┘  │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Whole-Book Pre-Trade Risk Gate                                     │  │
│  │ factor limits, gross/net, margin, drawdown, stress, staleness       │  │
│  └─────────────────────────────┬─────────────────────────────────────┘  │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Netting + Execution Planner                                        │  │
│  │ target-current deltas, order plan, risk-reducing first              │  │
│  └─────────────────────────────┬─────────────────────────────────────┘  │
│                                ▼                                        │
│  ┌──────────────────────┐    ┌──────────────────────┐                  │
│  │ Alpaca Adapter       │    │ IBKR Adapter          │   future: crypto │
│  │ equities/ETFs        │    │ futures/margin        │   options/etc.   │
│  └──────────┬───────────┘    └──────────┬───────────┘                  │
│             ▼                           ▼                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Fill Capture + Reconciliation + Immutable Decision Ledger           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

Postgres = immutable decisions, snapshots, events, orders, fills, reconciliation
Redis    = ephemeral locks/notifications/kill flag only, never source of truth
FastAPI  = dashboard/API process, not trading brain
```

This design has a hard separation of concerns:

| Component | Owns | Must not own |
|---|---|---|
| Strategy module | Signal logic and raw desired exposure | Capital, broker calls, final sizing |
| Portfolio constructor | Risk allocation and target book | Broker mechanics, fills |
| Risk gate | Whole-book admissibility | Alpha logic |
| Execution planner | Deltas and order sequencing | Portfolio optimization |
| Broker adapter | Venue mechanics | Strategy/risk policy |
| Reconciler | Broker reality vs ledger | Alpha decisions |
| Dashboard/API | Visibility and operator commands | Scheduled live trading loop |

The important shift is this:

> **Risk management is not a downstream approval queue. It is embedded into target construction and then enforced again as a whole-book pre-trade gate.**

---

## 2. Runtime topology: right-sized for a solo operator

### 2.1 Do not run the trading brain inside FastAPI

Your current topology couples the web server and trading orchestrator. That is fine for a research app. It is not acceptable before adding IBKR futures.

I would split into **two processes**, not ten services:

```text
Process 1: mtrader-api
- FastAPI
- React dashboard API
- Read-only book/risk views
- Operator commands: request rebalance, set kill switch, approve held plan, annotate
- No scheduled trading jobs
- No broker order placement except maybe manual emergency command routing to daemon

Process 2: mtrader-tradingd
- APScheduler or simple schedule loop
- Book-state builder
- Strategy intent collection
- Portfolio construction
- Risk gate
- Execution planner
- Broker adapters
- Reconciliation loop
- Kill-switch enforcement
```

Optional third process later:

```text
Process 3: mtrader-watchdog
- Minimal health/dead-man monitor
- Can set HALT_NEW_RISK if tradingd heartbeat is stale
- Can cancel open orders if explicitly configured
- Should not automatically flatten unless you deliberately enable that mode
```

Do **not** introduce Kafka, Kubernetes, Celery sprawl, or a distributed event bus right now. You are weekly/EOD, solo-operated, and running modest capital. Postgres plus one trading daemon is enough.

### 2.2 Scheduled vs event-driven

Use a **hybrid control model**:

- **Scheduled for book decisions**: weekly rebalance is a deliberate, atomic decision.
- **Event-driven inside the run**: fills, broker updates, kill-switch changes, reconciliation breaks, and operator approvals should be processed as events.
- **Periodic reconciliation heartbeat**: every 5–15 minutes while markets are open, and immediately before any rebalance.

Do not let individual sleeves fire at 09:45, 09:50, 09:55. That is exactly the silo problem.

Instead:

```text
Monday 09:30 ET: preflight + broker reconciliation
Monday 09:35 ET: freeze market/account/book snapshot
Monday 09:36 ET: collect all strategy intents
Monday 09:38 ET: construct target book
Monday 09:40 ET: risk gate + execution plan
Monday 09:45 ET: execute one netted book plan
Monday 10:00 ET: reconcile and report
```

The exact times can change, but the principle cannot:

> **One rebalance run, one frozen input snapshot, one target book, one execution plan.**

---

## 3. The single source of truth: book-state model

### 3.1 The uncomfortable truth: the broker is reality, Postgres is memory

You should not think of Postgres as “the truth” about positions. Brokers are truth for live positions, cash, margin, and orders. Postgres is the immutable memory of what you intended, what you sent, what you observed, and how you reconciled it.

So the source-of-truth model should be:

```text
Broker snapshots = external reality
Postgres ledger  = immutable decision/audit history
BookState actor  = deterministic current view derived from:
                   broker snapshots
                 + open orders
                 + fills
                 + instrument master
                 + strategy ownership ledger
                 + pending current rebalance plan
```

The **BookState** object is the canonical in-memory decision object. It is rebuilt every run from Postgres + broker snapshots. It is not a hidden in-memory dict that becomes a second reality.

### 3.2 The three states you must distinguish

You need three separate but linked concepts:

#### A. Physical book

What the brokers actually hold.

```text
Alpaca:
  SPY +100 shares
  SGOV +500 shares

IBKR:
  ES Sep26 +1 contract
  CL Aug26 -1 contract
```

#### B. Virtual strategy ownership

How you attribute the physical book back to strategies for P&L, risk budgets, and review.

```text
SPY physical +100:
  ETF_TREND owns +70 equivalent shares
  CASH owns 0
  FUT_XSMOM owns +30 SPY-beta-equivalent exposure through ES, not SPY shares

ES physical +1:
  FUT_XSMOM owns +0.7
  FUT_CARRY owns +0.3
```

Do not confuse virtual sleeve attribution with physical trading. Strategies should not have separate broker sub-books unless you are using true subaccounts. You are running one book.

#### C. Proposed book

What the book would look like if all current strategy intents were accepted and netted.

```text
Current physical book
+ pending open orders
+ strategy intents this cycle
+ proposed target changes
= proposed target book
```

The pre-trade risk gate should run on the proposed book, not on individual orders.

### 3.3 Canonical BookState object

A minimal book-state object should look like this conceptually:

```python
@dataclass(frozen=True)
class BookState:
    as_of: datetime
    run_id: str | None
    base_currency: str

    account: ConsolidatedAccount
    positions: list[CanonicalPosition]
    open_orders: list[CanonicalOrder]
    strategy_allocations: list[VirtualAllocation]

    exposures: ExposureVector
    factor_exposures: FactorExposureVector
    risk_state: RiskState
    margin_state: MarginState

    data_freshness: dict[str, FreshnessStatus]
    reconciliation_status: ReconciliationStatus
    kill_state: KillState
```

A canonical position:

```python
@dataclass(frozen=True)
class CanonicalPosition:
    instrument_id: str
    venue: str                 # ALPACA, IBKR
    broker_symbol: str
    asset_class: str           # ETF, EQUITY, FUTURE, CRYPTO, OPTION
    quantity: Decimal
    price: Decimal
    multiplier: Decimal
    currency: str
    market_value_base: Decimal
    notional_base: Decimal
    delta_equivalent_notional: Decimal
    strategy_tags: dict[str, Decimal]  # virtual ownership weights
```

A factor exposure vector:

```python
@dataclass(frozen=True)
class FactorExposureVector:
    equity_beta_usd: float
    rates_dv01_usd: float
    usd_fx_delta: float
    commodity_delta_usd: dict[str, float]
    vol_delta_usd: float
    vol_vega_usd: float
    inflation_beta: float | None
    credit_beta: float | None
```

For now, do not overfit a huge factor model. You need enough factor exposure to catch the obvious stacked risks:

- US equity beta: SPY/QQQ plus ES/NQ futures.
- Rates duration/DV01: bond ETFs plus Treasury futures.
- USD exposure: futures/FX/commodity contracts translated to USD.
- Commodity exposure: energy, metals, agriculturals.
- Vol/short-convexity exposure: VIX futures / VRP.
- Cash-equivalent exposure: SGOV/BIL separated from risk gross.

### 3.4 Recommended Postgres schema

Keep Postgres as the durable ledger. Use immutable append-only tables where possible.

#### Instrument master

```sql
instrument (
    instrument_id           text primary key,
    canonical_symbol        text not null,
    asset_class             text not null,
    base_currency           text not null,
    multiplier              numeric not null default 1,
    tick_size               numeric,
    point_value             numeric,
    expiry_date             date,
    root_symbol             text,
    roll_group              text,
    factor_map_id           text,
    is_cash_equivalent      boolean default false,
    is_tradeable            boolean default true
)

venue_instrument (
    venue                   text not null,
    instrument_id           text not null references instrument,
    broker_symbol           text not null,
    broker_conid            text,
    min_qty                 numeric,
    qty_increment           numeric,
    trading_calendar        text,
    margin_model_id         text,
    primary key (venue, instrument_id)
)
```

#### Strategy registry

```sql
strategy_registry (
    strategy_id             text primary key,
    status                  text not null, -- OFF, SHADOW, PAPER, PROBATION_LIVE, LIVE
    asset_class_scope       text[],
    venue_scope             text[],
    cadence                 text not null,
    risk_class              text not null, -- trend, carry, xsmom, short_vol, cash
    max_risk_weight         numeric not null,
    target_risk_weight      numeric not null,
    ramp_multiplier         numeric not null default 0,
    allow_short             boolean not null,
    allow_leverage          boolean not null,
    config_version          text not null,
    code_version            text not null,
    created_at              timestamptz not null,
    updated_at              timestamptz not null
)
```

#### Rebalance run state machine

```sql
rebalance_run (
    run_id                  text primary key,
    scheduled_for           timestamptz not null,
    decision_ts             timestamptz,
    status                  text not null,
    mode                    text not null, -- SHADOW, PAPER, LIVE, LIQUIDATION_ONLY
    policy_version          text not null,
    code_version            text not null,
    input_snapshot_hash     text,
    output_plan_hash        text,
    operator_approval       text,
    failure_reason          text,
    created_at              timestamptz not null,
    updated_at              timestamptz not null
)
```

Run statuses should be explicit:

```text
CREATED
PREFLIGHT_STARTED
SNAPSHOT_FROZEN
INTENTS_COLLECTED
TARGET_CONSTRUCTED
RISK_APPROVED
RISK_REJECTED
EXECUTION_PLANNED
EXECUTING
PARTIALLY_FILLED
RECONCILED
CLOSED
FAILED_HALTED
OPERATOR_HOLD
```

#### Snapshots

```sql
broker_account_snapshot (
    snapshot_id             text primary key,
    run_id                  text references rebalance_run,
    venue                   text not null,
    as_of                   timestamptz not null,
    raw_payload_json        jsonb not null,
    normalized_json         jsonb not null,
    hash                    text not null
)

broker_position_snapshot (
    snapshot_id             text primary key,
    run_id                  text references rebalance_run,
    venue                   text not null,
    instrument_id           text not null,
    broker_symbol           text not null,
    quantity                numeric not null,
    market_value_base       numeric not null,
    notional_base           numeric not null,
    raw_payload_json        jsonb not null,
    hash                    text not null
)

book_state_snapshot (
    snapshot_id             text primary key,
    run_id                  text references rebalance_run,
    as_of                   timestamptz not null,
    book_state_json         jsonb not null,
    exposure_json           jsonb not null,
    risk_state_json         jsonb not null,
    hash                    text not null
)
```

#### Strategy intents

```sql
strategy_intent (
    intent_id               text primary key,
    run_id                  text references rebalance_run,
    strategy_id             text references strategy_registry,
    as_of                   timestamptz not null,
    data_snapshot_hash      text not null,
    config_version          text not null,
    code_version            text not null,
    status                  text not null, -- ACCEPTED_FOR_CONSTRUCTION, REJECTED_STALE, ERROR
    intent_json             jsonb not null,
    model_portfolio_hash    text not null,
    diagnostics_json        jsonb,
    created_at              timestamptz not null
)
```

#### Target book, risk, and execution

```sql
book_target (
    target_id               text primary key,
    run_id                  text references rebalance_run,
    target_json             jsonb not null,
    strategy_weight_json    jsonb not null,
    exposure_json           jsonb not null,
    covariance_model_hash   text not null,
    policy_version          text not null,
    hash                    text not null
)

risk_check_result (
    risk_check_id           text primary key,
    run_id                  text references rebalance_run,
    check_name              text not null,
    severity                text not null, -- INFO, WARN, BLOCK
    passed                  boolean not null,
    observed_value          numeric,
    limit_value             numeric,
    details_json            jsonb,
    created_at              timestamptz not null
)

execution_plan (
    plan_id                 text primary key,
    run_id                  text references rebalance_run,
    plan_json               jsonb not null,
    hash                    text not null,
    created_at              timestamptz not null
)

planned_order (
    planned_order_id        text primary key,
    run_id                  text references rebalance_run,
    venue                   text not null,
    instrument_id           text not null,
    broker_symbol           text not null,
    side                    text not null,
    quantity                numeric not null,
    order_type              text not null,
    limit_price             numeric,
    time_in_force           text,
    idempotency_key         text not null unique,
    reduce_only             boolean not null default false,
    status                  text not null
)

broker_order (
    broker_order_id         text primary key,
    planned_order_id        text references planned_order,
    venue                   text not null,
    broker_native_order_id  text,
    idempotency_key         text not null,
    raw_request_json        jsonb not null,
    raw_response_json       jsonb,
    status                  text not null,
    created_at              timestamptz not null,
    updated_at              timestamptz not null
)

fill (
    fill_id                 text primary key,
    broker_order_id         text references broker_order,
    venue                   text not null,
    instrument_id           text not null,
    quantity                numeric not null,
    fill_price              numeric not null,
    commission_base         numeric,
    slippage_bps            numeric,
    raw_payload_json        jsonb,
    filled_at               timestamptz not null
)
```

#### Kill switch

```sql
kill_switch_state (
    id                      integer primary key default 1,
    mode                    text not null, -- NORMAL, HALT_NEW_RISK, CANCEL_ONLY, FLATTEN_ALL
    reason                  text,
    set_by                  text not null,
    set_at                  timestamptz not null,
    expires_at              timestamptz,
    version                 bigint not null
)

kill_switch_event (
    event_id                text primary key,
    prior_mode              text,
    new_mode                text not null,
    reason                  text,
    actor                   text not null,
    created_at              timestamptz not null
)
```

### 3.5 Redis should be demoted

Redis can remain useful, but not as a source of truth.

Use Redis for:

- Process locks.
- Pub/sub notifications.
- Fast kill-switch cache.
- Short-lived command queues.

Do not use Redis for the canonical list of proposals, approved trades, or “what is pending.” Those belong in Postgres as immutable run-scoped records.

---

## 4. Strategy contract: how strategies plug in safely

### 4.1 The new rule

Every strategy must implement one contract:

> **Given a frozen data context and current book context, emit a model intent. Do not size final capital. Do not call a broker. Do not place orders.**

A strategy intent is not an order. It is a desired model portfolio before book-level scaling.

Example interface:

```python
class Strategy(Protocol):
    strategy_id: str
    config_version: str

    def compute_intent(self, context: StrategyContext) -> StrategyIntent:
        ...
```

Context:

```python
@dataclass(frozen=True)
class StrategyContext:
    run_id: str
    as_of: datetime
    market_data: MarketDataSnapshot
    current_book: BookState
    instrument_master: InstrumentMaster
    strategy_config: dict
    mode: Literal["BACKTEST", "SHADOW", "PAPER", "LIVE"]
```

Intent:

```python
@dataclass(frozen=True)
class StrategyIntent:
    run_id: str
    strategy_id: str
    as_of: datetime
    model_portfolio: list[ModelPosition]
    expected_holding_period: str
    desired_risk_units: float | None
    confidence: float | None
    turnover_budget: float | None
    constraints: StrategyConstraints
    diagnostics: dict
    data_hash: str
    code_version: str
    config_version: str
```

Model position:

```python
@dataclass(frozen=True)
class ModelPosition:
    instrument_id: str
    direction: int             # -1, 0, +1
    raw_weight: float          # model weight before final scaling
    risk_weight: float | None  # optional inverse-vol normalized unit
    signal_strength: float | None
    min_trade_notional: float | None
    max_position_notional: float | None
    tags: dict
```

### 4.2 How this applies to your strategies

- **ETF trend** emits a 10-ETF long/flat inverse-vol model portfolio. It does not decide “50% gross.”
- **Cash/T-bills** emits a cash deployment preference for residual settled cash after risk book construction. It should run last and never consume risk budget.
- **Futures carry** emits cross-sectional long/short futures targets normalized to unit strategy volatility.
- **Futures xs-momentum** emits cross-sectional long/short futures targets normalized to unit strategy volatility.
- **VIX VRP** emits short-front-VIX exposure only when its deterministic crash gates pass. Its risk class is short-vol/convexity and gets stricter factor/stress constraints.
- **Crypto trend** emits model intent in SHADOW/PAPER until promoted.
- **Old ML proposal system** becomes one strategy adapter if it returns. It should emit a daily/intraday model book or proposed position set, not bypass the book brain.

### 4.3 Strategy registry becomes declarative

Adding a strategy should mean:

1. Implement `compute_intent`.
2. Add config to registry/YAML.
3. Add tests/golden-date replay.
4. Set status: `SHADOW`, `PAPER`, `PROBATION_LIVE`, or `LIVE`.
5. Assign risk class and risk caps.

Example YAML:

```yaml
strategies:
  etf_trend:
    module: app.strategies.etf_trend:ETFTrendStrategy
    status: LIVE
    cadence: WEEKLY
    venue_scope: [ALPACA]
    asset_class_scope: [ETF]
    risk_class: trend
    target_risk_weight: 0.40
    max_risk_weight: 0.50
    ramp_multiplier: 1.00
    allow_short: false
    allow_leverage: false

  futures_carry:
    module: app.strategies.futures_carry:FuturesCarryStrategy
    status: PAPER
    cadence: WEEKLY
    venue_scope: [IBKR]
    asset_class_scope: [FUTURE]
    risk_class: carry
    target_risk_weight: 0.20
    max_risk_weight: 0.25
    ramp_multiplier: 0.00
    allow_short: true
    allow_leverage: true

  vix_vrp:
    module: app.strategies.vix_vrp:VIXVRPStrategy
    status: PAPER
    cadence: WEEKLY
    venue_scope: [IBKR]
    asset_class_scope: [FUTURE]
    risk_class: short_vol
    target_risk_weight: 0.10
    max_risk_weight: 0.10
    ramp_multiplier: 0.00
    allow_short: true
    allow_leverage: true
```

---

## 5. The weekly decision loop

### 5.1 High-level sequence

```text
1. Acquire run lock
2. Check global kill switch
3. Broker connectivity and calendar preflight
4. Pull broker account/position/order snapshots
5. Reconcile broker reality vs ledger
6. Freeze current BookState snapshot
7. Freeze market data snapshot
8. Collect strategy intents
9. Normalize each strategy to unit risk
10. Apply strategy eligibility and ramp multipliers
11. Estimate robust strategy covariance / factor exposures
12. Allocate one book risk budget
13. Construct target book
14. Run whole-book pre-trade risk checks
15. Net target-current into execution deltas
16. Create idempotent planned orders
17. Optional operator hold if material change threshold is exceeded
18. Execute through broker adapters
19. Capture fills
20. Reconcile final broker state
21. Produce post-trade report
22. Release run lock
```

### 5.2 Pseudo-code

```python
def weekly_rebalance(scheduled_for: datetime, mode: RunMode) -> None:
    run_id = make_run_id("WEEKLY", scheduled_for)

    with acquire_advisory_lock(f"rebalance:{run_id}"):
        run = load_or_create_run(run_id, scheduled_for, mode)

        if run.status in TERMINAL_STATUSES:
            return

        assert_not_global_killed_or_enter_liquidation_mode(run)

        preflight = run_preflight_checks()
        persist_preflight(run_id, preflight)
        if preflight.has_blockers:
            halt_run(run_id, preflight.reason)
            return

        broker_snapshot = snapshot_all_brokers(run_id)
        reconciliation = reconcile_or_fail_closed(run_id, broker_snapshot)
        if reconciliation.material_break:
            set_halt_new_risk("material reconciliation break")
            halt_run(run_id, reconciliation.summary)
            return

        book_state = build_book_state(run_id, broker_snapshot)
        market_snapshot = freeze_market_data(run_id)

        intents = []
        for strategy in enabled_strategies():
            intent = strategy.compute_intent(
                StrategyContext(
                    run_id=run_id,
                    as_of=market_snapshot.as_of,
                    market_data=market_snapshot,
                    current_book=book_state,
                    instrument_master=instrument_master,
                    strategy_config=strategy.config,
                    mode=mode,
                )
            )
            validate_and_persist_intent(intent)
            intents.append(intent)

        target = construct_target_book(
            run_id=run_id,
            current_book=book_state,
            intents=intents,
            risk_policy=load_risk_policy(),
            covariance_model=load_covariance_model(),
        )
        persist_target(target)

        risk_result = run_whole_book_risk_gate(book_state, target)
        persist_risk_results(risk_result)

        if risk_result.has_blockers:
            halt_or_create_liquidation_plan(run_id, risk_result)
            return

        plan = create_execution_plan(book_state, target)
        persist_execution_plan(plan)

        if requires_operator_hold(book_state, target, plan):
            mark_operator_hold(run_id, plan)
            return

        execute_plan_idempotently(plan)
        final_snapshot = snapshot_all_brokers(run_id)
        final_recon = reconcile_final_state(run_id, final_snapshot, target)

        if final_recon.material_break:
            set_halt_new_risk("post-trade reconciliation break")
            alert_operator(final_recon)

        close_run(run_id)
```

### 5.3 Atomicity: decision atomic, execution not physically atomic

You cannot make Alpaca and IBKR execute atomically across venues. Do not pretend you can.

What you can make atomic is the **decision**:

- One run id.
- One frozen input snapshot.
- One target.
- One risk approval.
- One execution plan.

Execution then becomes a controlled process with partial-fill handling and recovery. The book may transition over minutes, but the intended end state is immutable.

---

## 6. Holistic sizing under one risk budget

This is the core of the system. I would avoid a fragile optimizer. I would use a robust, layered recipe:

1. **Manual base risk budgets by strategy class.**
2. **Normalize each strategy to unit risk.**
3. **Use a shrunk covariance model at the strategy level, not a giant instrument optimizer.**
4. **Size the total book to the worse of normal covariance and stressed covariance.**
5. **Apply deterministic de-grossing overlays for drawdown, realized correlation spikes, margin, liquidity, and event/crash gates.**
6. **Use factor exposure limits as hard constraints.**
7. **Only then convert to instrument targets.**

The allocator should be humble. You are not trying to maximize expected return. You are trying to keep a multi-stream book inside a controlled drawdown/vol/margin envelope.

### 6.1 Strategy-level first, instrument-level second

Because you only have 3–4 main risk-taking streams, the robust object to allocate across is the **strategy sleeve return stream**, not a 100-instrument covariance matrix.

Estimate:

```text
strategy_return_matrix:
  ETF_TREND
  FUTURES_CARRY
  FUTURES_XSMOM
  VIX_VRP
  CRYPTO_TREND, if promoted later
```

Then allocate risk across strategies. After strategy weights are set, each strategy’s model portfolio is scaled to its allocated risk.

### 6.2 Normalize each strategy to unit risk

Each strategy intent should produce a model portfolio. The portfolio constructor computes a unit-risk version:

```text
model_portfolio_i
estimated_vol_i
unit_portfolio_i = model_portfolio_i / estimated_vol_i
```

For each strategy:

```text
target_strategy_exposure_i =
    unit_portfolio_i
  × strategy_risk_budget_i
  × total_book_risk_scaler
  × ramp_multiplier_i
  × regime/crash_gate_multiplier_i
```

For futures, convert unit exposure into contracts using price, multiplier, FX, and contract specs.

### 6.3 Base risk budget

Start with explicit base weights, not optimized weights.

Illustrative starting point:

```text
ETF_TREND       35–45% of risk budget
FUTURES_CARRY   15–25%
FUTURES_XSMOM   15–25%
VIX_VRP          0–10%, capped hard because short-vol is nonlinear
CRYPTO_TREND     0 until proven; then 5–10% initially
CASH             no risk budget; residual cash deployment
```

Do not let the allocator decide that VIX VRP deserves 35% because its backtest Sharpe is high. Short-vol gets a hard cap.

### 6.4 Robust covariance model

Use a shrinkage model that is intentionally conservative.

Let:

```text
Σ_sample = sample covariance of strategy returns
D        = diagonal matrix of strategy variances
C_prior  = prior correlation matrix
Σ_prior  = vols × C_prior × vols
```

Recommended:

```text
Σ_normal = λ * Σ_sample + (1 - λ) * Σ_prior
```

Where `λ` is modest, maybe 0.25–0.50. For a solo book, I would rather underuse recent correlation than overfit it.

A simple prior correlation matrix might be:

```text
                 ETF_TR  CARRY  XSMOM  VIX_VRP  CRYPTO_TR
ETF_TR             1.00   0.35   0.45    0.40      0.25
CARRY              0.35   1.00   0.45    0.35      0.20
XSMOM              0.45   0.45   1.00    0.40      0.25
VIX_VRP            0.40   0.35   0.40    1.00      0.20
CRYPTO_TR          0.25   0.20   0.25    0.20      1.00
```

Then also define a crisis correlation matrix:

```text
C_crisis:
  risk-on strategies converge toward 0.75–0.90
  short-vol correlation to equity selloff risk rises materially
  crypto correlation rises in liquidity stress
```

The book should size to:

```text
estimated_book_vol = max(
    vol(weights, Σ_normal),
    vol(weights, Σ_crisis) * crisis_penalty_weight
)
```

Or more simply:

```text
risk_utilization = target_book_vol / max(vol_normal, vol_stress)
```

This is the key principle:

> **You get diversification credit only to the extent it survives shrinkage and stress.**

### 6.5 Equal risk contribution, but with strong caps

If you want a more formal allocator than fixed base weights, use **equal risk contribution with caps**, not max-Sharpe optimization.

Solve for weights where each strategy contributes roughly its budgeted share of portfolio variance:

```text
RC_i = w_i * (Σw)_i / sqrt(w'Σw)
```

But constrain it:

```text
0 <= w_i <= max_weight_i
abs(w_i - base_weight_i) <= rebalance_drift_limit
VIX_VRP <= hard_short_vol_cap
paper/probation strategies <= ramp cap
```

Do not optimize on expected returns. Expected returns are the noisiest input. Risk budgets should reflect conviction, capacity, and drawdown tolerance — not recent Sharpe-chasing.

### 6.6 Total book scaler

After strategy risk weights are selected, apply a total book scaler:

```text
total_scaler = min(
    vol_target_scaler,
    drawdown_scaler,
    realized_corr_scaler,
    margin_scaler,
    liquidity_scaler,
    event_risk_scaler,
    operator_capital_stage_scaler,
)
```

#### Vol target scaler

```text
vol_target_scaler = target_book_vol / estimated_book_vol
```

Capped:

```text
vol_target_scaler ∈ [0.0, 1.25]
```

I would not allow automatic leverage-up above 1.0 until the rest of the system has been live for a while. Let under-risking be acceptable. Over-risking through model confidence is dangerous.

#### Drawdown scaler

Example:

```text
if current_drawdown < 0.5 * max_dd_limit: scaler = 1.00
if current_drawdown between 0.5 and 0.8 limit: linear 1.00 -> 0.70
if current_drawdown between 0.8 and 1.0 limit: linear 0.70 -> 0.30
if current_drawdown >= limit: liquidation_only or risk_minimum
```

#### Realized correlation spike scaler

Compute trailing realized average pairwise correlation among live strategy returns and/or factor proxy returns.

```text
if avg_corr_63d <= 0.50: scaler = 1.00
if avg_corr_63d 0.50 -> 0.75: scaler = 1.00 -> 0.70
if avg_corr_63d > 0.75: scaler = 0.50
```

Do not let this be the only defense. It is an overlay.

#### Margin scaler

For IBKR futures:

```text
margin_utilization = initial_margin_required / net_liquidation_value
```

Example constraints:

```text
target initial margin <= 25–35% of equity initially
target maintenance margin <= 20–30%
post-stress margin <= hard cap
```

If margin estimates are stale or unavailable, fail closed.

#### Event/crash gate scaler

For deterministic crash governors — VIX term structure, market stress, exchange halts, major scheduled risk — de-gross. For LLM-driven event risk, if used at all, it should be bounded and only reduce risk.

### 6.7 Factor limits are hard gates, not soft suggestions

After constructing a candidate target, compute factor exposures.

Examples of limits to encode:

```text
gross_risk_notional_ex_cash <= 0.80 * equity initially
net_equity_beta_usd <= configurable cap
single_asset_class_risk <= cap
short_vol_notional <= hard cap
rates_DV01 <= cap
commodity_sector_exposure <= cap
single_contract_margin <= cap
single_strategy_risk_contribution <= cap
single_venue_margin_utilization <= cap
```

The exact limits are policy decisions, but the architecture must enforce them at target-book level.

### 6.8 Paper strategy ramp

A paper strategy should enter the live budget through an auditable ramp multiplier, not by a one-time manual size decision.

Status model:

```text
OFF             not run
SHADOW          emits intents, no paper orders, compared to backtest
PAPER           paper tracked, no live orders
PROBATION_LIVE  tiny live risk
LIVE            eligible for full configured risk cap
DISABLED        cannot emit target
```

Ramp fields:

```text
ramp_multiplier ∈ [0.0, 1.0]
ramp_reason
evidence_window_start
live_tracking_error
slippage_estimate
operational_break_count
max_observed_drawdown
operator_approved_at
```

Illustrative ramp:

```text
0.00  paper only
0.10  after 8–12 clean weekly shadow runs
0.25  after 3 months clean paper/live-parity evidence
0.50  after 6 months and no unresolved reconciliation/slippage issues
1.00  after 12 months or enough evidence, manually approved
```

Rules:

- Ramp cannot increase automatically after a material reconciliation break.
- Ramp cannot increase during strategy drawdown beyond threshold.
- Ramp can decrease automatically due to risk events.
- Every ramp change is a ledger event.

---

## 7. Netting and conflict resolution

### 7.1 Net exact instruments physically

If two strategies want the same canonical instrument in opposite directions, net them before trading.

Example:

```text
ETF_TREND wants +$40k SPY
OLD_ML wants -$10k SPY
Physical target = +$30k SPY
Virtual allocation:
  ETF_TREND +$40k
  OLD_ML    -$10k
```

This avoids paying spread twice and avoids fake diversification.

### 7.2 Do not over-net proxies by default

A subtler question: if ETF trend wants SPY and futures xs-momentum wants ES, should the brain net them?

My recommendation:

- **Exact canonical equivalents**: net automatically.
- **Declared substitute groups**: net if explicitly configured.
- **Correlated but not equivalent instruments**: do not net mechanically; manage through factor exposure limits and risk scaling.

Why? Because SPY and ES have different venue, margin, tax, liquidity, overnight, and financing characteristics. They are close economic substitutes, but not operational substitutes in every context. Automatic cross-instrument netting can create hidden behavior that breaks research/live parity.

Use equivalence groups sparingly:

```yaml
equivalence_groups:
  us_large_cap_beta:
    instruments: [SPY, IVV, VOO, ES]
    netting_mode: factor_aware
    preferred_execution:
      default: SPY
      futures_overlay: ES
    allow_cross_venue_substitution: false_initially
```

Initial policy:

> **Aggregate SPY and ES into the same equity-beta exposure, but do not automatically replace one with the other unless a specific strategy or execution policy declares them substitutable.**

### 7.3 Conflict resolution rule

When strategies disagree:

1. Preserve each strategy’s virtual target.
2. Net exact same instrument physically.
3. Enforce factor caps on the aggregate proposed book.
4. If factor cap is breached, scale down the risk budgets according to deterministic priority.
5. Do not let one strategy secretly override another.

Priority for scaling when constraints bind:

```text
1. Scale paper/probation strategies first.
2. Scale short-vol / nonlinear risk next if stress constraints bind.
3. Scale strategies with lowest current capital stage.
4. Scale all live strategies pro-rata if a general book-level constraint binds.
5. Cash sleeve absorbs residual.
```

This should be deterministic and auditable.

### 7.4 Execution netting

Execution planner should compute:

```text
delta = target_physical_position - current_physical_position - open_pending_delta
```

Then apply:

- No-trade bands.
- Minimum notional thresholds.
- Round to lot/contract increments.
- Reduce-only first.
- Cash sweep last.
- Cancel stale/conflicting orders before new orders.
- Venue-specific order strategy.

---

## 8. Factor exposure model

### 8.1 Why symbol/sector correlation is insufficient

The risk that matters is not just “how correlated is SPY to ES” or “what sector is this ETF.” The book needs a canonical exposure vector.

At minimum:

```text
Book exposure =
  gross risk notional
  net risk notional
  equity beta
  rates duration/DV01
  USD exposure
  commodity exposure
  vol/short-convexity exposure
  cash-equivalent exposure
  margin utilization
```

### 8.2 Factor map

Each instrument should have a factor map.

```json
{
  "instrument_id": "ES_U2026",
  "factor_loadings": {
    "equity_beta_us": 1.00,
    "rates_dv01": 0.00,
    "usd_fx": 0.00,
    "commodity_energy": 0.00,
    "vol_short": 0.00
  },
  "notional_formula": "price * multiplier * quantity",
  "delta_equivalent_formula": "notional * beta"
}
```

For bond futures/ETFs:

```json
{
  "instrument_id": "ZN_U2026",
  "factor_loadings": {
    "rates_dv01": "computed_from_duration_or_contract_spec",
    "equity_beta_us": 0.00
  }
}
```

For VIX futures:

```json
{
  "instrument_id": "VX_U2026",
  "factor_loadings": {
    "vol_short": -1.00,
    "equity_crash_beta": "estimated",
    "convexity_risk": "high"
  }
}
```

### 8.3 Exposure calculation

For each position:

```text
notional = quantity × price × multiplier × fx_rate

factor_exposure_k = notional × factor_loading_k
```

For rates, use DV01 where possible:

```text
DV01 = quantity × contract_DV01 × fx_rate
```

For ETFs, use beta/duration approximations from instrument metadata.

### 8.4 Factor limits should run twice

Run factor exposure checks:

1. On current book before collecting intents.
2. On proposed target book before execution.

If current book is already over a limit, the system should enter `LIQUIDATION_ONLY` for that factor: allow reducing trades, block increasing trades.

---

## 9. Cross-venue broker abstraction

### 9.1 Broker adapter interface

The brain should not know Alpaca or IBKR semantics directly. It should call a common interface.

```python
class BrokerAdapter(Protocol):
    venue: str

    def health(self) -> BrokerHealth:
        ...

    def snapshot_account(self) -> BrokerAccountSnapshot:
        ...

    def snapshot_positions(self) -> list[BrokerPositionSnapshot]:
        ...

    def list_open_orders(self) -> list[BrokerOrderSnapshot]:
        ...

    def estimate_order(self, order: CanonicalOrderIntent) -> OrderEstimate:
        ...

    def place_order(self, order: CanonicalOrderIntent) -> BrokerOrderAck:
        ...

    def cancel_order(self, broker_order_id: str) -> CancelResult:
        ...

    def cancel_all_open_orders(self) -> list[CancelResult]:
        ...

    def flatten_position(self, instrument_id: str, quantity: Decimal | None = None) -> list[BrokerOrderAck]:
        ...

    def normalize_symbol(self, broker_symbol: str) -> str:
        ...
```

The adapter handles:

- Broker symbols.
- Contract IDs.
- Futures expiry and roll mapping.
- Quantity increments.
- Margin semantics.
- Native order status.
- Fill parsing.
- Time-in-force differences.
- API idempotency behavior.
- Rate limits.
- Connectivity quirks.

The brain handles:

- Target book.
- Risk limits.
- Netting.
- Order intent.
- Kill mode.
- Reconciliation policy.

### 9.2 Canonical order intent

```python
@dataclass(frozen=True)
class CanonicalOrderIntent:
    run_id: str
    planned_order_id: str
    idempotency_key: str
    venue: str
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: Decimal
    order_type: Literal["MARKET", "LIMIT", "MOC"]
    limit_price: Decimal | None
    time_in_force: str
    reduce_only: bool
    strategy_reason: str
    risk_reason: str
```

### 9.3 Venue-specific concerns should not leak

IBKR futures require contract specs, margin, multiplier, expiry/roll, and session calendars. Those belong in:

- Instrument master.
- Venue instrument mapping.
- Margin model.
- Broker adapter.
- Futures roll service.

They should not appear inside ETF trend, futures carry logic, or the portfolio constructor except through normalized fields.

### 9.4 Cross-venue cash and margin

For a two-broker book, “cash” is not one number. You need:

```text
consolidated_equity
  = Alpaca equity
  + IBKR net liquidation value
  + external cash if manually included

risk_capital
  = configurable capital assigned to the book

available_risk_budget
  = risk_capital × capital_stage × risk_scaler

venue_cash:
  Alpaca settled cash
  Alpaca buying power
  IBKR cash
  IBKR initial/maintenance margin
  IBKR excess liquidity
```

The book constructor can allocate risk across venues, but the execution planner must ensure venue-specific funding and margin are feasible.

Do not let the T-bill cash sleeve blindly sweep cash needed for futures margin. Cash sweep should be downstream of the target risk book and venue funding model.

---

## 10. Single kill-switch and dead-man design

### 10.1 Kill-switch modes

Use explicit modes:

```text
NORMAL
HALT_NEW_RISK       no risk-increasing orders; allow exits/reductions
CANCEL_ONLY         cancel open orders; do not place new orders
FLATTEN_ALL         cancel open orders, then flatten positions across all venues
DISABLE_SCHEDULE    no scheduled rebalances
```

### 10.2 Kill-switch storage

Store kill state in Postgres, mirror it in Redis for fast access.

Every trading loop and broker adapter call must check kill state. Broker adapters should refuse risk-increasing orders if kill mode is not `NORMAL`.

### 10.3 Idempotent flatten

Flattening must be idempotent:

```python
def flatten_all(reason: str):
    set_kill_mode("FLATTEN_ALL", reason)
    for broker in brokers:
        broker.cancel_all_open_orders()
    snapshot = snapshot_all_brokers()
    for position in snapshot.positions:
        if abs(position.quantity) > dust_threshold:
            broker.flatten_position(position.instrument_id)
    reconcile_until_flat_or_timeout()
```

If called twice, it should not double-flatten. It should re-snapshot first and only close existing positions.

### 10.4 Dead-man

For your current scale, I would not auto-flatten on every heartbeat miss. That can create false-positive liquidation. Instead:

- If `tradingd` heartbeat is stale: watchdog sets `HALT_NEW_RISK`.
- If broker connectivity is bad: set `HALT_NEW_RISK`.
- If reconciliation material break exists: set `HALT_NEW_RISK`.
- If margin breach or catastrophic drawdown threshold: optionally set `FLATTEN_ALL`, but only for explicit severe rules.

Later, if real capital and futures risk grow, add configurable auto-flatten rules.

---

## 11. Determinism, idempotency, and auditability

### 11.1 Deterministic run IDs

A run id should be deterministic:

```text
WEEKLY_2026-06-22T09:35:00-04:00_POLICYv3
```

Or:

```text
run_id = sha256(schedule_key + portfolio_id + policy_version + rebalance_date)
```

Do not create a new run id on retry. Retrying should resume the same run.

### 11.2 Immutable snapshots

Every run must persist:

- Broker account snapshot.
- Broker position snapshot.
- Open orders snapshot.
- Market data snapshot hash.
- Strategy intent payloads.
- Covariance/risk model version and inputs.
- Book target.
- Risk check results.
- Execution plan.
- Orders and idempotency keys.
- Broker acks.
- Fills.
- Final reconciliation.

If you cannot reconstruct why the book did something, the design failed.

### 11.3 Idempotent order keys

Order key:

```text
idempotency_key =
  sha256(run_id + venue + instrument_id + side + qty + order_type + sequence_number + plan_hash)
```

Persist before sending. On crash/restart:

1. Load planned orders.
2. Query broker open/recent orders.
3. Match by client order id / idempotency key where broker supports it.
4. Do not send duplicates.
5. If unknown, halt and reconcile.

### 11.4 State machine recovery

On restart:

```text
If run status SNAPSHOT_FROZEN: continue intent collection.
If INTENTS_COLLECTED: rebuild target from persisted intents.
If EXECUTION_PLANNED: compare planned orders to broker orders before sending.
If EXECUTING: reconcile broker orders/fills; do not blindly resubmit.
If PARTIALLY_FILLED: construct residual plan only after reconciliation.
If RISK_REJECTED or FAILED_HALTED: do nothing except alert.
```

### 11.5 Reproducibility

Every deterministic decision should be replayable:

```bash
mtrader replay --run-id WEEKLY_2026_06_22 --assert-same-target
```

Replay should load persisted snapshots and policy versions, run the constructor, and verify that target hash and plan hash match.

This is also the foundation for research/live parity.

---

## 12. Whole-book pre-trade risk gate

### 12.1 Gate the target book, not just orders

The risk gate should receive:

```text
current_book
proposed_target_book
execution_plan
broker/margin estimates
risk policy
```

Checks:

```text
Data and state:
- Market data freshness
- Broker snapshot freshness
- Reconciliation clean
- No unknown open orders
- Instrument master complete
- Strategy intents complete or explicitly skipped
- Kill switch normal

Book risk:
- Gross ex-cash
- Net exposure
- Ex-ante vol
- Stress vol
- Drawdown budget
- Strategy risk contribution
- Factor exposure limits
- Concentration by instrument/root/asset class/venue
- Short-vol / convexity caps
- Turnover limits
- Liquidity / ADTV
- Margin utilization
- Cash/settlement constraints
- PDT / equity rules where relevant

Execution:
- Max order notional
- Max position change per instrument
- No order outside allowed venue/session
- No order in stale contract
- Roll constraints
- Reduce-only enforcement under halt modes
```

### 12.2 First-failure vs full report

The old RM first-failure style is bad for diagnosis. For the target-book gate, run all checks and produce a full report. Then block if any `BLOCK` severity check fails.

This gives you a complete “why this target failed” report.

### 12.3 Risk-reducing exception

If the current book is already in breach, you need liquidation-only logic.

Policy:

```text
If proposed trade reduces breached exposure:
    allow, even under some risk breaches
If proposed trade increases breached exposure:
    block
If proposed target reduces overall risk but violates a noncritical limit temporarily:
    allow only with explicit rule and audit
```

---

## 13. Execution planner

### 13.1 Execution should be boring

You are weekly. You do not need smart order routing. You need correctness.

Execution planner responsibilities:

1. Convert target physical book into deltas.
2. Subtract open pending orders.
3. Round to valid quantity increments.
4. Apply no-trade bands.
5. Sequence orders safely.
6. Generate idempotent planned orders.
7. Estimate margin and cash impact.
8. Persist before placing anything.

### 13.2 Sequencing

Default sequence:

```text
1. Cancel stale/conflicting open orders.
2. Execute risk-reducing sells/covers first.
3. Execute futures margin-reducing trades before margin-increasing trades.
4. Execute risk-increasing trades.
5. Execute cash/T-bill sweep last.
6. Reconcile.
```

For cross-venue books, order sequencing matters because the book can be temporarily more risky during transition. The planner should calculate max transient exposure if it buys before sells. Default to de-risking first.

### 13.3 Partial fills

Partial fill policy:

- Persist each fill.
- Recompute residual delta after broker snapshot.
- Do not assume unfilled quantity remains valid after a long delay.
- If fill is incomplete near cutoff, either:
  - leave residual if within tolerance,
  - retry with updated limit,
  - or halt and alert.

For weekly liquid ETFs/futures, this should be manageable.

### 13.4 Cash sleeve

Cash/T-bill sleeve should not be a normal alpha strategy. It is a residual allocator:

```text
residual_cash = cash_after_risk_book - liquidity_buffer - futures_margin_buffer
target_cash_equivalent = max(0, residual_cash)
```

Execute it last. It should never block margin needs.

---

## 14. Futures-specific design

### 14.1 Futures roll belongs in instrument/contract service

Do not let every futures strategy implement its own roll logic.

Create:

```text
FuturesContractService
- active contract for root
- roll calendar
- price adjustment metadata
- broker contract id
- multiplier
- tick size
- margin estimate
```

Strategies should speak in canonical roots when possible:

```text
strategy wants: CL root exposure +X
contract service maps to: CLQ6 or CLU6 depending roll policy
```

### 14.2 Roll as a book operation

A futures roll is not alpha. It is maintenance of target exposure.

Represent roll trades separately:

```text
trade_reason = ROLL
strategy_id = system_roll
links_to_strategy_allocations = yes
```

The portfolio brain should include roll in target construction and execution planning.

### 14.3 Margin stress

For futures, pre-trade risk must include margin stress:

```text
current_margin
target_margin
post_stress_margin
excess_liquidity_after_orders
```

If IBKR margin estimate is unavailable, conservative fallback:

```text
margin = configured_contract_margin × stress_multiplier
```

Fail closed if neither broker nor configured margin is available.

---

## 15. Extensibility: new venues and asset classes

### 15.1 Adding a venue

Adding a venue should require:

1. Broker adapter.
2. Venue instrument mappings.
3. Account/position/order normalization.
4. Margin/cash model.
5. Reconciliation parser.
6. Integration tests using recorded broker payloads.

It should not require changing the portfolio constructor.

### 15.2 Adding options

Options are the first asset class that will stress this design.

Before options go live, add:

- Greeks in canonical exposure.
- Assignment/exercise model.
- Expiry risk checks.
- Liquidity/spread checks.
- Scenario shocks.
- Position aggregation by underlying.
- Hard limits on short gamma/vega/theta.

Do not let options enter as just another symbol with notional.

### 15.3 Adding crypto

Crypto needs:

- 24/7 calendar.
- Different custody/settlement.
- Venue outage risk.
- Stale feed checks.
- Weekend risk policy.
- Separate volatility scaling.

The architecture can handle it, but the risk policy must treat it as a separate risk class.

---

## 16. Research/live parity

### 16.1 One strategy code path

The same strategy class should run in:

```text
BACKTEST
SHADOW
PAPER
LIVE
```

Mode should change data/execution context, not strategy logic.

Bad:

```python
if live:
    do_different_signal_logic()
```

Good:

```python
strategy.compute_intent(context)
```

Where `context` supplies backtest or live data through the same interface.

### 16.2 Golden-date replay

For every strategy and the full brain:

```bash
mtrader replay --date 2025-03-10 --strategy etf_trend
mtrader replay --date 2025-03-10 --full-book
```

Assertions:

- Same strategy intent hash.
- Same target book hash.
- Same exposure vector.
- Same risk check result.
- Same execution plan, after applying current rounding rules.

### 16.3 Live-vs-backtest divergence monitor

Each run should report:

```text
Expected target from research engine
Actual target from live brain
Difference due to:
  data revision
  missing data
  rounding
  liquidity/no-trade bands
  risk scaling
  margin constraints
  broker rejects
```

This is where many systematic systems quietly drift. Make drift visible.

---

## 17. LLMs in the loop

### 17.1 Verdict

Your priors are correct: **LLMs do not belong in the deterministic sizing/execution path.**

They can be useful around the path:

- Research/design review.
- Narrative/event-risk monitoring.
- Sanity review that can hold for human approval.
- Post-trade analysis.
- Operator copilot.
- Regime narration.

But they should not directly decide weights, increase exposure, pick order quantities, or place orders.

### 17.2 LLM-IN / LLM-OUT map

| Role | Verdict | Integration point | Allowed action | Fallback |
|---|---:|---|---|---|
| Research/design advisor | IN | Offline research workflow | Suggestions only | No live impact |
| Narrative/event-risk monitor | IN, later | Pre-construction overlay | Bounded de-risk flag or alert only | Ignore if unavailable |
| Anomaly/sanity layer over proposed book | IN, after deterministic brain | After target/risk report, before execution | Hold-for-human only | Proceed if disabled by policy; or no-op |
| Post-trade/reconciliation analyst | IN | After fills/reconciliation | Narrative + diagnostics | No trading impact |
| Regime narration/state summarization | IN | Dashboard/morning brief | Summary only | No-op |
| Operator copilot | IN, read-only first | Query interface | Read-only answers; draft actions requiring confirmation | No-op |
| LLM chooses sizes/orders | OUT | Deterministic path | None | Not allowed |
| LLM sends broker orders | OUT | Execution | None | Not allowed |
| LLM overrides risk gate to increase risk | OUT | Risk | None | Not allowed |

### 17.3 Event-risk monitor pattern

Inputs:

```text
- Curated news/macro feeds
- Economic calendar
- Current/proposed factor exposures
- Known high-risk instruments
- Existing deterministic regime indicators
```

Output must be structured:

```json
{
  "risk_flag": "NONE | WATCH | DEGROSS",
  "affected_factors": ["equity_beta_us", "short_vol"],
  "confidence": 0.0,
  "max_degross_multiplier": 0.85,
  "rationale": "...",
  "source_ids": ["..."]
}
```

Rules:

- LLM can never increase exposure.
- LLM can never choose instrument-level trades.
- LLM can only propose a de-gross multiplier within a cap, e.g. max 15–25%.
- For live capital, I would initially make it **alert-only** or **hold-for-human**, not autonomous.
- If unavailable, uncertain, or malformed output: ignore and log.
- Treat all news text as untrusted input; isolate it from system prompts and never let it issue tool commands.

### 17.4 Anomaly/sanity layer pattern

After deterministic construction:

```text
input:
  current book
  proposed target
  deltas
  risk report
  factor exposures
  recent market context
  run diagnostics

output:
  PASS or HOLD_FOR_HUMAN
  reason categories
  questions for operator
```

It can flag things like:

- “You are increasing equity beta while crisis governor is active.”
- “Most of this rebalance is from one new paper strategy.”
- “IBKR margin estimate seems inconsistent with prior runs.”
- “The book is reducing nominal gross but increasing short-vol stress.”

But it cannot mutate the plan. It can only hold.

### 17.5 Post-trade analyst

This is worth building after the deterministic ledger exists.

It should produce:

- What changed.
- Why the target changed.
- What risk budget each strategy used.
- What trades were executed.
- Slippage and commissions.
- Reconciliation breaks.
- Live-vs-backtest drift.
- Operator action items.

This is high-value and low-risk.

### 17.6 Build LLM monitoring now?

My honest answer:

> **Not before the deterministic brain exists.**

Right now, the risk is not that you lack an LLM narrator. The risk is that the live sleeves bypass the risk brain and IBKR is about to arrive as a second silo.

Build order:

1. Deterministic book state and target-book pipeline.
2. Broker abstraction and kill switch.
3. Whole-book risk report and post-trade report.
4. Then add LLM narrative/post-trade analyst.
5. Then consider LLM anomaly hold.
6. Event-risk de-gross comes last, and only bounded.

---

## 18. Failure modes I would fear most

### 18.1 DB says one thing, broker holds another

This is the top operational risk.

Guardrails:

- Broker snapshot before every rebalance.
- Material reconciliation break halts new risk.
- Broker reality wins.
- Unknown positions cannot be ignored.
- Dashboard highlights breaks.
- Post-trade reconciliation mandatory.

### 18.2 Duplicate orders after crash/retry

Guardrails:

- Persist planned orders before sending.
- Deterministic idempotency keys.
- Broker order lookup on restart.
- Never blindly resend.
- If broker idempotency is weak, halt on ambiguity.

### 18.3 Adding IBKR as a parallel silo

Guardrail:

- No IBKR live strategy until IBKR adapter feeds canonical BookState.
- Futures strategies emit intents only.
- All IBKR orders go through execution planner.

### 18.4 Correlation model gives false diversification

Guardrails:

- Shrink correlations.
- Stress correlations.
- Hard factor caps.
- Realized correlation de-gross overlay.
- Manual strategy caps.
- No max-Sharpe optimizer.

### 18.5 Cash sweep consumes margin liquidity

Guardrails:

- Cash sleeve executes last.
- Futures margin buffer reserved before cash deployment.
- Venue-specific cash/margin model.
- Cash equivalent excluded from risk gross but included in liquidity checks.

### 18.6 Futures contract roll error

Guardrails:

- Central futures contract service.
- Roll calendar.
- Contract expiry checks.
- No trading stale/expired contracts.
- Reconcile root exposure and contract exposure separately.

### 18.7 LLM false confidence

Guardrails:

- LLM outside deterministic path.
- Structured bounded outputs.
- No autonomous risk increase.
- No broker tools.
- Prompt-injection isolation.
- Full audit log.
- Human hold for material actions.

### 18.8 Dashboard/operator command accidentally trades

Guardrails:

- API cannot place orders directly.
- API writes operator command events.
- Trading daemon validates commands.
- Confirmation required for flatten/live mode/ramp increase.
- All commands audit logged.

---

## 19. Migration path: strangler, not big-bang

### Phase 0 — Freeze the dangerous expansion

**Rule:** Do not add IBKR futures as an independent trading silo.

You can continue paper tracking and adapter development, but no live/paper-live IBKR orders should bypass the future brain.

Deliverables:

- Written architecture decision record: “All live orders must go through portfolio brain.”
- Disable path for ad-hoc direct sleeve orders except current ETF trend until migrated.
- Define initial risk policy.

### Phase 1 — Split FastAPI from trading daemon

Deliverables:

- `mtrader-api`: dashboard only.
- `mtrader-tradingd`: scheduler, broker connectivity, reconciliation, current live jobs.
- Shared Postgres.
- Redis only for locks/notifications/kill cache.
- Heartbeat table.

Success criteria:

- Web restart does not stop trading daemon.
- Trading daemon restart does not require web restart.
- Dashboard can show tradingd health.

### Phase 2 — Build canonical instrument master and broker abstraction for Alpaca

Deliverables:

- Instrument table for ETFs/T-bills.
- Alpaca adapter implementing common broker interface.
- Canonical account/position/order snapshots.
- Reconciliation report against existing `trades` table.
- No trading behavior changed yet.

Success criteria:

- Every Alpaca position maps to canonical instrument.
- Current book exposure report matches broker.
- Reconciliation can classify breaks.

### Phase 3 — Build BookState ledger in shadow mode

Deliverables:

- `rebalance_run`.
- Broker snapshots.
- BookState snapshots.
- Exposure snapshots.
- Reconciliation break table.
- Dashboard view: current consolidated book.

No strategy migration yet.

Success criteria:

- At any time, you can answer:
  - What do I hold?
  - Which strategy owns it virtually?
  - What is gross/net?
  - What is equity beta?
  - What is cash-equivalent?
  - Are broker and DB reconciled?

### Phase 4 — Strategy intent interface and ETF trend shadow

Deliverables:

- Refactor ETF trend so it can emit `StrategyIntent`.
- Existing ETF trend rebalancer still trades.
- New brain computes what it would have done in shadow.
- Compare old vs new target.

Success criteria:

- For several weekly runs, target deltas match within rounding/no-trade tolerance.
- Differences are explained and logged.

### Phase 5 — Cash sleeve becomes residual allocator

Deliverables:

- Cash/T-bill no longer scheduled independently in shadow.
- Brain calculates residual cash target after risk book.
- Existing cash sleeve still trades until validated.

Success criteria:

- Cash target respects margin/reserve model.
- No unexplained differences vs current cash sleeve.

### Phase 6 — Brain controls ETF trend + cash in paper

Deliverables:

- Disable direct ETF trend orders.
- Brain generates execution plan for ETF trend.
- Alpaca adapter places orders.
- Cash executes last.
- Old rebalancer retained behind feature flag for rollback.

Success criteria:

- One full cycle completed with:
  - run snapshot,
  - strategy intent,
  - target,
  - risk approval,
  - execution plan,
  - fills,
  - reconciliation.

### Phase 7 — Add IBKR adapter read-only

Deliverables:

- IBKR account snapshot.
- IBKR positions snapshot.
- IBKR open orders.
- Futures instrument mapping.
- Contract service.
- Margin model.
- No IBKR orders yet.

Success criteria:

- IBKR state appears in consolidated BookState.
- Kill switch can at least halt future IBKR orders.
- Reconciler understands IBKR payloads.

### Phase 8 — Futures strategies emit intents through brain

Deliverables:

- Futures carry intent.
- Futures xs-momentum intent.
- VIX VRP intent.
- Futures risk/factor exposure calculation.
- No orders, or paper-only simulated orders.

Success criteria:

- Full proposed target book includes Alpaca + IBKR.
- Factor exposure catches stacked SPY/ES beta.
- Margin estimate produced.
- Risk gate can approve/reject.

### Phase 9 — Enable IBKR paper/live at tiny ramp

Deliverables:

- IBKR order placement through execution planner.
- Idempotent order keys.
- Partial fill handling.
- Post-trade reconciliation.
- Ramp multiplier set to tiny value.

Success criteria:

- No direct per-sleeve IBKR order path exists.
- All futures trades trace to a book decision run.
- Kill switch can cancel/flatten IBKR and Alpaca.

### Phase 10 — Retire legacy PM/RM/Trader path or adapt it

The old proposal-driven ML path should either remain off or become a strategy-intent emitter. It should not remain an independent live order pipeline.

---

## 20. Concrete build order

### Build 1: Book-state and reconciliation spine

Why first: the system cannot coordinate what it cannot see.

Tasks:

- Instrument master.
- Broker snapshot tables.
- BookState builder.
- Exposure calculator.
- Reconciliation classifier.
- Dashboard consolidated book view.
- Halt-new-risk on material breaks.

### Build 2: Trading daemon + kill switch

Why second: execution must leave FastAPI before futures.

Tasks:

- Separate process.
- Heartbeat.
- Postgres/Redis kill state.
- Alpaca adapter behind broker interface.
- Manual kill/flatten command path.
- Reconciliation loop.

### Build 3: Book decision pipeline in shadow

Why third: prove the new brain reproduces today before it controls capital.

Tasks:

- Rebalance run state machine.
- Strategy intent contract.
- ETF trend intent adapter.
- Cash residual allocator.
- Target book constructor v1.
- Whole-book risk report.
- Execution plan generation without sending orders.

Only after these should you route live ETF trend through the brain.

---

## 21. What I would deliberately not build yet

Do not build now:

- Kafka.
- Kubernetes.
- Multi-service OMS.
- Max-Sharpe optimizer.
- Intraday execution algos.
- Autonomous LLM trading agent.
- Complex factor model with dozens of unstable factors.
- Cross-venue proxy substitution engine.
- Options support before futures/equities are clean.
- Auto-flatten dead-man on every heartbeat miss.

These may feel “state of the art,” but for this book they are distraction or fragility.

---

## 22. Initial target architecture milestones

### Milestone A — Measurement brain

The system observes the whole book but does not control it.

Output:

```text
Current book
Factor exposures
Strategy virtual ownership
Broker reconciliation
Risk utilization
```

### Milestone B — Shadow decision brain

The system computes target book and execution plan, but old sleeves still trade.

Output:

```text
What brain would have done
Diff vs actual
Risk report
```

### Milestone C — Single-venue control

ETF trend and cash go through brain on Alpaca.

Output:

```text
One run
One target
One risk approval
One Alpaca execution plan
One reconciliation
```

### Milestone D — Multi-venue read-only

IBKR appears in BookState, no orders.

Output:

```text
Alpaca + IBKR consolidated state
Futures exposure/margin model
```

### Milestone E — Multi-venue controlled book

Futures strategies trade through the same brain.

Output:

```text
One book across Alpaca + IBKR
Unified risk budget
Unified kill switch
```

---

## 23. Opinionated design choices and trade-offs

### 23.1 Postgres event/snapshot ledger vs full event sourcing

Recommendation: **snapshot ledger with immutable events**, not pure event sourcing.

Why:

- Easier for solo operator.
- Easier to query/debug.
- Enough determinism for weekly book.
- Avoids building a replay framework before you need it.

What would flip it:

- Many intraday strategies.
- Need for high-frequency event replay.
- Multiple independent workers writing state.
- Regulatory-grade event reconstruction beyond your current needs.

### 23.2 One trading daemon vs many workers

Recommendation: **one trading daemon**.

Why:

- Weekly cadence.
- Lower operational burden.
- Easier locking/idempotency.
- Easier debugging.

What would flip it:

- Intraday strategies return.
- Many venues.
- Slow broker operations blocking decisions.
- Need to isolate strategy compute from execution.

Even then, split carefully: strategy worker, constructor, executor — not a microservice maze.

### 23.3 Strategy-level covariance vs instrument-level optimizer

Recommendation: **strategy-level robust covariance + factor limits**.

Why:

- Small number of streams.
- Avoid noisy instrument covariance.
- Easier to explain.
- Robust to regime shifts.

What would flip it:

- Many independent strategies.
- Large cross-sectional book.
- Strong validated factor risk model.
- Need to optimize borrow/margin/capital efficiency.

### 23.4 Manual base risk budgets vs optimizer-determined weights

Recommendation: **manual budgets with deterministic risk scalers**.

Why:

- Expected returns are noisy.
- Backtest Sharpe is not enough.
- Solo operator needs explainability.
- Prevents optimizer from over-allocating to fragile streams.

What would flip it:

- Long live track record.
- Stable capacity estimates.
- Independent risk committee process.
- Robust forward-looking return estimates.

### 23.5 Exact netting vs factor substitution

Recommendation: **exact netting now; factor substitution later**.

Why:

- Keeps research/live parity.
- Avoids hidden behavior.
- Cross-venue substitution has operational consequences.

What would flip it:

- Larger book.
- Explicit mandate to minimize financing/spread.
- Clear equivalence classes and tested substitution logic.

---

## 24. The question behind the question

You are asking for a “portfolio brain,” but the deeper question is:

> **Can MrTrader become a controlled investment process rather than a collection of successful scripts?**

The architecture must force that answer to be yes.

That means every live dollar must be traceable to:

```text
strategy signal
→ strategy intent
→ risk budget
→ target book
→ factor exposure
→ risk approval
→ netted order
→ fill
→ reconciled position
→ attribution
```

If any live order cannot be traced through that chain, it should not exist.

---

## 25. Final recommendation

Build the portfolio brain as a deterministic target-book system. Do not evolve the current silo scheduler by adding more sleeve-specific rebalancers. Do not add IBKR as a parallel path. Do not build a fragile optimizer. Do not put an LLM in charge of sizing or execution.

The correct architecture for MrTrader is:

```text
Frozen broker + market state
        +
Strategy intents from all sleeves
        ↓
Robust risk-budgeting constructor
        ↓
One proposed target book
        ↓
Whole-book risk and factor gate
        ↓
Netted execution plan
        ↓
Broker adapters for Alpaca/IBKR
        ↓
Fills, reconciliation, immutable ledger
```

Right-sized implementation:

- Python.
- Postgres.
- Redis for ephemeral coordination only.
- One trading daemon.
- FastAPI dashboard separate.
- Broker adapters.
- Immutable run ledger.
- Robust risk budgeter with shrinkage/stress/caps.
- Kill switch and fail-closed reconciliation.

That is the state-of-the-art version for your actual constraints: not trendy, not overbuilt, but structurally correct.

If I were signing off on this as a PM/CRO, my gating rule would be blunt:

> **No futures capital, even paper-live, until IBKR is behind the broker adapter and the consolidated BookState/risk gate can see Alpaca + IBKR as one book.**

That is the line between a controlled portfolio platform and a collection of correlated bets that happen to share a database.
