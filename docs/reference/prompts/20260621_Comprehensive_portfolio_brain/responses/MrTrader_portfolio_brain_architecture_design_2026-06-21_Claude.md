# MrTrader — The Portfolio Brain: Target Architecture

*A systematic-trading systems-architecture design for a solo, multi-strategy, multi-venue, low-frequency book.*
*Scope: the coordination layer only (sizing/netting/risk/execution/state), not strategy/alpha. Strategies are given.*

---

## 0. TL;DR — the thesis in five claims

1. **Replace `PM → RM → Trader` (propose → gate → execute) with a target-portfolio model.** Strategies stop being agents that trade. They become **pure functions that emit desired exposures.** One **constructor** solves for a single book target, a **risk layer** gates the *target* (not per-trade), an **execution planner** diffs target-vs-current into the minimal order set, and **broker adapters** translate. This single inversion — *strategies emit intent, the portfolio is constructed once holistically, trades are a derived diff* — collapses almost every gap in file 04 at the structural level.

2. **The robustness you want does not come from a smarter optimizer. It comes from the architecture.** Do **not** run a mean-variance optimizer that inverts a noisy correlation matrix. Use **Equal-Risk-Contribution (or plain inverse-vol) on a heavily-shrunk strategy-return covariance** as a stable base, and put **all crisis behavior into separate, dumb, *monotonic* de-gross governors** layered on top. Optimizers are for normal times; governors are for tails. That separation is the entire answer to "diversification without blow-up."

3. **The hard problem isn't sizing. It's reconciliation.** The thing that hurts a solo operator at 2 a.m. is trading on a book state that disagrees with the broker after a partial fill / disconnect / restart. **Reconciliation-before-trade, fail-closed, is the single most important invariant in the system** — and it's not even in your Tier-1. I'm elevating it.

4. **One daemon, not microservices. Postgres as the only durable truth; Redis optional and non-authoritative.** Construct/risk/execute are *steps in one deterministic pipeline*, not separate services. Splitting them adds failure modes for zero benefit at weekly cadence. The "state in three places" problem (G9) is solved by demoting Redis to ephemeral cache.

5. **LLMs sit *around* the deterministic core, never inside it — and most of the LLM layer is a distraction from getting the brain right first.** Build post-trade narrative now (pure upside, trivial). Defer everything that touches live risk until the deterministic brain is trustworthy and you have a baseline to measure an LLM signal against.

The rest of this document argues each of these and gives you the data model, pseudo-code, the staged migration, and the failure modes.

---

## 1. The target architecture

### 1.1 Why `PM/RM/Trader` is the wrong abstraction here

The propose → gate → execute pipeline is the correct shape for a **discretionary or proposal-driven** flow: individual trade ideas arrive asynchronously, each needs per-trade approval against a snapshot, and a trader works orders. That is genuinely what the swing/PEAD/intraday ML pipeline is. It is the *wrong* shape for a **fully-systematic, low-frequency, multi-strategy book**, for three reasons that map directly to your gaps:

- It reasons at the **trade** granularity (PM proposes per-symbol, RM approves per-trade, Trader executes per-order). Nobody reasons at the **book** level (G8). The book's actual risk is an emergent accident of independently-approved trades.
- It can't express **"given everything open and everything being considered across both venues, what should the *whole book* be, and what's the minimal set of trades to get there?"** — which is the only question that matters for a coordinated book.
- It made the live sleeves bypass it entirely (G2), because forcing a rules-based weekly rebalance through a per-proposal gate is an impedance mismatch. So your sophisticated risk logic gates the money that *isn't* trading.

Don't patch this. Replace it.

### 1.2 The replacement: a target-portfolio / risk-budgeting / execution-planner core

```
                    WEEKLY BOOK DECISION  (one run_id · atomic · deterministic)
                    ════════════════════════════════════════════════════════════

  [Strategy A]──┐                                    ┌─ reconcile BOTH venues FIRST
  [Strategy B]──┤  desired_exposures(snapshot)       │  (DB == broker? else FAIL-CLOSED,
  [Strategy C]──┼──────────────────────────────────► │   hold current, do not trade)
  [Strategy D]──┘  pure fns → ExposureVectors         │
                                                      ▼
                              ┌───────────────────────────────────────────┐
                              │            BOOK-STATE ASSEMBLER            │
                              │  positions ∪ accounts ∪ market ∪ "wants"   │
                              │        → immutable BookState snapshot       │
                              └─────────────────────┬─────────────────────┘
                                                    ▼
                              ┌───────────────────────────────────────────┐
                              │           PORTFOLIO CONSTRUCTOR            │
                              │  ERC / inverse-vol on SHRUNK strat-cov     │
                              │  → scale to book vol target                │
                              │  → governors (de-gross only; MIN wins)     │
                              │  → TARGET BOOK (net notional / instrument) │
                              └─────────────────────┬─────────────────────┘
                                                    ▼
                              ┌───────────────────────────────────────────┐
                              │          RISK / CONSTRAINT LAYER           │
                              │  factor limits · gross/net · per-venue     │
                              │  margin · position caps → clip / de-gross  │
                              │  deterministic invariants · fail-closed    │
                              └─────────────────────┬─────────────────────┘
                                                    ▼
                              ┌───────────────────────────────────────────┐
                              │            EXECUTION PLANNER               │
                              │  diff(target, current) per venue           │
                              │  net overlaps · no-trade bands · min size  │
                              │  → minimal order set + deterministic IDs   │
                              └──────────┬─────────────────────┬──────────┘
                                         ▼                     ▼
                                 [Alpaca Adapter]       [IBKR Adapter]
                                         │                     │
                                         ▼                     ▼
                                  orders (idempotent by client_order_id)
                                         │                     │
                                         └──────────┬──────────┘
                                                    ▼
                              ┌───────────────────────────────────────────┐
                              │               RECONCILER                   │
                              │  fills → positions · DB := broker reality  │
                              │  persist final BookState · close the run   │
                              └───────────────────────────────────────────┘

   Surrounding, continuous, SEPARATE cadence:  THE MONITOR LOOP
      heartbeat · kill-switch · staleness gate · intra-week drawdown governor ·
      dead-man watchdog · (NEVER opens new risk — can only halt or de-gross)
```

**Component responsibilities:**

| Component | Responsibility | Explicitly NOT its job |
|---|---|---|
| **Strategy** | Pure fn: `(market_snapshot, current_book) → ExposureVector`. Internally vol-balances its own legs. | Sizing to account, touching brokers, knowing other strategies exist. |
| **Book-State Assembler** | Reconcile positions across venues, gather accounts/margin/market data + all strategies' wants → one immutable `BookState`. | Deciding anything. It only *assembles truth*. |
| **Portfolio Constructor** | Allocate risk *across strategies* under one budget; scale to vol target; apply de-gross governors → a target book. | Per-instrument signal logic (that's the strategy); per-trade approval. |
| **Risk / Constraint Layer** | Apply hard book-level invariants to the *target*: factor limits, gross/net, per-venue margin, position caps. Clip or de-gross. | Generating exposure. It only *constrains*. |
| **Execution Planner** | Diff target vs current; net; apply no-trade bands and min-size; emit the minimal order set with deterministic IDs. | Choosing exposures or sizes. It implements the diff. |
| **Broker Adapters** | Translate canonical orders ↔ venue calls; own venue specifics (margin, multipliers, roll, settlement, symbology). | Leaking venue concerns into the brain. |
| **Reconciler** | Make DB == broker reality after fills; persist final state; close the run. | Trading. |
| **Monitor Loop** | Continuous safety: heartbeat, kill-switch, staleness, intra-week drawdown de-gross, dead-man. | Opening new positions. Ever. |

This is what a top systematic shop would recognize as the correct shape for a low-frequency multi-strategy book. It is deliberately boring.

### 1.3 The control model: scheduled trigger, atomic pipeline, separate monitor

Your current scheduler encodes the silo problem: `trend_rebalance 09:45`, `cash_rebalance 09:50`, `crypto 09:55` are **three separate jobs firing minutes apart.** That is three sleeves trading independently by construction.

**Fix: one weekly job that runs the entire pipeline above as a single atomic "book decision run" under one `run_id`.** Not event-driven on ticks (it's a weekly EOD book — there's nothing to react to intraday). The "event" is simply *"it's the rebalance window, data is fresh, market is open → run the book decision."* Internally it's a deterministic DAG with an immutable input snapshot and idempotent steps.

Separately, a lightweight **monitor loop** (your 60s heartbeat, repurposed) runs continuously and does only safety work: reconciliation, kill-switch checks, staleness/heartbeat watchdog, and an **intra-week drawdown governor** that can *de-gross or halt* but **cannot open new positions.** This cleanly separates *"decide the book weekly"* from *"watch the book continuously."* Decision cadence ≠ supervision cadence.

> **Trade-off named:** a pure weekly cadence means a tail event mid-week is only met by the monitor loop's de-gross/halt, not by a fresh construction. That's the right call for a low-freq book — you don't want narrative-driven intraweek rebalancing — but it means your *governors* (not your *constructor*) are your crisis response. Design them accordingly (§3.3). What would flip this: if you later add a genuinely higher-frequency sleeve, that sleeve gets its own faster construction loop feeding the *same* book state — but you do not speed up the weekly book to match it.

### 1.4 Topology: where it runs

**Two processes. That's the minimal robust split for a solo operator.**

- **Web/dashboard (FastAPI + React)** — reads from Postgres, renders book state, factor exposures, run history, and *why the last rebalance did what it did*. It can issue **control commands** (kill-switch, manual de-gross, ramp approval) by writing to a durable command/flag table in Postgres. It runs **no trading logic in-process.**
- **The Brain daemon (one long-running process)** — owns the scheduler, the monitor loop, the weekly decision pipeline, and the broker adapters. **It is the only thing that touches brokers.**

**Resist the temptation to split construct/risk/execute into separate workers.** They are deterministic steps over one shared in-memory snapshot. Making them separate services introduces network failures, serialization, and partial-failure states — real new failure modes — to buy you nothing at weekly cadence and solo scale. Keep them as **modules inside the daemon**, communicating by function calls over an immutable `BookState`. (G9 solved without microservice sprawl, which file 07 explicitly forbids.)

**Drop Redis as authoritative state.** Today Redis holds queues + kill-switch flag + capital stage. In the target design the queues vanish (no proposal→approved handoff; it's one pipeline). The kill-switch flag and capital stage are tiny bits of *durable, auditable* state that belong in Postgres anyway. So: **Postgres is the single source of durable truth; Redis is optional and may hold only ephemeral, non-authoritative cache.** This kills the "state in three places" coupling (G9) outright. *Keep Redis only if you have a concrete need for an ephemeral cache; do not let it hold anything you'd be sad to lose on restart.*

---

## 2. The consolidated book-state model (the source of truth)

### 2.1 Event-sourcing vs current-state — the pragmatic hybrid

- **Pure event-sourcing** (rebuild all state by replaying an event log on every read) is over-engineering for a solo weekly book and is *operationally fragile*: replay bugs, schema-evolution pain, and a lot of machinery to maintain alone.
- **Pure mutable current-state** loses the auditability and reproducibility that file 07 makes non-negotiable.

**Use both, for different jobs:**

- **Materialized current state** = the working truth the brain reads. Reconciled against brokers.
- **Immutable append-only decision log** = the audit/replay spine. One row per weekly decision, capturing the full lineage.

You already have the right instinct: `orders` is your immutable fill ledger; `risk_metrics` is an EOD snapshot. The change is to **make the *book decision* a first-class immutable record**, and to make the *consolidated state* an explicit object rather than something reconstructed ad hoc from three places.

### 2.2 The data model

```python
# ── In-memory during a run, snapshotted immutably to Postgres ──────────────

@dataclass(frozen=True)
class BookState:
    run_id: UUID
    as_of: datetime                  # decision timestamp
    data_date: date                  # the EOD date signals are computed from
    positions: list[Position]        # reconciled, across ALL venues
    wants: dict[StrategyId, ExposureVector]   # "what's being considered" this cycle
    accounts: dict[Venue, AccountState]
    market: MarketSnapshot           # prices, vols, betas, factor loadings, corr
    inputs_hash: str                 # hash of everything above → determinism check

@dataclass(frozen=True)
class Position:
    instrument_id: str               # CANONICAL id (venue-agnostic)
    venue: Venue
    qty: float
    market_value: float
    notional: float                  # signed; for futures = contracts * multiplier * price
    owning_strategy: StrategyId | Literal["SHARED"]
    asset_class: AssetClass
    factor_loadings: dict[Factor, float]   # equity_beta, rates_dv01, usd, commodity, vol

@dataclass(frozen=True)
class ExposureVector:                 # what a strategy WANTS — not orders
    strategy_id: StrategyId
    data_date: date
    targets: list[InstrumentTarget]   # internally vol-balanced by the strategy
    meta: StrategyMeta                # venue, asset_class, risk_class, live_fraction, confidence

@dataclass(frozen=True)
class InstrumentTarget:
    instrument_id: str
    target: float                     # target weight OR target risk fraction (strategy-internal)
    # For options later, this object carries greeks instead of a scalar notional (see §7.3).

@dataclass(frozen=True)
class AccountState:                   # PER VENUE — never netted across brokers
    venue: Venue
    nav: float
    cash: float
    settled_cash: float
    buying_power: float
    margin_used: float | None         # IBKR
    margin_available: float | None
    maintenance_margin: float | None

@dataclass(frozen=True)
class TargetBook:                     # the constructor's output
    run_id: UUID
    per_strategy_risk_weight: dict[StrategyId, float]
    target_positions: list[tuple[str, Venue, float]]   # (instrument, venue, target_notional)
    book_gross: float
    book_net: float
    factor_exposures: dict[Factor, float]
    diagnostics: ConstructorDiagnostics   # WHY: which governors fired, which limits bound,
                                          # correlation state, de-gross factor, shrinkage intensity
```

**Postgres tables:**

- `positions` — current reconciled holdings per venue (the materialized truth).
- `book_runs` — one immutable row per weekly decision: `run_id`, `as_of`, `data_date`, `inputs_hash`, the serialized `wants`, the `TargetBook`, the `diagnostics`, and the run state-machine status.
- `book_snapshots` — the post-run reconciled `BookState` (what the book actually became).
- `orders` — your existing immutable fill ledger (keep it; add `run_id` + deterministic `client_order_id`).
- `governor_log` — every de-gross event with its trigger, magnitude, and inputs (so "why did gross fall?" is a query, not a forensic exercise).
- `strategy_registry` — declarative: `{strategy_id, class, enabled, live_fraction, params, version}`.
- `control_flags` — kill-switch, capital stage, manual overrides (durable, replaces the Redis flags).

### 2.3 "What's being considered" — no longer scattered

In the target architecture there is no Redis queue of in-flight proposals and no in-memory `approved_symbols` fast-path to drift. **"What's being considered" is exactly `BookState.wants`** — the set of `ExposureVector`s emitted by all strategies for this run, captured atomically as part of the run snapshot. The **"proposed book"** is the constructor's `TargetBook` *before* execution: fully materialized, inspectable, and persisted. (This is precisely where a human confirmation or an LLM sanity-check can sit — §6.) G3 is solved by construction: the consolidated object isn't reconstructed from three stores; it's the single input the pipeline is built around.

### 2.4 "The DB is not reality" — reconciliation across two brokers

This is the part I want you to treat as load-bearing, not plumbing.

**Reconciliation-before-trade is a hard gate at the top of every run.** Before the constructor runs, the assembler pulls live positions and accounts from *every* adapter and compares to the DB. Rules:

- **Match → proceed.**
- **Mismatch within a tiny tolerance** (rounding, a known pending fill) → resolve toward **broker reality** (the broker is truth for positions/cash; the DB is a cache of it), log the adjustment, proceed.
- **Material mismatch you can't explain** → **FAIL-CLOSED: do not trade this cycle. Hold current positions. Alert.** A book that trades on a wrong view of what it holds is how you turn a small bug into a large loss. The cost of skipping a weekly rebalance is one week of slightly-stale exposure. The cost of trading on a phantom position is unbounded. The asymmetry is the whole point.

Two-broker subtlety: Alpaca and IBKR have different position/cash/margin semantics (cash-settled equities vs margined futures with multipliers and maintenance margin). The **adapter normalizes** each venue's truth into the canonical `Position` / `AccountState` *before* reconciliation, so the reconciler compares apples to apples. Margin is reconciled **per venue** (you cannot net margin across brokers); exposures are aggregated **across venues** (you absolutely net equity beta across SPY-on-Alpaca and ES-on-IBKR — that's the entire reason this project exists).

---

## 3. Holistic sizing under one risk budget (the core)

This is the section you said you care about most, so here is the concrete, defensible recipe — and the explicit answer to *"diversification without a fragile optimizer that blows up."*

### 3.1 The governing principle

> **Robustness is an architectural property, not an optimization property. Get the diversification benefit from a *stable base allocation*, and get the tail protection from *separate, dumb, monotonic governors.* Never ask the optimizer to handle crises — it will trust a noisy matrix and lever into a regime shift.**

A mean-variance optimizer inverts the covariance matrix. With 3–4 strategy streams and limited, regime-dependent history, the small eigenvalues of that matrix are pure noise, and inversion *amplifies* them — the optimizer piles risk into whatever looked least-correlated last year and de-risks whatever looked redundant, then both assumptions invert in the crisis. That is the blow-up. We will not do that.

### 3.2 The sizing stack (in strict order)

**Layer 0 — Within-strategy sizing stays with the strategy.** Each strategy already balances its own legs (inverse-vol trend, etc.) and emits an *internally-risk-balanced* `ExposureVector`. The brain does **not** re-derive within-strategy weights. Separation of concerns: the strategy knows its structure; the brain allocates *across* strategies.

**Layer 1 — Allocate risk across strategies via Equal-Risk-Contribution (ERC), or just inverse-vol.** Treat each strategy's *return stream* as the unit (3–4 streams, not hundreds of instruments). ERC sets weights so each strategy contributes equal risk to the book — which naturally *reduces* a strategy's allocation as its correlation with the others rises. Two reasons this is the robust default:
- It needs only the covariance of **strategy returns** (a 3×3 or 4×4 matrix), which is far more estimable than an instrument-level matrix.
- It degrades gracefully under estimation error — no matrix inversion chasing tiny eigenvalues.

> **The simpler sibling, which I'd actually start with:** plain **inverse-volatility weighting across strategies** *is* ERC under the assumption of equal pairwise correlation. It has zero matrix-inversion fragility. For a 4-strategy book I would **launch on inverse-vol and only graduate to full ERC once you trust your covariance estimate.** Fewer moving parts is worth more than a marginally better risk balance at this scale.

**Layer 2 — Shrink the strategy covariance, hard.** Whatever you feed Layer 1, shrink it. Ledoit-Wolf toward a structured target (constant-correlation, or even diagonal). With your limited history, **shrink aggressively** — a high shrinkage intensity toward constant-correlation, or toward the diagonal (which collapses ERC toward inverse-vol, i.e., the most robust possible allocation). The shrinkage intensity is itself recorded in the run diagnostics. *The point of shrinkage isn't precision; it's refusing to believe the off-diagonal noise.*

**Layer 3 — Scale to a book volatility target.** Translate the Layer-1 weights into a gross exposure that hits a target annualized **book vol** (e.g., 10%), using the *shrunk* covariance. **Not Kelly** — Kelly is too aggressive and too sensitive to edge mis-estimation for real capital, and you said so. Vol-targeting plus the governors below is the robust path.

**Layer 4 — Apply de-gross governors (each *monotonic*, *bounded*, *fail-closed*; compose by taking the MINIMUM scaler).** These are dumb on purpose. None can ever *increase* exposure. They are the crisis response.

| Governor | Trigger | Action |
|---|---|---|
| **Realized-correlation spike** | Rolling realized cross-strategy correlation rises above a threshold (diversification is *failing*) | Scale total book gross down by a bounded multiplier |
| **Drawdown** | Book drawdown from high-water exceeds tiered thresholds | Step gross down in tiers (e.g., −25% / −50% at successive thresholds) |
| **VIX-term crash governor** | Your existing VIX term-structure crash signal | De-gross (you already run this on trend — generalize it to the book) |
| **Margin** | IBKR margin utilization above a ceiling | Cap futures gross to stay well clear of maintenance margin |
| **Staleness/quality** | Stale data, reconciliation wobble, missing input | De-gross or hold — never trade up on bad info |

Composition rule: `final_gross = base_gross × min(g_corr, g_dd, g_vix, g_margin, g_stale)`. The most conservative governor wins. This is fail-closed by construction — the system can only get *smaller* when something is wrong.

**Layer 5 — Hand off to the risk/constraint layer (§4) for hard caps**, then to the execution planner.

### 3.3 Why this specifically survives a crisis

The diversification benefit is captured by Layers 1–2 in *normal* regimes (where it's real and modest — your ~0.49 average pairwise correlation). When the regime shifts and correlations spike toward 1, **the realized-correlation governor sees it directly in the realized data and de-grosses** — without re-running any optimizer, without trusting any forward estimate, monotonically and auditably. You get the upside of diversification in calm markets and a hard, dumb brake in violent ones. **That is the entire trick, and it is robust precisely because the brake is dumb.**

### 3.4 Paper strategies ramping into the live budget

Encode capital eligibility as **data, not code.** Each strategy carries a `live_fraction ∈ [0,1]` in the `strategy_registry`, multiplied into its risk weight by the constructor:

- **Paper / report-only:** `live_fraction = 0`. The constructor *computes what it would allocate* and logs it (so you see the shadow book), but allocates $0.
- **Ramping:** the fraction steps up — 0.25 → 0.50 → 1.00 — on a schedule **gated by live evidence** (e.g., N weeks live with tracking error vs backtest under a tolerance and realized Sharpe within a band).
- **The trigger may be a rule, but the *step* is human-confirmed.** Paper→real-money is the one irreversible move where you want a human in the loop. So: the system *proposes* "12 weeks clean, ready to ramp to 0.50," and you click to confirm. Automatic + auditable, with a human gate on the irreversible step. Every change to `live_fraction` is an audited row (who/what/when/why).

This is also how a previously-killed sleeve safely returns: it re-enters at `live_fraction = 0` (shadow), and ramps only on fresh live evidence.

---

## 4. Netting, conflict resolution, and the factor-exposure view

### 4.1 Netting is mostly automatic — that's a feature of the architecture

Because you construct **one book target** (the net desired notional per instrument, summed across strategies) and the execution planner **diffs target vs current**, two strategies both wanting SPY exposure collapse into a single net target SPY position and you trade the diff *once*. No double spread (G5 solved at the structural level, for free).

But cross-venue introduces a real distinction:

- **Instrument-level netting** (same canonical instrument, same venue): collapse to one net target, trade the diff. **Always do this. Pure win.**
- **Factor-level *awareness*** (economically-overlapping exposure in *different* instruments/venues — trend long SPY on Alpaca + xsmom long ES on IBKR): you **cannot** net these into one position (different contracts, different brokers), and you shouldn't pretend to. Instead the **factor view** makes the *combined* equity beta visible, and **factor limits gate the book target** (§4.3). So overlapping economic exposure is *risk-netted* even when it can't be *instrument-netted*.

### 4.2 Conflict resolution — net at the book level, but do not suppress a strategy

When strategies disagree, the resolution rule matters and it's a genuine fork. My strong position:

> **Strategies are independent alpha sources entitled to their views. The book nets their views into one position per instrument and risk-limits the *aggregate* factor exposure — but it never arbitrates "who is right" and suppresses a strategy.**

Concretely:
- **Same instrument, opposing views** (trend long crude, carry short crude): hold the **net** and trade the diff. They partially cancel — that's the book taking the net view, which is correct. You don't pay to express both sides of a wash.
- **Correlated but different instruments, opposing-ish views** (long energy equities vs short crude futures): let **both positions exist** — they are distinct bets carrying distinct risk premia — but the **factor view ensures the *net* exposure is what's limited.**
- **The only thing ever suppressed is aggregate risk that breaches a limit**, and that is handled by **proportional de-grossing of the contributors**, never by picking a winner. Suppressing a strategy because it disagrees with another destroys exactly the diversification you are paying for.

> **Trade-off named:** proportional de-gross on a breach is "fair" but blunt — it doesn't preferentially trim the strategy you trust least. What would flip me toward a priority scheme: if you had strong, *validated* conviction differences between strategies (you don't yet — they're all rules-based risk-premia harvesters of roughly comparable pedigree). Until then, proportional is the honest default.

### 4.3 The factor-exposure view

Define a **small, fixed, stable** factor set: `equity_beta, rates_dv01, usd, commodity, vol` (add `credit` if/when relevant). Each instrument has loadings:

- ETFs: proxies / slow regressions (SPY ≈ 1.0 equity beta; TLT → rates duration; etc.).
- Futures: mapped by contract (ES → equity beta; ZN → rates DV01; CL → commodity/energy; VX → vol).

Book factor exposure = Σ over positions of `notional × loading`, computed for the **current** book and the **target** book. Factor limits (max net equity beta, max DV01, max vol exposure, etc.) gate the target; a breach triggers proportional de-gross of the contributing strategies. **G6 solved — stacked beta across venues is now visible and limited.**

> **Keep the factor model simple and slow.** Loadings should come from stable priors or slowly-updated estimates, **not** re-fit weekly to noise. A simple stable factor map beats a sophisticated dynamically-estimated one that injects estimation noise into your risk limits. Robust over clever, again.

---

## 5. Cross-venue: broker abstraction, aggregation, kill-switch

### 5.1 The broker adapter interface

```python
class BrokerAdapter(Protocol):
    venue: Venue
    def get_positions(self) -> list[Position]: ...          # normalized → canonical
    def get_account(self) -> AccountState: ...              # cash / buying power / margin
    def place_order(self, order: Order) -> OrderResult: ...  # idempotent via client_order_id
    def cancel_order(self, client_order_id: str) -> None: ...
    def get_order_status(self, client_order_id: str) -> OrderStatus: ...
    def flatten_all(self) -> FlattenResult: ...             # for the kill-switch (idempotent)
    def normalize_instrument(self, venue_symbol: str) -> str: ...   # venue symbol ↔ canonical id
```

Everything venue-specific — **margin math, contract specs, multipliers, the roll calendar, settlement, symbology** — lives *inside* the adapter and is normalized before it reaches the brain. The brain speaks only canonical instruments, signed notionals, and a unified `AccountState`.

**Roll is a venue concern, not a brain concern.** The IBKR adapter owns the roll calendar and presents a *continuous* instrument to the brain: the brain thinks "long crude," the adapter knows it's the front contract and executes the roll as a mechanical trade. This keeps roll logic out of strategies and out of the constructor entirely.

### 5.2 Aggregation: risk globally, capital locally

- **`BookState.positions`** = union of all adapters' `get_positions()`, normalized → canonical. **Exposure/risk aggregates across venues.**
- **`AccountState` is per-venue and constrains per-venue.** You cannot move margin between Alpaca and IBKR, so the constructor must respect each venue's capital/margin even while reasoning about risk globally. The futures book is sized against IBKR margin; the equity book against Alpaca buying power; the *factor exposure* is limited across both.

### 5.3 The single kill-switch and the dead-man

- **Durable kill flag in Postgres** (`control_flags`), set by: the operator (dashboard), the monitor loop (auto-trip on breach), or the dead-man watchdog.
- **When set:** the monitor loop calls `flatten_all()` on **every** adapter (idempotent — flattening an already-flat book is a no-op), cancels open orders, and the brain **refuses to start any new decision run.** Fail-closed: while set, nothing trades. Every flatten attempt is logged.
- **Idempotency:** flatten reconciles to flat and is safe to call repeatedly — exactly what you want when a halt fires during a partial fill.
- **Dead-man:** the daemon writes a heartbeat timestamp every N seconds. A *separate, tiny* watchdog (systemd timer / cron / external uptime monitor) checks it. If the heartbeat goes stale (daemon crashed or hung), the watchdog acts.

> **A real trade-off, and my recommended default.** Auto-*flattening* on a stale heartbeat protects against tail risk but introduces *liquidation-by-bug* risk — a flaky watchdog could spuriously dump the book. For a solo operator I recommend the dead-man default be **halt-new-risk + loud alert + require human-confirmed flatten**, with **auto-flatten reserved for specific hard conditions** (e.g., margin-call territory at IBKR, where waiting is worse than liquidating). "If the brain stops thinking, the book stops *trading*" should be automatic; "the book gets *liquidated*" should usually want a human, except where margin physics removes that luxury.

---

## 6. Determinism, idempotency, auditability — at the book level

### 6.1 The run as a deterministic, idempotent state machine

Every weekly decision is a `book_run` with a unique `run_id` and an **immutable input snapshot** (the exact prices/vols/betas as of `data_date`, the reconciled positions, the account states, every strategy's `ExposureVector`). Hash it → `inputs_hash`.

- **Determinism:** given the same snapshot, constructor + risk + planner produce *identical* output. No wall-clock reads mid-run, no unseeded randomness, no reading mutable state during the run. Pure functions over the snapshot. (If a strategy needs randomness, seed it from `run_id`.) This is what makes the decision reproducible and replayable.
- **Idempotency:** orders carry a **deterministic `client_order_id = hash(run_id, instrument, venue, target)`.** Both Alpaca and IBKR dedup on client order IDs, so if the run crashes after placing some orders and is retried, re-submitting the same IDs is a no-op at the broker. The planner computes the *desired end state*; re-running **converges** to it rather than blindly re-firing. A crashed run is safe to re-drive to completion.
- **The run state machine:**
  `PENDING → SNAPSHOTTED → CONSTRUCTED → RISK_CHECKED → PLANNED → EXECUTING → RECONCILED → COMPLETE` (plus `HALTED` / `FAILED`). Each transition is persisted; a crash resumes from the last persisted state. Because each step is deterministic over the immutable snapshot, resuming is safe.

### 6.2 Auditability — full lineage from one query

`book_runs` (inputs_hash + wants + TargetBook + diagnostics) ∪ `orders` (fills, tagged with run_id) ∪ `book_snapshots` (final reconciled state) ∪ `governor_log` together reconstruct *exactly* why the book did what it did:

> **inputs → strategy targets → constructed target → which governors fired and which limits bound → planned trades → fills → final book.**

"Why did gross fall this week?" → read `governor_log` for that `run_id`. "Why is the book long equity beta?" → read the `factor_exposures` and the contributing positions. The diagnostics aren't a nice-to-have; they're the thing that keeps the book from being a black box (an explicit anti-goal in file 07).

### 6.3 The failure modes I'd fear most, and the guardrails

| Failure mode | Why it's the scary one | Guardrail |
|---|---|---|
| **DB ≠ broker, and the book trades anyway** | Trading on phantom/missing positions turns a small bug into an unbounded loss. **This is the #1 risk.** | Reconciliation-before-trade as a hard fail-closed gate (§2.4). No reconcile, no trade. |
| **Stale data presented as fresh** | Sizing/governing off last week's prices in a fast tape | Staleness gate on every input; stale → de-gross or hold, never trade up. |
| **Partial fill / disconnect mid-run** | Half a rebalance leaves the book in an unintended state | Idempotent re-drive via deterministic client_order_ids + resume-from-state (§6.1). |
| **Governor inversion / sign bug** | A "risk" control that accidentally *increases* exposure | Governors are structurally `min`-composed and clamped to ≤1.0; assert monotonicity in tests. |
| **Constructor produces a wild book** (bug, bad input) | A single run levers into one factor or one name | Deterministic sanity invariants as a hard pre-execution gate (gross/factor/position caps, and a "target not wildly different from last week without a logged cause" check) → fail-closed hold. |
| **Optimizer fragility** | The classic blow-up: levering into a correlation regime shift | We don't run that optimizer (§3). Stable base + dumb governors by design. |

**The unifying default: every failure path degrades to "hold current positions, do not trade, alert."** Never to "trade on partial/bad info." That is fail-closed made concrete at the book level.

---

## 7. Extensibility / future-proofing

### 7.1 The strategy contract (register-a-strategy)

Adding a strategy = implement one interface + add one declarative row.

```python
class Strategy(Protocol):
    id: StrategyId
    meta: StrategyMeta   # venue, asset_class, risk_class, live_fraction, factor_hints
    def desired_exposures(self, market: MarketSnapshot, current_book: BookState) -> ExposureVector: ...
    # PURE · deterministic · no trading · no account access · no knowledge of other strategies
```

Register in `strategy_registry`: `{id, class, enabled, live_fraction, params, version}`. The constructor iterates enabled strategies, calls `desired_exposures`, includes the result. **No bespoke rebalancer, no scheduler surgery, no teaching the reconciler about it** — the reconciler works on *positions tagged by `owning_strategy`* generically, so a new strategy's positions are just tagged with its id. **G10 solved; the file-07 future-proofing bar met** ("adding a strategy = implementing a small contract + registering it").

### 7.2 The venue/asset contract

Adding a venue = implement `BrokerAdapter` + register it. **The brain does not change.** Adding an asset class = the adapter normalizes it to canonical instruments and the factor map gets the new instruments' loadings.

### 7.3 The one honest exception: options break the linear-exposure assumption

Equities and futures are well-described by `notional × factor_loading` — a linear exposure. **Options are not** (their risk is delta/gamma/vega, non-linear in spot and vol). So options are *"contained but not free."* Two ways to handle it, and I recommend the first:

- **Design `InstrumentTarget` / `Position` to carry a general "exposure contribution" that *can* hold greeks** from day one. Linear instruments use the scalar-notional case; options populate delta/gamma/vega. Then options slot into the factor view (vega → the `vol` factor, delta → `equity_beta`, etc.) without a rewrite. **Future-proof the data model now for the one hard case later.**
- Or accept that options is the single extension that touches the exposure model, and cross that bridge if/when you trade them.

Given your VRP sleeve is *futures*-based (short front VIX future, not options), you may never need option greeks — but spending a little design budget now to make the exposure object greek-capable is cheap insurance against the one extension that would otherwise force a rewrite.

### 7.4 Research↔live parity is structural, not aspirational

The same `desired_exposures` function runs in backtest (fed historical snapshots) and live (fed the current snapshot) — **same code, same output structure, no separate "live sizing" path to drift.** The *constructor* is also the same code in both: you backtest the **whole book** by replaying the constructor over historical strategy targets.

**Replay/parity CI:** keep a set of **golden dates** with frozen input snapshots and expected `TargetBook` outputs. CI replays the brain over them and asserts the constructed book matches. Any code change that silently alters the book fails CI. This is the structural guarantee that a new sleeve's live behavior matches its validated backtest, and that refactors don't quietly change the money.

### 7.5 Swapping the sizing policy

The sizing policy sits behind a stable interface:

```python
class SizingPolicy(Protocol):
    version: str
    def allocate(self, wants, covariance, budget, governors) -> dict[StrategyId, float]: ...
```

Swap `InverseVol → ERC → RegimeAware` behind it; the old policy stays reproducible (versioned, and `book_runs` records which version ran). Strategy-pattern, independently testable — exactly the file-07 bar ("swap sizing policy = strategy-pattern swap behind a stable interface").

---

## 8. LLMs in the loop — the IN/OUT map, with honest sequencing

### 8.1 The principle

> **LLMs sit *around* the deterministic core (explain, narrate, monitor, query, flag-to-human), never *inside* it (decide, size, order). Any live influence is bounded to monotonic de-risking, fail-safe to "the deterministic system runs unchanged," logged, and human-confirmed until a long track record earns autonomy.**

### 8.2 LLM-OUT — absolutely, and here's the hard argument

**The deterministic sizing/execution path. No LLM chooses weights or sends orders. Non-negotiable.** Your instinct is right and the argument is stronger than "it's risky":

- It is **logically incompatible with the system's foundational property.** §6 makes the sizing/execution path a *pure deterministic function* — that's what gives you reproducibility, idempotent re-drive, and full audit. An LLM is non-deterministic by nature. You cannot have both. Putting an LLM in the path doesn't degrade determinism; it *abolishes* it, and with it every safety guarantee in §6.
- It opens a **prompt-injection surface in the money path** — a crafted headline or filing could manipulate sizing. Unacceptable.
- It makes "why did the book do this?" unanswerable, which is the explicit black-box anti-goal.

So: never. Full stop.

### 8.3 LLM-IN — in priority/sequencing order

**1. Post-trade / reconciliation analyst & book commentary (offline, after the fact) — BUILD FIRST.**
Highest value, lowest risk, and it teaches you what the brain is doing. After each run, an LLM reads the structured `book_run` (diagnostics, governors, factor deltas, fills, slippage) and drafts the book commentary: what changed and why, tracking-error and live-vs-backtest narrative.
- *Input:* structured run data (read-only). *Action:* writes prose. **Zero control authority.**
- *Fallback:* unavailable → no narrative; nothing else breaks.
- *Audit:* logged to `llm_call_log` (you already have this).
- **Pure upside. Build it alongside the brain.**

**2. Regime narration / morning situational brief (offline, read-only).**
Turns consolidated `BookState` + market context into a human-readable brief for you each morning. Read-only, no control, same fallback. High value for a solo operator's situational awareness; low risk.

**3. Operator copilot / natural-language query (read-only; any action human-confirmed).**
"What's my net equity beta across venues?" → LLM translates to a query over `BookState`, returns the answer. May *propose* actions ("consider de-grossing"), but any action is **human-confirmed and routed through the deterministic system** (it can suggest setting a governor; you confirm; the deterministic system executes). Bounded to **read + propose, never act.** Medium value, low risk *iff* strictly read/propose.

**4. Narrative / event-risk monitor → BOUNDED de-gross signal (the only one touching live risk) — DEFER.**
An LLM scans news/filings/macro for tail catalysts the rules don't encode and feeds a **bounded, monotonic de-gross signal** into the governor layer: it can only *reduce* gross, by at most a capped amount, and the deterministic system runs unchanged if the LLM is unavailable or uncertain.
- **My honest verdict: not yet.** This is the most dangerous integration (hallucinated risk-off that whipsaws the book; prompt-injection via crafted news; silent drift; over-trust), and its marginal value is *speculative* for a weekly book that **already has** a VIX-term crash governor and drawdown governors handling the *measurable* risk-off conditions. The unproven gain is "catching a *narrative* catalyst the rules miss"; the failure modes are nasty.
- **So: design the seam (a bounded de-gross input to the governor layer) but don't build it until the deterministic brain is solid and you've run it long enough to (a) trust it and (b) have a baseline to measure the LLM signal against.** When you do build it, make it **human-confirmed** for a long while ("I see elevated event risk, suggest de-gross X%" → you confirm → deterministic system applies). Autonomous-bounded only after a real track record.

**5. Anomaly / sanity layer over the proposed book (per-cycle, veto-to-human not veto-to-market) — LATER, and secondary to deterministic invariants.**
After the constructor produces the `TargetBook` but before execution, an LLM reviews it ("does this set of trades make sense given state/news?") and can **flag and HOLD for human review** — it can pause and escalate, never place or modify.
- *Critical asymmetry:* its **active flag** can hold a run, but its **absence/unavailability must NOT halt trading** (don't let the LLM become a single point of failure that blocks all rebalancing). Only a positive flag holds.
- **But build the *deterministic* sanity invariants first** (gross/factor/position caps, "target not wildly different from last week without a logged cause"). Those catch most of what matters, deterministically and reliably, and they fail closed. The LLM sanity layer is a *softer second look* on top — never the primary safety mechanism.

### 8.4 The LLM failure modes I'd fear, and the design that prevents them

| Failure mode | Prevention |
|---|---|
| **Hallucinated risk-off whipsaws the book** | Live influence is de-gross-only, capped, and (early on) human-confirmed; the deterministic governors are the real crisis response. |
| **Prompt-injection via news/filings** | Source whitelisting; LLM output is advisory-only with a hard cap on authority; it can never *increase* exposure or place orders. |
| **Silent drift** (the LLM slowly gets worse) | Everything logged to `llm_call_log`; the LLM never holds authoritative state; its outputs are reconstructable and reviewable. |
| **Over-trust** | The deterministic system *always runs unchanged* without the LLM; the LLM's absence changes nothing. You can unplug it at any moment and the book is unaffected. |

### 8.5 The honest bottom line on LLMs

**Most of the LLM layer is a distraction from getting the deterministic brain right first.** Build the brain. Add post-trade narrative + morning brief alongside (pure upside, trivial, and they accelerate your own understanding). Defer every live-influencing LLM role until the deterministic core is trustworthy and you have a measured baseline. **The order is: deterministic brain → LLM narration → (much later, maybe) bounded LLM monitoring.** Not the reverse.

---

## 9. The migration path — strangler, ETF-trend never goes dark

Each stage is independently valuable, reversible, and keeps the live book trading. No big-bang (file 07's hard anti-goal).

**Stage 0 — Decouple the daemon from FastAPI. (Mandatory before any IBKR capital.)**
Move scheduler + trading into a standalone **Brain daemon**; the web becomes read-only + control-command issuer. Pure infra, no behavior change, de-risks everything downstream. Your external panel and your own review already flagged this as mandatory. **This is first.**

**Stage 1 — Build the consolidated book-state / measurement layer in SHADOW mode. (Measure before you control.)**
Build `BookState` assembly: reconcile positions across venues (Alpaca now; stub IBKR), compute aggregate + factor exposures, snapshot to Postgres. Run it *alongside* the live sleeves in **observe-only** mode — it records what the book *is* and what a holistic risk view *would say*, but controls nothing. **You learn whether your factor model and consolidated view are correct *before* you let them drive trades.** (You suspected this is first — it's second only to decoupling, because measurement must run on a decoupled daemon.)

**Stage 2 — Convert the live strategies to emit desired exposures (behavior-preserving refactor) + build the constructor to *parity*.**
Refactor `trend_sleeve` and `cash_sleeve` so that instead of computing-and-trading, they emit `ExposureVector`s to the new constructor — and the constructor first **reproduces exactly today's behavior** (single strategy, `live_fraction = 1`, no cross-strategy sizing yet). **Parity gate: the new pipeline must reproduce the old sleeve's trades on golden dates before it takes over.** Strangle the old sleeve only once parity is proven.

**Stage 3 — Turn on holistic construction for the live Alpaca book (trend + cash).**
The constructor now does **joint sizing across the live Alpaca strategies under one budget**, with the governors. Still one venue. You get netting + book-level risk + the factor view on live money. The old per-sleeve rebalancers are retired. **The RM's sophisticated logic is reborn as the book-level risk/constraint layer — finally gating live money (G2 solved).**

**Stage 4 — Add the IBKR adapter; bring futures in at fractional `live_fraction`.**
Implement `BrokerAdapter` for IBKR (ib_insync). Bring carry + xsmom in at low `live_fraction` (paper→live ramp), now **coordinated with the Alpaca book through the same constructor** — cross-venue factor view live, stacked beta finally visible (**G1/G6/G7 solved**). VIX-VRP similarly, with its crash-gate intact. The single kill-switch now spans both venues.

**Stage 5 — Safety/observability polish + LLM narrative.**
Dead-man watchdog, deterministic sanity invariants, replay/parity CI, the dashboard's "why did the book do this" view, LLM post-trade commentary + morning brief. *Then* consider the deferred LLM monitoring roles (§8.3 #4–5).

Throughout: ETF-trend keeps trading on the old path until Stage 3 strangles it, and only after parity is proven. Evolution, not revolution — matching your production posture (roll back over ship untested).

---

## 10. The closer

### The one-paragraph north-star

> **MrTrader's portfolio brain is a single deterministic weekly pipeline that treats strategies as pure functions emitting desired exposures, assembles one reconciled book-state across all venues, constructs one book target by allocating risk across strategies with a robust shrunk inverse-vol/ERC base and a stack of dumb monotonic de-gross governors for the tails, gates that target with hard book-level factor and margin limits, and executes it as the minimal netted diff through pluggable broker adapters — every run immutable, reproducible, idempotently re-drivable, and fully reconstructable, with LLMs strictly around the core for narration and (eventually, bounded) monitoring, and with every failure path degrading to "hold and alert" rather than trading on bad information.** Boring, auditable, and correct — sized for one person, extensible to many strategies and venues without a rewrite.

### The first three things I'd build, in order

1. **Decouple the Brain daemon from FastAPI (Stage 0).** You cannot safely run any of this coupled to the web server, and it's mandatory before IBKR capital regardless.
2. **The consolidated book-state / measurement layer, in shadow mode (Stage 1).** You can't control what you can't measure. Run it observe-only against the live book and verify your factor model is right *before* it drives a single trade.
3. **Strategies-emit-exposures + the constructor, proven to parity with today (Stage 2).** You can't safely cut over what you can't reproduce. Parity on golden dates is the gate that lets you strangle the old sleeves without risk.

In that order, because each is a precondition for trusting the next.

### The single biggest mistake a solo operator makes building this

**Building the clever optimizer first (the seductive part) instead of the boring measurement / reconciliation / state / parity layer (the part that actually determines whether you blow up).** And its twin: **letting the portfolio optimizer be responsible for crisis behavior.** The blow-up is never a mediocre Sharpe — it's an optimizer that levered into a correlation regime-shift because it trusted a noisy matrix, on a book whose true positions it wasn't even sure of. This design avoids both by **sequencing measurement-and-reconciliation first**, and by **separating the stable base allocation from the dumb monotonic governors** so crisis behavior is robust by construction, not optimizer-dependent.

### The question behind the question — what you didn't ask, and I'd insist on

Three things your Tier-1 underweights:

1. **Reconciliation is the real architecture, more than sizing.** I've said it three times because it's true: a solo operator gets hurt at 2 a.m. by a DB-vs-broker mismatch after a partial fill, not by covariance shrinkage. **Reconciliation-before-trade, fail-closed, must be a first-class invariant**, and "hold and alert" must be the default of every failure path. Promote this above sizing in your build priority.

2. **The diversification you're sizing against is *conditional*, and inverts exactly when it matters.** Your strategies are not all the same *kind* of risk. **Trend is *divergent* (long volatility, pays off in crises); carry and VRP are *convergent* (short volatility, blow up together in crises).** Their pairwise correlation in calm markets (~0.49) *understates* their crash co-movement. So: encode a **`risk_class` per strategy** (divergent / convergent), give **convergent/short-vol sleeves a structural haircut and a crash-gate** (you already crash-gate VRP — *generalize* it to all convergent sleeves), and make the governors aware that convergent strategies cluster in the tail while trend hedges them. **The correlation that should drive your gross is the *tail* correlation, not the full-sample average.** This is where your strategy knowledge and this architecture meet, and it's the easiest thing to get subtly, expensively wrong.

3. **Build the "shadow book" as a permanent fixture, not just a migration tool.** The observe-only constructor from Stage 1 — computing what each strategy (including paper ones at `live_fraction = 0`) *would* contribute — should stay forever. It's how paper strategies prove themselves before ramping, how you catch a constructor regression (shadow vs live divergence), and how you sanity-check the live book every week against what a clean holistic view says it should be. The shadow book is your cheapest, most honest monitor — keep it running permanently.

---

*End of architecture document. Push back hard on any of it — the conditional-correlation point (§10) and the reconciliation-first priority are the two hills I'd defend most stubbornly; the inverse-vol-before-ERC and drop-Redis calls are the ones most worth your scrutiny.*
