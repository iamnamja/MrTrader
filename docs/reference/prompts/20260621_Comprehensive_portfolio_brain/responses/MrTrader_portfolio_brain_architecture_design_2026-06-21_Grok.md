# Portfolio Brain Architecture: Target Design for MrTrader's Systematic Book

**Author:** Grok (xAI) — Systematic Trading Systems Architect  
**Date:** June 21, 2026  
**Version:** 1.0 — Opinionated, first-principles redesign for solo operator

This document provides the complete, buildable target architecture based on a deep review of all provided files (01-08). It replaces the current PM→RM→Trader + silo sleeves model with a cleaner **Target-Portfolio / Risk-Budgeting** core. It is right-sized for a solo operator, evolves the existing Python/FastAPI/Postgres/Redis/Alpaca stack, prioritizes determinism/auditability/fail-closed, and future-proofs for growth.

## North-Star One-Paragraph Design
The Portfolio Brain is a **weekly atomic Target Portfolio Engine** that ingests signals from registered strategies (each emitting desired exposures + metadata), builds a consolidated Book State (open + proposed across Alpaca + IBKR), applies a robust risk-budget allocator (equal-risk-contribution with shrinkage + hard de-gross on realized correlation spikes), nets positions/factors at book level, generates minimal trades, executes via broker adapters, and logs everything immutably for full replay. Execution decouples into a daemon; the web dashboard remains lightweight. LLMs stay advisory/monitoring only. This yields one coherent book with controlled risk, netting, and extensibility via declarative registration — all reconstructible from a single run-id snapshot.

## Tier 1: Core Architecture (D1, D2, D3, D4)

### 1.1 Overall Component Decomposition (Replaces PM/RM/Trader)
I recommend replacing the per-proposal agent pipeline with a **book-level Target Portfolio model**. Justification: Strategies are low-frequency rules-based sleeves; per-symbol proposals + per-trade gating doesn't scale to holistic coordination. Book-level reasoning prevents stacked beta, unnetted trades, and emergent risk.

**Components & Responsibilities:**
- **Strategy Registry & Signal Collectors** (declarative): Register strategies that implement `Strategy` contract (see D6). Weekly job calls `get_desired_exposures(current_book_state)` → list of `DesiredPosition` (symbol/venue/contract, target_weight_or_qty, signal_strength, risk_class).
- **Book State Service** (Single Source of Truth — see D2): Central in-memory + persisted model.
- **Portfolio Constructor / Risk Allocator**: Takes proposed targets + current book → solves for book target under single risk budget (robust recipe below).
- **Netting & Conflict Resolver**: Nets overlapping exposures, resolves conflicts at factor/book level.
- **Execution Planner**: Computes minimal trade deltas → routes to Broker Adapters.
- **Broker Abstraction Layer** (D5): AlpacaAdapter, IBKRAdapter (ib_insync).
- **Reconciler**: Periodic + pre-trade sync with broker reality.
- **Audit & Snapshot Logger**: Immutable logs per run.
- **Kill Switch & Monitor**: Global across venues.

**Data Flow (Weekly Rebalance Loop — Hybrid Scheduled + Event-driven):**
1. Scheduler triggers `WeeklyBookDecision(run_id)` at fixed time (e.g., Mon 09:30 ET) — atomic.
2. Snapshot inputs → BookState.
3. Strategies emit desired exposures → ProposedBook.
4. Risk Allocator → BookTarget.
5. Netting → FinalTarget.
6. ExecutionPlanner → TradeSet (minimal deltas).
7. Pre-trade Reconcile + Risk Gate → Execute via adapters.
8. Post-fill Reconcile → Update BookState + Audit.

**Control Model:** Hybrid. Main loop is **scheduled atomic weekly** (avoids race conditions of staggered sleeves). Heartbeat (every 5-15min) for reconciliation/monitoring/kill-switch. Event-driven for urgent kills or anomalies. This makes the weekly decision atomic and reproducible.

**Deployment Topology (Solo-friendly):**
- FastAPI web app: Dashboard + API only (light queries to BookState).
- Separate **Execution Daemon** (`brain_daemon.py`): Runs scheduler, BookState, allocator, adapters. Uses APScheduler + Redis for commands. Minimal split: one process for now (evolves to 2-3 workers if needed). Decouples trading from web.

**Text Diagram:**
```
Scheduler (Weekly) --> BookStateService (Postgres + In-Mem) 
  --> Strategies (emit Desired) --> PortfolioConstructor/RiskAllocator 
  --> Netter/FactorView --> ExecutionPlanner --> BrokerAdapters (Alpaca/IBKR)
  <--> Reconciler (bidirectional) <--> AuditLog
KillSwitch (Redis flag + heartbeat) --> All Adapters
Dashboard API <-- BookState queries
```

### 1.2 Consolidated Book-State Model (D2)
**Single Source of Truth:** `BookState` class (in-memory primary, persisted to Postgres event log + periodic snapshots). Rebuilt on restart from event log + broker reconciliation.

**Data Model Sketch (Python/SQLAlchemy-inspired):**
```python
class BookState:
    run_id: str  # e.g., "2026-06-23-WEEKLY-001"
    timestamp: datetime
    positions: List[Position]  # open + pending
    proposed_targets: List[DesiredPosition]  # this cycle
    aggregate_exposures: Dict[str, float]  # gross_net, by_asset_class
    factor_exposures: Dict[Factor, float]  # equity_beta, vol, etc.
    risk_metrics: RiskMetrics  # vol, drawdown, margin_util, realized_corr
    version: int

class Position:
    id: str
    symbol: str
    venue: str  # 'alpaca' | 'ibkr'
    qty: float
    avg_price: float
    strategy_owner: str
    status: 'OPEN' | 'PENDING' | 'CLOSED'
    # + venue-specific (margin, contract)

# Postgres: events table (immutable append-only) + snapshots table
```

**"What's Being Considered":** `proposed_targets` populated before any trades. Brain reasons over `current + proposed`.

**Consistency with Brokers:** 
- Reconciliation job: Fetch positions/cash/margin from each adapter → diff vs DB → apply corrections (fail-closed on large diffs).
- "DB is not reality": Broker is ground truth. Pre-trade: full reconcile. On mismatch > threshold: halt + alert.
- Cross-venue aggregation: Normalize to USD notional/equity equivalents for unified view.

### 1.3 Holistic Sizing Under One Risk Budget (D3) — Robust Recipe
**Core Policy:** Risk budgeting by **vol-target + Equal Risk Contribution (ERC) with shrinkage**, plus hard rules. Avoid full optimizers.

**Concrete Recipe (Solo-defensible):**
1. Each strategy emits base target (e.g., inverse-vol weighted within sleeve).
2. Compute realized rolling corr matrix (60-90 day, shrunk towards equal-corr or historical average: `shrunk = 0.7*empirical + 0.3*target`).
3. Base allocation: ERC (risk parity) across strategies to meet book vol target (e.g., 10-15% annualized) or drawdown budget.
4. **Robustness Layer:** If realized pairwise corr spike (>0.7 or +0.3 from 30d avg) → hard de-gross multiplier (0.6-0.8) on total gross. Regime signal (from NIS/LLM advisory) can bound it.
5. Per-strategy cap + book factor limits gate final.
6. **Paper Ramp:** New strategy starts at 10-20% risk weight. Ramp linearly over 6-12 weeks of live evidence (e.g., based on realized Sharpe > threshold, tracked in metadata). Encoded as `live_weight_multiplier` in registry, updated by audit job.

This captures diversification without overfitting (shrinkage + hard trigger beats pure optimizer). Trade-off: Slightly conservative in normal times; flips to more aggressive with better regime modeling (future).

### 1.4 Netting, Conflict Resolution, Factor View (D4)
- **Netting:** Aggregate all Desired + Open by normalized instrument (futures roll-aware in adapter). Compute book-level target weights → deltas.
- **Conflicts:** Net at book level (honor net exposure). Independent risk-taking is preserved via strategy signals, but book vetoes excessive gross. Argue: Netting reduces costs/friction without killing edge (correlated longs/shorts cancel).
- **Factor View:** Pre-computed betas/durations/etc. (from data or adapters). Netted book limits (e.g., equity_beta <=1.2) gate allocator.

## Tier 2: Cross-Venue, Safety, Extensibility (D5, D6, D7)

### Broker Abstraction (D5)
`BrokerAdapter` abstract base: `place_orders(trades), get_positions(), get_cash_margin(), flatten()`.
- Venue specifics (IBKR rolls, margin) handled inside adapter + asset mappers.
- Aggregation: BookState normalizes.
- **Kill Switch:** Redis flag + daemon heartbeat. On trigger: `adapter.flatten_all()` idempotently for both. Dead-man: If daemon misses 2 heartbeats, external monitor kills.

### Extensibility (D6)
- **Strategy Contract:**
```python
class Strategy(ABC):
    @abstractmethod
    def get_desired_exposures(self, book_state: BookState) -> List[DesiredPosition]:
    metadata: dict  # venue, asset_class, risk_class, ramp_config
```
Register: `registry.register(ETFT rendStrategy())`

- Venue/Asset: New adapter + asset normalizer. Brain unchanged.
- Research-Live Parity: Strategies have `backtest_mode`/`replay_mode` using same logic + historical snapshots.

### Safety, Determinism, Audit (D7)
- **Idempotency:** Every decision keyed by `run_id`. Snapshot inputs to Postgres. Re-run replays exactly (deterministic strategies).
- **Audit Log:** Append-only events: `BookDecisionStarted(run_id, inputs_snapshot)`, `TargetsComputed`, `TradesExecuted`, `Fills`.
- **Failure Modes & Guardrails:**
  - Reconciliation mismatch: Halt.
  - Stale data: Skip cycle or de-gross.
  - Crash mid-rebalance: Idempotent retry uses snapshot.
  - Most feared: Silent divergence — mitigated by daily parity checks + immutable logs.

## Tier 3: LLMs & Migration (D8, File 06)

### LLMs in the Loop
**Verdict:** Keep deterministic core pure. LLMs **IN** for advisory/monitoring (worth building after core, not now — prioritize deterministic brain). **OUT** for sizing/execution.

**IN Roles (Bounded):**
1. **Research/Design:** Offline, full use.
2. **Narrative/Event Monitor:** Near-real-time scan → bounded de-gross signal (max -20% gross) or alert. Human confirm for anything stronger. Input: News feeds + book_state summary. Fallback: No signal.
3. **Anomaly/Sanity:** Review proposed book + news → flag/hold to human. No auto-trade.
4-6. Post-trade, Regime Narration, Copilot: All advisory, logged, human-in-loop.

**Guardrails:** Log all calls. Fail-safe (deterministic runs). Bounded actions only. Audit prompt+output.

**Not Yet:** Build monitoring layer after deterministic core solid (distraction risk high initially).

### Migration Path (Staged, Strangler)
1. **First:** Build BookState Service + Reconciler + Broker Abstraction (wrap existing Alpaca). Mirror live sleeves. ETF trend stays direct initially.
2. **Next:** Route ETF Trend + Cash through brain (atomic weekly). Add IBKR adapter + futures strategies (paper initially).
3. **Strangle:** Disable old silos one-by-one, route via registry. Live book never dark.
4. **Final:** Full risk allocator, netting, LLM monitoring.

## Closer
- **First Three Builds:** 1. BookState + Reconciler (foundation). 2. Broker adapters + kill switch. 3. Strategy registry + weekly decision skeleton.
- **Biggest Solo Mistake:** Over-engineering for scale too early / black-box risk. Avoided via simple in-mem + Postgres, declarative, full audit.
- **Unasked:** Instrument everything for observability (e.g., Prometheus metrics on risk utilization). Test replay on historical golden dates religiously.

**Implementation Notes:** Evolves current stack. Start with `brain/` package. Full pseudo-code and diagrams available on request.

**End of Document**
