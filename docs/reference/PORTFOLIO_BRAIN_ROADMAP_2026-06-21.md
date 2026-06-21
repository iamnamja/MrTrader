# MrTrader Portfolio Brain — Architecture, Design & Implementation Roadmap

**Status:** SSOT for the platform re-architecture. **Version:** v2.1 (post adversarial self-review +
verification gate). **Adoption:** ready (verification: 8/8 prior blockers closed; the one re-order
attribution gap fixed in v2.1).
**Date:** 2026-06-21.
**Inputs:** the 5-LLM Portfolio-Brain architecture panel (ChatGPT/Claude/DeepSeek/Gemini/Grok — raw
in `docs/reference/prompts/20260621_Comprehensive_portfolio_brain/`), fused with the Go-Live
strategy/risk synthesis (`docs/reference/GOLIVE_REVIEW_SYNTHESIS_2026-06-21.md`). Method: one
dedicated reader per response → synthesis (draft-1) → **three adversarial critics (implementation /
risk-correctness / scope) → this v2.** The self-review changelog is Part XI.

---

## Part 0 — Executive summary & the scope decision

### The design the panel converged on (5/5, unusually unanimous)
> **Replace PM → RM → Trader (propose → gate → execute) with a deterministic, weekly, atomic
> *Target-Portfolio* engine.** Strategies become **pure functions emitting desired exposures** (no
> sizing, no capital, no broker). One **Constructor** allocates risk across strategies under a single
> budget → one **target book**; a **Risk layer** gates the *target*; an **Execution Planner** diffs
> target-vs-reality into the **minimal netted order set**; **Broker Adapters** execute. Every run is
> immutable, reproducible, idempotently re-drivable, fully reconstructable, and **every failure path
> degrades to "hold current positions and alert."**

The reframe (ChatGPT): **from "a collection of successful scripts" to "a controlled investment
process"** — every live dollar traces *signal → intent → risk budget → target → factor exposure →
risk approval → netted order → fill → reconciled position → attribution*, and *"if any live order
cannot be traced through that chain, it should not exist."*

### The scope decision (this is the most important section — added in v2)
The full target-portfolio brain is the **correct end-state.** But three independent critics
converged on a hard truth the draft missed: **today this is a one-strategy problem.** The live book
is **trend (50%) + cash** — one real edge. Carry/xsmom/VRP are *paper*, and the "second engine"
(t = 2.29) is **statistically unproven until GL-0 runs** (and "more likely than not" fails a strict
family-wise bar → carry-only or nothing). **Building a 3–4-sleeve coordination engine — ERC,
covariance shrinkage, stress matrices, divergent/convergent taxonomy, netting — for strategies that
may not exist is the classic solo-operator failure: a perpetual rewrite that trades nothing new and
gets abandoned half-built (strictly *less* safe than not starting).**

Therefore the program is **staged into a shippable safety floor, a hard abort gate, and two
demand-pulled releases:**

- **R0 — Minimum Viable Safety (MVS). Shippable AND terminal.** Decouple the daemon; build the
  consolidated cross-venue book-state + reconciliation-before-trade + single kill-switch/flatten;
  and **route the *existing* live sleeves through a whole-book risk gate** (the minimal fix to the
  real as-built defect: live sleeves bypass the RiskManager). This captures **~80% of the safety**,
  protects the one live edge, and **is the hard no-go gate before any IBKR dollar.** *The operator is
  explicitly permitted to stop here indefinitely.*
- **🚦 GL-0 ABORT GATE.** Run the null zoo (free, ~1 week, in parallel with R0). **Fails / carry-only
  / neither survives → STOP at MVS.** Do not build the Constructor stack. (Optionally run carry-only
  on IBKR behind the MVS gate.) **Passes (basket real) → continue.**
- **R1 — New real trading on IBKR (gated on GL-0).** IBKR read-only adapter + connection supervisor
  → carry (and xsmom iff GL-0 clears it) live at fractional `live_fraction`, **behind the MVS
  measurement layer + whole-book gate + per-venue margin reserve**, with the full hardening
  (deterministic sanity invariants, absolute notional cap, external dead-man, fault-injection drill,
  pre-committed rollback triggers) **live BEFORE capital.** This delivers "closer than ever to real
  trading" *without* first rewriting the working Alpaca book.
- **R2 — The unified Constructor (demand-pulled, only once ≥2 live strategies genuinely contend).**
  Refactor strategies to emit exposures; build the Constructor to parity (execution-cutover *then*
  joint-sizing); retire legacy. **The covariance stack (ERC / Ledoit-Wolf / stress matrix) is
  data-gated: DO NOT build until ≥2 strategies have ≥6 months of *joint live* return streams** —
  you cannot shrink a covariance you have never observed out of sample. **Inverse-vol is the
  terminal sizing policy until then.**

This **converges to the identical end-state** as the panel's design — it just lets the clever brain
be *pulled into existence by a second proven strategy* rather than *pushed on a 10-phase schedule
while trading nothing new.* A **wall-clock stop-loss** guards the whole effort (Part VIII).

### The first three builds (the MVS core; unanimous across panel + critics)
1. **Decouple the trading daemon from FastAPI** (web read-only; daemon = the brain). *Budget this as
   the single largest, riskiest engineering item — not a warm-up.*
2. **The consolidated cross-venue book-state + reconciliation + kill-switch, in shadow** (you can't
   control — or safely add a second venue to — what you can't measure or reconcile).
3. **Route the existing live sleeves through a whole-book risk gate** (fixes the actual defect
   without the full strategies-emit-exposures rewrite).

### The governing principles (the "don'ts" that keep us safe)
1. No strategy places orders; no sleeve owns capital; no venue has a separate *ungoverned* book.
2. **Reconciliation-first, then sizing.** The 2 a.m. blow-up is a DB-vs-broker mismatch, not a
   covariance estimate. *(Promote reconciliation above sizing in build priority.)*
3. **Measure before you control.** *(And keep the consolidated measurement view forever.)*
4. **Robustness is architectural** — stable base allocation + dumb monotonic governors; never let an
   optimizer handle crises. **No Markowitz / max-Sharpe / HRP / Kelly.**
5. **Broker is reality; Postgres is memory.** Fail-closed on mismatch; "hold and alert" is the
   default of every failure path.
6. Determinism + idempotency + auditability are non-negotiable in the live path.
7. LLMs around the core, never inside it.
8. **Boring over clever. Evolution, not revolution. Build the coordination layer only when there is
   something to coordinate.**

---

## Part I — Target architecture

### The pipeline
```
                          ┌─────────────── monitor loop (continuous, safety only) ───────────────┐
                          │  heartbeat · reconciliation · kill-switch · staleness · DD de-gross   │
                          │  (can de-gross or HALT; CANNOT open new positions)                    │
                          └──────────────────────────────────────────────────────────────────────┘

  Strategies          Book-State Assembler        Portfolio Constructor      Risk / Constraint
 (pure functions) ──▶  (reconcile BOTH venues  ──▶ (risk-budget across   ──▶  layer (hard factor/
  emit desired          first; assemble one         strategies; scale to       margin/gross/NOTIONAL
  ExposureVectors)      immutable BookState)        book vol target;           caps on the TARGET;
                                                    de-gross governors,        clip / de-gross;
                                                    MIN-composed)              fail-closed)
                                                                                      │
                                                                                      ▼
        immutable Decision Ledger  ◀── Reconciler ◀── Broker Adapters ◀── Execution Planner
        (book_runs, governor_log,    (DB := broker    (Alpaca / IBKR;      (diff target vs current;
         orders, fills, snapshots)    reality;         idempotent orders;   net exact instruments;
                                       per-ORDER        per-order lifecycle  no-trade bands; min size;
                                       persistence)     persisted)          reduce-only first)
```
**Governing rule:** one weekly rebalance = **one run_id, one frozen input snapshot, one target book,
one execution plan.** Replaces today's staggered sleeves (09:45/09:50/09:55 — "the silo problem").
*(NB: this whole pipeline is R2. In R0/R1 the existing sleeves keep their own sizing and are
governed by the Book-State + risk gate + kill-switch, not yet the unified Constructor.)*

### Components (responsibilities + must-not-own)
| Component | Owns | Must NOT own | First built in |
|---|---|---|---|
| **Book-State Assembler** | reconcile + assemble one immutable cross-venue `BookState` | alpha, sizing | **R0** |
| **Reconciler** | broker-reality vs ledger; DB := reality after fills; per-order persistence | alpha | **R0** |
| **Risk / Constraint layer** | whole-book hard limits (factor/gross/net/margin/**absolute notional**) on the proposed book | alpha | **R0** (gate on existing sleeves) |
| **Broker Adapter** | venue mechanics (symbols, contracts, roll, margin, fills, idempotency, connection) | strategy/risk policy | **R0** (Alpaca) / **R1** (IBKR) |
| **Monitor loop** | continuous safety: heartbeat, kill-switch, staleness, intra-week DD/corr de-gross, dead-man | opening new positions — **ever** | **R0** |
| **Strategy** | signal logic; a desired `ExposureVector` (internally vol-balanced) | capital, sizing-to-account, broker calls, other strategies | **R2** |
| **Portfolio Constructor** | risk allocation across strategies → one target book + governors | per-instrument signals, per-trade approval | **R2** |
| **Execution Planner** | target−current delta; netting; no-trade bands; minimal ordered set | portfolio optimization | **R2** (a thin diff-planner exists in R1 for the carry sleeve) |
| **Dashboard/API** | read-only book/risk views; operator commands via durable rows | the trading loop; direct broker orders | **R0** |

### Control model — scheduled decision, atomic pipeline, continuous monitor
- **Decision cadence = weekly, atomic** (one snapshot, one run). **No intraday *alpha* decisions.**
- **Supervision cadence = continuous** — the monitor loop does *only* safety (reconciliation,
  kill-switch, staleness, a DD/correlation de-gross that can shrink or halt but **cannot open
  positions**). *"Decision cadence ≠ supervision cadence; the monitor reacts to market state for
  safety, never for alpha."*
- **Concurrency contract (added in v2):** a single advisory lock serializes the weekly decision run
  vs the monitor loop. The kill-state is **re-read from authoritative Postgres atomically before
  every adapter call** (not from the Redis mirror — see §II). If the monitor escalates kill-state
  mid-run, the next adapter call observes it and the run transitions to `HALTED`/`CANCEL_ONLY`.

### Topology
- **`mtrader-api`** — FastAPI + React; **read-only** Postgres views; operator commands via durable
  rows. **No scheduled trading jobs, no broker orders.**
- **`mtrader-tradingd`** — the brain: scheduler, monitor loop, assembler, (R2) constructor, risk
  layer, planner, adapters, reconciler. **The only process that touches brokers.** Construct/risk/
  execute are **modules over one in-memory immutable snapshot**, not microservices.
- **`mtrader-watchdog`** — a tiny external dead-man (see §V); **built in R0**, not deferred.
- **Tech:** **Postgres = the only durable source of truth.** **Redis = ephemeral fast-reads only**
  (UI cache, heartbeat) — **never on the order-placement safety path.** No Kafka/K8s/Celery/event-bus.
- **Production runtime (added in v2):** pin a **Linux host/VM** for the daemon (IB Gateway + a
  long-running supervised daemon + dead-man timer are first-class on Linux; the Windows dev box is
  not the production target). Process supervision = **systemd units** (daemon, api) + a **systemd
  timer** (watchdog). Restart/auto-start/crash-recovery defined per unit.

---

## Part II — Data model, book state & the load-bearing specs

### Source of truth: "the broker is reality, Postgres is memory"
Brokers are truth for live positions/cash/margin/orders. Postgres is the immutable memory of what we
intended/sent/observed/reconciled. `BookState` is a **derived deterministic view rebuilt every run**
from Postgres + broker snapshots — never a hidden in-memory dict that becomes a second reality.

**Reconciler policy change (made explicit in v2):** the *current* `startup_reconciler` "never
modifies broker, never auto-fixes — surfaces and waits." The new policy is **auto-resolve toward
broker reality within tolerance, fail-closed beyond.** This is a deliberate behavior change; it is
called out here so it does not surprise the operator, and the old surface-only logic is migrated, not
silently replaced.

### Three states: Physical book / Virtual strategy ownership / Proposed book
One physical book; virtual sub-ledgers via `strategy_tags` weights (no broker sub-books). **The risk
gate runs on the *proposed* book** (current physical + open orders + this cycle's intents), not on
individual orders.

### Core objects (frozen dataclasses; money = Decimal)
`BookState{run_id, as_of, data_date, positions:[CanonicalPosition], wants:{sid:ExposureVector},
accounts:{venue:AccountState}, market:MarketSnapshot, exposures, factor_exposures, inputs_hash,
data_freshness, reconciliation_status, kill_state}` ·
`CanonicalPosition{instrument_id (canonical), venue, broker_symbol, asset_class, quantity, price,
multiplier, currency, notional_base, delta_equivalent_notional, strategy_tags, factor_loadings}` ·
`ExposureVector{strategy_id, data_date, targets:[InstrumentTarget], meta{venue, asset_class,
risk_class, live_fraction, confidence}, data_hash, code_version, config_version}` ·
`InstrumentTarget{instrument_id, direction, target_weight|risk_fraction, exposure_contribution
(general — carries a scalar notional now, option greeks later)}` ·
`AccountState{venue, nav, cash, settled_cash, buying_power, margin_used?, margin_available?,
maintenance_margin?}` *(PER VENUE — never netted across brokers)* ·
`FactorExposureVector{equity_beta_usd, rates_dv01_usd, usd_fx_delta, commodity_delta_usd,
vol_delta_usd}` · `TargetBook{run_id, per_strategy_risk_weight, target_positions, book_gross,
book_net, factor_exposures, diagnostics}`.

### Persistence: hybrid (materialized current state + immutable decision ledger) — NOT pure event-sourcing
- Materialized `positions` (reconciled per venue) = working truth; immutable append-only `book_runs`
  (one row per weekly decision: `run_id`, hashes, `wants`, `TargetBook`, diagnostics, run-state,
  mode, policy/code version, operator_approval) = audit/replay spine.
- Tables: `instrument` + `venue_instrument` (contract master) · `strategy_registry` (incl.
  `risk_class`, `live_fraction`, caps, version) · `book_snapshots` · `orders` (existing ledger +
  `run_id` + deterministic `client_order_id` + **per-order lifecycle**) · `fill` ·
  `risk_check_result` · `governor_log` · `reconciliation_event` · `control_flags` (kill-switch,
  capital stage, **the signed GL-0/GL-1 policy artifacts** — see §III) · `kill_switch_event` ·
  **`market_snapshot`** (see below).
- **Schema migrations (added in v2):** adopt **Alembic** (the repo has only ad-hoc scripts today) as
  an R0 prerequisite; the `trades`→`orders`/`fill` backfill is a tested, reversible migration.

### The three load-bearing specs the draft hand-waved (now pinned)

**1. The market-snapshot layer (a named R0 component, not assumed).** `BookState.market` and the
immutable `inputs_hash` require capturing **prices/vols/betas/factor-loadings as of `data_date`** to
a `market_snapshot` table, content-hashed. **Replay, golden-date CI, and determinism are impossible
without it** — so it is built in R0 (Phase 2/3), not assumed from "Phase 4 onward."

**2. The factor map (source / seed / maintenance / fail-closed).** Every instrument carries
`factor_loadings`. Source = a **hand-curated YAML of priors** (SPY equity_beta 1.0; ES equity_beta
1.0; ZN rates_dv01 from contract spec; SGOV cash-equiv; VX vol_short −1.0) **+ a quarterly
regression-refresh job with a human diff-review gate** (slow priors, never re-fit weekly).
**Hard rule: an instrument with no complete factor-map entry is `LIQUIDATION_ONLY` — it cannot be
sized** (fail-closed). A test asserts every registered/held instrument has a complete map.

**3. Reconciliation tolerance + resolution algorithm (worked).** Pre-trade and post-fill: normalize
each venue's truth → canonical; diff vs DB. **Tolerances:** equities per-instrument qty delta **= 0**
(after excluding known-pending `client_order_id`s); futures contracts delta **= 0**; cash within
**max($X, Y bps of NAV)**; margin fields present (IBKR) or **no futures orders**. **Resolution:**
within tolerance → overwrite DB position/cash toward broker, write a `reconciliation_event` row,
proceed; **material/unexplained mismatch → FAIL-CLOSED: do not trade this cycle, hold current,
alert.** *Worked partial-fill example:* a run places reduce-only SPY −40 then a partial fills 25 →
next reconciliation sees broker SPY = current −25, DB pending −15; the −15 stays a logged open order
reconciled into the next run; the run never blocks indefinitely. **Margin reconciled per venue;
exposures aggregated across venues.** Asymmetry justifying fail-closed: a skipped weekly rebalance =
one week of stale exposure; trading on a phantom position = unbounded.

### Determinism / idempotency (with the v2 fixes)
- `run_id = hash(schedule_key + portfolio_id + policy_version + rebalance_date + **trigger_kind**)`
  — the `trigger_kind` discriminator (scheduled vs manual-de-gross) prevents a mid-week manual run
  colliding with the scheduled run while preserving retry-idempotency for the *same* trigger.
- `client_order_id = hash(run_id, instrument, venue, side, qty, order_type, seq, plan_hash)`; persist
  **before** sending; brokers dedup → re-submit after crash is a no-op.
- **Per-order lifecycle persisted** (`planned → sent → acked → filled/partial → reconciled`) so a
  crash during `EXECUTING` re-drives only unsent/unconfirmed orders (per-order, not per-phase).
- Run state machine `PENDING → SNAPSHOTTED → CONSTRUCTED → RISK_CHECKED → PLANNED → EXECUTING →
  RECONCILED → COMPLETE` (+ `HALTED`/`FAILED`/`OPERATOR_HOLD`); resume from last persisted state.

---

## Part III — Sizing & risk (the Constructor + governors) — substantially revised in v2

**Principle (both panels):** robustness is *architectural* — diversification from a **stable base
allocation**, tail protection from **separate, dumb, monotonic de-gross governors.** Never ask an
optimizer to handle a crisis. **No Markowitz / max-Sharpe / HRP / Kelly.**

### The single home for "tail correlation reduces gross" (fixes the v1 quadruple-count)
v1 charged the same co-crash risk in four places (stress covariance, the `g_corr` governor, the
convergent haircut, factor caps). v2 assigns **one home each:**
- **Base sizing = a stressed view.** When the covariance stack exists (R2, data-gated), the book is
  sized to the **worse of normal and a stressed covariance.** Until then (R0/R1) base sizing is
  **inverse-vol** and the "stress" is expressed only as conservative gross + factor caps.
- **The `g_corr` governor fires ONLY on realized-vs-assumed SURPRISE** — it de-grosses only when
  *realized* tail correlation exceeds the value already assumed in the base/stress view, so it never
  re-charges risk the base already removed.
- **The convergent haircut is applied ONCE, before the budget table** (an explicit multiplier on
  convergent-sleeve risk weight in policy), not on top of it.
- **Factor caps bound net exposure** (a different axis — beta/DV01/vol), not "correlation" again.

### The governor measures TAIL correlation, not the calm-market average (fixes the v1 contradiction)
v1 said "tail correlation drives gross" then implemented a **63-day realized *average*** corr — the
exact calm-market number the panel called "dangerously misleading," firing *after* the tail event.
v2:
- The `g_corr` trigger is **stress-conditional / exceedance correlation** among the live sleeves
  (correlation conditional on the worst-decile book or equity days), recomputed on a rolling window —
  **not** the unconditional average. For a **3–4-sleeve book, pairwise exceedance correlation is the
  primary trigger** (the absorption ratio / first-PC share is *not* used at this scale — it is an
  institutional many-asset metric with no statistical content for 3–4 streams; reserve it for later).
- **GL-1's *measured* exceedance-correlation matrix is the *input*** to the stress view (with a
  conservative floor), so the number that sizes the book is the **measured tail correlation**, not a
  literary assertion sitting in a dashboard.

### The sizing stack (strict order)
- **L0 — within-strategy sizing stays in the strategy** (inverse-vol legs → an internally balanced
  `ExposureVector`).
- **L1 — across strategies: plain INVERSE-VOL. This is the terminal policy until the covariance
  stack is data-gated in (R2).** Inverse-vol = ERC under equal-correlation, zero matrix inversion,
  maximally robust. `SizingPolicy` is a versioned, swappable interface.
- **L2/L3 — ERC + Ledoit-Wolf shrinkage + book-vol-target on the *shrunk/stressed* covariance.**
  **DO NOT BUILD until ≥2 strategies have ≥6 months of joint *live* return streams** (you cannot
  shrink a covariance you've never observed OOS; backtest returns reintroduce the in-sample
  correlation the panel distrusts). When built: shrink hard (toward diagonal collapses ERC back to
  inverse-vol); size to **max(normal-vol, stressed-vol)** where the **stress matrix is a literal
  numbered table in risk-policy** — per-sleeve **stressed vol multipliers (2–3×)** *and* the full
  pairwise **stressed-corr block** (convergent cluster → 0.75–0.90; **trend's cross-corr to the
  convergent cluster is negative/zero**, reflecting its divergent nature). Not a vague "0.75–0.90."
- **L4 — de-gross governors (each monotonic, bounded, fail-closed; composed by MINIMUM; each
  computed against `base_gross` — never the running product, asserted in tests).**

  | Governor | Trigger | Action |
  |---|---|---|
  | `g_corr` (tail) | **stress-conditional pairwise corr exceeds the assumed/stress value (surprise)** | cut 25% on first surprise; 50% with drawdown; round, un-optimized; require DD/vol/VIX confirmation |
  | `g_dd` | book DD from HWM | −8/−10% → 0.75; −12/−15% → 0.50; −16/−20% → flat/halt; asymmetric hysteresis (restore one rung after ~20d no new low; ladder state is book-level, a sleeve re-entry does not reset it) |
  | `g_vix` | existing VIX>VIX3M backwardation | de-gross (generalized from trend to the whole book) — **un-fitted, monotonic** |
  | `g_margin` | per-venue maintenance margin / NLV > ceiling (~20–25%) | cap that venue's futures gross clear of margin |
  | `g_stale` | stale data / recon wobble / missing input / **insufficient live history** | de-gross or hold; never trade up on bad info |

- **L5 — hard caps (Risk layer): factor caps + per-venue margin + an ABSOLUTE per-instrument and
  per-book notional cap (% of NAV).** Run the **lower of {vol-target gross, margin-cap gross}.** The
  absolute notional cap ships in **risk-policy v1 (R0)** — a dumb backstop active from the first live
  phase, independent of the fancier invariants (catches a constructor bug / bad price snapshot that a
  factor cap might miss).

### Divergent vs convergent risk_class — operationalized (not asserted)
- Tag `risk_class ∈ {divergent, convergent}`: **trend = divergent** (long-vol, hedges); **carry,
  xsmom, VRP = convergent** (co-crash). The driver of gross is **measured tail correlation** (above),
  not the full-sample average.
- **Convergent haircut** = an explicit multiplier (in policy) applied to convergent-sleeve risk
  weights **before** the budget table.
- **Do NOT generalize the fitted VRP crash-gate to all convergent sleeves** (v1 error — it spreads a
  2-event overfit across the cluster that co-crashes). Convergent crash protection = the **un-fitted,
  monotonic** `g_vix` + `g_dd` + `g_corr` governors; the VRP-specific gate stays confined to VRP and
  is re-validated leave-one-crisis-out per GL-1.

### Book volatility target & drawdown budget
Anchor on drawdown; we have **zero live multi-strategy track** → launch **conservative: ~6% book
vol, steady-state ~8–10%, hard max-DD budget −15% to −20%** (−20% = the kill line). **Explicitly NOT
Kelly.**

### Cold-start rule (added in v2 — paper sleeves have no joint history)
A sleeve below a minimum *live* history uses **inverse-vol only**; the `g_corr` / tail-correlation
machinery is **disabled (returns 1.0, fail-safe) below the minimum history and must not silently
lever up.** The shadow book computes what a paper sleeve *would* allocate via inverse-vol only — it
never fabricates a cross-sleeve covariance from backtest returns.

### Where these constraints are ENFORCED across the releases (added in v2.1 — the re-order's one gap)
The Constructor does not exist until R2, but the GL-0 hard-zero and the per-venue margin reserve must
bind in **R1** (carry live on IBKR). **The enforcement point is the governed sizing path, which
exists in every release:** in **R0/R1** it is the **whole-book risk gate + the per-sleeve sizing
path** (the same gate that fixes the "sleeves bypass the RM" defect — §VIII 0.5); in **R2** the
unified Constructor inherits the identical constraints. So everywhere below, read "the Constructor
enforces X" as "the governed sizing path enforces X — the R0 risk gate in R0/R1, the Constructor in R2."

### The Go-Live verdicts are ENFORCED ARTIFACTS, not prose (fixes the v1 "comments" gap)
- **GL-0 verdict** (basket-real / carry-only / none) is a **signed, versioned `control_flags`
  artifact the governed sizing path reads and *hard-zeros* disallowed sleeves regardless of
  `live_fraction`; fail-closed (cannot size futures) if the artifact is missing.** (Enforced by the
  R0 whole-book gate in R1; inherited by the Constructor in R2.)
- **Demotion stops** (from Go-Live, verbatim) are monitored conditions that **auto-set
  `live_fraction = 0`** (a kill; re-earn the ramp): live-vs-paper-shadow daily corr < 0.6; slippage
  > 50% of edge for 2 months; tracking error > 40% of expected sleeve vol; sleeve DD > 1.5×
  backtest; any margin breach; wrong contract traded.
- **VRP** enters at `live_fraction = 0`, ≤10% hard cap, never equal-risk, **only if reformulated**
  (VRP ⊕ conditional-long-vol, judged on CVaR-per-bleed) — enforced by the `risk_class` machinery.
- **Defensive sleeve is a GATE, not "if needed"** (fixes the v1 risk-ordering inversion): **funding
  any convergent futures sleeve in R1 requires the defensive bond/gold/FX trend sleeve live, OR GL-1
  proving it unnecessary.** You do not add short-crisis carry to the live book before the crisis
  hedge that justifies it.

### Per-venue margin reserve (added in v2 — a real margin-call guard)
Alpaca SGOV cash **cannot** fund an IBKR margin shortfall (different brokers, un-transferable cash).
The cross-venue cash view **must never count Alpaca cash toward IBKR buying power.** A **hard
constraint in the governed sizing path (the R0 risk gate in R1; the Constructor in R2) reserves a
minimum cash/margin buffer *at IBKR*** independent of the Alpaca cash sleeve; the cash sleeve runs
last and is margin-aware *per venue.* (This binds in R1, before the Constructor exists.)

### Fractional paper-ramp (capital eligibility = data)
`live_fraction ∈ {0, 0.25, 0.50, 1.0}` in `strategy_registry`, multiplied into risk weight. **Ramp
metrics are concrete (v2):** TE = annualized stdev of (shadow_target_weight − realized_weight) over
trailing 8 weeks < threshold; "clean week" = run reached `COMPLETE`, zero fail-closed events, recon
delta < tolerance. Steps gated by **N clean weeks + slippage < 30% of edge + ≥1 vol spike survived**;
**the step to real money is human-confirmed** (the one irreversible move); every change is an audited
row. Demotion stops (above) auto-revert to 0.

---

## Part IV — Netting, factor exposure, conflict resolution

- **Net exact same canonical instrument (same venue) — always** (collapses two strategies wanting
  SPY into one net target traded once; "for free" in the diff planner). **Do NOT mechanically net
  proxies** (SPY-on-Alpaca vs ES-on-IBKR) — close economic but not operational substitutes; auto-
  netting silently breaks research/live parity. Manage overlap via the **factor view + factor caps.**
  `equivalence_groups` are a *later*, explicitly-configured feature (dropped from near-term scope).
- **Factor-exposure view, netted across venues** (`equity_beta, rates_dv01, usd, commodity, vol`):
  computed for current and target book, checked twice (current before intents; target before
  execution); breach → `LIQUIDATION_ONLY` for that factor. This catches stacked SPY+ES beta. **Built
  read-only in R0** (it's the core of the measurement layer / GL-2 risk surface).
- **Conflict resolution = net at book level; never suppress a strategy.** Same instrument opposing
  views → hold the net, trade the diff. Correlated-but-different → both exist, limit the net via
  factor caps. The only thing suppressed is aggregate risk breaching a limit → **proportional
  de-gross of contributors, never picking a winner.** *(R2; in R0/R1 with ≤2 sleeves this is trivial.)*

---

## Part V — Cross-venue execution, safety, determinism

### Broker abstraction
- **`BrokerAdapter` Protocol:** `venue`, `health()`, `get_account()`, `get_positions()`,
  `list_open_orders()`, `estimate_order()`, `place_order()` (idempotent), `cancel_order()`,
  `cancel_all()`, `flatten_all()` (idempotent), `normalize_instrument()`. Venue specifics
  (symbols, contracts, **roll calendar**, margin, fills, TIF, rate limits, **connection lifecycle**)
  live inside the adapter; the brain speaks canonical instruments + signed notionals + per-venue
  `AccountState`. **Roll is a venue concern + a book operation, not alpha** (central
  `FuturesContractService` presents a continuous instrument).
- **IBKR is operationally NOT Alpaca (added in v2):** `ib_insync` needs a **stateful TWS/IB-Gateway
  session** (disconnects, re-login, auth lifecycle). The IBKR adapter includes a **connection
  supervisor** (auto-reconnect with backoff, session health surfaced via `health()`); a stale/down
  session → `HALT_NEW_RISK`. Secrets policy covers two brokers.
- **Risk globally, capital locally:** positions/exposures aggregated across venues; `AccountState`
  per-venue constrains per-venue (no margin transfer between brokers — see the IBKR reserve in §III).

### Kill-switch + dead-man
- **State machine in `control_flags` (Postgres authoritative):** `NORMAL → HALT_NEW_RISK →
  CANCEL_ONLY → FLATTEN_NON_CORE → FLATTEN_ALL → MANUAL_LOCK`. **Adapters read the authoritative
  Postgres row on the order-placement path** (the Redis mirror is for non-blocking UI only — never
  the safety hot path; fixes the v1 staleness hole). Adapters refuse risk-increasing orders unless
  `NORMAL`.
- **Idempotent flatten:** re-snapshot; close only above a dust threshold;
  `reconcile_until_flat_or_timeout()`; flatten IBKR futures first (esp. short-vol VX), leveraged
  next, ETFs last. **The out-of-band flatten script operates purely on broker-reported positions**
  (closes what the broker says you hold) — it needs **no internal instrument mapping** and no
  Redis/Postgres/FastAPI.
- **Dead-man (conservative default):** the external watchdog sets **`HALT_NEW_RISK`** on a stale
  daemon heartbeat (not auto-flatten — a flaky watchdog auto-flattening is "liquidation-by-bug").
  **Full flatten is operator-confirmed**, *except* hard margin-breach rules where waiting is worse.
  **Built in R0.**

### Determinism, idempotency, auditability, fail-closed
- Decision atomic, execution **not** physically atomic across venues (don't pretend) — one run/
  snapshot/target/approval/plan; controlled partial-fill handling; de-risk-first sequencing.
- **Whole-book risk gate** = a full report (not first-failure), blocks on any BLOCK-severity, allows
  a trade that *reduces* a breached exposure. Plus **deterministic sanity invariants** (gross/factor/
  position/**absolute-notional** caps + "target not wildly different from last week without a logged
  cause") as a hard pre-execution gate → fail-closed hold.
- **Replay / parity:** `replay --run-id … --assert-same-target` over persisted snapshots; **golden-
  date replay CI** (R2). The **permanent shadow measurement view** (R0) is the cheap forever-monitor.
- **Every failure path degrades to "hold current positions, do not trade, alert."**

### Feared failure modes → guardrails (panel + critics)
| Failure | Guard | Phase |
|---|---|---|
| **DB ≠ broker, book trades anyway (#1 risk)** | reconciliation-before-trade, fail-closed; broker wins; post-trade recon | R0 |
| duplicate orders after crash | deterministic `client_order_id` + per-order persistence + match-before-send | R0 |
| IBKR added as ungoverned silo | no IBKR live until it feeds canonical `BookState` + is under the risk gate + kill-switch | R1 |
| **Alpaca cash can't cover IBKR margin → forced liquidation** | per-venue IBKR margin reserve; never count Alpaca cash toward IBKR BP | R1 |
| false diversification from a noisy covariance | inverse-vol terminal; covariance stack data-gated; measured tail-corr governor; no max-Sharpe | R0→R2 |
| governor sign bug *increases* exposure | `min`-composed, clamped ≤1.0, each vs `base_gross`; monotonicity asserted in tests | R0 |
| constructor/price bug sizes a 10× position | **absolute notional cap in risk-policy v1, live from first live phase** | R0 |
| brain trades during shadow | **adapters physically read-only (no `place_order` wired) until cutover**, asserted in a test | R0 |
| monitor vs run race | single advisory lock + atomic kill-state re-read before each adapter call | R0 |
| cash sweep consumes futures margin | cash sleeve residual, runs last, margin-aware per venue | R1 |
| LLM false confidence | LLMs out of the deterministic path (Part VII) | — |

---

## Part VI — Extensibility & future-proofing
- **Strategy contract (R2):** one pure function `desired_exposures(market, current_book) ->
  ExposureVector` + one declarative `strategy_registry` row. No bespoke rebalancer, no scheduler
  surgery. The old swing/PEAD/intraday ML path, if it returns, becomes **one strategy adapter** —
  never bypasses the brain.
- **Venue/asset contract:** add a venue = implement `BrokerAdapter` + register; the constructor
  doesn't change. Add an asset class = adapter normalizes to canonical + a factor map.
- **Options = the one honest exception:** `InstrumentTarget`/`CanonicalPosition` carry a general
  `exposure_contribution` (greeks slot into the factor view) **from day one** — cheap insurance.
  **No other options-specific scaffolding** until an options sleeve is actually proposed (anti-creep).
- **Research ↔ live parity is structural (R2):** same `desired_exposures` in BACKTEST/SHADOW/PAPER/
  LIVE — mode changes data, not logic. Golden-date replay CI.
- **The permanent shadow *measurement* view (R0) stays forever** (cheap). The shadow *constructor*
  is only worth maintaining once there is a constructor worth regression-testing (R2).

---

## Part VII — LLMs in the loop
**Principle:** LLMs sit **around** the deterministic core (explain, narrate, monitor, query,
flag-to-human), **never inside** it. Any live influence is bounded to **monotonic de-risking**,
fail-safe to "deterministic system runs unchanged," logged, human-confirmed until a long track
record earns autonomy.
- **OUT — non-negotiable: the deterministic sizing/execution path** (an LLM is non-deterministic →
  abolishes every safety guarantee; opens a prompt-injection surface in the money path; makes "why
  did the book do this?" unanswerable). **Never.**
- **IN, in build-order:** (1) **post-trade analyst + book commentary** (offline, read-only — highest
  value/lowest risk, build alongside the brain); (2) **morning brief** (read-only); (3) **operator
  copilot** (read-only; propose, never act); (4) **anomaly/sanity over the proposed book** (DEFER;
  can HOLD-for-human, never mutate; **its absence must not halt trading**; build deterministic sanity
  invariants first); (5) **event-risk → bounded de-gross** (LAST; ≤15–25%, monotonic, fail-safe;
  verdict: **not yet** — most dangerous, marginal value for a weekly book that already governs
  measurable risk-off). Guardrails: source whitelist, advisory/de-gross-only, never increases
  exposure or places orders, logged, unplug-able with zero book impact.
- **Build-now verdict:** *deterministic brain → LLM narration → (much later, maybe) bounded
  monitoring. Not the reverse.* The LLM layer is **R2+** and mostly a distraction from the brain.

---

## Part VIII — The implementation roadmap (v2: MVS → abort gate → two demand-pulled releases)

**Strangler migration — ETF-trend never goes dark.** Two tracks run in parallel: the **build track**
and the **research-gate track** (GL-0/GL-1 + the look-ahead audit), which decide *which strategies*
run and at what `live_fraction`.

### 🟢 R0 — Minimum Viable Safety (shippable AND terminal)
*Protects the one live edge; the hard no-go gate before any IBKR dollar; captures ~80% of safety.*
- **0.1 Freeze & decide.** ADR: "all live orders go through the governed book; no ungoverned
  silo; do NOT add IBKR as a silo." Risk-policy v1 (vol target, DD ladder, factor caps, margin
  ceilings, **absolute notional cap**). Adopt **Alembic**. Pin the **Linux production host** +
  systemd units. *Exit: ADR merged; policy in `control_flags`; migration tooling in place.*
- **0.2 Decouple the daemon from FastAPI** *(the single largest, riskiest item — budget it as
  such).* Inventory every shared singleton/queue crossing the boundary; `mtrader-api` (read-only) +
  `mtrader-tradingd`; **fallback flag to run both in one process** if the split destabilizes; basic
  alerting (heartbeat-dead, fail-closed, recon-mismatch) wired **now**. *Exit: a web restart doesn't
  stop trading and vice versa; no behavior change.*
- **0.3 Canonical instrument master + market-snapshot layer + Alpaca adapter (read-only).**
  `instrument`/`venue_instrument`; the **factor-map YAML** (+ fail-closed-if-missing); the
  **`market_snapshot`** capture/hash; Alpaca behind the `BrokerAdapter` Protocol with **`place_order`
  not wired** (physically read-only). *Exit: every live position is a `CanonicalPosition`; Alpaca
  reached only through the adapter; snapshots persisted.*
- **0.4 Consolidated book-state + reconciliation + kill-switch (shadow).** `BookState` assembler;
  `FactorExposureVector`; the **reconciliation-before-trade gate** (observe-only) with the specified
  tolerances/resolution; the **state-machine kill-switch + out-of-band flatten script + external
  dead-man watchdog**; the dashboard "what do I hold / net equity beta across venues / margin /
  reconciled?" view. **= Go-Live GL-2.** *Exit: the consolidated risk surface is verified correct on
  the live book; I can halt+flatten both venues from one out-of-band command.*
- **0.5 Route the existing live sleeves through a whole-book risk GATE.** Before the existing trend/
  cash path places orders, check the proposed consolidated book against the hard caps; fail-closed to
  "hold." **Fixes the real as-built defect (sleeves bypass the RM) without the strategies-emit-
  exposures rewrite.** *Exit: nothing trades that the whole-book gate hasn't seen.*
> **R0 acceptance / terminal state:** *"At any moment I can answer what I hold across venues, my net
> equity beta, my margin, whether I'm reconciled — and I can halt and flatten both venues from one
> out-of-band command; and no live order escapes the whole-book risk gate."* **The operator may stop
> here indefinitely.**

### 🚦 GL-0 ABORT GATE (run in parallel with R0; ~1 week; free)
Selection-aware null-strategy zoo (block-permuted signals → full pipeline incl. Track-B + basket →
Deflated Sharpe + Hansen SPA + empirical "null-books" p) **+ the carry/xsmom look-ahead audit** (PIT
vol, expiry, stitch; ties to the data-quality `factor_scorer` finding). Persist the verdict as the
**signed GL-0 artifact** (§III).
- **Fails / carry-only / neither survives → STOP at R0.** Do not build R1's multi-strategy pieces or
  R2's Constructor. (Optionally: carry-only on IBKR behind the R0 gate.)
- **Passes (basket real) → proceed to R1.**

### 🟡 R1 — New real trading on IBKR (gated on GL-0; delivers "closer to real trading")
*All risk controls live BEFORE capital. Uses a thin diff-planner + the existing-style sleeve sizing;
the unified Constructor is R2.*
- **1.1 IBKR adapter read-only + connection supervisor + `FuturesContractService`.** IBKR appears in
  the consolidated `BookState`; futures factor + per-venue margin computed; **no IBKR orders.**
- **1.2 Pre-capital hardening (live before any IBKR dollar).** Deterministic sanity invariants;
  absolute notional cap (already in v1); external dead-man; **fault-injection drill** (kill
  mid-execution → re-run → assert broker no-op via `client_order_id`; inject a phantom position →
  assert fail-closed-hold); **pre-committed rollback triggers** (any fail-closed event / any recon
  mismatch > tolerance → auto-HALT). **Per-venue IBKR margin reserve** enforced.
- **1.3 Carry (and xsmom iff GL-0 clears it) → IBKR paper → tiny-live at fractional `live_fraction`,
  behind the R0 gate.** Promotion ladder (Rung-0 IBKR-paper ≥2–3 months / ≥2 rolls / ≥1 vol spike /
  slippage < 30% edge / intended-vs-actual PnL corr > 0.9 → Rung-1 tiny-live 1–2 contracts ≤10% risk
  budget → Rung-2 scale by **process fidelity, not live t-stat**, ≤25%/quarter, never on a streak),
  with the **auto-demotion stops** (§III). **Precondition: defensive sleeve live OR GL-1 says
  unnecessary** before funding a convergent sleeve. Single kill-switch + dead-man span both venues.
> **R1 acceptance:** new real (tiny) futures trading on IBKR, every order traced to a governed book
> decision, both venues halt-able from one switch, convergent risk hedged or proven safe.

### 🔵 R2 — The unified Constructor (demand-pulled: only once ≥2 live strategies genuinely contend)
*Built when carry + a second live strategy actually need joint sizing/netting. Converges to the
panel's end-state.*
- **2.1 Strategies emit `ExposureVector`s; Constructor (inverse-vol + governors) to PARITY.** Refactor
  the live sleeves; the Constructor first reproduces today's behavior. **Parity gate = target-hash
  golden-date equality.**
- **2.2 Execution cutover (5a) — the execution-parity / canary gate.** Run the new planner/adapter/
  reconciler placing **tiny real orders** for ≥N cycles; validate fills, idempotency, reconciliation,
  partial-fill handling — *the cutover risk parity-by-target-hash never exercises.* Only then disable
  the direct path (feature-flagged for rollback with pre-committed triggers).
- **2.3 Joint sizing (5b).** The Constructor does cross-strategy inverse-vol sizing + governors + the
  whole-book gate on live money. **The covariance stack (ERC/LW/stress) remains DATA-GATED out** until
  ≥2 strategies have ≥6 months joint live returns.
- **2.4 Safety/observability polish + LLM narrative.** Replay/parity CI; observability metrics; LLM
  post-trade analyst + morning brief (read-only). Defer LLM monitoring.
- **2.5 Retire legacy PM/RM/Trader** after **≥8 clean live weeks incl. ≥1 de-gross event** on the
  brain (defined sunset). One code path to capital.

### Dependency & abort map
```
R0 (MVS: decouple → instrument/snapshot/adapter → book-state+recon+kill-switch → gate existing sleeves)
        │  (terminal state — may stop here forever)
        ├──────────────  GL-0 null zoo + look-ahead audit  ──────────────┐
        │                                                                │
   GL-0 FAILS / carry-only / none ──▶ STOP at R0 (optional carry-only IBKR behind the gate)
        │                                                                │
   GL-0 PASSES ──▶ R1 (IBKR read-only → pre-capital hardening → carry/xsmom tiny-live behind the gate)
                          │   (GL-1 tail test decides VRP in/out + defensive-sleeve gate)
                          ▼
                   [≥2 live strategies actually contend?] ──no──▶ stay at R1
                          │ yes
                          ▼
                   R2 (Constructor to parity → execution canary → joint sizing → covariance stack
                       ONLY after ≥2 strategies × ≥6mo joint live → retire legacy)
```
**The CRO gate:** *no futures capital, even paper-live, until IBKR is behind the adapter and the
consolidated BookState/risk gate sees Alpaca + IBKR as one book* (→ no R1 capital before R0).
**The wall-clock stop-loss:** *if R0 is not shipped and protecting the live book within a fixed
window, halt architecture work and revert to the existing path.* Solo rewrites die by drift.

---

## Part IX — What we deliberately do NOT build (anti-goals)
Kafka · Kubernetes · microservices/event-bus · max-Sharpe / mean-variance / **HRP** optimizer ·
**Kelly** · pure event-sourcing · **the ERC/Ledoit-Wolf/stress-covariance stack before ≥2 strategies
have ≥6mo joint live returns** · the absorption ratio for a 3–4-sleeve book · `equivalence_groups` /
proxy-substitution · proportional-de-gross conflict resolution before there are ≥2 contending sleeves
· golden-date replay CI before there's a Constructor · the permanent shadow *constructor* (keep the
shadow *measurement view*) · intraday execution algos · autonomous LLM trading · options-specific
risk logic before an options sleeve exists · auto-flatten-on-every-heartbeat. *Right-size for one
operator; stop short of institutional complexity.*

---

## Part X — Governing principles
(See Part 0 — the eight "don'ts." The two the critics elevated above all: **reconciliation-first**
and **build the coordination layer only when there is something to coordinate.**)

---

## Part XI — Self-review changelog (draft-1 → v2)
Three adversarial critics (implementation feasibility / quant-risk correctness / scope devil's-
advocate) reviewed draft-1. The material changes they forced:

**Scope (all three converged):**
- Added the **MVS shippable-and-terminal milestone**, the **GL-0 abort gate**, and **two
  demand-pulled releases** — instead of one 10-phase push that builds coordination for strategies
  that may not survive GL-0.
- **Data-gated the entire covariance stack** (ERC/LW/stress) behind "≥2 strategies × ≥6mo joint live
  returns"; **inverse-vol is terminal until then.** (You can't shrink a covariance you've never
  observed OOS.)
- **Re-ordered for time-to-real-trading:** IBKR carry tiny-live (R1) comes *before* the Alpaca-book
  rewrite-to-parity (R2), so new trading arrives months sooner; the Constructor is *pulled* by a real
  2-strategy need.
- Added a **wall-clock stop-loss** on the rewrite.

**Risk-correctness:**
- **Removed the crisis-correlation quadruple-count** — one home each (stress-cov base / `g_corr`
  surprise-only / convergent haircut once / factor caps on a different axis).
- **The tail governor now measures stress-conditional/exceedance correlation, not the 63d average**
  (fixed the self-contradiction); **GL-1's measured matrix feeds the stress view**; dropped the
  **absorption ratio** for a 3–4-sleeve book.
- **Go-Live verdicts are enforced artifacts** (signed GL-0 flag the Constructor reads + hard-zeros
  disallowed sleeves; **auto-demotion stops**; defensive sleeve a **gate**, not "if needed").
- Added the **per-venue IBKR margin reserve** (Alpaca cash can't fund IBKR margin).
- **Pulled the hardening before capital** (sanity invariants + absolute notional cap + dead-man +
  fault-injection + rollback triggers are R0/R1, not "Phase 8").
- **Do not generalize the fitted VRP gate** to all convergent sleeves; added the **cold-start rule**
  (governors fail-safe to 1.0 below min live history, never lever up).

**Feasibility / missing specs:**
- **Pinned the three load-bearing specs:** the **market-snapshot layer** (a named R0 component), the
  **factor map** (YAML priors + quarterly review + fail-closed-if-missing), and the **reconciliation
  tolerance + resolution algorithm** (worked partial-fill example; explicit policy change from the
  current "never auto-fix" reconciler).
- **Split the live cutover** into execution-canary (real tiny orders) + joint-sizing — the
  target-hash parity gate doesn't cover execution risk.
- Added **Alembic**, a **calendar/holiday/roll service**, the **IBKR connection supervisor** (IBKR ≠
  Alpaca operationally), **process management on Linux/systemd** (the Windows dev box is not prod),
  **per-order persistence** for crash-resume, the **monitor/run advisory lock**, the **Redis-off-the-
  safety-hot-path** fix, the **run_id trigger discriminator**, **read-only adapters during shadow**,
  and **alerting from R0**.

**v2 → v2.1 (verification gate).** A fourth reviewer verified v2 against the prior blockers: **8/8
genuinely closed in the body.** It found one defect the R1-before-R2 re-order introduced — the GL-0
hard-zero and the per-venue margin reserve were textually owned by the Constructor (R2) yet must bind
in R1. Fixed by clarifying that the **governed sizing path** enforces them (the R0 whole-book risk
gate in R0/R1; the Constructor inherits at R2). No new re-architecture; adopted.

> Cross-references: Go-Live synthesis `docs/reference/GOLIVE_REVIEW_SYNTHESIS_2026-06-21.md`; raw
> architecture responses `docs/reference/prompts/20260621_Comprehensive_portfolio_brain/`; the P2
> IBKR execution spec `docs/reference/P2_IBKR_EXECUTION_DESIGN.md` becomes the IBKR adapter in R1;
> the 2026-06-21 data-quality `factor_scorer` look-ahead finding folds into GL-0's audit.
