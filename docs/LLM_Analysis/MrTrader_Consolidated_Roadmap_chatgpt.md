# MrTrader Consolidated Review and Phased Roadmap

**Purpose:** Consolidated approach based on two independent analyses of the MrTrader automated trading system: one critique document and one system review / next-phase plan.

**Audience:** Min and downstream LLM reviewers.

**Goal:** Create a structured, practical roadmap that combines the strongest parts of both reviews while being honest about what should not be built yet.

---

## Executive Summary

Both reviews are directionally aligned: MrTrader has a strong engineering base, but the next phase should be about **proving the system is safe, measurable, and currently effective**, not adding more alpha features yet.

The shared concerns are:

1. The current live gate is too weak.
2. The swing model's most recent walk-forward fold is a major red flag.
3. Execution quality and TCA are not yet measured deeply enough.
4. RM is still a checklist, not a portfolio-level risk engine.
5. NIS/news intelligence is under-validated.
6. The agent architecture has some complexity risk, but a full rewrite is not the immediate priority.

The consolidated path is:

> **Do not rewrite the whole system now. Do not go live based on the current 4-week gate. Do not add new alpha modules yet. Build a proof and safety layer first, then use that evidence to decide whether swing, intraday, NIS, and the agent architecture deserve promotion, refactor, or retirement.**

---

# Strategic Direction

Move the project from:

```text
Paper trading is profitable → prepare for live.
```

to:

```text
Every strategy, model, gate, order type, news decision, and risk rule must prove its marginal value under realistic execution, current-regime validation, and restart/broker failure conditions.
```

That framing matters because paper P&L can look good while hiding:

- unrealistic fills
- queue-position assumptions
- slippage
- partial fills
- broker rejections
- restart/replay issues
- data-feed mismatch
- model decay
- regime change
- overfitting
- risk concentration

The next phase should be about **measurement, proof, and risk containment**.

---

# What Both Reviews Got Right

## 1. Swing Fold 3 is the most important model issue

The swing model's average walk-forward Sharpe passed, but the most recent fold was effectively flat/negative.

From the current system state:

| Fold | Period | Sharpe |
|---|---:|---:|
| Fold 1 | 2022-07 → 2023-10 | +0.88 |
| Fold 2 | 2023-10 → 2025-01 | +2.69 |
| Fold 3 | 2025-01 → 2026-04 | **−0.03** |

The most recent fold deserves heavy weight because it is closest to the current market environment.

Possible explanations:

1. The edge decayed.
2. Market regime changed.
3. The model was overfit to earlier periods.
4. The SP-100 training / broader Russell inference universe mismatch hurt performance.
5. The path_quality label is misaligned with actual trade P&L.
6. Gates blocked winners or allowed poor trades.
7. Realistic costs would erase already-thin edge.

**Current recommendation:** swing should remain **paper-only or shadow-only** until Fold 3 is explained or a challenger model beats it under realistic assumptions.

---

## 2. TCA / execution quality must be promoted

Slippage reporting should not be a later enhancement. It is core infrastructure.

The system already stores `Order.slippage_bps`, but that is not enough unless it is surfaced by:

- symbol
- strategy
- time of day
- entry vs exit
- fill delay
- missed fills
- partial fills
- order timeout
- paper-vs-realistic fill gap
- paper-vs-live gap once live probes begin

Without TCA, it is impossible to know whether poor results are caused by bad model signal or execution drag.

---

## 3. RM needs to become a portfolio risk engine

The current RM is a useful rule gate, but not yet a true portfolio risk model.

Current RM checks include:

- kill switch
- circuit breaker
- buying power
- daily loss limit
- max open positions
- position size
- sector concentration
- correlation gate
- duplicate check

These are necessary but not sufficient.

The RM should eventually answer:

```text
What is the total portfolio risk if this trade is approved?
How much market beta am I carrying?
Are swing and intraday actually diversified?
What happens if SPY gaps down 3% tomorrow?
What is max expected loss if all long names correlate to 1 under stress?
```

The next RM evolution should be:

```text
Can this trade pass a checklist?
```

to:

```text
Does this trade improve or fit the portfolio within risk budget?
```

---

## 4. NIS should remain a risk overlay, not alpha

The NIS/news layer is valuable, but it is not proven as an alpha engine.

Near-term role:

```text
NIS = structured event-risk classifier
```

Not:

```text
NIS = autonomous trade selector
```

NIS should continue to support:

- `block_entry`
- `size_down`
- `exit_review`
- macro event classification
- earnings / major event risk controls

But it should not become a primary buy/sell signal or ML feature until there is enough attribution history to prove its value.

---

## 5. Intraday is more promising, but execution-sensitive

The intraday model's most recent fold is much stronger than the swing model's latest fold.

| Fold | Period | Sharpe |
|---|---:|---:|
| Fold 1 | 2024-10 → 2025-04 | +1.73 |
| Fold 2 | 2025-04 → 2025-10 | +0.72 |
| Fold 3 | 2025-10 → 2026-04 | +2.97 |

This makes intraday the better candidate for eventual live probing.

However, intraday is much more sensitive to:

- bar timing
- fill latency
- quote source
- spread
- slippage
- partial fills
- missed fills
- order timeout behavior
- data-feed mismatch

So the correct next step is **tiny live probe mode**, not full intraday live trading.

---

# Where the Consolidated View Modifies the Prior Reviews

## 1. Do not do a full architecture rewrite now

One critique strongly argued that the async agent/queue architecture is over-engineered and that many bugs are symptoms of architectural complexity. That concern is valid.

The known bug list includes several architecture-related issues:

- heartbeat ticks counted as bars held
- in-memory daily flags resetting
- reconciler duplicates
- runaway scanner
- uvicorn shutdown hang
- timezone mismatch on daily flag restore

However, a full rewrite now would create risk:

1. It delays proof work.
2. It introduces new bugs.
3. It distracts from the main question: does the strategy have current edge under realistic execution?

The better path is:

```text
Keep PM/RM/Trader as logical roles.
Harden state, order lifecycle, idempotency, and persistence.
Move toward modular services gradually.
Do not add more in-memory queue complexity.
Defer full scheduler refactor until after the proof phase.
```

This is **architecture containment**, not architecture rewrite.

---

## 2. Do not make VaR/CVaR the first risk build

A portfolio risk engine is important, but for a $20k system, the first risk model should be practical and enforceable.

Start with:

- per-trade dollar risk
- stop-distance sizing
- gross exposure
- beta exposure
- strategy exposure
- sector exposure
- correlated cluster exposure
- scenario shocks
- drawdown response ladder
- order-rate hard limits

Then add VaR/CVaR.

Reason: near-term risk is probably not a subtle VaR modeling issue. It is more likely to come from:

- too many orders
- wrong position size
- unexpected correlation
- restart duplication
- live fill drift
- broker mismatch
- bad state recovery

So yes to VaR/CVaR, but after hard exposure and risk-budget controls.

---

## 3. Do not require 3 months of paper before all live testing

The current 4-week paper gate is too weak. But there is a difference between:

```text
live trading with strategy capital
```

and:

```text
tiny live probe mode
```

The system does not need to wait 3 months to send a one-share test order, as long as order lifecycle, kill switch, cancel-all, flatten, reconciliation, and hard limits are in place.

Policy:

```text
No real strategy capital until stronger gates pass.
Tiny live probe mode can begin earlier after order-safety controls pass.
```

This allows real execution learning without pretending the strategy is live-ready.

---

## 4. Do not move NIS directly into model features yet

One analysis correctly noted that mature quant systems use calibrated vendor sentiment as model features, not just overlays. That is a valid long-term direction.

But for this system, moving LLM scores directly into model training too early is risky because:

- NIS score history is too short.
- LLM model behavior may drift.
- Finnhub free-tier news may be incomplete or stale.
- Headline novelty is not yet solved.
- The model may learn unstable artifacts.

Near term:

```text
Keep NIS as policy overlay.
Log NIS like a model feature.
Measure attribution for 60–90 days.
Only then test NIS as a model feature.
```

---

## 5. Do not prioritize tail hedging yet

SPY puts or other hedges are conceptually valid, but they add:

- options data complexity
- options execution complexity
- spread/decay issues
- expiration handling
- hedge-sizing complexity
- cost drag

For a $20k account, first build:

- lower gross exposure
- lower per-trade risk
- halt logic
- flatten logic
- scenario shocks
- drawdown ladders

Tail hedging can come later.

---

# Consolidated Phased Roadmap

## Phase 0 — Freeze Expansion and Define Promotion Rules

### Purpose

Stop adding new alpha features until the current system proves what works.

### Freeze for now

Do not build:

- true scalper
- options flow
- new autonomous LLM trade agent
- full live deployment
- full architecture rewrite
- new strategy expansion

### Allowed work

Focus only on:

- measurement
- safety
- execution realism
- model validation
- risk budgeting
- controlled live probes

### Deliverable

Create `promotion_policy.md`.

It should define:

```text
What qualifies swing for live?
What qualifies intraday for live?
What qualifies NIS for alpha usage?
What qualifies an order type for live use?
What blocks live deployment regardless of P&L?
What incident resets the readiness clock?
```

---

## Phase 1 — Live-Readiness and Evidence Dashboard

### Purpose

Replace the current simple gate with a true readiness framework.

Current gate:

```text
4-week Sharpe > 0.5
max drawdown < 5%
```

This is too weak alone.

### Build

`live_readiness_report.py`

### Required sections

#### 1. System safety

- duplicate order incidents
- stale proposal incidents
- runaway scanner incidents
- restart replay incidents
- unresolved reconciliation events
- timezone/date mismatch events
- unhandled broker errors
- kill switch test result

#### 2. Execution quality

- arrival price
- decision price
- submitted price
- fill price
- slippage bps
- fill delay
- partial fills
- missed fills
- rejected orders
- entry vs exit slippage
- paper-vs-realistic fill adjustment

#### 3. Strategy attribution

- swing P&L
- intraday P&L
- NIS impact
- regime impact
- PM gate contribution
- RM rejection contribution
- expectancy by strategy

#### 4. Model health

- latest fold status
- bootstrap confidence intervals
- rank IC
- feature drift
- model age
- retrain success/failure
- challenger model status

#### 5. Risk

- gross exposure
- per-trade dollar risk
- beta exposure
- sector exposure
- correlated cluster exposure
- strategy exposure
- scenario shock loss
- drawdown response state

#### 6. Operations

- broker connectivity
- market data quality
- rate-limit events
- host uptime
- dashboard/API health
- DB-vs-broker P&L reconciliation

### Promotion rule

No strategy goes live because of combined paper P&L alone.

A strategy must pass:

```text
strategy-specific evidence
execution evidence
risk evidence
safety evidence
operational evidence
```

---

## Phase 2 — Execution State Machine and TCA

### Purpose

Make order handling durable and measurable before live money is exposed.

This is the highest-priority engineering phase because it helps every strategy.

### Build formal order lifecycle

```text
PROPOSED
APPROVED
SUBMITTED
ACCEPTED
PARTIALLY_FILLED
FILLED
CANCELED
EXPIRED
REJECTED
REPLACED
FAILED_RECONCILIATION
CLOSED
```

### Add idempotency keys

```text
proposal_id
approval_id
order_intent_id
broker_order_id
position_intent_id
exit_intent_id
```

### Add execution analytics

```text
arrival_price
approval_price
submitted_limit_price
fill_price
decision_to_order_ms
order_to_fill_ms
quoted_spread_bps
effective_spread_bps
realized_spread_bps
slippage_bps
missed_fill_outcome
partial_fill_ratio
cancel_reason
reject_reason
broker_status_raw
```

### Add hard operational limits

```text
max orders per symbol per day
max orders per strategy per hour
max total orders per 5 minutes
max live notional per symbol
max live notional per strategy
max duplicate proposal count
max unresolved broker orders
```

### Why this phase comes before model improvements

Because without execution quality measurement, you cannot tell if a model is bad or if execution is eating the edge.

---

## Phase 3 — Model Truth Test

### Purpose

Separate “model has edge” from “backtest/paper looked good.”

This phase is primarily about the swing model, but it also includes intraday validation.

---

## Phase 3A — Swing Model Postmortem

### Required diagnostics

```text
Fold 3 bootstrap Sharpe CI
Fold 3 rank IC
Fold 3 feature drift
Fold 3 sector breakdown
Fold 3 regime breakdown
Fold 3 gate contribution
Fold 3 entry-timing impact
Fold 3 exit-rule contribution
Fold 3 cost sensitivity
Fold 3 baseline comparison
Fold 3 universe mismatch test
```

### Critical universe test

The swing model trains on SP-100 but infers on a broader Russell 1000-filtered universe. That may be extrapolation.

Run:

```text
Test A: Train SP-100, infer SP-100 only
Test B: Train Russell 1000-compatible universe, infer Russell 1000-compatible universe
```

Interpretation:

- If Test A works and Test B fails, the issue may be universe expansion.
- If both fail in Fold 3, the issue is likely edge decay, regime change, or label weakness.

### Swing promotion decision

At the end of Phase 3A, produce one of four outcomes:

```text
PROMOTE_TO_TINY_LIVE_PROBE
KEEP_PAPER_ONLY
KEEP_SHADOW_ONLY
REBUILD_OR_RETIRE
```

Current expected outcome:

```text
KEEP_SHADOW_ONLY until proven otherwise
```

---

## Phase 3B — Challenger Label Framework

Do not immediately replace `path_quality`. Build challengers and compare.

### Challengers

```text
1. Current path_quality regression
2. Triple-barrier classifier
3. Binary target-before-stop classifier
4. Cross-sectional ranking / LambdaRank-style model
5. Expected-value-after-cost model
6. Meta-label confidence gate
```

### Required comparison controls

Use the same:

```text
folds
universe
cost assumptions
entry rules
exit rules
risk rules
paper/live fill assumptions
```

### Promotion standard

Do not promote based on raw Sharpe alone.

Promote based on:

- current-period performance
- rank stability
- drawdown
- cost-adjusted expectancy
- robustness across adjacent folds
- feature stability
- simplicity

---

## Phase 3C — Intraday Validation

The intraday model looks better than swing, but the bar 12 opening-session edge needs validation.

### Tests

```text
bar 9 to bar 15 sensitivity
bar 12 vs neighboring bars
liquidity bucket breakdown
symbol concentration
regime breakdown
spread/slippage-adjusted performance
paper fill vs simulated delayed live fill
entry one bar later sensitivity
missed-fill sensitivity
```

### Decision

If intraday still works after realistic execution assumptions, it becomes the first candidate for **tiny live probe mode**.

Not full live. Probe mode.

---

## Phase 4 — Portfolio Risk Engine v1

### Purpose

Upgrade RM from trade checklist to portfolio decision engine.

### Keep existing RM gates

Do not remove:

```text
kill switch
circuit breaker
buying power
daily loss
max positions
sector cap
correlation gate
duplicate check
spread/volume filters
macro blocks
```

These remain useful.

### Add risk budget layer

```text
account_equity
max_total_gross_exposure
max_strategy_exposure
max_symbol_exposure
max_sector_exposure
max_beta_adjusted_exposure
max_correlated_cluster_exposure
max_per_trade_dollar_risk
max_daily_loss
max_weekly_loss
max_monthly_drawdown
drawdown_state
```

### Position sizing

Move away from fixed 10–15% notional swing sizing as the default.

Use:

```text
shares = min(
    risk_budget_dollars / stop_distance_dollars,
    max_symbol_notional / price,
    max_strategy_notional / price,
    liquidity_cap,
    broker_buying_power_cap
)
```

### Drawdown ladder

Example:

```text
0% to -2% drawdown: normal paper sizing
-2% to -4%: 50% sizing
-4% to -6%: 25% sizing, no new swing
below -6%: no new trades, exits only, human review
```

### Initial live risk suggestion

For live probe / early live stages:

```text
initial live per-trade risk: 0.10% to 0.25%
later live per-trade risk: 0.25% to 0.50%
max initial gross exposure: 10% to 25%
max open live positions: 1 to 2
```

This is intentionally conservative. At first live, the goal is not to make money. The goal is to prove system behavior.

---

## Phase 5 — NIS Outcome Attribution and Policy Hardening

### Purpose

Determine whether news intelligence helps, hurts, or simply adds complexity.

### Keep NIS in these roles

```text
block_entry
size_down
exit_review
macro risk classification
event calendar risk
```

### Do not yet use NIS as

```text
primary buy signal
autonomous sell signal without price/risk confirmation
unvalidated model feature
```

### Build NIS attribution

For every NIS action, store:

```text
symbol
event_type
headline/source
first_seen_time
model_used
model_version
materiality
direction
confidence
already_priced_in
policy_action
price_move_before_signal
price_move_after_signal
blocked_winner_flag
blocked_loser_flag
size_down_helped_flag
exit_helped_flag
premature_exit_flag
```

### Promotion standard

NIS can be promoted only if it shows:

- positive contribution after costs
- lower drawdown
- fewer bad entries
- no unacceptable blocked-winner rate
- stable calibration
- acceptable latency
- model-version traceability

---

## Phase 6 — Regime Engine v1

### Purpose

Replace blunt binary market gates with a policy engine that adjusts risk by environment.

### Inputs

```text
SPY trend
QQQ trend
IWM trend
VIX level
VIX change
realized volatility
market breadth
sector dispersion
overnight gap
macro calendar
rates proxy
credit spread proxy
dollar/oil if useful
```

### Outputs

```text
regime_state
risk_budget_multiplier
strategy_allowlist
max_new_positions
max_gross_exposure
stop_tightening_factor
entry_block_flag
exit_review_priority
```

### Example regimes

```text
BULL_LOW_VOL
BULL_HIGH_VOL
CHOP_LOW_VOL
CHOP_HIGH_VOL
BEAR_HIGH_VOL
MACRO_EVENT
LIQUIDITY_STRESS
```

Start rule-based and auditable. Do not begin with a complex HMM or black-box regime classifier.

---

## Phase 7 — Controlled Live Probe Mode

### Purpose

Learn live broker behavior without taking meaningful strategy risk.

### Requirements before probe mode

```text
order lifecycle complete
idempotency keys complete
kill switch tested
cancel-all tested
flatten tested
broker reconciliation tested
hard order-rate limits active
execution report active
live_readiness_report active
```

### Probe mode rules

```text
one share or minimum notional
whitelisted large-cap symbols only
one live position at a time
no leverage
no shorting
no options
no overnight intraday positions
auto-flatten
manual review after each probe day
```

### What to measure

```text
paper expected fill vs live fill
latency
spread
rejection behavior
partial fill behavior
broker order status transitions
data feed quote mismatch
P&L reconciliation
DB-vs-broker position reconciliation
```

### Important distinction

This is not “go live.”

This is:

```text
instrumented live execution testing
```

---

# Revised View of Existing Backlog Phases

## Phase 75 — EOD Swing Position Review

Original idea: re-score held swing positions at 15:45 and exit if score is weak.

Consolidated view: **keep, but do not prioritize before Fold 3 postmortem and execution/risk readiness.**

Reason:

```text
If the swing model has no current edge, adding better exit logic does not solve the core issue.
```

Better version:

```text
Build this as a generic ExitReviewService that can consume:
- model score
- NIS risk
- regime
- price action
- P&L state
- holding period
```

---

## Phase 76 — Slippage Analysis

Consolidated view: **promote immediately.**

This is one of the highest-value next tasks.

---

## Phase 77 — Graceful SIGTERM / Queue Drain

Consolidated view: **keep, but combine with durable order lifecycle and idempotency.**

Do not treat it as shutdown polish only. Treat it as order-safety infrastructure.

---

## Phase 78 — Live Readiness Checklist Update

Consolidated view: **replace with Live Readiness Report v2.**

A checklist is helpful, but a deterministic report with pass/fail gates is better.

---

## Phase 79 — AUC Drift Alert

Consolidated view: **keep, but broaden to full model health.**

Include:

```text
AUC/rank drift
feature PSI
model age
last retrain status
Fold 3/current-period performance
calibration
score distribution drift
```

---

## Phase 80/81 — Intraday or Held-Position Re-Evaluation

Consolidated view: **defer until model truth and risk layers are built.**

Do not add more PM decision loops yet.

If built later, make it a separate service rather than adding more PM complexity.

---

## True Scalper Agent

Consolidated view: **do not build yet.**

It requires:

- L1/L2 or tick data
- different execution model
- different backtest harness
- lower latency
- smaller universe
- different risk controls

The current system should be stable in live/probe mode first.

---

# Final Consolidated Task Sequence

## Batch 1 — Measurement and Safety Foundation

1. `live_readiness_report.py`
2. `execution_quality_report.py`
3. order lifecycle state machine
4. idempotency keys
5. hard order-rate and exposure limits
6. kill/cancel/flatten verification
7. DB-vs-broker reconciliation report

**Why first:** gives the instrumentation needed to judge everything else.

---

## Batch 2 — Swing and Intraday Proof

1. swing Fold 3 postmortem
2. bootstrap confidence intervals
3. 5+ fold / purged validation
4. SP-100 vs Russell universe test
5. challenger labels:
   - triple-barrier
   - ranking
   - binary
   - EV-after-cost
6. intraday bar 9–15 sensitivity
7. intraday realistic fill-delay and missed-fill sensitivity

**Why second:** tells you whether the strategies deserve more investment.

---

## Batch 3 — Risk Engine v1

1. per-trade dollar risk sizing
2. gross exposure cap
3. strategy exposure cap
4. beta exposure
5. correlated cluster exposure
6. scenario shocks
7. drawdown response ladder
8. risk contribution added to every RM decision

**Why third:** the system cannot scale without portfolio-level approval logic.

---

## Batch 4 — NIS Attribution and Regime v1

1. NIS action attribution
2. NIS calibration report
3. NIS A/B shadow comparison
4. structured event taxonomy
5. regime engine v1
6. regime impact report

**Why fourth:** NIS and regime logic are valuable only if measurable and auditable.

---

## Batch 5 — Controlled Live Probe

1. one-share probe mode
2. whitelist symbols
3. paper-vs-live comparison
4. live fill report
5. broker rejection handling
6. manual review gate
7. gradual ramp policy

**Why fifth:** starts collecting live-market truth without pretending the strategy is ready for capital.

---

## Batch 6 — Architecture Simplification

1. extract Strategy interface
2. split `SwingStrategy` and `IntradayOpeningSessionStrategy`
3. extract `ExitReviewService`
4. extract `RiskBudgetEngine`
5. extract `ExecutionService`
6. remove in-memory state where possible
7. replace volatile queues only where evidence shows they are causing failure

**Why sixth:** architecture refactor should follow proof, not replace proof.

---

# Guidance for the Next LLM Reviewer

Use this framing:

```text
We are not asking whether MrTrader is impressive. It is. We are asking whether it is safe and proven enough to move toward live trading.

The next roadmap should prioritize:
1. measurement before expansion,
2. execution realism before live capital,
3. current-regime validation before model promotion,
4. portfolio risk before scaling,
5. NIS attribution before making news an alpha input,
6. targeted architecture hardening before a full rewrite.

Do not recommend adding a true scalper, options flow, vendor sentiment feeds, or more autonomous agents until the proof layer is complete.
```

---

# Final Position

The best consolidated path is:

```text
Do not go live yet.
Do not rewrite everything yet.
Do not add more alpha yet.
Build the proof layer.
Diagnose swing Fold 3.
Validate intraday under realistic fills.
Harden order lifecycle and idempotency.
Upgrade RM into a risk-budget engine.
Measure NIS before trusting it.
Then run tiny live probes.
```

This gives the cleanest path from a promising paper system to something that could eventually deserve real capital.

---

# Short Version for Task Planning

If converting immediately into implementation tasks, use this order:

1. Live Readiness Report v2
2. Execution Quality / TCA Report
3. Durable Order Lifecycle + Idempotency
4. Hard Order-Rate / Exposure Limits
5. Swing Fold 3 Postmortem
6. Model Validation Upgrade: CIs, 5+ folds, universe tests
7. Challenger Label Framework
8. Intraday Bar 9–15 / Realistic Fill Validation
9. Portfolio Risk Engine v1
10. NIS Outcome Attribution
11. Regime Engine v1
12. Tiny Live Probe Mode
13. Gradual Architecture Simplification

That is the roadmap I would use.
