# MrTrader Automated Trading Bot — Independent System Review and Next-Phase Plan

**Purpose:** This document is designed to be shared with other LLMs or reviewers so they can analyze this review, compare it with their own recommendations, and help combine the responses into a stronger implementation roadmap.

**Scope:** This is not a code-level review. It is a trading-system architecture, risk, model, execution, and live-readiness review based on the current MrTrader system state.

**Primary source:** User-provided `system_review_prompt.md` describing the MrTrader architecture, current paper-trading state, ML models, PM/RM/Trader workflow, data sources, known fixes, and backlog.

---

## Executive Verdict

MrTrader is **ahead of a normal hobby algo project operationally**, but it is **not live-ready yet**.

The architecture has good bones:

- Portfolio Manager / Risk Manager / Trader separation
- Paper trading already active
- Audit tables
- Replay tooling
- Health checks
- Startup reconciliation
- Meaningful test suite
- FastAPI / dashboard monitoring
- Paper-trading bug discovery already underway

But the system is still vulnerable to the classic automated-trading trap:

> **Paper performance plus many safety rules can look institutional while hiding execution, overfitting, regime, and live-broker failure risk.**

The most important conclusion is:

> **Do not optimize for more alpha yet. Optimize for proving that the edge survives realistic execution, current market regimes, restarts, broker behavior, and failure conditions.**

The current live gate — **4-week Sharpe > 0.5 and max drawdown < 5%** — is too weak by itself. That gate mostly answers:

> “Did recent paper P&L look acceptable?”

The better live-readiness question is:

> “Can the system survive bad data, stale signals, missed fills, rejected orders, partial fills, restarts, duplicate proposal risk, timezone bugs, current-regime model decay, and live broker behavior?”

---

## Biggest Red Flag

The **swing model should not be treated as production-ready** just because the average walk-forward Sharpe passed.

The current swing model shows:

| Fold | Period | Sharpe |
|---|---:|---:|
| Fold 1 | 2022-07 → 2023-10 | +0.88 |
| Fold 2 | 2023-10 → 2025-01 | +2.69 |
| Fold 3 | 2025-01 → 2026-04 | **−0.03** |

The average Sharpe looks good, but the most recent fold is effectively flat/negative. In trading, the most recent out-of-sample fold deserves heavy weight because it is closest to the current market environment.

Possible explanations:

1. The model’s edge decayed.
2. The market regime changed.
3. The model was overfit to earlier periods.
4. The label is misaligned with the actual trading objective.
5. The entry and exit logic no longer match the environment.
6. Gating logic is suppressing winners or allowing poor-quality trades.
7. Realistic costs would erase already-thin current-period edge.

Current recommendation:

> **Swing should remain paper-only or shadow/probation-only until Fold 3 is explained or a challenger model beats it.**

---

## What Is Working Well

### 1. PM / RM / Trader Separation

The architecture is directionally right.

- PM generates trade ideas.
- RM has veto power.
- Trader handles execution and position management.

This is a good institutional pattern. It allows strategy logic, risk logic, and execution logic to evolve separately.

### 2. Risk Manager Veto Power

The RM having full veto power over all PM proposals is correct. It prevents the PM from becoming an unchecked alpha engine.

Current RM checks include:

- Kill switch
- Circuit breaker
- Buying power
- Daily loss limit
- Max open positions
- Position size
- Sector concentration
- Correlation gate
- Duplicate check

These are all useful.

### 3. Audit and Replay Infrastructure

The audit tooling is a strength. The system records:

- PM decisions
- RM approvals/rejections
- Trader actions
- Orders
- Trades
- P&L
- Gate outcomes
- NIS signals
- Agent status

The `replay_day.py` and `paper_trading_report.py` style tooling is exactly the kind of infrastructure that makes systematic improvement possible.

### 4. Paper Trading Is Already Finding Bugs

Known bugs already fixed include:

- `bars_held` counting heartbeat ticks instead of trading days
- Premarket exit signals
- Partial exit P&L not accumulating
- Daily flags resetting on restart
- Duplicate reconciler behavior
- Runaway post-close intraday scanner
- P&L chart using period P&L instead of cumulative equity
- Uvicorn shutdown hang
- Timezone mismatch in daily flag restore

This is positive. It means the paper-trading phase is exposing real operational issues before money is at risk.

---

## What Does Not Make Sense Yet

### 1. The Live Gate Is Too Simple

The current paper trading gate is:

- 4-week Sharpe > 0.5
- Max drawdown < 5%

That is not enough.

A better gate should include:

| Category | Required Before Live |
|---|---|
| Safety | No duplicate orders, stale proposals, runaway scanners, restart replay issues, or unresolved reconciliations |
| Execution | Slippage, fill delay, missed fills, rejected orders, partial fills, and realistic fill modeling |
| Strategy | Swing and intraday results separated, not just combined P&L |
| Model | Current-period OOS performance, drift, rank stability, calibration |
| Risk | Portfolio-level beta, factor, gross exposure, VaR/CVaR, stress scenarios |
| Operations | Durable queues, order state machine, kill switch verification, broker error handling |

Current gate asks:

> “Did paper performance look good?”

Better gate asks:

> “Would this system behave safely and predictably under live-market conditions?”

### 2. The PM Is Doing Too Much

The current PM is the brain for:

- Swing selection
- Intraday selection
- News overlay
- Premarket intelligence
- Rescoring
- Macro gating
- Held-position logic
- Exit signals

This is workable early, but over time it will become hard to understand what is actually adding value.

Recommended direction:

```text
StrategyBook
  ├── SwingMomentumStrategy
  ├── IntradayOpeningSessionStrategy
  ├── NewsRiskOverlay
  ├── RegimeModel
  └── PortfolioAllocator

PortfolioManager
  └── Aggregates normalized proposals and sends them to RM
```

The PM should become more of a **proposal aggregator and allocator**, not one giant strategy brain.

### 3. The RM Is a Rule Checklist, Not Yet a Portfolio Risk Engine

Sector caps, correlation checks, max positions, and daily loss limits are useful, but they are not a full portfolio risk model.

Missing portfolio-level risk controls include:

| Missing Control | Why It Matters |
|---|---|
| Dollar risk per trade | Position size should be based on stop distance and volatility, not just fixed notional percentage |
| Portfolio beta | Five different names can still behave like one SPY trade |
| Factor exposure | Growth, value, size, momentum, and sector exposures can dominate individual stock edge |
| VaR / CVaR | Needed to estimate tail loss |
| Scenario shocks | Needed to understand how the portfolio behaves in a gap-down or volatility shock |
| Liquidity stress | Needed to know whether positions can be exited without losing the expected edge |

The RM needs to evolve from:

```text
Can this trade pass a checklist?
```

to:

```text
Does this trade improve the portfolio within risk budget?
```

---

## Model Architecture Review

### Current Swing Model

- XGBoost regression
- Label: `path_quality = 1.0×upside_capture − 1.25×stop_pressure + 0.25×close_strength`
- Forward window: 5 trading days
- Stop / target: 0.5×ATR / 1.0×ATR
- Top 10 candidates selected by score

This formulation is not wrong, but it may not be the best final structure.

For a daily top-N strategy, the main problem is not simply predicting a perfect numeric score. The real tasks are:

1. Rank the best candidates for today.
2. Decide whether a candidate is tradable.
3. Estimate expected value after costs.
4. Size according to risk.
5. Decide whether to keep holding after entry.

### Recommended Model Stack

| Layer | Recommended Approach |
|---|---|
| Candidate ranking | Cross-sectional ranking model, such as LambdaMART/XGBoost ranking |
| Trade/no-trade | Binary or ternary classifier using triple-barrier outcome |
| Confidence gate | Meta-label model estimating probability the primary signal succeeds |
| Sizing | Expected value and volatility-adjusted sizing |
| Exit/hold | Separate hold/exit model, not just reused entry model |
| Portfolio allocation | Risk-adjusted optimizer with hard caps |

### Recommended Challenger Labels

Do not replace the current `path_quality` label immediately. Instead, build challengers.

1. **Triple-barrier classifier**
   - Did the trade hit target first?
   - Did it hit stop first?
   - Did it time out?

2. **Cross-sectional ranking label**
   - On each date, rank candidates by forward risk-adjusted return.

3. **Meta-label**
   - Given that the primary model likes the trade, should the system actually take it?

4. **Expected value label**
   - Expected net return after realistic slippage, spread, missed fills, and fees.

Compare all challengers using realistic assumptions, not raw backtest performance.

---

## Swing Fold 3 Postmortem

Before live swing allocation, run a dedicated postmortem on the most recent fold.

### Diagnostics Needed

| Diagnostic | Question |
|---|---|
| Regime breakdown | Did the model fail in high-vol, low-vol, trend, or chop? |
| Sector breakdown | Was loss concentrated in certain sectors? |
| Feature drift | Did top feature distributions change versus earlier folds? |
| Rank decay | Did high-score names stop outperforming low-score names? |
| Entry timing | Did the 09:50 delay help or hurt? |
| Exit logic | Did winners reverse because exits were too slow? |
| Cost sensitivity | Would realistic slippage turn marginal winners into losers? |
| Gate impact | Which gates blocked winners versus losers? |
| Baseline comparison | Did simple momentum or sector rotation beat the ML model? |

### Required Output

```text
Swing model current-state verdict:
- Promote to live?
- Keep paper-only?
- Keep shadow-only?
- Retire and replace?
```

Current recommendation:

> **Keep swing in shadow/probation mode until the most recent fold is explained.**

---

## Intraday Model Review

The intraday model appears more promising than the swing model because the most recent fold is strong:

| Fold | Period | Sharpe |
|---|---:|---:|
| Fold 1 | 2024-10 → 2025-04 | +1.73 |
| Fold 2 | 2025-04 → 2025-10 | +0.72 |
| Fold 3 | 2025-10 → 2026-04 | +2.97 |

However, the Bar 12 opening-session edge should be treated cautiously.

### Why Bar 12 Could Be Real

- The first hour digests overnight information.
- Opening volatility normalizes.
- Liquidity improves.
- Momentum/reversal structure can become clearer.
- Avoiding the open may reduce noise.

### Why Bar 12 Could Be an Artifact

- The system may have searched many bars and selected the best historical one.
- Neighboring bars may not show the same edge.
- Paper fills may exaggerate intraday profitability.
- The edge may be concentrated in a few names or regimes.
- It may be sensitive to exact feature construction.

### Validation Plan

1. Freeze the bar 12 rule.
2. Test truly out-of-sample after the rule was selected.
3. Compare bar 12 with bars 9–15.
4. Apply multiple-testing discipline.
5. Add realistic fill, spread, slippage, and missed-fill assumptions.
6. Validate by symbol liquidity bucket.
7. Validate by regime.
8. Run one-share live probes to compare expected vs actual fills.

Conclusion:

> Intraday may be the better candidate for eventual live testing, but only after execution realism is proven.

---

## Execution Quality Review

The current entry logic uses limit orders at `ask + 1 tick`.

For a $20k account trading liquid large caps, that is not unreasonable. But it is basically a marketable limit order. It protects against runaway prices but does not fully solve execution quality.

### Missing Execution Analytics

The system should measure:

| Metric | Why It Matters |
|---|---|
| Arrival price | Price when PM/RM approved the trade |
| Decision-to-order delay | Measures signal staleness |
| Order-to-fill delay | Measures execution drag |
| Effective spread | Measures implicit cost paid |
| Realized spread | Measures post-fill quality |
| Slippage bps | Already stored, but needs better reporting |
| Missed-fill outcome | Did unfilled orders avoid losers or miss winners? |
| Partial-fill handling | Live behavior can diverge from paper |
| Rejection reason | Broker rejections must be first-class events |
| Exit slippage | Often more important than entry slippage |

### Major Concern

Exit management currently runs every 5 minutes.

For swing positions, this may be acceptable for many checks, but overnight gap risk remains. For intraday trades, five minutes can be too slow in a fast move.

Recommended improvements:

- Broker-native bracket/OCO orders where possible
- Faster order/position monitor for intraday positions
- Emergency flatten-all command
- Cancel-all-open-orders command
- Reconcile broker state frequently
- Treat rejected/partial/canceled orders as normal state transitions, not errors

---

## News Intelligence Service Review

The NIS design is interesting but should be treated as a **risk overlay**, not alpha, for now.

### Current NIS

- Tier 1 macro classification using Claude Sonnet
- Tier 2 symbol headline scoring using Claude Haiku
- Policy engine for `block_entry`, `size_down`, and `exit_review`
- Held-position recheck every 30 minutes

### What Is Good

Using news to block entries or reduce size around material events makes sense. Examples:

- Earnings
- FOMC / CPI / NFP
- FDA decisions
- SEC/DOJ investigations
- Bankruptcy risk
- Major litigation
- Guidance cuts
- M&A / halt risk

### What Is Risky

| Weakness | Why It Matters |
|---|---|
| Free headline coverage | Important news may arrive late or be missed |
| No robust event taxonomy | “Bad news” is too vague |
| No novelty detection | Duplicate headlines can look like fresh signal |
| No price reaction context | News may already be priced in |
| LLM variability | Similar headlines may receive inconsistent scores |
| No historical calibration | You do not yet know whether NIS scores predict outcomes |

### Recommendation

For now:

```text
NIS = event risk classifier, not alpha generator
```

Track whether each NIS decision helped:

```text
Did block_entry block a loser?
Did it block a winner?
Did size_down reduce drawdown?
Did exit_review improve P&L?
Did exit_review cause premature exits?
```

Only after 60–90 days of NIS outcome history should NIS scores become model features or alpha inputs.

---

## Regime Detection Review

Current regime gating is simple:

- VIX ≥ 25
- SPY below 20-day moving average
- SPY 5-day return ≤ 0
- SPY intraday drawdown > 1.5%

This is a good starting safety rule, but too blunt for a mature system.

### Recommended Regime States

| Regime | Behavior |
|---|---|
| Bull trend / low vol | Full swing long budget, normal intraday |
| Bull trend / high vol | Smaller swing, faster exits |
| Bear trend / high vol | No long swing except special setups; intraday reduced |
| Range/chop | Reduce momentum; allow only validated mean-reversion |
| Macro event day | New entries reduced or blocked |
| Liquidity-stress day | No new trades; exits only |

### Recommended Regime Inputs

- SPY trend
- QQQ trend
- IWM confirmation
- VIX level
- VIX change
- Realized volatility
- Market breadth
- Sector dispersion
- Overnight gap
- Macro calendar
- Treasury yields
- Dollar index
- Oil, if relevant
- Credit spread proxy, if available

### Recommended Regime Output

The regime model should not simply say “trade” or “do not trade.”

It should output:

```text
regime_state
risk_budget_multiplier
allowed_strategy_types
max_gross_exposure
max_new_positions
exit_tightening_factor
```

---

## Three Live Failure Modes Paper Trading May Not Catch

### 1. Fill Quality and Queue Reality

Paper trading does not fully capture:

- Market impact
- Latency slippage
- Queue position
- Partial fills
- Real spread behavior
- Price improvement
- Routing quality
- Marketable limit behavior

This is the most likely source of “worked in paper, failed live.”

### 2. Data Feed Mismatch

The system uses Alpaca/IEX and Polygon data.

Potential issue:

- Signals may be generated from one data source.
- Quotes may be checked from another.
- Orders execute against live market reality.
- Paper fills may be based on simplified assumptions.

For intraday strategies, data-feed mismatch can materially distort signals and execution assumptions.

### 3. Broker and Account Rule Behavior

Live trading may expose:

- Buying-power rejections
- Pattern day trading / intraday margin constraints
- Order-type restrictions
- Symbol restrictions
- Shortability issues, if shorts are later added
- Trading halts
- Corporate action issues
- Fractional/share restrictions
- Extended-hours behavior
- Rejected bracket/OCO behavior

Broker responses need to be handled as normal lifecycle events, not exceptions.

---

# Recommended Next Phases

## Phase 1 — Live-Readiness Gate Rebuild

### Goal

Replace the current simple paper P&L gate with a true deployment gate.

### Build

`live_readiness_report.py`

### Report Sections

```text
System Safety:
- duplicate order incidents
- stale proposal incidents
- post-close scanner incidents
- restart replay incidents
- unhandled broker errors
- timezone/date restore mismatches
- unresolved reconciliations

Execution:
- slippage bps by symbol
- fill delay
- missed fills
- partial fills
- rejected orders
- paper-vs-realistic fill adjustment

Strategy:
- swing P&L
- intraday P&L
- combined P&L
- win rate
- expectancy
- average win/loss
- drawdown
- trade count
- exposure-adjusted return

Model:
- latest fold status
- rank IC
- calibration
- AUC/Sharpe drift
- feature drift

Risk:
- gross exposure
- beta exposure
- sector exposure
- correlation
- VaR/CVaR
- scenario loss
```

### Promotion Rule

No live launch until safety and execution gates pass, regardless of paper P&L.

---

## Phase 2 — Execution and Broker Reality Layer

### Goal

Make order handling robust enough for live markets.

### Build

1. **Order lifecycle state machine**

```text
proposed
approved
submitted
accepted
partially_filled
filled
canceled
expired
rejected
replaced
failed_reconciliation
```

2. **Idempotency keys**

```text
proposal_id
approval_id
order_intent_id
broker_order_id
```

3. **Execution quality analytics**

```text
arrival_price
decision_price
submitted_price
fill_price
decision_to_order_delay
order_to_fill_delay
quoted_spread
effective_spread
realized_spread
slippage_bps
missed_fill_outcome
```

4. **Broker-native protection**

- Bracket/OCO where possible
- Emergency flatten-all
- Cancel-all-open-orders
- Reconciliation every few minutes
- Explicit broker rejection handling

5. **Tiny live probe mode**

- One-share or minimum-notional live orders
- No meaningful strategy risk
- Used only to observe live fills, rejections, and slippage

---

## Phase 3 — Swing Model Truth Test

### Goal

Determine whether the swing model has current-market edge.

### Build

```text
swing_fold3_postmortem.py
swing_label_challenger_report.py
swing_gate_contribution_report.py
swing_feature_drift_report.py
```

### Questions

- Did the model lose rank power in Fold 3?
- Did gates help or hurt?
- Is the model just selecting high-beta names?
- Does it beat simple momentum?
- Does it survive realistic costs?
- Does triple-barrier labeling help?
- Does ranking beat regression for top-10 selection?

### Promotion Rule

If Fold 3 remains unexplained or negative after realistic evaluation, swing remains paper-only/shadow-only.

---

## Phase 4 — Portfolio Risk Engine v1

### Goal

Convert RM from rule checklist to portfolio risk engine.

### Add

```text
RiskBudget:
- max daily loss
- max weekly loss
- max per-trade dollar risk
- max strategy risk
- max gross exposure
- max beta-adjusted exposure
- max sector exposure
- max factor exposure
- max correlated cluster exposure
```

### Position Sizing

Use risk-based sizing:

```text
shares = min(
    risk_budget_dollars / stop_distance_dollars,
    max_notional_cap / price,
    liquidity_cap,
    broker_buying_power_cap
)
```

### Suggested Initial Live Risk

For a $20k live account, start far smaller than the current 10–15% notional swing sizing.

Recommended early live stage:

```text
Per-trade risk: 0.25%–0.50% of account
Max gross exposure: 25%–40%
Max open live positions: 2–3
Intraday live: disabled or one-share probe only until execution is proven
```

---

## Phase 5 — Strategy Separation

### Goal

Make each strategy independently measurable.

### Build Strategy Interface

```python
class Strategy:
    name: str
    horizon: str
    required_data: list[str]

    def generate_candidates(self, as_of_time):
        ...

    def score(self, candidates):
        ...

    def propose(self, scores):
        ...

    def explain(self, proposal):
        ...

    def update_from_outcome(self, outcome):
        ...
```

### Split Into

```text
SwingStrategy
IntradayOpeningSessionStrategy
NewsRiskOverlay
RegimeOverlay
PortfolioAllocator
```

### Why

Combined P&L can hide the truth. One strategy may be carrying another. Strategy-level attribution is required before scaling.

---

## Phase 6 — News Intelligence Rebuild

### Goal

Turn NIS into a structured event-risk engine.

### Event Schema

```json
{
  "symbol": "AAPL",
  "event_type": "earnings_guidance_cut",
  "source": "news_provider",
  "first_seen_utc": "...",
  "headline": "...",
  "novelty_score": 0.0,
  "materiality": 0.0,
  "direction": "negative",
  "confidence": 0.0,
  "price_move_since_first_seen_bps": 0,
  "policy_action": "block_entry",
  "reason": "material negative guidance event"
}
```

### Outcome Attribution

Track:

```text
Did block_entry block a loser?
Did block_entry block a winner?
Did size_down reduce drawdown?
Did exit_review improve P&L?
Did exit_review cause premature exit?
```

### Promotion Rule

NIS can become an alpha feature only after 60–90 days of tracked decisions and measurable predictive value.

---

## Phase 7 — Regime Model v1

### Goal

Move beyond binary VIX/SPY gating.

### Inputs

```text
trend_score
vol_score
liquidity_score
macro_score
dispersion_score
breadth_score
```

### Outputs

```text
regime_state
risk_budget_multiplier
strategy_allowlist
max_positions
stop_tightening
new_entry_block
```

### Example Regime Policy

| Regime | Swing | Intraday | Risk |
|---|---|---|---|
| Bull / low vol | Enabled | Enabled | Normal |
| Bull / high vol | Enabled | Enabled | Reduced |
| Bear / high vol | Mostly blocked | Reduced | Very low |
| Chop | Reduced | Selective | Low |
| Macro event | Block/reduce | Block/reduce | Event-specific |
| Liquidity stress | No new trades | No new trades | Exits only |

---

## Phase 8 — Controlled Live Rollout

### Goal

Avoid jumping from paper to full live.

### Rollout Ladder

| Stage | Capital/Risk | Purpose |
|---|---:|---|
| 0 | Paper only | Safety and reporting |
| 1 | Live one-share probe | Broker/fill/rejection validation |
| 2 | Live micro notional | Execution calibration |
| 3 | 25% normal risk | Validate real fill behavior |
| 4 | 50% normal risk | Scale only if stable |
| 5 | Full small-account risk | Only after 30–60 live trading days |

### Stage Promotion Criteria

```text
No duplicate orders
No uncontrolled exposure
No unreconciled positions
No unexplained P&L
No missing audit trail
Slippage within expected band
Strategy-level expectancy remains positive
No unresolved broker-state mismatch
```

---

# What Not To Build Yet

## Do Not Build the True Scalper Yet

The planned scalper requires:

- L1/L2 or tick data
- Much tighter execution
- Different backtesting
- Lower latency
- Smaller universe
- Different strategy logic
- Different risk controls

It should remain deferred until the current system is stable live.

## Do Not Add Options Flow Yet

Options flow adds cost and complexity before the base system has proven execution and risk discipline.

## Do Not Make the LLM an Autonomous Trade Decider

The LLM should not independently decide to buy/sell. Keep it as a structured risk overlay until measured performance supports promotion.

## Do Not Assume Daily Retraining Solves Drift

Daily retraining can make instability worse without:

- Model registry
- Challenger comparison
- Promotion rules
- Rollback path
- Drift alerts
- Feature stability checks

## Do Not Add More Agents Just Because the System Is Agent-Based

More agents will not fix weak validation. Build stronger contracts, state machines, audit trails, and risk gates first.

---

# Recommended Immediate Build Order

If converting this into the next implementation batch, use this order:

1. **Live Readiness Report v2**
   - Replaces simple paper gate
   - Separates swing/intraday
   - Adds safety, execution, model, and risk readiness

2. **Slippage + Execution Quality Report**
   - Uses existing `Order.slippage_bps`
   - Adds arrival price, fill delay, missed-fill outcome

3. **Durable Order State Machine + Idempotency**
   - Prevents replay, duplicate orders, partial-fill confusion, restart bugs

4. **Swing Fold 3 Postmortem**
   - No live swing until explained

5. **Triple-Barrier + Ranking Challenger Models**
   - Compare against current `path_quality` regression

6. **Portfolio Risk Engine v1**
   - Dollar risk per trade
   - Beta exposure
   - Gross exposure
   - VaR/CVaR
   - Scenario shock

7. **NIS Outcome Attribution**
   - Prove whether news blocks/exits help or hurt

8. **Regime Model v1**
   - State + risk multiplier + strategy allowlist

9. **Tiny Live Probe Mode**
   - Real fills and real broker behavior
   - Not full live trading

---

# Suggested Task Breakdown for LLMs

The following can be handed to another LLM to turn into implementation tasks.

## Task A — Build `live_readiness_report.py`

**Objective:** Create a CLI/report that determines whether MrTrader is operationally ready for live trading.

**Must include:**

- Strategy-level P&L
- Swing vs intraday separation
- Slippage
- Fill delay
- Missed fills
- Rejected orders
- Duplicate order incidents
- Restart/reconciliation incidents
- Model drift summary
- Gate contribution summary
- Risk exposure summary
- Pass/fail recommendation

**Acceptance criteria:**

- Produces deterministic report.
- Fails readiness if execution or safety gates fail.
- Does not allow combined P&L to hide strategy-level weakness.

---

## Task B — Build Execution Quality Report

**Objective:** Surface execution drag.

**Metrics:**

- Arrival price
- Approval price
- Submitted limit price
- Fill price
- Slippage bps
- Effective spread
- Fill latency
- Missed-fill outcome
- Partial-fill count
- Rejection count

**Acceptance criteria:**

- Report is available by date, symbol, and strategy.
- Separates entry slippage from exit slippage.
- Flags chronic slippage names.

---

## Task C — Durable Order Lifecycle

**Objective:** Make order handling robust under restarts and partial fills.

**Required order states:**

```text
proposed
approved
submitted
accepted
partially_filled
filled
canceled
expired
rejected
replaced
failed_reconciliation
```

**Acceptance criteria:**

- No duplicate broker order on restart.
- Every order intent has a stable idempotency key.
- Broker rejection is logged as a normal lifecycle event.
- Partial fills update position state correctly.
- Reconciliation resolves broker-vs-DB mismatch.

---

## Task D — Swing Fold 3 Postmortem

**Objective:** Explain why recent swing OOS performance degraded.

**Reports:**

- Regime breakdown
- Sector breakdown
- Feature drift
- Gate impact
- Cost sensitivity
- Baseline comparison
- Rank IC by period

**Acceptance criteria:**

- Produces recommendation: promote, paper-only, shadow-only, or retire.
- Does not rely on average Sharpe alone.
- Explicitly compares Fold 3 to Folds 1 and 2.

---

## Task E — Challenger Label Framework

**Objective:** Compare current `path_quality` regression against better labels.

**Labels:**

- Triple-barrier outcome
- Cross-sectional rank
- Binary positive-EV classifier
- Meta-label

**Acceptance criteria:**

- Same train/test splits as existing model.
- Same realistic cost assumptions.
- Reports performance by fold and by current period.
- Does not promote a challenger unless it beats current production model out-of-sample.

---

## Task F — Portfolio Risk Engine v1

**Objective:** Replace purely fixed notional sizing with risk-budget sizing.

**Features:**

- Per-trade dollar risk
- Stop-distance sizing
- Gross exposure cap
- Beta exposure estimate
- Sector/factor concentration
- Correlated-cluster exposure
- Scenario shock loss estimate

**Acceptance criteria:**

- RM approval includes portfolio-level risk impact.
- Position size is based on risk budget and stop distance.
- Trade can be rejected even if all old rule gates pass.

---

## Task G — NIS Outcome Attribution

**Objective:** Determine whether news intelligence decisions actually help.

**Track:**

- Blocked entries
- Sized-down entries
- Exit review signals
- Subsequent P&L
- Blocked-winner count
- Blocked-loser count
- Premature-exit count

**Acceptance criteria:**

- NIS decisions are evaluated against actual outcomes.
- NIS cannot be promoted to alpha input without positive measured value.
- Report shows action-level contribution.

---

## Task H — Regime Model v1

**Objective:** Replace binary VIX/SPY gate with state-based regime policy.

**Inputs:**

- Trend
- Volatility
- Breadth
- Liquidity
- Macro calendar
- Sector dispersion

**Outputs:**

- Regime state
- Risk multiplier
- Strategy allowlist
- Max positions
- Stop tightening factor
- Entry block flag

**Acceptance criteria:**

- PM/RM can consume the regime output.
- Regime decisions are logged.
- Regime policy impact is backtestable and auditable.

---

## Task I — Tiny Live Probe Mode

**Objective:** Test real broker behavior without meaningful capital risk.

**Features:**

- One-share or minimum-notional mode
- Strict whitelist of symbols
- Max one live probe at a time
- No leverage
- Auto-flatten
- Full audit
- Compare expected paper fill vs live fill

**Acceptance criteria:**

- System can run a live probe without enabling normal live strategy size.
- Probe produces execution comparison report.
- Any broker rejection or mismatch blocks promotion.

---

# Final Blunt Take

MrTrader is promising, but the next phase should be about **proof**, not expansion.

The project should move from:

```text
Paper P&L looks good, so prepare for live.
```

to:

```text
Every strategy, gate, model, news decision, and order type must prove its marginal value under realistic execution and current-regime validation.
```

Top five priorities:

1. **Execution realism**
2. **Swing model Fold 3 investigation**
3. **Portfolio-level risk engine**
4. **Durable order state and idempotency**
5. **Strategy-level attribution and live-readiness gate**

Only after those are solid should the system expand into more alpha features, richer news intelligence, options flow, or a true scalper.

---

# Reference Notes for Other LLMs

These were the main external concepts and references used in the analysis:

1. **Paper trading limitations**
   - Alpaca documentation notes that paper trading is simulated and does not fully reflect market impact, information leakage, latency slippage, order queue position, price improvement, fees, dividends, and other live-market factors.

2. **Deflated Sharpe / backtest overfitting**
   - Bailey and López de Prado’s Deflated Sharpe Ratio work is relevant when many strategies, parameters, labels, or feature combinations are tested.

3. **Triple-barrier labeling**
   - Triple-barrier labeling is relevant because the system already uses stop, target, and max-hold logic. It labels trades based on which barrier is reached first rather than only fixed-horizon return.

4. **Expected Shortfall / institutional risk**
   - Expected shortfall-style tail-risk thinking is more robust than only simple drawdown or VaR-style metrics.

5. **Factor exposure**
   - Basic equity factor exposure, such as market, size, value, momentum, and sector exposure, should be considered before scaling a long-equity book.

6. **News analytics**
   - Academic and industry evidence suggests news sentiment can affect short-horizon returns, volume, and liquidity, but late or poorly calibrated news signals can be dangerous. For this system, news should initially be risk control, not alpha.

