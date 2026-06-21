# MrTrader Risk Committee Review — 2026-06-21

**Role assumed:** external hedge-fund quant developer / PM / CRO.  
**Tone:** deliberately skeptical.  
**Primary decision:** whether the current paper multi-premia book should move toward real capital, and under what conditions.

---

## Executive verdict

You have made real progress since the prior review. The carry mechanics are cleaner, the in-sample vol-match artifact was removed, negative controls were added, the free futures zoo was mined without sign-flipping, and the second engine is no longer just a hand-wavy diversification story. That matters.

But the risk-committee answer is still **not “go size it.”** The correct answer is:

> **Trend can remain live-paper / live as-is. The futures book may move to IBKR paper and then tiny-live only as an execution-validation experiment. It is not yet cleared for meaningful capital allocation. VRP should not be treated as a fourth equal sleeve; it is a small, gated, short-crisis return enhancer until proven otherwise.**

The main reason is simple: **your t = 2.29 result is promising but not yet selection-aware.** A raw residual-alpha t-stat over 1.96 is not enough after ~20 sleeve families, six futures zoo factors, a discovered basket, and a reopened VRP idea. The single-sleeve Type-I control work is useful, but it does not answer the family-wise question. Until the null-strategy zoo / SPA-style max-stat test is run, I would not allow the futures book to receive more than plumbing-size capital.

The second reason: **you still do not have a book-level risk system.** Two brokers, futures margin, short-vol, weekly rebalancing, paper-to-live transition, and no single risk surface is not a harmless implementation detail. It is the difference between “I have backtests” and “I can operate a book.”

The third reason: **your diversification is still unproven in the only state that matters.** Carry, cross-sectional momentum, trend, and gated VRP are not obviously one bet day-to-day, but they may still become one bet in liquidity shocks. Unconditional correlations of 0.12–0.46 do not certify crisis diversification.

My Monday-morning risk committee decision:

1. **Do not scale futures capital until the family-wise null zoo passes.**
2. **Do not run VRP at equal risk. Cap it hard or leave it paper.**
3. **Build the unified risk layer before any non-trivial IBKR live trading.**
4. **Allow tiny-live futures only if explicitly labeled as execution validation, not edge validation.**
5. **Use cash as the default allocation until the above is built.**

---

## What I think you may still have wrong

### 1. You are too close to calling the futures book “validated.”

The futures book is **credible**. It is not yet **capital-cleared**.

The distinction matters. A t = 2.29 residual-alpha result from carry + xs-momentum is useful evidence, especially because both have plausible economic stories and are not random equity factor soup. But the actual research path matters: you tried many families, then mined six futures factors, then combined two marginal factors into a basket that crossed the conventional line.

That may be legitimate. It may also be the exact shape of a selection artifact.

The only acceptable answer is not a debate. It is a selection-aware null:

> “If I rerun my actual research selection procedure on null strategies with the same data structure, how often do I get a best surviving basket with residual-alpha t ≥ 2.29?”

Until that answer is known, the futures book is **yellow**, not green.

### 2. “Gate the basket” is valid only if the gate includes the search process.

Combining carry and xs-momentum can legitimately raise significance if:

- the equal-weight basket was specified before observing the winning combination, or
- the selection-aware test includes the act of testing singles, pairs, and baskets, then choosing the best book-improvement statistic.

It is not valid if the implicit procedure was:

1. test several families;
2. find two marginal survivors;
3. combine them;
4. declare the combined t-stat as if it were the only hypothesis tested.

The null must replicate the *researcher*, not just the strategy.

### 3. VRP is not a diversifier yet.

VRP is short-vol. A contango/backwardation gate is a good crash governor, but a short-vol strategy that survives Feb-2018 and COVID in a manual event study is not automatically a diversifier. It may be a return enhancer with left-tail truncation. That can still be valuable, but it should be sized like a dangerous sleeve.

My default before further evidence:

> **VRP max 5–10% of total ex-ante risk, never equal-risk with trend or futures book, and zero allocation if the tail-correlation / expected-shortfall test says it worsens the book left tail.**

### 4. The risk layer is now the bottleneck, not strategy discovery.

The app was previously ahead of the alpha. Now the research harness may be catching up, but the **risk architecture is behind the proposed book**.

The go-live layer you describe as unbuilt is not optional. It is the capital allocation system. Without it, IBKR futures should be paper or tiny-live only.

### 5. “The free zoo is exhausted” is not a capital argument.

Exhausting the zoo is good discipline. But from a statistical perspective it also confirms that there was a zoo. The dead tests do not disappear. They must enter the family-wise burden unless they were genuinely pre-allocated with separate alpha budgets and frozen hypotheses.

---

# Theme B — Is the edge real or multiple-testing residue?

This gates everything.

## B1. Does t = 2.29 survive honest family-wise correction?

**My current answer:** unknown, and I would not assume yes.

A rough sanity check:

- A one-sided t ≈ 2.29 is roughly p ≈ 1.1% under a large-sample normal approximation.
- If you had 20 independent opportunities, a Bonferroni 5% family-wise bar would be p < 0.25%, roughly t ≈ 2.8.
- If the effective number of tests were only six, the bar would be p < 0.83%, roughly t ≈ 2.4.
- If the effective number were four, t = 2.29 might pass.

So the t = 2.29 result sits in the uncomfortable middle: **too good to dismiss, too weak to bless without empirical deflation.**

Because the tests are correlated and because some premia have strong economic priors, Bonferroni is too crude. But “Bonferroni is too crude” does not mean “use 1.96.” It means build an empirical max-stat null.

### Practical bar

Use three bars:

| Decision | Required bar |
|---|---:|
| **Continue research / IBKR paper** | raw pass is enough, plus no implementation bugs |
| **Tiny-live execution test** | empirical family-wise p ≤ 0.10 under the null zoo; no tail-risk failure; risk layer live |
| **Meaningful capital scale** | empirical family-wise p ≤ 0.05 under the null zoo; survives at least two null designs; tail-risk pass; live execution consistency |
| **Call it a core engine** | empirical family-wise p ≤ 0.025, stable across subperiods / clusters / cost shocks, and live process metrics clean for 6–12 months |

If your observed t = 2.29 falls below the 90th percentile but not the 95th percentile of the selection-aware max-stat null, I would allow paper and maybe tiny-live plumbing. I would not allocate meaningful risk.

If it clears the 95th percentile under the strict null and the SPA p-value is ≤ 0.05, the futures book becomes a real candidate for scaled risk.

If it clears only the easiest null, it remains a research artifact.

## B2. Is combining carry + xs-momentum legitimate?

**Legitimate if selection-aware. Suspicious if not.**

The economic case is plausible:

- carry harvests term-structure / inventory / hedging pressure premia;
- cross-sectional momentum harvests persistence / underreaction / slow-moving capital;
- the two can be near-orthogonal across futures markets;
- equal-risk combination can improve residual alpha without being a trick.

But the statistical case depends on what was allowed.

### Clean case

The basket is clean if the pre-registered rule was:

> “If multiple canonical futures premia pass Track-A, combine them by fixed equal-risk weights and test the fixed basket against the live book.”

Then the combined t-stat is not automatically p-hacked. You still pay for testing the family, but not for arbitrary weight mining.

### Dirty case

The basket is dirty if the implicit process was:

> “Try six factors, keep the two that look best, combine them because the combined line is prettier, then test the basket as one hypothesis.”

That is exactly what max-stat / SPA must penalize.

### Required test

Run the null zoo using the same allowed selection rule:

1. generate null versions of the six futures factors;
2. compute Track-A for singles;
3. construct every allowed fixed basket from Track-A survivors;
4. compute Track-B residual-alpha t versus trend;
5. record the maximum statistic the researcher would have selected;
6. compare observed t = 2.29 to that null distribution.

If the observed carry+xsmom basket beats the null distribution after the same selection process, it is legitimate.

## B3. Null-strategy zoo protocol you can build this week

The goal is to answer:

> How often does my research harness manufacture a “validated second engine” from structured noise?

### Step 0 — Freeze the actual researcher procedure

Before generating nulls, write down the exact procedure you are testing. This must include:

- candidate family list;
- allowed config variants;
- allowed universe filters;
- allowed cost assumptions;
- allowed weighting rules;
- Track-A gate;
- Track-B gate;
- rule for combining sleeves;
- final statistic used for promotion.

For the immediate test, define the relevant search space as at least:

- futures carry;
- futures xs-momentum;
- curve momentum;
- value;
- skew;
- basis momentum;
- CFTC CoT if data quality is acceptable;
- VIX-VRP variants if VRP is being judged in the same capital decision.

Do not let the null omit dead families that informed your conclusion.

### Step 1 — Define the primary statistic

Use the statistic that actually drives allocation:

> **Maximum HAC residual-alpha t-stat versus the existing live trend book after applying the same Track-A / Track-B selection procedure.**

This is better than raw Sharpe because your real question is book improvement.

Also store secondary statistics:

- standalone HAC Sharpe t;
- appraisal ratio / residual Sharpe;
- bootstrap P(ΔSR > 0);
- max drawdown contribution;
- expected shortfall contribution;
- PBO / CSCV if configs were selected.

But the decision statistic should be one thing. I would use max residual-alpha t.

### Step 2 — Use at least three null designs

You want nulls that preserve the ways futures data can fool you: autocorrelation, cross-sectional structure, asset-class clustering, contract availability, roll timing, volatility targeting, and costs.

#### Null A — Cross-sectional random-rank null

At each rebalance date:

1. keep the eligible universe exactly as it was;
2. keep each market's volatility estimate, contract selection, cost model, and missing-data state;
3. randomly assign ranks within the eligible universe;
4. optionally preserve asset-class counts, e.g. randomize within {equity index, rates, FX, energy, metals, ags, vol};
5. construct long/short portfolios with the exact same inverse-vol and book-vol rules;
6. run the full backtest.

This tests whether the cross-sectional machinery itself generates false positives.

Use both variants:

- **A1:** randomize across the full universe;
- **A2:** randomize within asset-class clusters.

A2 is stricter because it preserves sector composition and prevents the null from being too dumb.

#### Null B — Circular-shift factor null

For each market and factor panel:

1. circularly shift the factor time series by a random offset, e.g. 252–2,000 trading days;
2. keep returns untouched;
3. keep factor autocorrelation, seasonality, missingness, and cross-market structure partly intact;
4. break the true timing relation between signal and subsequent return;
5. rerun the full strategy.

This is especially useful for rules-based factors because there is no model to retrain. You are destroying predictive alignment while preserving the factor's time-series texture.

#### Null C — Block-permuted return null

For each market or asset-class block:

1. split returns into 63-day or 126-day blocks;
2. resample blocks with replacement or circular permutation;
3. preserve within-block volatility clustering and cross-asset co-movement if possible;
4. keep signals fixed;
5. rerun the strategy.

This tests whether the apparent edge depends on a lucky historical alignment of return blocks.

Use carefully: if you break cross-market dependence too aggressively, the null becomes unrealistic. Prefer asset-class synchronized blocks over independent market-level blocks.

#### Null D — Sign-flip / direction null

For directional families:

1. randomly multiply signals by +1 or -1 by market, asset class, or epoch;
2. keep signal strength and turnover;
3. run the full harness.

This is useful for detecting whether your engine can pass on random directional choices.

### Step 3 — Replicate the actual selection process

For each null replication:

1. build all null candidate sleeves;
2. apply the same cost model and roll model;
3. apply Track-A;
4. construct all allowed fixed baskets from Track-A survivors;
5. apply Track-B against the current live trend book;
6. record the best selected statistic;
7. record whether the null researcher would have declared “second engine found.”

This is the critical point. Do **not** test carry and xs-momentum in isolation. Test the full procedure that found them.

### Step 4 — Number of replications

Minimum:

- **5,000 null replications** for a quick answer.

Preferred:

- **10,000–20,000 replications** for stable 95th / 99th percentile estimates.

If compute is expensive, start with 2,000 to debug, then run 10,000 overnight.

### Step 5 — Decision metrics

Compute:

```text
empirical_p = (1 + count(null_max_stat >= observed_stat)) / (1 + B)
```

Where `observed_stat` is the selected real futures_book residual-alpha t-stat, currently 2.29.

Also compute:

```text
family_false_positive_rate = count(null_researcher_declares_pass) / B
```

This tells you how often your entire harness manufactures a pass.

### Step 6 — SPA / Reality Check layer

After the null zoo, run a White Reality Check / Hansen SPA-style bootstrap over the candidate performance differentials versus the benchmark book.

Use:

- candidate differential = candidate sleeve or basket return minus benchmark-adjusted residual return;
- stationary bootstrap blocks, e.g. mean block length 20–60 trading days;
- studentized t-statistics;
- max statistic across candidates;
- SPA adjustment so terrible models do not overly dilute the test.

Decision:

- SPA p ≤ 0.05: candidate survives family-wise data-snooping adjustment.
- SPA p between 0.05 and 0.10: tiny-live only, no scale.
- SPA p > 0.10: paper only.

### Step 7 — Report the “deflated t bar”

Your output should say something like:

```text
Observed selected t: 2.29
Null max-stat 90th percentile: X
Null max-stat 95th percentile: Y
Null max-stat 99th percentile: Z
Empirical FWER p: P
SPA p: P_spa
Decision: paper / tiny-live / scale-candidate
```

This is more useful than arguing about a universal threshold.

## B4. Prospective research degrees-of-freedom ledger

Build this as an immutable table, not a note in a Markdown file.

### Required fields

For every research run:

```yaml
experiment_id:
timestamp_start:
timestamp_end:
researcher:
family_id:
family_name:
prior_class: [canonical_literature, economic_prior, exploratory, bugfix, reviewer_suggested]
hypothesis_text:
pre_registered: true/false
capital_decision_relevant: true/false
universe:
data_version:
feature_version:
label_version:
cost_model_version:
roll_model_version:
configs_allowed:
configs_tested_count:
selection_metric:
promotion_gate:
benchmark_book:
lookbacks_tested:
filters_tested:
exclusions:
post_hoc_changes:
performance_viewed_before_change: true/false
bugfix: true/false
bugfix_description:
result_status: [killed, parked, paper_pass, capital_candidate, live]
reason_for_status:
alpha_budget_spent:
notes:
code_commit_hash:
artifact_paths:
```

### Rules

1. **Every variant counts unless it was registered before seeing performance.**
2. **Bug-fix reruns count unless the bug was purely implementation and documented before inspecting the improved result.**
3. **Reviewer-suggested specs count.** Good ideas from outsiders are still tests.
4. **Post-hoc exclusions count heavily.** If you remove markets, dates, or regimes after seeing pain, that is a new variant.
5. **Family IDs persist.** A new lookback inside futures momentum is not a new family. It is a variant inside the same family.
6. **Keep an alpha-spending ledger.** Each family gets a pre-specified testing budget. When the budget is spent, further tests are exploratory and cannot promote without a fresh holdout or much stricter bar.

### Monthly governance

At month-end, auto-generate:

- number of new families;
- number of variants;
- number of tests viewed;
- number of killed / parked / promoted;
- current effective test count;
- family-wise adjusted bar;
- list of post-hoc changes;
- list of “dangerous reruns.”

This is the discipline that keeps a solo researcher from slowly converting curiosity into p-hacking.

---

# Theme A — Go-live sizing and risk architecture

## A1. Sizing: book vol, drawdown budget, sleeve weights, margin

### My sizing principle

Size by:

1. **max tolerable drawdown;**
2. **ex-ante volatility;**
3. **stress loss;**
4. **margin and contract discreteness;**
5. **live evidence multiplier.**

Never size by Kelly. Kelly is a diagnostic, not an allocation rule for this book.

### Recommended target volatility

For a solo operator with roughly $100k and paper-to-live transition risk:

| Stage | Book vol target | Hard cap | Comment |
|---|---:|---:|---|
| Current / before risk layer | 3–5% | 6% | Trend only plus cash; no meaningful futures risk |
| Tiny-live futures execution test | 4–6% | 7% | Futures sleeves at fractional evidence weights |
| Mature but still solo book | 6–8% | 10% | Only after null, tail, and live execution pass |
| Aggressive future state | 10% | 12% | Not appropriate now; requires long clean live record |

If your true emotional / financial max drawdown is 10%, target 6% vol. If you can tolerate 15%, target 8% vol. Do not pretend you can tolerate a drawdown you have never lived through with real money.

### Drawdown budget

Use a book-level drawdown budget independent of sleeve-level gates:

| Live book drawdown | Action |
|---:|---|
| -4% | Freeze new risk increases; review slippage / correlations |
| -6% | Cut total target risk by 25%; turn VRP off unless it is explicitly hedged |
| -8% | Cut total target risk by 50%; futures paper/tiny sleeves to minimum |
| -10% | Halt new trades; flatten non-core sleeves; manual risk review |
| -12% | Global kill / capital preservation mode; only cash or minimal trend after review |

Recovery rule: do not immediately re-risk after a bounce. Restore one rung only after either:

- 20 trading days without a new drawdown low and risk metrics normalized, or
- recovery of at least half the drawdown plus normalized correlations / vol.

### Sleeve evidence multipliers

Do not let paper sleeves enter at equal risk to live-proven trend.

Define target risk contribution as:

```text
allocated_sleeve_risk = strategic_sleeve_risk × evidence_multiplier × risk_state_multiplier
```

Suggested evidence multipliers:

| Sleeve | Now | After null zoo pass | After IBKR paper pass | After 3–6 mo tiny-live clean | After 12 mo clean live |
|---|---:|---:|---:|---:|---:|
| ETF trend | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Futures book | 0.00 | 0.20 | 0.25 | 0.40–0.50 | 0.75 |
| VRP | 0.00 | 0.05 | 0.05–0.10 | 0.10 | 0.15 max |

VRP should not get to 1.00 evidence multiplier just because its standalone Sharpe is decent. Short-crisis premia deserve structural caps.

### Strategic risk weights

With only 3–4 sleeves, HRP does not earn its complexity. Use equal-risk contribution with conservative covariance shrinkage.

Initial strategic risk weights after all gates pass:

| Sleeve | Strategic risk contribution |
|---|---:|
| ETF trend | 50–60% |
| Futures book | 30–40% |
| VRP | 5–10% |
| Cash | residual, excluded from risk caps |

Mature state, only after live evidence:

| Sleeve | Mature risk contribution |
|---|---:|
| ETF trend | 40–50% |
| Futures book | 35–45% |
| VRP | 5–10%, 15% absolute max |
| Defensive / convex sleeve if built | 10–20% funded by reducing VRP and trend |

Do not use optimizer weights. With three sleeves, a covariance optimizer will mostly optimize noise.

### Covariance estimate

Use the most conservative of:

- long-history covariance;
- rolling 3-year covariance;
- rolling 6-month covariance;
- stress-window covariance;
- tail-adjusted covariance where stress correlations are floored at observed stress values.

For allocation, use the covariance matrix that produces the **highest estimated book volatility** for the proposed weights.

### Margin caps

Futures margin is a separate constraint from volatility.

Recommended caps:

| Metric | Tiny-live | Mature max | Hard stop |
|---|---:|---:|---:|
| Initial margin / equity | 5–10% | 20–25% | 35% |
| Stress margin / equity, assuming 2× exchange margin | <25% | <45% | 60% |
| Available liquidity after margin | >75% | >60% | <50% = no new risk |
| Single contract 1-day 99% stress loss | <0.25–0.50% NAV | <1.0% NAV | >1.5% NAV = too large |

If the smallest tradable contract violates the risk budget, the answer is not “round up.” The answer is “do not trade that market live yet.”

## A2. Cross-venue risk aggregation

### Minimal correct unified risk surface

Build one daily canonical book snapshot across Alpaca and IBKR.

Required fields:

```text
as_of_time
broker
account_id
currency
NAV / equity
cash
settled_cash
buying_power
initial_margin
maintenance_margin
excess_liquidity
positions by canonical_instrument_id
quantity
multiplier
market_price
market_value
notional_exposure
asset_class
contract_expiry
days_to_expiry
sleeve_owner
strategy_target_quantity
actual_quantity
open_orders
unrealized_pnl
realized_pnl
```

Derived book-level metrics:

```text
total_NAV
cash_pct
gross_notional / NAV
net_notional / NAV
long_gross / NAV
short_gross / NAV
initial_margin / NAV
maintenance_margin / NAV
ex_ante_vol
realized_20d_vol
realized_60d_vol
current_drawdown
sleeve PnL and returns
sleeve risk contributions
cross-sleeve correlation
stress expected shortfall
factor betas: SPY, TLT/rates, DXY, broad commodities, VIX/VX
largest single-market stress loss
stale data flags
reconciliation breaks
```

### Broker is source of truth

Your DB is intent. The broker is reality.

Before every rebalance:

1. pull Alpaca positions / cash / open orders;
2. pull IBKR positions / cash / margin / open orders;
3. normalize into canonical book;
4. compare to intended positions;
5. block trading if mismatch exceeds tolerance.

Suggested tolerances:

| Break | Action |
|---|---|
| stale broker snapshot > 5 minutes during trading workflow | halt new orders |
| position mismatch > 1 share / 1 contract / $100 notional equivalent | reconcile before trading |
| unknown open order exists | cancel or manually resolve before new orders |
| cash / equity mismatch > 0.5% NAV | halt new orders |
| margin field unavailable from IBKR | no futures orders |
| contract expiry mapping uncertain | no futures orders in that market |

### Single kill-switch architecture

Implement a central risk-state machine:

```text
RUN
HALT_NEW_ORDERS
REDUCE_ONLY
FLATTEN_NON_CORE
FLATTEN_ALL
MANUAL_LOCK
```

Rules:

- All trader agents must read the global state before creating orders.
- All order routers must enforce the global state independently of the agents.
- The state must be persisted in an immutable table plus a fast cache.
- A manual override must be possible from the dashboard and from a CLI script.
- A dead-man trigger must set at least `HALT_NEW_ORDERS` if heartbeats fail.

Flattening sequence:

1. cancel open orders at both brokers;
2. confirm cancellation or mark unknown;
3. flatten IBKR futures first, especially VX / short-vol exposures;
4. flatten leveraged or margin-heavy futures next;
5. flatten ETFs last unless equity-market crash governor requires otherwise;
6. verify broker positions are flat;
7. write immutable post-mortem snapshot.

You need an out-of-band flatten script that does not depend on the full app stack. If Redis, Postgres, or FastAPI is down, you still need to flatten.

## A3. Forward risk: correlation spike and drawdown de-risk

### Realized-correlation-spike trigger

Use sleeve-level daily returns after applying actual allocations. Also compute equal-risk normalized sleeve returns so the correlation measure is not dominated by allocation size.

Windows:

- fast: 21 trading days;
- medium: 63 trading days;
- slow baseline: 252 trading days.

Metrics:

```text
avg_pairwise_corr_21d
avg_pairwise_corr_63d
max_pairwise_corr_21d
first_PC_variance_share_63d
stress_corr = corr on days when book return < 20th percentile or VIX change top quintile
```

Trigger levels:

| Condition | Action |
|---|---|
| avg pairwise corr 63d > 0.60 or first PC > 65% | freeze risk increases |
| avg pairwise corr 63d > 0.70 and book DD < -4% | cut gross / vol target 25% |
| avg pairwise corr 21d > 0.80 or first PC > 80% and DD < -6% | cut gross / vol target 50%; VRP off |
| stress corr > 0.75 in current stress window | treat as one-bet mode; only trend/cash unless manual review |

Do not cut just because correlations rise during a benign rally. Require either drawdown, realized-vol spike, VIX spike, or expected-shortfall deterioration.

### Avoid overfitting

Use round thresholds. Do not optimize them. Calibrate only to answer: “Would this have reduced exposure in known bad clustering regimes without constantly firing?”

The trigger should be a circuit breaker, not a trading signal.

## A4. Promotion ladder

### Rung 0 — Research pass to IBKR paper

Required before IBKR paper:

- deterministic contract master;
- expiry / roll calendar tested;
- multipliers verified;
- Norgate symbol to IBKR contract mapping verified;
- commission / spread model specified;
- risk layer can see hypothetical positions;
- null zoo result known, even if not fully passing for scale.

If null zoo fails badly, paper can still run as a research monitor, but do not call it a capital candidate.

### Rung 1 — IBKR paper to tiny-live

Minimum IBKR paper evidence:

| Requirement | Threshold |
|---|---:|
| calendar time | 8–12 weeks minimum |
| weekly rebalances | at least 10 clean cycles |
| rolls observed | at least 2 monthly/quarterly roll events relevant to traded contracts |
| VX roll if VRP included | at least 1 clean VX roll |
| order mapping errors | 0 unresolved |
| expired / wrong contract attempts | 0 |
| reconciliation breaks | 0 unresolved before next trade |
| intended vs paper actual position match | >99% notional match after rebalance |
| intended vs actual daily PnL correlation | >0.95 once positions are active |
| tracking error | <25% of expected sleeve vol, excluding known paper-fill artifacts |
| modeled slippage vs paper fills | paper fills no worse than 25–35% of expected gross edge |

Vol spike requirement:

- Ideally observe at least one of: SPY down >2%, VIX +15% in a day, VIX term backwardation transition, rates shock, or commodities shock.
- If no stress occurs after 12 weeks, tiny-live may begin **only** as plumbing validation at extremely small size. It cannot scale until a real stress / vol event is observed cleanly.

### Rung 2 — Tiny-live

Tiny-live sizing:

- trade only contracts whose minimum size fits the risk budget;
- 1 contract per market where possible;
- sleeve risk no more than 5–10% of eventual target;
- total futures initial margin <5–10% equity;
- VRP either excluded or capped at trivial size.

Tiny-live evidence required before scaling:

| Requirement | Threshold |
|---|---:|
| calendar time | 3–6 months minimum |
| rebalance cycles | 12–24 clean cycles |
| rolls | 3+ clean roll events across traded markets |
| live intended-vs-actual PnL corr | >0.90 |
| tracking error | <30% of expected sleeve vol |
| realized slippage | <30–40% of expected edge |
| margin surprises | 0 material surprises |
| operational breaks | no unresolved break that could have caused wrong exposure |
| drawdown | not worse than 1.5× backtest expectation at equivalent vol without explainable market regime |

Demotion triggers:

- wrong contract traded;
- kill-switch failed to prevent or flatten risk;
- broker reconciliation stale or wrong during rebalance;
- realized slippage >50% of expected edge for two consecutive months;
- tracking error >40% of expected sleeve vol;
- margin usage exceeds cap;
- VRP loses money during a state where the gate says it should be flat;
- manual intervention required to avoid a material unintended position.

### Rung 3 — Scale

Scale by process evidence, not by recent returns.

Rules:

- increase risk no more than once per quarter;
- increase by no more than 25% of target allocation per step;
- do not increase after a large positive outlier month; wait at least one rebalance cycle;
- do not scale if correlations, margin, or drawdown state are elevated;
- require live-vs-shadow consistency.

Scale thresholds:

| Scale level | Requirements |
|---|---|
| 25% target | null zoo p ≤ 0.10, tiny-live clean 3 months, risk layer stable |
| 50% target | null zoo p ≤ 0.05, 6 months clean live, slippage <35% edge, no risk breaks |
| 75% target | 9–12 months clean live, at least one stress event handled correctly |
| 100% target | 12–24 months clean live, no tail-test failure, no unresolved operational incident |

Do not demand live t-stat significance; you will not have enough live data. Demand process fidelity, slippage control, and regime consistency.

## A5. Hard no-go list

I would not deploy meaningful capital if any of these are true:

### Statistical no-go

- selection-aware null zoo empirical p > 0.10 for futures_book;
- SPA p > 0.10;
- observed t = 2.29 is below the 90th percentile of the max-stat null;
- carry+xsmom basket only passes when the null omits dead futures factors;
- result depends on one asset class, one decade, one roll assumption, or one cost assumption;
- ex-energy / ex-VIX / ex-top-5-markets version collapses below usefulness;
- paper-to-live tracker materially diverges from backtest intent.

### Tail-risk no-go

- stress conditional average correlation >0.70;
- first PC explains >75% of stress-window variance;
- lower-tail dependence is >0.40 between core sleeves;
- worst 5% book days show all sleeves losing together more than 60% of the time;
- VRP worsens book expected shortfall more than it improves median return;
- book loses more than trend-alone in the majority of historical crisis windows;
- no sleeve provides positive or flat contribution in equity/VIX shock windows.

### Operational no-go

- no unified risk surface;
- no single kill-switch across Alpaca and IBKR;
- no independent flatten script;
- IBKR margin unavailable or not reconciled daily;
- contract mapping not deterministic;
- expired / wrong contract risk unresolved;
- broker positions cannot be reconciled to intended positions;
- smallest futures contract size violates risk budget;
- open-order state cannot be trusted.

### Sizing no-go

- futures initial margin >25% equity before a long clean live record;
- stress margin >50% equity;
- any one market can lose >1% NAV on a plausible one-day shock at intended size;
- VRP sized above 10% risk without a compensating convex sleeve;
- book vol target above 8% before the full ladder is complete.

---

# Theme C — Diversification and co-crash risk

## C1. Are you genuinely diversified?

My current view:

> **You are more diversified than six weeks ago, but you are not yet proven diversified in crisis. Treat the book as a long-risk-premia portfolio with some trend convexity, not as a true all-weather book.**

Trend is the best sleeve because it can become defensive in slow bear markets. But fast crashes can whipsaw it. Carry can be short liquidity / inventory stress. Cross-sectional momentum can delever during reversals. VRP is explicitly short-crisis even if gated.

The tell is not average correlation. The tell is:

- who loses in the worst 5% of book days;
- who loses when VIX jumps;
- who loses during liquidity gaps;
- whether the first principal component dominates in stress;
- whether VRP makes the left tail worse;
- whether the book drawdown is materially smaller than trend-alone in crisis windows.

If all sleeves are different names for “risk appetite is stable,” you have one bet.

## C2. Tail-dependence tests to run

Build daily sleeve return series at equal ex-ante risk and at proposed live allocation.

### Stress definitions

Use multiple definitions:

```text
SPY worst 5% daily returns
SPY worst 5% 5-day returns
VIX daily change top 5%
VIX level top decile
VIX term structure backwardation windows
book worst 5% daily returns
book worst 5% 5-day returns
rates shock top 5%
DXY shock top 5%
commodity shock top 5%
known crisis windows
```

Known crisis windows should include at least:

- 2008 GFC;
- 2011 euro / US downgrade;
- 2013 taper tantrum;
- 2014–2015 oil shock;
- August 2015;
- February 2018 Volmageddon;
- Q4 2018;
- March 2020 COVID;
- 2022 inflation / rates shock;
- March 2023 bank stress;
- any 2024–2026 vol shock available in your data.

### Metrics

1. **Conditional correlation matrix** in each stress definition.
2. **Exceedance correlation:** correlation of returns conditional on either sleeve being in its bottom 5%.
3. **Lower-tail dependence:**

```text
lambda_L_ij(q) = P(R_i < Q_i(q) | R_j < Q_j(q))
```

Use q = 5% and q = 10%.

4. **Co-loss frequency:** percentage of stress days where 2, 3, or 4 sleeves lose money together.
5. **Worst-N-day contribution:** for worst 20 book days and worst 20 book 5-day windows, decompose PnL by sleeve.
6. **Stress PCA:** first principal component variance share in normal windows vs stress windows.
7. **Crash beta regressions:** regress sleeve returns on SPY return, ΔVIX, VIX backwardation dummy, rates shock, DXY shock, and commodity shock during stress windows.
8. **Expected shortfall contribution:** contribution of each sleeve to 95% and 99% book ES.

### Thresholds

I would call it “one bet” if any two of the following are true:

| Metric | One-bet threshold |
|---|---:|
| Stress avg pairwise corr | >0.65 |
| Stress max pairwise corr among core sleeves | >0.80 |
| Lower-tail dependence at 5% | >0.35–0.40 |
| First PC variance share in stress | >70% |
| All active sleeves lose together in worst book windows | >50% of windows |
| VRP loses in worst book windows | >60% of windows |
| Book crisis drawdown vs trend-alone | worse than trend-alone in most crisis windows |

I would call it genuinely diversified only if:

- stress average correlation stays below ~0.45;
- at least one sleeve is flat or positive in most equity/VIX stress windows;
- expected shortfall contribution is not dominated by VRP or one asset class;
- book crisis drawdown is reliably smaller than trend-alone and futures-book-alone;
- the result survives cost and execution stress.

## C3. Stress-testing without a clean crisis fold

Do not rely on standard CPCV. Build crisis-specific tests.

### 1. Leave-one-crisis-out folds

Manually define crisis blocks and force each as a held-out fold:

```text
train / calibrate / choose rules excluding crisis K
run fixed rules through crisis K
record sleeve and book PnL, drawdown, gates, turnover, margin
repeat for each crisis
```

Even rules-based sleeves have implicit calibration: vol estimates, gates, thresholds, universe choices, and costs. Treat those as fixed before the held-out crisis.

### 2. Historical scenario replay

Take today’s proposed risk weights and replay historical joint returns through crisis windows.

Important: replay the **rules**, not static exposures, where possible. If the VIX gate would have turned off, let it turn off using only point-in-time data.

Report:

- max drawdown;
- worst 1-day / 5-day / 20-day loss;
- margin usage;
- sleeve contribution;
- turnover / roll stress;
- time to recovery;
- whether kill-switch thresholds would fire.

### 3. Crisis-weighted block bootstrap

Build synthetic 10-year paths using block resampling:

- normal 20–60 day blocks;
- crisis blocks sampled at 3×–5× their historical frequency;
- preserve cross-sleeve joint returns within blocks;
- include cost multiplier in crisis blocks, e.g. 2×–5× spread assumptions;
- include margin multiplier, e.g. 1.5×–2× exchange margin.

Decision outputs:

- probability of -10%, -15%, -20% drawdown at target vol;
- expected shortfall;
- probability of margin cap breach;
- average number of de-risk triggers;
- recovery time distribution.

### 4. Synthetic correlation shock

Force:

- all sleeve correlations to 0.8;
- vol to 2× and 3× normal;
- VRP gap loss equal to worst historical plus 50%;
- futures costs to 5× normal;
- margin to 2× normal;
- one broker unavailable for a day.

This is not a forecast. It is an operational survivability test.

If the book cannot survive that at proposed size, proposed size is too high.

## C4. Defensive / convex sleeve candidates

You do not own perfect convexity unless you trade options. With futures and ETFs, you can build **crisis alpha**, not pure convexity.

### Candidate 1 — Defensive trend basket

Use rates, gold, USD, and maybe equity-index short exposure if allowed.

Possible universe:

- Treasury futures / TLT / IEF;
- gold futures / GLD;
- USD index / UUP / FX futures;
- possibly equity-index futures short when equity trend is negative.

Rule:

- long assets with positive 6–12 month trend;
- increase weight when equity trend is negative and VIX trend is positive;
- cap bleed in normal regimes;
- no forced long duration if inflation shock regime says bonds are in downtrend.

This is the most realistic defensive sleeve from your data.

Proof requirement:

- positive average PnL in SPY worst 5% days;
- positive or flat average PnL in VIX top 5% changes;
- positive contribution in at least half of historical crisis windows;
- standalone Sharpe can be low, but book expected shortfall must improve materially;
- bleed in normal regimes must be less than 20–30% of book expected return.

### Candidate 2 — VIX trend / crisis-alpha overlay

Not “long vol when cheap” as a naive carry trade. Long VIX futures when VIX momentum and term-structure state imply crisis acceleration.

Example rule:

- long VX only when VIX > VIX3M or VIX 20-day momentum is strongly positive;
- flat otherwise;
- size small;
- treat it as insurance, not a Sharpe sleeve.

Proof requirement:

- makes money in Feb-2018, COVID, and other VIX shock windows;
- does not bleed more than the benefit it provides to 95% / 99% expected shortfall;
- improves book expected shortfall after including realistic roll bleed and slippage.

### Candidate 3 — Equity crisis trend / short equity index futures

If allowed by mandate, add short equity-index futures only when slow trend is negative and volatility regime confirms.

This may overlap with existing trend, but ETF trend is long-flat, not short. A small short-equity crisis sleeve could provide convexity-like behavior in prolonged bear markets.

Proof requirement:

- improves 2008 and 2022-style scenarios;
- does not whipsaw the book to death in sideways markets;
- has strict cap and slow signal to avoid overtrading.

### Candidate 4 — Defensive curve trades

Rates curve steepener / flattener trades can hedge some regimes, but they are not generic crisis hedges. They are macro bets. I would rank them below defensive trend and VIX crisis-alpha unless you can show clean tail contribution.

## C5. Does VRP belong?

My default answer:

> **VRP belongs only as a small, gated, explicitly capped return sleeve. It does not belong as an equal-risk core diversifier unless the tail tests prove it reduces or at least does not worsen expected shortfall.**

Conditions for VRP inclusion:

- contango gate uses only point-in-time data;
- survives independent stress replay;
- expected shortfall contribution is acceptable;
- does not lose in most worst-book windows;
- no single VX roll / gap can dominate monthly PnL;
- live paper confirms contract selection, roll, and gate behavior;
- allocation capped at 5–10% risk until a long live record exists.

Conditions for dropping VRP:

- stress correlation to trend or futures book >0.70;
- lower-tail dependence with trend >0.35;
- worsens 95% or 99% expected shortfall;
- crash gate exits too late in fast shocks;
- returns are concentrated in a few quiet regimes;
- live slippage / roll cost eats more than 30–40% of expected edge;
- strategy requires discretionary overrides to feel safe.

---

# Implementation checklist

## Build first: family-wise null zoo

Priority: highest.

Deliverables:

```text
null_zoo_config.yaml
candidate_family_registry.csv
research_selection_procedure.md
null_generator_rank.py
null_generator_shift.py
null_generator_block.py
run_null_zoo.py
spa_reality_check.py
null_zoo_report.md
```

Report must include:

- observed t;
- null max-stat percentiles;
- empirical p;
- SPA p;
- family false-positive rate;
- decision label.

## Build second: unified risk state

Priority: highest before live capital.

Deliverables:

```text
canonical_instrument_master
broker_snapshot_table
book_snapshot_table
risk_metric_table
risk_state_machine
global_kill_switch
out_of_band_flatten_script
reconciliation_report
```

## Build third: tail-risk dashboard

Deliverables:

```text
stress_conditional_corr_matrix
lower_tail_dependence_matrix
worst_N_day_PnL_decomp
stress_PCA_report
crisis_replay_report
crisis_weighted_bootstrap_report
VRP_expected_shortfall_contribution
```

## Build fourth: promotion ledger

Deliverables:

```text
experiment_registry
family_registry
alpha_budget_ledger
monthly_research_governance_report
```

---

# Monday morning risk committee decision

## Single thing I would do first

**Run the selection-aware null zoo / SPA test for the futures book.**

Everything else depends on whether t = 2.29 is real after the family-wise burden. If it fails, the capital architecture should be built for trend plus cash and futures paper monitoring. If it passes, the futures book earns a tiny-live path.

## Single thing I would refuse to let you do

**I would refuse to let you size carry+xsmom+VRP as a real multi-premia live book before the unified risk layer and tail test exist.**

That would be sizing a research result, not operating a portfolio.

## Top 5 moves in order

1. **Freeze research and run the null zoo.** No new sleeve ideas until the current candidates are selection-adjusted.
2. **Build the unified Alpaca + IBKR risk surface and global kill-switch.** No non-trivial futures capital before this exists.
3. **Run tail-correlation / expected-shortfall tests and crisis replay.** Decide whether VRP is capped, dropped, or paired with a defensive sleeve.
4. **Move IBKR futures to paper with production reconciliation.** Treat it as an execution system test, not a capital decision.
5. **Only then start tiny-live futures.** Use minimum contract size, strict margin caps, and process-based promotion.

---

# Bottom line

You are no longer in “toy app with one trend sleeve” territory. You have something that could become a modest, coherent, alternative-risk-premia book.

But the next failure mode is obvious:

> **You convince yourself the second engine is validated because t = 2.29 crossed 1.96, then you add a short-vol sleeve because the standalone Sharpe looks good, then you discover in the first real stress that the book was one correlated risk-on trade with two brokers and no central risk governor.**

Do not let that be the story.

The right path is slower but much more professional:

- deflate the evidence;
- cap the paper sleeves;
- prove the book does not co-crash;
- build broker-level reality reconciliation;
- tiny-live for execution truth;
- scale only on process fidelity, not recent returns.

If the futures book survives that, it deserves capital. If it does not, trend plus cash is not failure. It is discipline.
