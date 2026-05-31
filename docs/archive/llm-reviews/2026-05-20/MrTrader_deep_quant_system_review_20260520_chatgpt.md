# MrTrader Deep Quant System Review and Next-Step Roadmap

**Date:** 2026-05-20  
**Prepared for:** Min  
**Input reviewed:** `system_design_review_20260520.md`  
**Review posture:** Brutally honest quant/system-design review, focused on what must be true before paper results or live promotion can be trusted.

---

## 1. Executive verdict

My blunt read: MrTrader is architecturally stronger than most early retail algorithmic trading bots, but the strategy/research layer is not yet investment-grade. The PM/RM/Trader separation, Redis workflow, ProposalLog lineage, database-backed configuration, restart reconciliation, and order-ledger P&L design are all good signs. Those are the bones of a real system.

The weak point is not the agent architecture. The weak point is that the system is at risk of becoming a sophisticated execution shell around unproven signals. Right now, the most important work is not adding more agents, more LLM reasoning, or more strategy variants. The most important work is to make the research environment reliable enough that a green result actually means something.

The P0 backtest defects called out in the source prompt are severe: simulated entries still use previous close plus an offset instead of next-session tradable prices, and stop checks use daily close rather than intraday high/low. That alone means past walk-forward Sharpe and drawdown numbers should not be used for promotion decisions. Add survivorship bias and unvalidated factor IC, and the correct conclusion is: do not optimize or expand the strategy until the truth engine is fixed.

The long/short pivot is directionally right. It is much more sensible than forcing long-only momentum through a bear-market-inclusive window. But the long/short implementation needs hard portfolio-level controls. Net exposure is currently a target, not a hard invariant. That is dangerous. A long/short system without explicit net, gross, beta, sector, borrow, and liquidity controls can drift into an unintended directional book even while every individual trade passes risk validation.

The News Intelligence Service is useful, but its current policy layer is too crude for long/short. Negative news should block or reduce a long; negative news may validate a short. NIS should become a directional alignment and materiality filter, not a generic block engine.

My highest-conviction recommendation: freeze feature expansion for one phase and run a disciplined validation sprint. Fix backtest mechanics, run factor IC, run PEAD event studies, add net/gross exposure gates, correct NIS direction logic, and build a paper-vs-sim attribution dashboard. Only after that should you decide whether the current PEAD + quality-short strategy deserves more buildout.

---

## 2. What the system gets right

### 2.1 The agent separation is good

The PM/RM/Trader split is the right mental model. The PM should decide what is attractive, the RM should decide whether the trade is allowed, and the Trader should execute and manage order/position state. This is a good architecture because it prevents strategy code, risk rules, and broker mechanics from blending into one fragile monolith.

The source design also shows a strong operational spine:

- Proposal UUID lineage from PM proposal to risk decision to trade/order.
- Fail-fast RM validation with explicit rejection reasons.
- DB-backed runtime configuration.
- Startup reconciliation against Alpaca positions.
- PENDING_FILL persistence before order placement.
- Immutable order ledger used to recompute partial P&L after restarts.

That is much better than a toy bot that just calls a model and sends orders.

### 2.2 The L/S pivot is conceptually correct

The decision to move away from long-only cross-sectional ranking is sensible. If the model saw a bear-market-inclusive training window and the label was always framed as long-only forward return, it likely learned a noisy mixture of stock selection and market direction. Adding shorts gives the system a way to monetize negative earnings surprise, deteriorating fundamentals, and relative weakness rather than treating them as merely “avoid” signals.

However, long/short is not just “add SELL_SHORT.” A real L/S system needs portfolio construction discipline. The current design is moving in that direction but still needs hard constraints and attribution.

### 2.3 PEAD is a legitimate anomaly candidate

PEAD is not random folklore. It is one of the better-documented anomalies in equity markets. But implementation details matter enormously: announcement timing, gap behavior, liquidity, revisions, guidance, analyst reaction, and post-announcement drift decay can make the difference between a usable edge and a backtest artifact.

The current PEAD scorer is a reasonable first cut, but a threshold like EPS surprise > +5% or < -5% is not yet a strategy. It is a hypothesis generator. It needs event-study validation before it deserves capital.

### 2.4 The system is paying attention to production realities

The prompt shows real attention to risk, settlement, PDT, restarts, wash-sale awareness, symbol halts, stale quotes, and daily lockouts. That matters. A lot of trading systems fail not because the signal is bad, but because state, order, or broker assumptions are wrong.

MrTrader already has some of the right production instincts. The task now is to bring the research and portfolio construction layers up to the same standard.

---

## 3. Highest-risk issues that can invalidate results

### 3.1 Backtest truth is not reliable yet

The current walk-forward bugs are not cosmetic. They are foundational.

If entries use `prev_close * 1.001` rather than the next tradable open or a realistic intraday execution price, the simulation can buy prices that were never available. If stop-outs are checked against daily close instead of daily low/high or intraday bars, losing trades that would have stopped out may remain alive in the backtest. Together, these defects can materially overstate returns, understate drawdowns, and distort strategy ranking.

This must be fixed before any further HPO, model comparison, or live-promotion decision.

Recommended P0 simulation rules:

```text
Long entry:
  use next-session open, or next available intraday bar VWAP/close after signal timestamp

Short entry:
  use next-session open, subject to shortability/borrow assumptions

Long stop:
  triggered if low <= stop_price

Short stop:
  triggered if high >= stop_price

If both stop and target hit in the same bar:
  use conservative sequencing unless intraday data can determine order

Limit orders:
  separate fill probability from decision price; do not assume fill just because limit is touched
```

Minimum acceptance test: create tiny synthetic price paths where the correct fill/stop outcome is obvious, then assert the simulator handles them correctly for long, short, gap-up, gap-down, stop-first, target-first, and ambiguous same-bar cases.

### 3.2 Survivorship bias makes the research universe suspect

Using current S&P/Russell constituents for a historical test gives the model the benefit of hindsight. Dead, acquired, degraded, and removed names disappear. That generally makes historical performance look better than reality.

You do not necessarily need a perfect institutional survivorship-bias-free universe on day one, but you do need to know what question the backtest is answering. “How would this have performed on today’s surviving names?” is not the same as “How would this have traded historically?”

Recommended approach:

1. Add a clear `universe_mode` to every experiment: `current_constituents`, `point_in_time_constituents`, or `custom_liquid_survivors`.
2. Treat current-constituent backtests as research diagnostics only, not capital evidence.
3. Prioritize point-in-time membership or a stable liquid universe that is explicitly defined before the test period.
4. Report performance with and without survivorship correction once available.

### 3.3 Factor IC is not optional

The factor portfolio strategy lists attractive-sounding inputs, but the prompt states that `scripts/compute_factor_ic.py` has not been run. Until it is run, the factor weights are opinion, not evidence.

Run IC before using the factor score for live allocation. Do not only compute average IC. Break it down by:

- Month and quarter.
- Bull, bear, high-vol, and low-vol regimes.
- Sector.
- Large vs smaller market cap buckets.
- Long horizon and short horizon: 1d, 3d, 5d, 10d, 20d.
- Long-leg and short-leg separately.

Minimum useful outputs:

```text
mean IC
median IC
IC t-stat
hit rate: % periods IC > 0
rank autocorrelation
long-short decile spread
turnover
capacity estimate
performance after realistic transaction costs
```

If IC is weak, unstable, or concentrated in one regime, do not tune the model harder. Either simplify the strategy or use the factor only as a risk/quality overlay.

### 3.4 Position sizing language appears internally inconsistent

The prompt says per-position risk is 2% of NAV. But the proposal construction describes base size as:

```text
account_value * position_risk_pct * confidence_scalar
```

That reads like notional sizing, not risk-at-stop sizing. If `position_risk_pct = 2%` means trade notional, then with a 2% stop the actual capital at risk is only about 0.04% of NAV before multipliers. If `position_risk_pct = 2%` means risk budget, then quantity should be calculated from stop distance.

This is not a naming nit. It changes strategy behavior materially.

Recommended sizing formula:

```python
risk_dollars = nav * risk_budget_pct * confidence_mult * regime_mult * nis_mult
stop_distance = abs(entry_price - stop_price)
qty_by_risk = floor(risk_dollars / stop_distance)
qty_by_notional_cap = floor((nav * max_position_notional_pct) / entry_price)
qty_by_liquidity = floor((adv20_dollars * max_adtv_pct) / entry_price)
qty = min(qty_by_risk, qty_by_notional_cap, qty_by_liquidity)
```

Use two separate config names:

```text
risk.per_trade_risk_pct_of_nav
risk.max_position_notional_pct_of_nav
```

Do not call notional exposure “risk.”

### 3.5 The account size conflicts with intraday ambitions

A $20k account is below the standard $25k PDT threshold. The current design prudently blocks intraday entries before too many day trades, but this means the intraday strategy cannot really express itself in a real-money account of the same size.

That does not mean intraday research is useless. It means intraday should be treated as shadow mode or a future sleeve until the account size and PDT constraints support it. For now, the highest-value live path is likely swing/event-driven trading, not frequent intraday trading.

### 3.6 Paper trading is useful for software QA, not performance proof

Paper trading should be used to test state management, order lifecycle, logging, reconciliation, and whether the strategy behaves as designed. It should not be trusted as a close estimate of live P&L.

Paper environments commonly miss or simplify market impact, order queue position, latency slippage, real borrow fees, regulatory fees, and realistic locate availability. For MrTrader, that matters because the strategy now includes shorts, limit orders, and intraday logic.

Recommended reporting split:

```text
Paper broker P&L
Economic P&L after modeled fees/slippage/borrow
Expected sim P&L for the same trades
Paper-vs-sim shortfall
Rejected-trade opportunity cost
Missed-fill opportunity cost
```

Do not promote based on broker paper P&L alone.

---

## 4. Strategy review: PEAD + quality short

### 4.1 PEAD should be treated as an event strategy, not a generic swing strategy

A PEAD trade is not the same as a momentum trade. The reason for the trade is an earnings event. Therefore exits should be event-aware.

Recommended PEAD-specific exit logic:

```text
Hard max hold: 5 trading days after earnings event
Early exit: signal invalidated by adverse news/guidance/revision
Risk exit: stop loss hit
Profit management: optional partial after strong drift, but do not let trailing logic turn PEAD into a generic trend trade
No extension unless a new catalyst appears
```

I would not allow the PM to casually extend PEAD trades just because the chart still looks okay. If the event drift decays after a few days, holding longer can dilute the anomaly and convert the position into untested factor exposure.

### 4.2 PEAD event study should be the next research artifact

Before optimizing PEAD thresholds, run a clean event study.

Required slices:

- Positive EPS surprise vs negative EPS surprise.
- Surprise magnitude buckets: 5-10%, 10-20%, 20%+.
- Pre-market vs after-hours announcements.
- Gap direction and gap size.
- High volume vs normal volume.
- Guidance/revenue confirmation vs EPS-only surprise.
- Large cap vs mid/small cap.
- High short interest vs normal short interest for negative surprises.
- Long and short results separately.

Required return windows:

```text
announcement close to next open
next open to same-day close
+1 trading day
+3 trading days
+5 trading days
+10 trading days
```

The key question is not “does PEAD exist academically?” The key question is “does this data provider, universe, execution assumption, and signal timestamp produce tradable drift after costs?”

### 4.3 Quality short is dangerous unless squeeze risk is modeled

Quality short signals based on declining revenue, low/negative margins, and valuation deterioration can work, but they are also exactly the names that can rip higher on short covering, liquidity squeezes, takeover rumors, meme interest, or “less bad than feared” earnings.

For quality shorts, add these controls before giving the sleeve meaningful size:

- Borrow availability and borrow fee estimate.
- Short interest / days-to-cover if available.
- Gap risk filter around earnings and major events.
- Hard-to-borrow or low-float exclusion.
- Price/volume squeeze detector.
- Max single-name short loss tighter than long loss.
- Forced review if stock rallies above a moving average or breaks a recent high on volume.

A good short book is not just “bad companies.” It is bad companies where the timing, borrow, liquidity, and catalyst are favorable.

---

## 5. Long/short portfolio construction

### 5.1 Add explicit net and gross exposure controls immediately

Net exposure should be enforced in three places:

1. PM: target portfolio construction should aim for the desired net/gross profile.
2. RM: pre-trade gate should reject any trade that violates hard limits.
3. Trader/monitor: post-fill drift should trigger alerts or rebalancing if fills are asymmetric.

Recommended hard checks:

```python
proposed_sign = 1 if direction == "BUY" else -1
proposed_notional = qty * entry_price
new_net = current_net + proposed_sign * proposed_notional / nav
new_gross = current_gross + abs(proposed_notional) / nav

if new_gross > max_gross_exposure:
    reject("gross exposure limit")

if new_net > max_net_long:
    reject("net long exposure limit")

if new_net < min_net_long:
    reject("net exposure below target band")
```

Suggested initial values for a $20k paper system:

```text
max gross exposure: 80% to 100% until validated
net target: +40%
net band: +20% to +55%
max beta-adjusted net: +50%
max single-name notional: 5%
max single-name risk-at-stop: 0.25% to 0.50% during validation
```

The current 2% risk budget may be too high if it truly means risk-at-stop. For a developing system, start smaller and earn the right to size up.

### 5.2 Do not let net sector exposure hide gross sector crowding

The design says shorts reduce sector concentration and that the check uses absolute signed sector exposure. That is useful for net exposure but insufficient for crowding.

Example: long 20% software and short 20% software can show net zero sector exposure, but the book is still highly exposed to software-specific volatility, earnings cycles, factor rotations, and liquidity shocks. Net checks and gross checks should both exist.

Recommended sector checks:

```text
sector_net_abs_pct <= 20%
sector_gross_pct <= 35%
sector_short_gross_pct <= 20%
sector_long_gross_pct <= 25%
```

Same principle applies to beta and factor exposure. Net beta is useful, but gross risk can still be high.

### 5.3 Replace top-N with target portfolio optimization

Top-N ranking is a reasonable prototype. It is not ideal for a long/short book.

Move toward a simple optimizer that converts scores into target weights subject to constraints. This does not need to be fancy. A robust linear or quadratic optimizer is better than a clever ranking rule.

Inputs:

```text
expected return proxy = calibrated score or historical sleeve expectancy
risk = volatility, beta, sector, correlation
cost = spread + slippage + borrow + turnover
constraints = net, gross, sector, beta, position, liquidity, shortability
```

Objective:

```text
maximize expected_return - cost_penalty - risk_penalty - turnover_penalty
```

This gives the PM a target portfolio. The RM then checks whether moving from current holdings to the target portfolio is allowed.

### 5.4 Use sleeves/pods instead of one blended score

I would structure MrTrader as multiple strategy sleeves, each with its own evidence, constraints, and paper/live attribution.

Recommended initial sleeves:

1. PEAD long.
2. PEAD short.
3. Quality short.
4. Momentum/quality long.
5. Intraday shadow sleeve only.

Each sleeve should report:

```text
proposals
approved trades
filled trades
win rate
avg win / avg loss
expectancy
Sharpe
max drawdown
turnover
slippage
borrow/fee drag
capacity
regime sensitivity
```

Do not blend sleeves until you can see which one is actually contributing.

---

## 6. News Intelligence Service review

### 6.1 NIS must become direction-aware

The current concern in the source prompt is correct: a negative news signal should not automatically block a short. The logic should be alignment-based.

Recommended policy:

```python
proposed_side = 1 if proposal.direction == "BUY" else -1
news_side = sign(direction_score)  # + bullish, - bearish
alignment = proposed_side * direction_score

if confidence < min_confidence or materiality < min_materiality:
    policy = "ignore"
elif alignment <= -0.60 and downside_or_upside_risk_is_material:
    policy = "block_or_size_down"  # news contradicts trade
elif alignment >= +0.60:
    policy = "confirming"          # news supports trade
else:
    policy = "neutral"
```

For now, I would not allow NIS to increase size above the base strategy size. Let it block or reduce risk first. Only allow size-up after you have measured that NIS-confirmed trades outperform non-confirmed trades out of sample.

### 6.2 Separate news materiality from sentiment

LLM sentiment alone is too crude. A news article can be negative but immaterial, or positive but already priced in. The NIS output already includes materiality and already-priced-in scores. Use them more explicitly.

Recommended buckets:

```text
A. Fresh material contradiction -> block
B. Fresh material confirmation -> allow; maybe tag as confirmed
C. Stale / already priced in -> ignore
D. Ambiguous high-impact event -> size down
E. Legal/regulatory/accounting issue -> special risk rule
F. M&A rumor -> special no-short or no-trade rule unless strategy supports it
```

### 6.3 Build a NIS evaluation dataset

Every NIS decision should become training/evaluation data:

```text
proposal_uuid
symbol
direction
strategy sleeve
raw headline/article IDs
LLM model and prompt version
news direction_score
materiality
confidence
policy
base strategy score
actual forward returns: 1d, 3d, 5d
trade outcome if filled
counterfactual outcome if blocked
```

After 200-500 events, you can ask whether NIS is adding value or just filtering randomly. Until then, it should be treated as a conservative risk overlay, not alpha.

---

## 7. Risk manager review

### 7.1 The RM is good, but it needs portfolio invariant checks

The current RM is broad and thoughtfully designed, but several rules are still trade-level rather than portfolio-construction-level. Add invariant checks that must always be true before and after each trade.

Hard invariants:

```text
gross exposure <= cap
net exposure inside band
beta-adjusted exposure inside band
single-name notional <= cap
single-name risk-at-stop <= cap
sector gross and sector net <= caps
strategy sleeve budget <= cap
correlation cluster gross <= cap
cash/buying-power sufficient after open orders
borrow/shortability valid for shorts
```

The RM should not only ask: “Is this trade individually safe?” It should ask: “Does the resulting portfolio still match the intended book?”

### 7.2 Review buying power and short margin assumptions

The source prompt says SELL_SHORT requires 1.5x buying power. That may be a useful conservative approximation, but the live implementation should not hardcode margin math when the broker account response can provide actual buying power and asset-level shortability. Margin requirements vary by account, security, price, volatility, and broker policy.

Recommended implementation:

- Use Alpaca account buying power fields as the first source of truth.
- Confirm `asset.shortable`, `asset.easy_to_borrow` if available, and order rejection behavior.
- Add pre-trade and post-rejection tracking for short locate failures or broker-side short rejections.
- Maintain an internal conservative model for simulation and risk forecasting, but reconcile it against actual broker responses.

### 7.3 PDT handling should shape the strategy roadmap

The PDT rule is not just a compliance check. It is a strategy design constraint. If live capital remains near $20k, the intraday system should not be the primary live sleeve. It can remain in shadow mode or paper-only until the account can support it.

Recommended stance:

```text
Swing/event-driven sleeve: primary candidate for live readiness
Intraday sleeve: shadow-only until PDT and execution realism are solved
```

### 7.4 Portfolio heat should use real stop distance and gap assumptions

Portfolio heat based on stop distance is good, but gap risk can exceed stop risk. This is especially true for shorts, earnings names, and overnight holds.

Add a stress heat calculation:

```text
normal_heat = abs(entry - stop) * qty / nav
stress_heat = stress_gap_pct(symbol, strategy, event_risk) * notional / nav
portfolio_stress_heat = sum(stress_heat across positions)
```

For PEAD and quality shorts, use larger stress gaps than for liquid long-only momentum names.

---

## 8. Trader/execution review

### 8.1 Good: persistence and reconciliation are treated seriously

The Trader design has several production-grade elements:

- PENDING_FILL is written before order placement.
- Startup reconciliation handles Alpaca positions not matching DB.
- Partial exits are represented in the Order table.
- P&L can be recomputed from filled order records.

This is exactly the kind of detail that prevents silent account drift.

### 8.2 Add an explicit order state machine

The next step is to formalize order state transitions. Every order should move through a finite-state machine, not ad hoc flags.

Recommended states:

```text
INTENDED
SUBMITTED
ACKNOWLEDGED
PARTIALLY_FILLED
FILLED
CANCEL_REQUESTED
CANCELLED
REJECTED
EXPIRED
RECONCILED_EXTERNAL
UNKNOWN_NEEDS_MANUAL_REVIEW
```

Every state transition should be idempotent and tied to broker order ID, client order ID, proposal UUID, and trade ID.

### 8.3 Limit-order simulation needs to be conservative

For swing entries, the design uses limit orders. That is reasonable, but the simulator must not assume fills too generously. If a limit order is touched, that does not mean your order got filled, especially if queue position matters.

Recommended fill model levels:

```text
Level 0: conservative touch model: fill only if price moves through limit by buffer
Level 1: volume-aware model: fill only up to X% of bar volume
Level 2: queue model: probabilistic fill based on spread, volume, and bar path
```

Use Level 0 first. It will be pessimistic, but pessimistic is better than self-deception.

### 8.4 Track paper-vs-sim shortfall by cause

Create a daily execution attribution table:

```text
expected_entry_price
actual_fill_price
expected_exit_price
actual_exit_price
spread_at_entry
spread_at_exit
slippage_bps
missed_fill_return
partial_fill_impact
latency_seconds
broker_reject_reason
borrow_or_short_reject
```

This will tell you whether the problem is alpha, risk filtering, or execution.

---

## 9. Data architecture review

### 9.1 Data feed consistency matters

The design uses Alpaca daily/intraday bars, FMP earnings/fundamentals, Finnhub news/economic calendar, and Polygon news. That is workable, but it introduces timestamp, coverage, and survivorship issues.

Add a data provenance record to every feature row:

```text
source_provider
source_endpoint
source_timestamp
as_of_timestamp
retrieved_at
adjustment_status
split_adjusted_flag
dividend_adjusted_flag
symbol_mapping_version
```

### 9.2 IEX vs SIP can matter for intraday and spread logic

If the system is relying on IEX-only data for quotes or historical bars, it can misread liquidity, volume, and spreads versus consolidated market data. This is especially important for intraday scans, spread gates, stop/target testing, and limit-order fill assumptions.

Recommendation: for any serious intraday validation, use SIP or another consolidated data source consistently across research and live monitoring. If you cannot, keep intraday in shadow mode.

### 9.3 Earnings timestamp precision is critical

PEAD is timestamp-sensitive. The system must distinguish:

```text
BMO: before market open
AMC: after market close
DMH: during market hours
unknown timestamp
```

A signal generated at 08:00 using an earnings result that was actually released after the prior close is different from a signal generated before the release. Every PEAD feature should include the exact event timestamp or at least release bucket.

---

## 10. Alternative methodologies worth considering

### 10.1 Keep PEAD, but make it more institutional

Enhance PEAD with:

- Revenue surprise.
- Guidance raise/cut.
- Analyst revision drift after earnings.
- Earnings call sentiment, if available and properly timestamped.
- Post-earnings volume shock.
- Gap-and-go vs gap-fade classification.
- Sector-relative surprise.

The best version is not “EPS surprise > 5%.” It is “which earnings surprises continue drifting after the tradable entry point?”

### 10.2 Add analyst revision momentum

Post-earnings drift often interacts with estimate revisions. A company that beats and then sees forward estimates revised upward may have a stronger drift profile than a company that beats on one-time items.

This could become a separate sleeve or a PEAD confirmation feature.

### 10.3 Consider market-neutral pairs or sector-relative trades

For a small account, full market-neutral implementation is hard, but research-wise it can be valuable. Instead of shorting bad names outright, pair them against sector ETFs or stronger peers.

Example:

```text
Long high-quality positive PEAD stock
Short sector ETF or weak peer
```

This can reduce market beta and isolate relative alpha, though it increases complexity and borrow/short execution needs.

### 10.4 Intraday should be regime-specific, not always-on

Intraday edge is hard. A generic intraday model across Russell 1000 with three scan windows may be too broad. Focus intraday on specific setups:

- Post-news continuation after confirmed volume expansion.
- Opening range breakout after catalyst.
- Mean reversion after extreme gap with failed continuation.
- VWAP reclaim/failure with liquidity filters.

Run these as explicit playbooks rather than one blended intraday score.

### 10.5 Avoid reinforcement learning and highly complex black-box trading for now

Do not add RL, deep sequence models, options, or autonomous LLM trade generation yet. Those are tempting, but they will make the system less explainable before the basics are validated.

The better “outside the box” move is not a more exotic model. It is a cleaner research factory with shadow tournaments across simple, well-specified strategy sleeves.

---

## 11. Revised promotion criteria

The current paper targets include 3 calendar months, annualized Sharpe >= 0.50, max drawdown <= 15%, max single position <= 8%, and live-vs-sim shortfall < 30 bps/day median.

I would make these stricter and more diagnostic.

### 11.1 Before any live money

Required:

```text
P0 simulator bugs fixed and unit-tested
survivorship mode explicitly labeled in every report
factor IC report completed
PEAD event study completed
net/gross exposure gates implemented
NIS direction logic fixed
paper-vs-sim attribution dashboard working
kill switches tested
broker/DB reconciliation tested under simulated failure
```

### 11.2 Paper trading pass criteria

Use trading days, not calendar months.

```text
Minimum: 60 trading days of paper/shadow evidence
Preferred: 90+ trading days
No unresolved state reconciliation incidents
No unexplained broker/DB mismatches
Economic P&L positive after modeled costs
Sharpe useful but not sufficient
Max drawdown within expected sim distribution
Per-sleeve attribution available
Shortfall vs simulation explained
```

A 3-month Sharpe of 0.50 alone is weak evidence. A low Sharpe can be noise, and a high Sharpe over 3 months can also be noise. What matters is whether the realized behavior matches the tested hypothesis.

### 11.3 Live pilot criteria

If you eventually go live, start tiny and treat it as an execution test, not a profit campaign.

```text
position risk-at-stop: 0.10% to 0.25% NAV
max gross exposure: 30% to 50% initially
intraday disabled or shadow-only under $25k
shorts limited to easy-to-borrow/liquid names
daily max loss tighter than paper target
manual review required after any unexplained order rejection
```

---

## 12. Prioritized implementation backlog

### P0 - Stop trusting bad research outputs

1. Fix next-session entry simulation.
2. Fix stop/target simulation using high/low/intraday path.
3. Add conservative same-bar ambiguity handling.
4. Add simulation unit tests for long/short/gap/stop/target cases.
5. Add survivorship mode to every experiment output.
6. Run factor IC by horizon, sleeve, sector, and regime.
7. Run PEAD event study by surprise bucket and announcement timing.
8. Add transaction cost, spread, slippage, and borrow-cost model.
9. Add experiment registry with config hash, data version, code commit, and output path.

### P1 - Make long/short a real portfolio

10. Add RM hard net exposure gate.
11. Add RM hard gross exposure gate.
12. Add beta-adjusted net exposure gate.
13. Add sector gross and sector net exposure checks.
14. Split long and short sleeve budgets.
15. Replace top-N construction with target portfolio optimizer.
16. Add post-fill exposure drift monitor.
17. Add hedge/rebalance alert if long and short fills are asymmetric.

### P1 - Fix NIS for L/S

18. Replace generic `block_entry` with direction-aware alignment policy.
19. Prevent NIS size-up until NIS value is proven out of sample.
20. Log NIS prompt version, model version, article IDs, and forward returns.
21. Build NIS confusion matrix and outcome attribution.
22. Add special handling for M&A, legal/regulatory, accounting, guidance, and downgrade/upgrade events.

### P2 - Execution realism and broker controls

23. Formalize order state machine.
24. Add client order ID idempotency checks.
25. Add shortability/borrow validation before short orders.
26. Add broker rejection analytics.
27. Add paper-vs-sim shortfall dashboard.
28. Validate IEX vs SIP differences for symbols traded.
29. Add stale data and provider outage kill switches.
30. Add stress heat and overnight gap-risk calculations.

### P3 - Strategy expansion only after P0/P1 pass

31. Add analyst revision momentum sleeve.
32. Add sector-relative PEAD sleeve.
33. Add intraday shadow playbooks instead of broad always-on intraday model.
34. Add model tournament where non-live selectors generate shadow proposals.
35. Add capital allocation by sleeve only after statistically meaningful evidence.

---

## 13. Direct answers to the prompt's open questions

### 13.1 Should NIS distinguish adverse vs confirming news by direction?

Yes. This is mandatory. A negative direction score is adverse to a long but confirming for a short. The correct abstraction is alignment, not sentiment.

### 13.2 Where should net exposure be enforced?

Both PM and RM. PM should construct a target portfolio inside the desired exposure band. RM should enforce hard pre-trade limits. Trader/monitor should detect post-fill drift when some orders fill and others do not.

### 13.3 Should PEAD have different exit logic?

Yes. PEAD should have event-specific exits and a hard time decay. I would use a 5-trading-day hard max unless the event study proves a better horizon.

### 13.4 Should live/paper track borrow cost?

Yes. Track two P&Ls: broker-reported P&L and economic P&L after modeled borrow/fees/slippage. For shorts, borrow and locate assumptions can decide whether an apparent edge survives.

### 13.5 Should current WF Sharpe be trusted?

No. Treat it as directional only until the P0 simulation bugs are fixed and survivorship/IC issues are addressed.

### 13.6 Should factor IC be run before using the factor portfolio?

Yes. Without IC and decile spread evidence, factor weights are just plausible stories.

---

## 14. Recommended near-term plan

### Week 1: Research truth gate

- Fix backtest entry and stop mechanics.
- Add simulation unit tests.
- Add data/version metadata to experiment reports.
- Run a small known-symbol PEAD event sanity check.

### Week 2: Signal evidence

- Run PEAD event study.
- Run factor IC.
- Split long and short results.
- Report by regime and sector.

### Week 3: Portfolio risk controls

- Add net/gross exposure gates.
- Add sector gross/net checks.
- Add beta-adjusted exposure checks.
- Fix position-sizing naming and formula.

### Week 4: Paper attribution

- Fix NIS direction alignment.
- Add paper-vs-sim attribution.
- Add borrow/slippage economic P&L.
- Keep intraday shadow-only unless account/PDT constraints change.

After those four weeks, decide whether to scale the current PEAD + quality-short strategy, simplify it, or pivot to a different sleeve.

---

## 15. What I would not do next

Do not add more LLM autonomy yet.  
Do not optimize XGBoost/HPO on flawed simulator outputs.  
Do not promote live based on current paper results.  
Do not expand to more symbols until data and fill assumptions are clean.  
Do not rely on broker paper P&L as proof of edge.  
Do not let long and short sector exposures cancel without also tracking gross exposure.  
Do not let NIS increase position size until it has proven incremental value.  
Do not run intraday live under a $25k account unless the strategy is specifically designed around PDT constraints.

---

## 16. LLM handoff prompt you can use with other reviewers

Use the following prompt with another LLM:

```text
You are reviewing a Python-based algorithmic trading system called MrTrader. It uses a PM/RM/Trader agent pipeline, Alpaca paper account, PEAD + quality-short long/short strategy, NIS news overlay, and DB-backed risk/execution state. Please review the attached system design and the accompanying critique. Act as a world-class quant and trading-system architect. Be brutally honest. Focus on whether the next-step roadmap is correct, what risks are missing, and what should be prioritized before live trading. Pay special attention to backtest validity, long/short portfolio construction, PEAD event-study design, NIS direction alignment, execution realism, paper-vs-live shortfall, PDT constraints, short borrow costs, and position sizing correctness. Provide a ranked set of changes and explicitly state what you agree or disagree with.
```

---

## 17. Bottom line

MrTrader should continue, but the next phase should be a validation-and-risk phase, not a feature phase. The system has a strong operational foundation. The danger is that the operational foundation creates confidence before the alpha evidence deserves it.

The best path is:

```text
Fix simulator truth -> validate signals -> enforce portfolio invariants -> measure paper-vs-sim -> only then scale complexity
```

If the PEAD + quality-short strategy survives that process, it becomes a credible first live sleeve. If it does not, the same research and execution factory will still be valuable because it will let you test the next strategy honestly.

---

## 18. External references consulted

- FINRA, “Day Trading”: Pattern day traders must maintain minimum equity of $25,000 in a margin account on any day the customer day trades.
- SEC, “Implementation of T+1 Settlement Cycle”: U.S. standard settlement cycle moved from T+2 to T+1 effective May 28, 2024.
- Alpaca Docs, “Paper Trading”: Paper trading is a simulation; paper-only accounts are entitled to IEX market data; paper does not account for market impact, information leakage, latency slippage, order queue position, price improvement, regulatory fees, or dividends.
- Alpaca Docs, “Margin and Short Selling”: Alpaca requires at least $2,000 equity for margin/short selling, describes Reg T margin and PDT buying power, and applies minimum initial margin requirements for marginable securities.
- Alpaca Docs, “Market Data FAQ”: Free market data uses IEX; SIP data requires the relevant subscription.
- Fink, Josef. “A review of the Post-Earnings-Announcement Drift,” Journal of Behavioral and Experimental Finance, 2021.
