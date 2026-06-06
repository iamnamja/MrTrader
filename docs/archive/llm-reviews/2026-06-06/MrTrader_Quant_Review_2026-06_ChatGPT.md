# MrTrader Automated Trading System - Brutally Honest Quant Review

**Date:** 2026-06-06  
**Prepared for:** Min / MrTrader  
**Input reviewed:** `QUANT_REVIEW_PROMPT_2026-06.md`  
**Reviewer stance:** skeptical systematic PM / quant researcher. The goal is not to make the system sound good. The goal is to avoid funding noise.

---

## 0. Bottom line

You are ahead of most solo retail algo projects in engineering discipline, auditability, and willingness to kill ideas. That is the good news. The bad news is that the current alpha book is not yet investable. It is a research platform with one weak live candidate, not a trading business.

My blunt verdict:

1. **The architecture is stronger than the alpha.** PM/RM/Trader separation, decision audit, counterfactuals, risk gates, slippage capture, and rollback discipline are real assets. Keep them.
2. **The generic cross-sectional equity ML program should remain dead.** Not because ML is useless, but because "daily Russell-1000 technical/fundamental features -> XGBoost ranker -> swing returns" is exactly where thousands of better-resourced teams and ETFs already compete. Your failed OOS IC and beta contamination are not bugs; they are what I would expect.
3. **The 52% CPCV fold-skip issue is severe.** It does not merely reduce confidence. It breaks the meaning of the absolute CPCV performance distribution because the missingness is regime-dependent. You can use the results as exploratory ranking clues only after proving each strategy has similar missingness exposure. Otherwise even relative comparisons are suspect.
4. **Current PEAD is not alpha yet.** With event-clustered bootstrap p around 0.19, HAC t around 1.04, uptrend dependence, and 87% of P&L in uptrends, it is a long-biased earnings drift/risk-on timing sleeve. It is rational to paper trade it as telemetry. It is not rational to ramp it as if it is proven.
5. **Your next edge is more likely to come from cleaner event definitions and portfolio construction than from a fancier model.** You need to move from "positive EPS surprise" to "genuine expectation shock plus post-event underreaction, conditional on attention, prior sentiment, guidance/revisions, price reaction, and crowding."
6. **You need a crisis-diversifying sleeve.** The cleanest one for your constraints is not single-name shorting or intraday ML. It is a small, liquid ETF/cross-asset time-series momentum / defensive trend sleeve that can be tested with daily bars and traded cheaply.
7. **Options are worth exploring only after you build a separate options simulator.** Alpaca and Polygon/Massive now make retail API options execution/data plausible, but a naive equity-bar simulator is useless for options. Do not bolt options onto the current daily equity engine and trust the output.

**Highest-EV ranked plan:**

| Rank | Initiative | Why it is high EV | Expected payoff | Main kill condition |
|---:|---|---|---|---|
| 1 | Fix validation and research governance | Without this, every next strategy can fool you | Prevents capital allocation to false positives | Cannot reproduce unbiased folds / event samples |
| 2 | Build PEAD 2.0: genuine surprise drift | Closest to current survivor; uses data you can get | Moderate alpha potential; natural extension | No alpha after factor/sector/beta neutralization and event bootstrap |
| 3 | Add ETF/cross-asset trend/crisis sleeve | Diversifies PEAD and is testable on daily bars | Lower Sharpe but high robustness and portfolio value | No robust drawdown reduction or net performance after costs |
| 4 | Explore defined-risk post-earnings options vol | Uses retail-accessible options data/execution; different return source | Potentially diversifying, but needs new stack | No edge after bid/ask, IV surface, assignment, and realistic spreads |
| 5 | Small event squeeze sleeve: high SI + positive catalyst | Cheap data, asymmetric behavior post-meme era | Opportunistic small sleeve | Too few events; returns vanish after gap/slippage |

---

## 1. What you got right

### 1.1 The PM/RM/Trader split is the correct architecture

The three-agent design is a good separation of concerns:

- **PM:** generates alpha proposals.
- **RM:** vetoes exposure and portfolio state.
- **Trader:** handles execution and exits.

This is the right architecture because it prevents the alpha layer from self-approving its own trades. A lot of failed trading systems hide risk decisions inside strategy code. You are not doing that. Keep this split.

What I would add: every strategy proposal should carry a structured **strategy contract**:

```yaml
strategy_id: pead_v2_revision_inconsistent_v1
hypothesis_id: HYP-2026-PEAD-003
signal_timestamp_utc: ...
known_at_timestamp_utc: ...
intended_entry_window: next_regular_session_open_plus_5m_to_30m
expected_holding_period_days: 10-30
primary_risk: earnings_gap_reversal
primary_factor_exposures_expected: beta, sector, momentum
kill_criteria_version: ...
research_freeze_hash: ...
```

The purpose is to make the trade auditable back to a frozen hypothesis, not just a selector name.

### 1.2 You are unusually honest about dead strategies

Most small algo projects never kill bad models; they tune until the backtest looks nice. You killed:

- the generic XGBoost/LambdaRank swing model,
- the 5-minute intraday model,
- the dollar-neutral high-breadth L/S ranker,
- the analyst rating drift artifact,
- the short-interest factor,
- small/mid-cap PEAD.

That discipline is valuable. The pattern is also informative: anything that looked like generic cross-sectional equity ranking collapsed to noise or beta. That is exactly the right lesson to internalize.

### 1.3 The decision audit and counterfactual logging are real assets

The `decision_audit` table, gated-trade counterfactuals, slippage logging, and PEAD-specific tracker are not "nice to have." They are the foundation of live learning. Many institutional teams would benefit from cleaner counterfactual capture than they actually have.

However, you need to tighten the distinction between:

- **model backtest alpha,**
- **paper-trading operational telemetry,** and
- **real-money executable alpha.**

Paper trading can tell you whether the system runs, whether gates behave, whether signal frequency is as expected, and whether proposed orders would have occurred. It cannot prove you can monetize the edge, especially around earnings gaps and options spreads.

---

## 2. The uncomfortable truth about the validation stack

### 2.1 CPCV is directionally right, but your implementation currently invalidates absolute conclusions

Using purged/embargoed walk-forward validation, per-fold retraining, N_eff based on fold count instead of correlated paths, deflated Sharpe, deployment-adjusted Sharpe, and event-clustered bootstrap is the right direction.

But the 52% fold-evaluation skip rate is not a cosmetic flaw. It means the distribution you are scoring is not the distribution you intended to sample. If the skipped folds are systematically older, bearish, volatile, lower-liquidity, or otherwise different, then your CPCV estimate is a conditional estimate on a friendlier subset of history.

**Blunt conclusion:** any strategy whose pass/fail depends on the current CPCV Sharpe should be treated as unproven.

You already suspect this. I agree with the skeptical interpretation. The fold skip is especially damaging because the surviving sample is biased toward recent bull regimes, and your one live candidate is explicitly uptrend-dependent.

### 2.2 Relative comparisons are not automatically safe

You wrote that you lean on relative comparisons on the same biased sample. That is only valid if every strategy has the same sensitivity to the missing folds. They probably do not.

Examples:

- PEAD long-only benefits from bull/uptrend survival.
- Short-interest reversal/short-squeeze behavior changes drastically between pre-2020 and post-meme eras.
- Analyst drift may look better in low-volatility, earnings-friendly windows.
- Dollar-neutral L/S may be less sensitive to market beta but more sensitive to dispersion and financing.

So the bias can change rankings, not just inflate levels.

### 2.3 Fix fold construction before testing anything else

Do not add new strategies until this is fixed. Suggested fix:

1. Define the maximum label horizon, maximum holding period, and maximum feature lookback for each strategy family.
2. Build fold groups such that every train/test split can be purged without dropping the test fold.
3. If CPCV coverage is impossible with the current k=8/choose-2 structure, reduce k or change the scheme. Full coverage with fewer folds is better than a sophisticated CV design that drops half the regimes.
4. Publish a **fold coverage report** with every backtest:
   - included days/events by year,
   - included days/events by VIX quintile,
   - included days/events by SPY trend state,
   - included days/events by bear/bull/crash regime,
   - skipped share by strategy.
5. Require the fold coverage report to pass before looking at performance.

A strategy should not be allowed to show a Sharpe until the sample map is clean.

### 2.4 Promotion gates are reasonable but overfit to metrics

Your current gates are sensible in spirit:

- average Sharpe,
- fold floor,
- DSR,
- profit factor,
- Calmar,
- worst-regime floor.

But the exact thresholds can become another optimizer. The more gates you have, the easier it is to reject obviously bad ideas, but also the easier it is to tune strategy definitions around the gates.

I would simplify promotion into three stages:

**Stage A - Research pass:**

- Positive net return after realistic costs.
- Positive factor-adjusted alpha with HAC t-stat.
- No single year/event cluster contributes more than a defined share of total alpha.
- Event/bootstrap significance passes a pre-declared threshold.
- Drawdown and turnover are compatible with the account.

**Stage B - Shadow/live telemetry pass:**

- Signal frequency matches backtest.
- Slippage and missed-fill rates are inside model assumptions.
- Gated-vs-entered counterfactuals do not show that gates are accidentally selecting losers/winners by leakage.
- Ops failure rate is low.

**Stage C - Capital pass:**

- Small real-money record survives pre-declared number of independent trades/events.
- Performance is evaluated against a frozen baseline and kill criteria.

### 2.5 Add a research ledger, not just a model registry

You need a table that records every research attempt, including dead ones. Not just outputs - the hypothesis before testing.

Minimum schema:

| Field | Purpose |
|---|---|
| hypothesis_id | Immutable ID |
| economic rationale | Why this should exist |
| data available_at timestamps | PIT sanity |
| first test date | Controls multiple testing |
| strategy family | For false-discovery grouping |
| variations tried | Number of researcher degrees of freedom |
| final frozen spec | What was actually tested |
| result | Pass/fail |
| kill reason | Why abandoned |
| eligible for retest? | Only if new data or regime |

Without this, even a careful process can become slow-motion data snooping.

---

## 3. PEAD triage: should you run it?

### 3.1 Current PEAD is a weak candidate, not a proven strategy

Current facts:

- Long positive EPS surprise.
- Enter next open.
- Hold weeks.
- Large-cap universe.
- CPCV mean Sharpe around 0.55 to 0.66 depending on filter.
- Event-clustered bootstrap p around 0.19.
- HAC t around 1.04.
- Around 87% of P&L in uptrends.
- No real live track record.

This is not enough to call it alpha. At best it is a candidate with a plausible economic story and weak statistical support.

Would I paper trade it? **Yes, but only as a telemetry experiment.**

Would I aggressively ramp it? **No.**

Would I allow small real-money execution eventually? **Yes, but only under a strict experimental budget, because paper fills around post-earnings gaps are not enough.**

### 3.2 Paper trading alone will not prove PEAD

Paper trading is useful for:

- signal count,
- production stability,
- logging completeness,
- proposed vs approved vs filled workflow,
- approximate timing,
- counterfactuals.

Paper trading is weak for:

- true opening liquidity,
- spread crossing,
- gap continuation/reversal fill quality,
- partial fills,
- behavioral slippage during volatile earnings names,
- short availability, if you later add shorts,
- options fills, if you later add options.

For an earnings drift strategy, the entry point matters. If the backtest assumes next open, but the live system enters at 9:50 after volatility settles, you must explicitly measure the cost or benefit of that timing shift.

### 3.3 How I would prove or kill PEAD

Create a frozen PEAD test plan before another tuning pass.

**Current PEAD should graduate only if all of the following hold:**

1. Unbiased fold rebuild still shows positive net returns.
2. Event-clustered bootstrap improves materially, ideally p < 0.05 for the frozen rule or p < 0.10 if treated as a small sleeve with strong economic prior.
3. Factor regression alpha is positive after market, sector, size, momentum, quality/profitability, and low-vol controls.
4. Uptrend dependency is acknowledged and sized as a risk-on sleeve, not sold internally as standalone alpha.
5. Entry timing sensitivity is stable: next open, open+5m, open+20m, VWAP first 30m, and previous after-hours reaction do not flip results.
6. No single earnings season, sector, mega-cap cluster, or year drives more than 25-30% of net alpha.
7. Slippage from paper/real micro-size is within the modeled budget.

**Kill or freeze current PEAD if:**

- the unbiased fold rebuild drops Sharpe below roughly 0.3 or alpha t-stat remains near 1;
- factor-adjusted alpha is indistinguishable from zero;
- the top decile of signal strength does not outperform weaker signals;
- returns are mostly explained by market beta plus buying stocks that gap up in bull markets;
- post-entry alpha disappears after the first 1-2 days, meaning you are just chasing gap momentum;
- live slippage exceeds backtest assumptions by more than 5-10 bps in normal conditions;
- the strategy cannot survive a simple hedge without losing all economics.

### 3.4 PEAD 2.0: make the event more selective

Vanilla EPS surprise is overused. A modern PEAD sleeve must define a **genuine expectation shock**, not just a positive surprise against stale/managed consensus.

I would add the following filters/features:

| Dimension | Feature idea | Why it matters |
|---|---|---|
| Surprise quality | EPS surprise AND revenue surprise same direction | Earnings beats with revenue confirmation are cleaner than margin/accounting beats |
| Expectations | Positive surprise despite negative/neutral analyst recommendation | More likely to represent genuine belief revision |
| Revisions | Pre-event estimate revision trend and post-event revision acceleration | Drift often follows analyst underreaction |
| Guidance | Raised/lowered guidance, qualitative guidance tone | Guidance can dominate historical EPS surprise |
| Price reaction | Event-day abnormal return vs implied/expected move | Separates stale consensus beat from true market shock |
| Attention | Smaller analyst coverage, high dispersion, lower media attention | More room for delayed incorporation |
| Crowding | Short interest / days-to-cover / options OI skew | Positive shock plus crowded short can create continuation |
| Quality | Accruals, gross profitability, balance sheet stress | Avoid low-quality one-off beats |
| Sector | Sector-specific post-earnings behavior | PEAD drivers differ across sectors |
| Regime | SPY trend, VIX, dispersion, breadth | Makes risk-on dependency explicit |

The rule should become narrower and more economically coherent, even if it trades less.

---

## 4. Where the alpha actually is for your situation

Given your constraints - solo operator, approximately $100k paper account, retail fills, daily bars, no prime broker, no cheap leverage, no securities-lending revenue - you should stop trying to compete with institutional stat-arb on generic single-stock ranking.

You want:

- sparse but cleaner events,
- liquid instruments,
- low operational complexity,
- cheap data,
- low turnover,
- explicit risk overlays,
- strategies that are not all long-beta earnings drift.

### Return source 1 - PEAD 2.0 / genuine expectation shock drift

**Expected value rank:** 1 among alpha sources  
**Use current simulator?** Yes, with upgrades  
**Correlation to current PEAD:** moderate/high, but cleaner if properly filtered  
**Correlation to crisis sleeve:** likely low to negative during risk-off because it should mostly shut down  
**Capacity:** high enough for your capital in Russell 1000; possibly lower in small/mid caps  
**Complexity:** moderate

**Core hypothesis**  
Markets underreact not to any EPS beat, but to earnings information that forces investors and analysts to revise a prior belief. The best candidates are events where the surprise contradicts prior sentiment, is confirmed by revenue/guidance, and is followed by revisions or continued price acceptance.

**Signal sketch**

Trade long when all or most conditions hold:

1. Positive EPS surprise above threshold.
2. Revenue surprise positive or revenue growth acceleration positive.
3. Guidance raised or management tone positive, if available.
4. Pre-event analyst revisions were flat/negative, or recommendation was neutral/negative.
5. Event-day return is positive but not an exhaustion gap.
6. Abnormal volume confirms institutional attention.
7. Spread/liquidity acceptable.
8. Market regime is not risk-off, or position is beta-hedged.

Optional short version: negative EPS/revenue/guidance shock where prior sentiment was positive and price reaction confirms. Be careful: shorting single-name post-earnings losers with retail borrow constraints is harder than long-side PEAD.

**Data needed**

- Existing FMP earnings surprise and fundamentals.
- Estimate revision history, not just point-in-time actual vs estimate.
- Revenue surprise.
- Guidance changes, if available cheaply.
- Analyst recommendation level and revision history.
- Earnings announcement timestamp: before open vs after close matters.
- Intraday/after-hours reaction or at least open-to-close event-day path.
- Sector/factor returns.

**Validation**

- Event-time analysis by event, not day.
- Cluster bootstrap by earnings date and company.
- Purged event CV by calendar quarter/year.
- Factor residual alpha using market, sector, size, momentum, value/quality/low-vol proxies.
- Entry sensitivity tests: next open, open+5m, open+30m, close of event day, next close.
- Placebo tests: randomly shifted earnings dates within same quarter and sector.
- Decile monotonicity: stronger genuine-shock score should outperform weaker score.

**Go/no-go gate**

Promote only if the frozen PEAD 2.0 rule beats current PEAD on:

- factor-adjusted alpha,
- event bootstrap p-value,
- drawdown,
- entry sensitivity,
- live/paper slippage,
- and concentration.

Do not promote just because raw Sharpe is higher.

### Return source 2 - ETF/cross-asset time-series momentum and crisis-defense sleeve

**Expected value rank:** 2  
**Use current simulator?** Yes, daily bars are enough for a first version  
**Correlation to PEAD:** low in normal times, valuable in crises  
**Capacity:** extremely high for your size  
**Complexity:** low to moderate  
**This is more portfolio engineering than secret alpha, but that is fine.**

**Core hypothesis**  
Liquid asset classes exhibit medium-term trend persistence and crisis regimes where defensive assets outperform equities. You do not need single-name alpha to improve the total system if this sleeve reduces drawdowns when PEAD is off or vulnerable.

**Instrument set**

Start with ETFs, not futures:

- Equity: SPY, QQQ, IWM, sector ETFs.
- Bonds/rates: TLT, IEF, SHY.
- Gold/commodities: GLD, DBC or more liquid commodity ETFs if acceptable.
- USD: UUP, if useful.
- Volatility proxy: VIX only as signal; avoid trading VIX ETPs initially.
- Crypto ETF exposure only later, if liquidity and data are clean.

Later, if this sleeve matters, test micro futures. But do not start there.

**Signal sketch**

For each asset:

- Compute trend score across 1, 3, 6, and 12 month lookbacks, skipping the most recent few days if needed.
- Position long if trend positive, flat or defensive if negative.
- Vol-target each instrument.
- Cap total exposure and rebalance weekly, not daily.
- Optional: allow inverse ETFs only after proving borrow/compounding effects; better to use flat/defensive rotation first.

**Why this belongs in MrTrader**

Your PEAD book is uptrend-dependent. A trend sleeve can be the system's immune system. It may not produce a spectacular standalone Sharpe, but it can improve capital survival and reduce the temptation to overfit single-name signals.

**Validation**

- Use long histories, ideally including 2008, 2011, 2015-16, 2018 Q4, 2020, 2022, and 2025-26.
- Walk-forward on calendar blocks; no complex CPCV needed initially.
- Test robustness across lookback ensembles, not one optimized lookback.
- Use close-to-close or next-open execution with realistic ETF spreads.
- Score not just Sharpe but portfolio contribution: drawdown reduction when combined with PEAD.

**Go/no-go gate**

This sleeve is worthwhile if it reduces combined portfolio drawdown and left-tail beta without consuming too much capital or creating whipsaw losses. It does not need to beat PEAD's raw Sharpe to be valuable.

### Return source 3 - Defined-risk post-earnings options volatility / IV normalization

**Expected value rank:** 3, but only after building a new options stack  
**Use current simulator?** No  
**Correlation to PEAD:** low to moderate  
**Capacity:** fine for your size in liquid names  
**Complexity:** high  
**Primary warning:** this is where glamorous backtests go to die if you model mid-prices.

**Core hypothesis**  
Single-name options around earnings can overprice or misprice the distribution of post-event moves. After earnings, IV normalization, skew changes, and realized-vs-implied move dynamics may provide a return source distinct from buying the stock.

But do not start with naked short vol. Start with defined-risk structures.

**Candidate strategies**

1. **Post-earnings IV crush continuation:** after the event, sell defined-risk premium when IV remains elevated but event uncertainty has resolved.
2. **Earnings move under/overreaction via options:** use debit spreads or call spreads for high-quality PEAD names to cap downside and reduce capital usage.
3. **Volatility risk premium filter:** only trade when implied move materially exceeds model-expected realized move and liquidity is good.
4. **Avoid pre-earnings short straddles as a first strategy.** It is seductive and dangerous.

**Data needed**

- Historical option chains with bid/ask, greeks, IV, OI, volume.
- Corporate actions and option contract adjustments.
- Earnings timestamps.
- Underlying intraday prices.
- Realistic fill model: enter at ask for buys, bid for sells, or conservative mid slippage assumptions.
- Assignment/exercise rules for short options.

**Execution support**

Retail API options trading is no longer fantasy. Alpaca supports options and multi-leg orders; Polygon/Massive provides options chain snapshots with greeks, IV, quotes, trades, and open interest in current docs. That makes this feasible, but not easy.

**Validation**

- Build an options simulator separate from the equity-bar engine.
- Never mark fills at mid unless you haircut aggressively.
- Include liquidity filters: min OI, min volume, max spread percent, max spread dollars.
- Backtest by contract, not by underlying only.
- Use conservative exercise/assignment assumptions.
- Stress test volatility surface shocks.

**Go/no-go gate**

Pursue only if the first simulator can reproduce known option P&L mechanics on historical contracts and if conservative bid/ask fills still leave edge. If the strategy only works at mid, kill it.

### Return source 4 - Positive catalyst plus crowded short/positioning squeeze

**Expected value rank:** 4  
**Use current simulator?** Mostly yes, with event upgrades  
**Correlation to PEAD:** moderate if catalyst is earnings; lower if other catalysts are added  
**Capacity:** low to moderate  
**Complexity:** moderate  
**Primary warning:** this is not the old low-short-interest factor. It is a catalyst-conditioned squeeze continuation sleeve.

Your short-interest factor died because generic high-SI shorting stopped working and may have reversed in the meme era. That does not mean short data is useless. It means short data is not a standalone factor.

**Core hypothesis**  
High short interest is toxic as a generic short signal, but high short interest plus an objectively positive catalyst can produce forced buying and continuation.

**Signal sketch**

Long only when:

- high days-to-cover or high short interest percentile,
- positive earnings/revenue/guidance shock or other catalyst,
- strong post-event price reaction,
- borrow/crowding is high enough to matter,
- liquidity is sufficient,
- not already fully squeezed before entry.

**Data needed**

- Existing bi-monthly short interest/days-to-cover.
- FINRA daily short volume as context only, not as true short interest.
- Options OI/skew if available.
- News/earnings catalyst classification.
- Intraday price reaction and volume.

**Validation**

- Event-level bootstrap.
- Separate meme-era/post-2020 from earlier periods.
- Study gap-only vs post-entry continuation.
- Explicitly model entry delay after extreme gaps.
- Compare high-SI positive-catalyst events to low-SI positive-catalyst controls.

**Go/no-go gate**

This should remain a small opportunistic sleeve unless it shows clear incremental alpha over PEAD 2.0. If all it does is pick the same PEAD winners with larger volatility, skip it.

---

## 5. What data would most improve your odds

Ranked by value to your actual roadmap:

### 5.1 Clean expectations/revisions/guidance data

This is the highest-value addition. PEAD is about expectation revision. If your data only sees actual EPS surprise, you are measuring a crude proxy.

Minimum fields:

- EPS consensus history by timestamp.
- Revenue consensus history by timestamp.
- Forward EPS/revenue estimate revisions after announcement.
- Price target and recommendation levels/revisions.
- Guidance raise/lower indicators.
- Analyst dispersion and coverage count.

FMP may be enough for a first pass, but you must audit timestamps. For a serious build, I/B/E/S, FactSet, Visible Alpha, or Estimize-type data would be better, but may not fit budget.

### 5.2 Earnings announcement timestamps and session classification

Before-open vs after-close vs intraday announcement matters enormously. A next-open backtest can be wrong if the event was known the prior evening, if after-hours price action already incorporated it, or if the entry actually occurs long after the event move.

Store:

- announcement datetime,
- session type,
- first tradable regular-session price after event,
- after-hours move if available,
- event-day open/high/low/close path.

### 5.3 Survivorship-bias-free universe and corporate actions

For large-cap Russell 1000 today, survivorship bias can still matter. For anything small/mid-cap it can dominate. You need historical membership or a survivorship-free database if you revisit smaller names.

For now, use current Russell 1000 only with humility, or buy/use a cleaner historical database before making claims about small/mid caps.

### 5.4 Options historical chains with conservative bid/ask

If you pursue options, you need historical chains, not just snapshots:

- bid/ask,
- greeks,
- IV,
- open interest,
- volume,
- contract metadata,
- corporate action adjustments.

Polygon/Massive and ORATS are plausible retail-accessible sources. ORATS-style standardized historical IV/surface fields may save a lot of engineering time if budget allows.

### 5.5 Borrow/short availability and rebate data

Without borrow data, single-name short strategies are research toys. If shorting matters, integrate IBKR or another source for borrow availability and borrow fee history. Otherwise keep short exposure tiny or avoid it.

### 5.6 News/transcript NLP only after you fix event data

LLM/NLP on filings and calls is tempting, but it is not the first missing piece. First fix expectations, timestamps, and event windows. Then use NLP for specific fields:

- guidance sentiment,
- management non-answer rate,
- margin pressure language,
- demand/supply chain change,
- confidence/uncertainty shift.

Do not let an LLM become an unconstrained stock picker. Use it to produce structured event features.

---

## 6. Model critique: did you kill ML or just the wrong ML?

You did not prove that ML is useless. You proved that your current broad cross-sectional ML setup is not competitive.

### 6.1 What is dead

I would not spend more cycles on:

- generic daily Russell-1000 XGBoost using RSI/MACD/EMA/momentum/fundamental features;
- LambdaRank over broad stocks without a strong economic event anchor;
- intraday 5-minute ML on retail data;
- single-factor equity rankings dressed up with ML;
- nightly retraining of a model that repeatedly fails the WF gate.

Stop retraining the dead model nightly. It creates operational noise and false hope. Freeze it as a benchmark only.

### 6.2 What ML might still be useful for

ML can help if it is constrained to a real economic mechanism.

Better uses:

1. **PEAD meta-labeling:** first identify economically valid events, then use ML to decide which events to trade or size.
2. **Conditional treatment effect:** estimate which event features cause stronger drift after earnings.
3. **Regime-conditioned sizing:** size down or hedge when regime variables predict poor payoff distribution.
4. **Execution timing model:** predict whether open+5m, open+30m, or close entry is better after a specific event reaction.
5. **Options liquidity/fill model:** predict fill probability and slippage from contract features.

Model classes to prefer:

- regularized logistic/linear models,
- monotonic gradient-boosted trees,
- small XGBoost/LightGBM with strict feature grouping,
- hierarchical shrinkage by sector,
- calibrated probabilities.

Avoid deep learning for now. Your bottleneck is not model expressiveness. It is signal definition, data quality, and sample size.

### 6.3 Use residual targets, not raw returns

For single-name event models, train on alpha residuals:

```text
forward_return - beta * market_return - sector_return - style_factor_return
```

If a model cannot predict residual return, it is not alpha; it is disguised exposure. Your own graveyard already proved this.

### 6.4 Use monotonicity and stability as acceptance criteria

A model should not be promoted just because it improves Sharpe. Require:

- monotonic performance by predicted score bucket,
- stable top-bucket event counts,
- stable feature importance across folds,
- no single feature or period driving the result,
- calibration: predicted edge should map to realized edge.

If the top score bucket does not beat the second bucket, you have a ranking model that does not rank.

---

## 7. Execution and risk critique

### 7.1 Daily-bar stops are probably overstated

If your simulator uses daily bars with ATR stops/targets, be careful. A daily bar can tell you that the low breached a stop, but not whether you would have filled at the stop, below it, or after a gap. Overnight earnings names gap. Stops do not protect you from overnight discontinuities.

Upgrade the simulator to explicitly model:

- gap-through stops,
- next-open stop execution,
- intraday high/low ambiguity,
- stop/target same-day ordering ambiguity,
- no-stop protection outside market hours.

For daily swing systems, stops often improve backtest psychology more than real outcomes. Treat them as exposure/risk caps, not magic loss control.

### 7.2 Entry timing must match the event economics

Your system sends swing proposals after open volatility settles. That is operationally sensible, but PEAD economics often depend on how much of the post-announcement move has already occurred.

For every event strategy, store these counterfactual entries:

- previous close,
- first regular-session open after event,
- open+5m,
- open+30m,
- VWAP first 30m,
- close of first regular session,
- next-day open.

You may discover that waiting reduces adverse selection enough to compensate for missed drift. Or you may discover that it gives away the edge. Either answer is valuable.

### 7.3 Risk gates are good, but add strategy-level risk budgets

Current RM gates are mostly portfolio-level. Add strategy-level budgets:

- max gross exposure by strategy,
- max daily new risk by strategy,
- max open event cluster exposure by date,
- max sector exposure by strategy,
- max earnings-week exposure,
- max correlated catalyst exposure,
- max cumulative experimental loss per strategy.

Every strategy should have its own kill switch and experimental drawdown budget.

### 7.4 Measure realized beta daily

PEAD is long-biased. You need a live beta monitor:

- rolling beta to SPY/QQQ/IWM,
- sector beta,
- exposure by market cap and momentum bucket,
- crisis beta measured on down-market days,
- P&L attribution: market, sector, residual.

If residual P&L is not positive, you are renting beta.

---

## 8. Recommended system redesign

I would not tear down the PM/RM/Trader architecture. I would tear down the research process around it.

### 8.1 New architecture layers

**Layer 1 - Data lake / point-in-time event store**

Everything starts here. Store every event with `known_at` timestamps:

- earnings announcement,
- EPS/revenue surprise,
- guidance,
- analyst revisions,
- short interest release date,
- option chain snapshot time,
- news/NLP extracted fields,
- corporate actions.

**Layer 2 - Research factory**

A strategy is a frozen YAML/spec plus code hash, not a mutable Python file.

Outputs:

- fold coverage report,
- data leakage report,
- factor attribution,
- event bootstrap,
- slippage sensitivity,
- kill criteria.

**Layer 3 - Strategy incubator**

Paper or micro-live only. No strategy can go from backtest to full notional. It must pass telemetry.

**Layer 4 - Production portfolio**

Only strategies with a frozen spec and live record get capital. Capital allocation is based on incremental portfolio contribution, not standalone Sharpe.

### 8.2 Target production book

I would aim for this eventual structure:

| Sleeve | Purpose | Starting capital share | Notes |
|---|---:|---:|---|
| PEAD 2.0 genuine surprise | Primary single-name event alpha | 5-15% initially | Only after rebuild; long-biased |
| ETF trend/crisis sleeve | Drawdown control and diversification | 10-30% | Liquid, daily bars, robust |
| Event squeeze sleeve | Opportunistic convex single-name longs | 0-5% | Small; high volatility |
| Options event vol | Separate return source | 0% until simulator exists | Do not rush |
| Cash/T-bills | Optionality and risk control | residual | Cash is a position |

At $100k, capital efficiency matters, but survival matters more. Your objective is not to maximize utilization. It is to produce credible evidence.

### 8.3 Use a "capital ladder"

Every strategy should move through capital levels:

1. **Backtest only.**
2. **Shadow mode.** No orders, full decision logging.
3. **Paper mode.** Orders simulated through broker paper.
4. **Micro-live.** Tiny notional; real fills only; predefined max loss.
5. **Pilot.** Small allocation.
6. **Production.** Still capped.

Promotion should require independent event count, not calendar time. For PEAD, 3 months can be meaningless if it includes too few independent earnings events.

---

## 9. Concrete 90-day roadmap

### Days 0-14: validation repair and research governance

**Goal:** stop fooling yourself before adding alpha.

Deliverables:

1. Rebuild fold generator to eliminate regime-biased skipping.
2. Add fold coverage report by year/regime/VIX/SPY trend/sector/event count.
3. Add strategy research ledger.
4. Add factor attribution report for every backtest.
5. Freeze dead swing ML as a benchmark; stop nightly retrain except as a non-production diagnostic.
6. Add explicit entry-timing sensitivity report for PEAD.

Decision gate:

- If fold skipping cannot be fixed quickly, temporarily abandon CPCV and use simpler blocked walk-forward/event-year splits with complete coverage.

### Days 15-30: brutal PEAD retest

**Goal:** decide whether current PEAD deserves incubation.

Deliverables:

1. Re-run current PEAD under repaired validation.
2. Run factor-adjusted returns and event-cluster bootstrap.
3. Test entry timings.
4. Decompose P&L by regime, sector, market cap, event quarter, gap size, and signal strength.
5. Produce a one-page go/no-go memo.

Decision:

- If current PEAD fails, do not delete it. Keep it as a benchmark and move to PEAD 2.0.
- If it passes weakly, run it only as a tiny risk-on sleeve.

### Days 31-60: build PEAD 2.0 data and first frozen strategy

**Goal:** move from crude EPS surprise to genuine expectation shock.

Deliverables:

1. Add revenue surprise and estimate revision history.
2. Add analyst recommendation level/revision and coverage/dispersion.
3. Add earnings timestamp/session classification.
4. Add event-day price reaction features.
5. Create a frozen PEAD 2.0 rule and a separate PEAD 2.0 meta-label model.
6. Compare PEAD 2.0 vs current PEAD on incremental alpha.

Decision:

- Promote only if PEAD 2.0 improves residual alpha and robustness, not just raw Sharpe.

### Days 45-75: build ETF trend/crisis sleeve in parallel

**Goal:** create a diversifying portfolio component using daily bars.

Deliverables:

1. ETF universe and data QA.
2. Lookback ensemble trend signal.
3. Vol-targeting and weekly rebalance.
4. Combined portfolio test with PEAD current/PEAD 2.0.
5. Drawdown contribution report.

Decision:

- Keep if it improves combined drawdown/left-tail exposure even with modest standalone Sharpe.

### Days 75-90: options feasibility spike, not full strategy

**Goal:** decide whether options deserve engineering time.

Deliverables:

1. Pull a sample historical option chain dataset.
2. Build contract-level P&L prototype for one underlying and one earnings event.
3. Implement bid/ask fill assumptions.
4. Validate greeks/IV/OI fields.
5. Produce an options-stack cost estimate.

Decision:

- Proceed only if data quality and fills can be modeled conservatively.
- Do not trade options from the equity-bar simulator.

---

## 10. Glamorous-but-wrong wastes of time

### 10.1 More generic cross-sectional XGBoost

You already tested it. It died. Do not revive it with more indicators.

### 10.2 Intraday ML on 5-minute bars

Retail 5-minute ML is a cost-and-adverse-selection trap. Unless you have a clear microstructure signal and a realistic queue/fill model, avoid it.

### 10.3 LLM stock picking

Use LLMs to extract structured features from transcripts/news/filings. Do not let an LLM decide trades.

### 10.4 Options flow without institutional-quality data

Options flow is seductive. Retail-accessible flow data is often noisy, delayed, ambiguous, and full of hedging trades. If you use options data, start with chains/IV/OI and event vol mechanics, not "unusual activity" hype.

### 10.5 Short-only single-name strategies without borrow data

If you cannot model borrow availability, borrow cost, recalls, and hard-to-borrow names, your short backtest is incomplete.

### 10.6 Strategy complexity before data quality

Your biggest missing edge is not a transformer, a multi-agent LLM PM, or a more complex risk manager. It is cleaner point-in-time event data and stricter validation.

---

## 11. Final prioritized recommendations

### Recommendation 1 - Treat validation repair as the next alpha project

This is the highest-EV task. A biased validation engine is worse than no validation because it produces confidence. Fix it before adding strategies.

### Recommendation 2 - Keep current PEAD alive only as a tiny experiment

Run it in paper or micro-live for telemetry. Do not ramp. Label it internally as "risk-on PEAD candidate," not "the live edge."

### Recommendation 3 - Build PEAD 2.0 around genuine expectation shock

Add revenue, revisions, guidance, analyst prior, event-day price reaction, attention, and crowding. This is the most natural extension of the one thing that has not completely died.

### Recommendation 4 - Add an ETF/cross-asset trend sleeve

This is the most practical way to diversify a long-biased earnings book with your current simulator and account size.

### Recommendation 5 - Explore options only through a separate simulator

Options are worth a feasibility spike because data/execution access exists, but they are dangerous if modeled lazily. Build the stack first; trade later.

### Recommendation 6 - Redefine success as portfolio contribution, not standalone Sharpe

A sleeve with Sharpe 0.5 that cuts drawdowns and has low correlation to PEAD may be more valuable than a fragile Sharpe 1.0 backtest that is just beta.

---

## 12. One-page handoff for other LLM reviewers

**System:** MrTrader is a solo automated equities trading platform using PM/RM/Trader agents, Redis, SQLite audit logs, Alpaca paper execution, daily Russell-1000 swing horizon, and a dormant 5-minute intraday stack.

**Current status:** Most strategies have been killed. Generic cross-sectional ML, factor ranking, dollar-neutral L/S, intraday ML, analyst drift, insider cluster, and short-interest factor failed or were artifacts. Only large-cap PEAD remains, but it is weak: p around 0.19, HAC t around 1.04, long-only, uptrend-dependent, no live proof.

**Core critique:** The architecture is good. The alpha is not proven. The 52% CPCV fold-skip bias is severe and invalidates absolute CPCV Sharpe conclusions. Relative comparisons are not safe unless missing fold exposure is equal across strategies.

**Best next steps:**

1. Fix validation/fold coverage and add a research ledger.
2. Re-test current PEAD brutally under unbiased folds and factor-adjusted attribution.
3. Build PEAD 2.0 around genuine expectation shock: EPS + revenue + guidance + revisions + analyst prior + event reaction + crowding.
4. Add a liquid ETF/cross-asset trend sleeve for crisis diversification.
5. Explore options only after building a real options simulator with bid/ask, greeks, IV, OI, assignment, and contract-level P&L.

**Do not do:** more generic XGBoost indicators, retail 5-minute intraday ML, LLM stock picking, options-flow hype, or single-name shorting without borrow data.

---

## 13. References and source notes

1. User-provided dossier: `QUANT_REVIEW_PROMPT_2026-06.md`.
2. Bailey, D. H. and Lopez de Prado, M. "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." SSRN/JPM.
3. Lopez de Prado, M. `Advances in Financial Machine Learning` - purging, embargoing, CPCV framework.
4. Bernard, V. L. and Thomas, J. K. foundational PEAD work; related summaries and replications in accounting/finance literature.
5. Recent PEAD evidence is mixed: some work argues broad PEAD has attenuated, while conditional variants may remain useful when the earnings surprise contradicts prior analyst beliefs.
6. Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. "Time Series Momentum." Journal of Financial Economics, 2012.
7. FINRA daily short sale volume files are aggregated TRF/ADF/ORF reported short-sale volumes; they are useful context but not a substitute for true short interest or borrow data.
8. SEC T+1 settlement cycle became effective in the U.S. on May 28, 2024, which is relevant for cash/reinvestment and operational assumptions.
9. Alpaca currently documents options and multi-leg options API functionality.
10. Polygon/Massive currently documents options chain snapshots including greeks, implied volatility, quotes, trades, and open interest.

---

## Appendix A - Suggested PEAD 2.0 scoring template

This is intentionally simple. Start rules-based; only add ML after the rule has explanatory power.

| Component | Example score | Notes |
|---|---:|---|
| EPS surprise percentile | 0 to 2 | Must be PIT and sector-normalized |
| Revenue surprise confirmation | -1 to 2 | Penalize EPS beat with revenue miss |
| Guidance change | -2 to 3 | Strongest qualitative input if reliable |
| Pre-event revision trend | -1 to 2 | Negative prior + positive shock is valuable |
| Analyst recommendation inconsistency | 0 to 2 | Positive surprise against negative prior belief |
| Event-day abnormal return | -2 to 2 | Need positive confirmation but avoid exhaustion |
| Abnormal volume | 0 to 1 | Confirms attention/institutional activity |
| Short/crowding pressure | 0 to 2 | Only with positive catalyst |
| Quality filter | -2 to 1 | Avoid low-quality one-offs |
| Regime filter | block / scale | Make uptrend dependency explicit |

Trade only top-score events after liquidity and execution gates.

## Appendix B - Minimum backtest report for any strategy

Every strategy report should include:

1. Hypothesis and economic rationale.
2. Data sources and known-at timestamps.
3. Full fold coverage map.
4. All parameter choices and number of variants tried.
5. Net performance after costs.
6. Factor-adjusted alpha.
7. Event/bootstrap significance if event-based.
8. Entry/exit timing sensitivity.
9. Slippage sensitivity.
10. Performance by year, regime, sector, market cap, volatility quintile.
11. Top 20 winning and losing trades/events.
12. Concentration analysis.
13. Failure modes.
14. Pre-declared live kill criteria.

## Appendix C - Kill criteria examples

For a new strategy entering micro-live:

- Stop if realized slippage exceeds backtest assumption by more than 10 bps median or 25 bps P75 over first 30 fills.
- Stop if signal frequency is less than 50% or more than 200% of backtest expectation without explainable market-regime cause.
- Stop if factor-adjusted live residual P&L is negative after 50-100 independent events/trades, depending on strategy frequency.
- Stop if one sector or event cluster contributes more than 40% of live P&L.
- Stop if operational errors exceed 1% of intended trades.
- Stop if drawdown exceeds the strategy's pre-funded experimental loss budget.

