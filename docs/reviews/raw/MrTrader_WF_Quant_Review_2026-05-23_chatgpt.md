# MrTrader Walk-Forward Validation — Brutally Honest Quant Review and 6-Month Roadmap

**Date:** May 23, 2026  
**Prepared for:** Min / MrTrader  
**Source reviewed:** `WF_LLM_REVIEW_PROMPT.md`  
**Purpose:** Independent quant-style critique of the walk-forward validation layer, model/label design, agent simulation approach, data requirements, multi-strategy extensibility, and a prioritized 6-month engineering roadmap.

---

## 0. Executive Verdict

The current MrTrader walk-forward system is a serious prototype, not yet a capital-trustworthy research platform. The most important positive is that you are trying to replay the production decision path rather than running a toy vectorized backtest. That is the correct instinct. The negative is that the current implementation is still testing something materially different from what the model was trained to predict and materially different from what live trading will experience.

My blunt assessment:

1. **The v215 failure is probably the first honest result.** The baseline result with wide stops and 160-bar holds was not an encouraging weak positive; it was mostly market beta and stale holding-period exposure. The post-realignment result exposed the real problem: the learned signal is not yet aligned with the realized trade path.
2. **v216 should not be judged by AUC 0.554.** For a cross-sectional ranking model, AUC is at best a weak diagnostic. The model only matters if top-decile/selected candidates produce positive net-of-cost, risk-adjusted, path-aware portfolio returns. Run decile spreads, Rank IC, hit rate by rank bucket, turnover, factor exposure, and actual policy P&L.
3. **The highest-priority flaw is not “more data.” It is label-policy mismatch.** Close[t] labels, open[t+1] entries, 20-day labels, 40-bar max holds, EOD-only stops, and stops/targets omitted from labels together mean the model is not trained on the same payoff function the WF realizes.
4. **The agent-stack simulation idea is valuable but dangerous.** It is valuable if the PM/RM/Trader decision kernel is deterministic, replayable, and shares code with live. It is dangerous if “agents” introduce stochasticity, prompt drift, hidden state, incomplete tool calls, or unlogged overrides. Quant shops do not trust opaque discretionary simulation unless every state transition is traceable and replayable.
5. **The 5-position portfolio is too concentrated for an ML equity ranking strategy.** It may be appropriate for a discretionary trader with high-conviction single-name theses. It is usually a poor fit for noisy cross-sectional ML alpha. With only five positions, idiosyncratic variance can swamp weak alpha, and your WF will be high variance fold-to-fold.
6. **Your immediate next step is not options, pairs, or alt data.** The next step is a boring but critical “trust layer”: aligned labels, deterministic event replay, point-in-time data validation, realistic stop and cost modeling, factor/risk attribution, and null benchmarks.

**Capital recommendation:** Do not deploy live discretionary-sized capital based on v216 until the minimum trust fixes in Section 3 are complete. Paper trading is fine. Tiny live canary trades are acceptable only after the WF and paper/live shadow run agree at the order and position ledger level.

---

## 1. What a Top Quant Shop Would Say Immediately

A Two Sigma/AQR/Man AHL-style review would not start by asking whether XGBoost is the right model. They would ask whether the research result is causal, reproducible, net of costs, and aligned with the real decision process. On those standards, the current system has good ambition but insufficient proof.

### 1.1 What is good

The following design choices are directionally right:

- **Walk-forward validation instead of a single static backtest.** This is necessary for any strategy where model retraining, feature recalculation, and regime evolution matter.
- **Purging/embargo awareness.** Overlapping labels can leak future information into training. Your purge/embargo design is a good sign, though the exact gap must match the maximum realized label and feature horizon.
- **Point-in-time Russell 1000 universe.** Survivorship bias can make equity strategies look far better than they are. A PIT universe is a baseline requirement.
- **Replaying the production stack.** Backtests often fail because research code and live code diverge. The intent to simulate PM → RM → Trader is correct.
- **Deflated Sharpe Ratio and multiple-testing awareness.** The DSR framework exists specifically because repeated trials and non-normal returns inflate backtest confidence.[1]
- **Explicit gates.** Having deployment gates is far better than eyeballing charts.

### 1.2 What would fail institutional review

A top quant reviewer would likely stop the deployment discussion on these points:

#### A. The target function is not the trading function

The model is trained to rank future cross-sectional returns over a fixed horizon. The simulator realizes returns through a separate policy involving open auction entry, fixed position sizing, ATR stops, targets, maximum holding period, daily gates, sector constraints, costs, and sometimes shorts. Those are different payoff functions.

This is the single biggest issue. You are asking the model to learn “which stocks have high forward returns,” then using it to drive “which constrained trades survive a stopped, targeted, costed, agent-mediated portfolio process.” Sometimes that works. Often it does not.

#### B. v215 did not fail because of one bug; it failed because the research object was incoherent

The 5-day label versus 40-bar hold mismatch is not a small implementation bug. It means the model was evaluated on a path it was never optimized to predict. Moving to a 20-day label helps, but it is still not fully aligned if the strategy can exit earlier by stop/target or later by 40-bar max hold.

#### C. Current risk management is more retail rulebook than portfolio construction

MAX_OPEN_POSITIONS, MAX_POSITION_SIZE_PCT, MAX_SECTOR_CONCENTRATION_PCT, daily loss limits, and account drawdown gates are useful guardrails. They are not a portfolio optimizer. They do not directly neutralize factor exposure, beta, volatility, correlation, liquidity, crowding, or expected cost.

#### D. Execution realism is underdeveloped

For daily swing trading at $20k–$500k, you do not need a high-frequency simulator. But you do need bid/ask/spread-aware open execution, auction assumptions, shortability, borrow availability, corporate actions, and stop logic based on high/low data or intraday bars. The current model of open[t+1] plus 3 bps slippage and flat 15 bps round-trip is a useful placeholder, not a final execution model.

#### E. The WF framework is too coupled to one strategy

A 1,400-line simulator and a 1,487-line WF script suggest the strategy, execution, risk, data access, metrics, and orchestration are probably too intertwined. Adding long-only, short-only, market-neutral, intraday, pairs, stat-arb, and options inside this structure will likely create branching complexity and subtle inconsistencies.

#### F. The system has weak falsification tests

Before asking “is Sharpe > 0.8,” you need to ask:

- Does a random portfolio with the same turnover and constraints perform similarly?
- Does the signal produce monotonic decile returns?
- Is the short book actually adding value or just adding noise?
- Is the strategy just long beta, sector momentum, size, or volatility?
- Does performance survive doubled costs?
- Does performance survive using next-day open labels?
- Does performance survive market-neutralization?
- Does performance survive removing the top five best trades?
- Does performance survive each fold, each regime, and each sector?

Without these, a Sharpe gate can be misleading.

---

## 2. Trust Calibration: How Much to Trust Current WF Results

### 2.1 Trust level by version

| Result | Trust level | My interpretation |
|---|---:|---|
| v215 baseline: Sharpe +0.036 with 160-bar holds | Very low | Mostly an accidental market-beta/stale-hold test. Not evidence of signal. |
| v215 post-realign: Sharpe -0.459, win rate 31% | Medium directionally, low numerically | Likely honest evidence that the prior signal/policy was weak. Exact Sharpe is not trustworthy, but direction is useful. |
| v215 post-all-fixes: Sharpe -0.571 | Medium directionally | Widening the pool pulling in low-conviction tail is a plausible diagnosis. It says the model’s ranking quality decays quickly beyond the top candidates. |
| v216 AUC 0.554 | Low | AUC is not a portfolio metric. It is only a weak sign that there may be some signal. |
| v216 WF not yet run | None | The only result that matters is path-aware, costed, constrained, fold-stable WF. |

### 2.2 Which gaps materially affect reliability

#### Must-fix before trusting results

1. **C1: Training/backtest entry mismatch.** This directly changes the label. For short-horizon equity strategies, overnight gaps can dominate expected edge. Label entry should be open[t+1] or, if entry is truly MOO, a realistic open auction fill proxy.
2. **C2: Label/hold mismatch.** This is not optional. The model must be trained on either the expected realized policy return or a family of labels that match the strategy’s exit distribution.
3. **C3: Stops below noise.** v215’s 0.5 ATR stop likely forced the strategy to trade noise. v216’s wider stops are more plausible, but they must be validated by stop-hit distribution, average adverse excursion, and sensitivity tests.
4. **H5: EOD-only stop triggers.** If the strategy conceptually has intraday stops, daily close-only checks overstate performance during volatile periods. If you only intend to evaluate stops at close, call them “close stops,” not real stops.
5. **H2: Warm-up / feature availability.** Feature burn-in must be strict and point-in-time. No row should be tradable unless every feature would have been available and stabilized at the decision timestamp.
6. **H4: Spread/cost model.** The current flat cost model may be tolerable for first-pass swing at small capital, but not for intraday or shorts. At a minimum, stress test 0, 15, 30, 50 bps round-trip and add symbol/date liquidity proxies.
7. **H1/M6: Shortability and borrow mechanics.** If the live broker only supports easy-to-borrow names and cancels orders when a symbol is not shortable, WF must model that. Alpaca’s current docs state it supports opening short positions only in easy-to-borrow securities and does not yet support hard-to-borrow securities; they also state ETB borrow fees are currently zero for Trading API users.[3]
8. **Metric consistency.** Sharpe annualization should use the correct trading-period convention and account for serial correlation where relevant. Lo shows that simple square-root annualization is valid only under special assumptions.[2]
9. **Null benchmarks and factor attribution.** Without these, you cannot tell whether the model is alpha or just beta/sector/size/momentum exposure.

#### Important but not blocking for small capital

- **M2: ADV/market impact realism.** At $20k–$500k, market impact is probably small for liquid Russell 1000 names if position sizes are modest. Still include participation-rate checks.
- **H7: Single-threaded runtime.** This matters for iteration velocity and research discipline but not directly for correctness.
- **H6: Static VIX threshold.** Crude but acceptable as a placeholder. It should not be the primary regime engine.
- **L1: Dividends.** For 20–40 day holds, dividends matter less than label/cost/stop alignment, but ignoring them biases long/short asymmetrically over longer samples.
- **L3: Feature importance stability.** Important for research confidence, but not the first blocker.

### 2.3 Minimum bar for “trusted enough to paper trade seriously”

Before you treat v216 WF as actionable, I would require:

1. **Causal data audit passes.** Every feature, universe membership, fundamental value, sector, VIX value, and event flag must be available as of the trade decision timestamp.
2. **Label entry equals execution entry.** Labels use open[t+1] or the exact same entry price abstraction as WF.
3. **Label exit matches policy.** Either train on realized policy labels or evaluate multiple labels matching expected holding periods and stop/target regimes.
4. **Stop model has conservative high/low handling.** For daily bars, if low breaches a long stop intraday, assume stop was hit unless you have intraday evidence otherwise. If both stop and target are touched in the same bar, choose a conservative ordering or use intraday bars.
5. **Cost model is stress-tested.** Base case plus pessimistic case.
6. **Shortability filter is modeled.** Do not allow historical shorts that the live broker could not place.
7. **Random/control portfolios are included.** Same universe, same constraints, same turnover, random signal. Your model must beat this.
8. **Decile and Rank IC diagnostics pass.** You need monotonicity or at least stable top-tail separation.
9. **Factor attribution is reported.** Market beta, sector, size, value, momentum, volatility, and liquidity exposure should be measured.
10. **Order-level replay is deterministic.** Same inputs produce same proposals, approvals, orders, fills, positions, and metrics.

---

## 3. Architecture Critique

### 3.1 Current architecture grade

| Area | Grade | Why |
|---|---:|---|
| Research ambition | A- | Replaying production stack and using WF/CPCV is the right direction. |
| Current signal validity | C- | v215 failed honestly; v216 unproven; label-policy mismatch remains. |
| Backtest realism | C | Better than a toy backtest, but stop, cost, borrow, and data-timestamp realism are incomplete. |
| Execution realism | C- | Slippage placeholder exists, but open auction, spread, shortability, partial fill, and intraday stop mechanics are incomplete. |
| Portfolio construction | C | Guardrails exist, but no true optimizer, factor risk, volatility scaling, correlation handling, or capacity checks. |
| Risk management | B- for retail; C for institutional | Hard limits are useful, but insufficient for systematic portfolio risk. |
| Extensibility | C | Large monolithic simulator will resist multi-strategy expansion. |
| Observability/replay | Unknown / likely C+ | Needs event sourcing, deterministic traces, and live/WF parity checks. |

### 3.2 The biggest architectural change needed

You need to split “walk-forward” into reusable layers:

1. **Data and security master layer**  
   Responsible for PIT universe, corporate actions, calendars, bars, fundamentals, sectors, borrow/shortability, bid/ask, and event calendars.

2. **Feature layer**  
   Produces timestamped feature panels with strict availability metadata and warm-up rules.

3. **Label layer**  
   Produces multiple label families: fixed horizon, policy-realized, triple-barrier, multi-horizon, long-only/short-only asymmetric labels, and cost-adjusted labels.

4. **Model layer**  
   Trains and scores models. It should not know broker details or portfolio constraints.

5. **Signal layer**  
   Converts model outputs to forecast objects: expected return, rank, confidence, horizon, expected risk, and reason codes.

6. **Portfolio construction layer**  
   Converts signals into target positions subject to risk, factor, liquidity, concentration, turnover, and leverage constraints.

7. **Execution simulator layer**  
   Converts target positions/orders into fills using an execution profile: MOO, MOC, VWAP, limit, intraday, auction, spread, participation, borrow, partial fills.

8. **Accounting and ledger layer**  
   Maintains cash, positions, realized/unrealized P&L, borrow, fees, dividends, corporate actions, margin, and rejected/cancelled orders.

9. **Agent orchestration layer**  
   Replays PM/RM/Trader decisions as deterministic policies against event snapshots.

10. **Analytics layer**  
   Computes P&L, attribution, fold diagnostics, deciles, IC, DSR/PSR, factor exposures, turnover, capacity, and failure reports.

The key is that strategies become configuration/specification objects rather than new simulators.

### 3.3 A clean strategy specification

A practical design:

```yaml
StrategySpec:
  name: swing_lambdarank_v216
  universe: russell_1000_pit
  bar_frequency: daily
  decision_time: after_close_t
  entry:
    type: next_open
    order_type: MOO
    slippage_model: open_spread_bps_model_v1
  exit:
    max_holding_bars: 40
    stop:
      type: atr
      multiple: 1.5
      trigger: intraday_high_low_conservative
    target:
      type: atr
      multiple: 3.0
  model:
    type: xgboost_lambdarank
    label_family: policy_return_open_to_exit_v1
    train_window: rolling_3y
    rebalance_frequency: 20d
  portfolio:
    mode: long_short
    max_positions: 20
    gross_exposure: 1.0
    net_exposure: 0.0_to_0.5
    max_name_weight: 0.03
    max_sector_weight: 0.20
    beta_target: 0.0_to_0.3
  execution:
    broker_profile: alpaca_equities
    shortability_required: true
  validation:
    folds: 5
    purge_days: 45
    embargo_days: 20
    null_benchmarks: true
```

This makes it possible to run long-only, short-only, market-neutral, intraday, and pairs without duplicating the engine.

---

## 4. Label Design: Current Issues and Better Alternatives

### 4.1 Is LambdaRank on cross-sectional return rank a reasonable choice?

Yes, but only if you understand what it is and is not doing.

XGBoost’s learning-to-rank implementation uses query groups and ranking objectives such as `rank:ndcg`; the docs describe LambdaMART as an adaptation of LambdaRank to gradient boosted trees and note that training requires query/group IDs.[6] For equities, a natural query group is “all eligible stocks on a given decision date.” That matches cross-sectional selection.

The problem is not LambdaRank itself. The problem is that your label is too far from your realized payoff.

### 4.2 Failure modes of learning-to-rank in equity alpha

1. **It ignores magnitude.** A stock ranked #1 by +0.4% expected return and a stock ranked #1 by +8% expected return can be treated similarly depending on the label transformation.
2. **It can optimize the wrong part of the distribution.** If your portfolio only trades the top five names, you care disproportionately about top-tail precision, not full-list NDCG.
3. **It does not automatically understand costs.** A high-rank, high-turnover name may be unprofitable after spreads, slippage, and borrow.
4. **It does not automatically understand path.** A stock can have good 20-day return but hit a -3% stop on day two.
5. **It can learn sector and factor bets.** Sector-neutral momentum features help, but you still need factor attribution and potentially residualized labels/features.
6. **It may be asymmetric for longs and shorts.** The best long signal is not necessarily the mirror image of the best short signal. Short books often require separate modeling due to borrow, squeeze risk, earnings risk, and hard-to-borrow availability.
7. **It may hide weak economic signal behind acceptable ML metrics.** AUC 0.554 can be meaningful in finance, but it can also be economically useless after costs and constraints.

### 4.3 Better label families to implement

You should not pick one label and declare victory. Build a label factory and test label families.

#### Label family A: Policy-realized net return label

For each decision date and symbol:

- Entry = same abstraction as WF, e.g., open[t+1] plus estimated spread/slippage.
- Exit = same exit policy as WF: stop, target, max holding period, and conservative intrabar order.
- Return = net of transaction costs, borrow cost, and estimated spread.
- Label = cross-sectional rank of this policy-realized net return.

This is the most aligned label. It also risks overfitting to your current exit rules, so it should be compared to other labels.

#### Label family B: Multi-horizon forward returns

Train or blend labels over 5, 10, 20, and 40 trading days. Then evaluate which horizon contributes to actual policy P&L. This is especially important if your realized holding period distribution is not fixed.

A simple approach:

```text
score = 0.20 * rank_5d + 0.30 * rank_10d + 0.30 * rank_20d + 0.20 * rank_40d
```

Weights should be determined out-of-sample, not hand-tuned repeatedly on the same WF.

#### Label family C: Volatility-adjusted return / return-to-risk rank

Rank by future return divided by realized volatility, ATR, downside deviation, or expected drawdown. This can reduce the tendency to select high-volatility names that look good on raw return but are poor after stops.

#### Label family D: Triple-barrier / path-aware label

The triple-barrier framework labels observations using upper, lower, and time barriers rather than only a fixed horizon; this is designed to make labels more path-aware and volatility-aware than naive fixed-horizon labels.[7]

For your system, use:

- Upper barrier = target multiple
- Lower barrier = stop multiple
- Vertical barrier = max holding period
- Optional label value = realized net return or classification of first barrier touched

This directly addresses the “ATR stops in WF but not in training” issue.

#### Label family E: Meta-labeling

Use the ranker to propose trades, then train a second model to answer: “Given this candidate was selected, should we actually take it?” The meta-label can be whether the candidate would have produced positive net return under the actual policy. This is useful when the ranker has weak top-tail signal but many false positives.

### 4.4 Should ATR stops be in training labels?

Not always. The correct answer depends on whether the stop is part of the alpha definition or only a risk overlay.

- If stops rarely bind and are only disaster control, pure horizon labels can be acceptable.
- If stops frequently bind, the model must learn the stopped payoff distribution.
- If stop-hit rate materially changes rank ordering, stops belong in the label or in a meta-label.

Your v215 stop-hit rate above 60% means stops were not a minor overlay. They dominated realized P&L. In that regime, not including stops in labels is wrong.

For v216, run this diagnostic:

| Diagnostic | Interpretation |
|---|---|
| Stop-hit rate < 15% | Stops are secondary; pure horizon labels may be acceptable. |
| Stop-hit rate 15–35% | Mixed; test stopped-return labels and meta-labels. |
| Stop-hit rate > 35% | Stops are central to payoff; labels must be path-aware. |

---

## 5. Specific Open Questions

### 5.1 20-day labels vs 40-bar hold: is 20 days right?

Twenty days is a reasonable experiment, not a principled final answer. The right label horizon should match the realized holding-period distribution and the decay profile of the alpha.

Do not think of `max_hold_bars=40` as the label horizon. It is a maximum. If most trades exit by day 9 due to stops/targets, then a 40-day label is wrong. If most trades hold to day 35, then a 20-day label may be too short.

Recommended approach:

1. Run v216 WF with full trade logs.
2. Measure realized holding period distribution by trade, side, sector, volatility regime, and rank bucket.
3. Compute return predictability at 5/10/20/40 days and under policy-realized exits.
4. Use the label family that produces the best out-of-sample top-tail monotonicity and policy P&L, not the one with the best AUC.

A clean solution is **multi-horizon modeling** plus **policy-realized labels**. Treat 20-day as one candidate.

### 5.2 Does LambdaRank have a train/live distribution mismatch because live only scores at rebalance points?

Not inherently. Training on daily cross-sections and scoring on rebalance dates can be fine if rebalance dates are a representative subset of training dates.

The mismatch appears if:

- Training includes all days, but live only trades every 20th day and those days have different signal distributions.
- Training ranks the full universe, but live only evaluates a prefiltered proposal pool.
- Training uses one universe/filter, live uses another.
- Training labels are created for after-close decisions, while live decisions happen before/at open.
- The model sees unavailable features at training time.

Run a distribution audit:

- Feature distributions: train days vs rebalance days.
- Score distributions: all days vs traded days.
- Rank bucket realized returns: daily scoring vs actual rebalance dates.
- Universe overlap: training eligible names vs live eligible names.

### 5.3 ATR stops in WF but not training labels: wrong or acceptable?

Acceptable only if stops are wide enough that they do not materially alter the payoff ranking. v215 proved they were not. v216 might be acceptable, but only after diagnostics.

The theoretical relationship:

- Stop width should be calibrated to the forecast horizon’s noise distribution.
- A 20-day alpha should not use a stop so tight that normal one-day noise exits most trades.
- A stop should correspond to a thesis invalidation threshold, not a random pain threshold.
- If the stop is a hard part of the strategy, the label should reflect first passage to stop/target.

### 5.4 Five positions: high conviction or idiosyncratic risk bug?

For this type of ML ranking strategy, five positions is more likely a bug than a feature.

A five-name portfolio can work if:

- Signals are extremely strong.
- Names are carefully diversified.
- There is deep single-name research behind each position.
- Position sizes are volatility-adjusted.
- Factor/sector exposure is controlled.

Your system is not yet proving that. It is using a weak/noisy cross-sectional ranker. With only five positions, idiosyncratic events can dominate the alpha. A single earnings surprise, FDA headline, antitrust headline, index rebalance, or gap can distort a fold.

Recommended structure for $20k–$500k:

| Capital | Suggested max positions | Max name weight | Comments |
|---:|---:|---:|---|
| $20k | 8–15 | 5–10% | Keep commissions/spreads in mind; fractional shares help. |
| $100k | 15–30 | 3–5% | Better diversification; still manageable. |
| $500k | 25–75 | 1–4% | Need liquidity and turnover controls. |

If you insist on five positions, require a much higher signal threshold and treat it like a concentrated discretionary strategy. Do not judge it by the same expectations as a diversified ML ranker.

### 5.5 Expanding window vs rolling window

Expanding windows are stable and use more data, but they can overweight stale regimes. Rolling windows adapt faster but can be noisy.

For Russell 1000 daily cross-sectional alpha, I would test:

- **1-year rolling:** likely too noisy but adaptive.
- **3-year rolling:** good baseline.
- **5-year rolling:** stable, may include stale regimes.
- **Expanding with exponential decay:** useful compromise.

Do not decide this philosophically. Treat train-window length as a hyperparameter evaluated with nested WF/CPCV and controlled multiple-testing discipline. Because repeated testing can inflate performance estimates, keep a model registry of every trial and use DSR/PSR-style discipline.[1]

---

## 6. Agent Workflow Integration

### 6.1 The promise of simulating PM → RM → Trader

The best version of this design is powerful: the same decision stack that trades live is replayed historically against causal data snapshots. That reduces research/live divergence.

But the phrase “simulate the exact agent stack” hides many traps.

### 6.2 Failure modes commonly missed

1. **Non-deterministic model/agent output.** If the PM agent can produce different proposals for the same date and inputs, WF is not reproducible.
2. **Prompt drift.** A prompt or system instruction changes in live but not in historical WF.
3. **Hidden memory/state.** Live agents may have conversation/session memory that the historical simulator does not reproduce.
4. **Tool availability differences.** Live tools can fail, time out, or return slightly different data than historical tools.
5. **Human override not modeled.** If live has manual intervention, historical WF must either model it or exclude it.
6. **Rejected/cancelled orders ignored.** Broker restrictions, shortability, partial fills, and order cancels are part of the strategy.
7. **Risk manager timing mismatch.** RM decisions must use the same equity, positions, marks, and pending orders available at that moment.
8. **Clock mismatch.** After-close, pre-open, open, intraday, and EOD data availability must be explicit.
9. **Corporate actions and symbol changes.** Agents may see current symbols/names unless historical symbol mapping is enforced.
10. **LLM explanation contaminating decision logic.** If natural-language reasoning changes outputs, it must be frozen and replayable.

### 6.3 How leading quant shops handle sim-to-live

The institutional pattern is:

- A **deterministic decision kernel** produces orders.
- Research and live share the same code path for feature generation, signal generation, portfolio construction, and order generation.
- The backtest is **event-sourced**: every input, decision, order, fill, position, and P&L event is logged.
- Live runs in **shadow mode** before trading: the system generates hypothetical orders and compares them to backtest expectations.
- There is a **paper/live reconciliation report**: expected order, submitted order, fill, broker status, position, cash, and P&L.
- Any discretionary or LLM component is either outside the trading-critical path or converted into structured, deterministic policy outputs.

Recommendation: let agents explain, monitor, and propose. Let deterministic policy modules decide orders. If you want LLM agents in the decision path, freeze prompts/model versions, use temperature 0, log raw inputs/outputs, validate JSON schemas, and require deterministic replay tests.

---

## 7. Multi-Strategy Flexibility

### 7.1 Do not build one simulator per strategy

If you build separate simulators for swing, intraday, long-only, short-only, market-neutral, pairs, and options, you will create inconsistent accounting, inconsistent costs, inconsistent risk, and impossible debugging.

Instead, build one event-driven simulation engine with strategy plugins.

### 7.2 Core abstractions needed

| Abstraction | Purpose |
|---|---|
| `Instrument` | Equity, ETF, option, pair/spread, future. |
| `UniverseProvider` | PIT eligible instruments by date. |
| `DataView` | Causal data snapshot available at decision time. |
| `FeaturePipeline` | Timestamped feature calculation with warm-up and availability checks. |
| `LabelFactory` | Fixed horizon, policy return, triple barrier, residual return, multi-horizon labels. |
| `ModelAdapter` | Fit/predict interface for rankers, regressors, classifiers, ensembles. |
| `Signal` | Structured forecast: side, score, expected return, confidence, horizon, risk. |
| `PortfolioConstructor` | Converts signals into target positions. |
| `RiskPolicy` | Hard gates, factor limits, sector limits, daily loss, drawdown, correlation heat. |
| `ExecutionProfile` | MOO, MOC, intraday market/limit/VWAP, options execution, auction assumptions. |
| `BrokerSimulator` | Fill logic, shortability, fees, partial fills, cancels, margin. |
| `Ledger` | Cash, positions, P&L, dividends, borrow, corporate actions. |
| `MetricsPack` | Strategy-specific analytics and common risk/performance metrics. |

### 7.3 Strategy modes

#### Long-only

- Disable short book.
- Add beta and sector exposure reports.
- Compare against SPY/Russell 1000 benchmark, not cash.
- Add cash drag and max exposure rules.
- Use long-only labels or top-tail labels.

#### Short-only

- Require shortability and borrow data.
- Model Rule 201/alternative uptick restrictions. SEC Rule 201 restricts short-sale prices after a covered security drops more than 10% in a day.[4]
- Add earnings/event blackout windows.
- Add squeeze/crowding diagnostics.
- Consider separate short model.

#### Long/short market-neutral

- Add optimizer with gross/net exposure constraints.
- Neutralize beta, sector, size, value, momentum, and volatility where possible.
- Evaluate long leg and short leg separately.
- Report alpha spread, not only total return.

#### Intraday

- Daily bars are insufficient.
- Needs bid/ask or at least spread proxy.
- Needs minute-level event calendar and market open/close behavior.
- Needs latency assumptions and no-lookahead bar construction.
- Costs dominate; 15 bps flat is too crude for a 30–50 bps target.

#### Pairs/stat-arb

- Requires pair universe construction, cointegration/spread diagnostics, hedge ratio estimation, borrow/shortability, market-neutral accounting, and spread-level risk.
- Labels are spread returns, not single-name returns.
- Do not bolt this onto the current single-name LambdaRank simulator.

#### Options

- Requires an options security master, contracts, IV surface, greeks, exercise/assignment, liquidity, wide spreads, corporate actions, and OCC symbology.
- This is out-of-scope until equity WF is trustworthy.

---

## 8. Data Requirements Ranked by Expected Impact

Below is my ranked view for this system. I separate **WF reliability** from **model alpha** because some datasets primarily make backtests honest, while others may add predictive signal.

| Rank | Dataset | Primary value | Priority | Why |
|---:|---|---|---|---|
| 1 | Corporate event calendar: earnings, dividends, splits, symbol changes | Reliability + alpha defense | Must-have | Prevents trading through known event risk blindly; fixes dividends/corporate-action accounting; enables event blackout tests. |
| 2 | Intraday bid-ask spreads / NBBO or reliable spread proxy | Reliability | Must-have for intraday; high for swing | Execution costs can erase short-horizon edge. Intraday targets of 0.5% cannot be trusted with crude 15 bps flat cost. |
| 3 | Shortability / borrow availability and borrow cost | Reliability | Must-have if shorts enabled | Live broker constraints must match WF. Also separates easy-to-borrow from impossible/hard-to-borrow historical shorts. |
| 4 | Factor exposures / risk model: sector, beta, size, value, momentum, volatility, liquidity | Reliability + portfolio quality | Must-have | Without factor attribution you do not know whether model P&L is alpha or compensated/uncompensated factor exposure. |
| 5 | Analyst estimates / earnings surprises | Alpha + event awareness | High | Useful for medium-horizon equity alpha and avoiding earnings landmines. Needs PIT timestamping. |
| 6 | Options implied volatility / term structure | Regime + event risk | High | Strong for vol regime, event risk, skew/crowding, and risk sizing. More useful than raw VIX alone. |
| 7 | Economic calendar: FOMC, CPI, NFP | Risk/regime | Medium-high | Important for exposure throttling and event blackout logic; less likely to drive single-name alpha directly. |
| 8 | News sentiment beyond current NIS | Alpha + risk | Medium-high | Can help if truly timestamped and evaluated causally. Dangerous if vendor timestamps/revisions are sloppy. |
| 9 | ETF flow data | Crowding/momentum | Medium | Useful for sector/theme flow and crowding; may be more valuable for regime/sector allocation than single-name picks. |
| 10 | Short interest | Alpha/risk | Medium | Useful but often delayed; better for crowding/squeeze risk than daily execution realism. |
| 11 | 13-F institutional holdings | Crowding/slow alpha | Low-medium | Lagged quarterly data; can help longer-horizon crowding but not urgent for 20–40 day WF. |
| 12 | Level 2 order book | Execution | Low for current capital; high for HFT | Overkill for $20k–$500k daily swing. Maybe useful for intraday later, but not before NBBO/spread basics. |
| 13 | Alternative data: satellite, credit card, web traffic | Alpha | Out-of-scope for now | Expensive, hard to validate, high overfitting risk. Do not add until research hygiene is institutional. |

### Data-provider blunt note

Using yfinance for any capital-trustworthy WF is not good enough. It is fine for prototyping. For serious validation, use a paid, versioned, point-in-time data source for OHLCV, corporate actions, dividends, delistings, symbol changes, fundamentals, and ideally bid/ask. The exact vendor matters less than having stable historical snapshots, timestamps, and auditability.

---

## 9. Risk and Portfolio Construction

### 9.1 Current fixed-fraction sizing is too simple

A flat 5% position size ignores volatility, liquidity, correlation, and signal confidence. It also makes low-vol and high-vol names contribute very different risk.

Better alternatives:

1. **Volatility-targeted sizing**  
   Size positions so each contributes similar ex-ante volatility or stop-risk.

2. **Signal-strength sizing**  
   Larger weights only for stronger forecast scores, capped by liquidity and concentration.

3. **Risk-budget sizing**  
   Allocate risk to names/sectors/factors rather than allocating dollars.

4. **Optimizer-based construction**  
   Maximize expected alpha minus costs and risk penalty subject to constraints.

For your current size, a simple robust version is enough:

```text
raw_weight_i = score_strength_i / forecast_vol_i
cap by max_name_weight
cap by ADV participation
cap by sector/factor limits
normalize to gross/net exposure target
```

### 9.2 Portfolio heat must include correlation

Current heat = sum(position size × stop distance) assumes independent positions. In a selloff, correlations rise. Five “different” tech/growth names can behave like one large position.

Minimum improvement:

- Estimate rolling correlation of position returns.
- Report portfolio expected volatility.
- Add beta and sector heat.
- Stress test all positions with market down 2%, VIX spike, sector shock, and gap-open scenarios.

### 9.3 Sector concentration should not be random

If a sector limit is hit, reject the worst risk-adjusted candidate, not a random candidate. The RM should have a deterministic marginal utility rule:

```text
candidate_utility = expected_alpha - cost_penalty - risk_penalty - concentration_penalty
```

Then choose the candidate set that maximizes total utility under constraints.

---

## 10. Execution Realism

### 10.1 Swing trading execution

For daily swing MOO/MOC trading, you need:

- Open auction fill assumptions.
- Spread/slippage by symbol/date/liquidity bucket.
- Gap handling.
- Partial fill/cancel possibility for illiquid names.
- Shortability at order time.
- Corporate action and dividend adjustments.
- Conservative stop/target ordering when using daily bars.

A basic but useful open-cost model:

```text
estimated_cost_bps = half_spread_bps(symbol,date)
                   + auction_slippage_bps(symbol,date)
                   + participation_impact_bps(order_notional / ADV)
                   + fixed_fee_bps
```

At $20k–$500k, participation impact will usually be small for Russell 1000 names. Spread and auction slippage matter more.

### 10.2 Intraday execution

The intraday subsystem cannot be trusted with 0.5% target and 0.3% stop until costs and bar construction are much tighter.

Intraday must include:

- Bid/ask or at least robust spread proxy.
- Entry/exit based on executable prices, not midpoint or bar close.
- No lookahead within 5-minute bars.
- Same-bar stop/target ordering rule.
- Latency and order type assumptions.
- Realistic no-entry windows around open/close if needed.
- News/halts/gaps handling.

With a 0.5% target, a 10–20 bps round-trip error is enormous. Execution modeling is not a detail; it is the strategy.

### 10.3 Market impact

For your capital range, full Almgren-Chriss execution optimization is probably overkill, but the concept matters: institutional execution models explicitly trade off expected transaction costs and volatility risk.[5] Implement simple liquidity/participation guardrails now; save advanced impact models for later.

---

## 11. Performance Metrics and Gates

### 11.1 Current gates are directionally good but incomplete

Existing gates:

- avg Sharpe > 0.8
- no fold Sharpe < -0.3
- DSR p > 0.95
- avg profit factor > 1.0
- avg Calmar > 0.3

These are useful, but add the following:

| Gate | Recommended threshold / rule |
|---|---|
| Minimum trades per fold | Enough to make fold Sharpe meaningful; flag thin folds. |
| Top-decile spread | Top bucket must beat middle and bottom buckets net of costs. |
| Rank IC / ICIR | Positive and stable across folds. |
| Long leg and short leg separately | Each must be evaluated; short leg cannot hide inside long beta. |
| Cost sensitivity | Strategy should not collapse under +15–30 bps extra cost. |
| Turnover | Must be explainable and capacity-aware. |
| Beta/factor exposure | Report and gate on unwanted exposures. |
| Random strategy comparison | Must beat random same-constraints baseline. |
| Best-trade dependency | Remove top 1%, 5%, 10 trades; strategy should not vanish. |
| Regime stability | Report bull, bear, high-VIX, low-VIX, crisis, and earnings-event performance. |

### 11.2 DSR is useful but not magic

DSR corrects for selection bias, non-normal returns, and multiple testing.[1] But it only works if the number of trials and result distribution are honestly tracked. If you run 200 experiments and only record 12, DSR is theater.

Create an experiment registry:

- model version
- features
- labels
- universe
- costs
- stops/targets
- train window
- hyperparameters
- code commit hash
- data snapshot hash
- WF results
- reason for run
- whether it was pre-registered or exploratory

### 11.3 Sharpe annualization

Use 252 trading days for daily trading returns, but be careful with overlapping holdings and serial correlation. Lo’s work shows simple square-root annualization can be wrong under serial correlation.[2] For holding periods up to 40 days, serial correlation can matter. Report both standard and autocorrelation-adjusted Sharpe where possible.

---

## 12. Immediate v216 Test Plan

Before you start building new strategies, run v216 through this exact diagnostic pack.

### 12.1 Pre-run validation

1. Confirm labels use open[t+1] or exact WF entry abstraction.
2. Confirm no symbol/date is tradable before all features have warm-up.
3. Confirm universe membership is PIT.
4. Confirm every fundamental field has availability timestamp or conservative lag.
5. Confirm purge/embargo covers maximum label horizon plus feature leakage risk.
6. Confirm costs are applied symmetrically and correctly for long/short.
7. Confirm shortable universe is filtered if shorts are enabled.

### 12.2 Signal diagnostics

Run these before portfolio simulation:

- Daily cross-sectional Rank IC.
- Rank IC by fold.
- Rank IC by sector.
- Rank IC by VIX regime.
- Decile/quintile forward returns net of simple costs.
- Top 5, top 10, top 20 candidate realized returns.
- Score distribution stability.
- Feature importance stability by fold.
- Prediction correlation with known factors.

### 12.3 Portfolio diagnostics

Run these WF variants:

| Variant | Purpose |
|---|---|
| Long-only top N | See if signal works on long side before short complexity. |
| Short-only bottom N | Test whether short book has independent edge. |
| Long/short beta-neutral | Separate alpha spread from market beta. |
| No stops | See whether stops help or destroy signal. |
| Wider/tighter stops | Sensitivity to exit policy. |
| 5/10/20/40 max positions | Test concentration. |
| 0/15/30/50 bps cost | Cost robustness. |
| Random same-turnover portfolio | Baseline. |
| Sector-neutral ranking | Check sector-bet contamination. |
| Rolling vs expanding train | Regime adaptation. |

### 12.4 Pass/fail interpretation

- If deciles are not monotonic and top bucket is not consistently positive net of costs, stop. Do not tune stops to rescue it.
- If long leg works and short leg fails, run long-only first.
- If performance only appears with five positions and disappears with 20, suspect overfitting/top-tail luck.
- If performance only appears with no costs or no stops, the strategy is not ready.
- If performance is mostly one fold or one sector, the strategy is not robust.

---

## 13. Six-Month Roadmap

The roadmap below assumes one small engineering/research team and a $20k–$500k target capital base. It is deliberately practical. Hedge-fund-level discipline does not mean building every hedge-fund feature; it means knowing which institutional practices matter for your scale.

### Month 1: Trust Foundation / Alignment Fixes

**Goal:** Make the WF evaluate the same payoff the model is trained to predict.

Must deliver:

- Label entry changed to open[t+1] or exact MOO execution abstraction.
- Label factory supporting fixed 5/10/20/40-day labels and policy-realized labels.
- Stop/target-aware label option using conservative high/low or intraday bars.
- Warm-up enforcement for all rolling/EMA features.
- Metric annualization cleanup.
- Purge/embargo recalibrated to max label horizon and feature lookback.
- Full trade ledger export: candidate, score, rank, order, fill, exit reason, P&L, costs, holding period.

Acceptance gate:

- A reproducibility test can rerun one fold and produce identical orders/fills/P&L.
- Label audit proves no close[t] / open[t+1] mismatch remains.
- v216 is run with full diagnostics, not just aggregate Sharpe.

### Month 2: Data and Execution Realism

**Goal:** Eliminate the biggest sources of fake P&L.

Must deliver:

- Corporate action/dividend handling or explicit adjusted-price/accounting policy.
- Shortability filter based on broker-realistic data or conservative proxy.
- Base spread/slippage model by liquidity bucket.
- Cost sensitivity framework.
- Daily bar stop logic using high/low conservative ordering.
- Intraday stop validation for a sample period where 5-minute data exists.

Acceptance gate:

- Strategy results are reported under base and pessimistic costs.
- Stop-hit rates and same-bar stop/target ambiguity are reported.
- Short trades that would not be executable are excluded or flagged.

### Month 3: Portfolio Construction and Risk Attribution

**Goal:** Move from retail guardrails to systematic portfolio construction.

Must deliver:

- Long leg / short leg separate analytics.
- Sector, beta, volatility, size/liquidity exposure report.
- Volatility-scaled position sizing.
- Deterministic candidate selection under sector constraints.
- Correlation-aware portfolio heat or at least portfolio volatility estimate.
- Variants with max positions 5/10/20/40.

Acceptance gate:

- You can state whether the strategy is alpha, beta, sector momentum, size, or volatility exposure.
- A diversified version is tested. If five positions remains best, you have evidence, not assumption.

### Month 4: Model and Label Research Sprint

**Goal:** Find out whether there is real signal after the WF is trustworthy.

Must deliver:

- Compare LambdaRank labels: 20-day, 40-day, multi-horizon, policy-realized, volatility-adjusted, triple-barrier.
- Compare model objectives: ranker vs regressor vs classifier/meta-label.
- Rolling vs expanding train windows: 1y, 3y, 5y, expanding with decay.
- Nested validation or controlled experiment registry.
- Feature stability and ablation tests.
- Null/random baselines.

Acceptance gate:

- One candidate model has stable top-tail diagnostics and survives cost/risk sensitivity.
- If no model passes, the correct decision is to pause live deployment and revisit alpha sources.

### Month 5: Framework Refactor for Multi-Strategy Support

**Goal:** Avoid turning the current simulator into a branching monster.

Must deliver:

- StrategySpec configuration object.
- Shared event-driven engine.
- Separate model/signal/portfolio/execution/accounting modules.
- Long-only and long/short strategy modes using the same engine.
- Intraday engine compatibility plan, but not necessarily full production readiness.
- Standard metrics pack across strategies.

Acceptance gate:

- The same engine can run long-only and long/short daily WF by changing config, not duplicating simulator code.
- Trade ledger schema is identical across strategies where possible.

### Month 6: Paper Trading, Shadow Reconciliation, and Deployment Gate

**Goal:** Prove sim-to-live parity before live capital.

Must deliver:

- Daily shadow run producing proposed orders before market.
- Broker/paper account reconciliation: proposed vs submitted vs filled vs held positions.
- Live data snapshot archiving for replay.
- Drift dashboard: feature drift, score drift, order count, exposure, P&L, slippage.
- Kill-switch and daily loss enforcement tested.
- Post-trade review report.

Acceptance gate:

- At least 4–6 weeks of paper/shadow runs with order-level reconciliation.
- No unexplained divergence between WF/paper/live decision path.
- Final deployment memo: what strategy, what capital, what max loss, what stop conditions.

---

## 14. What Is Must-Have vs Nice-to-Have vs Out-of-Scope

### Must-have before trusting WF

- Entry/label alignment.
- Horizon/exit-policy alignment.
- Causal PIT feature and data audit.
- Warm-up enforcement.
- Conservative stop/target handling.
- Cost stress tests.
- Shortability/borrow modeling if shorts are enabled.
- Trade-level ledger.
- Decile/Rank IC diagnostics.
- Random/null baseline.
- Factor/sector/beta attribution.
- Deterministic replay.

### Nice-to-have after core WF is trusted

- Options IV term structure.
- Analyst estimates and earnings surprise data.
- ETF flows.
- Better regime model.
- Advanced optimizer.
- CPCV path distribution improvements.
- Feature importance stability dashboard.
- Auto-generated model cards.

### Out-of-scope for now

- Level 2 order book for daily swing.
- Options strategies.
- Alternative data.
- Deep reinforcement learning.
- High-frequency market-making simulator.
- Complex stat-arb/pairs engine before the core equity WF is modular.

---

## 15. Final Brutal Summary

The current system is at an important turning point. The bad v215 WF result should be treated as a success of honesty, not a failure of the project. It revealed that the original backtest was not measuring durable alpha.

The mistake would be to respond by adding more features, more agents, more datasets, or more strategy types before the validation layer is trustworthy. That would create a larger machine that produces more sophisticated false confidence.

The right path is:

1. **Fix the research object.** Train the model on labels that match how trades are entered, exited, costed, and constrained.
2. **Make the simulator causal and replayable.** Every decision must be reproducible from timestamped inputs.
3. **Prove the signal before optimizing the portfolio.** Deciles, IC, top-tail returns, and null benchmarks before Sharpe worship.
4. **Then improve portfolio construction.** Diversify, volatility-scale, factor-control, and cost-control.
5. **Only then add more strategy types.** Long-only and market-neutral are logical next steps; intraday requires much better execution data; options and alt-data should wait.

If v216 passes after these fixes, you may have the beginning of a real systematic trading platform. If it fails, that is also valuable: it means the architecture is starting to tell the truth. At this stage, a truthful “no alpha yet” is far more valuable than a polished backtest that accidentally trades the past.

---

## 16. References

[1] David H. Bailey and Marcos López de Prado, “The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.” SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

[2] Andrew W. Lo, “The Statistics of Sharpe Ratios.” SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260

[3] Alpaca Docs, “Margin and Short Selling.” https://docs.alpaca.markets/us/docs/margin-and-short-selling ; Alpaca Support, “What are the fees for short selling?” https://alpaca.markets/support/short-selling-fees

[4] Nasdaq Trader, “Short Sale Circuit Breaker,” summarizing SEC Rule 201 / Alternative Uptick Rule. https://www.nasdaqtrader.com/trader.aspx?id=shortsalecircuitbreaker ; SEC, “SEC Approves Short Selling Restrictions.” https://www.sec.gov/news/press/2010/2010-26.htm

[5] Robert Almgren and Neil Chriss, “Optimal Execution of Portfolio Transactions.” https://papers.ssrn.com/sol3/papers.cfm?abstract_id=53501

[6] XGBoost documentation, “Learning to Rank.” https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html

[7] Marcos López de Prado, *Advances in Financial Machine Learning*, Chapter 3 labeling discussion. O’Reilly online excerpt: https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml
