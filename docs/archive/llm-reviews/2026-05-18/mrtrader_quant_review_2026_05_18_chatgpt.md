# MrTrader Deep Quant Review - Brutal Assessment and Next Steps

Date: 2026-05-18  
Reviewer stance: senior quant / systematic PM / trading-system builder  
Input reviewed: `llm_review_prompt_2026_05_18.md`  
Intended use: LLM-ingestible handoff file for comparing recommendations across multiple models

## 1. Executive verdict

The system is not fundamentally hopeless, but the current research loop is not yet measuring a tradable strategy. The biggest problem is not LightGBM, XGBoost, Optuna, or the number of features. The biggest problem is that the system repeatedly validates one economic object and then deploys a different one.

The clearest example is the factor portfolio. It was validated as a monthly rebalance, equal-weight, no single-name stop strategy, then deployed through a PM/RM/Trader path that uses ATR stops, targets, rescoring exits, and max-hold exits. That invalidates the conclusion. The negative AgentSimulator Sharpe does not prove the factor signal is bad. It proves the live execution contract is not the same contract that produced the positive backtest.

The second major problem is that the ML labels do not match the live decision. A cross-sectional top-quintile label says, "Which stocks are least bad or best relative to peers?" The live system needs to answer, "Should I put scarce capital into this long-only position after costs, or should I hold cash?" Those are not the same question. In a bear market, top quintile can mean "lost less money." A long-only account should often do nothing in that regime. A ranker that is forced to pick winners every day has no native concept of cash.

The third major problem is that exit policy is being treated as plumbing, but it is part of the alpha definition. A 0.5x ATR stop is not a harmless risk overlay. It transforms a medium-horizon factor or momentum signal into a short-horizon path-dependent trading strategy. If the label was trained on 10-day or 20-day forward return, but the system exits after normal two-day noise, the model is being judged on an outcome it was never trained to predict.

My blunt answer: pause live-style deployment decisions until every strategy has a formal Strategy Contract and one shared backtest/live execution path. Continue paper trading only as a plumbing test, not as evidence of alpha. The factor pivot is reasonable, but only if you trade it like a factor portfolio. The current ML campaign should remain closed until labeling, execution, and portfolio construction are redesigned around the actual trade you intend to place.

## 2. The single structural flaw explaining most of the pain

You are mixing three different businesses:

1. Factor portfolio management: slow-moving cross-sectional tilts, monthly or weekly rebalance, high tolerance for intra-month noise, portfolio-level drawdown/risk control.
2. Swing trading: discrete entries, path-dependent exits, target/stop/time-out logic, trade-level expectancy.
3. Intraday trading: microstructure-sensitive signals, short holding periods, spread/slippage-dominated P&L, broker/margin constraints.

These are not minor variations of one system. They require different labels, different data, different fill assumptions, different risk budgets, and different validation metrics.

The current architecture routes all of them through a common agent simulator that behaves like a stop/target trading engine. That makes sense for a swing trade. It does not make sense for a monthly factor portfolio. It may be far too crude for intraday.

A professional shop would not ask, "Does this model have Sharpe?" in isolation. It would ask: "For this exact strategy contract, under this exact execution policy, with this exact portfolio construction, does the net-of-cost distribution beat a relevant benchmark out of sample after accounting for data quality, capacity, slippage, turnover, and multiple testing?"

Right now, the answer is unknown because the research object keeps changing.

## 3. Strategy Contract - the mandatory next abstraction

Before more modeling, create a `StrategyContract` object and require every backtest, paper run, and live run to reference it. A strategy should not be deployable without one.

Minimum fields:

- `strategy_id`: e.g., `factor_monthly_v1`, `swing_trade_xgb_v1`, `intraday_orb_v1`.
- `alpha_horizon`: e.g., 1 hour, 5 days, 20 trading days, 1 month.
- `rebalance_frequency`: intraday scan, daily, weekly, monthly.
- `entry_rule`: next open, same-bar close, next 5-minute bar, limit/VWAP approximation.
- `exit_rule`: next rebalance, time stop, target/stop, trailing stop, model-rescore exit, portfolio risk-off exit.
- `position_count`: min/max names, target concentration, fractional shares allowed.
- `capital_policy`: cash allowed, always invested, benchmark allocation, T-bill/cash yield assumption.
- `risk_policy`: max gross/net exposure, sector cap, single-name cap, portfolio stop, daily loss rule.
- `cost_model`: commission, spread, slippage, market impact, borrow/margin costs if any.
- `data_policy`: PIT universe, corporate actions, delist handling, fundamental availability lag.
- `go_live_gate`: required OOS metrics and required live plumbing metrics.

Once this exists, the simulator cannot accidentally apply ATR stops to a factor portfolio unless the contract says so.

## 4. Factor portfolio - right pivot, wrong deployment

The pivot to a rule-based factor portfolio is directionally correct. With $20k, daily bars, long-only equities, and limited professional-grade data, a transparent factor/risk allocation system is much more plausible than a highly tuned cross-sectional ML selector.

But the current factor backtest is not yet proven. A Sharpe of 1.335, CAGR of 32.4%, worst year of +4.6% in 2022, and max drawdown of -25.9% over only 2019-2024 is attractive but suspicious enough to demand decomposition. It may be genuine. It may mostly be the SPY > 200-day MA and VIX < 30 regime gate. It may be a lucky six-year window. It may be parameter leakage. You do not know yet.

### What I would do immediately

Run the following ablation matrix before trusting the factor portfolio:

| Test | Purpose | Required output |
|---|---|---|
| Equal-weight Russell 1000 sample, no factor | Baseline universe return | CAGR, Sharpe, MaxDD, turnover |
| SPY buy-and-hold | Real investor benchmark | CAGR, Sharpe, MaxDD, beta |
| SPY with same regime gate | Tests whether gate is the whole alpha | Same metrics |
| QQQ with same regime gate | Tests if AI/big-tech beta explains result | Same metrics |
| Factor score, no regime gate | Is factor alpha independent? | Same metrics |
| Regime gate only, random top-N | Does selection matter? | Distribution across random seeds |
| Momentum-only sleeve | Is quality/value adding anything? | Same metrics |
| Quality-only sleeve | Tests quality alpha | Same metrics |
| Value/PE-only sleeve | Tests valuation contribution | Same metrics |
| Full factor + regime | Candidate production strategy | Full tear sheet |

If full factor + regime does not beat SPY with the same regime gate, the factor layer is not adding enough value. Then the strategy is really a market exposure timing model, not a stock-selection model.

### How it should be traded if kept

The factor portfolio should not use a 0.5x ATR single-name stop. For a monthly factor portfolio, the base execution should be:

- Rebalance monthly, possibly weekly if turnover/cost tests justify it.
- Equal weight or volatility-scaled weights across 10-25 names.
- No single-name ATR stop by default.
- Portfolio-level risk-off rule allowed, but it must be tested as part of the contract.
- Optional severe gap/catastrophe rule for single-name idiosyncratic events, but not a tight tactical stop.
- Sector caps and single-name caps are appropriate.
- Cash is allowed and should earn the assumed cash/T-bill rate in the backtest.

A 0.5x ATR stop is likely too tight for this signal. It clips normal volatility, converts a slow signal into noise trading, and increases turnover. If you want stops, test wide portfolio-aware stops such as 2.5x-4.0x ATR, trailing drawdown bands, or rebalance-only exits. But the default should be: hold until the next rebalance unless the portfolio regime contract says to reduce exposure.

## 5. LambdaRank - not the right primary architecture as currently used

LambdaRank is not inherently wrong. Cross-sectional ranking is common in quant equity. But your use case has three mismatches.

First, you are long-only with a small account. A ranker naturally produces relative winners. Professional rankers are often monetized through long-short portfolios, market-neutral books, or broad diversified baskets. You are selecting a few long positions. A relative winner in a bad tape can still be a bad absolute trade.

Second, your live system has a cash option, but the label does not. The model is forced to rank every day. It cannot say, "Do not trade." Your regime and opportunity gates try to bolt that decision on afterwards, but the model itself was not trained on the cash decision.

Third, NDCG@K is too far from the realized objective. NDCG can improve while trade Sharpe degrades because the model is optimizing rank purity, not net expected utility under the actual execution path.

The repeated gate failures are not surprising. Fold 2 is not a tuning problem. It is a target-definition problem. You are asking a ranker to find relative winners during a regime where the correct long-only action may be low exposure or cash.

### What to replace it with

Use a two-stage architecture:

1. Exposure controller: decides whether the system should be risk-on, reduced-risk, or cash-heavy.
2. Selector: ranks candidates only when exposure is allowed.

For the selector, start simpler than LambdaRank:

- Rule-based factor score as baseline.
- Logistic or gradient boosting model as a meta-labeler on top of the factor candidates.
- Objective: probability that this candidate beats a hurdle under the actual holding/execution policy.
- Hurdle: cash return + transaction cost + risk premium, or SPY/sector benchmark + cost.
- Output: expected return, expected downside, and confidence, not just a rank.

The ML model should initially be a veto or sizing overlay, not the primary alpha engine. Its job should be: "Among names the factor model likes, which ones should we avoid or overweight?" That is a more learnable problem with less overfit risk.

## 6. The 2022 Fold 2 problem - what professionals do

Professionals do not solve a bear market by forcing a long-only ranker to buy the least-bad stocks every day. They use one or more of the following:

- Reduce gross exposure.
- Go to cash or T-bills.
- Hedge with index futures/options.
- Run market-neutral long-short portfolios.
- Rotate to defensive sectors or low-volatility factors.
- Use separate regime-specific risk budgets.
- Change the objective from raw return to risk-adjusted or benchmark-relative return.

Because you are long-only and apparently not using shorts or options, your real bear-market tool is exposure control. That means Fold 2 should not be judged only by whether the selector found positive individual trades. It should be judged by whether the system avoided bad exposure.

Regime-conditional models may help, but they are not the full answer. The correct design is regime-conditional portfolio policy:

- RISK_ON: allow factor/swing exposure.
- NEUTRAL: reduce position count and size, require higher hurdle.
- RISK_OFF: cash or very limited defensive exposure unless a strategy has proven positive expectancy in risk-off.

For labels, stop using "top quintile regardless of absolute return" as the sole target. Better labels:

- `positive_excess_return`: 1 if forward return exceeds cash + cost + hurdle.
- `benchmark_excess_return`: 1 if forward return beats SPY or sector ETF by hurdle.
- `expected_utility`: forward return minus lambda times drawdown/volatility/MAE.
- `trade_policy_return`: realized return under the exact entry/exit rules you will use.
- `no_trade`: explicit class when the opportunity set is poor.

The most important change is that the label must allow "cash is better than this trade."

## 7. Walk-forward honesty - good foundation, but not enough yet

You have several good ingredients: expanding train windows, purge/embargo, transaction costs, PIT regime scores, and some survivorship-bias work. That is better than most retail algo projects.

But an institutional-grade review would still flag the following gaps.

### 7.1 Fundamental data may not be point-in-time enough

You mention fundamentals fetched quarterly. The key question is not the fiscal quarter. It is when the data became available to the market and when your pipeline would have known it. If FMP gives restated or current fundamentals without proper report-date availability, you may leak information. Use report filing dates, ingestion timestamps, and conservative lags. If you cannot guarantee PIT fundamentals, run price-only and fundamentals-lagged versions as a sanity check.

### 7.2 Universe construction still needs proof

`pit_union("russell1000", fold_start, fold_end) + DB historical symbols` is directionally good, but you need documented handling for:

- Delisted symbols and delist returns.
- Mergers, bankruptcies, ticker changes.
- Corporate actions.
- Names that were not tradable at the time.
- ETF/ADR/special share class contamination.

Survivorship bias is not fixed until failed names have realistic terminal returns and tradability constraints.

### 7.3 Fill assumptions are too crude

"Entry at next day open simulated as previous close x 1.001" is not acceptable for final validation. Use actual next-day open where possible. If you do not trust open prints, use next open plus spread/slippage model, or use first 5-minute VWAP where intraday bars exist.

For stops and targets using daily bars, you have an ordering problem. If both target and stop are inside a daily high/low range, you do not know which hit first unless you have intraday data. A conservative simulator should assume the adverse event happened first, or use intraday bars for stop/target strategies.

### 7.4 Repeated HPO creates selection bias

Nine campaigns and many prior runs create data snooping. Even when each run is clean, repeatedly trying labels, gates, features, and hyperparameters on the same 2020-2026 period turns the period into a development set. You need a locked final test window or nested walk-forward model selection.

Recommended split discipline:

- Research/dev: older data.
- Model selection: walk-forward validation.
- Final holdout: untouched recent period, ideally 2025-2026 or a later reserved window once enough data exists.
- Live shadow: paper-only forward test with no parameter changes counted as validation.

### 7.5 Paper trading is plumbing validation, not alpha validation

Four weeks of paper Sharpe is nearly meaningless for strategy validation. It can reveal order bugs, PDT/margin issues, data outages, and live/backtest drift. It cannot prove edge. A single month has too few independent observations, especially for swing trading.

### 7.6 Sharpe should not be the only gate

For this system, require a full tear sheet:

- CAGR and annualized volatility.
- Sharpe and Sortino.
- Max drawdown and drawdown duration.
- Worst month and worst quarter.
- Beta to SPY and QQQ.
- Alpha versus SPY, QQQ, and regime-gated SPY.
- Turnover and average holding period.
- Trade count and capacity.
- Exposure-adjusted returns.
- Hit rate, average win/loss, payoff ratio.
- Slippage sensitivity.
- Performance by regime, sector, month, and volatility bucket.

The current profit-factor bug matters operationally because it reduces trust in the reporting layer. Even if Sharpe is correct, misleading diagnostics lead to wrong decisions. Fix it.

## 8. Execution model - where the current system is most broken

The execution model is not a detail. It defines the strategy.

### Factor portfolio execution

Validated contract:

- Monthly rebalance.
- Equal weight.
- No single-name ATR stop.

Current live/simulator contract:

- Daily selection path.
- ATR stop/target.
- Model rescore exit.
- Max hold.

These are different strategies. Do not compare their Sharpes as if they are the same thing.

### Swing trade execution

If you want ATR target/stop swing trading, then train on ATR target/stop outcomes. The label should be: "Given entry at next open and this exact stop/target/max-hold policy, what is the realized R multiple or net return?" Not raw 20-day return. Not cross-sectional rank. Not target based on close when execution happens at next open.

Also, 0.5x ATR stop is very tight. In equities, normal noise can easily hit that before a multi-day signal plays out. Test 1.0x, 1.5x, 2.0x, 3.0x ATR stops and no-stop time exits. Do not choose by best Sharpe alone. Choose by stability across folds and by MAE/MFE evidence.

### Intraday execution

Intraday needs a separate engine based on 1-minute or 5-minute bars with realistic spread, slippage, order latency, and rejected-order handling. Daily OHLCV assumptions are irrelevant. At 30-minute to 3-hour holds, transaction cost and adverse selection dominate.

You should also log every order lifecycle event: signal time, quote/bar snapshot, order submitted, accepted, partial fill, fill price, cancel/replace, exit trigger, and realized slippage versus decision price.

## 9. Intraday at $20k - viable as research, unattractive as a business

Intraday long-only stock trading with $20k can be done as a learning/research exercise. As a money-making business, it is a low-probability path unless you have a very strong execution edge, very clean data, and strict trade filtering.

Important current constraint: FINRA has adopted changes replacing the old PDT framework with new intraday margin requirements effective June 4, 2026, with broker transition considerations. Alpaca says it will implement the new framework for Trading API users on June 4, 2026. Until implementation, Alpaca's current documentation still describes PDT protections, including paper trading protections and order rejection behavior for accounts below $25k. After implementation, the bottleneck shifts from day-trade count to real-time intraday margin and buying-power checks.

Even after the PDT change, the math remains hard:

- If average round-trip friction is 20-40 bps all-in for short-horizon trades, a 50 bps gross edge can disappear quickly.
- A 0.5%-2.0% target sounds large, but realized average profit after stops, time exits, partial fills, and missed exits may be much smaller.
- If the strategy trades only a few names per day, realized Sharpe will be noisy.
- If it trades many names, turnover and operational risk rise.
- Top-20% one-hour forward return labels again force a relative winner even when the right answer is no trade.

My recommendation: keep intraday as a separate research sleeve, capped and paper-only, until it proves out under a harsher simulator. Do not let it distract from fixing swing/factor architecture.

## 10. News Intelligence Service - useful as risk context, not alpha yet

The NIS is architecturally interesting, but it should not be treated as proven alpha until you have a point-in-time historical event dataset and can show incremental predictive value net of costs.

Use it initially for:

- Blocking obvious idiosyncratic risk: fraud, bankruptcy, catastrophic guidance, halted names, major litigation.
- Macro event handling: CPI, FOMC, NFP, Fed speeches.
- Sizing down around high-uncertainty events.
- Explaining why a trade was blocked or reduced.

Do not let an LLM generate discretionary buy/sell decisions directly. The output should be structured, bounded, and auditable:

```json
{
  "symbol": "AAPL",
  "published_at": "...",
  "event_type": "earnings_guidance",
  "direction_score": -0.4,
  "materiality_score": 0.8,
  "confidence": 0.7,
  "action_policy": "size_down",
  "expires_at": "...",
  "rationale": "short bounded explanation"
}
```

The RM can use `action_policy`; the PM can use the score as a feature only after the historical backtest proves it helps.

## 11. What I would build next - prioritized roadmap

### Phase 0 - Stop the bleeding: enforce strategy/execution alignment

Timeline: immediate.

1. Add `StrategyContract` and require every simulation and live/paper run to specify one.
2. Create three separate contracts:
   - `factor_monthly_v1`
   - `swing_trade_atr_v1`
   - `intraday_5m_v1`
3. Disable ATR stops for `factor_monthly_v1` unless explicitly testing a stop variant.
4. Make PM output either target portfolio weights or discrete trade proposals, not both.
5. Fix profit-factor/trade-return propagation so reporting is trustworthy.

Pass condition: the same strategy contract produces the same decisions in research simulation and paper/live dry-run replay, except for expected fill/slippage differences.

### Phase 1 - Factor portfolio truth test

Timeline: next.

Run a clean, dedicated factor portfolio validation:

- Monthly rebalance, weekly rebalance, and quarterly rebalance.
- Top 5, 10, 20, 30 holdings.
- Equal weight versus volatility-scaled.
- No stop versus portfolio risk-off exit.
- Costs at 5, 10, 20, and 50 bps per side.
- Compare to SPY, QQQ, equal-weight universe, SPY regime gate, QQQ regime gate.
- Decompose factor score into momentum, quality, value, and combined.
- Include turnover, tax-unaware but turnover-aware metrics.

Pass condition: full factor portfolio beats the relevant regime-gated benchmark with stable performance across rebalance frequencies, costs, and subperiods. If it only wins in one narrow configuration, it is not production-ready.

### Phase 2 - Execution mismatch ablation

Run the same factor selections under:

- No stop, rebalance-only.
- 1.0x ATR stop.
- 1.5x ATR stop.
- 2.0x ATR stop.
- 3.0x ATR stop.
- Trailing stop.
- Portfolio-level drawdown stop.
- Regime-off liquidation only.

For each, produce:

- Average hold.
- Stop-hit rate.
- Target-hit rate if target exists.
- MAE/MFE distribution.
- Return by days-held bucket.
- Difference versus no-stop base.

Pass condition: any stop overlay must improve drawdown or tail risk without destroying CAGR/Sharpe and without relying on one fold. My prior: tight ATR stops will fail.

### Phase 3 - Replace forced-rank ML with cash-aware labels

Do not revive LambdaRank as the main model yet. Build a cash-aware supervised target:

- Entry: actual next open.
- Exit: exact strategy contract.
- Label: net realized return after costs or realized R multiple.
- Positive class: return > cash + cost + hurdle.
- Optional second target: return > SPY/sector + hurdle.
- Include no-trade examples.
- Train simple models first: logistic regression, calibrated XGBoost, LightGBM binary/regression.
- Use calibration curves and expected value buckets, not just AUC/NDCG.

Pass condition: top predicted EV bucket has stable positive realized net return across regimes and beats the factor baseline, not just random.

### Phase 4 - ML as meta-model, not alpha oracle

Once the factor portfolio has a valid baseline, train ML to improve it.

Candidate design:

- Input universe: top 50 factor candidates per rebalance date.
- Model target: which candidates improve next-period portfolio return or avoid large drawdowns.
- Output: veto, size down, normal, overweight.
- Features: factor components, recent volatility, market regime, sector, earnings/news risk, liquidity, crowding proxy.
- Evaluation: full portfolio backtest, not classifier metrics.

Pass condition: factor + ML overlay improves net portfolio metrics out of sample after adding turnover and complexity penalties.

### Phase 5 - Regime/exposure controller

Treat regime as an allocator, not a post-hoc filter.

Build a simple exposure model:

- 100% target gross in strong risk-on.
- 50% target gross in neutral.
- 0%-25% target gross in risk-off.
- Optional defensive ETF/T-bill/cash allocation.

Test against:

- SPY > 200-day MA.
- VIX threshold.
- Breadth threshold.
- Combined model.
- Your XGBoost regime model.

Pass condition: regime controller improves drawdown and risk-adjusted return versus always-invested without eliminating most upside through over-filtering.

### Phase 6 - Intraday viability test

Do not optimize intraday further until you build the correct simulator.

Requirements:

- 5-minute or 1-minute bars.
- Realistic spread/slippage model by symbol and time of day.
- Order rejection and buying-power simulation.
- No forced trades on low-opportunity days.
- Benchmark against simple ORB/VWAP/pullback baselines.
- Evaluate by time bucket: 09:45, 10:45, 13:00.
- Evaluate by volatility regime and market trend.

Pass condition: strategy has stable positive expectancy after conservative costs in multiple folds and is not dependent on one market regime.

## 12. Concrete implementation backlog

### P0 - Must do before trusting any result

1. Implement `StrategyContract`.
2. Fix trade return propagation and profit factor.
3. Split simulators into strategy-specific execution policies.
4. Add actual next-open fills for daily strategies.
5. Add daily-bar ambiguity handling for stop/target order: conservative ordering or intraday confirmation.
6. Add experiment registry: every run stores strategy contract, data snapshot, feature list, label spec, commit hash, parameters, costs, and metrics.
7. Add benchmark tear sheets automatically.

### P1 - Data correctness

1. Validate adjusted OHLCV handling.
2. Document split/dividend treatment.
3. Build delisted-symbol test cases.
4. Add PIT fundamental availability lag.
5. Add ticker-change mapping.
6. Save raw vendor snapshots for reproducibility.

### P2 - Portfolio/risk

1. Separate target portfolio construction from trade execution.
2. Add exposure-level controller.
3. Add sector caps and correlation caps by contract.
4. Add cash/T-bill return assumption.
5. Add turnover cap.
6. Add max rebalance drift threshold to avoid unnecessary trades.

### P3 - ML research

1. Create cash-aware labels.
2. Train simple baselines first.
3. Calibrate probabilities.
4. Evaluate by EV bucket.
5. Move to meta-labeling on factor candidates.
6. Only then revisit rankers.

### P4 - Live/paper operations

1. Shadow-mode replay: store every signal and what would have happened under each strategy contract.
2. Live/backtest drift report: compare expected fill, actual fill, slippage, rejects, missing data.
3. Kill switch remains deterministic.
4. LLM explanations must not override RM.
5. Go-live requires both alpha validation and operational validation.

## 13. Revised go-live gates

The current idea of going live after four weeks of paper Sharpe > 0.5 and max drawdown < 5% is too weak. It can pass by luck.

Better gates:

### Research gate

- Minimum 5+ years if daily strategy; more is better.
- Multiple regimes included.
- Beats SPY/QQQ and regime-gated benchmark after costs.
- Stable across folds; no single fold contributes most profits.
- Cost sensitivity survives 2x and 4x assumed slippage.
- Turnover is economically realistic.
- No unresolved data leakage concerns.

### Paper trading gate

- Minimum 3 months for intraday plumbing, 6 months preferred for swing/factor observation.
- All orders reconcile.
- No unexplained rejected orders.
- Live feature values match offline replay.
- Fill slippage within modeled ranges.
- Drawdown and exposure match contract.
- No manual intervention needed except approved overrides.

### Initial live gate

- Start with tiny capital or tiny notional, not full $20k.
- Use one strategy contract only.
- No strategy changes during the validation window.
- Increase capital only after operational and performance drift are acceptable.

## 14. Direct answers to your specific questions

### 1. Is LambdaRank the right architecture?

Not as the primary architecture for this account and objective. It can rank relative winners, but your long-only system needs a cash-aware expected-return decision. Ranking is useful after exposure is approved and after a candidate set exists. It is not the right top-level decision engine.

### 2. Is the walk-forward honest?

Partly. The purge/embargo and expanding windows are good. But the validation is not fully honest until the execution policy, labels, fills, universe, fundamental availability, and live strategy contract are aligned. The biggest honesty problem is not classic leakage; it is strategy identity drift.

### 3. How should Fold 2 / 2022 be handled?

Do not force long exposure. Add a cash/no-trade decision and regime-based exposure control. A bear market should not require the selector to find magic long winners. The system should be allowed to say, "This is not our opportunity set."

### 4. Factor portfolio vs ML?

Given your constraints, factor portfolio first is the correct move. ML should become a meta-layer that improves selection, sizing, or veto decisions. Giving up on the current ML campaign is not giving up too early; it is stopping an invalid objective. Revisit ML after you have a correct baseline and correct labels.

### 5. Is entry at next open and ATR stop/target realistic?

Next-open entry is realistic if you use actual next-open data plus slippage. Simulating it as previous close x 1.001 is too crude for final validation. ATR stops/targets are realistic for a swing-trade strategy, but not for a monthly factor portfolio. Daily close-based stops are not the same as live stop orders and can miss gap/slippage behavior.

### 6. Is intraday viable at $20k?

Possible as a research project, but not the best first money-making path. Regulatory friction is changing in June 2026, and Alpaca says it will implement the new intraday margin framework, but execution friction, noise, and small sample size remain. Intraday should stay paper-only and separate until it clears a much harsher simulator.

### 7. What would I build next?

I would build, in order:

1. Strategy contracts and simulator alignment.
2. Clean factor portfolio validation and ablation.
3. Execution stop/no-stop mismatch study.
4. Cash-aware labels for swing ML.
5. ML meta-model on factor candidates.
6. Regime/exposure allocator.
7. Intraday simulator and only then intraday model improvement.

## 15. The brutal bottom line

You are not one hyperparameter away. You are one abstraction away.

The project will keep wasting cycles until every model is trained and judged on the exact trade policy it will execute. Right now, the strongest-looking strategy is the factor portfolio, but it is being damaged by routing it through a swing-trading stop engine. The ML ranker is failing because it answers a relative ranking question when the account needs an absolute long/cash decision. The intraday sleeve is interesting but likely a distraction until the simulator and broker constraints are more realistic.

The best next move is not another HPO run. It is to lock strategy identity, prove the factor baseline honestly, then let ML earn its way back in as a small, measurable improvement over that baseline.

## 16. References used for regulatory/market-structure notes

- FINRA, "Understanding the New Intraday Margin Requirements," April 20, 2026. Key points: changes effective June 4, 2026; firms may have transition period through October 20, 2027; new framework removes the $25,000 PDT minimum and replaces trade-count PDT logic with intraday margin monitoring. URL: https://www.finra.org/investors/insights/intraday-margin-requirements
- Alpaca, "FINRA Retires the PDT Rule: Introducing Alpaca's New Intraday Margin Framework," April 27, 2026. Key points: Alpaca says it will implement the new framework on June 4, 2026 and remove PDT-related restrictions/fields. URL: https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/
- Alpaca documentation, "User Protection." Key points: current PDT protections and paper-trading protections are described for accounts below $25,000, including order rejection behavior. URL: https://docs.alpaca.markets/us/docs/user-protection
- SEC, "SEC Chair Gensler Statement on Upcoming Implementation of T+1 Settlement Cycle," May 21, 2024. Key point: U.S. securities market moved to T+1 settlement on May 28, 2024. URL: https://www.sec.gov/newsroom/press-releases/2024-62

## 17. Suggested prompt to feed into other LLMs with this review

Use the following:

"Attached is a senior quant review of my MrTrader system. Critique it. Do not simply agree. Identify where the reviewer may be too conservative, where the proposed experiments are incomplete, and what alternative strategy designs I should consider under these constraints: $20k account, long-only equities/ETFs, Alpaca API, Python/FastAPI/SQLite, daily swing plus intraday research, no professional tick/Level 2 data, and a need for realistic walk-forward validation. Provide a prioritized implementation plan and explicitly separate research alpha, execution realism, risk management, and production engineering." 
