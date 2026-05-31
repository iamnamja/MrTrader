# MrTrader — Full System Review and Critical Roadmap

**Generated:** 2026-05-05  
**Purpose:** Independent critique of the MrTrader automated trading system based on the supplied full-system review prompt.  
**Intended use:** Upload to other LLMs or reviewers to synthesize a concrete next-step roadmap.

---

## Executive Verdict

MrTrader is **not yet a production-grade trading system**. It is a promising research/prototype platform with unusually good engineering discipline for a solo developer, but the current alpha evidence is not strong enough to justify live capital beyond very small pilot exposure.

The biggest problem is not XGBoost vs LightGBM, or whether another technical indicator is missing. The biggest problem is that **the research/backtest loop does not yet faithfully simulate the actual production decision loop**.

The current champion models also fail on the current data window:

- **Swing v142** originally passed on an older window with average Sharpe around **+1.181**, but now fails on the current data window with average Sharpe around **+0.310**.
- **Intraday v29** originally passed on an older window with average Sharpe around **+1.830**, but now fails on the current data window with average Sharpe around **-0.327**.

That means the current “champion” status is stale. These models should be treated as historical baselines, not production-valid champions.

The system currently has three separate realities:

1. **Model research reality** — walk-forward tests score the model in isolation.
2. **Production decision reality** — PM gates, RM gates, re-scoring, caps, blackouts, budget limits, and execution behavior alter which trades actually happen.
3. **Broker/execution reality** — Alpaca fills, state reconciliation, stop/target drift, pending orders, and bars-held lifecycle determine actual P&L.

Those three need to become one replayable historical system before the model numbers are trustworthy.

**Bottom line:** Do not go live yet in meaningful size. Fix the validation architecture first, then revisit model/feature work.

---

## 1. The 10 Biggest Issues

### 1.1 The current champions are no longer valid champions

The active swing and intraday models both failed when re-evaluated on the current data window. This is not a minor concern. It means the prior promotion process did not protect against regime-specific overfitting.

Interpretation:

- v142 and v29 are not necessarily useless.
- But they are no longer production-valid.
- They should be retained as baselines, not deployed as trusted models.

The right question is not “how do we fix v142 or v29?” It is: **how did models that later failed become champions in the first place?**

---

### 1.2 Walk-forward tests the model, not the actual strategy

The current tier-3 walk-forward evaluates model output in isolation. It does not replay the full production trading lifecycle:

- PM opportunity score
- PM candidate throttling
- PM re-score before execution
- RM sequential gates
- Position caps
- Earnings blackout
- Macro-event blackout
- Portfolio heat
- Budget limits
- Execution simulation
- Stop/target/trailing-stop lifecycle
- Broker-style fills

This is the most important architectural gap.

The promotion gate should evaluate the actual trading system, not just the model:

```text
historical bars
→ PM scan
→ PM opportunity score
→ proposal generation
→ RM gates
→ Trader execution simulator
→ stop/target/trailing-stop lifecycle
→ broker-style fills
→ full portfolio P&L
```

Until this exists, the reported model Sharpe is not the same thing as strategy Sharpe.

---

### 1.3 The regime failure is real

Both swing and intraday models degrade sharply during the 2025 shock/whipsaw period.

This should not be treated as an isolated bad fold. It is evidence that the models have a regime-dependent edge.

Likely current behavior:

- Better in calmer/trending regimes.
- Worse in high-volatility, macro-dominated, whipsaw regimes.
- Vulnerable when broad correlation rises and stock-specific selection becomes less useful.

A production system does not need to make money in every regime, but it must know when its edge is absent.

The correct choices in hostile regimes are:

1. Stop trading.
2. Reduce size materially.
3. Switch to a different strategy/model.
4. Trade a different style, such as mean reversion, hedged relative value, or no-trade mode.

---

### 1.4 The swing label is too narrow

The swing model is described as targeting 3–15 day holds, but the label asks whether price hits a 2.0×ATR target before a 1.2×ATR stop within 5 days.

That label may be suitable for a tactical 5-day path-quality model, but it is narrower than the intended strategy.

Problems:

- It rewards fast target hits.
- It may reject slower but valid swing trades.
- It turns a rich path into a binary outcome.
- It discards near-winners, smooth positive trades, and expected R information.
- It may not match the actual live exit policy.

The label is not obviously wrong, but it should be treated as one candidate label, not the definitive swing objective.

---

### 1.5 The intraday label is misaligned with long-only execution

The intraday model labels the top 20% of symbols by 2-hour forward return each day.

That is a cross-sectional ranking label. But the production strategy appears to enter long trades, not a true long/short, beta-neutral rank portfolio.

This creates a problem:

- On a bad market day, the top 20% of symbols may still lose money after spread and slippage.
- The model can be right cross-sectionally but wrong economically.
- A ranking model naturally wants a long/short or beta-neutral implementation.
- If the system is long-only, the label needs an absolute expected-return hurdle.

A better long-only label would be:

```text
positive = top-ranked cross-sectionally
           AND expected 2-hour return > estimated cost + slippage + risk buffer
```

---

### 1.6 There are too many features without enough ablation evidence

The swing model has around 84–89 features. Intraday v41 has 61 features. Feature richness is not automatically bad, but the evidence suggests feature accumulation is not currently improving robustness.

Examples:

- NIS and macro NIS did not rescue swing performance.
- v41 improved the bad intraday fold relative to v29 but worsened the later fold.
- SPY market-wide features were neutralized by cross-sectional normalization.

The system may be in “feature soup” territory: many intuitively reasonable signals, but unclear incremental out-of-sample economic value.

Before adding features, run disciplined feature-family ablations.

---

### 1.7 Three folds are not enough confidence

Three expanding walk-forward folds are directionally good, but not sufficient given:

- Many model versions
- Many feature additions
- HPO variance
- Multiple labels attempted
- Regime instability
- Sparse history of crisis regimes

The prompt notes that Optuna HPO variance can be around 2.0 Sharpe on identical features. That is a major warning. It means model-selection noise may dominate true signal.

The validation framework should add:

- Purged/embargoed cross-validation where labels overlap.
- More time splits.
- Regime splits.
- Deflated Sharpe or model-selection adjustment.
- Locked final holdout period.
- Stability reports by sector, regime, and volatility bucket.

---

### 1.8 The PM opportunity score is sensible but unproven

The opportunity score is directionally a good idea. It throttles trading based on VIX level, VIX trend, SPY trend, and SPY momentum.

However, it needs to be validated as part of the strategy, not treated as an external safety assumption.

Key questions:

- Would v142 have survived if the opportunity score had been applied historically?
- Would v29/v41 intraday performance have improved during Apr–Oct 2025?
- Were the thresholds chosen before or after seeing the bad regime?
- Does the score reduce losses without cutting off too much profitable activity?
- Does it work across 2022, 2024, 2025, and 2026 separately?

If the opportunity score was designed after seeing the bad fold, it must be treated as a new model component requiring fresh out-of-sample validation.

---

### 1.9 Alpaca/broker state must become the source of truth

The prompt identifies DB/Alpaca state divergence as a known issue. This is not a minor operational bug. It is a live-trading risk.

If the database and broker disagree, the system can:

- Double-enter.
- Fail to exit.
- Exit the wrong size.
- Miscount bars held.
- Apply stale stops.
- Think risk is lower than it is.
- Violate budget or PDT constraints.

For live trading, Alpaca must be the execution authority. The local DB should be an event log, cache, and audit trail — not the final source of truth for open positions.

---

### 1.10 The system may be over-engineered relative to alpha maturity

The PM → RM → Trader architecture is good. But the engineering maturity is ahead of the statistical evidence.

The next phase should not be more agents, more features, or more model families.

The next phase should be:

```text
Prove the strategy lifecycle end-to-end.
Then improve model alpha.
Then scale architecture.
```

---

## 2. Critical Assessment

### 2.1 What is fundamentally weak?

The fundamental weakness is **research-production mismatch**.

The live system includes PM scanning, PM opportunity scoring, PM re-scoring, RM gates, portfolio-level limits, earnings/macro blackouts, Trader execution, stop/target lifecycle, and broker reconciliation.

But the model gate is based on model-isolated walk-forward results.

This means the promotion system does not test the actual thing that will trade.

Secondary weaknesses:

- Regime dependence.
- Label/execution mismatch.
- Feature accumulation without sufficient ablation discipline.
- Sparse NIS history.
- Incomplete broker source-of-truth design.
- Insufficient multiple-testing controls.

---

### 2.2 What would a serious quant immediately ask?

A serious systematic reviewer would ask:

1. Are results net of realistic spread, slippage, partial fills, and rejected orders?
2. Is the universe survivorship-bias free?
3. Are corporate actions, delistings, symbol changes, and missing data handled correctly?
4. Are all fundamentals, earnings, analyst, options, and news timestamps point-in-time?
5. Is every model version selected using only information available at that time?
6. How many variants were tried before choosing the champion?
7. What is the deflated Sharpe, not just the raw Sharpe?
8. Does performance survive by regime, sector, volatility bucket, and calendar period?
9. Does the live decision stack match the research simulator?
10. What is expected degradation from paper to live?
11. Does the model beat simple baselines after costs?
12. What is the capacity of the strategy at $20k, $100k, $1M?
13. How robust is performance to slippage assumptions?
14. Does the strategy have hidden beta, sector, or momentum exposure?
15. What happens if the websocket disconnects or broker state conflicts with local state?

Several of these are currently unanswered or only partially answered.

---

### 2.3 Which design decisions likely limit alpha most?

The biggest alpha limiters are:

1. **Swing entry pre-filters** — the ML model only scores RSI_DIP and EMA_CROSSOVER candidates. It cannot discover other entry patterns.
2. **Intraday fixed entry time** — always entering at bar 12 assumes the opportunity starts 60 minutes after open.
3. **Intraday fixed 2-hour hold** — the model cannot learn optimal exit timing.
4. **Cross-sectional label with long-only execution** — the label solves rank, but execution requires positive expected return.
5. **Regime handled outside validation** — the opportunity score may help, but it is not part of the model promotion test.
6. **Sparse NIS history** — news features mostly missing historically cannot be relied on as core model drivers.
7. **Insufficient cost modeling** — especially dangerous for intraday trading.

---

## 3. The Regime Problem

### 3.1 What kind of problem is Apr–Oct 2025?

It is mostly a **strategy robustness problem**.

Breakdown:

| Component | Contribution |
|---|---|
| Model problem | Yes — single global models are not robust to regime shifts |
| Feature problem | Yes — features may not capture dispersion, liquidity, breadth, and correlation sufficiently |
| Label problem | Yes — labels do not encode regime-specific payoff shape well enough |
| Data problem | Partly — only a few years of history and limited crisis regimes |
| Unavoidable? | No — losses may be unavoidable, but trading through them at normal size is avoidable |

The model does not need to make money in every regime. It needs to identify when its edge is absent.

---

### 3.2 Should regime handling be inside or outside the model?

Use both, in layers.

Recommended structure:

```text
Layer 1: Hard no-trade / risk-off regime gate
Layer 2: Strategy-level risk budget adjustment
Layer 3: Model features include regime context
Layer 4: Separate regime models only after enough data exists
```

Do not immediately train many regime-specific models. The system likely does not have enough observations per regime to avoid overfitting.

Start with a regime-aware allocator:

| Regime | Action |
|---|---|
| Low VIX, SPY above trend, normal breadth | Normal trading |
| Moderate VIX, unstable trend | Reduce candidates and size |
| High/rising VIX, weak breadth, high correlation | No new longs or tiny pilot only |
| High dispersion, controlled VIX | Allow intraday relative-strength trades |
| Macro event window | Block or reduce |

---

### 3.3 Improve the opportunity score

Current opportunity score inputs:

- VIX level
- VIX trend
- SPY above/below MA20
- SPY 5-day return

Add diagnostics before adding complex ML:

1. VIX bucket.
2. VIX percentile.
3. Realized volatility percentile.
4. SPY above/below MA20, MA50, MA200.
5. Market breadth: percent of universe above MA20/50.
6. Cross-sectional dispersion.
7. Average pairwise correlation.
8. Sector leadership concentration.
9. Overnight gap magnitude.
10. Intraday first-hour range percentile.
11. Earnings season density.
12. Macro-event proximity.

The most important missing metric may be **cross-sectional dispersion**. Intraday ranking strategies need dispersion. If all stocks move together because macro dominates, stock selection alpha weakens.

---

### 3.4 Required regime report

Create a report that groups every historical trade by:

```text
VIX bucket: <15, 15–20, 20–25, 25–30, >30
VIX trend: rising / falling
SPY trend: above/below MA20, MA50, MA200
Market breadth: strong / neutral / weak
Cross-sectional dispersion: low / mid / high
Correlation regime: low / high
Gap regime: normal / large gap
Macro window: yes / no
Earnings window: yes / no
```

For each bucket, show:

- Trades
- Win rate
- Average R
- Median R
- Sharpe
- Max drawdown
- Profit factor
- Stop-hit rate
- Target-hit rate
- Average adverse excursion
- Average favorable excursion

This will likely reveal more than another retrain.

---

## 4. Label Design Review

### 4.1 Swing label review

Current swing label:

```text
Positive = price hits 2.0×ATR target before 1.2×ATR stop within 5 days
```

This is better than a naive forward-return label because it is path-aware. But it does not fully match the intended 3–15 day swing system.

Main problems:

- It may bias toward quick pop trades.
- It discards expected R.
- It discards near-winners.
- It may reject valid slower-developing trades.
- It may not match live exit policy.

#### Swing labels to test

**Label A — Triple-barrier multiclass**

```text
+1 = upper barrier hit first
 0 = time barrier hit first
-1 = lower barrier hit first
```

**Label B — Expected R regression**

```text
label = realized R over actual execution policy
```

Winsorize extreme outcomes so outliers do not dominate.

**Label C — Meta-label**

Keep RSI_DIP and EMA_CROSSOVER as candidate generators, but train the model to answer:

```text
Given that the rule fired, should this trade be taken?
```

This is probably the most practical first step because the current system already uses rule-based candidate generation.

---

### 4.2 Intraday label review

Current intraday label:

```text
Positive = symbol is in top 20% of 2-hour forward return for that day
```

This is useful for ranking, but dangerous for long-only execution.

The top 20% of stocks can still be unattractive if:

- SPY is selling off.
- Spreads are wide.
- The top-quintile return is too small.
- Volatility is high but directionless.
- Macro correlation dominates stock-specific behavior.

#### Intraday labels to test

**Label A — Rank plus absolute hurdle**

```text
positive = top 20% cross-sectionally
           AND 2-hour return > estimated trading cost + buffer
```

**Label B — Residual return label**

```text
future residual return = stock 2h return - beta × SPY 2h return
```

Then label top residual performers, not raw return performers.

**Label C — Realized execution R**

Simulate the actual stop/target/time exit and label:

```text
label = realized R after stop/target/time exit
```

The prior realized-R attempt failed, but it should not be abandoned until the implementation and feature suitability are audited.

**Label D — Long/short rank label**

If using a pure cross-sectional rank label, test the natural implementation:

```text
Long top N
Short bottom N
Beta neutral
Sector constrained
Exit after 2 hours
```

Even if shorting is not desired live, this tells whether the model is learning true cross-sectional structure.

---

## 5. Feature Review and Suggestions

### 5.1 Stop adding features until ablation is complete

Before adding features, run feature-family ablations.

Test:

```text
Base technicals only
Base + volume
Base + relative strength
Base + volatility
Base + fundamentals
Base + earnings
Base + WQ alphas
Base + regime
Base + NIS
Base + sector
Full model
```

Evaluate each by:

- Average Sharpe
- Minimum fold Sharpe
- Regime Sharpe
- Decile monotonicity
- Turnover
- Drawdown
- Hit rate
- Average R
- Feature stability

If a feature group does not improve minimum-fold or bad-regime behavior, remove or isolate it.

---

### 5.2 Swing feature suggestions

Focus less on adding more technical indicators and more on robust factor families.

#### Cross-sectional momentum

- 12-month minus 1-month momentum.
- 6-month momentum.
- 3-month momentum.
- 1-month reversal.
- Sector-relative momentum.
- Industry-relative momentum, if industry mapping exists.

#### Quality

- Gross profitability.
- Operating margin.
- Return on invested capital.
- Free cash flow margin.
- Accruals.
- Earnings stability.
- Debt service burden.

#### Value

Only if point-in-time correct:

- Earnings yield.
- Free cash flow yield.
- Sales yield.
- EBITDA/EV.
- Book-to-market.

#### Earnings drift

- Post-earnings announcement drift window.
- Days since last earnings surprise.
- Analyst revision breadth.
- Analyst revision acceleration.
- Guidance raise/cut flag, if available.

#### Liquidity and crowding

- Dollar volume percentile.
- Turnover.
- Short interest.
- Borrow cost, if available.
- Institutional ownership change stability.
- Volume shock persistence.

#### Residualized technicals

Add residual versions of existing signals:

```text
stock return - beta × SPY return
stock return - sector ETF return
stock volatility / sector volatility
stock drawdown - sector drawdown
```

For swing, this is likely more valuable than adding another oscillator.

---

### 5.3 Intraday feature suggestions

The strongest missing area is market microstructure / liquidity / intraday context.

With only 5-minute bars, true order book alpha is unavailable, but useful proxies exist.

#### Relative volume by time of day

Not just volume surge versus rolling average. Use:

```text
current cumulative volume / normal cumulative volume by same minute
```

A 9:45 volume spike and a 12:30 volume spike mean different things.

#### Opening auction/gap context

- Gap size versus ATR.
- Gap direction versus SPY gap.
- Gap direction versus sector ETF gap.
- Gap fill speed.
- Gap continuation after first 30/60 minutes.
- Prior-day close location in range.

#### Intraday sector-relative strength

At bar 12:

```text
stock return since open - sector ETF return since open
stock VWAP distance - sector ETF VWAP distance
stock first-hour range - sector first-hour range
```

#### Intraday market breadth

Feed market context separately, not through cross-sectional normalization:

- Percent of universe positive since open.
- Percent above VWAP.
- Percent breaking opening range high.
- Advance/decline ratio.
- Cross-sectional dispersion.
- Top-sector minus bottom-sector performance.

#### Liquidity/spread proxies

If quote data is unavailable:

- Average 5-minute dollar volume.
- Bar high-low spread proxy.
- Close-to-close micro-volatility.
- Zero-volume bars.
- Slippage proxy based on volume and volatility.
- ATR-normalized spread estimate.

#### Intraday reversal/continuation shape

- First-hour return divided by first-hour range.
- Close location within first-hour range.
- Number of VWAP crosses.
- Pullback depth after high.
- Recovery from low.
- Consecutive higher lows/lower highs.

---

### 5.4 Fix the cs_normalize design

The current intraday pipeline cross-sectionally normalizes features per day, which means market-wide features become zero.

Split features into two branches:

```text
Branch A: Cross-sectional stock features
- normalized across symbols

Branch B: Global market/regime features
- not normalized
- same for all symbols
- used by model as context or by outer gate
```

Start with a simple design:

- Keep the stock-selection model cross-sectional.
- Use an external regime/opportunity model to decide whether to trade.

Do not force market-wide features through cs_normalization.

---

### 5.5 NIS / news signal recommendation

Do not keep pushing NIS directly into the models yet.

The history is too sparse:

- Stock-level NIS only from May 2025 onward.
- Macro NIS only 259 days.

XGBoost can handle missing values technically, but that does not mean it can learn a stable news effect.

Use NIS first as a gate or overlay:

```text
If high-materiality negative NIS:
    block new long entries or reduce size

If high-materiality positive NIS:
    allow candidate but require price confirmation

If high already-priced-in score:
    suppress chasing

If high downside risk:
    tighten risk or block
```

Then run an NIS event study:

- NIS bucket.
- Forward 1h, 2h, 1d, 3d, 5d returns.
- By materiality.
- By already-priced-in score.
- By sentiment direction.
- By market regime.
- By sector.
- Net of SPY/sector move.

Only after NIS proves incremental value in event studies should it become a core model feature.

---

## 6. Architecture Review

### 6.1 PM → RM → Trader architecture

The architecture is conceptually good.

The separation is sensible:

```text
PM = opportunity generation
RM = permissioning and risk constraints
Trader = execution and lifecycle
```

Keep this structure.

The risk is not the high-level design. The risk is **async state drift**.

Queues can create stale decisions. The PM re-score loop is a good defense, but live trading requires stronger guarantees:

- Idempotent order IDs.
- Event-sourced order lifecycle.
- Broker reconciliation as first-class logic.
- Alpaca as source of truth.
- Local DB as audit/cache.
- Startup recovery test.
- Partial-fill handling.
- Cancel/replace handling.
- Duplicate-order prevention.
- Kill switch.
- Heartbeat monitor.
- “Do not trade if state uncertain” mode.

---

### 6.2 SQLite

SQLite is acceptable for research and small paper trading. It is not automatically disqualifying for a $20k pilot.

The key issue is not SQLite itself. The issue is whether order and position state transitions are:

- Transactional.
- Auditable.
- Recoverable.
- Broker-reconciled.

For small live pilot use, SQLite can be acceptable if:

- Alpaca remains source of truth.
- Every order event is written immutably.
- Reconciliation happens before trading starts.
- Reconciliation happens during the day.
- Any mismatch blocks new trading.

---

### 6.3 Walk-forward gate thresholds

Current gate:

```text
Swing: avg Sharpe > 0.8
Intraday: avg Sharpe > 1.5
Minimum fold > -0.3
```

Directionally fine, but incomplete.

Replace with a scorecard:

| Metric | Requirement |
|---|---|
| Average Sharpe | Must pass |
| Lower-confidence Sharpe | Must pass |
| Minimum regime Sharpe | Not catastrophic |
| Worst fold | Above threshold |
| Max drawdown | Below cap |
| Profit factor | Above threshold |
| Trade count | Sufficient |
| Turnover | Realistic |
| Cost-adjusted result | Must pass |
| Deflated Sharpe | Must pass |
| Decile monotonicity | Must pass |
| Live/paper shadow degradation | Must be acceptable |

The model should not pass because average Sharpe is good. It should pass because the distribution of outcomes is acceptable.

---

### 6.4 Should swing and intraday be combined?

No. Keep them separate.

They have different:

- Horizons.
- Features.
- Labels.
- Execution risks.
- Capacity.
- Failure modes.
- Regime sensitivities.

But combine them under a shared portfolio allocator.

The allocator should decide:

```text
Today:
- Swing budget: 0% / 25% / 50% / 70%
- Intraday budget: 0% / 10% / 20% / 30%
- Max new positions
- Max sector exposure
- Max single-name exposure
```

That allocator should be regime-aware.

---

### 6.5 Is XGBoost the right algorithm?

For now, yes.

Do not jump to LSTM or Transformer yet.

The current bottlenecks are:

- Label alignment.
- Regime handling.
- Validation realism.
- Feature stability.
- Execution simulation.
- Data quality.

A neural model would make those problems harder to diagnose.

Before neural models, add:

1. Regularized logistic regression baseline.
2. LightGBM/XGBoost comparison.
3. Simple rank model.
4. Monotonic constraints where sensible.
5. Calibrated probability layer.
6. Ensemble of simple + tree models only if each adds value.

Always ask: **does the complex model beat a dumb baseline after costs?**

---

### 6.6 Position sizing

Do not use full Kelly.

Kelly sizing is dangerous when edge estimates are unstable, and the current edge estimates are unstable.

Use volatility-targeted sizing with confidence and regime multipliers:

```text
base size = volatility target
adjustment = signal percentile / confidence bucket
cap = regime budget
cap = liquidity budget
cap = portfolio heat
```

Example:

| Signal bucket | Size |
|---|---|
| 60–70th percentile | No trade or tiny |
| 70–80th percentile | 0.25× base |
| 80–90th percentile | 0.50× base |
| 90–95th percentile | 0.75× base |
| 95th+ percentile | 1.00× base |

Then multiply by regime score:

```text
final size = base_risk × signal_multiplier × regime_multiplier
```

This is safer than Kelly.

---

## 7. Roadmap — Highest-Priority Next Steps

## Priority 1 — Build the full-fidelity historical replay engine

This is the highest-leverage task.

Build one simulator that replays history through the same architecture used live:

```text
For each historical day/bar:
    refresh data cache
    run PM scan
    compute opportunity score
    generate proposals
    run RM gates
    simulate Trader execution
    simulate stops/targets/exits
    write decision_audit
    produce portfolio P&L
```

The replay should support:

- Swing only.
- Intraday only.
- Combined portfolio.
- PM gates on/off.
- RM gates on/off.
- Cost/slippage assumptions.
- Different opportunity thresholds.
- Different position sizing.
- Regime reports.

No model should be promoted unless it passes this full replay.

---

## Priority 2 — Run brutally honest diagnostics on current models

For v142, v29, and v41, produce:

- P&L by regime.
- P&L by sector.
- P&L by symbol.
- P&L by signal decile.
- P&L by VIX bucket.
- P&L by SPY trend.
- P&L by opportunity score bucket.
- P&L by trade hold time.
- P&L by stop-hit/target-hit behavior.
- P&L by earnings proximity.
- P&L by spread/liquidity bucket.
- P&L before and after estimated costs.

Goal:

```text
Know exactly where the model makes and loses money.
```

The output may show something like:

```text
This model works only when:
- VIX < 20
- SPY above MA50
- breadth > 55%
- dispersion is medium/high
- no macro event
```

That is useful. That becomes the tradable regime.

---

## Priority 3 — Redesign labels to match execution

### Swing

Test:

1. Current binary path label.
2. Triple-barrier multiclass.
3. Realized R from actual execution.
4. Meta-label on RSI/EMA candidates.
5. Longer horizons: 5-day, 10-day, 15-day variants.

Pick the label that produces the most stable strategy P&L, not the best AUC.

### Intraday

Test:

1. Current top-20% rank label.
2. Top-20% plus positive absolute hurdle.
3. Residual return rank label.
4. Realized R execution label.
5. Long/short rank portfolio simulation.

If the current top-20% label only works as long/short, then either add shorting later or change the label.

---

## Priority 4 — Validate the opportunity score as a real strategy component

Backtest it explicitly.

Run:

```text
Model only
Model + opportunity score
Model + opportunity score + RM gates
Model + opportunity score + RM gates + costs
```

Compare:

- Total return.
- Sharpe.
- Drawdown.
- Trade count.
- Worst fold.
- Apr–Oct 2025 performance.
- Missed profits in good regimes.

If the opportunity score improves minimum-fold performance, it is valuable. If it simply avoids the known bad window because it was tuned after the fact, it needs fresh validation.

---

## Priority 5 — Fix broker/source-of-truth before live deployment

Before live trading, implement:

```text
Alpaca = source of truth
DB = event log / audit / cache
```

Required behavior:

- On startup, reconcile all positions and orders.
- If mismatch exists, block trading.
- During trading, reconcile periodically.
- Every order has a client order ID.
- Duplicate client order IDs are rejected.
- Partial fills update state correctly.
- Pending orders count toward exposure.
- Failed cancels are handled.
- If websocket disconnects, system enters safe mode.
- If local DB corrupts, system can recover from broker state.

This is mandatory before live deployment.

---

## 8. What to Stop Doing Right Now

### Stop adding NIS directly into the models

Not forever. Just for now.

The history is too sparse and the results are not convincing. Use NIS as an overlay/gate and run event studies first.

---

### Stop treating old champion results as meaningful

v142 and v29 failed on current data. They are no longer production-valid champions.

Keep them as baselines.

---

### Stop optimizing average Sharpe alone

Average Sharpe hid regime fragility.

Add:

- Worst-fold Sharpe.
- Worst-regime Sharpe.
- Max drawdown.
- Profit factor.
- Deflated Sharpe.
- Stability score.
- Live/paper degradation.

---

### Stop retraining before fixing validation

Nightly retraining is useful later. If the validation harness is incomplete, nightly retraining just creates more unstable artifacts.

---

### Stop adding complexity before proving simple baselines

Before another feature batch, compare against:

- RSI/EMA rules with no ML.
- Simple momentum rank.
- Simple residual momentum rank.
- Simple volatility-throttled version.
- Simple regime-gated version.

If XGBoost does not beat simple baselines after costs, the ML layer is not adding enough value.

---

## 9. Highest-Value Data Additions

### 9.1 Survivorship-bias-free universe

If the system uses today’s S&P 500 / NASDAQ 100 constituents historically, results are biased.

Need:

- Historical constituents.
- Delisted names.
- Symbol changes.
- Corporate actions.
- Accurate historical universe membership.

---

### 9.2 Better intraday liquidity/spread data

For intraday, spread and fill quality can erase the edge.

Add if possible:

- Bid/ask spread.
- Quote midpoint.
- NBBO.
- Opening auction data.
- Premarket volume/gap context.

---

### 9.3 Short interest / borrow / options-implied data

For swing, short interest and options data can identify crowded or unstable names.

Add only if point-in-time correct.

---

### 9.4 More PIT fundamentals

Useful, but not the immediate bottleneck.

---

### 9.5 News backfill

Worth doing only after event-study design is defined.

The issue is not the estimated $50–100 cost. The issue is whether timestamps, labels, and validation are clean enough to prove predictive value.

---

## 10. What Would Need to Be True for Systematic Fund Standard?

For institutional credibility, the system would need:

1. Full historical replay matching production logic.
2. Survivorship-bias-free data.
3. Strict point-in-time timestamping for every feature.
4. Transaction-cost and fill simulation.
5. Broker reconciliation and event-sourced order lifecycle.
6. Deflated Sharpe / multiple-testing adjustment.
7. Regime performance diagnostics.
8. Locked holdout performance.
9. Live paper shadow period with stable degradation.
10. Kill switch and operational monitoring.
11. Model versioning and reproducibility.
12. Clear feature ablation evidence.
13. Clear comparison against simple baselines.
14. Capacity and liquidity analysis.
15. Post-trade analysis loop.

MrTrader currently has pieces of this, but not the full package.

---

## 11. Risks the Developer May Be Underestimating

### 11.1 Paper trading fill quality

Paper fills can be much better than live fills. This is especially dangerous for intraday trading.

A small edge can disappear after spread, slippage, and queue priority.

---

### 11.2 PDT and account constraints

With roughly $20k capital, PDT constraints can materially affect intraday strategy behavior.

Even if the RM has a PDT gate, the strategy design itself may be constrained by account size.

---

### 11.3 Hidden common exposure

Swing and intraday may both load on the same hidden factors:

- Long beta.
- Growth.
- Momentum.
- Tech.
- High-volatility names.

Separate strategy names do not guarantee diversification.

---

### 11.4 False comfort from risk gates

Risk gates reduce blowups, but they do not create alpha.

A well-risk-managed losing strategy still loses money.

---

### 11.5 Feature leakage through data timing

PIT fundamentals are being improved, which is good. But all of the following require strict timestamp audits:

- News.
- Analyst upgrades/downgrades.
- Earnings surprise.
- Institutional ownership.
- Options data.
- Sector data.
- Macro data.

---

### 11.6 Model decay

The current champion degradation is already evidence of model decay or regime instability.

---

### 11.7 Complexity debt

The more gates, queues, re-scoring loops, and model versions exist, the harder it becomes to explain whether P&L came from alpha, filter, luck, or operational artifact.

---

## 12. What Is Actually Good

There is a lot worth preserving.

### 12.1 PM/RM/Trader separation

The architecture mirrors a real trading process:

- PM generates opportunities.
- RM controls permission and risk.
- Trader handles execution and lifecycle.

This is sound.

---

### 12.2 Decision audit trail

Logging every gate outcome to `decision_audit` is excellent. This can become a calibration dataset and post-trade analysis engine.

---

### 12.3 PM re-score loop

Re-scoring before execution is a smart defense against stale queued signals.

---

### 12.4 Paper trading first

The system is not rushing into live capital. This is correct.

---

### 12.5 Honest self-diagnosis

The original prompt already identifies many important issues:

- NIS sparsity.
- Walk-forward not applying PM gates.
- cs_normalize removing market-wide features.
- DB/broker state divergence.
- Label leakage risk.
- HPO variance.

This level of self-awareness is a major strength.

---

### 12.6 Risk framework is more mature than the alpha

The RM gate list is strong for a solo project:

- Position caps.
- Sector concentration.
- Correlation risk.
- Daily loss.
- Drawdown.
- Liquidity.
- Beta exposure.
- Factor concentration.

This should be preserved.

---

### 12.7 Walk-forward gating exists

Even though incomplete, having model promotion gates is good discipline.

---

### 12.8 PIT fundamentals work is directionally right

Moving from live-value fundamentals toward historical snapshots is exactly the right direction.

---

## 13. Concrete Next-Phase Plan

### Phase 1 — Truth Engine

Build the full PM/RM/Trader historical replay engine.

Goal:

```text
One backtest path that exactly mirrors live decision-making.
```

Deliverables:

- Historical replay runner.
- PM gate simulation.
- RM gate simulation.
- Trader execution simulation.
- Cost/slippage assumptions.
- Portfolio P&L output.
- Decision audit output.
- Regime performance report.

No model should be promoted unless it passes this.

---

### Phase 2 — Current Model Autopsy

Run v142, v29, and v41 through full diagnostics.

Deliverables:

- Performance by VIX bucket.
- Performance by SPY trend.
- Performance by breadth.
- Performance by sector.
- Performance by symbol.
- Performance by signal decile.
- Performance by opportunity score.
- Performance by cost assumption.
- Apr–Oct 2025 breakdown.

Goal:

```text
Know whether these models have a narrow tradable regime or no stable edge.
```

---

### Phase 3 — Label Repair

Redesign labels to match execution.

Swing tests:

- Current path label.
- Triple-barrier multiclass.
- Realized-R execution label.
- Meta-label on rule-fired candidates.
- 5/10/15-day variants.

Intraday tests:

- Current top-20% label.
- Top-20% plus absolute return hurdle.
- Residual return label.
- Realized-R execution label.
- Long/short rank test.

Goal:

```text
Train on what the system actually monetizes.
```

---

### Phase 4 — Feature Discipline

Run feature-family ablations and reduce noise.

Deliverables:

- Feature group contribution report.
- Minimum-fold improvement table.
- Regime stability table.
- Decile monotonicity report.
- Feature removal recommendations.

Goal:

```text
Smaller, more stable, more explainable models.
```

---

### Phase 5 — Live-Readiness Hardening

Make Alpaca the execution source of truth.

Deliverables:

- Broker reconciliation before trading.
- Broker reconciliation during trading.
- Mismatch safe mode.
- Idempotent order IDs.
- Partial-fill handling.
- Cancel/replace handling.
- Websocket disconnect handling.
- Kill switch.
- Recovery tests.

Goal:

```text
Operational trust before live capital.
```

---

## 14. Final Recommendation

The correct next step is **not** to add more features or chase a new model architecture.

The correct next step is to build a validation engine that can tell the truth.

Recommended sequencing:

1. Build full-fidelity historical replay.
2. Re-evaluate current models through the actual PM/RM/Trader stack.
3. Diagnose performance by regime.
4. Redesign labels to match actual execution.
5. Run feature ablations and simplify.
6. Fix Alpaca source-of-truth and live safety.
7. Only then consider live pilot exposure.

My honest view:

**MrTrader has a good skeleton, but the current alpha proof is not strong enough. The right move is not to abandon the project. The right move is to stop feature-chasing and build a research/production validation loop that can tell you the truth, even when the truth is uncomfortable.**

---

## Suggested Prompt for the Next LLM

Use the following when uploading this file to another LLM:

```text
I am building an automated trading system called MrTrader. Attached are two documents: 
1. The original system review prompt describing architecture, models, features, and current walk-forward results.
2. A critical review from another LLM.

Please do not simply validate or repeat the review. I want you to critique both documents and produce a concrete, prioritized implementation roadmap. Be direct about what is correct, what is wrong, what is missing, and what should be built first.

Specifically:
- Identify which recommendations are highest leverage.
- Identify which recommendations are overkill for a $20k paper/live pilot system.
- Identify any blind spots in the prior review.
- Produce a phased task plan that can be given to a coding agent.
- Separate research tasks, model tasks, data tasks, architecture tasks, and live-readiness tasks.
- Do not recommend generic ideas unless you explain exactly how they would be implemented, validated, and measured.
```
