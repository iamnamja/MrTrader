# MrTrader WF — Brutal Quant Review

> Reviewer posture: 15+ years quant equity, built WF infrastructure at multi-strat shops, allocated to and audited dozens of systematic books. No flattery. Specific opinions where other LLMs will hedge.

---

## 0. Topline (read this even if you read nothing else)

**Your negative WF Sharpe is the best news in your stack right now.** Most retail/semi-pro systems fail the opposite way: they show a +1.5 Sharpe on the backtest, get deployed, and bleed. Your post-realign run going from +0.036 to -0.459 to -0.571 as you tightened the simulation is exactly what a researcher *should* see when removing self-deception. **This is signal that your engine isn't lying to you. Don't lose that.**

That said, here's where I disagree with the framing in your prompt:

1. **C1–H7 are mostly symptoms, not root causes.** Your biggest problems are architectural, not parametric. Tuning ATR multipliers, label horizons, and stop logic in the same campaign will not get you to a trustworthy WF. You're treating a structural problem with parameter sweeps.

2. **You've confounded signal evaluation with strategy evaluation.** The WF you have measures "model + portfolio construction + execution + risk gates" as a single black box. When it fails, you cannot localize whether the alpha is dead, the portfolio construction is destroying it, the costs are eating it, or the risk gates are forcing bad exits. This is the single most important fix.

3. **LambdaRank → top-5 from Russell 1000 is a poor objective-to-decision match.** This is rarely discussed but it matters more than label horizon. Detail below.

4. **6 years of data, expanding window, 14 features, 5 folds is statistically thin.** Your DSR p-value is meaningfully overstated if you haven't tracked every model variant you tested. I'd bet you have <300 effectively independent training observations per fold under any honest accounting.

5. **The v215 → v216 jump changes ≥5 things simultaneously** (label horizon, feature count, ATR mult, target mult, step days). When v216 fails (or succeeds!), you won't know which knob mattered. Stop. Do controlled ablations.

6. **At $500k capital scaling target, yfinance daily + flat 15bps round-trip + MOO assumed at the open print is not credible.** Free at retail, expensive at scale.

The rest of this document is opinionated detail. I'm going to be more useful by being specific than by being polite.

---

## 1. Architecture Critique — What a Two Sigma Quant Says in 30 Seconds

If I handed your spec to a senior researcher at a top-tier shop, here's what they'd say verbatim before finishing the second page:

> "Why is your research stack the same code as your production stack? Where's the signal evaluation layer? Where's IC decay? Where's the factor decomposition? Where's the pre-cost backtest? You're computing strategy P&L without ever measuring the alpha you're feeding the strategy."

This is the deepest critique. Let me unpack it.

### 1.1 The Research/Production conflation

Best-in-class shops separate four distinct evaluations, each with its own validation:

| Layer | What it measures | Outputs | Validation question |
|---|---|---|---|
| **L1: Pure signal** | Does the model output correlate with future returns? | IC, rank-IC, IR-IC, IC decay curve | Is there any alpha at all? |
| **L2: Naive portfolio** | If we mechanically take top-K by score with no costs/stops, what's the return? | Decile spread, top-K vs bottom-K, factor exposure | Does the alpha survive the simplest portfolio construction? |
| **L3: Realistic portfolio** | Same + transaction costs + position limits + sector neutrality | Sharpe, IR vs benchmark, turnover | Does it survive frictions? |
| **L4: Full agent stack** | What you have today | Sharpe with stops, gates, MOO/MOC | Does it survive the actual decision policy? |

You jumped straight to L4. When L4 fails, you genuinely don't know if L1, L2, L3, or L4 broke the alpha. **This is the single most expensive architectural mistake in the system.**

Concrete fix: before retraining v217, run v215 through L1 and L2.

- **L1 test**: For each fold, compute rank-IC by forward horizon (1, 3, 5, 10, 20, 40, 60 days). Plot the IC decay curve. If your v215 model has rank-IC < 0.02 at any horizon, the alpha is dead and no amount of stop tuning will save it.
- **L2 test**: Take the top 50, top 20, top 10, top 5 by score, equal-weight, hold for N days with no stops, no costs. This is your **theoretical ceiling**. Compare to the SPX and to equal-weight R1K. The gap between L2 and L4 tells you exactly how much frictions+gates are costing you.

If L1 shows rank-IC of 0.03 at 5d and 0.005 at 40d, your label horizon problem isn't "5d vs 20d," it's "the alpha decays before 40 days." If L2 with top-5 has Sharpe of 0.4 and L4 has -0.5, you have a 0.9-Sharpe engine destruction. That's diagnostically valuable.

### 1.2 The Simulator is Doing Too Many Jobs

The agent_simulator at 1400 lines is doing: position management, P&L bookkeeping, stop checking, gate evaluation, sizing, regime detection, MOO/MOC simulation, transaction cost modeling, drawdown tracking, and probably more. This is a monolith. In production systems I've worked on, each of these is a swappable component with its own tests.

The cost shows: you can't run a "what if we removed stops entirely?" test in 30 minutes. You can't compare 5%/position vs 2%/position via a clean A/B. Every parameter change requires rerunning the monolith.

**Architectural recommendation**: refactor into ~6 components with clear contracts:

```
SignalGen          → DataFrame[date, symbol, score]
PortfolioBuilder   → DataFrame[date, symbol, target_weight]
ExecutionSim       → DataFrame[fill_date, symbol, fill_qty, fill_price]
CostModel          → DataFrame[fill] + commission/spread/impact
RiskOverlay        → Modifies PortfolioBuilder output
PerfAttribution    → Decomposes returns into alpha + factor + residual
```

Each can be swapped or stubbed. Want to A/B test "no stops"? Sub a no-op RiskOverlay. Want to test naive equal-weight vs vol-weighted? Swap PortfolioBuilder. This unlocks 10x iteration speed.

### 1.3 5-fold expanding WF on 6 years is statistically thin

Honest accounting:
- 6 years × 252 days ≈ 1500 daily observations
- With 20-day labels, ≈75 non-overlapping forward windows
- After purge/embargo of ~30 days total per fold, less
- Russell 1000 has ~1000 names but cross-sectional correlations mean effective N is far less (factor model gives ~10-50 effective independent bets)

You have a 14- to 19-feature LambdaRank model with **stop-loss optimization implicit in the strategy** trying to learn from this. The signal-to-noise is brutal. This is why you see fold-to-fold instability and why a 0.05 Sharpe shift in a single fold flips the gates.

What top shops do:
- **More history** (15-20 years where data is clean — see Compustat/CRSP for US equity); your 2018+ window misses 2008, 2011, the 2015-16 EM crisis, 2020 wasn't tested out-of-sample if it's in training. Note: clean PIT data pre-2010 is hard, but doable.
- **More folds via CPCV done seriously** — k=6, paths=2 is gentle. k=10-15 with proper purging gives 45-105 effective paths.
- **Synthetic regime augmentation** — bootstrap or block-bootstrap historical returns to generate alternative histories. Bailey-LdP write about this; few retail systems do it.
- **Hyperparameter robustness sweep** — run the full WF across 20-50 hyperparameter combinations and report the **distribution** of Sharpe, not the point estimate. If your strategy's Sharpe is fragile to ±20% changes in any of [ATR_mult, top_n, hold_bars, label_horizon], you don't have a strategy, you have an overfit.

### 1.4 The "agent stack" abstraction is leaking

PM/RM/Trader as conceptual layers is fine. But you should ask: in the WF, are these actually *agents* with decision processes that could be non-deterministic, or are they just **policy functions**? If they're policy functions, calling them agents is at best a code organization choice and at worst a source of bugs (state leaking between days, hidden globals, async issues).

In production, if these become LLM-driven agents (which your work history suggests is the direction), the WF needs to either:
- (a) Run the actual LLM in simulation (slow, expensive, non-deterministic) — only feasible for daily-bar swing
- (b) Distill the agent decisions to a deterministic policy function — fast but you're not testing the actual agent
- (c) Hybrid: run real agents on a sampled subset for validation, deterministic policy for full WF

Decide this now. The architecture you build depends on this answer.

---

## 2. Trust Calibration — How Much Should You Trust Current WF Output?

Honest answer: **for go/no-go on live deployment, you should trust it 1/10. For relative comparison of model versions, maybe 4/10.**

Here's the rank-order of issues by how much they distort the WF result:

| Issue | Distortion direction | Magnitude estimate |
|---|---|---|
| **L4-only evaluation (no IC/decile baseline)** | Indeterminate | You can't even tell sign of alpha |
| **C2 label/hold mismatch** | Overstate noise, understate signal | ±0.3 Sharpe |
| **C3 too-tight ATR stops (v215)** | Pure noise → bad fills | -0.4 to -0.6 Sharpe |
| **Purge 15d with 60d feature lookback + 20d labels** | Overstate (leakage) | +0.1 to +0.3 Sharpe |
| **yfinance data quality** | Mostly overstate (survivor effects) | +0.05 to +0.2 Sharpe |
| **Flat 15bps cost, no spread model** | Overstate | +0.1 to +0.3 Sharpe |
| **Stop checked EOD only** | Overstate | +0.05 to +0.15 Sharpe |
| **No earnings/event blackout** | Overstate volatility (tail trades) | +0.1 Sharpe avg, -2 Sharpe in single event days |
| **Flat 5% short borrow** | Overstate on shorts (cheap assumption) | +0.1 to +0.3 Sharpe on short PnL |
| **Hyperparameter trial accounting** | Overstate selected version | +0.2 to +0.5 Sharpe (depending on # of trials) |
| **C1 entry price mismatch** | Random, small | <±0.1 Sharpe |

Net: your reported Sharpe could be off by 0.5-1.0 in either direction. Given you're getting -0.5, the "true" Sharpe after fixes could be -1.0 or -0.2. Wide cone.

**Minimum fix set before any WF number deserves trust:**

1. **Add IC/decile baselines** (L1/L2 layer above). Without this, the system is uninterpretable.
2. **Fix purge** to ≥ max(feature_lookback, label_horizon) — should be ~70-80 days minimum given momentum_60d + 20d labels.
3. **Replace yfinance daily with Polygon daily** — you already pay for Polygon for intraday, the marginal cost is near-zero, the data quality is meaningfully better, and it handles delistings correctly.
4. **Real spread model** — even using a static spread table by market cap decile is dramatically better than 15bps flat. Polygon NBBO at open print costs you a Polygon Advanced sub.
5. **Earnings blackout** — IEX, Polygon, or Yahoo Earnings APIs are free or cheap. Trades through earnings announcements have completely different return distributions and your model wasn't trained for them.
6. **Hyperparameter trial registry** — every model variant you've tested, logged. Apply DSR using the *true* number of trials. This is uncomfortable because your DSR will deteriorate, but it's the truth.
7. **Benchmarks**: SPX buy-and-hold, equal-weight R1K rebalanced quarterly, random top-5 portfolio control. Without these, you can't tell good from bad.

Do these and rerun before anything else.

---

## 3. The Real Problems Beyond C1–H7

Things in your prompt that I think are mis-prioritized or missing:

### 3.1 The LambdaRank-to-decision mismatch (missing entirely from your list)

LambdaRank's loss function is pairwise — it learns to rank pairs of stocks correctly. When you train on Russell 1000 and you'll only ever use the top 5 names, the loss function is spending most of its capacity learning pairs in the middle of the ranking (rank 400 vs 500, rank 700 vs 600) that don't matter to your decision.

You want a loss that focuses on the extremes. Options:

- **LambdaRank with truncated DCG@K** — explicitly weight pairs where at least one element is in top-K. Most LightGBM/XGBoost ranking implementations support this via `lambdarank_truncation_level` or equivalent. If you haven't set this, you're using full-rank loss which is wrong for top-K selection.
- **Binary classification** — top decile = 1, rest = 0. Simpler, often works better at extremes.
- **Quantile regression** — predict 90th percentile of forward return.
- **Pointwise regression on rank-transformed returns** — sometimes the simplest thing.

Empirically across many ML equity systems I've seen, plain XGBoost regression on rank-IC-aligned targets often beats LambdaRank for top-K portfolio construction. LambdaRank shines for full-ranking applications (search, recommendation) where you care about the whole list.

**Worth testing as a controlled ablation**: same features, same window, swap LambdaRank for binary classification (top decile vs rest) or pointwise regression. If one of these meaningfully beats the other at L1 (rank-IC), you've found a real lever.

### 3.2 Train-on-label vs decide-on-path mismatch (mentioned in C2 but understated)

Your model is trained on "what's the 20-day forward return rank." Your strategy makes money via "buy → trail with ATR stops → exit at target or 40 bars." These are different objectives.

A stock with high 20-day forward return rank could have:
- Smooth uptrend → strategy captures it fully ✓
- V-shape: -3% then +5% → stopped out at -3%, strategy captures 0% ✗
- Jagged uptrend hitting stops on every pullback → multiple stop-outs, strategy underperforms ✗

The model has zero awareness of path. Two stocks with identical 20-day forward return can have wildly different P&L under your strategy.

**Better label** for what you actually do:

> **For each (symbol, entry_date), simulate your actual exit policy (ATR stop, target, max_hold) and label with realized P&L.**

This is your "strategy-aware label." It's computationally heavier (you need to simulate exits per training example) but it aligns the training objective with the decision policy.

Renaissance, Two Sigma, and AHL all do versions of this. It's sometimes called "label engineering" or "path-aware labels." Marcos López de Prado writes about meta-labeling and triple-barrier labels in *Advances in Financial Machine Learning* — your triple-barrier (stop, target, time) needs to be in the label, not just the simulation.

If you only do one thing on label design, do this.

### 3.3 No regime conditioning, anywhere

Your model treats 2020 March COVID, 2022 macro repricing, and 2019 melt-up as the same data-generating process. They aren't.

Top-tier shops always do one of:
- **Regime indicator features**: VIX level, VIX term structure (VIX9D/VIX30/VIX3M ratios), HY-IG credit spread, USD index, R1K-to-R2K ratio.
- **Regime-conditional models**: train separate models per regime cluster (you'd need >10 years to do this credibly).
- **Sample weighting**: weight training samples by similarity-to-recent-regime.

The simplest valuable add: a half-dozen regime features. VIX, VIX9D/VIX, HYG/LQD, $DXY 20d momentum, R1K/R2K 60d return spread. These cost you 6 numbers and meaningfully improve a model's ability to know "I'm in March 2020 right now, what does that imply about cross-section?"

### 3.4 Sector exposure is post-hoc, not in optimization

H1 mentions sector concentration is checked via RM rejection. M5 calls this out. But there's a deeper issue: even within the 20% sector cap, you have no notion of *factor exposure* — your top-5 long portfolio is probably massively long momentum, possibly short value, with hidden size tilts.

A 0.5 Sharpe long-momentum book in 2017-2021 was free money. In 2022 momentum lost 30%. If your "alpha" is actually a momentum factor exposure, you'd have lost everything in 2022 — and your fold 4 (2022-06 to 2023-01 test) is exactly that period.

You need factor decomposition. The cheap way:
- Compute daily factor returns (momentum, value, size, quality, low-vol) via Fama-French style portfolios on the R1K
- Regress your portfolio's daily returns on these factors
- Report alpha (residual) and factor loadings

If you have +0.5 Sharpe but it's all momentum factor, you have zero alpha. If you have -0.5 Sharpe but +1.0 alpha and -1.5 momentum (i.e., you were trying to be value/contrarian in a momentum-favorable period), you might actually have something.

This is one of the highest-value diagnostic tools you can add. It's ~200 lines of code.

### 3.5 Capacity analysis is missing

At $20k, you can trade anything. At $500k with 5 names at 5%, you have $25k positions. At top R1K liquidity (AAPL: $5B daily volume), this is invisible. At bottom R1K liquidity ($5M daily volume), you're 0.5% of ADV — meaningful impact.

What you should compute: for each fold's actual trades, what %ADV did the trade represent? If many of your alpha-generating trades are in the bottom-half-ADV of R1K, your strategy doesn't scale, and the WF results will systematically degrade with capital.

This matters because your stated capital range is 25x ($20k to $500k). The strategy that works at $20k may be completely different from the strategy that works at $500k.

### 3.6 The 5-position concentration question

You ask in (d) whether 5 positions is feature or bug. Both, but mostly bug.

Theoretical answer: with N independent positions of equal expected return μ and equal volatility σ, portfolio Sharpe scales as √N relative to single-position Sharpe (assuming zero correlation, which is generous). Going from 5 to 20 positions, theoretically, doubles portfolio Sharpe if alpha and correlations are well-behaved.

Empirical answer from ML equity literature: portfolios of 20-50 names typically dominate concentrated portfolios for ML signals because:
- Top-5 selection has high turnover (today's #5 is tomorrow's #15)
- ML signal noise is large relative to its mean — averaging across more names averages out signal noise too
- Idiosyncratic risk (earnings misses, M&A, fraud) dominates at 5 positions

The exception: if your signal has very high concentration of alpha in the extreme top (top 1%), 5 names captures it. Does yours? Test it. Look at average forward return rank-IC at top-5 vs top-20 vs top-50 by score. If rank-IC of top-50 average ≥ rank-IC of top-5, you should trade 50.

My strong prior: for a LambdaRank model on a 1000-name universe, top-30 will beat top-5 on a risk-adjusted basis. **The portfolio construction is almost certainly limiting your alpha more than the model.**

---

## 4. Hedge-Fund-Grade WF Infrastructure — What You're Missing

Ranked by value-to-build at your scale:

### 4.1 Experiment registry with proper trial accounting
Every model version, every hyperparameter, every WF run logged with full config and results. Apply DSR using N = total trials ever run, not 1. Bailey-LdP's PSR/DSR work assumes this; without it, your DSR is overstated. MLflow or simple DuckDB will do it. **High value, low cost.**

### 4.2 Adversarial / red-team review
Pay someone external to look for bugs. Common findings: look-ahead in feature construction (using close to compute a "next-day" momentum), survivorship in fundamentals (FMP backfills delisted), accidental future-knowledge in PIT membership. **You'll find at least one bug this way. Always.**

### 4.3 Synthetic data tests
Generate synthetic OHLC with **known alpha** injected. Run your full WF on it. Does it recover the known alpha? If not, you have an engine bug. If yes, you've validated the engine. **Critical for trust; nobody at retail/semi-pro level does this.**

### 4.4 Capacity & impact model
Per-trade slippage model: `slip = α × |trade_size / ADV|^β + spread/2`. Calibrate α, β from public TCA data or Almgren-Chriss style. **Required before $500k.**

### 4.5 Regime-conditional performance reporting
WF Sharpe by VIX regime, by SPX 60d return regime, by sector concentration regime. A single number masks fragility. **Reveals where strategy actually makes/loses money.**

### 4.6 Walk-forward hyperparameter optimization
Nested CV: each fold's training set internally CVs to pick hyperparameters, model trained on full fold-train, tested on fold-test. Without this you have a hyperparameter leak. **Hard to get right but transformative.**

### 4.7 Multi-strategy WF harness
Single config schema that supports long-only, market-neutral, intraday, pairs, etc. Each strategy is a plugin to the harness. **Architectural; see Section 5 below.**

### 4.8 Live shadow trading + reconciliation
Run paper trading via Alpaca paper for 6 months. Daily, compare paper P&L to "what the WF would have predicted given today's signals." Discrepancies expose bugs.

### 4.9 Risk attribution daily / weekly
Decompose portfolio returns into factor exposures, sector tilts, individual position contributions. Track over time. Identify drift.

### 4.10 Stress tests
Replay your strategy against:
- 1987-10-19
- 1998 LTCM
- 2008-09 to 2008-12
- 2010-05-06 (flash crash)
- 2020-03 COVID
- 2022 (full year)
- Custom: SPX -5% in one day, sector dispersion squeeze
If your strategy blows up on any of these and you can't explain why, you're not ready.

### 4.11 Out-of-distribution monitors
Once live: track feature distributions, predicted score distributions, realized vs predicted hit rates. Trigger alerts on drift. Lots of strategies die from quiet feature distribution shifts.

What real top shops have that you can't reasonably replicate at your scale (and shouldn't try):
- Tick-level order book reconstruction
- Custom market microstructure simulators
- Petabyte-scale alternative data
- Dedicated infra teams
- Multi-million-dollar Bloomberg/Refinitiv/FactSet contracts

You don't need these for $20k-500k. The above 1-11 list is what genuinely matters.

---

## 5. Multi-Strategy Architecture — How to Make WF Extensible

Your current WF is welded to "daily swing L/S on R1K with LambdaRank model." Making it support long-only, market-neutral, intraday, pairs, options requires real abstraction. Here's the cleanest framework I've seen work:

### 5.1 Strategy as composition of contracts

```
class StrategyConfig:
    universe:       Universe          # PIT membership over time
    data_layer:     DataLayer         # Bars at appropriate resolution
    signal_gen:     SignalGenerator   # (symbol, time) → score
    pf_builder:     PortfolioBuilder  # scores → target weights
    risk_overlay:   RiskOverlay       # weights → constrained weights
    exec_sim:       ExecutionSim      # weights → fills
    cost_model:     CostModel         # fills → realized cost
```

Each is an ABC. A strategy plugs in concrete implementations.

**Long-only swing**: same signal_gen, change pf_builder to long-only, change risk_overlay.
**Market-neutral**: same signal_gen, pf_builder enforces dollar/beta neutrality, risk_overlay adds factor constraints.
**Intraday**: change data_layer to 5-min, signal_gen to intraday model, pf_builder to bar-level rebalancing.
**Pairs trading**: signal_gen produces pair-residual z-scores, pf_builder is delta-neutral pair construction.
**Options**: separate exec_sim/cost_model entirely.

The WF harness doesn't care which combinations you plug in.

### 5.2 The Universe abstraction is critical

PIT membership is not just R1K. For pairs, it's a graph of co-integrated pairs. For options, it's an options chain. The Universe interface should be:

```
universe.get_tradable(as_of_date) → set of tradable_ids
universe.get_features_lookback(as_of_date) → window for valid features
universe.get_action_constraints(as_of_date) → halts, dividends, etc.
```

Get this right early. It's the highest-leverage abstraction.

### 5.3 Time resolution as a first-class config

Daily, hourly, 5-min, 1-min — same WF harness, different time resolution. The simulator should iterate "bars" abstractly, with a clock that knows when market opens/closes, session boundaries, halts.

Your current intraday_agent_simulator looks like a parallel implementation to agent_simulator. **This is the smell.** They should be one simulator with a `bar_resolution` parameter, period.

### 5.4 What this unlocks

Once this is in place:
- Multi-strategy portfolio allocation: WF each strategy individually, then compute a portfolio of strategies with correlation-aware allocation (risk parity, mean-variance, etc.). This is what real multi-strat books do.
- Strategy mixing: long-only sleeve + market-neutral sleeve + intraday sleeve, allocated by your tier/risk preference.
- Robust testing: "does this signal work as long-only? as long/short? as market-neutral?" — if it only works long/short, suspect factor exposure.

Estimated engineering effort to refactor: **3-4 weeks of focused work**, would save you 6+ months over the next two years of evolution.

---

## 6. Agent Workflow Pitfalls — PM → RM → Trader in Backtest

Specific to your design:

### 6.1 The information asymmetry between sim and live

In live: PM sees signals at 9:00, ranks, sends to RM at 9:15, RM evaluates at 9:20, Trader sends orders at 9:25 for MOC, MOC fills at 16:00.

In WF: PM sees today's close + EOD signals, generates ranking, executes at "open[t+1]."

These are not the same. The latency between "signal generation" and "fill" in live includes:
- Data arrival delay
- Agent decision time (worse if LLM)
- Order routing latency
- Queue position in order book at the open print
- Post-trade settlement

The aggregate: ~10-30 minutes of price moves between your signal-generation snapshot and your fill. In volatile open auctions, that's material.

**Fix**: instead of fill at `open[t+1] × (1 ± 0.03%)`, model as `VWAP(09:30-09:45)` from intraday data, with a slippage of ~5-15bps depending on size. You have Polygon intraday; this is achievable.

### 6.2 State leakage between agents

If PM's decision depends on RM's prior-day rejections (e.g., RM rejected XYZ yesterday so don't re-propose), and this state is global/cached, you have hidden state that the WF must reproduce perfectly. Even a small bug — e.g., RM state is per-process and a parallel WF run gets stale state — corrupts the entire run.

Common WF bug pattern: results vary by random seed when they shouldn't, or by execution order. Test for this. **If you re-run the same WF with same seed and get different Sharpe to 4 decimal places, you have a non-determinism bug.**

### 6.3 The risk gate evaluation order matters

Your RM evaluates: position sizing → daily loss → sector concentration → portfolio heat. The order matters when gates interact. If portfolio heat is binding and you sort candidates differently than live, you reject different names.

Document this. Make the gate logic ordering deterministic and explicit. Many WF bugs hide in "which 3 candidates of the 5 proposed got rejected by sector limit?"

### 6.4 Agent decisions that depend on agent observations of other agents

If your PM is going to be an LLM agent in live and it reads "RM rejected XYZ due to sector cap," that's an input the LLM sees. In WF, you must reproduce this exact text/signal.

The cleanest approach: **separate the agent decision protocol from its implementation.** The PM produces a `RankedProposalSet`. The RM produces a `RiskAssessment` from that. The Trader produces `ExecutionPlan` from that. These are data contracts. In WF you can substitute deterministic policy functions that produce equivalent outputs; in live you can run real LLMs that produce the same data contracts. Same harness, different implementations.

If you're not doing this already, do it before you scale agent complexity.

### 6.5 What top shops actually do

The actual best practice at multi-strat shops: **most decisions are pure functions, not agents.** "Agents" exist for tasks where judgment genuinely beats rules (e.g., interpreting an earnings call, evaluating a one-off corporate event). For position sizing, risk gates, and order placement — these are pure rule-based code, sometimes 50 lines, never an LLM. They're tested deterministically.

Your "agent stack" framing is fine for code organization. But if you're building agent intelligence into the PM/RM/Trader decisions themselves, **you're adding non-determinism and complexity to functions that should be pure code**. I'd push back on that.

---

## 7. Data Priorities — Ranked by Marginal Value at Your Scale

For a R1K daily ML strategy at $500k capital, here's the ranking. Each line: dataset, marginal value, cost, where to get it.

| Rank | Dataset | Marginal value | Cost | Source |
|---|---|---|---|---|
| 1 | **Corporate event calendar (earnings)** | Critical — single biggest source of unmodeled tail risk | $0-50/mo | Finnhub, IEX, Polygon, yfinance has it |
| 2 | **Short interest + borrow rates daily** | Fixes H1; required for short side; alpha signal in its own right | $0 (FINRA twice/month) to $100/mo daily | FINRA, Interactive Brokers, Polygon |
| 3 | **NBBO bid-ask at open/close** | Real execution cost model; fixes biggest cost realism gap | Bundled in Polygon Stocks Advanced ($199/mo) | Polygon |
| 4 | **Analyst estimates + earnings surprises** | Well-documented alpha factor (PEAD); regime-stable | $0-100/mo | Finnhub, FMP, IEX |
| 5 | **Daily Polygon equities (replace yfinance)** | Data quality, delisting handling, splits | Already paying | Polygon |
| 6 | **Economic calendar (FOMC, CPI, NFP)** | Free, easy, useful for blackout/regime | $0 | Trading Economics, FRED, BLS |
| 7 | **VIX term structure (VIX9D, VX1, VX2)** | Better regime detector than VIX alone | $0 | CBOE direct, Yahoo |
| 8 | **Options IV at-money** | Forward-looking vol signal per name; option market often leads | $50-200/mo | ORATS, IVolatility, Polygon Options |
| 9 | **13-F holdings (quarterly)** | Crowding signal, slow but cheap | $0 | SEC EDGAR direct |
| 10 | **ETF flow / ETF rebalance** | Crowding/momentum; IWM/SPY flows | $0-50/mo | ETF.com, Bloomberg if accessible |
| 11 | **Sector/Industry classification (GICS)** | Better than basic; matters for neutrality | $0 (sufficient quality in Polygon ref data) | Polygon, FMP |
| 12 | **News sentiment beyond NIS** | Diminishing returns if NIS is solid | varies | Already in pipeline |
| 13 | **Factor exposures (Barra-like)** | Risk decomp; can build your own from public factor returns | $0 (build) to $5k+/mo (commercial) | AQR Data Sets free, Kenneth French data |
| 14 | **L2 order book** | Overkill at $500k | $200+/mo | Polygon, IEX |
| 15 | **Alternative data (satellite, cc, web traffic)** | Not worth at your scale | $1k-$50k/mo | Various |

Pragmatic minimum stack at your tier:
- Polygon Stocks Advanced ($199/mo) — daily + intraday + reference + corporate actions
- Free: FINRA short interest, FRED macro, SEC EDGAR 13F, CBOE VIX term structure
- Optional add: Polygon Options Starter for IV — if you want regime signal

You can have a top-3 retail-tier data stack for ~$300/mo. There's no real excuse for staying on yfinance daily.

---

## 8. Label Design — Critical Deep Dive

You ask whether LambdaRank on cross-sectional return rank is the right choice. The answer requires multiple sub-questions:

### 8.1 Cross-sectional vs time-series

Cross-sectional (your choice): "given today's universe, which names will outperform the rest." Time-series: "given this stock's history, will it go up tomorrow."

Cross-sectional is correct for L/S equity. Time-series is correct for single-instrument futures. You picked correctly. ✓

### 8.2 Rank vs return regression

Rank-based labels are robust to outliers and regime non-stationarity in return magnitudes. Return regression captures magnitude info that rank discards.

Empirical evidence is mixed. For equity ML at daily horizon:
- Rank labels: more stable, lower variance, often slightly lower mean alpha
- Return labels (especially residualized vs sector/factor): higher ceiling, higher variance

For your scale and stability needs, **rank is the safer choice**. ✓ But test it.

### 8.3 LambdaRank vs alternatives

This is where I think you've underperformed. As discussed in 3.1, LambdaRank's full-list loss is mismatched to top-K selection.

**Strong recommendation**: in your next controlled ablation, swap LambdaRank for one of:
- XGBoost binary classification: label = (rank_pct ≤ 0.10), i.e., top-decile vs rest. Predicted probability used as score.
- XGBoost regression on rank itself.
- LightGBM LambdaRank with `lambdarank_truncation_level=50` (only optimize top-50 pairs).

Run all three through your WF, holding everything else constant. Report rank-IC at top-K from each. **This is one of the highest-EV experiments you can run in the next month.**

### 8.4 Label horizon — the actual answer

You ask (a): 20 days vs 40 days?

The right answer is "whichever horizon shows the highest rank-IC with reasonable persistence." You don't currently know this because you don't measure it.

Action: for v215 model (and any v216 retrain), compute rank-IC of model predictions vs forward returns at horizons {1, 3, 5, 10, 20, 40, 60} days. Plot decay. The horizon at which IC is maximized **is your effective alpha horizon**, and that's the horizon you should be running.

My prior: for an XGBoost model with a mix of momentum (5/20/60) and mean-reversion features (RSI, BB) features on R1K, peak IC is usually at 10-20 days, with substantial decay by 40 days. Your 40-bar hold is almost certainly past peak alpha, which is why v215 fails.

But maybe your specific features have longer persistence. Measure it.

### 8.5 Path-aware labels (triple-barrier)

As discussed in 3.2: your strategy makes money via a triple-barrier policy (stop / target / time). The label should reflect this.

Implementation (López de Prado, ch. 3):
```
For each (symbol, entry_date):
  upper_barrier = entry_price × (1 + target_pct)
  lower_barrier = entry_price × (1 - stop_pct)
  time_barrier = entry_date + max_hold
  
  Simulate forward day-by-day until any barrier hit
  Label = +1 (target), -1 (stop), 0 (time)
  OR
  Label = realized P&L
```

Use this label for training. Now the model is trained on "if I enter this name and run my actual policy, what happens." This is *strategy-aligned learning*.

This is the single most impactful label change you can make. Probably 0.2-0.4 Sharpe of improvement if your engine is sound.

### 8.6 Meta-labeling

Even more advanced: train a "primary" model to predict direction, then a "secondary" (meta-label) model to predict *when to take the primary's signal*. The secondary model has access to the primary's prediction plus features. This lets the strategy abstain on low-confidence signals.

For your case: PM produces a ranking → meta-label model says "yes/no take the top-5 trade today" based on portfolio state, regime, signal dispersion. This is a clean way to add a "should I be in the market at all today?" gate without hand-tuning regime rules.

Not necessary for v217 but worth keeping in mind for v220+.

---

## 9. Specific Open Question Answers

**(a) 20-day labels vs 40-bar hold:** Neither without IC analysis. Run rank-IC by horizon, pick the horizon at peak IC, set both label and hold there. My prior: 10-15 days is likely sweet spot for this feature set. Stop tuning before measuring.

**(b) Train/live distribution mismatch in LambdaRank:** Yes, real issue. LambdaRank trains on full 1000-name batches; you decide on top-5. The model never explicitly optimizes top-5 quality. Use `lambdarank_truncation_level` (LightGBM) or switch loss function. See 8.3.

**(c) ATR stops in WF but not in labels:** Wrong unless stops fire <10% of the time (then negligible). Currently 60%+ stop rate means stops dominate strategy and the model has no awareness. Fix via path-aware labels (8.5) OR widen stops so they don't fire on noise OR remove stops entirely in favor of vol-targeted sizing.

The theoretical relationship: stop width should be ≥ 2x the average daily noise over the label horizon. For 20-day labels on R1K, expected 20-day noise (ATR-implied) is ~5-8%. So stop should be ≥10-16%. Your v215 0.75-1.5% stops fire on noise constantly. v216's 2.25-4.5% is better but still tight. Realistically you want ATR_STOP_MULT in [3.0, 5.0] for label horizons of 20 days.

Alternative: **no stops, use vol-targeted position sizing instead.** Set position size = `risk_budget / (ATR × price)`. The position itself absorbs the risk; you don't need an additional stop. This is what many systematic equity books do and it's more aligned with cross-sectional ML.

**(d) MAX_OPEN_POSITIONS=5:** Bug at your scale. As discussed in 3.6, ML signals work better at 20-50 names due to noise averaging. Test 20 and 50 in your next sweep. My strong prior is 20-30 will beat 5 by 0.3-0.6 Sharpe.

Counter-argument: "high conviction" portfolios work when alpha is concentrated in the extreme tail and you have a discretionary overlay. You have neither. Trade more names.

**(e) Expanding vs rolling window:** Rolling for non-stationary equity markets. 3-5 year rolling. Expanding gives equal weight to 2018 and 2024 — implausible given regime changes. Rolling 1260 (5y) or 756 (3y) days is more typical. Run a sensitivity: WF Sharpe vs rolling window length {252, 504, 756, 1260, expanding}. **If your strategy is fragile to this choice, you're overfit.**

---

## 10. Counter-Intuitive Takes / Outside-the-Box

A few things I'd push you toward that don't appear in your prompt:

### 10.1 Stop modeling the strategy. Start modeling the alpha.

Your current research question is: "Will the MrTrader strategy make money?"

The better research question: "Does the v215 model produce a signal with rank-IC > 0.02 that decays slowly?"

If the answer is yes, *any* reasonable portfolio construction will make money. If the answer is no, *no* portfolio construction will save it.

You're optimizing the portfolio when you should be measuring the signal. Get to rank-IC measurement immediately, before any more model retraining.

### 10.2 Consider abandoning the agent framing for the research phase

The PM/RM/Trader agent structure is a production-deployment abstraction. For research, it's friction. Build a flat pipeline:

```
data → features → model → score → top-K weights → simulated returns → metrics
```

300 lines. Iterates in seconds. Once you have a viable signal, port it to the agent framework for deployment. **Don't research inside the production architecture.** Top shops have research notebooks/scripts that look nothing like their production code.

### 10.3 Consider a simple baseline you can never beat

Build a strategy that:
- Equal-weights top decile by 12-month momentum (excluding last month — standard academic momentum)
- Rebalances monthly
- Costs 5bps/trade

This is the classic Jegadeesh-Titman momentum strategy. It's been published for 30+ years. It's well-understood. It has ~0.5 Sharpe in normal regimes, drawdowns in 2022, etc.

**If your ML system can't beat this baseline at the same capital tier after costs, you have nothing.** Many ML equity systems can't. It's a useful sanity check.

### 10.4 Test the trivial null hypothesis

Run your full WF with a **random signal** (np.random.uniform()) instead of the model. What Sharpe do you get?

If you get Sharpe near 0 with reasonable distribution, your engine is unbiased.
If you get Sharpe meaningfully ≠ 0 from random signal, you have an engine bias (a bug somewhere).
If you get Sharpe of -0.1 from random, your stops + costs alone bleed you 10bps/year — that's the friction floor your model has to overcome.

This 10-line test reveals more than you'd think.

### 10.5 Test the perfect-information null hypothesis

Run WF where the "signal" is actual forward 20-day return rank (i.e., perfect information). What Sharpe?

This is your **theoretical ceiling**. If perfect-information gives Sharpe of 5 and your model gives -0.5, the gap is "how good your model is" plus "how good your strategy infrastructure is." Decompose by also running perfect-information through L2 (simple top-K) vs L4 (full agent). If perfect-information at L2 = 4.5 and at L4 = 2.0, your engine is destroying 2.5 Sharpe of value via gates/stops/costs/sizing — that's a strategy infrastructure problem, not a model problem.

This is a classic diagnostic that almost nobody runs. It's brutally clarifying.

### 10.6 Multi-horizon ensembling

Instead of one model at one horizon, train 3-5 models at different horizons (5, 10, 20, 40 days) and ensemble their predictions. Each captures different alpha. The ensemble averages out noise and captures alpha at the horizon where it's strongest for each name (some names are momentum at 5d, some are at 40d).

Cheap to implement, robust improvement empirically.

### 10.7 The portfolio-of-strategies endgame

The "right" architecture for your $20k-500k target isn't one strategy. It's a portfolio of strategies (swing L/S, market-neutral, intraday, maybe a long-only momentum sleeve) with capital allocated by risk-parity or correlation-aware optimization. Each individual strategy can have Sharpe of 0.6-0.8; combined, you can hit 1.2-1.5 because the strategies are imperfectly correlated.

This is what real multi-strat books do. Your single-strategy WF tunnel-vision is precluding this. Build for it.

---

## 11. Six-Month Prioritized Roadmap

Given Min is one person, balancing this with MarketAxess work, single-parent duty, and the orchestrator project — be realistic. Budget: assume ~15-20 hours/week sustainable on this.

### Month 1: Diagnostic foundation
**Goal: stop guessing, start measuring.**
- [ ] Migrate yfinance daily → Polygon daily; PIT validation
- [ ] Add benchmarks: SPX, equal-weight R1K, random top-5 control, Jegadeesh-Titman momentum baseline
- [ ] L1 measurement: rank-IC by horizon for v215 model on each fold
- [ ] L2 measurement: top-K (5/10/20/30/50) pure signal performance, no stops, no costs
- [ ] Trivial null + perfect-information null tests
- [ ] Compute factor exposures (Barra-light) on v215 historical positions
- [ ] Fix purge gap to ≥70 days

**Decision point at end of Month 1:** Is there alpha (rank-IC > 0.02 at some horizon)? If yes → continue. If no → swap LambdaRank for binary classification or regression; if still no → rethink features entirely.

### Month 2: Engine correctness + cost realism
- [ ] Refactor agent_simulator into separable components (signal/portfolio/risk/exec/cost)
- [ ] Unify daily and intraday simulators (single engine, bar_resolution param)
- [ ] Spread model from NBBO data
- [ ] Earnings blackout
- [ ] Short interest data (FINRA)
- [ ] Per-symbol borrow rates (Alpaca data if accessible, IBKR proxy otherwise)
- [ ] Intrabar stop simulation via 5-min bars

### Month 3: Label engineering + model variants
- [ ] Path-aware (triple-barrier) labels
- [ ] Controlled ablation: LambdaRank vs binary classification vs regression — same features, same window
- [ ] Multi-horizon ensemble: train at {5, 10, 20} jointly
- [ ] Sample weighting by regime
- [ ] Regime features (VIX term, credit spreads, $DXY)

### Month 4: Portfolio construction overhaul
- [ ] Vol-targeted position sizing
- [ ] Factor neutrality (sector + size + momentum)
- [ ] Test 20, 30, 50-name portfolios
- [ ] Mean-variance optimization layer (optional)
- [ ] Capacity analysis: %ADV tracking, expected impact

### Month 5: Multi-strategy framework
- [ ] Strategy plugin architecture
- [ ] Long-only variant
- [ ] Market-neutral variant
- [ ] Re-WF each through unified harness

### Month 6: Production discipline + paper trading
- [ ] Experiment registry with proper trial accounting
- [ ] DSR recomputed with full trial count
- [ ] Adversarial review (pay a quant friend $1-2k for a one-week audit)
- [ ] Stress tests (1987, 2008, 2020, 2022)
- [ ] Paper trading goes live; daily reconciliation
- [ ] Monitoring/alerts on feature drift

### What's out of scope at your tier:
- L2/order book modeling
- Custom factor models (Barra)
- Most alternative data
- Custom market impact models beyond linear-power-law
- Co-location, FPGA, kernel-bypass networking
- True HFT / market making

### Must-do vs nice-to-have summary:

| Priority | Items |
|---|---|
| **Must-do** | L1/L2 measurement, Polygon daily, purge fix, path-aware labels, factor decomp, regime features, capacity, earnings blackout, trial registry |
| **Should-do** | Engine refactor, vol-targeted sizing, more positions, multi-horizon ensemble, real spread model, short borrow rates |
| **Nice-to-have** | Multi-strategy framework, market-neutral variant, options regime signal, meta-labeling, synthetic data testing |
| **Out of scope** | L2 book, true alt data, factor model purchase, HFT-adjacent stuff |

---

## 12. Final Verdict

You have built more sophisticated infrastructure than 95% of retail systematic traders. DSR, CPCV, PIT membership, purge/embargo, an agent stack, a labeled known-issues list — most retail systems don't even have the vocabulary for this. Genuinely solid engineering foundations.

But sophistication ≠ alpha. The current system has:
- Negative WF Sharpe on its honest setup
- A confounded research stack that can't isolate where alpha breaks
- A model loss function (LambdaRank) mismatched to its decision policy (top-K)
- Statistically thin data (6y, 5 folds, 14 features)
- Costs and data sources unsuited for the upper end of its capital range

The path forward is **not** v217, v218, v219 retrains. The path forward is to **stop tuning the strategy and start measuring the signal**. Get to rank-IC, factor decomposition, perfect-information ceiling, and trivial null. Those four diagnostics will tell you in 2 weeks of work what 6 months of model retraining cannot.

**My honest probability assessment**: with the v215 features + LambdaRank top-5 architecture, you reach a deployable trustworthy system (Sharpe > 0.7 net of costs, stable across regimes, DSR-honest) with probability ~15-25%. The features are mostly standard public factors; a top-K LambdaRank on R1K of these factors is a heavily-mined corner of the space. Many shops have run this exact thing.

With path-aware labels + binary classification + 20-30 name portfolio + vol-targeted sizing + factor neutrality + proper data + earnings/regime conditioning: probability ~40-50%. This represents real engineering effort, but it's the work that genuinely moves the needle.

With multi-strategy portfolio (swing + market-neutral + intraday) at maturity: ~55-65% of having something modestly profitable, scale-appropriate.

**Don't deploy live with real money until L1 (rank-IC) and L2 (decile spread) show clear alpha, factor decomposition shows residual alpha (not just factor exposure), paper trading reconciles within 0.2 Sharpe of WF expectation over 3-6 months, and your DSR is computed against your true trial count.**

The system you're building is real and worth building. But discipline about what you actually know vs what you hope is true is what separates the strategies that work from the strategies that look like they work in backtest.

Be brutal with the system. Be honest with yourself. Measure before you tune. Build the diagnostic stack before the next model.

You'll get there. But not by retraining.

---

*Reviewer notes: Cross-reference the Bailey/López de Prado canon (Advances in Financial Machine Learning), Grinold-Kahn (Active Portfolio Management), and Pedersen (Efficiently Inefficient) if you want the institutional vocabulary. Most of the techniques discussed here trace to those three sources.*
