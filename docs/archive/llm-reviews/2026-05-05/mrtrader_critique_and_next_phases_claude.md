# MrTrader — Independent Critique and Next-Phase Plan

**Audience:** Min (system owner) + downstream LLM reviewers
**Purpose:** Honest gap analysis, prioritized next phases, and specific recommendations to move from "good paper-trading prototype" toward production-grade systematic strategy.
**Reviewer's stance:** Skeptical by default. The system has real engineering merit but several first-order issues that paper trading will not surface. The points below are framed as challenges, not verdicts — push back on any you disagree with.

---

## TL;DR — The Bottom Line

MrTrader is a well-engineered prototype with clean separation of concerns, thoughtful audit logging, and a defensible test culture (~1,200 tests). The infrastructure work is solid.

**However**, the case for live trading right now rests on a walk-forward result that is not as strong as the headline number suggests, a label formulation that is more handcrafted than learned, an execution model that has not been TCA-validated, and a risk framework that is a checklist of independent rules rather than a portfolio model. The "agent" architecture is over-engineered for what it actually does, and several of the "Known Bugs Fixed" entries are symptoms of that over-engineering, not isolated incidents.

**The single most important data point in the entire document:**
> **Swing model Fold 3 (the most recent 15 months, 2025-01 → 2026-04): Sharpe = −0.03, Win% = 55.2%.**

Average walk-forward Sharpe of +1.18 is being carried entirely by Fold 2 (+2.69). A real institutional gate would weight recent folds higher and require non-deterioration across folds. As stated, the gate is gameable by one strong middle fold. **This needs a direct, honest answer before any other phase work matters.** If the edge has degraded, you are about to deploy a model trained for a regime that has ended.

The top three things to do, in order:

1. **Validate the edge is real and current.** Bootstrap confidence intervals on Sharpe, monotonic-fold gating, sensitivity analysis on hyperparameters, NIS A/B test, gap-chase and entry-gate audit. Do not move to live until Fold 3 is understood.
2. **Build a portfolio-level risk model** (VaR/CVaR, beta exposure, stress tests). The current rules are necessary but not sufficient.
3. **Build TCA infrastructure** (implementation shortfall tracking, slippage distribution, backtest-vs-live P&L gap). Without TCA, you cannot tell whether the strategy is working or being eaten by frictions.

Everything else is secondary to those three.

---

## What Actually Works (Credit Where Due)

Before the critique, what's genuinely good:

- **Audit infrastructure is exceptional for a small-team build.** `decision_audit` with feature snapshots, gate-level outcome attribution, EOD backfill, replay tooling. This is rare even at funded prop desks. Keep this.
- **Test culture.** 1,200+ tests on a personal project is real discipline. The fact that you've shipped 9 documented post-incident fixes with regression tests means you're treating this seriously.
- **Separation of decision audit from agent decision tables.** Lets you reconstruct "what did the model see, what did it decide, what was the outcome" — critical for debugging and for any future regulatory scrutiny.
- **Reconciliation logic on startup.** Most retail bots break on restart; you've thought through ghost positions, missing positions, and synthetic Trade records. This is a sign of someone who has been bitten before.
- **Three-tier broker/data abstraction** (Alpaca + Polygon + SEC EDGAR) gives you optionality. If one provider degrades, you have a path.
- **Kill switch + circuit breaker.** Both exist and both are tested. Good.
- **EOD flat rule for intraday** is non-negotiable for a 2-hour-hold strategy. You have it. Many systems don't.

This is real work. Do not let the critique below make you think the foundation is bad — it isn't. The foundation is good enough that the critique is worth your time.

---

## Critical Findings (Sorted by Severity)

### 1. The Fold 3 Sharpe is the loudest signal in the document and needs a direct answer

| Fold | Period | Sharpe | What it means |
|---|---|---|---|
| 1 | 2022-07 → 2023-10 | +0.88 | Recovering bull, low-vol — model works |
| 2 | 2023-10 → 2025-01 | +2.69 | Mega-cap momentum regime — model thrives |
| 3 | 2025-01 → 2026-04 | **−0.03** | Most recent 15 months — **edge has gone flat** |

The "Avg Sharpe = +1.18 (gate: > 0.80) ✅" framing is misleading. Three observations:

- **Sharpe arithmetic averaging across folds is not how you should gate this.** Recent folds matter more than old ones. A regime-shifted strategy will look fine on average and fail in production.
- **N=134 trades in Fold 3.** A bootstrap 95% CI on the Sharpe estimate is going to be wide — probably something like [-0.4, +0.4]. The point estimate of -0.03 could plausibly be -0.3 (genuinely broken) or +0.3 (still working). You don't know yet.
- **What changed in 2025?** Mega-cap momentum has been the dominant regime since late 2022; if that flipped or compressed in early 2025, an XGBoost model trained on 2022-2024 features is going to struggle. Possible suspects: vol regime change, dispersion change, factor rotation, AI-trade unwind, narrower breadth.

**Required before live:** rerun Fold 3 with bootstrap CIs. If the lower bound is meaningfully negative, do not deploy. If it overlaps zero, you need fresh out-of-sample evidence (i.e., the 3-6 months of paper trading should be your real Fold 4).

The intraday model does not have this issue (Fold 3 = +2.97). But that means the intraday model is the only thing actually carrying the case for live deployment right now.

### 2. The path_quality label is a hand-tuned regression target masquerading as a learned objective

```
path_quality = 1.0 × upside_capture − 1.25 × stop_pressure + 0.25 × close_strength
```

Three problems with this:

- **The weights (1.0, 1.25, 0.25) are free hyperparameters.** They look like they were chosen by intuition or light tuning. They are not learned from data. If those weights were tuned on the same data used for walk-forward, you have a degree of label leakage: you've moved overfitting from the model to the label.
- **Regression on a continuous custom target is the wrong loss for a top-K selection problem.** You are picking the top 10 candidates and trading them. You don't care about the L2 distance between predicted and actual `path_quality` — you care about the *ranking* of candidates. **LambdaRank (XGBoost supports `rank:pairwise` and `rank:ndcg`) is the right framing.** This often produces materially different (better) results for top-K selection problems.
- **Path quality is correlated with realized P&L but is not realized P&L.** The justification "capture-the-move-cleanly" is reasonable, but you should quantify how often path_quality predictions disagree with actual P&L sign. If the disagreement rate is high, the label is a poor proxy.

A direct binary classification — "did this trade hit target before stop?" — is simpler, more interpretable, and avoids the label-weight tuning problem. You can use predicted probability for ranking.

### 3. Universe mismatch between training and inference

| | Training | Inference |
|---|---|---|
| Swing | SP-100 (~81 symbols) | Russell 1000 filtered (~430) |
| Intraday | Russell 1000 (~720) | Russell 1000 (~720) |

Training swing on SP-100 and trading the broader Russell 1000 is **extrapolation**, not interpolation. SP-100 names have:
- Higher liquidity → tighter spreads, lower slippage
- More analyst coverage → faster information dissemination
- Lower idiosyncratic vol → smoother paths
- Different ownership structure (more passive flow)

Features like "relative volume" and "RSI" do not behave the same on AAPL vs. on a $5B mid-cap. The walk-forward Sharpe of +1.18 is on the SP-100 universe; performance on the Russell 1000 inference universe is **unknown**.

**Fix:** train on the full inference universe, or restrict inference to the training universe. The first is preferred (more data, better generalization). The intraday model has no such mismatch — that is one reason its Fold 3 number is more credible.

### 4. The risk framework is a checklist, not a model

What you have:
- Position size limits, max positions, sector concentration, correlation gate, daily loss, kill switch, circuit breaker.

What's missing:
- **Portfolio-level VaR / CVaR.** Given your current 5 swing + 3 intraday positions, what's the 95% one-day VaR of the portfolio? You don't compute it. The sector cap and correlation gate are necessary but not sufficient — two stocks in different GICS sectors can still be 0.8-correlated under macro stress.
- **Beta exposure.** What's the net beta to SPY? If you're systematically long-biased in a high-vol regime, you're effectively running a leveraged SPY trade.
- **Factor exposures.** Net momentum, value, quality, size factor exposure. Without this, you're blind to factor crowding risk.
- **Stress testing.** What does the portfolio do in: -5% SPY day; +20 VIX spike; sector rotation event; rates +50bp surprise? Static-rule risk frameworks miss these.
- **Drawdown response.** Daily loss limit halts for the day. But what happens at -3% over a week? -5% over a month? No de-risking ladder. A real risk framework reduces position sizing as drawdown deepens (Kelly fraction or similar).
- **Tail hedge.** With 5 long positions, no hedge against a fat-tail event. SPY puts are cheap as insurance.
- **Strategy correlation.** Swing and intraday are both long-momentum on US equities. They are almost certainly correlated > 0.5. You're not getting diversification benefit you think you are.

The independent rules approach is brittle because it doesn't compose. Three rules each at 80% effectiveness combine into ~50% effectiveness if they catch overlapping risks. A portfolio model + the rules is a much stronger frame.

### 5. Execution is "limit at ask + 1 tick" — this is a market order in disguise

Setting a buy limit at `ask + 1 tick` on a liquid large-cap is functionally a market order. You're paying:
- Full bid-ask spread (~1-3 bps on liquid names)
- Plus the +1 tick aggression (~0.5-2 bps depending on price)
- Plus market impact (small for $20K orders, but non-zero)

For a $20K account this is not catastrophic — call it 5-10 bps round-trip on each trade — but it's a measurable haircut on a strategy targeting Sharpe 0.5-1.0. **More importantly, you don't measure it.** No implementation shortfall tracking, no per-symbol slippage distribution, no comparison of backtested fills vs realized fills.

This will become a much bigger issue if/when you scale capital. At $200K, market impact starts mattering. At $1M, you need real execution algos.

**The single most useful execution feature you could add right now:** implementation shortfall logging. Compare arrival price (price at proposal time) to filled price, broken down by symbol, time-of-day, and order size. This is one query on data you already have.

### 6. NIS is a black box of unknown alpha contribution

The News Intelligence Service uses Claude (Haiku for symbol-level, Sonnet for macro) to score news. Several concerns:

- **No A/B test of NIS additivity.** Is the strategy with NIS overlay better than the strategy without it? You don't have a controlled comparison. This is a measurable question and you should answer it before continuing to develop NIS.
- **LLM scores are not deterministic.** Same headline can get different scores on different days. This is noise, not signal. You can mitigate with temperature=0 and seed pinning, but model updates from Anthropic break reproducibility.
- **Calibration is unknown.** When NIS says "materiality=0.8, direction=+, confidence=0.9" — what's the empirical hit rate? You should track this and recalibrate.
- **Cost / latency.** Scoring 720 symbols × ~3 times/day at Haiku rates is fine, but it's not free, and the latency on the API call adds up.
- **Hallucination risk.** A confidently-scored hallucinated headline → a real trade decision. The signal-to-noise ratio of Finnhub free-tier headlines is also a question — most "news" on free tiers is rehashed wire content with low alpha.

What real quant funds do: vendor sentiment scores (RavenPack, Bloomberg ESS, S&P GMI Sentiment) that have been backtested over decades, calibrated, and integrated as model features rather than as overlays. **Recommendation:** treat NIS as a feature input to the ML model, not as a post-hoc overlay. And A/B test it.

### 7. The agent architecture is over-engineered and the bug list proves it

Three async coroutines + queues + audit tables + reconciliation + flag restoration is a lot of machinery for what is, structurally, a daily pipeline:

```
fetch_data → score → risk_check → execute → monitor_exits
```

Look at the "Known Bugs Fixed" list:
- #1 (`bars_held` counting heartbeat ticks): caused by async event loop conflating real time and trading time
- #4 (in-memory daily flags reset on restart): caused by storing state in coroutines instead of in DB
- #5 (reconciler creating duplicates): caused by complex startup logic
- #6 (runaway scanner): caused by missing market-hours guard in a coroutine that should not have been running
- #8 (uvicorn shutdown hang): caused by async task cancellation complexity
- #9 (timezone mismatch in flag restore): caused by complex startup logic

**Six of nine bugs are architectural, not domain.** The "agents" framing makes the system sound more sophisticated than it is — these are not LLM agents, they are three coroutines in one process passing dicts. The real thing it buys you (clean separation) you could get with three modules and a synchronous pipeline.

A simpler architecture for the same functionality:
- **Cron-driven (or APScheduler) synchronous pipeline:** at 08:00 ET, run `analyze_swing()`. At 09:50, run `propose_swing()`. At each scan window, run `scan_intraday()`. Each function is synchronous, transactional, restartable.
- **No queues.** Functions return values; calling code persists results.
- **No "active positions" in-memory dict.** Read from DB on every check.
- **No reconciliation needed** because state lives in DB, not memory.

This would cut the bug surface area by 50%+ and make the system far easier to reason about. The async/queue model is appropriate when you have I/O-bound long-running tasks with backpressure, not for "do this thing 4 times a day."

I recognize this is a major rewrite — but the question to ask before Phase 75/76/77 work is: are you adding features to the right architecture, or papering over the wrong one?

### 8. The 3-fold TimeSeriesSplit is too few folds, full stop

3 folds is the minimum and produces unstable Sharpe estimates. Standard practice for production-bound quant ML:
- **5-10 folds** with rolling or anchored window
- **Bootstrap CIs** on each fold's metrics
- **Sensitivity analysis:** does the result hold if you shift fold boundaries by ±2 weeks?
- **Combinatorial purged cross-validation (CPCV)** for trade-level overlap (de Prado's method) is the gold standard

With 3 folds and N=134 in the worst fold, your error bars dwarf your effect size. Until you've fixed this, every other ML decision you make is being made on noise.

### 9. The "Bar 12 / opening session edge" is probably partially overfit

The intraday model entry is bar 12 (60-min from open). Things to verify:

- Was bar 12 chosen because you tested all 78 bars and picked the winner? If so, you've done implicit multiple comparisons and the apparent edge is partly p-hacking.
- Does the edge survive at bars 11 and 13? If only bar 12 works, it's overfit. If bars 10-14 all work, it's robust.
- The "opening session edge" in the academic literature is real but has been studied for 30 years and is heavily traded. The persistence of edge in a crowded factor is suspect.

The Fold 3 Sharpe of +2.97 on intraday is encouraging but does not refute this concern — three folds of a single bar choice is not a robustness test.

### 10. Statistical power on paper trading gate is insufficient

> Gate: 4-week Sharpe > 0.5, max drawdown < 5%

With ~14 trades in 3 days, you'll have ~90-100 trades in 4 weeks. The standard error on the Sharpe estimate is roughly `sqrt((1 + 0.5×Sharpe²) / N)` ≈ 0.11 per year fraction. For a 4-week observation, scale that. **A 4-week Sharpe of 0.5 has a 95% CI of roughly [-0.3, +1.3].** You can pass that gate by luck.

**Recommended gate:** 3-month minimum paper trading window with stable parameters, Sharpe lower bound (not point estimate) > 0.3, max drawdown < 8%, and a Deflated Sharpe Ratio (DSR) test to account for multiple-comparison bias from your earlier model versions. The 4-week / Sharpe 0.5 / 5% drawdown gate is too easy.

---

## Specific Answers to Your 10 Questions

### Q1. Is regression on path_quality the right formulation?

**No.** For top-K selection: rank-based learning (XGBoost `rank:pairwise` or LambdaRank) is the right tool. Alternative: binary classification on "trade reaches target before stop" with predicted probability used for ranking. The current regression approach loses information about ordering and introduces label-weight free parameters. **Action:** retrain with both alternatives, compare on the same walk-forward, pick the winner.

### Q2. Where is the most alpha being left on the table?

In rough order of expected impact for a long-only US equity swing strategy of this size:
1. **Microstructure / order flow features** (CVD, lit-vs-dark imbalance, signed volume) — but Alpaca/Polygon free tier doesn't give you the data
2. **Cross-asset / macro features** (dollar strength, rates curve shape, credit spreads) — easy to add from FRED
3. **Earnings revision momentum** — hugely predictive, free from FMP
4. **Short interest changes** — FINRA Reg SHO + biweekly NYSE/Nasdaq short interest
5. **Insider buying clusters** — SEC Form 4, free
6. **Options skew / put-call ratio** — Polygon premium, but worth it

You have FMP and SEC EDGAR access already and aren't using them in features. **That's the biggest miss.**

### Q3. What causes Fold 3 Sharpe of -0.03?

Three plausible (and partially compatible) hypotheses:
1. **Regime change.** 2022-2024 was mega-cap momentum dominated; if early 2025 saw breadth thrust, factor rotation, or low-vol mean-reversion regime, your model is mis-specified.
2. **Crowding.** Path-quality momentum + top-decile selection is what every other systematic shop is doing. Excess returns compress as more capital chases the same signal.
3. **Universe drift.** Russell 1000 inference + SP-100 training is more lossy than it looks; if 2025 had idiosyncratic mid-cap dispersion that the SP-100-trained model wasn't equipped to handle.

**How to address:** retrain with a regime-conditioned feature set (interact features with VIX percentile, breadth, term-structure regime), expand training universe to match inference universe, and hold out the 2025-2026 period as final out-of-sample with no parameter tuning whatsoever.

### Q4. How to validate Bar 12 isn't a backtest artifact?

- Sensitivity test on bars 10-14: if performance is monotonic and similar across, robust. If only bar 12 works, overfit.
- White Reality Check or Hansen SPA test for multiple-comparison correction.
- Out-of-sample on a different universe (e.g., ETFs only) — does the same bar work? If yes, it's microstructural; if no, it's specific to the training set.
- Time-decay test: does the edge get weaker each year? If yes, it's being arbed away.

### Q5. Is "limit at ask + 1 tick" right?

For $20K on liquid large-caps: it's *defensible* but unmeasured. **Adding TCA infrastructure is more important than changing the order type.** Once you have implementation shortfall data, you can A/B test alternatives:
- Pegged-to-mid limit, stepping up if not filled in 30s
- POV (% of volume) participation if you ever scale beyond $100K
- Adverse-selection-aware orders (don't lift offer if the offer just dropped)

For now, ask+1 tick is fine. Just measure it.

### Q6. Is there a portfolio-level risk model wrapping all this?

No, and there should be. See Critical Finding #4. Minimum: parametric VaR daily, stress test scenarios weekly, beta exposure tracked continuously. CVaR > VaR for fat tails. Factor model (Barra-lite) eventually.

### Q7. How would a real quant fund use news data?

- **Vendor scores, not LLM scores.** RavenPack, Bloomberg ESS, S&P GMI. Decades of calibration, point-in-time corrections, cross-asset coverage.
- **Features, not overlays.** News features go *into* the ML model. The model learns the conditional relationship between news sentiment and forward returns. Overlays let humans interpose intuition that may be wrong.
- **Event studies before deployment.** "What does the strategy do in the 2 hours after a positive news event?" Quantified empirically, not assumed.
- **Decay modeling.** News alpha decays in minutes-to-hours. The half-life of a Finnhub headline is probably <2 hours. NIS scoring at 30-min intervals may be too slow for the actual signal.

You don't need to abandon NIS — but treat it as one feature, A/B test its additivity, and calibrate it. If NIS is alpha-additive, prove it. If it's not, kill it.

### Q8. What does a proper regime model look like?

Multiple layers, ideally:
1. **Volatility regime:** VIX level + term structure (VIX/VIX3M ratio = contango vs backwardation) → 4 states
2. **Trend regime:** SPY 50/200 MA + breadth (% stocks above 50-day MA) → trend / chop / mean-revert
3. **Macro regime:** rates regime (curve shape), dollar regime (DXY trend), credit regime (HY OAS)
4. **Sentiment regime:** AAII bull-bear, put/call ratio, NAAIM exposure index

Output is a regime label (e.g., "low-vol trend, easing rates") and the model has different parameters / different selection thresholds in each regime. HMMs and switching state-space models exist for this. Simpler version: train separate models per regime, route by current regime. Even simpler: regime as a categorical feature in the existing model.

Your current binary "VIX≥25 OR SPY<MA20 OR SPY 5d≤0 → abstain" is a 3-rule classifier. It's better than nothing. It's not enough.

### Q9. Three most likely failure modes paper trading won't catch

1. **Adverse fill quality / partial fill cascades.** Paper fills happen at the quoted price; live fills happen at marketable price after queue position. On illiquid mid-caps especially, expect 2-5x your paper-trading slippage.
2. **Latency-driven entry slippage on intraday.** The intraday strategy enters at bar 12. Paper trading fills you at bar-12 close price. Live trading at bar 12 close means your order arrives in bar 13, after the move you were trying to capture. The fix is to enter on bar 11 close or use limit orders priced for bar 12 close — but the paper model doesn't see this.
3. **Broker / data outages.** Alpaca has had multiple multi-hour outages in the last 18 months. Your circuit breaker handles it, but the *recovery* is what bites — positions in unexpected states, missed exits, max-hold timers misfiring. Plus IEX feed has gaps that don't appear in cached Polygon data; live, you'll see them.

Honorable mentions: borrow availability if you ever short, after-hours position drift (overnight gaps eating stop loss), and corporate actions (splits, dividends) silently breaking ATR calculations.

### Q10. Top 5 priorities to reach institutional grade

1. **TCA + execution analytics.** You cannot manage what you do not measure.
2. **Portfolio risk model** (VaR/CVaR + factor exposure + stress tests).
3. **Statistical rigor on the model** — bootstrap CIs, more folds, ranking objective, larger training universe, drift monitoring.
4. **Feature richness** — earnings revisions, short interest, insider activity, macro/cross-asset, options data.
5. **Infrastructure simplification + observability** — fewer moving parts, better metrics, clearer failure modes. Plus a real disaster recovery plan (broker outage, data outage, model staleness alarms).

---

## Architectural Critique: PM / RM / Trader Design

You asked me to take this apart. Here's the honest read:

### What it gets right

- Veto power for RM is the correct pattern. PM proposes, RM disposes. This separation is non-negotiable.
- Decision audit at every gate boundary is excellent.
- Reconciliation logic exists, even if it's complex.

### What's questionable

**The "agent" framing oversells what's there.** These are three async functions, not autonomous agents. The agent framing leads to architectural decisions that don't make sense:
- Agents have their own loops → race conditions
- Agents communicate via queues → message loss on restart
- Agents hold in-memory state → drift from DB
- Each agent needs its own reconciliation → complexity grows quadratically

**The PM is doing too much.** PM is responsible for:
- Watchlist scoring (data + ML)
- Candidate selection (ranking)
- Pre-trade gates (rules)
- NIS overlay (LLM)
- Re-scoring (Phase 70)
- EOD review (Phase 75 planned)
- Intraday position re-evaluation (Phase 80 planned)

Pull these apart. **A scoring service** (data → ML → score), **a candidate selector** (score → top-K → gates), **a reviewer** (held positions → re-score → exit signal). The PM coalesces concerns that should be separable for testing and replacement.

**RM is sequential validation.** That's fine, but it's not a portfolio model. RM should also expose `compute_portfolio_risk()` returning VaR, beta, factor exposure — even if the rule chain doesn't use it yet, downstream consumers (UI, alerts, daily reports) need it.

**Trader is doing exit management AND order placement AND reconciliation.** The exit logic deserves to be its own module — it has 8 prioritized rules, time-based logic, state machine semantics. Mixing it with order placement makes both harder to test.

### Recommended decomposition (if you do nothing else, consider this)

```
data/        # Polygon, Alpaca, SEC, FMP — pure I/O
features/    # OHLCV → feature vectors — pure compute
models/      # ML training + inference — pure compute
policies/    # NIS, gate definitions — pure compute
selector/    # candidate selection logic
risk/        # portfolio risk + rule chain
execution/   # order placement, fills, reconciliation
exits/       # exit decision state machine
scheduler/   # APScheduler orchestrating all of above synchronously
audit/       # logging, decision capture
```

No agents, no queues, no in-memory state. Schedule jobs, run them synchronously, persist outputs, repeat. This would be ~30% less code with ~70% less restart-edge-case risk.

I am not saying drop everything and rewrite — but the next architectural decision (where do Phase 75, 80, 81 live?) should be made in light of this. Adding more agent logic to an already over-agented system compounds the problem.

---

## Statistical / Model Critique (Detailed)

| Concern | Severity | Recommended action |
|---|---|---|
| Path quality label is hand-weighted | High | A/B test against rank objective + binary classification; check label correlation with actual P&L sign |
| 3-fold TimeSeriesSplit | High | 5-10 folds + bootstrap CIs + sensitivity test on fold boundaries |
| Training/inference universe mismatch (swing) | High | Retrain on full Russell 1000 inference universe |
| No purged cross-validation for trade overlap | Medium | Implement de Prado CPCV or at minimum embargo overlapping trades |
| Model retrains daily with no drift monitoring | Medium | Phase 79 (AUC drift alert) is correct — also add PSI on features |
| MetaLabel R² of 0.059 | Medium-Low | This is borderline noise. Either drop it or retrain with more signal. |
| No ensemble / single-model risk | Medium | Train LightGBM and XGBoost variants, ensemble; reduces single-model failure risk |
| Survivorship bias in training universe | Medium | Use point-in-time index membership, not current SP-100 |
| Label leakage check | Medium | Audit each feature for forward-looking content; build leakage detection test |
| Sample size in walk-forward | High | Bootstrap Sharpe CIs are mandatory before quoting numbers |

---

## Risk Framework Gaps (Detailed)

| Gap | Why it matters | Effort |
|---|---|---|
| Portfolio VaR/CVaR | Single most important missing piece. Lets you size to actual risk, not nominal capital. | Medium (parametric VaR is a few hundred lines) |
| Net beta to SPY | Tells you when you're long the market vs running idiosyncratic risk | Low |
| Factor exposure | Detects crowding (everyone long momentum at the same time) | Medium-High (need factor returns) |
| Stress tests | Quantifies tail risk; required for any institutional pitch | Medium (scenario library + revaluation logic) |
| Drawdown response ladder | Reduces sizing as drawdown grows; prevents ruin | Low |
| Tail hedge (SPY puts) | Cheap insurance for fat tails | Low |
| Strategy correlation | Are swing and intraday actually independent? Almost certainly not. Quantify it. | Low |
| Capacity analysis | At what AUM does the strategy break? | Medium |

---

## Execution Quality (Detailed)

You have:
- Limit orders at ask+1 tick
- Spread filter < 0.5%
- Volume filter ≥ 50% of 20-day average
- Final entry quality re-check at execution time

You don't have:
- Implementation shortfall (arrival vs filled price)
- Per-symbol slippage distribution
- Time-of-day slippage distribution
- Backtest-vs-live P&L gap analysis
- Order timeout analysis (how often does the 5-min TTL fire?)
- Fill-rate tracking (% of approved proposals that actually fill)

The fix is mostly logging. You already have `slippage_bps` per fill — you just don't surface it (Phase 76 is about this and should be promoted).

**Suggested execution dashboard panel:**
- Fill rate today / 7-day / 30-day
- Mean and 95th percentile slippage in bps
- Per-symbol slippage outliers
- Backtest expected vs realized P&L (running gap)

---

## NIS Critique (Detailed)

The NIS is the most novel and most under-validated component. Specific recommendations:

1. **A/B test additivity.** Run paper trading with NIS overlay enabled and disabled in parallel for 30 days. Compare Sharpe, win rate, drawdown. If NIS doesn't add measurable alpha, kill it or rebuild it.
2. **Calibration tracking.** When NIS scores `materiality=0.8, direction=+, confidence=0.9`, what's the empirical 5-day forward return distribution? Build a calibration plot. If it's flat (NIS scores don't predict returns), the LLM is producing noise.
3. **Move from overlay to feature.** Treat NIS scores as model features, not as post-hoc gates. Let the ML model learn how to weight them.
4. **Pin model versions.** Anthropic ships model updates; your "scorer" can drift silently. Log the exact model version used for every score.
5. **Cap LLM cost growth.** Symbol-level scoring at 720 symbols × 3x/day × Haiku rates is ~$X/day; track this as a metric and alert on anomalies.
6. **Backstop heuristics.** If NIS API fails, do you fall back? What's the behavior?

Bigger question: is there a simpler heuristic that captures 80% of NIS's value? E.g., "block entries on stocks with FOMC/earnings within 3 days, on stocks that gapped >5%, on stocks with 5x normal volume in the last 30 minutes." That's 3 rules and no LLM. If those rules capture most of the lift, the LLM is buying you complexity without proportional alpha.

---

## Operational Readiness Gaps

| Gap | Risk | Action |
|---|---|---|
| No runbook for broker outage | High in live | Document recovery procedure: detect → flat → wait → reconcile → resume |
| No model staleness alarm | Medium | Alert if model age > 48h or if last successful retrain > 24h ago |
| No PnL reconciliation alarm | Medium | Daily check: sum of trades P&L vs Alpaca account P&L; alert on mismatch |
| No data quality alarm | Medium | Alert on >5% missing bars, stale features, API rate-limit hits |
| Local Windows deployment | High in live | Single point of failure (power, ISP, OS update). Cloud migration is correctly deferred but should not be deferred indefinitely. |
| Time sync | Medium | NTP discipline on the trading host; clock skew → exit timer bugs |
| No incident response process | Medium | "Runaway scanner" (Bug #6) bought 7-9 duplicates. What's your max-loss-from-bug bound? |
| No paper-vs-live divergence dashboard | High when you flip live | Run paper and live in parallel for 2-4 weeks; track per-trade divergence |

The runaway scanner bug is worth dwelling on: a missing market-hours check turned into 7-9x duplicate buys. In live trading that would have been a real loss. **What is the structural protection against the next bug like that?** "Tests" is a good answer but not sufficient. Pre-trade hard limits (e.g., "no more than N orders in X minutes regardless of source") would have caught it. RM should have this as a rule.

---

## Prioritized Phase Plan

The current Phase 75/76/77/78/79/80/81 list is reasonable feature work, but it's working on the wrong tier of problem. You should reorder around impact.

### Phase 0: Pre-Live Gating (must complete before live trading; no exceptions)

| # | Item | Effort | Why |
|---|---|---|---|
| 0.1 | Bootstrap Sharpe CIs on all walk-forward folds | Days | Tells you if Fold 3 = -0.03 is "noise" or "broken" |
| 0.2 | Fold 3 root cause analysis (regime, crowding, universe drift) | 1-2 weeks | Required before deploying current model |
| 0.3 | Retrain swing on Russell 1000 universe (match inference) | Days | Eliminates extrapolation risk |
| 0.4 | Re-do walk-forward with 5+ folds, purged CV | 1 week | Current 3-fold is statistically thin |
| 0.5 | NIS A/B additivity test (30-day paper) | 1 month | Stop building on NIS until you know it adds alpha |
| 0.6 | Bar 12 sensitivity test (bars 10-14 + multi-test correction) | Days | Confirms intraday edge isn't artifact |
| 0.7 | Implementation shortfall + slippage dashboard (= Phase 76 promoted) | 1 week | TCA baseline before live |
| 0.8 | Tighten paper-trading gate to 3-month + DSR | (gate, not work) | Current gate is too easy to pass |

**If 0.1-0.4 reveal the swing edge has decayed, do not deploy swing.** Run intraday-only until you've rebuilt swing. This is a real possibility based on the data you have today.

### Phase 1: Risk Architecture (in parallel with Phase 0)

| # | Item | Effort |
|---|---|---|
| 1.1 | Portfolio VaR/CVaR (parametric, then historical) | 1-2 weeks |
| 1.2 | Net beta to SPY tracking | Days |
| 1.3 | Stress test framework (5 named scenarios) | 1-2 weeks |
| 1.4 | Drawdown response ladder (sizing reduction at -3%, -5%, -7% running) | Days |
| 1.5 | Strategy correlation (swing vs intraday actual correlation) | Days |
| 1.6 | Pre-trade hard limits (max orders/minute regardless of source) | Days |
| 1.7 | Tail hedge proof of concept (cheap SPY puts as % of portfolio) | 1 week |

### Phase 2: Live Readiness (after Phase 0 passes)

| # | Item | Effort |
|---|---|---|
| 2.1 | Disaster recovery runbook (broker outage, data outage, host failure) | 1 week |
| 2.2 | Paper-vs-live parallel run for 2-4 weeks (gradual capital ramp) | 1 month |
| 2.3 | Model staleness + drift alarms (= Phase 79 promoted) | Days |
| 2.4 | PnL reconciliation alarm (DB vs broker) | Days |
| 2.5 | Cloud migration POC (one provider, basic setup) | 2 weeks |
| 2.6 | Graceful SIGTERM (= Phase 77, keep) | Days |
| 2.7 | Live readiness checklist refresh (= Phase 78, keep) | Days |

### Phase 3: Architecture Refactor (deferred but plan now)

| # | Item | Effort |
|---|---|---|
| 3.1 | Decompose PM into Scoring + Selector + Reviewer modules | 2-3 weeks |
| 3.2 | Replace agent/queue model with synchronous scheduler pipeline | 3-4 weeks |
| 3.3 | Consolidate state in DB (kill in-memory `active_positions`) | 1 week |
| 3.4 | Exit logic as standalone state machine module | 1-2 weeks |

You may not want to do this — it's a real rewrite. But every new feature added to the current architecture compounds debt. The PM intraday review (Phase 81) and EOD review (Phase 75) should be designed to fit a future-decomposed system, not the current one.

### Phase 4: Alpha Improvements (after live is stable)

| # | Item | Effort |
|---|---|---|
| 4.1 | Earnings revision features (free from FMP) | 1 week |
| 4.2 | Short interest features (FINRA Reg SHO + biweekly) | 1 week |
| 4.3 | Insider activity features (SEC Form 4) | 1-2 weeks |
| 4.4 | Macro/cross-asset features (FRED rates, DXY, HY OAS) | 1 week |
| 4.5 | LambdaRank model variant + ensemble | 2 weeks |
| 4.6 | Regime conditioning (separate models per regime, or regime as feature) | 2-3 weeks |
| 4.7 | NIS as feature (not overlay) — only if Phase 0.5 confirmed alpha | 2 weeks |
| 4.8 | True intraday scalper (only after current system is stable in live) | 4-6 weeks |

### Phase 5: Scaling (when AUM justifies)

| # | Item | Why |
|---|---|---|
| 5.1 | Vendor news/sentiment data (RavenPack-class) | NIS plateau / replacement |
| 5.2 | Polygon premium (options flow, L2) | Better features + better execution |
| 5.3 | Strategy capacity analysis | Know your scale ceiling |
| 5.4 | Smart order routing (if scaling beyond $200K) | Execution alpha |
| 5.5 | Multi-account / margin structure | When personal capital exceeds $100K |

---

## Anti-Patterns to Watch For

These are traps you're at risk of falling into based on the document:

1. **Treating Fold 3 as "noise" without proving it.** The temptation is to say "average is fine, recent fold is just unlucky." It might be. It also might not be. Get the CIs.
2. **Adding more agents.** Phase 80, 81 are more agent logic. Stop and ask if a different architecture would handle these better.
3. **NIS feature creep.** Adding more LLM scoring without first validating additivity is sunk cost in disguise.
4. **Building features without TCA.** Every new feature you ship is a hypothesis; without TCA you can't tell if alpha is being eaten by execution.
5. **Trusting paper trading.** Paper trading does not produce realistic fills, doesn't expose data feed gaps, doesn't surface borrow/locate issues. Your gate criteria are also too easy.
6. **Over-relying on 4-week numbers.** 4 weeks of trading is N≈100 trades. Any conclusion drawn from 100 trades has wide error bars. Don't make irreversible decisions from short windows.
7. **Treating "passes the gate" as "ready for live."** Gates should be necessary, not sufficient. After the gate, you want a separate human review.
8. **Ignoring strategy correlation.** Swing and intraday almost certainly correlate; treating them as 2 strategies when they're 1.5 inflates your perceived diversification.
9. **Deferring cloud migration.** Local Windows is a single point of failure. Defer is fine; defer indefinitely is not.
10. **Forgetting model versioning of NIS.** Anthropic ships updates. Pin the model string. Without it, your audit trail is incomplete.

---

## What I'd Do First If This Were My System

In this order:

1. **Today / this week:** Bootstrap Sharpe CIs on all three swing folds. Build the implementation shortfall query. Write a 2-page Fold 3 RCA hypothesis document.
2. **Next 2 weeks:** Retrain swing on full inference universe. Re-run with 5 folds. Set up NIS A/B (or commit to disabling NIS for 30 days as the control).
3. **Next month:** Portfolio VaR + beta tracking + stress tests. Drawdown response ladder. Pre-trade hard limits in RM.
4. **Next 1-2 months:** Decide on architecture path. Either commit to the current agent design and fix Phase 75/77/79 properly, or start the synchronous-pipeline refactor. Don't be in between.
5. **Then:** Live readiness in earnest. Paper-vs-live parallel run. Cloud migration POC. Disaster recovery runbook.

The thing I would *not* do: ship Phase 75 / 80 / 81 as currently scoped before the model edge questions are resolved. You'd be adding logic to a model that may not have the alpha you think it does.

---

## Closing Notes

This system is further along than most personal trading bots. The audit infrastructure, test culture, and reconciliation logic are evidence of a builder who has been bitten before and is taking the right lessons. None of this critique is meant to suggest the foundation is bad.

The critique is: **the foundation is good enough that the edge questions and risk gaps deserve direct attention, not deferral.** A system at this maturity level is at the point where the next 10% of work decides whether it's a real strategy or a 12-month sunk cost. That 10% is mostly about validating that the alpha is real, building risk infrastructure that composes, and instrumenting execution so you can tell what's happening.

Push back on any of this you disagree with. Several of the points are based on inference from the document; you may have evidence I don't see (e.g., maybe the path_quality weights *were* learned, maybe there *is* TCA tooling not mentioned, maybe Fold 3 *has* been root-caused). Where I'm wrong, correct me.

---

*End of review. Total length is intentional — written for downstream LLM ingestion + your own use as a working document. Section anchors should make navigation straightforward.*
