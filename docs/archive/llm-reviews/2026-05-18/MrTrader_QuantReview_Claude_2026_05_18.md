# MrTrader — Senior Quant Review

**Reviewer perspective:** ex-buy-side quant, multi-strategy fund, equity systematic PM.
**Date:** 2026-05-18
**Tone:** brutal as requested.
**Reading time:** ~20 minutes. Sections are independent — skim or deep-read as needed.

---

## TL;DR — The Three Things That Matter

1. **You don't have a labeling problem, a model problem, or a feature problem. You have a structural problem: long-only, $20k capital, daily rebalance, R1000 universe, retail costs.** Each of those individually is fine. The combination is mathematically very close to unreachable. Your symptoms (Fold 2 collapse, LambdaRank failure, factor portfolio "working" only without stops) are all downstream of this.

2. **The factor portfolio Sharpe of 1.335 is almost certainly not real alpha.** It's likely 70–90% beta to SPY + MTUM + QMJ + a backtest period (2019–2024) that was extraordinarily friendly to momentum + quality. The execution-mismatch result (−1.43 with ATR stops) is your system *telling you the truth*: the strategy only "works" when you turn off realistic risk management.

3. **You have never run the single most informative diagnostic in this entire project: Information Coefficient (IC) of your ranker output vs. realized forward returns.** You're tuning Sharpe and NDCG, which are downstream of IC. Run IC analysis on a held-out fold before doing anything else — that one number tells you whether there is ANY signal in your features at this universe/horizon.

If you do nothing else from this review, do those three things. The rest of this document explains why and what to do next.

---

## Part 1 — Specific Answers to Your 7 Questions

### Q1: Is LambdaRank the right architecture?

**No, but not for the reasons you think.**

LambdaRank cross-sectional ranking is a legitimate institutional approach (it's what Renaissance, AQR, Two Sigma, etc. functionally do — rank stocks cross-sectionally and trade the spread). The mathematical machinery is fine. **The problem is that it implicitly assumes you can trade the spread (long top decile, short bottom decile).** You're long-only.

When you take only the top of the rank in a long-only system:
- In bull markets, top quintile = "fastest momentum, highest quality" → makes money
- In bear markets, top quintile = "fell least, most defensive" → completely different signal
- The model can't learn one coherent function because **the label semantics flip with regime**

This is not a labeling bug. It's a structural incompatibility between cross-sectional ranking (which is regime-neutral if you trade the long-short spread) and long-only execution (which is regime-exposed). Your "Fold 2 is structurally unreachable" conclusion is exactly right and no amount of HPO/feature engineering will fix it.

**What I'd use instead (in order of how much I'd actually believe in it):**

| Approach | Why it fits your constraints | Realistic Sharpe |
|---|---|---|
| **Concentrated long-only momentum + quality, monthly rebalance, regime gate to cash** | Lower turnover (cost-friendly), regime gate handles the long-only bear problem, no model needed | 0.5–0.8 live |
| **Post-Earnings Announcement Drift (PEAD)** | Event-driven (only trades on earnings days), small universe per day, persistent academic anomaly | 0.7–1.0 live with discipline |
| **Sector rotation (11 sector ETFs)** | Tiny universe, low cost, low turnover, $20k can actually take meaningful positions | 0.4–0.6 live |
| **Binary classifier (XGBoost) with absolute return labels + regime gate** | What you're considering. Survives long-only because absolute label is regime-neutral by construction | 0.4–0.6 live IF features have IC ≥ 0.02 |
| **LambdaRank with regime-conditional routing** | What you're considering. Marginal improvement over single model. The issue isn't model class. | 0.2–0.4 live, not worth the complexity |

### Q2: Is your walk-forward methodology honest?

**Mostly yes, but with three meaningful flaws.**

What you've done right (better than 90% of retail quant):
- Purge + embargo (Lopez de Prado standard)
- PIT regime score
- Transaction costs included
- Partial survivorship fix
- Expanding window
- TSNorm fit on train only

What's wrong or missing:

1. **You're using single-pass walk-forward, not CPCV.** Combinatorial Purged Cross-Validation (López de Prado, *Advances in Financial ML*, Ch. 12) generates many more train/test paths from the same data and computes a *distribution* of out-of-sample Sharpes. Your 5 folds give you 5 datapoints. CPCV with k=6, test=2 gives you 15 paths. This dramatically reduces overfitting to fold boundaries.

2. **You have no Deflated Sharpe Ratio (DSR).** With 9 LambdaRank training runs + multiple feature configurations + HPO + multiple labeling schemes, your effective number of trials is well over 100. DSR penalizes raw Sharpe for the number of trials. A raw +0.27 Sharpe from v205b has a DSR that's probably ≈ 0.0 — i.e., indistinguishable from random. **You correctly identified this intuitively ("v205b is likely a lucky draw"). DSR formalizes it.**

3. **Your AgentSimulator entry price of "previous close × 1.001" is wrong and probably biased.** Real next-day opens on momentum names have a positively-skewed gap distribution. You should use actual daily open prices from yfinance. The 10-bps adjustment likely *understates* slippage on the names you're most likely to pick (momentum top decile = often gappers).

4. **Your "min fold ≥ −0.30" gate is unrealistically strict.** AQR Style Premia had 2014, 2018, and 2020 drawdowns of multiple-standard-deviation magnitude. Asking for "never has a bad year" on 5-fold WF effectively requires the strategy to be regime-immune, which long-only equity systems definitionally cannot be. **You're filtering out viable strategies with this gate.**

5. **No bar-level slippage model for the simulator stops.** When you say "stop at entry − 0.5 × ATR," in real life you don't get filled at the stop price — you get filled wherever the next print is, which during a sell-off is often 50–200 bps worse. For a system targeting ~1.5 ATR profit and 0.5 ATR loss, this asymmetric slippage is structurally important.

### Q3: How do professional quant systems handle the Fold 2 (bear market) problem?

Honestly, three ways:

1. **They go long-short** so the regime doesn't matter — the spread is what they trade. **Not available to you.**
2. **They go to cash** via regime gating. (This is what you have via BenignGate / regime model. You should re-enable it for the factor portfolio.) The result is a *lower-Sharpe-but-positive* strategy, not the "consistently positive every year" strategy you're gating for.
3. **They run multiple strategies that are diversifying** (momentum + value + carry + low-vol) so that the long-only momentum drawdown in 2022 is partially offset by long-only low-vol winning. **This is what you should consider but you're trying to build ONE strategy that survives all regimes.**

Regime-conditional labeling sounds smart but is structurally problematic:
- Bear regimes are ~15–20% of history → tiny training set for bear model → overfit
- Regime classifier itself has noise → mis-classification propagates → instability at boundaries
- You add complexity without clear additional Sharpe

**My recommendation:** Don't do regime-conditional labels. Use regime as a feature in a single model AND as a gate for going to cash. If regime score < threshold, position = 0. Don't try to make money in bear markets long-only — accept the drawdown or sit out.

### Q4: Factor portfolio vs. ML — is factor better here?

**A rule-based factor portfolio is genuinely better than ML for your situation,** but probably not for the reasons it seems on the surface.

The truth about your 1.335 Sharpe factor portfolio:

| Component | Estimated contribution to your 1.335 Sharpe |
|---|---|
| Market beta (SPY rip 2019–2024) | ~0.6 |
| MOM factor exposure (momentum did well) | ~0.3 |
| QMJ / Quality factor exposure | ~0.2 |
| Period selection bias (2019–2024 was MOM+QMJ heaven) | ~0.2 |
| Genuine residual alpha | ~0.0 (most likely) |
| **Total** | **~1.3** |

**This is not a hypothesis — this is what factor decomposition almost always shows for retail-built factor portfolios.** AQR publishes monthly returns for momentum and quality factors going back decades. Long-only momentum on Russell 1000 has historically delivered Sharpe ≈ 0.4–0.6. Adding quality bumps it to ~0.5–0.7. **A Sharpe of 1.3+ on a 6-year period that included 2020 V-recovery and 2023 AI rally is the period, not your alpha.**

How to verify (this is the single most important experiment to run):
1. Pull AQR's published MOM and QMJ factor returns
2. Regress your factor portfolio's daily returns on (SPY, MOM, QMJ)
3. Look at the alpha intercept. If it's near zero (which it probably is), you don't have alpha — you have factor beta.
4. Run your portfolio on 2007–2019 (out of sample to the period you built it on). I expect Sharpe to be ~0.5–0.7, not 1.3.

**But here's the kicker:** Even if your portfolio is "just" 0.5–0.7 Sharpe of MOM + QMJ exposure with no alpha, **that's still better than what your ML system can produce after costs**, and it's a real, robust, academically validated thing. Long-only momentum + quality is a real factor portfolio. Just be honest about what it is.

So: factor portfolio > ML for you, but you need to:
- Honestly characterize it as factor exposure, not alpha
- Validate on a real out-of-sample period (2007–2019)
- Match the execution model to the design (monthly rebalance, no intra-month stops, accept drawdowns)

### Q5: Are your execution assumptions realistic?

Mostly yes for $20k size, mostly no for the assumptions underlying the stops.

**Realistic for $20k on R1000:**
- Entry at next open: yes (use actual open prices, not prev close × 1.001)
- 5 bps round-trip: actually conservative for Alpaca — you'll do better
- No market impact: correct, your size is irrelevant
- Fills happen: correct, R1000 is liquid

**Unrealistic:**
- **ATR stops fire at the stop price**: in real life, a stop-loss order in a fast-down market fills 50–200 bps worse. Your backtest doesn't model gap-down opens that skip your stop entirely.
- **Daily re-scoring and replacement**: you're modeling that the model re-ranks every day and you swap positions. Each swap costs ~10 bps round-trip. At top-10 portfolio with 5% per position and 20-day average hold, even 20% turnover per week = 2% drag per month = 24% annualized. **This alone could destroy any realistic Sharpe.**
- **Filling on "open" for catalyst names**: any name that gaps up 5%+ overnight from your signal, you're not buying anywhere near the previous close + 10bps. You're buying near the open which is the gap. You may want to model this as a "skip if gap > X%" rule and see how much it changes results.

### Q6: Is intraday viable at $20k?

**Mathematically: borderline impossible.**

Let's do the math:
- 15 bps per side = 30 bps roundtrip = $3 per $1,000 traded
- Target return = 0.5–2% per trade = $5–$20 on $1,000
- Net per winning trade = $2–$17
- Required hit rate at 1:1 risk/reward at 0.5% target = $5/$8 worst-case stop = **62% just to break even**

A 62% hit rate intraday on 5-min bar signals with $20k retail account is **not realistic for any retail system.** Real intraday HFT shops run at 51–54% with millions of trades and microseconds of edge. Retail intraday systems that *do* work (yes, they exist) typically:
- Have a *catalyst* (earnings, news, opening range breakout)
- Trade *fewer, higher-conviction* setups (1–3 per day, not 5)
- Have a *meaningful* asymmetric R:R (2:1 or 3:1), not 1:1
- Often pair with day-trading discipline, not algo execution

Your intraday +0.529 backtest Sharpe is probably 0.0–0.3 live after the slippage and queue-position realities you're not modeling.

**Recommendation:** Either shut down intraday, OR pivot it to event-driven (earnings gap, post-earnings drift, FDA decisions, etc.) where the edge is structural rather than statistical. Don't try to be a 5-minute bar momentum/mean-revert trader at $20k.

### Q7: What would I build next?

See Part 3 below for the full sequence. The short answer: **before you build anything new, you need to run three diagnostics that will tell you whether ANY of your work has value, and you need to do an honest decomposition of the factor portfolio you just deployed.**

---

## Part 2 — Structural Flaws I Need to Name

These are things you didn't ask about that I think are more important than what you did ask about.

### Flaw 1: You are doing alpha research and system engineering simultaneously
Your HPO is running inside the production walk-forward simulator. This is a *known anti-pattern* in quant research. You should have:
- **Alpha research notebooks** — focus on IC, factor returns, signal quality. No execution model.
- **Strategy engineering** — proven signals get implemented in production with realistic execution.

You're tuning Sharpe (a downstream metric of model × execution × risk × universe) instead of tuning IC (a direct measurement of signal quality). This is why your 9 runs all "fail" — you're optimizing a high-noise, multi-component metric instead of a clean signal-quality metric.

### Flaw 2: You have accumulated too many components without ablation
NIS, regime model, opportunity score, BenignGate, ATR stops, position sizing, daily rescoring, agent architecture. Each of these *might* be helping or *might* be hurting. You haven't ablation-tested. **The right experiment is to build a stripped-down "vanilla" version with NONE of these components, measure its Sharpe, then add components one at a time and measure marginal contribution.** I'd bet money that 3–4 of these are net negative.

### Flaw 3: You don't have a benchmark
What does "Sharpe 0.80" mean? Compared to what?
- Buy and hold SPY: ~0.50 Sharpe historical, ~1.0 the last 5 years
- Buy and hold QQQ: ~0.7 historical, ~1.3 the last 5 years
- Buy and hold MTUM (momentum ETF): ~0.6 historical
- 60/40 SPY/AGG: ~0.5

If your "factor portfolio" generates 1.0 Sharpe but QQQ generates 1.3 over the same period... you have *negative* alpha vs. just buying QQQ. **You should always benchmark to the dumbest reasonable alternative.**

### Flaw 4: The complexity of your system isn't proportional to the size of your data or capital
You're a single engineer with $20k of paper capital using daily bars. The agent architecture (PM → RM → Trader async queue), the multi-tier news intelligence, the regime model + opportunity score + BenignGate triple-gating — this is the architecture of a $100M+ AUM systematic fund. At your scale, the engineering complexity is consuming time that should be going to alpha research.

**Recommendation:** This is fine if the engineering is the point (i.e., you're learning to build production trading systems). But don't expect the architecture to compensate for alpha that isn't there. **No agent architecture can make a 0.0-Sharpe signal generate positive returns.**

### Flaw 5: You're using yfinance for fundamentals and OHLCV
Both have known data quality issues:
- yfinance fundamentals are point-in-time-wrong (they apply restated numbers backwards)
- yfinance OHLCV has occasional bad ticks, splits applied inconsistently
- yfinance survivorship is poor

For a serious system, you'd want Sharadar (~$50/mo) for fundamentals and Polygon (which you have) for prices. But your current data **does** introduce noise that biases backtests upward (forward-revision bias in fundamentals → fundamentals "look better" historically than they were available at the time).

---

## Part 3 — What I'd Do Next (Specific, Prioritized Sequence)

I'm going to give you a sequenced plan with explicit "kill criteria" — when to stop and switch gears.

### Phase 0: Diagnostics (Week 1–2, ~10 hours)

**These are non-negotiable. Do these before building anything new.**

| # | Experiment | What it tells you | Kill criterion |
|---|---|---|---|
| 0.1 | **Compute IC of your LambdaRank model output vs. forward 20d return on each fold** | Whether your features have ANY signal | If avg IC < 0.01, your features are noise. Stop tuning. |
| 0.2 | **Factor-decompose your 1.335 Sharpe factor portfolio: regress on (SPY, MOM, QMJ) using AQR factor data** | Whether you have alpha or just factor exposure | If alpha intercept < 0.1 Sharpe equivalent, you have no alpha |
| 0.3 | **Run factor portfolio on 2007–2019 honest out-of-sample** | Whether it's overfit to 2019–2024 | If Sharpe < 0.6 on 2007–2019, the 1.335 was period-specific |
| 0.4 | **Build a dumb benchmark: top-20 12m-1m momentum, monthly rebalance, regime gate (SPY > 200d MA), equal weight, no stops** | Whether your fancy stuff beats simple | If your factor portfolio doesn't beat this by ≥ 0.2 Sharpe, your tier-weighting is overfitting |

### Phase 1: Honest Validation (Week 3–4, ~8 hours)

Based on Phase 0 results, you'll know one of three things:

**Case A: You have measurable alpha (IC ≥ 0.02, factor-adjusted Sharpe ≥ 0.3)**
- Productionalize the factor portfolio with MONTHLY rebalance (matches design), NO ATR stops
- Use regime gate (SPY > 200d MA AND VIX < 30) to go to cash in bear markets
- Accept that this is a factor portfolio, not an ML system
- Target live Sharpe: 0.4–0.7

**Case B: You have no measurable alpha, just factor exposure**
- Honest answer: stop building. Buy MTUM (momentum) + QUAL (quality) ETFs in 50/50 mix.
- Live Sharpe will be the same as your system, with zero engineering effort.
- The project becomes "engineering exercise" not "alpha generation."

**Case C: You have *something* but it's small (alpha ~0.1 Sharpe equivalent)**
- Continue Phase 2 with one focused experiment

### Phase 2: Focused Alpha Research (Month 2, ~20 hours)

Pick ONE of these. Don't do all three. Don't add a fourth.

**Option A: Post-Earnings Announcement Drift (PEAD)**
- The most-replicated anomaly in academic finance
- Universe is only "stocks reporting earnings yesterday" → 5–20 candidates per day
- Trade direction = sign of earnings surprise (positive surprise = long for 1–60 days)
- Realistic live Sharpe: 0.6–1.0 for retail
- Why it fits you: smaller universe, event-driven (regime-independent), persistent
- Data needed: earnings surprises (FMP has them, IBES is gold standard)

**Option B: Sector rotation**
- 11 SPDR sector ETFs (XLF, XLE, XLK, etc.) + bonds (AGG) + gold (GLD)
- Rank monthly by 6m momentum × dual_momentum (vs. cash)
- Hold top 3, equal weight, monthly rebalance
- Realistic live Sharpe: 0.4–0.6
- Why it fits you: ETFs are cheaper than stocks (lower spread, 0bp comms at Alpaca), $20k can hold meaningful positions
- This is the *boring* answer that just works

**Option C: Concentrated quality + momentum, 10–15 stocks**
- Top-decile momentum × top-quintile quality (ROE + earnings yield), filter to ~15 names
- Quarterly rebalance, equal weight
- Realistic live Sharpe: 0.4–0.7
- Why it fits you: very low turnover, real factor exposure

### Phase 3: Engineering for Production (Month 3, ~15 hours)

If Phase 2 produced something:
- Implement the simplest possible execution layer for the winning strategy
- Strip out NIS / opportunity score / agent architecture for this strategy — it doesn't need them
- Paper trade for 60 days minimum before any real capital
- Track *live IC*, not just Sharpe (Sharpe takes a year to be meaningful)

---

## Part 4 — Decision Framework on Each Open Issue

### Should you keep the agent architecture (PM → RM → Trader)?

**For learning: yes.** It's a great engineering exercise that maps to how real shops are organized.
**For alpha: no.** It adds zero alpha. A vanilla "compute scores, take top-N, rebalance" function would give identical results with 5% of the code.

**Recommendation:** Keep it because you've built it and it's clean. But don't expect it to improve returns.

### Should you re-enable BenignGate?

**Yes, for the factor portfolio.** The 0.267 → 0.0 Sharpe degradation when you removed it was probably noise (σ=0.20 per run). The factor portfolio is long-only equity and *needs* a regime gate to handle bear markets. Re-enable for the factor portfolio path.

### Should you fix the execution mismatch?

**Yes. The strategy was designed for monthly rebalance + no stops. Run it that way.** Your AgentSimulator with ATR stops is fighting the strategy. Pick the strategy's natural execution mode and use it. Don't bolt day-trader risk management onto a factor portfolio.

### Should you continue LambdaRank?

**No. You've proven it doesn't work at this universe/horizon/capital.** Stop. 9 runs is more than enough evidence. The opportunity cost of continuing is large.

### Should you try XGBoost with absolute return labels?

**Only after Phase 0 diagnostics.** If your IC is < 0.01 with LambdaRank features, switching to XGBoost binary won't help — the features aren't predictive. If IC is > 0.02, then yes, this could be a useful experiment. But run IC first.

### Should you continue intraday?

**Reframe or shut down.** Don't continue same-day 5-minute-bar trading at $20k with 15 bps cost. Either:
- Pivot to event-driven intraday (earnings gaps, FDA decisions, halts) where structural edge exists
- Or accept that intraday at $20k retail is mathematically unfavorable and stop

---

## Part 5 — The Honest Bottom Line

You're a sophisticated engineer doing serious work. The system is more rigorous than 95% of retail quant attempts. But:

1. **You're solving the hardest version of the problem.** Long-only + small capital + daily bars + R1000 + ML + intraday is the *worst* combination of constraints for finding alpha.

2. **Your most valuable insight is already in your prompt:** "We are uncertain whether our WF is truly honest." Trust that uncertainty. Run the IC diagnostic and factor decomposition. Stop tuning Sharpe until you've validated signal quality.

3. **The factor portfolio is probably your best path forward**, but only after honest validation. The deployed 1.335 Sharpe number is likely period-biased.

4. **At $20k, the value of this project is engineering and learning, not financial return.** A 0.7 live Sharpe on $20k = ~$2,000/year of expected profit before taxes. Your time has been worth more than that. Be at peace with this framing.

5. **The single most likely outcome of doing all this right is: a long-only factor portfolio with live Sharpe of 0.4–0.6, matching a 50/50 MTUM/QUAL ETF allocation, with much more learned along the way.** That's a win, but only if you frame it correctly.

---

## Part 6 — Things to Watch for in Other LLM Reviews

Since you said you're feeding this to multiple LLMs and synthesizing, here's what to triangulate on:

**If another LLM says:**
- "Try transformers / deep learning instead" → ignore, won't help at this data scale
- "Use ensemble methods" → ignore, masks the signal-quality problem
- "Try different features" → ignore until you've measured IC of existing features
- "The labeling problem can be solved with X" → partially right, but doesn't fix the long-only structural issue
- "You need more data" → maybe, but PIT R1000 + 15 years is harder than it sounds

**Things I'd want any LLM review to address that I want to call out:**
- Did they mention Information Coefficient as a diagnostic? If not, push them on it.
- Did they factor-decompose your 1.335 Sharpe? If not, push them on it.
- Did they push back on the LambdaRank vs. ranking-as-architecture distinction (it's the structure, not the algorithm)?
- Did they recommend you out-of-sample validate on 2007–2019? Critical.
- Did they tell you to stop adding components and start ablating? This is unpopular but right.

**Convergent recommendations across LLMs are probably right. Divergent ones are where the interesting decisions are.**

---

## Appendix: Specific References

If you want to validate the recommendations in this review:

- **López de Prado, *Advances in Financial Machine Learning* (2018):** Ch. 4 (labeling), Ch. 7 (cross-validation), Ch. 12 (CPCV), Ch. 14 (Deflated Sharpe). The single most important book for your project.
- **AQR factor data:** https://www.aqr.com/Insights/Datasets — pull MOM, QMJ, BAB, HML returns for factor decomposition
- **Bernard & Thomas (1989)** on PEAD — the foundational paper
- **Faber, "A Quantitative Approach to Tactical Asset Allocation" (2007)** — simplest sector rotation framework that works
- **Asness, Moskowitz, Pedersen, "Value and Momentum Everywhere" (2013)** — institutional validation of momentum + value
- **Marcos López de Prado, "The 7 Reasons Most Machine Learning Funds Fail"** — your situation is described almost exactly

---

*Final note: I've tried to be as direct as you asked. Some of this will be wrong — quant work is empirical and I'm working from your written summary, not the code. Where I've made probabilistic claims (e.g., "your alpha is probably 0.0"), those are testable hypotheses. The Phase 0 diagnostics will validate or falsify them in 10 hours of work. Run those first.*
