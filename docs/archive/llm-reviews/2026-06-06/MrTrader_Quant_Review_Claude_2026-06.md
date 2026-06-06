# MrTrader — Deep Quant Review (Claude / Opus)

**Reviewer stance:** systematic PM/researcher, skeptical-IC voice. No diplomacy.
**Date:** 2026-06 · **Format:** structured for cross-LLM synthesis. Each major claim carries a confidence tag `[H/M/L]` so you can tally agreement across reviews. A falsifiable-claims ledger is at the end (§10).

---

## 0. One-paragraph verdict (read this first)

You have built an **excellent research-and-execution machine and pointed it at the single most picked-over market on earth.** Your graveyard is not a story of research mistakes — it is the *expected outcome* for a solo retail operator running daily-bar, standard-feature models on large-cap US equities. Every "edge" you killed died for the same reason (it was beta, or it was noise on a thin sample), and your one survivor (PEAD) is a decayed anomaly's residual in the most-arbed slice of it, sitting at `p≈0.19` because **there genuinely isn't enough signal-bearing data to conclude anything.** The bottleneck is not your technique. It is your **opportunity set.** The highest-EV move is not "fix the models" — it is **point the same machine at less-efficient instruments (futures, options, crypto) and stop validating strategies in isolation; build a portfolio of modest uncorrelated premia.** Details, ranked, below. `[H]`

---

## 1. What you got right (so you don't throw out the good parts)

Stated up front because the redesign section says "don't tear it down," and you should know *why*.

- **The validation philosophy is more honest than most pro shops.** CPCV, DSR, `N_eff = folds not paths`, event-clustered bootstrap, embargo/purge, sacred holdout, auto-rollback on gate fail. You are scoring significance the way a careful desk does. Most retail (and plenty of institutional) quants would have shipped the analyst-drift strategy on its `CPCV +0.894, t=2.85` and never noticed the fold-skip artifact. You caught it. That instinct is your real asset. `[H]`
- **The agent separation (PM / RM / Trader) is sound and reusable.** Independent risk veto, entry-quality gate, full decision-audit with counterfactuals for blocked trades. This plumbing is asset-class-agnostic and you should keep it verbatim. `[H]`
- **You diagnose beta correctly and repeatedly.** "Long-only edge was market beta," "full-window CAPM alpha t=0.20." Recognizing that you can't beat a single-factor regression *is* the skill. `[H]`

Keep all of the above. The problems below are about **where you aim it**, not how it's built.

---

## 2. Method critique — your harness has four real wounds

### 2.1 The 52% fold-skip is worse than "optimistic Sharpes" `[H]`

You're treating it as a level bias you can dodge by leaning on *relative* comparisons. That's only partly safe:

- The skip is driven by a **rolling-window overlap guard** → correlated with event timing and volatility → **non-random**. The surviving sample is a biased estimator *and the bias direction differs across strategy families.* Event strategies (clustered events → more overlap → more skips) get a structurally different effective sample than continuous factor strategies. **Your relative rankings across families (PEAD vs. a factor ranker) are contaminated.** Relative comparisons *within* a family are more trustworthy.
- More fundamentally: skipping half the evaluations **destroys the combinatorial structure CPCV exists to create.** The whole point is many quasi-independent paths to estimate the *distribution* of Sharpe. Skip half and you're back to a biased k-fold with extra steps.
- **Likely cheap fix:** the overlap guard is almost certainly too conservative (double-counting embargo against purge, or using a window wider than your max-hold). Audit it before you do anything else — recovering those folds may change every number you have. Until then, **do not trust cross-family relative rankings.**

### 2.2 `N_eff = 8` is honest and brutal — your stats aren't lying, your sample is empty `[H]`

With ~8 effective folds, the standard error on a Sharpe estimate is ≈ **0.3–0.5**. A measured `0.546` is ~1 SE from zero. This is *exactly* consistent with your `p≈0.19`. The implication people miss:

> **At this sample size you literally cannot distinguish SR 0.5 from SR 0.0.** No amount of cleverness fixes this — only more *independent* signal-bearing observations do. This is the "statistical thinness" problem, and it caps what you can ever conclude in this pond.

### 2.3 Your `avg Sharpe ≥ 0.80` promotion gate is selecting FOR overfitting `[H]` — this one is counterintuitive and important

Real, robust, *deployable* daily-bar equity strategies for a retail operator live at **gross SR 0.3–0.7**, less net. Demanding 0.80 on an 8-fold sample means the *only* things that clear the bar are strategies that **got lucky on the biased sample.** You have built a filter that **rejects honest mediocre-but-real edges and passes noise that happened to look good.** High Sharpe thresholds on small samples are reverse-overfitting filters. Lower the bar to something a real edge could plausibly hit (0.4–0.5) *and* weight robustness/breadth (fold consistency, neutralized t-stat) over level.

### 2.4 You validate strategies in isolation, not as a portfolio `[H]` — this is the biggest one

The question is never "is PEAD's Sharpe > 0.8." It's **"does adding PEAD improve my *book's* risk-adjusted return after costs."** A 0.4-SR sleeve uncorrelated to everything you own is gold; a 0.8-SR sleeve correlated 0.9 with what you have is worthless. **You need a portfolio-level harness, not a strategy-level one.** This reframes everything (see §4).

### 2.5 Bonus: the `VIX>30 → SPY<200d` swap that "improved" PEAD `+0.546 → +0.661` is in-sample overfit `[M]`

You found a filter that improved the backtest. That is the textbook researcher-degrees-of-freedom trap, and DSR only partly catches it. **The honest PEAD number is the pre-filter one (~0.55), or a number from a filter chosen on economic grounds before seeing results.** Treat every "improvement found by trying filters" as suspect.

---

## 3. PEAD triage — paper-trading it is rational as engineering, irrational as alpha discovery

### Is running a `p≈0.19` signal in paper rational? `[H]`

**Yes for the wrong reason, no for the reason you're implying.**

- **Rational (engineering):** it exercises the full live pipeline, generates a real-fills record, and stress-tests your execution assumptions. Run it as a *systems test*. Costs only compute.
- **Irrational (alpha):** **paper-trading will not resolve the statistical question on any horizon you care about.** The math:
  - To get `t = SR·√years > 1.65` (one-sided p<0.05) at SR=0.4 → **~17 years.**
  - Even `t > 1` → **~6 years.**
  - PEAD fires on limited events/quarter, so effective sample grows *slower* than calendar time.
  - **A year of paper P&L tells you essentially nothing statistically** — it will be dominated by beta and luck. Do not upgrade conviction or add capital based on it.

**Correct stance:** PEAD is a **literature-informed prior, not a proven edge.** It's one of the most-replicated anomalies, so a Bayesian holds a positive posterior even at `p≈0.19`. Keep it as a **low-conviction satellite, sized small** (which you're doing). Don't expect paper to validate it. Don't fall in love.

### The three tests that would actually prove or kill it (cheap, do these) `[H]`

1. **Factor decomposition.** Regress PEAD returns on `mkt + size + value + momentum + sector`. If the **residual alpha t < 1** (your analyst-version full-window CAPM t=0.20 suggests it might be), you're paying complexity for exposure you can rent cheaper. Run this *specifically for PEAD.*
2. **The neutralization test — the single most valuable thing in this section.** You report long-only PEAD is an "up-trend drift harvester," **~87% of P&L in up-trends**, rescued by an SPY<200d filter. **That is a giant red flag that "PEAD" is substantially conditional beta / trend exposure, not idiosyncratic post-earnings drift.** A *real* PEAD effect drifts the earnings name *relative to its peers* regardless of market direction.
   - **Test:** long positive-surprise, short the sector ETF (or a matched no-surprise basket). **If the neutralized version dies, your "survivor" is mostly beta+trend and you've rediscovered your own graveyard lesson.** This is decisive and costs you a day.
3. **Execution-realism stress (PEAD-specific, brutal):** your **next-open fill on post-earnings gappers is the most-violated assumption in the entire system.** Gappers have wide opening spreads exactly when you're trying to cross them; Alpaca *paper* fills are idealized and will **understate real slippage precisely where it bites.** Your realized SR ~0.40 is **probably an over-estimate of live SR.** Re-run the backtest with a punitive opening-spread model (e.g., 30–50 bps on gappers, not 10) and see what's left. I'd bet a meaningful chunk.

> **Net on PEAD:** keep it, size it small, run the neutralization + gapper-slippage tests this month, and stop treating paper P&L as evidence. If neutralized-PEAD survives, you have a genuine (if thin) diversifier. If it dies, redeploy the slot.

---

## 4. The strategic reframe that matters most: a portfolio of premia, not a hunt for alpha

This is the idea I'd most want to survive your synthesis.

You've been searching for **one strategy that clears SR 0.8.** That's the wrong target for your constraints. The only free lunch in finance is **diversification**, and you've been throwing it away by validating in isolation.

**The math:** N uncorrelated sleeves, each at SR 0.4, equal-vol-weighted → **book SR = 0.4·√N.**

| # uncorrelated sleeves @ SR 0.4 | Book Sharpe |
|---|---|
| 1 (where you are) | 0.40 |
| 2 | 0.57 |
| 3 | 0.69 |
| **4** | **0.80** |

**Four honestly-uncorrelated 0.4-SR sleeves beat one fragile 0.8-SR strategy — robustly, and without needing any single sleeve to be heroic.** This is the whole game for a retail operator. It also dissolves your "PEAD is long-biased and crisis-vulnerable" problem: you don't fix PEAD, you *pair it* with a crisis-positive sleeve. Your architecture is already perfect for this — add a **portfolio allocator** above the PM that vol-weights sleeves by their marginal contribution to book risk, and **validate at the book level.** `[H]`

The rest of this review is about finding those uncorrelated sleeves — in **less-efficient ponds.**

---

## 5. Where the alpha actually is — 4 ranked return sources for *your* constraints

Selection logic: a retail operator will **never** out-compute Two Sigma on the large-cap daily cross-section. Your edge must come from (a) instruments big funds can't/won't play, (b) structural/behavioral premia that require holding uncomfortable positions, or (c) patience institutions can't afford. All four below clear that bar.

### Side-by-side

| Sleeve | Return source (the "why") | Data needed | Sim work | Capacity @ $100k | Corr to PEAD | Crisis behavior | EV rank |
|---|---|---|---|---|---|---|---|
| **A. Trend / managed futures (TSMOM)** | Behavioral under-/over-reaction + risk-transfer; the most-documented **crisis-alpha** source | Roll-adjusted futures (cheap) | New: **futures roll** handling | Fine on **micro** futures (MES/MNQ/M2K/micro rates/FX/metals) | **~0** | **Crisis-POSITIVE** (made money '08, '22) | **#1** |
| **B. Earnings-vol (pre-earnings IV-crush, defined-risk)** | **Variance risk premium** — implied vol systematically > realized; market overprices the earnings move | **Historical options IV** (paid; ORATS/Polygon options flatfiles) + your existing earnings calendar | New: **options pricing/Greeks** | Large; single-name vol is under-capacity for big funds | **Low–moderate** (same names, different axis) | Short-vol = crisis-fragile, but **defined-risk caps it** | **#2** |
| **C. Merger arbitrage** | **Insurance premium** for bearing deal-break risk; idiosyncratic, patient-capital-friendly | Deal announcements (SEC/news, scrape-able) | **Reuses your equity sim** (vanilla merger arb = long/short stock, no Greeks) | Fine | **~0** | Mostly uncorrelated; sharp losses only in stress-driven breaks | **#3** |
| **D. Crypto basis / funding harvest** | **Carry premium** retail can harvest that **regulated funds structurally can't** (the edge is the regulation) | Funding rates / basis / OHLCV (free from exchanges) | New: exchange/custody infra | Fine | **~0** | Basis trade ~neutral; counterparty/tail risk is real | **#4** |

### Why each, concretely

**A — Trend-following on micro futures. This is your answer to "crisis-diversification," full stop.** `[H]`
- **Signal:** classic time-series momentum (Moskowitz-Ooi-Pedersen): long markets above their N-month MA, short below, vol-targeted, diversified across ~10–30 liquid futures.
- **Why it fits you:** rules-based (your stated preference), capacity-fine on **micros** ($100k is actually *well-suited* to MES/MNQ/M2K/MYM + micro rates/FX/metals/energy — micros exist for exactly this account size), negative-to-zero correlation with equity beta, **positive convexity in tail events.** It *complements* a long-biased PEAD book directly.
- **Honest caveat:** standalone trend SR is modest (~0.5 long-run, lumpy, with a brutal flat 2011–2019). **Its value is diversification, not standalone Sharpe** — which is exactly what §4 says you need. Don't judge it standalone; judge its marginal contribution to book SR.
- **Validation:** your existing harness works once you add roll handling. Well-trodden; lots of open references. **Lowest conceptual risk of the four.**

**B — Earnings-vol harvesting. This is "options are where durable retail edge lives," and it reuses your earnings infrastructure.** `[H]`
- **Signal:** pre-earnings implied vol is systematically too high (the market overprices the move). **Sell defined-risk premium (iron condors / short strangles with wings) into the announcement, close post-event to capture the vol-crush.** You already have the earnings calendar, surprise history, and the ability to get IV from Polygon.
- **Why it's the killer idea:** it's a **different return source (vol premium, not drift)** on the *same events* you're already modeling — so it's a natural diversifier, and you can even run it alongside directional PEAD on the same name. Single-name/earnings vol is *under-capacity* for big funds (too small to matter to them), which is your structural opening.
- **Cost:** needs an options backtest stack (Greeks) **and historical IV data** — the latter is the real expense (ORATS or Polygon options flatfiles; historical options data is the pricey part). **This is the "is it worth building a new sim" decision, and my answer is yes** — it's the highest-ceiling durable retail edge here.
- **Caveat:** short vol is crisis-fragile; **always defined-risk (wings), size down in high-VIX using the regime classifier you already have,** and you convert a blow-up risk into a bounded-loss risk.

**C — Merger arbitrage. The cleanest pure diversifier, and possibly your lowest-lift new sleeve.** `[M]`
- **Signal:** on announced deals, long the target / short the acquirer (or just long cash-deal targets near the spread), earn the spread as compensation for deal-break risk. Rules-based screening on announced-deal terms.
- **Why it fits:** **uncorrelated insurance-like premium**, $100k-friendly, **and it largely fits your *existing* equity-bars sim** (vanilla merger arb needs no Greeks). It directly delivers crisis-diversification because deal spreads are mostly idiosyncratic.
- **Caveat:** the tail is "deals break in a stress wave" (correlated losses in a crisis) — so cap concentration and avoid the highest-regulatory-risk deals. Lower ceiling than A/B but very clean.

**D — Crypto basis/funding. The high-optionality bet that plays to your structural advantage.** `[M]`
- **Signal:** when perp funding is positive (longs pay shorts), short the perp + long spot (cash-and-carry) to harvest funding/basis — a real, often-rich, ~market-neutral carry. Plus a less-efficient cross-section than equities for momentum/MR overlays.
- **Why it's genuinely outside-the-box:** crypto is the one liquid market where **the structural inefficiency persists because the big-fund/smart-money presence is thin and regulatory frictions keep US institutions out — so a retail automated operator can do things a regulated fund can't.** That regulatory asymmetry *is* the edge.
- **Caveat:** new infra (exchange accounts, custody, 24/7 ops), counterparty/exchange risk, regulatory uncertainty. Highest operational lift and tail risk. **Pursue only if you have appetite for new plumbing** — but the EV/optionality is real.

### And the thing to STOP doing `[H]`
**Stop hunting for cross-sectional alpha in large-cap US equity daily bars.** Not because you did it wrong (you mostly didn't) — because the pond is fished clean by players with more data, lower costs, and leverage. **Every marginal hour there is wasted.** Redeploy it to A/B above.

---

## 6. Is cross-sectional ML dead? — autopsy

**For large-cap US equities on daily bars with standard features: yes, dead for you.** `[H]` Not "ML is dead" — *that pond* is dead. Signal-to-noise in the daily large-cap cross-section is brutal (good factors run ~1–3% monthly IC, decaying), your features (RSI/MACD/momentum) are exactly what everyone uses, and you lack the alt-data moat + cost/leverage that make XS-ML pay for the big funds.

**Did you do it wrong? A few real things — but none resurrect the pond:** `[M]`
- **Label bug-class (most important):** if you trained on **raw forward returns**, the model *learned beta* — which in a bull sample looks like alpha, exactly matching "long-only edge was beta." **Cross-sectional labels must be residualized** against beta/sector/size so the model learns *idiosyncratic rank.* If you didn't, that's a genuine error — but fix it in a *new* pond, not this one.
- **LambdaRank/top-K mismatch (you flagged it, you're right):** LambdaRank optimizes full-list NDCG; you trade only the tails. The model burns capacity ordering the middle. Use a **tail-weighted loss** or just regress/classify the extremes. Legitimate fix; won't revive a no-signal pond.
- **Regime-conditioning is a *trap* at your data scale:** you have a regime classifier but use it only for sizing — and that's correct restraint. **At N_eff=8, every conditioning dimension is an overfitting vector.** Do NOT fit regime-conditional models; you don't have the data. Less is more.
- **Ensembles won't help:** ensembling noise yields smoother noise.

**Where XS-ML *is* alive for you:** ponds with more cross-sectional inefficiency — **crypto** (sleeve D), single-name **vol surfaces** (sleeve B), international/small-cap (you can't trade), or higher-frequency microstructure (you don't have the data/infra). **Move the ML to where there's signal; don't apply more ML where there isn't.**

**Time-series vs. cross-sectional:** you may have thrown out TS with XS. The large-cap *cross-section* is efficient; the *time-series* (when to be in vs. out — i.e., trend on the index/asset classes) is the more robust retail edge. That's literally sleeve A. **Your dead family was cross-sectional selection; your live opportunity is time-series timing.**

---

## 7. Missing data — ranked by how much it moves your odds

| Priority | Data | Cost | Unlocks | Verdict |
|---|---|---|---|---|
| **1** | **Historical options IV surfaces** (ORATS / Polygon options flatfiles / CBOE) | Paid (the real spend) | Sleeve B (earnings-vol / VRP) — highest-ceiling retail edge, reuses your earnings infra | **Buy it IF you commit to options** `[H]` |
| **2** | **Roll-adjusted continuous futures** | Cheap/available | Sleeve A (trend) — your crisis-diversifier | **Get it** `[H]` |
| **3** | **Crypto funding/basis/OHLCV** | Free from exchanges | Sleeve D | Get it if pursuing D `[M]` |
| **—** | More FMP equity endpoints (estimate revisions, guidance, price targets) | Cheap | More cross-sectional equity factors — **same dead pond** | **Skip** `[H]` |
| **—** | Glamorous alt-data (satellite, card-spend, options *flow*) | Expensive | Edge decays fast, requires scale you don't have | **Money pit for a solo operator — skip** `[H]` |

**Single highest-value purchase: options IV history**, because it's the key to the best new pond *and* it leverages the earnings machinery you've already built. **Do not chase alt-data** — it's where retail money goes to die.

---

## 8. Redesign? — No. Re-aim.

**Don't tear down the system. The plumbing is good; the alpha is the problem.** `[H]`

| Keep verbatim | Fix | Add |
|---|---|---|
| PM/RM/Trader agent separation | The **fold-skip overlap guard** (likely cheap; recover the 48%) | **A portfolio allocator above the PM** that vol-weights *sleeves* by marginal book-risk contribution |
| Decision-audit + counterfactuals | **Lower the SR gate** to ~0.4–0.5 + weight robustness over level | **Book-level validation** (stop validating sleeves in isolation) |
| Regime classifier (for sizing only) | Drop in-sample filter-tuning (the SPY<200d "improvement") | **Futures roll** handling in the sim (sleeve A); **Greeks** if pursuing B |
| Validation *philosophy* | | The **neutralization + gapper-slippage tests** for PEAD (§3) |

**The reframe restated:** stop hunting one 0.8-SR strategy; assemble **four uncorrelated 0.4-SR sleeves** (§4–§5). The architecture change is small; the mindset change is everything.

---

## 9. Concrete 90-day sequence (ranked by EV/effort)

1. **Week 1 — Audit the overlap guard.** Recovering the 48% skipped folds may change every number you hold. Cheapest highest-leverage action. `[H]`
2. **Week 1–2 — Run the two PEAD kill-tests:** (a) sector/market-neutralized PEAD; (b) backtest with 30–50 bps gapper slippage. If both survive, PEAD is a real thin diversifier; if not, free the slot. `[H]`
3. **Week 2–6 — Stand up sleeve A (trend on micro futures).** Add roll handling to the sim, build vanilla TSMOM, validate *as a book addition* alongside PEAD. This is your crisis-diversifier and lowest conceptual risk. `[H]`
4. **Week 4–8 — Build the portfolio allocator + book-level harness.** Vol-weight sleeves by marginal risk; this is the architectural unlock. `[H]`
5. **Month 2–3 — Decision gate on sleeve B (earnings-vol).** Price out historical IV data; if you commit, build the options/Greeks sim. Highest ceiling, real cost. `[M]`
6. **Parallel, low-priority — prototype sleeve C (merger arb)** in your *existing* equity sim (no new stack). Cheap diversification. `[M]`
7. **Optional — sleeve D (crypto)** only if you want the infra adventure. `[L]`
8. **Ongoing — stop all large-cap daily XS-ML research.** Redeploy the hours. `[H]`

---

## 10. Falsifiable-claims ledger (for your cross-LLM tally)

Tally how many reviewers agree/disagree with each. My confidence in brackets.

1. The graveyard is the *expected* outcome of fishing a fished-out pond, not a research-process failure. `[H]`
2. The 52% fold-skip contaminates **cross-family** relative rankings, not just absolute levels. `[H]`
3. `N_eff=8` makes SR 0.5 statistically indistinguishable from 0.0 — the data, not the method, is the wall. `[H]`
4. The `SR ≥ 0.80` promotion gate selects *for* overfitting on small samples. `[H]`
5. Validating strategies in isolation (vs. book-level) is the single biggest harness flaw. `[H]`
6. Paper-trading PEAD cannot resolve its `p≈0.19` in <~6–17 years; it's a systems test, not alpha validation. `[H]`
7. PEAD-as-implemented is substantially conditional beta/trend (87% up-trend P&L is the tell); the neutralization test is decisive. `[H]`
8. PEAD's next-open gapper fills make realized SR ~0.40 an over-estimate of live SR. `[M]`
9. The right target is 4 uncorrelated 0.4-SR sleeves (book SR→0.8), not one 0.8-SR strategy. `[H]`
10. Trend-following on micro futures is the highest-EV addition (crisis-diversifier, capacity-fine, rules-based, ~0 corr to PEAD). `[H]`
11. Earnings-vol (VRP / IV-crush) is the highest-*ceiling* addition and reuses the earnings infra. `[H]`
12. Cross-sectional ML is dead **in large-cap daily bars for retail**, alive in crypto / vol-surfaces / HF microstructure. `[H]`
13. A likely real XS-ML bug: labels on raw (not residualized) forward returns → the model learned beta. `[M]`
14. Historical options IV is the single highest-value data purchase; alt-data (satellite/card/flow) is a solo-operator money pit. `[H]`
15. Don't redesign the system — re-aim it; the plumbing is good, the opportunity set is the problem. `[H]`

---
*Reviewer: Claude (Opus). Prepared for multi-LLM synthesis — sections and the §10 ledger are structured for line-by-line cross-comparison.*
