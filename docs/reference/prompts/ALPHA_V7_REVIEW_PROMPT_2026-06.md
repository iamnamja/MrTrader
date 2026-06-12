# Alpha-v7 External Quant Review — Prompt & Briefing Kit (2026-06-12)

> **How to use this:** paste everything from "BEGIN PROMPT" down into each LLM (run the same prompt across 4–5 frontier models independently), and attach the files in the **Briefing Kit** list. Each model should answer on its own; the operator synthesizes the panel afterward. The whole point is *disconfirming, outside* views — we are explicitly asking these models to tell us where we are wrong.

---

## BEGIN PROMPT

### Your role
You are a **world-class quantitative researcher, quant developer, and systematic trader** — the kind who has run real buy-side capital, built and killed hundreds of strategies, and has strong, specific opinions about backtest validity and gate design. Be **skeptical, concrete, and willing to contradict us**. We do not want validation or encouragement — we want you to find our mistakes, our blind spots, and the edges we may have wrongly buried. If our conclusions are right, say so briefly and move on; spend your effort where you'd push back.

### The situation (read this honestly)
This is a **solo-operator systematic trading system** running a **$100k PAPER book** (Alpaca paper account). The goal is simply to **find durable alpha and trade it**. The codebase is mature: a full walk-forward / CPCV validation harness, a research registry with **pre-registration** (immutable one-shot hypotheses), simulators, gates, and a live execution layer.

We just completed a multi-week research program ("Alpha-v6") that was itself **the output of a prior 5-LLM panel review**. We executed the *entire* plan with pre-registered, one-shot, immutable hypotheses (so we cannot have fooled ourselves with post-hoc selection). **Every single candidate edge was rejected.** That is the uncomfortable fact that prompts this review.

**What is live right now:** a single **TSMOM trend-following ETF sleeve** — 10 liquid multi-asset ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP), long-flat, inverse-vol sized, weekly rebalance, ~25% of NAV — plus cash. Standalone **Sharpe ≈ +0.71 over 19 years** (the one thing that survives every honest test). Nothing else trades.

**The data we have:** *free* feeds — yfinance (daily + 5-min equity/ETF bars), FRED (macro), FMP Starter $29/mo (earnings calendar, fundamentals, analyst grades) — plus a **now-frozen 4-year options store** (Polygon/"Massive" OPRA EOD bars, 2022-06→2026-06, ~113M bars / 733 underlyings, with computed IV/greeks; the options subscription is being cancelled because options proved not to be an edge). We have **no** paid estimate-revision (I/B/E/S) data, no intraday tick/microstructure data, no alt-data, no L2/order-flow.

### What we have already tested and KILLED (the ledger — assume each was done with real rigor)
- **Cross-sectional swing ML ranker** (LambdaRank/XGBoost on price+fundamental features): per-fold CPCV +0.22, t=0.17 → **dead** (noise; long-only = market beta; the F2/Aug-2024 VIX spike destroys long-biased swing).
- **Intraday 5-min ML** (meta-model): honest CPCV **−2.80**, t=−6.85 → **dead** (cost-drag; an earlier +5.14 was a low-deployment / memorization artifact).
- **PEAD (post-earnings drift)** — the one-time "best" result. Long-only +0.546 (t=2.26) but **CAPM-hedged Sharpe −0.37** (it's small market beta riding the bull). Re-adjudicated at the **event level** (21k-event panel, two-way Cameron-Gelbach-Miller clustered, SPY-hedged): **10d mean −8.3bp, t=−0.77, p=0.78 → DEMOTED**. Now off the live book.
- **Options as a trading edge:** earnings IV-crush = KILL (single-name, cost-killed); index short-vol = real VRP (PF 2.24/1.75) but Sharpe-flat / a *risk premium*, not alpha; implied-move PEAD filter = fragile/overfit.
- **Options as a cross-sectional EQUITY signal (H4a–e, pre-registered, full 4y/208-week dollar-neutral decile L/S at equity cost):** CPIV, 25Δ put-skew, put-heavy O/S volume, term-structure slope, IV/RV — **all 5 KILLED**; four were *significantly negative* (the academic signs don't survive 2022–26 in this universe), one was noise.
- **Implied-move reaction-ratio as a continuous PEAD feature (H2):** t=−1.21 → not significant → OPT-5 parked.
- **Trend broadening (P5, pre-registered):** broaden the +0.71 sleeve with more legs (16 ETFs) + long-short + a book-vol overlay, judged on 19y. Broadened **Sharpe 0.30 / maxDD −24.7% vs the simple sleeve 0.72 / −13.9%** → **PARK** (long-short helped in crises but bled in the 19y bull; complexity didn't earn it).
- **Short-interest factor** (Boehmer/Asquith, FINRA data): −1.21, t=−3.53 → the anomaly *reversed* in the meme era → dead.
- **Analyst up/down-grade drift, short-term reversal, cross-asset carry, fundamental quality-short, small/mid-cap PEAD:** all tested honestly, all dead or cost-killed.

**Net:** the only thing that has *ever* survived an honest test is broad trend-following. Cross-sectional equity ML, event-driven (PEAD family), and options (execution *and* signal) are all exhausted on the data we have.

---

### The three questions we need you to answer

**Q1 — Are our model-acceptance parameters right, or do we need to change them?**
Our promotion gate (see `PIPELINE_ARCHITECTURE.md` §7 and `retrain_config.py`) is a **significance-first two-tier** gate:
- **PAPER tier:** path-Sharpe **t-stat ≥ 2.0** (N_eff = n_folds ≈ 8, *not* n_paths), **%positive ≥ 0.75**, **P5 (5th-pctile fold Sharpe) ≥ 0.0**, **mean Sharpe ≥ 0.35**, plus profit-factor / Calmar / worst-regime backstops.
- **CAPITAL tier:** the above + mean ≥ 0.50, n_folds ≥ 10, t ≥ 2.5 (or documented paper confirmation).
- **DSR** (Deflated Sharpe) is **report-only** — it saturates to p=1.0 above Sharpe ~2 regardless of trial count, so it provides no discrimination (Known Limitation #1).
- A **two-track** acceptance: Track A judges standalone alpha; **Track B** judges a candidate purely on its **book-level diversification contribution** (a diversifier needn't clear the standalone floor).
- We ran a **gate-calibration study** (positive + negative controls through the production gate) and found the gate is, on **≤4 years / ≈8 folds of data**, a **Type-II / false-negative machine**: because **t ≈ Sharpe·√years**, a *genuinely real* Sharpe-0.5–0.7 edge fails t≥2 more often than it passes; meanwhile 3/5 true-zero-Sharpe nulls cleared t≥2.0 by chance. So "just lower the t-bar" admits noise, and our own confirmed-real risk premia were getting killed.

  → **Is our acceptance framework sound? Are we too strict (killing real edges) or too loose (chasing noise) — and where exactly? What specific thresholds, statistics, or framework changes would you make?** (e.g., minimum data length before a verdict; SR/Sharpe vs alpha-t vs IR; how to size the multiple-testing penalty when the *real* trial count is ~dozens; whether the two-track Track-B book-delta gate is the right idea; how to gate a *diversifier* differently from an *alpha*; deflated-Sharpe alternatives; Bayesian / shrinkage approaches; the right way to think about "significant *enough* to risk paper capital" vs "real capital").

**Q2 — Given that everything we've tried is exhausted, what is next?**
Where does durable alpha realistically come from for an operation like this? Be concrete and **prioritized**. Specifically address: is the binding constraint **data** (and if so, exactly which dataset is worth paying for, and why — e.g., I/B/E/S estimate revisions, intraday/microstructure, options NBBO, alt-data, short-borrow, fundamentals depth), **instruments** (futures, FX, crypto, single-stock options, vol), **technique** (we've done trend, XS-ML, event-driven, options — what haven't we), **timeframe/capacity** (we're small — does that open or close doors), or **the premise itself** (is the honest answer "the liquid US-equity opportunity set is efficient for a retail quant; run the trend book and stop hunting"?). If you would do something fundamentally different with this exact setup, tell us what and why.

**Q3 — Could our walk-forward / backtesting process itself have made us wrongly DISMISS a real edge?**
This is the one that worries us most. Review **how we actually run validation** (see `PIPELINE_ARCHITECTURE.md` §1–12 + the source files). Known issues we're already aware of (be more worried about the ones we're *not*):
- **N_eff overstatement** (KL-4): %positive treats 15 correlated CPCV paths as independent; effective independent folds ≈ 6 — does this make us *over*-reject (demanding significance from ~6 noisy folds)?
- **Regime-label look-ahead** (KL-2): some VIX-quartile regime labels were computed over the full window (`pd.qcut`).
- **Calmar no-DD sentinel** (KL-3) and **DSR saturation** (KL-1).
- **Survivorship**: free yfinance has *no delisted-name bars*, which flatters long-only and breaks any short/event study that needs the losers — we patch with PIT index membership + (for some studies) Polygon delisted-inclusive data, but coverage is uneven.
- **Cost modeling**: did we cost-kill something that's actually tradeable at our (tiny) size? Are our slippage/spread assumptions too punitive for $100k?
- **The deeper question:** is there a *systematic* reason our harness produces nulls — e.g., we validate cross-sectional signals long-only inside a 5-position book, or we hedge out beta so aggressively that we also hedge out real exposure premia, or our purge/embargo is so conservative it starves the folds, or we demand standalone significance from things that are only ever edges *in combination*? **Walk through our methodology and tell us what a flawed process could have buried, and what specific re-test you'd run to recover it.**

---

### Briefing Kit — files to attach (provide all of these)
Paste/attach these so your answer is grounded in what we actually built, not assumptions:

1. **`docs/living/PROJECT_STATE.md`** — one-screen "what's happening now" (live book, latest verdicts). *Start here.*
2. **`docs/living/PIPELINE_ARCHITECTURE.md`** — **the most important file.** The SSOT for the WF/CPCV harness: component map, the two simulators, fold construction, the full **Gate Inventory (§7)**, and the **Known Limitations (§12)**. Central to **Q1 and Q3**.
3. **`docs/living/ML_EXPERIMENT_LOG.md`** — the append-only journal: every strategy tried, its honest numbers, and why it was kept/killed. The evidentiary basis for **Q2**.
4. **`docs/living/DECISIONS.md`** — why things are the way they are (the architectural + strategic decision log, incl. all the recent KILL/DEMOTE/PARK verdicts).
5. **`docs/living/MASTER_BACKLOG.md`** — the roadmap + the Alpha-v6 phase table (P0–P6) with per-phase status.
6. **`docs/living/MODEL_STATUS.md`** — what's live now and its gate results.
7. **`app/ml/retrain_config.py`** — the actual gate thresholds + feature flags (**Q1**).
8. **`scripts/walkforward/gates.py`** + **`scripts/walkforward/cpcv.py`** — the gate + CPCV implementation (**Q1, Q3**).
9. **`app/backtesting/agent_simulator.py`** (swing/event sim) — how fills, costs, stops, and equity are simulated (**Q3**).
10. **`app/strategy/tsmom.py`** — the one surviving sleeve (so you can judge whether the *survivor* is robust or lucky).
11. **`docs/reference/DATA_PROVIDERS.md`** — exactly what data we have and don't (grounds **Q2**'s data question).
12. *(Optional, for the full backstory)* **`docs/reference/NEXT_PHASE_BLUEPRINT_2026-06.md`** — the prior 5-LLM panel's plan that we just executed-and-falsified.

### The output we want
- Answer **Q1, Q2, Q3** explicitly and in order. Lead each with your single highest-conviction point.
- Be **specific and actionable**: name thresholds, datasets (with rough cost), instruments, statistical methods, and concrete re-tests — not generalities.
- **Prioritize.** If you give us 10 ideas, rank them by expected value for a solo operator with a $100k paper book and free-to-cheap data.
- **Disconfirm us.** Where do you think we've fooled ourselves, over-engineered, or quit too early? Where might the "everything is dead" conclusion itself be an artifact?
- A blunt **bottom line**: if you were handed this exact system and mandate tomorrow, what are the first three things you'd do?

## END PROMPT

---

### Operator notes (not part of the prompt)
- Run this across **4–5 models** (e.g., GPT-5/o-series, Gemini, Claude/Opus, Grok, DeepSeek), each cold, same files. Save raw responses under `docs/reference/prompts/20260612_Alpha_v7_Review/responses/` (mirrors the prior review folders).
- After collecting, the standard next step is a code-grounded synthesis (a single deep-dive pass that scores each suggestion against what's actually in the repo, separating "real gap" from "already tried / already exists") → a new blueprint. The last two reviews followed exactly this loop.
- The honest framing to hold while reading the responses: we have a **calibrated ruler** now and a **proven survivor** (trend). The two highest-value outcomes of this panel are (a) a credible *new* place to look that we haven't, or (b) a credible argument that our **process** wrongly buried something — anything else is confirmation that running the trend book is the right answer.
