# MrTrader — Synthesized Build Directive (Cross-LLM Review → Action Plan)

**For:** MrTrader engineering (Claude Code / VS Code session)
**From:** Synthesis reviewer (world-class quant lens), reconciling 5 external reviews
**Date:** 2026-06-02
**Inputs synthesized:** Claude, Gemini, Grok, DeepSeek, Copilot reviews of `EXTERNAL_ALPHA_REVIEW_BRIEF.md`

---

## 0. How to use this document

This is a **build directive**, not an essay. It reconciles five independent LLM reviews into one prioritized plan, separates **what to start now** from **what to design/spike first**, and — critically — names the ideas that showed up in multiple reviews that you must **NOT** build because they are dangerous or wrong.

Read in order: §1 (verdict) → §2 (what each review was worth) → §3 (consensus truths) → **§4 (DO NOT BUILD — read before touching anything)** → §5 (start now) → §6 (investigate first) → §7 (north star) → §8 (sequenced roadmap) → §9 (reusable test specs) → §10 (decisions that need the human, not the agent).

Every workstream below carries: **Goal · Method · Acceptance criterion · Backed by** (which reviews). Treat all projected Sharpe numbers in the source reviews as **hypotheses to test, not promises** — no reviewer can guarantee an out-of-sample Sharpe, and several tried to.

---

## 1. Synthesized verdict (the honest TL;DR)

The five reviews **agree on the diagnosis and split on the prescription.** The agreement is high-confidence and you should act on it immediately. The split is almost entirely about **one decision (options vs short interest as the next dataset)** and **one dangerous temptation (selling earnings volatility)**.

Three sentences:

1. **Your validation is sound and your low numbers are real — every serious reviewer agrees. The fix for "everything is dead" is not more signal hunting; it is structural: go dollar-neutral, fix breadth, residualize, and stop trading overlays you never validated.**
2. **Your one live edge (PEAD) rests on a hand-tuned crisis filter and an optimistic cost assumption — de-risk it *this week* (leave-one-crisis-out, cost-sensitivity, SUE) before you trust it with anything.**
3. **Three of five reviews push selling options volatility around earnings as "near-free money." It is the single most dangerous idea in the entire ensemble. Do not build it. Keep only the read-only version (IV as a PEAD signal-quality filter).**

---

## 2. What each review was actually worth (brutally honest scorecard)

| Review | Grade | Take | Discard |
|---|---|---|---|
| **Gemini** | **A−** | Strongest technical contributor. Newey-West path t-stat, eigenvalue effective-trials for DSR, **residualize features AND labels** (the correct F2 fix), and the best articulation of the PM/RM mismatch ("RM = safety net, not active PM; if you can't backtest a rule, you're banned from running it live"). Factor-neutral PEAD (long stock / short sector ETF) is implementable today. | Strategy 2 (short iron condor/straddle into earnings) — same steamroller trade as DeepSeek, dressed up. "Easily clears t≥2.5" overstates. |
| **Claude (mine)** | **A−** | The breadth/transfer-coefficient diagnosis (Fundamental Law: 5-position long-only mathematically discards ~85% of any cross-sectional edge — you can't tell "no signal" from "good signal, near-zero transfer"). The thesis-short vs cross-sectional-short conflation. The future-dated holdout catch. Over-purge catch. Live-overlay-= unvalidated-strategy. Short-interest-first data call. | Too categorically anti-options (a cheap read-only IV feed is worth it). Light on validation-math specifics and shipped no code scaffolds — Gemini and Copilot did better there. |
| **DeepSeek** | **C+** | Two real insights buried in bravado: the **event-breadth N_eff problem** (independent earnings cycles ≈ number of years, so effective N ≈ 4–5, not 8 — PEAD's t is more optimistic than even the team thinks), and "simplify the over-engineered architecture." VIX term-structure overlay idea is decent. | **"Invert the intraday signal → +2.8 Sharpe" is mathematically illiterate** (see §4.2). The options-selling pitch is reckless ("spend your *entire* budget"). "I guarantee t>3.0" is a charlatan tell. Internally contradictory (delete correlation risk *and* run a long-short book). |
| **Grok** | **C+** | Sober, correct, aligned with consensus: validation rigorous, failures real, realistic Sharpe 0.6–1.2, good triage. Safe to trust as a sanity check. | Surface-level. Generic redesign ("risk parity," "Kelly," "L/S skeleton") with no specs. Misses the deeper event-N_eff issue. Adds little the others don't say better. |
| **Copilot** | **D (diagnosis) / B (scaffolds)** | **Operated on assumptions because it claims it couldn't read the brief — and its assumptions are mostly wrong** (it asserts you have no CPCV, no per-fold retraining, a contaminated holdout, $50M AUM, and a "live/backtest Sharpe gap" — all contradicted by the brief; you fixed exactly these, it's $100k paper, and there is no live track record yet to show a gap). Its "single most damning signal" is **fabricated**. BUT it shipped genuinely useful, reusable **test/ablation scaffolds** (cost-sensitivity table, overlay ablation, SUE quintile test, SHAP feature-stability, gap decomposition). | **Reject its N_eff method outright** — Quenouille AR(1) on daily returns gives N_eff≈900–1200 and t-stats of 25–40, which would *re-inflate* everything you correctly deflated. Its data prices ($25–100k feeds) are scaled for a $50M fund, not you. |

**Meta-point for the agent:** weight Gemini and the breadth/structural arguments heavily; mine Copilot for code only; treat DeepSeek/Grok as idea-sources to be filtered, not followed.

---

## 3. The consensus truths (high-confidence — act on these)

Where 4–5 of 5 agree, the signal is strong:

1. **The failures are real, not measurement artifacts.** Believe the low numbers. (All except Copilot, which didn't read the brief.)
2. **The `t ≥ 2.5` / `N_eff = 6–8` CAPITAL gate is near-unreachable for a real-but-modest edge** (it implicitly demands Sharpe ≈ 0.9–1.0). Fix by **raising statistical power, not lowering the bar.** (Claude, Gemini, Grok, DeepSeek, Copilot.)
3. **`N_eff = n_folds` is the right *instinct* but still too generous for event strategies** — overlapping 40-day earnings holds mean the independent-bet count ≈ number of seasons/years. PEAD's t=2.26 is optimistic. (Claude, DeepSeek explicitly; Gemini via Newey-West.)
4. **You are over-purging** (85-day purge vs ~62-day test windows), starving power for no leakage benefit. (Claude, Gemini.)
5. **Your live system trades an unvalidated strategy** — clean PEAD got t=2.26; the live thing stacks regime sizing, news sizing, opportunity gates, macro blocks, and conviction sizing on top, none independently validated. (All five.)
6. **5 bps is too optimistic, especially for earnings-window names** where spreads blow out. (Claude, Gemini, Copilot.)
7. **The fix for long-only ML exhaustion (F2) is dollar-/factor-neutral, not abandonment.** (Claude, Gemini, Grok, DeepSeek.)
8. **Use SUE (standardized unexpected earnings), not raw % surprise.** (Claude, Copilot, Gemini.)
9. **Triage agreement:** intraday 5-min = dead (5/5); QualityShort = dead as a standalone short (5/5); small/mid-cap PEAD = correctly rejected (4/5); insider clusters = shelve (5/5).
10. **The architecture is over-engineered relative to a single fragile edge — simplify and align sim-to-live.** (5/5.)

---

## 4. DO NOT BUILD — rejected ideas (read this before writing any code)

A naive synthesis would average the reviews and carry these forward. **Do not.** Each is either dangerous or wrong.

### 4.1 ❌ Selling options volatility around earnings ("IV crush is free money")
Pushed by **DeepSeek** (sell OTM put verticals pre-earnings) and **Gemini** (short iron condor/straddle 15 min before close, "200–500 bps/trade, bypasses cost drag"). **Reject outright.**
- The **implied move is the market's price for the expected move.** The premium compensates the seller for *exactly* the tail they're short. There is no free lunch in "IV crush" — short-vol-into-earnings is a textbook **negative-skew** trade: win ~80% of the time, give it all back plus more on the gap that blows through your strike.
- You have **no options data, no options backtest harness, no Greeks risk management, no options execution.** Running a short-gamma book blind is how small accounts get vaporized overnight.
- Both reviews **misstate the costs** — single-name options spreads around earnings are wide (often 5–15% of premium); "bypasses cost drag" is false.
- It is the **opposite tail profile** of everything you've validated. Not diversification — a new, larger left tail.
- **What to keep instead:** the *read-only* version — use options-implied move to judge whether a +X% surprise is genuinely a surprise or already priced (Gemini's good point: "+10% surprise when 15% was priced in = an underperformance"). That needs a cheap IV feed and **zero options trading.** See §6.4.

### 4.2 ❌ "Invert the dead intraday signal to get +2.8 Sharpe"
DeepSeek's claim. **Mathematically illiterate.** The −2.80 is **net of costs**; gross PF was **0.94** (already below break-even *before* costs). Inverting a roughly-break-even gross signal and paying transaction costs **on the other side too** yields *another negative*, not +2.8. You cannot flip the sign of a cost-dominated Sharpe by flipping the signal. Do not spend a cycle "inverting" anything. The intraday horizon is dead for your infrastructure (no TAQ/L2/colocation) — leave it dead.

### 4.3 ❌ Copilot's `N_eff` (Quenouille AR(1) on daily returns)
It produces N_eff ≈ 900–1200 and t-stats of 25–40 for a Sharpe ~1 (its Test A5 literally prints "PEAD t-stat 37.9, PASS"). This would **undo the entire honesty campaign** — it's the false-confidence disease you spent 13 audit rounds curing. Keep `N_eff = n_folds` as a conservative floor; refine with Newey-West / block-bootstrap (§5.3), **never** with daily-AR(1).

### 4.4 ❌ Institutional-priced data acquisition right now
Copilot's $25–100k Compustat PIT / S3 / Ortex recommendations are scaled for a $50M fund. At $100k paper, start with **cheap/free** sources (FINRA short interest, a low-cost IV feed). See §6.3–§6.4 and §10.

### 4.5 ❌ DeepSeek's 5-name reversal book and any "guaranteed t-stat"
Short-term reversal is a real *category* worth testing (§6.7) — but a **5-position** reversal book contradicts the entire breadth argument, and "I guarantee t>3.0" is unfounded. Test reversal properly (high breadth, honest costs) or not at all. Treat every guaranteed/projected Sharpe across all reviews as a hypothesis.

---

## 5. START NOW — Phase 1 (≈ next 2 weeks): de-risk and re-measure what you already have

Low-regret, high-confidence, **no new data or redesign required.** These either protect live capital-readiness or sharpen the numbers you already trust.

### 5.1 Cost-sensitivity table + live-fill audit
- **Goal:** Find out how much of PEAD's +0.55 survives realistic earnings-window costs, and whether live fills already diverge from the model.
- **Method:** Re-run PEAD net Sharpe at one-way costs of **2 / 5 / 10 / 20 / 35 / 50 bps** (table). Separately, pull realized slippage on actual live PEAD fills vs the modeled next-open and compare to 5 bps. (Copilot shipped a usable scaffold — Test A3.)
- **Acceptance:** Net Sharpe ≥ 0.40 at **20 bps** one-way. If PEAD breaks at ~10 bps, it cannot survive live; reduce entry universe to the tightest-spread names or pause.
- **Backed by:** Claude, Gemini, Copilot.

### 5.2 PEAD crisis-block robustness (the highest-priority single test)
- **Goal:** Determine whether PEAD's edge is real or is one hand-tuned threshold sidestepping ~3 historical crises. The brief itself credits the VIX>30 block with the *entire* edge — that is ~3 events of evidence, not 6–8 folds.
- **Method:** (a) **Leave-one-crisis-out** — drop each VIX>30 episode (2018Q4, COVID-2020, 2022, Aug-2024) in turn; does the edge survive removing COVID alone? (b) **Threshold sweep** at 25/28/30/33/35 — stable across the range = robust; good only at exactly 30 = fit. (c) **Replace the discrete block** with a *regime-general* control: portfolio **volatility targeting** (scale gross to a target annualized vol) and/or a **trend filter** (SPY < 200dma → cut gross).
- **Acceptance:** PEAD survives under a *generic* regime control and under leave-one-crisis-out. If it only survives at VIX>30-exactly, or COVID alone carries it, **pause live PEAD** and treat it as unproven.
- **Backed by:** Claude (primary); regime-overlay echoed by DeepSeek, Gemini, Grok.

### 5.3 PEAD significance, done honestly for an event strategy
- **Goal:** Get a significance estimate whose unit of independence is the *earnings event*, not the fold or the day.
- **Method:** **Event-clustered block bootstrap** — resample earnings-event clusters (by season/quarter) with replacement to build the null distribution of the strategy Sharpe; this respects overlapping-hold dependence. Cross-check with a **Newey-West (HAC) t-stat** on the overlapping path-return series (Gemini's refinement — uses granular data while penalizing overlap). Keep `N_eff = n_folds` as the conservative floor.
- **Acceptance:** PEAD positive at block-bootstrap p < 0.05 **and** Newey-West t ≥ 2.0. Expect the honest number to be *weaker* than t=2.26.
- **Backed by:** Claude + DeepSeek (event clustering); Gemini (Newey-West). **Not** Copilot's AR(1).

### 5.4 Convert PEAD to SUE
- **Goal:** Replace the noisy `(actual − est)/|est|` (which explodes for tiny estimates) with the academically correct signal.
- **Method:** **SUE = (actual EPS − expected EPS) / σ(firm's trailing surprises)** (e.g., std of prior 8 quarters' surprises; analyst-estimate dispersion is an acceptable alternative denominator). Sort/enter on SUE deciles rather than a flat +5% cutoff. (Copilot's Experiment 1 / Test A1 is a usable scaffold.)
- **Acceptance:** SUE-PEAD net Sharpe ≥ current PEAD with a cleaner (less fat-tailed) path distribution.
- **Backed by:** Claude, Copilot, Gemini.

### 5.5 Fix the harness (purge + holdout)
- **Goal:** Stop starving statistical power and resolve the holdout ambiguity.
- **Method:** (a) **Shorten feature lookbacks** (e.g., shift 60-day technicals → 20-day) so the required purge shrinks to **≈ label horizon (20d) + embargo (~5d) ≈ 25–30 days**, not 85. Feature lookback looks *backward* and doesn't leak forward labels; purge should be driven by the label horizon. (b) **Resolve the holdout:** the brief's "sacred holdout" date (2026-11-09) is in the *future* — so there is currently **no historical holdout** and every backtest is in-sample at the meta level. Decide (human call, §10): carve a real historical holdout now (e.g., reserve everything after 2024-06-01) and never touch it, **or** explicitly relabel it "forward paper validation."
- **Acceptance:** Purge ≤ 30 days with no leakage detected in a synthetic-leak test; holdout policy documented and enforced by an assertion.
- **Backed by:** Claude, Gemini (purge); Claude (holdout).

### 5.6 Overlay ablation harness → demote RM to a safety net
- **Goal:** Prove each live overlay *adds* Sharpe, or strip it. Make the live system equal the validated one.
- **Method:** Build an **ablation harness** (Copilot shipped a clean `apply_overlay` scaffold): run PEAD-clean vs PEAD-with-each-overlay (regime sizing, news sizing, opportunity gate, macro block, conviction sizing) head-to-head in the **same CPCV pipeline**. Adopt Gemini's rule: **every sizing/concentration/correlation rule migrates into the PM/optimizer and is mirrored in the backtest; the RM keeps only catastrophic safety functions** (kill switch, margin breach, data-dropout, fat-finger). **If you can't backtest a rule, you can't run it live.**
- **Acceptance:** Each retained overlay improves net Sharpe **or** cuts max drawdown ≥ 15% without cutting Sharpe > 0.1. Everything else is removed. Live config == backtest config.
- **Backed by:** Gemini (best articulation), Claude, Copilot, all five.

---

## 6. INVESTIGATE / DESIGN FIRST — Phase 2 (≈ next 4–8 weeks): structural fixes needing a spike before commitment

These are higher-value but require a design decision or a data acquisition. Spike each before committing engineering.

### 6.1 ⭐ The breadth + dollar-neutral re-test of the swing ranker (THE decisive experiment)
- **Goal:** Determine whether your "dead" cross-sectional ML was ever dead, or just strangled by a 5-position long-only book. This is the single highest-value experiment in the plan.
- **Why:** The Fundamental Law (`IR ≈ IC × √breadth × TC`) says 5 long-only positions on a 1000-name ranking discards ~85% of any edge; +0.22/t=0.17 is exactly what a *real* modest signal looks like through that constraint. You cannot distinguish "no signal" from "good signal, near-zero transfer coefficient" with the current book.
- **Method:** Re-run the ranker as **dollar-neutral, sector-neutral, 40–80 names per side**, where the **short side = bottom of the same ranker** (this is standard market-neutral practice — *not* the QualityShort "short bad companies" thesis, which correctly failed; the two are different and must not be conflated). Apply Gemini's fix: **residualize both features and labels** — regress features and the forward return on market + sector (+ size) factors and train the ranker to predict the *idiosyncratic* residual return, not raw return.
- **Acceptance:** If dollar-neutral, residualized, high-breadth Sharpe clears ~0.6–0.8 net, the signal was always there and the long-only-5 frame was the killer. If it *still* fails, the ranker is genuinely dead — and you'll have proven it the right way.
- **Backed by:** Claude (breadth/transfer-coefficient), Gemini (residualize features+labels; factor-neutral), Grok (go L/S). **Requires short-interest data (§6.3) to run the short leg responsibly.**

### 6.2 Standalone analyst-revision momentum → an "information-diffusion sleeve"
- **Goal:** Harvest your best *untapped free* edge and combine correlated-mechanism signals to increase breadth of the same durable effect.
- **Method:** Build **analyst-revision momentum** (net estimate revisions / up-vs-down revision ratio) as a standalone cross-sectional signal from FMP data you already have (currently only a PEAD sub-filter). Then **combine revisions + SUE-PEAD + drift** into one sleeve — they exploit the *same* economic mechanism (slow diffusion of public information / underreaction), so stacking them increases the breadth of one real edge rather than diluting it.
- **Acceptance:** Revisions signal positive standalone (block-bootstrap p < 0.05); combined sleeve Sharpe ≥ best single component with lower variance.
- **Backed by:** Claude (information-diffusion sleeve), Gemini (idiosyncratic event focus), Copilot (revisions PIT discipline).

### 6.3 Short-interest data acquisition (cheap) → standalone signal + enables the short leg
- **Goal:** Unlock the dollar-neutral redesign (§6.1) and add a documented standalone large-cap edge.
- **Method:** Acquire **FINRA short interest** (near-free, semi-monthly) and a modest **borrow-availability/cost** feed. Test *changes* in short interest, days-to-cover, and crowded-short metrics as a cross-sectional signal (Boehmer-Jones-Zhang). Critically, you **cannot run a market-neutral short leg responsibly without knowing what's hard/expensive to borrow.**
- **Acceptance:** POC short-interest signal shows positive IC; borrow data integrated into the short-leg feasibility filter.
- **Backed by:** **All five rank short interest top-3.** Claude ranks it **#1** (cheap + standalone edge + unlocks the structural F2 fix); see the §10 decision vs options.

### 6.4 Read-only options IV as a PEAD signal-quality filter (NOT options trading)
- **Goal:** Capture the *salvageable* part of the options enthusiasm without the steamroller risk.
- **Method:** Acquire a **cheap historical + live ATM-IV / implied-move feed** (e.g., a low-cost vendor — not a full options-chain build). Use implied move to **gate/scale PEAD**: a +X% surprise against a much larger priced-in move is a weaker signal; a surprise against a small priced-in move is a stronger under-reaction. **No options positions, no Greeks, no selling vol** (see §4.1).
- **Acceptance:** IV-gated PEAD shows higher IC than price-only PEAD (Copilot's Experiment 3 is a usable read-only scaffold once you strip the trading).
- **Backed by:** Gemini (the good IV insight), Copilot (scaffold), Claude (deferred-but-as-signal). Full options-trading infra remains **deferred** until an equity book funds it.

### 6.5 Validation-power upgrade
- **Goal:** Make the CAPITAL gate reachable by a real edge without weakening it.
- **Method:** Move to **k = 12–15 folds with higher combinatoric depth** (more unique paths while keeping an N_eff floor); add **eigenvalue-based effective-trial count** for DSR (decompose the correlation matrix of your ~300 trial return streams so correlated hyperparameter variants don't count as full independent trials). The shortened purge (§5.5) is the prerequisite that makes more folds feasible.
- **Acceptance:** A known-good synthetic edge of true Sharpe ~0.7 clears the upgraded gate; correlated trial variants no longer over-deflate DSR.
- **Backed by:** Gemini (primary, both ideas), Claude (raise power not lower bar).

### 6.6 Regime overlay redesign (the general replacement for VIX>30)
- **Goal:** Replace the hand-tuned crisis threshold with a regime-general control.
- **Method:** **VIX term structure** — use VIX/VIX3M (you already have VIX3M): backwardation (ratio > 1) = stress → cut gross; contango = calm → normal. Combine with **portfolio vol-targeting** and an SPY-trend filter. Validate the overlay *independently* as a Sharpe-improver, not bolted on by faith.
- **Acceptance:** Term-structure + vol-target overlay improves OOS Sharpe of the core book and would have de-risked the Aug-2024 episode, without being fit to specific crises.
- **Backed by:** DeepSeek (term structure), Claude (vol-target/trend), Gemini/Grok (regime overlays).

### 6.7 (Lower priority) Honest tests of short-term and overnight reversal
- **Goal:** Check two reversal categories the brief left untested — *properly*, not in DeepSeek's broken form.
- **Method:** (a) **Short-term reversal** (1–5 day), large-cap, **dollar-neutral with 40–80 names/side** (NOT 5), with realistic costs and explicit modeling of the bid-ask bounce (a chunk of close-to-close "reversal" is microstructure noise you can't actually capture). (b) **Overnight reversal** after large intraday down-moves (Gemini's Strategy 3) — note this needs intraday data + a clean PIT news filter; it is *not* "cleanly validated on daily panels" as claimed, so spike the data dependency first.
- **Acceptance:** Either clears the honest gate net of realistic costs, or is documented dead. No 5-name versions; no guaranteed t-stats.
- **Backed by:** DeepSeek/Gemini (as categories), Claude (breadth + cost discipline required).

---

## 7. The redesign north star (where Phases 1–2 are heading)

The convergent recommendation across the strongest reviews (Claude + Gemini, with Grok concurring):

- **Core book — dollar-neutral, sector-neutral multi-factor.** Composite z-score of [revisions momentum + SUE-PEAD tilt + 12-1 momentum + quality + low-vol], sector-neutralized, **long top ~40–80 / short bottom ~40–80**, net beta ≈ 0, gross ≤ 80% NAV, inverse-vol sizing, weekly rebalance. This is the book that can clear ~Sharpe 1 net and survive regimes, because it harvests low-correlation premia with real breadth and the F2 beta cancels.
- **Event sleeve — information-diffusion (PEAD-family).** SUE-PEAD + standalone revisions drift, **20–40 small positions** (not 5), ~30–40-day hold, crisis-controlled via vol-target/term-structure (not a fitted VIX threshold).
- **Overlay — regime gross-scaling.** VIX term structure + vol-target + SPY-trend, validated independently.
- **Risk Manager — asymmetric safety net only.** All portfolio-construction logic lives in the PM/optimizer and is mirrored in the backtest.
- **Kill:** intraday 5-min, thesis shorts, the 5-position cap, the long-only default, the discrete VIX block, and every unvalidated live overlay. Simplify the agent topology to match.
- **Honest target:** net Sharpe **0.6–1.2** at the book level. The shops earning 2+ in US large-cap do it with TAQ, colocation, options desks, and securities-lending — not free data and $100k. A clean 0.8–1.2 dollar-neutral book on free data is a genuinely good result and is *achievable* with what you have; chasing 2+ is how disciplined shops talk themselves back into the overfit you just escaped.

---

## 8. Sequenced roadmap

| Phase | Workstream | Depends on | Acceptance gate |
|---|---|---|---|
| **P1 (wk 1–2)** | 5.1 Cost-sensitivity + live-fill audit | — | Net SR ≥ 0.40 @ 20bps |
| **P1 (wk 1–2)** | 5.2 PEAD crisis-block robustness | — | Survives leave-one-crisis-out + generic regime control |
| **P1 (wk 1–2)** | 5.3 Event-clustered block bootstrap + Newey-West | — | Block-bootstrap p<0.05 & NW t≥2.0 |
| **P1 (wk 1–2)** | 5.4 SUE conversion | — | SUE-PEAD ≥ current, cleaner tails |
| **P1 (wk 1–2)** | 5.5 Shorten purge + holdout policy | — | Purge ≤30d, no synthetic leak; holdout documented |
| **P1 (wk 1–2)** | 5.6 Overlay ablation → RM as safety net | — | Each overlay earns its place; live==backtest |
| **P2 (wk 3–8)** | 6.3 Short-interest data POC | budget call (§10) | Positive IC; borrow filter live |
| **P2 (wk 3–8)** | **6.1 Dollar-neutral residualized ranker re-test** | 6.3 | SR ≥0.6–0.8 net or proven dead |
| **P2 (wk 3–8)** | 6.2 Revisions signal + diffusion sleeve | 5.4 | Revisions p<0.05; sleeve ≥ best component |
| **P2 (wk 3–8)** | 6.4 Read-only IV PEAD filter | budget call (§10) | IV-gated IC > price-only IC |
| **P2 (wk 3–8)** | 6.5 Validation-power upgrade | 5.5 | Synthetic 0.7-SR edge clears gate |
| **P2 (wk 3–8)** | 6.6 Regime overlay redesign | 5.2 | Overlay improves OOS SR independently |
| **P2+ (wk 6+)** | 6.7 Reversal tests (honest, high-breadth) | 6.1 infra | Clears gate or documented dead |
| **P3** | §7 north-star integration | P1+P2 | Net book SR 0.6–1.2 in CPCV + forward paper |

---

## 9. Reusable test specs to implement (point the agent at the good scaffolds)

These exist as usable code scaffolds in the source reviews (mainly Copilot's appendix and experiments) — adapt, don't reinvent:

- **Cost-sensitivity table** — net Sharpe across 2/5/10/20/35/50 bps (Copilot Test A3). → §5.1
- **Overlay ablation** — `apply_overlay(signals, overlay_fn, label)` measuring each overlay's marginal Sharpe / drawdown / coverage-loss (Copilot §7.2). → §5.6
- **SUE quintile/decile test** — Bernard-Thomas-style spread + IC + hit-rate with strict OOS (Copilot Experiment 1 / Test A1), but use SUE denominator = trailing-surprise dispersion, not raw. → §5.4
- **Feature-importance stability** — SHAP CV across CPCV folds; drop features with CV > 1.0 (Copilot Test A4). → §6.1
- **Event-clustered block bootstrap** — resample earnings-event clusters by season to build the Sharpe null (synthesize; DeepSeek's "block bootstrap on earnings dates" instinct). → §5.3
- **Residualization pipeline** — regress features + forward return on market/sector/size; train on residuals (Gemini §6.2). → §6.1
- **Live/backtest gap decomposition** — cost / latency / decay / data / regime / noise (Copilot §2.7), for when live PEAD accrues a track record. → ongoing
- **Eigenvalue effective-trials for DSR** — correlation-matrix decomposition of the 300 trial return streams (Gemini §1.3). Requires storing trial return streams. → §6.5

⚠️ Do **not** implement Copilot's AR(1) `N_eff` (§4.3) or any options-selling backtest (§4.1). Strip the trading logic from Copilot's Experiment 3 — keep only the read-only IV-as-signal part.

---

## 10. Decisions that need the human (Min), not the agent

These are judgment calls the agent should surface, not make:

1. **Data budget allocation — the one real disagreement in the ensemble.** Four reviews rank **options #1**; this synthesis (and the structural argument) ranks **short interest #1**. Recommendation: **short interest first** (cheap, a standalone edge, *and* the precondition for the dollar-neutral fix to F2), then a **cheap read-only IV feed** as a PEAD signal enhancer (§6.4). **Defer full options-trading infrastructure** until an equity book funds it, and **never** sell vol blind (§4.1). Your call on dollar amounts and timing.
2. **Historical holdout vs forward-paper-only (§5.5).** Carving a real historical holdout now sacrifices some training data but gives you a true one-shot OOS test; relying on forward paper preserves data but means no historical holdout. Pick one and document it.
3. **Operational appetite for dollar-neutral / shorting.** §6.1 and the north star require running a short book (borrow, margin, Reg-T, short-squeeze risk). This is a real operational step up from long-only paper. Decide whether you're ready to take it on, even in paper.
4. **How much to keep instrumenting live PEAD while §5.2 runs.** If leave-one-crisis-out fails, pausing live PEAD is the conservative move consistent with your risk posture.

---

*Synthesis note for the agent: when the source reviews conflict, prefer the structural arguments (breadth, neutrality, sim-live alignment, honest event-significance) over the exciting ones (options vol, signal inversion, guaranteed Sharpes). The unglamorous path — fix breadth, go neutral, de-overfit the one edge, combine correlated-mechanism signals, align live to backtest — is the one most likely to move the honest number. Build accordingly.*
