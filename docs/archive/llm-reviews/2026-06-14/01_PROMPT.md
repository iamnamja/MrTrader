# Prompt — External Quant Research Review (MrTrader, 2026-06-14)

## Your role
You are a **world-class quantitative researcher / quant dev** — the kind who has shipped real systematic strategies, killed far more than you've shipped, and is allergic to overfitting, data-mining, and hopeful narratives. You are reviewing a solo-run systematic trading research program (MrTrader) that has just finished a rigorous gate rebuild and an honest sweep of its candidate strategies. **Almost everything has died.** The owner wants your brutally honest, deeply-reasoned answer to one question:

> **Given the data we actually have, the constraints we actually operate under, and everything we've already tried and killed — what are the best next research steps to find alpha? Where does testable, undiscovered edge plausibly still live?**

Think from first principles. Be brutal. If the honest answer is "the realistic remaining edge is small," or "you need data you don't have," or "trend is most of what's harvestable on free data" — say so plainly and explain why. Do not pad the list to look thorough. We would rather hear three real ideas (or one) than ten that won't survive contact with our gate or our data.

## What to read (in this pack)
1. **`02_STATE_SNAPSHOT.md`** — START HERE. Self-contained, hype-free: the data we actually have (with real coverage), every strategy we've tried and the exact number that killed it, what's live, how we now validate, and our binding constraints. If you read one thing, read this.
2. **`files/DATA_PROVIDERS.md`** — exactly which feeds we have, their plans/costs, what's free vs paid vs being cancelled. **The data envelope is the hard boundary on what's testable.**
3. **`files/RULER_V2_DESIGN.md`** — the promotion gate (now live). Any idea you propose must be *validatable* under this. Two tiers: PAPER (plausibility + a light HAC-SR significance floor + a diversifier regime waiver) and CAPITAL (Bayesian posterior + structural live-paper + multi-factor residual-α + bootstrap + power floor).
4. **`files/ML_EXPERIMENT_LOG.md`** + **`files/DECISIONS.md`** — the full kill ledger and the reasoning behind every verdict (LARGE — skim for the strategy lines + verdicts; the snapshot distills them).
5. **`files/ALPHA_V7_SYNTHESIS_AND_PLAN.md`** + **`files/NEXT_PHASE_BLUEPRINT_2026-06.md`** — the current strategic direction (operate a 3–5 sleeve risk-premia book at book SR ~0.8–1.0). Prior external-LLM panels fed this; we want you to pressure-test and extend it, not just agree.
6. **`files/PIPELINE_ARCHITECTURE.md`** — the WF/CPCV harness, gate inventory, known limitations (LARGE — reference for "how we test").
7. **`files/MODEL_STATUS.md`**, **`files/MASTER_BACKLOG.md`**, **`files/SYSTEM_BEHAVIOR.md`** — what's live, what's planned, runtime behavior.

## The hard truths to internalize before proposing anything
- **The only surviving edge is trend (TSMOM, ~0.71 Sharpe / 19y, 10 US ETFs), live at 25% + cash.** Everything else has died: cross-sectional ML ranking (noise/beta), PEAD (demoted at the event level, t=−0.77), options-as-signal H4a–e (significantly *negative*), earnings IV-crush (kill), index short-vol/VRP (kill; re-confirmed −0.21 Sharpe under spread-stressed CPCV), the implied-move filter (threshold-fragile), and trend-broadening to more ETFs + L/S (parked — genuinely worse than the simple sleeve).
- **These are HONEST negatives, not gate artifacts.** We deliberately rebuilt the gate to be *less* Type-II (it waives the worst-regime backstop for diversifiers and replaces a saturated DSR with a Bayesian posterior). Candidates were re-scored under best-case settings and still failed on real significance. So "just lower the bar" is empirically refuted.
- **Power is the binding statistical constraint.** ~8-fold CPCV on ≤4y of data cannot reliably validate a true Sharpe-0.4–0.7 edge. Realistic edges live at SR ~0.4–0.7, so **deep history (19y free ETF data, long FRED macro) is the only way to get the statistical power to confirm one.** A great idea we can't validate is, for us, not actionable.
- **Data envelope (post-cancellation):** effectively free feeds only — yfinance (ETFs/equities, ~19y daily), FRED (macro, decades), AlphaVantage (free tier), FMP Starter ($29, `/stable` only). We OWN ~4y of US index-options bars (frozen ~June 17 — no new options data after). No futures/FX/crypto feed wired. No live options NBBO.
- **Single quant-dev bandwidth.** Prefer a few high-expected-value, data-complete bets over a sprawling agenda.

## What we want from you

### 1. Brutally honest assessment (half a page)
Is our read correct — that on our data envelope, trend is most of the harvestable systematic edge and the rest genuinely died? Where, specifically, might we be wrong (a real edge we mis-killed, a data asset we're under-using, a methodology blind spot)? Call out anything in the kill ledger you think was killed for the *wrong reason*.

### 2. Ranked next research bets (the core deliverable)
Give **3–5 concrete research bets**, ranked by `expected_alpha × testability_with_our_data × not-already-dead`. For EACH:
- **Thesis** — the economic mechanism (why the edge exists and persists; who pays for it).
- **Data required** — and *do we have it* (cite the snapshot/DATA_PROVIDERS). If it needs data we don't have, say what and how cheaply it's obtainable on free/low-cost feeds — or mark it un-testable for us.
- **How to test it under our harness** — concretely (universe, signal, the CPCV window, can ~8-fold CPCV even validate it given the power constraint?). Be honest if it's under-powered.
- **Expected Sharpe range** and whether it's a standalone (Track-A) edge or a **diversifier/risk-premium** for the book (Track-B). We specifically want sleeves that *diversify trend*, not correlate with it.
- **Why it might fail / what would kill it** — pre-mortem.
- **Effort** (days/weeks for a solo dev).

### 3. The book view
We want a **3–5 sleeve risk-premia book at book SR ~0.8–1.0**, not a single home run. Given trend is the anchor, what 2–4 *genuinely diversifying* sleeves are most realistic to stand up on our data? What's the most credible path from "trend-only" to a real multi-sleeve book?

### 4. What NOT to pursue
Explicitly list ideas we should *not* spend time on given our data/power constraints or because we've already killed them for good reasons — so we don't re-tread dead ends.

### 5. The one bet
If you had to pick **one** research bet for a solo dev to pursue over the next 2–4 weeks to maximize the probability of adding a real, validatable, trend-diversifying sleeve — what is it, and why?

## Output format
- Lead with the **honest assessment (§1)** — don't bury it.
- Then the **ranked bets (§2)** as a structured list (thesis / data-have? / test / expected SR / pre-mortem / effort).
- Then **§3 book view**, **§4 do-not-pursue**, **§5 the one bet**.
- End with a one-paragraph **bottom line**.
- Cite specifics from the pack (data coverage, kill numbers, the gate) — generic advice ("try more factors", "use ML") is not useful and we will discount it. Assume we know the textbook; tell us what *you* would actually do with *our* data.
