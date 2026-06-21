# 04 — The validation harness + the multiple-testing problem (Theme B)

This is the file for judging whether our **t = 2.29 second engine is real or residue.** We have
been wrong in BOTH directions before (too strict, then a Type-I leak), so stress-test the ruler —
and especially the part we have NOT yet built: family-wise multiple-testing.

## The two-track gate (Ruler-v2, live)
- **Track-A (is it a real return stream?)**
  - **PAPER tier:** point-Sharpe floor (≥0.30) + a HAC-Sharpe significance floor (one-sided
    p<0.05, pooled OOS) + a non-catastrophic worst-regime backstop (waived for declared
    diversifiers) + a per-sleeve trial cap.
  - **CAPITAL tier:** a closed-form **Bayesian posterior P(SR>0)** combining backtest + a
    **structural live-paper requirement** (cannot reach capital on backtest alone) + a
    **multi-factor residual-α t** (HAC) + stationary bootstrap + PBO/CSCV (if multiple configs) +
    a **hard power floor** (n_folds ≥ 10).
- **Track-B (does it improve the BOOK?)** — a budget-invariant **appraisal ratio** (residual-α IR
  of candidate vs the existing book) + block-bootstrap P(ΔSR>0). How diversifiers are judged
  (not on a standalone-SR floor). **This is the track that gave the futures book t = 2.29.**

## Inference internals
- **HAC Sharpe** (Newey-West) for SR>0; **stationary bootstrap** (Politis-Romano) for P(SR>0);
  **PBO via CSCV** (Bailey/López de Prado); **multi-factor residual-α**; **CPCV** (combinatorial
  purged CV, 85-day purge + embargo). Rules-based sleeves are treated as OOS-by-construction
  (no per-fold retrain); a *trained model* must per-fold-retrain to promote.
- **Validate-the-validator:** we pushed known anomalies through the real feature→label pipeline
  and confirmed faithfulness (label-fidelity +0.76; 3/3 recovered; 0 deflationary bugs).
- **Negative controls (new, 2026-06):** true-null PAPER FP 23.6% floor-alone → **5.3% JOINT** with
  the HAC floor; anti-correlated zero-edge null → Track-B pass-rate **5.7%**. → **Type-I controlled
  at the *single-sleeve* level.**

## The gap we want you on: FAMILY-WISE multiple testing (still uncounted)
The negative controls above bound the false-positive rate of **one** test. They do **not** account
for the fact that, across 2 years, **we have tried ~20 sleeve families:**

> swing ML ranker · intraday ML · PEAD · options earnings IV-crush · index VRP (options) ·
> overnight · turn-of-month / calendar · ETF stat-arb / RV · rates carry · aggregate
> short-interest · FINRA short-volume · credit-spread overlay · crypto trend · **futures carry** ·
> **futures xs-momentum** · futures curve-momentum · futures value · futures skew · futures
> basis-momentum · CFTC CoT.

Most were KILLED — but **selection happens at the family level**, and our gate's "rules-based =
OOS-by-construction" assumption is silent on that. So the survivors (carry, xsmom, the t=2.29
book, VRP) have an **uncounted multiple-testing burden.** Concretely we are unsure:
1. After honest family-wise correction (Bonferroni is too blunt; we're thinking a **null-strategy
   zoo** → empirical per-family FP calibration, à la a deflated-Sharpe / White's Reality Check /
   Hansen SPA approach), **does t = 2.29 still clear?** What bar should it clear?
2. How do we build that null zoo *for rules-based cross-sectional futures factors* (where there's
   no model to retrain) — randomized sign/rank signals on the same universe? Block-permuted
   signals? How many nulls, and what statistic?
3. How do we **prospectively** log research degrees-of-freedom (discarded variants, bug-fix
   reruns, reviewer-suggested specs, post-hoc exclusions) so this doesn't rot again?

## Our own known failure modes (please pressure-test)
1. We were a **Type-II machine** (a t≥2 bar on ~8 folds of ≤4y data kills true SR-0.5 edges).
   Caught + recalibrated. *Are we now too *loose* at the family level — trading a survivor of 20 tries?*
2. We had a **Type-I leak** (the 0.30 SR floor admits ~23% of nulls), closed with the HAC floor.
3. **Risk-premia-vs-alpha framing** — our gate is an alpha gate; Track-B is the risk-premia fix.
   *Is the futures book being judged correctly, or is Track-B too easy to pass by stacking marginal factors?*
4. The t=2.29 came from **combining** carry (t 1.76) + xsmom (t 1.60). *Is "gate the basket" a
   legitimate significance gain, or a multiple-testing trick (you can always find a weighted combo
   that crosses 1.96)?* **This is the question we're most nervous about.**

## What we want answered (Theme B)
**Do you trust t = 2.29?** Give us the concrete protocol (null-strategy zoo design, statistic,
deflated bar) to either confirm the second engine is real or expose it as multiple-testing residue —
something we can build and run this week.
