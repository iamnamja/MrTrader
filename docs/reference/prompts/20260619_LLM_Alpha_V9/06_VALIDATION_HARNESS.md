# 06 — The Validation Harness ("is our ruler sound?")

This is the most important file for judging whether our verdicts are trustworthy. **We have been
wrong in BOTH directions before** (too strict, then a leak), so please stress-test the ruler itself.

## The two-track acceptance gate (Ruler-v2, live)
A sleeve is judged on two independent tracks:

- **Track-A (is it a real return stream?)**
  - **PAPER tier (plausibility):** point-Sharpe floor (≥0.30) + a light HAC-Sharpe significance
    floor (one-sided p<0.05, pooled OOS) + a non-catastrophic worst-regime backstop (waived for
    declared diversifiers/risk-premia) + a per-sleeve trial cap.
  - **CAPITAL tier (significance):** a closed-form **Bayesian posterior P(SR>0)** (replaces a
    saturated DSR) that combines the backtest with a **structural live-paper requirement** (you
    cannot reach capital on a backtest alone), + a **multi-factor residual-α t** (HAC), + a
    stationary bootstrap, + PBO/CSCV if multiple configs, + a **hard power floor** (n_folds≥10).
- **Track-B (does it improve the BOOK?)** — a budget-invariant **appraisal ratio** (residual-α IR
  of the candidate vs the existing book) + block-bootstrap P(ΔSR>0). This is how diversifiers /
  risk-premia are judged (not on a standalone-SR floor).

Plus, applied by hand on the decisive runs: a **pre-registered sub-period stability guard**
(an edge must hold across halves / the modern era) — this has killed real-looking edges
(rates carry = pre-2016 artifact) and flipped a tempting one (daily-carry).

## Inference internals
- **HAC Sharpe** (Newey-West, Bartlett) for SR>0 significance on autocorrelated daily OOS series.
- **Stationary bootstrap** (Politis-Romano) for P(SR>0) + CIs.
- **PBO via CSCV** (Bailey/Lopez de Prado) for overfitting probability.
- **Multi-factor residual-α** (the canonical residualization; a premia book needs multi-factor,
  not SPY-only).
- **CPCV** (combinatorial purged CV) with 85-day purge + embargo; rules-based sleeves are
  OOS-by-construction (per-fold-retrain not required); a trained model MUST per-fold-retrain
  to promote (`REQUIRE_TRUE_WF_FOR_PROMOTION=True`).
- **Validate-the-validator (P0):** we pushed known anomalies through the REAL feature→label
  pipeline and confirmed it's faithful (label-fidelity +0.76, 3/3 anomalies recovered, 0
  deflationary bugs) — i.e. IC≈0 on equity ML is the market, not a pipeline bug.

## Our own known failure modes (please pressure-test these)
1. **We were a Type-II machine (Alpha-v6).** A t≥2 bar on ~8 folds of ≤4y data kills true
   Sharpe-0.5 edges (t ≈ SR·√years). We caught this with positive/negative controls (TSMOM-on-4y
   as a decisive positive control) and recalibrated. *Are we still too strict somewhere — e.g.,
   did we wrongly kill VRP / options signals because the data was too short, not because they're
   dead?*
2. **We had a Type-I leak too.** A true-null seed once PASSED the plausibility floor (the 0.30 SR
   floor admits ~23% of zero-edge nulls at n≈1500); we closed it with the HAC significance floor.
   *Is the PAPER tier still too permissive?*
3. **Risk premia vs alpha framing.** Our gate is an *alpha* gate; we repeatedly mis-rejected
   *risk premia* (VRP) by applying alpha standards. Track-B was the fix. *Is carry being judged on
   the right track? Is short-vol genuinely uninvestable for us, or just mis-framed again?*
4. **In-sample normalization.** Some book-level Sharpes use full-sample vol-matching (flagged).
5. **Survivorship** in the equity universe (yfinance) and the futures liquidity filter.

## The question we most want answered
**Do you trust our verdicts?** Specifically: is "free daily US-equity directional alpha is mined
out" a sound conclusion or a function of a still-miscalibrated ruler / too-short data / wrong
framing? Where would a world-class quant *not* believe our KILLs — and what would you re-run, and
how, to settle it?
