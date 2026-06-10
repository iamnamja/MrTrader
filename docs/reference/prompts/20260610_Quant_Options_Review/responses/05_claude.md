# External Quant Review - MrTrader (Claude, 2026-06-10)

**Reviewer stance:** I read all attached docs (PIPELINE_ARCHITECTURE.md in full, including the gate inventory, KL list, and changelog; MODEL_STATUS.md; DECISIONS.md; the options program docs). This review is written against what you actually built, not a generic system. Where I disagree with your own conclusions, I say so and show the math.

---

## 0. Executive verdict (read this if nothing else)

Your validation harness is in the top decile of anything I've seen from a single operator - and it is now **miscalibrated in the opposite direction from the one you fear**. You spent months turning an inflating harness into an honest one. You succeeded. But you overshot: the gate as configured is not an alpha detector, it is a **false-negative machine** that will kill almost any *true* edge you could realistically find with your data budget. The evidence is in your own logs: everything dies, including textbook risk premia you confirmed are real (index VRP, PF 2.24, cost-robust). When a test rejects 100% of candidates including known-positives, the test is broken, not the candidates.

The single deepest problem is **statistical power**, and it's arithmetic, not opinion: a t-stat is `SR * sqrt(years)` (annual units). To clear t>=2.0 you need a true Sharpe-0.5 strategy observed for **~16 independent years**, or Sharpe 1.0 for ~4 years. You have 4 years of options data, ~19 quarterly event clusters for PEAD, and 6-8 CPCV folds. Under your own gate, **a genuinely real Sharpe-0.6 edge has well under a 50% chance of passing**. Your harness's Type I error is now excellent; its Type II error is catastrophic, and you've been interpreting Type II errors ("underpowered, KILL") as evidence about the world.

Three consequences flow from this, and they structure everything below:

1. **Stop gating sleeves at standalone significance. Gate the book.** Your own 2026-06-09 pause decision ("our gate is an alpha gate, short-vol is a risk premium") was the right insight applied too narrowly - it applies to *PEAD itself* and to everything you will ever test on <=5 years of data.
2. **Move from fold-level inference to event-level inference.** PEAD has hundreds of cross-sectional events per quarter. You are throwing away ~95% of your statistical power by collapsing them into ~8 fold Sharpes and then clustering at the quarter. Re-done correctly, PEAD may already be significant.
3. **Stop testing one binary threshold at a time.** The implied-move filter's "threshold fragility" is a methodology artifact, not necessarily a dead signal. Continuous features in one pre-registered panel model, not sequential filter-hunting.

---

## 1. Verdict on the validation harness

### 1.1 What's genuinely sound (don't touch)
- Per-fold retraining (KL-10b) - the frozen-CPCV-is-not-OOS finding and fix was the most important thing you did all spring.
- PIT discipline: `knowable_date`, expanding-quantile regime labels, `trained_through` save-guard, sacred holdout reserved at 2026-11-09 (a *forward* holdout - excellent; protect it).
- Survivorship: universe-from-traded-bars for options is correct.
- The overlap-guard bypass for rules-based scorers (KL-11).
- Mandatory 2x spread stress for options; mark-to-real-closes; the blow-up flag.
- A KILL-is-a-success culture.

### 1.2 Where the harness is INFLATING (smaller list, but real)
**(a) Forward-filled stale option marks smooth the equity curve.** Untraded contracts carry yesterday's close -> autocorrelated, vol-understated daily returns -> inflated options Sharpe/Calmar, understated drawdowns. Compute per-position fraction of marks that are forward-fills, report it on `SimResult`, add an unsmoothing check (Geltner AR(1) or weekly-return Sharpe comparison). The OPT-4 PF 2.24 deserves this check.
**(b) Trade-at-close circularity in options entries.** OPT-3 entered at T-1 close, selecting strikes/expiry off T-1 closes and filling at those same closes. Either select with T-2 / fill at T-1 close, or fill at next-day open with the spread cost doubled. Re-run OPT-3 canonical once under (ii).
**(c) Paper-fill optimism contaminates the capital-gate OR-path.** Alpaca paper fills at the touch with zero impact. Paper confirmation should be evidence about *implementation fidelity*, never about *edge existence*. Tighten the OR-path wording now.
**(d) Residual-alpha regresses on SPY only.** Single-factor. Regress on a small ETF factor set (SPY, IWM-SPY, MTUM-SPY, value proxy, maybe VIXY). One afternoon of work.
**(e) Modeled spread % was never calibrated to observed NBBO.** Log live NBBO daily for 4-6 weeks, fit half-spread % by (moneyness x DTE x underlying-class) buckets, replace the flat assumption. Cheapest high-impact fix in this document.
**(f) Verify underlying-level survivorship in the options backfill.** Confirm the 733-underlying backfill list was built from the R1K union including names delisted 2022-2026, not a current snapshot.

### 1.3 Where the harness is HIDING a real edge (the bigger problem)
**(a) The power problem, quantified.** Your PAPER gate needs t>=2.0 with N_eff = n_folds ~ 6-8. With realistic fold-Sharpe dispersion, a true mean of +0.55 gives an expected t of ~1.0-1.5 - the gate is set above the expected value of the statistic for a real edge of the size you're hunting. The things that pass are selected for luck.
**(b) Quarter-level clustering for PEAD is over-conservative - my single most actionable finding.** The standard treatment is a **panel of event-level hedged returns with two-way cluster-robust errors (by announcement date and by firm)** - not a quarterly block bootstrap on a portfolio equity curve. With ~1,500-3,000 PEAD-qualified events and effective clusters in the hundreds (announcement days), the same edge that bootstraps to p=0.19 at quarter level can be p<0.01 at day level. **Re-run PEAD inference at event level before writing another line of strategy code.**
**(c) Fold-Sharpe is the wrong test statistic for sparse event strategies.** CPCV remains right for path-dependent trained models (swing/intraday). For rules-based event scorers, CPCV should be the *robustness* check and the event panel the *significance* instrument.
**(d) No positive controls - the gate has never been calibrated.** Run known-real strategies through it: SPY buy-and-hold; plain 12-1 cross-sectional momentum; your own TSMOM (Sharpe 0.71 over 19y - does it pass its own gate on a 4-6y window?); 5 random-signal nulls (should fail ~95%+). If positives all fail on your data windows, you have measured the gate's Type II error empirically. Calibrate thresholds to where positive controls pass and nulls fail.
**(e) DSR's N_TRIALS=250 is theater.** Lean on pre-registration and the sacred holdout. Adopt: every experiment registers hypothesis, parameterization, pass criterion *before* the run (a `preregistered_at` timestamp that must precede the run timestamp).

### 1.4 The implied-move filter post-mortem - your FRAGILE verdict is premature
1.0 is the *unique economically meaningful point* (realized move inside vs outside the option-implied move). A plateau across 0.75/1.25 was never the right prediction; monotonicity of the *continuous* relationship is. The correct test: take all PEAD events with options coverage, compute the continuous `realized/implied` ratio, regress 5-day hedged forward return on it (or rank-correlate by decile), with day-clustered errors. On ~7-12 trades/fold, three binary cuts can flip sign on noise even when the continuous relationship is clean. Re-test inside the event panel as one pre-registered continuous feature before parking. (Honesty check: on the 2y options window your *baseline* PEAD had residual-alpha t = +0.04 - pure beta - so the filter manufacturing t=0.65 from a beta-only baseline means the filter was nearly the entire signal there. The event panel on 4y will tell you if that's real or a 2y artifact.)

---

## 2. The 3-5 highest-EV research directions (ranked)

### #1 - Build the earnings-event feature panel and make it the research factory (highest EV by a wide margin)
One table: every R1K earnings event 2022-2026 (~8-12k events), one row each. Columns: the label (1/3/5/10/20-day forward *beta-hedged* return from next open) and every PIT feature you can compute - equity side (SUE, revision momentum, announce-day gap, gap vs recent vol, prior-quarter drift, size, sector) and **options side** (pre-event implied move; IV run-up T-10->T-1; realized/implied reaction ratio; pre-event put-call IV spread at matched deltas; pre-event skew; options volume / share volume in the run-up; post-event IV retention). Pre-register: one model (OLS/rank regression per feature, then one regularized multivariate model), two-way clustered errors (announcement day x firm), train 2022-2024, validate 2025, sacred holdout untouched. No threshold picking. It converts every future event idea into a column and a regression, and it *increases* statistical power.

### #2 - Options-surface features as cross-sectional *equity* signals (alpha-shaped, sidesteps the cost wall)
Robust academic findings computable from your data: (a) call-put implied-vol spread at matched deltas (Cremers-Weinbaum CPIV: positive predicts positive stock returns over days-weeks); (b) implied skew (Xing-Zhang-Zhao: steep OTM-put skew predicts underperformance); (c) options/stock volume ratio (high O/S, put-heavy, is bearish); (d) IV term-structure slope around events. **Execution is in equities** - harvest options *information* at equity *costs*, sidestepping the spread wall. Use the existing dollar-neutral L/S harness; long top decile / short bottom decile of CPIV, weekly rebalance, multi-factor residual-alpha. Liquidity-filter IV inputs hard. Not reopening the dead price/fundamental XS-ML line - options-flow features are a genuinely different information set; but if simple decile sorts show nothing after costs, close it.

### #3 - Scale what already works: trend, broadened (the boring, capital-grade move)
TSMOM is your most statistically defensible asset (19 years, +0.71, crisis-convex) and under-built: 10 ETFs, one lookback, weekly Monday. Expansion is low-risk high-certainty EV: multiple lookbacks blended (3/6/12-month), vol-targeted sizing, broader sleeve (currency-hedged intl, more commodity legs, long-short on the ETF set). Each addition testable on 19+ years - the one place your significance machinery has the power to work.

### #4 - Book-level acceptance + small index-VRP diversifier (finish your own OPT-6 thought)
Index VRP is real and cost-robust (PF 1.75 @2x) and crisis-negative; trend is crisis-positive - textbook pairing. Build the **book-level gate**: does adding a small (5-10% risk budget), VIX-regime-gated short-vol sleeve improve the combined book's Sharpe/Calmar/worst-regime on the joint backtest? Acceptance is a *book delta* with a registered criterion. Guardrails: regime overlay must be your *existing* regime model (no new fitted parameters), defined-risk only, sized so a 2008-style event costs <=2% of NAV at the wings.

### #5 - Earnings long-vol selectivity (only if #1's panel hands it to you)
The mirror of OPT-3: are there identifiable events where implied systematically *underprices* the move (low pre-event IV run-up + high revision dispersion + small implied move vs historical)? Answerable for free inside the #1 panel. Build only if a selectable sub-population shows gross edge >= 2x the modeled spread.

**Explicitly NOT recommended:** dispersion as a trade (both legs pay single-name spreads - express as the panel's relative-VRP *feature* instead); dealer-gamma proxies (no OI); more short-vol parameterizations; small/mid PEAD re-runs; anything intraday.

---

## 3. Best alpha-shaped uses of the options data specifically
In EV order, feasible within your hard limits: 1. **Event-conditioning features for PEAD** (implied move, IV run-up, reaction ratio, post-event IV retention) - your data's killer app. 2. **Cross-sectional equity signals**: CPIV, skew, O/S volume ratio, term-structure slope - execution in equities. 3. **Book-level regime/risk inputs**: aggregate put-skew / index-IV-vs-realized as a *feature into the regime model* (the proper home for the "put-skew risk-off" idea). 4. **VRP-as-feature, not VRP-as-sleeve**: per-name computed IV minus forecast realized vol as a cross-sectional column. 5. **Index VRP as a small book diversifier** - the only options-*execution* use I'd endorse, behind the book-level gate.

Cannot support (don't burn cycles): true dealer-gamma (needs OI); 0DTE (intraday); quote-based execution alpha (NBBO); vol-surface arbitrage (real surface history).

---

## 4. Architecture / design gaps
**Under-built:** an event-research data layer (the panel table is the missing platform piece); gate calibration harness (positive/negative controls as one command); spread-model calibration job (nightly snapshot-NBBO logger -> bucketed half-spread table); multi-factor residual-alpha; live slippage reconciliation as a first-class metric; a borrow/locate model if the L/S equity line shows life. **Over-built / actively costing you:** standalone-sleeve significance as the universal acceptance framework; capacity gates at your capital scale; the infra-to-research ratio (budget ~70% research / 30% infra); three frozen ML training paths kept warm for dead strategies (archive aggressively). **Structural fragility:** the whole edifice keys on Polygon - persist the *computed* IV/greeks per contract-day (a one-time engine pass over the 112M bars) so the engine becomes optional for research reads and every panel query is a join, not a pricing run.

---

## 5. The first 5 things I'd change (prioritized), and the kill list
1. **Calibrate the gate with positive and negative controls.** One week. Until done, every KILL verdict has an unknown false-negative rate.
2. **Re-run PEAD significance at event level** (panel of hedged event returns, two-way day x firm clustered errors). Days, not weeks.
3. **Build the earnings-event feature panel** and re-adjudicate the implied-move signal inside it as a continuous pre-registered feature.
4. **Adopt a two-track acceptance framework**: Track A (alpha: event-panel significance + multi-factor residual-alpha + costs) for PEAD-family and the options-equity signals; Track B (book-delta: registered improvement to combined book Sharpe/Calmar/worst-regime) for risk premia and diversifiers (trend extensions, index VRP, regime features). Write it into PIPELINE_ARCHITECTURE.md as Section 7.0-B.
5. **Fix the three flattering mechanics before any new options verdict**: stale-mark fraction + unsmoothing check; calibrated spread table from snapshot NBBO; entry-fill convention re-run.

**Kill outright:** further single-name short-vol parameterizations; small/mid-cap PEAD; intraday anything; dispersion-as-a-trade; binary threshold sweeps as a validation method; the paper-confirmation OR-path *as evidence of edge*. **Double down on:** the trend sleeve (broaden it); the event-driven family via the panel; the PIT/survivorship data layer (your moat); the sacred holdout discipline; the Opus-adversarial-review habit.

---

## 6. Is the premise misframed?
Partly. "Single operator chasing capital-grade *standalone* alpha at t>=2.5, n_folds>=10" is unreachable arithmetic on your data budget. Restate the objective:

> **Run a diversified book of robust premia (trend, broadened; possibly a small regime-gated index-VRP sleeve) as the capital base, with one concentrated genuine-alpha effort (the earnings-event panel + options-conditioned PEAD) as the satellite - accepted at the book level, sized by live-tracked realized performance, with the sacred holdout and live reconciliation as the final arbiters instead of an unsatisfiable in-sample t-stat.**

**What would change my mind:** if the positive controls (TSMOM-on-4y, textbook momentum) *pass* the current gate, the power critique is overstated - run that test first. If event-level PEAD comes back p>0.15 even with day-level clustering, PEAD is genuinely marginal and the right book is trend-plus-cash while the panel hunts. If snapshot-calibrated spreads come back *wider* than the model, the cost wall is higher than assumed and even the equity-side signals deserve an extra haircut.
