# External Quant Review — MrTrader (Claude, 2026-06-10)

**Reviewer stance:** I read all attached docs (PIPELINE_ARCHITECTURE.md in full, including the gate inventory, KL list, and changelog; MODEL_STATUS.md; DECISIONS.md; the options program docs). This review is written against what you actually built, not a generic system. Where I disagree with your own conclusions, I say so and show the math.

---

## 0. Executive verdict (read this if nothing else)

Your validation harness is in the top decile of anything I've seen from a single operator — and it is now **miscalibrated in the opposite direction from the one you fear**. You spent months turning an inflating harness into an honest one. You succeeded. But you overshot: the gate as configured is not an alpha detector, it is a **false-negative machine** that will kill almost any *true* edge you could realistically find with your data budget. The evidence is in your own logs: everything dies, including textbook risk premia you confirmed are real (index VRP, PF 2.24, cost-robust). When a test rejects 100% of candidates including known-positives, the test is broken, not the candidates.

The single deepest problem is **statistical power**, and it's arithmetic, not opinion: a t-stat is `SR × √years` (annual units). To clear t≥2.0 you need a true Sharpe-0.5 strategy observed for **~16 independent years**, or Sharpe 1.0 for ~4 years. You have 4 years of options data, ~19 quarterly event clusters for PEAD, and 6–8 CPCV folds. Under your own gate, **a genuinely real Sharpe-0.6 edge has well under a 50% chance of passing**. Your harness's Type I error is now excellent; its Type II error is catastrophic, and you've been interpreting Type II errors ("underpowered, KILL") as evidence about the world.

Three consequences flow from this, and they structure everything below:

1. **Stop gating sleeves at standalone significance. Gate the book.** Your own 2026-06-09 pause decision ("our gate is an alpha gate, short-vol is a risk premium") was the right insight applied too narrowly — it applies to *PEAD itself* and to everything you will ever test on ≤5 years of data.
2. **Move from fold-level inference to event-level inference.** PEAD has hundreds of cross-sectional events per quarter. You are throwing away ~95% of your statistical power by collapsing them into ~8 fold Sharpes and then clustering at the quarter. Re-done correctly, PEAD may already be significant (details in §1.3).
3. **Stop testing one binary threshold at a time.** The implied-move filter's "threshold fragility" is a methodology artifact, not necessarily a dead signal (§1.4). Continuous features in one pre-registered panel model, not sequential filter-hunting.

---

## 1. Verdict on the validation harness

### 1.1 What's genuinely sound (don't touch)

- Per-fold retraining (KL-10b) — the frozen-CPCV-is-not-OOS finding and fix was the most important thing you did all spring. Most shops never catch this.
- PIT discipline: `knowable_date`, expanding-quantile regime labels, `trained_through` save-guard, sacred holdout reserved at 2026-11-09 (a *forward* holdout — excellent; protect it with your life).
- Survivorship: universe-from-traded-bars for options is the correct construction, strictly better than the reference endpoint. Delisted-inclusive equity universes, delisting haircut.
- The overlap-guard bypass for rules-based scorers (KL-11) — that fold-skip bias was real and you found it.
- Mandatory 2× spread stress for options; mark-to-real-closes rather than model prices; the blow-up flag.
- A KILL-is-a-success culture. Genuinely rare.

### 1.2 Where the harness is INFLATING (smaller list, but real)

**(a) Forward-filled stale option marks smooth the equity curve.** Untraded contracts carry yesterday's close. Stale marks → autocorrelated, vol-understated daily returns → **inflated options Sharpe and Calmar, understated drawdowns**. Minor for SPY/QQQ condors; material for single names (your own OPT-1a validation found 3%+ stale marks even on liquid names, and that was the *liquid* tail). Fix: compute per-position fraction of marks that are forward-fills, report it on `SimResult`, and add an unsmoothing check (Geltner-style AR(1) correction or simply re-compute Sharpe on weekly returns and compare — a big gap is the tell). Your OPT-4 PF 2.24 deserves this check before you ever cite it again.

**(b) Trade-at-close circularity in options entries.** OPT-3 entered at T-1 close, selecting strikes/expiry off T-1 closes and filling at those same closes. You cannot observe the close until it prints, and then you can't trade at it. This is the standard MOC approximation and usually mild — but pre-earnings option prices move fast into the close (IV ramps into the print), so for *this specific* strategy family the bias is plausibly non-trivial and **flattering to short-vol entries** (you sell at the marked-up close). Either (i) select with T-2 / fill at T-1 close, or (ii) fill at next-day open with the spread cost doubled. Re-run OPT-3 canonical once under (ii) — if the verdict moves, your fill assumption was doing work.

**(c) Paper-fill optimism contaminates the capital-gate OR-path.** Your CAPITAL tier accepts `t≥2.5 OR documented paper confirmation`. Alpaca paper fills at the touch with zero impact and zero adverse selection. Marketable-limit-cross entries in live PEAD will systematically do worse than paper. Paper confirmation should be evidence about *implementation fidelity* (did the live path reproduce the backtest book?), never about *edge existence*. Tighten the wording of that OR-path now, before it matters.

**(d) Residual-alpha regresses on SPY only.** Single-factor. A long-only PEAD book loads on size, momentum, and short-vol-like factors that SPY beta won't absorb. Cheap fix: regress on a small ETF factor set (SPY, IWM−SPY, MTUM−SPY, value proxy, maybe VIXY for vol). If PEAD's residual-α survives a 4-factor hedge, that's a much stronger statement; if it doesn't, you want to know today. One afternoon of work.

**(e) Modeled spread % was never calibrated to observed NBBO.** You have live NBBO via the snapshot. Log it daily for 4–6 weeks, fit half-spread % by (moneyness × DTE × underlying-class) buckets, and replace the flat assumption. Right now your decisive cost input — the thing that killed OPT-3 and that OPT-4 survived — is an uncalibrated guess stress-tested by multiplying the guess. The stress discipline is right; the anchor is unverified. This is the cheapest high-impact fix in this entire document.

**(f) Verify underlying-level survivorship in the options backfill.** Contracts are survivorship-safe by construction, but the 733-underlying *backfill list* — confirm it was built from the R1K union including names delisted 2022–2026, not from a current membership snapshot. If a 2023 delisting isn't in the backfill, every earnings-event options study silently excludes the most catastrophic events (which, for short-vol verdicts, would have made KILLs even more negative — but for *long*-vol or signal studies it inflates).

### 1.3 Where the harness is HIDING a real edge (the bigger problem)

**(a) The power problem, quantified.** Your PAPER gate needs t≥2.0 with N_eff = n_folds ≈ 6–8. The t-stat of a fold-mean is `mean/(σ/√n)`. With realistic fold-Sharpe dispersion (your swing runs showed σ ≈ 3 across folds; PEAD ≈ 1–1.5), a true mean of +0.55 gives an expected t of roughly 1.0–1.5 — i.e., **the gate is set above the expected value of the statistic for a real edge of the size you're hunting**. You will pass only on lucky draws, which means the things that *do* pass are selected for luck — the gate paradoxically enriches for overfit among passers while killing honest mediocrity. (You already noticed this exact paradox in the SR promotion gate context; it generalizes.)

**(b) Quarter-level clustering for PEAD is over-conservative — this is my single most actionable finding.** Your honest PEAD verdict (event-clustered bootstrap p≈0.19, "cluster-rate-limited at ~4 independent clusters/yr") treats the *earnings quarter* as the unit of independence. That's defensible only if drift outcomes are strongly correlated across the whole 6-week earnings season. They are not, once you beta-hedge: a 5-day hold means events ≥2 weeks apart share almost no holding window, and the within-day/within-week cross-sectional correlation of *hedged* event returns is the actual dependence that matters. The standard treatment (the entire event-study literature, and how every multi-manager stat-arb pod evaluates event signals) is a **panel of event-level hedged returns with two-way cluster-robust errors (by announcement date and by firm)** — not a quarterly block bootstrap on a portfolio equity curve. With ~1,500–3,000 PEAD-qualified events over 6 years and effective clusters in the *hundreds* (announcement days), the same underlying edge that bootstraps to p=0.19 at quarter level can easily be p<0.01 at day level. **Re-run PEAD inference at event level before you write another line of strategy code.** Either result changes your roadmap: significant → PEAD graduates, size it with confidence; not significant → PEAD is weaker than even you think, and the live book is trend + telemetry.

**(c) Fold-Sharpe is the wrong test statistic for sparse event strategies generally.** Collapsing a fold to one Sharpe throws away within-fold breadth; for a sleeve trading 7–12 times per fold, fold Sharpe is mostly noise about *which* events landed in the fold. CPCV remains the right machine for path-dependent, trained models (swing/intraday — where you used it correctly). For rules-based event scorers, CPCV should be the *robustness* check and the event panel the *significance* instrument. You've half-built this realization (event-clustered bootstrap exists); finish it.

**(d) No positive controls — the gate has never been calibrated.** You've run dozens of candidates through the gate and they all failed. You have never run a *known-real* strategy through it to confirm the gate can pass anything. Do this this week: (i) SPY buy-and-hold; (ii) plain 12-1 cross-sectional momentum on the R1K (a 30-year published premium); (iii) your own TSMOM (Sharpe 0.71 over 19y — does it pass its own house's gate on a 4–6y window? I'd bet it fails PAPER); (iv) 5 random-signal nulls (negative controls — should fail ~95%+). If (i)–(iii) all fail on your data windows, you have measured the gate's Type II error empirically and the conclusion is unavoidable: **the gate cannot certify anything real at your sample sizes**, and "FAILED our gate" stops being informative. Calibrate thresholds to where the positive controls pass and the nulls fail. This converts the gate from a philosophical statement into a measured instrument.

**(e) DSR's N_TRIALS=250 is theater (you know this).** The true trial count after parameterization sweeps, structure variants, and "canonical re-runs" is unknowable. The honest multiplicity controls you actually have are pre-registration and the sacred holdout. Lean on those, demote DSR to a reported curiosity, and adopt a hard rule: **every experiment registers its hypothesis, parameterization, and pass criterion in DECISIONS.md *before* the run** (you mostly do this; make it mandatory and machine-checked — a `preregistered_at` timestamp field that must precede the run timestamp).

### 1.4 The implied-move filter post-mortem — your FRAGILE verdict is premature

The sweep at {0.75, 1.0, 1.25} expecting a "plateau" embeds a wrong assumption: that the cut points are arbitrary. They're not — **1.0 is the unique economically meaningful point** (realized move inside vs. outside the option-implied move; the entire "priced-in" hypothesis lives at that boundary). A plateau across 0.75/1.25 was never the right prediction; monotonicity of the *continuous* relationship is. The correct test, which you haven't run: take all PEAD events with options coverage, compute the continuous `realized/implied` ratio, and regress 5-day hedged forward return on it (or just rank-correlate, by ratio decile). A real priced-in effect shows up as a monotone (or single-kink-at-1.0) relationship in the event panel with day-clustered errors. The sign flip at 1.25 *is* concerning — but on ~7–12 trades/fold, three binary cuts can flip sign on noise even when the continuous relationship is clean. **Don't park this line on the current evidence; re-test it correctly inside the event panel (§2.1), with the 4y backfill, as one pre-registered continuous feature.** If the panel coefficient is flat, *then* kill it with confidence.

(Related honesty check you should run: on the 2y options-covered window, your *baseline* PEAD had residual-α t = +0.04 — i.e., on that window PEAD itself was pure beta. The filter manufacturing t=0.65 from a beta-only baseline means the filter was nearly the entire signal there. That's either very exciting or a 2y artifact; the event panel on 4y will tell you which.)

---

## 2. The 3–5 highest-EV research directions (ranked)

### #1 — Build the earnings-event feature panel and make it the research factory (highest EV by a wide margin)

**Mechanism.** One table: every R1K earnings event, 2022→2026 (≈8–12k events), one row each. Columns: the label (1/3/5/10/20-day forward *beta-hedged* return from next open) and every PIT feature you can compute — equity side (SUE, revision momentum, announce-day gap, gap vs. recent vol, prior-quarter drift, size, sector) and **options side** (pre-event implied move; IV run-up over T-10→T-1; realized/implied reaction ratio; pre-event put-call IV spread at matched deltas; pre-event skew; options volume / share volume in the run-up; post-event IV retention). All computable from your data + engine.

**Why it could be real alpha.** PEAD is your one validated mechanism; the options run-up/reaction features are exactly the conditioning information theory says should separate "genuine surprise" from "priced." You already saw the strongest hint of your entire program here (alpha-like lift, β flat). The panel is the right instrument where the binary-filter workflow was the wrong one.

**How to test without fooling yourself.** Pre-register: one model (start with OLS/rank regression per feature, then one regularized multivariate model), two-way clustered errors (announcement day × firm), train on 2022–2024, validate 2025, sacred holdout untouched. No threshold picking — the model output is a continuous tilt/filter for PEAD sizing. Promotion criterion registered in advance: panel t on the held-out year, plus the PEAD-sleeve CPCV as a robustness check (not the significance instrument).

**Why this beats everything else:** it converts every future event idea (revisions, guidance, pre-announce drift, your put-skew risk-off idea) from a bespoke 2-week harness build into a column and a regression. It is also the only direction that *increases your statistical power* rather than fighting it.

### #2 — Options-surface features as cross-sectional *equity* signals (alpha-shaped, sidesteps the cost wall)

**Mechanism.** The robust academic findings on options-predict-stocks are all computable from your data: (a) **call−put implied-vol spread at matched deltas** (Cremers–Weinbaum: informed-trading pressure; positive CPIV predicts positive stock returns over days–weeks); (b) **implied skew** (Xing–Zhang–Zhao: steep OTM-put skew predicts underperformance for months); (c) **options/stock volume ratio** (Roll–Schwartz–Subrahmanyam; Johnson–So: high O/S, especially put-heavy, is bearish); (d) **IV term-structure slope** around known events. You compute IV daily from your bars (validated to ~0.7 vol-pt near-ATM — adequate for *cross-sectional ranks*, which is all you need); you have contract volume natively.

**Why it could be real alpha, not beta/premium.** These are information-flow signals (where informed traders act first), not risk compensation — they hedge clean in the literature and decay over weeks, matching your swing horizon. Crucially, **execution is in equities**: you harvest options *information* at equity *costs*, completely sidestepping the spread wall that killed every options-execution sleeve.

**How to test.** Your dollar-neutral L/S harness from the ranker post-mortem is the exact right machine and it's already built (net-sector cap, beta overlay, net-exposure capture). Long top decile / short bottom decile of CPIV (start there — strongest at your horizon), weekly rebalance, residual-α with the multi-factor hedge from §1.2(d). Liquidity-filter the IV inputs hard (near-ATM, day-volume ≥ 10, vega-weighted average across the two nearest tenors) because computed-IV noise on stale closes is your main measurement risk. Caveat to respect: your closed conclusion was "no alpha in cross-sectional ML on *price/fundamental* features" — options-flow features are a genuinely different information set, so this is not reopening a dead line; but if the simple decile sorts show nothing after costs, don't escalate to ML, just close it.

### #3 — Scale what already works: trend, broadened (the boring, capital-grade move)

Your TSMOM sleeve is the most statistically defensible asset you own (19 years, +0.71, crisis-convex). It is also under-built: 10 ETFs, one lookback, weekly Monday. Expansion is low-risk, high-certainty EV: multiple lookbacks blended (3/6/12-month), vol-targeted position sizing rather than inverse-vol weights alone, broader sleeve (add currency-hedged intl, more commodity legs, maybe managed-futures-style long-short on the ETF set rather than long-flat). Each addition is testable on 19+ years — the one place your significance machinery actually has the power to work. A Sharpe 0.7→0.9 improvement on your *largest live allocation* is worth more dollars than any new sleeve passing PAPER.

### #4 — Book-level acceptance + small index-VRP diversifier (finish your own OPT-6 thought)

You proved index VRP is real and cost-robust (PF 1.75 at 2× spread) and crisis-negative; trend is crisis-positive. That is a textbook pairing. Build the **book-level gate you already specced**: does adding a small (5–10% risk budget), VIX-regime-gated short-vol sleeve improve the combined book's Sharpe/Calmar/worst-regime on the joint backtest? Acceptance is a *book delta* with a registered criterion (e.g., book Calmar +20% with worst-regime not degraded), not standalone significance. Guardrails: the regime overlay must be your *existing* regime model (no new fitted parameters — that's how you avoid the thin-sample overfit trap you correctly fear), defined-risk only, sized so a 2008-style vol event costs ≤2% of NAV at the wings. One build, one verdict, banked either way.

### #5 — Earnings long-vol selectivity (only if #1's panel hands it to you)

Your OPT-3 arc ended at "single-name earnings VRP ≈ too thin to short, net of costs." The mirror question — *are there identifiable events where implied systematically underprices the move* (low pre-event IV run-up + high revision dispersion + small implied move vs. historical move) — is answerable for free inside the #1 panel (the label is just realized/implied > 1 profitability of a long straddle at modeled costs). If a selectable sub-population shows gross edge ≥ 2× the modeled spread, then and only then build the strategy. Do not build it speculatively; let the panel screen it.

**Explicitly NOT recommended:** dispersion as a trade (your two VRP findings are its two halves, but both legs pay single-name spreads — the cost wall again; express the idea as the panel's relative-VRP *feature* instead); dealer-gamma proxies (without OI you'd proxy positioning from volume — too noisy to distinguish from #2c, which captures the tradable part anyway); more short-vol parameterizations; small/mid PEAD re-runs; anything intraday.

---

## 3. Best alpha-shaped uses of the options data specifically

Condensed, in EV order, all feasible within your hard limits (daily OHLCV + computed greeks + universe; no OI/NBBO/IV history):

1. **Event-conditioning features for PEAD** (implied move, IV run-up, reaction ratio, post-event IV retention) — §2.1. Your data's killer app.
2. **Cross-sectional equity signals**: CPIV, skew, O/S volume ratio, term-structure slope — §2.2. Execution in equities.
3. **Book-level regime/risk inputs**: an aggregate put-skew / index-IV-vs-realized spread as a *feature into the regime model* (it retrains weekly and already gates sizing — adding 2–3 options-derived macro features is cheap and is the proper home for your "put-skew risk-off" idea; judged by the regime model's own F1/log-loss gate, no new framework needed).
4. **VRP-as-feature, not VRP-as-sleeve**: per-name computed IV minus forecast realized vol as a cross-sectional column in #2's sorts and #1's panel.
5. **Index VRP as a small book diversifier** — §2.4. The only options-*execution* use I'd endorse, and only behind the book-level gate.

What your data **cannot** support (don't burn cycles): true dealer-gamma (needs OI), 0DTE anything (needs intraday), quote-based execution alpha (needs NBBO), vol-surface arbitrage (needs a real surface history — your computed IVs are fine for ranks, not for surface-relative-value).

---

## 4. Architecture / design gaps

**Under-built:**
- **An event-research data layer** (§2.1). Right now each event experiment re-fetches and re-plumbs. The panel table is the missing platform piece — everything else in this review gets cheaper once it exists.
- **Gate calibration harness** (§1.3d): positive/negative controls runnable as one command, re-run after any gate change. You version-control gate *code* meticulously; you've never measured gate *behavior*.
- **Spread-model calibration job** (§1.2e): nightly snapshot-NBBO logger → bucketed half-spread table. Tiny build, anchors every options verdict past and future.
- **Multi-factor residual-alpha** (§1.2d).
- **Live slippage reconciliation as a first-class metric**: you have trackers for Sharpe-vs-expectation; add realized-fill-vs-assumed-cost per trade (entry slippage vs the 3bps assumption, marketable-limit crossing cost). This number is the bridge every paper→live decision will stand on.
- **A borrow/locate model** if the L/S equity line (#2) shows life — bottom-decile R1K shorts at a flat 5%/yr assumption is optimistic for the names a skew/O-S signal will pick.

**Over-built / actively costing you:**
- **Standalone-sleeve significance as the universal acceptance framework** — the thesis of this review.
- **Capacity gates at your capital scale.** Volume/notional caps for a modest single-operator book are solving a problem you won't have for years. Reallocate that rigor to cost realism, which you will have on day one.
- **The infra-to-research ratio.** The changelog shows weeks of world-class engineering (test-mode detection, log isolation, kill-switch coercion) per unit of new market knowledge. All real fixes — but you are one person, and the binding constraint on this project is *research throughput*, not code quality. Concrete suggestion: budget explicitly, e.g. 70% research / 30% infra per week, and let the backlog enforce it. The system is already more reliable than most prop-shop internal tooling; the marginal hour of hardening is worth far less than the marginal hour of panel-building.
- **Three frozen ML training paths** (swing/intraday/per-fold machinery) kept warm for dead strategies. Archive aggressively; the harness is the asset, not the strategies it killed.

**Structural fragility:** the whole edifice keys on Polygon. You've correctly made the local parquet store the durable asset. Add one more durable layer: persist the *computed* IV/greeks per contract-day as you generate them (a one-time engine pass over the 112M bars), so the engine becomes optional for research reads and every panel query is a join, not a pricing run.

---

## 5. The first 5 things I'd change (prioritized), and the kill list

1. **Calibrate the gate with positive and negative controls** (§1.3d). One week. Until this is done, every KILL verdict in your logs has an unknown false-negative rate, and every future verdict inherits it. This is epistemically upstream of everything.
2. **Re-run PEAD significance at event level** (panel of hedged event returns, two-way day×firm clustered errors) (§1.3b). Days, not weeks — the data exists. This either graduates your only live alpha sleeve or demotes it; both outcomes are worth more than any new build.
3. **Build the earnings-event feature panel** (§2.1) and re-adjudicate the implied-move signal inside it as a continuous, pre-registered feature (§1.4). This is the next month of research, and it replaces the one-threshold-at-a-time workflow permanently.
4. **Adopt a two-track acceptance framework**: Track A (alpha: event panel significance + multi-factor residual-α + costs) for PEAD-family and the §2.2 signals; Track B (book-delta: registered improvement to combined book Sharpe/Calmar/worst-regime) for risk premia and diversifiers (trend extensions, index VRP, regime features). Write it into PIPELINE_ARCHITECTURE.md as §7.0-B. This dissolves the alpha-gate-vs-risk-premium contradiction you've been working around for a month.
5. **Fix the three flattering mechanics before any new options verdict**: stale-mark fraction + unsmoothing check (§1.2a), calibrated spread table from snapshot NBBO (§1.2e), and the entry-fill convention re-run (§1.2b).

**Kill outright:** further single-name short-vol parameterizations; small/mid-cap PEAD; intraday anything; dispersion-as-a-trade; binary threshold sweeps as a validation method; the paper-confirmation OR-path *as evidence of edge* (keep it as evidence of implementation fidelity only).

**Double down on:** the trend sleeve (broaden it — §2.3); the event-driven family via the panel; the PIT/survivorship data layer (it's your moat); the sacred holdout discipline; the Opus-adversarial-review habit (it has a strong hit rate in your logs).

---

## 6. Is the premise misframed?

Partly, and you're closer to seeing it than your gate config admits. "Single operator chasing capital-grade *standalone* alpha at t≥2.5, n_folds≥10" is unreachable arithmetic on your data budget — not this year, not in five. That doesn't mean abandon alpha. It means restate the objective:

> **Run a diversified book of robust premia (trend, broadened; possibly a small regime-gated index-VRP sleeve) as the capital base, with one concentrated genuine-alpha effort (the earnings-event panel + options-conditioned PEAD) as the satellite — accepted at the book level, sized by live-tracked realized performance, with the sacred holdout and live reconciliation as the final arbiters instead of an unsatisfiable in-sample t-stat.**

That is how small professional books are actually run: nobody allocating to a 2-sleeve book demands each sleeve clear t=2.5 standalone; they demand honest construction, cost realism, regime awareness, and live tracking that matches the research — all of which you have already built, to a standard most funded teams don't reach. The irony of this system is that the engineering is institutional-grade while the acceptance statistics are calibrated for a data budget you'll never have. Fix the ruler; the workshop is excellent.

**What would change my mind:** if the positive controls (TSMOM-on-4y, textbook momentum) *pass* your current gate, my power critique is overstated and the gate is fine — run that test first, it adjudicates between us in a week. If the event-level PEAD panel comes back at p>0.15 even with day-level clustering, then PEAD is genuinely marginal and the right book is trend-plus-cash while the panel hunts. And if snapshot-calibrated spreads come back *wider* than your model, the cost wall is higher than assumed and even the §2.2 equity-side signals deserve an extra haircut.
