# MrTrader — Risk-Committee Panel Review (v11)

*Chair's read, candor over comfort. Solo systematic book, ~$100k paper, crossing from research to capital.*
*Scope: Theme B (is the edge real) → Theme A (go-live sizing/risk) → Theme C (diversification/tails).*

---

## Verdict in one breath

You have **one edge you can defend (trend), one factor that is probably real (carry), one factor that is selection-inflated (xsmom), and one sleeve I would bench (VRP).** The `futures_book` t = 2.29 is produced by *honest portfolio math applied to one contaminated input* — so the question is not "is combining a trick" (it isn't) but "does the basket survive once you deflate the xsmom input for the 6-factor search that produced it." The single largest *non-statistical* risk is operational: **you are about to run a two-broker, two-margin, leveraged book before the layer that can halt and reconcile it across both venues exists.** Fix the statistics first, refuse the second-venue capital until the kill-switch exists, and reformulate the vol exposure so you stop concentrating the exact tail you claim to be diversifying.

---

## 0. What I think you've already gotten wrong (lead-with-this)

Five things, in order of how much they should change your next two weeks.

1. **You are conflating "OOS-by-construction" with "selection-corrected." They are orthogonal.**
   "Rules-based sleeves are OOS-by-construction" is a statement about *estimation* (no per-fold parameter retrain → no in-sample parameter leakage). It says **nothing** about *which rule you chose to run.* You picked the survivors of ~20 families. That selection burden is completely uncounted by the OOS-by-construction property, and your gate is silent on it. You half-know this (it's why file 04 exists) — but the framing in your docs still leans on OOS-by-construction as if it were partial protection against multiple testing. It is not. Estimation honesty and selection honesty are two different audits; you've passed the first and not started the second.

2. **The xsmom t = 1.60 is a max-of-6 statistic reported as if it were a single test.**
   You tested {xs-momentum, curve-momentum, value, skew, basis-momentum, CoT} and kept the one winner. A t = 1.60 that is the best of 6 has an effective one-sided p far worse than its face value (see §B1 for the arithmetic — roughly p ≈ 0.29 family-adjusted, *not* 0.055). The basket leans partly on this inflated input. Carry (t 1.76) is far more defensible; xsmom is the marginal case and the one the whole "second engine" claim actually hinges on.

3. **VRP is the least-diversifying, most-risk-on, most-fitted sleeve — and you're proposing to *add* it to a book whose central disease is "one risk-on bet."**
   0.46 corr-to-trend (highest of the four), explicitly short-crisis, single-sleeve t = 1.46 (weakest of the four), and a crash-survival that is a **manual event study against two known events (Feb-2018 + COVID)** rather than a held-out fold. If the gate threshold was chosen with those two outcomes in view — and it almost certainly was — then "survives Feb-2018 and COVID" is an in-sample statement. Adding VRP makes the single-bet critique *worse*, not better. (Full treatment in §C5. The fix is not "delete" — it's "don't run it naked.")

4. **The 7.7×-Kelly anchor is a dangerous mental frame and it's still in your docs.**
   Describing 50% gross as "deeply haircut" *relative to 7.7× Kelly* (file 03) is the single instinct I'd most want to extinguish. 7.7× Kelly is a function of in-sample Sharpe and is wildly non-robust; it should never appear in a sizing conversation except as a cautionary tale. Anchor on **realized vol target and drawdown budget**, full stop. Delete the Kelly multiple from the sizing vocabulary.

5. **A look-ahead bug already existed in your stack (the parked fundamentals scorer) — which means the *newer, less battle-tested futures pipeline* is exactly where the next one is hiding, and the carry number is the most exposed to it.**
   "The data behind every live position is clean" is reassuring for the equity pipeline that's been audited for two years. The carry/xsmom numbers run through a newer roll/expiry/stitching pipeline that has **not** had the same paranoid look-ahead audit. The three classic futures look-aheads (expiry-schedule knowledge, contract stitching, full-sample vol in the inverse-vol sizer) all inflate carry specifically. Audit those three before you trust 0.58 (see §B5).

Everything below elaborates these and answers your numbered questions.

---

# THEME B — Is the edge real, or multiple-testing residue? (answer first; it gates everything)

## B1 — Does the surviving book clear an honest family-wise / deflated bar? What bar should t = 2.29 clear?

**Short answer: it is in the danger zone. It clears the naive 1.96 bar, almost certainly fails blunt Bonferroni-20, and whether it clears a properly-calibrated (correlated) null-zoo bar is precisely the test you must run. My prior is roughly 50/50 — and the coin lands largely on whether *carry alone* survives, not the basket.**

The arithmetic you need:

- **Bonferroni-20 (too blunt, but a useful ceiling).** 20 families → one-sided 5% threshold = 0.0025 → z ≈ **2.81**. Against this, **t = 2.29 fails.** Bonferroni assumes independent trials, which is too conservative here (your trials share universes and many were genuine nulls killed cleanly), so this is an *upper bound* on the bar, not the bar itself.

- **The max-of-6 problem for xsmom.** For 6 independent factors all with true SR = 0, the expected max t over your sample is ~1.27 and the 95th percentile of the max is materially above a single-test 1.64. A face-value one-sided p for t = 1.60 is ≈ 0.055; family-adjusted across 6 (Šidák): 1 − (1−0.055)^6 ≈ **0.29.** Your factors aren't independent (shared universe lowers the true correction), so the honest number is *between* 0.055 and 0.29 — but it is **not** 0.055. xsmom is not individually distinguishable from "best of a 6-factor noise search" at face value.

- **The mitigant that genuinely helps you.** Multiple-testing penalties bite hardest on patterns with *no prior.* Carry and cross-sectional momentum in futures are among the most externally-replicated risk premia in the literature (Koijen-Moskowitz-Pedersen-Vrugt on carry; Moskowitz-Ooi-Pedersen and Asness-Moskowitz-Pedersen on momentum), with decades of out-of-sample confirmation and an economic rationale. Strong external priors + economic signs + cost-robustness + positive-in-every-subperiod legitimately *reduce* the effective penalty. **But** — and this cuts both ways — the factors you *killed* (value especially) also have strong academic pedigree. The fact that a strong-prior factor (value) died from sample noise is simultaneously *reassuring* (your gate isn't keeping everything with a prior) and a *warning* (if a strong-prior factor can die from noise, a marginal-prior survivor can live from noise).

**The bar I would require:**
- **Deflated Sharpe Ratio (Bailey & López de Prado) > 0.95 at N_eff = 20** on the `futures_book` residual-Sharpe, AND
- **xsmom must beat the 95th percentile of the *max-of-6* null distribution** from your own null zoo (§B3).
- **If only carry clears its single-factor null** (likely — strong prior, p 0.0001, every subperiod): size **carry-only** and drop/de-weight xsmom. You would still have a real second factor; you just wouldn't have a "two-factor engine."
- **If neither clears:** the second engine is residue; stay single-sleeve and put the saved effort into sizing/execution, where real money is actually being left on the table.

*Evidence that confirms/refutes:* run §B3's zoo + the DSR formula in §Appendix-1. If `futures_book` DSR ≥ 0.95 at N_eff = 20 → proceed with the basket. If carry-only clears but basket doesn't → carry-only book. Quantitative, buildable this week.

## B2 — Is "gate the basket" a legitimate significance gain or a multiple-testing trick?

**It is legitimate — and I can show you it's legitimate from your own numbers — *provided you did not search over combinations.* The combination math is honest; the contamination is entirely upstream in the xsmom input.**

The portfolio identity. For two equal-vol streams with individual t-stats t₁, t₂ and correlation ρ, the equal-weight basket t is approximately:

```
t_basket ≈ (t₁ + t₂) / sqrt(2·(1 + ρ))
```

Plug in your numbers: t₁ = 1.76 (carry), t₂ = 1.60 (xsmom), ρ ≈ 0 (you report near-orthogonal):

```
t_basket ≈ (1.76 + 1.60) / sqrt(2·1) = 3.36 / 1.414 ≈ 2.38
```

You observed **2.29.** The tiny shortfall (2.38 → 2.29) is exactly what a slightly-positive ρ and non-identical Sharpes produce. **Your t = 2.29 is fully explained as the mechanical consequence of diversifying two positive, low-correlation streams.** That is not a trick — it is the same math that makes any risk-parity book's t exceed its components'. Equal-weight is a *zero-free-parameter, ex-ante* combination; you did not optimize weights to cross 1.96.

**The one way it becomes a trick** — and the only thing you must verify: did you *search over combinations*? If you tried carry+xsmom, then carry+xsmom+curve-mom, then re-weighted until something crossed, and reported the winner, that's p-hacking the combination. The tell is your research log (§B4): if **equal-weight(carry, xsmom) was pre-registered as the obvious combination before you saw the basket t**, you're clean. If the basket was assembled *after* seeing that neither sleeve cleared alone, and other combinations were tried, it's contaminated.

**Net:** the combination is honest math; the xsmom *input* to it is selection-inflated. Deflate the input (§B3), re-combine, and the basket t will fall toward something you can trust. My estimate is it lands in the high-1s to low-2s — survivable if carry carries it, fragile if it depends on xsmom.

## B3 — The null-strategy zoo protocol (build-this-week deliverable)

**Goal:** calibrate the null distribution of your *end-to-end Track-B statistic* (residual-α t vs the trend book) for a rules-based cross-sectional futures factor with zero true edge, on *this* universe, over *this* sample — then place t = 2.29 against it, corrected for the search.

Three nested nulls. Each pushes a *zero-information signal* through the **entire real pipeline** — same 76-market universe, same weekly rebalance, same inverse-vol sizing, same cost model, **and crucially the same residual-α-vs-trend regression** (null the statistic you actually gate on, not the raw Sharpe — otherwise you miss that the trend-orthogonalization is itself a fitted step).

### Null A — Single-factor null ("is one factor's t inflated by the universe's structure?")
Generate a random signal with the factor's structure but no information. **Preferred construction for cross-sectional factors: cross-section permutation.** Each rebalance, randomly reassign the factor scores across the markets (permute which market gets which score). This preserves the marginal score distribution and the entire portfolio-construction machinery while breaking the score↔market↔future-return link.
- Alternatives, in increasing fidelity: sign-randomization of the long-short legs (preserves turnover/gross); block-permutation of the *signal time series* (preserves signal autocorrelation, breaks alignment with returns; block length ≈ signal horizon, ~52 weeks for a 12-1 analog).
- Run **M = 5,000** random factors. For each, compute the residual-α t. → the null distribution of single-factor residual-α t under "zero edge, real engine."
- **Single-factor bar:** the 95th percentile of this distribution. **If xsmom's real t = 1.60 is below it, xsmom is not distinguishable from noise even before the family correction.** Run carry through this too — it should clear comfortably.

### Null B — Family-wise / max-of-N null (the multiple-testing correction; this is Hansen's SPA in disguise)
You searched 6 factors and kept the best. The relevant null is therefore **max-over-6 random factors**, not one.
- From the M random factors, draw groups of 6, take the max-t within each group, build the distribution of **max-of-6 residual-α t.** Its 95th percentile is your family-wise bar.
- Because the random factors live on the same universe, they inherit the real correlation structure → this is **less conservative than Bonferroni-6** (which assumes independence) but more honest than no correction. This *is* an empirical White's Reality Check; using the SPA refinement (down-weight obviously-dead nulls) is strictly better here because several of your 6 were clearly dead — Reality Check would be over-conservative.
- **xsmom must beat the 95th percentile of max-of-6,** not the single-factor null.

### Null C — Whole-program null (the brutal one; bound it, don't pretend to nail it)
The true trial count isn't 6 — it's ~20 families across 2 years (and arguably more once you count within-family spec grids *and* every Track-B test you ran, since each Track-B test is itself a trial). You cannot null-replicate every family with different data/engines. So **bound it via N_eff in the Deflated Sharpe Ratio.** Estimate the effective number of *independent* trials (your real trials are correlated — all futures factors share a universe — so N_eff < 20 raw); a defensible N_eff ∈ [10, 30]. Apply the DSR (Appendix-1) to the `futures_book` residual-Sharpe at N_eff = 20 as the headline, with a sensitivity band [10, 30].

**Report three numbers:**
1. xsmom's percentile vs the **single-factor** null (Null A) — is the marginal factor even individually real?
2. xsmom's percentile vs the **max-of-6** null (Null B / SPA) — does the survivor beat its search?
3. `futures_book` **DSR at N_eff = 20** (Null C / Bailey-LdP) — does the basket survive the whole program?

**Cost:** M = 5,000 end-to-end null runs on EOD weekly data is cheap (hours, not days). The only non-trivial engineering is making the null run the *full* pipeline including the residual-α regression.

## B4 — Prospectively logging research degrees-of-freedom

Make the family-wise count a *live, queryable number* so the bar auto-rises as you search.

- **Append-only trial registry** (git-tracked YAML or a Postgres table). One row per: family, spec variant, parameter grid searched, reviewer-suggested spec, bug-fix rerun, post-hoc exclusion.
- **Fields:** `trial_id, date, family, hypothesis (pre-registered sign/direction), universe, data_used, n_configs_searched (within-family grid size), outcome (pass/kill), counts_toward_FWER (bool)`.
- **Pre-register before the test.** Write hypothesis + sign + universe + acceptance bar to the registry *before* running the backtest. You already do "pre-registered sign, no sign-flipping" for the futures factors — that discipline (killing value/skew/CoT at their declared sign) is genuinely excellent and is the strongest evidence in the whole pack that your gate has teeth. Formalize it into the registry so it's enforced, not cultural.
- **Every new "significant" result is reported *with* the running N**, and the DSR/family bar is recomputed each time. The bar rises automatically as the program ages — which is correct.

## B5 — The look-ahead audit the futures pipeline still needs (not in your file; do it)

The carry number is the second engine's keystone, and it runs through the *newest* pipeline. The three classic futures look-aheads, each of which **inflates carry specifically**:

1. **Expiry-schedule knowledge.** Your carry signal uses "scheduled expiry from the contract code." Confirm Δt and the front/next selection use **only point-in-time-known** expiry, never actual realized expiry or any future-dated calendar. A subtle off-by-one in roll timing flatters carry.
2. **Contract stitching.** Confirm no future contract's price leaks into the front-month series (back-adjustment done with only past data; the "next" contract chosen by PIT rules).
3. **Inverse-vol sizing.** Confirm the vol estimate feeding the inverse-vol weights is **PIT rolling**, not full-sample. You learned this lesson on the credit overlay (the in-sample vol-match was an artifact); verify it didn't survive anywhere in the futures sizer.

*Evidence:* re-run carry with each of these deliberately *broken* and confirm the Sharpe *moves* (validates your harness is sensitive to them), then confirm the production path uses the clean version. If carry's 0.58 is robust to all three, the keystone holds.

---

# THEME A — Go-live: sizing & risk architecture (the unbuilt layer)

*All of this is conditional on Theme B. If the basket is residue, you build this layer for trend + carry-only, not for a four-sleeve book. Build the null zoo first.*

## A1 — Sizing (per-sleeve AND book-level)

**Target.** Anchor on **drawdown tolerance**, then derive vol. For a solo ~$100k book where this is serious-but-not-livelihood capital: **target ~10–12% annualized realized book vol**, implying a ~95% 1-yr max-DD in the **18–25%** range and a fat-tailed plausible worst case near **30%.** If a 30% peak-to-trough would make you capitulate (turn the system off at the worst moment), target **8%** instead. The right vol target is the one whose *implied worst drawdown you can sit through without intervening* — because the cost of capitulating at the bottom dwarfs the cost of running slightly under-levered.

**Combination rule — the comparison you asked for:**

| Rule | With 3–4 sleeves | Verdict |
|---|---|---|
| **Inverse-vol** | Ignores correlations; over-weights correlated sleeves | Too naïve once VRP (0.46 to trend) is in |
| **Equal-risk-contribution (ERC / risk parity)** | Transparent on a 3×3 covariance you can reason about directly; accounts for correlation | **The honest default** |
| **HRP** | Hierarchical clustering shines with *many* assets + unstable covariance; here it adds opacity over a matrix you can read by eye | **Does not earn its complexity** — skip |

**But ERC alone is wrong here**, because it would hand the *unproven paper sleeves the same risk budget as the live-proven trend sleeve.* The fix: **bucketed ERC with a probationary cap.**
- **Proven bucket** (trend): full ERC weight.
- **Probationary bucket** (futures_book, VRP): share a **capped total risk budget** — I'd cap the probationary bucket at **≤ 30–40% of total book risk** until live evidence accrues — and within the bucket, inverse-vol or ERC.
- This protects the one live edge from being diluted by sleeves that might be multiple-testing residue.

**Fractional entry that ramps with live evidence** (yes, do this):
- Paper-pass, no live track → **0% capital** (paper only).
- Passes IBKR-paper gate (Rung 1) → enter at **~25%** of ERC-implied weight.
- 3–6 mo live + ≥1 vol spike survived + tracking-error in bound → **50%.**
- 12 mo live + consistency check passed → **full** ERC weight.
- The ramp is **evidence-gated, not calendar-gated** — calendar is necessary, not sufficient.

**Margin-to-equity ceiling.** Independent of the vol target, cap **maintenance margin ≤ 20–25% of NAV** for a solo book. This is *liquidity/gap* protection, not vol protection: it guarantees an overnight futures gap-through-your-vol-estimate can't force a liquidation you can't meet (you have no treasury desk to wire variation margin at 3am). On a 10–12% vol diversified futures book the margin cap will rarely bind (the vol target binds first) — it's the backstop for the case where vol is underestimated. **Always run the lower of {vol-target gross, margin-cap gross}.**

**Delete Kelly from the vocabulary.** (See §0.4.)

## A2 — Cross-venue risk aggregation (Alpaca + IBKR)

**Minimal-correct unified risk surface (v1):** one nightly job pulling positions/cash/margin from both brokers, mapping every position to a **common factor-exposure vector** plus book aggregates:
- **One stitched equity curve** (combined daily P&L across both brokers → the series you compute realized vol and drawdown from). This is non-negotiable: you cannot risk-manage two equity curves.
- **Aggregate gross / net.**
- **Factor-exposure vector**: net equity-beta-$, net rates-DV01, net commodity exposure, net USD, net short-vol vega. *This is the single most important output* — it's what tells you whether your "diversified" book is secretly all long-equity-beta.
- **Margin-to-equity** per broker and aggregate.

That's the whole v1. You do not need a full commercial risk model; you need those four things rebuilt from broker truth each cycle.

**Single kill-switch governing both brokers:**
- A **broker-agnostic `flatten()` interface** with two implementations (Alpaca, IBKR) behind one orchestrator command. A global `HALT` flag in Redis that (a) blocks all new orders on both venues and (b) optionally fires flatten on both.
- Because the book is **weekly + EOD**, you do not need microsecond kill — flatten-at-next-open is acceptable. **But you need a dead-man's switch:** if the orchestrator process dies, positions must be governed by **resting protective orders at the broker** so a crash doesn't leave leveraged futures naked overnight. IBKR supports this materially better than Alpaca — use it.
- The kill-switch must be **idempotent** and **reconcile against broker truth** after firing (re-query both, confirm flat; never trust the DB).

**Reconciliation ("the DB is not reality"):** broker is always source-of-truth for positions/cash/margin; the DB is a cache. Each cycle: pull broker state, diff vs DB, on mismatch **broker wins + log + alert.** Extend your existing IBKR-spec reconciliation to a *cross-broker* reconciliation — the unified surface is rebuilt from broker truth, never from accumulated DB state.

## A3 — Forward-looking book risk

**Realized-correlation-spike de-gross trigger:**
- Compute rolling pairwise realized correlation among live sleeve returns on a **short window (20–30 trading days)** vs a **252-day baseline.**
- **Best single metric: the absorption ratio** (Kritzman-Li) — the share of book variance explained by the first principal component. When it spikes, your sleeves are co-moving and effective diversification has collapsed into one factor. Cleaner than average pairwise correlation because it's one interpretable number.
- **Calibrate statistically, not to maximize backtest Sharpe** (this is how you avoid overfitting the trigger): de-gross when the short-window avg correlation exceeds its long-run mean by **≥ +2σ** of the rolling-correlation series, **or** the absorption ratio crosses its **90th historical percentile.** Standardized triggers aren't curve-fit to specific events.
- **Magnitude:** proportional de-gross (cut book gross 25–50% on trigger), restore **gradually with hysteresis** as correlation normalizes. Proportional + hysteresis avoids binary whipsaw.
- This is the highest value-per-effort item in Theme A: it's a *live, regime-reactive* protection that **does not require your backtest to have seen a crisis.**

**Global book-drawdown de-risk ladder** (on whole-book DD from high-water-mark):

| Book DD from HWM | Gross |
|---|---|
| −8% | 75% |
| −12% | 50% |
| −16% | 25% |
| −20% | **flat, re-evaluate (the no-go DD)** |

**Avoiding selling the bottom:** (1) measure DD on a *smoothed* equity curve, not tick-by-tick; (2) re-gross with hysteresis (require recovery past the *previous* breakpoint by a margin before re-adding); (3) consider *vol-adjusted* DD so a high-vol regime doesn't trip it spuriously. **Be honest: any DD stop will sometimes sell the bottom.** Its job is tail-survival, not Sharpe — size it as insurance and accept a modest Sharpe drag for the guarantee you survive to keep trading.

## A4 — The promotion ladder (explicit rungs, evidence thresholds, demotion stops)

**Rung 0 → 1 — Paper-PASS → IBKR-paper.** *Entry:* Track-A PAPER-PASS (you have it). *To earn capital, IBKR-paper must show:*
- **Real fills on real contracts** for **≥ 13 rebalances (1 quarter) minimum, ideally 26 (6 months).**
- **≥ 2 contract rolls per traded market**, with **modeled roll cost confirmed against actual roll slippage** (this is where carry lives or dies — the 3bps/side assumption must hold in real fills).
- **≥ 1 volatility spike** (VIX > 25 day or fast multi-σ move) with the sleeve behaving as designed (gates fire, sizing responds).
- **Slippage ≤ ~30% of modeled edge.** If slippage eats >30% of the ~0.58 net edge, the sleeve fails this rung.
- *Demotion:* tracking error vs backtest exceeds the intended-vs-actual bound, or roll slippage > 2× modeled → demote to "investigate."

**Rung 1 → 2 — IBKR-paper PASS → tiny live.** *How tiny:* **1–2 contracts/market**, whole tiny-live futures book **≤ 10% of risk budget** (fractional entry from A1). Real capital, real margin, real gap risk — small enough that a total loss is tuition. *Demotion stops:*
- Live-vs-paper-shadow daily return correlation **< 0.6** over the window (should be > 0.8) → structural problem, demote.
- Realized slippage **> 2× modeled.**
- Sleeve DD **> backtest maxDD × 1.5** within first N months (regime-break).

**Rung 2 → 3 — tiny live → scale.** *Entry:* **≥ 12 months tiny-live** with consistency. *Consistency check:* a one-sided test that **live SR is not significantly worse than backtest SR − margin** (don't require equality; require live isn't materially worse), plus live factor exposures match intended. *Avoiding the scale-into-mean-reversion trap:* scale **gradually** (≤ double once per 6 months) and **never scale on the back of a strong streak** — gate scaling on *track length + consistency*, never on *recent performance level.* A sleeve coming off its best-ever quarter should **not** be scaled (momentum-crash risk in the sleeve's own returns).

**The futures analog of your trend tracking instrument — a two-instrument decomposition:**
- Run a **paper-shadow** (modeled fills) in parallel with live (real fills) for each futures sleeve.
- **Instrument (a) — implementation tracking error:** live vs paper-shadow → isolates *execution/slippage.*
- **Instrument (b) — edge tracking error:** paper-shadow vs backtest-predicted → isolates *regime/alpha decay.*
- The decomposition tells you *why* a sleeve underperforms — plumbing or alpha — which determines whether you fix the adapter or kill the sleeve. Without it you can't tell a slippage problem from a dead edge.

## A5 — The no-go list (hard gates, stated as disqualifiers)

1. **If `futures_book` t = 2.29 doesn't clear the null zoo** (residual-α below the family-wise / DSR bar at N_eff ≥ 20): do **not** deploy a two-factor engine. Fall back to carry-only if carry clears its single-factor null; else stay single-sleeve.
2. **If the four premia tail-correlate above ~0.6** in worst-5%-equity-day conditional correlation (Theme C): do **not** run them as "diversified" at full size — treat the book as one risk-on bet and size gross accordingly.
3. **If IBKR-paper roll slippage > ~2× modeled** or eats **> 30% of carry's edge:** do **not** promote carry to capital until execution is fixed.
4. **If margin-to-equity would exceed 25%** at target sizing: do **not** deploy at that size; cut gross.
5. **If the unified cross-venue risk surface + single kill-switch + dead-man's switch aren't built and tested:** do **not** put real capital on the second venue. **You do not deploy a two-broker book you can't halt with one action.** Hardest gate; enforce first; it's infrastructure, not statistics.
6. **If VRP's crash-survival can't be reconfirmed out-of-sample** (gate calibrated *without* Feb-2018 + COVID, then tested on them): keep VRP **paper-only.**
7. **If you can't produce the book's behavior in a 2008-style slow bear AND a Feb-2018-style fast spike from a *held-out stress fold* (not a manual event study):** do **not** scale beyond tiny-live.

---

# THEME C — Do the four premia diversify, or co-crash?

## C1 — Genuinely multi-bet, or leveraged long-risk-premium with extra steps?

**Honest read: closer to "leveraged long-risk-premium with extra steps" than to genuine diversification — and the one saving grace is trend, not the additions.**

The composition, characterized by crisis behavior:

| Sleeve | Crisis character | Diversifying? |
|---|---|---|
| **Trend (TSMOM)** | Convex to *slow* bears (can go short/flat risk — made money in 2008); **concave to *fast* shocks** (whipsaws Feb-2018, COVID-crash week) | The genuine crisis-alpha leg — *for slow crises only* |
| **Carry** | Risk-on (you're paid for bearing crash risk); broad, not only energy (ex-energy 0.54 ≈ full 0.58) | No — risk-on |
| **xsmom** | Most neutral (0.12 to trend, long-short cross-section) but **subject to momentum crashes** in sharp reversals (Daniel-Moskowitz) | Partially — the best of the additions |
| **VRP** | **Explicitly short-vol = short-crisis** | No — *anti*-diversifying for tails |

**Net: short fast-crisis, with a slow-crisis hedge from trend.** That is the honest one-line characterization of the book.

**The tell** — the single number that separates genuine diversification from a hidden one bet: **the book's equity beta conditional on stress.** Regress book returns on SPX in the *worst-decile-SPX days* vs the rest. If **down-market beta > up-market beta**, you have negative convexity (the thing that kills you) and it's a long-risk bet in disguise. If down-beta *falls or goes negative* (the trend leg decoupling and kicking in), you have genuine convexity. Watch down-capture vs up-capture.

## C2 — The stress-conditional test I'd trust, and the "one bet" threshold

**Primary — exceedance (tail-conditional) correlation** (Longin-Solnik / Ang-Chen):
- For each pair, z-score the returns, then compute correlation conditional on **both** below their q-quantile: ρ⁻(q) = corr(rᵢ, rⱼ | rᵢ < Qᵢ(q), rⱼ < Qⱼ(q)) for q = 10%, 5%. Compare to upper-tail ρ⁺(q) and unconditional ρ.
- **Diagnostic:** plot the exceedance-correlation curve (ρ vs q, both tails). A genuinely diversified book has **flat-or-declining lower-tail** exceedance correlation; a hidden one-bet has the classic **rising lower-tail smile** ("correlations → 1 in a crisis").
- **More important than pairwise:** the **book-level conditional equity beta** from C1.

**Rigorous backup — copula lower-tail dependence λ_L:** fit a tail-flexible copula (t-copula, or Clayton for lower tail) and estimate λ_L = lim_{q→0} P(rᵢ < Qᵢ(q) | rⱼ < Qⱼ(q)). λ_L > 0 = asymptotic tail dependence (they crash together even in the extreme); λ_L ≈ 0 = tail-independent. Use it to confirm the exceedance finding.

**The "one bet" line:**
- worst-5%-day **average pairwise correlation > ~0.6** (vs ~0.25 unconditional), **OR**
- book **down-market beta exceeds up-market beta by > ~0.3** (material negative convexity).

**Power caveat (take seriously):** ~19y daily → ~240 worst-5% days, ~48 worst-1%, and the *joint* worst days are fewer — noisy tail estimates. Mitigations: (1) **block-bootstrap CIs** on the exceedance correlations — don't trust a point estimate; (2) better, measure co-movement **within the named crisis windows** (2008, Aug-2011, Aug-2015, Feb-2018, Q4-2018, Mar-2020, 2022) specifically — *event-conditional*, since quantile-conditioning on daily returns also picks up idiosyncratic bad days that aren't systemic.

## C3 — Stress-testing a book with no isolated crisis fold

Four approaches, in order of value:

1. **Historical scenario replay (do first — cheapest, most interpretable).** Run **all four sleeves jointly** through each named crisis window → the **book's** P&L path (not per-sleeve event studies — the *joint* book path, which is what you're actually missing). Output: book maxDD + recovery per crisis, and **which sleeves offset vs compounded.** In-sample, but honest and joint.

2. **Crisis-isolated / leave-one-crisis-out CPCV fold (the held-out version you're missing).** For each crisis, **re-derive every data-driven parameter** (vol estimates, **gate thresholds**, correlation-based weights) using *only pre-crisis data*, then evaluate the book through the held-out crisis. For rules-based sleeves with no retrain, this specifically tests whether the **sizing/gating generalizes** — i.e., it directly attacks the VRP-gate concern: *if VRP's gate, calibrated only on pre-Feb-2018 data, still protects through Feb-2018, it generalizes; if it only works with COVID in-sample, this exposes it.*

3. **Block-bootstrap with crisis-block oversampling (for the *distribution* of outcomes).** Block-bootstrap the **joint** sleeve return *vectors* (resample date-blocks across all sleeves together → preserves contemporaneous cross-sleeve correlation, including whatever tail correlation exists), with crisis blocks. Output: bootstrap distribution of book max-DD; the **95th/99th-percentile DD is your "plausible worst case" for sizing** (feeds A1). Resampling the joint vector is the correct way to propagate co-crash risk into the drawdown distribution.

4. **Synthetic stress (most aggressive, optional).** Hand-build adverse scenarios from the term structure + VIX curve you own and shock every sleeve through its factor sensitivities — e.g., "VIX 15 → 40 overnight, backwardation, equity −10%, oil −20%, bonds rally," *and* a stagflation scenario (bonds + equities both fall, which 2022 only partially represents) to test a regime not in the history.

## C4 — A convex/defensive sleeve worth building? (and how to prove it pays in the tail, not just bleeds)

Yes — given the net-short-crisis tilt, a convex leg is worth investigating, under a hard discipline: **a convex sleeve must be proven to pay in the tail, not just bleed carry.** Ranked:

1. **Enhance trend's existing bond/gold legs (best risk-reward — you already own most of it).** Flight-to-quality convexity is *already partially in the book* via trend's TLT/IEF/GLD legs. **Caveat:** 2022 showed bonds + equities can fall together — a *static* bond hedge is regime-dependent (works in deflationary crises 2008/COVID, fails in inflationary 2022). A *trend-based* bond/gold leg (which can go short bonds) adapts and is far more robust than a static long. **Verdict: improve the trend rates/gold legs rather than bolt on a static bond hedge.** Free, uses data you own.

2. **Conditional long-vol — and pair it with VRP as ONE vol sleeve (this is the elegant fix to C5).** The mirror of VRP: long the front VIX future when vol is *cheap* (VIX bottom quartile + low realized vol + steep contango), flat otherwise. Structurally long-convexity — pays in crashes. It **bleeds carry** by construction (it's paying exactly the roll-down premium VRP harvests). **Run VRP-when-contango and long-vol-when-cheap as two sides of one vol sleeve** — VRP collects the premium in calm times, the long-vol leg pays off in the crash the VRP leg is exposed to. This **internalizes the VRP tail risk instead of leaving it naked.**
   - **The test that it pays rather than bleeds:** *do not* judge it on its standalone Sharpe (which will be near-zero or negative — expected). Judge it on the **book's CVaR / max-DD improvement per unit of carry-bleed.** Concretely: does a 5% allocation cut the book's COVID/2008 drawdown by enough to justify the Sharpe it costs in calm years? If 5% conditional-long-vol takes the book's COVID DD from −25% to −15% while costing ~0.05 of book Sharpe → good trade (fairly-priced tail insurance). If it costs 0.2 Sharpe to shave 3% off the tail → overpriced, don't.

3. **Standalone defensive curve trade — skip for now.** More complex, less clearly convex, and you already get most rate exposure from trend.

## C5 — Does VRP belong in the book?

**My answer, and the place I'd most override your instinct: VRP does not belong as currently constituted — not at full size, not naked, not standalone.**

Reasons (all already visible in your own numbers):
- **Highest corr-to-trend (0.46)** — the *least* diversifying of the four. Adding the least-diversifying, most-risk-on sleeve to a book whose central critique is "one risk-on bet" makes the disease worse.
- **Explicitly short-crisis** — it concentrates exactly the tail the book most needs to shed.
- **Crash-survival is a fitted event study, not a held-out result** — the gate was (almost certainly) calibrated with Feb-2018 + COVID in view; "survives" them is in-sample.
- **Lowest conviction (t = 1.46, weakest of four) + most selection bites** (VRP was tried, parked, re-tried with a different construction — multiple bites at the apple).

**But VRP isn't worthless** — the VIX-contango roll-down premium is real and documented. **The fix is not "delete," it's "don't run it naked":** reformulate as part of an integrated vol sleeve (C4-#2) so the vol exposure is **net tail-defensive or at worst tail-neutral**, harvesting contango in calm times via the short leg while owning crash protection via the long-when-cheap leg.

**Concrete recommendation:** VRP stays **paper-only** until (a) its gate is reconfirmed leave-COVID-out / leave-Feb2018-out (does a gate calibrated *without* those events still protect?), **and** (b) it's reformulated as an integrated vol sleeve whose CVaR-per-bleed test shows it *reduces* book tail risk. Until both, VRP is a tail-risk concentrator wearing a diversifier's clothing.

---

# The closer — if I were chairing your risk committee Monday morning

**The single thing I'd do FIRST:** build and run the **null-strategy zoo + Deflated Sharpe Ratio at N_eff = 20** on the `futures_book` (§B3, Appendix-1) — *before one line of IBKR sizing code.* Everything in Theme A is conditional on the basket being real. One week of work that gates the entire program. *Decision rule:* DSR ≥ 0.95 at N_eff = 20 → proceed carry+xsmom; only carry clears its single-factor null → carry-only book; neither clears → second engine is residue, stay single-sleeve and reinvest the effort in sizing/execution.

**The single thing I'd REFUSE to let you do:** **deploy real capital on IBKR before the unified cross-venue risk surface + single kill-switch + dead-man's switch exist and are tested.** Not one contract, not "just to get real fills." A solo operator's largest operational risk is a process death or broker desync leaving leveraged futures ungoverned overnight — and **no statistical edge survives an uncontrolled blow-up.** Infrastructure gate before any second-venue capital, always.

**The ordered 5 (run #1 and #2 in parallel this week):**
1. **Null-zoo + DSR on the `futures_book`** — gates everything. [B3]
2. **Tail diagnostic** — exceedance correlation + book down-market beta + joint crisis scenario replay → decide diversified-vs-one-bet and whether VRP stays. [C2/C3]
3. **Build the cross-venue risk surface + single kill-switch + dead-man's switch** — the infrastructure no-go gate, before any IBKR capital. [A2]
4. **Codify sizing** = bucketed-ERC-with-probationary-cap + margin cap + fractional paper entry; write the promotion ladder with explicit demotion stops. [A1/A4]
5. **Reformulate the vol exposure** — pair VRP with conditional-long-vol as one tail-defensive sleeve, or shelve VRP to paper. Don't run it naked. [C4/C5]

**And one strategic steer you already know but should hear from the committee:** *stop searching, start sizing.* The free factor zoo is exhausted *at the single-factor level* — but conditional/interaction factors (carry × regime, momentum × vol-state) are *more* degrees of freedom, not fewer, and would re-open the multiple-testing wound you're trying to close. The marginal factor hunt has hit diminishing returns and rising selection risk. The money left on the table now is in **sizing, two-venue risk, execution, and tail-defense** — not a 7th factor. Your own docs say "the next move is not a new strategy." Hold that line.

---

# Appendix 1 — Deflated Sharpe Ratio (drop-in formulas)

**Expected maximum Sharpe under the null** across N trials (Bailey & López de Prado 2014), the benchmark your observed SR must *beat*:

```
SR0 = E[max_n SR_n]
    ≈ sigma_SR * [ (1 - gamma) * Z_inv(1 - 1/N) + gamma * Z_inv(1 - 1/(N*e)) ]
```
- `sigma_SR` = std-dev of the Sharpe estimates **across your N trials** (cross-trial dispersion)
- `gamma`    = Euler-Mascheroni ≈ 0.5772
- `Z_inv`    = inverse standard-normal CDF
- `e`        = 2.71828…
- `N`        = effective number of independent trials (use N_eff = 20; sensitivity band [10, 30])

**Deflated Sharpe Ratio** = probability the *true* SR > 0 given selection + non-normality + sample length:

```
DSR = Z( (SR_obs - SR0) * sqrt(T - 1)
         / sqrt( 1 - g3 * SR_obs + ((g4 - 1)/4) * SR_obs^2 ) )
```
- `SR_obs` = observed (selected/best) Sharpe — **per-observation units**, consistent with T
- `SR0`    = expected-max from above (the deflation)
- `T`      = number of return observations
- `g3`     = skewness of returns
- `g4`     = kurtosis of returns (non-excess; 3 for normal)
- `Z`      = standard-normal CDF

**Pass bar:** DSR > 0.95 (one-sided 5%); use 0.975 if you want margin. For the `futures_book`, apply to its residual-Sharpe; cross-check against the **empirical max-of-N null-zoo percentile** (§B3). If analytic DSR and the empirical zoo disagree, **trust the zoo** (it captures the real correlation structure your trials share).

# Appendix 2 — Null-zoo loop (pseudocode)

```
M = 5000
null_residual_t = []
for m in range(M):
    sig = permute_cross_section(real_factor_scores)   # zero-info, structure-preserving
    pnl = run_full_pipeline(sig)                       # SAME universe, rebalance,
                                                       # inverse-vol, cost model
    t   = residual_alpha_t(pnl, trend_book_returns)    # null the END-TO-END statistic
    null_residual_t.append(t)

single_factor_bar = quantile(null_residual_t, 0.95)

# family-wise (max-of-6) correction == empirical Hansen SPA:
maxes = [max(sample(null_residual_t, 6)) for _ in range(M)]
family_bar = quantile(maxes, 0.95)

# verdicts:
xsmom_real        = (xsmom_observed_t   > single_factor_bar)  # individually distinguishable?
xsmom_survives_fwer = (xsmom_observed_t > family_bar)         # beats its 6-factor search?
# then DSR(futures_book, N_eff=20) > 0.95  -> whole-program survival
```

# Appendix 3 — One-line evidence map (what confirms/refutes each major claim)

| Claim | Confirms | Refutes |
|---|---|---|
| Basket combination is honest math | t_basket ≈ (t1+t2)/√(2(1+ρ)) reproduces 2.29 (it does: 2.38 vs 2.29) | — (already confirmed from your numbers) |
| xsmom input is selection-inflated | xsmom_observed_t < family (max-of-6) bar | xsmom_observed_t > family bar |
| Second engine survives the program | futures_book DSR ≥ 0.95 at N_eff=20 | DSR < 0.95 |
| Carry is the robust factor | carry clears single-factor null comfortably; survives the §B5 look-ahead audit | carry Sharpe collapses under PIT-vol / clean roll dating |
| Book is one risk-on bet | down-market beta − up-market beta > 0.3; worst-5%-day avg corr > 0.6 | down-beta ≤ up-beta; tail corr stays near unconditional |
| VRP concentrates tail risk | leave-crisis-out gate fails to protect; raises book CVaR | leave-crisis-out gate still protects AND integrated vol sleeve lowers book CVaR per bleed |

---

*Be the skeptic on your risk committee — but credit where due: the pre-registered-sign discipline (killing value/skew/CoT at their declared sign without sign-flipping), the Type-I negative controls, and the willingness to de-rate carry for honest roll cost are the three things that make this program worth taking seriously rather than dismissing. The gaps are real; the rigor is real too.*
