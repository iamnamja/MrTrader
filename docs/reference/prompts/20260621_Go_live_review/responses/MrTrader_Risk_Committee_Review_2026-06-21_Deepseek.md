# MrTrader — Risk Committee Review (2026-06-21)

**From:** Your external risk chair / PM / CRO  
**To:** Solo operator  
**Subject:** Brutally honest read on the paper → live transition

Thank you for the comprehensive files. You have done an extraordinary amount of high-quality execution since the last panel. The research harness is now robust; the ruler is no longer leaky; you've killed more than you've kept; and you've found a credible second engine. That is the good news.

The bad news: **you are about to deploy into the very layer you have not built.** The research is the easy part. The hard part is sizing, combining, and governing it across two venues — and you have not yet stress-tested the multiple-testing burden that produced the t=2.29 result. I am not saying "don't deploy." I am saying **do not deploy until the following gates are closed.**

I will answer in priority order, as requested. Where I make a claim, I tell you what evidence would confirm or refute it.

---

## Theme B — Is the edge real, or multiple-testing residue? (Answer FIRST — it gates everything)

### B1. Does the surviving book survive an honest family-wise correction? What bar should t = 2.29 clear?

**Answer: No, not yet. You have not run the test.** The t = 2.29 is the Track-B residual-α of the equal-weight `futures_book` (carry + xsmom) against the live ETF-trend book. That is a conventional significance threshold, but it is uncorrected for the ~20 sleeve families you tried.

The null-strategy zoo must be built before you can answer whether 2.29 clears a deflated bar. My preliminary read: **the bar is likely higher than 1.96.** How much higher? We need to estimate the effective number of independent trials. My working assumption is that the effective family size is ~6–10 (because many of the 20 were conceptually related or data-mined from the same universe). If the effective family is 10, a Bonferroni-corrected one-sided bar would be ≈ 2.58 (since α = 0.05/10 = 0.005 → z ≈ 2.58). Hansen SPA may give a slightly lower bar if the null distribution is right-skewed, but I would not be surprised if the deflated bar is **≥ 2.3–2.5.**

**What evidence would confirm/refute this?** Run the null-strategy zoo (protocol below). If the 95th percentile of the null distribution of max(SPA statistic) is below 2.29, then 2.29 clears. If the 95th percentile is above 2.29, it does not.

### B2. Is "gate the basket" a legitimate significance gain or a multiple-testing trick?

**Answer: It is legitimate *if* the two factors were pre-registered as a combination before seeing the combined result.** You combined carry (t=1.76) and xsmom (t=1.60) into an equal-weight book, and the book's residual-α t hit 2.29. That is not a trick; it is the standard diversification benefit of combining two uncorrelated signals. The trick would be if you had tried many weighted combinations (0.1/0.9, 0.2/0.8, …) and picked the one that maximized t. You did not — you used equal-weight, which is a neutral, defensible choice.

**How to tell:** Run a sensitivity analysis. Vary the weight on carry from 0.1 to 0.9 in 0.1 increments. If the t-statistic remains above the deflated bar across a wide range, the result is robust. If it only peaks at 0.5, it is fragile.

### B3. Concrete null-strategy zoo protocol for rules-based cross-sectional futures factors

**Design:**

1. **Universe:** The same 76 futures contracts you used for carry and xsmom. Same data (survivorship-free, full term structure).
2. **Null strategies:** Generate **1,000 null signals** per family. Each null is a **block-permuted version of the original signal**, not a random Gaussian signal. Why block-permuted? Because it preserves the autocorrelation structure of the real signal (which matters for the HAC standard errors).
   - For carry: block-permute the *sign* of the carry signal (long-backwardated / short-contango) within blocks of length 2, 4, or 8 weeks (randomly chosen per permutation). This preserves the persistence of the term-structure slope.
   - For xsmom: block-permute the *rank* of the 12-1 momentum signal within blocks. This preserves the cross-sectional rank structure but breaks the time-series alignment.
3. **Statistic:** For each null signal, compute the **Track-B residual-α t-statistic** of the equal-weight combine (null_carry + null_xsmom) against the live ETF-trend book. Also compute the Track-A Paper-pass rate (point-SR ≥0.30 + HAC p<0.05).
4. **Deflation procedure (Hansen SPA / White's Reality Check):**
   - For each null, compute the SPA statistic (or Reality Check statistic) = max over all nulls of the t-statistic.
   - Compare the **observed** t=2.29 to the **95th percentile** of the null distribution of the SPA statistic.
   - If 2.29 > 95th percentile, the result survives family-wise correction. If not, it does not.
5. **How many nulls?** 1,000 is sufficient for a 5% false-positive rate (Monte Carlo standard error ≈ sqrt(0.05*0.95/1000) ≈ 0.7%).

**Implementation timeline:** 1–2 days to code the block-permutation engine and run the zoo.

### B4. How to prospectively log research degrees-of-freedom

**Protocol:**

1. **Pre-registration log:** Maintain a simple markdown file in the repo (`research_log.md`) with:
   - Date of the research campaign.
   - The hypothesis being tested.
   - The exact signal specification (e.g., "carry = (P_front − P_next)/P_next / Δt").
   - The number of variants tested (e.g., "we tried 3 carry definitions: front-minus-next, front-minus-third, and a weighted average of the two").
   - The number of universes tested (e.g., "we tried 76 markets, then ex-energy, then ex-commodity").
   - The number of weighting schemes tested (e.g., "we tried equal-weight, inverse-vol, and risk-parity").
2. **Rule of thumb:** Any variant that changes the sign, weight, or universe of a factor counts as a new research degree of freedom. You must count it.
3. **Review:** Before any new sleeve goes to the Ruler-v2 gate, you must update the log and run the null-strategy zoo with the cumulative family size.

---

## Theme A — Go-live: capital sizing & risk architecture (the unbuilt layer)

### A1. Sizing

**Target book volatility / drawdown budget:**
- For a solo ~$100k book, I recommend a **target annualized volatility of 10–12%** (≈ 1% per month). This is conservative enough to survive a 20% drawdown without panic, but high enough to generate meaningful returns (~6–8% expected if Sharpe ≈ 0.6–0.7).
- **Max drawdown budget:** 20% (i.e., if the book is down 20% from peak, you de-risk to 0.5× gross and re-evaluate). This is a hard stop.

**Per-sleeve risk weights:**
- With only 3–4 sleeves, **equal-risk contribution (ERC)** is the honest default. HRP (hierarchical risk parity) does not earn its complexity with only 4 assets; it is overkill.
- **ERC weights** are computed as the inverse of the sleeve's volatility (PIT vol), normalized to sum to 1. This ensures each sleeve contributes equally to the overall portfolio volatility (assuming zero correlation, which is a simplification but acceptable for the first pass).
- **Paper sleeves** (carry/xsmom/VRP) should enter at **half** the ERC weight of the live trend sleeve. Why? Because you do not yet know their true live volatility, slippage, and execution quality. After 3 months of live paper with tracking error < 0.5, they can graduate to full ERC weight.
- **Margin-to-equity ceiling:** Futures notional can be up to 5× equity (i.e., gross notional / equity ≤ 5). This is a hard cap independent of the vol target. If the vol target would require >5× notional, you reduce the vol target.

**Implementation:**
- Compute PIT vol for each sleeve (exponential weighting, 60-day half-life).
- Compute ERC weights.
- Apply a 0.5× multiplier to paper sleeves.
- Normalize so the book's total vol = 10–12%.
- Ensure gross notional ≤ 5× equity.

### A2. Cross-venue risk aggregation (Alpaca + IBKR)

**Minimal correct unified risk surface:**

1. **Daily reconciliation:** At the end of each day, fetch positions, cash, and margin from both brokers. Compute:
   - Gross exposure (sum of absolute notional values across both venues).
   - Net exposure (sum of signed notional values).
   - Portfolio volatility (using the PIT covariance matrix across all sleeves, computed on daily returns from the backtest + live paper track).
   - Drawdown from peak equity (combined equity across both brokers + cash).
   - Margin utilization (used margin / total equity).

2. **Single kill-switch:** The kill-switch is an app-level command that:
   - Sends a "cancel all orders" command to both brokers.
   - Sets a `halted` flag in the database.
   - Does **not** automatically flatten positions — the operator must manually confirm the flatten command. Why? Because flattening a futures book at market during a crisis can be worse than staying in. The kill-switch should halt new trades; the decision to flatten is a separate human judgment.

3. **Reconciliation discipline:** The database is **not** the source of truth — the brokers are. Every day, the system reads positions from the broker APIs and reconciles them against the database. Any discrepancy > 0.1% of NAV triggers an alert and halts the system until manually resolved.

### A3. Forward-looking book risk

**Realized-correlation-spike de-gross trigger:**
- Compute the **rolling 60-day realized correlation** between each pair of sleeves.
- If the **average pairwise correlation** exceeds its 90th percentile historical value (computed from the backtest + live paper track), reduce gross exposure by 20%.
- If the average pairwise correlation exceeds its 95th percentile, reduce gross by 50%.
- This is a rule of thumb; calibrate the thresholds on historical data to ensure the trigger fires only in true crisis regimes (e.g., 2008, COVID, 2022).

**Global drawdown-based de-risk ladder:**
- Drawdown from peak equity:
  - 5%: no action (normal).
  - 10%: reduce gross by 20%.
  - 15%: reduce gross by 50%.
  - 20%: reduce gross to 0 (halt).
- To avoid selling the bottom, the de-risk is **asymmetric**: you re-risk only after the drawdown recovers by 50% of the peak-to-trough move (e.g., if drawdown goes from 15% to 10%, you re-risk to 0.5× gross; if it goes to 5%, you re-risk to full gross).

### A4. Promotion ladder (paper → tiny-live → scale)

**Rung 0 → 1 (paper PASS → IBKR paper):**
- The sleeve must pass Track-A PAPER-PASS (SR≥0.30, HAC p<0.05).
- It must run on IBKR paper for **3 months** with:
  - At least **10 rebalances** (weekly = ~12 weeks).
  - At least **2 roll events** (for futures sleeves).
  - Tracking error vs backtest ≤ 0.2 (i.e., live paper returns deviate from backtest by < 20% of expected vol).
  - Slippage ≤ 10% of the expected edge (i.e., if the edge is 5% annualized, slippage must be < 0.5% annualized).

**Rung 1 → 2 (tiny-live):**
- After 3 months of IBKR paper with no violations, put **1–2 contracts** of real capital on the sleeve (≈ $5k–$10k notional).
- Explicit stop: if the sleeve's live Sharpe over a 3-month rolling window falls below 0.2, or if tracking error > 0.4, or if a single drawdown > 10%, demote back to paper.
- Duration: 6 months of tiny-live before scaling.

**Rung 2 → 3 (scale):**
- After 6 months of tiny-live, if:
  - Live Sharpe ≥ 0.5 (or the sleeve's backtest Sharpe minus 0.2).
  - Tracking error ≤ 0.3.
  - No drawdown > 10%.
- Then scale to full ERC weight (with the margin cap).
- **Live-vs-backtest consistency check:** Compute the ratio of live to backtest Sharpe over the tiny-live period. If the ratio < 0.6, do not scale.

### A5. The no-go list (hard gates)

I would **not deploy** if any of the following are true:

1. **The t=2.29 does not survive the null-strategy zoo** (Theme B). This is the first gate.
2. **The tail correlation (Theme C) among the four premia exceeds 0.7** in the worst 5% of equity days. If it does, the book is effectively one bet.
3. **IBKR paper slippage exceeds 20% of the expected edge** (as defined above).
4. **The book's expected max drawdown (from the historical backtest) exceeds 25%** (we already know it is ~20%).
5. **The margin utilization exceeds 5× equity** at any point during the paper run.
6. **The live paper tracking error exceeds 0.3** over any 3-month window.

If any of these trigger, keep the sleeve on paper indefinitely and re-evaluate quarterly.

---

## Theme C — Do the four premia actually diversify, or co-crash?

### C1. Genuinely multi-bet, or a leveraged long-risk-premium with extra steps?

**Answer: You are on the knife's edge.** Trend, carry, and xsmom are all "risk-on" premia to varying degrees. VRP is explicitly short-vol = short-crisis. The pairwise average correlations are low, but the tail correlation is what matters. Until you measure that, you cannot claim diversification.

**The tell:** If the tail correlation (worst 5% of equity days) among the four sleeves is > 0.7, then this is effectively a single bet. If it is < 0.5, it is diversified.

### C2. Stress-conditional correlation / tail-dependence test

**Specific test:**

1. **Define stress days:** The worst 5% of daily returns on the S&P 500 (or VIX > 95th percentile). Also define a "VIX-spike" window: days when VIX rises > 20% in a day.
2. **Compute pairwise correlation** among the four sleeves *on those stress days only*.
3. **Exceedance correlation:** Compute the correlation of returns conditional on both sleeves being below their 5th percentile (i.e., simultaneous losses).
4. **Threshold:** If the stress-conditional correlation > 0.7 for any pair, the book is not diversified. If the exceedance correlation > 0.6, it is a concern.

### C3. Stress-test a book with no isolated crisis fold

**Solution:** Run a **scenario replay** across all sleeves jointly. Use the historical data you have (2007–2026) and overlay known crisis periods (2008, 2011, 2015, 2018, 2020, 2022). Simulate the book's returns over those periods. If the book loses > 20% in any of those periods, it is too fragile.

**Alternative:** Use a **block-bootstrap** that resamples crisis windows (e.g., 3-month blocks) with replacement. This gives you a distribution of crisis-period performance without needing a clean hold-out fold.

### C4. Is there a convex/defensive sleeve worth building?

**Candidate: Long volatility when it is cheap.** You already have the VIX curve data. Consider a sleeve that buys VIX futures when the contango is steep (i.e., VIX > VIX3M by > 2 standard deviations) and sells when it normalizes. This would be a long-vol strategy — the opposite of your VRP sleeve.

**How to test it:** Run a backtest of a long-vol strategy (buy VIX futures when the curve is in steep contango, sell when it flattens). If it has positive carry and negative correlation to the book in crises, it is a genuine diversifier. If it bleeds carry, it is not worth it.

### C5. Does VRP belong in the book?

**Answer: Not at full weight, and maybe not at all.** The 0.46 correlation to trend is too high for a sleeve that is explicitly short-crisis. I would either:

- **Drop VRP entirely** and run a 3-sleeve book (trend + futures-book) — this is the cleaner, more defensible portfolio.
- **Pair VRP with a long-vol sleeve** (as above) so that the net short-crisis exposure is hedged.

Given your solo operator constraints, I recommend dropping VRP from the live book for now. You can revisit it after 6 months of live trend + futures-book.

---

## The closer — "If you were chairing my risk committee Monday morning"

**Top 5 concrete moves, in order:**

1. **Run the null-strategy zoo (Theme B)** before any real capital is deployed. This is the single most important gate. If t=2.29 does not survive, the futures book stays on paper.

2. **Build the cross-venue risk surface (Theme A2)** — daily reconciliation, single kill-switch, margin cap. Without this, you are flying blind.

3. **Drop VRP from the live book** and run a simpler 3-sleeve book (trend + carry + xsmom) for the first 6 months. This reduces complexity and tail risk.

4. **Implement the promotion ladder (Theme A4)** — 3 months IBKR paper, 6 months tiny-live, then scale. Do not shortcut this. The evidence threshold is explicit.

5. **Run the tail-dependence test (Theme C2)** and the scenario replay (Theme C3). If the stress-conditional correlation among the three sleeves exceeds 0.7, or if the book loses > 20% in any crisis scenario, do not deploy.

**The single thing I would do FIRST:** The null-strategy zoo (Theme B). It gates everything else.

**The single thing I would refuse to let you do:** Deploy real capital before the null-strategy zoo and the cross-venue risk surface are built and validated. That is the definition of flying blind.

---

## Final verdict

You have done the research. You have a credible second engine. But you are not yet ready to deploy. The missing layer is not alpha — it is risk governance. Build the gates I have outlined, run the tests, and then re-evaluate. I expect the t=2.29 to survive the null zoo, but I cannot say that with confidence until you run it. If it does, you have a real book. If it does not, you have learned something valuable at the cost of 3 days of coding.

Thank you for the candor of your question. I am honored to be on your risk committee.

— Your external risk chair