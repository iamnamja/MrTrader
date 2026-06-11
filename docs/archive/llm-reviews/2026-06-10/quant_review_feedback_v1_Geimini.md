# External Quant Review: Brutally Honest Feedback

You've built a remarkably clean, disciplined, and rigorous engineering pipeline. Your handling of survivorship bias, point-in-time (PIT) data, embargoes, and out-of-sample enforcement is top-tier for a single-operator setup. You are asking the right questions and properly penalizing yourself for look-ahead bias and trading costs.

However, from a senior quant perspective, **you have built a beautiful, statistically rigorous machine specifically designed to reject perfectly good trades.** Your conceptual framing of "Alpha vs. Risk Premium" is flawed, your obsession with standalone CPCV on 4 years of data is a statistical trap, and your reliance on EOD options closes is dangerous. 

Here is my prioritized, unvarnished critique of your system and what you need to do next.

---

## 1. Verdict on the Validation Harness

Your pipeline is engineering-heavy but conceptually rigid. It is **hiding real edge** because you are evaluating strategies with the wrong ruler, and it is potentially **inflating edge** due to market microstructure realities you are missing.

### Where it is HIDING edge (Conceptual Flaws)
* **The "Pure Alpha" Fallacy & The Standalone Gate:** You killed the Index VRP (OPT-4) because its *standalone* Sharpe was ~0, calling it "just a risk premium." This is a rookie portfolio construction mistake. In the real world, capital-grade alpha is rarely a single magical signal with a 1.5 Sharpe. It is a portfolio of 5-10 orthogonal risk premia (VRP, Trend, Carry, PEAD) that yield a 1.5 Sharpe *together*. **You must gate on Marginal Contribution to Portfolio Sharpe, not standalone Sharpe.** If VRP is crisis-negative and TSMOM is crisis-positive, combining them creates a synthetic alpha. By killing VRP, you are throwing away the exact uncorrelated returns you need.
* **CPCV on 4 Years of Data is Theater:** Running CPCV C(8,2) on 2022-2026 data is not giving you statistical power; it is just resampling the exact same macro regime (post-ZIRP inflation fight). Your paths are heavily correlated. Stop using CPCV on the options dataset. For event-driven (PEAD) and short-horizon options, use event-clustered block-bootstrapping exclusively. 

### Where it is LEAKING/INFLATING edge (Simulation Flaws)
* **EOD Option Closes are Toxic:** You mark options to real EOD closes. In OPRA, the EOD close of an option is frequently stale (it might have traded at 2:00 PM while the underlying moved until 4:00 PM), or the bid/ask artificially widens at 3:59:59 PM. Marking to EOD closes creates massive artificial P&L volatility and false IV crushes. **Fix:** You must filter for staleness (time of last trade) and use a smoothed end-of-day mid, or rely purely on your engine's IV surface for marking.
* **Synthetic Stops on Daily Data:** Your `AgentSimulator` uses ATR stops. In reality, equity gaps over stops, and options gap *violently* over stops. Daily-bar simulators notoriously underestimate gap risk. If you are running stops on daily bars, your sim is flatter than reality.

---

## 2. The 3–5 Highest-Expected-Value Research Directions

Stop trying to find cross-sectional ML holy grails. You correctly identified that your edge is in the **Event-Driven / Structural** family. Double down there.

### 1. Implied Borrow & Corporate Actions (Alpha, not Beta)
* **Mechanism:** Use Put-Call Parity to extract the implied dividend/borrow rate from your options data. When the implied borrow rate diverges significantly from the general collateral rate, it signals high short interest, hard-to-borrow constraints, or impending corporate action.
* **Why it's real edge:** This is pure microstructure/flow-driven edge. Retail doesn't see it, and it isolates supply/demand shocks in the underlying equity.
* **How to test:** Rank equities by their option-implied borrow cost. Go long the top decile (short-squeeze candidates) holding for 3-5 days.

### 2. Earnings Dispersion Trading
* **Mechanism:** Instead of outright shorting single-name earnings vol (which you correctly proved dies to costs), trade the correlation. Go short index straddles (SPY/QQQ) and long constituent straddles heading into earnings season. 
* **Why it's real edge:** You've proven Index VRP is positive and Single-Name VRP is negative. By trading them against each other, you isolate the correlation risk premium. This explicitly hedges out the market risk and isolates the mispricing of constituent vs. index vol.

### 3. Analyst Revisions & Management Guidance (Event-Driven)
* **Mechanism:** PEAD is crowded. Drift following management guidance changes (extracted via NLP on filings or using standard revision data) decays much slower. 
* **Why it's real edge:** Institutions take weeks to re-allocate based on fundamental guidance shifts. 
* **How to test:** Treat it exactly like your PEAD adapter. Go long the highest upward revisions with a 10-20 day hold.

---

## 3. Best Alpha-Shaped Uses of the Options Data specifically

Given your hard limits (No historical OI, no intraday NBBO, daily OHLCV only), you cannot do dealer gamma positioning or intraday vol arb. You **must** use the options market as a predictive signal for the equity market.

* **Option-Implied Skew as a Regime Filter:** Calculate the 25-delta put minus 25-delta call IV. When single-name skew steepens drastically without a corresponding spot move, the smart money is buying tail protection. Use this as a **veto** on your PEAD longs.
* **Term Structure Contango/Backwardation:** If the front-month IV jumps above the second-month IV (backwardation) for a single name, it implies acute, immediate distress. This is a powerful filter for your trend and momentum sleeves.
* **The "Priced-In" Move:** Your OPT-5 finding (improving PEAD by checking realized vs implied move) is exactly the right path. You discarded it because it was "threshold fragile" at 1.0. **Do not discard it.** Instead of a hard 1.0 threshold, use the ratio `(Realized / Implied)` as a continuous feature in a simple logistic regression to size your PEAD conviction. 

---

## 4. Architecture / Design Gaps

* **Portfolio Allocation is Primitive:** Your allocator uses "static budgets" or basic regime tilts. You need to implement a true Mean-Variance or Risk-Parity optimizer at the portfolio layer. If PEAD has a low Sharpe but zero correlation to Trend, the optimizer will naturally size it correctly.
* **Options MTM Logic:** Your architecture marks options to their individual EOD close. You need to implement a synthetic mid-price curve fitter. If you don't have NBBO, marking off an illiquid OPRA close is mathematical suicide for your backtest metrics.
* **Over-reliance on T-Stats for Thin Data:** Your pipeline uses $t = 	ext{mean} / (	ext{std}/\sqrt{N})$. Financial returns are highly non-normal (fat-tailed). For low-N folds (like your 8-fold PEAD), the sample standard deviation heavily underestimates the true population standard deviation, leading to inflated t-stats. Use non-parametric bootstrapping for everything.

---

## 5. The First 5 Things I Would Change (Prioritized)

1.  **KILL the "Standalone Alpha" Promotion Gate:** Rewrite your gate logic to evaluate a strategy based on its **Marginal Contribution to Portfolio Sharpe**. Re-run Index Short-Vol (OPT-4) with a VIX-overlay *in combination* with TSMOM and PEAD. I guarantee the book Sharpe improves.
2.  **REVIVE the Implied-Move PEAD Filter (OPT-5):** Stop looking for a magic binary threshold (1.0 vs 1.25). Convert the implied/realized ratio into a continuous conviction multiplier for position sizing.
3.  **BUILD Option-Implied Borrow Cost Signals:** Extract the implied borrow rate from your options engine. Use it to build a short-squeeze event-driven sleeve. It is the most robust signal you can extract from daily EOD options data without OI.
4.  **FIX Options MTM Staleness:** Update your `OptionsSimulator` to heavily discount or interpolate EOD option closes if the option volume for that day was below a certain threshold.
5.  **KILL CPCV on 4-Year Datasets:** For your options data, turn off CPCV. Rely purely on expanding-window sequential walk-forward to preserve chronological regime continuity, combined with event-clustered bootstrapping to measure significance.
