# RISK COMMITTEE VERDICT: MRTRADER GO-LIVE READINESS
**Date:** June 21, 2026
**To:** MrTrader (Solo PM)
**From:** CRO / Risk Committee Chair
**Status:** **CONDITIONAL HOLD ON FUTURES GO-LIVE**

You asked for candor over comfort. Here it is: **Your platform is engineering-rich but statistically vulnerable.** You have built a beautiful engine, but you are about to drive it over a cliff because your risk gauge is broken at the family level. 

The most urgent threat to your capital is not slippage or margin calls—it is the very high probability that your "validated second engine" (the $t=2.29$ futures book) is a multiple-testing artifact. If you tested 20 families, a $t=2.29$ is not a discovery; it is the expected maximum of noise. 

Here is the prioritized roadmap to save your capital and build a genuinely robust multi-premia book.

---

## THE CLOSER: IF I WERE CHAIRING YOUR RISK COMMITTEE MONDAY MORNING

If we were sitting in a room right now, here are the exact directives I would issue.

1.  **THE SINGLE THING I WOULD REFUSE TO LET YOU DO:** I will not authorize you to fund the IBKR account until the $t=2.29$ book passes the Deflated Sharpe/Null Zoo test (detailed in Theme B). "Gating the basket" when the components ($t=1.76$, $t=1.60$) fail the hurdle individually is a classic data-mining trap.
2.  **THE SINGLE THING YOU MUST DO FIRST:** Build the Null Strategy Zoo this week.
3.  **Halt VRP deployment:** Drop the VIX-curve VRP from the go-live plan for now. You are already net-short tail risk; adding a premium with a 0.46 correlation to your main engine that structurally bleeds in a liquidity crisis is adding fuel to the tail. 
4.  **Build the "Risk Daemon":** You cannot run two venues with an app-level kill switch. Build an independent process that strictly reads aggregate net liquidity and margin utilization across Alpaca and IBKR, with a physical `flatten_all()` kill switch.
5.  **Equal-Risk Allocation:** For the sizing of the surviving sleeves, default to Equal Volatility Contribution. Do not use HRP for 3 assets.

---

## THEME B: IS THE EDGE REAL, OR MULTIPLE-TESTING RESIDUE? (ANSWER: PROBABLY RESIDUE)

This is the gate. Everything else is secondary. 

### B1. The Mathematical Reality of $t = 2.29$
A $t$-stat of 2.29 corresponds to a two-sided $p$-value of $pprox 0.022$. If you only ever tested one strategy, that's significant. But you tested **~20 sleeve families**. 
The probability of observing at least one $p \leq 0.022$ by pure random chance across 20 independent trials is $1 - (1 - 0.022)^{20} pprox 35.9\%$. 

**Verdict:** A $t$-stat of 2.29 across a 20-family search space is statistically indistinguishable from noise. Applying Bonferroni, your required $lpha$ is $0.05 / 20 = 0.0025$, requiring a $t$-stat $\ge 3.02$. You are well short of the deflated significance bar.

### B2. Is "Gating the Basket" a Trick?
**Yes, it is a trick.** If you take two uncorrelated random streams with positive mean (even noise), combining them lowers the variance, mechanically pushing the Sharpe ratio and $t$-stat up. If the *individual* factors (carry at 1.76, xsmom at 1.60) did not clear the family-wise hurdle, combining them post-hoc to clear 1.96 is data dredging. The basket only counts as a single hypothesis if "Equal Weight Carry + XSMOM" was pre-registered *before* you ran the backtest. 

### B3. The Concrete Null-Strategy Zoo Protocol
To rigorously define the deflated bar, build this this week:

1.  **The Generation:** Create 1,000 random cross-sectional signal matrices for your 76 futures markets. Do *not* randomize returns; randomize the *signals*.
    * *Method A (Gaussian):* Each week, assign each asset a random $Z$-score from $N(0,1)$.
    * *Method B (Permutation):* Take your actual `xsmom` signal matrix and shuffle the asset labels each week (breaks cross-sectional edge, preserves auto-correlation).
2.  **The Simulation:** Push these 1,000 null signals through the *exact same* sizing, vol-targeting, and roll-cost friction layer you use in production. 
3.  **The Search Sim:** Group the 1,000 null backtests into "families" of 20 (to mimic your historical search space). For each of the 50 groups, find the maximum $t$-stat achieved by any of the 20 nulls.
4.  **The Threshold:** Calculate the 95th percentile of those 50 maximum $t$-stats. **This is your new Deflated Hurdle.** I predict it will be around $t = 2.8$ to $t = 3.1$. If your $t=2.29$ does not clear this, the futures book is dead on arrival.

### B4. Prospective Logging
Create a simple `research_ledger.csv`. Columns: `Date`, `Hypothesis`, `Universe`, `Parameters_Tested`, `Result_T_Stat`. Every time you run a historical simulation to test an idea, log it. The required $t$-stat for the $N$th test scales with $\sqrt{2 \ln(N)}$. 

---

## THEME C: DO THE FOUR PREMIA ACTUALLY DIVERSIFY, OR CO-CRASH?

You suspect you are heavily short-crisis. **You are correct.**

### C1. The "Single Bet" Reality
You have 3 risk-on premia and 1 explicitly short-vol premium (VRP). 
* **Trend:** Slow crisis convex (good), fast crisis whipsaw (bad).
* **Carry:** Structurally short liquidity. In a crisis, the term structure flattens/inverts, and carry bleeds.
* **VRP:** The definition of short tail-risk. Your gate (flat in backwardation) saves you from the *depth* of the crash, but you still eat the gap-down that triggers the backwardation.
**Tell:** Your book is a synthetic short-put option on global liquidity. 

### C2. The Stress-Conditional Correlation Test
Average correlation (0.12 - 0.46) is dangerously misleading. You need to compute **Exceedance Correlation**:
1.  Isolate the bottom 5% of weekly returns for the S&P 500 (or the top 5% of VIX weekly spikes).
2.  Calculate the correlation matrix of your 4 sleeves *strictly within those weeks*.
3.  **Threshold:** If the conditional correlation between Carry and Trend, or Carry and VRP, jumps above 0.50 in these stress windows, you have a single bet.

### C3. Stress-Testing Without Folds
Do a **Scenario Replay**. You have the raw asset data for 2008, 2018, 2020. Even if your CPCV didn't hold them out cleanly, you can force the live engine to trade those dates. 
* Override the historical signals with what the engine *would* have generated.
* Apply the actual market gap-downs. 
* Look at the aggregate NAV drawdown. If the combined book drops > 25% in the COVID crash before the gates kick in, the sizing is fundamentally wrong.

### C4. The Convex / Defensive Sleeve
You need structural long convexity to offset the short-liquidity nature of Carry and VRP.
* **Idea 1:** US Treasury Trend. Isolate your rates sleeves and ensure they can go long duration heavily.
* **Idea 2:** FX Trend (Long USD vs Risk-on currencies). 
* *Warning:* Do not buy structural long VIX options; the negative carry will bleed your alpha to zero. Instead, rely on Trend's natural convexity in the fixed income and FX space.

### C5. Does VRP Belong?
**No.** Drop it for now. A 0.46 correlation to your main engine, combined with structural short-crisis exposure, is the opposite of a diversifier. Keep it on paper until the futures book is live and proven.

---

## THEME A: GO-LIVE, CAPITAL SIZING & RISK ARCHITECTURE

This is the unbuilt layer. If the futures book passes the Null Zoo, here is how you deploy it.

### A1. Sizing
* **Method:** **Equal Volatility Contribution.** For 3 sleeves, Risk Parity / HRP is mathematical masturbation. Equal vol is robust and honest.
* **Target Vol:** For a solo \$100k account, target **12% annualized book volatility**. This translates to a max expected drawdown of $pprox 18-20\%$. 
* **Paper Discount:** Paper sleeves do NOT get equal weight. If Trend is live and proven, and Futures is paper-passed, give Futures a **0.5x risk weight** until it survives 6 months of live execution matching the backtest.
* **Margin-to-Equity:** Cap initial margin at **20% of Net Liquidation Value**. Futures drawdowns require massive cash buffers. If margin usage breaches 30%, auto-scale down.

### A2. Cross-Venue Risk (Alpaca + IBKR)
You need a "Risk Daemon" completely decoupled from your PM agent.
* It runs every 5 minutes.
* It hits both Alpaca and IBKR APIs.
* It calculates: `Total_NAV = Alpaca_NAV + IBKR_NAV`.
* It calculates: `Total_Margin_Usage = (Alpaca_Margin + IBKR_Margin) / Total_NAV`.
* **The Kill Switch:** A single physical or API-endpoint button that executes `close_all_positions()` sequentially on both brokers and halts the scheduler. 

### A3. Forward Risk Triggers
* **Realized-Correlation De-gross:** Look at the trailing 21-day rolling correlation between the Trend book and the Futures book. If it exceeds 0.60, reduce gross exposure by 30% across the board. 
* **Global Drawdown Ladder:** * At -10% NAV from High-Water Mark: Cut gross by 25%.
    * At -15% NAV: Cut gross by 50%.
    * At -20% NAV: Halt trading entirely. Go to cash. Investigate.

### A4. The Promotion Ladder
1.  **Rung 0 $ightarrow$ 1 (Paper $ightarrow$ IBKR Paper):** Prove the execution. Minimum 2 months. You must execute at least two full roll cycles in the paper account. *Threshold:* Paper slippage vs Backtest must be $< 10$ bps.
2.  **Rung 1 $ightarrow$ 2 (IBKR Paper $ightarrow$ Tiny Live):** 1 micro-contract per asset (e.g., MES, MNQ). Allocate maybe $5k total margin. 
    * *Demotion Stop:* If live tracking error vs paper exceeds 1 standard deviation of expected weekly return, revert to paper. 
3.  **Rung 2 $ightarrow$ 3 (Scale):** Requires 6 months of live trading covering at least one VIX spike $> 25$. 

### SUMMARY DIRECTIVE
Do the math on Theme B before writing a single line of execution code for Theme A. I suspect the futures book will fail the deflated Sharpe test. If it does, be grateful you found out on paper, and keep mining the data until you find an orthogonal edge that clears $t > 3.0$.
