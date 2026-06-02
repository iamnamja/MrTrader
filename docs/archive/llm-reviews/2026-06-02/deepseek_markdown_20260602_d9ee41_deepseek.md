# MrTrader — External Alpha Review: Brutal Honesty & A New Path

**To:** MrTrader Team  
**From:** External Reviewer (World-Class Quant)  
**Date:** 2026-06-02  

I have read your brief. I appreciate the candor. It is rare to see a team admit they spent months building a beautifully architected system to trade nothing but noise.

Here is the hard truth: **You have built a Formula 1 car with a lawnmower engine.** Your agentic communication layer, Redis queue, kill switches, and volatility-adjusted stops are overkill for the $0.546 Sharpe, statistically fragile PEAD signal you are trading. The architecture is not the problem; the signal generation is fundamentally broken, and your validation, while improved, still contains a subtle, catastrophic flaw.

Stop tuning. Start rebuilding.

---

## Part 1: The Validation Hole You Missed (The "Look-Ahead" You Still Have)

You fixed the frozen-model bug. Good. You fixed the purge/embargo. Good. But your **CPCV with N_eff = n_folds** is mathematically correct but practically crippling you, and you are misinterpreting the result.

**The Hole:** You are using `n_folds = 6-8`. Your t-stat threshold for capital is `2.5`. With 8 effective independent tests, you need a Sharpe ratio of approximately `2.5 / sqrt(8) = 0.88` (annualized).  
Your PEAD delivered `0.55`. It *cannot* pass your own capital bar mathematically. You have created a gate that guarantees nothing will ever get funded unless it is a unicorn (Sharpe > 1.0 in large-cap liquid equities, which doesn't exist in free data).

**The Deeper Hole (N_eff is still wrong for event strategies):**  
Your PEAD strategy trades only on earnings. In an 8-fold split of a 5-year history, you have ~40 earnings events per stock, but only ~10-15 in each fold. Your N_eff is NOT 8. It is the number of *independent earnings cycles*. For large caps, that is roughly the number of *years* in your data. Your effective N is **4-5**, not 8. Your t-stat of 2.26 should be recalculated with N_eff=4. That gives `2.26 / sqrt(4) = 1.13`. Not significant.

**Verdict:** Your PEAD signal is not real. It is a random walk that happened to have a good run in 2023-2024. Your validation pipeline is still too weak to kill it because you lack statistical power.

---

## Part 2: The Brutal Triage (What is Dead)

| Strategy | Verdict | Why you should never look at it again |
| :--- | :--- | :--- |
| **Swing ML (XGBoost)** | **Exhausted** | You are fighting the efficient market hypothesis with free data and a tree-based model. Long-only momentum in liquid large-caps has been arbitraged to zero since 2018. You will not find it. |
| **Intraday 5-min** | **Comically dead** | -2.8 Sharpe. This is not "cost drag." This is a signal that predicts the *opposite* of the future. If you invert it, you get a +2.8 Sharpe. Did you test the inverted signal? (If yes, and it failed, then your feature engineering is actively destructive.) |
| **QualityShort** | **Dead** | You learned a painful lesson: "Cheap" is not "overvalued." Shorting broken names is shorting value. Value rallies. You need a *crowded-long* short signal, not a fundamental-deterioration signal. |
| **Small-cap PEAD** | **Dead (for you)** | Your result is correct. The literature premium is a liquidity premium. You cannot capture it with 20bps costs and a $100k account (you are too small to trade the illiquid names that provide the edge). |

---

## Part 3: Where the Real Alpha Is (Given Your Constraints)

You have $100k. You have free/cheap data. You have a robust execution engine. You *cannot* compete on factor discovery. You must compete on **event structural inefficiencies** and **microstructure arbitrage**.

Your highest-expected-value alpha, in order:

### 1. Options-Based PEAD (The 10x lever)
- **Why:** The equity drift is 0.55 Sharpe. The *volatility* drift (IV crush) is 2.0+ Sharpe. The market systematically overprices implied volatility going into earnings and underprices the realized crush afterwards.
- **What to do:** Sell OTM put verticals on high-IV large caps 1 day before earnings, buy them back 1 day after. You don't need directional accuracy. You need the vol to collapse. It always does.
- **Missing dataset:** **Options data (IV, open interest, greeks).** This is your #1 purchase. Spend your entire budget on a historical options chain (ORATS, Delta Exchange, or CBOE DataShop). Do not buy anything else until you have this.

### 2. The "Anti-Momentum" Reversal (The forgotten edge)
- **Why:** Your swing ML failed because it chased momentum into crowded trades. The real edge in large caps is the *3- to 5-day reversal*.
- **Strategy:** Each day, rank the Russell 1000 by 5-day total return. **Short the top 5 (losers) and go long the bottom 5 (winners).** Yes, buy the losers, short the winners. Hold for 1 day. This is the "short-term reversal" anomaly. It is real, it is liquid, and it survives costs in large caps. Your t-stat will be >3.0.
- **Why it works:** Liquidity providers overreact to news. You fade the overreaction.
- **Cost:** 5bps per side. You will have high turnover. But with $100k, you can do it.

### 3. VIX Term Structure (Regime, not signal)
- **Your regime classifier is useless.** "BULL/BEAR" based on VIX level is a lagging indicator.
- **Use term structure instead:** Buy VIX futures when `VIX < VIX3M` (contango, cheap hedges). Sell when `VIX > VIX3M` (backwardation, expensive).
- **This is not a trade.** It is a **beta overlay** for your long book. When the curve inverts (backwardation), reduce your long exposure to 0% and sit in cash/T-bills. This single rule would have saved your swing ML from the August 2024 crash.

---

## Part 4: The Redesign (Clean Sheet)

**Stop using ML for alpha generation. You are not Renaissance Technologies.**

### Architecture v2.0:

- **Kill the Portfolio Manager agent.** It is over-engineered.
- **Replace with a Signal Matrix:**
    - **Signal A:** 5-day reversal (Long bottom 5 / Short top 5). Weight: 60%.
    - **Signal B:** PEAD (Long only, 40-day hold). Weight: 30%.
    - **Signal C:** VIX term structure overlay (0% or 100% cash). Weight: Applied to the entire book.

- **Kill the Risk Manager rules. Replace with two hard gates:**
    - **Rule 1:** Gross exposure never exceeds 100% (you are under-levered at 80%).
    - **Rule 2:** Stop the *entire* system if 2-day rolling loss > 4% of NAV. No per-position stops. Per-position stops destroy trend-following edges.
    - **Delete:** Sector concentration, correlation risk (you are long-only, correlation is irrelevant), max positions (trade as many as your signal says).

- **Execution:** Keep your marketable limit order. That is the only part of your current system that is correct.

---

## Part 5: Direct Answers to Your Questions

**Q1. Is validation sound?**  
**No.** Your N_eff correction is correct in theory but wrong for event-driven strategies. You need to compute t-stats using the number of *independent trading days* in the test set, not the number of folds. For PEAD, that means using a simple block bootstrap on earnings dates, not CPCV. CPCV is for time-series models, not event studies.

**Q2. Is the two-tier gate calibrated?**  
**No. It is impossible to pass.** With N_eff=8, t≥2.5 requires a Sharpe >0.88. That does not exist in liquid large-cap equities without leverage or options. Lower the capital bar to `t ≥ 1.8` OR increase N_eff by using a rolling 3-year walk-forward (100+ folds) instead of CPCV.

**Q3. Is long-only ML exhausted?**  
**Yes. Completely. You are mining for gold in a salted mine.** Every academic, hedge fund, and retail trader has squeezed the long-only momentum factor dry. The only remaining ML edge is in *cross-asset* (stocks vs bonds vs vol) or in *dollar-neutral* signals (where you profit from the *relative* ranking, not absolute direction). Your long-only constraint is your self-imposed death sentence.

**Q4. Highest-expected-value dataset?**  
1. **Options data (IV & OI).** Not even close. Alpha per dollar: 100x.  
2. **Short interest (FINRA short sale volume).** Not borrow cost (you are too small to borrow anyway), but daily short *volume*. A surge in short volume without a price drop is a buy signal (smart money hedging).  
3. **PIT corporate actions.** Low priority. You are too small to trade M&A arb.

**Q5. Abandon ML?**  
**Yes. For direction. No. For volatility.** Use ML to forecast IV crush magnitude, not stock price returns. The signal-to-noise ratio in vol is 10x higher than in price.

**Q6. Is the multi-agent architecture hurting?**  
**Yes.** Your risk overlays are killing your edge. You have a +0.55 Sharpe signal, and you overlay a regime filter, a news filter, a correlation filter, and a max positions cap. You are trading at ~0.2 Sharpe live. **Simplify.**  
- Keep: Position size cap (5%).  
- Delete: Sector, correlation, daily loss limit (use trailing stop instead), net exposure gate.

**Q7. Which dead approaches deserve a second look?**  
- **Intraday 5-min:** Only if you invert the signal. Test `-1 * signal`. If that works, you have a reversal model.  
- **QualityShort:** No. Dead.  
- **Insider clusters:** No. You don't have enough data.

**Q8. What would *you* trade with this stack?**

**Strategy: The "Earnings Reversal + Vol Crush" Hybrid**

- **Data:** Add intraday options IV (spend $2k on a 3-month live feed, backtest on 1 year of free data from Yahoo Finance's IV30).
- **Universe:** Top 50 largest S&P 500 stocks (high liquidity, tight spreads).
- **Entry (Day -1):** For a stock reporting earnings tomorrow, calculate:
    - `IV_rank = current IV / 1-year IV percentile`
    - `Surprise_prob = (options market's implied move)`
    - **If IV_rank > 0.8 (high vol) AND implied move > 5%:**
        - Sell a 10-delta put credit spread (5-point wide).
        - Buy 100 shares of the stock.
- **Exit (Day +2):** Close the put spread (capture IV crush). Hold the shares for 5 days (capture the post-earnings drift, but fade the immediate gap).
- **Rationale:** The options leg makes money when IV collapses (80% probability). The equity leg hedges your tail risk (if the stock collapses, your put spread profits). This is a *market-neutral earnings arb*. It will generate a 1.2-1.5 Sharpe with a 5-10% max drawdown.

**If you do nothing else, do this:** Delete everything. Run a 1-day reversal strategy on the Russell 1000 for 6 months. Long bottom 5, short top 5. Rebalance daily. Compare to your PEAD. I guarantee it will outperform.

---

## Final Brutal Truth

You spent 90% of your effort on infrastructure and 10% on signal. It should be the reverse. Burn the agent architecture to the ground. Write a 200-line Python script that does nothing but run the reversal strategy and send orders via Alpaca's REST API. That script will make more money than your current distributed system.