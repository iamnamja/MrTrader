# MEMORANDUM

**TO:** MrTrader Principal  
**FROM:** [World‑Class Quant, ex‑Multi‑Manager Pod PM]  
**DATE:** 6 June 2026  
**RE:** Deep‑Dive Quant Review — Brutal Triage and Recommendations  

---

## 1. Executive Summary – The Hard Truths

Your system is well‑engineered for a solo operator. The architecture, audit trail, and safety gates are far ahead of what most retail quants build. *Unfortunately, you have spent most of that engineering effort validating noise.*  

The fundamental problem is that **your validation framework is broken** – and you already know it. The 52% fold‑skip bias is catastrophic. It means every Sharpe ratio you’ve ever reported is *materially* inflated, likely by 30‑50%, and your only “survivor” (PEAD) is, by your own honest admission, statistically indistinguishable from zero (p≈0.19).  

That leaves you with a system that is technically elegant but devoid of deployable alpha. My brutal advice: **stop paper‑trading PEAD as if it’s alpha, fix your validation method immediately, and redirect your scarce effort toward return sources that are genuinely exploitable with your constraints.**  

Below I give you a concrete, prioritised set of next moves. I will be direct and unsympathetic – that is what you asked for.

---

## 2. Method Critique – Your Validation Is Optimistic, and That’s Killing You

### 2.1 The Fold‑Skip Catastrophe
Skipping 52% of CPCV fold‑evaluations because of a rolling‑window overlap guard creates a **survivorship‑biased sample heavily tilted toward recent, low‑volatility bull markets**. The folds that are retained are precisely those where the overlap rule happened not to trigger – i.e. windows that often coincide with long, uninterrupted trends.  

In a purged cross‑validation design, you must *always* evaluate all folds, even if that means you have to relax the overlap definition or restructure the folds to be consecutive and non‑overlapping without skipping. If your current guard forbids this, replace it with a **strictly sequential walk‑forward** using anchored train/validation/test splits (no random fold combinations). You lose the “combinatorial” elegance of CPCV, but you gain an honest, unbiased timeline.  

**My recommendation:** abandon the current CPCV harness. Switch to a **Purged Walk‑Forward (PWF)** with:
- Fixed‑length training windows (e.g. rolling 5 years)
- Purged + embargoed out‑of‑sample period (e.g. 12 months)
- Evaluate the strategy on **every OOS period sequentially** – no skipping.
- Compute metrics only on that series of OOS returns.  
This will give you a realistic equity curve, and you can measure the *actual* Sharpe of deploying the strategy as you would have in real time.

### 2.2 Significance – You’re Still Fooling Yourselves
Using N_eff=#folds (8) to compute t‑statistics is a step in the right direction, but it is still generous when the folds are overlapping in time (because each fold’s return is auto‑correlated). A better approach for the sequential walk‑forward is simply to compute the **standard error of the out‑of‑sample mean return**, using a Newey‑West correction with an appropriate lag, and test whether the mean is > 0. That gives a t‑stat that accounts for serial correlation without pretending you have many independent folds.  

If the resulting t‑stat on the live‑simulated OOS returns is below 1.5, **the strategy has not been proven deployable**. Do not paper‑trade it. Do not allocate mental energy to it.

### 2.3 Promotion Gates – Sensible but not Stringent Enough
Your gate thresholds (Sharpe ≥0.80, DSR p>0.95, etc.) are reasonable *if your backtest were unbiased*. In practice, a 0.80 Sharpe in your current biased CPCV probably corresponds to a true OOS Sharpe around 0.3‑0.4 after accounting for selection bias and the fold‑skip effect. I suggest you **re‑calibrate all gates after you fix the validation** and also add a mandatory **live paper‑trading period** of at least 6 months as a final gate before any real capital is deployed. No paper‑track record, no real money.

---

## 3. Triage of PEAD – You’re Paper‑Trading Noise

Let’s look squarely at the facts:
- Event‑clustered bootstrap p≈0.19 → not significant at any conventional level.
- HAC t≈1.04 → no better than chance.
- 87% of P&L comes in up‑trends → you are harvesting equity beta, not a distinct drift anomaly.

**My verdict:** PEAD, as you have implemented it, is an expensive way to get long SPY with a slight tilt towards earnings‑winner rebalancing. The modest rules‑based overlay might give it a marginal positive expectancy (Sharpe ~0.2‑0.3) in up‑markets, but it will be crushed in a bear market, and the expected net alpha after costs is not statistically different from zero.

### 3.1 What to Do with It Now
- **Do not** size it as a core alpha source. At most, treat it as a beta‑enhancement overlay if you have a separate market‑timing mechanism that gets you out of equities during downtrends.  
- If you continue paper‑trading it, do so purely to gather live execution statistics (slippage, fill quality). Set a hard kill date: if after 12 months of live paper trading the net Sharpe (with realistic slippage) is <0.2 and the t‑stat on the live return stream is <1.0, **kill it permanently and delete the code**.  
- A p≈0.19 signal *can* be useful as a very small diversifier inside a portfolio of uncorrelated signals, but only if you already have other genuine alpha sources. You do not.

**Bottom line:** you are in love with your only “survivor”. It’s human. But it’s not alpha. Move on.

---

## 4. Where the Alpha Actually Is – Concrete, Exploitable Ideas

Given your constraints – ~$100k, retail fills, daily‑bar swing horizon, willingness to use simple rules, and a powerful automation stack – I would pursue the following four return sources. They are ordered by expected risk‑adjusted return net of realistic costs, and by their diversifying power relative to a long‑equity beta exposure.

### A. Volatility Risk Premium Harvesting (Selling Equity Index Options)
**Why:** The single most robust retail‑accessible risk premium is the variance risk premium – the tendency of index option implied volatility to over‑estimate realised volatility. Selling premium systematically, in a risk‑defined manner, has generated persistent alpha across decades and is uncorrelated to long equity direction over the short term. It is a natural crisis diversifier (you make money when vol falls, but you must manage the tail).

**What you need:**
- Data: Polygon Options chains/IV (you already have the plan). Historical daily OHLC options data for SPX/SPY going back to at least 2007.
- New backtest stack: must model option greeks (delta, gamma, theta), bid‑ask spreads, early assignment risk, and margin requirements. A simple simulation of a defined‑risk spread (e.g. short put spread on SPX) with 30‑45 DTE, managed at 21 days to expiry, is fully sufficient. The engineering effort is real but well worth it.
- Execution: Alpaca options trading. Use limit orders, and widen your simulated fills by at least 10‑20% of the spread to be conservative.

**Edge:** Structural demand for hedging from institutional investors; retail can harvest it without the same balance‑sheet constraints.

**Validation:** Purged walk‑forward on daily options data. The strategy’s returns are very path‑dependent; you must simulate the exact entry and exit prices on each roll date. Use a realistic commission (e.g. $0.65/contract) and assume you cross the full spread on entry, mid on exit. The metric to watch is the **Kelly‑optimal Sharpe** net of costs, and the maximum drawdown in 2008/2020/2022. Do not even think about going live unless the net Sharpe >0.8 on a realistic backtest.

**Expected capacity:** >$500k easily; you are the retail flow, not the market maker.

**Correlation to PEAD:** Near zero.

### B. Analyst Estimate Revision Drift (Post‑Revision Announcement Drift)
**Why:** A well‑documented anomaly (Givoly & Lakonishok, 1979; Chan, Jegadeesh & Lakonishok, 1996) that survives after trading costs and is economically intuitive. When analysts raise their earnings estimates, prices drift upward for weeks; when they cut, prices drift down. It is a cousin of PEAD, but the signal is spread out over time and less concentrated around earnings events, making it a good portfolio diversifier.

**What you need:**
- Data: FMP “analyst estimate revisions” endpoint (you already have access). You can pull point‑in‑time consensus estimate changes and label events when the 1‑month change in consensus EPS is >X%.
- Implementation: a simple rules‑based long/short portfolio of the top/bottom decile of revision intensity, held for 20‑40 trading days, with a delay of 1 day after the revision date (to avoid look‑ahead). Universe: Russell‑1000, filtered for minimum analyst coverage (≥5 analysts). Equal‑weight or vol‑target to 0.3‑0.5% per name.

**Validation:** Apply the same purged walk‑forward methodology on your daily bar simulator. Because the signal is point‑in‑time from FMP, you must ensure the revision date is truly published before you act. Check with FMP’s `filingDate` or `date`. This should be free of the fold‑skip problem once you fix the harness. I would expect a gross long‑short spread of 3‑5% annualised, netting to ~2‑3% after 10‑15 bps cost. Sharpe 0.5‑0.7 is plausible.

**Capacity:** High – you can run it on 50‑100 names per side easily.

**Correlation to PEAD:** Moderate (both are event‑driven), but because the timing differs, combined portfolio benefit is positive.

### C. Time‑Series Momentum (Trend Following) on Broad ETFs
**Why:** Time‑series momentum (Moskowitz, Ooi & Pedersen, 2012) is robust across asset classes and has a strong track record of providing crisis alpha. For a retail account, a simple 12‑month look‑back trend‑following rule on SPY, IWM, QQQ, and possibly TLT/GLD provides a strategy that is long when the trend is up, flat (or short) when it’s down, dramatically improving risk‑adjusted returns relative to buy‑and‑hold and providing diversifying positive skew.

**What you need:**
- Data: already have – daily adjusted prices for ETFs.
- Rules: each day, compute the excess return over the past 252 days. If positive, go long the ETF with a volatility‑targeted position (e.g. 15% annual vol). If negative, move to cash or short (if you allow shorting ETFs). Rebalance weekly to reduce turnover.
- Costs: negligible when using highly liquid ETFs, but you must include bid‑ask spread (≈1‑2 bps). Your current Alpaca execution can handle ETF orders easily.

**Validation:** Using your existing equity‑bars simulator (beta‑hedge off), run a purged walk‑forward on a multi‑ETF trend portfolio. Historical net Sharpe on a basket of US equity index futures has been around 0.5‑0.7 with occasional enormous years (2008). That alone is a better “alpha” than all your cross‑sectional ML combined.

**Capacity:** Unlimited for your size.

**Correlation to PEAD:** Low, especially during equity drawdowns.

### D. (Optional, lower priority) Pre‑Earnings Announcement Drift (Run‑up)
Many of the same stocks that exhibit PEAD also tend to drift upward in the weeks *before* the earnings announcement, especially for firms that later beat. A simple rules‑based strategy that buys stocks with the highest pre‑announcement price strength, exiting the day before earnings, could harvest a distinct premium that is uncorrelated to the post‑earnings drift. I’d put this in the “interesting but test it quickly” bucket – if it doesn’t show a net Sharpe >0.5 after costs in an honest PWF, bin it.

---

## 5. Missing Data – What Would Actually Change the Odds

You already have a solid data stack for equities. The one transformational addition is **historical options data** (full chain, with greeks) for liquid indices. This unlocks vol‑selling strategies, which I consider the single highest expected‑value project on the table.

Beyond that:
- **Analyst revision data** you already have; start using it now.
- **Intraday 1‑min bars** from Polygon (you have 5‑min) might be useful for a future mean‑reversion or execution‑timing model, but not a priority.
- **News sentiment** (NewsAPI + a simple BERT sentiment model or even Finnhub’s pre‑computed sentiment) could give you an intraday edge on discrete events. But the infrastructure needed to beat the decay time is high – I’d set it aside until you have live alpha elsewhere.

Do *not* spend money on expensive alternative datasets (satellite, card‑spend) at your capital level – you will never recoup the cost.

---

## 6. Is Cross‑Sectional ML Really Dead? (And What You Did Wrong)

Cross‑sectional ranking of equities on daily data is **not fundamentally impossible**, but you (and most quants) underestimate the severity of the multiple‑comparison problem and the small magnitude of true anomalies in highly liquid, heavily‑arbitraged US names. Your null was always market beta, and your features were mostly transformations of price and common risk factors.

**What you did wrong:**
- You trained on a universe already biased by survivorship and liquidity filters.
- You used purged folds but the skipping turned your “OOS” into post‑2016 bull runs, completely hiding regime sensitivity.
- You combined many weak, correlated features without a rigorous theory of *why* they should predict returns independently of known factors. XGBoost will happily latch onto a subtle combination of momentum and low vol that is just the equity risk premium.

**If you ever want to revisit ML:**
- Predict **residual returns** (orthogonalised to Barra‑style risk factors) – do not try to beat a long‑only benchmark directly.
- Use a much shorter horizon (1‑5 day holding) to reduce noise, but then costs eat you. At your cost structure, I’m sceptical any cross‑sectional ML can survive.
- Instead, use ML for **meta‑labelling**: a binary classifier to size or filter trades generated by a simple, economically‑motivated rule (e.g., “should I take this trend signal?”). That is more robust.

For now, **cross‑sectional ML should stay in the graveyard.** Use your engineering talent elsewhere.

---

## 7. Architectural Redesign – Simplify or Die

Your multi‑agent architecture with Redis pub/sub, PM, RM, Trader, and a 60‑second heartbeat is beautifully over‑engineered for a solo operator trading a handful of daily‑bar strategies. It adds cognitive load, maintenance burden, and potential for coordination bugs.

**What I would do:**
- Collapse the daily decision loop into a single Python script that runs once at 09:30: fetch data, compute signals for 2‑4 independent strategies (trend, revisions, vol selling, etc.), combine positions with a simple risk overlay (vol‑targeting and sector cap), and emit a single batch of orders.
- Keep your audit and risk database – that is excellent.
- Ditch the intraday horizon completely until you have a proven signal that *requires* it.

The simpler the moving parts, the more likely you are to spot a bug and to actually deploy with confidence.

---

## 8. Prioritised Action Plan (Ranked by Expected Value)

1. **Fix validation immediately** – replace CPCV with purged sequential walk‑forward, re‑run all strategies, and permanently kill anything that doesn’t show a net OOS Sharpe >0.3 after realistic costs (that threshold is already generous). This alone is the highest‑EV move, because it stops you from wasting time on illusions.

2. **Build a basic options backtest stack and design a simple index vol‑selling strategy** (SPX short put spread, 30‑45 DTE, managed). Data first, then simulate with conservative spread assumptions. If net Sharpe >0.8, begin paper trading with 1‑2% of account risk. This is the only genuine, capacity‑unconstrained retail alpha I know of.

3. **Implement the analyst revision drift strategy** on your existing daily bar simulator – use FMP data, simple decile L/S, hold 20 days. If PWF net Sharpe >0.4, add it as a diversifier. It can be live within weeks.

4. **Add a trend‑following ETF rotation module** – simple 12‑month lookback on SPY, QQQ, IWM. This gives you a robust, non‑correlated return stream and a natural hedge for your long‑biased drift strategies.

5. **Stop paper‑trading PEAD** (or reduce it to minimal size for execution metrics only). Re‑allocate your mental and coding capacity to the above.

6. **Optionally** explore pre‑earnings drift or a sentiment‑based intraday scalp only if the first four items are live and stable.

---

## 9. Final Word – What Would a Skeptical IC Say?

If I were a PM at a fund and you pitched me your current system, I would ask: “How much live money have you made, and show me the OOS track record that is not tainted by look‑ahead or selection bias.” Right now, you have zero. Your validation doesn’t meet the minimum bar for a serious systematic investor.  

But the silver lining is this: you have a solid engineering foundation and the intellectual honesty to suspect that you’re fooling yourself. Most retail traders never get that far.  

**Destroy your false idols (PEAD, CPCV with skipped folds, cross‑sectional ML on uncleaned data). Build the three simple, robust strategies above. Then let real money, slowly and carefully, be the judge.**  

You are closer to a viable, diversifying automated book than 95% of the people who attempt this. But only if you stop polishing a fancy machine that, so far, grinds out nothing but noise.

---
*End of memo. Save as `MrTrader_Brutal_Review.md`. Good luck.*