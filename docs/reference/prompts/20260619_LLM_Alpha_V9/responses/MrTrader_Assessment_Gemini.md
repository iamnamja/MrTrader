# MrTrader Assessment: Brutally Honest Quant PM Review

You asked for candor, so I will give it to you straight. You have built an impressive, rigorous institutional-grade research harness for a solo operator. Your intuition to fight p-hacking, use combinatorial purged CV (CPCV), and demand out-of-sample paper tracking puts you ahead of 95% of retail quants. 

However, you are suffering from a severe case of mislabeling your returns, operating with a few dangerous blind spots in your pipeline, and running an execution architecture that is a toy.

Here is my brutally honest assessment of your state, your edges, your architecture, and your next moves.

---

## 1. Are you finding real alpha or fooling yourself?

**You are not finding alpha. You are harvesting Alternative Risk Premia (ARP).**
Stop calling Trend and Carry "alpha". Alpha is the unexplainable residual after all known systematic risk factors are accounted for. Time-Series Momentum (TSMOM) and Term-Structure Carry are canonical, academically documented, crowded risk premia. 

**Is "free daily US-equity directional alpha mined out"?** Yes. For your holding period (weekly/swing) and your data (daily OHLCV), any standalone signal in highly liquid US equities was arbitraged away by 2012. You are correct to kill PEAD and cross-sectional ML on daily yfinance data. You are not miscalibrated; the market is just efficient at that timescale. If you want equity alpha, you either need to go faster (intraday microstructure), go weird (alt-data/sentiment), or go smaller (micro-caps/OTC). Given your constraints, abandoning daily US equity directional alpha is the most mathematically sound decision you've made.

## 2. Your Current Edges: Trend + Carry

Your results for TSMOM (SR ~0.72) and Carry (SR ~0.55-0.66) are exactly in line with institutional CTA/managed futures performance. They do not smell overfit; they smell like beta to known macro factors.

**The glaring red flag:** Your Carry backtest caveat about unmodeled roll cost is a cardinal sin. If your signal *is* the term structure (the roll yield), but your backtest trades the front month and ignores the cost of rolling to the next, your backtest is leaking the exact premium you are trying to capture. You estimate it's a -0.05 to -0.15 SR hit. In reality, market impact and slippage during roll weeks (when everyone else is rolling) often compress this further.
* **How to settle it:** Fix the roll logic immediately. Do not trade the front month. Create a continuous back-adjusted series that explicitly books the spread cost on the roll date. Until you do this, your 0.66 SR for Carry is a phantom number.

## 3. Strategies You HAVEN'T Tried (With Existing Data)

Since you have 30+ years of survivorship-free Norgate Futures data, you are under-utilizing it. You are trading TSMOM and Carry as distinct sleeves. 

**A. Curve Momentum (Trend of Carry)**
* **The Concept:** Don't just look at the current term structure; look at how the term structure is changing. If an energy curve is moving from contango to backwardation, that momentum of the spread is often highly predictive of inventory crunches.
* **Signal:** The 1-month and 3-month momentum of the annualized term-structure slope.
* **Universe:** Norgate Futures.
* **Kill Criteria:** If it doesn't survive a multi-factor residualization against your base Carry and base TSMOM, kill it. 

**B. Cross-Sectional Futures Momentum (XSMOM)**
* **The Concept:** TSMOM is absolute (is the asset going up?). XSMOM is relative (is this asset going up *faster* than other assets in its sector?). 
* **Signal:** Rank assets within their sector (e.g., ags, metals, equities) by their 12-month return, scaled by volatility. Long the top quartile, short the bottom.
* **Universe:** Norgate Futures.
* **Kill Criteria:** Must show positive dSR vs your TSMOM book in Track-B.

## 4. Architecture & Resilience

Your execution architecture (`uvicorn` FastAPI in-process orchestrator + APScheduler) is a **fragile toy**. 
* **The Risk:** You are running a stateful trading loop inside a web server process. If FastAPI crashes due to an unhandled web request, an out-of-memory error, or a dependency fault, your PM, Risk Manager, and Trader die with it. 
* **The Fix:** Decouple execution from the API. The PM/Trader agents should be standalone daemon processes (e.g., standard Python scripts managed by `systemd` or a lightweight container orchestrator) that read/write to Postgres. FastAPI should *only* be a read-only view into the DB for your React dashboard. 

Your research pipeline (Ruler-v2, Track-A/Track-B, Pre-registration) is **excellent**. Your Type-I/Type-II awareness is elite. Keep this. 

## 5. The Next Data Buy

**Do NOT buy Norgate US Stocks (Platinum) yet.** You already proved that daily equity alpha is dead for your holding period. Buying clean data to re-prove that PEAD and stat-arb don't work is an academic exercise, not a money-making one.

**Your highest-EV move is transitioning from Alpaca paper to an IBKR Pro account for Live Futures Execution/Data.**
You have a promising Carry strategy that relies on EOD Norgate data. You cannot trade futures on Alpaca. Your entire bottleneck right now is proving that your futures signals survive real-world execution, slippage, and margin requirements. Spend your money on IBKR market data subscriptions and margin capital.

## 6. Monday Morning Playbook (Top 4 Moves)

1. **Halt all new signal research and fix the Carry Roll Cost.** Rewrite your futures backtester to explicitly handle contract rolls and the associated slippage. Your Carry edge is not real until this is mathematically sealed.
2. **Decouple the Orchestrator.** Strip the APScheduler out of FastAPI. Make your trading agents resilient, standalone processes. 
3. **Open an IBKR Account & Connect the API.** You cannot validate futures carry without a broker that supports futures. Start logging live, real-time IBKR futures quotes to build an execution simulator.
4. **Implement Curve Momentum.** Once roll costs are fixed, test the momentum of the term structure on your Norgate data. It is the logical evolution of your current stack.

You have built a rigorous, beautiful machine. Stop digging for gold in the barren wasteland of daily US equities, fix your futures plumbing, and start acting like the CTA you have mathematically proven yourself to be.
