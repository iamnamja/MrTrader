# Deep-Dive Quant Review: MrTrader Automated Trading System
**Reviewer:** Systematic PM / Quant Researcher
**Date:** June 2026
**Target:** Solo Operator / Retail Constraints ($100k, No PB, Python/Alpaca)

---

## Executive Summary: The Brutal Truth
You have built a beautiful software engineering platform to execute noise. Your architecture (Redis pub/sub, strict risk gating, automated pipelines) is institutional-grade. Your research process, however, is fundamentally flawed, suffering from massive over-complexity, unstationary feature sets, and a backtesting framework that lies to you. 

The fact that your ML graveyard is full of dead models is actually a **good sign**—it means your out-of-sample guards are catching the garbage before you trade it. But the one model that survived (PEAD) is surviving on borrowed time, buoyed by a bull-market bias in your cross-validation scheme.

Here is the teardown of your methodology, the reality check on your survivor, and the roadmap to actual alpha given your constraints.

---

## 1. Method Critique: Overfitting by Complexity

### The CPCV / Fold-Skip Disaster
Your CPCV methodology is the biggest liability in your system. Combinatorial Purged Cross-Validation was popularized by Marcos López de Prado for **time-series macro and futures strategies**, not for cross-sectional equity ranking on a 750-stock universe. 
* **The 52% Fold-Skip Bias:** By skipping 52% of folds (due to rolling-window overlaps and regime guards), you have systematically deleted the very environments that test model robustness. A model evaluated primarily in an overlapping, bull-market regime is just a beta-harvester. 
* **The Illusion of Rigor:** Deflated Sharpe Ratio (DSR) and path-Sharpe t-stats are mathematically meaningless if the underlying distribution of paths is heavily biased by fold-skipping. You are applying high-end statistics to a compromised sample.
* **The Fix:** **Ditch CPCV.** Move to a strict, continuous **Walk-Forward Optimization (WFO)** with an expanding or rolling window (e.g., 5 years train, 1 year test, slide forward). Do not skip regimes. If your model cannot survive 2022 or 2020 without a hard-coded "regime filter," the model has no alpha—it's just a conditional beta exposure.

### The Gates
Your risk gates (Agent 2) are solid engineering, but they act as backtest-overfitting parameters. If you tweak the `per-name size cap` or `sector concentration` to smooth a backtest, you are curve-fitting the risk manager.

---

## 2. Brutal Triage of PEAD: The "Survivor"

**Verdict: Do not deploy real capital to this specific implementation.**

Here is why your PEAD is showing a p≈0.19 (which is statistically insignificant noise, meaning 1 in 5 random monkeys could generate your equity curve):
1.  **The "Next Open" Execution Fallacy:** You assume entering at the next open with a marketable-limit order costs ~10 bps. On an earnings gap-up (positive EPS surprise), retail marketable limits placed at 09:30:00 ET are toxic flow. Wholesalers (Citadel, Virtu) will fill you at the absolute peak of the retail morning rush. Your backtest assumes the official opening print; your live slippage will be 30-50 bps on these specific event names.
2.  **SPY Trend Filter = Beta:** You improved the Sharpe from 0.546 to 0.661 by blocking entries in SPY downtrends. This proves the "alpha" is actually just high-beta exposure to market rallies.
3.  **The Fix to Save It:** If you want to trade PEAD, you cannot trade the raw EPS surprise. The market prices raw EPS instantly. You must trade **Standardized Unexpected Earnings (SUE)** combined with **Management Guidance Revisions**. Furthermore, delay your entry. Let the 9:30–10:00 AM retail/institutional volume settle. Enter via TWAP from 10:30 to 15:30.

---

## 3. Where the Alpha Actually Is (Given Your Constraints)

Forget cross-sectional predictive ML on price bars. You are competing against deep learning clusters at Two Sigma parsing order book updates in nanoseconds. You need strategies that institutions ignore because of **capacity constraints** (they can't deploy $100M into it) or because they are **operationally annoying**.

Here are 3 return sources you should actually pursue:

### A. Index Rebalance Arbitrage (Event-Driven / Rules-Based)
* **The Edge:** Russell and S&P index rebalancing creates massive forced buying/selling by passive funds. The rules for inclusion are public.
* **The Method:** Predict the additions/deletions weeks in advance (using market cap and float data). Go long predicted additions, short predicted deletions.
* **Why it fits you:** It requires zero ML. It is a data-engineering problem. Institutional PMs trade this, but smaller names in the Russell 2000 rebalance are too illiquid for mega-funds, leaving meat on the bone for a $100k account.

### B. "The New PEAD": Estimate Revisions & Analyst Dispersion (Factor)
* **The Edge:** Rather than trading the earnings *event*, trade the *slow institutional reaction* to changing forward estimates.
* **The Method:** Get IBES/FactSet consensus forward 12-month EPS estimates. Rank your universe by the 30-day change in the *consensus estimate* (not the rating). Go long the top decile. 
* **Why it fits you:** It's a slow-decay anomaly. Holding periods are weeks to months. It survives daily-bar backtesting and 10 bps retail execution.

### C. Mean Reversion of Intraday Sector Dislocation (Stat-Arb Light)
* **The Edge:** ETFs and their underlying constituents often decouple temporarily due to retail flow or basket hedging.
* **The Method:** Calculate the implied price of an ETF (e.g., XLF) based on the real-time weighted average of its holdings. When the ETF price deviates from the basket NAV by > x bps, buy the cheaper, sell the expensive. 
* **Constraint Warning:** You would need to revive your 5-minute data pipeline, but you drop the XGBoost overhead. Pure linear algebra and threshold rules.

---

## 4. Missing Data & Tooling

Stop buying alternative data until you fix the fundamentals.
* **Garbage to Drop:** MACD, RSI, Bollinger Bands, "Macro NIS." XGBoost cannot extract stationary edge from these.
* **Crucial Missing Data 1 — Options Implied Volatility Surface:** (Available cheap via Polygon). You are trading equities, but the options market is smarter. Cross-sectional equity returns are highly predicted by **IV Skew** (the difference in implied vol between OTM puts and OTM calls). If call skew is violently steepening, smart money is positioned for upside.
* **Crucial Missing Data 2 — Detailed Short Borrow Rates:** FINRA days-to-cover is useless retail data. You need institutional borrow rates (e.g., IBKR fee rates API). High borrow fees predict negative equity returns reliably. 

---

## 5. Modeling: Why Cross-Sectional ML Died on You

You noted that every cross-sectional ML / single-factor ranking went to the graveyard. Here is why:

1.  **Non-Stationarity:** You fed raw technicals (Price vs EMA, RSI) into tree-based models (XGBoost). Trees make hard splits on absolute values (e.g., `If RSI > 70`). The market regime shifts, the absolute values change, and the tree breaks out of sample.
2.  **Improper Feature Scaling:** For cross-sectional ML in equities, **every single feature must be cross-sectionally z-scored (or rank-normalized) daily**. The model should not know if the VIX is at 15 or 40; it should only know that AAPL's momentum is +2 standard deviations above the cross-sectional mean *today*.
3.  **The Wrong Target:** Predicting forward returns directly is a low signal-to-noise task. A better approach is to use ML to predict *fundamental outcomes* (e.g., predict whether next quarter's SUE will be positive), and then use a simple rules-based system to trade those predictions.
4.  **Ensembles over LambdaRank:** LambdaRank optimizes for the entire list ordering. In trading, you only care about the extremes (Top decile vs Bottom decile). Use binary classifiers tuned to precision at the tails.

---

## 6. Strategic Redesign: The Path Forward

If I were to tear down MrTrader and rebuild it for a solo operator with $100k, here is the blueprint:

**1. The Universe:** Shrink from 750 to the S&P 500. You cannot afford the slippage and spread of the lower Russell 1000 names with retail routing. 
**2. The Horizon:** Stick to daily swing trading. You correctly identified that intraday ML dies to cost drag.
**3. The Sim Engine:** Build a vectorized backtester using `pandas` or `polars`. Event-driven backtesting is for final staging; it is too slow for rapid research. You need to test 100 variations of a rule in 10 minutes, not overnight.
**4. The Architecture:**
* **Kill:** The Regime sizing multiplier. It's a crutch for bad models. A good strategy is market-neutral or dynamically hedged.
* **Keep:** The independent Risk Manager agent. Your gating logic is excellent for live safety.
* **Build:** A proper limit-order execution model. Stop crossing the spread at the open. Model VWAP execution over the first 2 hours of the day.

**Summary Directive:** Freeze your ML pipeline. It is an overfitting machine right now. Shift to economically motivated, rules-based anomalies (Estimate Revisions, Index Arbitrage) utilizing high-quality fundamental and options-market data. Rebuild your cross-validation to standard Walk-Forward without fold-skipping. Once you find an edge in the raw rules, *then* you can re-introduce simple linear ML (Ridge Regression) to optimize the signal weighting.
