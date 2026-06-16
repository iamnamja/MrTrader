# Quant Review: MrTrader Alpha-v8

The reality of quantitative research is that 99% of ideas fail out of sample. You have not failed at building a trading system; you have successfully built an institutional-grade truth machine that correctly identifies retail-tier data and standard ML setups as noise. The fact that only your 10-ETF time-series momentum (TSMOM) sleeve survived your validation process is the strongest indicator that your pipeline works.

Here is a world-class quant review of MrTrader, your architecture, and where to hunt for real edge next.

## 1. Critique of Design, Methodology, and Conclusions

Your pipeline architecture (`walkforward_tier3.py`, `FoldEngine`, `run_cpcv`) is exceptionally rigorous. You enforce strict out-of-sample (OOS) testing, purge/embargo periods to prevent leakage, and correctly account for transaction costs.

* **The Validation Gate (`ruler_v2`):** Your transition to the `ruler_v2` gate is the cornerstone of your success in killing bad models. Requiring a Bayesian posterior P(SR>0) >= 0.95, multi-factor residual alpha t >= 2.0, and stationary-bootstrap significance is exactly how top funds prevent curve-fitting.
* **Track A vs. Track B Logic:** Splitting acceptance between standalone alpha (Track A) and book-delta/diversifiers (Track B) is structurally correct. You recognized that crisis-diversifiers will fail absolute Sharpe thresholds but add massive value to a combined book at a 25% risk budget.
* **The Daily ML Fallacy:** You have correctly concluded that free daily US equity data is mined out for additive alpha. Standard machine learning models (XGBoost, LSTMs) applied to daily OHLCV data will find nothing but spurious correlations because any linear or non-linear predictive edge on that timeframe has been arbitraged away by multi-billion dollar statistical arbitrage desks.
* **The Data/Engine Mismatch:** Your primary architectural flaw is a mismatch between your validation engine and your data. You have a Ferrari engine (the CPCV pipeline) running on contaminated fuel. Relying on `yfinance` for your equity universe introduces fatal survivorship bias, automatically invalidating any cross-sectional or mean-reversion studies.

## 2. The Data Solution: What to Buy

Given your retail budget of $100–300/yr, you must patch the structural leaks in your data.

1.  **The Highest ROI Purchase: Norgate Data (~$270/yr).** This is non-negotiable. Norgate provides survivorship-free US equity history, including delisted names, with historically accurate index constituents. Without this, your long/short and event studies are quietly biased. Furthermore, Norgate provides clean, continuous futures data with proper roll schedules, which unlocks the carry premium you literally cannot test on free data.
2.  **Point-In-Time Fundamentals: Sharadar (Core US).** If you want to pursue cross-sectional equity strategies, you need PIT fundamentals. FMP's starter tier is insufficient for deep historical, survivorship-free quantitative research.
3.  **Halt Options ML:** Freeze your options research. Your Polygon 4-year snapshot is frozen, and generating greeks via Black-Scholes on delayed/indicative NBBO is a path to simulation bias. True options alpha requires live, tick-level surface data, which is completely out of a $300/yr budget.

## 3. Concrete Trading Strategy Ideas

Stop hunting for pure predictive alpha in price series and start harvesting structural risk premia and behavioral inefficiencies.

### A. Cross-Asset Trend & Carry (The AQR/Winton Approach)
Your 10-ETF TSMOM sleeve is your only survivor because trend following harvests a persistent behavioral risk premium. To expand this, you must move beyond equities.
* **The Strategy:** Trade time-series momentum and roll-yield (carry) across commodities, fixed income, and FX.
* **The Catch:** Alpaca has no futures execution. You must open an Interactive Brokers (IBKR) account. Use Norgate for the historical backtest, size positions using inverse volatility targeting, and combine the trend and carry signals.

### B. Statistical Arbitrage: Cross-Sectional Quality & Momentum
With Norgate removing survivorship bias and Sharadar providing PIT fundamentals, you can build a classic equity market-neutral (EMN) book.
* **The Strategy:** Rank the Russell 1000 cross-sectionally by a composite of fundamental quality (e.g., Gross Profitability, low accruals) and 12-month momentum (skipping the most recent month).
* **Execution:** Go long the top decile and short the bottom decile, beta-hedging to ensure zero market exposure. Your pipeline already has the residual-alpha CAPM/HAC diagnostic to test if this survives hedging out the market.

### C. Crypto Cross-Sectional Momentum & Basis
Alpaca supports crypto execution. Retail crypto markets remain highly inefficient compared to US equities.
* **The Strategy:** Build a cross-sectional momentum model on top-50 liquid altcoins. Alternatively, if you can access funding rate data (via free tier APIs from Binance/Bybit), build a strategy that harvests the persistent long-bias funding rate premium by holding delta-neutral short perpetual futures vs. long spot positions.

### D. Regime-Switching Risk Parity
Instead of trying to predict the *direction* of the market, predict the *volatility regime*.
* **The Strategy:** Utilize your existing macro cache (VIX, HYG, IEF) to build a Hidden Markov Model (HMM) that classifies the market into high/low volatility states. Use this to dynamically scale the leverage of a risk-parity portfolio (allocating risk equally between equities and treasuries).
