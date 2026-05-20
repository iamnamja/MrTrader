# STRATEGY REVIEW: MrTrader
**CONFIDENTIAL | QUANTITATIVE RESEARCH DEEP-DIVE | CRITICAL ASSESSMENT**

## 1. Executive Summary & The "Broken Simulator" Hypothesis
As requested, I have reviewed the MrTrader system architecture and walk-forward results. I will be brutally honest: **you are currently front-running massive backtest bugs.** Do not deploy live capital, even paper capital, until the simulator is fixed.

> **RED FLAG:** The reported metrics (Sharpe 8.1 for PEAD, total returns of 38x, and Max Drawdowns of 0.01% on QualityShort with a 21% win rate) are mathematically impossible for a realistic unlevered long/short equity strategy.

The core issue lies in your `AgentSimulator`. Based on the provided metrics, your backtest engine suffers from severe structural flaws:
* **The Drawdown Impossibility:** QualityShort has a 21% win rate, a 5-20 day hold, yet a Max Drawdown of 0.01%. This proves the simulator is **ignoring unrealized mark-to-market (MTM) losses**. It is likely only calculating equity at trade closure, or there is an egregious look-ahead bias preventing losing trades from entering the equity curve during their lifespan.
* **The Capital Reinvestment Bug:** 38x total return ($100k to $3.8M) in one fold implies unbounded leverage or a compounding bug. If your `MAX_POSITION_SIZE_PCT = 5%` uses *current account equity*, an unconstrained simulator might be sizing trades based on unrealized ghost capital, or overlapping trades are exceeding 100% gross exposure without triggering margin calls.
* **Survivorship Bias:** Using today's Russell 1000 for a 2021 backtest means the worst possible short candidates (those that went bankrupt or delisted) are excluded. While you noted this, the magnitude of this exclusion entirely invalidates the short strategy's backtest.

---

## 2. Responses to Specific Quant Questions

### Data Integrity
**1. Is Yahoo Finance acceptable for Russell 1000 daily OHLCV?**
No. YF is prone to survivorship bias, missing corporate actions, and dividend adjustment errors. For a PEAD strategy relying on gap opens and daily precision, you must use Norgate (for delisted constituent tracking) or CRSP/Polygon.

**2. Is FMP `earnings-surprises` data truly PIT-safe?**
No. The `date` field is the announcement date, but historical API calls return the database entry, which is often populated hours or days later. You are assuming instantaneous data availability that didn't exist in reality.

**3. Most-recent-quarter vs TTM for FMP flags?**
If using Q-statements, it is MRQ. However, YoY revenue growth using raw Qs can be highly seasonal. TTM (trailing twelve months) is standard practice to smooth seasonality.

**4. Latency of FMP API?**
FMP can lag press releases by 30 minutes to 4 hours. By the time it hits the API, algorithms scraping PR wires (Bloomberg, Refinitiv) have already priced in the surprise.

**5. Look-ahead from filingDate vs acceptedDate?**
Huge risk. If `filingDate` is missing and you fall back to `period_end + 45d`, you are fabricating a release date. If the company actually filed at `period_end + 60d` (late filer), you are trading on data you couldn't possibly have known yet.

### Walk-Forward Validity
**6. Is 5 folds x 1-year window adequate?**
It's acceptable, but an embargo of 10 days for a 5-day hold is borderline. I recommend a 20-day embargo. Autocorrelation in factor momentum lasts ~1 month.

**7. Does purge_days=10 adequately prevent leakage?**
Yes, for a strict 5-day hold, 10 days purge prevents direct overlap. But it doesn't prevent macroeconomic regime leakage.

**8. Is the 8.1 Sharpe a code bug?**
100% a code bug. Standard post-cost PEAD Sharpe is 0.8 to 1.2. Sharpe 8 implies a straight 45-degree equity curve. Look for: double-counting returns, summing percentage returns instead of dollar returns, or missing MTM pricing.

**9. Fold 2 (2022) Sharpes highest - suspicious?**
For QualityShort, 2022 was indeed a short-seller's paradise, so outperformance is logical. For PEAD, high volatility regimes often increase dispersion. However, the *magnitudes* remain fictitious.

**10. Multiple-testing correction for 17 configs?**
Use the Deflated Sharpe Ratio (Bailey & Lopez de Prado) using `N=17` and track the variance of the tested Sharpes. However, fixing the simulator bugs will likely reduce your Sharpe below 1.0, making DSR irrelevant for now.

### PEAD Specific
**11. Why is PEAD Sharpe 5-10x higher than academia?**
Primary causes: (a) Total return compounding bug, (b) entering at "next day open" misses the fact that BMO (Before Market Open) earnings gap *that day*, not the next day. If your simulation applies today's BMO earnings to tomorrow's open, you have a massive one-day lookahead.

**12. Total returns 11-38x plausible?**
Mathematically impossible without massive, continuous leverage. If risk limit is 5% per trade, 20 winners in a row yield roughly 100% return, not 3800%.

**13. Is 5 bps/side realistic for next-day open?**
No. Russell 1000 mid/small caps have open auction slippage of 15-30 bps. Round trip costs should be modelled at 40-50 bps total.

**14. Delay entry by 1 day?**
PEAD decays logarithmically. Day 1 captures 50% of the drift. Day 2-5 captures the rest. Delaying ruins the trade.

### Quality Short Specific
**15 & 16. QS win rate 21% vs Sharpe, Borrow Costs**
A 21% win rate means you lose 4 out of 5 trades. The 0.01% max DD confirms MTM is completely broken. Realistic borrow costs for HTB names (declining fundamentals) are 8% to 25% annualized. At 5 days, that's 10-30 bps purely in borrow drag per trade.

**17. Exposure to short squeezes?**
Extreme. Your universe excludes delisted/bankrupt names, removing the true tail risk (e.g., heavily shorted names that squeezed and subsequently failed or were acquired). You are testing on a sterilized universe.

### Portfolio Construction & Risk Management
**18. Concurrent open positions and risk budgeting?**
Earnings are highly clustered. Weeks 2-4 of earnings season will easily generate 30+ signals daily. If you cap at 5 or 20, you face massive execution selection bias. Which of the 30 signals do you take? If random, your live path will diverge wildly from the backtest.

**19. De-duplication between legs?**
You must implement a position manager that nets exposure. Being long and short the same name burns 10 bps in spread/tx costs for zero net exposure.

**20. PEAD regime gating?**
Correct. PEAD is relatively cross-sectionally neutral and relies on idiosyncratic events; it does not need a strict SPY>MA200 gate. Factor longs definitely do.

### Paper Trading Plan
**24. Is 3 months sufficient?**
No. 3 months covers exactly one earnings cycle. You need 6-9 months to capture different macro backdrops and sufficient sample size.

**25. Expected paper Sharpe vs WF?**
If you fix your bugs and WF Sharpe drops to 1.5, expect paper Sharpe of 0.7 to 1.0.

**26. Kill-switch recommendations?**
Set a hard stop at -12% account equity DD, or 5 consecutive days of limit-down gross exposure contraction. Monitor borrow utilization daily.

---

## 3. Probability Estimate: 0% Chance of Success in Current State
You asked for a calibrated probability estimate. In its current state, the probability that this system achieves a live paper Sharpe > 0.50 is **exactly 0%**. The underlying simulator is structurally compromised.

## 4. Actionable Next Steps (Immediate Prioritization)
1.  **Fix the Mark-to-Market Bug:** Halt all signal research. Open `AgentSimulator` and rewrite the equity curve generation. It MUST mark every open position to the daily close price, including accrued borrow costs, and calculate daily total portfolio equity. Ensure Max Drawdown is calculated on this daily MTM equity, not trade-level PnL.
2.  **Audit the Position Sizing / Cash Management:** Ensure cash is strictly deducted upon entry. A $100k account cannot allocate $10k to 15 different trades simultaneously.
3.  **Implement Norgate Data:** You cannot run short-side research on a static, survivorship-biased universe. Purchase Norgate Data ($700/yr) to get historical point-in-time index constituents and delisted tickers.
4.  **Model Realistic Costs:** Hardcode transaction costs to 15 bps per side (to simulate open auction slippage) and borrow costs to 10% annualized for shorts.
5.  **Manual Trade Audit:** Dump the OHLCV and trade entry/exit dates for the top 5 most profitable trades in Fold 2. Trace them manually in a spreadsheet to verify exactly how the PnL was calculated.

*End of Review - Authored by Senior Quant Reviewer*
