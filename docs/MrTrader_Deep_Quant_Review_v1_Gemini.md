# Deep Quant Review: MrTrader Architecture & Methodology

**To:** MrTrader Quantitative Trading Team
**From:** Senior Systematic Portfolio Manager
**Date:** May 18, 2026
**Subject:** Brutal Assessment of Trading Architecture & Walk-Forward Validation

I have reviewed your methodology, architecture, and the specific bugs you're encountering. You asked for a brutally honest assessment from the perspective of someone who has built and shipped institutional systematic trading models. 

Here is the hard truth: **Your system is suffering from an identity crisis.** You are attempting to run institutional-grade cross-sectional ranking (stat-arb) architecture on a retail account ($20k) with retail constraints (long-only) and retail data (yfinance). 

The good news is that your engineering pipeline, walk-forward mechanics (embargo/purge), and factor ideas are actually quite advanced for an independent team. Your failures are not code bugs; they are **structural financial logic errors**.

Below is my detailed teardown of your architecture, direct answers to your questions, and the exact steps you need to take to unblock this system.

---

## 1. The Elephant in the Room: Account Size & Intraday ($20k)

Before we talk about machine learning, we need to talk about market structure. You have a $20k account and you are trying to run a systematic intraday strategy (closing same day) in the US market (Alpaca).

**You are violating the Pattern Day Trader (PDT) rule.**
Any US brokerage account under $25,000 cannot execute more than 3 day trades in a rolling 5-business-day period. If your intraday system selects "2-5 stocks per day", Alpaca will hard-lock your account within 48 hours. 

* **The Fix:** You must completely pause the Intraday strategy. It is mathematically and legally un-tradeable at your capital level. Furthermore, 15 bps of slippage/commission on a target of 0.5% means you are giving up 30% of your gross alpha to the spread. At $20k, you are the liquidity provider's lunch. **Kill the intraday system immediately and focus 100% on Swing.**

---

## 2. The Fallacy of Long-Only Cross-Sectional Ranking (LambdaRank)

Your LambdaRank implementation is structurally flawed for your mandate, which perfectly explains your "Fold 2 (2022 Bear Market) Problem."

LambdaRank is designed to rank a universe of stocks. In a bull market, the top 20% go up 15%, and the bottom 20% go up 2%. In a bear market (-25%), the top 20% go down "only" 5%. 

Because you are **Long-Only**, buying the top 20% in a bear market means you are systematically buying assets that lose 5%. 

Worse, the features that predict "falling least" in a bear market (low beta, high dividend, defensive sectors) are the exact *opposite* of the features that predict "rising most" in a bull market (high momentum, high volatility). By forcing the model to learn a single ranking function across both regimes, you are confusing the tree splits. 

* **Why Institutional Quants use Rankers:** We use rankers because we are **Market Neutral**. We buy the top 20% and *short* the bottom 20%. If the market drops 25%, our longs drop 5% and our shorts drop 35%, generating a massive net positive return.
* **The Fix for Long-Only:** You cannot use purely relative cross-sectional ranking for long-only unless you have an aggressive, binary cash-toggle (Regime Model) that moves you to 100% cash in bear markets. Alternatively, you must switch to an **Absolute Return ML objective** (e.g., probability that $R_{t+20} > 3\%$). 

---

## 3. The Execution Mismatch Mystery (Sharpe 1.335 to -1.43)

You deployed a Factor Portfolio that validated at a 1.335 Sharpe using a **monthly rebalance, equal-weight, no-stop** framework. When you ran it through your live AgentSimulator with **daily ATR stops**, the Sharpe went to -1.43.

**This is not a bug; this is market microstructure mechanics at work.**
Factors like Value (P/E ratio) and Quality (Margins) are notoriously poor at entry timing. They identify structurally sound companies, but those companies routinely experience random walk drawdowns of 1-2 ATRs before realizing their factor premium over a 20-30 day horizon.

When you add a tight ATR stop (0.5x) to a monthly factor model:
1. You crystallize the temporary volatility into permanent losses.
2. You get shaken out of the position *right before* the mean reversion happens.
3. You completely sever the statistical edge the factor model found over the 30-day holding period.

**The Fix:** Your backtest execution MUST match live trading. If you want to use ATR stops, you must optimize your factors *with the stops active in the backtest*. However, I strongly advise against trailing stops on factor portfolios. Factor edges are harvested over time, not through precise entries. Use time-based exits (e.g., hold 20 days, close) or signal-decay exits (close when factor rank drops below top 50).

---

## 4. Honest Walk-Forward Assessment

Your walk-forward framework (expanding window, embargo, purge) is conceptually sound, but your **execution simulation is entirely dishonest.**

You simulate entry as: `previous close × 1.001`

**This is lethal.** You are assuming you can enter at the open using yesterday's close plus 10 basis points. In reality, news happens overnight. If your model likes a stock because of an earnings beat, it will gap up 4% at the open. Your backtest assumes you bought it at yesterday's close + 0.1%, gifting you 3.9% of phantom alpha.

* **The Fix:** You must use the actual `Open` price of the next day for your entry, and add 5-10 bps of slippage *on top of the Open price*. If using `yfinance`, be aware that historical Open prices are heavily adjusted for splits, making gap analysis difficult. 

Furthermore, `yfinance` has massive **survivorship bias**. You mentioned adding historical DB symbols to fix this, but if your feature engineering relies on accurate, point-in-time constituent lists (e.g., "was this stock actually in the Russell 1000 on March 4, 2021?"), you are likely leaking future inclusion data. 

---

## 5. Direct Answers to Your Questions

**1. Is LambdaRank the right architecture?**
No. It is for Long/Short market-neutral portfolios. For Long-only, you need absolute return thresholds (XGBoost/LightGBM binary classifier targeting absolute gains) paired with a regime filter.

**2. Is our walk-forward methodology honest?**
Your data partitioning is honest. Your execution simulation (`prev_close * 1.001`) is deeply flawed and likely hiding massive slippage and gap-risk.

**3. The Fold 2 (2022 bear) problem — how do pros handle it?**
We short. Since you cannot short, you must use a Regime Model as a hard gate. If Regime = RISK_OFF, cash allocation = 100%. Do not try to force a long-only ML model to "find the winners" in a crash; the math doesn't support it.

**4. Factor portfolio vs ML at $20k?**
Stick to the Factor Portfolio. ML on daily bars with 750 stocks is incredibly noisy. Rule-based factor portfolios (combining Value, Momentum, and Quality) are far more robust on low-frequency (daily) data. You are overcomplicating the math for the data quality you possess.

**5. Execution model realistic?**
No. $20k retail needs low turnover. Ditch the daily ATR stops. Move to a weekly or bi-weekly rebalance. Enter at the TWAP of the first 30 minutes, or just take the explicit Open price + slippage.

**6. Intraday at $20k viable?**
Mathematically illegal (PDT rule) and statistically unprofitable due to spread/commission drag on small absolute targets. Kill it.

**7. What would you build next? (See Action Plan below)**

---

## 6. Actionable Next Steps (Priority Order)

**Step 1: Reconcile the Execution Mismatch (Days 1-3)**
Take the Factor Portfolio (Phase D). Strip out the PM's ATR stops in your AgentSimulator. Run the walk-forward using the exact same rules as the backtest: Enter at `Open`, hold for X days (or until rebalance), exit at `Open`. See if the Sharpe reverts to ~1.3. If it does, your factor edge is real, and your stops were destroying it. 

**Step 2: Fix Simulator Pricing (Day 4)**
Change your entry fill logic in the AgentSimulator.
* *Current:* `fill_price = prev_close * 1.001`
* *New:* `fill_price = today_open * (1 + slippage_bps)`
Re-run your baselines. Prepare to see your Sharpe drop. This is the "honest" baseline.

**Step 3: Implement Absolute Return XGBoost (Days 5-10)**
If you insist on ML, abandon LambdaRank. Implement XGBoost targeting a Triple Barrier, but use **Absolute** targets, not ATR targets.
* Target = +4% from Open
* Stop = -3% from Open
* Max Hold = 15 days
* Label = 1 if Target hit first, 0 otherwise.
This stabilizes the labels during high volatility regimes. 

**Step 4: Hard-Gate the Regime**
Your NIS (News Intelligence) and Regime Model should act as absolute capital allocators. If Regime = RISK_OFF, max positions = 0. Not 3. Zero. Protect the $20k capital.

### Final Word
You have a Ferrari pipeline but you're feeding it low-octane fuel (`yfinance`) and driving it in a school zone ($20k constraints). Simplify the execution. Match the live agent to the backtest exactly. Trust the factors over the complex ML for now.

Good luck,  
*Senior PM*
