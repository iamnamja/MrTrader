# Brutal Quant Assessment — MrTrader System

**From:** A senior quant (ex-hedge fund, built & shipped multiple live algo systems)  
**To:** MrTrader team  
**Date:** 2026-05-18

---

## Executive Summary

You are grappling with the classic problem: **signal discovery in equity long-only space with minuscule capital**. I will be direct: the current architecture is unlikely to produce real, net-of-cost alpha. The cross-sectional ranking approach (LambdaRank) was always a poor fit for a long-only $20k portfolio, and the walk-forward numbers you’re seeing—persistent negative Sharpe in bear regimes—are telling you something structural, not a tunable hyperparameter problem. The sudden pivot to a factor portfolio with a 1.335 backtest Sharpe is a dangerous distraction; I’d bet my own money that backtest is contaminated by look-ahead bias.

That said, you’ve built a lot of solid infrastructure. The challenge now is to stop adding complexity and start being ruthlessly honest about whether you have any edge at all. Below I dissect every piece, answer your specific questions, and give you a prioritized action plan. **My tone will be blunt. I’m not here to validate effort; I’m here to save you months of wasted time.**

---

## 1. Model Training Pipeline Assessment

### Labels
- **Triple barrier with ATR multiples:** The problem you identified—barriers widening in high volatility, causing label collapse—is real and well-known. In a bear market, stops become rarely hit (since you’re using a tight -0.5× ATR stop) and targets are unreachable. The result is massive class imbalance and a loss of information. The correct fix isn’t to abandon triple barrier, but to use **absolute percentage thresholds** (e.g., +8% target, -5% stop) or volatility-scaled thresholds that adapt to the current VIX but not raw ATR. Many production systems use a fixed return threshold independent of ATR; that works fine if you control position sizing by volatility separately.
- **Cross-sectional quintile labels (LambdaRank):** This is the root of your Fold 2 disaster. In a -25% bear market, the “top 20%” are stocks that lost the least. Training a model to pick those teaches it to love low-beta, defensive stocks—exactly the opposite of the momentum/quality signals that work in bull markets. A single model cannot learn contradictory patterns across regimes. You *must* separate regimes or use labels that are regime-invariant (e.g., absolute return > +3%, or binary directional forecast). Cross-sectional ranking is appropriate only if you can go long/short market-neutrally; as a long-only investor, it’s at best misleading, at worst guaranteed to fail in down markets.
- **What you should have tried:** Absolute forward return binary labels (1 if 10d return > +4%, 0 otherwise) with careful threshold selection to maintain a balanced class ratio. Use a fixed threshold, not a function of ATR. This preserves signal in all volatility regimes and allows the model to learn “will this stock go up meaningfully?” regardless of relative performance.

### Feature Engineering
- Your 17 features are reasonable, but the IC study that claims IRs of 1.99 for 12-month momentum and 1.87 for VIX z-score smells like in-sample overfitting. Did you compute IC on the entire 6-year period *before* any train/test split? Those figures are likely inflated. Nonetheless, momentum, volatility, and quality are standard factors; they probably have some predictive power gross of costs, but net is another story.
- Sector-neutral features being dropped because HPO “landed in a worse basin” is not evidence they have no value. It’s evidence your HPO is unstable and your search space is too small. Do not discard a feature because one Optuna run didn’t like it. Run an ablation study with proper cross-validation to measure marginal impact.
- The TSNormalizer (z-score with trailing mean/std) is correct if fitted only on training data and applied PIT. You confirmed that bug is fixed; good.

### ML Architecture
- **XGBoost binary classifier with AUC 0.49–0.53:** That’s exactly what you’d expect from a dataset where the signal is very weak relative to noise. 0.50 is random. You’re not going to magically get AUC 0.55 with a few extra features. The problem is likely not the model but the label definition and the inherent difficulty of the task.
- **LambdaRank:** The model itself is fine, but the objective (ranking relative returns) is mismatched with your long-only execution. Even if it perfectly ranks stocks, you can only buy the top N; if the top quintile has negative returns, you lose money. NDCG@K is a ranking metric; it does not guarantee profitability. You observed a 0.1–0.3 correlation between NDCG and Sharpe—that’s typical. Ranking well doesn’t guarantee the top bucket is positive.

**Verdict:** The pipeline is not fundamentally broken, but the label design and objective function are misaligned with the investment problem. You need absolute return forecasting, not cross-sectional ranking.

---

## 2. Walk-Forward Methodology Honesty

Your walk-forward framework has the right structure: expanding windows, purge/embargo, transaction costs, PIT regime scores, delisted symbols, feature caching. This is above average for a retail system. However, there are critical honesty gaps:

### Execution simulation
- **Entry price = previous close × 1.001:** This is a joke. You have daily OHLCV from yfinance—you **have the actual open price**. Why use a constant multiplier that always gives you a 0.1% slippage loss? Use the real next-day open. The constant gap artificially penalizes your returns and may mask genuine alpha. Worse, if your features are computed on closes, using the next open is a legitimate test of tradability. Simulate exactly as you would trade: enter at open, exit at close (or intraday if you had that data). Remove artificial frictions.
- **ATR-based stops and targets fired on same-day close:** In a daily bar simulation, if the low of the day hit your stop, would you have been stopped out? You only check close. This underestimates stop-outs and overestimates profitability. In reality, a stop loss would trigger intraday if price touches the level. With only OHLCV, you must check whether the day’s low breached the stop. If so, assume exit at the stop price (or worse, if gap). Your current method gives you an unrealistically high chance of staying in a trade that should have been stopped out. This bias could be substantial, especially in volatile periods.
- **Kelly-like position sizing based on confidence:** At $20k, sizing is tiny. The main issue is whether the sizing logic is PIT. If confidence is derived from the model score on that day, that’s fine. But using ATR for sizing is reasonable.

### Data integrity
- **Survivorship bias:** You mentioned `pit_union("russell1000", fold_start, fold_end) + DB historical symbols`. This is good in theory, but I doubt your historical symbol database is complete. How do you ensure that a stock that existed in 2020 but was delisted by 2021 is included in the 2020 fold with correct prices? You need a true point-in-time universe membership and price history for all constituents, including delisted ones. Using yfinance + Alpaca may not give you that. Many stocks that went to zero or were acquired are missing from standard APIs. If your backtest universe is even slightly biased toward survivors, your Sharpe is overstated.
- **Fundamental data lag:** You fetch quarterly fundamentals via FMP. Are you lagging them appropriately? If a company reports Q1 earnings in late April, the features `profit_margin`, `pe_ratio`, etc., must not be used until after the report date. If you naively use the value at the end of the quarter, you’re peeking 30-45 days into the future. This is the most common source of spectacular backtest results. I strongly suspect your “factor portfolio” 1.335 Sharpe suffers from this.

### Gate criteria
- Your gate of avg Sharpe ≥ 0.80 and min fold ≥ -0.30 is sensible, but the bar is very high for a long-only strategy with 5-20 stock portfolios and realistic costs. Most institutional long-only factor strategies struggle to achieve net Sharpe 0.5. Expecting 0.8 might be unrealistic.

**Verdict:** Your WF is more honest than 90% of retail attempts, but the execution simulation is flawed. Fix it before drawing any conclusions.

---

## 3. Trade Entry/Exit Logic Reality Check

With $20k, you are a price-taker. Market impact is zero. The only frictions that matter are commissions (Alpaca has none) and the **bid-ask spread**.

- **5 bps per side for swing:** Reasonable for large-cap Russell 1000 stocks if you use market orders. The average spread for a $50 stock with 0.01% spread is about 0.02-0.05%. So 5 bps might be a bit high, but okay as a conservative estimate.
- **15 bps per side for intraday:** This is optimistic for 5-minute bar trading of 100 liquid names. Even highly liquid stocks have spreads of 2-4 bps; your cost includes slippage from market orders and the fact that you’re trading mid-price signals but crossing the spread each time. 15 bps one-way is plausible, but if turnover is high, round-trip costs eat 30 bps per trade. Can your intraday signals overcome 30 bps friction with 0.5-2% targets? It’s possible, but you need a very high win rate.

- **Stop and target logic:** In reality, a stop-loss order resting at the exchange triggers when the bid (for a long) hits that price. You cannot guarantee execution at exactly your stop level; gaps overnight can cause slippage. Your simulation must account for overnight gaps. Since you only trade daily bars and hold overnight, you should assume the worst-case fill if the next day’s open is through your stop. With OHLCV, you can approximate by comparing the stop to the next open; if open < stop, exit at open with slippage.

**Verdict:** Fix the stop simulation. Use actual open for entry. Model spreads explicitly.

---

## 4. Overall Strategy Design Critique

### Long-only, $20k, Russell 1000
This is a brutal combination. The signal-to-noise ratio for daily equity selection is extremely low. Any edge you find will be small and fragile. With only 5-20 positions, diversification is minimal; you’re making concentrated bets. The strategy’s fate hinges on a few trades.

You’re trying to beat the market by picking a small subset. That’s stock-picking, not systematic factor investing. Systematic strategies typically need hundreds of positions to wash out idiosyncratic noise and capture factor premia. At $20k, you cannot trade 500 stocks profitably due to costs and operational overhead. This is a fundamental structural mismatch: the methodology that works (broad diversification) is infeasible, and the feasible approach (concentrated bets) is too noisy to be systematic.

### Swing + Intraday
Having two separate strategies is fine, but both share the same capital pool. The RM limits positions to 5 total. If both are running, they will compete for slots, potentially causing conflict. More importantly, intraday trading with 5-minute bars on 100 stocks while you have $20k is mathematically possible, but the edge must be enormous to overcome costs. The fact that your best intraday model has a Sharpe of 0.529 (below the 1.0 gate) suggests there is some signal, but not enough to justify the complexity. I’d drop intraday entirely and focus on swing.

### Ancillary systems (NIS, regime model)
These add layers that can mask overfitting. The regime model is a binary classifier trained on macro data; if its PIT implementation is correct, it’s fine. But the “Opportunity Score” that gates intraday scans with VIX, SPY vs MA20, etc., is essentially a rule-based tactical overlay. That’s okay as a risk management layer, but it introduces discretionary elements that can be curve-fit. Be aware that any combination of conditions reduces the number of trades, which inflates Sharpe in backtest if you accidentally select favorable periods.

---

## 5. Pivot to Factor Portfolio — Is It the Right Call?

**No.** This is a knee-jerk reaction to ML frustration. Here’s why:

- The dedicated backtest Sharpe of 1.335, CAGR 32.4%, worst year +4.6% (in 2022!) is **impossibly good** for a long-only rule-based factor portfolio of 20 stocks. If it were real, you’d be the best fund manager on earth. The fact that this portfolio loses massively when run through your (more realistic) AgentSimulator with ATR stops tells you the original backtest is not just an execution mismatch—it’s likely contaminated by forward-looking bias. The monthly rebalance and no stops would not turn a -1.43 Sharpe into +1.335; a bad signal plus no stops should still show a poor equity curve, just with larger drawdowns. The drastic reversal suggests the signals themselves are garbage when evaluated point-in-time.

- The factor portfolio uses **fundamental data that probably isn’t lagged**. I’d put a large wager on this. Profit margin, operating margin, PE ratio—these are available only after earnings reports. In a “dedicated backtest,” it’s trivial to use the current known value without lag, producing perfect foresight. This is the number one kill-joy in factor backtesting. You must ensure that for any date, you only use data that would have been known on that date, with the correct reporting lag (e.g., for a company with fiscal Q1 ending March, data becomes available in late April; so from Jan-Mar you still use Q4 data). Without PIT fundamentals, every backtest is a fairy tale.

- Even if the factor portfolio’s signals were genuine, executing it live with ATR stops that were not part of the validation is insane. You’re combining a validated strategy with an execution mechanism that fundamentally changes its return distribution. If the factor portfolio relies on holding through drawdowns to capture 20-day momentum, tight stops will kill it. You wouldn’t take a long-term trend-following model and apply a 2% stop; that’s a different strategy.

**My advice:** Immediately suspend live paper trading of the factor portfolio until you have validated it in the exact same execution environment (AgentSimulator, no stop first, then with realistic stops) and confirmed no fundamental data peeking. If it falls apart, you’ll have saved yourself from a dangerous deployment.

---

## 6. Answers to Your Specific Questions

**Q1: Is LambdaRank the right architecture?**
No. For long-only, you need a model that predicts absolute forward return, not relative rank. If you still want to use a ranking model, you must combine it with a market timing overlay that prevents you from buying anything when the whole market is falling. Even then, ranking the “best losers” is a losing game. Drop LambdaRank entirely. Switch to a binary classifier with an absolute return threshold (e.g., forward 10d return > 5% = 1, else 0). Train separate models per regime if needed.

**Q2: Is our walk-forward honest?**
Mostly. Major honesty gaps:
- Stop-loss simulation only checks close, not intraday lows.
- Entry price fabrication (1.001 multiplier) instead of actual open.
- Fundamental data lag not proven to be PIT.
- Survivorship bias in historical universe likely incomplete.
- No explicit modeling of the bid-ask spread; you assume a fixed bps cost, but that’s an approximation.
Fix these, then re-evaluate.

**Q3: How do pros handle bear-market labeling?**
Standard approaches:
1. **Absolute return thresholds:** Label = 1 if return > +X% (e.g., 5%), irrespective of market. This naturally produces fewer positives in bear markets, which is honest; the model learns that it’s harder to find winners. You may need to adjust threshold to maintain a workable class balance.
2. **Regime-conditional models:** Train separate models on bull/bear subsets, or include regime features and interact them with other features. A single model can learn interactions if it has enough data.
3. **Market-neutral relative labels:** Compute each stock’s return minus the market return, then threshold. This predicts outperformance, but as a long-only investor, you still need the market to not tank. So you’d need a market beta hedge or explicit direction forecast. Most long-only shops simply accept that their strategy will lose money in bear markets; they rely on asset allocation to reduce exposure.
4. **Multi-task learning:** Predict both direction (up/down) and magnitude, then combine. But the simplest and most robust is absolute threshold with regime-aware features.

**Q4: Factor portfolio vs ML given constraints?**
A rule-based factor portfolio is not inherently better; it’s just simpler to validate if done correctly. The mistake is comparing a overfit factor backtest to an underperforming ML pipeline. Once you fix data lags and execution, both might show zero edge. At $20k, the real question is: can any active strategy beat a simple SPY buy-and-hold after costs? Probably not. If you insist on trying, the ML approach is more likely to capture nonlinear interactions, but only if you solve the label and cost problems. I’d pursue ML with a low complexity model (e.g., logistic regression with Ridge penalty) first, then only add XGBoost if signal is confirmed.

**Q5: Is execution realistic?**
Partially. At $20k, your orders will be tiny (100-200 shares of most stocks). You can use market orders and get filled at near the quoted price. The main issue is that your simulation doesn’t account for the actual open price (you have it, use it) and doesn’t correctly stop out on intraday lows. A retail system should assume:
- Entry at next open (actual open price, not a synthetic one).
- Exit at stop if the day’s low ≤ stop price, with fill at stop price (or open if gap down below stop).
- Costs: half-spread + commission. For Alpaca, commission is $0, but spread is about 0.01-0.05% per side. Model this dynamically using a proxy like (high-low)/close if you don’t have tick data.

**Q6: Intraday at $20k viable?**
Mathematically, it’s extremely difficult. The main constraint is the bid-ask spread. If you target 1% gross return per trade and costs are 30 bps round-trip, you need a win rate and average win size that gives positive expectancy. With a 5-minute holding period, the signal-to-noise ratio is terrible. The fact that your best model has a Sharpe of 0.529 but you need 1.00 to overcome costs suggests you’re not close. Additionally, intraday models often suffer from overfitting to micro-noise. At $20k, the additional operational complexity is not worth it. I’d kill intraday entirely.

**Q7: What would I build next?**  
See prioritized action plan below.

---

## 7. Actionable Next Steps (Priority Order)

Assume you still want to find edge in this space. Here is the critical path:

### Step 1: Stop everything and fix backtest execution realism (1 week)
- **Entry:** Use actual open prices from yfinance (already available). Remove the 1.001 multiplier.
- **Stop simulation:** Check if the day’s low ≤ stop. If yes, assume exit at min(open, stop) or the stop price, whichever is more conservative. For gaps down, exit at open with a penalty.
- **Costs:** Explicitly model half-spread. For each stock, estimate spread as a percentage of price using (high-low)/close as a proxy for daily range, then a fraction (e.g., 0.1× range) as half-spread. Add this to each trade.
- **Fundamental data lag:** Implement a hard lag: for quarterly data, use the filing date or assume a conservative 45-day delay after quarter end. Validate that no future information leaks into the feature matrix.
- **Survivorship:** Acquire a true point-in-time list of Russell 1000 constituents for each date (use an index provider dataset or carefully reconstruct from Wikipedia/EDGAR). Your current method is inadequate.

**Do not run any new ML until this is bulletproof.**

### Step 2: Re-evaluate the factor portfolio honestly (2 days)
- Run the factor portfolio through your *corrected* AgentSimulator using **two configurations**:
   a) Hold for exactly 20 days, no stop/target (mimic the monthly rebalance logic but daily to see signal purity).
   b) With no stop, but rebalance to top-N every day.
- If the Sharpe is near zero (or negative), the factor signal is fake. Find and eliminate any look-ahead bias in fundamentals. If you can’t, discard the factor approach.
- If the Sharpe is modestly positive (e.g., 0.2-0.4), then the signal might be real, but the ATR stops destroyed it. In that case, you can consider using the factor as a signal but with a wider stop or no stop and simply holding for a fixed period.

### Step 3: Redesign the ML label and objective (1 week)
- **Label:** Binary, based on whether forward 10-day return exceeds an absolute threshold (try 4%, 5%, 6%) and does not hit a stop-loss of -5% (absolute) during the period. This creates a realistic “win” condition.
- **Model:** Start with logistic regression (or LGBM with binary objective) on your 17 features. No ranking. Use class_weight if imbalanced.
- **Validation:** Walk-forward with the corrected execution assumptions. Evaluate pure signal by entering at open next day, exiting at close after N days (no stops) to see if the model adds value.
- If the model’s signal (without stops) is not significantly better than random, stop. You have no edge.

### Step 4: If signal is found, layer on risk management (1 week)
- Once you have a model with a positive expected return per trade (gross of costs), add realistic stops and position sizing.
- Test different exit logic: time-based exits, trailing stops, profit targets. Simulate them honestly using daily high/low.
- Optimize the number of positions. With $20k, 5 positions may be too diversified given the signal strength. Consider top-3 or even top-2.

### Step 5: Abandon intraday entirely (immediate)
- Your intraday Sharpe is far below the gate. The cost model is generous, and you lack tick data to properly model execution. This is a distraction. Shelve it.

### Step 6: Lower your expectations (mental shift)
- A net Sharpe of 0.5 after costs on a concentrated long-only portfolio is world-class. You’ve been chasing 0.80 and 1.00. Those are unrealistic for this capital level and data frequency.
- The gate should be: positive expected value with a Sharpe significantly above buy-and-hold SPY. Compare your strategy’s equity curve to a passive benchmark.

### Step 7: If all fails, accept the brutal truth
- It is entirely possible that there is no harvestable alpha in daily stock selection with your constraints. The market is highly efficient, and your small account cannot implement the diversified, low-cost strategies that institutions use. In that case, the correct decision is to stop trying to beat the market and instead invest passively. The educational value of building this system is immense, but don’t let it become a money-losing exercise.

---

## Closing Thoughts

You’ve done an impressive amount of engineering, and your awareness of common pitfalls (purge, embargo, PIT) puts you ahead of most retail quants. But you’ve fallen into a classic trap: **building complex solutions before validating the simplest possible hypothesis.** Always start with a linear model, a single label definition, and the most realistic simulation possible. If that shows a glimmer of edge, then scale complexity.

Right now, your system is a collection of interesting parts that don’t add up to a profitable whole. Tear it down to the bare metal, fix the data and execution lies, and ask: *Does any signal survive?* If the answer is no, walk away with the knowledge you’ve gained. If yes, you’ll have a foundation worth building on.

Go fix those backtest assumptions. Then let the numbers talk.

— A quant who’s been burned by his own backtests more times than he can count.