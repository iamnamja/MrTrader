
Each concrete strategy (DailySwing, IntradayMomentum, PairsStatArb) implements these methods. The WF engine iterates through timestamps, calls the strategy, records fills, and computes metrics. This is similar to how backtesting frameworks like Zipline or QuantConnect work, but you retain full control.

**Intraday support** requires that the WF engine can step through *bars of any frequency*, not just daily. You already have an `intraday_agent_simulator.py` – it should be a strategy plug‑in, not a separate script.

**Pairs trading / stat‑arb** adds complexity because you need to simulate entry on a spread. That’s a matter of defining a `PairSignal` that knows how to generate orders for both legs simultaneously. The execution simulator must handle multi‑leg fills with realistic legging risk. Start with simple cointegration‑based signals and a `submit_spread_order` primitive.

**Options** – Forget about it in this 6‑month timeline. Options simulation requires an entirely different level of complexity (implied vol surface, early exercise, pin risk). If you must, integrate with a library like QuantLib and treat it as a separate, v2.0 project.

---

## 5. Agent Workflow Integration

The “simulate the exact agent stack” goal is correct, but the failure modes are subtle.

### Common Failure Modes
- **State divergence**: The live agent may receive data at a different time (e.g., before or after the open) than the simulation assumes. If your PM uses a signal that depends on today’s open price, and the live system runs at 9:31 AM while the sim uses “known open”, you introduce look‑ahead.
- **Decision branching**: In live, if an order is rejected by the broker (e.g., symbol not tradable), the RM may fallback to a different action. The simulation must model all possible exceptions.
- **Clock skew**: Market holidays, early closes – the simulation must use a proper trading calendar identical to the broker’s.
- **Agent memory**: If any agent carries state (e.g., “last signal strength”), that state must be reset identically in sim and live on restart.

### How Leading Shops Handle It
They build a **simulation harness** that:
1. Records all inbound data (market data, news, orders) from the live environment.
2. Replays that data into the exact same production binary, with a flag that switches the clock to “replay time”.
3. The output orders are captured and compared against what actually happened.
4. Any discrepancy triggers an alert.

Your current architecture (separate `agent_simulator.py` that reimplements the logic) is *almost certainly* diverging from the live code path. The only safe way is to have the live agents operate in a “paper trading” mode that can be fed historical data. A quick‑and‑dirty solution: make the live agents’ `act()` function accept a `data_provider` interface. In production, it’s real‑time; in simulation, it’s a replay provider. The logic stays identical.

---

## 6. Data Requirements (Ranked by Impact)

Here is the brutal ranking. I assume you have limited time and money.

| Rank | Dataset | Why | Expected Improvement |
|------|---------|-----|----------------------|
| **1** | **Intraday bid‑ask spreads (1‑min or 5‑min)** | This single dataset converts your backtest from “optimistic fiction” to a realistic cost estimate. Implementation: use Polygon.io’s aggregate bars with spread or estimate from high/low. | **Transformational** |
| **2** | **Short interest / borrow rates (daily)** | Your short P&L is completely fabricated without this. At minimum, use a tiered borrow cost table by market cap and historical short interest. | **High** |
| **3** | **Factor exposures (Barra/Axioma risk model or simple PCA)** | Instantly tells you if your alpha is just leverage on known factors. You can start with a freely available Fama‑French 5‑factor + momentum series. | **High** |
| **4** | **Corporate event calendar (earnings, dividends, splits)** | Avoid holding through binary events. A single earnings miss can blow through your wide ATR stop. Simple rule: no new entries 5 days before earnings, exit 1 day before. | **High** |
| **5** | **Options implied volatility (term structure)** | VIX alone is crude. IV skew and term structure give better regime signals, especially for short positions. Can be scraped from CBOE delayed data. | **Medium** |
| **6** | **Analyst estimates / earnings surprises** | Adds an event‑driven overlay. Can be used to bias the model away from stocks with negative revisions. | **Medium** |
| **7** | **ETF flow data** | A good crowding indicator, but noisy and often lagged. | **Low‑Medium** |
| **8** | **13‑F institutional holdings (quarterly)** | Too slow for a daily strategy. Might be useful as a slow factor. | **Low** |
| **9** | **Economic calendar (FOMC, CPI, NFP)** | Useful for switching off the system on macro event days, but your current market‑neutral approach shouldn’t be too sensitive. | **Low** |
| **10** | **Alternative data (satellite, credit card)** | Only if you have a dedicated data science team to extract signal. Not for a 1‑person shop. | **Out of scope** |
| **11** | **Level 2 order book** | Overkill for end‑of‑day execution. Only needed if you move to intraday market‑making. | **Out of scope** |
| **12** | **News sentiment (beyond NIS)** | You already have an NIS; improving it gives marginal gains until the core problems are solved. | **Nice to have** |

---

## 7. Label Design

**LambdaRank on cross‑sectional return rank** is a defensible choice for a long‑short equity strategy because it directly optimises the ordering that drives your portfolio construction. However, you are misapplying it.

### Failure Modes of Your Current Approach
1. **Label does not reflect trading reality.** You rank on “20‑day close‑to‑close return”, but your P&L is generated by a path‑dependent process with stops, targets, and a time cap. A stock that ranks in the top quintile because it had a huge gap up on day 19 will look great in the label, but in simulation you might have been stopped out on day 3 during a drawdown.
2. **Look‑ahead in cross‑sectional ranking.** If you compute the rank on day `t` using returns from `t` to `t+20`, you must ensure that no information from the future is used to construct the universe or features. You are doing point‑in‑time Russel 1000 membership, which is good. But what about survivorship bias? Stocks that delist before `t+20`? You must include them with a return of -100% (or worse).
3. **Loss function ignores tail events.** LambdaRank is pairwise logistic loss on relative order. It doesn’t heavily penalise large drawdowns. You can have a high AUC but still pick stocks that suffer occasional -20% gaps, which your risk management won’t fully prevent.

### What Top Shops Do
- **Use future returns that incorporate the intended exit logic.** For a 40‑bar max hold with an ATR trailing stop, the ideal label is the return from entry to the earlier of {stop hit, target hit, max hold}. This requires running a mini‑simulation for each stock during training, which is expensive. A practical compromise:  
  `label = min(return, stop_loss_return)` where `stop_loss_return` is the return of a synthetic stop condition (e.g., if low[t:t+20] ≤ entry * (1 - stop_pct), then stop_pct else actual return). This censoring aligns the model with the strategy.
- **Directly predict the probability of hitting the target before the stop**, or the expected risk‑adjusted return. This is often done with a binary classifier (meta‑labelling) on top of the ranking model.
- **Use a transformer or LSTM that consumes the whole price path** and outputs a score for each stock, trained on a loss that approximates the Sharpe ratio (differentiable sorting, or a reinforcement learning approach). This is overkill for your capital level.

### Your Specific Questions
- **20‑day labels vs 40‑bar hold:** The right horizon is the *median holding period* in your simulation. With 1.5×ATR stops and a 3×ATR target, the expected holding period for a winning trade might be around 15–25 days. 20 days is a reasonable first guess, but you must empirically measure it and set `FORWARD_DAYS` to the nearest multiple of your step frequency.
- **LambdaRank batch scoring vs live pointwise scoring:** There is no inherent distribution mismatch because the model learns a scoring function `f(x)` that is applied pointwise. The batch is only for constructing pairwise losses during training. However, if you later only pick the top 5 scores, you are sampling from the extreme tail, which can expose miscalibration if the model is not well calibrated. A solution: calibrate the scores using isotonic regression on an out‑of‑sample validation set.
- **ATR stops not in training labels:** It is **inherently wrong** if the stops are tight enough to frequently truncate the return distribution. With 1.5×ATR stops, truncation is less frequent (maybe <20% of trades). For a first pass, accept the bias but be aware it will favour stocks with smooth upward trends and penalise volatile names that would actually get stopped out. The correct fix is to use a censored regression or survival model. If you want to stay with ranking, use the truncated return as the label.
- **5 positions – feature or bug?** **Bug.** The academic literature (e.g., “Optimal Portfolio Diversification?” by Lhabitant) and practical experience show that for a daily ML strategy with moderate information coefficient (IC ~0.03–0.05), you need at least 50–100 positions to achieve a reliable Sharpe. With 5 positions, the standard error of your Sharpe estimate is so large that a backtest of 6 years is meaningless. I’d insist on a minimum of 20 positions, equally weighted, then move to a risk‑parity sizing.
- **Expanding vs rolling window:** For a cross‑sectional model with 1 year of training data, **expanding window is wrong.** Market regimes change; 2018 data is irrelevant for predicting 2023. Use a **rolling window** of fixed length (e.g., 3–5 years) so that each fold uses only recent, regime‑relevant data. The purge gap prevents overlap. This is standard in academic studies (e.g., Gu, Kelly, Xiu “Empirical Asset Pricing via Machine Learning”).

---

## 8. Prioritised 6‑Month Roadmap

If I were your Quant PM, I would not allow a single dollar of live capital until Month 4. Here is the brutal plan.

### Month 1: Stop the Bleeding (Foundation Fixes)
- **Fix C1, C2 completely.**  
  Retrain v216 with labels based on `open[t+1]` to the truncated return (stop‑aware) at 20 days or the actual exit price of a simulated 40‑bar hold with 1.5×ATR stops. This is non‑negotiable.
- **Implement intrabar stop simulation** using existing 5‑minute bars (Polygon). For daily backtest, at each day you replay the 5‑min bars to detect stop/target triggers accurately. This single change will drastically reduce optimistic bias.
- **Scrape or buy daily short borrow rates.** Start with a conservative model: large caps 0.5%, mid caps 2%, small caps 5%. Apply in WF.
- **Increase MAX_OPEN_POSITIONS to 20 (10 long, 10 short if market‑neutral).** Adjust risk limits accordingly. This will reduce noise and make the strategy statistically testable.
- **Add a simple spread model:** extra slippage = half the 20‑day average spread (you can get this from Polygon daily aggregates or estimate from (high‑low)/close).

**Deliverable:** First honest, trustworthy WF run on v216. If average Sharpe < 0.5, stop and reassess model alpha. No gate to live yet.

### Month 2: Risk Management & Execution Realism
- **Integrate a basic factor model.** Use Fama‑French 5 factors + momentum downloaded from Kenneth French’s library. Regress your daily returns against them to decompose alpha vs factor beta. Your signal should have a statistically significant alpha intercept.
- **Volatility‑adjusted position sizing.** Size each position inversely proportional to its recent ATR or historical volatility, so each contributes roughly equal risk.
- **Portfolio‑level VaR/CVaR stop.** Replace the simple sum‑of‑stop‑distances heat limit with a historical VaR (using 2 years of daily returns of current positions).
- **Regime‑adaptive exposure.** When VIX > 25, reduce gross exposure by 50%. Simple but effective.

**Deliverable:** A strategy that can survive a 2020‑style shock with controlled drawdown. Re‑run WF, expect Sharpe > 0.8 with low factor correlation.

### Month 3: Refactor for Multi‑Strategy & Extensibility
- **Abstract the backtest engine** as described in Section 4. Implement the `Strategy` base class.
- **Port the current daily swing strategy** into a `SwingStrategy` implementation. Verify WF output matches exactly the previous monolithic script (a critical test).
- **Add an `IntradayMomentumStrategy`** using 5‑min bars, leveraging the same engine. Run a separate WF for that.
- **Build a centralised feature store** (can be as simple as HDF5 or Parquet files) so that all strategies share the same clean data.

**Deliverable:** One unified WF script that can run `--strategy swing` or `--strategy intraday`. Code duplication eliminated.

### Month 4: Advanced Label & Meta‑Labelling
- **Implement stop‑aware label generation** (truncated returns) as standard for all models.
- **Train a meta‑label model:** Use the primary LambdaRank to propose 50 candidates. Train an XGBoost binary classifier to predict “trade profitable after costs” on out‑of‑sample data. Filter trades through it.
- **Hyperparameter sweep within WF** – For each fold, use the first 80% of train data as train, last 20% as validation to select best `max_positions`, `stop_mult`, and `target_mult`. This will prevent manual overfitting.
- **Start paper trading** in Alpaca with the exact same code that will go live, using a small amount of capital ($5k). Compare simulated P&L vs paper P&L daily.

**Deliverable:** A robust, validated strategy with a demonstrable edge. WF Sharpe > 1.0 on paper.

### Month 5: Production Hardening & Monitoring
- **Build live‑to‑sim drift detection.** Every night, replay the day’s data through the simulator and compare with the actual paper/live fills. Alert on any deviation > 2bp per trade.
- **Add corporate event blackout** (earnings, dividends). Use a free calendar API to avoid holding through earnings.
- **Implement proper borrow cost data feed** from Alpaca or IBKR, updated daily.
- **Stress‑test with custom scenarios**: flash crash, VIX spike, sector rotation.

**Deliverable:** System ready for live capital deployment at the $20k–$50k level.

### Month 6: Scale‑Up & Advanced Features (time‑permitting)
- **Factor risk constraints** using a commercial risk model (e.g., Axioma) or a custom statistical risk model with PCA.
- **Dynamic portfolio optimisation** (minimise risk subject to a target score exposure) using a quadratic optimiser.
- **L2 order book integration** for intraday execution only.
- **Out‑of‑sample holdout test**: After final model selection, reserve the most recent year of data that has never been used in any WF fold. Run a single evaluation on that. That is your true “live” expectation.

**Deliverable:** Institutional‑grade trust layer. Ready for $500k+ allocation.

---

## Final Word

You have the bones of a serious system, but right now the WF is a placebo. The fact that v215 failed when you tightened the simulation should be taken as good news – the system is self‑aware enough to reveal its own weakness. Fix the label mismatch, simulate intraday stops, diversify to at least 20 positions, and integrate real borrow/spread costs. Only then will the walk‑forward output have any correlation with reality.

Do not deploy a single dollar until the average WF Sharpe (with all fixes) is above 0.8 and the Deflated Sharpe Ratio p‑value is below 0.05. Anything less is gambling, not trading.