# mrtrader_review_gemini_2026-06-22.md

## Overview & The Verdict
David, you have built an institutional-grade validation engine, but you are falling into the classic solo-quant trap: mistaking rigorous backtesting for live alpha, and using research as a procrastination mechanism against the terror of live deployment. The internal panel is right that the app is ahead of the alpha, but they are dramatically underestimating the fragility of your execution and safety architecture. 

Stop looking for the 5th sleeve. You are fighting for basis points in the factor zoo while leaving your flanks wide open to catastrophic operational failure. Here is the brutally honest breakdown.

---

## Block A — An overlooked trading method
**The internal panel missed the fundamental nature of convexity.** They are looking for a statistical artifact (like FX value) to provide structural convexity, which is a category error. 

* **A1. What you missed:** Intraday Time-Series Momentum (CTA-lite) on highly liquid futures (ES, NQ, CL, GC). Your existing trend sleeve holds overnight gap risk—that is where the structural short-convexity bleed happens during a crash. Pure intraday trend scales into directional moves and goes flat by the close. It pays the spread/commissions (the economic payer is the institutional flow executing massive VWAP orders over the day), but it holds zero overnight tail risk.
* **A2. Long-crisis-convexity without negative carry:** **Trend on the yield curve spread (e.g., 2Y/10Y steepener).** In a true deflationary or credit crash, central banks cut rates aggressively at the front end. The curve bull-steepens. Trend-following the spread itself (not outright rates) gives you positive carry (if you are correctly positioned) and explosive positive convexity when the crisis hits.
* **A3. Reversion is dead:** The panel is right. Short-horizon liquid-ETF reversal at 2 bps is dead for a retail/solo setup. It was arbed out by HFTs in 2012. Unless you have co-location and sub-millisecond data, you are the liquidity provider for Citadel. Park it permanently.
* **A4. FX Value vs Carry:** FX PPP (Value) plays out over 3-to-5-year horizons. It is completely useless for a portfolio targeting ~10% vol with weekly rebalances. Carry is the only viable G10 FX edge at your frequency. To falsify FX Value, test it over rolling 6-month horizons—the signal-to-noise ratio is indistinguishable from zero. Drop it.

---

## Block B — Swing equity
**The internal panel is correct. Swing equity is a sunk-cost trap for your setup.** * **B1. The Trap:** You are a solo operator. Single-name equity swing trading requires managing corporate actions, M&A gaps, earnings surprises, and borrowing costs on 3000+ names. It is fundamentally an infrastructure game, not a math game. You do not have the infrastructure. 
* **B2. Vol-managed momentum:** It is redundant. Vol-scaling Barroso/Santa-Clara momentum just dynamically de-levers during crashes. Your VIX governor already does this at the portfolio level. Paying $693 to find a 0.65+ correlation to your existing ETF trend sleeve is burning capital. 
* **B3. Non-momentum swing:** The only viable non-momentum swing strategy for a solo operator is **Statistical Arbitrage (Cointegration)** on highly liquid, fundamentally linked ETF pairs (e.g., EFA/SPY, or sector pairs). It requires lower turnover than single-name reversion and sidesteps idiosyncratic corporate risk. 
* **B4. Pre-screening:** There is no valid cheap pre-screen. Survivorship bias in single names removes the exact catastrophic left-tail events that destroy swing strategies. Testing on biased data is worse than not testing at all because it generates false confidence. Buy the clean Norgate data, or kill the single-name dream.

---

## Block C — Better-trade what we have
**The red team's hard veto on the 8% vol target is the most intelligent thing produced by your internal review.**

* **C1. The Vol-Targeting Danger:** Your book is effectively one bet (convergent continuation). If you lever an undiversified bet to 8% vol, you are guaranteeing a drawdown that will break your psychological risk tolerance, not just your margin limits. I would run at 4–5% vol. Why? Because you have no live track record. Survive the first 12 months. You can scale up; you cannot recover from a blown account.
* **C2. Drawdown ladder vs. VIX Governor:** A rigid drawdown ladder on a single-sleeve book is just a slow-moving stop-loss. It forces you to de-gross at the exact moment of maximum mean-reversion potential (the bottom). The VIX governor is ex-ante (reactive to implied volatility); the ladder is ex-post (reactive to realized losses). Keep the VIX governor, ditch the rigid ladder. 
* **C3. Skip-month TSMOM:** Standard in single names (to avoid the 1-month reversal effect), but it is a coin flip in macro ETFs. Do not touch the crown-jewel edge. Prioritize **execution algorithms**. TWAPing your weekly rebalance over 2 hours instead of crossing the spread at the open will yield more marginal EV than tweaking the signal lookback.
* **C4. Combining sleeves:** Do not use Inverse-vol or ERC. You have zero joint live history. Correlations in backtests are lies waiting to be exposed. Use a **hard, static capital split (e.g., 50/50)**. Isolate the risk pools. Do not let the trend sleeve's PnL fund the futures sleeve's margin until you have 12 months of live data.

---

## Block D — Make the app stronger
**You are vastly underestimating your operational tail risk.** * **D1. Blast-radius × Likelihood:** The #1 failure mode for a solo automated system is a **stale-data/retry loop**. FMP serves yesterday's price -> your system reads a massive delta -> submits an order -> the broker rejects it -> your orchestrator retries infinitely, or worse, the broker accepts it and you leg into max gross at the wrong price. 
* **D2. The No-Go Gate:** The current gate is wildly insufficient. You **must** build the out-of-band broker-only flatten script. If your Postgres DB crashes, your FastAPI app dies. You need a completely independent cron job (running on a separate VPS) that expects a heartbeat from the main app. If no heartbeat in 5 minutes, it sends a pure `close_all_positions()` API call directly to IBKR. 
* **D3. Minimum monitoring:** A Telegram/Pushover bot that sends three messages daily: Open PnL, Margin Utilization %, and System Uptime. If you don't get the message, assume the system is actively dying. 
* **D4. Over-engineering:** The unified "Portfolio Brain" Constructor. You have a $100k book and two sleeves. You do not need Citadel's cross-margining risk engine. You need simple, brute-force position limits and a dead-man's switch. Stop building abstract state machines and start building concrete circuit breakers.

---

## Block E — The meta-question
* **E1. Stop hunting alpha:** The panel is 100% correct. You are using alpha research to hide from the reality of live deployment. The friction of real futures trading (slippage, roll costs, margin latency) will destroy assumptions you hold sacred. Stop researching. Deploy the minimum size, and learn how your system bleeds in the real world.
* **E2. Three forced actions in the next two weeks:**
    1.  **Build and test the external dead-man watchdog.**
    2.  **Wire the fail-closed reconciliation-before-trade logic into the live path.**
    3.  **Delete the swing/single-name scaffolding from your codebase.** (It is a distraction).
    *Waste of time:* Any further optimization of the credit/curve overlays.
* **E3. The blind spot:** Your single most intellectually dishonest framing is the belief that your "Track-A/Track-B, CPCV-validated, multiple-testing-corrected" pipeline immunizes you from overfitting. It does not. It simply shifts the overfitting to the meta-level. You have optimized your validation geometry to select the exact models that performed well over a specific 19-year macro regime (a historic central bank liquidity bubble). You are mathematically certain you haven't overfit the data, but you are completely blind to the fact that you have overfit the *era*.