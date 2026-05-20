# MrTrader Architecture & Quant Strategy Review
**Comprehensive Assessment & Institutional Feedback**
**Date:** 2026-05-20

---

## 1. Executive Summary: The Institutional Verdict

From an engineering and systems architecture perspective, MrTrader is exceptionally robust. The separation of concerns (Portfolio Manager, Risk Manager, Trader) via async Redis queues, the fail-fast risk gates, and the authoritative use of the Order ledger for P&L reconstruction are institutional-grade design patterns. You have built a highly resilient execution pipeline.

However, from a **quantitative and algorithmic perspective**, the system currently suffers from critical blind spots. Your backtest results are highly likely to be a mirage due to P0 execution bugs (look-ahead bias on entry pricing) and survivorship bias. Furthermore, the combination of PEAD (a fast-decaying anomaly) with ATR trailing stops (a trend-following exit mechanism) demonstrates a misalignment between signal thesis and exit mechanics. Promoting this system to live capital before addressing these flaws will result in immediate strategy bleed.

> **CRITICAL WARNING: The Backtest Delusion (Ref 10.5 & 10.6)**
> Bug P0.1 (using `prev_close * 1.001` for entry price) completely invalidates your PEAD performance. The vast majority of Post-Earnings Announcement Drift (PEAD) alpha occurs in the overnight gap. Your simulation is currently "buying" before the gap occurs, capturing returns that are impossible to realize in live trading. You must fix this to use `next_open` immediately.

---

## 2. Addressing Open Architectural Questions

### 10.1 NIS Direction Alignment (High Priority)
**Diagnosis:** You are absolutely correct that the current implementation is broken for short selling. Blocking a `SELL_SHORT` proposal because of "negative news" is filtering out your best, highest-conviction trades.
**Solution:** You must decouple the absolute news sentiment from the action policy. Introduce a dynamically computed alignment score:
* `Aligned_Score = NIS_Direction_Score * (1 if BUY else -1)`
* If `Aligned_Score < -0.5`: Trigger `block_entry` (The news heavily contradicts the ML signal).
* If `Aligned_Score > 0.5`: Trigger `size_up` (The news strongly validates the ML signal).

### 10.2 Net Exposure Gate Missing
**Diagnosis:** The Portfolio Manager constructs trades in a vacuum, focusing on individual alpha. The Risk Manager is the final arbiter of portfolio state. Therefore, the net exposure gate belongs strictly in the Risk Manager.
**Solution:** Implement Rule 0d in the RM sequence: *Portfolio Net Exposure Check*. Before approving a trade, calculate the simulated net exposure. If a new long pushes net exposure above +55%, or a new short pushes it below +25% (around your 40% target), the RM must reject it with reason `NET_EXPOSURE_BREACH`.

### 10.3 PEAD Hold Duration vs. Stop Logic
**Diagnosis:** Misalignment of thesis and mechanics. PEAD is an information assimilation anomaly. The market takes time to price in the surprise, but this edge decays exponentially. Trailing ATR stops are for riding long-term momentum; they give the asset "room to breathe," which in PEAD means giving back your edge as the anomaly fades.
**Solution:** Scrap ATR trailing stops for PEAD. Implement a hard time-based exit (e.g., end of Day 5). For risk management, use a tight, static invalidation stop based on the post-earnings gap (e.g., if the price closes below the gap-up open, the thesis is dead—exit immediately).

### 10.4 Borrow Cost Reality
**Diagnosis:** Ignoring borrow costs on a "Quality Short" strategy is fatal. Companies with deteriorating fundamentals (low margin, declining revenue) have high Short Interest. Hard-to-Borrow (HTB) fees can easily range from 5% to 40% annualized.
**Solution:** In live trading, you must pull HTB rates via Alpaca's API (if supported) or model a synthetic minimum 3-5% borrow drag. Furthermore, add a pre-trade gate: if a stock's borrow fee exceeds your expected alpha (e.g., fee > 15%), reject the short.

---

## 3. Out-of-the-Box Quant Perspectives

### A. Intraday Market Orders are Toxic
Your design states that intraday entries use market orders. For a $20k account, the liquidity might seem sufficient, but small-cap momentum names suffer from severe intraday micro-structure volatility. A market order will force you to cross the spread at the worst possible micro-second. 
**Recommendation:** Use limit orders pegged to the midpoint of the NBBO, or implement a basic TWAP (Time-Weighted Average Price) execution for larger sizing.

### B. Dynamic Factor Weighting (Regime Awareness)
Currently, you use the NIS Tier 1 macro context merely to scale sizing (0.3x - 1.0x). A world-class approach uses macro regimes to shift *feature weights*. If the macro context detects a high-inflation, high-rate regime, the Quality Short (profit margin, high PE inverse) should receive a heavy multiplier, while momentum factors should be down-weighted. Your ML model should ideally be trained conditionally on these regimes.

### C. The "Stale Short" Trap
Your Quality Short candidates are selected based on quarterly fundamental deterioration (margins, revenue). These are slow-moving, structural themes. However, your exit logic (ATR stops) operates on daily/intraday noise. You are using micro-structure risk management on macro-fundamental signals. 
**Recommendation:** Quality shorts need wider stops and longer hold times (weeks/months) to allow the fundamental thesis to play out. If you enforce a max 30-day hold, you will get stopped out by noise before the deterioration is priced in.

---

## 4. Immediate Roadmap (Prioritized)
1. **Halt Live Promotion:** Do not move to live capital until the Walk-Forward simulator is fixed.
2. **Fix P0.1 & P0.2:** Re-write the simulator to use `next_open` for entry fills and intraday highs/lows for stop-outs. Your current Sharpe ratio is artificially inflated.
3. **Implement NIS Directional Alignment:** Fix the logic inversion that is currently blocking your short signals.
4. **Refactor PEAD Exits:** Implement thesis-aligned exits (Time-based decay + tight gap invalidation) instead of ATR trailing stops.
5. **Introduce RM Net Exposure Gate:** Ensure portfolio net long delta remains tightly bounded around the 40% target.
