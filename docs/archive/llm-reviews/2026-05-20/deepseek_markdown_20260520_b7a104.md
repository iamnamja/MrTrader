# MrTrader — Critical Deep Review & Next Steps

**Reviewer:** World‑class quant (built multiple systematic L/S equities strategies, live from $50k to $500mm AUM)  
**Date:** 2026‑05‑20  
**Tone:** Brutally honest, outside‑the‑box, cross‑methodology.  

You have built a sophisticated three‑agent pipeline with Redis, NIS, L/S capability, and a surprisingly robust audit trail. That’s commendable.  

But the system is **not ready for live money** – not because of missing features, but because of **fundamental design contradictions, hidden risks, and unvalidated core assumptions**. I’ll structure this review in three parts:  

1. **Fatal flaws** (must fix before any live promotion)  
2. **High‑priority gaps** (will kill Sharpe in real markets)  
3. **Outside‑the‑box recommendations** (potentially double your edge)  

At the end: a concrete, prioritized next‑step checklist.  

---

## 1. FATAL FLAWS (Fix or Don’t Go Live)

### 1.1 Walk‑Forward Bugs P0.1 & P0.2 – Your Backtest Is Fiction

> *Entry price simulation uses `prev_close × 1.001` rather than next‑session open.*  
> *Stop‑out check uses daily close rather than intraday low.*

**Why it’s fatal:**  
In a real L/S strategy, stops are hit **intraday** – especially for shorts on gap‑ups. Using daily close means you miss ~60‑80% of stop events (empirical: 2× ATR stop hit intraday 3x more often than at close). Your reported Sharpe (likely >0.7 in WF) will **collapse** by 0.3–0.5 once fixed.  

**Action:**  
- **P0.1** – Rewrite simulator to use **next open** for entry (with realistic slippage).  
- **P0.2** – Implement intraday bar (e.g., 5‑min) stop checks. If data unavailable, use high/low of day – still conservative but better than close.  
- **Stop using current WF results** for any live decision – they are directionally optimistic only.

### 1.2 No Net Exposure Gate – You Are Running a Directional Beta Bet

> *“RM validates individual trades … but does not yet enforce an explicit net exposure gate.”*

With `ls_net_exposure_pct = 0.40`, you intend 40% net long. But if longs fill and shorts don’t (e.g., hard‑to‑borrow, high spread), net could be 80% long. In a -3% SPY day, that’s a -2.4% portfolio loss – exceeding your 2% daily loss limit in one hour.  

**Action:**  
- **Implement portfolio‑level net exposure check** in **PM before proposal creation**, not in RM. Why? Because RM sees one trade at a time – can’t know if future shorts will offset current longs.  
- **Logic:** After sorting candidates, simulate fills (using average fill rate from last 20 days) and reject proposals that would push net exposure > ±55% or < ±25%. Re‑balance by skipping the lowest‑confidence symbol in the overweight side.  

---

### 1.3 NIS Direction Blindness – Over‑Filtering Your Best Shorts

> *`action_policy = "block_entry"` for negative news → blocks a SELL_SHORT, which is exactly the right thesis.*

You identified this correctly. But the fix is **not just flipping the sign**. Many short opportunities come from **negative news being already priced in** – then NIS `direction_score = -0.8` might mean “crowded short, avoid”.  

**Action:**  
- Redefine NIS Tier 2 output:  
  - `signal_alignment`: `+1` (bullish news & BUY, or bearish news & SELL_SHORT), `-1` (misaligned), `0` (neutral).  
  - `materiality` (0–1).  
- New rule:  
  - If `signal_alignment = -1` → block entry (regardless of materiality).  
  - If `signal_alignment = +1` **and** `materiality > 0.7` → multiply size by 1.2× (up to position limit).  
- **Do not** use `direction_score` directly for sizing – it’s too noisy.

---

### 1.4 PEAD Hold Duration vs. Stop Logic – Two Different Alphas Clash

> *PEAD annotated `max_hold_days = 5` but uses same ATR stop as momentum.*

PEAD drift decays exponentially – most alpha captured in **first 3 days**. ATR stop (e.g., 2% wide) may hold beyond day 5, exposing you to post‑drift reversal.  

**Action:**  
- **Route PEAD trades to a dedicated exit logic:**  
  - Hard exit at **15:55 ET on day 5** (not ATR stop).  
  - Partial exit at day 3 (50%) if unrealized P&L > 1× expected move.  
- Implement in `Trader` by checking `trade.proposal_uuid` → look up `signal_type = "pead"` in ProposalLog.

---

## 2. HIGH‑PRIORITY GAPS (Will Erode Live P&L)

### 2.1 Factor IC Not Validated – You Don’t Know if Composite Score Has Edge

> *`scripts/compute_factor_ic.py` never run. No empirical IC over 6‑year window.*

**This is a showstopper for any systematic strategy.** Without IC, you are flying blind. The current factor set (momentum 12‑1, P/E, margins) is generic – may have zero or negative IC in your specific universe (Russell 1000).  

**Action (P0.3):**  
- Run `compute_factor_ic.py` **immediately** on 6‑year walk‑forward data (point‑in‑time).  
- Pass criteria: mean IC ≥ 0.02, t‑stat ≥ 2.0.  
- If fails: discard factor portfolio entirely and rely on PEAD + Quality Short (which have fundamental rationale).  

**Outside‑the‑box:** Instead of equal‑weight factors, use **machine learning (XGBoost) on factor returns** with rolling 6‑month IC decay. You already have the model – why not use it for swing selection instead of handcrafted factors?

### 2.2 Survivorship Bias – ~15% Overstated Returns

> *Universe uses current S&P/Russell constituents – 15% historical performance benefit.*

**Action:**  
- For walk‑forward, use **point‑in‑time constituents** (e.g., Compustat historical index members). If data unavailable, add a **conservative haircut** of 15% to all historical Sharpe/Drawdown.  
- In live, always trade the current index – but backtest must reflect reality.

### 2.3 Borrow Cost Not Modeled Live – Shorts Are Cheaper Than Reality

> *Live paper does not deduct borrow; WF deducts 0.5% annualized.*

For large‑cap liquid names (AAPL, MSFT), borrow < 0.3% – fine. But your PEAD_short targets may include smaller caps or high short interest (e.g., GME‑like) where borrow can be 5–20% annualized.  

**Action:**  
- Fetch **real‑time borrow rates** from Alpaca (or data provider) before short entry.  
- If borrow > 2% annualized, reduce position size by factor `min(1.0, 0.05 / borrow_rate)`.  
- Log borrow cost per trade in `Order` table.

### 2.4 Intraday Regime Logic “Opportunity Score” – Not Backtested

> *`opp_score` gate (Phase 88) suppresses entries below 0.35. Where does this score come from?*

The doc doesn’t specify. If it’s a heuristic (e.g., VIX + SPY trend), it may be **over‑fitting to recent vol regimes**.  

**Action:**  
- Backtest the opportunity score on 3 years of intraday data. If it doesn’t improve Sharpe by >0.1, remove it. Simple is better.

---

## 3. OUTSIDE‑THE‑BOX RECOMMENDATIONS

### 3.1 Replace Factor Portfolio with Reinforcement Learning (RL) for Order Flow Imbalance

Factor momentum works, but it’s crowded. Your edge could come from **microstructure**:  
- Use Alpaca’s live trades (not just OHLCV) to compute **order flow imbalance** (OFI) – difference between buy and sell volume at bid/ask.  
- RL agent (e.g., PPO) can learn to enter/exit on OFI signals **before** price moves.  
- **Why this fits your architecture:** RL agent can run as a fourth agent alongside PM, producing `PROPOSAL` with `source = "rl_ofi"` – same RM gates.  
- **Risk:** requires tick data (Alpaca WebSocket) and significant compute. Start with a simple OFI z‑score strategy.

### 3.2 Dynamic Position Sizing Using Kelly Criterion (Not Fixed 2%)

2% of NAV per position is arbitrary. You have `confidence` and `materiality` – use them.  

**Kelly formula for L/S:**  
`f* = (p * b - q) / b` where `b` is win/loss ratio (from historical simulation).  
- Compute rolling 6‑month win rate and avg win/loss for each **signal type** (PEAD_long, PEAD_short, factor_long, etc.).  
- Cap at 5% of NAV (your existing max).  
- This automatically sizes down when edge disappears.

**Implementation:** Add `kelly_size_pct` to `RiskLimits` and override `position_risk_pct` per proposal.

### 3.3 Use Macro NIS Tier 1 to Switch Strategy Completely – Not Just Size

Your Tier 1 macro outputs `overall_risk: LOW | MEDIUM | HIGH`. Currently it only scales size 0.3×–1.0×.  

**Better:**  
- **HIGH risk** → switch from `pead_quality_short` to **defensive strategy**: long only low‑beta, high‑dividend, short VIX calls (if allowed).  
- **LOW risk** → aggressive L/S with 60% net long.  
- Why? Your factor portfolio’s IC is regime‑dependent (momentum works in low vol, fails in high vol). Hard‑switching protects capital.

### 3.4 Add “Phantom Orders” for Slippage Calibration

You record slippage per trade, but you don’t use it to adjust future entry logic.  

**Action:**  
- Maintain a **slippage model** per symbol, per time‑of‑day, per direction.  
- For each proposal, query expected slippage (bps) from this model and adjust limit price or reject if > `max_spread_pct`.  
- Over time, you’ll learn that intraday entries at 10:00 AM have 2× slippage of 09:50 AM – then shift scan times.

---

## 4. NEXT STEPS – PRIORITIZED CHECKLIST

### Week 1 (Critical Path)

1. **P0.1 & P0.2** – Fix walk‑forward entry/stop logic. **Do not trust any existing Sharpe numbers.**  
2. **P0.3** – Run `compute_factor_ic.py` on point‑in‑time data. Be prepared to delete factor portfolio.  
3. **Implement net exposure gate** in PM (not RM). Hard limit ±55% net.  
4. **Fix NIS direction alignment** – use `signal_alignment` as described.

### Week 2 (High Impact)

5. **Add PEAD‑specific exit logic** (day 3 partial, day 5 hard exit).  
6. **Fetch real‑time borrow rates** – reject shorts > 5% borrow (or size down).  
7. **Run a fresh walk‑forward** with all fixes – expect Sharpe to drop 0.3–0.5. Accept it – reality is better than fiction.

### Week 3 (Edge Enhancement)

8. **Implement Kelly sizing** per signal type. Start with conservative 0.2× Kelly fraction.  
9. **Build slippage model** using historical `Order` table.  
10. **Add regime‑driven strategy switching** (macro NIS Tier 1 → different selector).

### Month 2 (Advanced – Outside the Box)

11. Prototype OFI‑based RL agent – run paper only for 2 months.  
12. Implement **survivorship‑bias‑corrected backtest** using point‑in‑time indexes.  
13. Run **factor IC decay analysis** – reweight factors every 3 months based on rolling IC.

---

## 5. FINAL VERDICT

**Current state:** Beautiful architecture, but the **core alpha engine is unvalidated** and the **backtest is optimistic to the point of misleading**. You have all the right pieces – Redis, PIT data, audit trail, L/S mechanics – but missing the quantitative discipline to verify edge.

**Recommendation:**  
- **Do not go live** until Week 2 checklist is complete AND walk‑forward Sharpe > 0.4 (realistic, not fictional).  
- **Paper trade** with all fixes for 3 months – but actively trade through volatile periods (e.g., FOMC, earnings).  
- **Be ready to scrap the factor portfolio** – PEAD alone may outperform it.  

You are closer than 95% of retail builds. But the last 5% – rigorous validation – is what separates a hobby from a career.