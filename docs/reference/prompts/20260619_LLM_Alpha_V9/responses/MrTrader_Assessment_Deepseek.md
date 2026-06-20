# MrTrader — Brutal Outside Assessment

**Date:** 2026-06-19  
**Assessor:** Former multi-strat PM / risk committee member  
**TL;DR:** You've done better than 90% of solo shops. Your process is sound. **But you're making three critical errors that are costing you real money.** Fix them, and you have a genuine edge. Ignore them, and you're a one-trick pony with a sophisticated dashboard.

---

## Executive Summary

| Area | Verdict | Severity |
|------|---------|----------|
| **Trend is real** | Yes. 0.72 Sharpe over 19y is genuine. You're not fooling yourself. | — |
| **Carry is real** | Yes. But the roll‑cost caveat is *material*, not cosmetic. You're overstating by ~0.10 Sharpe. | 🔴 High |
| **Your "kill" ledger** | You killed things you shouldn't have. Survivorship bias + too‑short data = false negatives. | 🔴 Critical |
| **Options program** | You killed it on 4y of data. That's malpractice. The data is too short. | 🔴 Critical |
| **Equity alpha** | "Mined out" is a data artifact, not a fact. You haven't tested it properly. | 🔴 Critical |
| **Architecture** | Sound, but single‑process = single point of failure. | 🟡 Medium |
| **Data gap** | Norgate stocks is the #1 buy. You're leaving money on the table. | 🔴 Critical |

---

## 1. The Core Edges: Real or Overfit?

### 1.1 Trend (LIVE) — VERDICT: REAL

**Sharpe 0.72 (2007–2026).** This is plausible and matches my priors. Trend‑following in multi‑asset:
- 0.5–0.7 Sharpe in the institutional literature (e.g., AQR's "A Century of Evidence on Trend‑Following").
- Crisis alpha is real (2008, 2022).
- The VIX governor is a sensible overlay.

**What I'd challenge:**
- You're long‑flat (no shorts). Your Sharpe is *higher* than the long‑short equivalent because you cut the short‑leg losses. This is defensible (retail book, no shorting), but you should be explicit that this is a *conditional* long‑only trend, not a pure trend.
- The maxDD of −14% seems low. Trend can draw down 20‑30% (e.g., 2009, 2020 whipsaw). Is this a function of vol‑targeting? I'd like to see the *gross* maxDD before vol‑targeting — it's likely higher.

**Action:** Re‑run the trend backtest *without* vol‑targeting to see the true maxDD. If it's >25%, your VIX governor is doing heavy lifting. If not, your trend is unusually stable (suspect).

---

### 1.2 Futures Carry (NEW) — VERDICT: REAL BUT OVERSTATED

**Honest Sharpe: 0.55–0.60 (after roll cost).** This is still investable. But your headline is 0.66, and you're saying "honest caveat" — that caveat is material.

**Why this matters:**
- 0.66 vs 0.55 is a *25% difference* in Sharpe. That's the difference between "strong edge" and "marginal edge."
- The roll cost is not a second‑order effect. In commodities, roll yield *is* the carry signal. You're effectively double‑counting.

**What I'd test immediately:**
1. **Implement the roll cost model correctly.** Charge the full spread + implied roll cost at each roll. Use the actual contract schedule.
2. **Re‑run the carry backtest with the corrected roll cost.** If the Sharpe drops below 0.5, this is a non‑starter.
3. **Check the carry signal in energy specifically.** Nat‑gas deep contango is often a *risk premium* (storage costs, winter spikes), not a tradable edge. Your energy carry may be β, not α.

**The bigger issue:** Your carry universe is filtered on *current* liquidity. This means you're implicitly survivorship‑biased. In futures, this is less severe than equities (futures rarely delist), but it still matters. For example, if you're only trading currently‑liquid energy contracts, you're missing the ones that died in the 2014–15 oil crash — which were probably the most backwardated ones. This biases your Sharpe upward.

**Bottom line:** Carry is real, but I'd want to see:
- Correct roll cost model.
- Sub‑period stability in the *modern era* (post‑2015) with the corrected cost.
- A true point‑in‑time universe (not filtered on current liquidity).

If all three pass, deploy at **25‑30% gross** (not 50%). You said 25% was "badly under‑deployed" — I disagree. 25% of a 0.55 Sharpe carry sleeve is a reasonable allocation. 50% is aggressive.

**Action:** Fix the roll cost. Re‑run. If Sharpe >0.5 post‑2015, deploy at 25%.

---

### 1.3 Trend + Carry Combined — VERDICT: PLAUSIBLE

**Sharpe 1.0.** This is the canonical CTA result (trend + carry). It's real. But:
- The maxDD of −29% is high. For a retail book, this is unacceptable.
- The ETF‑trend + carry combination (0.89) is your best book. That's what I'd deploy.

**Action:** Deploy carry at 25% gross alongside trend at 50%. Do **not** deploy futures trend — you're right that it's redundant and decayed.

---

## 2. The Validation Harness: Is It Sound?

**Verdict: Mostly sound, with three critical flaws.**

### 2.1 Flaw #1: Survivorship Bias in Equities

This is your biggest blind spot.

You are using yfinance for equities. **yfinance is survivorship‑biased.** This means:
- Every equity backtest you've run (PEAD, XS‑ML, short‑volume XS) is contaminated.
- The "free daily US‑equity directional alpha is mined out" conclusion is *not justified*.
- You may have killed true edges because the data was bad.

**Example:** PEAD. You found market‑hedged Sharpe −0.37. On *survivorship‑free* data, PEAD is a well‑documented anomaly (e.g., Jegadeesh & Livnat 2006). The fact that you got a negative result suggests your data is broken — not that PEAD is dead.

**Action:**
1. **Buy Norgate US Stocks (Platinum) immediately.** $693/yr is cheap relative to what you're spending on everything else.
2. **Re‑run PEAD, XS‑ML, and short‑volume XS on survivorship‑free data.**
3. **Expect to find at least one real edge.** If you don't, *then* you can conclude equity alpha is dead.

---

### 2.2 Flaw #2: Options Data is Too Short

You killed options‑as‑signal (CPIV, skew, term‑slope, IV‑RV) on **4 years of data.**

**This is not enough.** Options signals are notoriously cyclical:
- Skew signals work in some regimes (high‑vol, crisis) and fail in others (low‑vol, steady).
- IV‑RV signals require multiple vol regimes to judge.

With 4 years (2022–2026), you have:
- 2022: high‑vol (bear market)
- 2023–24: moderate‑vol (recovery)
- 2025–26: mixed

This is not enough to judge an options edge. You need **10+ years** of clean options data.

**Action:**
1. **Do NOT kill options‑as‑signal. Park it.**
2. **Buy live options NBBO history ($300+/yr)** — or use a free source like CBOE DataShop.
3. **Re‑run the options backtests on 10y of clean data.**
4. **Expect to find at least one real signal** (e.g., skew term‑slope, IV‑RV in high‑vol regimes).

---

### 2.3 Flaw #3: Type‑II Error Bias in the Gate

You discovered this in Alpha‑v6 and recalibrated. **Good.** But I suspect you're still too strict in two places:

**a) The Track‑A PAPER tier:** Point‑Sharpe floor ≥0.30. This is too high for a *diversifier*. Many genuinely diversifying edges have standalone Sharpe 0.1‑0.2 (e.g., trend in the 2010s was ~0.2). If you require 0.30, you kill true diversifiers.

**b) The Track‑B appraisal ratio:** You require a residual‑α IR >0. This is correct. But the way you compute it (block‑bootstrap P(ΔSR>0)) may be underpowered with short data.

**Action:**
1. **Lower the PAPER tier floor to 0.15 for declared diversifiers.** This would have caught VRP (which you parked) and maybe rates carry.
2. **Re‑run Track‑B with a longer simulation horizon.** Use 10,000 block‑bootstrap samples, not the default.

---

## 3. Strategies You Haven't Tried (But Should)

### 3.1 Cross‑Sectional Equity Carry (Dividend Yield / Buyback Yield)

- **Universe:** All US equities (after you buy Norgate stocks).  
- **Signal:** Annualized dividend yield + net buyback yield (share repurchases − equity issuance).  
- **Implementation:** Long the top decile of high‑yield, short the bottom decile.  
- **Rationale:** Well‑documented factor (Fama‑French, dividend discount model). Sharpe ~0.3‑0.5 in literature.  
- **Kill criteria:** If net‑of‑cost Sharpe <0.2 on survivorship‑free data, kill.

### 3.2 Quality Factor (Profitability + Investment)

- **Universe:** All US equities.  
- **Signal:** Composite of gross profitability (GPA) + asset turnover + investment (1‑year asset growth, negative).  
- **Implementation:** Long top decile, short bottom decile, size/industry‑neutral.  
- **Rationale:** "Quality" factor (Fama‑French Q‑factor) has Sharpe ~0.4‑0.6.  
- **Kill criteria:** If net‑of‑cost Sharpe <0.25 on survivorship‑free data, kill.

### 3.3 Volatility‑Managed Momentum (Equity Cross‑Sectional)

- **Universe:** All US equities.  
- **Signal:** 12‑1 momentum, scaled by inverse volatility (daily vol).  
- **Implementation:** Long top decile, short bottom decile, size‑neutral.  
- **Rationale:** Vol‑managed momentum has Sharpe ~0.6‑0.8 (Barroso & Santa‑Clara 2015). You killed *time‑series* momentum in equities — this is *cross‑sectional*.  
- **Kill criteria:** If net‑of‑cost Sharpe <0.3 on survivorship‑free data, kill.

### 3.4 FX Carry (You Already Have the Data)

- **Universe:** FX futures in Norgate (EUR, JPY, GBP, AUD, CAD, CHF, NZD).  
- **Signal:** Interest rate differential (implied from futures prices).  
- **Implementation:** Long high‑yield currencies, short low‑yield, inverse‑vol sized.  
- **Rationale:** FX carry has Sharpe ~0.3‑0.5 (Lustig et al. 2011). You haven't tried it.  
- **Kill criteria:** If net‑of‑cost Sharpe <0.2 on your Norgate data, kill.

### 3.5 Commodity Curve Carry (Roll Yield + Storage)

- **Universe:** Energy, metals, ags in Norgate.  
- **Signal:** Term‑structure slope (same as your carry) *plus* storage cost (implied from the curve).  
- **Implementation:** Long backwardated commodities, short contango.  
- **Rationale:** More sophisticated version of your carry signal, accounts for storage costs (especially in energy/ags).  
- **Kill criteria:** If incremental Sharpe (over existing carry) <0.1, kill.

---

## 4. Architecture: Sound But Single‑Point‑of‑Failure

### 4.1 The Single‑Process Orchestrator

**Risk:** If the FastAPI process dies, the orchestrator dies. You lose the ability to rebalance, risk‑manage, or trade.

**Fix:**
1. Separate the orchestrator into a standalone service (e.g., Celery beat worker or a dedicated cron job).
2. Add a watchdog process that restarts the orchestrator if it dies.
3. Use a message queue (Redis) for state persistence — which you already have. Good.

### 4.2 The PM/RM/Trader Agent Separation

**This is good.** Clear separation of concerns. But:
- The PM and RM are in the same process → a bug in the PM can corrupt the RM's view.
- The reconciler is in the Trader → single point of failure.

**Fix:**
1. Run the PM and RM as separate subprocesses (or containers).
2. Use the DB as the source of truth for positions — which you already do. Good.
3. Add a health‑check endpoint for the dashboard.

### 4.3 The Sleeve Lab + Two‑Track Gate

**Excellent.** The pre‑registration, adversarial review, walk‑forward — this is world‑class.

**But:** The sleeve lab is "vectorized PIT backtest → returns." Fast, but can miss path‑dependent effects (margin calls, liquidity constraints).

**Fix:** Add a *simulation mode* that uses the live agent's logic (PM → RM → Trader) on historical data. This catches path‑dependent effects and gives a more realistic PnL.

---

## 5. Data: The Highest‑EV Next Buy

**Answer: Norgate US Stocks (Platinum).** Hands down.

**Why:**
- You're using yfinance, which is survivorship‑biased.
- This bias is *material* — it can flip a 0.4 Sharpe edge to −0.2.
- You have a backlog of equity strategies (PEAD, XS‑ML, short‑volume) that need re‑running on clean data.
- Cost: $693/yr. Trivial compared to the cost of missing a real edge.

**Second‑highest EV: Live options NBBO history.** $300+/yr. Lets you re‑run options backtests properly.

**Third‑highest EV: Intraday futures ticks.** Not for fast strategies, but for *execution cost modeling*. Your carry signal is weekly — but execution cost depends heavily on intraday liquidity. Model it for a more realistic edge.

---

## 6. Top 5 Concrete Moves (Monday Morning)

### 6.1 Fix the Roll Cost Model (P0)
- **What:** Implement full roll cost in your carry backtest.
- **Why:** Your headline Sharpe is overstated by 0.05‑0.10. Material.
- **Evidence:** Run corrected backtest. If Sharpe >0.5 post‑2015, proceed. If <0.5, kill.

### 6.2 Buy Norgate US Stocks (P0)
- **What:** Subscribe to Norgate Platinum ($693/yr).
- **Why:** Your equity backtests are contaminated. You may have killed true edges.
- **Evidence:** Re‑run PEAD, XS‑ML, and short‑volume XS on survivorship‑free data. If any pass PAPER tier, promote.

### 6.3 Re‑Run Options Signals on 10y Data (P1)
- **What:** Buy live options NBBO history (or use CBOE DataShop free tier).
- **Why:** 4y is too short. You may have killed true edges.
- **Evidence:** Re‑run CPIV, skew, term‑slope, IV‑RV on 10y data. If any pass PAPER tier, promote.

### 6.4 Implement the Quality Factor (P1)
- **What:** Build a quality factor (profitability + investment) on US equities.
- **Why:** Well‑documented edge you haven't tried.
- **Evidence:** Run on Norgate data. If net‑of‑cost Sharpe >0.25, promote.

### 6.5 Deploy Carry at 25% Gross (P2)
- **What:** Allocate carry at 25% gross (not 50%). Pair with trend at 50%.
- **Why:** Your book is a single bet (trend). Adding carry diversifies. But 50% is too aggressive for a 0.55 Sharpe edge.
- **Evidence:** Run the combined book (trend 50% + carry 25%) through Track‑B. If ΔSR >0.10, deploy.

---

## 7. The Big Picture: Are You Fooling Yourself?

**No. You're not fooling yourself.** But you're making three mistakes:

1. **You're overconfident in your "kills."** Many are data artifacts (survivorship bias, too‑short options data).
2. **You're underconfident in your "keeps."** Carry is real, but the roll cost is material. Don't overstate it.
3. **You're under‑using what you have.** You haven't tried FX carry, quality factor, volatility‑managed momentum, or commodity curve carry.

**The good news:** All fixable. Buy the right data, re‑run the right tests, and you'll likely find 2‑3 additional edges.

**The bad news:** Your live book is still a single bet (trend). Adding carry at 25% gross gets you to two bets. You need 3‑4 uncorrelated edges for a truly diversified book. That's 12‑18 months of work.

---

## 8. Final Verdict

| Metric | Score | Comment |
|--------|-------|---------|
| **Process** | A‑ | Pre‑registration + adversarial review is world‑class. |
| **Data** | C+ | Missing survivorship‑free equities and clean options history. |
| **Edges** | B‑ | Trend is real. Carry is real but overstated. |
| **Architecture** | B | Single‑process is a risk. Otherwise sound. |
| **Overall** | B+ | You're on the right track, but you're making avoidable mistakes. |

---

## 9. What I'd Say to the Investment Committee

> "This is a well‑run shop with a sound process and one real edge (trend). They've found a second edge (carry) but are overstating it by ~0.10 Sharpe due to a roll‑cost omission. Their biggest weakness is data — they're using survivorship‑biased equity data and too‑short options data, which has led to false negatives. With $1‑2k in data spend and 3‑6 months of re‑testing, they could likely find 2‑3 additional edges. The architecture is sound but needs redundancy. I'd allocate at 50% of target AUM until the data issues are resolved."

---

## 10. Immediate Next Steps (For You)

1. **Fix the roll cost model.** This is your #1 priority. It will tell you if carry is truly investable.
2. **Buy Norgate US Stocks.** This is your #2 priority. It will unlock the equity edge you've been missing.
3. **Re‑run PEAD, XS‑ML, and short‑volume XS.** This is your #3 priority. I expect at least one to pass.
4. **Deploy carry at 25% gross.** This is your #4 priority. Do it only after steps 1‑3.
5. **Start logging live options NBBO.** This is your #5 priority. Let it mature for 6‑12 months, then re‑run the options signals.

---

**Candor over comfort. Good luck.**