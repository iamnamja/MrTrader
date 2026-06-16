# External Quant Research Review: MrTrader (2026-06-14)

## 1. Brutally Honest Assessment

Your diagnosis is entirely correct, and you should take a victory lap for building the guillotine that proved it. Your validation engine (Ruler v2, the registry, the forward holdout, Track B) is institutional-grade. Most retail quant operations run beta-masked-as-alpha or overfit noise because they lack the discipline to kill their darlings. You killed yours honestly.

The hard truth: **Your validation harness has outgrown your data envelope.** Your 4-year frozen options database is a statistical trap. Four years (2022–2026) gives you roughly one-and-a-half macro regimes. You cannot validate a structural risk premium (like VRP) or an options-conditioned equity signal over a single regime using an ~8-fold CPCV. The math physically prevents it. A true SR 0.4–0.6 edge over 4 years will consistently fail your t-stat and bootstrap gates because the standard error is simply too wide. 

You built a Ferrari engine but are forced to drive it on a go-kart track. To find anything that passes Ruler v2, you *must* exploit the 19-year free daily ETF history or the multi-decade FRED macro data. That is the only sandbox where your gate has the statistical power to pass a true edge.

**Did you mis-kill anything?** You killed index short-vol (OPT-4) properly under the rules you had, but as your own notes suggest, you measured it with the wrong ruler initially. VRP is a Track B risk premium, not a Track A alpha. However, pausing it was the correct operational move: 4 years of data cannot price the left tail of a short-vol strategy. Until you have 15+ years of options data to capture 2008 and 2020, VRP remains un-testable for your capital tier.

---

## 2. Ranked Next Research Bets

Given your constraints (free data, 19y ETF/Macro depth, single-dev bandwidth), here is where undiscovered edge plausibly lives.

### Bet 1: Structural Calendar / Liquidity Premia
* **Thesis:** Institutional mandates dictate rigid cash flows (month-end rebalancing, 401k deployments) and macro-hedging (FOMC days). These flows create structural supply/demand imbalances that pay a premium to liquidity providers.
* **Data Required/Have?:** YES. 19-year daily ETF data + FMP Economic Calendar / FRED.
* **How to Test:** Test long SPY/QQQ over the Turn-of-Month (T-1 to T+3) and FOMC drift (T-1 to T0). Run via Track B (diversifier) to assess book-level delta against the TSMOM base.
* **Expected SR / Track:** SR ~0.3–0.5. Track B (Risk Premium / Diversifier).
* **Pre-Mortem:** Fails because the premium has attenuated post-2015 due to HFT crowding, or trading costs eat the brief holding period yield.
* **Effort:** 3–5 days. The logic is simple date-math overlaid on your existing ETF bars.

### Bet 2: The "Night Effect" (Overnight Gap Carry)
* **Thesis:** Nearly all historical equity risk premium is generated outside regular trading hours (close-to-open), while intraday returns (open-to-close) are flat or negative. You get paid a premium to hold gap risk overnight.
* **Data Required/Have?:** YES. 19-year daily ETF OHLC data.
* **How to Test:** Long SPY/QQQ at the close, exit at the open. Condition this on FRED macro regimes (e.g., only execute when VIX < 25 or Yield Curve is positive) to cut the left tail. 
* **Expected SR / Track:** SR ~0.3–0.4. Track B (Risk Premium). Highly diversifying to TSMOM because it captures mean-reverting overnight gaps rather than multi-week trends.
* **Pre-Mortem:** Fails the Track B risk budget constraints due to transaction costs (daily turnover) or fails the joint-tail overlap test during crashes.
* **Effort:** 3–5 days. You already have the daily OHLC cache and Track B harness.

### Bet 3: Multi-Speed Trend (Time-Frame Diversification)
* **Thesis:** Your live TSMOM is a lookback-ensemble (21/63/126/252d) rebalanced weekly. Trend-following is fractal. Adding a distinctly different speed (e.g., a very fast 5-day breakout, or a very slow 12-month structural trend) captures different market participants' flow.
* **Data Required/Have?:** YES. 19-year ETF data.
* **How to Test:** Build a fast-trend sleeve (e.g., 5-day/10-day channel breakouts) and test it strictly as a Track B diversifier against your core TSMOM sleeve. 
* **Expected SR / Track:** SR ~0.2–0.3 (Standalone). Track B (Diversifier). The goal isn't standalone alpha; it's to smooth the equity curve of the trend book.
* **Pre-Mortem:** Fails because it correlates too highly with the existing TSMOM sleeve, violating the Track B `< 0.30` correlation ceiling.
* **Effort:** 1 week.

---

## 3. The Book View

You cannot hit a 0.8–1.0 SR book with one home run on free data. You must build it through uncorrelated ~0.3–0.4 SR bricks. The path from "trend-only" to a multi-sleeve book looks like this:

1. **The Anchor (40% Risk Budget):** TSMOM 10-ETF (Currently live, SR +0.71).
2. **The Structural Flow Sleeve (30% Risk Budget):** Calendar Premia (Bet 1). It has effectively zero correlation to 20-day/60-day momentum because it is time-fixed in the market.
3. **The Defensive/Overnight Sleeve (30% Risk Budget):** Overnight carry or a Low-Vol equity tilt. 

If you blend a +0.71 SR asset with two +0.35 SR assets that share <0.20 correlation, the portfolio mathematics will naturally push the combined book SR toward the 0.85+ range. You build the book by hoarding small, uncorrelated, structurally sound premia that survive 19 years of out-of-sample testing.

---

## 4. What NOT to Pursue

* **Do NOT touch the 4-year options data.** Stop torturing it. You cannot validate a risk premium over one regime, and you cannot test event-conditioning robustly without more crisis clusters. Park the options engine until you can afford a 15-year OPRA backfill.
* **Do NOT attempt to "rescue" PEAD.** You demoted it correctly (t=-0.77 at the event level). Small/mid-cap PEAD also failed your survivorship-safe test. Let it go.
* **Do NOT run another Cross-Sectional ML on price features.** You ran it honest, and it came out beta-only or cost-dead. The free technical/price data has been arbitraged out by institutions decades ago.
* **Do NOT attempt cross-asset carry again.** You screened it, and it was pre-cost negative. 
* **Do NOT build intraday mean-reversion.** It died to the cost drag (t=-6.85). Unless you are executing at institutional spread tiers, intraday on free data is a dead end.

---

## 5. The One Bet

If you have a solo quant-dev working for the next 2–4 weeks, **build the Structural Calendar Risk Premia sleeve (FOMC, Turn-of-Month) on the 19-year ETF data.** Why? It directly attacks your statistical power constraint. With 19 years of data, you have 228 turn-of-month events and ~152 FOMC days. That is enough independent clusters to easily clear your Ruler v2 significance floors and Bayesian posterior if an edge exists. It costs zero new data dollars, trades highly liquid ETFs (bypassing the spread wall), and is structurally uncorrelated to your TSMOM sleeve. 

***

**Bottom Line:** Your validation engine is too strict to pass weak anomalies, and your data envelope is too short to prove complex ones. You have successfully eliminated the noise. You must now pivot completely away from "clever" machine learning on thin data, and aggressively test "dumb," well-documented structural risk premia on the 19-year ETF dataset.