# MrTrader — External Quant Review: Detailed Research Recommendations

**Reviewer:** World-class quant researcher (simulated)  
**Date:** 2026-06-14  
**Based on:** `02_STATE_SNAPSHOT.md`, `01_PROMPT.md`, `DATA_PROVIDERS.md`, `RULER_V2_DESIGN.md`, plus supporting logs

**Bottom line upfront:**  
Your system is honest, your gate is working, and your free‑data US‑equity alpha surface is genuinely exhausted. The only remaining testable, undiscovered edge on free data is **cross‑asset carry** on your 19‑year ETF history (#1 below). If that fails, accept `trend + cash` as the optimal portfolio on your data budget.

---

## 1. Brutally Honest Assessment

### What you got right (95%)

- **Your gate is not the problem.** Ruler v2 with PAPER’s light HAC significance floor (p<0.05) correctly rejects true nulls (3/5 balanced nulls cleared t≥2.0 under the old path‑t). The all‑kill outcome is **honest**, not a Type‑II artifact.
- **Power is not the binding constraint.** For SR=0.5, N=1000 days → t≈15.8. The problem is that none of your candidates had true SR>0.1 after costs. Your test is working, not underpowered.
- **The live book (trend-only 25% + cash) is defensible.** TSMOM’s 19‑year Sharpe 0.71 with crisis‑positive behavior (2008 +7%, 2022 +0.9%) is a solid base.
- **Options as standalone equity signal is dead.** H4a–e with t = −2.7 to −4.4 are conclusive. No revival.
- **PEAD demotion at event level (t=−0.77)** is definitive. The waiver retirement was correct.

### Where you might be wrong (the 5%)

1. **The one overlooked asset: 19‑year free daily ETF history includes *non‑equity* legs you haven’t properly tested for *carry*.** You tried cross‑asset carry once and dismissed it, but you tested the wrong signal (distribution yield) instead of actual curve roll‑down. Re‑test with the correct definition.
2. **Your frozen 4‑year options store is an information asset you’re under‑using.** Options‑as‑signal decile sorts died, but **options‑as‑conditioning** on your 9,774‑event earnings panel is still alive. The implied‑move filter (OPT‑5) showed alpha‑like lift (beta‑hedged Sharpe +0.587) even though threshold‑fragile. The signal is real; the parameterisation was overfit. Continuous interaction regressions are the fix.
3. **You never tested *trend speed optimisation* on the existing 10 ETFs.** You tested *broadening* (more assets + L/S) – that failed. But pure lookback blend optimisation (vol‑adaptive, rolling Sharpe‑optimal) is cheap, high‑EV, and uses your 19‑year power asset.
4. **Your power analysis is mathematically inverted.** You repeatedly say “~8‑fold CPCV cannot validate SR 0.4–0.7 on ≤4y”. That’s false. For SR=0.5, N=1000 days → t=0.5·√(252·4) ≈ 0.5·31.7 = **15.8**. The test is fine; your candidates were just zero‑edge.

**Conclusion:** On free US‑equity data, you’re done. The remaining alpha (if any) lives in (a) cross‑asset carry, (b) conditioning existing equity signals with options data, and (c) micro‑optimising the one thing that works (trend speed). Everything else is a waste of cycles.

---

## 2. Ranked Next Research Bets

### #1 — Cross‑asset carry on the 19‑year ETF universe (Track B diversifier)

| Field | Detail |
|-------|--------|
| **Thesis** | Carry (rolling futures curves, not distribution yield) is structurally uncorrelated to trend. On your ETF set: bond carry (TLT − IEF/SHY yield slope), commodity carry (DBC roll yield via front/back month proxy), FX carry (UUP vs cash). Academic premium: Sharpe 0.3–0.6, crisis‑negative (complements trend’s crisis‑positive profile). |
| **Data required – do we have it?** | **YES (mostly).** yfinance daily is enough for a proxy. Bond carry: difference between 20y (TLT) and 2y (SHY) yield via `^TNX`, `^IRX`. Commodity carry: DBC’s underlying futures curve isn’t directly available, but `(DBC.forward − DBC.spot)/DBC.spot` can be approximated using the ETF’s monthly rebalancing data (available from issuer). FX carry: UUP minus T‑bill rate (`^IRX`). **No new data purchase.** |
| **How to test under Ruler v2** | `CarryStrategy` (rules‑based, subclass `EventEdgeStrategy`). Signal: long highest‑carry assets (bond slope, commodity roll, FX rate), short lowest, dollar‑neutral, weekly rebalance. Universe: your 10 ETFs + duration legs (TLT, IEF, SHY). **Window: 19 years** (2007–2026). CPCV k=8, paths=2, **Track B routing** (worst‑regime waived, diversifier). Gate: appraisal IR ≥ 0.20, P(ΔSR>0) ≥ 0.90 vs trend‑only book at 10‑15% budget. |
| **Expected Sharpe range** | Standalone gross SR 0.2–0.4, net SR 0.15–0.3. **Diversifier value** is the bet: correlation to trend < 0.0 in crisis windows (2008, 2020, 2022). |
| **Why it might fail** | Free‑data carry proxy too crude (DBC roll yield approximation may be noise). Bond carry may already be captured by TSMOM (trend already buys bonds in downturns). Edge may have been fully arbitraged since Koijen et al. (2018). |
| **Effort** | **1–2 weeks** (PIT yield fetcher, DBC proxy research, CPCV adapter, Track B gate). |

---

### #2 — Options‑conditioned earnings drift (event‑level interaction, not a new strategy)

| Field | Detail |
|-------|--------|
| **Thesis** | PEAD died unconditionally (t=−0.77). But OPT‑5 showed conditional promise (beta‑hedged Sharpe +0.587 when `realized/implied < 1.0`). The correct fix: **continuous conditioning**, not binary filtering. Regress hedged event return on `reaction_ratio × pre_event_skew × iv_runup`. Options surface tells you *which* earnings surprises are unpriced. |
| **Data required** | **YES.** Your event panel (9,774 events) + options feature table (583k rows) already joined (enriched 2026‑06‑12). Columns exist: `reaction_ratio`, `cpiv_pre`, `skew_25d_pre`, `iv_runup_t10_t1`, `opt_volume_z_pre`. PIT, holiday‑aware, validated. |
| **How to test under Ruler v2** | **NOT a CPCV run.** Use `event_inference.py` (two‑way CGM clustering). Regression: `fwd_ret_10d_spyhedged ~ reaction_ratio + cpiv_pre + skew_pre + iv_runup + reaction_ratio:skew_pre`. Pre‑register **ONE** hypothesis (H5‑OPTIONS‑CONDITIONAL). Acceptance: interaction term t ≥ 2.0 (two‑sided) with sign consistent (high IV‑runup pre‑event → drift attenuated). No trading strategy yet – pure inference. Only if t ≥ 2.5 and economic magnitude ≥ 10bp per interquartile shift would you build a conditional PEAD v2 scorer. |
| **Expected Sharpe range** | N/A – this is a **significance test**, not a strategy. The payoff is unblocking PEAD’s conditional resurrection without reintroducing threshold overfitting. |
| **Why it might fail** | Unconditional negative PEAD (−8.3bp) may be so robust that no conditioning helps. Options features may be noise (H4 kills suggest no standalone signal, but *interaction* is a different object). Sample may be too thin post‑2022 (4y = ~4 earnings cycles per name). |
| **Effort** | **3–5 days** (run regressions, robustness checks, pre‑registration). Panel already exists. |

---

### #3 — Trend speed optimisation on the existing 10 ETFs (pure parameter tuning)

| Field | Detail |
|-------|--------|
| **Thesis** | Your TSMOM blends 4 fixed lookbacks (21/63/126/252d). Optimal blend for 2007–2026 is likely regime‑adaptive: shorter in high‑vol (VIX>25), longer in bull markets. Before building complexity, test **vol‑scaled lookback** and **rolling Sharpe‑optimal blend** on 19y data. |
| **Data required** | **YES.** Same 10‑ETF daily bars (2005–2026). No new data. |
| **How to test under Ruler v2** | Fork `tsmom.py` into `tsmom_speed.py` with parameterised `lookback_ensemble` (e.g., `[10,20,40,80]` vs `[21,63,126,252]`). Test 3 variants: (a) shorter, (b) vol‑adaptive (lookback ∝ 1/√(VIX/20)), (c) rolling 2‑year optimal blend (re‑optimise weights every 6 months on trailing Sharpe, **pre‑registered rule** – no look‑ahead). CPCV on 19y, Track A (significance). Gate: mean Sharpe improvement ≥ +0.10 over baseline, t ≥ 2.0 on delta. |
| **Expected Sharpe range** | Baseline SR 0.71. Upper bound SR 0.85–0.95 if optimal regime‑adaptive exists. More likely +0.05–0.10 (SR 0.76–0.81). |
| **Why it might fail** | Equal‑weighted blend across 4 lookbacks may already be optimal robust solution – adding parameters only overfits. Vol‑adaptive may hurt in slow‑burn bears (2022) where long lookbacks performed best. |
| **Effort** | **1 week** (modify TSMOM, add optimisation harness, CPCV run). |

---

### #4 — Simple put‑spread tail hedge on trend book (risk management, not alpha)

**Status: NOT testable on your current data.** You have 4y frozen options (2022‑2026) with zero fast crashes (COVID not included). To backtest you would need pre‑2022 options data or VIX futures – both require purchase. **Do not pursue** until you acquire that data. Low priority.

---

### #5 — Short‑term mean reversion in ETFs (low conviction, cheap)

| Field | Detail |
|-------|--------|
| **Thesis** | Intraday stock reversal died on costs (159x turnover → -0.90 net). ETFs have lower spreads, lower turnover, and wider mean‑reversion windows (5‑10 days). Simple z‑score of RSI(5) on 10‑ETF universe, rebalanced weekly, might scratch 0.2–0.3 Sharpe. |
| **Data required** | **YES.** 19y ETF history. |
| **How to test** | `MeanReversionETFStrategy` – long oversold (RSI<30), short overbought (RSI>70), dollar‑neutral, weekly, 5bps costs. CPCV, Track B. Gate: P(ΔSR>0) ≥ 0.85 vs cash, appraisal IR ≥ 0.15. |
| **Expected Sharpe** | 0.15–0.25 net. Correlated to trend (negative during trends, positive during ranges) → limited diversification. |
| **Why it might fail** | ETFs are more efficient than stocks; anomaly may not exist. Signal may be negative net of costs (same as intraday). |
| **Effort** | **2–3 days.** Low priority. |

---

## 3. The Book View – Most Credible Path to 3–5 Sleeves

**Current:** 1 sleeve (trend, 25% of book + cash). Target book SR 0.8–1.0 needs 2–4 sleeves at SR 0.3–0.6 with pairwise correlations < 0.3.

**Realistic 3‑sleeve book (2–3 months of work):**

| Sleeve | Est. SR | Corr to trend | Effort | Status |
|--------|---------|--------------|--------|--------|
| Trend (existing) | 0.71 | 1.0 | – | live |
| Cross‑asset carry (#1) | 0.25–0.40 | −0.1 to +0.1 | 2 weeks | build |
| Options‑conditioned earnings (#2) | 0.30–0.50 | +0.2 to +0.3 | 2 weeks (inference) → 2 weeks (scorer) | panel exists |

**Combined book SR (equal‑weight, vol‑targeted):**  
`√(0.71² + 0.35² + 0.40² + 2·(0.71·0.35·0.1) + …)` ≈ **0.85–1.05** – clears your target.

**Path:**

- **Week 1–2:** Build carry sleeve (#1). Test Track B vs trend‑only. If appraisal IR ≥ 0.20, wire to paper at 10‑15% budget.
- **Week 3–4:** Run options‑conditioned earnings panel regression (#2). If interactions significant (t ≥ 2.5), build conditional PEAD v2 scorer. Test Track B.
- **Week 5–6:** If both pass, run book‑level CPCV (trend + carry + conditional earnings) with `sleeve_allocator` (equal‑weight, vol‑target). Gate: book SR ≥ 0.8, maxDD ≤ trend‑alone maxDD.
- **Week 7–8:** Paper trade 2‑sleeve (trend+carry) while earnings conditioner accumulates OOS events.

**If carry fails (IR < 0.15):** Accept trend‑only as base. Add a simple SPY put‑spread tail hedge (cannot backtest, but first‑principles justified) at 2‑3% NAV cost. Book SR stays ~0.7, but Calmar improves.

---

## 4. What NOT to Pursue (Explicit Kill List)

| Idea | Why not |
|------|---------|
| **Any cross‑sectional ML ranking** | Killed 5+ ways (swing +0.22/t=0.17, intraday -2.80, dollar‑neutral 0.14/t=0.18). Line closed. |
| **Options‑as‑signal equity decile sorts** | H4a–e killed (t=−2.7 to −4.4). Inverse not tradeable (post‑hoc sign‑mining). |
| **Single‑name short volatility** | OPT‑3 killed (mean −1.02 @1×, dies at 2× cost). Spread wall is real. |
| **Dispersion trading (broad single‑name baskets)** | OPT‑3’s cost structure multiplied by 30–50 names. No NBBO history to calibrate. |
| **Short‑interest factor** | Reversed in meme era (t=−3.53). Flipping it = overfitting. |
| **Analyst ratings drift** | CAPM alpha t=0.20 (noise). Dead. |
| **Insider clustering** | Too rare in R1K (mean Sharpe 0.23, t=0.88). Weak. |
| **Buyback announcement drift** | Untestable (no PIT feed on your FMP plan). |
| **Binary threshold sweeps as validation** | OPT‑5’s 1.0 spike + negative at 1.25 proves fragility. Continuous or pre‑registered only. |
| **I/B/E/S or paid revisions data** | Not EV‑positive at $100k paper. H3 blocked – fine. |
| **VRP via condors on single names** | Index VRP via ETP (#4 in blueprint) is the only VRP path, and even that is low priority. |

---

## 5. The One Bet (If You Only Do One Thing)

**Cross‑asset carry on your 19‑year free ETF history (#1).**

**Why:**

- Uses your one asset with statistical power (19 years).
- Genuinely uncorrelated to trend (theoretically negative correlation in crises).
- Strong academic backing (Koijen / Moskowitz / et al., SR 0.3–0.6).
- Zero new data spend.
- Can be tested and deployed within 2 weeks.
- If it works, it adds a second sleeve and moves you from a 1‑sleeve trend book to a real multi‑sleeve portfolio.

**Pre‑mortem:** The free‑data carry proxy may be too crude (DBC roll yield approximation fails). Bond carry may already be captured by trend (trend already goes long TLT in downturns). Signal may be 0.15 net Sharpe, not 0.3–0.4. **Even 0.15 Sharpe with correlation −0.1 to trend improves book SR from 0.71 to 0.75+** – material. If it fails, you’ve spent 2 weeks and proven carry is not harvestable on free data – a publishable null that closes that line forever.

**The one‑liner:** Build a PIT carry signal from TLT/IEF/SHY yield curves and DBC’s rolling methodology, run it through CPCV on 19 years, judge it on Track B (appraisal IR ≥ 0.20, P(ΔSR>0) ≥ 0.90). If it passes, wire it to paper at 10% of book risk. If it fails, accept that your free‑data universe has no second sleeve and focus on operational excellence (trend execution, fill quality, cost calibration) until you’re ready to buy Norgate for proper futures data.

---

## 6. Final Meta‑Comment

Your system is now **honest** – a rare achievement in retail quant. The fact that almost everything died is evidence of a working kill switch, not failure. The next phase should not be “find more alpha” but **“operate a small book of robust premia”** – exactly what Alpha‑v7 states. Carry is the most promising missing piece on free data. If that fails, you have reached the terminal state for your current data budget. Then you decide: buy Norgate or accept the 0.71 Sharpe book. But stop re‑treading dead ends.

*Review completed.*