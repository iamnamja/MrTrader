# MrTrader — World-Class Quant Review: Next Research Steps (2026-06-14)

**Author:** Grok (xAI) as external quant reviewer  
**Date:** 2026-06-14  
**Input:** Full state snapshot, prompt, data providers, ruler v2, logs, etc.  
**Goal:** Brutally honest, actionable guidance for solo systematic research on free/long data constraints.

## 1. Brutally Honest Assessment

Your assessment is largely correct and the program demonstrates high research integrity. The gate rebuild (Ruler v2) and honest sweep killed almost everything for good reasons: statistical power limits (~8 effective folds on ≤4y data) make validating realistic SR 0.4–0.7 edges extremely difficult, and the failures survived less Type-II settings. The sole survivor — 19y free-ETF TSMOM (+0.71 SR, long-flat inverse-vol, weekly) — is a genuine, testable risk premium on deep history.

**Key truths:**
- Standalone "alpha" in US equities on your data envelope is exhausted. XS-ML (long/L/S), PEAD (event-level demoted t=-0.77), options signals (negative), etc., died honestly.
- Power is the binding constraint; long free history (yfinance ETFs 2005+, FRED) is your primary advantage.
- Frozen 4y options are useful for conditioning/events, not execution or standalone signals.
- Paid data ending forces free-data focus.
- Single-dev reality favors simple, rules-based, deep-history bets over ML or high-turnover.

Realistic remaining edge is marginal premia/diversifiers around trend, validated at book level (Track B) with Ruler v2 waivers. You are not "missing" obvious alpha; the opportunity set is what it is on public/free data.

## 2. Ranked Next Research Bets

### Bet 1: Cross-asset carry/value/term-structure on 19y ETF universe (Highest priority)
**Thesis:** Roll yield, forward curves, yield spreads in rates/commodities/FX/bonds provide persistent premia (hedgers pay for liquidity/insurance). Diversifies trend (often low/negative corr in regimes).
**Data required:** Yes — fully available (yfinance daily on core 10 ETFs + extensions like HYG/LQD if ≥15y history; FRED macro). No new data.
**How to test:** Extend `tsmom.py` (ensemble lookbacks/vol already there) with carry/value signals (backwardation, real yield, term slope). 19y CPCV (power sufficient). Ruler v2 Track B vs current sleeve (ΔSR, crisis contrib, corr<0.3, tail overlap). Pre-register 2-3 variants.
**Expected Sharpe:** Standalone 0.3–0.6; book diversifier +0.1–0.25 IR.
**Why might fail:** High correlation to TSMOM in bull periods; costs/liquidity on secondary ETFs; post-2022 regime change.
**Effort:** 1–2 weeks.

### Bet 2: Conditional event/PEAD structures with frozen options conditioning
**Thesis:** Unconditional PEAD weak, but stronger conditional on dispersion, IV skew/term, macro regime (under-reaction under uncertainty).
**Data:** Yes — event_panel + options_features (frozen).
**Test:** Extend event_inference (two-way clustered). Pre-register continuous interactions (CPIV_pre × SUE, skew × VIX, etc.). Panel reg + deciles; CPCV robustness. Ruler v2 event-level.
**Expected:** Marginal 0.2–0.4 standalone; regime-specific diversifier.
**Pre-mortem:** 4y options window overfit; cluster power still limited; live fills worse.
**Effort:** 2–3 weeks.

### Bet 3: Regime-aware allocator / trend timing overlay
**Thesis:** FRED + breadth/VIX term tilt exposure/cash without new signal.
**Data:** Yes.
**Test:** Extend sleeve_allocator_live. Pre-register features. Track B book-delta.
**Expected:** Small lift via DD control.
**Effort:** 1 week.

### Bet 4: Index VRP as tiny ETP/Track-B diversifier (post-gate)
**Thesis:** Real premium (PF>2), crisis-negative pairs with trend.
**Data:** Partial (mechanics fixes needed).
**Test:** ETP first (no options), then condors if needed. Track B.
**Effort:** 2 weeks (after fidelity).

## 3. Book View — Path to 3–5 Sleeve Premia Book

Anchor: TSMOM 25%+. Add carry/value (#1), conditional event (#2 small), regime overlay (#3). Judge via Track B (appraisal IR, ΔSR, crisis, corr). Equal/vol weights. Live fidelity (replay-diff) mandatory before scaling. Realistic book SR 0.7–0.9 with diversification, not 1.5+ home run. Trend ~50% risk.

## 4. What NOT to Pursue

- XS-ML ranking revival, intraday ML.
- Standalone options execution/signals.
- Killed factors (SI, analyst drift, small/mid PEAD, thresholds).
- New paid data without free proof-of-concept.
- High-turnover or complex ML.

## 5. The One Bet

**Cross-asset carry/value on ETFs (#1).** Highest probability of adding a real, testable, diversifying sleeve using your strongest asset (19y history). Start here.

**Bottom line:** Trend is most of the harvestable edge. Build a small premia book around it with the above. Realistic, disciplined progress is the win. Execute with pre-reg, Track B, fidelity. Kill fast. 

---

*Full detailed version for LLM ingestion: this file.*  
(End of document)