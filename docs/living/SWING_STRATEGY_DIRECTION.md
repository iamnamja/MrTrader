# Swing Strategy Direction — post-mortem + forward plan

**Status:** Long-only cross-sectional swing ranking is DEAD (confirmed by honest per-fold CPCV). This doc is the evidence-based forward plan. Opus 4.8 deep-dive, 2026-06-01.

---

## Diagnosis: why long-only cross-sectional died

**Structural, not a tuning miss.** The first genuinely out-of-sample, per-fold-retrained CPCV of the production v224 LambdaRank architecture returned **mean Sharpe +0.22, t-stat 0.17, 50% positive, DSR p=0.30** — statistically zero. Convergent with every honest result: LX1 equal-weight +0.079, LX9-A beta-neutral +0.031 (F2=-0.70), v221 honest -0.311, factor IC -0.0064 (t=-0.28).

**Mechanism:** cross-sectional ranking is regime-neutral *only if you trade the long-short spread*. Long-only execution is regime-exposed by construction — the long book always carries net equity beta into bear/VIX-spike folds (Aug-2024 "F2"), where the top-quintile label flips meaning from "fastest momentum" to "fell least." LX9-A proved this is NOT fixable by ranking engineering: beta-residualized + diversified, F2 still -0.70.

**The property a swing survivor needs:** P&L must NOT be a monotone function of forward equity-beta. Earn from something orthogonal to "did the market go up" — (a) a self-financing dollar-neutral spread, (b) an event clock independent of the market clock (PEAD), or (c) explicit allocation OUT of equity beta when it turns toxic (book-level cash/treasury switch).

---

## Ranked directions

| # | Direction | P(success)* | Effort | Infra | Verdict |
|---|---|---|---|---|---|
| **1** | **PEAD long-only (hold-extension + guidance-quality split)** | **35%** | S | ✅ Ready | **RUN FIRST** |
| 2 | True dollar-neutral L/S (β≈0, purpose-built short signal) | 20% | L | Partial | High ceiling, fund if PEAD clears |
| 3 | PEAD via options (IV-crush / short verticals) | 30%† | L | ❌ No options infra | Highest ceiling, defer (needs options data+sim) |
| 4 | Sector rotation (11 SPDR ETFs, top-3 6mo momentum) | 20% | S | ❌ ~50 LOC | Cheap benchmark floor — run in parallel |
| 5 | Portfolio-level regime switch to cash/treasuries | 15% standalone | M | Partial | Overlay only, not standalone alpha |
| 6 | New label/horizon/target on long book | <5% | — | — | **DEAD — do not run** |
| 7 | Alt-data features (IV skew, short interest, analyst rev) | 10% | M-L | Hollow slots | Only as PEAD inputs, not ranker rescue |

\* P(success) = clears honest CPCV gate or defensibly-relaxed long-only gate. † Conditional on building options infra.

---

## #1 recommendation: PEAD hold-extension (run first)

**Why:** only direction runnable TODAY on the honest pipeline with zero new infra, best honest number already on record (**+0.349 CPCV** vs ~0 for every swing variant), and genuine F2 immunity (event clock + VIX>30 hard-block sidesteps crisis folds).

**What the honest history says works/fails:**
- Removing shorts was the biggest win (0.129→0.349)
- Priced-in filter HURTS (large announce-day gaps have strongest drift) — keep disabled
- Symmetric VIX confidence-damping hurt longs
- Unexplored levers: longer hold (5→10→15d), earnings-quality split (beat+guidance-raise vs beat alone via `get_analyst_features_at`)

**Experiment:** `scripts/run_pead_cpcv.py`, `long_short=False`, `max_announce_day_move=1.0` (priced-in filter off), **re-enable `vix_block_all=30.0`** (committed CPCV config wrongly has 100.0), sweep hold {5,10,15}, add guidance-quality gate.

**Kill criterion (pre-registered):**
- ADVANCES if mean CPCV Sharpe ≥ 0.50 AND P5 ≥ -0.50 AND %positive ≥ 65% → paper at 1% sizing, 60-90d monitor
- KILLED if mean ≤ +0.20 OR P5 ≤ -0.90 at best (hold,quality) cell → +0.349 was the ceiling; pivot to sector-rotation floor, reconsider options-PEAD

**PEAD readiness:** runnable as-is. Rules-based (no per-fold retraining needed — no model to leak); `trained_through=date.min` so OOS guard passes correctly; same hardened CPCV harness, PIT universe, 5bps costs.

**Gaps to verify before promotion:**
1. PIT-safety of FMP surprise (spot-check `get_earnings_features_at` only returns `filingDate <= as_of`)
2. Committed CPCV config disables VIX block (100.0 vs WF's 30.0) — re-enable
3. Survivorship ~15% (upper bound, same as all swing)
4. +0.349 was best-of-4 configs — DSR-deflate; treat as ceiling

---

## STOP doing (proven dead ends — wasted effort)

1. **All long-only cross-sectional single-name variants** — v224 per-fold t=0.17 is the tombstone. ML weighting ruled out (XGBoost 5-feat -2.344; 82-feat +0.171 < equal-weight).
2. **Timing/entry/exit gates** — LX6a -0.127, LX6b -0.103, LX8 -0.207, LX9-B1 +0.057. All ruled out.
3. **New label/horizon/target on the long book** — sector-relative, momentum-enhanced, triple-barrier, 5d/10d all died in F2.
4. **"+40% net long" L/S** — undefended free param; still carries F2 beta. If L/S revisited: true 0% net + purpose-built short signal (NEVER inverted long composite — LX7/F-infra prove bottom-N = post-crash names that rally fastest).
5. **Treating any long-biased equity strategy as "regime-immune."**

---

## Next-month steer
PEAD hold-extension (#1) first, sector-rotation floor (#4) as cheap parallel benchmark. Hold dollar-neutral L/S (#2) and options-PEAD (#3) as highest-ceiling infra-heavy bets to fund only if PEAD clears its kill criterion.
