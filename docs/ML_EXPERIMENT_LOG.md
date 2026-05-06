# ML Experiment Log — Active Campaign

Tracks model improvement iterations for active and recent phases.
**Archive (Phases 18–26b/c/d, Iterations 1–6):** `docs/ML_EXPERIMENT_LOG_archive.md`

---

## How to Read This Log

- **Verdict**: ✅ Keep | ❌ Revert | 🔄 Pending
- Walk-forward gate (intraday): avg Sharpe > 1.50, no fold < -0.30
- Walk-forward gate (swing): avg Sharpe > 0.80, no fold < -0.30

> **2026-05-05 Meta-update:** Multi-LLM review revealed the walk-forward gate numbers to date are NOT reliable baselines because: (1) no transaction costs, (2) no PM opportunity score simulated, (3) no purge/embargo between folds, (4) NIS features encode time (NaN = pre-2025 regime). Phases 1–2 of MASTER_BACKLOG fix this. Re-run all champions after Phase 1+2 complete to get honest numbers. See `docs/llm_review_synthesis.md`.

---

## Current Champion Models

| Model | Version | Features | Label | Original Sharpe | Honest Sharpe (Phase 1 corrections) | Status |
|---|---|---|---|---|---|---|
| Swing | v142 | 84 | path_quality (5d) | +1.181 | **+0.422** ❌ | Paper only — stale |
| Intraday | v29 | 50 | cross-sectional top-20% | +1.830 | **-0.984** ❌ | Paper only — stale |

> **Phase 1 corrections applied (2026-05-05):** Walk-forward now includes (1) 5bps/15bps round-trip transaction costs, (2) 10-day swing / 2-day intraday purge at fold boundaries, (3) NIS features removed from training (time-leak). These are the first honest Sharpe numbers. Both models fail. See Phase 1 Corrections section below for full fold detail.

---

## Phase 1 Corrections Baseline Walk-Forward — 2026-05-05

**Purpose:** First honest re-validation of champions after applying Phase 1a (cost model), Phase 1b (purge/embargo), Phase 1c (NIS removal). Establishes true baseline before any new model work.

**Corrections vs original walk-forward:**
| Correction | Before | After |
|---|---|---|
| Transaction costs | 0bps | 5bps RT (swing) / 15bps RT (intraday) |
| Fold purge gap | None | 10 calendar days (swing) / 2 trading days (intraday) |
| NIS features | In training | Removed (time-leak — encode regime not sentiment) |
| Walk-forward window | Varies | Same 730d intraday / 5yr swing as original |

### Intraday v29 — Phase 1 Corrections

**Run date:** 2026-05-05 | **Cost:** 15bps RT (7.5bps/side) | **Purge:** 2 trading days

| Fold | Test Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | 2024-10-24 → 2025-04-22 | 244 | 41.4% | **-1.36** | ❌ |
| 2 | 2025-04-25 → 2025-10-17 | 245 | 45.7% | **-2.19** | ❌ |
| 3 | 2025-10-22 → 2026-04-17 | 245 | 50.2% | **+0.60** | ❌ |
| **Avg** | | **734** | **45.8%** | **-0.984** | ❌ GATE FAILED |

**Gate:** avg Sharpe > 1.0 (updated honest gate) | **Min fold:** -2.19 (gate: > -0.30)

**Interpretation:**
- Fold 2 (Apr–Oct 2025 tariff shock) is the regime killer at -2.19. Win rate 45.7% — model has some signal but costs destroy it in a choppy trending market.
- Fold 3 (Oct 2025–Apr 2026) recovers to +0.60 at 50.2% win rate — model works in recent regime.
- Original +1.830 was computed without costs or purge. True net Sharpe on the same data would have been ~+0.8–1.0 if costs had been included.
- **Root cause:** 15bps RT is too high relative to the 0.8×ATR target in a chop regime. Need either lower-cost execution or wider target, or a dispersion gate to skip low-opportunity days.

### Swing v142 — Phase 1 Corrections

**Run date:** 2026-05-05 | **Cost:** 5bps RT (2.5bps/side) | **Purge:** 10 calendar days

| Fold | Test Period | Trades | Win% | Sharpe | Max DD | Gate |
|---|---|---|---|---|---|---|
| 1 | 2022-08-17 → 2023-11-05 | 186 | 46.8% | **+1.25** | 3.0% | ✅ |
| 2 | 2023-11-16 → 2025-02-03 | 226 | 43.4% | **+0.24** | 3.8% | ✅ |
| 3 | 2025-02-14 → 2026-05-05 | 186 | 39.8% | **-0.23** | 5.0% | ✅ |
| **Avg** | | **598** | **43.3%** | **+0.422** | | ❌ GATE FAILED (avg < 0.6) |

**Gate:** avg Sharpe > 0.6 (updated honest gate) | **Min fold:** -0.23 (gate > -0.30 ✅ just passes)

**Interpretation:**
- Fold 1 (2022–2023 bear/recovery): +1.25 — model generalises well to trending-down/recovery regime.
- Fold 2 (2023–2025 melt-up): degrades to +0.24 — EMA/RSI_DIP signals are late entries in a sustained bull run.
- Fold 3 (2025 tariff/vol shock): -0.23, win rate 39.8%. The RSI_DIP pre-filter catches falling knives in gap-and-trend regimes. Just barely passes the min-fold gate.
- Cost correction matters: with wrong 10bps RT, fold 3 was -0.47 (fails gate). With correct 5bps RT, fold 3 is -0.23 (passes). Swing is less cost-sensitive than intraday.
- **Root cause of avg Sharpe below gate:** Mid-period degradation (fold 2 at +0.24) pulls avg down. RSI_DIP + EMA_CROSSOVER pre-filters cap alpha ceiling. ML signal is present but entry timing is constrained by hardcoded rules.
- **Next steps:** (1) PM opportunity score gate for 2025 regime (Phase 2a), (2) remove RSI_DIP/EMA_CROSSOVER pre-filters and let ML pick entries (Phase 3a), (3) shorten training window to exclude 2021 bull distortion.

---

---

## Phase 1d — Deflated Sharpe Ratio + Bootstrap Analysis — 2026-05-05

**Purpose:** Quantify selection bias from testing ~15 model variants. DSR (Bailey & López de Prado 2014) corrects the raw Sharpe for the expected maximum Sharpe from N independent random trials.

### DSR Results

| Model | Version | Raw Sharpe | N Trials | DSR z-score | P(SR>0) | Significant? |
|---|---|---|---|---|---|---|
| Swing | v142 | +1.181 (original) | 15 | -9.96 | 0.000 | ❌ No |
| Swing | v142 | +0.422 (Phase 1 corrected) | 15 | -30.95 | 0.000 | ❌ No |
| Intraday | v29 | +1.830 (original) | 15 | +0.87 | 0.807 | ⚠️ Borderline |
| Intraday | v29 | -0.984 (Phase 1 corrected) | 15 | -56.77 | 0.000 | ❌ No |

**Interpretation:**
- **Swing v142 original +1.181 is statistically meaningless** even before Phase 1 corrections. With 15 trials, the expected maximum Sharpe from random noise exceeds +1.181. We cannot tell whether swing v142 has any genuine edge.
- **Intraday v29 original +1.830 was borderline** (p=0.807) — meaningful signal existed but not definitively proven. The original result was partly real, partly selection bias.
- With Phase 1 corrections, both models fail DSR comprehensively.
- **DSR is now printed in every walk-forward run** (added to `walkforward_tier3.py`). Any future model must pass DSR p > 0.95 as an additional gate condition.

### Bootstrap Distribution (Small Sample — 10 Resamples)

Full 200-resample bootstrap would take ~50 hours (200 × 17min swing). A 10-resample run is pending — will document the distribution shape below. The DSR analytic result above is the primary Phase 1d deliverable.

*Bootstrap results: pending — run `python scripts/bootstrap_sharpe.py --model swing --n-resamples 10` overnight.*

### Updated Gate Criteria (Post Phase 1d)

```
avg net Sharpe > 0.6 (swing) / > 1.0 (intraday)   [lower but honest — with costs + purge]
worst fold Sharpe > -0.30
DSR p-value > 0.95                                  [NEW — corrects for N trials tested]
Max drawdown < 15%
```

---

> **Retrain in progress (2026-05-05 EOD):** Swing v146 (89 feat = 84 + 5 macro NIS) and Intraday v30 (63 feat = 53 + 5 NIS + 5 macro NIS) are training now. Results will be logged below when walk-forward completes.

---

## Retrain Campaign — 2026-05-05 EOD (Macro NIS Integration)

**Context:** Phases 64 (stock NIS) + 90 (macro NIS) added 10 new features with full historical backfill. Current champions (v142/v29) were trained before these features existed — live inference uses them but training didn't see them.

### Swing v146 — 89 features (84 + 5 macro NIS) 🔄 IN PROGRESS

**Training started:** 2026-05-05 ~16:08 ET
**Features added:** `macro_avg_direction`, `macro_pct_bearish`, `macro_pct_bullish`, `macro_avg_materiality`, `macro_pct_high_risk`
**Training window:** 3 years (Polygon daily bars, 430 symbols)
**Gate:** avg Sharpe > 0.80, no fold < -0.30

**Walk-forward result:**

| Fold | Trades | Sharpe |
|---|---|---|
| 1 | 189 | +0.18 |
| 2 | 254 | +0.44 |
| 3 | 208 | **-1.07** |
| **Avg** | 651 | **-0.148** ❌ GATE FAILED |

v142 restored as ACTIVE champion. Fold 3 (2025 regime) continues to collapse — worse than v145 (-0.87). Macro NIS features are not helping and may be adding noise. **Root cause is not the features — the 3yr training window includes 2021-2022 bull market data that teaches patterns incompatible with the 2025 tariff/vol regime.**

**Next swing retrain should try a shorter window (2yr or 18mo) to exclude 2021-2022.** Also: fundamentals (`profit_margin`, `revenue_growth`, `debt_to_equity`) and `sector_momentum` are now available via PIT parquets — include those in v147.

---

### Intraday v30 — 58 features (50 + 5 NIS + 3 SPY-relative) 🔄 IN PROGRESS

**Training started:** 2026-05-05 ~16:08 ET
**Features added over v29 (50 feat):**
- Phase 64: 5 stock NIS features (`nis_direction_score`, `nis_materiality_score`, `nis_already_priced_in`, `nis_sizing_mult`, `nis_downside_risk`) in `intraday_features.py`
- Phase 86b: 3 stock-relative SPY features (`stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`) in `intraday_features.py`
- **Note:** Macro NIS features (Phase 90) live in `features.py` (swing only) — intraday uses `intraday_features.py`, so macro NIS is NOT included in this retrain
**Label scheme:** `cross_sectional_top20pct_phase89` (same as gate-passing v29)
**Training window:** 730 days (Alpaca 5min bars, 750 symbols)
**Gate:** avg Sharpe > 1.50, no fold < -0.30

*Results pending — will update when walk-forward completes.*

---

## Intraday Improvement Campaign — 2026-05

**Context:** v29 passed walk-forward gate Apr 2026 (+1.807 avg Sharpe, 730d window).
Re-tested May 2026 on most recent 365 days — fails (+0.611 avg Sharpe).
**Root cause:** Cross-sectional top-20% label forces positive labels on low-opportunity days.
**Full plan:** `docs/intraday_improvement_plan.md`

### Baseline — v29 Re-Test (2026-05-01)

| Fold | Period | Trades | Win% | Sharpe | Regime |
|---|---|---|---|---|---|
| 1 | Jul–Oct 2025 | 126 | 33.3% | **-0.68** | Low-vol melt-up: VIX 16.5, 0.65%/d |
| 2 | Oct–Jan 2026 | 126 | 49.2% | +1.88 | Moderate: VIX 17.3, 0.76%/d |
| 3 | Jan–Apr 2026 | 126 | 44.4% | +0.63 | Higher vol: VIX 20.4, 0.90%/d |
| **Avg** | | **378** | **42.3%** | **+0.611** | FAIL |

---

### Phase 85 — PM Abstention Gates (No Retrain) ✅ DONE

**Branch:** `feat/phase-85-intraday-gates` | **Completed:** 2026-05-02

**What was changed:**
- VIX ≥ 25 → abstain from all intraday entries
- SPY < MA20 → abstain from all intraday entries

**Walk-forward result (730d window, 3 expanding folds, model v29):**

| Fold | Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 250 | 48.0% | +2.95 | ✅ |
| 2 | Apr 2025–Oct 2025 | 250 | 40.4% | +0.78 | ✅ |
| 3 | Oct 2025–Apr 2026 | 250 | 46.8% | +1.75 | ✅ |
| **Avg** | | 750 | 45.1% | **+1.830** | ✅ GATE PASSED |

**Key observation:** Gates rescued fold 2 (was -0.68) without hurting high-vol folds. Gates are a runtime patch — v29 training signal is still structurally corrupted. Phases 87+86b fix the root cause.

**Verdict:** ✅ GATE PASSED — merged, v29 + abstention gates are the active live configuration.

---

### Phase 86 — Market Context Features + Retrain ❌ REVERTED

**Branch:** `feat/phase-86-market-context-features` | **Completed:** 2026-05-02

**What was tried (v34–v36):**
Added 5 market-wide SPY features: `spy_first_hour_range`, `spy_5d_return`, `spy_5d_realized_vol`, `market_is_trending`, `spy_day_vol_vs_avg`. Lookahead bug found and fixed (`d <= day` → `d < day`). Train/test mismatch found and fixed (simulator now receives `spy_daily_bars`).

**Walk-forward result (v36, 6 folds):**

| Fold | Test Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 250 | 40.8% | -1.04 | ❌ |
| 2 | Apr 2025–Oct 2025 | 252 | 46.0% | -0.41 | ❌ |
| 3 | Oct 2025–Apr 2026 | 250 | 42.8% | -0.13 | ✅ |
| **Avg** | | 752 | 43.2% | **-0.529** | ❌ GATE FAILED |

**Root cause:** All 5 features have identical values for every symbol on a given day. After `cs_normalize` (z-score within each day's symbol set), all become exactly zero — zero cross-sectional discriminating signal.

**Also discovered:** v37 (same 53 features as v29, fresh 50-trial HPO) scored **-0.219** vs v29's +1.830. Confirms HPO variance (~2.0 Sharpe spread) is a second root cause independent of features.

**Verdict:** ❌ REVERTED — market-wide features incompatible with cs-normalisation. Plumbing kept (`spy_daily_bars` wired through training, simulator, walk-forward) for Phase 86b. Proceeding to Phase 87 (stability fix) first.

---

---

## Swing Model Campaign — 2026-05

**Context:** Swing champion v142 (84 features, path_quality 5d) was promoted ~2026-05-01 by automated retrain but not logged. Walk-forward baseline run 2026-05-04 showed avg Sharpe 0.281 (below 0.8 gate) — model was already degraded. NIS features (Phase 64) added; retrain campaigns below.

### Swing Baseline Walk-Forward (2026-05-04, no retrain, existing champion v142)

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Aug 2022–Nov 2023 | 190 | 46.3% | +1.10 |
| 2 | Nov 2023–Feb 2025 | 231 | 44.2% | +0.26 |
| 3 | Feb 2025–May 2026 | 172 | 39.5% | **-0.51** |
| **Avg** | | 593 | 43.3% | **+0.281** | ❌ GATE FAILED |

Root cause: Fold 3 (Feb 2025–now) captures 2025 tariff/volatility regime. Model trained on pre-2025 data doesn't generalize.

### Phase 64 NIS Swing Retrain — v144 (2026-05-04, clean cache) ❌ GATE NOT MET

**What changed:** Added 5 NIS features (direction, materiality, already_priced_in, sizing_mult, downside_risk) with `0.0/1.0` defaults for missing data.

| Fold | Period | Trades | Sharpe | vs Baseline |
|---|---|---|---|---|
| 1 | Aug 2022–Nov 2023 | 180 | **-0.11** | ↓ was +1.10 |
| 2 | Nov 2023–Feb 2025 | 251 | +1.21 | ↑ was +0.26 |
| 3 | Feb 2025–May 2026 | 199 | +0.03 | ↑ was -0.51 |
| **Avg** | | 630 | **+0.376** | ❌ GATE FAILED |

**Root cause of Fold 1 regression:** NIS DB has no data for 2022–2023. All Fold 1 training rows got default values (0.0/1.0), teaching XGBoost a spurious pattern: "when NIS=defaults, predict X." Fixed in Phase 88.

---

### Phase 88 — NIS NaN Encoding + Loosen Intraday Label ❌ GATE NOT MET (swing)

**Branch:** `feat/phase-88-label-nis-fix` (PR #125) | **Date:** 2026-05-05

**What changed:**
1. NIS missing-data encoding: `0.0/1.0` → `NaN` so XGBoost uses learned missing-value direction
2. Intraday label: `MIN_REALIZED_R` 0.5 → 0.40 (experiment log recommendation from Phase 87)

**Swing v145 walk-forward (3 folds, 5yr window, 81 symbols):**

| Fold | Period | Trades | Sharpe | vs v144 (0.0 defaults) | vs Baseline |
|---|---|---|---|---|---|
| 1 | Aug 2022–Nov 2023 | 190 | +0.64 | ↑ from -0.11 | ↓ from +1.10 |
| 2 | Nov 2023–Feb 2025 | 236 | +0.78 | ↓ from +1.21 | ↑ from +0.26 |
| 3 | Feb 2025–May 2026 | 190 | **-0.87** | ↓ from +0.03 | ↓ from -0.51 |
| **Avg** | | 616 | **+0.182** | ❌ | ❌ GATE FAILED |

**v142 retained as ACTIVE swing champion.**

**Analysis:** NaN encoding fixed the Fold 1 regression (−0.11 → +0.64) as expected. But Fold 3 (Feb 2025–now) collapsed further (−0.51 → −0.87). The 2025 market regime is structurally hostile to the current feature set and label scheme. NIS features are not the bottleneck — the swing model needs a deeper fix for recent-regime generalization.

**Intraday v40 walk-forward (MIN_REALIZED_R=0.40, NaN NIS encoding):**

| Fold | Period | Trades | Sharpe | vs v39 (0.5 threshold) | vs v29 baseline |
|---|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 248 | +0.18 | ↑ from +0.06 | ↑ from -0.77 |
| 2 | Apr–Oct 2025 | 249 | **-1.00** | ↓ from +0.64 | ↑ from -1.45 |
| 3 | Oct 2025–Apr 2026 | 248 | -0.74 | ↓ from -0.77 | ↓ from +0.99 |
| **Avg** | | 745 | **-0.519** | ❌ worse | ❌ GATE FAILED |

OOS AUC: 0.508 (≈ random). **v29 restored as ACTIVE intraday champion.**

**Analysis:** Looser label (0.40) did not restore signal quality — AUC barely moved from 0.517 (v39) to 0.508 (v40). The realized-R label scheme appears fundamentally unsuited to the 730d training window used here, or the feature set doesn't predict it. Fold 2 (Apr–Oct 2025) collapsed from +0.64 → −1.00, which is unexplained by the threshold change alone.

**Critical insight:** v29 (cross-sectional top-20% labels + Phase 85 abstention gates) achieved +1.830 avg Sharpe. Every retrain with realized-R labels has failed. The cross-sectional label scheme combined with runtime gates may be the right architecture — the problem is v29 has 50 features but live inference now produces 58 (NIS Phase 64b causes mismatch).

**Recommended next step:** Retrain with cross-sectional labels (revert realized-R) + 58 features. This would give us a proper v29-equivalent but with the correct feature schema.

---

### Phase 87 — Realized-R Labels + 3-Seed Ensemble + Frozen HPO ❌ GATE NOT MET

**Branch:** `feat/phase-87-label-fix-ensemble` (PR #121) | **Date started:** 2026-05-02

**Architecture rationale:** Phase 86 failure revealed HPO variance (~2.0 Sharpe) dominates results. Must fix training stability before adding new features. Three changes bundled:

**What was changed:**
1. **Realized-R labels (B+A):** `realized_R ≥ 0.5 AND abs_move ≥ 0.30%` → label 1. Zero positives allowed on bad days. Replaces cross-sectional top-20% ranking.
2. **3-seed XGBoost ensemble (permanent):** Seeds 42/123/777, blend probabilities by simple average. Applied to every retrain going forward.
3. **100-trial Optuna HPO → FROZEN_HPO_PARAMS:** Thorough search once, freeze params for stability.

**Training class balance (v38, 2026-05-02):** pos=68,459 / neg=209,927 (scale_pos_weight=3.07)
Note: healthier than forced top-20% which always had ~20% positive regardless of market quality.

**Walk-forward results — full comparison (all 4 runs, same 730d window):**

| Run | Fold 1 | Fold 2 | Fold 3 | Avg Sharpe | Gate |
|---|---|---|---|---|---|
| v29, no gates | -0.82 | -1.38 | +0.82 | -0.458 | ❌ |
| v38, no gates | +0.05 | +0.34 | +0.26 | +0.215 | ❌ |
| v29 + Phase 85 gates | +2.95 | +0.78 | +1.75 | +1.830 | ✅ |
| **v38 + Phase 85 gates** | **+0.50** | **+0.51** | **+1.02** | **+0.675** | ❌ |

v38+gates detail: 524 total trades (144/224/156), win rates 49.3%/46.9%/51.9%, max DD 0.2%.

**Key observations:**
- v38 (realized-R labels) is strictly better than v29 without gates: +0.215 vs -0.458. The label change is directionally correct.
- With gates, v38 trails v29: +0.675 vs +1.830. Gates filter trade count (524 vs ~750) — realized-R labels combined with gates are over-filtering.
- All v38 folds positive (no blowups). Min fold Sharpe +0.50 — structurally healthier than v29+gates which had fold 2 at +0.78 only due to gates masking a -1.38 without-gates fold.
- Trade count drop (524 vs 750) suggests `MIN_REALIZED_R=0.5` is too strict — many 0.3–0.4R winners labeled 0 at training time, then correctly rejected by the model at test time.

**Verdict:** ❌ GATE NOT MET — v29 + Phase 85 gates remains champion. v38 is a structural improvement but needs threshold tuning. Options: (A) lower `MIN_REALIZED_R` to 0.35–0.40 and retrain, (B) proceed to Phase 86b (stock-relative features) which adds discriminating signal and may recover selectivity. **Decision: proceed to Phase 86b first** — adding features is higher leverage than threshold micro-tuning.

---

### Phase 87a — Regression Labels (Realized R-Multiple) [DEFERRED]

**Precondition:** Phase 87 ✅ + Phase 86b (stock-relative features) ✅

**Rationale:** Binary labels discard magnitude — a +3R and +0.6R win both get label=1. Regression target (predict realized R-multiple) teaches model to distinguish great setups from marginal ones.

**What to change:**
- XGBoost objective: `binary:logistic` → `reg:squarederror`
- Label: `realized_R` (clipped to [-3.0, +3.0])
- Scoring: rank by predicted R-multiple directly (no probability threshold)
- Keep ensemble 3 seeds + frozen HPO (adapted for regression)

**Verdict:** 🔄 Deferred

---

## Key Takeaways — Intraday Campaign

| Learning | Implication |
|---|---|
| cs_normalize zeros market-wide features | All new features must vary across symbols within a day |
| HPO variance ~2.0 Sharpe on identical features | 100-trial search once → freeze → 3-seed ensemble permanently |
| Forced top-20% labels corrupt bad-day training | Realized-R outcome labels allow zero positives on bad days |
| Gates rescue bad folds but don't fix training | Model must learn to self-abstain, not rely on PM-level runtime gates |

---

---

## Re-Validation on Current Data Window — 2026-05-05

**Purpose:** Re-run existing champions on current data window (folds ending 2026-04-17) to see if they still hold up. Result: neither does. This is the key finding that triggered the multi-LLM review and Phase 1–3 work.

### Intraday v41 — 61 features (NIS + SPY-relative + cs top-20%) ❌

**Trained:** 2026-05-05 EOD  
**HPO best:** AUC=0.6465 (OOS: 0.6289)  
**Top 5 features:** `seg_x_atr_norm` (14%), `atr_norm` (11%), `range_compression` (6%), `minutes_since_open` (3%), `time_of_day` (3%) — NIS not in top 5

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 248 | 41.1% | +0.69 |
| 2 | Apr 2025–Oct 2025 | 249 | 45.4% | **-0.34** |
| 3 | Oct 2025–Apr 2026 | 249 | 49.8% | +0.15 |
| **Avg** | | 746 | 45.4% | **+0.167** ❌ GATE FAILED |

v29 restored as ACTIVE champion. New features (NIS, SPY-relative) did not help and v41 is worse on fold 3 (+0.15 vs v29's +0.97). Fold 2 (Apr–Oct 2025) remains the regime problem.

**Key insight from NIS NaN analysis:** NIS features at 80% NaN teach the model `NaN = 2021-2024 regime`. v41's slight improvement on fold 2 (-0.34 vs v29's -1.27) may be the model learning to treat non-NaN rows differently by time period, not by sentiment. NIS must be removed from training. See Phase 1c.

### Intraday v29 — Re-validated on Current Window ❌

**Original gate result (older window):** +1.830  
**Same model, current data window:**

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 248 | 41.1% | **-0.67** |
| 2 | Apr 2025–Oct 2025 | 249 | 45.4% | **-1.27** |
| 3 | Oct 2025–Apr 2026 | 249 | 49.8% | **+0.97** |
| **Avg** | | 746 | 45.4% | **-0.327** ❌ |

The original +1.830 was computed before Apr–Oct 2025 existed in any test fold. v29 was never validated against that regime. Fold 3 (+0.97) shows the model architecture works in calmer conditions — the problem is regime-specific, not fundamental.

**Decomposition of Sharpe collapse (per multi-LLM review):**
- ~60%: Selection bias (v29 was best of several variants; regression to mean as new data arrives)
- ~20%: PIT leakage (v142 trained with non-PIT-correct fundamentals; similar effect for intraday timing data)
- ~15%: Genuine non-stationarity (tariff shock is a real regime shift)
- ~5%: Data sparsity (limited crisis regime in training history)

### Swing v142 — Re-validated on Current Window ❌

| Fold | Train | Test | Trades | Win% | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| 1 | 2021-04→2022-08 | 2022-08→2023-11 | 189 | 47.1% | +1.20 | 3.1% |
| 2 | 2021-04→2023-11 | 2023-11→2025-02 | 231 | 43.7% | +0.24 | 4.0% |
| 3 | 2021-04→2025-02 | 2025-02→2026-05 | 172 | 39.5% | **-0.51** | 5.1% |
| **Avg** | | | 592 | 43.4% | **+0.310** ❌ |

Fold 3 (Feb 2025–now) collapses across ALL swing versions. This is the 2025 tariff shock + elevated VIX period. Win rate 39.5% (vs 47.1% in fold 1) shows the model's entry pattern selection is failing, not just timing.

---

## Key Takeaways — Swing Campaign (Historical)

1. **Embargo worked:** Test set reduced by 428 samples — confirms leakage was present without it
2. **CS normalization worked:** `sector_momentum` SHAP dropped from 0.90 to 0.18
3. **Revenue growth is real alpha:** Leading SHAP feature at 0.2353
4. **AUC drift threshold:** Realistic steady-state for SP-500 universe is 0.56–0.58 (not 0.65)
5. **5-day forward window:** Doubled training samples, better label-execution alignment
6. **Triple-barrier labels:** Tried and reverted — stop exits didn't improve

Full swing improvement history: `docs/ML_EXPERIMENT_LOG_archive.md`
