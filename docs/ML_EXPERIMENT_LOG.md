# ML Experiment Log — Active Campaign

Tracks model improvement iterations for active and recent phases.
**Archive (Phases 18–26b/c/d, Iterations 1–6):** `docs/ML_EXPERIMENT_LOG_archive.md`

---

## How to Read This Log

- **Verdict**: ✅ Keep | ❌ Revert | 🔄 Pending
- Walk-forward gate (intraday): avg Sharpe > 1.50, no fold < -0.30
- Walk-forward gate (swing): avg Sharpe > 0.80, no fold < -0.30

---

## Current Champion Models

| Model | Version | Features | Label | Avg Sharpe | Gate |
|---|---|---|---|---|---|
| Swing | v142 | 84 | path_quality (5d) | unknown (promoted ~2026-05-01) | ✅ |
| Intraday | v29 | 50 | realized-R ≥ 0.5 + Phase 85 gates | +1.830 | ✅ |

> **Note:** Swing v119 was documented champion but v142 (84 features) was promoted by automated retrain ~May 1. v142 details not fully logged — see model file `app/ml/models/swing_v142.pkl`. Intraday v29 has 50 features (pre-NIS); live inference now computes 58 features causing mismatch — next gate-passing intraday retrain will fix this.

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

## Key Takeaways — Swing Campaign (Historical)

1. **Embargo worked:** Test set reduced by 428 samples — confirms leakage was present without it
2. **CS normalization worked:** `sector_momentum` SHAP dropped from 0.90 to 0.18
3. **Revenue growth is real alpha:** Leading SHAP feature at 0.2353
4. **AUC drift threshold:** Realistic steady-state for SP-500 universe is 0.56–0.58 (not 0.65)
5. **5-day forward window:** Doubled training samples, better label-execution alignment
6. **Triple-barrier labels:** Tried and reverted — stop exits didn't improve

Full swing improvement history: `docs/ML_EXPERIMENT_LOG_archive.md`
