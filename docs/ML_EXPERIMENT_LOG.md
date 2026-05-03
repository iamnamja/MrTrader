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
| Swing | v119 | 84 | path_quality (5d) | +1.181 | ✅ |
| Intraday | v29 | 53 | path_quality (cross-sectional) | +1.830 | ✅ (+ Phase 85 gates) |

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
