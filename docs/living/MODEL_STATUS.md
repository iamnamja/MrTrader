# MrTrader — Model Status

**This is the single source of truth for active model versions.**

> **Update rule:** Updated by Claude as the final step of any retrain, promotion, or revert. Updated by human when manually changing the active paper-trade model. If this file and the DB disagree, trust the DB and update this file.

**Last updated:** 2026-05-31

---

## Active Models (Paper Trading)

| Model | Version | Status | Last WF Sharpe | Last CPCV Result | Notes |
|---|---|---|---|---|---|
| swing | v224 | ⚠️ UNVERIFIABLE | INVALID (in-sample) | Cannot run — trained_through=None | Saved 2026-05-29, predates trained_through feature (PR #311, 2026-05-30). Retrain required. |
| intraday_meta | v63 | ⚠️ UNVERIFIABLE | **INVALID (in-sample memorization)** | +5.143 STRUCK FROM RECORD | Saved 2026-05-22. +5.14 was scored on its own training data — see below. Retrain required. |
| regime | v5 | ACTIVE | — | — | Regime classifier; AUC gate separate |

> ## 🔴 CRITICAL (2026-05-31): Both ML models are UNVERIFIABLE; prior results are in-sample
>
> Both `intraday_v63` (saved 2026-05-22) and `swing_v224` (saved 2026-05-29) were trained and
> saved **before** the `trained_through` feature existed (PR #311, 2026-05-30 23:18). Neither
> artifact records its training cutoff, so the OOS guard correctly refuses to validate them.
>
> **The intraday +5.143 CPCV / +6.618 WF result is in-sample memorization, NOT edge.** Confirmed
> by Opus 4.8: the model's 730-day training window (2024-05 → 2026-05) fully contains all three
> CPCV test folds (2024-11 → 2026-04). The run also started 9 minutes before the OOS guard commit
> landed, so it had zero out-of-sample verification. Per-fold Sharpe of 2.05/9.64/8.16 is the
> signature of memorization. **These numbers are struck from the record.**
>
> **Swing v224 has the identical disease** (same frozen-full-window architecture, 6-year window).
>
> **DEEPER STRUCTURAL FINDING:** frozen-model CPCV (one pre-trained model scored across all folds)
> is *structurally incapable* of being out-of-sample when the model is trained on the full window.
> The OOS invariant `te_start > trained_through + purge` can only hold if test folds are AFTER the
> training cutoff — i.e. in the sacred holdout. The honest fix is **true per-fold retraining**
> (retrain inside each fold on only that fold's train window). See PIPELINE_ARCHITECTURE.md KL-10.

---

## Current Gate Status

| Gate | Status | Notes |
|---|---|---|
| Swing WF | ⏳ Pending | v224 not yet walk-forward validated post-audit |
| Intraday WF | ✅ PASSED | avg +6.618, DSR p=1.000, PF=2.67, Cal=32.43 |
| Intraday CPCV | ✅ PASSED | mean +5.143, P5=+0.533, 92.9% pos, DSR p=0.984 |

> **NOTE (2026-05-31):** All WF/CPCV results prior to the 13-round audit (PRs #323–327) must be re-run. Pipeline hardening changes gate behavior for Calmar, PF, and OOS guard. The intraday CPCV result above was produced on the corrected pipeline.
>
> **Known limitations in current gate results:** Deployment-adjusted Sharpe not yet computed (CRITICAL-2 fix pending Phase 1). **Regime gate now ACTIVE (Phase 2, 2026-05-31):** coarse3 BULL/BEAR/NEUTRAL labeler with expanding-quantile VIX (PIT-correct); `worst_regime_sharpe=None` now HARD-FAILS the gate (no more silent pass) unless `ALLOW_NO_REGIME_GATE=True`. WF/CPCV results must now populate `FoldResult.regime_sharpes`. See `docs/living/PIPELINE_ARCHITECTURE.md` §12 for full known-limitations list.

---

## Recently Gate-Failed Models

| Model | Version | Avg WF Sharpe | Gate Result | Reason | Date |
|---|---|---|---|---|---|
| swing | v216 | -0.91 | FAILED | LambdaRank 18-feat 20d. PF=0.00 every fold. | 2026-05-23 |

---

## Version History Quick Reference

See `docs/living/ML_EXPERIMENT_LOG.md` for full fold results and training details.

| Model | Versions | Key milestone |
|---|---|---|
| swing | v186–v224 | v186 last "honest" pre-audit baseline; v217–v224 trained post Phase C features |
| intraday_meta | v51, v61–v63 | v63 first post-audit CPCV pass |
| regime | v1–v5 | v5 current; AUC gate: ≥0.75 + Brier < baseline |
