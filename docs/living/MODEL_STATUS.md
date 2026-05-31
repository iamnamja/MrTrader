# MrTrader — Model Status

**This is the single source of truth for active model versions.**

> **Update rule:** Updated by Claude as the final step of any retrain, promotion, or revert. Updated by human when manually changing the active paper-trade model. If this file and the DB disagree, trust the DB and update this file.

**Last updated:** 2026-05-31

---

## Active Models (Paper Trading)

| Model | Version | Status | Last WF Sharpe | Last CPCV Result | Notes |
|---|---|---|---|---|---|
| swing | v224 | ACTIVE (paper) | Unknown (pre-audit baseline) | Not yet run post-audit | v222/v223 superseded; v224 trained 2026-05-29 |
| intraday_meta | v63 | ACTIVE (paper) | +6.618 (WF, 2026-05-30) | +5.143 mean (2026-05-30, PASSED) | CPCV passed all gates post-13-round audit |
| regime | v5 | ACTIVE | — | — | Regime classifier; AUC gate separate |

---

## Current Gate Status

| Gate | Status | Notes |
|---|---|---|
| Swing WF | ⏳ Pending | v224 not yet walk-forward validated post-audit |
| Intraday WF | ✅ PASSED | avg +6.618, DSR p=1.000, PF=2.67, Cal=32.43 |
| Intraday CPCV | ✅ PASSED | mean +5.143, P5=+0.533, 92.9% pos, DSR p=0.984 |

> **NOTE (2026-05-31):** All WF/CPCV results prior to the 13-round audit (PRs #323–327) must be re-run. Pipeline hardening changes gate behavior for Calmar, PF, and OOS guard. The intraday CPCV result above was produced on the corrected pipeline.
>
> **Known limitations in current gate results:** Deployment-adjusted Sharpe not yet computed (CRITICAL-2 fix pending Phase 1). Regime gate inactive (worst_regime_sharpe=None, silently passing). See `docs/living/PIPELINE_ARCHITECTURE.md` §12 for full known-limitations list.

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
