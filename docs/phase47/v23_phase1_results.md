# Phase 47 — Phase 1: Meta-Model Drop (No Retrain)

**Date:** 2026-04-26
**Action:** Drop MetaLabelModel from intraday stack (no retrain required)
**Baseline:** v22 + MetaLabelModel v1 + PM abstention gate → avg Sharpe +0.301

---

## Decision

The Phase 0 diagnostic confirmed that the MetaLabelModel v1 contributes exactly **+0.000 Sharpe** across all three walk-forward folds:

| Fold | With meta | Without meta | Meta contribution |
|---|---|---|---|
| 1 | +0.242 | +0.242 | +0.000 |
| 2 | +0.434 | +0.434 | +0.000 |
| 3 | +0.227 | +0.227 | +0.000 |
| **Avg** | **+0.301** | **+0.301** | **+0.000** |

**Root cause:** MetaLabelModel v1 intraday has R2=0.001, corr=0.044. The base model's AUC of 0.544 produces insufficient signal variation for a meta-model to filter on. The meta-model's threshold (E[pnl] > 0) is essentially noise, neither accepting nor rejecting trades in a patterned way.

## Action Taken

- MetaLabelModel removed from intraday walk-forward stack (simulator flag: no `--intraday-meta-model-version`)
- `intraday_meta_label_v1.pkl` file retained on disk (no deletion) — may be retrained once base model signal improves
- `SYSTEM_BEHAVIOR.md` updated: meta-model gate step annotated as "currently inactive — R2=0.001"

## Result

**No Sharpe change** (as expected). Baseline remains +0.301.

Stack going into Phase 3: **v22 XGBClassifier + PM abstention gate only**

---

## Why This Matters for Phase 3

With the meta-model dropped, Phase 3 (stop/target compression + stop_pressure fix) experiments will run cleaner:
- Fewer confounders
- Attribution of any Sharpe change goes to the label/exit changes only
- If Phase 3 improves base model quality, we can retrain the meta-model afterward on the better signal
