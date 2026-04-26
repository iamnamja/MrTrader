# Phase 45 - Phase 3 (v120 meta): MetaLabelModel Walk-Forward Results

**Date:** 2026-04-25
**Primary model:** v119 (path_quality regression)
**Meta model:** MetaLabelModel v1 (XGBRegressor, pnl_pct target)
**Config:** stop=0.5x ATR, target=1.0x ATR
**Min expected R threshold:** 0.0

## Meta Model Training Metrics

- R2: 0.059
- MAE: 0.0306
- Corr: 0.286
- Train samples: 412
- Val samples: 103

## Walk-Forward Results (v119 + MetaLabelModel)

| Fold | Gate | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | OK | 138 | 50.0% | +0.330 |
| 2 | OK | 162 | 61.7% | +1.850 |
| 3 | OK | 177 | 54.2% | -0.140 |
| **Avg** | | | | **+0.680** |

## Gate Assessment: FAILED

- Avg Sharpe: +0.680 (gate: > +0.80)
- Min Fold: -0.140 (gate: > -0.30)

## vs Baselines

| Model | Label | Meta | Avg Sharpe | Min Fold | Fold 3 | Gate |
|---|---|---|---|---|---|---|
| v110 | cross_sectional binary | None | +0.34 | -0.73 | -0.81 | FAIL |
| v119 (Phase 2) | path_quality regression | None | +0.476 | -0.806 | -0.81 | FAIL |
| **v119 + meta (Phase 3)** | **path_quality regression** | **v1** | **+0.680** | **-0.140** | **-0.14** | **FAIL (close)** |

## Key Findings

1. **Meta-model dramatically improved fold 3** — the hard 2025-2026 regime period improved from -0.81 to -0.14. The meta-model is learning to avoid bad entries in unfavorable conditions.

2. **Min fold now within gate** — -0.140 vs gate of -0.30. The "no fold below -0.30" criterion is now met. Only the avg Sharpe gate (+0.80) is missed by 0.12.

3. **Win rate held at 55%** — the meta-model filtered out losing trades without proportionally reducing winners (138/162/177 trades vs 188/223/271 without meta). About 26% of trades filtered.

4. **Meta training metrics are weak** (R2=0.059, corr=0.286) — the model predicts pnl weakly in absolute terms, but the filtering threshold (E[R] > 0) is still useful for removing the worst expected-loss trades.

5. **Gap to gate: +0.12 Sharpe** — the system needs ~+0.12 more avg Sharpe to clear the +0.80 gate. Options: (a) tune meta-model threshold, (b) add more training data, (c) Phase 3-Parallel PM abstention gate.

## Next Steps

Phase 3-Parallel: PM daily abstention gate — skip all new entries on days with unfavorable market breadth/VIX regime. Independent of RM, may push avg Sharpe above +0.80.