# Phase 46 - 46-B+C: Intraday MetaLabelModel + Abstention Gate Results

**Date:** 2026-04-25
**Model:** Intraday v19 (ATR-adaptive labels, 38 features)
**Meta model:** MetaLabelModel v1 (XGBRegressor, pnl_pct target)
**PM abstention gate:** VIX >= 25.0 OR SPY < 20-day SMA

## Meta Model Training Metrics

- R2: 0.001
- MAE: 0.0129
- Corr: 0.044
- Train samples: 648
- Val samples: 162

## Walk-Forward Results (v19 + MetaLabelModel + PM Abstention)

| Fold | Gate | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | OK | 152 | 51.3% | +0.240 |
| 2 | OK | 222 | 49.5% | +0.430 |
| 3 | OK | 152 | 52.6% | +0.230 |
| **Avg** | | | | **+0.300** |

## Gate Assessment: FAILED

- Avg Sharpe: +0.300 (gate: > +0.80)
- Min Fold: +0.230 (gate: > -0.30)