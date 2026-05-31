# Phase 46 — Intraday v22 Full Stack Results

**Date:** 2026-04-25
**Model:** Intraday v22
**Changes vs v18 baseline:**
- 46-A: path_quality regression labels (continuous score, cross-sectional top-20% → label 1)
- 46-B: MetaLabelModel v1 (XGBRegressor, pnl_pct target, min_expected_r=0.0)
- 46-C: PM abstention gate (VIX >= 25 OR SPY < MA20)
- 46-D: orb_direction_strength feature (42 features total)
- Simulator: removed hard ORB breakout + volume/whale gate (now soft via model features)

**Training metrics:**
- OOS AUC: 0.5438 (HPO CV AUC: 0.5649)
- Train samples: ~279k | Symbols: 722/766
- Training time: ~1,100s

---

## Stage 1: v22 No Meta Gate

Walk-forward with soft ORB gate, no meta-model:

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | 2024-10-14 -> 2025-04-15 | 252 | 49.2% | -0.96 |
| 2 | 2025-04-16 -> 2025-10-15 | 252 | 49.2% | +0.57 |
| 3 | 2025-10-16 -> 2026-04-17 | 252 | 48.0% | -0.49 |
| **Avg** | | **252** | **48.8%** | **-0.292** |

Gate: FAIL (avg Sharpe -0.292, gate > +0.80)

Trade count recovered from 5 (v19/v20 with hard ORB gate) to 252/fold.

---

## Stage 2: v22 + MetaLabelModel v1 + Abstention Gate (46-B+C)

Meta model training: 810 in-sample trades → XGBRegressor (R2=0.001, corr=0.044)

| Fold | Gate | Trades | Win% | Sharpe | Max DD |
|---|---|---|---|---|---|
| 1 | OK | 152 | 51.3% | +0.24 | 0.5% |
| 2 | OK | 222 | 49.5% | +0.43 | 0.6% |
| 3 | OK | 152 | 52.6% | +0.23 | 0.5% |
| **Avg** | | **175** | **51.1%** | **+0.301** | **0.5%** |

## Gate Assessment: FAILED (but significant progress)

- Avg Sharpe: +0.301 (gate: > +0.80) FAIL
- Min Fold: +0.227 (gate: > -0.30) **PASS**

All folds positive for the first time. Min fold gate cleared.

---

## Progress vs Baseline

| Model | Stack | Avg Sharpe | Min Fold | Trades/fold | All Folds+ |
|---|---|---|---|---|---|
| v18 | Base XGB+LGBM, fixed labels | -1.16 | — | ~50 | No |
| v19 | Binary ATR labels | -0.875 | -1.620 | 2 | No |
| v20 | path_quality labels | -0.138 | -0.820 | 2 | No |
| v22 (no meta) | path_quality + soft ORB | -0.292 | -0.960 | 252 | No |
| **v22 + meta + abstention** | **Full 46 stack** | **+0.301** | **+0.227** | **175** | **Yes** |

---

## Root Cause of Remaining Gap (+0.301 vs gate +0.80)

1. **Meta model signal is weak** (R2=0.001, corr=0.044): The base model's predictions are not
   informative enough to train a strong meta-model. The meta gate filters trades but mostly randomly.
2. **Win rate ~51%** with ATR-adaptive stops/targets: Need ~54%+ to generate Sharpe >0.80
   at current trade frequency.
3. **Feature predictive power**: OOS AUC 0.5438 leaves limited room for the model to separate
   winners from losers.

## Recommended Next Steps

1. **Improve base model signal**: Feature engineering focused on intraday momentum quality
   (volume profile, order flow proxies, relative strength at time of entry).
2. **Tighter entry filter**: Instead of model confidence ≥ 0.50, raise to 0.55+ to trade
   higher-conviction setups only (fewer trades, higher win rate).
3. **Asymmetric R:R**: Current 1.2x target / 0.6x stop = 2:1 R:R. At 51% win rate this
   barely breaks even. Consider 3:1 R:R (wider target or tighter stop) to improve expectancy.
4. **Re-run meta-model with v23** once base AUC improves — a stronger base model will produce
   a more informative meta-model (R2 > 0.05 target).
