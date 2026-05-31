# Phase 46 - 46-A: Intraday v19 Walk-Forward Results

**Date:** 2026-04-25
**Model:** Intraday v19
**Changes vs v18:**
- 46-A: ATR-adaptive binary stop/target labels (1.2x/0.6x prior-day range) replacing fixed 0.5%/0.3%
- 46-D: Added orb_direction_strength feature (42 features total)

**Training metrics:**
- OOS AUC: 0.5106 (HPO CV AUC: 0.5445)
- Train samples: 279,657 | Test samples: 64,988
- Top features: spy_session_return, is_close_session, spy_rsi_14, atr_norm, whale_candle
- Training time: 1,244s

## Walk-Forward Results (v19 — ATR binary labels)

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | 2024-10-14 -> 2025-04-15 | 4 | 25.0% | -1.620 |
| 2 | 2025-04-16 -> 2025-10-15 | 0 | 0.0% | 0.000 |
| 3 | 2025-10-16 -> 2026-04-17 | 1 | 0.0% | -1.000 |
| **Avg** | | **5** | **8.3%** | **-0.875** |

## Gate Assessment: FAILED

- Avg Sharpe: -0.875 (gate: > +0.80) FAIL
- Min Fold: -1.620 (gate: > -0.30) FAIL

## Root Cause Analysis

The binary ATR label fix was conceptually correct (training/inference mismatch was real)
but the chosen multipliers (1.2x/0.6x prior-day range) are calibrated for daily swing trades,
not 2-hour intraday windows.

For a typical stock with 2% prior-day range:
- Training label target: 2.4% in 2 hours — very rarely achieved
- Training label stop: 1.2% in 2 hours — rarely hit either

Result: near-zero positive training examples -> model learned "never trade" ->
5 total trades across 3 folds (vs 150 for v18).

## vs Baseline

| Model | AUC | Avg Sharpe | Total Trades | Gate |
|---|---|---|---|---|
| v18 (fixed labels, 41 features) | ~0.52 | -1.16 | ~150 | FAIL |
| v19 (ATR binary labels, 42 features) | 0.5106 | -0.875 | 5 | FAIL (worse) |

## Decision: Revert ATR Binary Label -> Path Quality Regression

v19 confirmed the ATR mismatch was real, but the binary label approach is not viable
for 2-hour intraday. Next approach (v20): use path_quality continuous score (same as
swing v119) with ATR-adaptive levels. Continuous regression avoids the sparse positive
problem while still aligning with the simulator's ATR-based exit logic.

Next step: retrain v20 with path_quality score label.
