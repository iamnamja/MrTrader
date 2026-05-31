# Phase 45 - Phase 1 (v119): Stop/Target Structure Grid Results

**Date:** 2026-04-25
**Model:** v110 (unchanged - inference-only test)
**Folds:** 3 walk-forward OOS folds

## Config Definitions

| Config | stop_mult | target_mult | R:R | Gambler's Ruin P(stop) |
|---|---|---|---|---|
| Baseline | 0.5 | 1.5 | 3.0:1 | 75.0% |
| Config A | 0.75 | 1.25 | 1.67:1 | 62.5% |
| Config B | 0.5 | 1.0 | 2.0:1 | 67.0% |

## Per-Config Walk-Forward Results

### Baseline (stop=0.5x, target=1.5x)

**Avg walk-forward Sharpe: +0.096**
Avg trades/fold: 199 | Avg stop exits: 68.7% | Avg win rate: 42.0% | Avg profit factor: 1.364

| Fold | OOS Period | Trades | Win% | Sharpe | PF | Stop% | Target% |
|---|---|---|---|---|---|---|---|
| 1 | 2022-07-07 -> 2023-10-12 | 167 | 43.7% | +0.46 | 2.04 | 65.3% | 31.7% |
| 2 | 2023-10-13 -> 2025-01-17 | 243 | 41.6% | +0.28 | 1.11 | 68.7% | 30.0% |
| 3 | 2025-01-18 -> 2026-04-25 | 187 | 40.6% | -0.45 | 0.94 | 72.2% | 27.3% |

### Config A (stop=0.75x, target=1.25x)

**Avg walk-forward Sharpe: +0.512**
Avg trades/fold: 212 | Avg stop exits: 61.4% | Avg win rate: 45.3% | Avg profit factor: 1.464

| Fold | OOS Period | Trades | Win% | Sharpe | PF | Stop% | Target% |
|---|---|---|---|---|---|---|---|
| 1 | 2022-07-07 -> 2023-10-12 | 178 | 46.6% | +0.80 | 2.14 | 57.9% | 39.3% |
| 2 | 2023-10-13 -> 2025-01-17 | 250 | 44.0% | +0.41 | 1.14 | 63.2% | 36.0% |
| 3 | 2025-01-18 -> 2026-04-25 | 208 | 45.2% | +0.33 | 1.11 | 63.0% | 36.5% |

### Config B (stop=0.5x, target=1.0x) -- WINNER

**Avg walk-forward Sharpe: +0.567**
Avg trades/fold: 227 | Avg stop exits: 55.3% | Avg win rate: 48.3% | Avg profit factor: 1.236

| Fold | OOS Period | Trades | Win% | Sharpe | PF | Stop% | Target% |
|---|---|---|---|---|---|---|---|
| 1 | 2022-07-07 -> 2023-10-12 | 194 | 49.0% | +0.76 | 1.41 | 53.1% | 44.8% |
| 2 | 2023-10-13 -> 2025-01-17 | 264 | 48.5% | +0.91 | 1.25 | 55.7% | 43.6% |
| 3 | 2025-01-18 -> 2026-04-25 | 223 | 47.5% | +0.03 | 1.05 | 57.0% | 43.0% |

## Summary Comparison

| Config | Avg Sharpe | Min Fold | Avg Trades | Avg Stop% |
|---|---|---|---|---|
| Baseline | +0.096 | -0.452 | 199 | 68.7% |
| Config A | +0.512 | +0.326 | 212 | 61.4% |
| **Config B** | **+0.567** | **+0.029** | **227** | **55.3%** |

## Gate Assessment: PASSED

Config B beats Baseline by +0.471 avg Sharpe. All three folds positive. Trade count up 14% vs Baseline.

Key finding: tighter target (1.0x vs 1.5x ATR) improves win rate from 42% to 48%, reduces stop exits from 68.7% to 55.3%. The model has more directional edge than the original structure was capturing.

## Phase 2 Locked Parameters

- `STOP_MULT = 0.5`
- `TARGET_MULT = 1.0`

These multipliers will be used for path_quality label construction in Phase 2 (v120).