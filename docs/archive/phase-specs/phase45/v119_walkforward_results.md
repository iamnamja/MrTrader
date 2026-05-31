# Phase 45 - Phase 2 (v119): Path Quality Regression Label Walk-Forward Results

**Date:** 2026-04-25
**Model:** v119 (XGBRegressor, path_quality label, 84 OHLCV features)
**Label:** `score = 1.0*upside_capture - 1.25*stop_pressure + 0.25*close_strength`
**Config:** stop=0.5x ATR, target=1.0x ATR (Config B from Phase 1)
**OOS AUC:** 0.513 (binarized at 80th pct — near random)

## Walk-Forward Results

| Fold | OOS Period | Trades | Win% | Sharpe | DD |
|---|---|---|---|---|---|
| 1 | 2022-07-28 -> 2023-10-26 | 188 | 50.0% | +0.46 | 1.9% |
| 2 | 2023-10-27 -> 2025-01-24 | 223 | 57.4% | +1.77 | 2.0% |
| 3 | 2025-01-25 -> 2026-04-25 | 271 | 52.0% | -0.81 | 4.4% |
| **Avg** | | **682** | **53.1%** | **+0.476** | |

## Gate Assessment: FAILED

- Avg Sharpe: +0.476 (gate: > +0.80) — MISS
- Min Fold: -0.806 (gate: > -0.30) — MISS

## vs v110 Baseline

| Model | Label | Avg Win% | Avg Sharpe | Min Fold | Gate |
|---|---|---|---|---|---|
| v110 (baseline) | cross_sectional binary | ~40% | +0.34 | -0.73 | FAIL |
| v119 (Phase 2) | path_quality regression | 53.1% | +0.476 | -0.81 | FAIL |

**Improvements:** Win rate up from ~40% to 53%, avg Sharpe up from +0.34 to +0.476, trade count up (682 vs ~290 for v110 baseline run). Fold 2 was exceptional (+1.77).

**Regression:** Fold 3 worse (-0.81 vs v110 fold 3 which was also poor). Same regime-drift issue — the model trains on older data but the most recent market environment diverges.

## Key Findings

1. **AUC 0.513 doesn't prevent trading performance** — the model still generated useful signals (53% win rate, fold 2 Sharpe +1.77) despite near-random classification AUC. The regression ranking is doing something.

2. **Same fold-3 collapse as v110** — the 2025-2026 period is consistently hard. This is a regime issue, not a label issue.

3. **Top features shifted** toward trend/structure indicators: `choch_detected`, `uptrend`, `price_above_ema20`, `bars_since_choch`, `near_52w_high`. These make intuitive sense for path quality.

4. **Win rate improvement is real** — 53% vs 40% cross-sectional binary. The path_quality label is better aligned with actual trade outcomes than raw 5-day return ranking.

## Next Steps

Phase 3 (meta-labeling) targets the fold-3 regime problem by training a secondary model to filter out trades in unfavorable entry contexts.
