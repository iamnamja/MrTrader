# Phase 45 - Phase 3-Parallel: PM Abstention Gate Walk-Forward Results

**Date:** 2026-04-25
**Primary model:** v119 (path_quality regression)
**Meta model:** MetaLabelModel v1 (XGBRegressor, pnl_pct target)
**PM abstention gate:** VIX >= 25 OR SPY < 20-day SMA -> skip all new entries
**Config:** stop=0.5x ATR, target=1.0x ATR

## Walk-Forward Results (v119 + MetaLabelModel + PM Abstention Gate)

| Fold | Gate | Trades | Win% | Sharpe | DD% |
|---|---|---|---|---|---|
| 1 | OK | 94 | 53.2% | +0.880 | 1.2% |
| 2 | OK | 113 | 65.5% | +2.690 | 0.6% |
| 3 | OK | 134 | 55.2% | -0.030 | 2.7% |
| **Avg** | | **341** | **58.0%** | **+1.181** | |

## Gate Assessment: PASSED

- Avg Sharpe: +1.181 (gate: > +0.80) PASS
- Min Fold: -0.031 (gate: > -0.30) PASS

## vs Progression

| Phase | Config | Avg Sharpe | Min Fold | Gate |
|---|---|---|---|---|
| Baseline v110 | binary label, no meta | +0.340 | -0.730 | FAIL |
| Phase 1 (Config B) | stop=0.5x, target=1.0x | +0.567 | +0.030 | FAIL |
| Phase 2 (v119) | path_quality regression | +0.476 | -0.806 | FAIL |
| Phase 3 (v119 + meta) | + MetaLabelModel filter | +0.680 | -0.140 | FAIL |
| **Phase 3-Parallel** | **+ PM abstention gate** | **+1.181** | **-0.031** | **PASS** |

## Key Findings

1. **Gate cleared** — avg Sharpe +1.181 vs gate of +0.80. The PM abstention gate added ~+0.50 Sharpe
   over the meta-model alone.

2. **Fold 3 (hard 2025-2026 regime) recovered to -0.03** — from -0.81 (no gates) to -0.14 (meta only)
   to -0.03 (meta + abstention). The combined filtering is close to flat in the hardest regime period.

3. **Trade count reduced moderately** — 341 total vs 477 with meta only (28% further reduction).
   Win rate held at 58.0%, showing the gate removes unfavorable-regime trades without hurting quality.

4. **The abstention gate is orthogonal to meta-label** — they address different failure modes:
   - Meta-label filters individual stock entries with weak expected return
   - PM abstention gate filters entire trading days in unfavorable macro regimes (high VIX or SPY downtrend)

5. **Final system stack:** v119 path_quality regression + MetaLabelModel v1 + PM abstention (VIX>=25 OR SPY<MA20)

## System Configuration

- Model: swing_v119.pkl (XGBRegressor, path_quality regression label)
- Meta-filter: swing_meta_label_v1.pkl (E[R] > 0.0 threshold)
- PM gate: skip entries when VIX >= 25 OR SPY close < 20-day SMA
- Stop: 0.5x ATR, Target: 1.0x ATR (2:1 R:R)
- Universe: SP-100, 81 symbols with >= 210 bars
