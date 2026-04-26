# Phase 49 — Regime-Conditional Walk-Forward Analysis

Generated: 2026-04-26

## Purpose
Identify which market conditions the v23 intraday edge depends on.
Fold 1 Sharpe was +0.79 vs Fold 3 +1.73 — understanding why guides future development.

## Method
Each trading day in each fold's test period is tagged with:
- **VIX level**: low (<18), mid (18-25), high (>25)
- **SPY trend**: bull (above MA20) vs bear (below MA20)
- **SPY 5-day return**: positive vs negative

Then days are counted per regime bucket per fold.

## Fold-by-Fold Regime Breakdown

### Fold 1  (2024-10-15 → 2025-04-16)  Sharpe = +0.79

**VIX distribution:**

| VIX Regime | Trading Days | % of Period |
|---|---|---|
| low (<18) | 66 | 52% |
| mid (18-25) | 47 | 37% |
| high (>25) | 13 | 10% |

**SPY trend**: 71 bull days (56%), 55 bear days (44%)

**SPY momentum (5d)**: 61 positive days (48%)

**Avg VIX**: 19.5 | **Avg SPY 5d return**: -0.26%

### Fold 2  (2025-04-17 → 2025-10-16)  Sharpe = +1.30

**VIX distribution:**

| VIX Regime | Trading Days | % of Period |
|---|---|---|
| low (<18) | 83 | 66% |
| mid (18-25) | 36 | 29% |
| high (>25) | 7 | 6% |

**SPY trend**: 114 bull days (90%), 12 bear days (10%)

**SPY momentum (5d)**: 91 positive days (72%)

**Avg VIX**: 18.1 | **Avg SPY 5d return**: +0.90%

### Fold 3  (2025-10-17 → 2026-04-20)  Sharpe = +1.73

**VIX distribution:**

| VIX Regime | Trading Days | % of Period |
|---|---|---|
| low (<18) | 63 | 50% |
| mid (18-25) | 48 | 38% |
| high (>25) | 15 | 12% |

**SPY trend**: 75 bull days (60%), 51 bear days (40%)

**SPY momentum (5d)**: 70 positive days (56%)

**Avg VIX**: 19.3 | **Avg SPY 5d return**: +0.29%

## Cross-Fold Comparison

| Fold | Sharpe | Avg VIX | Bull % | Low-VIX % | High-VIX % | SPY 5d avg |
|---|---|---|---|---|---|---|
| 1 | +0.79 | 19.5 | 56% | 52% | 10% | -0.26% |
| 2 | +1.30 | 18.1 | 90% | 66% | 6% | +0.90% |
| 3 | +1.73 | 19.3 | 60% | 50% | 12% | +0.29% |

## Insight

- **Model is improving over time** (Fold 3 > Fold 1). This is a positive signal — v23 is not degrading. Could reflect model improving as market adapts to post-2024 patterns that match training features.

## Implications for Future Phases

- **Phase 51 (multi-scan)**: If edge is regime-specific, add regime check before each re-scan
- **Phase 55 (swing gate)**: Use the VIX/bull-pct pattern here to tune the abstention gate
- **Phase 50 (time-of-day)**: Regime may interact with time-of-day — worth testing in retrain