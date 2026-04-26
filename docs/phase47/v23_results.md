# Phase 47 — Phase 3: Stop/Target Compression → v23

**Date:** 2026-04-26
**Gate result:** ✅ PASSED — avg Sharpe +1.275 (gate > 0.80)

---

## Changes vs v22

| Parameter | v22 | v23 | Rationale |
|---|---|---|---|
| `ATR_MULT_TARGET` | 1.2× | 0.8× | Target reachable in 2h window (was only 10.6% hit rate) |
| `ATR_MULT_STOP` | 0.6× | 0.4× | Maintains ~2:1 R:R with compressed target |
| `stop_pressure` coeff | −1.25 | −0.50 | Reduce reversion bias (diagnostic: stop-zone trades won 3.1%) |
| Meta-model | Active | Dropped | +0.000 Sharpe contribution confirmed by diagnostic |

---

## Walk-Forward Results (Tier 3)

| Fold | Test Period | Trades | Win% | Sharpe | DD |
|---|---|---|---|---|---|
| 1 | 2024-10-15 → 2025-04-16 | 150 | 44.0% | +0.79 | 0.4% |
| 2 | 2025-04-17 → 2025-10-16 | 226 | 43.8% | +1.30 | 0.6% |
| 3 | 2025-10-17 → 2026-04-20 | 154 | 50.6% | +1.73 | 0.4% |
| **Avg** | | **530** | **46.2%** | **+1.275** | **0.5%** |

**GATE: avg > 0.80 ✅ | min fold > −0.30 ✅ (min: +0.79)**

---

## vs Baseline (v22)

| Metric | v22 | v23 | Δ |
|---|---|---|---|
| Avg Sharpe | +0.301 | +1.275 | **+0.974** |
| Min fold Sharpe | +0.227 | +0.792 | +0.565 |
| Avg win rate | 51.1% | 46.2% | −4.9pp |
| Total trades | 526 | 530 | ≈ same |
| AUC | 0.5438 | 0.5995 | +0.056 |

**Key insight:** Win rate actually *dropped* (51% → 46%) but Sharpe nearly *quadrupled* (+0.30 → +1.28).
This confirms the root cause: the target compression dramatically increased the reward on winning trades
(more target exits at 0.8× ATR vs time exits at random). R:R improved without needing higher win rate.

---

## Decision

**Gate passed. v23 is the new intraday production candidate.**

Phases 2, 4, 5 (XGBRanker, liquidity filter, feature pack) are optional improvements —
they may push Sharpe higher but the gate is already met. Recommend:
1. Ship v23 to paper trading (gate cleared)
2. Continue Phase 2 (XGBRanker) in parallel as an improvement experiment
3. Gate for paper trading stays at avg Sharpe > 0.80
