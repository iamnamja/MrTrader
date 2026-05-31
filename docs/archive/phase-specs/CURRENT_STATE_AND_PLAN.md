# MrTrader — Current State and Plan
*Last updated: 2026-05-23 by Opus 4.7*

**TL;DR:** v216 GATE FAILED (avg Sharpe -0.91, PF=0.00 every fold). Phase 1 rank-IC shows near-zero signal with 2022 catastrophic inversion. **Next: PIT audit + L2 decile spread before any retraining.**

---

## 1. Where We Are

### Model Status
| Model | Status | Notes |
|-------|--------|-------|
| swing_v215 | ACTIVE (paper) | Restored after v216 gate fail |
| swing_v216 | TRAINED, GATE FAILED | LambdaRank, 18 features, 20d horizon. Avg Sharpe -0.91, all 5 folds negative, PF=0.00 |
| intraday_meta_v63 | ACTIVE (paper) | |
| regime_model_v5 | ACTIVE | |

### Walk-Forward Gate Results (v216)
| Fold | Sharpe | Max DD | PF | Trades |
|------|--------|--------|----|--------|
| 1 | -1.24 | -18.3% | 0.00 | 312 |
| 2 | -0.45 | -8.1% | 0.00 | 95 |
| 3 | -0.87 | -14.2% | 0.00 | 287 |
| 4 | -1.13 | -19.7% | 0.00 | 341 |
| 5 | -0.86 | -13.8% | 0.00 | 298 |
| **Avg** | **-0.91** | **-14.8%** | **0.00** | **267** |

**Gate: FAILED** (requires Sharpe ≥ 0.80, no fold < 0.50)

### Phase 1 Signal Diagnostic (rank-IC, v216 scores)
| Year | IC@5d | IC@10d | IC@20d | t-stat@20d |
|------|-------|--------|--------|-----------|
| 2021 | +0.007 | +0.016 | **+0.023** | **+4.92** |
| 2022 | -0.016 | -0.020 | **-0.028** | **-8.73** |
| 2023 | -0.005 | -0.006 | -0.009 | -2.11 |
| 2024 | +0.012 | +0.013 | +0.010 | +1.98 |
| 2025 | +0.006 | +0.011 | **+0.017** | **+4.02** |
| **Overall** | +0.001 | +0.003 | **+0.001** | **+0.63** |

**Verdict: NO SIGNAL** — IC@20d = 0.0012, t = 0.63. 2022 catastrophic inversion.

---

## 2. The Definitive Plan (Opus 4.7)

### Core Mistake
PF=0.00 on every fold = execution pathology stacked on near-zero signal. **Jumped to L4 (full agent stack) without ever measuring L1 (rank-IC) or L2 (decile spread).** This is the root cause of all wasted iterations.

### Four-Layer Diagnostic Framework
- **L1**: Rank-IC — does the model rank stocks correctly?
- **L2**: Decile spread — long top decile, short bottom decile, no stops, no costs
- **L3**: Realistic portfolio — costs on, no stops, top-40
- **L4**: Full agent stack — current WF simulation

**Never jump layers.** Each layer must pass before proceeding to the next.

---

## 3. Phase 1: WF Trustworthiness (current)

**Goal**: Make v216 rerun results believable and defensible.

### Tasks (in priority order)
1. **PIT audit of fundamentals** ← HIGHEST RISK
   - Script: `scripts/audit_fundamentals_pit.py`
   - Check: `pe_ratio`, `gross_margin`, `revenue_growth` have ≥45d filing lag
   - If FAILS: all prior IC numbers are contaminated. Rerun Phase 2 after rebuild.

2. **Fix purge to 85 days**
   - Currently: 10d purge + 10d embargo
   - Required: 85d total (60d feature lookback + 20d label horizon + 5d buffer)
   - Files: `scripts/walkforward_tier3.py`, `scripts/walkforward/cpcv.py`

3. **Survivorship correction**
   - Add delisted stocks to training labels with realized return through delisting
   - Currently: only stocks still trading are in labels

4. **Null benchmark**
   - Script: `scripts/random_portfolio_runner.py` (100 seeds, same constraints)
   - Gate requires beating null 2σ

5. **Hyperparameter trial registry**
   - Fix DSR N_trials (currently uses arbitrary count, not logged trials)

6. **Fold 2 audit**
   - Diagnose 95 trades in fold 2 vs 287-341 in other folds
   - Hypothesis: ATR stop collapse causing near-zero participation

### Phase 1 Gate
Pass = "v216 rerun is believable and defensible regardless of Sharpe value"

---

## 4. Phase 2: Signal Measurement (after Phase 1 gate)

**Goal**: Answer "does this model have signal, yes or no?" with evidence.

### Tasks
1. **L2 Decile spread backtest** — `scripts/diag_decile_spread.py`
   - Long top decile, short bottom decile, equal-weight, daily rebalance
   - No stops, no costs, no slippage
   - **Sharpe ≥ 0.60** → signal exists, problem is execution → go to Phase 4
   - **Sharpe 0.20-0.60** → marginal, proceed cautiously to Phase 3
   - **Sharpe < 0.20** → **STOP. Features don't have alpha. Pivot.**

2. **L1 Rank-IC re-measured** (post-PIT fixes)
   - IC IR ≥ 0.5 at h=10/20d = pass

3. **Decile monotonicity** — D1 through D10 should be monotonically ordered

4. **Factor attribution** — residual alpha t-stat > 2 after Mkt/SMB/HML/MOM

5. **L3 bridge test** — top-40, costs on, no stops → should be ≥50% of L2 Sharpe

### Phase 2 Gate
Pass = Sharpe ≥ 0.20 on L2 AND factor attribution shows residual alpha.

---

## 5. Phase 3: Modelling (only if Phase 2 passes)

1. **Policy-realized labels** — simulate actual exits in training labels, not clean 20d close-to-close
2. **Model comparison** — Regression vs Binary Classifier vs LambdaRank; pick by L2 decile spread
3. **Rolling 4-year train window** (not expanding — regimes are non-stationary)
4. **Earnings blackout filter** (±3 days around announcement)

**DO NOT**: regime-conditional retraining (that's HARKing based on 2022 inversion observation).

---

## 6. Phase 4: Portfolio/Execution (only if Phase 3 passes)

1. Fix gates so median open positions ≥ 25 (currently 5-15)
2. L/S conversion: top-N long + bottom-N short, dollar-neutral, borrow filter
3. Intrabar stop simulation (or EOD with explicit bias accounting)
4. Tiered cost model by ADV decile
5. **Final WF gate**: Sharpe ≥ 0.80, ≥ 0.50 any single fold, beats null 2σ

---

## 7. Decision Tree

```
PIT audit fails?
  → YES: All prior results invalid. Rebuild fundamentals. Rerun Phase 2.
  → NO: continue

L2 Sharpe < 0.20?
  → STOP. Rebuild features (microstructure, alt data, options-derived) or change strategy class.

L2 Sharpe 0.20-0.60 AND factor attribution fails?
  → Edge is factor beta, not alpha. Use ETFs instead.

L2 Sharpe ≥ 0.60 AND L3 Sharpe < 0.10?
  → Signal exists, portfolio construction broken. Phase 4 first.

All checks pass?
  → Proceed through Phases 3 and 4.
```

---

## 8. Most Dangerous Assumption

**Fundamentals are PIT.** If `pe_ratio`, `gross_margin`, or `revenue_growth` are not strictly lagged by filing date (≥45 days), the in-sample IC is contaminated by look-ahead. OOS IC ≈ 0 is the expected symptom. This single finding would explain all v216 results.

---

## 9. What Comes After

If Phases 1-4 succeed and WF gate passes, the next horizon is:
- Options-derived features (IV skew, put/call OI)
- Microstructure features (bid-ask spread, order flow imbalance)
- Alternative data integration (satellite, credit card, web traffic)
- News/earnings intelligence (Phase G PEAD signals)

But none of that matters until the current signal problem is resolved.
