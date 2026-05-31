# Phase 47 — Diagnostic Report (Phase 0)

**Date:** 2026-04-26
**Model:** Intraday v22 (path_quality labels, soft ORB gate, 42 features)
**Data:** 3 walk-forward folds (Oct 2024 → Apr 2026), 526 trades total

---

## Three Structural Questions

1. **Momentum or Reversion?** → ⚠️  **REVERSION-FLAVORED**: Inside-ORB entries win more than breakout entries.
2. **True Realized R:R?** → 1.20:1
3. **Meta-Model Marginal Sharpe?** → +0.000 Sharpe
4. **Stop-Pressure Label Bias?** → ⚠️  **Stop-zone-touched trades win 63.8% LESS** — consistent with stop_pressure label bias. Consider reducing -1.25 coefficient.

---

### Cut 1 — Momentum vs Reversion (ORB Direction Alignment)

Does the model win more when entry *aligns* with ORB breakout direction (momentum)
or when it *opposes* it (reversion)?

- `orb_breakout = +1`: price above ORB high (bullish breakout)
- `orb_breakout = 0`: price inside ORB (no breakout)
- `orb_breakout = -1`: price below ORB low (bearish — rare in long-only)

|   orb_breakout |   n |   win_rate |   avg_pnl |   avg_mfe |   avg_mae |
|---------------:|----:|-----------:|----------:|----------:|----------:|
|             -1 | 372 |     0.4919 |    0.0005 |    0.0171 |    0.0135 |
|              0 | 145 |     0.5655 |    0.0041 |    0.0205 |    0.0126 |
|              1 |   9 |     0.3333 |   -0.0045 |    0.0079 |    0.0149 |

**Win rate by price_momentum quintile (across all trades):**

| mom_quintile   |   n |   win_rate |   avg_pnl |
|:---------------|----:|-----------:|----------:|
| All            | 526 |     0.5095 |    0.0014 |

**Verdict:** ⚠️  **REVERSION-FLAVORED**: Inside-ORB entries win more than breakout entries.


---

### Cut 2 — Exit Type Breakdown & True Realized R:R

| exit_reason   |   n |   pct |   avg_pnl |   avg_hold |   win_rate |
|:--------------|----:|------:|----------:|-----------:|-----------:|
| STOP          | 165 |  31.4 |   -0.016  |     8.1697 |     0      |
| TARGET        |  56 |  10.6 |    0.0326 |     8.5357 |     1      |
| TIME_EXIT     | 305 |  58   |    0.0051 |    21.8131 |     0.6951 |

**Avg winning pnl_pct:** 0.0143
**Avg losing pnl_pct:** -0.0119
**Realized R:R:** 1.20:1
**Target exits:** 56 (10.6%)
**Stop exits:** 165 (31.4%)
**Time exits:** 305 (58.0%)
  - Time exits positive: 212 (69.5%)
  - Time exits negative: 93 (30.5%)

**Verdict:** ⚠️  **TRUE R:R IS 1.20:1** — much lower than stated 2:1. Stop/target compression (Phase 3) is mandatory.

⚠️  **58% TIME EXITS** — target is too far for 2-hour hold. Compression strongly indicated.


---

### Cut 3 — Regime Gate Attribution (VIX & SPY vs MA20)

*Note: PM abstention gate (VIX≥25, SPY<MA20) should have removed the worst regime days.
This cut checks win rates across the regimes that passed the gate.*

**Win rate by ATR volatility quintile (proxy for VIX regime):**

| atr_quintile   |   n |   win_rate |   avg_pnl |
|:---------------|----:|-----------:|----------:|
| All            | 526 |     0.5095 |    0.0014 |

**Win rate by SPY session direction:**

| spy_regime   |   n |   win_rate |   avg_pnl |
|:-------------|----:|-----------:|----------:|
| SPY flat     | 526 |     0.5095 |    0.0014 |

**Per-fold summary:**

|   fold |   n |   win_rate |   avg_pnl |   sharpe_proxy |
|-------:|----:|-----------:|----------:|---------------:|
|      1 | 152 |     0.5132 |    0.0016 |         1.4546 |
|      2 | 222 |     0.4955 |    0.0015 |         1.3666 |
|      3 | 152 |     0.5263 |    0.0012 |         1.2209 |

---

### Cut 4 — Feature Stratification (Which Features Carry the Edge)

**Win rate by cross-sectional momentum rank quintile:**

| mom_quintile   |   n |   win_rate |   avg_pnl |
|:---------------|----:|-----------:|----------:|
| All            | 526 |     0.5095 |    0.0014 |

**Win rate by ORB position quintile (0=at low, 1=at high):**

| orb_pos_decile   |   n |   win_rate |   avg_pnl |
|:-----------------|----:|-----------:|----------:|
| All              | 526 |     0.5095 |    0.0014 |

**Win rate by volume_surge quintile:**

| vol_quintile   |   n |   win_rate |   avg_pnl |
|:---------------|----:|-----------:|----------:|
| All            | 526 |     0.5095 |    0.0014 |

**Win rate by RSI-14 bin:**

| rsi_bin       |   n |   win_rate |   avg_pnl |
|:--------------|----:|-----------:|----------:|
| Oversold(<30) | 526 |     0.5095 |    0.0014 |

**Key question:** Does win rate increase monotonically with cs_rank_momentum?
If yes → model has learned momentum signal. If U-shaped or flat → reversion or noise.


---

### Cut 5 — MetaLabelModel Attribution

Comparison of walk-forward Sharpe with meta-model (Phase 46 result) vs without.

| Fold | With meta | Without meta | Meta contribution |
|---|---|---|---|
| 1 | +0.242 | +0.242 | +0.000 |
| 2 | +0.434 | +0.434 | +0.000 |
| 3 | +0.227 | +0.227 | +0.000 |
| **Avg** | **+0.301** | **+0.301** | **+0.000** |

**Verdict:** ⚠️  **META-MODEL CONTRIBUTES NEAR-ZERO** (+0.000 Sharpe). Drop it — it is adding complexity without signal.


---

### Cut 6 — MFE / MAE Distribution by Exit Type

MFE = Maximum Favorable Excursion (how far it went in our favor before exit).
MAE = Maximum Adverse Excursion (how far it went against us before exit).

| exit_reason   |   n |   avg_mfe |   avg_mae |   mfe_p25 |   mfe_p75 |   mae_p25 |   mae_p75 |
|:--------------|----:|----------:|----------:|----------:|----------:|----------:|----------:|
| STOP          | 165 |   0.01014 |   0.03224 |   0.00261 |   0.01359 |   0.01989 |   0.04029 |
| TARGET        |  56 |   0.06373 |   0.00016 |   0.03068 |   0.05626 |  -0.00081 |   0.00625 |
| TIME_EXIT     | 305 |   0.01357 |   0.0054  |   0.00606 |   0.01959 |   0.00111 |   0.00914 |

**Time-exit losers avg MFE:** 0.0079
⚠️  **Time-exit losers showed meaningful upside before reversing** — these could be target captures with tighter exit or earlier trailing stop.

**Stop-exit avg MFE:** 0.0101
⚠️  **Stop exits showed MFE before being stopped** — classic false stop pattern. Stop may be too tight OR -1.25×stop_pressure label is penalizing these setups.


---

### Cut 7 — Per-Fold Regime Analysis

**Fold 1:**
- Trades: 152 | Win rate: 51.3% | Avg pnl: 0.0016
- Avg MFE: 0.0223 | Avg MAE: 0.0116
- ORB+ entries: 3.3% | Inside-ORB entries: 30.3%
- Exit mix: {'TIME_EXIT': np.float64(0.553), 'STOP': np.float64(0.316), 'TARGET': np.float64(0.132)}

**Fold 2:**
- Trades: 222 | Win rate: 49.5% | Avg pnl: 0.0015
- Avg MFE: 0.0164 | Avg MAE: 0.0132
- ORB+ entries: 0.9% | Inside-ORB entries: 28.4%
- Exit mix: {'TIME_EXIT': np.float64(0.59), 'STOP': np.float64(0.302), 'TARGET': np.float64(0.108)}

**Fold 3:**
- Trades: 152 | Win rate: 52.6% | Avg pnl: 0.0012
- Avg MFE: 0.0155 | Avg MAE: 0.0151
- ORB+ entries: 1.3% | Inside-ORB entries: 23.7%
- Exit mix: {'TIME_EXIT': np.float64(0.592), 'STOP': np.float64(0.329), 'TARGET': np.float64(0.079)}


---

### Stop-Pressure Label Bias Analysis

The path_quality label uses `-1.25 × stop_pressure` which penalizes any trade
where price came within stop distance during the hold, even if it recovered.
This section checks if high-MAE trades that ultimately won are being mislabeled.

**Trades that touched stop zone (MAE ≥ 80% of stop distance):** 131 (24.9%)
- Win rate: 3.1% | Avg pnl: -0.0130

**Trades that did NOT touch stop zone:** 395 (75.1%)
- Win rate: 66.8% | Avg pnl: 0.0062

**Stop-zone-touched winners (label likely penalized):** 4
- Avg pnl: 0.0082

**Exit mix for stop-zone-touched trades:** {'STOP': np.float64(0.885), 'TIME_EXIT': np.float64(0.107), 'TARGET': np.float64(0.008)}

**Verdict:** ⚠️  **Stop-zone-touched trades win 63.8% LESS** — consistent with stop_pressure label bias. Consider reducing -1.25 coefficient.


---

## Phase 47 Experiment Ordering Recommendation

Based on the above diagnostics:

- **Phase 1 (drop meta-model):** CONFIRMED — meta contributes near-zero. Execute first.
- **Phase 3 (stop/target compression):** ELEVATED PRIORITY — true R:R is well below stated 2:1.
- **Phase 2 (XGBRanker):** Proceed as planned — highest expected impact.
- **Phase 4 (top-300 liquidity):** Proceed as planned.
- **Phase 5 (feature pack):** Proceed as planned — run last.