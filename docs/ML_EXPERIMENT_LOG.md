# ML Experiment Log

Tracks every model improvement iteration: what was changed, which model, baseline vs. result, and verdict.

---

## How to Read This Log

- **Baseline AUC** = measured *before* the change on the same test set
- **Result AUC** = measured *after* the change on the same test set
- **Verdict**: ✅ Keep | ❌ Revert | ⚠️ Partial (kept with caveats) | 🔄 Pending
- Model versions are DB `swing` or `intraday` version numbers from `ModelVersion` table

---

## Baseline Snapshot (pre-improvement, 2026-04-21)

### Swing Model
| Item | Value |
|---|---|
| Model version | v92 |
| Architecture | XGBoost LambdaRank |
| Universe | SP-500 (~446 symbols) |
| Training samples | 39,089 |
| Test samples | 13,074 |
| AUC | **0.5748** |
| Precision | 0.2945 |
| Recall | 0.1969 |
| Threshold | 0.35 |
| Label scheme | Raw top-20% 10-day forward return (cross-sectional) |
| Feature count | ~126 |
| Train/test split | Simple chronological split, no embargo |
| Feature normalization | None (raw values) |
| Top SHAP features | sector_momentum (0.90), revenue_growth (0.19), vol_of_vol (0.15), wq_alpha40 (0.15) |
| Known issues | sector_momentum dominates SHAP 5× next feature; no train/test embargo; raw return labels |

### Intraday Model
| Item | Value |
|---|---|
| Model version | v15 |
| Architecture | XGBoost + LightGBM soft-vote ensemble |
| Universe | Russell 1000 (~1000 symbols) |
| Bar size | 5-min |
| Horizon | 24 bars (~2 hours) |
| Feature count | 37 (pure OHLCV, no external APIs) |
| Backtest Sharpe | 2.55 |
| Win rate | 54% |
| Cumulative return | +4.4% over 60 days |
| Label scheme | Raw top-20% 2h forward return (cross-sectional per day) |
| Train/test split | Simple chronological split, no embargo |
| Feature normalization | None |
| Known issues | No 24-bar embargo; no session-time features; no cross-sectional normalization |

---

## Improvement Roadmap

| # | Technique | Models | Status | Branch |
|---|---|---|---|---|
| 1 | 10-day / 24-bar train/test embargo | Swing + Intraday | 🔄 Pending | — |
| 2 | Cross-sectional feature normalization (z-score within date×sector) | Swing + Intraday | 🔄 Pending | — |
| 3 | Risk-adjusted labels — Sharpe-based (swing) / BOLB breakout (intraday) | Swing + Intraday | 🔄 Pending | — |
| 4 | Session-time features (minutes since open, session segment flag) | Intraday only | 🔄 Pending | — |
| 5 | Sector-relative momentum feature (stock return minus sector ETF return) | Swing only | 🔄 Pending | — |
| 6 | DDG-DA regime-adaptive resampling | Swing + Intraday | 🔄 Pending | — |

---

## Iteration 1 — Embargo + Cross-Sectional Normalization

**Branch:** `feature/model-improvement-iter1`  
**Models:** Swing + Intraday  
**Date started:** 2026-04-21  
**Date completed:** —

### What We Changed

#### 1a. Train/Test Embargo
- **Swing**: Added 10-day embargo between train and test windows (matches label horizon)
- **Intraday**: Added 24-bar embargo between train and test windows (matches 2h forward label horizon)
- **Why**: Consecutive windows share label formation periods → data leakage inflates in-sample AUC
- **Files changed**: `app/ml/training.py`, `app/ml/intraday_training.py`

#### 1b. Cross-Sectional Feature Normalization
- Z-score all features within each `(date, sector)` group before training and inference
- Formula: `(feature - group_mean) / group_std` per date × sector
- **Why**: Removes systematic sector/regime bias; directly attacks `sector_momentum` SHAP dominance
- **Files changed**: `app/ml/training.py`, `app/ml/intraday_training.py`, `app/ml/feature_engineer.py` (if applicable)

### Results

#### Swing
| Metric | Baseline (v92) | Result | Delta |
|---|---|---|---|
| AUC | 0.5748 | — | — |
| Precision | 0.2945 | — | — |
| Recall | 0.1969 | — | — |
| n_train | 39,089 | — | — |
| Top SHAP #1 | sector_momentum (0.90) | — | — |

#### Intraday
| Metric | Baseline (v15) | Result | Delta |
|---|---|---|---|
| Backtest Sharpe | 2.55 | — | — |
| Win rate | 54% | — | — |
| AUC | — | — | — |

**Verdict:** 🔄 Pending  
**Notes:** —

---

## Iteration 2 — Risk-Adjusted Labels

**Branch:** `feature/model-improvement-iter2`  
**Models:** Swing + Intraday  
**Date started:** —  
**Date completed:** —

### What We Changed

#### 2a. Swing — Sharpe-Based Labels
- Replace raw top-20% 10-day return with: `(forward_return - cross_sectional_mean) / cross_sectional_std` per date
- Stocks ranked by risk-adjusted outperformance, not raw return
- **Why**: Raw returns overweight high-volatility stocks during bull runs; Sharpe label captures skill not luck
- **Files changed**: `app/ml/training.py`

#### 2b. Intraday — BOLB Breakout Labels
- Replace raw top-20% 2h return with: did price break +N% threshold within 24 bars? → {1, 0}
- Threshold TBD (likely 0.3–0.5% for 2h window based on typical intraday ranges)
- **Why**: Binary breakout label is more actionable and less noisy than continuous return for intraday signals
- **Files changed**: `app/ml/intraday_training.py`

### Results

#### Swing
| Metric | Post-Iter1 | Result | Delta |
|---|---|---|---|
| AUC | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |

#### Intraday
| Metric | Post-Iter1 | Result | Delta |
|---|---|---|---|
| Backtest Sharpe | — | — | — |
| Win rate | — | — | — |

**Verdict:** 🔄 Pending  
**Notes:** —

---

## Iteration 3 — Session-Time Features (Intraday Only)

**Branch:** `feature/model-improvement-iter3`  
**Models:** Intraday only  
**Date started:** —  
**Date completed:** —

### What We Changed

- Add `minutes_since_open` (0–390) as a continuous feature
- Add `session_segment` one-hot: `open` (0–30 min), `midday` (30–330 min), `close` (330–390 min)
- **Why**: Open auction and close have distinct microstructure; lunch lull has mean-reversion tendency; model currently has no time-of-day awareness
- **Files changed**: `app/ml/intraday_training.py`

### Results

#### Intraday
| Metric | Post-Iter2 | Result | Delta |
|---|---|---|---|
| Backtest Sharpe | — | — | — |
| Win rate | — | — | — |
| AUC | — | — | — |

**Verdict:** 🔄 Pending  
**Notes:** —

---

## Iteration 4 — Sector-Relative Momentum (Swing Only)

**Branch:** `feature/model-improvement-iter4`  
**Models:** Swing only  
**Date started:** —  
**Date completed:** —

### What We Changed

- Replace or augment `sector_momentum` with `momentum_vs_sector_etf`: stock's 20-day return minus its sector ETF's 20-day return
- Sector ETF map: XLK (tech), XLF (finance), XLV (health), XLE (energy), XLI (industrial), XLY (consumer disc.), XLP (consumer stap.), XLU (utilities), XLB (materials), XLRE (real estate), XLC (comms)
- **Why**: Raw sector_momentum captures whether the sector is rising (regime), not whether the stock is beating its sector (alpha). SHAP 0.90 dominance suggests it's acting as a regime proxy.
- **Files changed**: `app/ml/training.py`, `app/ml/fundamental_fetcher.py` (or feature computation layer)

### Results

#### Swing
| Metric | Post-Iter2 | Result | Delta |
|---|---|---|---|
| AUC | — | — | — |
| sector_momentum SHAP | — | — | — |
| momentum_vs_sector_etf SHAP | — | — | — |

**Verdict:** 🔄 Pending  
**Notes:** —

---

## Techniques Evaluated and Ruled Out

| Technique | Reason Not Pursued |
|---|---|
| v37 model (AUC 0.757) | Degenerate: precision=0.0006, recall=1.0, threshold=0.2 — predicted everything as buy |
| Lowering AUC gate to 0.60 | Masks real problem; fixing labels/embargo is better than accepting lower bar |

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-21 | Expanded swing universe SP-100 → SP-500 | More diverse training data; SP-100 only had 4K train samples |
| 2026-04-21 | Set AUC drift threshold at 0.65 | Based on v37 which was later found to be degenerate |
| 2026-04-21 | AUC drift threshold needs revisiting | True SP-100 steady-state was ~0.60; 0.65 gate is too tight |
