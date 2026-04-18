# ML Model Training Experiments

## Baseline: v16 (Cross-Sectional LGBM, top-30 features)
- **AUC**: 0.495 — BELOW RANDOM. Failure mode identified.
- **Root causes**:
  1. `STEP_DAYS=5` with `FORWARD_DAYS=10` = consecutive windows share 5 forward days (label leakage)
  2. Cross-sectional ranking with mean-reversion: high-momentum stocks ranked higher but tend to revert
  3. Asymmetric ATR (1.5x target / 0.75x stop) created class imbalance (~30% winners)

---

## v17: Symmetric ATR + 12-Month Momentum + Non-Overlapping Windows
- **Branch**: `experiment/v17-symmetric-atr-12mo-momentum`
- **Model**: LightGBM, ATR-hit labels, top-25 features
- **Key changes**: STEP_DAYS 5→10, symmetric ATR (1.0x/1.0x), momentum_252d_ex1m, rs_vs_spy_60d
- **Results**: AUC **0.503** — essentially random
- **Analysis**: Symmetric labels fixed class imbalance but LightGBM found no predictive signal.
  High recall (71.1%) but near-random precision (48.4%) = model is guessing.

---

## v18: XGBoost+LightGBM Ensemble + Sector-Neutral Momentum
- **Branch**: `experiment/v18-ensemble-sector-neutral-momentum`
- **Model**: lgbm_ensemble (50/50 XGBoost+LightGBM), ATR-hit labels, top-30 features
- **Key changes**: lgbm_ensemble model, momentum_20d_sector_neutral, momentum_60d_sector_neutral,
  mean_reversion_zscore, up_day_ratio_20d
- **Results**: **CRASHED** (AttributeError: missing `_select_top_features` method — now fixed)
- **Status**: Fixed, needs re-run

---

## v19: Volume Confirmation + Quality Features + Tight Stop Labels
- **Branch**: `experiment/v19-quality-factor-better-labels`
- **Model**: lgbm_ensemble, ATR-hit with volume confirmation, top-30 features
- **Key changes**: Asymmetric ATR 1.5x/0.5x, VOL_CONFIRM_THRESHOLD=0.9, trend_consistency_63d,
  vol_price_confirmation, price_efficiency_20d
- **Results**: AUC **0.569** — BUT DEGENERATE (Precision 0%, Recall 0%, Accuracy 82.4%)
- **Analysis**: Volume confirmation filter was too aggressive — eliminated nearly all positive labels.
  Model learned to predict all-negative. AUC 0.569 = the underlying probability score HAD signal,
  but decision threshold was useless due to extreme class imbalance from the filter.

---

## v20: Longer Forward Window (21 days)
- **Branch**: `experiment/v20-longer-forward-window`
- **Model**: XGBoost, ATR-hit labels (1.5x/0.5x), no volume filter, top-25 features
- **Key changes**: FORWARD_DAYS 10→21, STEP_DAYS 10→21, removed volume confirmation filter
- **Results**: AUC **0.505** — essentially random
- **Analysis**: Longer window did not help. ATR-hit labeling appears fundamentally noisy
  regardless of window length. The binary ATR-hit outcome over 10-21 days is dominated by
  market noise, not stock-specific predictable factors.

---

## KEY INSIGHT: ATR-Hit Labels Are Too Noisy

All v17-v20 experiments with ATR-hit labels yielded AUC ~0.50-0.57. The labeling scheme itself
is the core problem:
- Whether a stock hits 1-1.5x ATR within 10-21 days is heavily driven by market/macro moves
- With only 82 symbols × ~60-120 windows = 5k-10k samples, XGBoost/LGBM overfits noise
- The signal-to-noise ratio of absolute ATR-hit is too low for these features to predict

## v21: Cross-Sectional Labels (spy_relative scheme)
- **Branch**: `experiment/v21-cross-sectional-spy-relative`
- **Model**: XGBoost, `spy_relative` label scheme (top quartile by SPY-adjusted 21-day return)
- **Rationale**:
  - Cross-sectional removes market noise: labels stock as "winner" only if it beats SPY by >2%
  - Removes absolute threshold sensitivity — pure relative performance
  - Top-quartile labeling ensures ~25% positive rate (no class imbalance)
  - Matches how institutional investors actually think about stock selection
- **Key changes**:
  - Label scheme: `spy_relative` (already implemented in training.py)
  - FORWARD_DAYS=21, STEP_DAYS=21 (longer window = less noise)
  - XGBoost with `scale_pos_weight` for ~75/25 class split
  - Top-25 features by mutual information
- **Status**: PLANNED

---

## What to Try Next (v22+)

### v22: More Data (7 years)
- If v21 AUC > 0.55: add more data (7yr) to reduce variance
- More windows = less overfitting, more stable feature importance

### v23: Purged Cross-Validation
- Embargo windows to prevent any train/test contamination
- Standard in quantitative finance for time-series ML

### v24: Factor Model Features
- Accruals ratio (cash earnings vs reported earnings quality)
- Debt momentum (change in leverage predicts underperformance)
- Analyst revision momentum (consensus EPS estimate changes)

### Key features to watch in v21:
- `momentum_252d_ex1m` — classic 12-1 month momentum factor
- `rs_vs_spy_60d` — 60-day relative strength
- `momentum_20d_sector_neutral` — pure alpha momentum
- `vol_of_vol` — was #1 in v16
- `fmp_surprise_2q_avg` — earnings consistency
