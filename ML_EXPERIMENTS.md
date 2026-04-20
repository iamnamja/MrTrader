# ML Model Training Experiments

## Results Summary

| Version | Label Scheme | Model | AUC | Verdict |
|---------|-------------|-------|-----|---------|
| v16 | cross_sectional | LGBM | 0.495 | Below random — label leakage |
| v17 | atr (symmetric 1x/1x) | LightGBM | 0.503 | Random |
| v18 | atr (symmetric 1x/1x) | lgbm_ensemble | CRASHED | Missing _select_top_features (fixed) |
| v19 | atr (1.5x/0.5x) + vol confirm | lgbm_ensemble | 0.569* | Degenerate all-negative |
| v20 | atr (1.5x/0.5x), 21-day window | XGBoost | 0.505 | Random |
| v21 | spy_relative (beats SPY >2%) | XGBoost | 0.504 | Random |

*v19 AUC 0.569 is misleading — Precision=0%, Recall=0%, model predicted all-negative

---

## Root Cause Analysis: Why Everything Is ~0.50

After 6 experiments, the pattern is clear: **label scheme and model architecture are not the problem**.
Every reasonable combination yields AUC 0.50-0.51 (or degenerate predictions).

### The Real Issues:

1. **Too few samples for the feature count**
   - 82 symbols × ~60-120 windows = 5k-10k total samples (before filtering)
   - After 75/25 train/test: ~4k-7k train samples
   - With 83 features, XGBoost/LightGBM can easily memorize noise → overfits, generalizes randomly
   - Rule of thumb: need ~100 samples per feature minimum = 8,300+ train samples for 83 features
   - **Fix**: More symbols (S&P 500 instead of S&P 100), more years (7-10 vs 5)

2. **Short-term equity returns are near-efficient**
   - 10-21 day absolute returns are dominated by market/sector moves, not stock-specific factors
   - Academic factor literature (Fama-French, AQR) shows factor premia on 1-12 MONTH horizons, not 10-21 days
   - Our features (momentum, technical indicators, fundamentals) are appropriate — the prediction horizon is too short
   - **Fix**: Use 63-day (quarterly) forward returns

3. **Cross-sectional rank information is discarded**
   - We label each stock independently (binary: hit ATR or not)
   - Relative performance of stocks within a window is much more predictable than absolute returns
   - If we trained on "rank of return vs all other stocks this window" we'd use 82× more signal
   - **Fix**: True cross-sectional labeling across all stocks per time period

4. **Feature redundancy reduces effective signal**
   - 83 features with high mutual correlation (multiple momentum features, multiple vol features)
   - Correlation dilutes feature importance, harder for tree models to find signal
   - **Fix**: PCA or stricter feature selection (top 10 instead of top 25)

---

## v22: Planned — Quarterly Labels + Full S&P 500
- **Hypothesis**: The core issue is horizon + sample count
- **Changes**:
  - FORWARD_DAYS=63 (quarterly return prediction — matches factor literature)
  - STEP_DAYS=21 (still non-overlapping from label perspective, more windows)
  - Use S&P 500 (500 symbols) instead of S&P 100 (82 symbols) → 6× more data
  - XGBoost with top-15 features by mutual information
  - spy_relative labels: beat SPY by >3% over 63 days
- **Expected train samples**: 500 symbols × 60 windows × 0.75 = ~22,500 (vs current 4k-7k)
- **Status**: PLANNED

---

## What to Fix Before v22

1. **Merge v17-v21 branches to main** — code improvements are real (stale cache fix,
   extended ModelTrainer constructor, new features, spy_relative label scheme)
2. **Re-run v18** after crash fix — need lgbm_ensemble results for comparison
3. **Expand symbol universe** — add S&P 400/500 tickers to constants.py

---

## Branches Status

| Branch | Status | Notes |
|--------|--------|-------|
| experiment/v17-symmetric-atr-12mo-momentum | Ready to merge | AUC 0.503 |
| experiment/v18-ensemble-sector-neutral-momentum | Fixed, needs re-run | Missing method fixed |
| experiment/v19-quality-factor-better-labels | Ready to merge | AUC 0.569 (degenerate) |
| experiment/v20-longer-forward-window | Ready to merge | AUC 0.505 |
| experiment/v21-cross-sectional-spy-relative | Ready to merge | AUC 0.504 |
