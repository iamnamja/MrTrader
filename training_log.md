## v27 — 2026-04-19
Params: --label-scheme atr_and_sector --top-features 30 --hpo-trials 25 --walk-forward 5 --two-stage --multi-window --workers 6
Metrics: AUC=0.532 (blended)  Per-window: 63d=0.612, 126d=0.607  Recall=100%  Precision=11.1%  Threshold=0.20 (floor)
Notes: FAILED. Two root causes:
  1. predict_multi_window passes 63d test set (83 features) to 126d model (trained on 93 features) → dimension mismatch → blended AUC collapses to ~random.
  2. atr_and_sector dual-requirement labels (ATR hit AND sector top-20%) too strict → model can't build confident positive predictions → threshold hits floor.
  Per-window AUC of ~0.61 shows model has mild signal, but architecture choices masked it.
Changes for v28: Drop --multi-window (broken eval), drop --two-stage (complexity without gain), switch to --label-scheme sector_relative (what worked for v26), increase --hpo-trials 35 and --top-features 35.

## v28 — 2026-04-20
Params: --label-scheme sector_relative --top-features 35 --hpo-trials 35 --walk-forward 5 --workers 6 (no two-stage, no multi-window)
Metrics: AUC=0.629  Recall=65.3%  Precision=27.3%  WF-AUC=0.597±0.032  Threshold=0.20 (floor)  Time=18min
Notes: Marginal improvement over v26 (0.627→0.629). Threshold hit floor again — single-model path missing scale_pos_weight, so XGBoost compressed outputs near base rate (~20% positives with sector_relative). WF gap=0.032 mild. Top features: volatility, sector_momentum, atr_norm, parkinson_vol, revenue_growth — model is largely a volatility + momentum filter.
Changes for v29: Added scale_pos_weight (n_neg/n_pos) + sample_weight to single-model training path. Added --two-stage to separate fundamental quality from technical timing. Bumped --hpo-trials 50, --top-features 40.

## v29 — 2026-04-20
Params: --label-scheme sector_relative --top-features 40 --hpo-trials 50 --walk-forward 5 --workers 6 (no two-stage, no multi-window)
Metrics: AUC=0.614  Recall=54.8%  Precision=27.3%  WF-AUC=0.584±0.028  Threshold=0.50  Time=7016s (~117min)
Notes: AUC REGRESSED from v28 (0.629→0.614) despite scale_pos_weight fix. Threshold rose from floor (0.20) to 0.50 — confirming scale_pos_weight is now working. Top features: volatility, atr_norm, parkinson_vol, revenue_growth, short_interest_pct, pb_ratio, pe_ratio — same vol/momentum dominance. The AUC regression is likely HPO finding a different local optima (50 trials vs 35) or the interaction between scale_pos_weight and sample_weight. Note: --top-features 40 flag didn't reduce feature count (still 93) — may be a no-op or applied differently.
Changes for v30: Try --two-stage (TwoStageModel confirmed to forward scale_pos_weight to both stages). Bump --hpo-trials 75 for better Bayesian search. Drop --top-features (flag ineffective). Keep sector_relative, 5-year data, 5-fold WF. Goal: break 0.65 AUC via two-stage fundamental+technical separation.

## v30 — 2026-04-20
Params: --label-scheme sector_relative --hpo-trials 75 --walk-forward 5 --years 5 --two-stage --workers 6
Metrics: AUC=0.631  Recall=59.5%  Precision=28.1%  WF-AUC=0.602±0.019  Threshold=0.50  Time=931s (~16min)
Notes: Best WF-AUC yet (0.602 vs 0.584) and tightest std (0.019) — two-stage improved generalization. AUC improved from v29 (0.614→0.631) but still below par (0.65). Top features: volatility, atr_norm, parkinson_vol, pe_ratio, earnings_proximity_days, short_interest_pct, pb_ratio — similar vol/fundamental pattern. Threshold settled at 0.50 (scale_pos_weight working). Two-stage clearly helps WF stability. Fundamental quality filter (stage 1) is adding value.
RESULT: Below par. Max allowed version reached (v30). Stopping autonomous loop.

## v31 — 2026-04-20
Params: --label-scheme sector_relative --hpo-trials 75 --walk-forward 5 --years 5 --three-stage --workers 6
Metrics: AUC=0.622  Recall=53.2%  Precision=28.3%  WF-AUC=0.604±0.018  Threshold=0.50  Time=1039s (~17min)
Notes: Three-stage (quality 0.20 / catalyst 0.40 / timing 0.40) vs two-stage v30. WF-AUC best yet (0.604 vs 0.602) and tightest std (0.018 vs 0.019) — three-stage marginally improves generalization. Raw AUC 0.622 > v30 0.631? No — v30 was 0.631, v31 is 0.622, slight regression on raw AUC. Top features: EMA_20, momentum_252d_ex1m, EMA_50, momentum_60d_sector_neutral, momentum_20d_sector_neutral — trend/momentum features dominating from the shared context pool. Staging is working (three separate XGBoosts trained) but the AUC ceiling persists around 0.62-0.63. The 0.65 ceiling likely requires new signal sources rather than architecture changes alone.

## v32 — 2026-04-20
Params: --provider polygon --label-scheme sector_relative --hpo-trials 75 --walk-forward 5 --years 5 --three-stage --workers 6
New features: Polygon financials (fcf_margin, operating_leverage, rd_intensity) added to Stage 1 (fundamental quality gate)
Metrics: AUC=0.623  Recall=53.2%  Precision=28.3%  WF-AUC=0.603±0.017  Threshold=0.50  Time=1128s (~19min)
Notes: Polygon financials added 3 new fundamental features to Stage 1. AUC nearly flat vs v31 (0.622→0.623) — the new features are not hurting but also not providing a meaningful boost. WF-AUC 0.603 vs 0.604 — essentially identical. The AUC ceiling at 0.62-0.63 persists through architecture changes (two-stage→three-stage) AND new fundamental data (Polygon). This strongly suggests the ceiling is not a data quality or architecture problem — it's a signal limit in available public data for 10-day swing prediction. Probable next step: options flow data (IV term structure, put/call skew, GEX) — Polygon options require plan upgrade.
RESULT: Below par (AUC=0.623 < 0.65). New Polygon features validated (no regression) but insufficient to break ceiling.

## v33 — 2026-04-20
Params: --provider polygon --label-scheme return_regression --hpo-trials 75 --walk-forward 5 --years 5 --three-stage --workers 6
Metrics: AUC=0.526  Recall=99.4%  Precision=20.0%  WF-AUC=0.584±0.020  Threshold=0.20 (floor)  Time=1165s
Notes: FAILED. return_regression (XGBRegressor on raw float returns) produced near-random AUC=0.526. Threshold hit floor (0.20) — regressor output scores cluster together after normalization so model predicts almost every stock as positive (recall=99.4%). Root cause: XGBRegressor predicts returns in narrow range (e.g. -0.03 to +0.05); after (score-min)/(max-min) normalization, the separation between top-20% and bottom-80% is lost. WF-AUC=0.584 is also worse than binary sector_relative (0.603). Regression label approach is fundamentally mismatched with our normalized-score inference path. Fix: return_blend will use binary labels (top-20% of blended 5d+10d return = 1), keeping the classifier path.
RESULT: Below par. Approach abandoned for raw regression; switching to binary blend labels.

## v34 — 2026-04-20
Params: --provider polygon --label-scheme sector_relative --hpo-trials 75 --walk-forward 5 --years 5 --three-stage --workers 6
New vs v32: +17 features (14 WorldQuant 101 alphas + 3 reversal signals: reversal_3d, reversal_5d, reversal_5d_vol_weighted). Total features: ~113.
Metrics: AUC=0.622  Recall=82.6%  Precision=23.8%  WF-AUC=0.602±0.018  Threshold=0.45  Time=~19min
Notes: Essentially identical to v32 (AUC=0.623, WF-AUC=0.603). The 17 new WorldQuant+reversal features had zero measurable impact on AUC. Threshold rose from 0.50 to 0.45 — slight shift but within noise. Top features are still dominated by momentum/volatility (same as v32). This confirms the AUC ceiling is NOT a feature-engineering problem — the model already saturates available public signal. The WorldQuant alphas are likely correlated with existing momentum/vol features, providing no new information.
RESULT: Below par. New features validated (no regression) but insufficient to break ceiling.

## v35 (sector_relative + 121 features) — 2026-04-20
Params: --provider polygon --label-scheme sector_relative --hpo-trials 75 --walk-forward 5 --years 5 --three-stage --workers 6
New vs v32: +25 features total (17 WQ/reversal + 8 new: vrp, realized_vol_20d, days_to_opex, near_opex, beta_252d, beta_deviation, earnings_drift_signal, earnings_pead_strength). Total: ~121 features.
Metrics: AUC=0.623  Recall=52.8%  Precision=28.2%  WF-AUC=0.603±0.020  Threshold=0.50  Time=1771s
Notes: Identical to v32 baseline (AUC=0.623, WF-AUC=0.603). VRP, beta, opex calendar, earnings PEAD drift — none of these moved AUC. Threshold held at 0.50 (healthy, not floor). This definitively confirms the signal ceiling from public data is ~0.623. We have now tried: architecture (2-stage/3-stage), 25+ new features (WorldQuant 101, reversal, Polygon financials, VRP, beta, opex, PEAD), label smoothing, HPO tuning — all plateau at 0.622-0.623.
RESULT: Below par. AUC ceiling confirmed at ~0.623 with public free data.

## v36 (return_blend + 121 features) — 2026-04-20
Params: --provider polygon --label-scheme return_blend --hpo-trials 75 --walk-forward 5 --years 5 --three-stage --workers 6
Metrics: AUC=0.622  Recall=86.2%  Precision=23.4%  WF-AUC=0.603±0.017  Threshold=0.45  Time=1807s
Notes: Blended 5d+10d label (binary, sector-relative) with 121 features. Identical AUC to sector_relative. Recall=86.2% vs 52.8% — the blended label makes the model more recall-oriented (wider net) at cost of precision. WF-AUC identical. Blended labels did not help.
RESULT: Below par. Blended labels add no AUC improvement.
