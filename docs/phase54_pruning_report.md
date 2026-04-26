# Phase 54 -- Intraday Feature Pruning Report

Generated: 2026-04-26
Pruning threshold: 0.005

## Full Feature Importance Ranking

| Rank | Feature | Importance | Status |
|---|---|---|---|
| 1 | prev_day_high_dist | 0.04667 | kept |
| 2 | prev_day_low_dist | 0.04447 | kept |
| 3 | volume_surge | 0.02730 | kept |
| 4 | cum_delta | 0.02329 | kept |
| 5 | minutes_since_open | 0.02324 | kept |
| 6 | ema20_dist | 0.02304 | kept |
| 7 | ret_30m | 0.02277 | kept |
| 8 | is_close_session | 0.02149 | kept |
| 9 | gap_pct | 0.02141 | kept |
| 10 | spy_rsi_14 | 0.02140 | kept |
| 11 | time_of_day | 0.02130 | kept |
| 12 | ema9_dist | 0.02103 | kept |
| 13 | daily_parkinson_vol | 0.02101 | kept |
| 14 | pullback_from_high | 0.02094 | kept |
| 15 | daily_vol_regime | 0.02093 | kept |
| 16 | stoch_k | 0.02089 | kept |
| 17 | rel_strength_vs_spy | 0.02088 | kept |
| 18 | ema_cross | 0.02035 | kept |
| 19 | spy_session_return | 0.02028 | kept |
| 20 | whale_candle | 0.01970 | kept |
| 21 | orb_breakout | 0.01947 | kept |
| 22 | range_vs_20d_avg | 0.01917 | kept |
| 23 | macd_hist | 0.01896 | kept |
| 24 | vwap_distance | 0.01878 | kept |
| 25 | ret_15m | 0.01837 | kept |
| 26 | daily_vol_percentile | 0.01834 | kept |
| 27 | green_bar_ratio | 0.01832 | kept |
| 28 | orb_position | 0.01826 | kept |
| 29 | range_compression | 0.01823 | kept |
| 30 | session_hl_position | 0.01818 | kept |
| 31 | williams_r | 0.01807 | kept |
| 32 | lower_wick_ratio | 0.01806 | kept |
| 33 | session_return | 0.01793 | kept |
| 34 | gap_fill_pct | 0.01792 | kept |
| 35 | orb_direction_strength | 0.01786 | kept |
| 36 | vol_x_momentum | 0.01773 | kept |
| 37 | upper_wick_ratio | 0.01768 | kept |
| 38 | obv_slope | 0.01760 | kept |
| 39 | rel_vol_spy | 0.01747 | kept |
| 40 | gap_followthrough | 0.01746 | kept |
| 41 | atr_norm | 0.01741 | kept |
| 42 | rsi_14 | 0.01721 | kept |
| 43 | consecutive_bars | 0.01718 | kept |
| 44 | vol_trend | 0.01712 | kept |
| 45 | bb_position | 0.01709 | kept |
| 46 | body_ratio | 0.01704 | kept |
| 47 | trend_efficiency | 0.01704 | kept |
| 48 | above_vwap_ratio | 0.01701 | kept |
| 49 | vwap_cross_count | 0.01668 | kept |
| 50 | is_open_session | 0.00000 | PRUNED |

## Summary
- Total features: 50
- Features to prune: 1 (importance < 0.005)
- Features kept: 49

## Features to Prune

- `is_open_session` (importance=0.000000)

## Walk-Forward Gate
Gate: avg Sharpe >= +1.275 (v23 baseline). Must not regress.
## Implementation Note

`is_open_session` was removed from both:
1. `FEATURE_NAMES` list (line 546 in `intraday_features.py`)
2. The feature computation itself (line 288) — `feats["is_open_session"] = ...` line deleted

Removing only from FEATURE_NAMES would still include the feature at training time
(the training pipeline builds feature names from the feats dict keys, not FEATURE_NAMES).
Both locations must be updated to truly prune a feature.
