# MrTrader ML Improvement Brainstorm Prompt

You are an expert quant researcher. I need creative, technically grounded ideas for improving a swing trading ML model. Please read all the context below carefully — there's a lot of history of what's been tried and why things failed. I want fresh ideas that don't repeat what's already been ruled out.

---

## The System

A fully automated multi-agent swing trading system (Portfolio Manager → Risk Manager → Trader) running on Alpaca paper trading with SP-500 universe. Written in Python with XGBoost. The system runs daily: PM scores all SP-500 stocks, proposes top-N entries, RM validates against 10 rules, Trader executes.

**Capital**: $20,000 real, $100,000 simulated  
**Hold period**: Typically 1–5 days (swing)  
**Position sizing**: Up to 5 positions, 20% each (3 in bear regime)  
**Entry/exit**: Enter at next open after signal, exit on ATR-based stop or target or time  
**Stop**: 0.5× ATR below entry (on average ~1.5% below)  
**Target**: 1.5× ATR above entry (on average ~4.5% above)  
**Risk/reward**: 3:1 target-to-stop ratio

---

## Current Best Model — v110

**The only model we have that beats random.** Everything else regresses from this.

| Property | Value |
|---|---|
| Architecture | XGBoost classifier (400 trees, depth 4, lr=0.03) |
| Features | 84 price-only OHLCV features (see full list below) |
| Label scheme | Cross-sectional top-20% 5-day Sharpe-adjusted return within each rolling window |
| Training label in detail | For each 63-day rolling window, rank all stocks by their Sharpe-adjusted 5-day forward return. Label = 1 if stock is in top 20%, else 0. This is a RELATIVE label — not absolute direction. |
| Universe | SP-500 (~753 symbols) |
| Forward window | 5 days |
| Training window | 63 trading days (rolling, step 1 day) |
| Train samples | ~129,000 |
| Train period | 5 years |
| Test split | Last 25% of windows (time-based, with embargo) |
| OOS AUC | 0.638–0.641 |
| Scale_pos_weight | 4.0 (class ratio correction) |
| Threshold | 0.40 (tuned on val set to maximize F1) |
| Normalization | Cross-sectional z-score within each time window across all stocks |

**Top features by XGBoost importance:**
```
realized_vol_20d    0.083
volatility          0.081
vol_of_vol          0.080
atr_norm            0.079
parkinson_vol       0.068
ema_20              0.022
reversal_3d         0.019
vrp                 0.015
choch_detected      0.015
downtrend           0.014
```

The model is dominated by **volatility features** — it's essentially a volatility-rank screener. The top 5 features are all volatility measures. This may be a clue about what's really happening (see diagnostics below).

---

## v110 Tier 3 Backtest Results (the honest number — full agent simulation)

The "Tier 3" backtest runs the actual PM/RM/Trader code on historical bars. This is the go/no-go metric, not AUC.

| Metric | v110 |
|---|---|
| **Tier 3 Sharpe** | **+0.34** |
| Profit factor | 1.11 |
| Win rate | 40.3% |
| Trades | 290 |
| Stop exits | **70%** |
| Target exits | 30% |
| Total return | +1.9% over 2yr |
| Avg hold | ~2 bars |
| SPY benchmark Sharpe | ~1.5 (it's a bad alpha period) |

**Gate to reach**: Avg walk-forward OOS Sharpe > 0.8 across 3 folds, no fold below -0.3.  
**Current walk-forward result**: Avg -0.727 (all folds negative). Gate not met.

**Critical diagnostic: 70% stop exits.** The model picks stocks that reverse and hit the stop before reaching target. In a 3:1 R:R system, 70% stops with 30% targets means you need avg win >> avg loss to break even, which this barely achieves (PF 1.11).

---

## Full Feature List (84 features, all price-only OHLCV)

```
Technical oscillators:
  rsi_14, rsi_7
  macd, macd_signal, macd_histogram
  stoch_k, stochrsi_k, stochrsi_d, stochrsi_signal
  williams_r_14, cci_20, adx_14, adx_slope

Moving averages & price position:
  ema_20, ema_50
  price_above_ema20, price_above_ema50
  price_change_pct, price_to_52w_high, price_to_52w_low, near_52w_high
  dema_20_dist, keltner_position, bb_position

Momentum:
  momentum_5d, momentum_20d, momentum_60d, momentum_252d_ex1m
  momentum_20d_sector_neutral, momentum_60d_sector_neutral
  momentum_5d_sector_neutral, sector_momentum_5d
  reversal_5d, reversal_5d_vol_weighted, reversal_3d
  up_day_ratio_20d, trend_consistency_63d, consecutive_days
  uptrend, downtrend, price_acceleration

Volatility:
  volatility (annualized std 20d), realized_vol_20d (realized vol, log-returns)
  vol_of_vol (std of 10d vol windows), atr_norm (ATR/price)
  parkinson_vol (high-low range estimator), vrp (vol risk premium: realized - implied proxy)
  vol_percentile_52w, vol_regime, vol_of_vol

Volume:
  volume_ratio, volume_trend, volume_surge_3d
  vpt_momentum, cmf_20

Mean reversion / structure:
  mean_reversion_zscore, price_efficiency_20d
  range_expansion, vwap_distance_20d
  choch_detected, bars_since_choch, hh_hl_sequence
  consolidation_position

Pressure dynamics:
  pressure_persistence, pressure_displacement, pressure_index

WorldQuant 101 formulaic alphas (14):
  wq_alpha3, wq_alpha4, wq_alpha6, wq_alpha12, wq_alpha33, wq_alpha34,
  wq_alpha35, wq_alpha40, wq_alpha43, wq_alpha44, wq_alpha46,
  wq_alpha53, wq_alpha54, wq_alpha55

Interaction features:
  rsi_x_vix_regime, momentum20_x_vix_bucket, vol_pct_x_vix_bucket, adx_x_vix_bucket
```

Note: VIX features (vix_regime_bucket etc.) are available as **inference-time** features but during training `regime_score = 0.5` (neutral) is used to prevent look-ahead bias, since the live VIX score would be identical for all historical windows.

---

## Everything We've Tried (What Didn't Work and Why)

### Label Schemes

**Cross-sectional (current, v110):** Label = 1 if stock in top 20% by Sharpe-adjusted 5-day forward return within each window. Works best so far. Doesn't require predicting absolute direction — only relative ranking. AUC 0.638.

**Triple-barrier (v115, v116):** Label = 1 if price hits 1.5× ATR target before 0.5× ATR stop within 5 days, else 0. The idea was to train directly on what the Tier 3 backtester measures.
- **Asymmetric (1.5× target / 0.5× stop)**: 85% of stocks stop out → class imbalance 85:15 → model collapses to predict all-positive (recall=100%, AUC ~0.50). `scale_pos_weight` couldn't fix it.
- **Symmetric (1.5× / 1.5×)**: Expected ~50/50 split. Model still collapsed to all-positive. Root cause: 84 OHLCV features cannot predict absolute direction within 5 days reliably enough to overcome noise. The cross-sectional ranking approach sidesteps this problem.

**LambdaRank (v113, v114):** Ranking objective (LightGBM NDCG). AUC ~0.50 — random. LambdaRank with AUC metric doesn't converge for classification. Ruled out.

### Model Architecture

**XGBoost + LR ensemble (v118, Phase 44):** 70/30 soft-vote blend. Win rate went up +5pp but Sharpe dropped -0.42 (to -0.08). LR suppressed valid signals and cut trade count by 40%. The linear decision boundary can't capture the non-linear structure this feature set has.

### Sample Weighting

**VIX regime upweighting (v111, Phase 26a):** 1.5× weight for low-VIX training windows. Result: Sharpe +0.34 → -0.43, stop exits 70% → 76%. Biases toward calm-market patterns that fail when vol picks up. VIX is already in the features — using it as a weight was redundant and harmful.

### Feature Engineering

**Feature pruning: 140 → 84 (v117, Phase 43):** Removed 56 features with zero XGBoost importance. Result: Sharpe +0.34 → -0.15, stop exits 70% → 79%. Zero XGBoost importance ≠ useless for trading. The pruned features likely constrained which candidates reached the RM and at what scores.

### Inference-Time Filters (No Retrain)

**Wider stops at inference (1.0× ATR stop):** Training/inference mismatch. Model was trained with 0.5× ATR labels — wider stops expose more loss-space before exiting. Stop exits: 70% → 78%. Sharpe: +0.34 → +0.30.

**Min confidence 0.60:** Model never reaches 60% confidence. XGBoost probabilities cluster below 0.55 for this architecture. 0 trades.

**Vol filter ≤75th percentile:** Fewer trades (290 → 93), but quality worsened. Stop exits: 70% → 82%. Diversification lost without quality gain.

**Phase 34/35 filters (major improvement, but OOS collapses):**
Applied at inference time only (no retrain):
- No-chase filter: skip if open > prior_close × (1 + 0.75×ATR)
- EMA extension filter: skip if entry_price > EMA20 × (1 + 1.5×ATR)
- Bear regime gate: SPY < EMA200 → max_positions = 3
- VIX fear spike: VIX > 30 → skip all entries

Result: Full-window Tier 3 Sharpe +1.69 (5× improvement!). But **walk-forward OOS collapses**:
- Fold 1 (Jan–Oct 2024): Sharpe -0.19
- Fold 2 (Oct 2024–Jul 2025): Sharpe -1.45
- Fold 3 (Aug 2025–Apr 2026): Sharpe -0.55
- **Avg OOS Sharpe: -0.727**. Gate: > 0.8.

The in-sample Sharpe boost was an overfitting artifact of the filter parameters being calibrated on the same window used for evaluation. The underlying v110 signal has OOS win rate ~34-36% — this is the real problem.

---

## Key Diagnostics / What We Know

1. **70% stop exits is the core symptom.** Every experiment we've tried that changes labels, architecture, or filters either doesn't fix it or makes it worse. We need to understand WHY 70% of entries immediately reverse.

2. **The model is a volatility ranker.** Top 5 features by importance are all vol measures. It may be selecting high-vol stocks that rank high in cross-sectional Sharpe-adjusted returns *because they're volatile* — they have high Sharpe returns when they go up, but 70% of the time they also have high downward moves. This is a label artifact: cross-sectional Sharpe favors high-vol stocks.

3. **Cross-sectional labels create implicit vol bias.** The label = top-20% Sharpe-adjusted return within a window. High-vol stocks naturally appear in the top 20% more often because their returns have higher variance. The model is learning "pick the most volatile stock right now" which is not the same as "pick the stock most likely to hit target before stop."

4. **AUC 0.638 is real but limited.** The model does rank slightly better than random. But AUC measures ranking ability within the training label space (cross-sectional rank), not ability to predict stop-vs-target outcomes. The two are different metrics.

5. **Walk-forward collapses.** Even the best inference-time filtered result (Sharpe +1.69) collapses OOS. This suggests the filters and/or the underlying model are overfit to the 2021-2025 training regime. The OOS periods (2024-2026) seem to have a different return structure.

6. **The 5-day horizon creates noise.** Average hold period from Tier 3 is ~1.4 bars (entry day + exit day), not 5. So the model is trained to rank 5-day outcomes but most trades exit by day 2. The label horizon and the actual exit horizon are mismatched.

7. **Fundamentals didn't help.** We tried PE ratio, PB ratio, profit margin, revenue growth, D/E ratio, earnings proximity — all got pruned (zero XGBoost importance) and removing them didn't help either. This signal is purely price/volume based.

8. **Class balance:** ~20% positives (top-20% CS rank). With scale_pos_weight=4.0. Model outputs probabilities clustered below 0.55 even for "positive" predictions.

---

## What We Haven't Tried Yet

- **Recency bias / time-decay weighting**: exponentially downweight training samples older than 1yr
- **15-day forward window** (10-day was tried, was worse than 5-day; 15-day not yet tried)
- **ATR-adaptive labels**: scale target/stop thresholds by per-stock ATR percentile
- **Different training universe filters**: e.g., only train on stocks with ADX > 20 (trending)
- **Sector-neutral training**: train separate model per sector or add sector fixed effects
- **Time-series cross-validation**: use expanding windows instead of rolling windows
- **Target metric alignment**: optimize directly for Sortino or Calmar rather than AUC
- **Directional features**: put/call ratio, options flow, insider buys (tried earlier, got pruned; may need different form)
- **Longer lookback for mean-reversion features**: 63-day momentum already, but 126-day not in feature set

---

## Technical Constraints

- **Must use Tier 3 backtest result (not AUC or Tier 1/2)** as go/no-go. Tier 1/2 regularly show Sharpe 2–7 while Tier 3 shows -1 to +0.34.
- **One change per retrain** (isolation requirement for attribution).
- **Training time budget**: ~4 minutes for 5yr, 753 symbols, 8 workers, no fundamentals.
- **Must match inference schema**: whatever features/labels we train on must be computable at inference time without look-ahead.
- **Triple-barrier labels**: ruled out for now — can revisit if feature set improves dramatically, but 84 OHLCV features can't predict absolute 5-day direction reliably.
- **LambdaRank**: ruled out — doesn't converge for this use case.

---

## My Specific Questions for You

1. **Volatility bias fix**: The model appears to be a volatility ranker, not a momentum/direction predictor. How do I make the cross-sectional label scheme less biased toward high-vol stocks? Should I Sharpe-normalize by vol rank, or use a different relative ranking approach entirely?

2. **Label-to-exit alignment**: The model trains on 5-day outcomes but exits in ~1.4 bars on average. What label scheme would better align with how the Tier 3 backtester actually closes positions?

3. **70% stop exits root cause**: What's your diagnosis? Is this a label problem, a feature problem, an entry timing problem, or something structural about the trade setup (stop at 0.5× ATR is just too tight for 5-day swing trades)?

4. **Recency bias implementation**: If recent market regimes are very different from 2021-2023 training data, what's the best way to upweight recency without overfitting to the most recent period? Exponential time decay? Rolling train windows? Some form of domain adaptation?

5. **Feature ideas**: Given that the top 5 features are all volatility measures and the model plateaus at AUC 0.641, what fundamentally different features could break through? We've tried: all standard technicals (RSI, MACD, ATR, BB, Stoch, ADX, CCI, Williams%R), momentum (5d through 252d), volume dynamics (VPT, CMF, volume surge), market structure (ChoCh, HH/HL, consolidation), WorldQuant 101 alphas, pressure dynamics, VWAP distance, sector momentum, and VIX regime interactions. What's left that could add signal?

6. **Alternative architectures**: Beyond XGBoost alone, what architectures are known to work for 5-day equity ranking? (e.g., temporal fusion transformers, LightGBM with different objectives, neural ranking models) Is the problem architecture-limited at AUC 0.641 or data/feature-limited?

7. **Walk-forward generalization**: The biggest failure mode is that in-sample results don't transfer OOS (even inference-time filters that look great in-sample collapse in walk-forward). What techniques specifically address regime-shift-driven walk-forward failure?

8. **Stop distance calibration**: With a 3:1 R:R ratio (target = 1.5× ATR, stop = 0.5× ATR) and 70% stop exits, the theoretical edge barely breaks even (0.30 × 3R - 0.70 × 1R = -0.1R). Is the stop/target structure itself wrong for this holding period? What's the empirically optimal stop/target for 1-2 day swing trades in SP-500?

9. **What would you do differently from scratch?** If you were building a swing equity ranking model for SP-500 with 5yr daily OHLCV data, $20k capital, 1-5 day holding period, and the constraint that you must use a Tier 3 agent simulation as your eval metric — what would your approach be from scratch? Don't be constrained by what I've tried.

10. **Quick wins vs architectural changes**: Given the constraints above, what do you think is most likely to move the needle in the next 2-3 experiments? Please prioritize: quick wins (1-2 day implementation) vs deeper architectural changes (1 week+).

---

## Output Format I'd Like

Please structure your response as:
1. **Diagnosis**: Your read on why v110 is stuck at +0.34 Sharpe and 70% stop exits
2. **Top 3 highest-priority experiments**: ranked by expected impact × implementation difficulty, with specific implementation details (what exactly to change, in what file, what to expect)
3. **Longer-term ideas**: things worth exploring if the top 3 don't pan out
4. **Things to avoid**: based on the history above, what would you also rule out
5. **Wild card**: one unconventional idea that might surprise us

Be specific. "Try different features" is not useful. "Add the ratio of 5-day realized vol to 20-day realized vol as a vol-of-vol measure — this captures vol acceleration that precedes directional moves" is useful.
