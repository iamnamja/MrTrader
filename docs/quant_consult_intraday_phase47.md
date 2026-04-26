# MrTrader — Intraday Model Quant Consultation Brief
**Date:** 2026-04-26  
**Purpose:** You are a world-class quantitative researcher. I need your help designing Phase 47 of an intraday ML trading model. Below is a complete technical brief of the system, what we've tried, what worked, what failed, and the current results. Your job is to suggest the highest-leverage next experiments to close the remaining Sharpe gap.

---

## System Overview

**MrTrader** is a multi-agent paper/live trading system targeting US equities with a $20k account on Alpaca. It has two independent strategies:

- **Swing model**: 5-day hold, SP-100 universe. **GATE PASSED** (+1.181 avg Sharpe). Ready for paper trading.
- **Intraday model**: 2-hour hold (24 bars of 5-min data), Russell 1000 universe. **Gate NOT met.** This is what we're working on.

### Agent Architecture
Three agents running sequentially each day:
1. **Portfolio Manager (PM)**: Scores all Russell 1000 stocks with ML model → proposes top-5 entries
2. **Risk Manager (RM)**: Validates proposals against risk rules (position size, sector concentration, daily loss limit, drawdown)
3. **Trader**: Executes entries via Alpaca API

### Intraday Trade Mechanics
- **Universe**: Russell 1000 (~766 symbols with 2yr Polygon 5-min OHLCV cache)
- **Entry window**: After 12 bars (1 hour) of market open — features built on first 60 minutes
- **Hold period**: Up to 24 bars (2 hours) from entry
- **Exit logic**: ATR-adaptive stop/target OR time exit at bar 24
  - Stop: 0.6× prior-day range
  - Target: 1.2× prior-day range
  - Time exit: final close at bar 24 if neither hit
- **Position sizing**: 3% of equity per position, max 5 simultaneous positions
- **Transaction cost**: 0.05% per side

### Walk-Forward Validation (Tier 3)
3-fold expanding-window walk-forward. Each fold trains on all data up to fold start, tests on next 6 months:
- Fold 1: test 2024-10-14 → 2025-04-15
- Fold 2: test 2025-04-16 → 2025-10-15  
- Fold 3: test 2025-10-16 → 2026-04-17

**Gate to pass:** avg Sharpe > +0.80 across all folds, no single fold below -0.30.

---

## Current Model Architecture

### Base Model: XGBoost Classifier (v22)
- **Architecture**: XGBClassifier, binary classification
- **Label**: path_quality regression score → cross-sectional top-20% = label 1
  - `path_quality = 1.0 × upside_capture − 1.25 × stop_pressure + 0.25 × close_strength`
  - where `upside_capture = min((max_high - entry) / target_dist, 1.0)` over 24-bar horizon
  - `stop_pressure = min((entry - min_low) / stop_dist, 1.0)` over 24-bar horizon
  - `close_strength = clip((final_close - entry) / target_dist, −1, 1)` at bar 24
- **OOS AUC**: 0.5438 (HPO CV AUC: 0.5649)
- **Training data**: 722 symbols × 2 years of 5-min bars = ~280k samples
- **HPO**: Optuna, 20 trials, CV AUC objective

### MetaLabelModel v1
- **Architecture**: XGBRegressor trained on 810 in-sample trade outcomes
- **Label**: realized pnl_pct per trade
- **R2**: 0.001, corr: 0.044 (very weak — limited by base model signal quality)
- **Gate**: Enter only if predicted E[pnl] > 0
- **Effect**: Reduced trades from 252 → 175 per fold, improved Sharpe

### PM Abstention Gate
- Skip all new intraday entries on days where: VIX ≥ 25 OR SPY < 20-day SMA
- Applied both in backtest and live system

---

## Feature Set (42 features, all derived from 5-min OHLCV + SPY context)

### Opening Range Breakout (ORB)
- `orb_high`, `orb_low`: high/low of first bar (first 5 minutes)
- `orb_range`: `(orb_high - orb_low) / orb_low` — normalized ORB width
- `orb_breakout`: +1 if price above ORB high, -1 if below ORB low, 0 inside
- `orb_position`: normalized position within ORB range (0=at low, 1=at high)
- `orb_direction_strength`: `(last_close - orb_mid) / orb_range` — added in Phase 46

### Price Action (1-hour window)
- `price_momentum`: 60-min return from open to current
- `high_low_range`: normalized 60-min high-low range
- `close_position`: current close relative to 60-min range
- `price_velocity`: last 12-bar price change rate
- `gap_pct`: overnight gap vs prior close

### Volume
- `volume_surge`: current 60-min volume vs 20-day average volume (same time window)
- `whale_candle`: binary — did any 5-min bar have volume > 3× its 20-bar rolling average
- `volume_trend`: slope of volume over 60 minutes
- `volume_acceleration`: second derivative of volume

### Technical Indicators (computed over 60-min window)
- `rsi_6`, `rsi_14`: RSI at 6 and 14 periods
- `macd_signal`: MACD signal line
- `bb_position`: position within Bollinger Bands
- `atr_norm`: ATR normalized by price (proxy for volatility)
- `vwap_distance`: (price - VWAP) / price
- `vwap_cross_count`: number of times price crossed VWAP in 60 min

### Session Timing
- `is_open_session`, `is_close_session`: binary session markers
- `bars_since_open`: how many bars since market open

### SPY Context
- `spy_session_return`: SPY return from open to current bar
- `spy_rsi_14`: SPY RSI over 60-min window
- `spy_volume_surge`: SPY volume relative to its 20-day average

### Prior Day Context
- `prior_day_range_pct`: prior day high−low / prior close
- `prior_day_return`: prior day close-to-close return
- `prior_day_volume_ratio`: prior day volume / 20-day average

### Cross-Sectional (computed at scoring time)
- `cs_rank_momentum`: rank of `price_momentum` among all symbols scored today
- `cs_rank_volume`: rank of `volume_surge` among all symbols
- `cs_rank_atr`: rank of `atr_norm` among all symbols
- (Cross-sectional normalization applied to all features before model scoring)

---

## Full Experiment History

### v18 Baseline (before Phase 46)
- **Architecture**: XGBoost + LightGBM soft-vote ensemble (50/50)
- **Label**: Fixed binary labels — stop at -0.3%, target at +0.5%
- **Training label**: Fixed thresholds
- **Simulator exit**: ATR-adaptive (1.2× prior-day range target, 0.6× stop) ← **MISMATCH**
- **Walk-forward result**: Avg Sharpe **-1.16**, ~50 trades/fold, 58% stop exits
- **Root cause identified**: Training labels used fixed 0.3%/0.5% thresholds; simulator used ATR-adaptive exits 4-5× larger. Model learned to predict the wrong thing.

### v19 — Binary ATR Labels (Phase 46-A attempt 1)
- **Change**: Replace fixed stop/target with ATR-adaptive binary labels (1.2×/0.6× prior-day range)
- **Problem**: 1.2× prior-day range ≈ 2.4% in 2 hours for typical stock. Near-zero positive training examples.
- **Result**: 5 total trades across 3 folds. Model learned "never trade." Avg Sharpe -0.875.
- **Verdict**: Binary ATR labels not viable for 2-hour intraday window.

### v20 — path_quality Regression Label (Phase 46-A attempt 2)
- **Change**: Continuous path_quality score (same formula as swing model), top-20% by score = label 1
- **AUC improvement**: 0.5106 → 0.5427
- **Result**: Still only 5 trades across 3 folds. Root cause: hard ORB breakout gate in simulator.
- **Discovery**: `IntradayAgentSimulator._process_day()` required `orb_breakout > 0` as hard gate. In choppy/range-bound markets (Fold 2: Apr–Oct 2025), most stocks never broke their ORB → 0 trades.
- **Verdict**: Labels are correct. Gate is the bottleneck.

### v22 — Soft ORB Gate + Full Phase 46 Stack
**Changes from v20:**
1. Removed hard ORB breakout gate from simulator (features remain for model to use)
2. Removed hard volume/whale gate from simulator
3. Added `orb_direction_strength` feature (Phase 46-D)
4. MetaLabelModel v1 for intraday (Phase 46-B)
5. PM abstention gate: VIX ≥ 25 OR SPY < MA20 (Phase 46-C)

**Stage 1 — v22 no meta:**
| Fold | Trades | Win% | Sharpe |
|---|---|---|---|
| 1 (Oct14–Apr15) | 252 | 49.2% | -0.96 |
| 2 (Apr16–Oct15) | 252 | 49.2% | +0.57 |
| 3 (Oct16–Apr17) | 252 | 48.0% | -0.49 |
| **Avg** | **252** | **48.8%** | **-0.292** |

Trade count recovered from 5 → 252/fold. Fold 2 (sideways/choppy market) went from 0 trades to +0.57 Sharpe.

**Stage 2 — v22 + MetaLabelModel v1 + PM abstention:**
| Fold | Trades | Win% | Sharpe |
|---|---|---|---|
| 1 (Oct14–Apr15) | 152 | 51.3% | +0.24 |
| 2 (Apr16–Oct15) | 222 | 49.5% | +0.43 |
| 3 (Oct16–Apr17) | 152 | 52.6% | +0.23 |
| **Avg** | **175** | **51.1%** | **+0.301** |

**All folds positive for the first time.** Gate requires +0.80. We're at +0.301. Gap: +0.50 Sharpe.

---

## Remaining Gap Analysis

### Why are we at +0.301 instead of +0.80?

**1. Base model signal is weak (AUC 0.544)**
- Random = 0.50. We're at 0.544. The model can barely separate good trades from bad.
- Win rate: 51% at 2:1 R:R (1.2× target / 0.6× stop). At 51% win rate:
  - Expectancy = 0.51 × 1.2 − 0.49 × 0.6 = 0.612 − 0.294 = +0.318 per "unit"
  - This generates Sharpe ≈ 0.30 at 175 trades/fold — which matches exactly what we see.
- To get Sharpe +0.80 at 175 trades, we need either: ~56% win rate OR better R:R.

**2. Meta-model has near-zero predictive power (R2=0.001, corr=0.044)**
- The meta-model is trying to predict pnl_pct of individual trades
- With base AUC 0.544, there's almost nothing for it to learn from
- It's filtering trades mostly randomly — reduces count without meaningfully improving quality

**3. Choppy middle fold (Fold 2, Apr–Oct 2025)**
- This was the period when the hard ORB gate produced 0 trades (range-bound market)
- With soft gate: +0.57 Sharpe — the model actually does well in sideways markets
- But Folds 1 and 3 (trending markets) are +0.24/+0.23 — model struggles with directionality

**4. Feature set is OHLCV-only with no microstructure signal**
- No bid-ask spread proxies
- No order flow imbalance
- No relative strength vs sector at entry time
- No pre-market gap or overnight catalyst features

---

## Constraints and Practicalities

- **Data available**: Polygon 5-min OHLCV for Russell 1000, 2yr history. No tick data. No Level 2.
- **No fundamentals**: Not used (OOMs during training, ruled out)
- **Compute**: Single machine, training takes ~20 min for intraday model
- **Retraining cost**: Each retrain = 20 min training + 45 min walk-forward = ~65 min
- **Account**: $20k. Intraday positions = 3% of equity = $600 per position, max 5 = $3k deployed
- **No shorting**: Long-only system (Alpaca paper trading, pattern day trader rules apply)
- **Python stack**: XGBoost, LightGBM, scikit-learn, pandas, numpy. Optuna for HPO.
- **Feature store**: SQLite cache keyed by (symbol, date, schema_version). Adding features requires cache invalidation and full rebuild (~2hr for Russell 1000 at 5-min resolution).

---

## What the Swing Model Did Right (for reference)

The swing model (now gate-passed at +1.181 Sharpe) used these same techniques that might apply to intraday:

1. **path_quality regression label** — continuous score capturing upside/downside path quality, not just binary outcome. Avoids sparse positive problem. **Already adopted for intraday.**

2. **MetaLabelModel** — second-stage XGBRegressor filtering entries. **Already adopted, but weak due to base model.**

3. **PM abstention gate** — VIX/SPY regime filter. Adds +0.50 Sharpe. **Already adopted for intraday.**

4. **Tight universe** — SP-100 for swing (100 stocks) vs Russell 1000 for intraday (766 stocks). Smaller universe = higher quality data, less noise. **Not yet tried for intraday.**

5. **Stop 0.5× ATR, Target 1.0× ATR (2:1 R:R)** — swing. Intraday uses 0.6× stop, 1.2× target (also 2:1). Similar structure.

---

## Specific Questions for You

1. **Signal quality**: With OHLCV-only 5-min data, what features or transformations are most likely to push intraday AUC from 0.544 to 0.56+? We have 60 minutes of bars before entry. What signals matter most for 2-hour momentum trades?

2. **Label engineering**: Is path_quality the right label for a 2-hour hold? What alternative labeling schemes (e.g., maximum favorable excursion, risk-adjusted return, classification with wider bins) might produce a stronger training signal?

3. **Stop/target structure**: At 51% win rate and 2:1 R:R, we need 56%+ win rate OR 3:1 R:R to hit Sharpe 0.80. Which is more achievable — improving win rate or widening R:R? What ATR multipliers should we try?

4. **Universe**: Should we shrink from Russell 1000 to SP-500 or SP-100 for intraday? The hypothesis: larger, more liquid stocks have cleaner price action and more reliable ORB breakouts. Risk: fewer symbols = fewer trade opportunities per day.

5. **Entry timing**: Currently we enter after exactly 12 bars (60 minutes). Should we try different entry windows — e.g., 6 bars (30 min), 18 bars (90 min)? Or dynamic entry based on when a signal threshold is crossed?

6. **Meta-model approach**: Given that our meta-model R2 is near zero, is there a better way to build a second-stage filter? E.g., calibrated probability thresholds, regime-conditional models, or a completely different meta-learning approach?

7. **Architecture**: Should we try a different model architecture? E.g., gradient boosted trees with ranking objective (LambdaRank), LightGBM with dart boosting, or even a simple logistic regression on derived features?

8. **Multi-timeframe**: We have 5-min data. Should we compute features on multiple resolutions (e.g., 15-min, 30-min aggregated) and feed all to the model?

9. **Cross-sectional ranking**: We already rank momentum/volume/ATR cross-sectionally. Are there other cross-sectional features that capture relative opportunity at the stock level?

10. **Regime-conditional modeling**: Fold 2 (sideways market) is +0.57, Folds 1 and 3 (trending markets) are ~+0.24. Should we train separate models per regime? Or use regime as a feature?

---

## What to Deliver

Please suggest 3–5 concrete, prioritized experiments for Phase 47. For each:
- **Hypothesis**: what signal or structural change you expect to help and why
- **Implementation**: what specifically to change (features, label, model config, simulator)
- **Expected impact**: rough magnitude of Sharpe improvement if hypothesis holds
- **Risk**: what could go wrong / reasons it might not work
- **Priority**: order from highest-leverage to lowest

Keep in mind:
- Each experiment takes ~65 min compute time to validate
- We can run them sequentially (one at a time)
- Don't combine multiple changes in one experiment — we need to isolate causality
- The system is Python/XGBoost — stick to what's implementable in this stack

The goal is avg Sharpe ≥ +0.80, all folds ≥ -0.30, with the intraday model staying robust across different market regimes (trending and sideways).
