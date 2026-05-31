# MrTrader — Phases 18–23: Model Improvement Roadmap

Last updated: 2026-04-22

This document covers the model improvement and validation roadmap following the completion of
the three-tier backtesting system (Phases 15–17). The goal is to take both models from their
current negative Tier 3 performance to paper-trading-ready (Tier 3 Sharpe > 1.0 swing,
Sharpe > 1.5 intraday).

---

## Context: Why These Phases Exist

After building the three-tier backtest (Phase 17), running both models through honest Tier 3
agent-driven simulation revealed:

| Model | Tier 3 Sharpe | Win Rate | Root Problem |
|---|---|---|---|
| Swing v94 | -0.32 | 35.3% | Label-exit mismatch, 77% stop exits |
| Intraday v17 | -0.06 | 51.1% | 55-day data limit, label mismatch |

**Three root causes identified:**
1. **Label-exit mismatch**: models trained on endpoint returns, but Tier 3 exits on stop/target. A stock can rank top 20% at day 10 but stop out on day 2.
2. **Poor entry timing**: 77% stop exits on swing = entering extended/reverting stocks.
3. **Data starvation**: 55 days of intraday history = 47 trades = statistical noise, not signal.

---

## Phase 18 — Fix Label-Exit Mismatch

**Status:** Pending  
**Branch naming:** `feature/phase-18-label-exit-fix`  
**Gate:** Swing Tier 3 win rate > 42% after retrain

### Problem

Swing model trained to predict "top 20% 10-day forward return" (endpoint only). Tier 3 exits
at 2% stop / 6% target. A stock can be top 20% at day 10 but reverse through the 2% stop on
day 2 before recovering. The model has no concept of drawdown path.

Intraday model trained to predict "top 20% best 2h return (max HIGH)." Tier 3 exits at +0.5%
target / -0.3% stop. The best HIGH over 24 bars is not the same as "hit target before stop."

### Swing Label Fix

Replace cross-sectional top-20% return label with:
```
label = 1 if (max_high_10d / entry - 1 >= TARGET_PCT)
             AND (min_low_10d / entry - 1 > -STOP_PCT)
        else 0
```
Where `TARGET_PCT = 0.06`, `STOP_PCT = 0.02`. Binary classification: "did this stock reach
+6% before hitting -2% within 10 trading days?"

**Files to modify:**
- `app/ml/training.py` — `ModelTrainer._build_labels()`

### Intraday Label Fix

Replace cross-sectional top-20% max-HIGH label with:
```
label = 1 if (max_high_24bars / entry - 1 >= TARGET_PCT)
             AND (min_low_24bars / entry - 1 > -STOP_PCT)
        else 0
```
Where `TARGET_PCT = 0.005`, `STOP_PCT = 0.003`.

**Files to modify:**
- `app/ml/intraday_training.py` — `IntradayModelTrainer._build_labels()`

### Deliverables
- [ ] Update swing label construction in `ModelTrainer`
- [ ] Update intraday label construction in `IntradayModelTrainer`
- [ ] Update `scale_pos_weight` to reflect new label distribution (likely fewer positives)
- [ ] Retrain both models
- [ ] Run full Tier 3 backtest on both
- [ ] Update ML_EXPERIMENT_LOG.md with results
- [ ] Tests: confirm label shape and positive rate

---

## Phase 19 — New Features: Pressure Index + ChoCh (Swing)

**Status:** Pending  
**Depends on:** Phase 18 complete  
**Branch naming:** `feature/phase-19-pressure-choch-features`  
**Gate:** `choch_detected` or `pressure_index` appear in SHAP top 15

### Why These Features

Both features were derived from TradingView indicators reviewed on 2026-04-22:
- **Pressure Index** (from "Overbought & Oversold Zones MTF Signals"): measures HOW SUSTAINED
  an extreme is, not just how big. Catches stocks that have been extended for multiple bars.
- **ChoCh / Market Structure** (from "Market Structure Trend Matrix"): detects fresh structural
  breaks (trend starts). Entering at a fresh ChoCh vs mid-trend is the difference between a
  good swing entry and a reverting one.

### Pressure Index Features

**`pressure_persistence`**: number of bars price has stayed consistently above/below a
volatility-adjusted baseline. Higher = more extended.

```python
baseline = close.ewm(span=20).mean()
atr = compute_atr(bars, 14)
above = (close > baseline + 0.5 * atr).astype(float)
pressure_persistence = above.rolling(10).sum()  # bars above baseline in last 10
```

**`pressure_displacement`**: how far current price deviates from baseline in ATR units.
```python
pressure_displacement = (close - baseline) / atr
```

**`pressure_index`**: combined score (persistence × displacement), normalized.
```python
pressure_index = pressure_persistence * pressure_displacement.clip(-3, 3)
```

**Files to modify:** `app/ml/features.py` — `FeatureEngineer.engineer_features()`

### ChoCh Features

**`choch_detected`**: boolean (1/0) — did price in the last 5 bars break a recent swing
high (bullish ChoCh) or swing low (bearish — avoid)?
```python
# Bullish ChoCh: close crosses above highest high of prior 20 bars (new high)
recent_high = high.rolling(20).max().shift(1)
choch_detected = (close > recent_high).astype(float)
```

**`bars_since_choch`**: how many bars since the last structural break.
```python
# Days since last time close made a 20-bar high
last_choch = choch_detected.replace(0, np.nan).ffill()
bars_since_choch = (choch_detected.cumsum() != choch_detected.cumsum().shift(1)).cumsum()
# Simplified: rolling count since last 1
```

**`hh_hl_sequence`**: count of consecutive higher-highs in last 5 pivots (trend confirmation).

**Files to modify:** `app/ml/features.py`

### Whale Candle (Intraday only)

From "AlgoPath Pro+" indicator. Candle body > 2×ATR = unusual institutional activity.
```python
candle_body = abs(close - open_)
whale_candle = (candle_body > 2 * atr_norm * close).astype(float)
```
Add to `app/ml/intraday_features.py`.

### Deliverables
- [ ] Add `pressure_index`, `pressure_persistence`, `pressure_displacement` to `features.py`
- [ ] Add `choch_detected`, `bars_since_choch`, `hh_hl_sequence` to `features.py`
- [ ] Add `whale_candle` to `intraday_features.py`
- [ ] Swing feature count: 126 → ~131
- [ ] Intraday feature count: 40 → 41
- [ ] Retrain swing + intraday models
- [ ] Check SHAP — new features must rank in top 15 to keep
- [ ] Run Tier 3 backtest

---

## Phase 20 — Entry Quality Filter in Tier 3

**Status:** Pending  
**Depends on:** Phase 18 (need better baseline before tightening further)  
**Branch naming:** `feature/phase-20-entry-quality-filter`  
**Gate:** Tier 3 Sharpe crosses above 0.0 on swing

### Problem

Currently `AgentSimulator._trader_signal()` only requires `price > EMA-200`. Too permissive —
allows entries on stocks in short-term downtrends within a long-term uptrend, and overbought entries.

### Swing Entry Filter Additions

In `app/backtesting/agent_simulator.py` — `_trader_signal()`:

```python
# Near-term trend (not just long-term)
ema20 = close.ewm(span=20).mean().iloc[-1]
ema50 = close.ewm(span=50).mean().iloc[-1]
near_trend_ok = close_price > ema20 and close_price > ema50

# RSI zone: not overbought at entry, not in freefall
rsi = compute_rsi(bars, 14).iloc[-1]
rsi_ok = 40 <= rsi <= 65

# Volume confirmation
vol_ratio = bars["volume"].iloc[-1] / bars["volume"].rolling(20).mean().iloc[-1]
volume_ok = vol_ratio >= 1.0

# Regime gate
vix_ok = vix_level < 25  # block high-fear entries
spy_trend_ok = spy_close > spy_ema50  # block broad market downtrend
```

### Intraday Entry Filter Addition

In `app/backtesting/intraday_agent_simulator.py` — entry gate:
- Add `whale_candle` check: if feature_bars show institutional volume, allow entry
- Tighten ORB: require `orb_breakout > 0` AND `volume_surge > 1.2` (stronger confirmation)

### Deliverables
- [ ] Update `AgentSimulator._trader_signal()` with EMA-20/50, RSI zone, volume, regime gate
- [ ] Update `IntradayAgentSimulator` ORB + volume gate
- [ ] Tests for new gate conditions
- [ ] Run Tier 3 backtest — target stop-exit rate < 55%, Sharpe > 0.0

---

## Phase 21 — Intraday Data Infrastructure (Polygon)

**Status:** Pending  
**Depends on:** Nothing (can run in parallel with 19-20)  
**Branch naming:** `feature/phase-21-polygon-intraday-data`  
**Gate:** 500+ Tier 3 intraday trades achievable from stored data

### Problem

yfinance 5-min data is limited to the last 60 days. This gives 47 Tier 3 trades — not enough
for statistical significance. Need 2 years of 5-min data = ~500+ trades.

### Solution

Use Polygon.io REST API (API key already configured in `.env`) to pull historical 5-min bars.
Store as Parquet files per symbol. Update all intraday training/backtesting to use this cache.

### Deliverables
- [ ] `scripts/fetch_intraday_history.py` — pulls 2yr 5-min Polygon data per symbol
- [ ] Parquet cache: `data/intraday/{symbol}.parquet` with 24h TTL refresh
- [ ] Update `IntradayModelTrainer` to load from Parquet instead of yfinance
- [ ] Update `IntradayBacktester` to use same cache
- [ ] Update `IntradayAgentSimulator` to use same cache
- [ ] Retrain intraday model on 2yr data
- [ ] Run Tier 3 backtest — expect 500+ trades
- [ ] Update `scripts/backtest_ml_models.py` to use Polygon cache

---

## Phase 22 — Walk-Forward Validation on Tier 3

**Status:** Pending  
**Depends on:** Phases 20 + 21 both complete  
**Branch naming:** `feature/phase-22-walkforward-tier3`  
**Gate:** Avg OOS Tier 3 Sharpe > 0.8, no fold below -0.3

### Problem

Current walk-forward (5-fold) uses Tier 1 label-aligned replay, not Tier 3 agent logic.
Tier 1 is optimistic. Need to confirm results hold under actual agent constraints.

### Design

Rolling train/test with Tier 3 evaluation:
- Fold 1: train on Y1-Y2, test Tier 3 on Y3
- Fold 2: train on Y1-Y3, test Tier 3 on Y4
- Fold 3: train on Y1-Y4, test Tier 3 on Y5 (most recent)

Minimum 3 folds. Per-fold report: trades, win rate, Sharpe, max drawdown, stop-exit rate.

### Deliverables
- [ ] `scripts/walkforward_tier3.py` — rolling train/Tier 3 test runner
- [ ] Report: per-fold + aggregate stats
- [ ] Store results in `ML_EXPERIMENT_LOG.md`
- [ ] **Go/no-go decision**: avg Sharpe > 0.8 → proceed to Phase 23

---

## Phase 23 — Paper Trading Gate

**Status:** Pending  
**Depends on:** Phase 22 gate passed  
**Branch naming:** `feature/phase-23-paper-trading`  
**Gate:** 60 trading days + live Sharpe > 0.5 + max DD < 10%

### Design

Wire improved swing + intraday models into live paper trading loop (already built in Phase 8).
Monitor live results vs Tier 3 walk-forward predictions. If live/backtest gap > 30%, halt and
investigate before continuing.

### Deliverables
- [ ] Deploy swing + intraday models to paper trading
- [ ] Weekly automated performance report (vs Tier 3 walk-forward baseline)
- [ ] Kill-switch ready: if drawdown > 8%, halt new entries
- [ ] 60-day minimum before any live capital deployment

---

## Decision Gates Summary

| Phase | Gate Metric | Required Value |
|---|---|---|
| 18 | Swing Tier 3 win rate | > 42% |
| 19 | New features in SHAP | Both in top 15 |
| 20 | Swing Tier 3 Sharpe | > 0.0 |
| 21 | Intraday Tier 3 trades | > 500 |
| 22 | Avg OOS Tier 3 Sharpe | > 0.8, no fold < -0.3 |
| 23 | Live paper Sharpe (60d) | > 0.5, max DD < 10% |

---

## What's Been Tried and Ruled Out

| Approach | Outcome | Notes |
|---|---|---|
| Raw return labels (v92) | Sector_momentum dominated SHAP at 0.90 | Regime proxy, not alpha |
| CS normalization alone | SHAP rebalanced but AUC dropped slightly | Structural win despite AUC drop |
| Sharpe-adjusted labels | Combined with CS norm → revenue_growth now #1 SHAP | Keep |
| EMA crossover gate in Tier 3 | 0 trades — crossover too rare on daily bars | Replaced with EMA-200 uptrend filter |
| 2% stop / 6% target without label alignment | 77% stop exits, Sharpe -0.32 | Root cause identified: label mismatch |
| yfinance 5-min data for intraday backtest | 55 days max, 47 trades — insufficient | Polygon.io needed (Phase 21) |
