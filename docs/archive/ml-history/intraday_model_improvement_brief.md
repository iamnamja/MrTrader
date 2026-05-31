# MrTrader Intraday Model — Full Improvement Brief

**Purpose:** Comprehensive context for AI consultation on how to improve the intraday model.
Read this in full before suggesting next steps.

---

## 1. System Overview

MrTrader is an autonomous multi-agent trading system (PM → RM → Trader → Alpaca paper).
Capital: $20,000 paper account. Two strategies run in parallel:
- **Swing**: holds 2–5 days, daily bars, XGBoost classifier (v119, gate PASSED)
- **Intraday**: enters at bar 12 (10:30 AM ET), 2-hour hold window, 5-min bars, XGBoost classifier (v29, gate CONTESTED — see below)

This document is about the **intraday model only**.

---

## 2. Current Active Model: v29

| Attribute | Value |
|---|---|
| Architecture | XGBoost classifier |
| Features | 50 (see Section 4) |
| Training data | 2 years of Polygon 5-min bars, Russell 1000 universe (~720 symbols) |
| Training window | Apr 2024 – Apr 2026 |
| Entry timing | Bar 12 only (60 min post-open = 10:30 AM ET) |
| Hold window | Up to 2 hours (24 bars) |
| Exits | ATR-adaptive: stop = 0.4× prior-day ATR, target = 0.8× prior-day ATR, time exit at bar 24 |
| Labels | `path_quality` regression → cross-sectional top-20% = label 1 |
| Hyperparameters | n_estimators=577, max_depth=6, learning_rate=0.0176 (HPO-tuned) |
| AUC | 0.5970 (OOS) |
| Top features | prev_day_low_dist, prev_day_high_dist, atr_norm, range_compression, pullback_from_high |

### v29 Walk-Forward Gate History

**Validated Apr 2026 (Oct 2024 – Apr 2026 window, 18 months):**
| Fold | Test Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024 – Apr 2025 | 146 | 48.6% | +1.73 |
| 2 | Apr 2025 – Oct 2025 | 224 | 39.7% | +0.72 |
| 3 | Oct 2025 – Apr 2026 | 158 | 53.8% | +2.97 |
| **Avg** | | **528** | **47.4%** | **+1.807** ✅ |

**Re-tested May 2026 (Apr 2025 – Apr 2026 window, most recent 365 days):**
| Fold | Test Period | Trades | Win% | Sharpe | Market Context |
|---|---|---|---|---|---|
| 1 | Jul 2025 – Oct 2025 | 126 | 33.3% | **-0.68** ❌ | Low vol (0.65% daily), VIX 16.5, SPY +10.4% melt-up |
| 2 | Oct 2025 – Jan 2026 | 126 | 49.2% | +1.88 ✅ | Moderate vol (0.76%), VIX 17.3, SPY +3.8% |
| 3 | Jan 2026 – Apr 2026 | 126 | 44.4% | +0.63 | Higher vol (0.90%), VIX 20.4, SPY +4.4%, mostly below 20MA |
| **Avg** | | **378** | **42.3%** | **+0.611** ❌ |

**Gate requirement:** avg Sharpe > 1.50, no fold below -0.30.
**Status:** FAILS on most recent 365-day window. The edge exists but is not consistent across regimes.

### Key Diagnostic: Regime Sensitivity

The fold 1 failure (Jul–Oct 2025) is a **low-volatility melt-up regime**:
- SPY gained +10.4% in 3 months with only 0.65% daily realized vol — the lowest of any fold
- VIX averaged 16.5, ranging 14–25 — calm market
- Win rate collapsed to 33% — the bar-12 opening momentum signals become noise when markets grind up with no intraday range

Contrast with fold 2 (best): VIX 17.3, moderate vol 0.76%, SPY flatish. The model thrives when there is enough intraday volatility for the first-hour patterns to be meaningful.

**Hypothesis:** The bar-12 opening range breakout edge **requires a minimum level of intraday volatility** to distinguish real setups from noise. In low-vol melt-ups, every stock looks like a good setup (price near prior-day high) but none follow through because there's no intraday range to capture.

---

## 3. Label Engineering

### Current: `path_quality` (regression → binary)

```
path_quality = upside_capture - 0.50 × stop_pressure + 0.25 × close_strength
```

- `upside_capture`: how close to the ATR target the price got
- `stop_pressure`: fraction of time price was within the stop zone (penalizes risky winners)
- `close_strength`: where the bar-24 close was relative to session high/low

Cross-sectional top-20% per day → binary label 1.

### Label History

| Version | Label Scheme | Notes |
|---|---|---|
| v16–v18 | Fixed stop/target (0.5%/0.3%) | Label/exit mismatch — fixed exits don't match ATR simulator |
| v19 | Binary ATR (1.2×/0.6×) | Near-zero positives (2.4% target-hit rate in 2h) — too sparse |
| v20–v22 | path_quality (−1.25 × stop_pressure) | Reversion bias — penalizes trending winners that pull back |
| v23–v29 | path_quality (−0.50 × stop_pressure) | Current. 46–49% win rate, 2:1 R:R, ~10% target-hit rate |

**Known issue with current labels:** Only ~10% of trades hit the ATR target within 2 hours. The majority (~58%) are time exits at bar 24. The label tries to capture "path quality" rather than "did it hit target" — this is reasonable but means we're training on a noisy proxy for actual P&L.

---

## 4. Feature Set (50 Features)

### Price / Structure (10)
- `orb_position` — price position within opening 30-min range
- `orb_breakout` — +1 above ORB high, -1 below ORB low, 0 inside
- `orb_direction_strength` — signed distance from nearest ORB boundary (breakout velocity)
- `vwap_distance` — (close − VWAP) / VWAP
- `vwap_cross_count` — times price crossed VWAP today
- `gap_pct` — overnight gap vs prior close
- `gap_fill_pct` — fraction of opening gap already filled
- `session_hl_position` — close position within today's session H-L range
- `prev_day_high_dist` — (close − prior-day high) / close ← **#2 feature by importance**
- `prev_day_low_dist` — (close − prior-day low) / close ← **#1 feature by importance**

### Trend / Moving Averages (5)
- `ema9_dist`, `ema20_dist`, `ema_cross`, `macd_hist`, `bb_position`

### Momentum (6)
- `rsi_14`, `session_return`, `ret_15m`, `ret_30m`, `stoch_k`, `williams_r`

### Volume / Order Flow (4)
- `volume_surge`, `cum_delta`, `vol_trend`, `obv_slope`

### Candlestick / Structure (4)
- `upper_wick_ratio`, `lower_wick_ratio`, `body_ratio`, `consecutive_bars`

### Volatility (2)
- `atr_norm` ← **#3 feature** | `range_compression`

### Market Context (3)
- `spy_session_return`, `spy_rsi_14`, `rel_vol_spy`

### Session Timing (4)
- `time_of_day`, `minutes_since_open`, `is_open_session`, `is_close_session`

### Daily Vol Context (3)
- `daily_vol_percentile`, `daily_vol_regime`, `daily_parkinson_vol`

### Quality / Structure (8, added Phase 47-5)
- `trend_efficiency`, `green_bar_ratio`, `above_vwap_ratio`, `pullback_from_high` ← **#5 feature**
- `range_vs_20d_avg`, `rel_strength_vs_spy`, `vol_x_momentum`, `gap_followthrough`

**Notably missing:** VIX level at entry time, market-wide realized vol, sector ETF context, pre-market volume/gap, order book imbalance.

---

## 5. Entry Gate (PM Abstention Logic)

The PM currently abstains from intraday entries when:
- VIX ≥ 25 (extreme fear — avoids crashes/spikes)
- SPY below 20-day MA (bearish trend)

**These gates address the HIGH-vol extreme. They do NOT address the LOW-vol problem** (the new failure mode: VIX 14–18, low realized vol, melt-up regime).

---

## 6. Everything That Has Been Tried (and Why It Failed)

### Architecture Changes

| Experiment | Version | Result | Why It Failed |
|---|---|---|---|
| XGBRanker (pairwise) | v25 | Sharpe 0.184 | Replaced prev_day_high/low_dist with EMA features. Ranking objective may be correct but feature degradation dominated |
| LightGBM ensemble | v30 | Sharpe -0.068 | Distribution mismatch (trained on 3 session windows, deployed on 1) |
| MetaLabel model | v22 | +0.000 Sharpe contribution | Base model signal too weak for meta to amplify (R²=0.001) |
| Multi-scan (bars 12/18/24) | v29+v30 | Sharpe -0.186 to +0.322 | Opening-range features are only valid at bar 0. Mid-day re-use produces garbage predictions |

### Label Changes

| Experiment | Result | Why It Failed |
|---|---|---|
| Binary ATR (1.2×/0.6×) | 5 trades, Sharpe -0.875 | Target too far for 2h window — near-zero positives |
| path_quality with −1.25 × stop_pressure | v22 Sharpe 0.301 | Reversion bias — model learns to avoid trending winners |
| path_quality with −0.50 × stop_pressure | v23+ Sharpe 1.275–1.807 | **Current best. Gate passed.** |

### Universe / Filtering

| Experiment | Result | Why It Failed |
|---|---|---|
| Top-300 by dollar volume | Sharpe -1.414 | Large-caps have too little intraday range. Edge lives in mid-caps |
| Full Russell 1000 (720 symbols) | Sharpe 1.807 | **Current best** |

### Feature Engineering

| Experiment | Result | Notes |
|---|---|---|
| 8 quality features (Phase 47-5) | All contribute > 0 importance | Kept. pullback_from_high = #5 |
| Remove is_open_session (zero XGB importance in v26) | v28 Sharpe 0.634 — FAILED | Critical for full universe. Zero importance was artifact of top-300 universe |
| Session timing features (v17) | Modest improvement | Kept |

### Entry/Exit Mechanics

| Experiment | Result | Notes |
|---|---|---|
| Fixed stops (0.5%/0.3%) | Label/exit mismatch → gate fail | Retired |
| ATR stops 0.6×/1.2× | 10.6% target-hit rate, true R:R 1.2:1 | Baseline before v23 |
| ATR stops 0.4×/0.8× (v23) | Better target-hit rate, R:R maintained | **Current** |
| Bar-sweep 9–15 (Phase 80) | Only bar 12 generates trades with positive Sharpe | Bar 12 confirmed. Bars 9–11 generate 0 trades |

### Abstention Gates

| Gate | Effect | Status |
|---|---|---|
| VIX ≥ 25 abstain | Avoids crash days | Active |
| SPY < 20MA abstain | Avoids bear markets | Active |
| SPY 5d return ≤ 0 abstain | Applied to swing, NOT intraday | Swing only |
| VIX < 15 abstain (LOW vol) | **Not yet tried** | Hypothesis: low-vol melt-ups kill intraday edge |

---

## 7. Walk-Forward Gate Definition

- **Gate:** avg OOS Sharpe > 1.50, no single fold below -0.30
- **Folds:** 3 expanding-window folds on most recent 365 days of 5-min data
- **Simulator:** Full agent simulation with ATR stops, PM abstention gates, 5-symbol-per-day cap, commission model
- **Data:** Polygon 5-min bars, ~720 symbols cached

---

## 8. Current Hypothesis for Failure

The model's opening-range features (prev_day_high/low_dist, orb_position, orb_direction_strength, gap_pct) only generate alpha when there is **sufficient intraday volatility** for those patterns to resolve into clean moves.

In low-vol melt-up regimes (VIX 14–18, daily realized vol < 0.65%):
- Every stock is near its prior-day high → `prev_day_high_dist` signal becomes noisy (everyone looks like a breakout)
- ORB ranges are narrow → `orb_direction_strength` is near-zero for everything
- Stocks don't move enough within 2 hours to hit 0.8× ATR target
- Win rate collapses (33%) → deeply negative Sharpe

---

## 9. Ideas for Improvement (Please Evaluate and Prioritize)

The following are hypotheses. Some may conflict. Please analyze and suggest which are highest-priority and how to implement/test them.

### A. Volatility Regime Gate (Fast — No Retrain)
Add PM abstention when market realized vol is too low:
- Gate: skip intraday entries when VIX < 15 OR SPY 5-day realized vol < 0.60%
- Could also use the stock's own `atr_norm` at bar 12 as a per-trade filter
- **Risk:** reduces trade count significantly. Need to verify enough trades remain across folds.
- **Expected:** eliminates the low-vol regime losses without retraining

### B. Volatility-Conditioned Labels (Medium — Retrain Required)
The current label treats a 0.8× ATR target the same in high-vol and low-vol days. In low-vol days the ATR itself is small, so 0.8× ATR may only be a 0.2% move — barely covering commissions.
- Idea: use **vol-adjusted labels** — require a minimum absolute move (e.g. 0.5%) in addition to ATR-relative target
- Or: condition the cross-sectional ranking on realized-vol quintile so we only label stocks "good" if they actually moved meaningfully
- **Risk:** may reduce positive examples further in low-vol environments

### C. Regime-Conditioned Sample Weights (Medium — Retrain Required)
Weight training samples by the realized vol on that day:
- High vol days → higher weight → model learns from environments where the edge exists
- Low vol days → lower weight → model is not confused by false signals
- Implementation: add `daily_realized_vol_weight` to sample_weight array in XGBoost fit
- **Risk:** may overfit to high-vol patterns and fail in moderate-vol environments

### D. VIX / Realized Vol as Model Features (Medium — Retrain Required)
The model has no awareness of whether today is a high-vol or low-vol day at the market level:
- Add features: `vix_level` (normalized), `spy_5d_realized_vol`, `market_vol_regime` (expanding/contracting)
- If the model sees VIX=16 vs VIX=22, it could learn to be more selective in low-vol environments
- **Note:** `daily_vol_percentile`, `daily_vol_regime`, `daily_parkinson_vol` already exist but are **stock-level** vol features, not market-level (SPY/VIX)

### E. Adaptive ATR Multipliers by Vol Regime (Medium — No Retrain, Logic Change)
In low-vol regimes, 0.4× ATR is an even tighter stop. Instead of changing the model:
- High vol (VIX > 20): keep 0.4×/0.8× ATR
- Low vol (VIX < 18): widen to 0.6×/1.2× ATR (same as old regime) or skip entirely
- **Risk:** adds complexity, changes the exit distribution the model was trained on

### F. Confidence Threshold Tuning by Regime (Medium — No Retrain)
Currently all predictions above a fixed threshold (0.35) trigger entries:
- In low-vol regimes, raise the threshold to 0.50 or 0.60 — only take the highest-conviction signals
- This is a filter, not a gate — still enters but much more selectively
- Could be combined with a VIX floor gate

### G. Better Labels: Realized P&L Regression (Hard — Major Retrain)
Instead of `path_quality` (a proxy), train directly on realized P&L percentage (the ATR simulator exit):
- Label = actual_pnl_pct from the simulator using the exact same exit logic
- Cross-sectional top-20% within each vol-regime quintile
- **Risk:** requires running the full simulator to generate labels for every training sample — expensive
- **Benefit:** eliminates the proxy → true P&L gap that currently causes noisy training

### H. Separate Low-Vol and High-Vol Models (Hard — Architecture Change)
Train two separate models:
- Model A: trained only on high-vol days (VIX > 18 or realized vol > 0.70%) — optimized for momentum breakouts
- Model B: trained only on low-vol days (VIX < 18) — if any edge exists here (perhaps mean-reversion?)
- Deploy whichever matches today's regime
- **Risk:** smaller training sets per model, more complexity, harder to maintain

### I. Feature Engineering: Market-Level Vol Context
The model lacks market-level volatility awareness. Add:
- `vix_norm` — VIX normalized to 52-week range (already have `daily_vol_percentile` for the stock; need SPY equivalent)
- `spy_5d_realized_vol` — 5-day realized vol of SPY (objective vol measure, not fear measure)
- `vol_regime_expanding` — is SPY vol expanding or contracting vs 20-day avg?
- `sector_etf_return` — the stock's sector ETF session return (e.g. XLK for tech stocks)
- `spy_open_drive` — SPY's directional efficiency in the first 30 minutes (is the market trending or chopping?)

### J. Better Entry Timing Within Bar 12 Window
Bar 12 = the 12th 5-min bar = 10:30 AM. But "bar 12" means we're looking at the close of that bar.
- Could we add a sub-bar signal: only enter if bar 12 is a strong directional bar (e.g. body_ratio > 0.6)
- Or: enter only if volume_surge at bar 12 is above 1.5× (volume-confirmed signal)
- These would be model features or hard gates on top of the model score

---

## 10. What We'd Like From Your Review

1. **Which of the above ideas (A–J) are most likely to fix the low-vol regime problem specifically?**
2. **Are there approaches we haven't considered?** Especially around regime-conditional modeling.
3. **What is the right order to try these?** (fastest to implement + highest expected lift first)
4. **Are there any of the above that are clearly wrong or that would make things worse?**
5. **Should we consider a fundamentally different model architecture** (e.g. LSTM on the 5-min bar sequence, regime-switching HMM, or reinforcement learning for exit timing)?
6. **Label quality:** is `path_quality` a reasonable proxy for what we want, or is there a better approach to labeling intraday trades?
7. **Is 2 hours the right hold window?** Bar 24 = 12:30 PM. Does the morning momentum edge typically resolve in 1 hour or 2?

---

## 11. Constraints

- **Data:** Polygon 5-min bars, 2 years history, ~720 symbols cached locally. No tick data, no Level 2.
- **Compute:** Single Windows machine, 16 cores, 32GB RAM. Training takes ~20 min. Walk-forward takes ~8 min per fold (~25 min total).
- **Live system:** Must produce a walk-forward gate result (avg Sharpe > 1.50, no fold < -0.30) before deployment. Any proposed change must be testable against this gate.
- **Gate window:** 365 most recent days, 3 folds. The problematic fold is Jul–Oct 2025 (low-vol melt-up).
- **Trade count:** Each fold produces 126 trades at bar 12. Too few trades = noisy Sharpe estimates. Changes that reduce trade count further need justification.
- **No lookahead:** All features at bar 12 must only use data available at 10:30 AM ET (prior daily bars + today's bars 0-11).
