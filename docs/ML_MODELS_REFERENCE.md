# MrTrader — ML Models Reference

Last updated: 2026-04-26 (post Phase 46 — intraday v22, all folds positive)

---

## Overview

MrTrader runs two independent ML models:

| | Swing Model | Intraday Model |
|---|---|---|
| **Horizon** | 5 trading days | ~2 hours |
| **Bar resolution** | Daily OHLCV | 5-minute OHLCV |
| **Universe** | SP-100 (~81 symbols with >= 210 bars) | Russell 1000 (~766 symbols) |
| **Training history** | 5 years | 2 years |
| **Features** | 84 (OHLCV only, no fundamentals) | 42 |
| **Architecture** | XGBRegressor (path_quality label) + MetaLabelModel filter + PM abstention gate | XGBClassifier + MetaLabelModel filter + PM abstention gate |
| **Label scheme** | path_quality regression (continuous score) | path_quality regression → cross-sectional top-20% = label 1 |
| **Retrain schedule** | 17:00 ET daily | 17:00 ET daily |
| **Walk-forward gate** | PASSED (avg Sharpe +1.181, min fold -0.031) | NOT MET (avg Sharpe +0.301, gate > 0.80) |

---

## Swing Model (v119 — ACTIVE, gate passed 2026-04-25)

The swing model is a three-layer stack: a primary signal model, a meta-filter that screens individual entries, and a PM-level gate that suppresses all entries on bad macro days.

### Layer 1 — Primary Signal Model (v119)

- **Model file:** `app/ml/model.py` — `PortfolioSelectorModel`
- **Trainer:** `app/ml/training.py` — `ModelTrainer`
- **Underlying estimator:** `XGBRegressor` (400 trees, depth 4, lr=0.03)
- **Features:** 84 OHLCV-derived features (no fundamentals, no external APIs at inference time)
- **Preprocessing:** `StandardScaler` (fit on training data, saved separately)
- **Output:** Continuous path_quality score; higher = better expected trade quality

**Train command:**
```bash
python scripts/train_model.py --model-type xgboost --label-scheme path_quality \
  --years 5 --no-fundamentals --workers 8
```

#### Label Construction (path_quality)

For each (symbol, date) in the training set:
1. Compute ATR-14. Set stop = 0.5 × ATR, target = 1.0 × ATR
2. Simulate the next 5 bars bar-by-bar, tracking high/low/close
3. Compute three components:
   - `upside_capture = min(max_high_reached / target_price, 1.0)` — did price reach the target?
   - `stop_pressure = min(min_low_reached / stop_price, 1.0)` — how close did price come to the stop?
   - `close_strength = clip((final_close - entry) / target_distance, -1.0, 1.0)` — did price close strong?
4. `label = 1.0 × upside_capture − 1.25 × stop_pressure + 0.25 × close_strength`
   - Range roughly [-2.5, 1.5]; positive = good trade quality, negative = poor
   - The 1.25× stop_pressure weight penalizes stop hits more than it rewards target hits

#### Training Pipeline

```
Fetch daily OHLCV (Alpaca/Polygon, 5 years)
    ↓
Rolling windows (63-day feature window, 5-day forward)
    ↓
Feature engineering per window (point-in-time, as_of_date passed)
    ↓
Feature store cache check (SQLite SCHEMA_VERSION=v4, avoids recomputing)
    ↓
path_quality label computation (bar-by-bar, 0.5x/1.0x ATR, 5-day horizon)
    ↓
StandardScaler fit on X_train
    ↓
XGBRegressor.fit() with early stopping on X_val
    ↓
Save: swing_v{N}.pkl + swing_scaler_v{N}.pkl + swing_meta_v{N}.pkl
    ↓
Register in ModelVersion DB table
```

#### Model Persistence

| File | Contents |
|---|---|
| `app/ml/models/swing_v119.pkl` | Raw XGBRegressor object |
| `app/ml/models/swing_scaler_v119.pkl` | StandardScaler (84 features) |
| `app/ml/models/swing_meta_v119.pkl` | feature_names list, model_type=xgboost |

---

### Layer 2 — MetaLabelModel v1 (entry filter)

- **File:** `app/ml/models/swing_meta_label_v1.pkl`
- **Class:** `app/ml/meta_model.py` — `MetaLabelModel`
- **Underlying estimator:** `XGBRegressor`
- **Trained on:** 515 in-sample v119 trades (first 80% of 5yr data), features at entry date → pnl_pct outcome
- **Training metrics:** R2=0.059, MAE=0.0306, corr=0.286 (weak absolute accuracy but useful as filter)
- **Decision rule:** `should_enter(features) → True if predicted pnl > 0.0`
- **Effect:** Filters ~26% of entries; removes the worst expected-loss setups without proportionally cutting winners

**How it integrates:**
- Backtest: `AgentSimulator(meta_model=MetaLabelModel.load(...))`
- Live: PM calls `meta_model.should_enter(feature_vector)` before submitting each proposal
- Walk-forward CLI: `--meta-model-version 1`

---

### Layer 3 — PM Abstention Gate (macro regime filter)

Suppresses **all** new swing entries on days with unfavorable macro conditions. Independent of per-stock signals.

**Trigger conditions (either → no entries today):**
- VIX >= 25 (elevated fear)
- SPY close < 20-day simple moving average (short-term downtrend)

**How it integrates:**
- Live: `portfolio_manager._market_regime_allows_entries()` called at start of `_send_swing_proposals()`
- Backtest: `AgentSimulator(pm_abstention_vix=25, pm_abstention_spy_ma_days=20)`
- Walk-forward CLI: `--pm-abstention-vix 25 --pm-abstention-spy-ma-days 20`

**Effect on fold 3 (hard 2025-2026 regime):** Sharpe -0.81 (no gates) → -0.14 (meta only) → -0.03 (meta + abstention)

---

### Full Walk-Forward Results (v119 stack, 3-fold OOS, 2026-04-25)

Run command:
```bash
python scripts/walkforward_tier3.py --model swing --stop-mult 0.5 --target-mult 1.0 \
  --meta-model-version 1 --pm-abstention-vix 25 --pm-abstention-spy-ma-days 20
```

| Fold | Period | Trades | Win% | Sharpe | DD% |
|---|---|---|---|---|---|
| 1 | 2022-07-28 → 2023-10-26 | 94 | 53.2% | +0.880 | 1.2% |
| 2 | 2023-10-27 → 2025-01-24 | 113 | 65.5% | +2.690 | 0.6% |
| 3 | 2025-01-25 → 2026-04-25 | 134 | 55.2% | -0.030 | 2.7% |
| **Avg** | | **341** | **58.0%** | **+1.181** | |

Gate: avg Sharpe > 0.80 **PASS** | min fold > -0.30 **PASS**

---

### Stop/Target Configuration

- **Stop:** 0.5× ATR-14 (tight, ~5% below entry for typical stock)
- **Target:** 1.0× ATR-14 (2:1 reward-to-risk)
- Chosen via Phase 45 Phase 1 grid search over 3 configs; Config B (0.5/1.0) won on avg Sharpe

---

### Entry Signal Logic (AgentSimulator / live PM)

Each trading day, for each symbol in the SP-100 universe:
1. Compute 84 features via `FeatureEngineer.engineer_features()`
2. Score with v119 `model.predict(features)` → path_quality score
3. Filter: `MetaLabelModel.should_enter(features)` → skip if E[pnl] <= 0
4. Filter: PM abstention gate → skip entire day if VIX >= 25 or SPY < MA20
5. Signal types: `RSI_DIP` (RSI < 50 + price near 20-day low) or `EMA_CROSSOVER` (EMA9 > EMA20 + momentum)
6. Size position to 5% of portfolio, stop at 0.5× ATR, target at 1.0× ATR

---

## Intraday Model (v22 — ACTIVE, Phase 46, gate not yet met)

The intraday model is a three-layer stack mirroring the swing model structure:
a primary signal model, a meta-filter, and a PM-level macro gate.

**Current best result:** avg Sharpe +0.301, all 3 folds positive (first time). Gate requires +0.80.
**Active improvement phase:** Phase 47 (diagnostic dive in progress, XGBRanker next).

### Layer 1 — Primary Signal Model (v22)

- **Model file:** `app/ml/model.py` — `PortfolioSelectorModel` (xgboost)
- **Trainer:** `app/ml/intraday_training.py` — `IntradayModelTrainer`
- **Underlying estimator:** XGBClassifier (binary classification)
- **OOS AUC:** 0.5438
- **Retrain command:** `python scripts/retrain_intraday.py --days 730`

### Layer 2 — MetaLabelModel v1 (intraday)

- **Model file:** `app/ml/models/intraday_meta_label_v1.pkl`
- **Architecture:** XGBRegressor trained on 810 in-sample trade outcomes (pnl_pct target)
- **R2:** 0.001, corr: 0.044 — **very weak signal** (base model AUC too low to train meta effectively)
- **Gate:** Enter only if predicted E[pnl] > 0
- **Status:** Likely contributing near-zero Sharpe. Phase 47 Phase 1 will confirm and likely drop it.

### Layer 3 — PM Abstention Gate

Same as swing: skip all new intraday entries when VIX ≥ 25 OR SPY < 20-day SMA.

### Label Construction (path_quality — Phase 46)

For each (symbol, date) in the training set:
1. Take first 12 bars (60 min) as feature window; entry = close of bar 12
2. Compute prior-day range → stop_dist = 0.6× range, target_dist = 1.2× range
3. Simulate next 24 bars (2 hours) bar-by-bar:
   - `upside_capture = min((max_high - entry) / target_dist, 1.0)`
   - `stop_pressure = min((entry - min_low) / stop_dist, 1.0)`
   - `close_strength = clip((final_close - entry) / target_dist, -1, 1)`
4. `path_quality = 1.0 × upside_capture − 1.25 × stop_pressure + 0.25 × close_strength`
5. Within each day, rank all symbols by path_quality; top 20% → label = 1

**Note:** The `-1.25 × stop_pressure` coefficient is under investigation in Phase 47 —
it may create a reversion bias by penalizing trades that touched the stop zone even if they
recovered and hit target (common in trending markets).

### Training Pipeline

```
Fetch 5-min OHLCV from Polygon cache (Russell 1000, 2 years)
    ↓ (cached as Parquet per symbol)
Fetch SPY 5-min + prior-day OHLC for ATR proxy
    ↓
Per symbol × per day: compute_intraday_features() → 42 features
    ↓ (parallel workers)
path_quality label computation (bar-by-bar, 0.6x/1.2x prior-day range)
    ↓
Cross-sectional daily labeling (top-20% path_quality per day = label 1)
    ↓
Recency sample weights (exp decay, 180-day half-life)
    ↓
Optuna HPO (20 trials, 3-fold StratifiedKFold)
    ↓
XGBClassifier.fit() with best params
    ↓
Save as intraday_v{N}.pkl + register in ModelVersion DB
```

### Walk-Forward Results (v22 full stack, 2026-04-26)

| Fold | Period | Trades | Win% | Sharpe | Max DD |
|---|---|---|---|---|---|
| 1 | 2024-10-14 → 2025-04-15 | 152 | 51.3% | +0.24 | 0.5% |
| 2 | 2025-04-16 → 2025-10-15 | 222 | 49.5% | +0.43 | 0.6% |
| 3 | 2025-10-16 → 2026-04-17 | 152 | 52.6% | +0.23 | 0.5% |
| **Avg** | | **175** | **51.1%** | **+0.301** | **0.5%** |

Gate: **FAIL** (needs +0.80). Min fold +0.227 (passes -0.30 floor). All folds positive for first time.

### Intraday Version History

| Version | Phase | Change | Avg Sharpe | Trades/fold | Status |
|---|---|---|---|---|---|
| v18 | Baseline | XGB+LGBM ensemble, fixed binary labels | -1.16 | ~50 | Superseded |
| v19 | 46-A | Binary ATR labels (1.2x/0.6x range) | -0.875 | 2 | Superseded — too sparse |
| v20 | 46-A | path_quality regression label | -0.138 | 2 | Superseded — ORB gate blocked |
| v22 | 46 full | Soft ORB gate + path_quality + meta + abstention | +0.301 | 175 | **Active** |

### What Was Ruled Out (Intraday)

- **Binary ATR labels** — 1.2× prior-day range target (~2.4% in 2h) produces near-zero positive training examples
- **Hard ORB breakout gate** — starves trades in range-bound markets (5 total trades across 3 folds in v19/v20)
- **XGBoost+LightGBM ensemble** — no improvement over single XGBClassifier, added complexity

---

## Features: Swing Model

### Category Summary

| Category | Count | Data Source | Requires Key? |
|---|---|---|---|
| Technical indicators (RSI, MACD, EMA, BB, etc.) | 50 | Daily OHLCV (Alpaca/Polygon) | No |
| Volume & price dynamics | 10 | Daily OHLCV | No |
| Relative strength vs SPY / sector | 6 | Daily OHLCV + SPY | No |
| WorldQuant 101 alphas | 14 | Daily OHLCV | No |
| Short-term reversal | 3 | Daily OHLCV | No |
| Volatility / options proxy | 11 | Daily OHLCV (computed) | No |
| VIX regime | 5 | yfinance `^VIX` | **No** (⚠️ 401 issue) |
| Regime score | 1 | RegimeDetector (internal) | No |
| Fundamentals (P/E, P/B, margins, D/E) | 5 | SEC EDGAR XBRL API | No ✓ |
| Earnings history & surprise | 5 | SEC EDGAR XBRL (quarterly 10-Q EPS) | No ✓ |
| Short interest | 1 | FINRA Reg SHO list | No ✓ |
| Sector ETF momentum | 1 | Alpaca `get_bars()` (XLK/XLF/etc.) | No ✓ |
| Insider activity (Form 4) | 1 | SEC EDGAR free API | No ✓ |
| Earnings surprise | 1 | Alpha Vantage | Optional |
| News sentiment | 4 | Polygon.io news API | Yes |
| FMP fundamentals (point-in-time) | 8 | FMP API | Yes |
| Polygon financials | 3 | Polygon.io financials API | Yes |
| **Total** | **126** | | |

### Yahoo Finance — Removed (2026-04-21)

Yahoo Finance was returning HTTP 401 "Invalid Crumb" for all `Ticker.info`, `earnings_history`, and `calendar` calls. It has been **fully replaced**:

| Was (yfinance) | Replaced with |
|---|---|
| `Ticker.info` → P/E, P/B, margins, D/E | SEC EDGAR XBRL API (`companyfacts` endpoint) |
| `earnings_history` → EPS surprise | SEC EDGAR XBRL quarterly EPS (10-Q filings) |
| `ticker.calendar` → next earnings date | Estimated from last 10-Q filing date + 90-day cadence |
| `yf.download(ETF)` → sector momentum | Alpaca `get_bars()` (same API used everywhere else) |
| `Ticker.info` → short interest | FINRA Reg SHO threshold list (free, bi-monthly) |

yfinance is no longer imported anywhere in the feature pipeline.

### Full Feature List (Swing)

```
MOMENTUM
  rsi_14, rsi_7
  momentum_5d, momentum_20d, momentum_60d, momentum_252d_ex1m

MACD
  macd, macd_signal, macd_histogram

MOVING AVERAGES
  ema_20, ema_50
  price_above_ema20, price_above_ema50, price_above_ema200
  dist_from_ema200

MEAN REVERSION & PRICE POSITION
  price_to_52w_high, price_to_52w_low, near_52w_high
  mean_reversion_zscore, up_day_ratio_20d, consolidation_position
  price_efficiency_20d, price_acceleration

VOLATILITY & ATR
  volatility, atr_norm, bb_position, stoch_k, adx_14
  vol_expansion, adx_slope, atr_trend

OSCILLATORS
  williams_r_14, cci_20, stochrsi_k, stochrsi_d, stochrsi_signal

KELTNER / CHAIKIN
  keltner_position, cmf_20, dema_20_dist

VOLUME & PRICE DYNAMICS
  volume_ratio, volume_trend, vpt_momentum, range_expansion
  vwap_distance_20d, vol_price_confirmation, volume_surge_3d
  uptrend, downtrend, consecutive_days

RELATIVE STRENGTH
  rs_vs_spy, rs_vs_spy_5d, rs_vs_spy_10d, rs_vs_spy_60d
  momentum_20d_sector_neutral, momentum_60d_sector_neutral

VOLATILITY REGIME
  vol_percentile_52w, vol_regime, vol_of_vol, atr_trend, parkinson_vol
  beta_252d, beta_deviation
  vrp, realized_vol_20d, days_to_opex, near_opex
  [live only: options_put_call_ratio, options_iv_atm, options_iv_premium]

VIX REGIME
  vix_level, vix_regime_bucket, vix_fear_spike
  vix_percentile_1y, spy_trend_63d

REGIME SCORE
  regime_score

WORLDQUANT 101 ALPHAS (computed from OHLCV)
  wq_alpha3, wq_alpha4, wq_alpha6, wq_alpha12
  wq_alpha33, wq_alpha34, wq_alpha35, wq_alpha40
  wq_alpha43, wq_alpha44, wq_alpha46, wq_alpha53, wq_alpha54

SHORT-TERM REVERSAL
  reversal_5d, reversal_5d_vol_weighted, reversal_3d

ENTRY TIMING
  trend_consistency_63d, price_change_pct

--- FUNDAMENTALS (disabled when yfinance returns 401) ---
  pe_ratio, pb_ratio, profit_margin, revenue_growth, debt_to_equity
  earnings_proximity_days, sector_momentum
  earnings_surprise_1q, earnings_surprise_2q_avg, days_since_earnings
  earnings_drift_signal, earnings_pead_strength
  short_interest_pct

SEC EDGAR (free, always available)
  insider_score

POLYGON NEWS (requires key)
  news_sentiment_3d, news_sentiment_7d
  news_article_count_7d, news_sentiment_momentum

FMP (requires key, point-in-time safe)
  fmp_surprise_1q, fmp_surprise_2q_avg, fmp_days_since_earnings
  fmp_analyst_upgrades_30d, fmp_analyst_downgrades_30d, fmp_analyst_momentum_30d
  fmp_inst_ownership_pct, fmp_inst_change_pct
  fmp_consecutive_beats, fmp_revenue_surprise_1q

POLYGON FINANCIALS (requires key)
  fcf_margin, operating_leverage, rd_intensity
```

---

## Features: Intraday Model

### Full Feature List (42 features — as of v22 / Phase 46)

All computed purely from 5-min OHLCV + SPY 5-min bars. **No external API calls at inference time.**
Feature window: first 12 bars (60 minutes) of the trading day.

```
OPENING RANGE BREAKOUT (ORB)
  orb_high, orb_low          — high/low of first bar (first 5 min)
  orb_range                  — (orb_high - orb_low) / orb_low, normalized width
  orb_position               — position in ORB range [0=at low, 1=at high]
  orb_breakout               — +1 above ORB high / -1 below ORB low / 0 inside
  orb_direction_strength     — (close - orb_mid) / orb_range  [Phase 46-D]

PRICE ACTION (60-min window)
  price_momentum             — 60-min return from open to current
  high_low_range             — normalized 60-min high-low range
  close_position             — current close relative to 60-min range
  price_velocity             — last 12-bar price change rate
  gap_pct                    — overnight gap vs prior close

VOLUME
  volume_surge               — 60-min vol vs 20-day avg same window
  whale_candle               — any 5-min bar with vol > 3× rolling 20-bar avg
  volume_trend               — slope of volume over 60 min
  volume_acceleration        — second derivative of volume

TECHNICAL INDICATORS
  rsi_6, rsi_14              — RSI at 6 and 14 periods
  macd_signal                — MACD signal line
  bb_position                — Bollinger Band %B
  atr_norm                   — ATR normalized by price
  vwap_distance              — (price - VWAP) / price
  vwap_cross_count           — times price crossed VWAP in 60 min

SESSION TIMING
  is_open_session            — binary session marker
  is_close_session           — binary session marker
  bars_since_open            — bars elapsed since market open

SPY CONTEXT
  spy_session_return         — SPY return from open to current bar
  spy_rsi_14                 — SPY RSI over 60-min window
  spy_volume_surge           — SPY volume vs 20-day avg

PRIOR DAY CONTEXT
  prior_day_range_pct        — prior day (high - low) / prior close
  prior_day_return           — prior day close-to-close return
  prior_day_volume_ratio     — prior day volume / 20-day avg

CROSS-SECTIONAL (computed at scoring time, over daily candidate universe)
  cs_rank_momentum           — rank of price_momentum among all symbols today
  cs_rank_volume             — rank of volume_surge
  cs_rank_atr                — rank of atr_norm
```

### Feature Engineering Code

`app/ml/intraday_features.py` — `compute_intraday_features(feat_bars, spy_day, prior_close, ...)`

Returns a dict of 42 features. Called per (symbol, day) at both training time and inference time.
Cross-sectional ranks are applied in `app/ml/cs_normalize.py` at scoring time, not in the feature function.

### Removed Features (v22 vs v18)

The v18 feature set included several features that were restructured in Phase 46:
- `gap_fill_pct`, `session_hl_position` — replaced by `orb_position` and `close_position`
- `prev_day_high_dist`, `prev_day_low_dist` — replaced by `prior_day_range_pct`
- `ema9_dist`, `ema20_dist`, `ema_cross` — dropped (low importance in intraday context)
- `stoch_k`, `williams_r`, `cum_delta`, `obv_slope` — dropped
- Added: `orb_high`, `orb_low`, `orb_range`, `orb_direction_strength`, `whale_candle`,
  `volume_acceleration`, `spy_volume_surge`, `prior_day_return`, `prior_day_volume_ratio`

---

## Features: Intraday Model (legacy — v18, kept for reference)

### v18 Feature List (37 features)

```
PRICE STRUCTURE
  orb_position       — position in opening 30-min range [0,1]
  orb_breakout       — +1 above ORB / -1 below / 0 inside
  vwap_distance      — (close - VWAP) / VWAP
  vwap_cross_count   — times price crossed VWAP today
  gap_pct            — overnight gap / prior close
  gap_fill_pct       — how much of gap was filled [0,1]
  session_hl_position — (close - low) / (high - low)
  prev_day_high_dist, prev_day_low_dist — distance from S/R

TREND
  ema9_dist, ema20_dist  — (close - EMA) / close
  ema_cross              — (EMA9 - EMA20) / close
  macd_hist              — MACD histogram / close
  bb_position            — Bollinger Band %B

MOMENTUM
  rsi_14                 — RSI(14) / 100
  session_return         — (close - open) / open
  ret_15m, ret_30m       — 3-bar and 6-bar returns
  stoch_k                — Stochastic %K / 100
  williams_r             — Williams %R normalized [0,1]

VOLUME / ORDER FLOW
  volume_surge           — last bar vol / 20-bar avg
  cum_delta              — % bars with close > open
  vol_trend              — volume EMA(10) slope
  obv_slope              — OBV EMA(10) slope

CANDLESTICK
  upper_wick_ratio, lower_wick_ratio
  body_ratio, consecutive_bars

VOLATILITY
  atr_norm               — ATR(14) / close
  range_compression      — 5-bar H-L range / close

MARKET CONTEXT (SPY)
  spy_session_return     — SPY return from open
  spy_rsi_14             — SPY RSI(14) / 100
  rel_vol_spy            — stock vol / SPY vol

SESSION TIMING
  time_of_day            — fraction of 6.5h session elapsed

DAILY VOL CONTEXT
  daily_vol_percentile   — realized vol percentile vs 52w
  daily_vol_regime       — short/long vol ratio
  daily_parkinson_vol    — Parkinson volatility estimator
```

---

## Data Sources Reference

| Source | What We Fetch | How | Key Required? |
|---|---|---|---|
| **Alpaca** | Daily OHLCV (swing), 5-min OHLCV (intraday), SPY bars, sector ETF bars | `StockHistoricalDataClient` | Yes (in `.env`) |
| **Polygon.io** | Daily OHLCV (swing training), news sentiment, financials | REST API | Yes (in `.env`) |
| **SEC EDGAR XBRL** | P/E proxy, P/B, margins, revenue growth, D/E, EPS history | Free REST API (`companyfacts`) | No |
| **SEC EDGAR Form 4** | Insider buy/sell score | Free REST API (EFTS search) | No |
| **FINRA** | Short interest flag (Reg SHO threshold list) | Free REST API | No |
| **Alpha Vantage** | EPS surprise (if key configured) | REST API | Optional |
| **FMP** | Point-in-time earnings, analyst ratings, institutional ownership | REST API | Yes (in `.env`) |
| **RegimeDetector** | VIX level, SPY trend, composite regime score | Computed from Alpaca bars | No |


---

## Feature Store (Cache)

**File:** `app/ml/models/feature_store.db` (SQLite)

**Purpose:** Cache computed feature vectors by (symbol, as_of_date) so retraining doesn't re-call APIs for historical windows already computed.

**Schema:**
```sql
features(symbol TEXT, as_of_date TEXT, features_json TEXT, created_at TEXT)
  PRIMARY KEY (symbol, as_of_date)
meta(key TEXT PRIMARY KEY, value TEXT)  -- stores schema_version
```

**Cache invalidation:** Auto-clears when SCHEMA_VERSION changes (i.e., when features are added/removed).

Feature store was cleared on 2026-04-23 (Phase 43 pruning, 140→84 features). All entries now use 84 features (SCHEMA_VERSION=v4). Fundamentals columns are excluded from active training.

---

## Model Versions in DB

The `model_versions` table tracks all trained models:

```sql
model_name    -- "swing" or "intraday"
version       -- auto-incrementing integer
status        -- "ACTIVE" or "ARCHIVED"
model_path    -- path to .pkl file
training_date -- when trained
performance   -- JSON: {auc, accuracy, precision, recall, threshold, n_test}
```

**Loading logic in PM:** `_try_load_model()` queries the highest ACTIVE version and calls `PortfolioSelectorModel.load(dir, version, model_name)` which reads all three pickle files.

**Current latest:** swing v66 (126 features, trained 2026-04-21) — not usable until Yahoo is fixed or we retrain with `fetch_fundamentals=False`.

---

## Backtesting Methodology

Three-tier architecture — each tier adds more realism:

| Tier | Class | Purpose |
|---|---|---|
| 1 | `SwingBacktester` / `IntradayBacktester` | Signal quality; trade every model signal |
| 2 | `StrategySimulator` | Portfolio replay; position sizing, Sharpe, drawdown |
| 3 | `AgentSimulator` / `IntradayAgentSimulator` | Full agent simulation; real PM/RM/Trader logic on historical bars |

Tier 3 is the benchmark for go/no-go decisions. Tier 2 is optimistic (replays winners). Tier 3 is what you'd actually get.

**Warm-up (Tier 3 swing):** 420 calendar days before trading start (EMA-200 needs ~300 business days of history).

---

## Current Model Versions

| Model | Version | Features | Trained | Notes |
|---|---|---|---|---|
| Swing | **v119** | 84 | 2026-04-25 | **Active — gate passed.** XGBRegressor, path_quality regression label (0.5x/1.0x ATR, 5-day horizon). + MetaLabelModel v1 (E[R]>0 filter) + PM abstention gate (VIX>=25 or SPY<MA20). Avg Sharpe +1.181, min fold -0.031. |
| Swing meta | **v1** | 84 (entry features) | 2026-04-25 | MetaLabelModel. XGBRegressor trained on 515 in-sample trade outcomes. R2=0.059, corr=0.286. Threshold E[R]>0. |
| Intraday | **v22** | 42 | 2026-04-25 | **Active — gate not yet met.** XGBClassifier, path_quality labels (0.6x/1.2x prior-day range). + MetaLabelModel v1 (weak, R2=0.001) + PM abstention gate. Avg Sharpe +0.301, all folds positive. Phase 47 in progress. |
| Intraday meta | **v1** | 42 (entry features) | 2026-04-25 | MetaLabelModel. XGBRegressor on 810 in-sample trades. R2=0.001 (very weak). Likely to be dropped in Phase 47. |

### Swing Model Version History

| Version | Phase | Change | Tier 3 Sharpe | Status |
|---|---|---|---|---|
| v108 | 24a | Label bug fix + no fundamentals | -1.26 | Superseded |
| v109 | 24b | 6 regime interaction features | -0.52 | Superseded |
| v110 | 25 | 5-day forward window, cross_sectional | +0.34 | Superseded |
| v111 | 26a | VIX regime sample weights | -0.43 | Superseded |
| v113 | 33 | Triple-barrier + LambdaRank | -0.76 | Superseded |
| v114 | 33 | LambdaRank triple-barrier (OMP fixed) | -0.47 | Superseded |
| v115 | 33 | XGBoost triple-barrier asymmetric | ~0.00 | Superseded |
| v116 | 33 | XGBoost triple-barrier symmetric | ~0.00 | Superseded |
| v117 | 43 | Feature pruning 140→84 | -0.15 | Superseded |
| v118 | 44 | Ensemble XGBoost+LR blend | -0.08 | Superseded |
| **v119** | **45** | **path_quality + meta + PM abstention** | **+1.181** | **ACTIVE** |
