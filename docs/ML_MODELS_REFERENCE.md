# MrTrader — ML Models Reference

Last updated: 2026-04-21

---

## Overview

MrTrader runs two independent ML models:

| | Swing Model | Intraday Model |
|---|---|---|
| **Horizon** | 10 trading days | ~2 hours |
| **Bar resolution** | Daily OHLCV | 5-minute OHLCV |
| **Universe** | S&P 500 (~500 symbols) — planned; currently SP-100 (~81) | Russell 1000 (~1000 symbols) |
| **Training history** | 3–5 years | 2 years |
| **Features** | ~83 (no fundamentals) → ~126 (with fundamentals) | 37 |
| **Architecture** | XGBoost (XGBClassifier) | XGBoost + LightGBM soft-vote ensemble |
| **Label scheme** | Cross-sectional: top-20% 10-day return = 1 | Cross-sectional: top-20% intraday return = 1 |
| **Prediction threshold** | Tuned on test set (default 0.35) | Tuned on test set (default 0.35) |
| **Retrain schedule** | 17:00 ET daily | 17:00 ET daily |

---

## Swing Model

### Architecture

- **Model file:** `app/ml/model.py` — `PortfolioSelectorModel`
- **Trainer:** `app/ml/training.py` — `ModelTrainer`
- **Underlying estimator:** `XGBClassifier` (400 trees, depth 4, lr=0.03)
- **Preprocessing:** `StandardScaler` (fit on training data, saved separately)
- **Feature weights:** MI-score weighting applied before scaling

### Label Construction

For each non-overlapping 10-day window across all symbols:
1. Compute forward return over the next 10 trading days
2. Within that window, rank all symbols by return
3. Top 20% (80th percentile) → label = 1, rest → label = 0
4. ATR-adaptive fallback: target = 1.5× ATR14, stop = 0.5× ATR14 (3:1 R:R)

### Training Pipeline

```
Fetch daily OHLCV (Polygon/Alpaca, 3-5 years)
    ↓
Rolling windows (63-day feature window, 10-day forward)
    ↓
Feature engineering per window (point-in-time, as_of_date passed)
    ↓
Feature store cache check (SQLite, avoids recomputing)
    ↓
Cross-sectional labeling (top 20% per window)
    ↓
MI-score feature weighting
    ↓
StandardScaler fit on X_train
    ↓
XGBClassifier.fit() with early stopping on X_test
    ↓
Threshold tuning on X_test (maximize F1, search 0.20–0.65)
    ↓
Save: {model}_v{N}.pkl + {model}_scaler_v{N}.pkl + {model}_meta_v{N}.pkl
    ↓
Register in ModelVersion DB table
```

### Model Persistence

Three files per version in `app/ml/models/`:

| File | Contents |
|---|---|
| `swing_v{N}.pkl` | Raw XGBClassifier object |
| `swing_scaler_v{N}.pkl` | StandardScaler (fit on training features) |
| `swing_meta_v{N}.pkl` | feature_names list, predict_threshold, feature_weights, model_type |

The `PortfolioSelectorModel.load()` reads all three and reconstructs the wrapper.

---

## Intraday Model

### Architecture

- **Model file:** `app/ml/model.py` — `PortfolioSelectorModel` (xgboost)
- **Trainer:** `app/ml/intraday_training.py` — `IntradayModelTrainer`
- **Underlying estimator:** XGBoost + LightGBM soft-vote ensemble (50/50)
- **Preprocessing:** StandardScaler

### Label Construction

For each trading day, across all symbols with sufficient bars:
1. Take the first ~1 hour of 5-min bars as the feature window
2. Compute best intraday return = (max high over next 24 bars − entry close) / entry close
3. Within that day, rank all symbols by best return
4. Top 20% → label = 1, rest → label = 0

### Training Pipeline

```
Fetch 5-min OHLCV from Alpaca (Russell 1000, 2 years)
    ↓ (cached as Parquet per symbol, 24h TTL)
Fetch SPY 5-min + daily OHLCV for market context
    ↓
Per symbol × per day: compute_intraday_features() → 37 features
    ↓ (parallel: min(24, cpu_count()) threads)
Cross-sectional daily labeling (top 20% per day)
    ↓
Recency sample weights (exp decay, 180-day half-life)
    ↓
Optuna HPO (n_trials configurable, 3-fold StratifiedKFold)
    ↓
XGBoost.fit() + LightGBM.fit() in parallel
    ↓
Soft-vote ensemble evaluation on X_test
    ↓
Save + register (same as swing)
```

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

### Full Feature List (37 features)

All computed purely from 5-min OHLCV + SPY 5-min bars. **No external API calls at inference time.**

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

Feature store was cleared on 2026-04-21 after the Yahoo Finance migration. All entries now use 126 features consistently.

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

## Current Model Versions

| Model | Version | Features | Trained | Notes |
|---|---|---|---|---|
| Swing | v67 | 126 | 2026-04-21 | Clean retrain with EDGAR + Alpaca sources |
| Intraday | v15 | 37 | 2026-04-07 | Unaffected by Yahoo change (pure OHLCV) |
