# MrTrader — Full System Review Prompt for LLM Critique

**Purpose:** This document is a self-contained briefing for an LLM (or team of analysts) to critique the MrTrader algorithmic trading system end-to-end, identify weaknesses, and suggest concrete improvements. The goal is to reach a standard that a top-tier systematic hedge fund would consider production-grade.

**Be direct. Be critical. Suggest things we haven't thought of. Don't just validate what exists.**

---

## 1. SYSTEM OVERVIEW

### What is MrTrader?

MrTrader is a fully automated, AI-driven paper trading system targeting retail-sized capital (~$20k) with ambitions to scale. It runs two concurrent ML-driven strategies:

- **Swing trading** — multi-day holds (3–15 days), daily bar signals, XGBoost ensemble
- **Intraday trading** — 2-hour holds from bar 12 (60 min after open), 5-min bar signals, XGBoost+LightGBM ensemble

It operates via a multi-agent async architecture (Portfolio Manager → Risk Manager → Trader) connected to Alpaca paper trading via REST/WebSocket.

### Operator Profile
- Solo developer, quantitative background, first production trading system
- Currently paper trading, planning live transition after models pass performance gates
- Capital: ~$20k, targeting risk-adjusted returns, not raw return maximization

### Primary Data Sources
| Source | Use |
|---|---|
| Polygon S3 | Historical daily bars (430 swing symbols, 5yr), 5-min bars (750 intraday), sector ETF bars (11 ETFs, 3yr) |
| Alpaca | Live execution, position state, recent 5-min bars for intraday cache refresh |
| Finnhub | Earnings calendar (bulk prefetch) |
| FMP (Financial Modeling Prep) | Fundamentals: profit_margin, revenue_growth, debt_to_equity, analyst upgrades/downgrades, institutional ownership, earnings surprise |
| NewsSignalCache (internal) | Proprietary NIS (News Intelligence Signal) — LLM-scored news sentiment per symbol, available from May 2025 onwards |
| MacroSignalCache (internal) | Macro NIS — aggregated macro sentiment from news LLM pipeline, available 259 days back |

---

## 2. SYSTEM ARCHITECTURE

### Agent Pipeline

```
[Nightly Scan / 5-min tick]
        │
        ▼
┌─────────────────────┐
│  Portfolio Manager  │  ← ML signal generation + opportunity scoring
│  (PM)               │    Scans universe, scores each symbol via model
└─────────┬───────────┘    Produces proposals: {symbol, direction, size, stop, target}
          │ async queue
          ▼
┌─────────────────────┐
│   Risk Manager      │  ← Sequential gate validation (22 checks)
│   (RM)              │    Approves, modifies, or blocks each proposal
└─────────┬───────────┘    Writes full audit trail to decision_audit table
          │ async queue
          ▼
┌─────────────────────┐
│     Trader          │  ← Order execution + position lifecycle
│                     │    Reconciles vs Alpaca on startup
└─────────────────────┘    Manages stops, targets, trailing stops, bars_held
```

### PM Re-scoring Loop
After RM approves a proposal and the Trader queues it, the PM runs a secondary re-score at execution time. If score drops below threshold, it emits a WITHDRAW signal. This prevents stale signals from executing after queue lag.

### Retraining
- Runs nightly as a subprocess (decoupled from uvicorn)
- Swing: ProcessPoolExecutor, 3yr daily bars, 430 symbols
- Intraday: sequential per symbol, 730d 5-min bars, 750 symbols
- Walk-forward gate must pass before model is promoted

### Persistence
- SQLite for all state: trades, proposals, audit trail, gate calibration
- Parquet files for bar cache (Polygon bulk download, updated nightly)
- Model artifacts in `app/ml/models/` (versioned JSON + .ubj/.txt files)

---

## 3. SWING MODEL — FULL DETAIL

### Architecture
- **Algorithm:** XGBoost (3-seed ensemble, blended probabilities)
- **Label:** `path_quality` — requires price to reach target before stop within 5 days. Binary classification.
- **Stop/Target:** ATR-based. Target = 2.0×ATR above entry, Stop = 1.2×ATR below entry (approximate R:R 1.67:1)
- **Entry signals:** RSI_DIP (RSI 45–58 + above EMA20) or EMA_CROSSOVER (EMA9 crosses above EMA20)
- **Universe:** 430 symbols (S&P 500 + NASDAQ 100 large-caps)
- **Training window:** 3 years of daily bars
- **Walk-forward:** 3 expanding folds, each fold adds ~18 months of train data

### Current Champion: v142
- **Version:** v142
- **Features:** 84
- **Avg Sharpe (original gate, earlier data window):** +1.181
- **Status:** Active champion, but see "Current Walk-Forward Results on Current Data" below

### Feature List (~89 features in latest versions, some pruned)

**Price & Technical (13)**
`rsi_14`, `rsi_7`, `macd`, `macd_signal`, `macd_histogram`, `ema_20`, `ema_50`, `price_above_ema20`, `price_above_ema50`, `price_change_pct`, `price_to_52w_high`, `price_to_52w_low`, `near_52w_high`

**Volume & Momentum (12)**
`volume_ratio`, `volume_trend`, `uptrend`, `downtrend`, `volatility`, `momentum_5d`, `momentum_20d`, `momentum_60d`, `momentum_252d_ex1m`, `atr_norm`, `bb_position`, `stoch_k`

**Advanced Technicals (10)**
`adx_14`, `consecutive_days`, `reversal_5d`, `reversal_5d_vol_weighted`, `reversal_3d`, `williams_r_14`, `cci_20`, `price_acceleration`, `stochrsi_k`, `stochrsi_d`

**Market Structure (11)**
`keltner_position`, `cmf_20`, `dema_20_dist`, `vol_expansion`, `adx_slope`, `volume_surge_3d`, `consolidation_position`, `trend_efficiency`, `trend_consistency_63d`, `vol_price_confirmation`, `pressure_index`

**Relative Momentum (11)**
`rs_vs_spy`, `rs_vs_spy_5d`, `rs_vs_spy_10d`, `rs_vs_spy_60d`, `momentum_20d_sector_neutral`, `momentum_60d_sector_neutral`, `momentum_5d_sector_neutral`, `sector_momentum`, `sector_momentum_5d`, `mean_reversion_zscore`, `up_day_ratio_20d`

**Volatility & Options (7)**
`vol_percentile_52w`, `vol_regime`, `vol_of_vol`, `parkinson_vol`, `options_put_call_ratio`, `options_iv_atm`, `options_iv_premium`

**Fundamentals (8) — now PIT-correct**
`profit_margin`, `revenue_growth`, `debt_to_equity`, `earnings_proximity_days`, `insider_score`, `fmp_surprise_1q`, `fmp_surprise_2q_avg`, `fmp_days_since_earnings`

**Earnings Quality (8)**
`fmp_analyst_upgrades_30d`, `fmp_analyst_downgrades_30d`, `fmp_analyst_momentum_30d`, `fmp_inst_ownership_pct`, `fmp_inst_change_pct`, `fmp_consecutive_beats`, `fmp_revenue_surprise_1q`, `earnings_surprise_1q`

**WorldQuant-Inspired Alphas (13)**
`wq_alpha3`, `wq_alpha4`, `wq_alpha6`, `wq_alpha12`, `wq_alpha33`, `wq_alpha34`, `wq_alpha35`, `wq_alpha40`, `wq_alpha43`, `wq_alpha44`, `wq_alpha46`, `wq_alpha53`, `wq_alpha54`

**Regime Context (8)**
`vix_level`, `vix_regime_bucket`, `vix_fear_spike`, `vix_percentile_1y`, `vrp`, `beta_252d`, `spy_trend_63d`, `rs_vs_spy`

**Regime Interactions (6)**
`rsi_x_vix_regime`, `momentum20_x_vix_bucket`, `adx_x_spy_trend`, `rsi_x_spy_trend`, `vol_pct_x_vix_bucket`, `adx_x_vix_bucket`

**News Intelligence Signal / NIS (5)**
`nis_direction_score`, `nis_materiality_score`, `nis_already_priced_in`, `nis_sizing_mult`, `nis_downside_risk`
*Note: NIS only available from May 2025 — training rows before that date have NaN (handled via XGBoost missing-value direction)*

**Macro NIS (5)**
`macro_avg_direction`, `macro_pct_bearish`, `macro_pct_bullish`, `macro_avg_materiality`, `macro_pct_high_risk`
*Note: 259 days of history only*

**Sector (2 — recently un-pruned)**
`sector_momentum` (20d ETF return), `sector_momentum_5d` (5d ETF return)

### Swing Walk-Forward Results — ALL VERSIONS on Current Data (2026-05-05)

**v142 (84 features, current champion) — current data window:**
| Fold | Train Period | Test Period | Trades | Win Rate | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| 1 | 2021-04→2022-08 | 2022-08→2023-11 | 189 | 47.1% | **+1.20** | 3.1% |
| 2 | 2021-04→2023-11 | 2023-11→2025-02 | 231 | 43.7% | **+0.24** | 4.0% |
| 3 | 2021-04→2025-02 | 2025-02→2026-05 | 172 | 39.5% | **-0.51** | 5.1% |
| **Avg** | — | — | — | 43.4% | **+0.310** | — |
| **Gate** | — | — | — | — | **FAIL (need >0.8)** | — |

**Previous versions (all gate-failed):**
- v144 (+5 stock NIS, wrong 0.0/1.0 defaults): avg ~-0.148
- v145 (+5 macro NIS, NaN encoding): avg ~-0.148
- v146 (+5 macro NIS, correct NaN encoding): avg -0.148, fold 3 = -1.07

**Key insight:** Fold 3 (Feb 2025–present) collapses across ALL swing versions. This period covers the 2025 tariff shock, elevated VIX, trend reversals. The model trained on 2021–2025 may be over-fit to the low-volatility momentum regime of 2021–2024.

---

## 4. INTRADAY MODEL — FULL DETAIL

### Architecture
- **Algorithm:** XGBoost (3-seed ensemble) + LightGBM, blended probabilities
- **Label:** Cross-sectional top-20% per day. A symbol is labeled positive if its 2-hour return (bars 12–24 from open, i.e. 60–180 min) ranks in the top 20% of all symbols that day.
- **Entry:** Bar 12 (exactly 60 minutes after open) — gives time to observe opening range
- **Hold:** 24 bars = 2 hours
- **Stop:** 0.4×ATR from entry
- **Target:** 0.8×ATR from entry (R:R 2:1)
- **Universe:** 750 symbols, 730 days of 5-min bars
- **Walk-forward:** 3 expanding folds

### Current Champion: v29
- **Version:** v29
- **Features:** 50
- **Original gate result (earlier data window):** +1.830 avg Sharpe ✅
- **Same model, current data window:** -0.327 avg Sharpe ❌

### Feature List (61 features in v41, the latest trained)

**Opening Range & Structure (4)**
`orb_position`, `orb_breakout`, `orb_direction_strength`, `gap_pct`

**VWAP (3)**
`vwap_distance`, `vwap_cross_count`, `gap_fill_pct`

**Price Structure (3)**
`session_hl_position`, `prev_day_high_dist`, `prev_day_low_dist`

**Trend (3)**
`ema9_dist`, `ema20_dist`, `ema_cross`

**Momentum (5)**
`macd_hist`, `bb_position`, `rsi_14`, `session_return`, `ret_15m`, `ret_30m`

**Volatility (2)**
`atr_norm`, `range_compression`

**Oscillators (2)**
`stoch_k`, `williams_r`

**Volume / Order Flow (4)**
`volume_surge`, `cum_delta`, `vol_trend`, `obv_slope`

**Candlestick Structure (4)**
`upper_wick_ratio`, `lower_wick_ratio`, `body_ratio`, `consecutive_bars`

**Market Context — SPY (3)**
`spy_session_return`, `spy_rsi_14`, `rel_vol_spy`

**Time-of-Day (4)**
`time_of_day`, `minutes_since_open`, `is_open_session`, `is_close_session`

**Session Segment (3)**
`session_segment`, `seg_x_high_dist`, `seg_x_atr_norm`

**Daily Vol Context (3)**
`daily_vol_percentile`, `daily_vol_regime`, `daily_parkinson_vol`

**Quality / Structure (8)**
`whale_candle`, `trend_efficiency`, `green_bar_ratio`, `above_vwap_ratio`, `pullback_from_high`, `range_vs_20d_avg`, `rel_strength_vs_spy`, `vol_x_momentum`

**SPY-Relative (3) — Phase 86b**
`stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`

**NIS Features (5) — Phase 64**
`nis_direction_score`, `nis_materiality_score`, `nis_already_priced_in`, `nis_sizing_mult`, `nis_downside_risk`

### Intraday Walk-Forward Results — ALL VERSIONS on Current Data (2026-05-05)

**v29 (50 features, active champion) — current data:**
| Fold | Test Period | Trades | Win Rate | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024→Apr 2025 | 248 | 41.1% | **-0.67** |
| 2 | Apr 2025→Oct 2025 | 249 | 45.4% | **-1.27** |
| 3 | Oct 2025→Apr 2026 | 249 | 49.8% | **+0.97** |
| **Avg** | — | — | 45.4% | **-0.327 FAIL** |

**v41 (61 features: +NIS, +SPY-relative, +macro NIS) — same data:**
| Fold | Test Period | Trades | Win Rate | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024→Apr 2025 | 248 | 41.1% | **+0.69** |
| 2 | Apr 2025→Oct 2025 | 249 | 45.4% | **-0.34** |
| 3 | Oct 2025→Apr 2026 | 249 | 49.8% | **+0.15** |
| **Avg** | — | — | 45.4% | **+0.167 FAIL** |

**Previously retired versions:**
- v34–v36: SPY market-wide features → avg -0.529 (cs_normalize zeros market-wide signals)
- v38: Realized-R labels + ensemble → +0.675 (gate not met)
- v39: Realized-R + NIS → negative
- v40: Realized-R + NaN encoding → -0.519

**Key insight:** The Apr–Oct 2025 period (tariff shock, VIX elevated, whipsaw conditions) is universally destructive across all model versions. Fold 3 (Oct 2025–Apr 2026) shows recovery in v29 (+0.97) suggesting the architecture works in calmer conditions. v41 is worse on fold 3 despite more features — the new features may be adding noise.

**Critical observation:** v29's original +1.83 gate pass was on an earlier data window where Apr–Oct 2025 did not yet appear in any test fold. The model was never validated against that regime.

---

## 5. PORTFOLIO MANAGER GATES

### Opportunity Score (Phase 88 — Graduated, not binary)
```
score = 0.35 × vix_score + 0.20 × vix_trend_score + 0.30 × ma_score + 0.15 × momentum_score

vix_score      = clip(1 - (VIX - 15) / 20, 0, 1)     → 1.0 at VIX≤15, 0.0 at VIX≥35
vix_trend      = clip(1 - (VIX - VIX_5d_avg) / 5, 0, 1)
ma_score       = 1.0 if SPY ≥ MA20, else 0.4
momentum_score = clip(0.5 + SPY_5d_return × 25, 0, 1)

score ≥ 0.65  → normal (full candidate pool)
0.35–0.64     → caution (cap at 2 candidates)
< 0.35        → suppress all entries
```

### Other PM Gates
- Earnings calendar blackout (Finnhub): intraday blocks ±1d before / 3d after; swing blocks 3d before
- Macro event windows (FOMC, CPI, NFP): 4h window around scheduled events
- Intraday position cap: max 3 concurrent
- Gross exposure cap: total deployed ≤ 80% of account
- Strategy budget caps: swing ≤ 70%, intraday ≤ 30%

### RM Sequential Gates (22 checks)
Symbol halt → Earnings → Macro event → Position cap → PDT → Gross exposure → Strategy budget → Buying power → Position size → Sector concentration → Correlation risk → Daily loss → Account drawdown → Open positions → Portfolio heat → Bid-ask spread → ADTV liquidity → Beta exposure → Factor concentration → Dynamic stop calculation

All gate outcomes logged to `decision_audit` table with gate_category (alpha/quality/risk/structural/scan) for calibration.

---

## 6. LABEL DESIGN

### Swing: path_quality
- Binary: 1 if price hits target (2.0×ATR above entry) before stop (1.2×ATR below entry) within 5 days
- Positive class rate: ~20–25%
- Problem: This is a fixed-horizon exit label, not a true path-quality label. It captures only the first 5 days.

### Intraday: Cross-sectional top-20%
- For each trading day, label = 1 if the symbol's 2h return (bar 12→36) ranks in top 20% of all symbols that day
- Positive class rate: exactly 20% by construction
- Problem: The model is trained to rank, but execution is binary (enter or not). We pick top N candidates but don't size by predicted rank.
- Note: Realized-R labels (predict raw R-multiple) were tried and failed (AUC ~0.51) — features don't predict magnitude, only relative rank.

---

## 7. TRAINING PIPELINE DETAILS

### Swing Training
- ProcessPoolExecutor (8 workers max to avoid memory crash)
- Per-symbol: computes features for each rolling window end date (daily)
- PIT (point-in-time) correct: fundamentals loaded from `fundamentals_history.parquet` (391/430 symbols, 11,285 snapshots), sector ETF momentum from `sector_etf_history.parquet`
- PRUNED_FEATURES: list of features computed live at inference but not during training (historically: sector_momentum was pruned because ETF history didn't exist — now un-pruned)
- Walk-forward expanding window: train from earliest date, evaluate on next 18-month window
- HPO: 100-trial Optuna search, then frozen (HPO variance ~2.0 Sharpe on identical features — frozen after search)
- 3-seed ensemble: XGBoost with seeds 42, 123, 777 — blended probabilities

### Intraday Training
- Sequential per symbol (no multiprocessing — 5-min bar data too large)
- cs_normalize: after feature computation, z-score normalize each feature cross-sectionally (per day, across all symbols). This means market-wide signals (VIX, SPY level) become zero — only per-symbol-varying signals survive.
- Feature matrix: 276,946 train rows / 61,147 test rows for v41
- AUC: 0.6465 (HPO best), 0.6289 (OOS)
- Top features: `seg_x_atr_norm` (14%), `atr_norm` (11%), `range_compression` (6%), `minutes_since_open` (3%), `time_of_day` (3%) — NIS features not in top 5

---

## 8. KNOWN ISSUES & CONSTRAINTS

### Data Issues
1. **NIS stock-level data gap:** NewsSignalCache only has data from May 2025 onwards. Training windows go back 2-3 years — model sees NaN for NIS in ~80% of training rows. Signal quality is limited until backfill is complete. Estimated cost: $50-100 in LLM API calls.
2. **Macro NIS history:** Only 259 days. Same gap problem.
3. **Fundamentals PIT correctness:** `pe_ratio` and `pb_ratio` excluded (require live price at filing date — hard to backfill correctly). 391/430 symbols have fundamentals history; 39 symbols use live values.
4. **Options data sparsity:** `options_put_call_ratio`, `options_iv_atm`, `options_iv_premium` — coverage may be limited for smaller symbols.

### Architecture Issues
1. **Fold 3 collapse (Apr–Oct 2025 regime):** The tariff/volatility shock created a structurally different regime. All models trained on 2021–2025 data perform poorly on this period. The dynamic opportunity score (Phase 88) is designed to reduce/suppress trading in such conditions, but the walk-forward simulation doesn't account for this (it simulates all trades regardless of score).
2. **Walk-forward doesn't simulate PM gates:** The tier-3 walk-forward runs the model in isolation. PM gates (opportunity score, position caps, earnings blackouts) are not applied during simulation. Real live performance likely differs.
3. **Binary entry signal (swing):** RSI_DIP and EMA_CROSSOVER are rule-based pre-filters. The ML model only scores candidates that pass these rules — it cannot discover other entry patterns.
4. **Fixed hold period (intraday):** Always exit at bar 36 (or stop/target first). Cannot learn optimal hold duration.
5. **cs_normalize constraint:** Any feature that is market-wide (same value for all symbols on a given day) becomes zero after normalization. This eliminates VIX, SPY level, macro data as direct features — they can only enter as interactions or via the PM opportunity score layer.
6. **Database state divergence:** DB is the primary source of truth for position state, but Alpaca is the execution authority. Reconciliation is imperfect — bars_held, stop/target drift, and pending fills have all caused bugs. Phase 100 (Alpaca as single source of truth) is planned but not yet implemented.

### Model Issues
1. **Label leakage risk:** The path_quality label for swing uses the 5-day forward path. Care must be taken that no forward-looking features (e.g., future volume) enter the feature matrix. This has not been formally audited.
2. **Regime non-stationarity:** A single XGBoost model trained on 3 years including multiple regimes (bull 2021, crash 2022, recovery 2023, AI bull 2024, tariff shock 2025) may not generalize to any single regime.
3. **Class imbalance:** Swing positive rate ~20-25%, intraday exactly 20%. scale_pos_weight=3.99 is applied but this alone doesn't resolve regime-specific imbalance.
4. **HPO variance:** Optuna 100-trial search has variance ~2.0 Sharpe on identical features. Params are frozen after search to prevent re-searching on every retrain (would change results randomly).

---

## 9. WHAT HAS BEEN TRIED AND THE RESULTS

### Swing
| Attempt | Change | Result |
|---|---|---|
| v119→v142 | Routine retrain | +1.181 avg Sharpe ✅ (on older data window) |
| v144 | +5 stock NIS (0.0/1.0 defaults — wrong encoding) | ~-0.148 ❌ |
| v145 | +5 macro NIS (NaN encoding correct) | ~-0.148 ❌ |
| v146 | +5 macro NIS, correct NaN, 89 features | avg -0.148, fold3 -1.07 ❌ |
| v142 on current data | No change, just re-evaluated | avg +0.310 ❌ (fold3 -0.51) |
| Regime interactions | 6 VIX/SPY/ADX interaction features | In model, unclear contribution |
| Sector momentum | sector_momentum, sector_momentum_5d un-pruned | In next retrain |
| Fundamentals PIT | profit_margin, revenue_growth, debt_to_equity now PIT-correct | In next retrain |

### Intraday
| Attempt | Change | Result |
|---|---|---|
| v29 | Cross-sectional top-20%, 50 features | +1.830 ✅ (older window), -0.327 ❌ (current) |
| v34–v36 | SPY market-wide features | -0.529 ❌ (cs_normalize zeros them) |
| v37 | Fresh HPO, same 53 features | -0.219 ❌ |
| v38 | Realized-R labels + 3-seed ensemble | +0.675 (gate not met) |
| v39 | Realized-R + NIS (58 features) | Negative ❌ |
| v40 | Realized-R + NaN encoding + 63 features | -0.519 ❌ |
| v41 | Cross-sectional + NIS + SPY-relative (61 features) | +0.167 ❌ |

---

## 10. THE ASK — WHAT WE WANT FROM YOU

**Please review everything above and provide:**

### A. Critical Assessment
- What is fundamentally wrong or weak about this system?
- What would a quant at a serious systematic fund immediately flag?
- Which design decisions are likely limiting alpha the most?
- Is the label design appropriate? Would you design it differently?
- Is the walk-forward methodology sound? What is it missing?

### B. The Regime Problem
- The Apr–Oct 2025 period (tariff shock, elevated VIX) collapses all models. Is this a model problem, a feature problem, a label problem, or an unavoidable data problem?
- What approaches would you use to make models more robust to regime change?
- Should we train separate regime-conditional models? Use regime as a gating layer? Use different architectures for different VIX levels?
- The dynamic opportunity score partially addresses this at the PM layer — is that enough, or does it need to be inside the model?

### C. Feature Suggestions
- Are there obvious features missing from either model?
- Are there features present that are likely noise or leakage risks?
- For the swing model: given the 3yr daily bar dataset and fundamentals, what alpha factors are well-established in academic literature that we should add?
- For the intraday model: given 5-min bars and the cs_normalize constraint, what features could capture true intraday alpha?
- How should we think about the NIS (news intelligence signal) features given the data sparsity problem?

### D. Architecture Suggestions
- Is the PM→RM→Trader async queue architecture appropriate? What would you change?
- Is the walk-forward gate design good? Gate is avg Sharpe > 0.8 (swing) / > 1.5 (intraday), min fold > -0.3. Are these thresholds sensible?
- Should swing and intraday be combined into a single model or kept separate?
- Is XGBoost the right algorithm? What about neural approaches (LSTM, Transformer on sequences), linear models for interpretability, or ensemble approaches combining different model families?
- Should position sizing be dynamic (Kelly, vol targeting) rather than fixed percentage?

### E. Roadmap Suggestions
- If you had to prioritize 5 things to work on next, what would they be and why?
- What data sources would have the highest marginal value to add?
- What would need to be true for this system to be used by a systematic fund?
- What risks does this system have that the developer may not be aware of?

### F. What's Actually Good
- What design decisions are sound and should be preserved?
- What has been done well that is non-obvious?

---

## 11. CONTEXT: WHAT THE SYSTEM IS TRYING TO ACHIEVE

The target is a system that:
1. Generates consistent risk-adjusted returns (Sharpe > 1.5 swing, > 1.5 intraday) across market regimes
2. Has rigorous risk controls (drawdown limits, concentration limits, correlation limits)
3. Is explainable enough that a human operator can understand why any given trade was taken or blocked
4. Can be trusted to run autonomously overnight and during market hours without supervision
5. Eventually scales to $100k+ live capital with institutional-grade execution quality

The developer understands this is ambitious. The goal of this review is to identify the highest-leverage changes that move the system closest to that standard.

---

*Generated 2026-05-05. All walk-forward results are on current data window (folds ending 2026-04-17). System is in paper trading mode.*
