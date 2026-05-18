# MrTrader — Deep Quant Review Request

**Date:** 2026-05-18  
**Purpose:** We are building an algorithmic trading system and have hit a wall on model training and walk-forward validation. We need brutal, expert assessment from the perspective of a senior quant at a major hedge fund who has built and shipped these systems. Do not spare feelings. If our architecture is fundamentally broken, say so. If we are on the right track with minor fixes, say that too.

**Please assess:**
1. Our model training pipeline (label design, feature engineering, ML architecture)
2. Our walk-forward methodology (is it honest? does it reflect real trading?)
3. Our trade entry/exit logic (do the backtest assumptions match reality?)
4. Our overall strategy design (swing + intraday, $20k capital, Russell 1000 universe)
5. Our current decision to "pivot to factor portfolio" — is this the right call?
6. What you would do next, with specific actionable steps in priority order

---

## System Overview

**Capital:** $20,000 paper trading account (Alpaca paper)  
**Goal:** Systematic long-only equity trading — swing (multi-day holds) + intraday (same-day closes)  
**Universe:** Russell 1000 (~750 symbols) for swing; top ~100 liquid names for intraday  
**Infrastructure:** Python, FastAPI backend, SQLite DB, Alpaca brokerage API, yfinance for historical data  
**Agent architecture:** Portfolio Manager (PM) → Risk Manager (RM) → Trader agents communicating via async queue

---

## Strategy 1: Swing Trading

### What we're trying to do
Select top-5 to top-20 stocks from the Russell 1000 universe each morning. Hold for ~5–20 trading days. Long-only. Target: daily bar signals, entry at next day open, exit at ATR-based target/stop or time limit.

### Label Design (ML path)
We have tried several labeling schemes:

**triple_barrier (original):**  
- Entry price = day's close  
- Target = +1.5× ATR14 above entry  
- Stop = −0.5× ATR14 below entry  
- Max hold = 10 bars  
- Label = 1 if target hit before stop/time, 0 otherwise  
- **Problem:** In high-vol regimes (VIX spike, 2022 bear), ATR widens → barriers become unreachable → large fraction of labels become None/0 → model loses training data exactly when regime matters most

**cross_sectional (LambdaRank, current attempt):**  
- 20-day forward window, compute each stock's return  
- Label = quintile rank (1=top 20%, 0=bottom 20%), middle 60% excluded from training  
- ~150 positive + 150 negative samples per day  
- Fed into LGBMRanker (LambdaRank objective)  
- **Problem:** In bear markets (2022, −25% SPY), top quintile = "fell least" (e.g., −5%). Model learns defensive/low-beta signals that CONTRADICT the momentum+quality features it was trained on during bull regimes. This creates a train/test mismatch that is NOT tunable away.

**What we haven't tried:**  
- Absolute return threshold labels (e.g., label=1 if 10d return > +3%)
- Regime-conditional labels (different thresholds in bull vs. bear)
- Ranking by forward Sharpe ratio rather than raw return
- Multi-task learning (predict both direction and magnitude)

### Feature Engineering (swing)
17 core features (PHASE_C_PLUS_FEATURE_KEEP_LIST):

**Momentum (Tier 1, high IC):**
- `momentum_252d_ex1m` — 12-month momentum ex last month (IR=1.99 in IC study)
- `momentum_20d`, `momentum_60d`, `momentum_5d`
- `price_to_52w_high` (IC IR=1.11)

**Quality/Fundamentals (Tier 1):**
- `profit_margin`, `operating_margin`, `pe_ratio` (annual, fetched quarterly)

**Volatility/Regime:**
- `vol_regime` — VIX z-score (IR=1.87 in IC study)
- `atr_norm`, `vol_percentile_52w`

**Technical:**
- `range_expansion`, `price_to_52w_low`, `volume_trend`
- `rsi_14`, `ema_ratio_20_50`

**Sector-relative (experimental, dropped):**
- `sector_momentum`, `momentum_20d_sector_neutral`, `momentum_60d_sector_neutral`
- Added in v208, dropped in v209a — hurt HPO (expanded search space, landed in worse basin)

**Feature engineering window:** 63-day rolling window (1 quarter) per training sample. TSNormalizer applied per feature across time (z-score with trailing mean/std to prevent lookahead).

### ML Model Architecture
**LGBMRanker (LambdaRank):**
- Training data: ~750 symbols × multiple 20-day forward windows across 6 years
- Group structure: each day is a "query group" — model learns to rank symbols relative to each other within a day
- Eval metric: NDCG@3 or NDCG@5
- HPO: Optuna TPE, 50 trials, tuning n_estimators (200–800), num_leaves (8–63), learning_rate, subsample
- Seeded (TPESampler(seed=42)) to make runs reproducible

**Prior model (deprecated):** XGBoost binary classifier — AUC consistently 0.49–0.53 across 15+ training runs. Rejected as producing no signal above chance.

### Walk-Forward Methodology (Swing)
- **5 folds, 6 years total (2020–2026)**
- **Expanding train window** (fold 1 trains on 1yr, fold 5 trains on 5yr)
- **Purge:** 10 trading days between train end and test start (prevents label leakage from 10-day forward window)
- **Embargo:** 10 trading days after test end before next fold trains (prevents test rows entering next train set)
- **Transaction costs:** 5 bps per side (0.05%)
- **Universe:** pit_union("russell1000", fold_start, fold_end) + DB historical symbols (survivorship bias fix)
- **Simulation:** AgentSimulator — daily bar simulation, ATR-based entry/exit, regime score PIT

### Walk-Forward Simulation (AgentSimulator) — what it actually does
- Each test day: score all universe stocks using the trained model → rank by predicted probability
- Take top-N stocks above confidence threshold (default: top-10, min confidence 0.40)
- For each selected stock: "PM proposes" → "RM validates" → "Trader enters"
- Entry: next day open (simulated as previous close × 1.001 due to OHLCV-only data)
- Exit rules (triggered daily on each open position):
  - **Stop:** if current close < stop_price (entry − 0.5× ATR14 × entry, or EMA20−based)
  - **Target:** if current close > target_price (entry + 1.5× ATR14 × entry)
  - **Max hold:** 20 bars (time-based forced exit)
  - **Re-score exit:** if model score drops below 0.35 threshold
- Position sizing: Kelly-like, based on confidence × account value / ATR, capped at 5% per position
- Multiple positions: up to 5 open simultaneously, regime-aware (RISK_OFF → 3 max positions, halved size)

### LambdaRank Campaign Results (9 training runs, all GATE FAILED)

Gate: avg Sharpe ≥ 0.80, min fold ≥ −0.30

| Version | Config | Avg Sharpe | Min Fold | Status |
|---|---|---|---|---|
| v203 | First correct run (bugs fixed) | −0.041 | −0.847 | ❌ FAILED |
| v204 | Bug5b fix (TSNorm) | +0.103 | −0.625 | ❌ FAILED |
| v205b | BenignGate removed | **+0.267** | −0.625 | ❌ FAILED |
| v206 | 50 HPO trials | +0.158 | −0.625 | ❌ FAILED |
| v207 | + sector-neutral features | −1.322 | −2.597 | ❌ (sector ETF bug) |
| v208 | Bug fixed, 19 features | −0.156 | −0.625 | ❌ FAILED |
| v209a | Reverted to 17 features | −0.163 | −0.847 | ❌ FAILED |
| v209b | NDCG@3, seeded HPO | −0.294 | −1.822 | ❌ FAILED |
| v210 | Same as v209b (auto-retrain) | −0.294 | −1.822 | ❌ FAILED |

**Per-fold breakdown (v209a, 5-fold, representative):**

| Fold | Period | Trades | Sharpe |
|---|---|---|---|
| 1 | 2021-05-30→2022-05-19 | 231 | **−0.847** |
| 2 | 2022-05-30→2023-05-19 | 204 | **−0.625** |
| 3 | 2023-05-30→2024-05-18 | 215 | **+0.724** |
| 4 | 2024-05-29→2025-05-18 | 180 | **+0.339** |
| 5 | 2025-05-29→2026-05-18 | 212 | **−0.404** |

**Pattern:** Folds 3+4 (2023–2025, clean AI-led bull) pass. Folds 1, 2, 5 (bear/transition/late-cycle) always fail. Fold 2 (2022 bear) has never exceeded −0.625 across 9 runs. The gate requires min fold ≥ −0.30.

**Root cause assessment (Opus 4.7 + our analysis):**
- Fold 2 is structurally unreachable: cross-sectional quintile labels in a −25% bear market label "fell least" as positive. The model learns defensive signals during training on this period, contradicting momentum/quality features.
- HPO variance: σ≈0.20 per run (true mean ≈ +0.09). v205b's +0.267 is likely a lucky draw.
- NDCG→Sharpe transfer correlation is ~0.1–0.3 at this feature/data scale.

---

## Strategy 2: Intraday Trading

### What we're trying to do
Select 2–5 stocks per day at 09:45, 10:45, 13:00 ET scan windows. Enter with a 5-minute bar signal. Exit intraday (before close). Long-only. Target: 0.5–2% per trade, hold 30 min to 3 hours.

### Label Design (Intraday)
- **cross-sectional top-20% rank** on 1-hour forward return
- Features computed from 5-minute bars
- Label = 1 if stock's 60-min return is in the top 20% of the intraday universe that session
- Training data: 5-min bars for ~100 liquid names, 2021–2024

### Feature Engineering (Intraday, 59 features — "Branch B")
- 5-min momentum (close/open, 5-bar, 10-bar momentum)
- VWAP deviation
- ATR(14) normalized by price
- Volume ratio (5-bar vs 20-bar avg)
- Opening range breakout signal
- SPY/QQQ correlation and relative momentum
- Intraday price action patterns (gap, pullback, breakout)

### Walk-Forward (Intraday)
- **3 folds, 2 years total (2022–2024)**
- Purge: 2 trading days
- Transaction costs: 15 bps per side (higher than swing due to shorter hold)
- Gate: avg Sharpe ≥ 1.00 (recalibrated from 1.50), min fold ≥ −0.30

### Intraday Walk-Forward Results (current champion v51)
Avg Sharpe = +0.529. GATE FAILED (below 1.00 gate). Active in paper trading but below gate.

---

## Ancillary Systems Built Around Trading

### NIS (News Intelligence Service)
A multi-tier news scoring pipeline:
- **Tier 1 (macro):** Daily macro sentiment from Finnhub, FMP economic calendar. Produces risk_level (LOW/MEDIUM/HIGH) and sizing_multiplier (0.5–1.0). Blocks all entries on HIGH macro risk.
- **Tier 2 (stock-specific):** Sentiment analysis for individual symbols from news headlines. Produces action_policy (block_entry / size_down / normal). Applied at morning digest (09:00 ET) to pre-score the swing candidate universe.
- **Post-event refresh:** After FOMC/NFP/CPI releases (detected from economic calendar), NIS macro context is invalidated and re-fetched 3 minutes later.
- **Integration:** PM agent checks NIS before sending proposals to RM. block_entry symbols are excluded. size_down symbols get 50% position size.

### Regime Model (regime_model_v4)
Binary classifier (XGBoost) predicting RISK_ON vs RISK_OFF regime:
- Features: SPY 5-day return, VIX level, VIX change, HYG/IEF spread, VIX3M/VIX ratio, market breadth (% stocks above MA50)
- Trained on 2019–2024 daily macro data
- Output: 0.0–1.0 score → RISK_OFF (<0.35), NEUTRAL (0.35–0.65), RISK_ON (>0.65)
- Applied in WF via PIT `regime_score_history` (score computed from data available on each test day)
- Live: max positions reduced (5→3) and sizes halved in RISK_OFF; entries blocked below 0.35 threshold

### Opportunity Score
Continuous 0.0–1.0 daily score combining:
- VIX level (0 at VIX=35, 1 at VIX=15)
- SPY vs MA20 (1.0 above, 0.4 below)
- SPY 5-day momentum (clamped 0–1)
- Market breadth (% universe above MA50)
- Cross-sectional return dispersion
Used to gate intraday scan: score ≥ 0.65 → normal (top-5 intraday picks), 0.35–0.64 → 2 picks max, <0.35 → no picks.

### BenignGate (removed from LambdaRank path)
Originally blocked all swing entries on "adverse regime" (regime score below threshold). Removed from LambdaRank WF after v205b showed removing it improved avg Sharpe from ~0.0 to +0.267. Still in codebase but bypassed via `no_prefilters=True` in WF.

### Risk Manager
- Max single position: 5% of account value
- Max sector concentration: 20% of account value
- Max daily loss: 2% of account value (blocks new entries after breach)
- Max peak-to-trough drawdown: 5% (blocks new entries after breach)
- Max open positions: 5 simultaneously

### Factor Portfolio (Phase D — just deployed)
Rule-based composite score (no ML):
- **Tier 1 (2× weight):** momentum_252d_ex1m, price_to_52w_high, profit_margin, operating_margin, pe_ratio (reversed)
- **Tier 2 (0.5× weight):** price_to_52w_low, volume_trend, range_expansion, gross_margin, revenue_growth_yoy
- Z-scored cross-sectionally, then equal-weight composite
- **Regime gate:** SPY > 200-day MA AND VIX < 30
- **Validated backtest:** Sharpe=1.335, CAGR=32.4%, MaxDD=−25.9% (COVID crash 2020), WorstYear=+4.6% (2022), on 6-year dedicated backtest (2019–2024) with monthly rebalance + equal-weight
- **Deployed today** as PM agent's default swing selector (config: pm.swing_selector='factor_portfolio')
- **Problem found:** When run through AgentSimulator walk-forward (ATR stop 0.5×, target 1.5×), avg Sharpe = **−1.43**. This is likely an execution model mismatch — the validated backtest used monthly rebalance + no stops; the AgentSimulator uses ATR-based stops that fire prematurely on normal intra-month volatility.

---

## Current System Bugs / Known Issues

### Execution model mismatch (CRITICAL — unresolved)
The factor portfolio was validated in a "monthly rebalance, equal-weight, no stop" framework. It was deployed to the live PM agent which uses ATR-stop/target exits. The AgentSimulator walk-forward with ATR stops shows avg Sharpe −1.43. We don't know yet if this means:
(a) The factor signals are real and only the stop execution is wrong
(b) The 1.335 backtest Sharpe is curve-fitted / overfitted to the historical period
(c) Both
We have not reconciled the execution assumptions between the backtest validation and live trading.

### Training/WF alignment (partially resolved)
Earlier bugs (now fixed) in walk-forward simulation:
- AgentSimulator was not applying TSNorm state at inference (fixed: WF-A1)
- WF was using SP100 (~81 symbols) while training used Russell 1000 (fixed: WF-A3)
- WF was not downloading historical delisted symbols (partial survivorship bias fix: WF-A2)
- Feature cache (`build_feature_cache`) was not computing sector-neutral features (all sector_momentum=0.0 in WF test folds). This caused v207's −1.322 avg Sharpe. Fixed in current codebase.

### Profit factor always = 0.00 in WF
`trade_returns` list is not propagated from strategy object to SimulationResult. The Sharpe is correct (computed from equity curve), but all PF displays show 0.00. Not affecting model evaluation but misleading.

### Label look-forward (resolved)
TSNormalizer previously fitted on full dataset before feature-list filter. Fixed: TSNorm now fitted only on training set, applied identically to test set.

---

## What We've Decided (Current State)

1. **Closed LambdaRank campaign** after 9 runs. True mean ≈ +0.09 Sharpe with gate at 0.80 — 3× structural gap.
2. **Deployed rule-based factor portfolio** as swing selector default (today). Validated Sharpe 1.335 in dedicated backtest. But execution model mismatch unresolved.
3. **ML training paused** for swing. Next ML idea is XGBoost with triple-barrier labels but with absolute return thresholds instead of ATR-relative thresholds (to avoid high-vol label collapse).
4. **Intraday (v51) continues paper trading** but below gate (0.529 vs 1.00 gate).

---

## Our Planned Next Steps (what we THINK we should do)

1. **Fix execution model:** Decide whether the factor portfolio in live trading should use ATR stops (current) or monthly-hold exits (what was validated). Need one consistent assumption across backtest and live.
2. **XGBoost triple-barrier with absolute return labels:** Use label=1 if 10d return > +X% (absolute threshold), not ATR-relative. Eliminates high-vol label degradation. Still needs to handle bear regime.
3. **Regime-conditional model:** Train separate models for RISK_ON vs RISK_OFF regimes, route at inference time.
4. **Intraday improvement:** Investigate Fold 2 (2022) intraday failure. Similar pattern to swing — bear regime destroys top-20% rank signal.
5. **Walk-forward honest validation question:** We are uncertain whether our WF is truly honest. We have transaction costs (5 bps swing, 15 bps intraday), purge/embargo, survivorship bias partial fix, PIT regime scores. But we still have concerns about whether the AgentSimulator execution logic (ATR stops, daily rescoring, position sizing) matches what a real trade would look like.

---

## Specific Questions for You

1. **Is LambdaRank the right architecture for this problem?** Cross-sectional ranking over 750 stocks with daily rebalance at $20k capital seems like it has a transaction cost problem even if signals are real. What would you use instead?

2. **Is our walk-forward methodology honest?** What are we missing that a proper quant shop would do?

3. **The Fold 2 (2022 bear) problem** — how do professional quant systems handle labeling in bear markets? Is regime-conditional labeling the standard solution or is there something better?

4. **Factor portfolio vs ML:** Given our constraints ($20k capital, Russell 1000, daily bars, no tick data, no short selling), is a rule-based factor portfolio legitimately better than ML here, or are we giving up too early on ML?

5. **Execution model:** Our backtest uses "entry at next open, exit at ATR stop/target/max-hold." Is this realistic for a retail system at $20k? What execution assumptions should we be making?

6. **Intraday at $20k:** Is same-day intraday trading at this capital level even viable? What are the mathematical constraints?

7. **What would you build next?** Given everything above, lay out a specific sequence of experiments (in priority order) that would tell us whether there is a real edge in this system, and if so, how to capture it.

**Please be direct. If the answer is "this system cannot make money at $20k with these constraints," say that. If there's a specific structural flaw that explains all our results, name it. We want to fix real problems, not spend more months tuning around them.**

---

## Technical Notes for Reference

- Data: yfinance daily OHLCV + Alpaca paper brokerage. No tick data, no Level 2. Fundamentals from FMP quarterly (fetched ~4x per year per symbol).
- Compute: Windows 11, 16-core CPU, 32GB RAM. ProcessPoolExecutor for WF, capped at 8 workers (OOM fix). Each 5-fold WF run takes ~10–12 minutes.
- Model serialization: joblib pickle. TSNormalizerState saved separately.
- Walk-forward uses feature cache (pre-computed per-fold feature matrix for all symbols × test days) for ~25x speedup vs live feature engineering.
- All LambdaRank training: LGBMRanker with lambdarank objective, evaluated on NDCG@K.
- All XGBoost training: XGBClassifier with binary:logistic objective.
