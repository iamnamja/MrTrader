# MrTrader Walk-Forward Validation — Deep-Dive LLM Review Prompt

> **Instructions for the LLM:** You are a quantitative finance expert with deep experience in
> institutional-grade systematic trading, machine learning for finance, and production backtesting
> infrastructure. You have worked at or consulted for top-tier firms (Two Sigma, AQR, Citadel,
> Renaissance Technologies, Man AHL). Be **BRUTALLY HONEST**. Do not flatter. Identify every flaw,
> gap, and risk, no matter how uncomfortable. Think like a head of quant research reviewing a system
> before allocating $100M.

---

## System Overview

This is **MrTrader**: a systematic equity trading system targeting ~$20k–$500k retail/semi-institutional
capital. The system is a full-stack ML-to-execution pipeline:

```
NIS (News Intelligence System)
  ↓ sentiment signals
PM (Portfolio Manager agent)
  ↓ ranked proposals
RM (Risk Manager agent) → position sizing, daily loss limits
  ↓ approved orders
Trader agent → Alpaca brokerage execution (MOO/MOC orders)
  ↓
Walk-Forward Validation (WF) — the historical trust layer
```

The WF replays this **exact agent stack** on historical data, day by day, to simulate what the live
system would have done. The goal is not just a backtest but a faithful simulation of the production
code path.

---

## ML Model: v215 / v216 (in retraining)

| Parameter | v215 | v216 (in progress) |
|-----------|------|--------------------|
| Model type | XGBoost LambdaRank | XGBoost LambdaRank |
| Universe | Russell 1000 (point-in-time) | Russell 1000 (point-in-time) |
| Training window | 1 year rolling (252 days) | 1 year rolling |
| Label scheme | Cross-sectional 5-day return rank | Cross-sectional **20-day** return rank |
| Feature count | 14 (PHASE_C list) | 19 (PHASE_C_V2 — adds sector-neutral momentum) |
| Step frequency | STEP_DAYS=5 | STEP_DAYS=20 |
| ATR stop mult | 0.5× (WF default) | **1.5×** (WF default) |
| ATR target mult | 1.5× (WF default) | **3.0×** (WF default) |

**Key features (v215):**
`momentum_5d`, `momentum_20d`, `momentum_60d`, `rsi_14`, `atr_pct_14`, `volume_ratio_20d`,
`adv_20d`, `market_cap_log`, `pb_ratio`, `ps_ratio`, `macd_signal`, `bb_width_20`,
`close_vs_52w_high`, `sector_encoded`

**Additional features in v216:**
`momentum_20d_sector_neutral`, `momentum_60d_sector_neutral`
*(removes concentrated sector-bet bias in LambdaRank)*

**Training label:** Position in cross-sectional return distribution for FORWARD_DAYS ahead.
v215=5 days, v216=20 days. No transaction costs in training labels.

**Critical note:** Entry in training = `close[t]`. Entry in live/WF = `open[t+1]`. **MISMATCH** (see C1).
ATR stops in training: **None** — LambdaRank labels are pure return-rank, no path constraints.

---

## Simulation Engine (agent_simulator.py, ~1400 lines)

### Simulation loop (per trading day):

**1. MORNING PREP**
- Load daily OHLCV bars for all Russell 1000 symbols
- Check positions vs prior day's EOD prices
- RM evaluates gates:
  - `MAX_OPEN_POSITIONS = 5` (concurrent positions)
  - `MAX_POSITION_SIZE_PCT = 5%` of portfolio
  - `MAX_SECTOR_CONCENTRATION_PCT = 20%`
  - `MAX_DAILY_LOSS_PCT = 2%`
  - `MAX_ACCOUNT_DRAWDOWN_PCT = 5%`
  - `MAX_PORTFOLIO_HEAT_PCT = 6%`

**2. ENTRY DECISIONS (MOO simulation)**
- Entry price = `open[t+1] × (1 ± 0.03%)` ← 3bps slippage (recently added)
- PM scores candidates via `model.predict_proba()` → top N proposals
- Proposal pool = `max(top_n × 5, 50)` symbols scored per day
- `top_n` = 10 long + 5 short candidates evaluated
- Position sizing: `fixed_fraction(account_equity, max_position_pct=0.05)`
- Direction: long if `rank_score > LONG_THRESHOLD`, short if `rank_score < SHORT_THRESHOLD`

**3. DAILY P&L AND STOP/TARGET CHECKING**

| Parameter | v215 default | v216 default |
|-----------|-------------|-------------|
| ATR_STOP_MULT | 0.5× ATR14 (≈0.75–1.5%) | **1.5× ATR14 (≈2.25–4.5%)** |
| ATR_TARGET_MULT | 1.5× ATR14 | **3.0× ATR14** |
| Stop slippage | 0.05% | 0.05% |
| Stop clip range | [0.75%, 4.0%] | [0.75%, 4.0%] |
| Target check | EOD only | EOD only |

- `max_hold_bars = 40` bars (≈2 trading weeks at daily bars)
- Stop trigger timing: **EOD bar close only** (NOT intrabar)
- Gap-through handling: if open gaps through stop, fill at `open + slippage`

**4. EXIT**
- Stop: fill at `stop_price × (1 + stop_slippage)` for longs
- Target: fill at target_price (EOD check)
- max_hold: force close at EOD on bar 40
- Transaction cost: **0.15% round-trip** (flat, no spread model)

**5. PERFORMANCE METRICS (per fold)**
- Sharpe ratio (annualized, 252 trading days)
- Max drawdown, Win rate, Profit factor, Calmar ratio, K-ratio
- **Deflated Sharpe Ratio** (Bailey & López de Prado, corrects for multiple hypothesis testing)

---

## Walk-Forward Architecture (walkforward_tier3.py, ~1487 lines)

**Configuration:**
- Folds: 5 | Total history: 6 years | Structure: **Expanding window**
- Purge gap: 15 days | Embargo: 15 days

**Fold boundaries (approximate):**
| Fold | Train | Test |
|------|-------|------|
| 1 | 2018-01 → 2019-07 | 2019-08 → 2020-01 |
| 2 | 2018-01 → 2020-05 | 2020-06 → 2021-01 |
| 3 | 2018-01 → 2021-05 | 2021-06 → 2022-01 |
| 4 | 2018-01 → 2022-05 | 2022-06 → 2023-01 |
| 5 | 2018-01 → 2023-05 | 2023-06 → 2024-01 |

**Gate thresholds (ALL must pass for live deployment):**
- `avg_sharpe > 0.8`
- `no fold sharpe < -0.3` (no catastrophic fold)
- `dsr_p > 0.95` (Deflated Sharpe Ratio p-value)
- `avg_profit_factor > 1.0`
- `avg_calmar > 0.3`

**CPCV (Combinatorial Purged Cross-Validation):**
- k=6, paths=2
- Generates multiple non-overlapping test paths for overfitting detection
- Compares CPCV Sharpe distribution vs standard WF Sharpe

---

## Intraday Subsystem (intraday_agent_simulator.py)

| Parameter | Value |
|-----------|-------|
| Bar resolution | 5-minute bars (Polygon.io) |
| HOLD_BARS | 24 (≈2 hours) |
| TARGET_PCT | 0.5% |
| STOP_PCT | 0.3% |
| MAX_ENTRY_BAR | 60 (no entries after ~14:30 ET) |
| EOD force-close | Bar 78 (16:00 ET) |
| Transaction cost | 0.15% flat (no spread model) |

---

## Known Issues and Gaps

### 🔴 CRITICAL — Directly Affects P&L

**C1. TRAIN/BACKTEST ENTRY PRICE MISMATCH**
Training labels computed using `close[t]` as entry. Simulation enters at `open[t+1]`.
For a 5-day label, this means the model was trained on "buy at close and hold 5 days" but tested
as "buy next open and hold 40 bars." The first-day gap is absorbed into P&L but not into the label —
can be ±0.3–1% systematic bias.
*Status: Will be fixed in v216 by using open[t+1] in label construction. Not done yet.*

**C2. LABEL/HOLD HORIZON MISMATCH (CONFIRMED ROOT CAUSE OF v215 FAILURE)**
v215: `FORWARD_DAYS=5`, `max_hold_bars=40`. Model predicts 5-day rank but P&L realized over 40 bars.
A stock "ranked top 20% for 5 days" is not necessarily top 20% for 40 days — it may mean-revert.
v215 WF: avg Sharpe -0.459, win rate 31% (gate fails badly).
v216 fix: `LABEL_HORIZON_DAYS=20` (partial alignment; not full 40).

**C3. ATR STOPS BELOW DAILY NOISE (v215)**
0.5×ATR14 ≈ 0.75–1.5% for Russell 1000 large caps. Average daily range ≈ 1.5–2%.
v215 WF showed `stop_exit_rate > 60%` — most exits were noise stops, not signal.
v216 fix: 1.5×ATR (≈2.25–4.5%) — outside daily noise range.

### 🟠 HIGH PRIORITY — Significant Modeling Risk

**H1. FLAT BORROW RATE FOR SHORTS**
All short positions assume 5% annual borrow cost regardless of symbol. Easy-to-borrow large caps
≈0.3–1%, hard-to-borrow small caps can be 20–100%+. Assumption undocumented and unvalidated
against actual Alpaca borrow rates.

**H2. EMA BURN-IN PERIOD NOT ENFORCED**
EMA-200 requires ≈400 bars to stabilize. No warm-up exclusion enforced. Affects features derived
from EMAs (bb_width_20, momentum signals).

**H3. ANNUALIZATION INCONSISTENCY**
Some metrics use 365 calendar days, others 252 trading days. Sharpe and DSR use 252 (correct).
Some log output uses 365.

**H4. NO BID-ASK SPREAD MODEL**
Transaction costs modeled as flat 0.15% round-trip. Reality for Russell 1000: spread ≈0.05–0.2%.
Market impact for >$1M positions not modeled. Fine at $20k; matters at $500k+.

**H5. STOP TRIGGER AT EOD ONLY**
Intrabar stop triggers not simulated for daily bars. If a stock hits stop intraday and recovers,
stop is missed. This **overestimates returns** in volatile markets / bear regimes.

**H6. VIX REGIME: STATIC THRESHOLDS**
VIX > 20 = high regime, VIX < 20 = low regime. Hard-coded thresholds, not fitted from data.
No vol-of-vol regime detection. No cross-asset regime signals.

**H7. SINGLE-THREADED SYMBOL LOOP**
Feature cache exists (parallel pre-compute), but main simulation loop is single-threaded.
1000 symbols × 1500 days × 5 folds = 7.5M iterations. Large universe = slow WF (~45 min for 5 folds).

### 🟡 MEDIUM PRIORITY

**M1. POSITION SIZING IS FIXED FRACTION (5% flat)**
No Kelly criterion, no volatility scaling, no liquidity adjustment.

**M2. NO EXECUTION REALISM FOR SIZE**
All trades assumed fillable at open regardless of order size vs ADV.

**M3. PORTFOLIO HEAT ASSUMES UNCORRELATED POSITIONS**
Heat = sum of (position_size × stop_distance). No correlation between positions modeled.

**M4. NO REGIME-CONDITIONAL POSITION SIZING**
In high-VIX regimes, system still runs 5 max positions at 5% each. No leverage reduction in stress.

**M5. SECTOR CONCENTRATION LIMIT IS POST-HOC**
RM rejects candidates randomly within a sector that's at limit, rather than rejecting worst-ranked.

**M6. SHORT POSITION MECHANICS**
No uptick rule simulation. No forced buy-in simulation. Short squeeze risk not modeled.

### 🟢 LOW PRIORITY / NICE-TO-HAVE

**L1. NO DIVIDENDS** — Long positions don't capture dividends (≈1.5–2%/year for Russell 1000)

**L2. NO CORPORATE ACTIONS** — Splits/spin-offs handled by adjusted close but not position adjustments

**L3. CROSS-FOLD MODEL STABILITY NOT MEASURED** — No feature importance stability check across folds

---

## Actual WF Results (v215)

| Run | Code State | Avg Sharpe | Win Rate | Assessment |
|-----|-----------|-----------|---------|-----------|
| Baseline | Pre-fix (wide stops, max_hold=160 bars) | **+0.036** | 43% | **MISLEADING** — 32-week holds captured market beta |
| Post-realign | ATR stops + max_hold=40 bars | **-0.459** | 31% | **HONEST** — weak signal exposed, label/hold mismatch |
| Post-all-fixes | +entry slippage, pool=200 | **-0.571** | 31% | **WORSE** — pool widening pulled in low-conviction tail |

**Current:** v216 retrain complete (AUC 0.554). WF not yet run on v216.

---

## What I Need From You

### 1. Architecture Critique
Be brutally honest about this simulation architecture. What would a Two Sigma or AQR quant think?
What critical flaws would they find immediately? What does institutional-grade look like vs this?

### 2. Trust Calibration
Given the known gaps (C1–H7), how much should we trust the WF results? Which gaps materially affect
reliability? Which are truly minor? What's the minimum set of fixes needed before results are "trusted"?

### 3. Hedge-Fund-Level Enhancements
What does best-in-class WF infrastructure look like at Two Sigma, AQR, or Man AHL? What practices
are we missing entirely? Think outside the box — what would you add if budget/time were unconstrained?

### 4. Multi-Strategy Flexibility
The system currently runs swing (daily, L/S, 40-bar hold). We want WF to also support:
- Long-only (no shorts)
- Short-only
- Long/Short market-neutral
- Intraday (5-min bars, 2-hour hold)
- Potentially: pairs trading, stat-arb, options

What architectural changes are needed? What would a clean, extensible WF framework look like that
supports all these without code duplication?

### 5. Agent Workflow Integration
The live system uses PM → RM → Trader agents. The WF must simulate exactly what the live system does.
What are the failure modes of "simulate the agent stack"? What's commonly missed? How do leading quant
shops handle the sim-to-live transition?

### 6. Data Requirements (ranked by importance)

**Current data:** daily OHLCV (yfinance), 5-min bars (Polygon.io), fundamental ratios (FMP),
VIX (daily), Russell 1000 membership (PIT), basic sector classification.

Please rank these additional datasets by expected improvement to WF reliability and model alpha:

| Dataset | Notes |
|---------|-------|
| Intraday bid-ask spreads | Execution cost realism |
| Level 2 order book | Market impact modeling |
| Short interest / borrow rates (daily) | Fix H1 |
| Options implied volatility (term structure) | Regime + vol signal |
| 13-F institutional holdings (quarterly) | Flow/crowding signal |
| Analyst estimates / earnings surprises | Event alpha |
| Alternative data (satellite, credit card, web traffic) | Non-correlated alpha |
| News sentiment | Beyond current NIS |
| ETF flow data | Crowding + momentum |
| Factor exposures (Barra, Axioma) | Risk decomposition |
| Corporate event calendar (earnings, dividends, splits) | Event blackout windows |
| Economic calendar (FOMC, CPI, NFP) | Macro regime |

### 7. Label Design
LambdaRank on cross-sectional return rank — is this the right choice? What are the failure modes
of learning to rank vs direct return prediction? What do top quant shops use for daily equity alpha
signals? How should the label be constructed to minimize look-ahead while maximizing alignment with
how we actually make money in simulation?

### 8. Specific Open Questions

**a)** 20-day labels vs 40-bar hold: is 20 days the right compromise, or should we use 40-day labels?
What's the right way to think about train/test horizon alignment?

**b)** LambdaRank requires the model sees a full cross-sectional batch each training day. But in live
we only score at rebalance points. Is there a train/live distribution mismatch?

**c)** ATR stops in WF but not in training labels. Is this inherently wrong, or acceptable if the stop
multiplier is wide enough? What should the theoretical relationship be between stop width and label horizon?

**d)** With `MAX_OPEN_POSITIONS=5`, the portfolio is extremely concentrated. At institutional scale
this would be reckless. Is this a feature (high conviction) or a bug (idiosyncratic risk swamps alpha)?
What does the literature say about optimal portfolio concentration for ML-based equity strategies?

**e)** The WF uses an expanding window (all train data used each fold). Would rolling window be better?
What's the right train window length for a LambdaRank cross-sectional model on Russell 1000?

---

## Final Ask

Give me a prioritized roadmap: if I were a quant PM allocating **6 months of engineering time** to
make this WF hedge-fund level, what would the milestones look like? What must be done vs nice-to-have
vs out-of-scope for a system of this size ($20k–$500k capital)?

Don't soften the feedback. This system's live performance depends on the WF being trustworthy.
If you see something wrong, say so directly.
