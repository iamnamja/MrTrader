# MrTrader — System Behavior Reference

Documents how the Portfolio Manager, Risk Manager, and Trader actually behave at runtime. Updated 2026-04-22.

---

## Architecture Overview

```
[Portfolio Manager] → proposals → [Risk Manager] → approved → [Trader] → [Alpaca API]
        ↑                                                           |
        └─────────── re-eval requests ──────────────────────────────┘
```

All agents communicate via an in-process async message queue. No HTTP between agents.

---

## Portfolio Manager (PM)

**File:** `app/agents/portfolio_manager.py`

### Daily Schedule

| Time (ET) | Action |
|---|---|
| 09:00 | Pre-market intelligence: earnings calendar, 8-K filings, news sentiment |
| 09:25 | Swing model scoring: daily bars → rank SP-500 → propose top 10 |
| 09:45 | Intraday model scoring: 5-min bars → rank Russell 1000 → propose top 5 |
| 09:30–16:00 | Position review every 30 min: rescore open positions, request exits/extensions |
| 09:30–16:00 | Re-eval drain: handles Trader's re-score requests (e.g. large adverse move) |
| Weekly | Walk-forward backtest + model retrain if drift detected |

### Swing Selection (09:25)
1. Fetch last `WINDOW_DAYS` of daily bars for all SP-500 symbols
2. Run `FeatureEngineer.engineer_features()` → 126-feature vector per symbol
3. CS-normalize features across batch (removes sector/regime bias)
4. Run `LambdaRankModel.predict()` on full batch → ranking score [0, 1] per symbol
5. Filter: `score >= MIN_CONFIDENCE (0.55)`, take top 10
6. Size position using `_calculate_quantity()`: confidence scalar × base allocation
7. Send proposals to Risk Manager

**Model:** v94 — XGBoost LambdaRank, 126 features, trained on SP-500 5-year history
**Label:** Top 20% cross-sectional Sharpe-adjusted 10-day forward return
**Top SHAP signals:** revenue_growth (0.235), sector_momentum (0.178), operating_leverage (0.139)

### Intraday Selection (09:45)
1. Fetch 5-min bars for Russell 1000 symbols (cache-first, Alpaca fallback)
2. Run `compute_intraday_features()` → 40-feature vector per symbol
3. Run `PortfolioSelectorModel.predict()` → probability per symbol
4. Filter: `score >= MIN_CONFIDENCE (0.55)`, take top 5
5. Tagged `trade_type="intraday"` → PM will force-close by 15:45

**Model:** v17 — XGBoost + LightGBM ensemble, 40 features, 766 symbols
**Label:** Cross-sectional Sharpe-adjusted 24-bar (2h) best return (top 20%)
**Top SHAP signals:** atr_norm, orb_position, bb_position, rel_vol_spy, session_hl_position

### Position Review (every 30 min)
- Rescore all open swing positions using same model
- If score drops below MIN_CONFIDENCE: send EXIT request to Trader
- If price near target: optionally send EXTEND_TARGET (extend by 1 ATR)
- If earnings event coming: send EXIT or HOLD based on risk policy

### Pre-Market Intelligence (09:00)
- Checks earnings calendar: blocks intraday trading on earnings day stocks
- Fetches 8-K filings: downweights stocks with recent material filings
- News sentiment: adjusts sector weights going into open
- If high VIX detected → blocks all new intraday entries for session

---

## Risk Manager (RM)

**File:** `app/agents/risk_manager.py`  
**Config:** `app/agents/risk_rules.py → RiskLimits`

### Validation Sequence

Every proposal from PM goes through these checks **in order** — first failure rejects:

| # | Rule | Default Limit | Configurable |
|---|---|---|---|
| 1 | Buying power | Must have cash | Yes |
| 2 | Position size | Max 5% of account per trade | Yes |
| 3 | Daily loss | Max 2% account loss today | Yes |
| 4 | Max open positions | 5 simultaneous positions | Yes |
| 5 | Sector concentration | Max 20% account in one sector | Yes |
| 6 | Portfolio beta | Max 1.30 vs SPY | Yes |
| 7 | Factor concentration | Max 60% in same factor | Yes |
| 8 | Return correlation | Max 0.75 correlation with existing positions | Yes |
| 9 | Bid-ask spread | Max 0.15% spread | Yes |
| 10 | ADTV liquidity | Max 1% of 20-day avg daily volume | Yes |

All limits stored in DB via `risk.*` config keys; RM reloads from DB on each startup and when receiving a config update message.

### Position Sizing
- Base: PM calculates shares based on `confidence_scalar × budget_per_slot`
- `confidence_scalar(prob)`: linear 0.5× at 0.55 → 2.0× at 1.0
- Swing budget: 70% of account equity across all swing slots
- Intraday budget: 30% of account equity across all intraday slots
- RM re-validates the final dollar cost before approving

### Approved Flow
- Approved → publishes to `trader_entry_queue`
- Rejected → logs rejection reason, notifies PM for re-queuing at next cycle

---

## Trader

**File:** `app/agents/trader.py`

### Entry Execution
1. Receives approved proposal from RM
2. Checks Alpaca market state (open/halted)
3. For swing: places **limit order** at midpoint (bid + half spread)
4. For intraday: places **market order** at open (speed priority)
5. Polls for fill status; if partially filled after 30s → cancels remainder
6. On fill: records entry to DB, begins monitoring in `_open_positions` dict

### Exit Logic — Swing
Each 30-second scan checks open swing positions:

| Condition | Action |
|---|---|
| Price ≤ stop_price | EXIT — stop hit |
| Price ≥ target_price | EXIT — target hit (full or partial) |
| Held ≥ MAX_HOLD_DAYS | EXIT — time stop |
| PM sends EXIT request | EXIT — model signal degraded |
| PM sends EXTEND_TARGET | Move target up by 1 ATR |
| Price down >3% from entry | Tighten stop to near-breakeven |
| Regime shifts to HIGH | Tighten all swing stops to 1×ATR from current |

**Stop calculation:** `entry × (1 - STOP_LOSS_BASE_PCT)` scaled by `ATR / NORMAL_VOLATILITY_ATR_RATIO` — wider stops in high-vol environments.

### Exit Logic — Intraday
| Condition | Action |
|---|---|
| Price hits target (+0.5%) | EXIT — target |
| Price hits stop (-0.3%) | EXIT — stop |
| 15:45 ET | Force-close all intraday positions |
| Pre-market intel blocked | Never entered |

### Slippage Tracking
- Records `intended_price` (midpoint at signal time) vs `filled_price`
- Slippage in basis points logged to `trades` table
- Available via API: `GET /api/v1/performance/slippage`

---

## ML Models — Current State

### Swing Model (v94)
| Field | Value |
|---|---|
| Architecture | LightGBM LambdaRank |
| Universe | SP-500 (~446 symbols) |
| Feature count | 126 |
| Train period | 5 years rolling |
| Retrain schedule | Weekly (Sunday) |
| OOS AUC | 0.5682 |
| Backtest Sharpe | 1.87 (2yr, SP-100 sample) |
| Key improvement | CS normalization dethroned sector_momentum (0.90→0.18 SHAP) |

### Intraday Model (v17)
| Field | Value |
|---|---|
| Architecture | XGBoost + LightGBM ensemble |
| Universe | Russell 1000 (~766 symbols) |
| Feature count | 40 |
| Train period | 730 days rolling |
| Retrain schedule | Daily (after market) |
| OOS AUC | 0.5646 |
| Backtest status | Backtest label mismatch — see Known Issues |
| Key improvement | Session-time features (minutes_since_open, is_open/close_session) |

---

## Known Issues & Planned Fixes

| Issue | Impact | Fix |
|---|---|---|
| Intraday backtest mismatch | Backtester uses target/stop exits; model trained on 2h Sharpe rank | Option A: fix backtester to use 2h time exit; Option B: retrain with target/stop outcome labels |
| AUC drift gate too tight | 0.65 gate fires on every retrain; realistic steady-state is 0.56–0.58 | Lower gate to 0.54 |
| Swing model misses sector_momentum_5d | v94 trained before Iter 4 features were merged; 2 features missing from model | Retrain to get 128-feature model |
| Intraday min_confidence (0.55) too high | Almost no signals pass in backtest | Tune threshold post-retrain or lower to 0.50 |

---

## Configuration Reference

All tunable parameters live in the DB `agent_config` table. Key entries:

| Key | Default | Effect |
|---|---|---|
| `pm.min_confidence` | 0.55 | Minimum model score to propose a trade |
| `pm.top_n_stocks` | 10 | Max swing proposals per cycle |
| `risk.max_position_size_pct` | 0.05 | Max single position size |
| `risk.max_daily_loss_pct` | 0.02 | Daily loss kill-switch |
| `risk.max_open_positions` | 5 | Concurrent position limit |
| `risk.max_correlation` | 0.75 | Max pairwise return correlation |
| `risk.max_portfolio_beta` | 1.30 | Portfolio beta cap vs SPY |
| `risk.max_spread_pct` | 0.0015 | Max bid-ask spread pre-trade |
| `risk.max_adtv_pct` | 0.01 | Max size vs 20-day avg volume |
