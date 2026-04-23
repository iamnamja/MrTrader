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
1. Fetch last `WINDOW_DAYS` (63) of daily bars for all SP-500 symbols
2. Run `FeatureEngineer.engineer_features()` → 140-feature vector per symbol
3. CS-normalize features across batch (removes sector/regime bias)
4. Run `XGBClassifier.predict_proba()` → probability score per symbol
5. Filter: `score >= MIN_CONFIDENCE (0.55)`, take top 10
6. Size position using `_calculate_quantity()`: confidence scalar × base allocation
7. Send proposals to Risk Manager

**Model:** v111 — XGBoost, 140 features, trained on SP-500 5-year history
**Label:** Top 20% cross-sectional Sharpe-adjusted 5-day forward return
**Top SHAP signals:** atr_norm (0.114), volatility (0.064), realized_vol_20d (0.062)
**Note:** Tier 3 Sharpe +0.34 (as of v110). Paper trading gate requires > 0.8.

### Intraday Selection (09:45)
1. Fetch 5-min bars for Russell 1000 symbols (Polygon Parquet cache-first, Alpaca fallback)
2. Run `compute_intraday_features()` → 41-feature vector per symbol
3. Run `PortfolioSelectorModel.predict()` → probability per symbol
4. Filter: `score >= MIN_CONFIDENCE (0.55)`, take top 5
5. Tagged `trade_type="intraday"` → PM will force-close by 15:45

**Model:** v18 — XGBoost + LightGBM soft-vote ensemble, 41 features, 766 symbols
**Label:** ATR-adaptive: target = 1.2× ATR14, stop = 0.6× ATR14 (2:1 R:R)
**Top SHAP signals:** atr_norm, orb_position, bb_position, rel_vol_spy, session_hl_position
**Note:** Tier 3 Sharpe -1.16. Paper trading gate requires > 1.5. Not yet production-ready.

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

### Swing Model (v111) — Updated 2026-04-23
| Field | Value |
|---|---|
| Architecture | XGBoost (400 trees, depth 4, lr=0.03) |
| Universe | SP-500 (~753 symbols used in training) |
| Feature count | 140 (no fundamentals — faster iteration) |
| Forward window | 5 days (Phase 25: was 10 — halved to double training samples) |
| Train samples | ~129k |
| Train period | 5 years rolling |
| Retrain schedule | 17:00 ET daily |
| OOS AUC | 0.640 |
| Tier 3 Sharpe | +0.34 (v110 — gate requires > 0.8) |
| Key phase history | Phase 24a: label bug fix. Phase 24b: regime interactions. Phase 25: 5-day window. Phase 26a: VIX sample weights. |

### Intraday Model (v18) — Updated 2026-04-23
| Field | Value |
|---|---|
| Architecture | XGBoost + LightGBM soft-vote ensemble (50/50) |
| Universe | Russell 1000 (~766 symbols) |
| Feature count | 41 |
| Train period | 2 years of 5-min bars (Polygon cache) |
| Retrain schedule | 17:00 ET daily |
| OOS AUC | ~0.56 |
| Tier 3 Sharpe | -1.16 (gate requires > 1.5) |
| Status | Not production-ready. Improvement work queued after swing gate is met. |

---

## Backtesting Architecture (Three-Tier)

**Files:** `app/backtesting/` — three complementary simulators

| Tier | Class | What It Tests |
|---|---|---|
| Tier 1 | `SwingBacktester` / `IntradayBacktester` | Label-aligned raw signal quality; trade every signal the model produces |
| Tier 2 | `StrategySimulator` | Portfolio-level replay; position sizing, capital allocation, Sharpe/drawdown stats |
| Tier 3 | `AgentSimulator` / `IntradayAgentSimulator` | Runs actual PM + RM + Trader agent code on historical bars; honest end-to-end simulation |

### Tier 3 — AgentSimulator (Swing)

**File:** `app/backtesting/agent_simulator.py`

Daily loop (per business day in [start_date, end_date]):
1. **PM scoring**: batch `FeatureEngineer.engineer_features()` + `model.predict()` + CS-normalize → ranked proposals filtered by `MIN_CONFIDENCE=0.50`, top 10
2. **Trader gate**: checks EMA-200 uptrend (`price > EMA-200`); blocks downtrend entries
3. **RM validation**: `validate_buying_power`, `validate_position_size`, `validate_sector_concentration`, `validate_daily_loss`, `validate_open_positions`, `validate_account_drawdown`, `validate_portfolio_heat`
4. **Exit scan**: `check_exit()` trailing-stop logic + intrabar H/L stop/target override; time exit at `MAX_HOLD_BARS=20`

**Warm-up:** 420 calendar days (~300 business days) before trading start so EMA-200 has history.

**Position sizing:** `size_position()` result capped to RM's `MAX_POSITION_SIZE_PCT` (5%) before RM validation.

**Known gap:** Swing Tier 2 shows higher returns than Tier 3 because Tier 2 replays historical winners while Tier 3 runs real agent logic. Tier 3 is the honest number.

### Tier 3 — IntradayAgentSimulator

**File:** `app/backtesting/intraday_agent_simulator.py`

Operates on 5-min bars. Per trading day:
1. **PM scoring**: `compute_intraday_features()` per symbol → `model.predict()` → top-N filtered by `MIN_CONFIDENCE`
2. **Trader gate**: ORB breakout check + session-time constraint (no entries after 14:30 ET)
3. **RM validation**: same RM rules as swing, applied intraday
4. **Exit scan**: target (+0.5%), stop (-0.3%), or HOLD_BARS=24 time exit

**Script entry point:** `scripts/backtest_ml_models.py --model intraday` runs Tier 1 + Tier 2 + Tier 3 for intraday.

---

## Backtesting Results — Current Models

### Swing v110 (run 2026-04-23) — Best Result to Date

| Tier | Trades | Win Rate | Sharpe | Return | Notes |
|---|---|---|---|---|---|
| Tier 3 (AgentSimulator) | 290 | 40.3% | **+0.34** | +1.9% | First positive Sharpe. 70% stop exits. |

Gate requires Tier 3 Sharpe > 0.8. Currently at +0.34. Need +0.46 more.

### Intraday v18 (run 2026-04-23)

| Tier | Trades | Win Rate | Sharpe | Notes |
|---|---|---|---|---|
| Tier 3 (IntradayAgentSimulator) | 266 | 49% | -1.16 | 58% stop exits. Gate requires > 1.5. |

---

## Known Issues & Status

| Issue | Status |
|---|---|
| Swing 70% stop exit rate | Ongoing. Volatility features dominate SHAP. Model picks volatile stocks that noise-stop. |
| Intraday Sharpe -1.16 | Not yet addressed. Phases 23-26 focused on swing only. |
| AUC drift gate (0.65) fires constantly | Steady-state AUC is 0.63-0.64. Gate is informational only — don't use to block trading decisions. |
| Fundamentals disabled | `--no-fundamentals` used for faster iteration. Re-enable when swing gate is met for potential gain. |

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
