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

### PEAD Selector (`pm.swing_selector="pead"`) — wired 2026-06-02, NOT activated
When the swing selector is set to `"pead"`, `_analyze_swing_premarket` routes to
`_analyze_swing_pead`, which faithfully runs the **validated +0.546 long-only PEAD config**
(`scripts/run_pead_cpcv.py`) as a *risk-managed live variant*:

1. Fetch 60 daily bars for the universe (`get_bars_batch`).
2. **VIX-series injection (FIX A):** fetch a **daily VIX close series** (`_fetch_vix_series`, ≥60
   bars, lowercase `close`, DatetimeIndex) and inject it under `symbols_data["^VIX"]`. The
   `PEADScorer._vix_today` crisis block (VIX>30 → return `[]`) only fires with this series — a
   prior scalar/no-key bug meant the block, which is credited with the entire edge (P5
   −0.29→+0.01), never fired live.
3. **Validated config pinned (FIX C):** construct `PEADScorer` with **every** validated parameter,
   read from `pm.pead_*` agent_config keys (defaults = validated values): `long_threshold=0.05`,
   `short_threshold=-0.05`, `max_days_after=3`, `long_short=False`, `vix_block_all=30`,
   `vix_block_short=100`, `vix_conf_ref=100`, `max_announce_day_move=1.0` (priced-in filter OFF —
   the backtest KEEPS the large-gap high-drift names), `require_positive_revision=False`.
4. Score; long-only filter applied (shorts require `pm.pead_enable_shorts=true` + margin account).
5. **Hold = 40 trading days (FIX B):** proposals annotated `max_hold_days=40` (the hold-5/hold-15
   variants were killed — drift wants a longer hold). The Trader enforces this per-position.
6. Proposals built via `_build_directional_proposals` using the **live `_calculate_quantity`**
   sizing (NOT 5% equal-weight), then pass through the **kept risk overlays**: regime sizing
   multiplier, NIS news sizing, opportunity-score gate (`_send_swing_proposals`), macro-calendar
   block, and the RM 10-rule chain. These overlays cause expected tracking error vs the clean
   backtest.
7. **Marketable entries (Trader, scoped to `selector=="pead"`):** PEAD entries route as a
   **marketable limit** that crosses the spread (long = ask+10bps, short = bid−10bps) instead of
   the standard below-ask swing limit, so fills track the backtest's next-open assumption and
   avoid the adverse-selection trap (below-ask limits fill only the names that DON'T run).
   Swing/intraday entry routing is byte-identical.

**Observability (PEAD-only):** `app/live_trading/pead_tracker.py` (sqlite `data/pead_tracking.db`)
records a daily row — date, #signals, #entered, #filled, fill-rate, gross deployed, realized +
unrealized + cumulative P&L, VIX level, `vix_block_fired`, and per-overlay suppression counts
(opportunity / macro / RM) for tracking-error attribution. The row is written in **two stages**:

- **Signals stage (premarket, `_analyze_swing_pead`):** writes `n_signals`, `vix_level`,
  `vix_block_fired`, `extra`. Leaves the P&L/fill fields untouched (NULL).
- **EOD stage (16:30–16:59 ET, `_run_eod_jobs` → `_compute_pead_eod_stats`):** upserts the same
  day's row with the PEAD book's REAL numbers after close — `gross_deployed` (cost basis of open
  PEAD positions), `realized_pnl` (today's CLOSED PEAD trades), `unrealized_pnl` (mark-to-market of
  open PEAD positions via Alpaca `get_positions`), `n_entered`/`n_filled`. `record_daily` is a
  **partial upsert** (`ON CONFLICT(trade_date) DO UPDATE … COALESCE(excluded.X, X)`): each stage
  sets only its own fields and preserves the other's, so the EOD P&L update does not clobber
  `n_signals`/`vix`. Without this EOD stage the weekly Sharpe was "n/a" forever (gross stayed 0).

**PEAD trade attribution — `Trade.selector` column:** PEAD trades are identified via a new
`Trade.selector` column (`VARCHAR(32)`, default `""`). The Trader populates it from
`proposal["selector"]` when writing the Trade (`_write_pending_fill` + the reconciler fallback
path). A startup migration in `session.py::_migrate_columns` adds the column; pre-existing rows
default to `""` (non-PEAD). `_compute_pead_eod_stats` filters on `Trade.selector == "pead"`.

**`signal_type` vs `selector` (do not confuse):** `Trade.signal_type` records the *signal
mechanism* that `generate_signal()` classified — `EMA_CROSSOVER | RSI_DIP | ML_RANK | NONE`. PEAD,
factor-portfolio, and other ML-driven entries have no EMA/RSI crossover, so `generate_signal`
returns `NONE` and the Trader records `signal_type="ML_RANK"`. This is correct and intentional: the
`[ML_RANK]` tag means "ML-ranked signal", **not** "swing-ML selector". Strategy-source attribution
(PEAD live-vs-backtest, etc.) lives **only** in the dedicated `Trade.selector` column — `signal_type`
is deliberately *not* overloaded with the selector because `app/analytics/signal_attribution.py`
buckets closed trades by `signal_type` and depends on the EMA/RSI/ML_RANK taxonomy. So a PEAD
proposal legitimately carries `selector="pead"` + `signal_type="ML_RANK"` simultaneously.

**Weekly rollup (Friday EOD):** at the END of `_run_eod_jobs`, Friday-only (ET weekday == 4) and
AFTER the EOD daily-row update above, the PM calls `pead_tracker.weekly_rollup()`. Running it inside
the same coroutine, after the day's final row is written, gives **guaranteed ordering** (no
cross-process race). It computes the trailing-7-day realized Sharpe of the PEAD book vs the +0.546
backtest expectation and emails it via `notifier.enqueue("pead_weekly", …)` (dedup_key
`pead_weekly_{week_ending}`). **Vacuous-email guard:** if fewer than `min_days` (=3) daily rows in
the window have `gross_deployed > 0`, the email is SKIPPED (payload returned with
`skipped="insufficient data"`). Both the EOD update and the rollup are wrapped in try/except and are
non-fatal to the PM loop. Kill-switch / circuit-breaker reuse the existing swing path.

**Activation (separate deliberate step):** set agent config `pm.swing_selector="pead"` (DB key
`agent.pm.swing_selector`). The PM reads `swing_selector` at decision time, so **no uvicorn restart
is required** for the config flip; restart only if the new module files were added while the agent
process was already running.

> **Restart required for the EOD P&L update + Friday rollup:** the two-stage daily row and the
> Friday weekly rollup are new code in `_run_eod_jobs`. A **uvicorn restart is required** to load
> them into the running PM process (the EOD window code only takes effect after restart).

### Intraday Selection (09:45)
1. Fetch 5-min bars for Russell 1000 symbols (Polygon Parquet cache-first, Alpaca fallback)
2. Run `compute_intraday_features()` → 50-feature vector per symbol (model selects its 42 by name)
3. Cross-sectional normalize features across today's candidate universe (Russell 1000, ~720 symbols)
4. Run `PortfolioSelectorModel.predict()` → probability per symbol
5. Filter: `score >= MIN_CONFIDENCE (0.50)`, take top 5
6. PM abstention gate: skip all entries if VIX ≥ 25 OR SPY < 20-day SMA
7. Tagged `trade_type="intraday"` → PM will force-close by 15:45

*(MetaLabelModel dropped in Phase 47 — confirmed +0.000 Sharpe contribution, R2=0.001)*

**Model:** v23 — XGBClassifier, 42 features, Russell 1000 (~720 symbols with Polygon cache)
**Label:** path_quality regression → cross-sectional top-20% per day = label 1
  - path_quality = 1.0×upside_capture − 0.50×stop_pressure + 0.25×close_strength  *(Phase 47-3: coeff reduced from −1.25)*
  - Stop = 0.4× prior-day range, target = 0.8× prior-day range *(Phase 47-3: compressed from 0.6/1.2)*
**Top features:** spy_session_return, is_close_session, spy_rsi_14, atr_norm, whale_candle
**Tier 3 Sharpe:** +1.275 avg (3-fold walk-forward, gate > 0.80 ✅). Paper trading candidate.

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

### Intraday Model (v23) — Updated 2026-04-26 ✅ GATE PASSED
| Field | Value |
|---|---|
| Architecture | XGBClassifier + PM abstention gate (MetaLabelModel dropped — +0.000 contribution) |
| Universe | Russell 1000 (~720 symbols with Polygon 5-min cache) |
| Feature count | 50 (42 base + 8 Phase 47-5 quality features) |
| Train period | 2 years of 5-min bars (Polygon cache) |
| Retrain schedule | 17:00 ET daily (`scripts/retrain_intraday.py`) |
| OOS AUC | 0.5995 |
| Tier 3 Sharpe | **+1.275 avg** (gate > 0.80 ✅, min fold +0.79 > −0.30 ✅) |
| Label | path_quality = 1.0×upside_capture − **0.50**×stop_pressure + 0.25×close_strength |
| Exit config | Stop = **0.4**× prior-day range, Target = **0.8**× prior-day range (~2:1 R:R) |
| Status | **Gate passed. Paper trading candidate.** |

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
1. **PM abstention gate**: skip all entries if VIX ≥ 25 OR SPY < 20-day SMA
2. **PM scoring**: `compute_intraday_features()` per symbol → CS-normalize → `model.predict()` → top-N filtered by `MIN_CONFIDENCE=0.50`
3. **Meta-model gate**: skip if MetaLabelModel predicts E[pnl] ≤ 0 (currently weak — R2=0.001)
4. **RM validation**: same RM rules as swing, applied intraday
5. **Exit scan**: ATR-adaptive stop (0.6× prior-day range), target (1.2× prior-day range), or HOLD_BARS=24 time exit

**Note:** Hard ORB breakout gate was removed in Phase 46 (was starving trades in range-bound markets). ORB features remain as model inputs.

**Script entry point:** `scripts/backtest_ml_models.py --model intraday` runs Tier 1 + Tier 2 + Tier 3 for intraday.

---

## Backtesting Results — Current Models

### Swing v110 (run 2026-04-23) — Best Result to Date

| Tier | Trades | Win Rate | Sharpe | Return | Notes |
|---|---|---|---|---|---|
| Tier 3 (AgentSimulator) | 290 | 40.3% | **+0.34** | +1.9% | First positive Sharpe. 70% stop exits. |

Gate requires Tier 3 Sharpe > 0.8. Currently at +0.34. Need +0.46 more.

### Intraday v23 (run 2026-04-26) — ✅ GATE PASSED

| Fold | Trades | Win Rate | Sharpe | Notes |
|---|---|---|---|---|
| Fold 1 (Oct '24–Apr '25) | 150 | 44.0% | +0.79 | |
| Fold 2 (Apr '25–Oct '25) | 226 | 43.8% | +1.30 | |
| Fold 3 (Oct '25–Apr '26) | 154 | 50.6% | +1.73 | |
| **Avg** | **530** | **46.2%** | **+1.275** | **Gate > 0.80 ✅** |

Phase 47-3 changes: target 1.2x→0.8x, stop 0.6x→0.4x prior-day range; stop_pressure coeff −1.25→−0.50. Meta-model dropped (+0.000 contribution).

v22 stack: XGBClassifier + MetaLabelModel v1 + PM abstention gate (VIX≥25 OR SPY<MA20). All folds positive for first time. Phase 47 targeting +0.80.

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

---

## Logging & Ops

The running server writes to `logs/mrtrader_<date>.log` via `_DailyFileHandler`
(`app/main.py`), installed by `_configure_logging()` in the FastAPI lifespan.

**Test-log isolation (2026-06-02):** the pytest suite invokes the lifespan (via
`TestClient(app)`) and instantiates agents directly, which previously wrote
hundreds of fake "Orchestrator started" banners and test-only errors into the
**live** `mrtrader_<date>.log`. `_DailyFileHandler` now detects pytest
(`PYTEST_CURRENT_TEST` in env, or `pytest` in `sys.modules`) and switches the
file prefix to `test_mrtrader_<date>.log`. Production behavior is unchanged — the
env var is never set outside pytest, and the live server keeps writing
`mrtrader_<date>.log`. Covered by `tests/test_log_isolation.py`.

---

## Trend sleeve (TSMOM) — rebalance cadence & why Monday

The trend sleeve (`app/live_trading/trend_sleeve.py`) rebalances the 10-ETF basket
(SPY,QQQ,IWM,EFA,EEM,TLT,IEF,GLD,DBC,UUP) **weekly, on Monday 09:45 ET** to inverse-vol
long-flat TSMOM target weights. The orchestrator registers the job **daily** (cron Mon–Fri
09:45) with an in-handler `weekday == pm.trend_rebalance_weekday` (default 0 = Monday) guard,
so the rebalance day is live-tunable (`agent_config`) without redeploy, and a market-open
clock guard fails closed on holidays. `misfire_grace_time=1800` (vs the 60s default for daily
jobs): a *weekly* job dropped by a >60s-late fire would skip the ENTIRE week, so it tolerates
a 30-min-late fire (TSMOM is slow → benign). Orders are placed directly via Alpaca, tagged
`selector/trade_type="trend"`, committed **per order** (a restart mid-loop must not orphan a
placed ETF), and excluded from the Trader's stop/target exit loop (the rebalancer is their
sole manager).

**VIX-term crash governor (Alpha-v7 F1b, 2026-06-14):** before order generation the sleeve's
effective budget is scaled by `_crash_governor_multiplier(db)` ∈ [`pm.crash_governor_derisk_to`,
1.0] — when the VIX term structure inverts (VIX > VIX3M on the last `pm.crash_governor_confirm_days`
SETTLED closes), exposure is de-risked (default 0.5), else full. Flag `pm.crash_governor_enabled`
defaults ON (owner-approved); reversible live. It is **fail-safe** (missing/stale/error VIX-VIX3M →
multiplier 1.0 = pre-governor behavior) and can only ever REDUCE exposure — the per-name and 80%
gross caps still bind. PIT-correct: it reads only settled closes (strictly before today), so the
Monday 09:45 ET intraday VIX print is never used. The applied multiplier is in the rebalance summary
(+ a detailed log line when active). Loading the code needs an orchestrator restart; the flags do not.

**Credit overlay (Alpha-v8 G1/G4, 2026-06-15) — wired, flag DEFAULT OFF:** a second overlay,
`_credit_governor_multiplier(db)` (HYG/IEF below its 120d MA by >2% = credit spreads widening →
de-risk to 0.5), composes multiplicatively with the VIX governor: `overlay_mult = max(0.25,
vix_gov × credit)` and `alloc *= overlay_mult` (the 0.25 floor bounds the stacked de-risk). Flag
`pm.credit_governor_enabled` **defaults OFF** (owner-gated candidate) — so today it contributes 1.0
and the live book is unchanged. Same fail-safe + PIT (settled closes) + can-only-reduce properties
as the VIX governor. To activate: review the G1 caveats (small tail-insurance effect; multiplicity;
paper-only), then set the flag true.

**Why weekly:** `TSMOMConfig.rebalance_days=5` is the validated backtest cadence — the
standalone +0.71 Sharpe (2007–26, all crises) was measured on a 5-day rebalance grid; the
Monday wall-clock rebalance is its faithful live analog. **Why Monday:** a fixed weekday gives
a deterministic weekly cadence; 09:45 is after the 09:30 selection nudge and after the open
settles so daily bars/quotes are stable.

**Why not more frequent:** mechanically safe (the path is rebalance-to-target + idempotent
per-day `client_order_id` + fail-closed + gross-capped), but economically counter-productive —
TSMOM signals use 21/63/126/252-day lookbacks and barely move day-to-day, so daily rebalancing
mostly churns transaction costs (the backtest charges 2bps/one-way turnover; daily would
multiply turnover ~5× against a near-static signal, degrading the validated Sharpe). The
per-day `client_order_id` dedup is also keyed on the date, so a finer cadence would need a
finer dedup key. **Keep weekly** unless a future signal redesign justifies otherwise.
