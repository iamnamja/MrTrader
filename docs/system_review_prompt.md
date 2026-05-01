# MrTrader System Review — Prompt for External LLM Analysis

## Your Role

You are a world-class quantitative analyst with deep experience spanning:
- Systematic hedge fund architecture (multi-strategy PM + execution desks)
- ML-driven alpha generation and walk-forward validation
- Market microstructure and execution quality
- Risk management frameworks (institutional grade)
- News and alternative data signal processing
- Real-time distributed systems for trading

You are being asked to review an in-development automated trading system currently in paper trading. Your job is to:
1. Identify **gaps, weaknesses, and risks** in the current architecture and workflow
2. Suggest **specific, concrete next phases** to move this toward institutional-grade quality
3. Flag anything that would **fail in live trading** that paper trading wouldn't catch
4. Prioritize recommendations by **risk-reduction vs alpha-generation** impact

Be brutally honest. This is not a review for praise — it is a gap analysis to make the system production-ready.

---

## System Overview

**Name:** MrTrader  
**Status:** Paper trading active (live since 2026-04-28). $20,000 starting capital.  
**Goal:** Move to live trading once paper trading gate is met (4-week Sharpe > 0.5, max drawdown < 5%).  
**Long-term vision:** Institutional-grade multi-strategy automated trading system with full audit trail, adaptive ML, and news intelligence.

---

## Infrastructure

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API Framework | FastAPI (REST + WebSocket) |
| Broker | Alpaca (paper + live; IEX data feed) |
| ML | XGBoost, LightGBM, scikit-learn |
| Feature Engineering | NumPy, Pandas, TA-Lib |
| HPO | Optuna (Bayesian, 3-fold TimeSeriesSplit) |
| Databases | SQLite (feature store, dev trades) / PostgreSQL (prod) via SQLAlchemy ORM |
| Data: daily bars | Alpaca StockHistoricalDataClient |
| Data: 5-min bars | Polygon.io REST + S3 bulk download (cached as Parquet, 23h TTL) |
| Data: fundamentals | SEC EDGAR XBRL API (free), FINRA Reg SHO, FMP API |
| Data: news | Finnhub REST API (free tier) |
| News LLM scorer | Anthropic Claude Haiku (Tier 2 signals), Claude Sonnet (Tier 1 macro) |
| Frontend | React + Vite monitoring dashboard (local) |
| Deployment | Local Windows 11, 32 cores, 64GB RAM. Cloud migration deferred post paper-trading gate. |
| Testing | pytest, 1171+ tests |
| CI | GitHub Actions |

**Rate limiting:** Alpaca 200 req/min (we cap at 180 with token bucket). Polygon free tier (5 req/sec). Finnhub free tier (60 req/min).

---

## Agent Architecture

Three autonomous agents run as async Python coroutines, communicating via in-process async queues. Each agent runs its own infinite loop and shares no mutable state directly — all state is passed via message dicts or written to the shared SQLite/PostgreSQL DB.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Process (uvicorn)                   │
│                                                                 │
│  ┌──────────────────────┐    async queue     ┌──────────────┐  │
│  │  Portfolio Manager   │ ─────────────────► │ Risk Manager │  │
│  │  (PM)                │                    │ (RM)         │  │
│  └──────────────────────┘                    └──────┬───────┘  │
│                                                     │          │
│                                              async queue       │
│                                                     ▼          │
│                                             ┌──────────────┐   │
│                                             │    Trader    │   │
│                                             │   Agent      │   │
│                                             └──────────────┘   │
│                                                                 │
│  ┌──────────────────────┐    APScheduler (async)               │
│  │  AgentOrchestrator   │ ── health check every 5 min          │
│  │                      │ ── EOD jobs at 16:30 ET              │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Portfolio Manager (PM) — Full Workflow

The PM is the "brain." It decides what to trade and when.

### Daily Schedule (ET)

| Time | Task |
|---|---|
| 08:00 | `swing_premarket_analysis` — score entire watchlist, identify top 10 candidates |
| 09:00 | `premarket_intelligence` — run NIS (News Intelligence Service) macro analysis |
| 09:50 | Send swing proposals to RM (held 50 min to avoid gap-chase) |
| Every 60 min | `_intraday_position_review` — re-check held positions (planned Phase 81, not yet built) |
| 09:45, 11:00, 13:30 | Intraday scan windows (3 per day, 2-hr cooldown per symbol) |
| 15:45 | EOD swing position review (Phase 75 — planned, not yet built) |
| 16:00 | Market close — all intraday positions must be flat |
| 16:30 | EOD jobs: outcome backfill, daily summary write, model retrain trigger |
| 17:00 | Swing model retrain (XGBoost, warm cache ~5 min) |

### Swing Candidate Selection (08:00 ET)

1. **Universe:** ~430 symbols (Russell 1000 filtered for Alpaca availability)
2. **Feature fetch:** 84 features per symbol — OHLCV-derived, no fundamentals in model
3. **ML score:** XGBoost `path_quality` regression score (0–1) for each symbol
4. **Top 10 by score** selected as candidates
5. **PM Abstention gates** (any fail = no swing trades today):
   - VIX ≥ 25
   - SPY below 20-day MA
   - SPY 5-day return ≤ 0
6. **Entry quality gates** (per-symbol, checked at proposal time):
   - Gap-chase: price must be within 2% of prior close (prevents chasing overnight gaps)
   - Spread: bid-ask spread < 0.5% of mid
   - Volume: current bar volume ≥ 50% of 20-day avg volume
   - Momentum: price must be trending in direction of signal
7. **NIS overlay:** morning news digest pre-scores symbols; `block_entry` or `size_down` applied
8. **Proposals sent to RM at 09:50 ET** (50-min hold prevents opening gap chase)

### Intraday Candidate Selection (3 windows)

1. **Universe:** ~720 symbols (Polygon-cached Russell 1000)
2. **Features:** 50 intraday features including `is_open_session` indicator
3. **Entry:** Bar 12 (60 min from open) — the system found the "opening session edge" superior to afternoon entries
4. **Hold window:** 24 bars (2 hours)
5. **Intraday Abstention gates:**
   - SPY intraday drawdown > 1.5% (live macro gate)
   - NIS macro `block=True` (e.g., FOMC day)
   - Lunch block (11:15–13:00 ET) — no new entries
6. **Proposals sent to RM immediately** (no hold period for intraday)

### PM Re-Scoring (Phase 70 — Active)

Every 30 minutes, PM re-scores symbols that were approved by RM but not yet executed:
- If confidence drops below threshold: send `WITHDRAW` signal → Trader cancels pending entry
- Prevents staleness: a proposal approved at 09:50 but not filled by 11:30 may be based on stale signals

### News Intelligence Service (NIS)

Three-tier LLM-based news analysis:
- **Tier 1 (Sonnet):** Macro event classification — FOMC, CPI, NFP, ISM. Outputs `risk=LOW/MEDIUM/HIGH`, `sizing_factor`, `block=True/False`. Wired into premarket routine and live SPY gate.
- **Tier 2 (Haiku):** Symbol-level news scoring — fetches Finnhub headlines per symbol, scores `materiality`, `direction`, `confidence`, `already_priced_in`. Applied as overlay on PM proposals.
- **Policy engine:** `exit_review` (bad news on held position → exit signal), `block_entry` (block new entry), `size_down` (reduce position size).
- **30-min held-position re-check:** NIS re-scores all held swing positions every 30 min during market hours. Breaking negative news → EXIT signal queued.

---

## Risk Manager (RM) — Full Workflow

The RM is the gatekeeper. It has **veto power** over all trade proposals from PM.

### Validation Chain (sequential — any failure = reject)

1. **Kill switch:** If manually triggered, reject all proposals.
2. **Circuit breaker:** If 3+ network errors in 5 min (Alpaca/data feed issues), reject all.
3. **Account state:** Buying power check — enough capital for the proposed position.
4. **Daily loss limit:** If realized + unrealized P&L today < −2% of portfolio: reject all (trading halt).
5. **Max open positions:** ≤ 5 simultaneous swing positions, ≤ 3 intraday.
6. **Position size:** 10–15% of portfolio per swing trade; 5% per intraday. Dynamic: reduced in high-VIX regimes.
7. **Sector concentration:** ≤ 30% of portfolio in any single GICS sector.
8. **Correlation gate:** Veto if pairwise 30-day return correlation > 0.75 with an existing position > 5% of portfolio.
9. **Duplicate check:** One approval per symbol per day (prevents re-entering same symbol after exit).

### RM Output

- `APPROVED` → proposal forwarded to Trader Agent queue
- `REJECTED` → logged to `decision_audit` table with reason; PM notified

---

## Trader Agent — Full Workflow

The Trader executes approved proposals and manages all open positions.

### Entry Execution

1. Receives approved proposal from RM via async queue
2. **Final entry quality check** at execution time (re-checked because market may have moved since approval):
   - Price run: current price > entry_price × 1.005 → skip (too much slippage)
   - Spread: bid-ask spread < 0.5%
   - Momentum: 5-min bar close > open (for buys)
   - Volume: current bar ≥ 50% of 20-day avg
3. If checks pass: places **limit order** at `ask + 1 tick` (not market orders, to control slippage)
4. Tracks pending limit orders in memory with 5-minute TTL; cancels unfilled orders after TTL

### Exit Management (runs every 300 seconds / 5 min)

For each active position, checks in order:

1. **Stop loss:** current price ≤ stop_price → EXIT
2. **Target hit:** current price ≥ target_price → EXIT (full)
3. **T1 Partial exit:** when price reaches 50% of the way to target: sell 50% of position, extend target by 0.5×ATR
4. **Trailing stop (ATR-based):** if highest_seen_price − current_price > 1.5×ATR → EXIT (trailing)
5. **VIX stop tightening:** if VIX spikes > 20 mid-session, stop_price moves to entry + 0.25×ATR (tighten to protect profits)
6. **Max hold:** bars_held ≥ 20 (calendar trading days — fixed from earlier bug where it counted heartbeat ticks) → EXIT
7. **NIS exit signal:** if PM/NIS sends EXIT message for symbol → EXIT
8. **Market hours guard:** exit signals suppressed outside 09:30–16:00 ET (prevents pre-market order placement)

### Startup Reconciliation

On every uvicorn restart, Trader reconciles Alpaca's live positions with DB:
- Positions in DB (ACTIVE) but not in Alpaca → flagged as `RECONCILE_GHOST`
- Positions in Alpaca but not in DB → synthetic Trade record created (handles limit orders that filled during downtime)
- All active positions reloaded into in-memory `active_positions` dict from DB

---

## ML Models

### Swing Model (v119 — Active, Gate Passed)

| Attribute | Value |
|---|---|
| Architecture | XGBoost regression |
| Label | `path_quality = 1.0×upside_capture − 1.25×stop_pressure + 0.25×close_strength` |
| Forward window | 5 trading days |
| Universe | SP-100 (~81 symbols) for training; ~430 for inference |
| Features | 84 (OHLCV, technical indicators, regime features) — no fundamentals |
| Stop / Target | 0.5×ATR / 1.0×ATR (2:1 R:R) |
| Walk-forward Avg Sharpe | **+1.181** (gate: > 0.80) ✅ |
| Walk-forward Win% | 58.0% |
| Retrain | Daily at 17:00 ET on fresh bars |
| MetaLabel | XGBRegressor v1 (R²=0.059, corr=0.286) — acts as secondary confidence gate |

Walk-forward validation results by fold:
| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | 2022-07 → 2023-10 | 94 | 53.2% | +0.88 |
| 2 | 2023-10 → 2025-01 | 113 | 65.5% | +2.69 |
| 3 | 2025-01 → 2026-04 | 134 | 55.2% | −0.03 |

**Notable concern:** Fold 3 (most recent period) has Sharpe of −0.03 — the model may be losing edge in current market conditions.

### Intraday Model (v29 — Active, Gate Passed)

| Attribute | Value |
|---|---|
| Architecture | XGBoost classifier |
| Label | `path_quality = 1.0×upside − 0.50×stop_pressure + 0.25×close_strength` |
| Forward window | 24 bars (~2 hours) |
| Universe | Russell 1000 (~720 symbols with Polygon cache) |
| Features | 50 (includes `is_open_session` indicator) |
| Entry bar | Bar 12 (60 min from open) — opening session edge |
| Stop / Target | 0.4×ATR / 0.8×ATR (2:1 R:R, compressed) |
| Walk-forward Avg Sharpe | **+1.807** (gate: > 1.275) ✅ |
| Retrain | Daily at 17:00 ET |

Walk-forward results:
| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | 2024-10 → 2025-04 | 146 | 48.6% | +1.73 |
| 2 | 2025-04 → 2025-10 | 224 | 39.7% | +0.72 |
| 3 | 2025-10 → 2026-04 | 158 | 53.8% | +2.97 |

---

## Data & Feature Engineering

### Swing Features (84 total)
- Price action: RSI, MACD, Bollinger Bands, ATR, EMA crossovers, SMA ratios
- Volume: relative volume, OBV, volume trend
- Regime: SPY MA distance, VIX level, market breadth proxy
- Pattern: higher highs/lows sequences, gap-up detection
- **No earnings date proximity** (earnings gate handled separately via Finnhub calendar)
- **No fundamentals in the model** (SEC EDGAR data available but not used in features)

### Intraday Features (50 total)
- 5-min OHLCV derived: ATR, VWAP distance, opening range position
- Session indicator: `is_open_session` (first 60 min from open)
- Relative volume, gap-from-open, price momentum over 5/15/30-min windows

### Feature Store
- SQLite WAL mode, ~170k entries
- 23h TTL on Parquet price cache
- Cold cache rebuild: ~20-25 min (ProcessPool, 32 workers)
- Warm cache retrain: ~5 min

---

## Risk Controls (Complete List)

| Control | Implementation |
|---|---|
| Kill switch | Manual + API endpoint; instantly halts all new orders |
| Circuit breaker | 3 network errors in 5 min → trading pause; auto-resets after 10 min |
| Daily loss limit | −2% portfolio → trading halt for the day |
| Position size limit | 10-15% swing, 5% intraday |
| Max open positions | 5 swing, 3 intraday |
| Sector concentration | ≤ 30% any GICS sector |
| Correlation gate | Veto if pairwise 30-day corr > 0.75 with existing position > 5% |
| Gap-chase prevention | Entry price must be within 2% of prior close |
| Spread filter | Bid-ask spread < 0.5% |
| PM abstention | VIX ≥ 25 OR SPY < MA20 OR SPY 5d return ≤ 0 |
| Live macro gate | SPY intraday drawdown > 1.5% → intraday halt |
| NIS macro block | Claude-scored macro events can block all new entries |
| Earnings gate | Earnings within 3 days → no entry (Finnhub calendar) |
| Macro event window gate | FOMC/CPI/NFP within window → sizing_factor applied |
| Market hours guard | No exits placed outside 09:30–16:00 ET |
| Intraday lunch block | No new intraday entries 11:15–13:00 ET |
| EOD flat rule | All intraday positions closed by 16:00 ET |
| Duplicate check | One approval per symbol per day |
| ATR-based stop | Dynamic stop at 0.5×ATR below entry (swing) |
| Trailing stop | Exit if price falls 1.5×ATR from highest seen |
| VIX stop tightening | If VIX spikes > 20 mid-session, stop tightens to entry + 0.25×ATR |
| T1 Partial exit | At 50% of the way to target: sell 50% of shares |
| Max hold (swing) | 20 calendar trading days |
| Max hold (intraday) | EOD (24 bars / 2 hours) |

---

## Audit & Observability

- **`decision_audit` table:** Every PM enter/block decision recorded with: symbol, gate name, reason, top-8 model features, outcome (filled later via backfill job)
- **`AuditLog` table:** Every agent action (reconcile, errors, kill switch events)
- **`AgentDecision` table:** PM/Trader/RM high-level decisions (used for startup flag restoration)
- **`Order` table:** Every order with fill price, slippage_bps, status
- **`Trade` table:** Entry/exit, P&L, bars_held, stop/target, signal_type
- **EOD outcome backfill:** Nightly job writes realized P&L back into `decision_audit` rows
- **`gate_performance_summary()`:** Per-gate block rate + outcome analysis (did this gate block winners or losers?)
- **`RiskMetric` daily summary:** Swing vs intraday P&L split, trade count, win rate, block rate
- **`paper_trading_report.py`:** CLI report script — slippage, decision audit, daily summary, day replay
- **`replay_day.py`:** Chronological timeline of a given day — macro context, NIS signals, PM decisions with features, trades, fills

---

## Dashboard (React Frontend)

- Real-time P&L (cumulative equity curve sourced from Alpaca's equity series)
- Open positions with stop/target
- Trade history (win rate, avg hold, realized P&L)
- Agent status panel (PM/RM/Trader heartbeat, last action)
- PM Decision Audit panel (per-symbol gate pass/fail table)
- NIS signals panel (macro risk, symbol scores)
- Agent System log (live WebSocket feed)
- Manual controls: kill switch, close position, adjust limits

---

## Known Bugs Fixed During Paper Trading

1. **`bars_held` counting heartbeat ticks instead of calendar days** — caused max-hold exit after 100 minutes instead of 20 days. Fixed: now uses `_last_bar_date` to count calendar days only.
2. **Exit signals firing pre-market** — no market hours guard existed. Fixed: exits suppressed outside 09:30–16:00 ET.
3. **Partial exit P&L not accumulating** — partial exits computed P&L but didn't write to `trade.pnl`; final exit overwrote with only remaining-shares leg. Fixed: partial P&L accumulated on trade record.
4. **In-memory daily flags reset on restart** — `_analyzed_today`, `_selected_today` etc. reset on every uvicorn restart causing duplicate task execution. Fixed: `_restore_daily_flags()` reads `AgentDecision` table on startup.
5. **Reconciler creating duplicate positions** — startup reconciler created new Trade records without checking `RECONCILE_GHOST` status. Fixed.
6. **Runaway scanner** — April 28: intraday scanner fired every 5 min after market close, buying same stocks 7-9x. Root cause was missing market-hours check before scan. Fixed.
7. **P&L chart using per-period values** — Alpaca's `profit_loss` series gives per-period P&L, not cumulative. Fixed: now uses `equity` series with `equity[0]` as baseline.
8. **Uvicorn shutdown hang** — agent tasks weren't cancelled cleanly. Fixed: 5-second timeout on task cancellation in `orchestrator.stop()`.
9. **Daily flag restore timezone mismatch** — `_restore_daily_flags()` compared TZ-aware ET datetime against naive UTC DB timestamps. SQLite returned 0 rows silently. Fixed: compare naive UTC midnight.

---

## What Is NOT Yet Built (Planned Backlog)

### Near-Term (Tier 3)

**Phase 75 — EOD Swing Position Review**
At 15:45 ET, re-score all held swing positions using end-of-day bars. If score < EXIT_THRESHOLD AND held ≥ 2 days → send EXIT. Currently, positions can be held overnight with no "should I still hold this?" check.

**Phase 76 — Slippage Analysis**
`Order.slippage_bps` is stored per fill but never surfaced. Per-symbol slippage report to identify chronic drag.

**Phase 77 — Graceful SIGTERM / Queue Drain**
In-flight proposals can replay on restart. Signal handler should drain the trade_proposals queue and log cancellations before shutdown.

**Phase 78 — Live Readiness Checklist Update**
`test_live_readiness.py` was written pre-NIS. Needs updates for all new components.

**Phase 79 — AUC Drift Alert**
Model retrains daily but silent AUC degradation is invisible. Alert needed when OOS AUC drops below gate − 0.03.

**Phase 80 — Intraday Position Re-Evaluation (Planned)**
Every 60 min during market hours, re-score held swing positions using ML inference + NIS check + SPY regime. Two consecutive weak scores → EXIT. Currently, exits are price-rule-only — no mid-session "is this trade still thesis-valid?" check.

### Deferred (Tier 4)

**Phase 80 (separate) — True Technical Day Trader (New Agent)**
The current "intraday" model holds positions for ~2 hours. A true scalper is architecturally different:
- Separate `IntradayScalper` agent (not a modification of existing intraday PM)
- Entry triggers: VWAP breakout/reclaim, 1-min/5-min momentum, volume surge
- Universe: 20–30 liquid names only (AAPL, TSLA, NVDA, SPY, QQQ)
- Hold time: 5–45 minutes
- Exits: VWAP lose, 0.3–0.5% target, 0.2% hard stop, 45-min time stop
- No ML scoring — pure signal-to-execution, sub-second latency target
- All positions flat by 15:45 ET
- **Blocked by:** requires L1/L2 tick data (Alpaca 1-min bars too coarse), different backtesting harness
- **Precondition:** current system stable in live trading first

**Other Deferred:**
- News sentiment as ML training features (need 60 days NIS history first)
- Options flow data (unusual activity as entry filter — Polygon premium required)
- Finer-grained regime detection (bull/bear/chop sub-modes)
- Intraday model retrain with multi-scan distribution data

---

## Current Paper Trading State (as of 2026-05-01)

- Paper trading since 2026-04-28 (3 days)
- Realized P&L: ~$958 on 14 completed trades (50% win rate)
- One open swing position: ALK (Alaska Airlines, 44 shares, entered 2026-04-30)
- ISM Manufacturing event today (2026-05-01) flagged MEDIUM risk, sizing_factor=0.75
- Today's swing candidates: 10 found, 6 blocked by gap-chase, proposals for remaining 4 sent to RM at 09:50
- Intraday: 3 scan windows active, SPY regime clean

---

## Specific Questions for the Analyst

1. **Model architecture:** The swing model uses regression-style path_quality as a label. Is this the right formulation for a 5-day swing trade? Would a directional binary label (positive return over threshold vs not) or a ranking-based approach (LambdaRank style) work better?

2. **Feature coverage:** 84 features, all OHLCV-derived — no order flow, no L1/L2, no fundamentals in the model. Where are we leaving the most alpha on the table?

3. **Walk-forward concern:** Swing model Fold 3 (most recent 15 months) has Sharpe of −0.03. This is a red flag. What causes this and how would you address it?

4. **Intraday edge:** The model found that Bar 12 (60-min from open) is the best entry. The "opening session edge" is well-documented but also well-known and crowded. How do we validate this edge is real vs. backtest artifact?

5. **Execution quality:** We use limit orders at ask+1 tick. For a $20k account trading liquid large-caps, is this the right execution approach? What are we missing on the execution side?

6. **Risk architecture:** We have many independent risk controls (sector, correlation, daily loss, position size) but they operate independently. Is there a portfolio-level risk model (VaR, CVaR, factor exposure) that should be wrapping all of this?

7. **News intelligence:** We use an LLM to score Finnhub headlines. How would a quant fund actually use news data? What are we doing wrong here?

8. **Regime detection:** Our market regime is binary (abstain vs trade based on VIX + SPY MA). Sophisticated systems use multi-factor regime models. What would a proper regime model look like for this use case?

9. **Live trading readiness:** What are the 3 most likely failure modes that paper trading would NOT catch but live trading would expose immediately?

10. **Path to institutional grade:** If you had to prioritize 5 things to get from "good paper trading system" to "institutional-grade automated strategy," what are they?

---

*System built by a small team. Python codebase. ~1,200 pytest tests. All decisions logged. Paper trading gate criteria: 4-week Sharpe > 0.5, max drawdown < 5%.*
