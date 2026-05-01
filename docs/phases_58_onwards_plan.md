# MrTrader — Consolidated Phase Roadmap

**Last updated:** 2026-05-01 (v3 — code-grounded review by Opus, full V1-V31 verification)
**Status:** Paper trading active. Critical safety gaps identified — see Phase 75 and 76 before any live promotion.

---

## Part 1 — What Is Complete

### Foundation (pre-2026-04-28)
All phases 1–56 complete. Walk-forward gates passed for both models.

| Model | Version | Avg Sharpe | Gate |
|---|---|---|---|
| Swing | v119 | +1.181 | ✅ >0.80 |
| Intraday | v29 | +1.776 | ✅ >0.80 |

### Paper Trading Hardening (2026-04-28)

| Phase | Name | What Was Built |
|---|---|---|
| A | State Durability | `_reconcile_positions()` on startup; DailyState table for PM flags |
| B | Swing Recovery | Swing re-proposes correctly after late restart |
| C | Proposal Hygiene | 60-min TTL, 4 PM Redis purge, one-approval-per-symbol-per-day dedup |
| 67 | Trader Entry Quality Check | Price run, spread, momentum, volume gates at execution time |
| 68 | Live Macro/Market Gate | SPY intraday drawdown check; `is_swing_blocked()` / `is_intraday_blocked()` |
| 69 | Dynamic Stop Trailing + Partial Exits | T1 partial exit, VIX stop tightening, target extension |
| 70 | PM Re-Scoring / WITHDRAW Signal | Stale proposals withdrawn after confidence drops below threshold |

### News Intelligence Service (2026-04-29)

| Phase | Name | What Was Built |
|---|---|---|
| 58 | Earnings Calendar Gate | `fetch_earnings_calendar()` via Finnhub; `CalendarEvent` DB table |
| 59 | Macro Calendar Awareness | NIS Tier 1 wired into `premarket.py` — consensus-aware FOMC/CPI/NFP scoring |
| 60 | Structured NewsSignal + Haiku Scorer | Full `app/news/` stack: `signal.py`, `llm_scorer.py`, `intelligence_service.py`, `finnhub_source.py` |
| 63 | News as PM Scoring Overlay | `news_score_overlay()` applied in `_build_proposals()`; sizing adjusted per NIS policy |
| 71 | Correlation Gate in RM | `validate_correlation_risk()`: veto if pairwise 30-day return corr >0.75 with >5% position |
| 61 | Decision Audit Trail | `decision_audit` DB table; PM writes every enter/block row; `gate_performance_summary()` for calibration |
| 51 | Multi-Scan Intraday | 3 windows (09:45, 11:00, 13:30 ET); 2hr per-symbol cooldown; daily 1% P&L loss cap |
| 62 | Morning NIS Digest | 09:00 ET pre-scores full candidate universe via NIS Tier 2; warms cache before 09:50 send |

### Backlog Phases Shipped (2026-04-29 session 2)

| Phase | Name | What Was Built |
|---|---|---|
| 72 | NIS Tier 2 re-check for held swing positions | In 30-min review loop: re-fetches `NewsSignal`; `exit_review`/`block_entry` policy triggers EXIT |
| 73 | Overnight gap open price fix | Gap calc now uses actual 1-min open bar (not prior day close proxy); auto-exit already wired |
| 74 | NIS & Decision Audit API endpoints | `GET /api/nis/macro`, `/api/nis/signals`, `/api/nis/cost`, `/api/decision-audit/summary`, `/api/decision-audit/recent` |

**Also shipped (bug fixes):**
- Trader 3 PM cutoff now intraday-only (was incorrectly blocking swing)
- `log_decision` emitted when macro gate blocks an entry
- `_scan_new_opportunities()` gates on both macro + PM abstention before scanning
- `_get_symbol_sector()` helper added to PM

### Audit Trail Hardening (2026-04-29 session 3)

| Gap | What Was Built |
|---|---|
| 1 — Feature explainability | `top_features` JSON column added to `decision_audit`; PM caches `features_by_symbol` and writes top-8 model-important feature values per decision |
| 2 — Outcome backfill scheduled | `_run_eod_jobs()` in PM at 16:30 ET calls `backfill_outcomes(14)`; `retrain_cron.py` also calls it post-retrain |
| 3 — Daily summary row | `app/database/daily_summary.py` → `write_daily_summary()` upserts `RiskMetric` with swing/intraday P&L, trade count, win rate, block rate breakdown |
| 4 — Decision audit in report | `section_decision_audit()` + `section_daily_summary()` added to `scripts/paper_trading_report.py` |
| 5 — Day replay script | `scripts/replay_day.py --date YYYY-MM-DD [--symbol X]` — chronological timeline of macro context, NIS signals, PM decisions (with top features), trades + fills, agent events |

**Test count:** 1171 passing, 4 skipped.

---

## Part 2 — Active Backlog (Priority Order)

### Tier 1 — Let It Run (no code needed, just time)

#### Phase 57 — Paper Trading Review + Calibration
**Precondition:** 2–4 weeks of live paper trading + `decision_audit` table populated.

**What to do when ready:**
1. Run `gate_performance_summary()` from `app/database/decision_audit.py` — did each gate block winners or losers?
2. Check `llm_call_log` — daily LLM cost, cache hit rate, any latency spikes
3. Review intraday multi-scan: did 11:00 and 13:30 windows add alpha or just noise?
4. Check correlation gate veto rate — firing too often means the threshold needs tuning
5. Swing vs intraday P&L split — is the 70/30 capital allocation right?

**Gate for live trading:** 4-week paper Sharpe > 0.5, max drawdown < 5%.

> **⚠️ 2026-05-01 code-review update:** The above gate is necessary but NOT sufficient. Phases 75–78 + 82–83 must all be green before any real-money promotion. See code-grounded plan below.

---

### Tier 2 — Near-Term Engineering ✅ COMPLETE

Phases 72, 73, 74 shipped 2026-04-29.

| Phase | Name | Status |
|---|---|---|
| 72 | NIS re-check for held swing positions | ✅ Done — `exit_review` policy triggers EXIT in 30-min review |
| 73 | Overnight gap open price accuracy | ✅ Done — uses actual 1-min open bar; auto-exit already wired |
| 74 | NIS & audit dashboard API | ✅ Done — 5 new endpoints in `app/api/nis_routes.py` |

---

---

## Part 2b — Code-Grounded Priority Plan (Phases 75–84)

> Added 2026-05-01 from full codebase review (V1-V31 verification, ~14 min Opus sweep). These phases supersede the old Tier 3 ordering below. Full detail: `docs/phase_plan_v3_code_grounded.md`.

### ⛔ Critical Findings (must fix before live money)

| Finding | File:Line | Severity |
|---|---|---|
| Kill switch is decorative — `kill_switch.is_active` checked in zero agent loops | `app/agents/*.py` (grep: 0 matches) | **CRITICAL** |
| MetaLabel filter NOT in production — docs claim ~26% filter; code shows backtest-only | `app/agents/portfolio_manager.py` (grep: 0 matches for MetaLabel) | **HIGH** |
| Slippage measurement broken — uses post-fill `get_latest_price` not `filled_avg_price` | `app/agents/trader.py:646` | **HIGH** |
| Survivorship bias — universe is current static list, not point-in-time membership | `app/utils/constants.py:7,141,145` | **HIGH** |
| Earnings gate fails open — yfinance 401s silently; `block_swing=False` on exception | `app/calendars/earnings.py:120-122` | **HIGH** |
| `_peak_equity` resets on restart — drawdown rule loophole | `app/agents/risk_manager.py:54,312` | **MEDIUM** |
| Multi-universe inconsistency — walk-forward on SP-100; training on R-1000; live on SP-500 | `scripts/walkforward_tier3.py:228` vs `train_model.py:580` vs `portfolio_manager.py:551` | **MEDIUM** |
| Bar-12 intraday entry never sensitivity-tested (bars 9-15 never swept) | `app/ml/intraday_features.py:86` | **MEDIUM** |

### Phase 75 — Wire Kill Switch Into All Agents (Critical, 1 day)
**Gap:** `kill_switch.is_active` is checked in zero agent loops. Calling `activate()` closes positions but PM/RM/Trader continue accepting and executing new proposals. Fatal for a real account during a flash crash.

**What to build:**
- `app/agents/portfolio_manager.py`: early-return + `log_decision` in `_send_swing_proposals` and `select_intraday_instruments` when `kill_switch.is_active`
- `app/agents/risk_manager.py`: reject all incoming proposals with `failed_rule="kill_switch"`
- `app/agents/trader.py`: skip `_check_entry`; cancel pending limit orders when active
- `app/live_trading/kill_switch.py`: also cancel all open Alpaca orders inside `activate()` (currently it only closes positions, not pending orders)
- `tests/test_kill_switch_blocks.py`: new — activate kill switch, send fake proposal, assert Order count unchanged within 30s

**Acceptance criteria:** After `kill_switch.activate()`, no new `Trade` rows created within 30s. All open Alpaca orders cancelled. AuditLog has `KILL_SWITCH_ACTIVATED` entry.

**Blocker for live trading:** Yes.

### Phase 76 — Fix Slippage Measurement for Market Orders (Critical, 0.5 days)
**Gap:** `trader.py:646` uses `alpaca.get_latest_price(symbol)` AFTER `place_market_order()` returns — this is the *next* quote bar, not the fill price. `intended_price = result.entry_price` was set at signal time (often 5–30 min stale). So `slippage_bps` column is measuring (post-fill price drift vs signal-time price) — not actual execution quality.

**What to build:**
- Replace `get_latest_price` call with poll-loop on `alpaca.get_order_status(order_id)` using `filled_avg_price`
- Capture `intended_price` from `alpaca.get_quote()["mid"]` immediately before order submission
- `tests/test_slippage_market.py`: mock order status → assert correct `slippage_bps` computation

**Acceptance criteria:** Median `|slippage_bps|` on market orders in next 5 paper-trading days documented and < 30bps for SP-100 names.

### Phase 77 — Decision-Audit Dashboard Tile (High, 2 days)
**Gap:** `decision_audit` table is populated but nothing reads it. Cannot answer "are the gates actually working?"

**What to build:**
- `GET /api/audit/summary` endpoint — aggregate win-rate by `block_reason`, avg realized P&L by `news_action_policy`, `model_score` bucket vs realized 4h return
- Frontend tile on dashboard
- `scripts/backfill_decision_outcomes.py` — 16:30 ET cron that populates `outcome_pnl_pct`, `outcome_4h_pct`, `outcome_1d_pct`

**Acceptance criteria:** After 2 weeks, can read "NIS `sizing_multiplier 0.7-0.8` correlates with X bps lower realized return." Backfill runs in < 10 min.

**Dependencies:** Phase 76 (so realized P&L is trustworthy)

### Phase 78 — Order Lifecycle State Machine + Periodic Reconciliation (High, 2 days)

**Context:** Today's order handling has multiple production gaps discovered during paper trading (TNDM duplicate positions, JBLU target miss, orphaned open orders). This phase makes the order lifecycle explicit, persistent, and resilient to crashes and restarts.

#### 78a — Fix Partial Fill Handling (0.5 days)
**Gap:** `_poll_pending_limit_orders` treats `partially_filled` identically to `filled` — records the partial qty and removes the order from tracking. The remaining unfilled shares stay open in Alpaca, can fill silently later, and become untracked positions.

**What to build:**
- When `order_status == "partially_filled"`: record the partial qty as a Trade ✅ (already done), then **immediately cancel the remainder** via `alpaca.cancel_order(order_id)`, log `PARTIAL_FILL_REMAINDER_CANCELLED` to decision audit
- If partial fill qty is 0 (race condition): leave in pending, do not record or cancel
- Test: mock status returning `partially_filled` with filled_qty=50/qty=100; assert Trade records 50 shares AND cancel is called for the order

#### 78b — Persist `_pending_limit_orders` to DB (0.5 days)
**Gap:** `_pending_limit_orders` is in-memory only. On uvicorn restart while a limit order is unfilled, the order ID is lost. Startup reconciler creates a placeholder Trade with default 2%/6% stop/target, overriding the strategy's ATR-based values.

**What to build:**
- Add `PendingLimitOrder` DB table (or use `Order` with `status='PENDING_LIMIT'`): columns `symbol, order_id, shares, limit_price, intended_price, stop_price, target_price, atr, trade_type, created_at`
- On limit order placement: write to DB immediately (before adding to in-memory dict)
- On startup: load any rows where `status='PENDING_LIMIT'` and created today; re-populate `_pending_limit_orders` from DB; startup reconciler skips symbols that have a `PENDING_LIMIT` row
- On fill/cancel/expire: delete the DB row
- Test: simulate restart with one `PENDING_LIMIT` row; assert it's re-loaded into `_pending_limit_orders` and polled correctly

#### 78c — Idempotency Keys on Order Placement (0.5 days)
**Gap:** `place_limit_order` and `place_market_order` don't pass a `client_order_id`. If the app crashes between order submission and DB write, a restart can place a duplicate order.

**What to build:**
- Generate `client_order_id = f"mrtrader-{symbol}-{date.today()}-{uuid4().hex[:8]}"` before calling Alpaca
- Pass as `client_order_id` in the order request
- On duplicate detection (Alpaca returns 422 "order already exists for client_order_id"): look up the existing order by `client_order_id`, re-use its `order_id`, continue normally
- Store `client_order_id` in `Order` table
- Test: mock Alpaca to return 422 on second call; assert no duplicate Order row in DB

#### 78d — Periodic Mid-Session Reconciliation (0.5 days)
**Gap:** Reconciliation runs only at startup. Manual closes in Alpaca UI, fills of forgotten orders, or split/dividend adjustments aren't detected until next restart.

**What to build:**
- Add `_reconciliation_loop` task to Trader (not PM) — runs every 15 min during market hours (09:30–16:00 ET)
- On each run: call `startup_reconciler.reconcile(alpaca, db)` + new `cancel_orphaned_orders(alpaca, db)` helper
- `cancel_orphaned_orders`: for each open Alpaca order **not** in `_pending_limit_orders` (by order_id) and not in `Order` table, log `ORPHANED_ORDER_CANCELLED` to AuditLog and cancel
- On ghost detection (DB ACTIVE but no Alpaca position): mark `Trade.status = 'RECONCILE_GHOST'`, do NOT re-enter, log to decision audit
- Test: simulate mid-session manual close; assert Trade marked CLOSED within 15 min

#### Summary — Order State Machine
All orders must transition through explicit states, persisted in DB:

```
PENDING_LIMIT  →  FILLED         (polled, limit hit)
               →  PARTIAL_FILLED (partial fill — record partial, cancel remainder)
               →  CANCELLED      (EOD cutoff or manual)
               →  EXPIRED        (DAY order expired)
               →  REJECTED       (broker reject — alert + clean up)

PENDING_MARKET →  FILLED         (poll get_order_status after submit)
               →  REJECTED       (broker reject)
```

**Acceptance criteria (all sub-phases):**
- Restart with 1 pending limit order → it's re-tracked from DB within 60s of startup
- Partial fill (50 of 100 shares) → Trade records 50 shares, remainder cancelled, logged
- Duplicate order attempt (crash+restart) → only 1 Order row in DB
- Manual Alpaca close detected within 15 min of periodic reconciliation
- Orphaned open order (not in our tracking) → cancelled + AuditLog within 15 min

**Dependencies:** Phase 75 (kill switch must block entries before reconciler can safely cancel orders)

### Phase 79 — Point-in-Time Index Membership (High, 3 days)
**Gap:** `SP_500_TICKERS`, `SP_100_TICKERS`, `RUSSELL_1000_TICKERS` are static Python lists reflecting ~early 2026 membership. Stocks delisted/acquired between 2021-2026 are silently absent from training, inflating walk-forward Sharpe.

**What to build:**
- `app/data/universe_history.py` with `members_at(date)` function
- `data/universe/sp500_membership.parquet` and `data/universe/russell1000_membership.parquet` — seeded from Polygon reference data or Wikipedia historical snapshots
- Update `scripts/train_model.py:580` and `scripts/walkforward_tier3.py:228` to call `members_at(fold_train_start)`
- Re-run swing walk-forward; report new Sharpe in `docs/ML_EXPERIMENT_LOG.md` (may be lower — that's the correct result)

**Acceptance criteria:** `members_at(date(2022,1,1))` returns ~500 symbols including names since delisted (e.g. WORK, PAGS). New walk-forward Sharpe documented.

### Phase 80 — Bar-12 Intraday Sensitivity Test (High, 1 day)
**Gap:** Bar 12 entry was chosen by intuition before any sweep. Phase 50 tested bars 12/24/36 but those were multi-scan offsets (all failed gate). Bars 9-15 around the current entry have never been tested. If only bar 12 shows Sharpe > 1.5 and bars 11 and 13 don't, the edge may be in-sample noise.

**What to build:**
- `scripts/bar_sensitivity.py`: takes `--entry-offset` arg, runs full walk-forward for one bar, outputs Sharpe + win-rate + trade count
- Run for offsets 9, 10, 11, 12, 13, 14, 15
- New section in `docs/ML_EXPERIMENT_LOG.md` with results table

**Acceptance criteria:** If 4 of 5 bars (10-14) pass Sharpe > 0.8 → document as robust. If only bar 12 passes → file follow-up, downgrade intraday Sharpe expectation in docs.

**Dependencies:** Phase 79 preferred first (PIT universe for honest results)

### Phase 81 — Earnings Calendar: Finnhub Primary + FMP Fallback, Fail-Closed (Medium, 1.5 days)
**Gap:** `app/calendars/earnings.py:103` uses yfinance for earnings dates. yfinance returns 401 on most requests; on exception `block_swing=False` — gate fails open silently. Both Finnhub and FMP are already paying APIs with earnings calendar endpoints.

**What to build:**
- Replace `_fetch` with Finnhub call (reuse `app/news/sources/finnhub_source.py` pattern) as primary
- On Finnhub failure → FMP fallback (`app/data/fmp_provider.py` already integrated)
- On both failing → fail-closed for swing (`block_swing=True, reason="earnings_data_unavailable"`); fail-open for intraday (2hr hold, lower exposure)
- Remove yfinance from earnings path entirely
- New scheduled task at 06:00 ET: prefetch earnings dates for all watchlist symbols
- New metric in `/api/health`: `earnings_data_freshness_pct` (% symbols with data < 24h old)
- `tests/`: mock Finnhub 503 + FMP 503; assert swing `block_swing=True`; assert intraday `block_intraday=False`

**Acceptance criteria:** Earnings gate works even when yfinance is dead. No swing entry within 2 trading days of earnings even when both primary sources fail.

### Phase 82 — Persist `_peak_equity` (Medium, 0.5 days)
**Gap:** `risk_manager.py:54` initializes `_peak_equity = None`, assigned to current equity if higher. On every uvicorn restart the drawdown high-water mark resets. If account goes from $20k → $19k and restarts, `_peak_equity` resets to $19k and the 5% drawdown rule now allows losing to $18.05k before tripping — creating a restart-loophole.

**What to build:**
- Add `Configuration` table key `risk.peak_equity` (float)
- Load on startup; fall back to Alpaca portfolio history max
- Update `_peak_equity` assignment to persist on every change
- Test: simulate restart; assert `peak_equity == max(persisted, current)` after reload

**Acceptance criteria:** Peak equity survives restart. Drawdown rule trips at the correct level (from historical peak, not restart-time equity).

### Phase 83 — Deadman Switch + External Watchdog (Medium, 1 day)
**Gap:** No external watchdog. If FastAPI process dies Friday at 14:00 ET, positions stay open indefinitely until Min notices manually.

**What to build:**
- PM writes heartbeat row to `process_heartbeat` table every 60s during market hours
- `scripts/watchdog.py` (run as separate cron, every 2 min) — if last heartbeat > 5 min old AND market open, calls `kill_switch.activate(reason="deadman: PM heartbeat stale")` via API
- Run on a cheap separate host (free-tier cloud VM or always-on laptop)

**Acceptance criteria:** Kill FastAPI process; within 7 min all positions closed.

**Dependencies:** Phase 75 (kill switch must actually block before this matters)

**Blocker for live trading:** Yes.

### Phase 84 — Integration Test: Full PM→RM→Trader Round Trip (Medium, 2 days)
**Gap:** No e2e test that drives PM→RM→Trader→Alpaca-paper with persistent state. All "integration" tests are unit tests with mocks. V25 verified: no regression tests exist for known bugs.

**What to build:**
- `tests/test_e2e_round_trip.py` — uses Alpaca paper account (real Redis) to drive one swing proposal end-to-end
- Asserts: `Trade.status='ACTIVE'` after fill; `Trade.status='CLOSED'` and `pnl` populated after stop/exit
- Run on demand only (not in normal CI — takes minutes)

**Acceptance criteria:** One test that can certify a release candidate.

### Explicit Deferred (Don't Do)
- **No new model training** until Phase 79+80 land (PIT universe + bar sensitivity)
- **No new NIS features** until Phase 77 shows NIS scores predict realized P&L (needs 2 weeks data)
- **No live-money switch** until Phases 75, 76, 78, 82, 83 all green

---

### Tier 3 — Medium-Term (2–6 weeks out)

#### Phase 64 — News as Model Features
**Precondition:** 60 trading days of point-in-time `NewsSignal` history in the DB.

Add `materiality_decayed_4h`, `direction_decayed_3d`, `article_count_4h`, `already_priced_in_score` as XGBoost training features. Walk-forward gate: Sharpe improvement > 0.10, no lookahead leakage.

**Do not start before:** mid-July 2026 at earliest.

#### Phase 65 — Source Expansion + Basic Clustering
After Phase 63 baseline measured (is news overlay adding alpha?):
- Reuters/MarketWatch RSS as second source
- Trading halts feed (NASDAQ Trader RSS) — hard gate, no LLM needed
- Simple dedup: same symbol + same hour = same event, score once

#### Phase 66 — Sonnet Escalation
After Haiku error rate is measurable from `llm_call_log`:
- Escalate to Sonnet for `materiality_score >= 0.70`, `confidence < 0.60`, or M&A/FDA/legal event types
- Budget Sonnet separately (20× cost per call)
- Gate: A/B test shows accuracy improvement

---

### Tier 3 — Medium Term (2–6 weeks)

#### Phase 75 — EOD Swing Position Review Signal
**Gap:** No logic asks "should this swing position be closed before tomorrow?" at end of day. Current exit triggers are only: stop hit, target hit, 5-day max hold, VIX tightening, T1 partial exit.

**What to build:**
- At 15:45 ET (15 min before close), re-score all held swing positions using end-of-day bars
- If score < `EXIT_THRESHOLD` AND position has been held ≥ 2 days: send EXIT signal
- Log `EOD_EXIT_WEAK_SIGNAL` to decision audit
- **Files:** `app/agents/portfolio_manager.py` (new `_eod_swing_review()` task)

#### Phase 76 — Slippage Analysis in Reporting
**Gap:** `Order.slippage_bps` is stored for every fill but never surfaced. Chronic slippage on specific symbols is invisible drag.

**What to build:**
- In `scripts/paper_trading_report.py`: add slippage section — avg bps per symbol, worst offenders
- Flag any symbol with avg slippage > 20 bps over 20+ fills for universe review
- Add `GET /api/dashboard/slippage` endpoint returning per-symbol slippage summary
- **Files:** `scripts/paper_trading_report.py`, `app/api/routes.py`

#### Phase 77 — Graceful SIGTERM / Queue Drain on Shutdown
**Gap:** If the process is killed, in-flight Redis queue proposals can replay on next start causing duplicate entries.

**What to build:**
- Register `signal.SIGTERM` / `signal.SIGINT` handlers in `app/main.py`
- On signal: set agent status="stopping", drain `trade_proposals` queue (log any pending proposals as CANCELLED), log clean shutdown event to `audit_logs`
- Restart reconciliation already handles positions — this only covers in-flight proposals
- **Files:** `app/main.py`, `app/agents/trader.py`

#### Phase 78 — Live Readiness Checklist Audit
**Gap:** `tests/test_live_readiness.py` was written before NIS, correlation gate, multi-scan intraday, and decision audit. It doesn't verify the new components are configured correctly.

**What to build:**
- Audit `test_live_readiness.py` against current system — add checks for:
  - Anthropic API key set and `macro_classify([])` returns valid dict
  - Finnhub key set and `fetch_economic_calendar()` returns without error
  - `decision_audit` table exists and is writeable
  - Correlation gate fires correctly for a known-correlated pair
  - All 3 intraday scan windows are distinct and non-overlapping
- **Files:** `tests/test_live_readiness.py`

#### Phase 81 — Dynamic Position Re-Evaluation (Market-Adaptive Hold/Exit)
**Gap:** Once a swing position is entered, exit logic is purely price-rule-based (stop/target/trailing/max-hold). The system never asks "given what the market is doing right now, does this position still make sense?" A strong ML signal that has since flipped negative, a news event on the held symbol, or a broad regime shift can make a held position stale — but nothing acts on it intraday.

**What to build:**
- Every 60 minutes during market hours, re-score each held swing position using:
  1. **ML re-score:** run inference on the held symbol with current bars → if score drops below `EXIT_THRESHOLD` (default 0.48) and position has been held ≥ 1 day: send EXIT signal
  2. **NIS re-check:** already partly done via Phase 72 (30-min NIS loop) — deepen to also check symbol-specific filings or breaking news since entry
  3. **Regime shift:** if SPY intraday regime flips to `bearish` (>1.5% drawdown from open) and position is still green, consider early partial exit to lock in gains
- Log every re-evaluation to `decision_audit` as `POSITION_REVIEW` with `outcome=HOLD|EXIT|PARTIAL_EXIT`
- **Files:** `app/agents/portfolio_manager.py` (new `_intraday_position_review()` task, runs at :15 past each hour), `app/agents/trader.py` (consume EXIT signal from queue)
- **Gate:** Don't exit on first re-score drop — require 2 consecutive weak scores (avoid noise exits)

**Philosophy:** Entries are based on pre-market signals. Intraday, the world changes. This phase makes the system responsive rather than mechanical once a position is open.

#### Phase 79 — AUC Drift Live Alert
**Gap:** The swing model retrains daily at 5 PM but silent failures aren't surfaced. If OOS AUC drops below gate threshold, no alert fires.

**What to build:**
- Post-retrain: compare new model's validation AUC to gate threshold (0.57 for swing, 0.59 for intraday)
- If AUC < threshold − 0.03: log `MODEL_DRIFT_ALERT` to `audit_logs` and send Slack alert (if configured)
- **Files:** `app/ml/training.py` (post-retrain hook), `app/agents/portfolio_manager.py`

---

### Tier 4 — Deferred (after live trading + calibration)

| Phase | Name | Why Deferred |
|---|---|---|
| 51b | Multi-scan intraday tuning | Need live data to know if 11:00 / 13:30 windows help or just add noise |
| Sentiment v2 | Upgrade news_sentiment_3d/7d swing features | Need 60 days NIS history first |
| Regime v2 | Finer-grained market regime (bull/bear/chop sub-modes) | Need live performance data to see where the model fails |
| Options flow | Polygon options unusual activity as entry filter | Polygon premium required; validate concept first |
| Intraday v2 | Retrain intraday with multi-scan data | Run 4 weeks of 3-window data, then retrain on richer intraday distribution |
| Phase 80 | True Technical Day Trader (new strategy) | Fundamentally different architecture — not an enhancement, a new agent |

#### Phase 80 — True Technical Day Trader
**Context:** The current intraday model is an ML-scored, fixed-window position trader. It holds positions for hours and exits by EOD. A true day trader operates on a completely different paradigm: event-driven entries, purely technical signals, 5–45 minute holds, 5–20 trades per day.

**What it is:**
- Separate `IntradayScalper` agent (not a modification of the current intraday PM)
- Entry triggers: VWAP breakout/reclaim, 1-min/5-min momentum, volume surge vs 20-day avg, L2 imbalance (if available)
- Hold time: 5–45 minutes; max position 90 minutes
- Universe: high-ADV liquid names (AAPL, TSLA, NVDA, SPY, QQQ, ~20–30 tickers, never 700+)
- Exits: VWAP lose, target (0.3–0.5%), hard stop (0.2%), time stop (45 min)
- No ML model scoring — pure signal-to-execution, sub-second latency target
- All positions flat by 15:45 ET, no exceptions

**Why deferred:**
- Requires L1/L2 tick data (Alpaca's 1-min bars are too coarse for 5-min scalps)
- Risk framework needs separate position limits (scalper holds many small positions simultaneously)
- Requires backtesting harness different from walk-forward on daily bars
- Current paper trading infrastructure should be validated first

**Precondition:** Phase 57 calibration complete + live trading approved (swing/intraday system stable).

---

## Part 3 — One-Page Status Board

```
SYSTEM STATE: Paper trading active as of 2026-04-28
MODELS: Swing v141 (gate passed ✅), Intraday v33 (gate passed ✅)
TESTS: 1171 passing (as of 2026-04-29 session 3; count may have grown)

COMPLETE (all shipped as of 2026-04-30):
  Foundation        — All phases 1–56, both model gates passed
  Hardening         — Phases A/B/C/67/68/69/70
  NIS               — Phases 58/59/60/63 (full Finnhub+Haiku stack)
  Risk              — Phase 71 (correlation gate)
  Audit             — Phase 61 (decision_audit table)
  Intraday          — Phase 51 (3 scan windows, cooldown, P&L cap)
  Digest            — Phase 62 (9 AM NIS pre-score, cache warm)
  Swing review      — Phase 72 (NIS re-check for held positions)
  Gap accuracy      — Phase 73 (actual open bar, auto-exit already wired)
  API visibility    — Phase 74 (5 NIS + audit endpoints)
  Audit hardening   — Gaps 1-5: feature explainability, EOD backfill, daily summary, report sections, day replay
  Institutional intraday — SPY regime gating, morning candidates cache, adaptive re-scan,
                            lunch block (11:15–13:00), staleness 90→30 min, intraday decision audit

LET IT RUN:
  Phase 57  — Paper trading calibration (run 2-4 weeks, then review)

CRITICAL / SAFETY (must fix before live money):
  Phase 75  — Wire kill switch into PM/RM/Trader + cancel open orders on activate [1 day]
  Phase 76  — Fix slippage measurement (use filled_avg_price, not get_latest_price) [0.5 days]
  Phase 78  — Order lifecycle state machine: partial fill cancel, persist pending orders to DB,
              idempotency keys, periodic mid-session reconciliation [2 days]
  Phase 82  — Persist _peak_equity across restarts [0.5 days]
  Phase 83  — Deadman switch + external watchdog [1 day]

HIGH (hardening + measurement):
  Phase 77  — Decision-audit dashboard tile (are gates actually working?) [2 days]
  Phase 79  — Point-in-time index membership (eliminate survivorship bias) [3 days]
  Phase 80  — Bar-12 sensitivity test (sweep bars 9-15, validate intraday edge) [1 day]

MEDIUM (correctness):
  Phase 81  — Earnings calendar: replace yfinance with Finnhub, fail-closed [1.5 days]
  Phase 84  — Integration test: full PM→RM→Trader round trip [2 days]

LONG TERM (after calibration, needs 60 days NIS history):
  Phase 64  — News as model features
  Phase 65  — Source expansion + dedup clustering
  Phase 66  — Sonnet escalation for high-stakes events
  Tier 4    — Intraday v2 retrain, regime v2, options flow
  Phase 85  — EOD swing review signal (15:45 ET weak-score exit) [was Phase 75]
  Phase 86  — Kill switch mode selector: named modes A/B/C (hard-block-exits / auto-flatten / full-freeze)
              Currently Phase 75 implements mode A (hard block entries, allow exits).
              Add a `kill_switch.activate(mode="flatten")` param so operator can choose per-situation.
  Phase 87  — Graceful SIGTERM / queue drain [was Phase 77]
  Phase 88  — Live readiness checklist audit [was Phase 78]
  Phase 89  — AUC drift live alert [was Phase 79]
  Phase 90  — Tax + P&L Impact Review (pre-live gate, NJ taxable account)
              Before any real-money promotion: quantify wash sale exposure from paper trading history
              (how many times did we exit a loss and re-enter the same symbol within 30 days?),
              compute after-tax Sharpe assuming 37% Fed + 10% NJ on short-term gains, and decide
              whether to enable hard wash-sale blocking (Phase 7b option) before going live.
              Also: confirm account type (cash vs margin) and T+1 settled-cash impact.
              Revisit: when paper trading is stable and 4+ weeks of clean data exists.
  Phase 91  — True technical day trader (new IntradayScalper agent, post-live-approval) [was Phase 80]

OPEN QUESTIONS FOR MIN (from code review — must answer before proceeding):
  Q1: Universe scope after Phase 79 — retrain on PIT-R1000, keep SP-100-PIT gate, or accept bias?
  Q2: Kill switch style — (a) hard block, (b) allow exits, (c) hard block + auto-flatten?
  Q3: MetaLabel disposition — wire in live, delete docs, or train v2?
  Q4: Going-live gate — after phases 75/76/78/82/83 green only, or also require 4 weeks clean paper?
  Q5: Earnings gate fail behavior — fail-closed always, or fail-open for intraday?
```
