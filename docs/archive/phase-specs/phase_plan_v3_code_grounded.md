# MrTrader — Phase Plan v3 (Code-Grounded)

**Author:** Quant review against actual repo state (commit 5174e28, branch `feat/phase-58-59-calendar-gates`)
**Date:** 2026-05-01
**Scope:** Verification of 31 reviewer assumptions against code, then a re-prioritised backlog.

All claims below are anchored to file:line. Where code disagrees with the previous reviewers, that is stated bluntly.

---

## Section A — Executive Summary

### TL;DR

MrTrader is closer to "well-instrumented data pipeline + decent ML stack" than to "production trading system." The labelling, walk-forward harness, embargo, regime gates, earnings/macro calendars, and audit logging are real and largely correct. The **execution and safety layer** is where the system breaks down: the kill switch is decorative (no agent ever checks `kill_switch.is_active`), the MetaLabel model documented as a live filter is not wired into the live PM at all (only into backtests), slippage is measured against `get_latest_price` AFTER the market order completes (so it's noise, not signal), training universes are built from *current* membership (survivorship bias), and the bar-12 intraday entry was never sensitivity-tested. The walk-forward Sharpe numbers are cleaner than they look in some places (proper embargo) and dirtier in others (current-membership universes, intraday top-300 fold variance). Paper-trading since 2026-04-28 is generating real outcome data — that data is the most valuable thing to instrument right now, not the model.

### Top 3 things to do this week (ranked by risk-reduction × completability in 5–15 hrs)

1. **Wire kill switch into PM, RM, Trader main loops + executor.** Currently `KillSwitch.activate()` closes positions but new proposals continue to flow through. ~3 hrs. Fixed surface area.
2. **Fix slippage measurement for market orders.** Today `filled_price = alpaca.get_latest_price(symbol) or intended_price` (trader.py:646) — this fetches a *new* quote AFTER fill and is not the actual fill price. Use the order's `filled_avg_price` from `get_order_status`. ~2 hrs. Without this, every other execution-quality conclusion is wrong.
3. **Build a paper-trading dashboard tile that compares decision-time NIS / model-score / regime gate against realized 1-day and 4-day P&L.** The `decision_audit` table (database/models.py:291) already records all this; nothing yet reads it. This is what tells you whether the gates are actually filtering well or just suppressing volume. ~4 hrs.

### Top 3 things NOT to do (deferred until fundamentals are solid)

1. **Don't train a v120 swing model or a v30 intraday model.** v119 / v29 already passed gates. Adding features or trying lambdarank again won't move the needle until you confirm the MetaLabel-not-actually-running issue isn't hiding even more leakage.
2. **Don't go live (paper → real money) until kill switch is enforced AND wash-sale/PDT enforcement is validated end-to-end on the paper account.** Currently both are advisory.
3. **Don't add more LLM/news features.** Tier 1 + Tier 2 NIS just shipped. Let it run two weeks and measure whether `news_action_policy` actually predicts `outcome_pnl_pct` in `decision_audit` before adding more news plumbing.

### Single biggest risk the reviewers missed

**The kill switch is decorative.** `kill_switch.is_active` is checked in zero agent loops (verified by `grep` in `app/agents/`). It is referenced only in `app/api/routes.py` (health endpoints), `app/main.py` (startup logging), and `app/live_trading/readiness.py`. Calling `activate()` will market-close existing positions but will not stop the next proposal from flowing PM → RM → Trader → Alpaca order. On a real account during a flash crash this would be a multi-thousand-dollar bug.

---

## Section B — Verification Results (V1–V31)

| ID | Task | Reviewer Assumption | What Code Actually Shows | Verdict |
|----|------|---------------------|--------------------------|---------|
| V1 | path_quality weights hardcoded? Tuned? | 1.0 / 1.25 / 0.25 hardcoded | Hardcoded at `app/ml/training.py:254` (swing) and `app/ml/intraday_training.py:853` (intraday, but coefficient is **0.50 not 1.25** for stop_pressure — Phase 47-3 change). Comment at training.py:211 documents intent. No tuning script exists; chosen via Phase 45 grid over `(stop_mult, target_mult)` only, not `(upside_w, stop_w, close_w)`. Intraday weights diverged from swing without re-running the swing grid. | CONFIRMED hardcoded; PARTIALLY_RIGHT on tuning |
| V2 | Walk-forward = 3-fold TimeSeriesSplit with embargo? | Yes, 3-fold | **No, two different things.** `scripts/walkforward_tier3.py:262-269` uses **expanding-window manual fold construction** (default 3 folds, segment_days = total_years*365/(folds+1)). The `TimeSeriesSplit(n_splits=3)` at `app/ml/training.py:1505` is only used inside Optuna HPO (hyperparameter search inside one training run), not for the OOS Sharpe gate. Embargo `EMBARGO_WINDOWS = max(1, round(FORWARD_DAYS / STEP_DAYS))` exists at `app/ml/training.py:45` and is applied at line 607 — **=1 window=5 trading days**, which is exactly the forward-label horizon. Not purged; just gapped. | PARTIALLY_RIGHT |
| V3 | Swing trains on SP-100 (~81), infers on R1000 (~430)? Intraday R1000+R1000? | As stated | **Three different universes are in use simultaneously.** Swing default training in `train_model.py:580` uses `RUSSELL_1000_TICKERS` (~620 unique). The walk-forward script `walkforward_tier3.py:228` uses `SP_100_TICKERS` (~95). Live PM at `portfolio_manager.py:551` falls back to `SP_500_TICKERS` (~480). Intraday training and inference both use `RUSSELL_1000_TICKERS`. The reviewer's "SP-100 train, R1000 infer" mismatch is partially correct: walk-forward is on SP-100 (passes the gate), but the actual deployed swing model `v119` was trained on what was passed to `train_model.py` — likely SP-500 or R1000 depending on operator. Docs at `ML_MODELS_REFERENCE.md:15` claim "SP-100 (~81 symbols)" — this contradicts `train_model.py:580` default. | WRONG (multi-universe inconsistency worse than reviewer claimed) |
| V4 | Feature leakage in 84-feature swing set | 10 spot-checks | Spot-checked: `regime_score` is set to constant 0.5 in training (training.py:616) to avoid leakage — good. `vix_level/vix_percentile_1y` use `as_of_date` mask on `vix_history` (features.py:475) — good. `momentum_*`, RSI, MACD, EMAs, ATR — all trailing window — good. `fmp_surprise_1q/days_since_earnings` use `get_fmp_features_at(symbol, pit_date)` (features.py:566) — good IF FMP provider respects the date. **Bug: `get_earnings_history(symbol)` (fundamental_fetcher.py:537) takes no `as_of_date` and computes `days_since_earnings` from `datetime.now()` (line 587) — leakage.** Mitigated because `earnings_surprise`, `days_since_earnings`, `earnings_surprise_1q`, `earnings_surprise_2q_avg` are all in `PRUNED_FEATURES` (training.py:58-75) and stripped before training. So in practice the leaky features are not in the v119 84-feature set. | PARTIALLY_RIGHT (latent bug, currently masked by feature pruning) |
| V5 | Survivorship bias — point-in-time membership? | Yes (probably) | **No.** `SP_500_TICKERS`, `SP_100_TICKERS`, `RUSSELL_1000_TICKERS` are static Python lists in `app/utils/constants.py` (lines 7, 145, 141). They reflect membership ~early 2026. Comment at line 80 explicitly says "yfinance will silently skip any delisted or unresolvable symbols." There is no point-in-time membership table. Stocks that *were* in the universe in 2021 and got delisted, acquired, or relegated by 2026 are silently absent from training. This biases training results upward. | WRONG (not point-in-time; reviewer was right to flag) |
| V6 | Bar-12 intraday entry chosen via search? | Likely overfit | **No systematic sweep.** `MIN_BARS = 12` set in `app/ml/intraday_features.py:86` and `scripts/backtest_intraday.py:44`. Phase 50 (commits `2f8d8bc`, `c7cd1af`) added multi-offset training with `ENTRY_OFFSETS = [12, 24, 36, ...]` but commit `19d2f15` reverted to `[12]` after the multi-scan v30 model failed the gate. So bars 12, 24, 36 were tried — but the *single-scan bar 12* was the original choice, made before any sensitivity test. No documented results comparing bar 9, 11, 13, 15. The reviewer concern is valid: bar 12 might be cherry-picked. | CONFIRMED reviewer concern |
| V7 | MetaLabel R²=0.059, corr=0.286 — still active as confidence gate? How often does it change a decision? | Active in live PM | **Not in live PM.** `grep MetaLabel\|meta_model\|should_enter` in `app/agents/` returns zero matches in `portfolio_manager.py`, `risk_manager.py`, `trader.py`. MetaLabel is only referenced in `app/backtesting/agent_simulator.py` and `app/backtesting/intraday_agent_simulator.py`. Docs claim ~26% of entries are filtered (`ML_MODELS_REFERENCE.md:97`) — this is a **backtest-only** statistic. Live PM uses (a) ML score threshold, (b) `_market_regime_allows_entries()` abstention gate (VIX>=25 or SPY<MA20), and (c) Phase 59 macro window gate. No MetaLabel anywhere live. Documentation is misleading. | WRONG — MetaLabel is not in production |
| V8 | In-memory state inventory | Mixed | PM has many in-memory daily flags (`_analyzed_today`, `_selected_today`, `_intraday_windows_run`, `_morning_intraday_candidates`, `_swing_proposals`, `_pending_approvals`, `_last_swing_features`, `_last_intraday_features`) at `portfolio_manager.py:90-100, 255-275`. Trader keeps `active_positions: Dict` and `approved_symbols: Dict` and `_pending_limit_orders: Dict` (trader.py:55-60). RM keeps `_open_intraday_count: int` and `_peak_equity: float` (risk_manager.py:54-56). On startup: kill switch state restored from DB config (`kill_switch.py:32`); PM daily-flags restored from DB AgentDecision log (`portfolio_manager.py:212-238`); Trader rebuilds `active_positions` from Alpaca + DB Trade reconciliation (trader.py:64-226); Startup reconciler creates synthetic Trades for untracked Alpaca positions. **Not restored:** `_swing_proposals`, `_pending_limit_orders` (cleared on restart — Alpaca limit orders may fill into untracked positions, hence reconciler line 70-113), `_pending_approvals` (proposals in Redis are flushed at startup, main.py:118-124), RM `_peak_equity` (resets to current equity — silent reset of drawdown high-water mark). | CONFIRMED partial restore; key gap = `_peak_equity` resets |
| V9 | Order placement flow, partial fill, idempotency | Some form of state | `Order` model has `status: PENDING/FILLED/FAILED` (database/models.py:46). No explicit `PARTIALLY_FILLED` state. No idempotency key (no Alpaca `client_order_id` recorded — verified absence by grep). Limit order polling (trader.py:768) handles `partially_filled` by treating it the same as `filled` and recording the partial qty as `filled_qty`, but never re-polls for the remainder. Market orders never query `get_order_status` at all — they assume fill and use `get_latest_price` as a fill-price proxy (line 646). | PARTIALLY_RIGHT — major gap on partial fills + idempotency |
| V10 | Reconciliation logic, RECONCILE_GHOST handling, schedule | Startup-only | Confirmed startup-only (`app/main.py:107`). Two distinct reconcilers: `app/startup_reconciler.py:reconcile()` flags ghost positions (DB ACTIVE but no Alpaca pos) by setting `Trade.status = "RECONCILE_GHOST"` (line 56) and writing `AuditLog` — does NOT close them, does NOT alert. Untracked Alpaca positions get a placeholder Trade with default 2% stop/6% target (line 89-91). `app/agents/trader.py:_reconcile_positions()` runs once at trader startup, more thorough — uses `generate_signal()` to compute proper stop/target. No periodic reconciler exists. If Alpaca state and DB diverge mid-session, nothing notices until next restart. | CONFIRMED startup-only |
| V11 | `_restore_daily_flags` — DB or in-memory authoritative after fix? | Unclear | DB authoritative on startup (`portfolio_manager.py:222-225` queries `AgentDecision` rows for today). After startup, in-memory flags become authoritative until midnight reset (line 252-276). Bug history: pre-fix, flags only existed in memory and a uvicorn restart at 14:00 ET would re-trigger 09:50 swing proposals. Fix correctly restores from DB. Limitation: only three flags are restored (`_analyzed_today`, `_selected_today`, `_premarket_run_today`) — `_intraday_windows_run`, `_benchmark_recorded_today`, `_eod_jobs_run_today`, `_weekly_report_generated_today`, `_retrained_today` are NOT restored. A restart between 09:45 and the lunch block re-fires intraday scans. | PARTIALLY_RIGHT — fix is incomplete |
| V12 | Runaway scanner fix — guard? | Unclear | Two protections: (a) `_intraday_windows_run` set is checked at line 340 — once a window runs, it won't re-run that day (in-memory only, see V11); (b) adaptive re-scan has `_last_adaptive_scan_at` timestamp gate (`portfolio_manager.py:374`, 1-hour cooldown via `_time.monotonic()`). Both are weekday + market-hours gated. **Weakness:** if uvicorn restarts between 09:45 and 13:00, `_intraday_windows_run` is empty so all three windows can re-fire (V11 limitation propagates here). No global rate limit on `select_intraday_instruments()` calls. | PARTIALLY_RIGHT |
| V13 | Position sizing — 10-15% notional, stop-distance? | 10-15% | **Sizing is risk-based, not notional.** `app/strategy/position_sizer.py:size_position()` computes `shares = floor((account_equity * 0.02 * conviction_mult) / (entry - stop))` then caps at min(shares_from_risk, 90% available cash, 10% account equity). With $20k equity and 2% risk = $400 dollar-risk per trade. Conviction multiplier ranges 0.75×–1.25× based on ML score (lines 21-37). The `MAX_POSITION_PCT = 0.10` (10%) hard cap binds when stops are tight — most positions hit this cap, not the risk cap. So *effective* position size is closer to 10% notional than 2%-risk-implied — reviewer's intuition was right but the mechanism is different. | PARTIALLY_RIGHT (cap binds, not risk-based as designed) |
| V14 | Slippage logging — what is it computed against? Logged but unused? | Logged but unused | Computed at `app/agents/trader.py:647` (market) and 771 (limit). For limit orders, `intended = pending["intended_price"]` which was set as `quote.ask` at order placement time (line 615) — that's correct. For market orders: `filled_price = alpaca.get_latest_price(symbol) or intended_price` (line 646) where `intended_price = result.entry_price` set at signal-generation time minutes earlier. **This is wrong twice over:** (1) `get_latest_price` returns a NEW Minute bar AFTER the order placed — that's not the fill price, that's the next-tick observed price. (2) `intended_price` is the signal-time price, often 5-30 minutes stale. So `slippage_bps` for market orders is closer to "post-fill price drift" than to actual slippage. Persisted to `Order.slippage_bps` (database/models.py:50) and logged via `log_decision`. **No code reads it** — verified by `grep slippage_bps` in `app/`. Not used to gate, alert, or back-pressure entries. | WRONG measurement for market orders + not used |
| V15 | Sector concentration cap, sector source | 30% cap, GICS | **20% cap, hand-curated sector dict.** `RiskLimits.MAX_SECTOR_CONCENTRATION_PCT = 0.20` (`app/agents/risk_rules.py:17`). Factor concentration cap = 60% (`max_factor_concentration: float = 0.60`, also using sector as factor). Sector source = `SECTOR_MAP` dict in `app/utils/constants.py:172` — manually-maintained Python dict mapping ticker → sector string (not GICS, not from any API). Unknown symbols default to "UNKNOWN" via `.get()` and may slip through any sector check that matches "UNKNOWN" with itself. Also: dict probably out-of-date as the universe was last refreshed manually. | WRONG (20% not 30%, not GICS) |
| V16 | Correlation gate — 0.75 pairwise 30-day, lookback, computation, source | As stated | Threshold = 0.75 (`risk_rules.py:26`). Lookback = **65-day fetch, ~63-day correlation** (`risk_manager.py:543`, fetches 65 daily bars to get ~64 returns after `pct_change().dropna()`, takes overlap with each open position). Asks for >=20 bars on the new symbol and >=10 overlap with each open position. Data source = Alpaca `get_bars(timeframe="1Day", limit=65)` — IEX feed. Computed at proposal time *live* — not pre-cached. With N open positions, this is N+1 separate API calls per proposal; for an empty portfolio it's 1 call. Reviewer claimed "30-day correlation" — it's actually 60+ day. | WRONG (60-day, not 30-day) |
| V17 | Daily loss limit — measured against what? | -2% rolling? | -2% (`MAX_DAILY_LOSS_PCT = 0.02`, `risk_rules.py:18`). Source = `RiskMetric.daily_pnl` from `risk_metrics` table, queried by today's date string at `risk_manager.py:691`. **Realized P&L only** (the column is updated when trades close, not from unrealized). The "today" boundary is `str(date.today())` which uses local server time (UTC if container, ET if Windows host) — not market-day-aware, so a 23:30 ET loss flips into "next day" at 00:00 server-local. Unrealized losses do not contribute. So a portfolio that's down 5% unrealized but flat realized today is not blocked. | PARTIALLY_RIGHT — realized only, fragile date boundary |
| V18 | Kill switch — what does it block? In-flight orders? Remote trip? | Should block trades | **Decorative.** `KillSwitch.activate()` (kill_switch.py:63) closes all Alpaca positions and sets `_active = True` and persists to config_store. `is_active` is checked NOWHERE in agent loops (verified with `grep kill_switch app/agents/`). PM/RM/Trader will continue accepting and executing proposals after activation. The only gates that look at it are: `app/api/routes.py:505` (returns 503 on /trades/execute), `app/api/routes.py:511` (status display), `app/main.py:189` (degraded health flag), `app/live_trading/readiness.py:97` (paper-to-live promotion check). There is a `/api/kill-switch/activate` POST endpoint (routes.py:731) so remote trip works via API. In-flight Alpaca orders are NOT canceled — `activate()` only iterates `get_positions()` and submits sell market orders for held quantities, ignoring open orders. | CONFIRMED CRITICAL GAP |
| V19 | Data feed — IEX-only? | IEX | Confirmed `feed="iex"` for bar requests (`app/integrations/alpaca.py:271, 285`). Quotes via `get_quote` use `StockLatestQuoteRequest` with no explicit feed parameter (line 324) — defaults to whatever Alpaca free tier provides, also IEX. Live fills go through Alpaca paper engine (in-process, simulated). Polygon is used for training only (provider at `app/data/polygon_provider.py`). FMP for fundamentals. yfinance for VIX (when not cached) and earnings calendar. **No SIP feed anywhere** — NBBO is IEX-only, which represents ~3% of consolidated tape volume during RTH and is wider than SIP NBBO. | CONFIRMED |
| V20 | Polygon cache TTL, invalidation | 23h | TTL = **24h** not 23h (`CACHE_TTL_HOURS = 24` at `app/data/intraday_cache.py:26`). FMP provider uses 24h (`_CACHE_TTL = 86_400` at fmp_provider.py:30). Polygon financials 24h (polygon_financials.py:32). Cache invalidation: file mtime check; if file older than 24h, `cache_is_fresh()` returns False and caller fetches via Polygon REST. **If refetch fails:** `load()` returns None; calling code (e.g. `IntradayModelTrainer._cache_is_fresh`, `intraday_training.py:312`) logs warning and skips the symbol — silently drops it from training. Backtests continue with degraded universe; no alert. | PARTIALLY_RIGHT (24h not 23h; silent degradation) |
| V21 | Corporate actions handling | Unclear | No explicit handling code anywhere. Polygon REST requests pass `adjusted=true` (polygon_provider.py:223, 267) — splits are handled, dividends are partially handled (Polygon's "adjusted" includes split-adjustments; cash dividends not back-adjusted into prices). yfinance uses `auto_adjust=True` (yfinance_provider.py:129). Alpaca daily bars are split-adjusted by default — but the code does not verify or normalize. **No reconciliation between providers' adjustment conventions.** A stock that splits during a Polygon-trained / Alpaca-inferred period would have a price discontinuity in the live feature path. No tests for this. | INCONCLUSIVE — provider defaults relied on, no internal handling |
| V22 | Earnings calendar — Finnhub gate, 3-day blackout, refresh | As stated | **Wrong source.** `app/calendars/earnings.py:103` uses `yfinance Ticker.calendar`, NOT Finnhub. Blackout = **2 trading days swing, 1 day intraday** (lines 20-21), not 3. Cache TTL = 1 hour per symbol (line 23). Finnhub IS the source for the macro/economic calendar (`app/news/sources/finnhub_source.py` per phase 58 docs), but earnings dates use yfinance — the same yfinance that returns 401 on most endpoints today (per ML_MODELS_REFERENCE.md:293). On a 401, `_fetch` returns None and `block_swing` becomes False — earnings gate **fails open silently.** | WRONG (yfinance not Finnhub, fails open on 401) |
| V23 | NIS API resilience — Anthropic down | Unclear | Fail-open. `app/news/intelligence_service.py:_build_macro_context()` returns `MacroContext.neutral()` on any exception (line 84) and logs warning. `_build_stock_signal()` returns `NewsSignal.neutral()` similarly. Neutral context = `block_new_entries=False, sizing_factor=1.0` — system continues trading as if there were no macro news. No alert escalation. The 1-day macro cache and 1-hour stock cache mean a brief Anthropic outage has minimal impact, but a sustained outage = sustained fail-open. | CONFIRMED — fail-open with no escalation |
| V24 | Test coverage by module | Unclear | CI gate = 58% line coverage (`.github/workflows/ci.yml:90` — `--cov-fail-under=58`). 66 test files in `tests/`. Heavy coverage on backtesters, walk-forward, feature engineering, NIS, calendars (test_phase_58_59.py, test_nis.py). Light/missing coverage on: `app/agents/portfolio_manager.py` 2,309 lines with most paths exercised only by integration test, kill_switch closure-loop not tested under partial fill, reconciler edge cases, Trader's `_execute_partial_exit` (unit-tested only via mocks). No e2e test that drives PM→RM→Trader→Alpaca-paper round trip with persistent state. Mostly unit tests with heavy mocking. | PARTIALLY_RIGHT — 58% gate is not strict |
| V25 | Bug regression tests for known bugs | Unclear | `_restore_daily_flags` — no explicit regression test (grep returns 0 matches in tests/). Runaway scanner — no explicit regression test for in-memory `_intraday_windows_run` not being persisted across restarts. Kill switch — only tested as a config persistence (test_integration_db.py:133-134), not for actual blocking effect (because there is no blocking effect to test, see V18). The "use patch.object on singleton instances for earnings/macro gate mocks" commit (5174e28) IS a regression fix but not a new bug regression test — it's a flake fix. | WRONG — no real regression suite for known bugs |
| V26 | Walk-forward in CI | Unclear | `tests/test_phase22_walkforward_tier3.py` is included in CI (237 lines, mostly unit tests of `FoldResult.passed_gate` and `WalkForwardReport` aggregation logic — uses MagicMock for the actual simulation). The full `scripts/walkforward_tier3.py` is NOT run in CI; comments in the script note "intraday: --days 730" which would take >>10 minutes. The CI `backtest-regression` job (ci.yml:100) runs only on `main` branch and is restricted with `-k "signal or sizer or check_exit or position"` — it skips actual walk-forward. So the gate (Sharpe>0.8 swing, >1.5 intraday) is enforced only on operator request, not in CI. | WRONG — no walk-forward in CI |
| V27 | Account type — paper, cash or margin? | Paper (margin sim) | `app/integrations/alpaca.py:67-72` instantiates `TradingClient(paper=True)` when `settings.trading_mode != "live"`. Paper account type at Alpaca = margin account simulator by default. Means PDT rules apply (4 day-trades in 5 days under $25k); compliance code at `app/agents/compliance.py:25-28` enforces PDT correctly. T+1 settlement is simulated by `compliance_tracker.record_sale_proceeds()` (compliance.py:191) tracking unsettled cash and `settled_buying_power()` deducting it. Code is right; reviewer assumption correct. | CONFIRMED |
| V28 | PDT day-trade counter — enforced? | Unclear | Counter exists (`compliance.py:71-100`), enforced at `risk_manager.py:298` (intraday-only check, blocks new intraday proposal if `equity < $25k AND day-trades-in-window >= 2`). Notice: SEC PDT limit is **3 day-trades in 5 business days** for accounts under $25k (4 = pattern day trader designation). Code uses `PDT_MAX_DAY_TRADES = 3` and `PDT_WARN_AT = 2`. The warn-at-2 means it blocks at 2, so you'd never hit 3. Conservative but correct. State loaded from DB on startup via `load_day_trades_from_db()` (line 118). Enforced only for `trade_type == "intraday"`. **A swing position closed same-day (e.g. stop-out within hours) is recorded as day trade (trader.py:1207-1210) but NOT checked at proposal time** because the PDT check is gated on `proposal["trade_type"] == "intraday"`. | PARTIALLY_RIGHT — same-day swing closes not pre-checked |
| V29 | Wash sale handling — duplicate-check rule prevents re-entry within 30 days? | Unclear | **No.** `compliance.check_wash_sale()` (compliance.py:150) is called at `risk_manager.py:520` and produces a *non-blocking warning* in the reasoning JSON. Re-entry within 30 days of a loss is allowed. The duplicate-check that exists is `if symbol in self.active_positions` (trader.py:427) — only same-position dedup, not wash-sale-window dedup. No tax-aware logic anywhere. | WRONG — wash sale is advisory only |
| V30 | After-hours — system started 03:00 ET, what runs? | Unclear | At 03:00 ET on a weekday: PM main loop runs but most `if is_weekday and self._in_window(now, ...)` gates fail (premarket starts 08:00, intraday windows 09:45/10:45/13:00). RM consumes Redis with no time gate. Trader runs `_scan_cycle` continuously: `_check_exit` is suppressed via `not during_market_hours` (trader.py:1032); `_check_entry` for intraday is gated by `now_et.hour > 15 or ==15.minute>=0` (trader.py:434) — that blocks late-day intraday but **does not block 03:00 entries** (because `15 > 3` is False — wait, `now_et.hour > 15` is False at 3 AM so the guard passes through and entries can fire). For swing: limit orders are placed with default TIF=DAY, queued by Alpaca, and fill at 09:30 open. **There is no global market-hours guard on entry execution.** | INCONCLUSIVE for swing (limits queue), CONFIRMED gap for intraday (no AH/PM block) |
| V31 | Operator-unavailable mode — deadman switch, watchdog? | None expected | None. `grep deadman\|watchdog\|external.*monitor` returns no matches. Heartbeats exist (PM logs `PM heartbeat HH:00 — flags: ...` hourly, `portfolio_manager.py:281-290`) but they only write to logs — nothing checks for missing heartbeats and nothing escalates. Min must monitor uvicorn process himself. If the FastAPI process dies on a Friday afternoon, no positions get closed, nothing alerts, and Min finds out Monday. | CONFIRMED gap |

---

## Section C — Updated Phase Plan (Ranked)

Numbered by execution order. Effort estimated for 5–15 hr/week part-time. Phase numbers continue from completed Phase 74; this is **Phases 75–84**.

### Phase 75 — Wire Kill Switch (Critical, 1 day)
- **Importance:** Critical — pre-go-live blocker
- **Effort:** 1 day
- **Dependencies:** None
- **Acceptance criteria:** PM `_send_swing_proposals`, PM `select_intraday_instruments`, RM `_validate_trade`, Trader `_check_entry` each return early with `KILL_SWITCH_ACTIVE` decision logged when `kill_switch.is_active` is True. Test: activate kill switch via API, send a fake proposal through Redis, assert Order count does not increase. Test: ensure `kill_switch.activate()` cancels all open Alpaca orders before submitting close orders (currently it just closes positions — open buy orders will still fill).
- **Success:** Kill switch activation results in zero new entries within 30 seconds.
- **Failure:** Kill switch tripped, a proposal still becomes a filled order. (This is current state.)

### Phase 76 — Fix Slippage Measurement for Market Orders (Critical, 0.5 days)
- **Importance:** Critical — every execution-quality decision is downstream of this
- **Effort:** 0.5 days
- **Dependencies:** None
- **Acceptance criteria:** `trader.py:_execute_entry` market path calls `alpaca.get_order_status(order_id)` after submission, retries up to 3× with 0.5s sleep until `status in ('filled', 'partially_filled')`, then uses `filled_avg_price` from the status response — not `get_latest_price`. `intended_price` for market orders captured from `alpaca.get_quote(symbol)["mid"]` at the moment of order submission, not from `result.entry_price` set minutes earlier. Acceptance: re-run a paper-trading day; mean abs slippage on intraday market orders should drop from current high values to <30 bps.
- **Success:** Can answer "what's our average slippage" with one DB query and trust the answer.
- **Failure:** Continue making decisions on noise.

### Phase 77 — Decision-Audit Dashboard (High, 2 days)
- **Importance:** High — answers "are the gates actually working?"
- **Effort:** 2 days
- **Dependencies:** Phase 76 (so realized P&L feeding `outcome_pnl_pct` is trustworthy)
- **Acceptance criteria:** New dashboard page `/audit` shows: (a) win-rate by `block_reason` from `decision_audit` (which gates are over-blocking?), (b) avg `outcome_pnl_pct` for `final_decision='enter'` vs blocked, grouped by `news_action_policy` and `macro_risk_level`, (c) chart of `model_score` bucket vs realized 4h/1d return, (d) histogram of `news_sizing_multiplier` × realized P&L. EOD job populates `outcome_pnl_pct`, `outcome_4h_pct`, `outcome_1d_pct` (currently nullable — need to verify backfill exists). Acceptance: at least 2 weeks of paper data populated, can read off "NIS sizing multiplier 0.7-0.8 correlates with X bps lower realized return."
- **Success:** Min can sit down for 5 min on a Saturday and see whether NIS or macro gate is justifying its complexity.
- **Failure:** Pretty graphs that nobody acts on.

### Phase 78 — Periodic Reconciliation + Open-Order Sync (High, 1 day)
- **Importance:** High — startup-only reconciliation is a known gap
- **Effort:** 1 day
- **Dependencies:** Phase 75 (so reconciler can react to kill-switch state)
- **Acceptance criteria:** Add `_reconciliation_loop` task to PM that runs every 15 minutes during market hours, calling `startup_reconciler.reconcile()` plus a new `cancel_orphaned_orders()` that compares Alpaca open orders vs DB `orders` table and cancels orders not tracked locally. Persist `_pending_limit_orders` to DB so it survives uvicorn restarts. Acceptance: simulate restart with 1 pending limit order in Alpaca; verify it is cancelled OR re-tracked within one reconciliation cycle.
- **Success:** Drift between Alpaca and DB never persists more than 15 minutes.
- **Failure:** Same as today — drift persists indefinitely.

### Phase 79 — Point-in-Time Universe (High, 3 days)
- **Importance:** High — silent survivorship bias is inflating walk-forward Sharpe
- **Effort:** 3 days
- **Dependencies:** None (parallelizable with Phase 75/76/77)
- **Acceptance criteria:** Build `app/data/universe_history.py` that reads a CSV/parquet of (symbol, date_added_to_index, date_removed_from_index) for SP-500 + Russell-1000, populated from Polygon's reference data (or one-time manual seed from Wikipedia historical snapshots). Replace `SP_500_TICKERS` references in `train_model.py` and `walkforward_tier3.py` with a function `members_at(date)`. Re-run swing walk-forward and intraday walk-forward; report new Sharpe. **Be honest if it drops** — that's the whole point.
- **Success:** Walk-forward gate metrics now reflect investable universe; can defend numbers under scrutiny.
- **Failure:** Numbers worsen and Min has to decide whether to re-engineer or ship the worse number.

### Phase 80 — Bar-12 Sensitivity Test (High, 1 day)
- **Importance:** High — answers "is the intraday edge real or p-hacked?"
- **Effort:** 1 day
- **Dependencies:** Phase 79 (point-in-time universe should land first)
- **Acceptance criteria:** New script `scripts/bar_sensitivity.py` runs walk-forward at bar 9, 10, 11, 12, 13, 14, 15 (ENTRY_OFFSET param). Report avg Sharpe per bar. Acceptance: if bars 10-14 all > 0.8 → robust, ship. If only bar 12 > 0.8 → flag as overfit, downgrade intraday Sharpe expectation in docs and treat live results with corresponding skepticism.
- **Success:** Edge is robust across nearby entry times.
- **Failure:** Edge is brittle, intraday model needs rework or sizing reduction.

### Phase 81 — Earnings Calendar to Finnhub + Fail-Closed (Medium, 1.5 days)
- **Importance:** Medium — current gate fails open silently on yfinance 401
- **Effort:** 1.5 days
- **Dependencies:** None
- **Acceptance criteria:** Replace yfinance source in `app/calendars/earnings.py:_fetch` with Finnhub (already integrated for macro). Default behavior on fetch failure: **fail closed** for swing (block entry) with `reason="earnings_data_unavailable"`. Add daily prefetch warm-up at 06:00 ET so all watchlist symbols have fresh earnings dates before 08:00 premarket. Add metric: % of symbols with stale (>24h) earnings data, alert if >20%.
- **Success:** Earnings gate works even when yfinance is dead.
- **Failure:** Quietly trades through earnings releases.

### Phase 82 — Real `_peak_equity` Persistence + Drawdown Authority (Medium, 0.5 days)
- **Importance:** Medium — silent drawdown high-water mark reset on restart
- **Effort:** 0.5 days
- **Dependencies:** None
- **Acceptance criteria:** Persist `_peak_equity` to `Configuration` table (key=`risk.peak_equity`, value=float, updated_at). Load on startup. Use Alpaca portfolio history if DB is empty (mirrors the pattern from commit `3ca26ac`). Acceptance: kill uvicorn at 90% of $20k; restart; assert peak_equity is still $20k * historical max, not $18k.
- **Success:** Drawdown high-water mark survives restarts; account drawdown rule works as designed.
- **Failure:** Restart laundering — peak resets, drawdown rule never fires.

### Phase 83 — Deadman Switch + External Watchdog (Medium, 1 day)
- **Importance:** Medium — Min is part-time; system needs to fail safe when he's not watching
- **Effort:** 1 day
- **Dependencies:** Phase 75 (kill switch must actually block)
- **Acceptance criteria:** PM writes a heartbeat row to a `process_heartbeat` table every 60 seconds during market hours. New script `scripts/watchdog.py` (run as separate cron, e.g. every 2 min) reads the heartbeat; if last heartbeat > 5 min old AND market open, calls `kill_switch.activate(reason="deadman: PM heartbeat stale")` via the API endpoint. Cron is set up on Min's laptop or a free-tier cloud VM independent of the FastAPI host. Acceptance: kill the FastAPI process; within 7 minutes positions are closed.
- **Success:** A crashed system fails safe instead of fail-running.
- **Failure:** FastAPI dies Friday at 14:00 ET, nothing happens until Monday.

### Phase 84 — Integration Test: Full PM→RM→Trader Round Trip (Medium, 2 days)
- **Importance:** Medium — V25 found no real e2e regression suite
- **Effort:** 2 days
- **Dependencies:** Phase 78 (so `_pending_limit_orders` is persistable)
- **Acceptance criteria:** New test file `tests/test_e2e_round_trip.py` that uses Alpaca paper account (not mocks) and a real Redis (compose service). Drives one swing proposal end-to-end with: ML score above threshold, regime gate clear, RM approves, Trader places limit order, simulates fill via Alpaca paper market move, asserts `Trade.status='ACTIVE'`, then triggers `_force_close_intraday`, asserts `Trade.status='CLOSED'` and `pnl` populated. Run on demand only, not in normal CI.
- **Success:** One test that can certify a release candidate.
- **Failure:** Continue shipping on the unit-test wishful-thinking pyramid.

### Deferred (Explicit Don'ts)

- **No new model training** until Phase 79 + 80 land. The swing v119 / intraday v29 numbers are good enough to keep paper-trading and accumulate `decision_audit` data.
- **No new NIS features** until two weeks of `decision_audit` data with NIS scores joined to outcomes can show whether direction_score, materiality, sizing_multiplier each individually predict realized P&L. Phase 77 enables this measurement.
- **No live-money switch** until Phases 75, 76, 78, 82, 83 are all green.

---

## Section D — Things the Reviewers Got Wrong

Direct list. Reviewer claim → reality → implication.

1. **"MetaLabel R²=0.059 is still active as confidence gate."** Wrong — MetaLabel is referenced only in backtests (`grep` confirms 0 matches in `app/agents/`). The "filters 26% of entries" stat is a backtest artifact. Implication: live PM is more permissive than docs and walk-forward results assume; live results may underperform walk-forward by exactly the meta-filter contribution.

2. **"Bar 12 was selected via search across 78 bars."** Wrong — bars 12, 24, 36 were tested in Phase 50 multi-offset (multi-scan), and that v30 model failed the gate (commit `19d2f15`). The original single-scan bar 12 was chosen by intuition before any sweep. Sensitivity at bars 9-14 has never been measured. Implication: V6 reviewer concern stands; intraday edge may be brittle.

3. **"30% sector cap, GICS classification."** Wrong on both counts. 20% sector cap (`risk_rules.py:17`); sector source is a hand-curated Python dict (`constants.py:172`), not GICS. Implication: when a new ticker enters the watchlist without a SECTOR_MAP entry, it's "UNKNOWN" and slips through any sector concentration that compares against itself.

4. **"30-day pairwise correlation gate."** Wrong — it's 60+ day (~63 returns from 65 daily bars). Implication: the gate is more conservative than reviewer thought, blocking pairs with longer-term comovement; may be over-restrictive for short-horizon intraday plays where 60-day correlation is dominated by sector beta.

5. **"3-fold TimeSeriesSplit walk-forward."** Wrong — manual expanding-window construction with operator-set fold count (`walkforward_tier3.py:262-269`). The TimeSeriesSplit at training.py:1505 is for Optuna HPO inside a single fit, not for the walk-forward gate. Implication: the walk-forward methodology is not standard sklearn; understanding it requires reading the script. The embargo is correct (1 window = FORWARD_DAYS), but it's a gap, not a purge.

6. **"Earnings gate uses Finnhub."** Wrong — uses yfinance (`earnings.py:103`). Same source flagged as broken in `ML_MODELS_REFERENCE.md:293`. Implication: the gate that's supposed to keep MrTrader out of earnings reports relies on a feed that 401s most of the time. Fail-open behavior makes this a silent risk.

7. **"Position sizing is 10-15% notional."** Partially right by accident — designed as 2% risk per trade, but the 10% notional cap (MAX_POSITION_PCT) typically binds because intraday/swing stops are tight. Implication: sizing is mechanically correct but the operator's mental model ("we're risking 2%") is inaccurate; they're closer to "we cap at 10% notional, risk is whatever stop distance happens to be."

8. **"Daily loss limit is rolling."** Wrong — it's a calendar-day boundary using `str(date.today())`, server-local timezone. Implication: A loss at 23:30 ET / 04:30 UTC resets at midnight UTC. If the server is on UTC (Docker default), the daily loss limit "resets" at 20:00 ET — mid-trading-day on the West Coast.

9. **"Walk-forward runs in CI."** Wrong — only the `WalkForwardReport` aggregation logic is unit-tested. Full walk-forward must be triggered manually. Implication: a PR that breaks walk-forward (e.g. a bad feature change) won't be caught until Min runs it manually.

10. **"Wash sale rule prevents re-entry within 30 days."** Wrong — only logs a warning. Implication: tax-relevant; not a trading bug, but every loss-then-re-entry is a wash sale waiting to be flagged at year-end if this ever goes live with real money in a taxable account.

---

## Section E — Things Nobody Caught

Newly-found issues from reading the actual code.

### E1 — Slippage measurement is broken for market orders (V14)
`trader.py:646` uses `alpaca.get_latest_price(symbol)` AFTER `place_market_order()` returns, then computes slippage_bps against `intended_price = result.entry_price` set at signal time. This measures (post-fill price drift) against (signal-time price), not (fill price - decision price). It is the wrong number, persisted to a column called `slippage_bps`. Anyone querying that column (including the dashboard at `/api/orders`) is making decisions on garbage. Severity: blocks all execution-quality work.

### E2 — `_pending_limit_orders` lost on restart
`trader.py:60` initializes `self._pending_limit_orders: Dict[str, Dict[str, Any]] = {}` in memory, never persists. On uvicorn restart with a Phase 75 limit order live in Alpaca, the order ID is lost from the application; the order may fill silently, creating a position that the startup reconciler will pick up as "untracked" and assign default 2% stop / 6% target — overriding the actual ATR-based stop the strategy intended. Comment at `startup_reconciler.py:71` acknowledges this scenario but the fix is "create placeholder Trade with default values" — a degraded experience.

### E3 — `_peak_equity` resets to current equity on restart (V8)
`risk_manager.py:54, 312-313` initializes `_peak_equity = None` and assigns to current equity if higher. This is the high-water mark for the `MAX_ACCOUNT_DRAWDOWN_PCT = 0.05` rule. On every uvicorn restart, the operator's drawdown allowance resets. If the account is at $19k after losing from a $20k peak, restarting would set peak=$19k and the drawdown rule would now permit losing another 5% from there before tripping. Over a few restarts this loophole becomes meaningful. Also: there's no integration with Alpaca portfolio history — Min already added that for daily P&L (commit `3ca26ac`), but not for peak equity.

### E4 — Sector map is a static Python dict
`constants.py:172` — manually curated. New tickers added to `WatchlistTicker` table won't have a sector entry unless the operator updates `constants.py`. The fallback "UNKNOWN" sector matches itself in `validate_sector_concentration` and `_check_factor_concentration`, so multiple unknown-sector positions stack into the same "UNKNOWN" bucket and may breach concentration silently — or, in a worse failure mode, since SECTOR_MAP unknowns return "UNKNOWN" for every unknown ticker, two unknown tickers will be considered same-sector and cap correctly; but their *real* sectors might be different and this creates artificial blocks. Either way, undefined behavior.

### E5 — Earnings gate fails open on yfinance 401 (V22)
`earnings.py:120-122` catches all exceptions and returns None. None → days_until=None → block_swing=False. yfinance is documented as broken for most endpoints (`ML_MODELS_REFERENCE.md:293`). So in practice the earnings gate is a no-op for any symbol that yfinance refuses to serve. No metric tracks the % of symbols where this happens.

### E6 — No idempotency on order placement
`trader.py:619` calls `alpaca.place_limit_order` and `trader.py:641` calls `alpaca.place_market_order` without passing a `client_order_id`. On a network blip / retry, the same logical entry could be placed twice. The Trader currently has `if symbol in self.active_positions: return` guard but that guard checks AFTER the order is submitted, not before — so a network-induced retry of `_execute_entry` with stuck Redis re-delivery would double-buy.

### E7 — No periodic Alpaca reconciliation (V10)
Only at startup. If a manual Alpaca trade happens (Min logs in and closes a position from the Alpaca UI), MrTrader's DB will show the position as ACTIVE for the rest of the day, will continue evaluating exit signals against Alpaca data that no longer reflects DB state. Phase 78 fixes.

### E8 — No global market-hours guard on entry execution (V30)
Trader's `_check_entry` blocks intraday entries after 15:00 ET (line 434) but doesn't block before 09:30 ET. Swing entries place limit orders any time; Alpaca queues them as DAY orders. If a swing proposal makes it through at 03:00 ET (e.g. operator runs `select_instruments` manually), the limit order will be placed and queued.

### E9 — `MetaLabelModel` documentation drift
`ML_MODELS_REFERENCE.md:97` says "Effect: Filters ~26% of entries; removes the worst expected-loss setups." This implies live behavior. Code says backtest-only (V7). Anyone reading the docs will design tomorrow's improvements assuming a filter that isn't there. Either delete the doc claim or wire the filter in (and re-run gates because v119 walk-forward used MetaLabel).

### E10 — Multi-universe inconsistency
Three universes (SP-100 / SP-500 / Russell-1000) are used by different scripts and the live PM, with no shared configuration. The walk-forward gate (SP-100) is easier to pass than R-1000 inference performance would suggest; live PM trades a wider universe than was used to demonstrate the model passes the gate. This is the single biggest "the numbers don't represent what's deployed" issue in the system.

### E11 — Redis queue durability
PM puts proposals on Redis queue `trade_proposals`; main.py:118-124 flushes all queues on startup. So a proposal stuck in Redis when the system crashes is lost. In a normal restart this is right (avoids re-processing stale proposals); in a crash where the proposal was important, it's data loss without a record. The persisted `TradeProposal` row in DB is created by the RM on receipt — so a crash between PM-send and RM-receive loses both the message and the audit trail.

### E12 — Test for Phase 70 PM re-scoring is missing in tests/
Recent commit `71e496a` introduced "PM re-scoring of unexecuted approvals with WITHDRAW signal." `grep test_phase_70` returns nothing. The change works against a Redis queue and PM `_pending_approvals` dict; without a regression test, the next refactor of Redis or `_pending_approvals` semantics will silently break it.

---

## Section F — Task Backlog Specs (next 5–10 tasks)

(Tasks numbered 75 onwards to align with Phase numbers in Section C. Min has no `tasks/` directory yet; these are the spec files to create when needed.)

```
## Task 075: Wire kill switch into all agents
**Context:** kill_switch.is_active is checked in zero agent loops; trips
do not block new proposals. Current behavior: KillSwitch.activate() closes
existing positions but PM/RM/Trader keep flowing through to Alpaca.
**Deliverables:**
- app/agents/portfolio_manager.py: early-return + log_decision in
  _send_swing_proposals and _select_intraday_instruments when active
- app/agents/risk_manager.py: reject all proposals with failed_rule="kill_switch"
- app/agents/trader.py: skip _check_entry when active; cancel pending limit orders
- app/live_trading/kill_switch.py: also cancel all open Alpaca orders inside activate()
- tests/test_kill_switch_blocks.py: new — assert no order placed within 10s of activate()
**Acceptance Criteria:**
- After kill_switch.activate() is called, no new Trade rows are created within 30s
- Open Alpaca orders (returned by GetOrdersRequest(status=OPEN)) are all cancelled
- AuditLog has KILL_SWITCH_ACTIVATED entry; per-agent log_decision entries with
  decision_type=KILL_SWITCH_ACTIVE for every blocked proposal
**Effort:** 1 day
**Dependencies:** None

## Task 076: Fix slippage measurement for market orders
**Context:** trader.py:646 uses get_latest_price (post-submission new minute bar)
as the "fill price". intended_price for market orders comes from result.entry_price
(signal time, often minutes stale). slippage_bps column is therefore noise.
**Deliverables:**
- app/agents/trader.py:_execute_entry market path: replace get_latest_price call
  with poll-loop on alpaca.get_order_status(order_id), use filled_avg_price
- intended_price for market orders should be alpaca.get_quote()["mid"] captured
  IMMEDIATELY before place_market_order
- app/integrations/alpaca.py: add get_order_status if missing (it exists per
  trader.py:760, verify signature)
- tests/test_slippage_market.py: mock order status to return filled_avg_price=99.5,
  intended=100.0, assert slippage_bps == -50.0
**Acceptance Criteria:**
- Median |slippage_bps| on market orders in the next 5 paper-trading days is
  documented and reasonable (<30bps for liquid SP-100 names)
- slippage_bps for limit orders unchanged (already uses filled_avg_price)
**Effort:** 0.5 days
**Dependencies:** Task 075 (helps to have kill switch real before changing
order paths)

## Task 077: Decision-audit dashboard tile
**Context:** decision_audit table is populated but never read. Need to know
whether NIS, regime gate, macro gate are actually filtering well or just
suppressing volume.
**Deliverables:**
- app/api/routes.py: new endpoint GET /api/audit/summary returning aggregations
  by block_reason, news_action_policy, macro_risk_level
- frontend/dist/audit.html or React component
- scripts/backfill_decision_outcomes.py: runs as 16:30 ET cron, populates
  outcome_pnl_pct, outcome_4h_pct, outcome_1d_pct from Trade and bar data
- Test: factory data with known outcomes; assert summary matches
**Acceptance Criteria:**
- Dashboard shows: count and avg realized P&L by block_reason, win rate by
  news_action_policy bucket, scatter of model_score vs outcome_4h_pct
- Backfill script runs to completion in < 10 min for one day's decisions
**Effort:** 2 days
**Dependencies:** Task 076 (so realized P&L is trustworthy)

## Task 078: Periodic reconciliation + open-order sync
**Context:** Reconciliation runs only at startup. Drift between Alpaca and
DB persists indefinitely. Pending limit orders are lost on restart.
**Deliverables:**
- app/agents/portfolio_manager.py: add _reconciliation_loop task fired every
  15 min during market hours, calls existing reconcile() plus a new
  cancel_orphaned_orders helper
- app/startup_reconciler.py: add cancel_orphaned_orders(alpaca, db_session)
  function — for each open Alpaca order without a matching DB Order row,
  log + cancel via alpaca.cancel_order(order_id)
- app/agents/trader.py: persist _pending_limit_orders to a new
  pending_limit_orders DB table (or extend Order with status='PENDING_LIMIT')
- Test: simulate restart with one pending limit order; assert it is either
  re-tracked from DB or cancelled within next reconciliation cycle
**Acceptance Criteria:**
- Mid-session manual close in Alpaca UI is detected within 15 min and DB
  Trade row is marked CLOSED with note='reconciled_manual_close'
- Orphaned Alpaca order cancelled and audit logged within 15 min
**Effort:** 1 day
**Dependencies:** Task 075

## Task 079: Point-in-time index membership
**Context:** Training universes are static lists of ~current S&P 500 / R1000
membership. Survivorship bias inflates walk-forward Sharpe.
**Deliverables:**
- New file app/data/universe_history.py with members_at(date) function
- data/universe/sp500_membership.parquet — manually seeded from Polygon
  reference data or Wikipedia historical snapshots; columns: symbol,
  added_date, removed_date
- data/universe/russell1000_membership.parquet — same
- Update scripts/train_model.py:580, scripts/walkforward_tier3.py:228,348
  to call members_at(fold_train_start) instead of using static list
- Re-run swing walk-forward with point-in-time universe; report new Sharpe
  in docs/ML_EXPERIMENT_LOG.md
**Acceptance Criteria:**
- members_at(date(2022,1,1)) returns ~500 symbols including names since
  delisted/acquired (e.g. WORK, PAGS)
- Walk-forward avg Sharpe with PIT universe is documented and may be
  lower than current +1.181; that is a successful outcome
**Effort:** 3 days
**Dependencies:** None

## Task 080: Bar-12 sensitivity test
**Context:** Bar 12 was chosen by intuition; the multi-offset experiment
in Phase 50 was reverted but did not test bars 9-15.
**Deliverables:**
- scripts/bar_sensitivity.py: takes ENTRY_OFFSET arg, runs full walk-forward
  for one bar, outputs Sharpe + win-rate + trade count
- Run for offsets 9, 10, 11, 12, 13, 14, 15
- New section in docs/ML_EXPERIMENT_LOG.md with table of results
**Acceptance Criteria:**
- If avg Sharpe at bars 10-14 is monotonically smooth and 4 of 5 > 0.8,
  ship as robust
- If only bar 12 > 0.8, file follow-up phase to either pick a different
  bar with regime-conditional logic OR downgrade intraday Sharpe expectation
**Effort:** 1 day
**Dependencies:** Task 079 (preferable to have PIT universe for honest result)

## Task 081: Earnings calendar to Finnhub + fail-closed
**Context:** earnings.py uses yfinance which 401s most of the time;
fails open on exception → block_swing=False → trades through earnings.
**Deliverables:**
- app/calendars/earnings.py: replace _fetch with Finnhub call (reuse pattern
  from app/news/sources/finnhub_source.py)
- On fetch failure: return EarningsRisk with block_swing=True,
  reason="earnings_data_unavailable"
- New scheduled task in PM run loop at 06:00 ET: prefetch earnings dates
  for all watchlist symbols
- New metric in /api/health: earnings_data_freshness_pct (% symbols with
  data <24h old)
- Test: mock Finnhub 503; assert block_swing=True
**Acceptance Criteria:**
- Earnings gate continues to function when yfinance is down
- Health endpoint shows earnings data freshness
- No swing entry made within 2 trading days of earnings, even when data
  fetch fails
**Effort:** 1.5 days
**Dependencies:** None

## Task 082: Persist _peak_equity
**Context:** RM's _peak_equity resets on restart, allowing multi-step
drawdown loophole.
**Deliverables:**
- Add Configuration table key risk.peak_equity (float)
- app/agents/risk_manager.py:__init__ — load from config, fall back to
  Alpaca portfolio history max equity
- Update _peak_equity assignment to also persist (every change)
- Test: simulate restart sequence; assert peak_equity preserved
**Acceptance Criteria:**
- After uvicorn restart, _peak_equity == max(persisted, current)
- MAX_ACCOUNT_DRAWDOWN_PCT rule trips at correct level
**Effort:** 0.5 days
**Dependencies:** None
```

---

## Section G — Open Questions for Min

Decisions only Min can make. One question per item.

1. **Universe scope for retraining after Phase 79 (PIT universe):**
   - (a) Re-train both swing and intraday on point-in-time R-1000 — most honest, slowest, may worsen Sharpe
   - (b) Keep walk-forward on SP-100 PIT for the gate but train deployed model on full PIT R-1000
   - (c) Stay on current static universe; document the survivorship bias as a known accepted risk

2. **Kill switch enforcement style:**
   - (a) Hard block — kill switch tripped → all agents reject everything until Min manually resets
   - (b) Soft block — kill switch tripped → reject new entries but allow exit signals (existing positions can still close on stop/target)
   - (c) Hard block + auto-flatten — kill switch tripped → reject new entries AND immediately submit market-close orders for everything

3. **Slippage threshold for back-pressure:**
   - (a) Just measure correctly (Phase 76) and review weekly
   - (b) Auto-pause strategy when 5-trade rolling avg |slippage| > 50bps
   - (c) Per-symbol auto-halt when single trade slippage > 100bps

4. **Earnings gate fail behavior:**
   - (a) Fail-closed always — no entry without confirmed earnings date
   - (b) Fail-closed for swing, fail-open for intraday (intraday holds <2h, exposure smaller)
   - (c) Fail-open with daily Slack/email alert if >20% of universe has stale earnings data

5. **Wash-sale enforcement (relevant if you ever go to taxable real money):**
   - (a) Keep advisory only (current state)
   - (b) Block re-entry within 30 days of a closed loss for the same symbol
   - (c) Block re-entry within 30 days for losses > $X; allow if loss < $X

6. **MetaLabel in live PM:**
   - (a) Wire it in — match the documented behavior, re-validate walk-forward, accept the slight live performance hit
   - (b) Delete the model and remove all docs claiming it's active — be honest that the gate isn't there
   - (c) Train a new MetaLabel v2 with stronger features (R²=0.059 is weak) before deciding

7. **Going-live timeline:**
   - (a) After Phase 75-76-78-82-83 all green (kill switch + slippage + reconciler + drawdown + deadman), regardless of paper performance
   - (b) After 4 weeks of clean paper trading + the 5 hardening phases
   - (c) Defer indefinitely; treat MrTrader as a research platform only

---

*End of document.*
