# Whole-App Bug Audit — 2026-06-22 (Alpha-v10)

Multi-agent adversarial audit of all of `app/` (39 components). Each component reviewed by one Opus
agent; **every candidate finding independently re-verified by a second Opus agent that tried to
refute it.** Raw result: `tasks/wcht6web3.output` (run id `wf_c79f00ac-cd7`).

**122 candidates → 102 confirmed (20 false positives killed): 2 BLOCKER / 40 MAJOR / 60 MINOR.**

`scripts/` (156 files) was deliberately NOT in scope — a separate second pass.

---

## Progress
- ✅ **WAVE 1 DONE 2026-06-22** (PR pending merge): 10 fixes (BLOCKER #1 + 9 MAJOR) — live order-path fail-OPEN → fail-CLOSED + idempotency on every order. Opus deep-dive SAFE TO MERGE (no new fail-open / dropped-order collision / exit regression). 12 new tests; full suite green. See DECISIONS 2026-06-22 (Audit Wave 1).
- ✅ **WAVE 2 DONE 2026-06-22** (PR pending merge): BLOCKER #2 (pause/resume permanently killed the agent loops + auto-pause irreversible) + the intraday 2% daily-loss hard stop. Cooperative pause (loops idle alive), auto-pause auto-resumes on Alpaca recovery, force-close runs even while paused, daily-loss uses live equity. Opus deep-dive SAFE TO MERGE; MINOR-1 closed. 7 new tests; full suite green. DECISIONS 2026-06-22 (Audit Wave 2).
- ✅ **WAVE 3a DONE 2026-06-23** (PR pending merge): live agent-path state-corruption (6 fixes): partial-exit shares decrement, exit fill-price read (kills PM-EXIT $0 P&L), intraday filled_qty, _record_entry duplicate-ACTIVE, PM rescore sleeve isolation, premarket prior-close/current-price (+AUTO_EXIT guard). Opus deep-dive SAFE TO MERGE. 4 new tests + phase20 gap suite updated; full suite green.
- ✅ **WAVE 3b DONE 2026-06-23** (PR pending merge): runtime/data state-corruption (7 fixes): capital-ramp persistence, FRED most-recent obs, earnings trading-day blackout, single-row model sigmoid fallback, PEAD+trend daily-P&L delta, go-live cross-process bridge, ghost-state-machine close + partial-fill-at-restart. Opus deep-dive: all correct, no regression (caught a flaky-test MAJOR + the trend_tracker sibling → both fixed). 5 new tests; full suite 3993 green. **Wave 3 complete.**
- ✅ **WAVE 4a DONE 2026-06-23** (PR pending merge): backtest metrics + simulators (research-tier, non-live): break-even classification, frequency-based Sharpe annualisation, exit-ordered max-drawdown (metrics.py); short-borrow `/365`→`/252` (agent_simulator); intraday dispersion-gate look-ahead → prior-day dispersion (intraday_agent_simulator, both Phase-2c + R5-B). Unit-tested; full suite 3997 green. **Re-run note:** the dispersion-gated intraday WF/CPCV should be re-run (the leak inflated its Sharpe).
  - **DEFERRED to dedicated PRs (intricate, non-live research P&L/sizing — too risky to bundle):** (1) `agent_simulator` signal-mode short-margin reservation (interacts with `buying_power` property + `_effective_cash` + shared `_close_position` used by the live-developed L/S rebalance research); (2) `intraday_agent_simulator` positions-never-populated (MAX_OPEN/heat caps dead + the always-0 deployment metric is its consequence). Both need their own deep-dive + eval re-run.
- ⏳ Wave 4b (research stats + ML/data look-ahead) + Wave 5 + re-audit: in progress.

## Triage → fix waves

Priority is **live-capital risk before the IBKR live flip**, then live state-corruption, then
research/verdict-trustworthiness, then low-risk cleanups. Research-tier look-ahead fixes change
backtest/CPCV numbers, so they may require re-running the affected evals (flagged per item).

### WAVE 1 — LIVE order-path: fail-OPEN safety gates + missing idempotency (the "double-trade / can't-halt" cluster) — HIGHEST PRIORITY
| # | Sev | File | Bug |
|---|---|---|---|
| 1 | **BLOCKER** | integrations/alpaca.py + agents/trader.py | `get_position()` swallows ALL errors → None; entry "final guard" reads None as "flat" → **duplicate live position** on a non-transient read error at restart. Fix: distinguish 404 from error; entry guard fail-CLOSED. |
| 2 | MAJOR | live_trading/trend_sleeve.py | `_current_trend_positions` returns `{}` on DB/get_positions error → rebalancer **re-buys the entire sleeve**. Fix: fail-CLOSED (None → block). |
| 3 | MAJOR | live_trading/kill_switch.py | flatten places market orders with **no client_order_id** → concurrent kill (manual + watchdog) **double-sells / flips short**. Fix: route through `close_all_positions(cancel_orders=True)`. |
| 4 | MAJOR | live_trading/kill_switch.py | silent `_persist_state()` failure → the 3 s state-sync poll reloads DB `False` → **kill switch un-arms itself ~3 s after activation**. Fix: persisted-active is a latch; never downgrade in-memory active→False without an explicit reset. |
| 5 | MAJOR | api/routes.py | `/control/kill-switch` (close-all) + `/control/close-position` send `sell` of a **negative qty** for shorts → Alpaca rejects → **shorts never flattened**. Fix: side = buy if qty<0; abs(qty); delegate to kill_switch.activate(). |
| 6 | MAJOR | live_trading/trend_sleeve.py | whole-book gate mode read **without `.strip().lower()`** → `"Enforce"`/`"ENFORCE"` silently runs as shadow → **gate fails OPEN** in the live path. Fix: normalize (one line). |
| 7 | MAJOR | agents/trader.py (+ kill_switch) | exit market orders (partial + full) placed with **no client_order_id** → lost-response retry **double-sells**. Fix: deterministic exit client_order_id. |
| 8 | MAJOR | integrations/alpaca.py | `place_limit_order` has **no duplicate-id idempotency** (unlike market) → lost-response retry leaves an **orphaned live order** booked as FAILED. Fix: mirror the market-order idempotent-reuse block. |
| 9 | MAJOR | integrations/alpaca.py | `get_positions/get_position/get_order_status` do `int(pos.qty)` → **crash on any fractional share** (aborts the whole positions fetch → breaks reconciliation). Fix: `int(float(...))` + per-row guard. |
| 10 | MAJOR | notifications/notifier.py + scripts/notify_watcher.py | CATASTROPHIC emails (kill_switch/dead_man/gate_error) **silently dropped after 5 SMTP failures**, no escalation. Fix: escalate/never-cap CRITICAL. |

### WAVE 2 — BLOCKER: lifecycle + intraday hard-stop
| # | Sev | File | Bug |
|---|---|---|---|
| 11 | **BLOCKER** | orchestrator.py | `pause_trading()` flips status→paused; agent `while status=="running"` loops **return → tasks die**; `resume_trading()` only flips a flag (no task recreation) → **trading never resumes without a restart**. Worse: a transient Alpaca health blip auto-pauses → permanently kills the trader loop → **3:45 pm intraday force-close stops running** (overnight risk). Fix: cooperative pause (loop keeps iterating while paused) + resume re-creates dead tasks + auto-resume. |
| 12 | MAJOR | agents/risk_manager.py | daily-loss gate reads the **EOD-only** RiskMetric row → returns 0.0 intraday → the **2% hard daily-loss stop cannot fire during the session**. Fix: compute live intraday day-loss from Alpaca account equity; optionally auto-trip kill. |

### WAVE 3 — LIVE state-corruption / agents (real, mostly self-healing or fail-safe direction)
capital-ramp not persisted (resets to Stage 1 on restart) · PEAD daily-P&L double-counts the unrealized **level** (corrupts Sharpe + would misallocate if the vol/regime allocator is armed) · 30-min rescore re-scores **intraday** proposals with the **swing** model → spurious WITHDRAW · partial-exit `pos['shares']` never decremented → P&L corruption in the no-position fallback · intraday market entry records requested (not filled) qty · `_record_entry` can create a **duplicate ACTIVE Trade** on a status race · PM EXIT falls back to entry_price → **fabricated $0 P&L** · premarket gap/`_fetch_spy_premarket`/`today_open` use the **wrong prior bar** (3 bugs) → spurious AUTO_EXIT + mis-gated sizing · ghost state-machine **never reaches CLOSED** (count frozen at 1) → every ghost lingers 24 h → UNRESOLVED · partial-fill-at-restart tracks only the partial · go-live/capital-start **not bridged** in subprocess mode · model single-row min-max normalization → 0.0 → spurious EXIT (latent: active model is guarded LambdaRank) · FRED API fetches the **oldest 24 obs** (1960s macro values) when a FRED key is set · earnings blackout counts **calendar** not trading days → can hold through a print.

### WAVE 4 — RESEARCH / verdict-trustworthiness (look-ahead + stats; fixes change eval numbers → may need eval re-runs)
event-scorer VIX gate reads **same-day** close (pead/insider/analyst/short_interest) · dispersion gate floors on **full-series** quantile (active WF gate) · `_walk_forward_cv` splits **symbol-major** rows → not OOS · polygon/FMP financials + 13F filter on **period-end not filing-date** (look-ahead into model features) · `vol_percentile_52w` ranks against the **oldest** year · `ix_momentum_vol` **sign-flips** high-vol momentum · daily/macro caches **append-only with auto_adjust** → split/dividend drift · VIX-VRP sleeve charges **no cost** (inflates go-live Sharpe) · short borrow **/365** but accrued per trading day (−31%) · signal-mode shorts **don't reserve margin** (defeats buying-power cap) · intraday sim **never populates positions** (MAX_OPEN/heat caps dead) · backtest Sharpe annualization **scales with n** (gates paper-readiness) · drawdown on overlapping concurrent trades · max-of-6 null **shrinks** on degenerate panels · TSNormalizer **mutates inference state** → std→0 collapse · NaN-mask blind spots in the look-ahead audits · ddof inconsistencies · DSR fail-open on sparse nulls · several more.

### WAVE 5 — MINOR cleanups (conservative-direction / latent / research-only)
idempotency-key omits side/qty · allocator staleness off-by-fraction & warmup-doesn't-revert · notifier dedup/throttle edge cases · RedisQueue drops undecodable msg · RSI=0 on zero-loss · regime VIX cache no upper-staleness bound · BenignGate caches a failed-closed 0.0 for the day · timezone-mixed `closed_at` · config upsert race · dead `daily_state` migration · break-even classified as loser · NIS train/serve skew · vix_term_ratio inverted naming · earnings cache casing · macro already_priced_in None==None · etc.

---

## Notes
- **Recurring root cause** across Wave 1: *fail-OPEN on an unknown/error state* in the live path (position read, gate mode, persist failure) and *missing idempotency on non-entry orders* (exits, limits, kill-flatten). A shared "unknown ⇒ fail-closed" helper + always passing a client_order_id would prevent the whole class.
- Several findings are **latent** (guarded by a default-off flag) or **fail-safe direction** (over-block, never over-trade) — fixed for correctness but not urgent.
- Research-tier (Wave 4) fixes alter backtest/CPCV outputs → the affected sleeve verdicts should be **re-run** after fixing (carry/xsmom/VRP costs, the look-ahead gates). None change the *live* book.
