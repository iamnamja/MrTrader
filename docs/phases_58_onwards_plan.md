# MrTrader — Consolidated Phase Roadmap

**Last updated:** 2026-04-29  
**Status:** Paper trading active. All major risk and news intelligence layers shipped.

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

---

### Tier 2 — Near-Term Engineering ✅ COMPLETE

Phases 72, 73, 74 shipped 2026-04-29.

| Phase | Name | Status |
|---|---|---|
| 72 | NIS re-check for held swing positions | ✅ Done — `exit_review` policy triggers EXIT in 30-min review |
| 73 | Overnight gap open price accuracy | ✅ Done — uses actual 1-min open bar; auto-exit already wired |
| 74 | NIS & audit dashboard API | ✅ Done — 5 new endpoints in `app/api/nis_routes.py` |

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

---

## Part 3 — One-Page Status Board

```
SYSTEM STATE: Paper trading active as of 2026-04-28
MODELS: Swing v119 (Sharpe +1.181 ✅), Intraday v29 (Sharpe +1.776 ✅)
TESTS: 1171 passing

COMPLETE (all shipped as of 2026-04-29):
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

LET IT RUN:
  Phase 57  — Paper trading calibration (run 2-4 weeks, then review)

MEDIUM TERM (phases 75–79):
  Phase 75  — EOD swing review signal (15:45 ET weak-score exit)
  Phase 76  — Slippage analysis surfaced in reporting + API
  Phase 77  — Graceful SIGTERM / queue drain on shutdown
  Phase 78  — Live readiness checklist audit (extend test_live_readiness.py)
  Phase 79  — AUC drift live alert wired into daily retrain

LONG TERM (after calibration):
  Phase 64  — News as model features (needs 60 days NIS history)
  Phase 64  — News as model features (needs 60 days NIS history)
  Phase 65  — Source expansion + dedup clustering
  Phase 66  — Sonnet escalation for high-stakes events
  Tier 4    — Intraday v2 retrain, regime v2, options flow
```
