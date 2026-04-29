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

### Tier 2 — Near-Term Engineering (worth doing in next 1–2 weeks)

#### Phase 72 — Earnings Exit Gate (holding positions into earnings)
**Gap:** We gate new *entries* near earnings, but we have no logic to exit *existing* swing positions before an earnings print. A 5-day hold entered Monday can roll straight into a Thursday AMC report.

**What to build:**
- In the 30-min position review loop, check if any held swing position has earnings within `EARNINGS_EXIT_DAYS=3` trading days
- If yes: send EXIT signal and log `EARNINGS_EXIT_APPROACHING` decision
- Already have `fetch_earnings_calendar()` in `finnhub_source.py` — just need the PM-side check
- **Files:** `app/agents/portfolio_manager.py` (30-min review loop)

#### Phase 73 — Overnight Gap Protection
**Gap:** The premarket routine checks gaps but only triggers a re-evaluation request to PM. There's no automatic exit if a stock gaps down hard overnight (e.g., -4%) on news we missed while the server was idle overnight.

**What to build:**
- In `_check_overnight_gaps()`, if adverse gap > `OVERNIGHT_GAP_EXIT_PCT=5%`, auto-submit an exit order directly (not just a reeval request)
- Log `OVERNIGHT_GAP_AUTO_EXIT` to decision audit
- **Files:** `app/agents/premarket.py`, `app/agents/trader.py`

#### Phase 74 — API / Dashboard Endpoint for NIS State
**Gap:** The dashboard has no visibility into NIS state — you can't see today's macro risk level, which symbols are blocked by news, or what the morning digest found.

**What to build:**
- `GET /api/nis/macro` — returns current `MacroContext` as JSON
- `GET /api/nis/signals` — returns latest `NewsSignal` per cached symbol
- `GET /api/nis/digest` — returns this morning's digest log (block_entry list, sized_down list)
- `GET /api/decision-audit/summary` — returns `gate_performance_summary()` output
- **Files:** `app/api/` (new router)

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

### Tier 4 — Deferred (after live trading + calibration)

| Phase | Name | Why Deferred |
|---|---|---|
| 51b | Multi-scan intraday tuning | Need live data to know if 11:00 / 13:30 windows help or just add noise |
| Sentiment v2 | Upgrade news_sentiment_3d/7d swing features | Need 60 days NIS history first |
| Regime v2 | Finer-grained market regime (bull/bear/chop sub-modes) | Need live performance data to see where the model fails |
| Options flow | Polygon options unusual activity as entry filter | Polygon premium required; validate concept first |
| Intraday v2 | Retrain intraday with multi-scan data | Run 4 weeks of 3-window data, then retrain on richer intraday distribution |

---

## Part 3 — What Is Probably Missing (Honest Assessment)

These are gaps that aren't in any phase but matter for a real trading system:

### 1. Position-Level Risk Review at End of Day (not built)
The PM reviews positions every 30 min during the day, but there's no end-of-day structured review that asks: "Should any swing position be closed before tomorrow given today's close price / volume / macro setup?" Right now, the only exit triggers are: stop hit, target hit, 5-day max hold, VIX tightening, or T1 partial exit. A weak EOD signal isn't caught.

### 2. Slippage / Execution Quality Tracking (built but not acted on)
`Order.slippage_bps` is stored but nothing reads it. After 4 weeks of paper trading, the `paper_trading_report.py` should surface which symbols have chronic slippage (indicating a liquidity problem the model doesn't see) and flag them for removal from the universe.

### 3. No Position-Level News Monitoring for Swing Trades (partial gap)
`NewsMonitor` watches symbols for intraday exits. But held swing positions don't get NIS Tier 2 re-scoring during the 30-min review — only the ML score is re-evaluated. A swing position held for 3 days could have material adverse news appear on day 2 with no NIS re-check. The fix: call `nis.get_stock_signal(symbol, force_refresh=True)` during the 30-min swing position review and check `action_policy == "exit_review"`.

### 4. Model Staleness Detection (not built)
The swing model retrains daily at 5 PM, but there's no alert if the retrain fails silently or if the new model's OOS AUC has drifted significantly below the gate threshold. The `auc_drift` check in `test_phase_b4_mlops.py` tests this in isolation but it isn't hooked into the daily retrain pipeline as a live alert.

### 5. No Graceful Shutdown / SIGTERM Handling (not built)
If the server process is killed (OS restart, crash), open positions are fine (reconciled on next start), but any in-flight proposals in the Redis queues may be in an inconsistent state. A SIGTERM handler that drains the queues and logs a clean shutdown event would prevent phantom proposals re-firing on restart.

### 6. Live Trading Readiness Checklist (partially built)
`test_live_readiness.py` exists but may not cover the new NIS components, the correlation gate, or the multi-scan intraday changes. Before live money, this test file should be audited against the full current system.

---

## Part 4 — One-Page Status Board

```
SYSTEM STATE: Paper trading active as of 2026-04-28
MODELS: Swing v119 (Sharpe +1.181 ✅), Intraday v29 (Sharpe +1.776 ✅)
TESTS: 1171 passing

COMPLETE (everything shipped):
  Foundation        — All phases 1–56, both model gates passed
  Hardening         — Phases A/B/C/67/68/69/70
  NIS               — Phases 58/59/60/63 (full Finnhub+Haiku stack)
  Risk              — Phase 71 (correlation gate)
  Audit             — Phase 61 (decision_audit table)
  Intraday          — Phase 51 (3 scan windows, cooldown, P&L cap)
  Digest            — Phase 62 (9 AM NIS pre-score, cache warm)

NEXT (no code yet):
  Phase 57  — Paper trading calibration (run 2-4 weeks, then review)
  Phase 72  — Earnings exit gate for held swing positions
  Phase 73  — Overnight gap auto-exit
  Phase 74  — NIS/audit dashboard endpoints

MEDIUM TERM:
  Phase 64  — News as model features (needs 60 days NIS history)
  Phase 65  — Source expansion + dedup clustering
  Phase 66  — Sonnet escalation for high-stakes events

KNOWN GAPS (not in any phase yet):
  - EOD swing position review signal
  - Slippage analysis surfaced in reporting
  - NIS Tier 2 re-check during swing 30-min review
  - Model staleness / AUC drift live alert
  - Graceful SIGTERM / queue drain on shutdown
  - Live readiness checklist audit against current system
```
