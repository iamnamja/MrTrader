# Phases 58+ — Consolidated Roadmap (News Intelligence + Remaining Backlog)

**Status:** Active. Synthesizes external NIS spec + existing Phase 48–57 backlog + live paper-trading session learnings (2026-04-28).
**Last updated:** 2026-04-28

---

## Part 1 — Honest Review of the External NIS Spec

The spec is excellent engineering architecture for a mature system. Most of the *what* is right. Some of the *when* and *how much* needs trimming for our stage.

### What the spec gets right (adopt these ideas)

1. **Earnings calendar as the #1 gap.** We have zero earnings awareness. A model that holds AAPL into an earnings print has no edge — it's a coin flip with 3× normal volatility. This is the single highest-leverage addition we can make right now.

2. **Macro calendar (FOMC, CPI, NFP).** These events move entire sectors in 15-minute windows. Entering a swing position 30 minutes before a CPI print is amateur-hour risk management.

3. **LLM as interpreter, not trader** (Section 1.1 of the spec). The framing is exactly right. The LLM scores event materiality; Python code makes the trading decision. This is already how our Claude Haiku audit tasks work — we just need to extend it to news.

4. **Structured signal (direction + materiality + confidence) instead of a boolean.** The current `has_negative_news() -> bool` loses all nuance. A 0.95-confidence earnings miss is categorically different from a 0.40-confidence analyst note.

5. **Decision audit trail.** We have no structured record of why PM proposed what it did, what news it saw, what calendar risk existed, what the model score was, and whether the trade worked. This is the precondition for honest calibration in Phase 57.

6. **Information decay.** A news event from 6 hours ago should have less weight than one from 30 minutes ago. The decay function (Section 8.7) is simple and right.

7. **"Already priced in" as a signal dimension.** If price already moved hard in the direction implied by the news, the actionable edge is smaller. This is subtle but correct.

8. **Cluster-level scoring, not article-level.** Avoid double-counting when 8 outlets run the same wire story.

### What's overkill for our current stage (defer or simplify)

| Spec Item | Why It's Premature |
|---|---|
| Full 6-layer module layout with ABCs (`NewsSource`, `LLMScorer`) | We have one Polygon source + one Alpaca source. Abstractions before we have 3+ sources are premature. |
| TF-IDF cosine clustering across sources | Significant engineering for a 50-symbol universe. Start with exact-hash dedup + same-symbol same-hour heuristic. |
| OpenAI + Ollama provider fallbacks | We're all-in on Claude. Add Sonnet as escalation first. Ollama is a multi-week project on its own. |
| Self-consistency check (run Sonnet twice, reversed article order) | Overkill until we have evidence the LLM is producing conflicting outputs. |
| `schema_version` + `prompt_version` on every DB row (Phase 1) | Right idea, but can be added in Phase 60 when we actually store LLM outputs. |
| `news_layer_alpha` metric (Section 14.2) | Requires 30+ days of decision audit history. Phase 63+ concern. |
| Phase 4: news as model features | Explicitly requires 60 days of point-in-time history. Don't even think about this until Phase 63. |
| Phase 5: opportunity detection (sympathy plays) | Cool idea, expand later. |
| Local Llama fallback | Don't need it while we're under $5/day on Haiku. |

### What the spec missed (not in the external review, but important for us)

1. **Phase 50 (time-of-day) and Phase 51 (multi-scan) are still pending.** These are pre-existing backlog items that must complete before we can claim full paper-trading readiness. The NIS spec doesn't know about these.

2. **Phase 57 calibration is a checkpoint, not a destination.** After 4 weeks of paper trading with decision audit data, we need a formal review before live money.

3. **Existing `NewsMonitor` (just built) should not be thrown away.** It's a working dual-source (Polygon + Alpaca) live poller. The right move is to evolve it, not replace it.

4. **We already have swing news training features** (`news_sentiment_3d/7d/momentum`). Don't delete them until Phase 4-equivalent replacements are validated.

---

## Part 1b — Live Paper Trading Session Learnings (2026-04-28)

First full paper trading session revealed systemic gaps in the PM→RM→Trader pipeline. All of the following were identified and either fixed (✅) or queued:

| Gap | Status | What We Did |
|---|---|---|
| No durable state — restart wiped active_positions | ✅ Fixed (Phase A) | `_reconcile_positions()` on startup; `DailyState` table for PM flags |
| PM had no time-of-day routing — swing fired at 2 PM | ✅ Fixed (Phase A) | Time-window guards; swing only pre-9:50, intraday only 9:45-10:30 |
| No stale proposal TTL — morning approvals re-fired hours later | ✅ Fixed (Phase C) | 60-min TTL + 4 PM Redis purge + one-approval-per-symbol-per-day dedup |
| No per-position size cap — oversized bets on tight-stop stocks | ✅ Fixed | 10% equity cap added to `size_position()` |
| Force-close missed positions set via DB (in-memory state stale) | ✅ Fixed | Force-close now cross-checks DB for ACTIVE intraday trades |
| Trader entered blindly regardless of current conditions | ✅ Fixed (Phase 67) | Entry quality check: price run, spread, intraday momentum, volume |
| No live macro gate — entered on bad market days | ✅ Fixed (Phase 68) | `premarket_intel.is_swing_blocked()` + live SPY intraday drawdown check |
| Stop/target not adjusted as market moved | ✅ Fixed (Phase 69) | `check_dynamic_adjustments()`: VIX-driven stop tightening + target extension |
| Swing/intraday mixed up in DB (all showing "swing") | ✅ Fixed | `trade_type` persisted to Trade on entry; reconcile reads from DB |
| Summary endpoint 500 error (missing DB column) | ✅ Fixed | `exit_reason` column added via ALTER TABLE |
| Equity curve 1w/1m missing today's P&L | ✅ Fixed | Route appends today's live P&L (unrealized + realized) to history |
| Last signal showing 150h (stale AgentDecision) | ✅ Fixed | Falls back to most recent Trade.created_at |
| **Trader re-score of unexecuted candidates as market moves** | 🔜 Phase 70 | See below |
| **Intraday correlation/sector concentration in RM** | 🔜 Phase 71 | See below |
| **Full news intelligence pipeline** | 🔜 Phase 60+ | Existing plan unchanged |

---

## Part 2 — Consolidated Backlog (All Open Phases)

### Immediate — Complete These Before Paper Trading Starts

| Phase | Name | Type | Status | Depends On |
|---|---|---|---|---|
| 50 | Time-of-Day Segmentation (Intraday) | Retrain + walk-forward | Training running (v30 log) | — |
| 52 | Condition-Based Entry Gates | Code only | Partially built in PM | — |
| 53 | Live News Monitor | Engineering + live validation | Code done, needs market hours | — |
| 56 | Paper Trading Monitoring Report | Engineering | Done (`scripts/paper_trading_report.py`) | — |

**Merge gate for paper trading start:** Phases 52, 53, 56 merged + Phase 50 gate passed.

---

---

### Phase 67 — Trader Entry Quality Check ✅ DONE (2026-04-28)

Real-time entry quality filter in `app/strategy/entry_quality.py`. Called by Trader at execution time, after PM/RM approval, to validate that current conditions still support entering.

**Checks:**
1. **Price run** — stock hasn't moved >1.5% past signal price since PM scored it (hours ago)
2. **Adverse move** — stock hasn't dropped >1.0% from signal price (gap-down since approval)
3. **Spread/liquidity** — bid-ask spread ≤ 0.5% of mid-price
4. **Intraday momentum** — 30-min slope of 5-min closes not sharply negative (−0.8% intraday, −1.5% swing)
5. **Volume context** — current bar not suspiciously thin (<30% of 20-bar avg)

On failure: logs `ENTRY_REJECTED_QUALITY` decision to DB. Price-run failures drop the proposal; other failures retry on next scan cycle.

---

### Phase 68 — Live Macro/Market Gate ✅ DONE (2026-04-28)

Enhanced `premarket_intel` module with two new methods:

- **`is_swing_blocked()`** — blocks swing entries on: SPY pre-market gap < −2.5%, FOMC day, or live SPY intraday drawdown > 2% from open
- **`_get_spy_intraday_drawdown()`** — live SPY 5-min bar check, cached 5 min
- **`get_market_context()`** — returns full context dict (spy pct, drawdown, macro flags, block status)

Both intraday and swing gates now checked in `_check_entry` before sizing.

---

### Phase 69 — Dynamic Stop Trailing + Partial Exits ✅ DONE (2026-04-28)

New `check_dynamic_adjustments()` function in `signals.py`. Called every scan cycle per open position.

**What it does:**
1. **Partial exit at T1** — when price hits entry + 2×ATR, sells `partial_exit_pct` (default 50%) of shares; caller tracks `_partial_exited` flag to fire once
2. **VIX-driven stop tightening** — when VIX > 20 and position is profitable, tightens trailing stop to 1.5×ATR below highest price (vs. normal 2.5×ATR)
3. **Normal trail at +4%** — tightens to 2×ATR below highest once up 4%
4. **Target extension** — when price within 0.5×ATR of target and strong momentum (price > entry + 3×ATR), extends target by 1×ATR to let winners run

Stop and target updates are persisted to DB immediately so they survive a restart.

---

### Phase 70 — PM Re-Scoring of Unexecuted Candidates (queued)

**Gap:** PM scores candidates at 9:50 AM. If Trader hasn't entered by 1 PM (e.g., entry quality check failed all morning), the score is 4 hours stale. Market conditions may have fundamentally changed.

**What to build:**
- PM tracks `_pending_approvals: Dict[symbol, approved_at]` — set when proposal approved by RM
- Every 30-min review cycle: re-score pending symbols using fresh daily bars
- If fresh score < `MIN_CONFIDENCE * 0.9` → send `WITHDRAW` message to RM/Trader to drop the symbol
- If score still high but entry quality was failing on price-run → update the target entry price

**Files:** `app/agents/portfolio_manager.py`, new `PENDING_WITHDRAWALS_QUEUE`

---

### Phase 71 — Correlation/Sector Concentration in RM (queued)

**Gap:** RM checks sector concentration in `validate_sector_concentration()` but uses a simple count, not correlation. Long NVDA + AMD + AVGO + SMCI = four separate 10% positions that are 90%+ correlated — effectively 40% in semis.

**What to build:**
- `validate_correlation_risk()` rule in `risk_rules.py`
- Fetch 30-day daily returns for current portfolio + proposed symbol
- Reject if pairwise correlation > 0.70 with any existing position that's already >8% of portfolio
- Sector contribution cap: sum of market_value for any sector ≤ 25% of portfolio

**Files:** `app/agents/risk_rules.py`, `app/agents/risk_manager.py`

---

### Sprint 1 — Weeks 1–2: Calendar Intelligence (Highest Leverage)

These two phases are the top priority from the NIS spec and have the clearest ROI.

#### Phase 58 — Earnings Calendar Gate

**Why first:** We are blind to earnings. One bad print on a held swing position can wipe a week of gains. This is the single most important risk addition before live money.

**What we build:**
1. Daily fetch of earnings dates for SP100 + Russell 1000 universe via Polygon `/v3/reference/dividends` or Finnhub free tier (`/calendar/earnings`)
2. Store in SQLite `calendar_event` table (schema below)
3. `CalendarService.get_earnings_risk(symbol, hold_days) -> EarningsRisk` dataclass
4. PM rules:
   - Swing: block new entry if earnings within 2 trading days
   - Swing: force exit review if current position would be held through earnings
   - Intraday: block new entry if earnings today or pre-market tomorrow
5. RM rule: veto any proposal that `CalendarService` flags as `holds_through_earnings=True`

**Gate:** Manual review of 10 symbols across the last 6 months — confirm earnings dates are correct and that the gate would have prevented entries into known blow-up prints.

**Files:**
- New: `app/calendars/earnings.py` — `EarningsCalendar` + `EarningsRisk` dataclass
- New: `app/calendars/__init__.py`
- Modified: `app/agents/portfolio_manager.py` — call earnings gate before sending to RM
- Modified: `app/agents/risk_manager.py` — veto rule for holds_through_earnings
- New table: `calendar_event` in existing SQLite DB

**Minimal schema:**
```sql
CREATE TABLE calendar_event (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,    -- 'earnings', 'fomc', 'cpi', 'nfp', etc.
    symbol TEXT,                 -- null for macro events
    event_time TEXT NOT NULL,    -- scheduled UTC ISO8601
    importance TEXT NOT NULL,    -- 'low', 'medium', 'high'
    source TEXT NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT NOT NULL
);
CREATE INDEX idx_cal_symbol ON calendar_event(symbol);
CREATE INDEX idx_cal_time ON calendar_event(event_time);
```

---

#### Phase 59 — Macro Calendar Awareness

**Why second:** FOMC and CPI are market-wide events. Trading into a scheduled Fed announcement with full position size is a sizing error, not a model error.

**What we build:**
1. Fetch high-impact macro events from a free source (FRED releases calendar + ForexFactory RSS or hardcoded schedule for FOMC)
2. Store in same `calendar_event` table (event_type = 'fomc', 'cpi', 'nfp', etc.)
3. `MacroCalendar.get_market_context() -> MacroContext` with:
   - `high_impact_today: bool`
   - `within_event_window: bool` (±15 min of release)
   - `next_event: Optional[str]`
4. PM rules:
   - Within ±15 min of high-impact event: skip new entries for both swing and intraday
   - FOMC press conference: ±60 min window
   - If `high_impact_today=True` and VIX > 20: reduce position sizes by 15%
5. Existing `MarketRegime` / abstention gate updated to consume `MacroContext`

**Gate:** Backtest replay check — confirm at least 5 known macro-event-day scenarios would have triggered the pause window correctly.

**Files:**
- New: `app/calendars/macro.py`
- Modified: `app/agents/portfolio_manager.py`
- Modified: `app/backtesting/agent_simulator.py` (macro-aware backtester for Phase 57)

---

### Sprint 2 — Weeks 3–4: Structured News Signal + Decision Audit

#### Phase 51 — Multi-Scan Intraday (depends on Phase 50 gate passing)

**Status:** Blocked on Phase 50 v30 results. If avg Sharpe ≥ +1.275, proceed.

**What we build:**
- Remove `_selected_intraday_today` once-per-day guard
- Scan windows: 09:45, 11:00, 13:30 ET
- Per-symbol 2-hour cooldown after exit
- Stale-price guard: skip if price drifted >0.5% from scoring price
- Daily intraday P&L cap: stop new entries if day's intraday loss > 1% of account
- Walk-forward gate: avg Sharpe ≥ +1.275

---

#### Phase 60 — Structured NewsSignal + LLM Haiku Scorer

**Why now:** We have `NewsMonitor` working with a boolean output. This phase replaces the boolean with a structured signal and adds LLM materiality scoring.

**What we build:**

1. **`NewsSignal` dataclass** (new `app/agents/news_signal.py`):
   ```python
   @dataclass
   class NewsSignal:
       symbol: str
       evaluated_at: datetime
       direction_score: float        # [-1.0, 1.0]
       materiality_score: float      # [0.0, 1.0]
       downside_risk_score: float
       confidence: float
       novelty_score: float
       already_priced_in_score: float
       event_type: str
       rationale: str
       scorer_tier: str              # 'rules' | 'haiku' | 'cached' | 'fallback'
       stale: bool = False
       fallback_used: bool = False
   ```

2. **LLM Haiku scorer** (extend `app/ai/claude_client.py`):
   - New function: `score_news_event(symbol, company, headline, summary, price_context) -> NewsSignal`
   - Uses `materiality_v1` prompt template
   - Caches by `sha256(headline + summary + symbol)` with 1-hour TTL
   - Falls back to keyword classifier if Haiku fails or times out (2s timeout)
   - Logs call to `llm_call_log` table (cost, latency, cache_hit)

3. **`NewsMonitor` upgraded:**
   - `_poll_one()` now returns `Optional[NewsSignal]` instead of `Optional[Dict]`
   - `has_negative_news()` kept as backward-compat shim: `return signal.downside_risk_score > 0.6 and signal.confidence > 0.5`
   - `get_signal(symbol) -> Optional[NewsSignal]` added as primary API

4. **PM updated:**
   - Entry gate uses `signal.downside_risk_score` and `signal.confidence` thresholds instead of boolean
   - Exit flag threshold: `downside_risk_score >= 0.75 and confidence >= 0.65`
   - Intraday size reduction: if `0.5 <= downside_risk_score < 0.75` → 0.75× size (don't fully block, just de-risk)

5. **New table `llm_call_log`** added to existing SQLite DB (minimal — call_id, called_at, symbol, model, latency_ms, cost_usd, cache_hit, error)

**Prompt template** (`config/prompts/materiality_v1.txt`) — simplified from spec Section 8.5, removing price context for initial version (add in Phase 62 when we have more context available at scoring time):

```
You are a financial event interpretation system. Your role is to classify
whether a news event is material to a specific stock.

You DO NOT recommend trades. You classify events only.

Symbol: {symbol}
Headline: {headline}
Summary: {summary}
Published: {published_at}

Return STRICT JSON ONLY:
{
  "event_type": <earnings|guidance|analyst_action|m_and_a|legal_regulatory|product|other>,
  "direction_score": <float [-1.0, 1.0], negative=bearish for this symbol>,
  "materiality_score": <float [0.0, 1.0]>,
  "downside_risk_score": <float [0.0, 1.0]>,
  "confidence": <float [0.0, 1.0]>,
  "novelty_score": <float [0.0, 1.0]>,
  "already_priced_in_score": <float [0.0, 1.0]>,
  "rationale": "<one sentence max 150 chars>"
}
```

**Gate:** Live validation — confirm Haiku calls complete under 2 seconds, cost stays under $1/day at 10-symbol scan frequency, fallback fires correctly when API is down.

---

#### Phase 61 — Decision Audit Trail

**Why:** Phase 57 calibration is meaningless without this. Every PM decision (enter, block, size_down, exit) needs a structured record with its inputs so we can answer: "did the news gate add alpha or just block winners?"

**What we build:**

1. New table `decision_audit`:
   ```sql
   CREATE TABLE decision_audit (
       decision_id TEXT PRIMARY KEY,
       decided_at TEXT NOT NULL,
       symbol TEXT NOT NULL,
       strategy TEXT NOT NULL,          -- 'swing' | 'intraday'
       model_score REAL,
       news_signal_id TEXT,             -- nullable ref to llm_call_log
       calendar_risk_json TEXT,
       final_decision TEXT NOT NULL,    -- 'enter'|'block'|'size_down'|'exit_review'|'hold'
       size_multiplier REAL NOT NULL DEFAULT 1.0,
       block_reason TEXT,
       -- Filled in EOD by post-trade evaluator:
       realized_pnl_pct REAL,
       realized_outcome_4h REAL,
       realized_outcome_1d REAL
   );
   ```

2. PM writes a `decision_audit` row for every `send_to_rm()` call and every `block_entry()` decision

3. EOD script (extend `paper_trading_report.py`): fill in `realized_pnl_pct` for completed trades

4. Phase 57 query: `SELECT block_reason, AVG(realized_pnl_pct), COUNT(*) FROM decision_audit WHERE final_decision='block' GROUP BY block_reason` — immediately shows whether each gate is blocking winners or losers

**Files:**
- Modified: `app/agents/portfolio_manager.py`
- Modified: `scripts/paper_trading_report.py`
- New: `app/database/decision_audit.py` (repository functions)

---

### Sprint 3 — Weeks 5–6: Morning Intelligence + Calibration

#### Phase 62 — Morning Pre-Market Intelligence Digest

**What we build:**
1. Pre-market scan at 08:00 ET: fetch overnight news for full universe (top 100 swing + Russell 1000 intraday)
2. Score each cluster through `score_news_event()` (Phase 60)
3. Build structured per-symbol risk summary before market open
4. Ground the existing Haiku briefing (audit task #3) with these structured signals — the narrative becomes a synthesis of real scored events, not just a free-form summary
5. PM's 09:45 scan consumes the pre-market signal cache rather than making fresh calls

**Why this matters:** The 09:45 scan is the most time-sensitive moment. Having pre-scored signals cached from 08:00 means zero LLM latency at scan time, and the briefing becomes actually informative instead of generic.

**Gate:** Verify pre-market signals are populated in cache before 09:30 ET on 5 consecutive trading days.

---

#### Phase 57 — Paper Trading Review + Calibration (existing backlog)

**Precondition:** 4 weeks of paper trading + `decision_audit` populated.

**What to review:**
- Actual vs. simulated Sharpe per strategy
- Slippage impact (actual fills vs. proposal price)
- RM veto rate by rule — any rule firing too frequently or never?
- Earnings gate: how many trades were blocked, how many would have been profitable?
- News gate (Phase 60): `news_layer_alpha` — did the news gate add or subtract alpha?
- Which symbols generate the most intraday trades — any concentration risk?

**Gate for continuing live trading:** 4-week paper Sharpe > 0.5, max drawdown < 5%.

---

### Future — After Paper Trading + Calibration (Phase 63+)

These are deferred until we have real performance data.

#### Phase 63 — News as PM Scoring Overlay

**Precondition:** Phase 61 decision audit populated with 30+ days of data showing news gate adds value.

Replace the hard entry-block gate with a continuous scoring overlay on top of the model score:

```python
news_overlay = (
    0.10 * signal.upside_catalyst_score
  - 0.15 * signal.downside_risk_score
) * signal.confidence * (1.0 - 0.5 * signal.already_priced_in_score)

final_score = model_score + news_overlay
```

Size adjustments based on confidence thresholds (as in spec Section 11.3).

**Gate:** Backtested `news_layer_alpha > 0` on decision audit data.

---

#### Phase 64 — News as Model Feature (NIS Phase 4 equivalent)

**Precondition:** 60+ trading days of point-in-time `news_signal` history.

Only then should we consider promoting news features from runtime overlay into the XGBoost model itself. The spec is exactly right about this: you can't train on data you don't yet have.

Add `SymbolNewsFeatures` (materiality_decayed_4h, direction_decayed_3d, article_count_4h, etc.) as training features. Walk-forward gate: Sharpe improvement > 0.10, no lookahead leakage.

---

#### Phase 65 — Source Expansion + Clustering

Once we have Phase 63 working and have baseline accuracy measurements, add:
- Reuters/MarketWatch RSS (source diversity for clustering)
- Trading halts feed (NASDAQ Trader RSS) — hard gate, no LLM needed
- Simple story clustering (same symbol + same hour = likely same event)

The full TF-IDF clustering from the spec is probably overkill even then, but the halts feed is genuinely high-value.

---

#### Phase 66 — Sonnet Escalation

Once Haiku scoring is stable and we have enough volume to measure its error rate:
- Escalate to Sonnet for: `materiality_score >= 0.70`, `confidence < 0.60`, or `event_type in {m_and_a, fda_action, legal_regulatory}`
- Budget separately (Sonnet costs ~20× more per call)
- Gate: escalation improves decision accuracy measurably in A/B test

---

## Part 3 — One-Page Summary

```
DONE (2026-04-28 paper trading session):
  Phase A  — State durability (reconcile, DailyState, PM flag persistence)
  Phase B  — Swing recovery on late restart
  Phase C  — Proposal hygiene (dedup, 4 PM Redis purge)
  Phase 67 — Trader entry quality check (price run, spread, momentum, volume)
  Phase 68 — Live macro/market gate (SPY drawdown, FOMC, swing-blocked)
  Phase 69 — Dynamic stop trailing + partial exits (T1, VIX tightening, target ext.)

NEXT (before next trading session):
  Phase 70 — PM re-scoring of unexecuted candidates (stale entry withdrawal)
  Phase 71 — Correlation/sector concentration in RM

SPRINT 1 — Weeks 1-2: Calendar Intelligence
  Phase 58 — Earnings Calendar Gate        ← HIGHEST PRIORITY
  Phase 59 — Macro Calendar Awareness
  Phase 51 — Multi-scan Intraday (if Phase 50 passes)

SPRINT 2 — Weeks 3-4: Structured News + Audit
  Phase 60 — Structured NewsSignal + Haiku LLM scorer
  Phase 61 — Decision Audit Trail

SPRINT 3 — Weeks 5-6: Morning Intelligence + First Review
  Phase 62 — Morning Pre-Market Intelligence Digest
  Phase 57 — Paper Trading Review + Calibration (4-week mark)

FUTURE (after calibration):
  Phase 63 — News as scoring overlay (continuous, not boolean)
  Phase 64 — News as model features (needs 60 days history)
  Phase 65 — Source expansion + basic clustering
  Phase 66 — Sonnet escalation for high-stakes events
```

---

## Part 4 — Explicit Decisions on Open Questions from NIS Spec (Section 18)

| Question | Decision |
|---|---|
| SQLite vs Postgres | Stay on SQLite. At 500 articles/day + Phase 64 backfill we're well within SQLite limits. Revisit if we expand universe beyond Russell 1000. |
| `NewsMonitor` vs `NewsIntelligenceService` | Evolve `NewsMonitor` in place. Don't rename or replace — just upgrade its output from bool to `NewsSignal`. NIS full architecture is overkill at our stage. |
| Deprecate Polygon swing features now? | No. Keep `news_sentiment_3d/7d/momentum` in the swing model until Phase 64 produces validated replacements. Never remove a feature that's in a gate-passed model without a retrain. |
| Golden test fixtures for news? | Capture during Phase 60 live validation. We don't have a historical corpus yet. |
| One branch or many tasks per Phase 1? | One PR per phase, consistent with existing practice. Phase 58 and 59 can be one PR (both are calendar-related). |
| Pre-market briefing: replace or augment? | Augment. Keep the Haiku narrative (existing audit task #3), but seed it from structured Phase 60 signals so it's grounded. |
| Universe expansion for Phase 65? | Stay within current S&P 500 / Russell 1000 universe. No expansion until live trading shows the edge holds. |
