# Real-Time Trading Architecture — Phased Plan

**Created:** 2026-04-28
**Context:** MrTrader is paper trading as of 2026-04-28 with validated swing (v119, +1.181 Sharpe)
and intraday (v29, +1.807 Sharpe) models. The current architecture is batch-driven —
PM scores once at 8:00 AM, proposals are sent at 9:50, and the Trader executes mechanically.
This plan moves toward a fluid, position-aware, continuously reassessing system.

---

## The Problem We're Solving

The current flow is:
```
08:00 PM scans universe → 09:50 proposals sent → RM approves → Trader executes
         (nothing changes after this until next morning)
```

What we want:
```
Market hours → PM + RM + Trader continuously collaborate
             → aware of open positions and their current risk
             → dynamic entries when conditions are right
             → dynamic exits when conditions deteriorate
             → all three agents have shared context
```

---

## Phase 59 — Remove Rule-Based Entry Gate (TODAY ✅ DONE)

**Problem:** `generate_signal()` (EMA/RSI Tier 1 rules) was blocking every ML-approved proposal.
The walk-forward validation never applies this gate, so live behavior didn't match backtest.

**Fix:** Trader now gates on ML confidence score (>= 0.55) instead of `is_buy`.
`generate_signal()` is still called for ATR-based stop/target price calculation only.

**Files changed:** `app/agents/trader.py`

---

## Now — Paper Trading Observation Period (2–4 weeks)

Before building more, we need real data from the system running live.

**Watch for:**
- Are ML-approved trades actually entering now? (yes, post Phase 59)
- What's the win rate vs. backtest? (expect some degradation — that's normal)
- Which symbols are being proposed repeatedly? Are they sensible?
- Are stops/targets being hit intraday or held overnight?
- Does the PM abstention gate (VIX/SPY) trigger during volatile sessions?
- Are there proposal symbols that keep failing RM rules?

**Do not implement further phases until you have 2 weeks of paper trade data.**

---

## Phase 60 — Position Risk Monitoring (Week 2–3)

**Why next:** Once trades are entering, the gap is that nothing watches them actively.
A position that's down 1.5% intraday just sits there until its stop is hit or it's 5pm.

**What to build:**
- PM heartbeat (currently 30-min) checks each open position's current price vs. entry
- If a position drops > 0.7× ATR intraday (approaching stop), PM sends a "watch" flag to RM
- RM can tighten the stop to breakeven if the position has already moved in our favor
- No new ML scoring needed — purely position-level arithmetic

**Gate to proceed:** >= 10 closed trades in paper trading with stop/target data.

---

## Phase 61 — Shared Position Context for PM (Week 3–4)

**Why:** PM currently proposes entries without knowing the current book's risk profile.
It might propose adding NWSA when you're already long 3 high-beta names into a weak tape.

**What to build:**
- PM reads open positions (from Trader's `active_positions`) before scoring new entries
- Reduces confidence score for symbols in the same sector as existing positions
- Reduces universe to exclude stocks with open positions already
- No new model training — context injection into existing scoring logic

**Gate to proceed:** At least 5 simultaneous positions observed in paper trading.

---

## Phase 62 — Intraday Regime Re-Check (Week 3–4)

**Why:** The PM abstention gate runs once at session start. If SPY drops 2% at 11 AM,
new intraday entries should be blocked — but currently they aren't, because the gate
already passed at 9:45.

**What to build:**
- Trader checks a lightweight regime flag before each entry (not just at session start)
- If SPY is down > 1.5% from open OR VIX spikes > 5 points intraday, Trader skips new entries
- Existing positions are unaffected (don't exit just because regime shifted)
- This is a guard, not a full re-score: fast, cheap, runs in the Trader loop

**Gate to proceed:** Observe at least one volatile session in paper trading.

---

## Phase 63 — Intraday Re-Scoring on Adverse Moves (Month 2)

**Why:** The current model scores stocks once. If a position moves -1% in the first 30 min
and fundamentals haven't changed, we have no mechanism to decide "stay" vs "exit early."

**What to build:**
- For each open intraday position, re-run the intraday model score every 30 minutes
- If re-score drops below 0.45 (was above 0.55 at entry), send an early-exit proposal to RM
- RM evaluates: if position is also below breakeven, approve early exit
- If re-score stays high, hold — the model still believes in the trade

**Gate to proceed:** Phase 62 complete + >= 30 intraday closed trades in paper.

---

## Phase 64 — Event-Driven Entry Triggers (Month 2–3)

**Why:** The current 9:45–10:30 intraday scan window is arbitrary. Some of the best
intraday setups happen at 11 AM or 2 PM when a catalyst breaks.

**What to build:**
- NewsMonitor already polls every 5 minutes — wire a "catalyst flag" to the PM
- If a strong positive news signal arrives for a symbol in the universe, PM does an
  on-demand mini-scan for that symbol (not the whole universe)
- RM sees the proposal with a `trigger: news_catalyst` tag and applies tighter size limits
- This is additive — doesn't change the scheduled scan, just adds an opportunistic path

**Depends on:** Phase 58 (earnings calendar) complete so we don't enter before earnings.

---

## Phase 65 — Proactive RM (Month 3)

**Why:** RM currently only reacts to proposals. It has no ability to say "this book looks
too correlated, reduce exposure proactively" or "we've had 3 consecutive losses, pause."

**What to build:**
- RM gets a new periodic loop (every 30 min during session) independent of proposals
- Checks: sector concentration > 30%, 3 consecutive losses today, portfolio beta > 1.5
- Can send voluntary "reduce" signals to Trader for specific positions
- PM is notified so it doesn't re-propose the same symbols

**Gate to proceed:** Phase 61 complete (shared position context exists).

---

## Phase 66 — Full Decision Audit Trail (Month 3–4)

**Why:** Once all three agents are making dynamic decisions, we need structured records
of *why* each decision was made — not just what was decided. This is the precondition
for calibration and improving the models with live feedback.

**What to build:**
- Every entry/exit/skip decision records: ML score, regime state, open positions at the time,
  news flags active, RM rule outcomes, final action
- Stored in a `decision_audit` table (not just the existing `DecisionLog`)
- Weekly report: what did the system decide, why, and was it right?
- This feeds Phase 57 calibration (comparing live decisions to paper trading expected outcomes)

---

## What We Are NOT Building (and Why)

| Idea | Why Deferred |
|---|---|
| Tick-by-tick streaming (WebSocket price feed) | Overkill for 5-min intraday bars; adds infra complexity for marginal gain |
| Reinforcement learning entry/exit | Requires months of live data + careful reward shaping; premature now |
| Multi-stock portfolio optimizer (mean-variance) | SP-100 universe is already diversified; formal optimization is Phase 70+ |
| GPT-4 real-time market commentary | Cost + latency don't justify vs. structured signals we already have |
| Options hedging | Different product entirely; out of scope until equity system is stable |

---

## Summary Timeline

| Phase | What | When |
|---|---|---|
| **59** | Remove rule-based entry gate — ML score drives entries | ✅ Today |
| **Observe** | 2–4 weeks paper trading before building more | Week 1–2 |
| **60** | Active position monitoring + stop tightening | Week 2–3 |
| **61** | PM reads open book before proposing new entries | Week 3–4 |
| **62** | Intraday regime re-check in Trader loop | Week 3–4 |
| **63** | Intraday re-scoring on adverse moves | Month 2 |
| **64** | Event-driven entries from news catalyst | Month 2–3 |
| **65** | Proactive RM — portfolio-level risk reduction | Month 3 |
| **66** | Full decision audit trail | Month 3–4 |
