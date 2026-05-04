# MrTrader — Active Phase Roadmap

**Last updated:** 2026-05-02
**Status:** Paper trading active. Safety phases all green. Phase 87 training in progress.
**Completed phases:** `docs/phases_archive.md`

---

## Current Model State

| Model | Version | Avg Sharpe | Gate | Notes |
|---|---|---|---|---|
| Swing | v119 | +1.181 | ✅ >0.80 | Active |
| Intraday | v29 | +1.830 | ✅ >1.50 | Active (+ Phase 85 gates) |
| Intraday | v38 | TBD | 🔄 | Phase 87 training 2026-05-02 |

---

## Intraday ML Improvement Sequence (Active)

### Phase 86 — Market Context Features ❌ REVERTED / REDESIGN PENDING

**Root cause of failure:** All 5 SPY features (spy_first_hour_range, spy_5d_return, etc.) are market-wide — same value for every symbol per day. After `cs_normalize` they become zero. v36 avg Sharpe: **-0.529**.

**Redesign (Phase 86b — after Phase 87):** Stock-relative interaction features that survive cs_normalize:
- `stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`
- SPY daily bars plumbing already wired through simulator, walkforward, PM — no re-plumbing needed.

---

### Phase 87 — Realized-R Labels + 3-Seed Ensemble + Frozen HPO 🔄 IN PROGRESS

**Branch:** `feat/phase-87-label-fix-ensemble` (PR #121, 2026-05-02)

**Three changes bundled:**

1. **Realized-R labels (B+A):** `realized_R ≥ 0.5 AND abs_move ≥ 0.30%` → label 1. Days with no qualifying setups → all zero labels. No more forced top-20% on bad days.

2. **3-seed XGBoost ensemble (permanent):** Seeds 42/123/777, blend probabilities. Eliminates ~2.0 Sharpe HPO variance (v29 +1.830 vs v37 -0.219 on identical features).

3. **100-trial Optuna HPO → FROZEN_HPO_PARAMS:** Thorough search once, freeze params for all future retrains.

**Label formula:**
```python
realized_R = (realized_exit - entry) / stop_dist_use
absolute_move = abs(realized_exit - entry) / entry
label = 1 if (realized_R >= 0.5 and absolute_move >= 0.003) else 0
```

**Acceptance criteria:** avg Sharpe > 1.50, no fold < -0.30.

**Walk-forward:** Pending (training in progress 2026-05-02).

---

### Phase 86b — Stock-Relative Interaction Features ⏳ AFTER PHASE 87

**Precondition:** Phase 87 walk-forward result in hand.

Features that vary by symbol (survive cs_normalize):
- `stock_vs_spy_5d_return`: stock 5d return − SPY 5d return
- `stock_vs_spy_mom_ratio`: stock 1d momentum / SPY 1d momentum
- `gap_vs_spy_gap`: stock overnight gap − SPY overnight gap

**Files:** `app/ml/intraday_features.py`, `app/ml/intraday_training.py`
**Branch:** `feat/phase-86b-stock-relative-features`

---

### Phase 87a — Regression Labels (Realized R-Multiple) ⏳ AFTER PHASE 86b

**Precondition:** Phase 87 ✅ + Phase 86b ✅.

Replace binary classification with regression target (predict realized R-multiple). XGBoost objective: `reg:squarederror`. Rank by predicted R directly (no probability threshold). Label clipped to [-3.0, +3.0].

**Why deferred:** Changes the entire scoring pipeline. Do after binary classifier is stable with a baseline to compare against.

**Branch:** `feat/phase-87a-regression-labels`

---

## Phase 88 — Dynamic Regime Gates ⏳ AFTER PHASE 86b

**Precondition:** Phase 87 ✅ + Phase 86b (stock-relative features) ✅ — conceptually related, implement together.

**Problem with current Phase 85 gates:**
- VIX ≥ 25 is a hard binary cutoff — VIX 24.9 trades, VIX 25.1 doesn't. No reason that threshold is special.
- SPY < MA20 blocks the entire universe even when specific sectors are trending well independently.
- Both gates are fully on/off — no gradation for borderline days.
- All market-wide context stays in PM-level gates because cs_normalize would zero it out if baked into the model.

**What to build:**

### 88a — Sector-Level Abstention Gates

Replace the universe-wide SPY MA gate with per-sector gates:
- Map each candidate symbol to its sector ETF (XLK, XLF, XLE, XLV, XLI, etc.) using existing `_get_symbol_sector()`
- For each sector: compute sector ETF vs its own 20-day MA
- Block candidates in sectors where the sector ETF is below its MA
- Allow candidates in sectors where the sector ETF is above its MA — even if SPY overall is weak

This lets energy stocks trade on a strong oil day while tech is blocked after a sector rotation.

**Files:** `app/agents/portfolio_manager.py` — `select_intraday_instruments()`, new `_sector_allows_entry(symbol)` helper

### 88b — Continuous Opportunity Score (Replace Binary VIX Gate)

Instead of hard VIX ≥ 25 block, compute a 0–1 day opportunity score:

```python
def _intraday_opportunity_score(vix, vix_5d_avg, spy_first_hour_range, spy_first_hour_eff):
    # VIX: score drops as VIX rises, but also penalizes rapidly expanding VIX
    vix_score = max(0.0, 1.0 - (vix - 15) / 20)           # 1.0 at VIX=15, 0.0 at VIX=35
    vix_trend = max(0.0, 1.0 - (vix - vix_5d_avg) / 5)    # penalty if VIX spiking vs 5d avg
    # First-hour range: more range = more opportunity
    range_score = min(1.0, spy_first_hour_range / 0.008)   # full score at 0.80%+ first-hour range
    # First-hour efficiency: trending (not choppy)
    eff_score = spy_first_hour_eff                          # already 0–1
    return 0.35*vix_score + 0.20*vix_trend + 0.30*range_score + 0.15*eff_score
```

Use score to scale `max_candidates` rather than binary block:
- Score ≥ 0.70 → normal (max_candidates = 5)
- Score 0.40–0.69 → reduced (max_candidates = 2)
- Score < 0.40 → skip (abstain entirely)

### 88c — Intraday Regime Signal (First 30 Minutes)

SPY's first 30 minutes (bars 0–5) characterizes the day's regime:
- **Trending open:** SPY moves directionally >0.4% with efficiency >0.7 → increase max_candidates
- **Choppy open:** SPY range >0.5% but efficiency <0.4 (whipsaw) → reduce max_candidates to 1
- **Quiet open:** SPY range <0.2% → VIX-like low-opportunity signal, reduce candidates

**Files:** `app/agents/portfolio_manager.py` — `_market_regime_allows_entries()` refactored into `_compute_opportunity_score()`

**Walk-forward validation:** Run with Phase 85 gate replaced by Phase 88 dynamic gates. Gate: avg Sharpe ≥ v29+Phase85 baseline (+1.830). If equal or better with fewer blocked days → improvement confirmed.

**Important constraint:** All of 88a/b/c stay as PM-level runtime logic, not model features. Market-wide and sector-wide signals are zeroed by cs_normalize in the model — they must live in the PM layer.

**Branch:** `feat/phase-88-dynamic-regime-gates`

---

## Phase 89 — Training/Inference Data Alignment

**Motivation:** Audit (2026-05-03) revealed training data must mirror live inference data exactly. Currently several features are zeroed during training but fetched live, or vice versa. Goal: backfill historical values point-in-time so the model trains on the same signals it scores with.

### Phase 89a — Historical Fundamentals Backfill (High, 3 days)

**Problem:** `revenue_growth` is the top SHAP feature at live inference (fetched from SEC EDGAR/yfinance), but is zeroed during training (`--no-fundamentals`). Model learns on price signals only, then scores live setups using fundamental values it was never trained on.

**What to build:**
- `scripts/backfill_fundamentals_history.py` — quarterly P/E, P/B, revenue growth, profit margin, debt/equity from SEC EDGAR XBRL. Store by filing date (not report date) for point-in-time safety.
- `data/fundamentals/` parquet store keyed by `(symbol, as_of_date)`
- Update `train_model.py` to load from parquet instead of live API — remove `--no-fundamentals` flag
- Un-prune `pe_ratio`, `pb_ratio`, `revenue_growth`, `profit_margin`, `debt_to_equity` from `PRUNED_FEATURES`

**Acceptance criteria:** Walk-forward avg Sharpe improvement > 0.05 vs v119 baseline (+1.181).

**Branch:** `feat/phase-89a-historical-fundamentals`

---

### Phase 89b — Sector ETF History (Medium, 1 day)

**Problem:** `sector_momentum` and `sector_momentum_5d` are fetched live at inference (XLK, XLF, XLE etc. ETF returns) but zeroed during training. SPY daily bars plumbing already exists — extend it to sector ETFs.

**What to build:**
- Extend Polygon daily bar cache to include 11 sector ETFs (XLK, XLF, XLE, XLV, XLI, XLB, XLRE, XLU, XLP, XLY, XLC)
- Pass sector bars into training the same way `spy_daily_bars` is passed today
- Un-prune `sector_momentum`, `sector_momentum_5d` from `PRUNED_FEATURES`
- Also enables Phase 88a (sector-level abstention gates) with historical data for validation

**Acceptance criteria:** Sector momentum SHAP > 0.05 in trained model (confirms signal survives cs_normalize).

**Branch:** `feat/phase-89b-sector-etf-history`

---

### Phase 89c — Historical Options IV (Low, 2 days)

**Problem:** `options_iv_atm`, `options_put_call_ratio`, `vrp` fetched live via yfinance options chain, but training defaults them to 0.0. Options history on free tier only goes back ~2 years.

**What to build:**
- `scripts/backfill_options_iv.py` — daily ATM IV and put/call ratio from yfinance options history
- Wire into training feature computation for walk-forward folds within the 2yr window
- Alternative: drop these features entirely if backfill coverage is insufficient

**Acceptance criteria:** Either confirms IV adds signal (Sharpe improvement) or cleanly removes from feature set.

**Branch:** `feat/phase-89c-historical-options-iv`

---

## Safety + Hardening Backlog

### Phase 99 — Decouple Nightly Retraining from Uvicorn (Medium, 1 day)

**Problem:** Retraining runs inside the uvicorn process via `loop.run_in_executor`. On a 32-core machine, even 24 workers saturate file descriptors enough to drop the HTTP listener socket, killing the dashboard mid-training.

**What to build:**
- `scripts/retrain_scheduled.py` — standalone script that trains and writes results to DB, callable from CLI or subprocess
- Orchestrator spawns it via `asyncio.create_subprocess_exec` instead of `run_in_executor`
- Stdout/stderr piped to `logs/retrain_YYYY-MM-DD.log`
- Uvicorn stays isolated: no worker threads competing for FDs

**Files:**
- `scripts/retrain_scheduled.py` (new)
- `app/orchestrator.py` — replace `run_in_executor` with `create_subprocess_exec`

**Acceptance criteria:** Retraining completes without dashboard dropping. Dashboard accessible throughout training run. Log file captures full training output.

---

### Phase 77 — Decision-Audit Dashboard (High, 2 days)

`decision_audit` table is populated but unread. Cannot answer "are the gates working?"

**What to build:**
- `GET /api/audit/summary`: aggregate win-rate by `block_reason`, model_score bucket vs realized 4h return
- Frontend tile + `scripts/backfill_decision_outcomes.py` (16:30 ET cron)

**Acceptance criteria:** After 2 weeks, can read "NIS sizing_multiplier 0.7–0.8 correlates with X bps lower realized return."

---

### Phase 79 — Point-in-Time Index Membership (High, 3 days)

Static universe lists (`SP_500_TICKERS`, `RUSSELL_1000_TICKERS`) reflect ~early 2026 membership. Delisted stocks 2021–2026 silently absent from training — inflates walk-forward Sharpe.

**What to build:**
- `app/data/universe_history.py` with `members_at(date)`
- `data/universe/sp500_membership.parquet`, `data/universe/russell1000_membership.parquet`
- Update `train_model.py` and `walkforward_tier3.py` to use `members_at(fold_train_start)`

**Acceptance criteria:** `members_at(date(2022,1,1))` returns names since delisted (e.g. WORK, PAGS).

---

### Phase 80 — Bar-12 Intraday Sensitivity Test (High, 1 day)

Bar 12 was chosen by intuition. Bars 9–15 have never been swept. If only bar 12 shows Sharpe > 1.5, the edge may be in-sample noise.

**What to build:** `scripts/bar_sensitivity.py --entry-offset N` → Sharpe + win-rate + trade count. Run for offsets 9–15.

**Acceptance criteria:** If 4 of 5 bars (10–14) pass Sharpe > 0.8 → robust. If only bar 12 → downgrade Sharpe expectations.

---

### Phase 84 — Integration Test: Full PM→RM→Trader Round Trip (Medium, 2 days)

No e2e test drives PM→RM→Trader→Alpaca-paper with persistent state. All "integration" tests use mocks.

**What to build:** `tests/test_e2e_round_trip.py` — drives one swing proposal end-to-end. Asserts Trade.status='ACTIVE' after fill; Trade.status='CLOSED' + pnl after stop/exit. Run on demand only (not in CI).

---

## NIS + Features Backlog

### Phase 64 — NIS as Swing Model Features ⏳ (overnight backfill first)

**Precondition:** Run `python scripts/backfill_nis_history.py --days 252 --workers 4` overnight.

**NIS features to add to swing training:**
- `nis_materiality_decayed_4h`, `nis_direction_score_3d`, `nis_article_count_4h`
- `nis_already_priced_in`, `nis_sizing_mult`
- Use point-in-time lookup only — no lookahead

**Gate:** Walk-forward avg Sharpe improvement > 0.05 vs baseline.

---

### Phase 65 — Source Expansion (after Phase 64 baseline)
- Reuters/MarketWatch RSS as second source
- Trading halts feed (NASDAQ Trader RSS) — hard gate, no LLM needed

### Phase 66 — Sonnet Escalation (after Haiku error rate measurable)
- Escalate to Sonnet for `materiality_score >= 0.70` or M&A/FDA/legal event types

---

## Agent Architecture + Intelligence Improvements

*Inspired by analysis of TradingAgents (TauricResearch/TradingAgents, arXiv 2412.20138) — a multi-agent LLM trading framework. We reviewed their architecture and extracted what's genuinely useful without abandoning the XGBoost + walk-forward rigor that gives MrTrader its reproducible edge.*

---

### Phase 91 — Pre-Execution Bull/Bear Debate Agent ⏳ AFTER PHASE 86b

**Origin:** TradingAgents' most novel contribution is a structured adversarial debate before any investment decision. Two LLM agents argue opposite sides using the same evidence; a judge synthesizes. We adapt this to MrTrader's existing approval workflow as a second-pass filter on top of the ML signal, not a replacement.

**Problem with current flow:**
The PM generates a proposal from the XGBoost model score + NIS score alone. There is no mechanism to catch cases where the ML says "strong setup" but the news/fundamentals context says "this is a bad day for this stock specifically." A single ML score collapses all signals into one number and can't express internal contradiction.

**What to build:**

A `DebateAgent` that runs after the PM generates a proposal but before it is sent to the RM queue. For each proposed trade the agent:

1. Receives a structured `DebateContext`:
   - Symbol, strategy (swing/intraday), ML model score + top 5 SHAP features
   - NIS scores: `direction_score`, `materiality_score`, `sizing_multiplier`, top headlines
   - Key technicals: RSI, ATR%, recent momentum, proximity to support/resistance
   - Market regime: VIX level, SPY vs MA20, sector ETF trend
   - Proposed entry/stop/target and position size

2. Runs two Claude Haiku calls in parallel:
   - **Bull agent:** "Argue the strongest case FOR entering this trade given the above context."
   - **Bear agent:** "Argue the strongest case AGAINST entering this trade given the above context."

3. A **judge call** (Haiku or Sonnet depending on materiality) reads both arguments and returns a structured `DebateVerdict`:
   - `proceed: bool`
   - `confidence_adjustment: float` (e.g. -0.15 = reduce position 15%)
   - `key_risk: str` (one-line bear argument that was most compelling)
   - `key_thesis: str` (one-line bull argument that was most compelling)

4. PM uses verdict to:
   - Withdraw the proposal if `proceed=False` (logs reason to `decision_audit`)
   - Scale down position size by `confidence_adjustment` if negative
   - Attach `key_risk` and `key_thesis` to the proposal for audit trail

**Why this works alongside XGBoost (not instead of it):**
- XGBoost provides the quantitative edge: reproducible, backtestable, no hallucination risk
- Debate provides the qualitative filter: catches macro/news context the model can't see
- Debate only fires on proposals that already passed the ML gate — it's a second filter, not the first

**Cost estimate:** ~3 Haiku calls per proposal × ~$0.001 each × ~5 proposals/day = $0.015/day. Negligible.

**Precondition:** Phase 86b complete (need stable v39 model as baseline before adding debate overhead).

**Files:**
- `app/agents/debate_agent.py` — new `DebateAgent` class
- `app/agents/portfolio_manager.py` — `_build_proposals()` calls debate before enqueuing
- `app/database/models.py` — add `debate_verdict` JSON column to `decision_audit`

**Acceptance criteria:** After 2 weeks of paper trading with debate enabled, `decision_audit` shows debate-withdrawn proposals had materially worse realized returns than debate-approved ones. Gate: debate-withdrawn avg realized return < debate-approved avg realized return.

**Branch:** `feat/phase-91-debate-agent`

---

### Phase 92 — PM Decomposition ⏳ AFTER 4+ WEEKS PAPER TRADING DATA

**Origin:** TradingAgents splits decision-making into Analyst → Researcher → Risk Manager → Portfolio Manager agents. Our PM is a 2,495-line file doing everything. The split makes sense — but only after paper trading is stable and we have behavioral baselines to verify nothing broke.

**Why not do this now:**
- PM is stable and tested. A refactor mid-paper-trading creates verification risk.
- Need 4+ weeks of paper trading logs to establish behavioral baseline (proposal counts, block rates, position sizes) before restructuring.
- Phase 92 is purely architectural — no new signal, no Sharpe improvement. Do it when the benefit (testability, extensibility) outweighs the risk.

**Problem with current PM:**
`portfolio_manager.py` owns five distinct responsibility layers that are tightly coupled:
1. **Data fetching** (VIX, SPY bars, earnings calendar, features, NIS)
2. **Universe filtering** (regime gates, symbol-level entry gates, candidate selection)
3. **Intelligence synthesis** (ML scoring, NIS weighting, proposal construction)
4. **Capital allocation** (position sizing, portfolio concentration limits)
5. **Orchestration** (event loop, task scheduling, EOD jobs, retrain triggers)

This makes it hard to test any one layer in isolation, impossible to parallelize data fetching, and expensive to add new signal sources (everything goes into one file).

**Proposed decomposition:**

**`MarketIntelligenceAgent`** — data gathering only, no decisions:
- Absorbs: `_run_premarket_intelligence()`, `_run_morning_nis_digest()`, `_fetch_swing_features()`, `_prefetch_earnings_calendar()`, `_fetch_vix_level()`, `_get_spy_intraday_state()`, `_fetch_target_upside()`
- Produces: `IntelligenceReport` (Pydantic model) written to DB or passed via queue
- Runs concurrently during premarket window; PM consumes the report, not the raw data
- Independently testable: mock the APIs, assert `IntelligenceReport` fields
- Natural home for Phase 91 debate context construction

**`UniverseFilterAgent`** — eligibility decisions only, no sizing:
- Absorbs: `_market_regime_allows_entries()`, `_check_swing_entry_gates()`, `_check_intraday_entry_gates()`, `select_instruments()`, `select_intraday_instruments()`
- Produces: `EligibleUniverse` (list of symbols + block reasons for rejected ones)
- All gate logic isolated here — Phase 88 dynamic regime gates are added here, not to PM
- Makes Phase 88 and Phase 91 composable: filters run before debate, debate runs before sizing

**`PortfolioManager`** (slimmed) — decisions and orchestration only:
- Retains: `_build_proposals()`, `_calculate_quantity()`, `_rescore_pending_approvals()`, `_review_open_positions()`, `run()`, task scheduling
- Consumes `IntelligenceReport` + `EligibleUniverse`, produces trade proposals
- Target: ~800 lines (down from 2,495)

**Migration approach:**
- Extract agents one at a time, each behind a feature flag
- Run both old and new code in shadow mode for 1 week per agent, assert identical outputs
- Only retire old code after shadow validation passes

**Files:**
- `app/agents/market_intelligence_agent.py` — new
- `app/agents/universe_filter_agent.py` — new
- `app/agents/portfolio_manager.py` — slimmed, consumes the above
- `app/database/models.py` — `IntelligenceReport` table for audit/replay

**Acceptance criteria:** Full pytest suite passes. Paper trading proposal rate, block rate, and position sizes are statistically identical before and after (±5% over 2-week shadow period).

**Branch:** `feat/phase-92-pm-decomposition`

---

### Phase 93 — Structured Decision Schemas (Pydantic) ⏳ ALONGSIDE PHASE 92

**Origin:** TradingAgents uses Pydantic models (`ResearchPlan`, `TraderProposal`, `PortfolioDecision`) for all inter-agent communication. MrTrader passes raw dicts between PM→RM→Trader, which means bugs only surface at runtime when a key is missing or misnamed.

**Problem:**
```python
# Current: runtime dict — no type checking, no validation
proposal = {"symbol": "AAPL", "model_score": 0.72, ...}
# A missing key causes KeyError deep in Trader, not at proposal creation
```

**What to build:**
Pydantic schemas for every inter-agent message:
- `SwingProposal` / `IntradayProposal` — PM → RM queue
- `ApprovedTrade` — RM → Trader queue
- `TradeOutcome` — Trader → PM (fill confirmation, stop hit, exit)
- `IntelligenceReport` — MarketIntelligenceAgent → PM (Phase 92)
- `EligibleUniverse` — UniverseFilterAgent → PM (Phase 92)
- `DebateVerdict` — DebateAgent → PM (Phase 91)

**Why this matters:**
- Validation at construction time, not at consumption time — bugs surface at the source
- Self-documenting inter-agent contracts
- Makes unit testing trivial: build a valid schema object, pass it, assert outputs
- Required foundation for Phase 92 decomposition

**Files:** `app/agents/schemas.py` — all inter-agent Pydantic models

**Acceptance criteria:** All queue messages constructed via schemas. `mypy` passes on agent files. No raw dict construction in PM/RM/Trader for inter-agent messages.

**Branch:** `feat/phase-93-pydantic-schemas` (can be done as part of Phase 92)

---

## LLM Decision Enhancement Phases

**Design principle:** LLM adds value where context and nuance matter; ML adds value where pattern recognition over large datasets matters. Use LLM for one call per decision boundary event (morning, pre-trade, post-loss, weekly) — never in tight loops. Keep XGBoost as the primary signal generator. Cost budget: <$1/day across all phases below.

*All phases below use Claude Haiku unless noted. Haiku cost: ~$0.001/call. Sonnet: ~$0.01/call.*

---

### Phase 94 — Macro Morning Digest ⏳ AFTER PHASE 91

**Problem:** `_run_premarket_intelligence()` fetches macro data (VIX, SPY pre-market, Fed calendar, sector moves) but interprets it with hard rules only (VIX ≥ 25 = bad, SPY < MA20 = bad). There is no synthesis of *why* today looks the way it does or what it means for specific sectors and strategies.

**What to build:**
One Haiku call at 8:30 AM ET each trading day. Input:
- VIX level + 5-day trend
- SPY pre-market move %
- Overnight macro headlines (top 3 from NIS global feed)
- Fed calendar (FOMC dates, scheduled speakers)
- Sector ETF pre-market moves (XLK, XLF, XLE, XLV)
- Previous day's win/loss ratio for reference

Output: `MacroContext` struct:
```python
{
  "regime_label": "risk-off",          # one of: risk-on, risk-off, neutral, event-driven
  "regime_confidence": 0.82,
  "narrative": "Tariff escalation overnight pushing VIX to 23. Tech pre-market -0.8%, energy flat. Fed speaker at 10am could add volatility. Favor defensive sectors; raise intraday score threshold.",
  "sector_bias": {"XLK": -1, "XLE": 0, "XLV": 1},  # -1 avoid, 0 neutral, +1 favor
  "caution_flags": ["fed_speaker_10am", "vix_elevated"]
}
```

`MacroContext` is:
- Attached to every proposal's debate context (Phase 91 bull/bear agents read it)
- Used by Phase 88b opportunity score as qualitative input
- Logged to `decision_audit` for each trading day

**Cost:** 1 call/day × $0.001 = ~$0.02/month. Negligible.

**Files:**
- `app/agents/market_intelligence_agent.py` — `_generate_macro_context()` method
- `app/database/models.py` — `MacroContext` table (one row per trading day)
- `app/agents/portfolio_manager.py` — pass `MacroContext` into proposal construction and Phase 91 debate

**Acceptance criteria:** `MacroContext` populated for every trading day in `decision_audit`. Phase 91 debate agent uses `narrative` field in bull/bear prompts. After 4 weeks: `regime_label=risk-off` days show lower avg proposal win rate than `risk-on` days (validates the signal).

**Branch:** `feat/phase-94-macro-morning-digest`

---

### Phase 95 — Intraday Exit Timing LLM Advisory ⏳ AFTER PHASE 91

**Problem:** When an open intraday position hits a re-evaluation trigger mid-hold (news event, approaching stop, sharp move in SPY), the PM re-scores mechanically with the ML model. The ML model was trained on bar-12 entry, not on mid-trade exit decisions. It has no concept of "this news just dropped and changes the thesis." Currently: either hold to bar-36 exit or stop out — nothing in between.

**What to build:**
A Haiku advisory call triggered when any of:
- New NIS score for the symbol during the hold window changes by > 0.3 vs entry NIS
- Position is down > 0.6R (approaching stop) and SPY is also down > 0.3%
- Position is up > 0.6R (approaching target) and NIS shows negative news

Input to advisory:
- Entry price, current price, unrealized P&L in R-multiples
- Time remaining in hold window (bars left)
- NIS score at entry vs. current NIS score + latest headline
- Current SPY move vs. entry SPY level
- ML model score at entry + top SHAP features

Output: `ExitAdvisory`:
```python
{
  "recommendation": "hold" | "tighten_stop" | "exit_now",
  "reasoning": "...",  # one sentence
  "stop_adjustment": -0.02  # optional: move stop X% closer to market
}
```

PM acts on `recommendation`:
- `exit_now` → send market exit order immediately, log reason
- `tighten_stop` → update stop level in Trade record, RM enforces
- `hold` → no action, log for audit

**Cost:** Fires rarely — only on trigger events. Estimate ~2 triggers/day × $0.001 = ~$0.04/month.

**Files:**
- `app/agents/portfolio_manager.py` — `_handle_reeval_requests()` calls advisory before ML re-score
- `app/database/models.py` — add `exit_advisory_json` column to `Trade` table

**Acceptance criteria:** After 4 weeks, trades where `exit_now` was recommended show better realized R than trades that held to natural exit on the same trigger events. Gate: advisory-exit avg R > hold-to-expiry avg R on trigger days (±0.1R tolerance).

**Branch:** `feat/phase-95-exit-advisory`

---

### Phase 96 — Post-Loss Postmortem Agent ⏳ AFTER PHASE 77 (AUDIT DASHBOARD)

**Problem:** When a trade loses significantly (> 1.0R loss), the only record is numbers in the `trades` table. There is no synthesis of *why* it lost or what could have been caught earlier. Losses are currently unlearnable events — they happen, get logged as numbers, and the model eventually retrains. A postmortem turns each loss into a structured learning artifact.

**What to build:**
A Haiku call triggered automatically when any trade closes with realized R < -0.8.

Input:
- Full trade record: symbol, entry time, exit time, entry price, exit price, realized R
- ML model score at entry + top 5 SHAP features (what the model liked about this setup)
- NIS scores at entry (was news a factor?)
- MacroContext at entry date (what was the regime?)
- SPY move during the hold window (was this market-driven or stock-specific?)
- Any re-evaluation events during the hold (stop adjustments, advisory calls)

Output: `PostMortem` stored in `decision_audit`:
```python
{
  "loss_category": "macro_driven" | "stock_specific" | "model_overconfident" | "news_reversal" | "stop_too_tight",
  "primary_cause": "SPY dropped 1.2% during hold, dragging stock despite strong setup",
  "what_model_missed": "High materiality NIS score at entry was already-priced-in per direction_score",
  "future_filter": "Consider blocking entries when NIS already_priced_in_score > 0.7 despite positive direction"
}
```

`future_filter` suggestions are logged and reviewed weekly (Phase 97). The best ones get promoted into actual gate rules.

**Cost:** Fires only on significant losses. Estimate ~3 losses/week × $0.001 = ~$0.01/month.

**Files:**
- `app/agents/portfolio_manager.py` — `_review_open_positions()` triggers postmortem on close
- `app/database/models.py` — `PostMortem` table
- Phase 77 dashboard — add postmortem tab showing recent loss_categories

**Acceptance criteria:** Every trade with R < -0.8 has a `PostMortem` record within 1 minute of close. After 8 weeks, `loss_category` distribution is readable (not all "unknown"). At least one `future_filter` suggestion has been validated and promoted to a real gate.

**Branch:** `feat/phase-96-postmortem-agent`

---

### Phase 97 — Weekly Performance Reflection ⏳ AFTER PHASE 96

**Problem:** `_generate_weekly_report()` produces a raw stats dump (win rate, total P&L, top winners/losers). There is no synthesis layer that reads across the week's decisions and identifies patterns — which gate is working, which setup type is failing, whether the model is performing differently in different regimes.

**What to build:**
One Sonnet call every Sunday at 6 PM ET (after market close Friday, before Monday premarket).

Input:
- Week's trade summary: all closed trades with realized R, symbols, strategies
- Block summary: how many proposals were blocked by each gate (VIX, SPY MA, NIS, debate)
- PostMortem summaries from the week (Phase 96 outputs)
- MacroContext labels for each day (Phase 94 outputs)
- Comparison to prior 4-week avg win rate and Sharpe

Output: `WeeklyReflection`:
```python
{
  "week_label": "2026-05-03",
  "regime_summary": "3 risk-off days, 2 neutral. Model performed well on risk-off (60% win rate) but poorly on neutral days (38%).",
  "what_worked": "NIS gate correctly blocked 4 of 5 eventual losers. Intraday setups in XLE outperformed.",
  "what_failed": "Swing model over-traded consumer discretionary in low-volatility chop. 3 of 4 losses were stock-specific, not macro.",
  "gate_effectiveness": {"nis_block": "effective", "vix_gate": "over-cautious", "debate_agent": "2 correct withdrawals"},
  "suggested_adjustments": "Consider raising swing score threshold from 0.65 to 0.70 on neutral-regime days.",
  "action_items": ["Review earnings gate sensitivity for next week (3 major reports)", "Monitor XLE sector ETF for Phase 89b signal"]
}
```

`WeeklyReflection` is:
- Stored in DB for audit trail
- Displayed on Phase 77 dashboard as weekly summary tile
- Used as context in the following week's Phase 94 MacroContext calls ("last week's suggested adjustments were...")

**Cost:** 1 Sonnet call/week × $0.01 = ~$0.04/month.

**Files:**
- `app/agents/portfolio_manager.py` — `_generate_weekly_report()` replaced by Sonnet call
- `app/database/models.py` — `WeeklyReflection` table
- Phase 77 dashboard — weekly reflection tile

**Acceptance criteria:** `WeeklyReflection` generated every Sunday. After 8 weeks, `suggested_adjustments` show measurable directional accuracy — i.e., weeks where a threshold was raised show higher win rate the following week vs. weeks where no adjustment was suggested.

**Branch:** `feat/phase-97-weekly-reflection`

---

### Phase 98 — Earnings Event Classification ⏳ AFTER PHASE 96

**Problem:** The earnings gate (Phase 81) currently blocks ALL trades within N days of earnings, both pre- and post-event. Post-earnings, many of the best setups occur — earnings beats with raised guidance often lead to multi-day momentum. Currently we gate out and miss these. The issue is distinguishing "risky pre-earnings hold" from "high-quality post-earnings entry."

**What to build:**
A Haiku call triggered when an earnings announcement is detected in the NIS feed for a symbol in our universe.

Input:
- Earnings headline + full article text (from NIS/Finnhub)
- Prior quarter's reported vs. estimated EPS
- Revenue beat/miss if available
- Guidance language (raised/maintained/lowered/withdrawn — extracted from article)
- Stock pre-market reaction (price move %)

Output: `EarningsClassification`:
```python
{
  "verdict": "strong_beat" | "beat" | "inline" | "miss" | "strong_miss",
  "guidance": "raised" | "maintained" | "lowered" | "withdrawn" | "not_provided",
  "one_time_items": True | False,   # large write-downs, restructuring etc.
  "post_entry_signal": "favorable" | "neutral" | "unfavorable",
  "entry_window": "immediate" | "wait_1d" | "avoid",
  "reasoning": "Beat on EPS (+12%) with raised FY guidance. Pre-market +4.5%. No one-time items. Strong post-earnings momentum setup."
}
```

PM uses `EarningsClassification`:
- `entry_window=immediate` + `post_entry_signal=favorable` → lift earnings gate for this symbol for 2 trading days
- `entry_window=avoid` → extend earnings gate by 3 additional days
- Log to `decision_audit` for each earnings event

**Cost:** ~3 earnings events/week in our universe × $0.001 = ~$0.01/month.

**Files:**
- `app/agents/market_intelligence_agent.py` — `_classify_earnings_event()` triggered by NIS feed
- `app/agents/portfolio_manager.py` — `_check_swing_entry_gates()` reads `EarningsClassification` before applying gate
- `app/database/models.py` — `EarningsClassification` table

**Acceptance criteria:** Every earnings announcement for a universe symbol gets classified within 15 minutes of release. After 8 weeks: `post_entry_signal=favorable` + `entry_window=immediate` trades show avg R > universe avg R (validates that post-earnings re-entry adds alpha vs. blanket gating).

**Branch:** `feat/phase-98-earnings-classification`

---

## Deferred (After Live Trading + Calibration)

| Phase | Name | Why Deferred |
|---|---|---|
| 57 | Paper trading calibration review | Need 4+ weeks live data + all safety phases green ✅ |
| 90 | Tax + P&L Impact Review | Pre-live gate. NJ ~47% combined rate on short-term gains. Wash sale frequency from paper data. |
| 51b | Multi-scan intraday tuning | Need live data to know if 11:00/13:30 windows add alpha |
| Regime v2 | Finer-grained regime (bull/bear/chop) | Need live performance to see where model fails |
| Phase 80b | True Technical Day Trader (IntradayScalper) | Different architecture; requires L1/L2 tick data |
