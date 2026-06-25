# Macro Intel — Workflow + Dashboard Scope & Fix Plan (2026-06-25)

**Status:** SCOPED, not yet implemented. Two Opus 4.8 design passes (macro workflow + page review).
**Owner decision required before build** — see §6 Open Decisions.

This doc covers BOTH halves of the Macro Intel system:
- **Part A — the macro RISK workflow** (how risk is computed, re-assessed post-event, and consumed by PM/RM/Trader).
- **Part B — the Macro Intel PAGE** (the 5 dashboard panels + the stubborn AAPL Decision-Linkage bug).

All citations are `file:line` against HEAD `fdca91b`. Nothing here touches the **live safety floor** (the hardcoded `app/calendars/macro.py` ±window block that drives RM Rule 0 + the PM proposal gates) — that stays deterministic and fail-safe. The fixes below act only on the NIS/LLM layer (Trader + premarket + the panel) and on display/persistence.

---

## 0. How it works today (verified)

Two independent macro systems:
- **NIS Tier-1 (LLM, all medium+ events)** — `macro_classify` ([llm_scorer.py:159](../../app/news/llm_scorer.py)) → `_build_macro_context` ([intelligence_service.py:64](../../app/news/intelligence_service.py)) → `MacroContext(overall_risk, block_new_entries, global_sizing_factor, events_today)`. Drives the **Trader** gate (`is_swing_blocked`/`is_intraday_blocked`), premarket sizing, and the **panel**.
- **Hardcoded gate** — `app/calendars/macro.py` FOMC/CPI/NFP ±window block. Drives **RM Rule 0** + the **PM proposal gates**. THE SAFETY FLOOR. (Now feed-extended per Wave 6e, hardcoded floor preserved.)

Post-event refresh: `_maybe_refresh_nis_post_event` ([portfolio_manager.py:860](../../app/agents/portfolio_manager.py)) re-runs the LLM ~60s after each event releases (+3min, retries till `actual` lands) and pushes the new context to `premarket_intel`.

Persistence (already wired): `persist_nis_macro_snapshot` ([decision_audit.py:130](../../app/database/decision_audit.py)) writes `nis_macro_snapshots` (day latest) **and** `nis_macro_history` (append-only timestamped lineage) on premarket + each post_event refresh.

---

## Part A — Macro risk workflow

### A1. Why risk stays HIGH all day (root cause)
The LLM prompt bakes in ([llm_scorer.py:222](../../app/news/llm_scorer.py)): *"HIGH risk but outcome known (actual released): sizing_factor=0.85, block=false."* Once it's a high-impact day, the model keeps HIGH/0.85 forever post-release. There is **no rule to step risk DOWN** to LOW/MEDIUM once results land benign/in-line, and the prompt never reasons about **surprise direction** (it gets raw prior/consensus/actual but no polarity).

### A2. Proposed: day-level risk STATE MACHINE
`SCHEDULED → PRE_RELEASE (uncertain → block/size-down) → AT_RELEASE (vol spike) → DIGESTING → DIGESTED (step DOWN to residual)`. The all-benign morning should end **DIGESTED / LOW / 1.0 by mid-morning**, not HIGH all day. Aggregate to the day context with weakest-link semantics (risk = max, block = any, sizing = min) so we never *raise* exposure vs the current contract.

### A3. Proposed: surprise-direction polarity (single source of truth)
New `app/news/macro_polarity.py` — `POLARITY` table: **lower-is-risk-on** = {CPI, Core CPI, PCE, Core PCE, PPI, Initial/Continuing Jobless Claims, Unemployment Rate}; **higher-is-risk-on** = {GDP, Retail Sales, Nonfarm Payrolls, Personal Income, ISM, Consumer Confidence, Durable Goods}; FOMC/rate-decisions = neutral/context. `surprise(event)` → `{polarity, raw_surprise, z, market_lean BULLISH/NEUTRAL/BEARISH, magnitude}`. Deterministic = the fail-safe + the source of truth shared by the LLM prompt AND the UI Beat/Miss label. **Clamp rule:** LLM-vs-deterministic → take the MORE conservative (deterministic wins on the safe side).

### A4. Proposed: prompt rewrite + triggers
- Feed the computed `market_lean` into the prompt; **delete the sticky "HIGH→0.85 forever" line**; add "all released + benign → LOW/1.0"; add JSON fields `net_market_lean`, `digested`.
- Triggers: keep +3min capture; add a **+~20min digestion tick** (one extra rebuild/event) so risk actually decays; time-only transitions computed locally (no LLM). Cache key gets a coarse state bucket → ≤3 LLM calls/event/day, no thrash.
- Fail-safe: LLM/feed down → deterministic path from the polarity table + state machine; never less conservative than today (`MACRO_UNAVAILABLE_BLOCK=True` stays).

### A5. PM / RM / Trader interaction (keep the existing contract)
Keep `block_new_entries: bool` + `global_sizing_factor: float ∈ [0.5,1.0]`; add optional `net_market_lean`, `macro_state`, `residual_risk` (default-safe).
- **RM Rule 0** + **hardcoded ±window** = deterministic floor — **DO NOT** route through the LLM/step-down.
- **PM** — unify swing/intraday sizing onto the **graded NIS** `global_sizing_factor` (keep the hardcoded block as the floor above it) so the step-down actually reaches sizing.
- **Trader** — graded block-lift at DIGESTED; sizing interpolates.
- **Exits** — a digested adverse read may set a NEW `tighten_exits` flag consumed ONLY by exit-review (tighten stops) — never force-liquidate, never part of the entry-block contract.

---

## Part B — Macro Intel dashboard page

Data sources: `/api/nis/macro` ([nis_routes.py:27](../../app/api/nis_routes.py)), `/api/nis/signals` ([nis_routes.py:96](../../app/api/nis_routes.py)), `/api/decision-audit/recent` ([nis_routes.py:230](../../app/api/nis_routes.py)). Frontend `MacroIntelPanel` ([App.tsx:1586](../../frontend/src/App.tsx)).

### B-Panel findings
1. **Header** — SWING and INTRADAY badges render the **same** global `block_new_entries` ([App.tsx:1720,1724](../../frontend/src/App.tsx)) → implies a per-book split that doesn't exist. "AS OF" is ET but not the requested `mm/dd/yyyy hh:mm` format.
2. **TODAY'S NIS ASSESSMENT / REFRESH HISTORY** — the real lineage (`nis_macro_history`) is **never queried**; the UI synthesizes a fake history from current `as_of` + decision-audit rows ([App.tsx:1607-1621](../../frontend/src/App.tsx)) with hardcoded `sizing=1`/`events=0`. Post-event reassessments that produced no trade never appear. Shows one blob, not a timestamped newest-on-top list.
3. **MACRO EVENT SCHEDULE** — Beat/Miss by `sign(actual−estimate)` ([App.tsx:1791-1796](../../frontend/src/App.tsx)) → wrong for inflation/claims/unemployment (lower is better). The per-event "meaning" isn't in the API (`events_json` carries only raw actual/est/prior + the LLM `direction`).
4. **BLOCK ENTRY / SIZE DOWN** — **driven by the per-symbol Tier-2 NEWS scorer (`_stock_cache`), NOT the macro factor** ([nis_routes.py:127-128](../../app/api/nis_routes.py)). The macro `global_sizing_factor` is applied at order-build time ([portfolio_manager.py:2652](../../app/agents/portfolio_manager.py)) but never written to `_stock_cache`, and the post-event refresh rebuilds the macro context only — **so SIZE DOWN does NOT update when macro results land.** (Answer to operator Q1: no.)
5. **DECISION LINKAGE — the AAPL bug — ROOT CAUSED:**
   - `_write_skip_audit` ([portfolio_manager.py:2558](../../app/agents/portfolio_manager.py)) writes **one `decision_audit` row per proposal symbol** with `block_reason="pm_skip: {reason}"`.
   - AAPL is the **first symbol of the swing universe** ([constants.py:9](../../app/utils/constants.py)); it's a REAL scored proposal; ~70% = mid of the `[0.55,0.95]` confidence map ([portfolio_manager.py:1541](../../app/agents/portfolio_manager.py)). The "17 symbols" = the swing top-N, all stamped with the same skip reason.
   - `pm_skip: kill_switch` rows are **historical fact** — written when the switch was active or on manual re-triggers / restarts that reset `_selected_today=False` ([orchestrator_routes.py:234](../../app/api/orchestrator_routes.py), [portfolio_manager.py:170,407](../../app/agents/portfolio_manager.py)).
   - **They persist because `/recent` has NO date filter** ([nis_routes.py:240](../../app/api/nis_routes.py)) → returns last 100 rows across all sessions, so a stale kill-switch batch shows forever even with the switch now OFF.
   - **Why prior fixes missed it:** they targeted the kill-switch STATE (Wave 5j un-arm) — but the symptom is **stale skip-audit row DISPLAY**, orthogonal to switch state. Nothing changed `_write_skip_audit`, the `/recent` query, or the frontend's skip handling.
   - Secondary frontend bugs: Outcome column compares `'BLOCKED'/'APPROVED'` but DB stores lowercase `enter/block/skip/size_down` ([App.tsx:1903](../../frontend/src/App.tsx)) → skip rows render blank/raw; header "blocked/sized" counters always 0 (wrong case + `size_multiplier<1` default 1.0); Macro Risk / NIS Policy blank because skip rows carry no macro/news context.

---

## 5. Consolidated fix plan (phased)

**Phase 0 — Minimal-correct (kills the loudest bugs; ~half day; low-risk display/query):**
- **F1** date-scope `/api/decision-audit/recent` to today ET (`?days=N` override) — stops the stale AAPL rows. *Highest leverage.*
- **F2** frontend lowercase mapping for `enter/block/skip/size_down` + label `*` rows as "All (strategy abstained)" + fix the header counters.
- **F3** `mm/dd/yyyy hh:mm` ET timestamps across the page (header AS OF + history rows).
- **F4** collapse strategy-level skip spam (kill_switch/no_proposals_cached/model_not_trained/opportunity_score_low/macro_event_window) to ONE `symbol='*'` row (keep per-symbol only for per-symbol gates). *Calibration-safe — calibration only aggregates `final_decision='block'`, not `skip`.*

**Phase 1 — Assessment history (requested feature):**
- **F5** new `GET /api/nis/macro-history?date=today` returning `nis_macro_history` newest-first (table already populated).
- **F6** render the REAL lineage newest-on-top (timestamp ET `mm/dd/yyyy hh:mm`, trigger source/event, risk, sizing, block, rationale), replacing the synthesized history.

**Phase 2 — Correctness of meaning:**
- **F7** server-side polarity table (§A3) → emit per-event `market_outcome` + `surprise` into `events_json` + `/macro`; UI consumes the server verdict (fixes Beat/Miss). Shared with the prompt rewrite.
- **F8** prompt rewrite + state-machine step-down (§A2/A4) — the risk-stays-HIGH fix.
- **F9** honest SWING vs INTRADAY badge (drop the false split or wire a real per-book flag).

**Phase 3 — Reactivity / depth (optional):**
- **F10** SIZE DOWN macro reactivity — decision E3 (relabel "news-driven" / add a macro-scalar banner / fold the factor in).
- **F11** idempotent skip-audit on repeated manual triggers (behavioral; must not suppress the actual abstention).
- **F12** unify PM sizing onto graded NIS + `tighten_exits` (behavioral; feature-flagged; paper-first).

---

## 6. Open decisions for the operator

1. **Skip-audit granularity (F4):** OK to collapse blanket abstentions to one `*` row? (Calibration-safe.)
2. **`/recent` window (F1):** default today-ET with `?days=N`, or rolling last-N + a session/date badge?
3. **SIZE DOWN reactivity (F10):** (a) relabel "News-driven sizing" for honesty, (b) add a "macro scalar applies to all" banner from `/macro`, or (c) fold the macro factor into per-symbol display (larger)?
4. **Polarity set (A3/F7):** confirm the lower-is-better list (CPI/Core CPI/PCE/Core PCE/PPI/Initial+Continuing Claims/Unemployment) — must match the prompt rewrite.
5. **SWING vs INTRADAY (F9):** independent per-book macro block flags intended, or stop implying a split until the backend models one? (Per Wave 5g, NIS sizing is intraday-only and the block gates both.)

---

## 7. Key file:line index
`app/news/llm_scorer.py:159,222` · `app/news/intelligence_service.py:64` · `app/agents/portfolio_manager.py:860,1541,2558,2602,2652,2769` · `app/database/decision_audit.py:130,145,187` · `app/api/nis_routes.py:27,96,127,230` · `app/api/orchestrator_routes.py:234` · `app/utils/constants.py:9` · `app/calendars/macro.py` (SAFETY FLOOR — untouched) · `frontend/src/App.tsx:57,1586,1607,1720,1791,1887,1903` · `frontend/src/types.ts:302`
