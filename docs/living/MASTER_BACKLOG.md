# MrTrader — Master Backlog & Roadmap

**Last updated:** 2026-06-22

---

## 🎯 ACTIVE PLAN — Alpha-v10 "harden, don't hunt" (2026-06-22, 10-LLM panel)

**SSOT:** [`docs/reference/prompts/20260622_LLM_Alpha_V10/COMPREHENSIVE_ROADMAP_2026-06-22.md`](../reference/prompts/20260622_LLM_Alpha_V10/COMPREHENSIVE_ROADMAP_2026-06-22.md). 4 internal Opus panelists + red-team + 5 external reviews (ChatGPT/Claude/DeepSeek/Gemini/Grok), **unanimous: stop hunting a 5th sleeve for 1–3 months — the binding constraint is capital + a real *tiny-live* track record + not blowing up.** Three phases: **H (Harden) → D (Deploy) → R (Research).**

### Macro Intel — workflow + dashboard fix ✅ COMPLETE 2026-06-25
**Scope doc:** [`docs/reference/MACRO_INTEL_SCOPE_2026-06-25.md`](../reference/MACRO_INTEL_SCOPE_2026-06-25.md) (2 Opus passes). **Shipped #587–#597:** Phase 0/1/2a (decision-linkage date-scope + skip-row collapse + ET timestamps + polarity-aware Beat/Miss) → 2b (macro risk STEPS DOWN after benign releases; prompt rewrite + deterministic floor/clamp, gate never less conservative) → F10 (SIZE-DOWN folds the macro factor per-symbol) → F11 (idempotent skip-audit) → F12a `UNIFIED_MACRO_SIZING` (entry sizing) → F12b `MACRO_TIGHTEN_EXITS` (exit stop-tightening). **Both F12 flags ENABLED 2026-06-25 (paper).** Plus #597: phantom `INTRADAY_FORCE_CLOSED AAPL` fix (DB-authoritative purge) + 304-row AAPL DB cleanup. Each phase: Opus deep-dive + tests + docs. Safety floor (`app/calendars/macro.py` + RM Rule 0) untouched. F9 (honest SWING/INTRADAY badge) parked (both badges correctly read `block_new_entries`).

### Phase H — make the safety layer LOAD-BEARING (the hard IBKR no-go gate; before any IBKR dollar)
| # | Item | Status |
|---|---|---|
| **H10** | cash-ETF mapping SSOT (register all 8 `cash_sleeve.CASH_ETFS` in `instrument_master`/`book_state`) + fix the trend-allocation doc drift (0.25→0.50) | ✅ DONE 2026-06-22 (zero live-change; unblocks H2) |
| **H3** | pre-trade order sanity: **dollar-notional cap + max-order-size** (Alpaca now), **futures-multiplier verify-on-connect** (at IBKR), fail-closed | ✅ DONE 2026-06-25 (Alpaca side) — `_assert_order_within_caps` at the place_market/limit chokepoint, fail-closed before submit; `H3_MAX_ORDER_NOTIONAL_USD=$500k` / `H3_MAX_ORDER_SHARES=100k` (retrain_config), `est_price` plumbed to all market callers. Opus deep-dive SHIP-READY (can't fail-open; rejected-exit stays monitored). 10 tests. Effective next restart. IBKR futures-multiplier half = covered by P2.2 verify-on-connect. |
| **H1** | **reconciliation-before-trade** (fail-closed, broker=truth) wired into EVERY live order path (wire the built-but-inert `reconciliation.py`) | 🔭 SHADOW DONE 2026-06-22 (wired into trend+cash; `pm.reconciliation_mode`). **Enforce-prep DONE 2026-06-25:** in-flight orders now modelled (held=ACTIVE vs pending=PENDING_FILL; break only when broker is outside the `[held, held+pending]` band) → no just-placed-order false break; cross-sleeve sums by (venue,iid). Shadow soak clean (0 breaks since 06-22). **→ flip→enforce after a clean 7-day soak (~06-29, owner-present).** Remaining (non-blocking): cash-side check (cash not modelled DB-side → "not checked" note, never a false break). |
| **H2** | wire the **kill-switch state machine**; flip `whole_book_gate_mode` shadow→enforce after H10 + a clean shadow week | 🔭 **code-half SHADOW DONE 2026-06-25** — `kill_switch_sm` singleton + `evaluate_new_risk` consult wired at trend/cash/RM gates; auto-triggers fed (recon-FAIL_CLOSED when recon enforce; heartbeat); flag `pm.kill_switch_sm_mode` default shadow; Opus SHIP-ABLE (zero live change in shadow). **Remaining (owner-present): flip `pm.kill_switch_sm_mode`→enforce + `pm.whole_book_gate_mode`→enforce after a clean shadow week (~06-29);** reduce-only refinement (let HALT_NEW_RISK pass protective exits at the sleeves) is the documented enforce-time follow-up. |
| **H4** | **out-of-band broker-only flatten** (no DB/Redis/app dep), tested weekly on Alpaca paper | ✅ DONE 2026-06-22 (`scripts/emergency_flatten.py`, dry-run default; live dry-run validated) |
| **H5** | **external dead-man watchdog** (separate process; heartbeat → call/SMS, optional auto-flatten) | ✅ DONE 2026-06-22 (durable heartbeat + `scripts/dead_man_watchdog.py`, alert-only default) |
| **H6** | **per-order idempotency** (client order IDs) + transactional order log | ✅ DONE 2026-06-22 (idempotent `place_market_order` on dup + centralized `order_ids.idempotency_key`; full per-order lifecycle log deferred to P2.3) |
| **H7** | **broker-side dollar limits** (IBKR precautionary settings) your code can't override | ⬜ (at IBKR) |
| **H8** | tiered alerting (catastrophic→call/SMS, warning→push, info→digest) + reconciliation-break + "gate-didn't-run" alerts | ✅ DONE 2026-06-22 (severity tiers + subject prefix + gate_error alert + env-gated critical webhook `MRTRADER_CRITICAL_WEBHOOK`; SMS/push = plug in a webhook) |
| **H9** | wire the **drawdown de-gross ladder** into the live budget (2–3 steps, smoothed/broker-confirmed trigger, hysteresis) | ✅ DONE 2026-06-22 (`_drawdown_ladder_multiplier` on trend `alloc`, reads `risk.peak_equity` HWM, RISK_POLICY_V1 rungs −8/−12/−16/−20%→×0.75/0.50/0.25/0.00; SHADOW-default flag `pm.drawdown_ladder_enabled`; un-floored so −20% flattens; fail-safe to 1.0) |

### Phase D — DEPLOY (start the real clock; on IBKR approval)
- D1 step trend ~4.7%→~6% vol **with a hard leverage cap (~1.3–1.5×)** — only after H done.
- D2 verify cash sleeve draining idle cash to T-bills (free RFR).
- D3 pre-register IBKR tiny-live launch (instruments, max contracts, margin reserve, rollback, 30-day probation) → IBKR paper → **microscopic** live (carry+xsmom, micro contracts).
- D4 sleeve combine = inverse-vol + per-sleeve cap + **isolated margin pools**; never MVO; covariance only as live joint history accrues.
- D5 bucket **robust premia (trend, carry)** vs **candidate anomalies (xsmom, VRP, new)** — smaller budget / higher discount for the latter.
- **IBKR plumbing (the futures venue):** **P2.2 ✅** read-only adapter + verify-on-connect (DONE 2026-06-22, validated live). **P2.3 ✅** order-construction core + SHADOW executor (DONE 2026-06-22: `futures_sizing.target_lots`/`futures_order_deltas`, `order_ids.futures_run_id`, `futures_sleeve.run_futures_rebalance` — places NOTHING; inert by `ibkr.enabled`/`ibkr.futures_enabled`/`ibkr.trading_mode`=off + no write API). **R1 (owner-gated):** real placement, fills capture, margin preview, roll emission, the live carry/xsmom signal path, per-run snapshot table, scheduler wiring — needs TWS Read-Only API OFF + owner-present + the live-paper soak.
  - **🚨 R1.3 built out (2026-07-07): full IBKR write surface + roll policy/monitor/holiday-calendar/hybrid-roll + live signal wired (all inert/shadow) — but the BEFORE-LIVE GATE FAILED.** The carry+xSMOM book does NOT survive on the 16 IBKR markets (Track-B t 2.61 → −0.20; DECISIONS 2026-07-07). **Futures Breadth Program** (SSOT [`FUTURES_BREADTH_PROGRAM_2026-07-07.md`](../reference/FUTURES_BREADTH_PROGRAM_2026-07-07.md)): FB0 breadth sweep → **~48 markets (3×) needed, commodity-first, marginal + cost-inflated even at full breadth → RECOMMEND SHELVE futures-live.** ETF-trend + cash stays the sole live book; R1.1/R1.2 (ETF→IBKR) UNAFFECTED. Owner call: FB-SHELVE (recommended) vs FB1 (expand universe).

### Phase R — RESEARCH (ONLY after Phase H; time-boxed, pre-registered, ONE at a time)
1. **Recover trend's convexity** — long-short futures incl. short-equity/long-bond-gold-USD legs + a declared defensive-macro **crisis overlay** (highest EV; barely "new").
2. **Commodity calendar-spread / seasonal-storage premia** (most genuinely-distinct family on owned Norgate data).
3. **VIX-gated stress reversal** (Nagel liquidity-provision; contested — pre-register, expect possible kill).
4. **G10 FX value** as a hedge to FX carry (low-conviction; one clean test, expect park).

### 🚫 DON'T list (standing guardrail — consensus)
vol-to-8% (esp. pre-harden = most dangerous) · swing-equity-ML revival / Norgate buy *for* vol-managed momentum · touching the live TSMOM signal (skip-month) · Constructor/covariance/HA/streaming-risk at $100k · re-adding crypto-trend or options-dispersion as "diversifiers" (short-convexity in a crisis) · capitalizing paper-futures Sharpe before real fills · VIX-VRP back into the live book.

### Meta-critiques to internalize
Operator-capacity/behavioral risk is the dominant *unmodeled* risk (assume backtest 0.72 → live ~0.4–0.5) · true multiple-testing N is in the hundreds not 26 (mark carry/xsmom confidence down) · possible *era*-overfit (2006–2025 liquidity regime) · "risk premia not alpha" is partly a euphemism.

---

## 🎯 PRIOR PLAN — Alpha-v9: multi-engine premia book (2026-06-16)

**SSOT:** [`docs/reference/ALPHA_V9_ROADMAP.md`](../reference/ALPHA_V9_ROADMAP.md) (roadmap) + [`docs/reference/ALPHA_V9_ARCHITECTURE.md`](../reference/ALPHA_V9_ARCHITECTURE.md) (design/impl). Opus synthesis of 5 external world-class-quant reviews (`docs/reference/prompts/20260616_Alpha_v8/responses/`). Core reframe: stop hunting a 4th weak uncorrelated *equity* sleeve (the corr<0.30 wall, hit 4×); run a **second return engine in a different risk class (defined-risk VRP) paired with trend by skew**, point trend at **crypto**, and buy **Norgate** to un-bias event work + unlock carry — **after** proving the validator can detect a known edge.

| Phase | What | Status |
|---|---|---|
| **P0** | **Validate the validator** — **P0-1 ✅ DONE 2026-06-16 → PASS** (`scripts/walkforward/positive_control.py`: known anomalies through the REAL feature→label pipeline; FIDELITY-centered verdict — pipeline IC matches an independent reference IC, 3/3 faithful, 0 deflationary, label-fidelity +0.761 → the feature path is sound, IC≈0 is market/model/cost not a bug; finding: features window-limited to ~63 bars). **P0-2 ✅ DONE 2026-06-16 → SUCCESS** (powered stability test replaces the binary both-halves guard [sim 22%→3.8% FN]; Track-B standalone-SR floor report-only for diversifiers; probation P(ΔSR>0)≥0.75 → carry F3 now `PROMOTE_PAPER_PROBATION` — the old guard was the problem; live-paper-ratified, report-only). **P0-3 ✅ DONE 2026-06-16 → SUCCESS** (`REQUIRE_TRUE_WF_FOR_PROMOTION` flipped False→True: a trained model can no longer promote on a frozen generalization test — must per-fold-retrain; `is_true_walkforward = per_fold OR rules_based` so rules-based sleeves are unaffected; Opus C1 fix: a frozen run blocked by this alone → INCONCLUSIVE not RETIRE, so cron can't roll back the champion; +5 tests, 3544 pass; report-only — retrain currently disabled). **Phase 0 COMPLETE → next is Phase 1.** | ✅ Phase 0 done (P0-1/P0-2/P0-3); Phase 1 next |
| **P1** | **Harden & monetize** — **P1-2 ✅ DONE** (trend allocation 25%→50%; Kelly not binding, vol-target pick; live DB + schema). **P1-3 🔭 SHADOWING** (credit overlay; `scripts/shadow_credit_governor.py`; verdict ~mid-July). **P1-4 ✅ DONE** (`app/live_trading/back_validation.py`: intended-vs-actual live-vs-sim tracking-error + PASS/WATCH/FAIL/BUILDING verdict; 2 Opus deep-dives pivoted off a contaminated independent-sim; BUILDING until ~early July). **P1-1 ✅ DONE** (`app/live_trading/cash_sleeve.py` + `cash_tracker.py`: park idle settled cash in T-bills [SGOV/BIL] for the risk-free rate; buffer + risk-off sell policy; T-bills excluded from the 80% risk gross cap + position/budget/sector counts; default OFF; Opus deep-dive DON'T-SHIP→fixed 6-site gross-exclusion gap + Trader buffer-liquidation path; 14 tests, 3578 pass). **Buildable work COMPLETE; P1-3 shadow verdict ~mid-July.** | ✅ P1-1/P1-2/P1-4 done; P1-3 shadowing |
| **P2** | **Cost model + adapters** — **P2-4 ✅ DONE 2026-06-16** (`app/options/spread_model.py`: calibrated moneyness/DTE/underlying-aware option spread surface from the NBBO log → premium-% cost; per-underlying buckets + conservative p75 fallback; 2 Opus deep-dives fixed anachronism/global-too-cheap/IV-claim; framework shipped, calibration PRELIMINARY + NOT wired to a live VRP verdict; 16 tests, 3603 pass). Remaining: P2-5 adapter discipline; crypto + FINRA-short-volume data adapters | 🟡 P2-4 done; adapters next |
| **P3** | **New return engines on Alpaca** — **P3-1 ✅ DONE 2026-06-16** (`crypto_trend` sleeve: TSMOM on 10 Alpaca spot pairs, ann=365, hard vol-target; standalone Sharpe 0.64 / corr-to-trend 0.18 → falsifiable criterion MET = a real low-corr stream; but Track-B FAIL [dSR −0.17] + CAPITAL power-floor → **PAPER-candidate, not capital**; 2 Opus deep-dives fixed gate annualization C1 + partial-bar PIT; 7 tests, 3611 pass). defined-risk VRP (P3-2, needs mature spread surface); overnight/intraday (P3-3); risk-premia composite (P3-4); FINRA short-volume (P3-5) | 🟡 P3-1 done (PAPER-candidate); P3-2+ next |
| **P4** | **Strategic data bet** — 👤 buy Norgate → re-run PEAD + F2 on survivorship-free data (P4-1); futures trend+carry research (P4-2); 👤🔴 IBKR adapter ONLY if P4-2 passes (P4-3) | ⬜ |
| **P5** | **Modeling reframe** (gated by P0-1) — predict vol/regime not return; meta-labeling for sizing; HRP allocator | ⬜ |

**The single most important decision (👤):** the **trend ⊕ VRP** pairing — positive-skew (trend, long-crisis) ⊕ negative-skew (VRP, short-crisis) under the governor — the most likely route from ~0.7 to ~1.0+ book Sharpe. **DO-NOT list** (roadmap §6): no more daily-equity XGBoost sweeps, no intraday-ML revival, no PEAD threshold filters, no per-regime models, no RL, no path-inflated CPCV, no naked short vol, no IBKR before futures research passes.

**📦 PARKED — Norgate-data justification accumulator (👤 owner data buy; revisit when the stack of reasons justifies the $693):**
A single **Norgate US Stocks (Platinum, ~$693/yr)** purchase — delisting-inclusive / survivorship-bias-free — unlocks *all* of the following. Park until the accumulated value clears the cost; keep adding reasons here.
- **Option B — single-stock momentum** on survivorship-free history (the bigger prize; P4 equity audit).
- **Option A-v2 — thematic/country dual-momentum rotation** — the one rotation variant with a credible path to being non-redundant to trend (high-dispersion universe). Free phase **A2-0** (build the delisted-ETF roster + size the prize on biased survivor data) can be done anytime; the gated work needs Norgate. SCOPE: [`docs/reference/OPTION_A_V2_THEMATIC_ROTATION_SCOPE_2026-06-22.md`](../reference/OPTION_A_V2_THEMATIC_ROTATION_SCOPE_2026-06-22.md).
- **Re-run PEAD + F2 (ETF relative-value)** on survivorship-free data (the original P4-1 reason).

**Discovered tech-debt (data-quality audit 2026-06-21):**
- 🔧 **`app/ml/factor_scorer.py` PIT look-ahead** — all 5 scorers resolve the as-of date column from `("date","report_date","filed_date")`, none of which exist in the FMP schema (`as_of_date`) → the PIT filter is silently skipped (returns the full unfiltered frame = look-ahead), and they bypass `load_fmp_fundamentals`'s new quality guards. The fundamentals-factor sleeve is **parked/unused** so no live result changes, but any past number from this path is look-ahead-contaminated. **Fix before any fundamentals-factor revival:** add `as_of_date` to the candidate columns + route through `load_fmp_fundamentals()`, then re-run. (See `docs/reference/DATA_QUALITY_AUDIT_2026-06-21.md`.)

---

## 🎯 PRIOR PLAN — Alpha-v8 Research Program: Overlay & Timing track (2026-06-14) — COMPLETE

**SSOT:** [`docs/reference/ALPHA_V8_RESEARCH_PLAN.md`](../reference/ALPHA_V8_RESEARCH_PLAN.md) (Opus-architected). Three free-data angles, phased + exhaustively tested, executed sequentially. Driven by the F-series lesson: additive equity sleeves hit the IC≈0 / Track-B-correlation wall; **overlays** are what work (the VIX governor); power + a pre-registered both-halves stability guard are non-negotiable.

| Phase | What | Status |
|---|---|---|
| **G0** | Shared infra: overlay **marginal-stacking** API (`compose_overlays`, `evaluate_overlay_marginal`) + overlay registry | 🟢 **DONE 2026-06-14** (Opus-reviewed; marginal math correct + PIT-safe; 11 tests) |
| **G1** | **Credit/curve de-risk overlay** (HYG/IEF + ^TNX/^IRX; daily, deep, zero-friction) — highest EV; judged marginal to the VIX governor | 🟢 **DONE 2026-06-14:** curve KILLED (inert); credit-orig L60 KILLED (over-eager); **credit-SELECTIVE (L120/2%-band) = ⭐ promote_paper CANDIDATE** (marginal dSharpe +0.064, all-3-crises, both-halves stable; Opus-defensible w/ caveats; multiplicity-disclosed). → owner-gated G4 (flag OFF). |
| **G2** | **Short-interest de-risk overlay** (data layer exists; bi-monthly ~190 obs → depth go/no-go gate first) | 🔴 **KILLED 2026-06-15** — depth gate passed (202 obs) but uniformly Sharpe-negative marginal to the governor across a 12-cell grid; COVID missed (bi-monthly+lag); RRT effect absent/reversed post-2017. Opus-verified honest (sign/PIT/aggregation OK). Line closed. (Per-name SI data layer stays reusable for a future XS factor.) |
| **G3** | **Additive long-flat timing sleeve** (full Track-A+B; corr<0.30 wall) — hardest, last | 🟠 **PARK 2026-06-15** — credit-timing SPY sleeve: Track-A PAPER **PASS** (point_SR +0.913, HAC p 0.0001 — real standalone edge) but Track-B **FAIL** on the corr wall (corr 0.518; largely SPY beta, residual-α 1.50). Opus-verified honest (no look-ahead). Not promoted; credit's productive form is the G1 overlay, not a sleeve. |
| **G4** | Owner-gated **live wiring** of any winner (compose multiplier stack, clamped, fail-safe) | 🟢 **DONE 2026-06-15** — credit-selective overlay wired into `trend_sleeve` (composes ×VIX governor, clamped to 0.25 floor), flag `pm.credit_governor_enabled` **DEFAULT OFF** (zero live change; owner flips to activate). Opus-verified SAFE + fail-safe. ⚠️ **OWNER DECISION: review G1 caveats, then set the flag true.** |

Cross-cutting: pre-register every confirmatory run (R7) + mandatory both-halves stability guard + exhaustive tests + independent Opus adversarial deep-dive + fail-safe + one PR/sub-phase. **First actions: G0 → G1a.**

---

## 🎯 PRIOR PLAN — Alpha-v7: Sleeve Lab + orthogonal deep-history premia (2026-06-14) — COMPLETE (F0–F5 swept; governor live, carry killed)

**SSOT:** `docs/reference/ALPHA_V7_RESEARCH_SYNTHESIS_2026-06-14.md` (synthesis of the 5-LLM panel — Opus 4.8/ChatGPT/DeepSeek/Gemini/Grok; inputs archived `docs/archive/llm-reviews/2026-06-14/`). Supersedes the prior Alpha-v7 blueprint for *direction*. **Context:** Ruler v2 is LIVE (both gates); the honest candidate sweep found no new model (trend is the only edge). Panel consensus: build a 3–5 sleeve premia book around trend at a realistic **book SR 0.7–0.9** on **deep free history** (19y ETFs / FRED); frozen 4y options for *conditioning* only.

| Phase | Item | Why | Effort | Status |
|---|---|---|---|---|
| **F0** | **Sleeve Lab** — unify the sleeve research→Ruler-v2(Track-A+B)→sleeve_allocator→report pipeline into ONE tested module + a sleeve registry; retire bespoke `run_*_cpcv` scripts. | The future-proof, hardened substrate that makes every future premia a small uniform declaration. | ~1–2wk | 🟢 **CORE LANDED 2026-06-14** (`scripts/walkforward/sleeve_lab.py` — `Sleeve`/`evaluate_sleeve`/registry/`assemble_book`/`SleeveReport`; 20 tests; Opus deep-dive applied. Follow-ups: overlay eval path (lands with F1 governor) + multi-factor residual-α wiring.) |
| **F1** | **Structural premia + crash governor** — turn-of-month / FOMC / overnight-gap sleeves + a VIX-term de-risking overlay (governor). | Most orthogonal-to-trend, most powered (high event count), cheapest, owned data, no spread wall. | ~1–2wk | 🟡 **IN PROGRESS** — **F1a (2026-06-14): turn_of_month + overnight BOTH FAIL** (miss HAC significance + don't diversify the trend book; overnight is timed SPY beta; Opus-verified honest; no promotion). **F1b (2026-06-14): VIX-term crash governor MODESTLY HELPS — FIRST POSITIVE / ⭐ candidate overlay** (maxDD −13.9→−12.1%, Calmar 0.469→0.501, Sharpe flat, COVID DD −10.7→−6.5%; ~0.5%/yr give-up; Opus-verified honest+PIT, robust; `app/strategy/crash_governor.py` + overlay eval path in `sleeve_lab.py`). **→ OWNER APPROVED → WIRED LIVE (2026-06-14):** governor integrated into `trend_sleeve.run_trend_rebalance` (`alloc *= _crash_governor_multiplier`); flag `pm.crash_governor_enabled` ON by default, fail-safe, PIT-correct, reversible. Opus pre-merge review (1 HIGH PIT look-ahead fixed → SHIP). 50 tests. ⚠️ needs orchestrator restart to take effect. **F1c (2026-06-14): FOMC drift DEFERRED** — same additive-SPY-beta shape that failed Track-B + no clean historical FOMC date list (bug risk); low EV, skipped (recorded). **F1 net: governor is the one candidate → now LIVE; additive premia all fail.** |
| **F2** | **Slow ETF relative-value** — pre-registered log-spread mean-reversion across ~6–8 economically-linked ETF pairs; slow, vol-targeted. | Orthogonal (mean-reversion vs trend), deep-history; NOT the killed high-turnover reversal. | ~1wk | 🔴 **FAIL 2026-06-14** — `app/strategy/etf_relative_value.py`, 5 pre-registered pairs, deep history: genuinely orthogonal (corr **−0.23** — the diversification F1a lacked) BUT ~zero standalone edge (point_SR 0.026, HAC p 0.46); every grid cell below the 0.30 floor net of cost. Track-B FAIL (a zero-return diversifier adds nothing). Opus-verified honest (sign/PIT/cost OK). No promotion. **Learning: standalone return, not orthogonality, is the binding constraint.** |
| **F3** | **Carry done right (small)** — rates/curve roll-down (FRED) + FX rate-diff; skip commodity (no clean futures). Judge on CRISIS correlation. | Real but crisis-correlated + likely overlaps TSMOM → small. | ~1–2wk | 🟠 **NEAR-MISS 2026-06-14 — strongest additive candidate.** `app/strategy/carry.py` (duration-carry: IEF sized by 10y−3m term spread, yfinance ^TNX/^IRX). Pre-registered config (IEF/1.5/long-short): point_SR +0.314, **resid-α +2.10 (real orthogonal alpha, corr +0.01)**, Track-B 7/8 (IR +0.39, ΔSR +0.084) — but misses BOTH bars marginally (HAC p 0.0998>0.05; P(ΔSR>0) 0.886<0.90) → **FAIL, NOT promoted**. Robustness grid: edge real+robust (+resid-α in all 12 configs; Track-B passes 8/12). Opus-verified honest (no look-ahead, no unit bug). **Owner-recommended: a FRESH pre-registered carry confirmation (R7), NOT a cherry-pick.** (FX/commodity carry deferred: no clean free data.) **→ CONFIRMATION RUN (owner-authorized) `F3-CARRY-CONFIRM-20260614`: 🔴 KILLED.** Principled spec long-flat IEF/1.5: full-sample Track-B PASS (IR +0.434, P(ΔSR>0) 0.930) but Track-A still fails (HAC p 0.083) and the pre-registered **sub-period stability guard FAILED — H1 2007-2016 SR +0.689 vs H2 2017-2026 SR −0.098** (edge is a pre-2016 artifact, dead in the modern regime). Line CLOSED; not promoted. |
| **F4** | **Options-conditioned event interaction** — continuous, pre-registered interaction regression on the owned event panel × frozen options features (event_inference two-way CGM; NOT a threshold filter). | Long shot; uses owned data; weak priors. | ~1–2wk | ⏸ **DEFERRED 2026-06-14** — already answered+killed in P4 (H4a–H4e ALL KILL + PEAD demoted t=−0.77) on the SAME frozen 4y options store (underpowered by construction). Panel-ranked long-shot. Negative-EV re-tread; revisit only with longer options history. (Recorded, not silently dropped.) |
| **F5** | **Book assembly + live fidelity** — sleeve_allocator book CPCV when ≥2 sleeves pass; replay-diff before scaling. | Turn passing sleeves into a book. | ongoing | ⏸ **NOT TRIGGERED 2026-06-14** — gated on ≥2 sleeves passing; ZERO passed (carry near-missed; others failed; governor is an overlay). Illustrative owner-pending book computed (trend+carry+governor: maxDD −13.9→−10.9%, SR ~flat; corr trend~carry +0.029). Real book build triggers only on a passing sleeve. |

> **F-series WRAP (2026-06-14):** F0 Lab ✅ (permanent infra) · F1a ❌ · F1b VIX governor 🟡 helps (overlay, owner-gated) · F1c ⏸ · F2 ❌ · F3 carry 🟠 robust near-miss (owner-rec re-registration) · F4 ⏸ · F5 ⏸. **No additive sleeve cleared the pre-registered bar on free daily US data — confirms "trend is the only standalone edge".** Two real-signal items remain, both OWNER decisions: (1) adopt the VIX governor overlay? (2) authorize a fresh pre-registered carry confirmation? If both stall → the next lever is a deliberate DATA buy (Norgate futures / longer options), not more free-data searching.
| **Deferred** | cross-asset trend via **futures** (Norgate + futures acct) · aggregate **short-interest timing** (FINRA backfill) · index-**VRP ETP** (dangerous, last). | Data/infra-gated or dangerous — only when a precondition is met. | — | ⬜ |

**Each sleeve:** ONE pre-registered design (no sweeps), CPCV on the deepest window, Track-B vs the *current* book, kill-fast if no value outside one crisis. Build → independent Opus deep-dive → tests → no-drift docs → merge.

---

**Prior — Last updated:** 2026-06-06
**Capital:** $100k (paper)
**Status:** 🎯 **ALPHA-v7 — OPERATE A PREMIA BOOK** (active, 2026-06-12). SSOT: **`docs/reference/ALPHA_V7_SYNTHESIS_AND_PLAN.md`**. Alpha-v6 is COMPLETE — every pre-registered hypothesis (H1/H2/H3/H4a–e/P5) KILLED/DEMOTED/PARKED. A 4-LLM panel (2026-06-12) converged: the gate is a **Type-II machine on ≤4y data** (so the all-kill was partly geometry), standalone US-equity/free-data alpha is **genuinely exhausted**, and the move is to **re-charter from "find alpha" to operate a 3–5 sleeve premia book judged at the BOOK level (Track B), target book SR ~0.8–1.0.** Phases: A recovery re-tests (P5 L/S overlay + noise-pile + TSMOM audit, no new data) · B Ruler-v2 (retire path-t → HAC-SR+bootstrap+PBO; invert tiers; residual-α primary) · C trend expansion (futures/FX/crypto) · D live fidelity (replay-diff) · E index-VRP via ETP. Sharpens Alpha-v6's two-track + registry substrate. (Prior Alpha-v6 section below remains the executed record.)

---

## 🎯 ALPHA-v6 PLAN — Fix the Ruler + Options-as-Signal (2026-06-10)

**SSOT:** `docs/reference/NEXT_PHASE_BLUEPRINT_2026-06.md` (full design + implementation spec — this table is the index). Inputs: 5 independent world-class-quant LLM reviews (Gemini/DeepSeek/Grok/ChatGPT/Claude), synthesized by a Fable 5 deep-dive and grounded against the code. Reviews archived: `docs/reference/prompts/20260610_Quant_Options_Review/responses/`.

**Thesis (5/5 consensus):** the harness, hardened against inflation, is now a **Type-II / false-negative machine** — a t≥2.0 gate on N_eff≈6–8 folds of ≤4y data rejects *true* Sharpe-0.5–0.7 edges (t ≈ SR·√years); 100% KILL incl. the confirmed-real index VRP = a miscalibrated ruler, not an empty opportunity set. Fix the measurement, then harvest the options data as **signal at equity cost** (the spread wall that killed OPT-3 can't reach an equity book).

| Phase | Item | Track | Effort | Status |
|---|---|---|---|---|
| **P0** | **Calibrate the ruler** (blocks trustworthy verdicts): positive/negative gate controls (**TSMOM-on-4y decisive**), **two-track acceptance** (A: standalone alpha / B: book-delta diversifier), research registry + pre-registration, DSR→report-only, `event_regime_sharpes()`. | — | M ~1.5wk | ✅ **COMPLETE** — calibration harness (#444) + result (#447: refutes 'lower t-bar' → two-track) + Track B book-delta gate (#448) + research registry (#449) + Track B amend→TSMOM passes (#450/#451) + **`--hypothesis-id` enforcement on all 9 run scripts + `event_regime_sharpes()` (report-only; waiver retirement deferred to H1/P3) + H1/H2/H3 pre-registered (#454)**. NEXT = P1c/P2 fuses + P3 event panel / H1. |
| **P1** | **Live-book fidelity** (make it boring before new capital): StrategySpec replay-diff, fill-quality table, nightly NBBO→calibrated spread table, multi-factor residual-α, paper-confirmation reworded as fidelity-only. | infra | M ~2wk | 🔨 **P1c FUSE LIT (#455)** — nightly NBBO snapshot logger (`log_options_nbbo.py`) via **Alpaca** (Polygon serves no NBBO); 15:55 ET job (activates on uvicorn restart). Accumulating toward the calibrated spread table (~4-6 wks). Replay-diff / fill-quality / residual-α still TODO. |
| **P2** | **Options feature layer**: persist computed IV/greeks (one pass over 112.8M bars), surface-quality reader, BMO/AMC event-time-aware snapshots, daily per-underlying feature table, R1K-union survivorship audit. | infra | M ~2wk | ✅ **greeks store COMPLETE (#455, 733/733)** + ✅ **daily feature table BUILT (P4a, 2026-06-12)** — `app/data/options_features.py`/`scripts/build_options_features.py` → `data/options_features.parquet` (one row/underlying-day, 5 XS features + quality cols; holiday-aware knowable_date + split-adjusted RV, 2 PIT/split BLOCKERs caught by Opus deep-dive) + `app/data/options_quality.py` coverage filter. BMO/AMC event-time snapshots / standalone surface-quality reader still optional TODO. |
| **P3** | **Earnings-event panel + PEAD v2** (centerpiece): one-row-per-event table; **event-LEVEL inference** (two-way clustered announce-day×firm); pre-registered H1 (re-adjudicate live PEAD), H2 (continuous reaction-ratio — settles OPT-5), H3 (continuous options-conditioned scorecard, NOT XGBoost). | A | L ~3–4wk | ✅ **DONE — H1 DEMOTE (#456) + H2/H3 adjudicated (2026-06-12).** H1 10d SPY-hedged p=0.7804 → PEAD not an event-level edge → book=trend-plus-cash. **H2 (reaction_ratio continuous) → NOT_CONFIRMED** (10d t=−1.21; OPT-5 settled/parked, no threshold-hunting). **H3 (PEAD v2 scorecard) → BLOCKED** (frozen feature `revision_momentum` = forward-estimate-revision data, DATA-BLOCKED; one-shot preserved). Required the **event-time options join** (`enrich_event_panel_options.py`) to populate the panel's options-pre-event features (PIT, Opus-deep-dived). Event panel + join + CGM inference = the standing event-hypothesis instruments. |
| **P4** | **Options-as-equity-signal XS sleeve**: CPIV / 25Δ put-skew / O-S volume / term-slope / IV-RV decile L/S, executed in equities via the existing dollar-neutral engine. Kill if simple sorts fail net of costs (NOT an XS-ML revival). | A | M–L ~2–3wk | ❌ **DONE → ALL 5 KILL (2026-06-12).** Data layer + pre-registration (P4a) then the five confirmatory R4 runs (`scripts/run_options_xs_cpcv.py`, full 4y/208wk). **H4a CPIV −2.70 / H4b skew −4.10 / H4c put-O/S −4.43 / H4d term-slope −2.83 / H4e IV/RV −0.12** (week-clustered t) — H4a–H4d significantly NEGATIVE (academic signs don't hold 2022–26 at equity cost), H4e noise. **Options-as-signal equity edge is DEAD; all lines CLOSED** (the inverse is not tradeable — post-hoc sign-mining; Opus deep-dive = growth-crash regime). Live book unchanged (trend+cash). Options = data asset for event-conditioning (H2/H3) only. |
| **P5** | **Trend broadening** (capital-grade EV): more legs / vol-targeting / long-short on the existing set, judged on 19y where t≥2 is reachable + Track-B book delta. | B | M ~1–2wk | ❌ **DONE → PARK (2026-06-12).** ONE frozen spec (`P5-TRENDBROADEN-20260612`): 16 ETFs + long-short + 10% book-vol overlay. 19y confirmatory: **broadened Sharpe 0.30/t=1.31/maxDD−24.7% vs baseline 0.72/t=3.18/−13.9%** → fails all 3 legs → PARK. L/S improved crises but bull-bleed+leverage swamped Sharpe/DD. **Simple 10-ETF long-flat sleeve stays live.** Book-vol-overlay+L/S capability retained in `tsmom.py` (off). Opus deep-dive SAFE (PIT-clean; baseline reproduces +0.72). "Complexity must earn it." |
| **P6** | **Index-VRP micro-sleeve** behind the book gate: one pre-registered defined-risk structure, regime-gated, ≤2%-NAV tail budget; preconditioned on sim-mechanics fixes (stale-mark/unsmoothing, entry-fill convention) + calibrated spreads. 12-mo re-test lockout on fail. | B | M ~2wk | last |

**Reconciliations (contested points):** dispersion = **feature-first / trade-maybe-never** (Claude's cost-wall objection over Gemini/DeepSeek/Grok, on the app's own OPT-3 evidence); CPCV **demoted to robustness** for event strategies (kept primary for path-dependent trained models — not killed); DSR → **report-only** (DeepSeek's `N_eff=k(k−1)/2` fix rejected — wrong direction). **Corrections to reviewers:** forward sacred holdout EXISTS (`SACRED_HOLDOUT_START=2026-11-09`); quarter-level event bootstrap exists (`pead_significance.py` — day-level panel is the upgrade); `EventEdgeStrategy`/options adapter/allocator already built.

**Cross-cutting:** two-track gate spec → `PIPELINE_ARCHITECTURE.md` §7.0-B (P0); **research registry = the program's true N_TRIALS ✅ DONE 2026-06-22 (P0.5: `app/research/family_registry.py`, 25 families enumerated, feeds null_zoo DSR; SSOT `docs/reference/P0_5_FAMILY_REGISTRY_2026-06-22.md`)**; StrategySpec replay-diff contract (4 boring weekly diffs → capital-eligible); **multi-strat eval harness — Phase A ✅ DONE 2026-06-22 (`app/research/multistrat_eval.py`: unified combined-book WF — combined-book CPCV-as-a-unit + return-level drawdown governor + one report: per-sleeve attribution + leave-one-out Track-B + GL-1 tail + P0.5 deflation + fold-in union book; 11 tests); Phase B ERC/covariance sizing = R2 (data-gated); Phase C research↔live replay parity (R1/R2). Scope SSOT `docs/reference/HOLISTIC_MULTISTRAT_STATE_AND_SCOPE_2026-06-22.md`**; per-phase NO-DRIFT doc obligations in blueprint §6.

**First move:** P0 — `scripts/walkforward/gate_calibration.py`, **TSMOM-on-4y positive control** (the cheapest experiment that can falsify the premise of the whole plan).

---

## 🎯 ALPHA-v4 PLAN — Portfolio of Uncorrelated Premia (2026-06-06)

**SSOT:** `docs/reference/QUANT_REVIEW_SYNTHESIS_2026-06.md` (consensus tally, regime policy §4b, full per-phase implementation detail). This table is the index; the synthesis doc is the spec.

**Thesis (5/5 reviewer consensus):** the architecture is good — *re-aim, don't rebuild*. The wall is the **opportunity set + a biased ruler**, not the technique. Stop validating sleeves in isolation; build **4 uncorrelated ~0.4-SR sleeves → book SR ≈ 0.8**. PEAD is *not* proven alpha (p≈0.19, 87% P&L in up-trends = conditional beta) — keep it small, pair it with a crisis-positive sleeve.

**Locked decisions (2026-06-06):** (1) **PEAD dialed back to telemetry** — `pm.pead_size_mult` 3.0→**1.0**, `pm.pead_max_position_pct` 0.10→**0.05** (live, restart-free, done). (2) **Gates: lower bar (~0.45) + reweight to robustness** (residual-alpha-t + fold-consistency primary; keep worst-regime survivability floor). (3) **Targeted re-architecture** (keep execution core; rework research harness; add sleeve + regime-allocator layer). (4) **Free-data only** until a feasibility spike justifies a paid purchase.

| Phase | Item | Effort | Status |
|---|---|---|---|
| **P0** | **Validation integrity** (blocks all): `is_trained` guard fix → full-coverage CPCV for rules-based scorers; purged sequential-WF baseline for trained models; fold-coverage report (year×regime×VIX×trend) gated before perf; **gate recalibration** (avg-Sharpe→~0.45 + residual-alpha-t + fold-consistency + survivability floor); **freeze dead XS-ML** retrain. | ~1wk | **NEXT** |
| **P1** | **PEAD honest reckoning**: ✅ dial-back live (done) · neutralization kill-test (long surprise / short sector ETF) · FF5 factor attribution (residual-α t) · gapper-slippage stress (30–50bps) · entry-timing sensitivity. Decision gate: small real sleeve **or** benchmark-only. | ~1–2wk | After P0 |
| **P2** | **Trend / TSMOM sleeve** (the crisis-diversifier): built `app/strategy/tsmom.py` (vectorized PIT, lookback ensemble, inverse-vol, weekly) — standalone Sharpe **+0.71**; validated as a book addition (DD reduction + marginal book-SR). **✅ LIVE-WIRED (shadow-first) 2026-06-06**: standalone weekly rebalancer `app/live_trading/trend_sleeve.py` (NOT a selector), equal-capital 40/40 w/ PEAD, dormant+shadow by default. Follow-ups: P3 vol-weight/regime tilt; add Alpaca trading-calendar helper; verify reconciler selector-scoping under live fills. | ~3–4wk | ✅ DONE + live-wired |
| **P3** | **Regime-aware allocator + book-level validation** (the unlock): allocator above the live sleeves — equal / inverse-vol / regime-tilt. **Must beat static-equal-weight OOS net of turnover or be dropped.** **✅ LIVE-WIRED (gate-controlled, DISABLED) 2026-06-08**: `app/live_trading/sleeve_allocator_live.py` sets live sleeve weights weekly with a fixed-weight fallback; ships disabled (= today). Gate verdict on 2 sleeves = **equal** (vol/regime OFF until a 3rd sleeve earns them). Re-runnable gate `scripts/run_book_allocator.py --emit-config`. Opus-reviewed (PEAD double-tilt guarded). Follow-ups before enabling vol/regime: re-validate the live one-shot regime path; coarse live trend-vol estimate (warmup-guarded). | ~4wk | ✅ DONE (gate-controlled) |
| **P4a** | **PEAD 2.0** (gated on P1 neutralized-PEAD surviving): genuine-shock scorer — SUE + revenue confirm + fwd estimate revisions + guidance + analyst-prior inconsistency, on `EventEdgeStrategy`. | TBD | Conditional |
| **P4b** | **Options-VRP feasibility spike** (NOT a full strategy): contract-level P&L prototype w/ conservative bid/ask; **decision gate before** any options sim. Free-data-only until the spike says go. | TBD | Conditional |
| **P4c** | **Squeeze-conditioning**: SI days-to-cover as a PEAD-*long* conditioner (uses already-acquired SI data). | ~S | Opportunistic |
| ~~**P4-estrev**~~ | ❌ **Estimate-revision drift — DATA-BLOCKED (2026-06-08).** Forward consensus-EPS-*revision* data is not available in-repo or via wired providers (FMP earnings=report-snapshot, grades=ratings only, fundamentals=actuals); can't be reconstructed retroactively. Needs a collected-going-forward PIT estimate panel or paid I/B/E/S. CPCV harness ready, data absent → not pursued. **ALSO gates H3 (`revision_momentum`, 2026-06-12)** — H3 cannot run as pre-registered without it (one-shot preserved). Unblock = buy **Zacks Estimates** (retail-accessible; Zacks Rank is a revisions model) / I/B/E/S, or collect-going-forward (~1–2 yrs). **Recommendation: keep PARKED — don't buy data for H3** (improves demoted PEAD; the 8-feature peek showed nothing). Worth acquiring only for an independent revision-momentum sleeve. See ML log Phase 4c + DECISIONS 2026-06-12. | done | ❌ |
| ~~**P4-carry**~~ | ❌ **Cross-asset carry — DEAD-ON-ARRIVAL (2026-06-08), not pursued.** Free-data carry reduces to an income/distribution-yield proxy (proper curve/commodity/FX carry needs unavailable free data). Quick screen of a dollar-neutral income-carry book (pre-cost): **−0.22 Sharpe**, alpha t(HAC) −1.4 — high-yield ETFs didn't outperform low-yielders 2007-26. No edge, no module built. NOT filter-hunted. See ML log Phase 4b. | done | ❌ |
| ~~**P4-rev**~~ | ❌ **Short-term reversal sleeve — NOT VALIDATED (2026-06-08), cost-dead.** Dollar-neutral cross-sectional 5d reversal (long losers/short winners, top-500 liquid, PIT membership): gross **+0.40/t=1.28 @2bps** but **-0.90 @10bps** (~159x turnover → ~16%/yr cost drag); genuinely uncorrelated (β~0.1, corr +0.13/+0.03 to PEAD/trend) but a money-loser that drags the book (+1.145→+0.138). Opus-verified real (not a bug); NOT filter-hunted (B5 trap). Harness retained (`app/strategy/reversal.py`, `scripts/run_reversal.py`, 7 tests). **3rd-sleeve slot still OPEN.** | done | ❌ |
| **P5** | Optional/later: futures-roll trend upgrade · merger arb · CJL **forward-estimate-revision** residualized retest (≠ killed A1 ratings) · crypto basis. | TBD | Backlog |

**Cross-cutting (introduced P0/P1):** strategy contracts (frozen spec + code hash) · research ledger (hypothesis registry, incl. nulls) · capital ladder (backtest→shadow→paper→micro-live→pilot→production).

**Explicit STOP list (5/5):** large-cap daily XS-ML · intraday 5-min ML · LLM stock-picking · expensive alt-data · single-name shorting without borrow data · "improving" a backtest by trying filters (the B5 trap).

---

## ✅ ALPHA-v3 PLAN — Event-Edge Family (2026-06-03) — COMPLETE (superseded by Alpha-v4)

> **Outcome:** Track A swept (A1 analyst-drift ❌, A2 short-interest ❌ — both NULL); Track B shipped (B1/B2 clock, B4 ramp, B5 trend filter). Net result: **PEAD remained the sole validated edge.** The 5-LLM review (2026-06-06) then redirected to Alpha-v4 (portfolio of premia) and dialed the B4 ramp back to telemetry. Retained below for archaeology.

**Thesis:** every attempt to extract alpha from *cross-sectional ML ranking* on price/fundamental features came back **noise or beta**. The one thing that worked — **PEAD** — is **event-driven, rules-based, economically grounded**. That's where alpha lives in this stack. So: stop ranking, hunt **discrete-event → measurable-drift** edges, each run through the same honest CPCV + event-bootstrap harness.

**Rigor (non-negotiable, identical to PEAD):** leak-free per-fold CPCV · event-clustered bootstrap significance (not just path-t) · F2-immunity · economic grounding **pre-registered before** the run · **capital only via the live-paper route**, never via k-inflation (multiple-testing trap). A null is a publishable result, not something to patch around.

### Track A — Build edge #2 (the research engine)
| Phase | Item | Effort | Status |
|---|---|---|---|
| **A0** | Generalize `run_pead_cpcv.py`'s `PEADStrategy` → reusable **EventEdgeStrategy** (parameterize event source → entry/exit → drift window over AgentSimulator + CPCV gate + event-bootstrap). One refactor; every future edge becomes a thin adapter. | ~1d | **NEXT** |
| ~~**A1**~~ | ❌ **Analyst up/downgrade drift — NOT VALIDATED (2026-06-03).** CPCV long-only looked best-in-campaign (+0.894, t=2.85) but was a **52% fold-skip artifact**: dollar-neutral L/S collapsed to +0.342/t=1.24 and full-window CAPM gave alpha t=0.20 / residual Sharpe ~0 — noise, not alpha. See ML log. Reusable cross-check: `scripts/analyst_beta_check.py`. | done | ❌ |
| **A2-data** | **Short-interest / short-volume data** — ✅ **DONE 2026-06-03**. Source = **Polygon** (FINRA-originated, existing key; FMP/Finnhub/FINRA-OAuth all unavailable on plan). Backfilled **SI 140k rows / 741 tickers (2017→2026, bi-monthly, days-to-cover)** + **SV 403k rows / 720 tickers (2024→2026, daily Reg SHO)**, survivorship-safe (PIT R1K union). PIT leak-killer: conservative `knowable_date = settlement + 10 bdays` (GME-squeeze validated). `app/data/short_interest_provider.py`, `scripts/backfill_short_interest.py`, `docs/reference/SHORT_INTEREST_DATA.md`. | done | ✅ |
| ~~**A2**~~ | ❌ **Short-interest factor — NOT VALIDATED (2026-06-03).** Dollar-neutral long-low-DTC / short-high-DTC (Boehmer/Asquith anomaly): CPCV **−1.213, t=−3.53, PF 0.66** — significantly negative. The anomaly **reversed** in 2020–2026 (meme-era squeezes: high-DTC names rose). Flipping it = overfitting, not an edge. See ML log. `app/ml/short_interest_factor_scorer.py`. | done | ❌ |
| **A3** | If ≥2 edges survive: **combined-book** portfolio construction (PEAD + edge#2) → test portfolio IR > either alone (the diversification prize; self-certifies faster). | TBD | Conditional |

### Track B — Productionize & aggressively ramp PEAD in paper (the self-certification clock)
| Phase | Item | Effort | Status |
|---|---|---|---|
| ~~**B1**~~ | ✅ **DONE** (already built/scheduled): EOD `record_daily` upsert in `_run_eod_jobs` writes PEAD realized/unrealized P&L + fills (`selector="pead"`); `_realized_sharpe` + `weekly_rollup`. The self-certification clock runs at 16:30 ET. 12 tests. | done | ✅ |
| ~~**B2**~~ | ✅ **DONE**: Friday `weekly_rollup` in `_run_eod_jobs` (weekday==4) + vacuous-email guard (min_days=3). | done | ✅ |
| ~~**B4**~~ | ✅ **DONE 2026-06-04** (deploys at restart): **aggressive paper ramp** — config-driven `pm.pead_size_mult` (3.0) + `pm.pead_max_position_pct` (0.10), live-tunable, PEAD-specific (`apply_pead_size_ramp`). RM made PEAD-aware (`validate_position_size` override) so the 10% cap isn't clipped to the global 5%; aggregate still bounded by the 80% gross cap. **ADV-participation** instrumentation logged + on proposals (slippage already per-fill). 19 tests. PAPER ONLY. | done | ✅ |
| **B3** | Announce-day-move / earnings-surprise capture per PEAD signal (cockpit's defining column; live-path write + migration). | ~S–M | Opportunistic |
| ~~**B5**~~ | ✅ **DONE 2026-06-04** (deploys at restart): SPY<200d trend filter replaces the VIX>30 block — **validated +0.661 vs +0.546** (every metric better) on the same window. Config-reversible (`pm.pead_regime_control="trend"`, `pm.pead_trend_ma=200`); fail-CLOSED to VIX if SPY unavailable. 19 tests. | done | ✅ |

### Closed / retired (green-lit 2026-06-03)
- ❌ Cross-sectional ML ranking line — **closed** (DECISIONS.md 2026-06-03).
- ❌ Spike B (residualized ranker features) — **shelved permanently** (nothing to unmask: neutralized book rewarded zero IC).
- ♻️ §3.3 short-interest-as-ranker-feature — **re-scoped** into A2 as its own event edge.
- ✅ Insider-buying cluster — already tested → FAIL/weak (ML log); not repeated.
- ⏸ Index reconstitution — dropped (≈annual → too few events to power).

---

## STRATEGIC DIRECTION — 2026-05-18 PIVOT

After 4 independent LLM reviews (DeepSeek, Gemini, ChatGPT, Claude/Opus 4.7) of the entire system, the active strategy is shifting from **long-only cross-sectional factor portfolio** to **directional Long/Short**.

**Why the pivot:**
- The Phase C LambdaRank campaign (9 runs) collapsed structurally in bear-market folds because cross-sectional top/bottom quintile labels in a 25% drawdown teach the model "fell least = winner" — defensive signals that contradict the momentum-quality features in non-bear regimes.
- Long-only cross-sectional ranking cannot survive bear regimes by design.
- L/S is regime-independent: shorts produce P&L when the market falls, longs when it rises. The 0.80 Sharpe gate becomes achievable.

**New strategy: directional Long/Short**
- Net exposure target: **+40% net long** (configurable via `pm.ls_net_exposure_pct`)
- Gross exposure: ~150% (e.g. 95% long + 55% short)
- Top-N: 20 longs / 15 shorts (configurable)
- WF gate stays at **avg Sharpe ≥ 0.80, min fold ≥ -0.30** (now reachable)
- Capital: $100k paper

Full synthesis lives in `docs/QUANT_REVIEW_SYNTHESIS_2026_05_18.md`.

---

## Key Architectural Decisions (2026-05-18)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Directional Long/Short, 40% net long** | Regime-independent; configurable via `pm.ls_net_exposure_pct` |
| 2 | **Factor portfolio → paper after PIT audit + L/S WF ≥ 0.80** | Already integrated (PR #224). Continuous improvement after promotion. |
| 3 | **Intraday on backburner** | Paper continues running, zero new dev. Revisit after swing validated. PDT rule change June 2026 may help. |
| 4 | **PEAD as second strategy** | FMP already has EPS surprise data (PIT-safe via `filingDate`, $0 extra cost). |
| 5 | **Equal-weight strategy allocation** | 50/50 factor + PEAD when both active; simpler than risk-parity |
| 6 | **Survivorship audit required** | `scripts/audit_survivorship.py` written; must run before any new WF |
| 7 | **Live promotion criteria** | 3 months paper, Sharpe ≥ 0.50 annualized, max DD ≤ 15%, max single position ≤ 8% NAV |
| 8 | **WF gate stays 0.80 avg Sharpe** | Achievable for L/S; was structurally unreachable for long-only ranking |

**Convergence across all 4 reviewers (treat as facts):**
- IC must be computed for every model — has never been done; most critical gap
- Long-only cross-sectional ranking is broken in bear regimes (label inversion)
- Factor decomposition required: regress factor returns on SPY + AQR MOM + QMJ; require alpha t-stat > 2
- Sector rotation as floor benchmark (11 SPDR ETFs, top-3 by 6mo momentum)

---

## Active Phase Plan

```
Phase E (DONE)        IC + survivorship scripts                  2026-05-18
Phase F (DONE)        L/S infrastructure                          2026-05-20
Phase 0-align (DONE)  10 WF simulation bugs fixed, PR #288       2026-05-27
Phase 1 (DONE)        Model rank-IC diagnostic + honest WF       2026-05-27
Phase LS0 (DONE)      L/S experiment — enable-shorts on v221     2026-05-27  FAIL (-1.31 min fold)
Phase LS1 (DONE)      swing_short_v1 design spec (Opus 4.7)      2026-05-27  DEFERRED (see below)

── ACTIVE MISSION: Find honest long-side edge (3-week timebox) ──────────────

Phase LX1 (DONE)      Experiment: equal-weight 5 IC features + B2 overlay    2026-05-27  avg Sharpe +0.557
Phase LX2 (DONE)      Experiment: v186 honest clean re-run                   2026-05-27  avg Sharpe +0.171 FAIL — XGBoost 82 features worse than equal-weight
Phase LX3 (DONE)      Experiment: Retrain XGBoost on 5 IC-validated features only        2026-05-27  avg Sharpe -2.344 FAIL — XGBoost 5 features ≪ equal-weight; ML weighting ruled out
Phase LX4 (DONE)      Experiment: Concentrated LX1 (target_n=15) + factor-stability gate  2026-05-27  avg Sharpe -0.251 FAIL — but win rate 64.7% on 1,332 trades confirms real signal edge; problem is DD (17-25%), not direction
Phase LX5 (DONE)      Experiment: Inverse-vol position sizing on LX1 (target_n=30, 20d vol lookback, 0.5x-2x cap)  2026-05-28  avg Sharpe +0.032 FAIL — inv-vol helps F1/F3 (+0.24/+0.67) but fold-2 killer remains; fold-spec mismatch vs LX1 invalidates direct comparison
Phase LX1-rb (DONE)   Re-baseline LX1 on identical folds (--as-of 2026-05-28)    2026-05-28  avg Sharpe +0.079 FAIL — original +0.557 was fold-period artifact; honest equal-weight baseline is +0.079
Phase LX6a (DONE)     Entry-only regime gate (VIX≥30→30% on new entries)              2026-05-28  avg -0.127 FAIL — WORSE than baseline; blocks recovery entries; ruled out
Phase LX6b (DONE)     Hard-exit regime gate (VIX≥30→liquidate all longs at rebalance)  2026-05-28  avg -0.103 FAIL — F2 worsened (-0.72→-0.88); exits at bottom, misses bounce; ruled out. PIVOT TRIGGERED.
Phase LX7 (DONE)        L/S: long top-20 + short bottom-20 by 5-feature composite, +40% net long  2026-05-28  avg +0.036 FAIL — short-book thesis wrong; bottom-20 composite = value/post-crash names that rally fastest; L/S ruled out
Phase LX8 (DONE)        7% per-position trailing stop on LX1 (bug-fixed as LX8b)                 2026-05-28  avg -0.207 FAIL — stop cuts winners (PF 0.957); bug found: stop fired on swing-model positions too (fixed in PR #305). Root cause confirmed: timing interventions don't fix beta exposure problem.
Phase LX9-B1 (DONE)      10-day rebalance cycle on LX1                                              2026-05-29  avg +0.057 FAIL — F2=-0.72 unchanged; cadence definitively not the lever; all timing fixes ruled out
Phase LX9-A (IN PROGRESS) Beta-neutralize feature ranking (OLS residualize vs trailing 252d beta-to-SPY)  implementation by Opus 4.7; WF launch pending  P(success)~45%; highest-conviction structural fix remaining
Phase LX-gate         Long side honest WF Sharpe > 0.8 → UNLOCK short model

── SHORT SIDE (deferred until LX-gate passes) ───────────────────────────────

Phase SD0             Data infra: VIX3M + FINRA short interest (safe, parallel)  ~4h
Phase SD1             Rules-based short overlay (3-day, no ML)               ~3 days
Phase SD2             swing_short_v1 ML model (full 10-day build)            after LX-gate
Phase SD3             30-day paper trade gate for shorts                      3 months

── DOWNSTREAM (unchanged) ───────────────────────────────────────────────────

Phase G               PEAD strategy                               after LX-gate
Phase H               3-month paper trading gate                  after SD2
Phase I               Live $100k                                  after H passes
```

### LX-gate definition
Long side passes when ALL of:
- Honest WF avg Sharpe ≥ **0.80** (same gate as before)
- No fold < **-0.30**
- DSR p ≥ **0.95**
- Beats naive B2 (SPY>200d MA = 0.808) by ≥ **0.10 Sharpe**

### Short model deferral rationale (Opus 4.7, 2026-05-27)
Building `swing_short_v1` on a long book with DSR z=-10.4 means every future bug is unattributable (long? short? interaction?). The same research process that produced the +0.374 false-positive will embed false positives in short-model IC selection. Fix the foundation first. Full design spec is preserved in `ML_EXPERIMENT_LOG.md` and ready for implementation the moment LX-gate clears.

### Phase 1 Key Findings (2026-05-23)

v216 rank-IC@20d = **0.0012** overall (noise), BUT by-year reveals a **regime problem**:

| Year | IC@20d | t-stat | Regime |
|------|--------|--------|--------|
| 2021 | **+0.023** | **4.92** ✅ | Bull, low vol |
| 2022 | **-0.028** | **-8.73** ❌ | Bear, rate shock |
| 2023 | -0.009 | -2.11 ❌ | Mixed |
| 2024 | +0.010 | +1.98 🟡 | Bull recovery |
| 2025 | +0.017 | +4.02 🟡 | Bull |

**Interpretation:** Features work in trending/low-vol regimes. The 2022 rate-shock year **inverted** the model signal (high-momentum stocks fell hardest), destroying 5-year aggregate IC. This is a **label design problem** (LambdaRank trained cross-sectionally without regime conditioning), not a dead feature set.

**Next step (Phase 2):** Compute regime-conditional IC (filter to BENIGN regime days only). If IC > 0.02 in BENIGN regime → switch to policy-realized binary labels + regime filter in training.

---

## STRATEGIC PIVOT — 2026-05-24: Rebalancing Execution Architecture

**Context:** After Phase 4 v3 WF (all bugs fixed, Sharpe=-0.036, 7 trades/fold), Opus 4.7 analysis identified the root problem: the execution layer (RSI/EMA signal triggers) is architecturally mismatched with LambdaRank, which is a cross-sectional portfolio selection model. L3 Bridge Test confirmed alpha exists at Sharpe=0.577 when the model is used correctly (rank → pick top-N → rebalance). Fix the execution layer, not the model.

**Design decisions (Opus 4.7, 2026-05-24):**

| Decision | Detail |
|----------|--------|
| Execution mode | `REBALANCE` (new PM mode alongside existing `SIGNAL`) |
| Rebalance cadence | 20 trading days (matches 20d label horizon) |
| Target positions | N=30 at $20k, N=50 at $100k+ |
| Long/short | Long-only first; shadow-track 3 short candidate sets |
| Short deployment | $50k+ account, separate short model (not LambdaRank bottom-N) |
| Position sizing | Cascade: regime gate → inverse-vol base → NIS modulation |
| Regime exposure | Bull=100%, Neutral=70%, Bear=30% invested |
| Exit logic | 20d hold baseline + profit harvest (+12% in ≤7d) + NIS exit (<-0.4) + regime-flip forced rebalance; NO price stops |
| Score weighting | Deferred — LambdaRank scores are ordinal, not calibrated expected returns |

**Why no price stops:** 8% stop on a 20d-horizon model fires on noise (typical R1000 20d vol = 12-18%). Replace with NIS-based exits (information-driven, not price-noise-driven).

**Why no LambdaRank bottom-N as short book:** L2 L/S (0.397) < L3 long-only (0.577) proves the bottom ranking is not a good short signal. Bottom-ranked stocks are "mediocre" not "likely to fall" — and mediocre has positive drift (equity risk premium) working against shorts.

---

## Phase RA — Rebalancing Baseline ← ACTIVE NEXT

**Goal:** Replace signal-triggered execution with ranking-based rebalance. Reproduce L3 alpha (0.577) inside the WF framework. Pass criteria: WF avg Sharpe ≥ 0.50, ≥30 trades/fold.

**What to build:**
1. `app/strategy/portfolio_construction.py` — liquidity filter, sector cap (30%), hysteresis (add at rank ≤15, drop at rank ≥30), equal-weight sizing
2. PM `REBALANCE` mode — on rebalance date: score all symbols → apply constraints → compute target set → emit close/open orders
3. Config flags: `EXECUTION_MODE`, `REBALANCE_DAYS=20`, `TARGET_POSITIONS=30`, `SECTOR_CAP=0.30`
4. Attribution logging: per-trade decomposition into selection/exit/cost components
5. Regime gate (Layer 1): Bull=100%, Neutral=70%, Bear=30% gross exposure

**WF validation:** Run `walkforward_tier3.py` with rebalance mode. Expect ~11 rebalance events × ~6 rotations × 2 sides = ~130 fills/fold. Sharpe computed on daily equity curve, not per-trade.

---

## Phase RB — Sizing Overlay (after RA passes WF)

1. Inverse-volatility base weights (60d realized vol, cap 0.5×–2× equal-weight)
2. NIS modulation: NIS > +0.4 → ×1.25 size; NIS < -0.4 → exit position
3. Renormalize to regime gross target

---

## Phase RC — Exit Overlays (after RB validated)

1. Profit harvest: +12% in ≤7 days → rotate to next unheld ranked name; cap rotations at 25% of book per window
2. NIS-driven exits (already in RB; formalize as explicit exit event)
3. Regime-flip forced rebalance: bull→bear → immediate rebalance + de-risk to 30%

---

## Phase RD — Shadow Short Infrastructure (parallel to RA-RC)

Track three short candidate sets (shadow P&L only, no live trades):
1. Bottom-30 of LambdaRank (naive null)
2. Bottom-30 filtered by NIS < -0.4
3. High-short-interest + negative momentum names

Store with `is_shadow=True`, `shadow_strategy` enum. Daily P&L persistence. After 6-12 months data → decide short deployment.

---

## Phase RE — Short Deployment (deferred, $50k+)

Separate short model with different features: accruals (Sloan), leverage deterioration, dilution, high short-interest + rising days-to-cover. Deploy after Phase RD data justifies. Operationally viable at $50k+.

---

## Phase D — Factor Portfolio Integration ✅ COMPLETE (2026-05-18) — superseded by L/S pivot

- PRs #224 / #225 merged: `app/ml/factor_scorer.py` + PM routing + SPY>MA200 + VIX<30 gate
- Validated Sharpe 1.335 from `scripts/factor_portfolio_backtest.py` (monthly rebalance, equal-weight)
- AgentSimulator WF mismatch documented (ATR stops vs monthly rebalance — wrong tool, not signal failure)
- **Decision:** Factor portfolio stays as long sleeve of L/S; will be re-validated under Phase F WF after L/S infrastructure is built.

---

## Phase E — P0 Scripts ✅ COMPLETE (2026-05-18)

P0 bug audit (per LLM review synthesis):

| Item | Status |
|---|---|
| **P0.1 Entry price** | ✅ ALREADY CORRECT — `agent_simulator.py:804` uses `today_bar["open"]` |
| **P0.2 Intrabar stop** | ✅ ALREADY CORRECT — `agent_simulator.py:935-944` checks `today_low <= stop_price` / `today_high >= target_price` |
| **P0.3 Factor IC computation** | ✅ Script written: `scripts/compute_factor_ic.py` (Spearman IC, monthly rebalance dates, fwd 10d returns; pass ≥ 0.02, t-stat ≥ 2.0). Not yet run. |
| **P0.4 Survivorship audit** | ✅ Script written: `scripts/audit_survivorship.py` (checks delisted ticker coverage in daily cache). Not yet run. |

**Deliverables:**
- `scripts/compute_factor_ic.py`
- `scripts/audit_survivorship.py`
- `docs/QUANT_REVIEW_SYNTHESIS_2026_05_18.md` (synthesis + P0–P4 backlog)

---

## Phase F — Long/Short Infrastructure ✅ COMPLETE (2026-05-20)

**WF Result:** avg Sharpe 0.579, GATE FAILED (Fold 4 = -0.98 during April 2025 tariff shock). Factor IC = -0.0064 → no signal. L/S infrastructure stays as foundation for PEAD; factor portfolio deprioritized as alpha source.

**Goal:** Convert the factor portfolio + (future) PEAD into a directional L/S engine.

### F.1 — `FactorPortfolioScorer` returns shorts
- `app/ml/factor_scorer.py`: extend to return `[(symbol, confidence, direction)]` where `direction ∈ {LONG, SHORT}`
- Bottom-N composite score → SHORT candidates (with inverted score sign)
- Existing top-N → LONG candidates

### F.2 — PM proposals carry SELL_SHORT direction
- `portfolio_manager.py`: emit `Proposal.direction = SHORT` for short candidates
- New proposal type: `SELL_SHORT` (entry) and `BUY_TO_COVER` (exit)
- Routing path: factor → both long and short sleeves to RM

### F.3 — `AgentSimulator` supports short P&L
- P&L sign inverted for shorts: `pnl = (entry - exit) × qty`
- Borrow cost accrual: **0.5%/yr** (configurable) deducted daily
- Inverted stop/target: stop above entry, target below
- Update `agent_simulator.py:804/935-944` for short branch

### F.4 — `risk_rules.py` net-exposure + short-heat gates
- Net exposure gate: target ± 15% tolerance around `pm.ls_net_exposure_pct`
- Short heat: cap total short notional (e.g. 75% of NAV)
- Hard locate-availability check (Alpaca easy-to-borrow list)

### F.5 — Config keys in `agent_config.py`
```
pm.ls_net_exposure_pct       = 0.40
pm.ls_top_n_long             = 20
pm.ls_top_n_short            = 15
pm.ls_borrow_cost_annual_pct = 0.005
pm.ls_net_exposure_tolerance = 0.15
```

### F.6 — Walk-forward re-run with L/S
- 5-fold, 6-year, R1K universe with PIT membership
- **Gate:** avg Sharpe ≥ 0.80, min fold ≥ -0.30
- Block PEAD work until F.6 passes

---

## Phase G — PEAD Strategy

**Goal:** Second orthogonal strategy. Post-Earnings Announcement Drift.

### G.1 — `PEADScorer`
- New module: `app/ml/pead_scorer.py`
- Uses `fmp_provider.get_earnings_features_at(symbol, asof)` (PIT-safe, `filingDate` based)
- Features: standardized EPS surprise, revenue surprise, guidance revision direction, post-announcement gap
- Score = signed surprise z-score × confidence

### G.2 — Multi-strategy routing in PM
- PEAD has priority for symbols within ≤ 5 trading days post-earnings
- Factor fills remainder of allocation
- **Capital allocation: 50/50 equal-weight by strategy when both active**

### G.3 — `scripts/run_pead_walkforward.py`
- 5-fold, 6-year WF on R1K
- Gate: avg Sharpe ≥ 0.80, min fold ≥ -0.30 (same as factor)
- IC check: ≥ 0.02 on forward 5d returns to surprise score

---

## STRATEGIC DECISION — 2026-05-22: Stay Single-Name L/S, Fix Alignment

**Kill criterion triggered:** Model WF avg=-0.275 (folds: -0.842, -0.469, -0.624, +1.074, -0.513). 4/5 negative.

**Decision:** Do NOT pivot to ETFs. The kill criterion result is not evidence that single-name L/S has no alpha — it is evidence that the training/WF pipeline was misaligned (see root causes below). Fix the alignment first, then re-evaluate.

**Root causes identified (Opus 4.7 review, 2026-05-22):**
1. WF has always evaluated `FactorPortfolioScorer` (rules-based), never `model.predict`. Spearman=0.035 between the two — completely independent systems.
2. Training objective (triple-barrier labels) ≠ WF objective (AgentSimulator ATR-stop P&L). Different things were optimized vs evaluated.
3. Universe survivorship bias not audited.
4. Feature construction parity between training/WF/live not verified.
5. LambdaRank single-row predict = 0.0 (fixed 2026-05-22).
6. TSNormalizerState empty for LambdaRank causing all-zero inference (fixed 2026-05-22).

---

## Phase 0 — Freeze, Audit, Instrument (1-2 days) ← NEXT

**Goal:** Stop changing behavior until misalignments are measurable.

### 0.1 — Canonical contract (`app/ml/contracts.py`)
- `@dataclass FeatureRow`: symbol, asof_ts, feature_name→value, label, label_meta
- `@dataclass ScoreRow`: symbol, asof_ts, raw_score, normalized_score, rank
- `schema_hash()` — hash of feature names + dtypes + order
- Training, WF, live PM all import this. Mismatch = startup fail.

### 0.2 — Parity test harness (`tests/parity/test_training_wf_live_parity.py`)
- 5 fixed (symbol, date) pairs spanning 2019–2025
- Assert: training pipeline features == WF features == live PM features (within 1e-9)
- Assert: schema_hash identical across all three paths
- **Start by expecting failure — every diff surfaces a real bug.**

### 0.3 — Structured logging in all three paths
- After feature construction: log schema_hash, row count, NaN count
- After normalization: log normalizer ID, feature mean/std
- After model.predict: log score distribution, top-10 symbols
- **Files:** `app/ml/training.py`, `scripts/run_model_walkforward.py`, `app/agents/portfolio_manager.py`

---

## Phase 1 — Catalog All Misalignments (2 days)

**Goal:** Evidence-backed list of everything broken. No fixes yet.

### 1.1 — Feature construction audit → `docs/feature_audit.md`
For every feature: source, lookback, training code path, inference code path, PIT compliance.
- Rolling z-scores: cross-section in training vs cs_normalize on N symbols in live?
- Fundamentals: lagged by report date or filing date?
- Sector/industry: static-as-of-today (survivorship) or PIT?

### 1.2 — Label audit → `docs/label_audit.md`
Compute correlation: triple-barrier label vs AgentSimulator P&L for same trades.
**Expected:** correlation < 0.3 → training objective is provably wrong → Phase 2 mandatory.

### 1.3 — Universe/survivorship audit → `docs/universe_audit.md`
Is "Russell 1000" today's membership backfilled, or PIT? Count delisted symbols in training.
**Script:** `scripts/audit_survivorship.py` (already written)

### 1.4-1.6 — Normalization, inference path, execution audits
- cs_normalize on 5 symbols ≠ cs_normalize on 750 — live scoring is wrong
- AgentSimulator vs live: stops, slippage, sizing, rebalance cadence
- Live PM inference paths: selection (full universe) vs reeval (single symbol)

---

## Phase 2 — Replace Training Objective (3-5 days)

**Goal:** Train the model to predict what WF and live actually reward.

### 2.1 — Canonical label: forward N-day residual return vs sector
- `app/ml/labels.py` (new): `compute_residual_return_label(prices, sectors, horizon=5)`
- N matches live holding period (~5-10 days for swing)
- Continuous label (not binary triple-barrier)

### 2.2 — LambdaRank group = (date, universe), gain = return decile rank
- `app/ml/training.py`: group per date, continuous residual label
- **Pass criterion:** Training IC (rank corr predicted vs realized 5d residual) > 0.02

### 2.3 — Retrain swing_v215
- Log in `docs/living/ML_EXPERIMENT_LOG.md` with IC, fold Sharpes, gate result

---

## Phase 3 — Lockstep WF (2-3 days)

**Goal:** WF evaluates exactly the same code as training, on PIT-correct universe.

### 3.1 — Single feature builder (`app/ml/features.py`)
One function: `build_feature_matrix(symbols, asof_date)` → DataFrame with schema_hash.
Training, WF, live PM all call this. Delete duplicate code.

### 3.2 — PIT universe in WF
Top-1000 by trailing 60d ADV at each rebalance date. No symbol before IPO or after delisting.
**File:** `scripts/run_model_walkforward.py`

### 3.3 — WF calls model.predict on FULL universe each day
cs_normalize on K symbols ≠ cs_normalize on 750. WF must score entire eligible universe, take top-K.
**File:** `app/backtesting/agent_simulator.py` — inject `signal_fn(date, universe) -> Series[score]`

### 3.4 — Parity test green
Phase 0.2 test passes for all 5 (symbol, date) pairs across training/WF/live.

---

## Phase 4 — Honest WF Diagnosis (2 days)

Run aligned WF with v215. Required outputs regardless of pass/fail:
- **Per-fold IC:** rank corr predicted score vs realized 5d residual
- **Decile P&L:** top-decile minus bottom-decile per fold
- **Hit rate by sector and VIX bucket**
- **Attribution:** selection vs sizing vs stops

**Decision rules:**
- IC > 0.02 but P&L < gate → execution/sizing issue → Phase 5
- IC ≈ 0 → signal issue → Phase 7
- IC > 0.02 + decile P&L > SPY but full WF < SPY → stops destroying alpha → Phase 5

---

## Phase 5 — Execution Alignment (2-3 days, if IC > 0 but P&L < gate)

- Strip simulator to pure mode: equal-weight top-K long/bottom-K short, hold N days, no stops
- Sweep: ATR multiplier {1.5, 2.5, none}, holding period {3, 5, 10, 20}, K {5, 10, 20, 50}
- Find config where pure mode beats SPY; layer stops back only if they improve it
- Align live PM to winning config

---

## Phase 6 — Pre-Production Validation

- CPCV on v215 with lockstep pipeline
- 6-month strict holdout (most recent, not used in training)
- 1-week live shadow mode: log proposals without executing, verify match to WF
- **Pass:** holdout Sharpe within 1σ of CPCV median

---

## Phase 7 — Signal Engineering (if IC ≈ 0 in Phase 4)

Feature additions in priority order:
1. Earnings surprise / drift (PIT EPS vs estimate)
2. Short interest / days-to-cover (weekly, PIT)
3. Analyst revision breadth (PIT)
4. Cross-sectional residual momentum (12-1, sector-neutralized)
5. Quality: accruals, ROIC trend (PIT fundamentals)
6. Microstructure: realized vol-of-vol, Amihud illiquidity

Each addition: must improve fold IC ≥ 0.005 to stay.

---

## Sequencing Rationale

| Phase | Blocks | Why |
|-------|--------|-----|
| 0 | All | Without contracts + parity tests, every fix is unverifiable |
| 1 | 2,3 | Must know all misalignments before choosing fix order |
| 2 | 3,4 | Model must predict the right thing before WF can validate it |
| 3 | 4 | WF must call same code as training before result is meaningful |
| 4 | 5,6,7 | Diagnosis determines which branch to take |
| 5/7 | 6 | Don't validate until execution OR signal is fixed |
| 6 | Live | Don't risk capital until CPCV + holdout + shadow pass |

---

## Phase H — 3-Month Paper Trading Gate

**Live promotion requires all of:**
- 3 calendar months of continuous paper trading
- Annualized Sharpe ≥ 0.50
- Max drawdown ≤ 15%
- Max single position ≤ 8% NAV (RM-enforced)
- Live-vs-sim shortfall < 30 bps/day median
- Zero unreconciled Alpaca/DB position mismatches

---

## Phase I — Live $100k

Only after Phase H passes. Start small (e.g. $25k of $100k); scale up after 4 weeks of clean live data.

---

## Open Technical Debt (parallel, non-blocking)

| Item | Owner | Notes |
|---|---|---|
| `russell1000_membership.parquet` — 198/750 tracked | Data | Rebuild from iShares IWB monthly holdings |
| Daily price cache gap (~71 R1K symbols missing) | Data | `scripts/backfill_yfinance.py` + Polygon fallback |
| R1K fundamentals backfill (~320 R1K-only symbols) | Data | FMP — overnight job |
| StrategyContract abstraction (ChatGPT P2 suggestion) | Eng | Not blocking — refactor after L/S validated |
| Factor decomposition vs SPY + AQR MOM + QMJ | Quant | Required before live; alpha t-stat > 2 gate |
| Sector rotation floor benchmark | Quant | 11 SPDRs, top-3 by 6mo momentum, monthly rebalance |

---

## Phase Plan G-Pre → K: Foundation Hardening (2026-05-22, Opus 4.7 Reviewed)

### Background & Diagnosis

After Factor IC = -0.0064, PEAD avg 0.346, and Factor Portfolio best 0.701 (all gate FAILED), a
4-LLM synthesis + Opus 4.7 architectural review identified the root causes in descending probability:

| # | Root Cause | Est. Probability |
|---|---|---|
| 1 | **Label-evaluation mismatch**: training on triple-barrier 5d, evaluating on ATR-stop simulator Sharpe — these are different objectives | 70% |
| 2 | **Score artifact mismatch**: WF measures FactorPortfolioScorer callable, not model.predict; Spearman correlation unknown | 50% |
| 3 | **Feature pipeline asymmetry**: train and WF features computed via different code paths; never diffed | 30% |
| 4 | **Genuine no-edge**: momentum/quality/value doesn't work on R1K post-2020 at 5bps costs | 30% |
| 5 | **Survivorship/PIT leakage** | 10% |

> Note: these are not mutually exclusive — likely 2-3 are simultaneously true.

**Kill criterion (define now, before sunk-cost bias):** If after G-Pre + Phase G, SPDR rotation
beats our system by > 0.2 Sharpe → pivot sector rotation as primary strategy and deprioritize
single-name picker development.

---

### Phase G-Pre.0 — Fold Reproduction by Hand (½ day, FIRST)

Pick Fold 3 (well-behaved). Trace one stock, one trade:
- Compute features via training pipeline code path
- Compute same features via WF code path
- Feed both to scorer and model.predict
- Trace AgentSimulator P&L for that trade (entry price, stop, exit price, P&L)

If P&L cannot be reproduced within 1bp → **stop everything and fix the reproduction gap before proceeding.**

---

### Phase G-Pre.1 — Score Reconciliation (2 hours, BLOCKING)

For 20 random (date, universe) tuples sampled from WF fold windows:
```python
scorer_ranks = FactorPortfolioScorer(**SCORER_CONFIG)(day, symbols_data, vix_history)
model_ranks  = model.predict(wf_features_for_day)
spearman = scipy.stats.spearmanr(scorer_ranks, model_ranks).statistic
top10_overlap = len(set(top10_scorer) & set(top10_model))
```
**Pass criterion:** Spearman ≥ 0.85 AND top-10 overlap ≥ 7/10 on ≥ 90% of dates.

**If fails:** all prior WF numbers are measuring the scorer, not the model. Re-run all experiments
using `model.predict` rank to understand what the model actually does.

---

### Phase G-Pre.2 — Label Construction Decision (THE critical fork)

Current: triple-barrier labels (5d, +2σ/-1σ) → evaluated by AgentSimulator ATR-stop Sharpe.
These are different objective functions. The model optimizes for one thing; WF grades another.

**Option A (cleanest, ~4 hrs compute):** Replace triple-barrier with simulator-derived labels.
For each (symbol, date), forward-simulate with same ATR stops used in WF. Label = realized P&L.
This makes training and evaluation measure the same thing.

**Option B (1 day):** Replace with forward return matching typical simulator holding period.
Measure mean holding period from recent WF run logs. If ~7d, use r_{t+7} as continuous label
and train LambdaRank on those. Simpler but doesn't capture stop-out effects.

**Decision rule:** If IC of triple-barrier model on 10d forward returns stays at ≈ 0 after
G-Pre.1 reconciliation → must switch to Option A or B. Do not proceed to Phase H with broken labels.

---

### Phase G-Pre.3 — Feature Equality + PIT + Universe (parallelizable, 1 day)

**Feature equality:**
- Pickle train feature DataFrame from retrain artifacts at one fold start date
- Recompute same features via WF code path at same (symbol, date) index
- `pd.testing.assert_frame_equal(train_df, wf_df, rtol=1e-6)` on intersection
- Known failure modes: z-score window size, cross-sectional rank denominator, NaN handling

**PIT spot-check (30 min):**
```python
for (symbol, date) in random_sample_10:
    fundamentals = wf_pipeline.get(symbol, date)
    filing_date = FMP.get_filing_date(symbol, fundamentals['quarter'])
    assert filing_date <= date - timedelta(days=1)  # T-1 minimum
```
Any failure = full fundamentals stack is leaked.

**Also:** shuffle fundamentals dates by ±90 days for one fold. If Sharpe changes by < 0.05,
fundamentals were not contributing signal (consistent with IC = -0.0064).

**Universe consistency:**
```python
symmetric_diff = train_universe ^ pit_union_universe
assert len(symmetric_diff) / len(train_universe) < 0.05  # < 5%
```

**G-Pre.4:** Run `scripts/audit_survivorship.py` and `scripts/compute_factor_ic.py`.

---

### Phase G — Benchmarks (run in parallel with G-Pre, 1–2 days)

Two naive baselines. **These must be run before building anything more complex.**

**G.1 — SPDR Sector Rotation (50 LOC):**
- 11 SPDR ETFs: XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC
- Rank by 6-month total return with 1-month skip (r_{t-7m to t-1m})
- Top-3, equal-weight, monthly rebalance, 5bps costs
- 5-fold, 6-year WF, same gate (avg Sharpe ≥ 0.80)
- Also compute: equal-weight 11-SPDR portfolio as second floor
- **If top-3 rotation passes gate → it becomes primary strategy; pivot immediately**

**G.2 — Pure 12-1 Momentum Picker (30 LOC):**
- Rank R1K by 12-month return minus last 1 month
- Top-N longs, bottom-N shorts (test both long-only and L/S)
- Also test: 6-1 momentum, 3-1 momentum as variants
- 5-fold, 6-year WF
- **Purpose:** if pure 12-1 outperforms factor composite → composite weighting destroys the signal

**G.3 — Fold 4 calibration check (10 min):**
- Run equal-weight SPY (or SPY itself) through Fold 4 only
- If SPY Sharpe in Fold 4 ≈ -0.5 or worse → gate floor of -0.30 on Fold 4 is impossible without
  explicit short exposure; gate needs adjustment

---

### Phase H — Regime Allocator (1 week, only after G-Pre resolves labels)

**Do not build this until G-Pre.2 label decision is made. An allocator on a broken picker
modulates noise, not alpha.**

**Inputs (max 4 features — hard limit):**
- % of R1K universe above 200DMA (breadth) ↑ → exposure ↑
- Cross-sectional return dispersion (std dev of 20d returns) ↑ → exposure ↓
- VIX/VXV ratio (term structure, not level) ↑ → exposure ↓
- Credit spread proxy: HYG/IEF total return ratio ↑ → exposure ↑
- **Do NOT include NIS** (circular — NIS is our own constructed signal)
- SPY 50d/200d MA state can substitute breadth if breadth unavailable

**Model: logistic regression with L2, NOT a decision tree.**
Reasoning: ~15 years × 252 days = 3,800 obs, but effective ~40 regime episodes (autocorrelated).
A decision tree with 3 splits memorizes 8 leaves on 40 episodes.

**Better alternative:** continuous exposure scalar with hand-set monotonic coefficients:
```python
exposure = sigmoid(-2 * z_dispersion - 1.5 * z_vixts + 0.5 * z_breadth - 1.0 * z_credit_spread)
```
Tune ~3 coefficients only. Enforce sign constraints — reject any fit that violates direction.

**Train/freeze split:** 2007-2017 train, validate 2018-2020 (includes COVID), freeze 2021-2026.
(Not 2007-2019 freeze: COVID + 2022 rate shock are the most informative regime episodes.)

**Wiring:** Weekly rebalance of target gross exposure. Cap day-over-day change at 5pp.
Daily application via position sizing multiplier into PM.

**Overfitting safeguards:**
- Coefficient sign constraints enforced
- Max 4 features — period
- Sensitivity: drop one year, refit, compare paths — if 2020 exposure changes > 20% when dropping
  2008, model is unstable
- Smoothness penalty on day-over-day exposure changes

---

### Phase I — Retry with Allocator + Label Fix

After G-Pre.2 fixes labels and Phase H builds allocator:
- Re-run Factor Portfolio WF with new labels + allocator
- Re-run L/S WF with new labels + allocator
- Gate: avg Sharpe ≥ 0.80, min fold ≥ -0.30

---

### Phase J — PEAD Retry with Allocator

After Phase I passes (or independently, same allocator):
- PEAD had avg 0.346 — failed because Fold 4 drowned event signal
- With allocator reducing gross exposure in Fold 4, signal may survive

---

### Required WF Output Additions (apply to all future runs)

Every WF run must output these (currently missing):

1. **P&L attribution per fold:** `total = alpha - costs - stop_losses + carry`
2. **Transaction cost sensitivity:** run 5bps, 10bps, 15bps variants. If Sharpe collapses at 15bps → edge is fictitious.
3. **Bootstrap CIs on fold Sharpe:** with 5 folds × 250 days, Sharpe stderr ≈ ±0.4. Report 95% CI.
4. **Null-strategy distribution:** 100 random-pick portfolios (random N stocks, equal-weight) through same WF. Factor model must beat 80th percentile.
5. **Fold 4 SPY benchmark:** always report SPY Sharpe in Fold 4 alongside strategy Sharpe.

---

### Kill Criterion (Pre-registered 2026-05-22 BEFORE running model WF)

> If, after model-based WF with per-fold artifacts and verified feature parity:
> - Mean WF Sharpe < 0.40 AND
> - Mean WF Sharpe < (EW-all-11-SPDR mean Sharpe − 0.20) AND
> - At least 2 of 5 folds are negative
>
> Then: **STOP single-name picker development**. Pivot to ETF-level strategy
> (EW-all-11 base + drawdown overlay + regime-conditional cash tilt).
> Archive swing_v* model family with post-mortem in ML_EXPERIMENT_LOG.
>
> If 0.40 <= Sharpe < 0.80: continue with label/feature work, cap at 4 more weeks
> before second go/no-go.
>
> If Sharpe >= 0.80 with passing diagnostics: paper trading prep.

---

### Phase K — Paper Trading Gate (3 months)

Same criteria as before. Note: 3 months has stderr ~0.5 on Sharpe — treat as smoke test, not
statistical validation. Be explicit about this in reporting.

---

## Historical Phases (Archive)

Earlier phases (A diagnostics, B build, C LambdaRank campaign, R regime model, WF-A alignment, Phases 1–5 statistical truth + simulation realism, Phase 6 live readiness, data tasks D0–D5) are preserved in:

- `docs/archive/phase-specs/phases_archive.md` — full completed phase history
- `docs/living/ML_EXPERIMENT_LOG.md` — every retrain + WF result
- `docs/archive/ml-history/ML_EXPERIMENT_LOG_archive.md` — older campaigns

**Key historical context:**
- Phase A diagnostics (2026-05-13): naive baseline +0.808 beat best ML +0.106 → confirmed ML was destroying alpha vs trivial SPY timing
- Phase C LambdaRank (9 runs, closed 2026-05-18): structural Fold 2 (bear) collapse — campaign closed
- Phase D factor portfolio (2026-05-18): integrated and paper-running; will be the long sleeve of Phase F L/S
- WF-A1/2/3 (PRs #198/199/200): TS-norm parity, PIT universe, survivorship — all merged
- Vol targeting (3d), live-vs-sim harness (2d), opportunity score (5b) — all merged, feature-flagged where appropriate

---

## Walk-Forward Gate (canonical)

```
avg Sharpe (NET of 5–15 bps costs) >= 0.80
min fold Sharpe                    >= -0.30
max drawdown                       <= 15%
profit factor                      >= 1.2
trade count per fold               >= 100
deflated Sharpe                    >  0 (p < 0.05)
IC vs forward 10d returns          >= 0.02 (t-stat >= 2.0)
opportunity / regime gate applied during simulation
PIT universe (`pit_union` with delisted seed)
```

---

## Training / WF Command Reference

```bash
# Phase E scripts (ready to run)
python scripts/compute_factor_ic.py --years 6 --horizon 10
python scripts/audit_survivorship.py --universe russell1000

# Phase F (after F.1–F.5 implemented)
python scripts/walkforward_tier3.py --model swing --strategy ls_factor \
    --net-long 0.40 --top-long 20 --top-short 15

# Phase G (after F passes)
python scripts/run_pead_walkforward.py --years 6 --folds 5
```
