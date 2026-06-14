# MrTrader — Project State

**One-screen view of what's happening right now. Updated at session start/end when focus changes.**

> **Update rule:** Human updates this at session boundaries. Keep it to one screen. This is NOT a planning doc (that's MASTER_BACKLOG.md) and NOT a history doc (that's ML_EXPERIMENT_LOG.md). It answers: "If I open the laptop cold, what do I need to know in 30 seconds?"

**Last updated:** 2026-06-12 (**ALPHA-v7 Phase B / Ruler v2 — Phase 1 (PR #471) + Phase 2 (PR #472) LANDED, both DARK. Phase 2 = `bayes_sr.py` (Bayesian posterior P(SR>0), replaces saturated DSR) + `ruler_v2.py` (two-tier gate) + `CPCVResult.oos_returns_dated` + `GATE_MODE="ruler_v2"` dispatch; legacy gates byte-for-byte untouched (89 tests). Opus deep-dive caught a CRITICAL: CAPITAL was "unreachable on backtest alone" only by threshold luck → made live-paper a STRUCTURAL gating criterion (posterior = P(SR>0 | backtest AND live paper)). No live behavior change (flag not flipped; owner OD-1…OD-9 sign-off pending). Earlier today: H1 RUN → PEAD DEMOTED at event level (p=0.78). ✅ PEAD FLIPPED OFF LIVE + uvicorn restarted → live book = trend-only (25%) + cash. P0+P1c+P2+P3-H1 shipped (#454/#455/#456) + P4a options feature table + H4a–H4e pre-registered. P4 H4a–H4e → ALL 5 KILL; H2 NOT_CONFIRMED (OPT-5 parked); H3 BLOCKED (revision data). All Alpha-v6 hypotheses adjudicated (P5 PARK). NEW DIRECTION: ALPHA-v7 — operate a premia book (`docs/reference/ALPHA_V7_SYNTHESIS_AND_PLAN.md`; Phase B design = `docs/reference/RULER_V2_DESIGN.md`). Live book unchanged.**)

## 🧭 NOW (2026-06-12): Alpha-v7 Phase B — Ruler v2 Phase 1 + Phase 2 (gate wired, DARK) landed

**Phase 2 (PR #472, dark) wires the inference core into a two-tier gate behind `GATE_MODE="ruler_v2"`** (not flipped; legacy gates untouched, 89 tests green). `bayes_sr.py` = the closed-form Bayesian posterior `P(SR>0)` that replaces the saturated DSR (registry trial-count shrinks a mean-zero prior; backtest + live-paper combine precision-weighted). `ruler_v2.py` = PAPER (plausibility: point-SR floor + implausibility ceiling + non-catastrophic regime + sleeve cap; PF/Calmar demoted to report-only) vs CAPITAL (significance: Bayesian posterior ≥0.95 + multi-factor residual-α t + bootstrap + PBO-if-M>1 + hard power floor). `CPCVResult.oos_returns_dated` (pure-additive) carries the dated OOS book series the gate runs inference on. **The independent Opus deep-dive caught a CRITICAL** — CAPITAL was "unreachable on a backtest alone" only by threshold luck (a clean t≈3 backtest cleared 0.95 ~1-in-5 draws); fixed by making **live-paper a STRUCTURAL gating criterion** (capital ⇒ posterior must include a live-paper observation), plus the missing `run_cpcv` population integration test. **Phase 3 (PR #473, dark) — Track-B v2** (`scripts/walkforward/track_b_appraisal.py`, behind `TRACKB_MODE="ruler_v2"`): replaces the budget-DEPENDENT legacy ΔSharpe≥0.10 bar (which secretly encoded a position size) with the **budget-invariant appraisal ratio** (residual-α IR via `multifactor_alpha(cand_vt, factors={base})`) + **block-bootstrap P(ΔSR>0)**. Worst-regime floor waived for declared diversifiers/risk-premia; fails closed for other components lacking regime data (unless explicitly waived). Opus deep-dive verified budget-invariance empirically + ruled out junk-sleeve gaming (2 MINOR fixed). Legacy `book_delta_gate` untouched. **2026-06-13 — OD decisions ratified (#475) + R4 calibration run → NOT CLEAN → DO NOT FLIP.** Ratified OD-1…OD-9 (independent Opus advisory): R7 trial-count fix, OD-9 premia factor policy (trend excludes TSMOM), OD-5 PdSR→0.90, OD-2 documented. Built the R4 pre-flip instrument (report-only `rv2_*` columns in `gate_calibration.py`, #476) and ran the decisive real controls. **R4 NOT CLEAN** (artifact `logs/gate_calibration_20260613.json`; independent Opus-reviewed): (A) the tsmom positives failed Ruler-v2 PAPER on the regime backstop — a RUN ARTIFACT (a crisis-diversifier scored through Track-A instead of Track-B + thin stress-regime coverage), and (B) a TRUE null (random_balanced_seed_5) PASSED Ruler-v2 PAPER — a REAL Type-I leak (the 0.30 plausibility floor admits ~23% of zero-edge nulls at n≈1500). **GATE_MODE/TRACKB_MODE NOT flipped** (the validation gate correctly blocked the flip). **2026-06-13 (later) — remediation (1)+(2) DONE → R4 now CLEAN** (DECISIONS 2026-06-13 later): (1) ruler_v2 PAPER waives the worst-regime backstop for declared diversifiers/risk-premia (`CPCVResult.component_type`; tsmom→`risk_premium`) → both tsmom positives now PAPER-pass; (2) ruler_v2 PAPER adds a light one-sided HAC-SR significance floor (`RULERV2_PAPER_MAX_HAC_P=0.05`, pooled-OOS instrument) → all 5 true-nulls now PAPER-fail incl. the lucky `seed_5` (leak closed; MC 5.7% null-pass vs 23.4%). Opus deep-dive=SHIP. **R4 RULER-v2 CHECK = CLEAN** on the decisive controls (positives clear, nulls dead, leaky rejected). **2026-06-13 (full set) — ran pead_baseline + xmom_12_1: R4 NOT strictly clean — `xmom_12_1` (labeled positive) FAILED ruler_v2 PAPER.** On the merits this is a CORRECT rejection (xmom 12-1 is genuinely insignificant now — meanSR 0.17, t 0.77; the legacy gate fails it too; cross-sectional momentum was ruled dead 2026-06-03). **DID NOT flip.** A drafted R4 carve-out that would have made it read CLEAN was caught by an independent Opus methodology review as GOALPOST-MOVING (post-hoc, single-control-targeted, and used the discredited path-t to define "significant") → **reverted**. The flip is back to an explicit OWNER decision (DECISIONS 2026-06-13 full-set): (a) accept xmom as a documented correct-reject + flip; (b) reclassify xmom positive_alpha→known_marginal as a dated pre-stated amendment, then flip; (c) hold dark. Ruler v2 stays DARK; strict R4 unchanged on main; live book unchanged (trend-only 25% + cash).

**Phase 5 (PR #474, REPORT-ONLY) — kill-ledger re-score** (`scripts/walkforward/ruler_v2_rescore.py`): tabulates the PAPER-tier flip (REVIVED / DEMOTED / unchanged) of each CPCVResult under Ruler v2 vs its recorded significance verdict — the question Alpha-v7 exists to answer (which significance-kills a less-Type-II plausibility ruler would revive). HARD CONTRACT: never flips a gate, never writes the registry, never promotes — a REVIVED row is a CANDIDATE for an owner-initiated, R4-logged re-test, not an action. Opus deep-dive caught + fixed a CRITICAL (the re-score mutated the caller's result + a swallowed-exception spurious-REVIVED path). **Phase B build is COMPLETE (Phases 1-3+5 all merged, all DARK/report-only).** **Remaining = owner calls: (a) OD-1…OD-9 sign-off, (b) the live `GATE_MODE="ruler_v2"` / `TRACKB_MODE="ruler_v2"` flip, (c) reviewing any REVIVED rows from running the re-score on the real ledger.** Live book unchanged (trend-only 25% + cash).

---

**Phase 1 (PR #471, dark) — the pure inference keystone.** `app/research/inference.py` is the pure inference keystone (numpy/scipy in, frozen dataclasses out, no I/O, no gate wiring): `hac_sharpe` (autocorrelation-robust SR>0 significance, Lo-2002 mean-HAC t), `stationary_bootstrap_sr` (Politis-Romano P(SR>0)+CI, data-driven block length), `pbo_cscv` (Bailey/LdP Probability of Backtest Overfitting via CSCV), and the canonical `multifactor_alpha` (moved here from `options_xs_ls.py`, which now re-exports it). Built Opus 4.8 → **two independent Opus deep-dives**: round 1 fixed the SE labeling (`se_sr_ann_implied` — it's the mean-HAC t, not Lo's full SR SE) + integrated-autocorrelation-time block length + a flaky null test; round 2 caught three contract/bias issues feeding a future capital gate — `multifactor_alpha` raised on an inf factor (`.dropna()` doesn't drop ±inf), PBO silently corrupted to a finite gating number on a NaN perf cell, and PBO's `<=` rank biased PBO DOWN (optimistic) under ties — all fixed (finite-mask, fail-closed, tie-fair mid-rank). 22 known-answer tests; round-2 verification = **SHIP**. **Next: Phase 2** — persist `CPCVResult.oos_returns_dated` + `ruler_v2.py` + `bayes_sr.py` behind a `GATE_MODE="ruler_v2"` flag (coexists with legacy; zero diffs at default). Owner decisions OD-1…OD-9 needed at Phase 2/3, NOT Phase 1.

## 🧭 EARLIER TODAY (2026-06-12): H1 verdict in — PEAD is NOT an event-level edge → book = trend-plus-cash

**The centerpiece answered the quarter's most important question.** H1 (`H1-PEAD-EVENTLEVEL-20260611`, one-shot R4, `panel_sha256=af206149…`) re-adjudicated the LIVE PEAD edge at the EVENT level on a 21,330-event / 9,774-qualified R1K panel (2019→2026) with two-way (announce_date×firm) CGM-clustered SEs (validated to the published Petersen-2009 pins): **PRIMARY 10d SPY-hedged mean −8.3bp, t=−0.77, one-sided p=0.7804 → DEMOTE.** Negative at every horizon; conservative bootstrap p=0.66; robust to all LOCO; the announce+1-vs-+2 gap is +2.6bp (negative even at the favorable entry). **PEAD's case for capital is closed** (corroborates Alpha-v4 Phase-1's CAPM hedged-Sharpe −0.37). The inference-upgrade success metric is met: no strategy is killed/kept on an 8-fold t-stat again.

**▶️ OWNER ACTIONS — status (live config verified 2026-06-12):**
1. ✅ **DONE — live PEAD sleeve flipped OFF** (overnight; notify id 59). Mechanism: `pm.swing_selector` `'pead'`→`'ml_model'` (+ `pm.swing_ml_live_enabled='false'`) → the swing/PEAD proposing path runs the dormant dead-ML branch and fires nothing; PEAD no longer trades. `pm.pead_size_mult` stays 1.0 (moot — PEAD isn't the selector; the pead_tracker telemetry is retained). **Live book = trend-only (`trend_enabled='true'`, `trend_shadow='false'`, 40%) + cash.**
2. ✅ **DONE (2026-06-12) — live trend weight reconciled 40%→25%** (`pm.trend_allocation_pct` 0.4→**0.25**; Track-B 25% framing, #451). Trend is the sole live sleeve; the new 25% gross applies from Monday's first real rebalance. Schema default + `set_trend_config` + docstrings brought in line. (`pm.allocator_enabled='false'`.)
3. ✅ **DONE — uvicorn restarted** (2026-06-12 07:52) → notify_watcher live (drained the queue incl. the P4a email), clean logs + FMP calendar (#445/#446), and the 15:55 ET NBBO snapshot logger (#455) active.
4. **H2/H3 are NOT auto-run** — they were PEAD-improvement hypotheses; with PEAD demoted, whether to still run them as pure research (no live capital) is your call. The event-panel + CGM instrument is the standing tool for any future event hypothesis.

**Also shipped 2026-06-12 — Alpha-v6 P4a (options-as-signal DATA LAYER, no verdict yet):** the daily options **feature table** (`app/data/options_features.py` + `scripts/build_options_features.py` → `data/options_features.parquet`: CPIV / 25Δ-skew / term-slope / IV-RV / O-S volume, PIT, holiday-aware knowable_date + split-adjusted RV) + the **options-quality universe filter** (`app/data/options_quality.py`), and **H4a–H4e PRE-REGISTERED** (`preregistered_at=2026-06-12T12:00Z`; kill = simple decile sorts net-of-costs show nothing → CLOSE, NOT an XS-ML revival). Built Opus 4.8 → independent Opus 4.8 deep-dive (**2 PIT/split BLOCKERs caught + fixed before any run**) → Opus verification = SHIP; 47 tests.

**Also shipped 2026-06-12 — Alpha-v6 P4 H4a–H4e VERDICTS → ❌ ALL 5 KILL (options-as-signal equity edge DEAD):** the five confirmatory R4 runs (`scripts/run_options_xs_cpcv.py`, full 4y/208wk decile L/S) all KILL — H4a CPIV t=−2.70, H4b skew −4.10, H4c put-O/S −4.43, H4d term-slope −2.83 (all significantly NEGATIVE — academic signs don't hold 2022–26 at equity cost), H4e IV/RV −0.12 (noise). The inverse is NOT tradeable (post-hoc sign-mining; Opus deep-dive = 2022–23 growth-crash regime). Independent Opus 4.8 deep-dive confirmed NO look-ahead + fixed a decision-label BLOCKER and the monotonicity Type-II trap before recording; 29 tests. **Options confirmed (again) as a data asset, not a tradeable equity signal; live book unchanged (trend-only 25% + cash).**

**Also shipped 2026-06-12 — H2/H3 (event-conditioned) adjudicated:** built the **event-time options join** (`scripts/enrich_event_panel_options.py` + `app/research/event_options_join.py`) populating the event panel's options-pre-event features (PIT, knowable_date-gated; ~45% coverage; Opus deep-dive SAFE-TO-RECORD). **H2 → NOT_CONFIRMED** (reaction_ratio continuous: 10d t=−1.21, no significant edge → **OPT-5 settled, stays PARKED**, no threshold-hunting). **H3 → BLOCKED** (its frozen feature list needs `revision_momentum` = forward-estimate-revision data, which is DATA-BLOCKED; the runner refuses to consume the one-shot — H3's pre-registration stands for if that data is ever acquired). **All Alpha-v6 confirmatory hypotheses (H1, H2, H3, H4a–e) are now adjudicated.** Live book unchanged.

**Also shipped this session (all CI-merged):** P0 stubs + H1/H2/H3 pre-registration (#454); both slow fuses (#455) — the **Alpaca NBBO logger** (Polygon serves no options NBBO → free Alpaca feed; day-1 captured) and the **computed-greeks store COMPLETE** (`data/options_greeks/`, 733/733, 112.8M rows). Process held throughout: Fable-5 build → independent Fable-5 adversarial review → fix loop; **3 BLOCKERs caught before merge/run** (event-regime gate units, yfinance split-adjustment, empty-pre crash).

## 🧭 PRIOR (2026-06-11): Alpha-v6 Phase-0 machinery COMPLETE — lead with P3 (earnings-event panel / H1)

**Shipped overnight (8 PRs, all CI-merged; main @ `ab8151c`):** the P0 *measurement* machinery is built + validated end-to-end —
- **Gate-calibration harness** (#444) + result (#447): the gate's false-negative is the **worst-regime backstop, NOT the t-stat**. "Lower the t-bar" is empirically **REFUTED** — 3/5 TRUE zero-SR nulls clear t≥2.0, and PEAD's t=3.33 ≈ a noise null's 3.47 (the 8-fold path-t can't separate PEAD from noise).
- **Two-track Track B book-delta gate** (#448) — judges diversifiers/risk-premia on book contribution, not the standalone floor.
- **Research registry / pre-registration ledger** (#449) — the program's true N_TRIALS (DSR demoted to report-only).
- **First real Track B run** (#450) + **owner-approved REGISTERED budget amendment 10%→25%** (#451) → **TSMOM PASSES Track B** (Sharpe 0.41→0.64; sweep crossover ~12.5%, so 25% is in the pass region with margin). TSMOM is now Track-B-validated (trend is already live at 40% capital; reconciling the live weight to the 25% *risk* framing is a SEPARATE owner call).
- **Ops:** uvicorn `[uvicorn.error]`→`[uvicorn]` (#445); FMP econ-calendar fixed (#446 — it was a **dead legacy `/api/v3`**, NOT a paywall; migrated to `/stable/`, works on the $29 plan) + new `docs/reference/DATA_PROVIDERS.md`; stale `risk.peak_equity` $50k→$101k reset.

**▶️ NEXT DIRECTION (Fable 5 recommendation, ADOPTED): LEAD WITH P3 — the earnings-event panel — breaking the blueprint's "P3-after-P2" dependency.** Why: P0 proved the path-t can't separate PEAD from noise, and **PEAD is live capital on that broken instrument**; event-level inference (H1) is the replacement and needs **ZERO options features** → it is NOT blocked on P2.
- **✅ DONE 2026-06-11 (PR #454) — P0 stubs (1)+(2) + pre-registration (3 below):** `--hypothesis-id` registry enforcement wired into all 9 `run_*_cpcv` scripts (shared `registry_enforcement.py`; warn-only until **2026-06-25** then required, or `--exploratory`); `event_regime_sharpes()` built + surfaced as `CPCVResult.event_worst_regime_sharpe` — **REPORT-ONLY (does NOT gate; waiver retirement deferred to H1/P3** — a per-event Sharpe isn't unit-comparable to the annualized-daily −0.5 floor, and retiring the waiver is H1's pre-registered consequence; the Fable-5 review caught + I reverted a first-cut gate-fallback). **H1/H2/H3 PRE-REGISTERED** (frozen criteria, `preregistered_at=2026-06-11T12:00Z`): `H1-PEAD-EVENTLEVEL-20260611`, `H2-IMPLIEDMOVE-CONTINUOUS-20260611`, `H3-PEADV2-SCORECARD-20260611`. See DECISIONS / ML_EXPERIMENT_LOG / PIPELINE_ARCHITECTURE §7.0c (2026-06-11).
- **✅ DONE 2026-06-11 (PR #455) — both slow fuses LIT:** (P1c) `scripts/log_options_nbbo.py` — nightly 15:55 ET NBBO snapshot logger via **Alpaca** (`feed=indicative`, free) because the **Polygon $79 plan serves NO options NBBO** (verified: `/v3/quotes` → 403; snapshot has no `last_quote`/IV/greeks). ⚠️ **The recurring schedule activates only on a uvicorn restart** (owner) — until then it can be run manually. (P2) `scripts/backfill_computed_greeks.py` — STARTED (background) → `data/options_greeks/`; European-warm-start American refine (~0.6h; naive was ~122h — a put-IV CRR-degenerate landmine). Inputs are **as-traded** (Polygon `adjusted=false` closes — yfinance is split-adjusted, a BLOCKER the Fable-5 review caught: ~2.9M rows would have had S wrong by the split ratio). 2× Fable-5 review (1 BLOCKER + 6 fixes). Fitter/CalibratedSpreadModel deferred (~4-6 wks of NBBO needed).
- **▶️ NEXT (this session):** (PR3) build **`app/research/event_panel.py`** (equity cols, 2019→2026) + **`scripts/walkforward/event_inference.py`** (two-way cluster-robust OLS, Cameron-Gelbach-Miller, unit-tested vs a published reference) → **run H1** under `H1-PEAD-EVENTLEVEL-20260611`. Pre-committed acceptance: p<0.05 → PEAD graduates to an honest Track-A paper pass (no waiver); p>0.15 → live book = trend-plus-cash (trend is the capital base either way). H1 may well **demote** PEAD (Phase-1 CAPM had hedged Sharpe −0.37) — that's the decision the live book needs.

**Process (standing):** Fable 5 (model:fable) for design/analysis/implementation → INDEPENDENT Fable-5 deep-dive for bugs → fix + re-review until perfect → tests → NO-DRIFT docs → email phase_complete → CI-gated merge. One branch/PR at a time, squash-merge, delete branch, sync main.

**Guardrails (what NOT to do):** don't touch significance thresholds (refuted); no new Track-B sleeves / index-VRP re-test on the TSMOM-pass high (preconditions unmet + cooling-off); no more binary threshold sweeps (the OPT-5 trap) — the `run_pead_implied_threshold_sweep.py` tooling was REMOVED (#452; H2's continuous pre-registered coefficient is the sanctioned settlement; the OPT-5 FRAGILE verdict stays logged). Don't start P5 trend-broadening until Monday's first real rebalance + the trend replay-diff are boring.

**Live ops:** first REAL trend rebalance **Mon 2026-06-15 09:45 ET** (unchanged). ⚠️ **Restart uvicorn** to pick up #445/#446 (cleaner logs + the working FMP calendar). SSOT for the full plan: [NEXT_PHASE_BLUEPRINT_2026-06.md](../reference/NEXT_PHASE_BLUEPRINT_2026-06.md).

---

## 🧭 PRIOR (2026-06-10): operating the live book; options = data asset (not a sleeve); external review out
- **Live book unchanged = PEAD (telemetry size) + TSMOM trend.** First REAL trend paper rebalance still **Mon 2026-06-15 09:45 ET** — verified READY (read-only shadow sim) and reliability-hardened (per-order commit + 1800s weekly misfire grace, #430). Today is quiet/no-trades and that's correct: swing ML ranker dormant by design (`pm.swing_ml_live_enabled=false`), PEAD found 0 qualifying earnings signals, trend not due until Monday.
- **OPT-5 implied-move filter → ❌ FRAGILE / PARKED (#433).** The threshold-robustness sweep showed the PEAD lift exists only at ratio=1.0 and *inverts* at 1.25 → overfit-suspect. Don't pursue without a powered + pre-registered re-test. Broader options program stays PAUSED (short-vol = risk premium, not alpha).
- **Options data → full 4y local store acquired (#440).** 2022-06-09 → 2026-06-08, ~112.8M bars, 733 names, ~6.18M contracts — the max the Polygon Developer plan serves. We now own the complete copy **even if the subscription is cancelled.**
- **Reliability / test-infra hardening (this session):** kill-switch strict-bool restore (#434); test→prod log/DB bleed closed via one shared `is_test_mode()` (#435/#436); weekly regime retrain + fixed 3-class gate revived from an abandoned PR (#439); Finnhub+FMP economic-calendar 403 log-spam silenced (#437/#441 — the calendar needs a paid tier on BOTH providers; falls back to FRED macro + hardcoded FOMC/NFP).
- **External quant review prepped:** `docs/reference/prompts/EXTERNAL_QUANT_REVIEW_PROMPT.md` + the `20260610_Quant_Options_Review/` kit — soliciting world-class-quant feedback on the next phase across multiple LLMs (harness soundness, options models, architecture gaps).
- **✅ DIRECTION CHOSEN → ALPHA-v6** (SSOT: [NEXT_PHASE_BLUEPRINT_2026-06.md](../reference/NEXT_PHASE_BLUEPRINT_2026-06.md)). The 5-LLM synthesis (Gemini/DeepSeek/Grok/ChatGPT/Claude, deep-dived by Fable 5, code-grounded) converged: our harness over-corrected into a **Type-II / false-negative machine** — a t≥2.0 gate on ~8 folds of ≤4y data kills *true* Sharpe-0.5–0.7 edges (t≈SR·√years), so 100% KILL (incl. confirmed-real index VRP) is a miscalibrated ruler, not an empty opportunity set. **7-phase plan:** **P0** calibrate the ruler (gate positive/negative controls — TSMOM-on-4y decisive — + two-track acceptance [alpha vs book-delta] + research registry) → **P1** live-book fidelity (replay-diff / fill-quality / NBBO spread-calibration) ∥ **P2** options *feature* layer (persisted greeks / surface-quality / BMO-AMC snapshots) → **P3 (centerpiece)** earnings-event panel + event-LEVEL PEAD inference + PEAD v2 continuous score → **P4** options-as-equity-signal XS sleeve (CPIV/skew/O-S/term-slope) → **P5** trend broadening (19y) → **P6** index-VRP micro-sleeve behind the book gate. **Options = signal-first, not execution.** Reviews archived under `docs/reference/prompts/20260610_Quant_Options_Review/responses/`.
- **▶️ P0 STARTED — gate-calibration harness SHIPPED (PR #444):** `scripts/walkforward/gate_calibration.py` scores positive (`tsmom_4y` decisive, `tsmom_19y`, `xmom_12_1`, `pead_baseline`, `spy_buyhold`) + negative (balanced/beta nulls, `leaky_tplus1`) controls through the PRODUCTION gate to MEASURE its false-negative/false-positive rates. Built + 2× Fable-5 adversarial review (1 BLOCKER + 3 MAJOR fixed), 49 tests, changes no threshold. **✅ RESULT (full 16-control run): the gate IS a false-negative machine for our edges (0/4 positives pass PAPER) — but the t-stat is NOT the cause.** `tsmom_4y` posts t=6.72 and fails ONLY on the `worst_regime_sharpe` backstop; meanwhile 3/5 TRUE zero-SR nulls clear t≥2.0 (so lowering t* admits noise → pre-registered rule verdict `NO_ADMISSIBLE_TSTAR`). PEAD's t=3.33 ≈ a noise null. **✅ Two-track acceptance gate (Track B book-delta) SHIPPED (#448)** — `book_gate.py` judges diversifiers/risk-premia on book contribution (not the worst-regime floor); 8 pre-registered criteria incl. the registered tail-overlap test; 2× Fable-5 review (3 MAJOR fixed). **✅ Research registry (pre-registration ledger) SHIPPED (#449)** — `app/research/registry.py` enforces pre-registration integrity (the program's true N_TRIALS; DSR is report-only); 2× Fable-5 review (1 BLOCKER + 3 MAJOR fixed), 45 tests. **✅ FIRST Track B run + registered amendment done (#450, #451): TSMOM now PASSES Track B.** First run at the 10% budget FAILED only on ΔSharpe (the bar structurally rejects any diversifier at that thin slice). Owner-approved REGISTERED amendment raised the risk budget 10%→25%; re-run → **TSMOM PASSES all 8 criteria (Sharpe 0.41→0.64, Calmar 0.28→0.59, maxDD shallower)**; budget sweep crossover ~12.5%. TSMOM is now **Track-B-eligible for paper book inclusion** (actually wiring it into the live book at a 25% weight is a SEPARATE owner call). **NEXT P0: wire `--hypothesis-id` into the run scripts + `event_regime_sharpes()`; then bring forward event-level inference (P3).** See DECISIONS 2026-06-11.
- **Operate Monday's first real trend rebalance** (still Mon 2026-06-15 09:45 ET) in parallel.

---

## 🧭 PRIOR (2026-06-09): options program PAUSED; operating the validated live book
- **Live book = PEAD (telemetry size) + TSMOM trend.** The **dead cross-sectional swing ML ranker is now OFF in the live path** (`pm.swing_ml_live_enabled=false`) — it was silently producing ~30/32 recent trades of a validated-null strategy (#418). The Trader's ML-score gate is now observable (`REJECTED_ML_SCORE`).
- **Trend armed for paper:** `pm.trend_enabled=true`, `pm.trend_shadow=false` → first REAL paper rebalance **Mon 2026-06-15 09:45 ET** (weekly cadence, ~7 ETFs, 40% alloc), via the orchestrator scheduler.
- **PEAD** clears the 0.55 entry gate (confidence 0.65–0.90); event-sparse (fires on earnings). Live-sizing fidelity fixed (sizes off PEAD's own ATR stop, not the swing stop).
- **Alpha-v5 Options Program (OPT-0..4): built, validated, PAUSED.** Two Opus-certified verdicts: single-name earnings IV-crush = KILL (cost-killed); index short-vol = KILL standalone but **VRP real + cost-robust** (PF 2.24/1.75), just Sharpe-weak. Key insight: our gate is an *alpha* gate, short-vol is a *risk premium* (wrong ruler). Harness retained; revisit later as a book diversifier with a risk-premium framework + more data. See DECISIONS 2026-06-09.
- **Next:** OPT-5 (options-data-as-signal: implied-move filter for PEAD + put-skew risk-off, judged on the host sleeve's gate — sanctioned parting options win, data kept) + continued live-book hardening. ⚠️ Recurring CI flake (`test_agent_simulator_rebalance.py` DB-lock) still costs a rerun on most PRs.

---

## 🧭 P3 (2026-06-08): Live regime-aware sleeve allocator — wired, shipped DISABLED
`app/live_trading/sleeve_allocator_live.py` turns the backtest allocator into a live,
kill-switchable book layer. **Ships `pm.allocator_enabled=false` → byte-identical to
today's static budgets** (trend 0.40 / PEAD telemetry). When enabled it recomputes weekly
(before the trend rebalance), persists effective sleeve weights to agent_config, and both
sleeves read them with a fixed-weight fallback (disabled/stale/warmup/error → static).
Default scheme `equal` (the validated winner); `vol`/`regime` are live-capable but OFF
until `scripts/run_book_allocator.py --emit-config` selects them (expected after the 3rd
sleeve). PEAD regime double-tilt guarded (Opus review). To enable later:
`python -m scripts.set_allocator_config --enable`. **NEXT: the 3rd uncorrelated sleeve**,
then re-run the gate to (likely) activate vol/regime.


---

## 🧭 ACTIVE DIRECTION (2026-06-06): Alpha-v4 — Portfolio of Uncorrelated Premia
**SSOT:** `docs/reference/QUANT_REVIEW_SYNTHESIS_2026-06.md` · **Index:** `MASTER_BACKLOG.md` (Alpha-v4 table).

Five independent world-class-quant LLM reviews (ChatGPT, Gemini, Grok, DeepSeek, Claude) converged: **the architecture is good — re-aim, don't rebuild.** The wall is the *opportunity set + a biased ruler*, not technique. Stop hunting one hero edge; assemble **~4 uncorrelated sleeves → book SR ≈ 0.8**, on an honest harness. PEAD is **not** proven alpha (p≈0.19, 87% P&L in up-trends = conditional beta) — keep small, pair with a crisis-positive sleeve.

**Locked (2026-06-06):** ① **PEAD → telemetry** (`pm.pead_size_mult` 3.0→**1.0**, `pm.pead_max_position_pct` 0.10→**0.05**, applied live restart-free — reverses B4). ② **Gates** → lower bar (~0.45) + reweight to robustness (residual-α-t + fold-consistency primary; keep survivability floor). ③ **Targeted re-arch** (keep execution; rework research harness; add sleeve+regime-allocator). ④ **Free-data only** until a spike justifies a paid purchase.

**Phases (EV/effort):** **P0** validation integrity (`is_trained` guard → full-coverage CPCV for rules-based; sequential-WF baseline; fold-coverage report; gate recalibration; freeze dead XS-ML) → **P1** PEAD reckoning (neutralization + FF5 attribution + gapper slippage; decision gate) → **P2** Trend/TSMOM ETF sleeve (crisis-diversifier, book-level eval) → **P3** regime-aware allocator (the unlock; must beat static weights net of turnover) → **P4** gated high-ceiling bets (PEAD 2.0 · options-VRP spike · squeeze-conditioning). **Regime policy:** "attribute, don't amputate" — switch *allocation* across full-history sleeves, never train a model per regime (§4b).

**PHASE 0.1 + PHASE 1 DONE (2026-06-06):** `is_trained` guard fix shipped (#394) → PEAD CPCV now full-coverage (skips 50%→8). On the honest harness PEAD is **NOT a standalone edge**: unbiased CPCV mean +0.578 but t=1.81 (<2)/p5=−0.796 (worse tail revealed); **CAPM beta-isolation is decisive — α=−1.29%/yr, HAC α t=−0.95, β=0.14; the beta-removed (market-hedged) Sharpe is −0.37** (hedge out SPY and it loses money). The positive Sharpe is small β riding the 2020–26 bull. Gapper-slippage collapses the edge by 50bps. → **PEAD = weak market-beta-driven risk-on satellite, kept at telemetry size (1.0×/5%, live), never a centerpiece.** Retires "PEAD is the sole edge." PEAD 2.0 dropped (was gated on neutralized PEAD showing life). See ML_EXPERIMENT_LOG Phase 1. Tool: `scripts/pead_phase1_attribution.py`.

**PHASE 2 DONE (2026-06-06) — TSMOM trend sleeve ✅ KEEP.** Built a vectorized PIT-safe TSMOM sleeve (`app/strategy/tsmom.py`, 10-ETF multi-asset, long-flat, inverse-vol, weekly) + validation (`scripts/run_tsmom.py`). Independent deep-dive: no look-ahead; fixed a cost-timing off-by-one. **Standalone Sharpe +0.714** (2007-26, incl. all crises), crisis-positive in *slow* bears (2008 +7% vs SPY −37%) but whipsawed in *fast* shocks (COVID −6%). **corr to PEAD +0.25** (diversifying). **Exit gate PASSES:** PEAD+trend (50/50, vol-matched) beats PEAD-alone on Sharpe (0.31→0.92) AND drawdown (−13.7%→−8.3%). Trend is the stronger sleeve; PEAD the weak satellite. See ML_EXPERIMENT_LOG Phase 2.

**PHASE 3 DONE (2026-06-06) — sleeve allocator built; SHIP SIMPLE FIXED-WEIGHT.** Built `app/strategy/sleeve_allocator.py` (general N-sleeve: equal/vol/regime, inverse-vol risk parity, regime hysteresis + EWMA blend, PIT) + `scripts/run_book_allocator.py`. Independent deep-dive CLEAN (no look-ahead). On the PEAD+trend overlap (2020-26): **equal-capital Sharpe +1.082 / maxDD −5.1% / 0× turnover** beats static vol-weight (+0.715) beats regime-tilt (+0.593 / −6.4% / 2.6×). **Regime tilt FAILS its margin** (worse on every metric) → regime layer OFF. **Inverse-vol is fooled by PEAD's sparse-low vol** (over-weights the weak sleeve) → loses to naive equal-capital. → **ship simple fixed-weight book; vol+regime layers kept as OFF scaffold; revisit with more sleeves / longer overlap.** ("Complexity must earn it.") See ML_EXPERIMENT_LOG Phase 3.

**ALPHA-v4 ARC (Phases 0-3) COMPLETE.** Net: PEAD = weak beta satellite (kept at telemetry size); TSMOM trend = the strongest sleeve (Sharpe +0.71 19y, diversifying); the book = the two at simple fixed weights (smart weighting doesn't yet earn its complexity on a 2-sleeve / thin overlap).

**✅ LIVE MULTI-SLEEVE WIRING SHIPPED (2026-06-06, shadow-first):** the trend sleeve now trades live as a **standalone weekly ETF rebalancer** (`app/live_trading/trend_sleeve.py`) alongside PEAD — NOT a `swing_selector` value. Direct Alpaca placement with a lightweight risk gate (kill-switch / gross cap trend+PEAD≤80% / fat-finger), fired by the orchestrator daily 09:45 ET → runs only on `pm.trend_rebalance_weekday` (Mon) when the market is open (fail-closed `get_clock`, covers holidays). **Equal-capital 50/50** (`pm.trend_allocation_pct=0.40`); PEAD dialed to telemetry (schema defaults rebaselined 1.0/0.05). Positions tagged `selector="trend"` and excluded from the Trader's stop/target loop (rebalancer-managed). Tracking via `trend_tracker.py` (+0.71 ref, weekly rollup). **Deploys DORMANT + SHADOW** (`pm.trend_enabled=false`, `pm.trend_shadow=true`). **To activate: restart uvicorn, then `python -m scripts.set_trend_config`** (set `pm.trend_enabled=true` for shadow; add `--arm` for real orders). 32 new tests pass; flake8 clean. **REMAINING (owner-gated):** verify a shadow run on the live data tier (≥253 daily bars per ETF), then arm.

**PRIOR NEXT (now done):** ~~live multi-sleeve wiring — get the trend sleeve trading at a simple fixed weight alongside PEAD~~. This places live ETF orders in the paper book → a deliberate live-system change needing owner go-ahead. Then **Phase 4** (options-VRP feasibility spike / squeeze-conditioning; PEAD 2.0 dropped after Phase 1). Phase 0 remainder (sequential-WF baseline, fold-coverage report, gate recalibration, freeze dead XS-ML) remains as harness hygiene. **CI `database is locked` flake — FIXED (2026-06-06):** per-worker SQLite isolation — feature_store / pead_tracker / notifier DB paths are now env-overridable and conftest points each xdist worker at its own temp dir (set before collection); the bare `universe_history` connect hardened (timeout=30). Validated under -n 8 (2767 passed, 0 locks).

---

## 🗄️ PRIOR DIRECTION (2026-06-02, superseded): Alpha v2 — `docs/living/ALPHA_V2_PLAN.md`
PEAD long-only is **live in paper**. The next phase is **structural, not signal-hunting** (from a 5-LLM external review + a code-grounded plan): de-risk PEAD honestly, then test whether the "dead" swing ranker was just strangled by a 5-position long-only book by re-running it **dollar-neutral, sector-neutral, high-breadth, residualized**. **Locked decisions:** short-interest data first; carve a post-2024-06-01 historical holdout; dollar-neutral shorting approved in paper; keep live PEAD running (pause only if leave-one-crisis-out fails). **First moves:** PEAD cost-sensitivity sweep → crisis-block robustness (leave-one-crisis-out). See the plan for the full sequenced roadmap.

**PHASE-1 PEAD DE-RISK COMPLETE (2026-06-02):** §1.1 cost ✅ GO · §1.2 crisis ✅ GO · §1.3 significance ❌ FAIL. PEAD is **real-but-underpowered** (event-level bootstrap p=0.19, HAC t=1.04; CPCV t=2.26 was optimistic) — a long-biased up-trend drift harvester (~0.40 SR, 87% P&L up-trends). **Keep paper-trading as a small diversifier; never a capital centerpiece** (CAPITAL-HOLD confirmed).

**🏁 RANKER VERDICT — DEAD (2026-06-03):** the §3.1 dollar-neutral high-breadth ranker (the Phase-2 capital hope) is **dead**, established rigorously. The first run was invalid (the "dollar-neutral" book actually ran ~35% net-long at 0.35 gross — the L/S engine was fed a one-sided long proposal pool and never re-sized held positions). Fixed end-to-end across 3 phases (observability → neutral-at-target-gross → full-cross-section scoring + breadth). On the **corrected, genuinely-neutral book** (net$ −0.01, gross 0.73), the decisive CPCV (k=8, N_eff=8) gave **Sharpe +0.14, t=0.18, %pos 67%, dep-adj +0.12** → **no cross-sectional alpha**; the long-only +0.22/+1.06 was confirmed **market beta**. Cross-sectional ML ranking is now exhausted (swing noise · intraday cost-drag · ranker null). **→ PEAD is the SOLE validated edge.** Also shipped this session: PEAD-aware entry gate (PEAD had been 0-filling — post-earnings gappers hit the swing 1.5%/0.5% thresholds; now live + filling) and PEAD cockpit v2 (live book + signal log + live-vs-backtest). **NEXT: strategy reassessment** — pivot research to the event-driven edge *family* (PEAD's DNA: discrete event → drift, F2-immune, rules-based) + productionize PEAD; the §3.3 short-interest / Spike-B residualization items are shelved (they were gated on ranker life). Alpha-v3 plan pending owner direction.

**🧭 ALPHA-v3 TRACK-A SWEEP COMPLETE — PEAD STILL SOLE EDGE (2026-06-03 PM):** owner approved Alpha-v3 (A: build a 2nd event edge; B: aggressively ramp PEAD in paper). Built the reusable **EventEdgeStrategy** harness (A0, PEAD byte-identical) + acquired **short-interest/short-volume data** (Polygon/FINRA, 540k rows, PIT-safe). Tested the two top candidates — **both NULL**: **A1 analyst up/downgrade drift** ❌ (CPCV looked best-in-campaign at +0.894/t=2.85 but was a **52% fold-skip artifact**; neutralized L/S +0.342/t=1.24, full-window CAPM alpha t=0.20 → noise) and **A2 dollar-neutral short-interest factor** ❌ (**−1.213, t=−3.53** — the Boehmer/Asquith anomaly *reversed* in the meme era). Cross-sectional ML ranker + A1 + A2 all dead. **PEAD (+0.546, real-but-underpowered) remains the SOLE validated edge.** Also fixed: nightly intraday retrain crash (orphan `n_workers` kwarg). **NEXT → Track B (owner-gated, at restart):** B1 realized-Sharpe EOD pipeline (the self-certification clock) + B2 Friday cron + B4 aggressive paper-allocation ramp. The event-edge engine (harness + beta-isolation discipline + FINRA/SI + analyst-grades data) is retained for future candidates.

**🟢 TRACK B SHIPPED (2026-06-04):** B1 (realized-Sharpe EOD pipeline) + B2 (Friday rollup) were **already built/scheduled** (16:30 ET, 12 tests) — the self-certification clock is live. **B4 aggressive paper ramp** built + merged: config-driven `pm.pead_size_mult`=3.0 + `pm.pead_max_position_pct`=0.10 (live-tunable via agent_config), PEAD-specific `apply_pead_size_ramp`; RM made PEAD-aware so the 10% per-name cap isn't clipped to the global 5% (aggregate still bounded by the 80% gross cap); ADV-participation instrumentation added (slippage already per-fill). 19 tests; PAPER ONLY. **Deploys at the next uvicorn restart.** Tune the ramp live by editing `pm.pead_size_mult` / `pm.pead_max_position_pct` in agent_config (no redeploy). **B5 SPY<200d trend filter** (2026-06-04) replaces the VIX>30 block — CPCV-validated **+0.661 vs +0.546** (every metric better, same window); wired live config-reversible (`pm.pead_regime_control="trend"`, ma=200), fail-CLOSED to VIX if SPY unavailable. **PEAD now stands down in downtrends (SPY<200d), not just VIX spikes** — protects the ramped book. Deploys at restart.

---

## ☀️ MORNING SUMMARY (2026-06-01) — read this first

**Headline: the honest pipeline killed two illusions and found one real edge.**

| Strategy | Honest OOS CPCV (per-fold, leak-free) | Verdict |
|---|---|---|
| Swing (long-only cross-sectional) | +0.22, t=0.17, 50% pos | ❌ DEAD (noise) |
| Intraday v63 | -2.80, t=-6.85, PF 0.94 | ❌ DEAD (cost-drag); struck fake +5.14 (memorization) |
| **PEAD (post-earnings drift)** | **+0.546, t=2.26, 95% pos, P5 +0.009** | ✅ **REAL EDGE** (clears 0.50 paper gate; short of 0.80) |

**PEAD is the first genuine, statistically-significant, economically-grounded positive result** the project
has produced. Best config: long-only, VIX>30 crisis block ON, k=8, no priced-in filter. The VIX block was
the key lever — it trimmed the crisis-fold left tail (P5 -0.288 → +0.009, %pos 80% → 95%). Mean 0.546 is
short of the 0.80 promotion gate but comfortably clears the 0.50 PAPER gate. PEAD is event-driven (F2-immune),
rules-based (no leakage risk).

### UPDATE (2026-06-01 workday) — PEAD lever sweep + second-edge hunt COMPLETE
All high-EV experiments now run. PEAD config tuning is exhausted; the short-side second-edge hunt failed.

| Experiment | Mean | t-stat | Verdict |
|---|---|---|---|
| **PEAD long-only (baseline)** | **+0.546** | **2.26** | ✅ KEEPER, paper-ready |
| PEAD hold-15 | +0.411 | 1.19 | ❌ killed (drift wants longer hold) |
| PEAD long-short | +0.456 | 2.61 | robust lower-return variant (regime-pass) |
| PEAD earnings-quality | +0.449 | 1.02 | ❌ killed (power collapse) |
| QualityShort shorts-only | **-0.903** | -3.19 | ❌ ANTI-EDGE (old +5.95 was inflated) |

**Conclusion:** PEAD long-only +0.546 is the SOLE validated edge. Config tuning can't reach 0.80 (~0.5-0.7
academic ceiling confirmed). Two short approaches ruled out honestly (inverted-long LX7 +0.036; fundamental-
deterioration QualityShort -0.903) — shorting beaten-down names bleeds (they rally on bounces).

### UPDATE (2026-06-01) — Small/mid-cap PEAD expansion REJECTED (honest result)
Built a survivorship-safe small/mid-cap PEAD harness (Polygon grouped-daily flat files, 8755 delisted names
retained, [$2M,$50M] ADV band, top-300/day, 20bps cost, delisted-haircut wired) — 2 Opus pre-run correctness
fixes (PR #362: haircut no-op + eligibility lookahead), full suite green. **Result: mean +0.361, t-stat +0.95
(coin flip), P5 -1.368 → FAILS, and is WEAKER than R1K large-cap PEAD (+0.546, t=2.26).** Opus oddity review:
REAL failure, result trustworthy (the symbology-gap suspect was disproven — FMP covers the tradeable universe;
low trade count is the shared 5-position cap). The literature's "event edges stronger in small-caps" did NOT
survive honest survivorship + cost modeling. **R1K large-cap PEAD remains the sole edge. Small/mid not worth a re-run.**

**The free-data experiment ladder is now fully exhausted** (swing dead, intraday dead, PEAD levers exhausted,
short edges dead, insider weak, buyback no-data, small/mid-cap rejected). The two decisions below are now the
only remaining moves without new infra/data spend.

### UPDATE (2026-06-02) — Gate recalibrated + PEAD long-only WIRED & ACTIVATED for paper
Both decisions executed (Opus design → 2 review passes each → merged):
- **PR #365 — significance-first two-tier promotion gate** (replaces mean-Sharpe≥0.80 relic). PAPER tier
  (t≥2.0, %pos≥0.75, P5≥0.0, mean≥0.35) + CAPITAL tier (t≥2.5, n_folds≥10, mean≥0.50, explicit sign-off).
  Event-sparsity regime waiver (paper-only, flagged) lets PEAD's `worst_regime_sharpe=None` pass paper with
  `requires_human_review`. **Verdicts (real gate): PEAD → PAPER PASS / CAPITAL HOLD; all else FAIL.** Legacy
  `GATE_MODE=mean_sharpe` is a verified no-op. WF-only runs are INCONCLUSIVE (no longer auto-RETIRE in cron).
- **PR #366 — PEAD long-only wired into live paper** (the live path existed but ran a wrong config). Fixed:
  VIX>30 crisis block now fires (daily VIX series injected; **fail-closed** if unavailable), hold=40 (was 5),
  priced-in filter OFF, scorer pins every validated param. Owner-chosen **risk-managed variant** (keeps
  regime/NIS/opportunity/macro/RM overlays → expected tracking error vs the clean +0.546, logged for
  attribution), **marketable entries** (PEAD-scoped, avoids below-ask adverse selection), **full swing budget**.
  9 `pm.pead_*` config keys (defaults=validated). Observability: `app/live_trading/pead_tracker.py` daily row +
  weekly Sharpe-vs-0.546 rollup email.

**🟢 ACTIVATED 2026-06-02:** `pm.swing_selector` set `pead_quality_short` → **`pead`** (pure long-only; drops
the dead QualityShort anti-edge shorts). **⚠️ REQUIRES UVICORN RESTART** to load the new wiring — run
`.\serve.ps1` (or restart `uvicorn app.main:app`). PEAD begins paper-trading on the next premarket cycle
(08:00 ET analyze → 09:50 ET send). Watch `data/pead_tracking.db` daily rows + the weekly rollup email.
**Capital is withheld** (PEAD is CAPITAL-HOLD pending k≥10 re-run OR live-paper confirmation) — this is paper only.

**🔵 DECISIONS NOW YOURS (autonomous experiment ladder exhausted):**
1. **Paper-trade PEAD long-only** — clears the 0.50 paper gate today (t=2.26, 95% pos, PF 1.54, DSR-pass).
2. **Is 0.80 the right promotion gate?** For a PF-1.54 / 95%-positive / DSR-pass / Calmar-0.77 real edge,
   0.80 + P5>-0.30 may be too strict (multi-LLM reviews flagged it). A deliberate gate decision is warranted.
3. **STOP** long-only price-feature ML (swing+intraday dead) AND fundamental/inverted-long shorts (both dead).
4. **Higher-ceiling future bets** (need new infra/data, not just config): options-PEAD (IV-crush), or the two
   remaining untested shorts (MeanReversion, AnalystRevision momentum) — but LOW priority after 2 short failures.

**Caveats on PEAD:** edge rests on ~8 fold outcomes (N_eff=8); ~15% survivorship upper-bound; verify FMP
`date`=announcement-date PIT (5-min spot check). Solid, economically-grounded, but under-powered.

**9 PRs merged overnight (#335–#348):** save-guard, swing+intraday per-fold retraining, 6 latent integration
bugs (all surfaced only on real runs — mocked tests passed vacuously), daily-source fix (aggregate_5min),
real-data smoke test (#345, guards the empty-matrix class), swing deep-dive doc (#344), PEAD instrumentation
fix (#348, real DSR/regime). Full suite green throughout. notify_watcher auto-starts with uvicorn.

(DEFINITIVE PEAD run with real DSR finishing as of this writing — see logs/p0_pead_cpcv_DEFINITIVE.log;
result will be appended to ML_EXPERIMENT_LOG.)

---

## 🌙 OVERNIGHT PLAN (completed)
All 7 planned items done: intraday definitive (−2.80, dead) → daily-source fixes (#343/#346) → intraday
oddity analysis → real-data smoke test (#345) → PEAD eval (+0.546, the win) → swing deep-dive (#344). Plus
PEAD instrumentation fix (#348) for honest DSR. Process held: branch → Opus implement → Opus review → tests
→ merge → document.

---

## Active Phase
**Honest baseline validation on the fully-corrected pipeline**

All 3 phases of the gate integrity overhaul are merged (PRs #329–#334). Pipeline cleared for CPCV by 3 Opus 4.8 review passes. Now establishing the two honest baselines we've never had: swing v224 CPCV and a clean intraday v63 CPCV re-run (the prior +5.14 predates deployment tracking + the active regime gate).

---

## Model Status
→ See `docs/living/MODEL_STATUS.md` for full details.

- **swing v229** — trained 2026-06-06 (age 4d); XS swing ranker is OFF in the live path (`pm.swing_ml_live_enabled=false`).
- **intraday_meta v65** — trained 2026-06-06 (age 4d).
- **regime v5** — active (age 20d).

---

## RUNNING NOW (2026-05-31): First honest swing CPCV
`logs/p0_swing_v224_cpcv_perfold.log` — swing per-fold CPCV C(6,2), `--per-fold-retrain --as-of 2026-05-29`.
This is the FIRST genuinely out-of-sample ML result in project history. Each fold trains a fresh
model on only its own window (Opus-verified no-leak + per-fold OOS guard, verdict "RUN IT"). ETA ~20-60min.
**The gap between this number and the old frozen +0.08-0.55 swing numbers = the leakage that was inflating them.**
Expect it to be honest-low. A pass (mean Sharpe > 0.80) would be the first promotable swing result.

## Superseded Jobs (2026-05-31, earlier)
1. **CPCV swing v224** (~4h) → `logs/p0_swing_v224_cpcv.log` — first honest swing baseline
2. **CPCV intraday v63 re-run** (~4h) → `logs/p0_intraday_v63_cpcv_postaudit.log` — deployment-adjusted Sharpe on fully-corrected pipeline
3. Macro history + yfinance incremental data updates (~12min)

**Tomorrow:** `/morning` for results → Opus 4.8 analyzes for oddities (esp. if intraday Sharpe stays implausibly high → dig for residual artifacts) → decision tree below.

---

## Decision Tree (after tonight's results)
```
Intraday v63 deployment-adjusted Sharpe?
  > 1.0  → Intraday is REAL edge → lead strategy → begin PEAD as 2nd strategy
  0.3-1.0 → Marginal; the +5.14 was largely a deployment artifact (Opus est. 3-7% real)
  < 0.3  → Artifact confirmed → intraday not deployable as-is

Swing v224 CPCV?
  Passes (mean > 0.80) → run paper-trade prep
  Fails → long-only cross-sectional ranking exhausted (all 9 LX experiments failed,
          incl. LX9-A beta-neutral on corrected pipeline: +0.031, F2=-0.70).
          Next swing work = PEAD (earnings momentum, F2-immune) or proper L/S short model.
```

---

## Strategic Context: Why NOT re-run old models
The pipeline bugs (PF gate non-functional, DSR n_obs=0, pit_union look-ahead, Calmar formula)
affected **measurement**, not the underlying signal. The F2 structural loss (Aug 2024 VIX spike
destroys long-only beta-exposed swing models) is real market behavior — no bug fix changes it.
LX9-A confirms this: beta-neutralized, post-audit, still -0.70 on F2.

**Worth re-running:** only v224 CPCV (never run) + intraday v63 (deployment unknown).
**Not worth re-running:** LX2–LX9, v186–v223 in isolation — failures were F2-structural, not bug-induced.

---

## Data Review (queued for analysis)
After tonight's results, evaluate whether the current data set is the bottleneck:
- Current: yfinance daily/5min, Polygon 5min cache, FMP fundamentals, macro_history, sector ETFs
- Question: are we missing data points that would unlock edge? (options IV/skew, short interest,
  VIX3M term structure, alt data). Opus to assess what's worth downloading vs purchasing.

---

## Blockers / Risks
- Intraday +5.14 Sharpe is implausible for retail equity (daily IR 0.32) — likely a low-deployment
  artifact. Tonight's re-run with deployment tracking will quantify it.
- Swing long-only approach may be structurally exhausted — need forward-looking pivot decision.

---

## Recently Completed
- 2026-05-31: 3-phase gate integrity overhaul merged (PRs #329–#334); 3 Opus review passes; CLEARED FOR CPCV
- 2026-05-31: notify_watcher auto-starts with uvicorn; docs restructured (living/reference/archive)
- 2026-05-31: 13-round adversarial audit complete; PIPELINE_ARCHITECTURE.md is SSOT
