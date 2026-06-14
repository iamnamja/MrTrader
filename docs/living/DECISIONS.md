# Decision Log
Append-only record of significant architectural and strategic decisions.
Format: `## YYYY-MM-DD — Title` then context, decision, rationale, consequences.

---

## 2026-06-14 (F2) — ETF relative-value: FAIL (genuinely orthogonal, but no standalone edge)

**Context**: F2 of the plan — `app/strategy/etf_relative_value.py`, a slow dollar-neutral log-spread mean-reversion sleeve over 5 pre-registered economically-linked pairs (QQQ/SPY, IWM/SPY, HYG/IEF, TLT/IEF, EEM/EFA), canonical config (lookback 120, entry z 1.5, exit 0.5, 2bp/leg), equal-weight, declared `diversifier`. Chosen over more calendar premia precisely because it is MARKET-NEUTRAL (not timed beta), so it could in principle diversify the trend book where F1a could not.

**Verdict (FAIL, no promotion)**: deep history 2007→2026 (~3661 OOS): point_SR +0.026, mean_SR +0.110, path_t +0.62, HAC p(1s) 0.458 → PAPER FAIL (no standalone edge). Track-B FAIL: **corr −0.230** (it IS genuinely orthogonal — the diversification the calendar premia lacked) but standalone SR ≈ 0, so appraisal_IR +0.10 / ΔSR −0.011 / P(ΔSR>0) 0.46 — a zero-return diversifier contributes nothing to book SR.

**Rationale / integrity**: Independent Opus adversarial review — HONEST, no bug. The spread SIGN is correct (long-spread when z below −entry = long the cheap leg, P&L +(rA−rB)), PIT clean (z lagged by shift(1)), cost fair (4bp/round-trip on liquid ETFs). A grid diagnostic (NOT used for selection — the canonical config is pre-registered) shows EVERY {lookback ∈ 60/120/250 × entry ∈ 1.0/1.5/2.0} cell stays below the 0.30 paper floor net of cost (best, L=60, net SR 0.19 — still a fail); per-pair only IWM/SPY (+0.22) and EEM/EFA (+0.20) are positive, so the equal-weight combine is not masking a hidden winner. The edge is genuinely weak/absent net of cost, not a wiring artifact.

**Consequences**: Nothing promoted; live book unchanged. The sleeve + registry entry are kept as tested infrastructure; verdict recorded so it isn't blind-retested. KEY LEARNING: the binding constraint for an additive sleeve is **standalone return**, not orthogonality — F2 had the orthogonality (corr −0.23) the calendar premia lacked but still failed for lack of edge. This sharpens the remaining priors: free daily US data offers no easy standalone premium. NEXT: F3 (carry done right, small).

---

## 2026-06-14 (F1c) — FOMC pre-announcement drift: DEFERRED (not built)

**Context**: F1c of the plan was a conditional ("only if a clean 2007–2026 FOMC date list is feasible without bug risk").

**Decision (owner-discipline call): DEFER — do not build now.** Two reasons: (1) **Low EV** — FOMC pre-announcement drift is a long-SPY-in-a-window strategy, structurally the SAME additive-SPY-beta shape that just made turn-of-month and overnight FAIL Track-B (positive correlation with the trend book → won't diversify). (2) **Bug risk** — the repo has no clean in-repo FOMC date series before 2026 (`app/calendars/macro.py` holds only FOMC_2026); hand-assembling ~144 dates is error-prone and a wrong date silently biases the window. Building a low-EV, bug-prone sleeve violates the owner's "best, not fast / avoid bugs" mandate.

**Consequences**: F1c skipped (recorded, not silently dropped). If a clean historical FOMC/event-date source is later wired (e.g. for F4's event panel), this can be revisited cheaply as a registry declaration. Net F1 result: the additive calendar premia (TOM, overnight) FAIL; the VIX-term crash governor (overlay) MODESTLY HELPS and is the one F1 candidate.

---

## 2026-06-14 (F1b) — VIX-term crash governor: MODESTLY HELPS → first candidate overlay (owner-gated)

**Context**: Built the overlay evaluation path the F0 review flagged as missing — `Overlay` / `evaluate_overlay` / `OverlayReport` in `sleeve_lab.py` (a book-MODIFYING signal judged book-WITH vs WITHOUT on tail metrics, not Track-A significance) — plus the first overlay, `app/strategy/crash_governor.py`: de-risk the book's exposure when the VIX term structure inverts (VIX > VIX3M = backwardation = acute stress), signal read at close t and applied to t+1 (PIT-safe via shift).

**Verdict (MODESTLY HELPS — first positive of the F-series; NOT promoted)**: canonical config (derisk_to=0.5, ratio_threshold=1.0, confirm_days=1) on the live 10-ETF TSMOM book, 2007→2026 (n=4891):
- maxDD −13.9% → −12.1% (+1.8pp), Calmar 0.469 → 0.501 (+0.032), Sharpe 0.723 → 0.720 (−0.003), AnnVol 9.3%→8.7%, at a ~0.5%/yr return give-up. De-risks 11% of days (mean multiplier 0.944).
- Crisis maxDD: COVID-2020 −10.7% → −6.5% (notably shallower), GFC-2008 +0.4pp, BEAR-2022 +1.3pp.

**Rationale / integrity**: Independent Opus adversarial review — HONEST, no fix needed. PIT verified (removing the t+1 shift jumps Sharpe to 1.038, i.e. the lag strips exactly the future info; the surviving benefit is real because backwardation persists). No GFC data gap (^VIX3M back-stitches the VXV index to 2007; 146 real GFC obs, no forward-fill). Config canonical (1.0 = the contango/backwardation boundary; 0.5 a round default) not tuned; robust across a 36-cell grid (maxDD improves in nearly every threshold≥1.0 cell; only threshold 0.95 — de-risking in contango — fails). The trend book does NOT already neutralize these days (realized vol 9.6% on de-risk days vs 8.3% otherwise), so the benefit is genuine marginal protection, not double-counting.

**Honest caveats (recorded)**: the real cost is the −0.5%/yr return give-up, not the negligible Sharpe change; and the headline tail benefit is event-concentrated (COVID drives most of it). It is a risk-manager, not a Sharpe-booster.

**Consequences**: The governor is the **first candidate worth the owner's consideration** as a book overlay. **Promotion to live is the owner's decision** — it re-times live exposure (a live behavior change), so it is NOT auto-applied; recorded report-only. Live book unchanged (trend-only 25% + cash). The overlay eval path is now permanent Sleeve-Lab infrastructure (closes the F0 overlay gap). NEXT: F2 (slow ETF relative-value).

---

## 2026-06-14 (F1a) — calendar/overnight premia: BOTH FAIL → not added to the book

**Context**: First sleeves built as registry declarations through the F0 Sleeve Lab — `app/strategy/calendar_premia.py` (vectorized PIT-safe turn-of-month + overnight backtests) + `scripts/walkforward/sleeves.py` (the registry + runner). Run report-only through the Lab on deep history (2007→2026, ~3661 pooled-OOS obs), each declared `risk_premium`, with Track-B vs the live 10-ETF trend book.

**Verdict (both FAIL, no promotion)**:
- **turn_of_month_SPY**: point_SR +0.320, mean_SR +0.280, path_t +3.07, HAC p(1s) 0.097. PAPER FAIL (misses HAC significance). Track-B FAIL: appraisal_IR −0.026, ΔSR −0.058, P(ΔSR>0) 0.25.
- **overnight_SPY**: point_SR +0.344, mean_SR +0.393, path_t +2.58, HAC p 0.087, residual-α_t −1.23. PAPER FAIL. Track-B FAIL: appraisal_IR +0.090, corr 0.33, P(ΔSR>0) 0.47.
- Both **clear the plausibility floor (point_SR ≥0.30) but miss the light HAC significance bar (p>0.05)**, and **decisively fail Track-B** — neither diversifies the trend book (overnight is timed SPY beta the book already holds; TOM actively lowers book Sharpe).

**Rationale / integrity**: Independent Opus adversarial review confirmed the FAIL is HONEST, not an artifact — PIT clean (the TOM calendar position is known ex-ante so it correctly needs no shift, unlike a price-derived weight; overnight = open[t]/close[t-1]−1), costs fair (TOM ~2bp/yr immaterial; overnight ~5%/yr but a physically-real daily round-trip), the residual-α sign correctly reads "diluted beta," and the Track-B fail is cost-independent + decisive. No goalpost-moving: the pre-registered canonical spec (one config each, n_trials=1) was judged as-is.

**Consequences**: Nothing promoted; live book unchanged (trend-only 25% + cash). The two premia builders + the registry are kept as permanent, tested infrastructure and the verdict is recorded so these are never blind-retested. NEXT: F1b (overlay eval path + VIX-term crash governor). Honest read: the two cheapest calendar premia don't clear the bar — consistent with the panel's "trend is the only edge"; the crash governor (an overlay, not an additive sleeve) is the more promising F1 piece.

---

## 2026-06-14 (F0) — Sleeve Lab built: the uniform sleeve→Ruler-v2→report pipeline (Alpha-v7 F0)

**Context**: Executing the game plan above. Every sleeve to date (PEAD, options-XS, index short-vol, trend-broaden) was a hand-written `run_*_cpcv.py` re-implementing the same plumbing — a fresh bug surface each time and a real risk of wiring the gate inconsistently as the book grows to 3–5 sleeves.

**Decision**: Build `scripts/walkforward/sleeve_lab.py` — a `Sleeve` (validated declaration: label / component_type / PIT net returns / optional spy_prices + n_trials_registered) → `evaluate_sleeve()` that runs the IDENTICAL audited path every time (`SeriesReturnStrategy` → `run_cpcv` → Ruler-v2 Track-A PAPER+CAPITAL → optional Track-B `appraise_track_b` → uniform `SleeveReport`), a `@register_sleeve` registry, and `assemble_book()` (F5) over the proven `sleeve_allocator`. It **composes the existing proven pieces — no gate-semantics change — and is report-only** (promotion stays owner-gated). New sleeves register here instead of spawning new top-level scripts; the old `run_*_cpcv` scripts stay as frozen historical runners.

**Adversarial Opus deep-dive — decisions taken**:
- **`overlay` is fail-loud-EXCLUDED** from valid component types. An overlay (the planned VIX-term crash governor) MODIFIES the book rather than adding a return stream, so the additive Track-A significance + Track-B blend model is the wrong instrument and would emit a confidently-wrong verdict. Overlays get a dedicated book-with-vs-without evaluation path, landing with the F1 governor. Failing loud beats silently-wrong.
- **Residual-α is SPY-CAPM-only on this path** (documented KNOWN LIMITATION). `result.residual_alpha_t_hac` is the single-factor SPY CAPM diagnostic (and only when spy_prices is given); the multi-factor harvested-premium-excluded residual-α (`ruler_v2.residual_alpha_t` / `RULERV2_HARVESTED_FACTOR`) is NOT yet wired into the sleeve path — deferred (needs a factor panel). Latent today because CAPITAL is structurally unreachable on a backtest alone (requires live-paper). `evaluate_sleeve` warns when spy_prices is absent.
- **Loud hardening warnings**: thin pooled-OOS (< power floor), `n_folds` below the CAPITAL power floor (10) at the default geometry (FULL_N_FOLDS=8, kept for parity with the calibration controls — a capital-aspiring run must pass n_folds≥10), and `assemble_book` inner-join coverage loss (a short sleeve silently truncating the book window).
- **C1 (Track-B worst-regime) confirmed NOT a bug**: the lab passes the candidate's standalone CPCV worst-regime to Track-B, which matches Track-B's documented contract (candidate regime safety backstop; book contribution is measured separately by appraisal-IR / P(ΔSR>0) / tail-overlap).

**Consequences**: 20 offline tests. `PIPELINE_ARCHITECTURE` component map + changelog updated. Follow-ups tracked: overlay eval path (F1) and multi-factor residual-α wiring. NEXT: F1 sleeves built as registry declarations through the Lab. Live book unchanged.

---

## 2026-06-14 — 5-LLM research panel synthesized → game plan: Sleeve Lab + orthogonal deep-history premia (Alpha-v7 next)

**Context**: After the Ruler-v2 go-live + the honest candidate sweep (only trend survived), posted a research-request pack to 5 external quant LLMs (Opus 4.8, ChatGPT, DeepSeek, Gemini, Grok) asking for the best next steps to find alpha. All five returned. Synthesized into `docs/reference/ALPHA_V7_RESEARCH_SYNTHESIS_2026-06-14.md` (the new direction SSOT); inputs archived at `docs/archive/llm-reviews/2026-06-14/`.

**Decision (owner, after deep read of all five)**:
- **Accept the unanimous picture**: kills are honest, trend is the only edge, build a **3–5 sleeve risk-premia book around trend at a realistic book SR ~0.7–0.9** (not a home run), every new bet on **deep free history** (19y ETFs / decades of FRED), the 4y frozen options only for *conditioning*.
- **Adjudicated divergences**: (1) power is binding ONLY for the retired path-t / 4y options — on 19y daily data it is NOT (the dead candidates died on zero edge, not power); so new bets must be deep-history AND mechanism-backed. (2) Carry deserves a *proper* retest (roll-down, not distribution-yield) but is crisis-correlated + likely overlaps TSMOM → medium priority, sized small. (3) Trend-breadth is largely SPENT (live universe already cross-asset; P5 broadening failed; real extension needs futures) → deferred.
- **The future-proof investment (owner's emphasis on best-not-fast/hardened)**: build a **Sleeve Lab** FIRST — a uniform, tested sleeve research→Ruler-v2(Track-A+B)→sleeve_allocator→report pipeline + a sleeve registry, retiring the bespoke `run_*_cpcv` scripts. Makes every future premia idea a small, uniform, hardened declaration.
- **Phased plan** (see SSOT): **F0** Sleeve Lab · **F1** structural/calendar/overnight premia + a VIX-term crash governor (highest-EV: most orthogonal, most powered, cheapest, owned data) · **F2** slow ETF relative-value · **F3** carry-done-right (small) · **F4** options-conditioned event interaction (long shot) · **F5** book assembly + live fidelity. Deferred/data-gated: cross-asset trend via futures (Norgate), aggregate short-interest timing (FINRA backfill), index-VRP ETP (dangerous, last).

**Rationale**: on free daily US data the cross-sectional IC is ≈0, so breadth is the only lever and trend is the canonical breadth play; the highest-marginal-value work is the reusable Lab + the two cheapest *orthogonal* premia + a crash governor — not more gate machinery (the rigor is sufficient) and not more model variants on already-killed lines.

**Consequences**: `MASTER_BACKLOG` updated with F0–F5 as THE active plan; the prior Alpha-v7 7-phase blueprint is superseded for direction by this synthesis. No live/code change yet — this is the plan; execution (starting F0) is the next session's work on owner go-ahead. Live book unchanged (trend-only 25% + cash). Honest expectation: likely a 2–3 sleeve book; if nothing new passes on free data, the decision becomes operational-excellence + a deliberate data buy (Norgate), not more searching.

---

## 2026-06-13 (Track-B GO-LIVE) — wired the run_book_gate dispatcher + flipped TRACKB_MODE → ruler_v2

**Context**: The GATE_MODE go-live audit (entry below) found that flipping `TRACKB_MODE` would be an INERT no-op — no runner dispatched on it (`run_book_gate.py` hardcoded `book_delta_gate`). This change wires the dispatcher and completes the Track-B side of the migration.

**Decision**: `run_book_gate._evaluate(base, candidate, *, mode, …)` now dispatches on `TRACKB_MODE` — `"ruler_v2"` → `track_b_appraisal.appraise_track_b` (trend candidate declared `component_type="risk_premium"` → worst-regime backstop waived; budget-invariant appraisal IR + block-bootstrap P(ΔSR>0)); `"book_delta"` → legacy `book_delta_gate` (retained + tested). Added an ASCII `format_report(TrackBAppraisalResult)`. **`TRACKB_MODE` flipped `book_delta` → `ruler_v2` (LIVE)** — now that a dispatcher actually reads it. A ruler_v2 result is recorded under its OWN hypothesis_id (`TRACKB-TSMOM-VS-PEAD-RULERV2-20260613`, parent = the book-delta row) so criteria + result describe the same gate.

**End-to-end live run (manual, report-only)**: tsmom_trend vs the PEAD book, 1496-day overlap, budget 0.25 → **VERDICT: PASS** (appraisal IR +0.881 ≥0.20; P(ΔSR>0) 0.976 ≥0.90; corr 0.286 <0.30; standalone vol-targeted SR 0.93; tail-overlap 0.071; residual-α t_HAC +2.11; worst-regime waived). i.e. under the budget-invariant gate, trend genuinely diversifies the (now-demoted) PEAD book.

**Rationale**: completes the audit's MAJOR-1. `run_book_gate` stays a MANUAL, report-only tool (`decision='park'`, owner-gated, no auto-promotion, no cron) — flipping the mode only changes which gate that manual tool applies. The only behavioral reader of `TRACKB_MODE` is this runner.

**Consequences**: Track-B v2 is the live Track-B gate; `book_delta` retained as legacy. Track-B book inclusion remains owner-gated (the PASS above is report-only; PEAD-as-base is stale post-demotion — point the runner at the live book when a real diversifier decision is on deck). Independent Opus deep-dive on the wiring: SHIP (no CRITICAL; code correct, ASCII-safe, manual/report-only).

---

## 2026-06-13 (GO-LIVE) — GATE_MODE flipped significance → ruler_v2 (Ruler v2 is now the production Track-A gate)

**Context**: After the R4 remediation (diversifier regime waiver + PAPER HAC-significance floor) and reclassifying xmom_12_1 → known_marginal (a documented correct-reject), the full-control-set R4 came back **CLEAN under the strict definition** (no carve-out): significant positives tsmom_4y/19y pass Ruler-v2 PAPER, all 5 true-nulls fail, leaky rejected. Owner approved the flip (reclassify-then-flip).

**Decision**: **`GATE_MODE` flipped `significance` → `ruler_v2` (LIVE).** Ruler v2 is now the production Track-A promotion gate; PAPER = plausibility + a light HAC-SR significance floor + diversifier regime waiver; CAPITAL = Bayesian posterior + structural live-paper + residual-α + bootstrap + power floor. The `significance` branch is RETAINED as legacy (reachable via `significance_gate_*` and explicit mode), exactly as `mean_sharpe` was kept when significance went live.

**`TRACKB_MODE` stayed `book_delta` at GATE_MODE go-live (NOT flipped then).** The go-live Opus audit found that flipping it would have been an INERT no-op — no runner dispatched on it. **→ SUPERSEDED same day: the run_book_gate dispatcher was wired and `TRACKB_MODE` flipped to `ruler_v2` — see the "Track-B GO-LIVE" entry above.**

**Verification**: independent Opus go-live audit = the GATE_MODE flip is mechanically correct (all callers route to ruler_v2; legacy branch reachable), causes NO live break (every `gate_detail` consumer iterates generically; no significance-only key is indexed live), is cleanly reversible, and rests on a legitimately-CLEAN strict R4. `gate_calibration` was DECOUPLED from GATE_MODE (significance OC columns now via `significance_gate_*` directly) so the calibration/recalibration rule stays correct under the new default. Full suite under the flip: only the known pre-existing local-only NFP test fails (3398 passed).

**Rationale**: This is the analogue of the significance go-live — promote the validated gate, keep the prior as legacy. R4 (the pre-registered pre-flip gate) is clean; the xmom reclass is independently grounded (cross-sectional momentum dead, 2026-06-03); nothing is in flight (SWING/INTRADAY retrain disabled), so the flip affects only FUTURE gate calls.

**Consequences**: The NEXT retrain/promotion is gated by Ruler v2. Live book unchanged (trend-only 25% + cash — no promotion pending). Reverting = set `GATE_MODE` back to `"significance"`. Remaining: wire the Track-B runner dispatcher before flipping `TRACKB_MODE`. See PIPELINE_ARCHITECTURE.md (gate inventory + changelog), MODEL_STATUS.md, PROJECT_STATE.md.

---

## 2026-06-13 (full-set R4) — full control set run → NOT strictly clean (xmom fails, a documented correct-reject); DID NOT flip

**Context**: Owner chose "complete the full control set, then flip." Ran the remaining controls (`pead_baseline`, `xmom_12_1`) into `logs/gate_calibration_20260613.json`. Result: tsmom_4y/19y PAPER-pass (diversifier waiver + significant), all 5 true-nulls PAPER-fail (incl. seed_5), leaky rejected — BUT **`xmom_12_1` (declared `positive_alpha`) FAILED Ruler-v2 PAPER**, so the strict R4 reads NOT CLEAN.

**Decision**: **DID NOT flip `GATE_MODE`/`TRACKB_MODE`.** xmom's failure is, on the merits, a CORRECT rejection — xmom 12-1 cross-sectional momentum is genuinely insignificant in this window (meanSR 0.17, t 0.77), the LEGACY significance gate independently fails it too (`significance_core_pass=False`), and the project already ruled cross-sectional momentum dead (DECISIONS 2026-06-03; the control spec itself says "post-2010 attenuation expected"). So Ruler-v2 is behaving correctly, not committing a Type-II.

**What I explicitly did NOT do (integrity)**: I drafted a refinement to `ruler_v2_r4_summary` that excluded significance-core-failing positives from the Type-II count (which would have made R4 read CLEAN). An independent Opus methodology review flagged it as **GOALPOST-MOVING**: it was a post-hoc carve-out added in the same step that declared victory, it uniquely targeted the one failing control, and — decisively — it used the discredited legacy path-t core to define "genuinely significant," which structurally blinds R4 to the exact regime-heterogeneous Type-II Ruler v2 exists to catch. **Reverted it.** A flip on a same-step redefinition is precisely the "silent goalpost move" RULER_V2_DESIGN.md §3 forbids.

**Rationale**: The substantive gate behavior is correct on the full set, but turning "NOT CLEAN" into "CLEAN" requires a JUDGMENT CALL about xmom's classification that must not be made unilaterally right before a live flip. The honest state: R4 is clean on the real edges (tsmom) + nulls + leaky; the one blemish is a labeled-positive that is independently-documented-dead.

**Consequences**: The flip is BACK to an explicit owner decision with these options: **(a)** accept xmom as a documented correct-rejection (footnote, strict R4 definition unchanged) and flip; **(b)** RECLASSIFY `xmom_12_1` from `positive_alpha` → `known_marginal` as a deliberate, dated, pre-stated registry/spec amendment (justified by the 2026-06-03 cross-sectional-momentum-dead decision) BEFORE re-reading the verdict, then flip; **(c)** hold dark. Ruler v2 stays DARK; live book unchanged (trend-only 25% + cash). The strict `ruler_v2_r4_summary` is unchanged on main (no carve-out shipped).

---

## 2026-06-13 (later) — Ruler-v2 R4 remediation (1)+(2) → R4 now CLEAN on the decisive controls

**Context**: The earlier R4 run (same day) was NOT CLEAN for two reasons (artifact A: diversifiers mis-routed through Track-A; real leak B: plausibility-only PAPER admitted a lucky null). Owner directed: "do 1 then 2." Implemented both as DARK ruler_v2 PAPER changes; independent Opus deep-dive = SHIP (Monte-Carlo-verified). Re-ran the decisive controls (artifact `logs/gate_calibration_20260613.json`).

**Decision**: Both fixes ship (DARK); **R4 is now CLEAN on the decisive controls**:
- **(1) Component-type regime waiver** — ruler_v2 PAPER waives the worst-regime backstop for declared diversifiers/risk-premia (`RULERV2_REGIME_WAIVED_TYPES`; carried by the new pure-additive `CPCVResult.component_type`). CAPITAL still needs explicit `regime_waiver_approved`. tsmom_4y/19y declared `risk_premium` → **both now PAPER-pass** (artifact A resolved: a Track-A mis-routing, not a gate flaw).
- **(2) PAPER light significance floor** — ruler_v2 PAPER now also requires one-sided HAC-SR p < `RULERV2_PAPER_MAX_HAC_P=0.05` on the POOLED-OOS instrument (NOT the path-t). **All 5 true-nulls now PAPER-fail** including `random_balanced_seed_5` (the lucky null that previously PASSED) — leak B closed. MC: the 0.05 floor → ~5.7% null-pass (was ~23.4% plausibility-only).

**R4 RULER-v2 CHECK verdict: CLEAN** — positives `['tsmom_4y','tsmom_19y']` pass; all 5 true-nulls fail; leaky rejected on the implausibility ceiling. PAPER is now "plausibility + a LIGHT significance floor," still far more lenient than CAPITAL (Bayesian posterior + live paper + residual-α + bootstrap).

**Rationale**: The two fixes are orthogonal (a diversifier with a waived regime STILL must clear significance; a lucky null fails significance regardless of regime) and use the honest pooled-OOS HAC instrument, not the discredited path-t. Both DARK; legacy gates byte-for-byte untouched.

**Consequences**: The owner-ratified pre-flip gate (R4 clean both ways) is now MET on the decisive control set (tsmom positives + true-nulls + leaky). REMAINING before a live flip: (a) optionally complete the FULL pre-registered control set (pead_baseline, xmom_12_1) so the recalibration rule evaluates (currently "INCOMPLETE - partial set"); (b) the live `TRACKB_MODE` then `GATE_MODE` flip is the owner's call. Ruler v2 still DARK; live book unchanged (trend-only 25% + cash).

---

## 2026-06-13 — Ruler-v2 R4 calibration → NOT CLEAN; DO NOT flip GATE_MODE (real Type-I leak at PAPER + a run artifact)

**Context**: Before flipping `GATE_MODE="ruler_v2"` live, the owner-ratified checklist (RULER_V2_DESIGN.md §6, risk R4) requires the gate-calibration controls to pass BOTH ways through Ruler-v2 PAPER: real positives clear, true-nulls stay dead. Ran the decisive controls (tsmom_4y/tsmom_19y positives, random_balanced_seed_1..5 true-nulls, leaky_tplus1) via the new report-only R4 instrument in `gate_calibration.py` (artifact `logs/gate_calibration_20260613.json`). Verdict: **R4 NOT CLEAN.** Independent Opus review confirmed the read.

**Decision**: **DO NOT flip `GATE_MODE` (or `TRACKB_MODE`) live.** Two structurally distinct problems, reported separately:
- **(A) Positives failed Ruler-v2 PAPER on `regime_not_catastrophic` — a RUN ARTIFACT, not a gate flaw.** tsmom is a crisis-DIVERSIFIER whose worst-regime backstop failure is exactly the 2026-06-10 P0 finding (judge it on Track-B, which waives the regime floor for `risk_premium`/`diversifier`). R4 scored it through Track-A (component_type unset → no waiver; Ruler-v2 PAPER's −0.5 floor is identical to the legacy backstop, which PAPER never loosened for Track-A). Compounded by thin coverage (run warned "no stress-regime fold evaluated"), so the worst-regime values (−2.08 vs a +1.6 meanSR) are not gate-grade.
- **(B) A TRUE null (random_balanced_seed_5) PASSED Ruler-v2 PAPER — a REAL Type-I leak.** PAPER is plausibility-only (no significance on the AND); the `RULERV2_PAPER_MIN_SR=0.30` floor sits only ~0.74σ above zero at n≈1500 (SR sampling SD ≈ √(252/1500) ≈ 0.41), so ~23% of zero-edge nulls clear it. Observed 1/5 (20%) = the expected rate, not bad luck. This is the Type-I risk the 2026-06-12 independent advisory flagged as the under-examined side of the redesign.

**Rationale**: A flip cannot be justified while (B) — a confirmed gate weakness — is open AND (A) leaves the positives uncertified through the correct path. The validation gate did its job: it caught the problem before any live change. Live book unchanged (trend-only 25% + cash); no urgency (no candidate queued; dark-coexistence is the ratified default).

**Consequences**: Pre-flip remediation (owner-ratified before any future flip; ranked): **(1)** re-run R4 with `component_type` set + diversifiers routed through Track-B; **(2)** require stress-regime fold coverage for a calibration to be admissible; **(3)** for the Type-I leak, add a LIGHT significance floor to PAPER (preferred — a HAC-SR p-floor on the pooled OOS series, not the saturated path-t) OR **(4)** raise `RULERV2_PAPER_MIN_SR` (blunter; trades Type-I for Type-II on genuine 0.4–0.7 edges). Prefer (3) over (4). The R4 instrument (report-only `rv2_*` columns + `ruler_v2_r4_summary` in `gate_calibration.py`) is retained as the permanent pre-flip gate. Ruler v2 stays DARK. See ML_EXPERIMENT_LOG / the artifact for the run record.

---

## 2026-06-12 — P5 trend-broadening → PARK (the simple 10-ETF sleeve wins; complexity didn't earn it)

**Context**: P5 (Track B) — broaden the validated TSMOM trend sleeve (the live book's sole sleeve, +0.71 Sharpe/19y) on the 19y window where t≥2 is reachable. Owner-approved spec: all three levers — more legs (16 ETFs: the 10-ETF core + HYG/LQD/SHY/SLV/VGK/EWJ), long-short, and a new 10% book-vol overlay. Pre-registered as ONE frozen spec (`P5-TRENDBROADEN-20260612`, `preregistered_at=2026-06-12T16:00Z`) — NOT a sweep (the OPT-5 trap). Built the book-vol overlay in `app/strategy/tsmom.py` (optional `book_vol_target`; the live sleeve runs `None` = byte-identical), 13 tests.

**Result (the one confirmatory run, recorded R4, run_at 2026-06-12T19:04Z)** — broadened vs the 10-ETF live baseline on the identical 19y window:
- **Broadened (16 ETF, L/S, 10% book-vol): Sharpe 0.30, t=1.31, maxDD −24.7%, Calmar 0.11.**
- **Baseline (10 ETF live): Sharpe 0.72, t=3.18, maxDD −13.9%, Calmar 0.47.**
- The broadened sleeve fails ALL THREE frozen pass conditions: not significant (t<2), lower Sharpe, deeper drawdown. Notable nuance: the long-short DID improve crisis behavior (2020 COVID +2.5% vs −6.2%; 2022 +8.1% vs +0.9%) — but the short-side bleed in a 19y bull + the added leverage swamped it on Sharpe/DD.

**Decision (frozen rule: not-significant OR doesn't-beat-baseline → PARK)**: **PARK. The current 10-ETF, long-flat, per-instrument-vol-targeted sleeve STAYS — broadening is rejected.** No lever-hunting / universe sweep (the kill rule forbids it). "Complexity must earn it" confirmed again (echoes the Alpha-v4 regime-tilt allocator that also failed its margin). Independent Opus 4.8 deep-dive = **SAFE-TO-RECORD** (the PARK is over-determined; the new book-vol overlay is PIT-clean; baseline reproduces the validated +0.721/19y, proving the harness is sound).

**Consequences**: **no live-book change** — trend-only (25%) + cash, the existing 10-ETF sleeve. The broadened-sleeve *capability* (book-vol overlay + long-short) is retained in `tsmom.py` (off by default) for any future use. The standalone long-short crisis-positivity is a real, logged observation — not pursued now (it didn't beat the simple sleeve on the book metrics that matter). See ML_EXPERIMENT_LOG + MASTER_BACKLOG 2026-06-12.

---

## 2026-06-12 — H2 NOT_CONFIRMED (OPT-5 settled, stays parked); H3 BLOCKED (data-unavailable); event panel options-enriched

**Context**: The two remaining 2026-06-11 pre-registered event hypotheses (PEAD-improvement; PEAD is demoted so these are pure research, no capital). Both adjudicate on the earnings-event panel — but its options-pre-event features were 100% empty (the H1-era panel was equity-only). Built the prerequisite **event-time options join** (`app/research/event_options_join.py` + `scripts/enrich_event_panel_options.py`): a PIT as-of join populating the panel's OPTION_COLUMNS (cpiv_pre, skew_25d_pre, reaction_ratio, iv_runup, opt_volume_z_pre, post_iv_retention, pre_event_implied_move) from the options feature table at the pre-event snapshot — the last chain knowable strictly BEFORE the announce day (gated on the holiday-aware `knowable_date`; UNK BMO/AMC → conservative). 46%→45% event coverage (options data starts 2022). 8 tests; independent Opus 4.8 deep-dive = **SAFE-TO-RECORD** (no look-ahead: the forward return starts at announce+1 open and excludes the announce-day gap, so reaction_ratio cannot mechanically correlate with the outcome).

**Decisions / verdicts**:
1. **H2 (`H2-IMPLIEDMOVE-CONTINUOUS-20260611`) → NOT_CONFIRMED (recorded R4, decision=park).** Two-way clustered OLS of 10d SPY-hedged drift on the CONTINUOUS reaction_ratio (= |announce-day move| / pre-event implied move), n=9,034: **coef −0.0010, day-clustered t=−1.21, decile ρ≈0** — weakly negative but NOT significant (frozen bar: coef<0 AND t≤−2 AND monotone). **OPT-5 is now settled PROPERLY as a continuous feature: no edge → the FRAGILE/PARKED verdict stands, no threshold-hunting** (the OPT-5 binary-sweep trap stays closed).
2. **H3 (`H3-PEADV2-SCORECARD-20260611`) → BLOCKED, NOT recorded (one-shot preserved).** Its FROZEN 9-feature list includes `revision_momentum`, which is 100% null in the panel — forward-estimate-revision data is DATA-BLOCKED (the long-standing P4-estrev finding; needs paid I/B/E/S). The runner DETECTS the missing frozen feature and **refuses to consume the one-shot R4** on an un-runnable test. (An exploratory 8-feature peek dropping the blocked feature showed nothing: val decile ρ=+0.25, +4bp, non-monotone — but that is NOT the pre-registered test and is not recordable.) H3's pre-registration stands for if/when revision-revision data is ever acquired.

   **What `revision_momentum` needs (unblock path, 2026-06-12):** the trend in analysts' FORWARD-EPS *estimate* revisions ahead of the print — i.e. a HISTORICAL time series of the consensus forward-EPS estimate (snapshots through time per ticker), so the pre-event revision path is measurable PIT. We have only the *level* (FMP `epsEstimated` = one snapshot at the report) and analyst *ratings* (`/stable/grades` — already tested as A1, null), NOT the estimate-revision series. Three ways to unblock: (a) **buy** a revision dataset — **Zacks Estimates** (most retail-accessible; the Zacks Rank IS a revisions model) or I/B/E/S / FactSet (institutional) — gives retroactive 2022–26 history so H3 runs immediately; (b) **collect-going-forward** — snapshot the consensus forward estimate weekly from FMP/Finnhub starting now (~$0, but ~1–2 yrs to accumulate testable events, no retroactive history); (c) a quick check of a higher FMP tier's analyst-estimates endpoint (likely current-only, not PIT-historical). **Recommendation: keep H3 PARKED — do NOT buy data for it.** It improves PEAD (demoted/off-book), and the exploratory 8-feature peek (dropping revision) already showed nothing — revision_momentum would have to carry essentially the whole event, a stretch. Revision data is only worth acquiring for an INDEPENDENT reason (e.g. the backlog's P5-optional standalone forward-estimate-revision sleeve, distinct from the killed ratings-A1); if it arrives for that, H3 rides along for free.

**Consequences**: the event-conditioned PEAD-improvement thread is closed for now — H2 confirms no continuous implied-move edge; H3 is blocked on unavailable data. **No live-book impact** (trend-only 25% + cash, unchanged). The options-enriched event panel + the event-time join are retained as standing instruments for any future event hypothesis. See ML_EXPERIMENT_LOG + OPTIONS_DATA 2026-06-12.

---

## 2026-06-12 — H4a–H4e VERDICTS: options-as-signal cross-sectional equity edge = DEAD (all 5 KILL)

**Context**: P4's confirmatory adjudication of the five pre-registered options-as-signal hypotheses (frozen 2026-06-12T12:00Z) — does an options-derived feature carry a cross-sectional EQUITY edge, executed as a weekly dollar-neutral decile L/S sleeve at equity cost? Built `scripts/run_options_xs_cpcv.py` + `app/research/options_xs_ls.py` (decile high-minus-low construction, multi-factor residual alpha, H4c put-heavy composite) on the 730-name / 583k-row options feature table, over the full 4y window (208 weeks).

**Result (the five one-shot R4 runs, recorded; `run_at` 2026-06-12T15:3x UTC > prereg)** — week-clustered t / net spread per week / residual-α t:
- **H4a CPIV** (dir +): t=**−2.70**, −57bp/wk, α t=−2.79 → **KILL**
- **H4b put-skew** (dir −): t=**−4.10**, −55bp/wk, α t=−4.30 → **KILL**
- **H4c put-heavy O/S** (dir −): t=**−4.43**, −50bp/wk, α t=−3.16 → **KILL**
- **H4d term-slope** (dir +): t=**−2.83**, −48bp/wk, α t=−2.78 → **KILL**
- **H4e IV/RV** (dir −): t=**−0.12**, −2bp/wk, α t=−0.72 → **KILL**

**Decision (the FROZEN kill rule: simple decile sorts show nothing net of costs → CLOSE the line)**: **all five options-as-signal lines are CLOSED.** H4a–H4d are *significantly negative* — the hypothesized books LOSE money (the academic CPIV/skew/O-S/term-slope signs do NOT survive 2022–26 in this R1K universe at equity cost); H4e is pure noise. **The inverse "working" is NOT a tradeable result**: flipping a pre-registered sign post-hoc is the sign-mining pre-registration forbids, and the independent Opus 4.8 deep-dive identified the strong-but-inverted signs as a **2022–23 growth-crash regime effect**, not a stable edge. NO escalation to ML combinations (the kill rule's explicit prohibition; the dead XS-ML is not revived).

**Process / integrity**: built Opus 4.8 → independent Opus 4.8 deep-dive (verdict SHIP-AFTER-FIXES) → fixes → 29 tests. The deep-dive **confirmed NO look-ahead** (feature as-of, forward return, factor alignment all PIT-clean) and that an alarming smoke result (ρ −0.99 / −75bp/wk) was a genuine regime effect, not a bug. Two fixes pre-R4: a BLOCKER (an invalid decision label that would have crashed a PASS write) and the monotonicity reading (operationalized as the standard sign-of-Spearman trend, not strict every-step — the Type-II trap; both readings recorded for audit). NO verdict hinged on that interpretation (all fail the t≥2 gate outright).

**Consequences**: the **options data is confirmed (again) as NOT a tradeable equity-signal edge** in this window — corroborating the whole Alpha-v6 arc (options-as-execution already killed; PEAD demoted). The live book stays **trend-only (25%) + cash**, unchanged. Options remain a *data asset* for event-conditioning research (H2/H3) only. See ML_EXPERIMENT_LOG + OPTIONS_PROGRAM 2026-06-12.

---

## 2026-06-12 — Alpha-v6 P4a: options feature table + quality filter shipped; H4a–H4e pre-registered

**Context**: With PEAD demoted (H1) the live book is trend-plus-cash, and the blueprint's next research thread is **P4 — options-as-signal**: do options-derived features carry a cross-sectional EQUITY edge, adjudicated as a weekly dollar-neutral L/S sleeve (information at equity cost, NOT an options trade)? The computed-greeks store (P2, 733/733, 112.8M rows) is the substrate; this PR builds the daily FEATURE layer on top of it and freezes the confirmatory hypotheses BEFORE any L/S run exists.

**Decisions / what shipped**:
1. **The daily options feature table** (`app/data/options_features.py` + `scripts/build_options_features.py` → `data/options_features.parquet`): one row per (underlying, date) from each name's greeks-store slice, carrying the five P4 features — `cpiv_matched_delta` (Cremers-Weinbaum), `skew_25d_put` (Xing-Zhang-Zhao), `term_slope_30_60`, `iv_rv_20d_ratio`, `opt_share_volume_ratio` (Roll-Schwartz-Subrahmanyam O/S) — plus `atm_iv_30d`, `implied_move_front`, `put_call_volume_ratio`, `opt_volume_z`, and coverage/quality columns. Frozen quality contract: a contract is valid only if `solver_status=="ok"` and not stale; a name-date with <6 valid contracts is dropped. PIT throughout (knowable_date = store's holiday-aware D+1 session; RV + opt_volume_z use strictly-prior windows).
2. **The options-quality universe filter** (`app/data/options_quality.py`): the PIT coverage floor (≥6 valid contracts, non-NaN atm_iv, ≥100 traded contracts) deciding who is in the cross-sectional sort.
3. **H4a–H4e PRE-REGISTERED** (`scripts/preregister_options_xs_features.py`, label=confirmatory, `preregistered_at=2026-06-12T12:00Z`, each ONE confirmatory shot R4): **H4a** `H4a-OPTIONS-CPIV-20260612` (CPIV → POSITIVE coeff); **H4b** `H4b-OPTIONS-SKEW-20260612` (skew → NEGATIVE); **H4c** `H4c-OPTIONS-OSRATIO-20260612` (put-heavy O/S → NEGATIVE); **H4d** `H4d-OPTIONS-TERMSLOPE-20260612` (term-slope → POSITIVE / backwardation→negative return); **H4e** `H4e-OPTIONS-IVRV-20260612` (IV/RV richness → NEGATIVE). **Frozen kill rule: simple decile sorts show nothing net of costs → CLOSE the line; do NOT escalate to ML combinations (NOT a revival of the dead XS-ML).** The line is a SIMPLE decile sort on purpose.

**Process / rationale**: Built by Opus 4.8 → **independent Opus 4.8 adversarial deep-dive** → fix loop → **Opus 4.8 verification pass (SHIP)**. The deep-dive caught **2 BLOCKERs before any H4 run**: (B1) `knowable_date` was recomputed with a holiday-blind `BDay(1)` (landed on market holidays → a 1-trading-day-early PIT leak into the sort) — fixed by carrying the store's holiday-aware value through; (B2) RV was computed from the store's AS-TRADED `underlying_close`, so a split step (NVDA 10:1: RV 8.23 vs true 0.48) corrupted `iv_rv` for ~20 sessions per split — exactly the names H4e sorts on — fixed by using split-adjusted equity closes for RV (+ a split-jump guard on the fallback). Plus the subset-overwrite footgun in `assemble_final` (a single-name rebuild truncated the full table). Same class of split-adjustment landmine prior reviews caught in the greeks backfill and the event panel. 47 tests; flake8 clean.

**Consequences**: P4a is the DATA LAYER only — **no H4 verdict yet**. The five confirmatory runs (R4, `run_at > 2026-06-12T12:00Z`) are the next P4 step and need ZERO live capital. This does not change the live book (still trend-plus-cash, PEAD→0 owner-gated). See ML_EXPERIMENT_LOG + OPTIONS_DATA §(feature table) + OPTIONS_PROGRAM.

---

## 2026-06-12 — H1 VERDICT: PEAD DEMOTED at event level → live book = trend-plus-cash (#456)

**Context**: The Alpha-v6 centerpiece. PEAD has been live (telemetry size) on an instrument P0 proved couldn't separate it from noise (8-fold path-t). H1 (`H1-PEAD-EVENTLEVEL-20260611`, pre-registered 2026-06-11T12:00Z) re-adjudicates the LIVE PEAD edge at the EVENT level — the right inference unit — on a 21,330-event / **9,774-qualified** R1K panel (2019→2026), with two-way (announce_date × firm) cluster-robust SEs (CGM, validated to the published Petersen 2009 pins).

**Result (the ONE confirmatory run, recorded R4; `panel_sha256=af206149…` pinned in the registry)**:
- **PRIMARY 10d SPY-hedged event return: mean −8.3bp, two-way-clustered t=−0.77, one-sided p = 0.7804.** NEGATIVE point estimate at every horizon (5d −9.4bp, 10d −8.3bp, 20d −14.0bp).
- Conservative quarter-cluster bootstrap agrees: p=0.66, CI [−6.2%, +3.8%]. Beta-adjusted 10d −15.2bp. Live-B5 (SPY<200d trend-gated) slice −21.3bp (t=−1.80) — worse. Robust to leave-one-quarter/sector/top-10. Deciles vs pead_score_v1 non-monotone (ρ=+0.22).
- F4 caveat MOOT: the announce+2 (live) vs announce+1 (primary) gap is only +2.6bp — the edge is negative even at the favorable announce+1 entry, so no execution-timing upgrade rescues it.

**Decision (the FROZEN pre-registered rule: p>0.15 → DEMOTE)**: **PEAD is NOT an event-level edge. Per the pre-committed rule, the live book becomes trend-plus-cash** — TSMOM trend (Track-B-validated, #451) is the capital base; PEAD's research case for capital is closed. This corroborates and sharpens the Alpha-v4 Phase-1 finding (CAPM beta-hedged Sharpe −0.37): the positive long-only Sharpe was beta riding the bull, not drift alpha. **No strategy is ever again killed or kept on an 8-fold t-stat alone** — the success metric of the inference upgrade is met (PEAD now has an honest event-level verdict).

**Consequences / OWNER ACTION (decision=None recorded — live-capital changes are owner-gated)**: the research verdict is DEMOTE; **actually flipping the live PEAD sleeve to zero (keep telemetry logging) is an owner action**, not auto-executed. Recommended: set PEAD allocation → 0 (retain the tracker for telemetry), book = trend (40% / Track-B 25% risk framing — the open reconciliation) + cash. H1/H2/H3's H1 is now answered; **H2 (continuous reaction_ratio) and H3 (PEAD v2 scorecard) are NOT auto-run** — they were PEAD-improvement hypotheses; with PEAD demoted, whether to still test them (as pure research, no live capital) is an owner call. The event panel + CGM inference instrument is now the standing tool for every future event hypothesis. See ML_EXPERIMENT_LOG + PIPELINE_ARCHITECTURE 2026-06-12. The PRE-COMMITMENT below was logged before this number was seen.

**UPDATE 2026-06-12 — the owner action was EXECUTED (overnight; notify id 59 "PEAD flipped OFF live", 00:16:28).** The H1 DEMOTE was actioned by setting `pm.swing_selector` `'pead'`→`'ml_model'` (with `pm.swing_ml_live_enabled='false'`): the pre-market swing path now hits the dormant dead-ML branch and proposes nothing, so PEAD no longer trades live (`pm.pead_size_mult` left at 1.0 — moot, since PEAD isn't the active selector; the `pead_tracker` telemetry is retained). **Live book = trend-only (`pm.trend_enabled='true'` / `trend_shadow='false'` / 40%) + cash** — exactly the prescribed trend-plus-cash. Verified against live config 2026-06-12. The ONE remaining owner item from this decision is the SEPARATE trend-weight reconciliation (40%→25% risk framing); `pm.allocator_enabled` stays `'false'`.

**UPDATE 2026-06-12 — trend weight reconciled 40%→25% (owner-approved).** With trend now the SOLE live sleeve, `pm.trend_allocation_pct` set 0.40→**0.25** (the Track-B 25% framing, #451) — a CAPITAL-gross fraction (the sleeve is internally vol-targeted, so this is a conservative reconciliation of the gross budget, not a precise risk-parity match). Live value set immediately (applies at Monday 6/15's first real rebalance); the schema default, `set_trend_config.py` BASE, and the sleeve docstrings were brought in line so nothing silently reverts to 40%. `pm.allocator_enabled` stays `'false'` (static budget). This closes the H1 live-book follow-up; the live book is now trend (25%) + cash.

---

## 2026-06-11 — H1 interpretation PRE-COMMITMENT (logged BEFORE the run; integrity guard)

**Context**: PR3 built the event panel + CGM two-way-clustered inference and is about to fire the ONE-shot confirmatory run `H1-PEAD-EVENTLEVEL-20260611`. The independent Fable-5 review (F4) noted a structural gap: H1's PRIMARY return stream enters at **announce+1 open** (the registered frozen decision), but the LIVE PEAD book enters at **announce+2 open at the earliest** (`pead_scorer` requires `days_since>=1`; `AgentSimulator` fills next-day open after the scoring day). So a GRADUATE could be earned partly by the day-1 open→open drift the current live implementation forfeits.

**Pre-commitment (made BEFORE seeing any H1 number, to preserve pre-registration integrity — the frozen stat spec is unchanged):**
1. A **GRADUATE** verdict (primary 10d one-sided p<0.05) means "PEAD is a real event-level edge **at announce+1-open entry**" — and is **conditional on an execution upgrade** (a market-on-open order at announce+1) before any capital step. It does NOT by itself license the current announce+2 live path to capital.
2. At adjudication I will read the runner's reported **`entry_open_next2` (announce+2) cut** and the **`day1_momentum_gap`** alongside the primary. If the edge survives only at announce+1 and collapses at announce+2, the honest read is "edge exists but the live entry timing forfeits it" → an execution-timing fix is the prerequisite, not a capital increase.
3. **DEMOTE** (p>0.15) and **INCONCLUSIVE** (0.05–0.15) are unaffected by this nuance and stand as registered (demote → live book = trend-plus-cash; inconclusive → PEAD stays telemetry size).

This entry is the audit record that the announce+1-vs-announce+2 caveat was committed before the result, not rationalized after.

---

## 2026-06-11 — P0 finished: `--hypothesis-id` enforcement + `event_regime_sharpes()` (report-only) + H1/H2/H3 pre-registered (#454)

**Context**: The last two P0 stubs from the blueprint Phase-0 list, plus the Phase-3 pre-registration the blueprint requires "BEFORE the panel exists." Closes P0; the next work is the slow fuses (P1c NBBO logger + P2 greeks backfill) and the event panel / H1.

**Decisions**:
1. **`--hypothesis-id` registry enforcement** is now wired into all nine `run_*_cpcv` scripts via one shared helper (`scripts/walkforward/registry_enforcement.py`). `begin_run()` FAILS FAST (before the multi-hour fetch/CPCV) on an unregistered id, a confirmatory-but-not-preregistered id, an already-recorded id (R4 — its result could never be recorded), or a run starting at/before the prereg instant (R2 ordering). With no id it WARNS during the 2-week grace window (`GRACE_UNTIL=2026-06-25`) and REQUIRES an id (or `--exploratory`) on/after that date. `HypothesisRun.record()` is best-effort (a completed CPCV is never lost to a registry hiccup). **Behavior with zero new flags is byte-for-byte unchanged** except one warning line.
2. **`event_regime_sharpes()`** added to `scripts/walkforward/regime.py` — per-event (UN-annualized) cross-event Sharpe bucketed by entry-day regime, the instrument that will retire the event-sparsity waiver. EventEdge strategies now emit per-event `(entry_date, pnl_pct)` and `run_cpcv` surfaces the min as `CPCVResult.event_worst_regime_sharpe`. **It is REPORT-ONLY in the gate** — it does NOT feed `worst_regime_sharpe`/`regime_ok` and does NOT retire the waiver in PR1.
3. **H1/H2/H3 pre-registered** with FROZEN acceptance criteria (`scripts/preregister_event_hypotheses.py` → `H1-PEAD-EVENTLEVEL-20260611`, `H2-IMPLIEDMOVE-CONTINUOUS-20260611`, `H3-PEADV2-SCORECARD-20260611`; `preregistered_at=2026-06-11T12:00Z`, label=confirmatory). H1 decision rule (pre-committed): p<0.05 → PEAD graduates to honest Track-A paper (waiver retired); p>0.15 → demote to trend-plus-cash; 0.05–0.15 → inconclusive. H2: continuous reaction_ratio, NEGATIVE coeff t≤−2, no thresholds. H3: monotonic scorecard (NOT XGBoost), train 2022-24 / validate 2025-26, t≥2.

**Rationale / adversarial findings (Fable-5 loop)**: The first implementation WIRED the event-level Sharpe into the live gate as a fallback (fires whenever daily buckets are starved — exactly PEAD's case). The independent Fable-5 review flagged this as a BLOCKER and I agree: it compared a **per-event un-annualized** Sharpe against `MIN_WORST_REGIME_SHARPE=-0.5` (calibrated for **annualized daily** Sharpe). For PEAD's 10–40d holds, −0.5 in event units ≈ −1.25 to −2.5 annualized → the backstop becomes 2.5–5× looser and near-unbinding; it also silently removed the CAPITAL-tier human-sign-off fail-closed, and it **pre-empted H1's own pre-registered decision rule** (waiver retirement is H1's *consequence at p<0.05*, not PR1's). The blueprint (X6) says adopt it "inside Phase 3, once validated." → reverted gate consumption to **report-only**; the FIX-2 waiver path is unchanged. Also fixed (review): `begin_run` fail-fast on R4/ordering; `pnl_pct is None` guard in `event_edge.py` (a None would have silently dropped a whole fold); narrowed the prereg-script R1 catch.

**Consequences**: P0's measurement machinery is complete. No threshold changed; no live/gate verdict changed (the report-only field is informational). Each Hn gets ONE confirmatory run (R4); the H1 run lands in PR3 with `run_at > 2026-06-11T12:00Z`. Per CLAUDE.md the `scripts/walkforward/` touch updates PIPELINE_ARCHITECTURE (§7.0c + changelog; `GRACE_UNTIL` is a `registry_enforcement.py` constant, not a retrain_config feature flag, so §13 is unchanged). **NOTE — 2026-06-25 cutover:** any overnight/manual CPCV run after that date needs `--hypothesis-id` or `--exploratory`.

---

## 2026-06-11 — Track B budget amendment APPLIED (0.10→0.25, owner-approved registered): TSMOM now PASSES (#451)

**Context**: The first Track B run (#450) showed the 10% budget structurally rejects any realistic diversifier on ΔSharpe (TSMOM improved the PEAD book on every metric yet missed ΔSharpe by 0.0115). The owner approved raising the risk budget — a REGISTERED amendment, not ad-hoc tuning.

**Change**: `TRACKB_MAX_RISK_BUDGET` 0.10 → **0.25** (a quarter of book risk — the conservative end of the owner-endorsed 25–40% range; the validated Phase-2/3 book ran trend at ~40–50%). Chosen on PRINCIPLE, not to barely clear TSMOM: the budget→ΔSharpe sweep flips PASS at ~12.5%, so 0.25 is comfortably in the pass region (passes at any budget ≥ 12.5%). Full rationale in the retrain_config comment.

**Result (re-run, recorded as the registry re-test `TRACKB-TSMOM-VS-PEAD-20260611-AMEND25`, parent = the original)**: at 0.25, TSMOM **PASSES Track B on ALL 8 criteria** — Sharpe 0.411→0.640 (Δ+0.229), Calmar 0.278→0.588 (Δ+0.310), maxDD −5.75%→−3.7% (shallower), corr +0.274, tail-overlap 1/14, standalone vt-SR 0.92. No criterion newly binds at the higher budget. decision=park (book inclusion remains owner-gated — Track B never auto-promotes).

**Consequences**: Track B is now well-calibrated (the first real diversifier clears it with margin). The registry dogfooding worked end-to-end AND its integrity fired in the wild — a redundant concurrent re-run was correctly rejected by R1 (duplicate) + R4 (one-shot). PIPELINE_ARCHITECTURE §7.0-B (calibration RESOLVED) + changelog updated (gate-threshold change). **TSMOM is now Track-B-eligible for PAPER book inclusion; actually wiring TSMOM into the live book at a 25% weight is a SEPARATE owner decision.** Next P0: wire `--hypothesis-id` into the run scripts + `event_regime_sharpes()`; then event-level inference (P3).

---

## 2026-06-11 — Track B first real run: TSMOM vs PEAD book FAILS only on ΔSharpe@10% — calibration question answered; amendment PENDING OWNER (#450)

**Context**: First real application of the Track B gate — the registered open calibration question from 2026-06-10. `scripts/run_book_gate.py` ran `book_delta_gate(PEAD, TSMOM)` at the 10% risk budget on the 2020-06→2026-06 sleeve overlap (1495 evaluated days). Pre-registered + recorded in the research registry (first real ledger row `TRACKB-TSMOM-VS-PEAD-20260611`, decision=park — book inclusion is owner-gated, NOT auto-promoted).

**Finding (no constant changed)**: TSMOM **FAILS Track B on `sharpe_delta` ONLY** (7/8 criteria pass): ΔSharpe **+0.0885** vs the ≥0.10 bar. Yet it improves the book on EVERY metric — Sharpe **0.411→0.500**, maxDD −5.75%→−4.95% (0.81pp shallower), Calmar 0.278→0.371, lower vol; corr **+0.274** (<0.30), tail-overlap **1/14** (clean crisis profile), standalone vol-targeted SR **0.92**. The gate's math is correct; the **calibration** is the issue: ΔSR ≈ budget·(SR_cand − corr·SR_book), so a 10% slice at +0.27 corr against a 0.41-SR base implicitly demands SR_cand ≳ 1.11 — unreachable for any realistic diversifier. **The ΔSharpe-at-fixed-10%-budget bar rejects a sleeve that demonstrably diversifies.**

**OPEN — a *registered* amendment is the OWNER's call (NOT made here; the discipline forbids ad-hoc tuning).** Options to weigh: (a) **raise the risk budget** — the Phase-2 validation used ~40-50% where PEAD+trend Sharpe went 0.31→0.92; 10% is a very thin slice and the likeliest mis-calibration; (b) **evaluate at the budget that maximizes book Sharpe** and gate on the improvement curve rather than a single 10% point; (c) **gate on a composite "book improves"** (Sharpe OR Calmar OR maxDD) since TSMOM clearly passes the latter two; (d) lower the ΔSharpe bar (threshold-shopping risk — least preferred). No constant changed; the result stands as recorded.

**Consequences**: The two-track machinery + registry worked end-to-end (the gate produced the correct math; the registry recorded the first confirmatory run with valid pre-registration). The amend-or-not decision (and which option) awaits the owner. `scripts/run_book_gate.py` added (reusable). ML_EXPERIMENT_LOG + PROJECT_STATE updated.

---

## 2026-06-11 — Research registry shipped: the pre-registration ledger = the program's true N_TRIALS (#449)

**Context**: Third Alpha-v6 P0 unit. DSR is report-only (it can't represent iterative human/LLM research), and CPCV protects a single run — not the research PROGRAM. The real multiplicity defenses are pre-registration + a registry + the forward sacred holdout. This builds the registry: `app/research/registry.py::ResearchRegistry` (sqlite, env-isolated) + `scripts/registry.py` CLI.

**Decision**: Every experiment is recorded; confirmatory runs must be pre-registered. Enforced integrity rules (each raises): **R1** unique `hypothesis_id`; **R2** a confirmatory result requires `preregistered_at` STRICTLY before `run_at` AND non-empty `acceptance_criteria`; **R3** exploratory results can NEVER promote (kill/park/exploratory_only only); **R4** one result per hypothesis (a re-test = new id + `parent_id` + a `cooling_off_until` preceding the re-test's `run_at`); **R5** pre-registration is immutable / not post-hoc. Unknown labels + orphan parents fail closed; concurrency-safe (single-transaction guarded UPDATEs). Going forward, confirmatory WF/CPCV/panel runs register a `hypothesis_id`; the `--hypothesis-id` enforcement wiring into `run_*_cpcv` is a follow-up PR.

**Rationale / adversarial findings**: Built + 2× Fable-5 review. 1 BLOCKER + 3 MAJOR found+fixed: (BLOCKER) conftest didn't isolate the registry DB → a bare construction in a future test would pollute the real ledger; (MAJOR) criteria-less confirmatory promotion (R2 checked only the timestamp); cooling-off compared to caller-`now` instead of `run_at` (a run executed DURING cooling-off could be recorded); a check-then-write TOCTOU on the one-shot UPDATE. All probed closed under raw-SQL adversarial testing. 45 tests.

**Consequences**: Additive (no existing behavior changed beyond a 1-line conftest isolation addition). Timestamps are caller-supplied — the registry is honest-recording infrastructure (a self-reported ledger), with the forward sacred holdout (2026-11-09) as the tamper backstop. Not a PIPELINE-rule file → `PIPELINE_ARCHITECTURE.md` unchanged. **P0 now has its three core units shipped** (calibration harness #444 + result #447; Track B gate #448; research registry #449). Remaining P0 / early-P3: the `--hypothesis-id` run-script wiring + `event_regime_sharpes()` + bringing forward event-level inference.

---

## 2026-06-10 — Track B (book-delta) acceptance gate shipped — the two-track framework's first half (#448)

**Context**: The P0 gate-calibration RESULT (below) localized the false-negative to the worst-regime backstop and prescribed two-track acceptance. This builds **Track B**: `scripts/walkforward/book_gate.py::book_delta_gate` — a PURE gate that judges a candidate sleeve (risk-premium / diversifier / tail-hedge) on its contribution to the COMBINED book at a ≤10% risk budget, not on the standalone significance gate.

**Decision**: Adopt two-track routing (`component_type` → Track A significance gate for `alpha`; Track B book-delta for `risk_premium`/`diversifier`/`tail_hedge` — PIPELINE_ARCHITECTURE §7.0-B). Track B PASS iff all 8 pre-registered criteria hold (Sharpe Δ≥0.10, Calmar Δ≥0, maxDD not deeper, corr<0.30, standalone vol-targeted SR∈(0.20, 3.0], risk budget≤10%, tail-overlap≤0.30). Track B gates **PAPER-level book inclusion only — NEVER auto-promotes to CAPITAL** (owner sign-off + tail budget). Constants frozen in retrain_config (`TRACKB_*`), pre-registered 2026-06-10.

**Rationale / adversarial findings**: Built + 2× Fable-5 review. Three MAJOR bugs found+fixed: (1) a 5× leverage cap broke vol-target invariance → removed (leverage now bounded by the 2% floor; invariance to ~4e-16); (2) the joint-tail criterion was first implemented as a mean-of-tail-returns test — BOTH maskable (one lucky day → false ADMIT of a tail-amplifier) AND ~43% false-REJECT on independent diversifiers → replaced with the blueprint's actually-REGISTERED **overlap** test (candidate's worst days must not coincide with the book's; independent false-reject 0/100 in re-review); (3) no implausibility ceiling → added (reuses `SHARPE_IMPLAUSIBILITY_CEILING`). 19 tests; metrics reuse `sleeve_allocator.combine()` (identical to the book harness).

**Consequences**: Track A (significance gate) UNCHANGED; Track B is additive (no live/gate code touched). **Open calibration question (registered):** at a 10% budget, ΔSharpe≥0.10 implicitly demands standalone SR ≈0.94 (corr 0) / ≈0.70 (corr −0.3), so TSMOM (SR≈0.71, corr≈+0.25 to PEAD) may be structurally rejected — resolve via a registered amendment after the first real TSMOM-vs-book run, NOT ad hoc. Per CLAUDE.md, PIPELINE_ARCHITECTURE §7.0-B + changelog updated this PR. Next P0: the research registry (pre-registration ledger) + bringing forward event-level inference (P3).

---

## 2026-06-10 — P0 gate-calibration RESULT: do NOT lower the significance bar; the false-negative lever is the worst-regime backstop (→ two-track acceptance) and the path-t is unreliable (→ event-level inference)

**Context**: Ran the pre-registered calibration controls (the #444 harness) through the production gate. Positive PAPER pass-rate **0/4** (confirms a real Type-II problem for the edges we care about) but significance-CORE pass-rate **2/4** — `tsmom_4y` (t=6.72) and `tsmom_19y` (t=4.46) clear the significance core and fail ONLY on `worst_regime_sharpe`. Meanwhile **3/5 TRUE zero-SR balanced nulls posted t=2.6–3.5** (≥ the 2.0 bar), and PEAD's t=3.33 is statistically indistinguishable from a noise null (t=3.47). The pre-registered recalibration rule self-returned **`NO_ADMISSIBLE_TSTAR`** (the binding failure is not the t-stat). Type-I control is sound (0/10 nulls pass the full gate; the leaky control is flagged implausible). Full OC table in ML_EXPERIMENT_LOG; artifact `logs/gate_calibration_20260610.json`.

**Decision**: P0's next step is **NOT** a t-threshold recalibration (empirically shown to admit noise). Instead: **(a)** build the **two-track acceptance** gate (Track B book-delta) so crisis-diversifiers / risk premia (TSMOM, index VRP) are judged on book contribution, not a standalone worst-regime floor — this is the lever that actually unblocks TSMOM; **(b)** treat the 8-fold CPCV path t-stat as non-promotion-grade for event/series strategies and bring forward **event-level / cluster inference** (P3) as the significance instrument. **No gate threshold is changed.**

**Rationale**: The calibration **refuted the blueprint's specific "the t≥2.0 bar is arithmetically too strict" hypothesis** and localized the real false-negative to the worst-regime backstop. The harness did its job — it prevented a plausible-but-wrong change and pointed at the two levers the blueprint already names (two-track acceptance + event-level inference). The result also empirically validates the reviewers' warning that the CPCV path t-stat (correlated paths, N_eff=8) is not a clean discriminator — pure-noise nulls cleared it.

**Consequences**: Blueprint P0 framing recast (the "recalibrate t*" sub-task becomes "two-track acceptance + a better significance instrument"); ML_EXPERIMENT_LOG carries the OC table; PROJECT_STATE updated. No code/gate change this entry → PIPELINE_ARCHITECTURE §7 thresholds untouched (the harness + §7.0a landed in #444). Next concrete P0 PR: the Track-A/Track-B two-track acceptance scaffold (`book_gate.py`) + research registry.

---

## 2026-06-10 — Alpha-v6 direction: fix the ruler (two-track acceptance + event-level inference) + options-as-signal; calibrate first

**Context**: Five independent world-class-quant LLM reviews of the 2026-06-10 package (Gemini, DeepSeek, Grok, ChatGPT, Claude — archived under `docs/reference/prompts/20260610_Quant_Options_Review/responses/`) were synthesized by a Fable 5 deep-dive and grounded against the code. They **converge** on one diagnosis: the harness, hardened so far against *inflation*, is now a **Type-II / false-negative machine** — a t≥2.0 gate on N_eff≈6–8 folds of ≤4y data rejects *true* Sharpe-0.5–0.7 edges (t ≈ SR·√years), so 100% KILL (including the confirmed-real index VRP, PF 2.24/1.75) is the signature of a miscalibrated ruler, not an empty opportunity set. Second: the 4y options store is a **signal/information asset, not an options-execution edge** — OPT-3 died on single-name spreads; OPT-5 (signal-only) produced the program's only alpha-like lift. Third: for event strategies the independence unit is the announcement-day cluster, not the fold.

**Decision**: Adopt **Alpha-v6** (SSOT: `docs/reference/NEXT_PHASE_BLUEPRINT_2026-06.md`), a 7-phase plan that fixes the *measurement* and aims the options data signal-first. **P0** calibrate the ruler — positive/negative gate controls (TSMOM-on-4y is the decisive control), **two-track acceptance** (Track A standalone alpha / Track B book-delta diversifier), research registry + pre-registration, DSR→report-only, `event_regime_sharpes()`. **P1** live-book fidelity (StrategySpec replay-diff, fill-quality table, nightly NBBO→calibrated spreads, multi-factor residual-α). **P2** options feature layer (persist computed greeks, surface-quality reader, BMO/AMC event-time snapshots). **P3 (centerpiece)** earnings-event panel + event-LEVEL PEAD inference (two-way clustered announce-day×firm) + PEAD v2 continuous options-conditioned scorecard. **P4** options-as-equity-signal XS sleeve (CPIV/skew/O-S/term-slope/IV-RV, executed in equities). **P5** trend broadening (Track B, judged on 19y). **P6** index-VRP micro-sleeve behind the book gate (after sim-mechanics + spread fixes). **Reconciliations:** dispersion = feature-first / trade-maybe-never (Claude's cost-wall objection wins over Gemini/DeepSeek/Grok on the app's own OPT-3 evidence); CPCV **demoted to robustness** for event strategies (not killed — kept primary for path-dependent trained models); DeepSeek's DSR `N_eff=k(k−1)/2` fix **rejected** (makes DSR less conservative — wrong direction).

**Rationale**: Every KILL/KEEP verdict is only as trustworthy as the gate; until the positive controls run, every historical KILL carries an unknown false-negative rate. Two-track acceptance is the correct ruler for risk premia/diversifiers (the 2026-06-09 short-vol pause discovered this narrowly — Alpha-v6 generalizes it). Event-level inference uses *hundreds* of independent announcement days instead of 8 fold-Sharpes — the single highest-EV measurement upgrade, and it can re-adjudicate the live PEAD edge either way. Options-as-signal sidesteps the spread wall entirely.

**Consequences**: This **supersedes the open-ended "choose next research direction" item** in `PROJECT_STATE.md` and **sharpens (does not reverse) Alpha-v4** (sleeves + allocator remain the substrate). First move = **P0 TSMOM-on-4y control** — the cheapest experiment that can falsify the whole premise. **Reviewer corrections logged:** the forward sacred holdout DOES exist (`SACRED_HOLDOUT_START=2026-11-09`, enforced in code); a quarter-level event bootstrap already exists (`pead_significance.py` — the over-conservative end; day-level panel is the upgrade); `EventEdgeStrategy`/options adapter/allocator are already built. **Doc drift fixed:** `CLAUDE.md` DSR quick-ref 250→**300** (code SSOT `retrain_config.py`). Per the NO-DRIFT rule, the blueprint §6 lists each phase's required doc updates; `MASTER_BACKLOG.md` + `PROJECT_STATE.md` updated this PR. No code/gate change yet → `PIPELINE_ARCHITECTURE.md` untouched until P0 lands (§7 gate-calibration + §7.0-B two-track spec).

---

## 2026-06-10 — Regime-model retrain: weekly cadence, fixed 3-class gate, one shared evaluator (revived from abandoned PR #240)

**Context**: Investigating stale branches surfaced that `feat/phase-n-audit8b` (PR #240, closed unmerged ~3 weeks ago) carried a real, never-landed fix. Verified against main: **(1)** `scripts/train_regime_model.py` is BROKEN — it reads `payload["wf_auc_min"]`/`["brier_score"]` from the saved **pickle**, but `regime_training.py` only writes those keys to the **DB row** (the pickle has `wf_log_loss_mean`/`wf_macro_f1_mean`) → `KeyError` whenever the script runs. **(2)** The gate cutoff `brier < 0.22` is a 2-class Brier value mis-applied to the Regime-V2 **3-class cross-entropy log-loss** (random baseline = log(3) ≈ 1.099; v5 mean = 0.358) — wrong metric, wrong threshold. **(3)** There is no automated regime retrain at all — `regime_model_v5` (live, carries book sizing) only retrains via the broken manual script, and was 20 days stale.

**Decision**: Re-implement on a fresh branch off main (NOT merge the 345-behind branch). **(a)** Write the gate inputs into the pickle (`wf_auc_min`=macro_F1 min, `brier_score`=log_loss mean, repurposed names kept for back-compat). **(b)** Introduce ONE shared `regime_gate(payload)` (in `regime_training.py`) used by BOTH the CLI and the PM — so the threshold can never drift into two copies again (the root shape of the 0.22-vs-0.45 bug) — backed by config constants `REGIME_GATE_MACRO_F1_MIN=0.60`, `REGIME_GATE_LOG_LOSS_MAX=0.45`. It reads with safe defaults so a missing/garbage payload FAILS the gate rather than raising. **(c)** Add `PortfolioManager._retrain_regime`, scheduled weekly at 17:30 ET on a FILE-AGE cadence (`REGIME_RETRAIN_INTERVAL_DAYS=7`) **independent of `RETRAIN_WEEKDAY`** — so the regime model stays current even while swing/intraday retraining is frozen (Alpha-v4 P0). Gate-failed models are deleted so the filename-based loader keeps the prior passing version.

**Rationale**: The regime classifier feeds live sizing, so a silently-broken retrain path is an operational gap. Centralizing the gate + putting thresholds in config is the durable fix (same "single source of truth" principle as the test-mode detector above). The weekly file-age cadence (vs a fixed weekday) means a missed 17:30 window self-heals the next day, and the model never exceeds ~8 days old.

**Consequences**: `train_regime_model.py` works again; the regime model auto-retrains weekly with a correct 3-class gate. Known minor: on gate failure the freshly-trained pickle is deleted but its `RegimeModelVersion` DB audit row remains (the loader is filename-based, so this is cosmetic). Per the CLAUDE.md rule, `PIPELINE_ARCHITECTURE.md` changelog updated (retrain_config gate-threshold change). The abandoned `feat/phase-n-audit8b` branch can now be deleted. 9 tests (`tests/test_regime_retrain.py`).

---

## 2026-06-10 — Hardening sweep: centralize test-mode detection (one subprocess-safe `is_test_mode()`) — prevent the whole test→prod-bleed class

**Context**: After fixing the kill-switch false-ACTIVE log (#434) and the log-isolation leak (#435), an Opus deep-dive audited the codebase for the *same family* of bug — test behaviour/output bleeding into production resources, and the mock-fragility that enables it. Findings:
- **Inconsistent, fragile test-mode detection at 3 sites**, each rolling its own check: `_DailyFileHandler._prefix` and the notify-watcher-skip (both `PYTEST_CURRENT_TEST` env **or** `pytest` in `sys.modules`), and — worst — `kill_switch._running_under_pytest`, which keyed on `os.environ.get("_")` (a unix-ism rarely set on Windows) and **guards real DB/audit writes**. All three are runtime-only signals that do **not** survive a process boundary (Windows `spawn` → fresh interpreter, env may lack `PYTEST_CURRENT_TEST`), the exact fragility behind the log leak.
- **Main-DB test isolation is achieved by patching `get_session` → `MagicMock`** in the `test_client` fixture (not a global DATABASE_URL override). This is what produces the `MagicMock` objects seen in mocked startups, and is the upstream source of the `bool(mock)`/comparison-on-mock surfaces (kill_switch.load_state #434, the startup `>`/`>=` errors, #429).
- **Verified-robust (no action needed):** `get_agent_config` coerces int/float with try/except→default (and there are currently **no** bool-typed configs, so the uncoerced-bool path is unreachable today); `risk_manager` peak-equity restore wraps `float(val)` in a broad except; no other unguarded `bool(config)` surfaces remain after #434.

**Decision**: Introduce **one** authoritative detector, `app.utils.runtime.is_test_mode()` (stdlib-only, import-safe from early logging code), with `MRTRADER_TEST_MODE` as the PRIMARY signal (inherited across spawns) and the runtime signals as fallback. Route all three sites through it — notably hardening the kill-switch persist/audit guard, which could previously have persisted `kill_switch.active=True` to the real config store / written a real audit row from a subprocess or Windows test.

**Rationale**: A single shared detector stops detection logic from drifting per-site (the root reason three subtly-different, subtly-broken copies existed) and makes "is this the test session?" correct on both sides of a process boundary everywhere. The env-var-primary design is the same pattern already proven for per-worker DB isolation (`MRTRADER_*_DB`) and the log prefix (#435).

**Consequences**: The test→prod-bleed class (logs today, and the latent DB/audit-write path) is closed at its common root. 5 new tests (`test_runtime_test_mode.py`) incl. the subprocess case and the kill-switch guard delegation. **Documented residual (not fixed here, lower priority):** main-DB isolation still relies on per-fixture `get_session` patching — a test path that uses `get_session` without the `test_client` fixture, or a real subprocess app-boot, would hit the configured DATABASE_URL; a global test-DB override would be more defense-in-depth but has broad blast radius. Not a WF/CPCV change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-10 — Test isolation: route logs by an inherited env var so subprocess app-boots can't leak into the live log

**Context**: The kill-switch false-ACTIVE log (above) was traced to a pytest run whose app-boot wrote into the **production** `mrtrader_<date>.log`. Empirically the isolation works ~99.9% of the time (a full suite produced 915 startup banners, all correctly routed to `test_mrtrader_<date>.log`); the leak is one specific path. `_DailyFileHandler._prefix()` decided the prefix from `PYTEST_CURRENT_TEST` (env, per-test) **or** `"pytest" in sys.modules`. Both are runtime signals that **do not survive a process boundary**: a pytest-spawned subprocess (Windows 'spawn' → fresh interpreter, no `pytest` imported, possibly no PYTEST_CURRENT_TEST) booting `app.main` falls through to the LIVE prefix.

**Decision**: Add `MRTRADER_TEST_MODE=1`, force-set in `conftest.py` at import (before any test imports `app.main`), and make `_prefix()` check it FIRST. Env vars are **inherited by spawned children**, so any app boot under the test session — in-process or subprocess — routes to the isolated log. The two runtime signals are kept as belt-and-suspenders. Same env-propagation pattern already proven for the per-worker DB isolation (`MRTRADER_*_DB`).

**Rationale**: The root fragility is relying on in-memory state (`sys.modules`) for a cross-process decision. An inherited env var is the only signal that is correct on both sides of a spawn. Production is unaffected (conftest never runs, the var is never set → live prefix; covered by an explicit "no signal → live prefix" test). 2 new regression tests in `test_log_isolation.py` (the subprocess blind spot + the production-default).

**Consequences**: Closes the test→live-log leak class deterministically. Part of a broader test/prod-bleed and mock-fragility hardening sweep (see ML_EXPERIMENT_LOG / this date). Not a WF/CPCV change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-10 — Reliability: KillSwitch.load_state() strict-bool — never coerce a malformed persisted value to ACTIVE

**Context**: The live log showed `Kill switch restored as ACTIVE from persisted state` at a 09:41 startup, yet the kill switch had **never been activated** — the persisted config (`kill_switch.active`) was a real bool `False` (last written 2026-05-29) and the only kill-switch audit event ever is an April test-reset. Root cause: that "startup" was a **pytest run that leaked into the production log** (its banner carried `MagicMock` objects), and `KillSwitch.load_state()` did `self._active = bool(val)`. The mocked config store returned a `MagicMock`, and `bool(MagicMock())` is `True` → it falsely logged "restored as ACTIVE." The same `bool()` coercion would misfire in **production** for any non-bool value: a legacy/corrupted string `"false"` also evaluates `bool("false") == True`, which would spuriously HALT live trading on startup. (Same "unguarded `bool(mock)` surface" family as #429.)

**Decision**: Harden `load_state()` to a **strict `isinstance(val, bool)`** check. A genuine `activate()`/`reset()` always persists a real JSON bool, so a clean `True` is never lost; any non-bool is malformed → log a WARNING and treat as **INACTIVE**. We deliberately fail toward *not* halting on garbage rather than fail toward a spurious halt: we only ever ignore values that could not have come from a legitimate activation, and a real emergency activation (clean bool `True`) is always honored.

**Rationale**: A kill switch is a safety device, but a spurious self-engaging halt on every restart (from a mock under test, or one bad config row in prod) is itself a serious reliability failure and erodes trust in the control. Strict-bool removes the only path by which a non-activated switch could read as active. 6 regression tests (`tests/test_kill_switch_load_state.py`) cover the exact MagicMock case + string `"false"`/`"true"` + genuine bool True/False + missing row.

**Consequences**: Startup can no longer be tricked into a false halt by a malformed `kill_switch.active`. Live trading was never actually halted by this (the real value was always False) — the only prior effect was misleading log noise. **Separately noted (not fixed here):** a test exercising the startup/lifespan path still logs to the production `mrtrader_<date>.log` instead of the isolated `test_mrtrader_<date>.log` — a log-isolation gap worth a follow-up. Not a WF/CPCV change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-10 — Live trading: PEAD execution now honors the 5% telemetry position cap

**Context**: Pre-Monday verification (Opus live-path review) found the Trader re-sizes every entry via the generic `size_position` (2%-risk / ATR-stop, hard-capped at the global `MAX_POSITION_PCT=0.10`) and **discarded** the PM's PEAD-ramped `proposal["quantity"]`. So `pm.pead_max_position_pct=0.05` (the owner's deliberate Alpha-v4 telemetry-size decision) and `pm.pead_size_mult` were **inert at execution** — live PEAD names could be sized to the global 10% cap, double the validated 5% telemetry size. (The earlier #420 fix corrected the PEAD *stop* used for sizing — PEAD's own 0.5×ATR vs the swing stop — but not the position-% cap.)

**Decision**: `size_position` gains a `max_position_pct` parameter (default `MAX_POSITION_PCT`, so non-PEAD is byte-identical). For PEAD entries the Trader reads `pm.pead_max_position_pct` / `pm.pead_size_mult` and passes the 5% cap plus `risk_fraction = RISK_FRACTION * size_mult`. The caps now match the PM's `apply_pead_size_ramp` formula exactly (`account_value * max_position_pct / price`).

**Rationale**: The owner set 5% intending PEAD capped at 5%; execution silently ignoring it is a fidelity bug, not a design choice. With `size_mult=1.0` (current) the only behavioral change is the cap (10%→5%) when it binds (tight PEAD stop → large risk-based size → cap binds), making live PEAD position size match its validated telemetry size. Opus diff re-review: correct + safe, no regressions (REST-config read robust; conviction multiplier inert once the cap binds; non-PEAD unchanged).

**Consequences**: live PEAD per-name size is now ≤5% of equity (was ≤10%) — paper-only, more conservative, faithful to the validated book. 2 sizing tests added. Not a WF/CPCV change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-10 — Alpha-v5 OPT-5: implied-move filter is THRESHOLD-FRAGILE — do not pursue as-is (overfit-suspect)

**Context**: The 2026-06-09 OPT-5 decision (below) flagged the implied-move PEAD filter as a "promising lead, unconfirmed" and listed the explicit remaining confirmation: **threshold robustness**. #423 found the lift at a SINGLE threshold (realized/implied < 1.0) on a thin 2y/8-fold sample — a textbook multiplicity/overfit risk. `scripts/run_pead_implied_threshold_sweep.py` swept the baseline + filter at {0.75, 1.0, 1.25} on the same window (PEAD's own CPCV, k=8/p=2). A real effect should PLATEAU across thresholds; an artifact SPIKES at one.

**Decision**: **Do not advance the implied-move filter toward deploy on the current evidence.** The lift is concentrated entirely at 1.0 (Δmean +0.455, Δhedged +0.551); it is marginal at 0.75 (Δmean +0.098, below the +0.10 bar) and goes **clearly negative at 1.25** (Δmean −0.264, Δhedged −0.311 — worse than no filter). That is a single-threshold spike, not a plateau → **FRAGILE / overfit-suspect**. The scorer hook stays default OFF (unchanged). If revisited, gate behind **(1)** a demonstrated PLATEAU on a powered (4y R1K) re-run AND **(2)** a single PRE-REGISTERED threshold (no post-hoc 1.0-picking).

**Rationale**: Robustness is necessary, not sufficient — and the filter failed it. Even at the best threshold (1.0) the alpha is not statistically established (resid-α t 0.65 < 1, DSR saturated, per-arm CPCV gate fails on tstat/pct_positive/p5), so a fragile lift at an in-sample-selected threshold is the weakest possible evidence. A ±0.25 perturbation that flips the sign is the signature of fitting noise on ~7-12 trades/fold. This also **lowers the priority of the Phase-4 4y-data refactor**: more data resolves power, but this sweep already shows the parameterization is fragile, so power alone wouldn't rescue it without a pre-registered threshold. An Opus deep-dive of the sweep harness caught a critical bug (hedged Sharpe read from a non-existent `residual_alpha_sharpe` attr → always NaN → verdict hardcoded to FRAGILE regardless of data); fixed (real attr is `residual_sharpe`), verdict hardened to report missing-hedged honestly, classifier extracted as a pure tested function (17 tests) — so this FRAGILE verdict is genuinely data-driven (hedged populated, n=283), not the bug artifact.

**Consequences**: The OPT-5 implied-move filter line is parked as overfit-suspect (not killed — a powered, pre-registered re-test could revive it). The reusable artifacts remain: `ImpliedMoveProvider`, the scorer hook (OFF), and now a general threshold-robustness sweep harness + tests. Not a WF/CPCV pipeline change → `PIPELINE_ARCHITECTURE.md` untouched. See `ML_EXPERIMENT_LOG.md` (OPT-5 threshold robustness, 2026-06-10).

---

## 2026-06-10 — Live trading: harden the trend-sleeve rebalance for its first real run (Mon 2026-06-15)

**Context**: Pre-Monday verification (two independent Opus deep-dives of the live PEAD and trend paths) found the system has **no hard blocker** — PEAD and trend both fire correctly — but surfaced two latent reliability risks in the never-yet-run trend rebalance path: **(F1)** `run_trend_rebalance` deferred its `db.commit()` to *after* the order-placement loop, so a crash/restart mid-loop orphaned already-placed Alpaca ETF positions with uncommitted Trade rows — which startup reconciliation then adopts as `trade_type="swing"` with a synthetic 2%/6% stop/target, letting the Trader liquidate a trend leg mid-week and re-buy it next Monday (double-trade). **(F2)** the shared `schedule_daily_at_time` used `misfire_grace_time=60`; for the *weekly* (Monday-only, in-handler-gated) trend job a >60s-late fire at 09:45 (busy loop / restart) makes APScheduler DROP the entire week's rebalance.

**Decision**: **(F1)** commit the `PENDING_FILL` Trade row immediately **per order** (with `db.rollback()` on failure) instead of one deferred post-loop commit — shrinking the orphan window from "the whole loop" to "one in-flight order." **(F2)** parameterize `schedule_daily_at_time(misfire_grace_time=…)` (default unchanged at 60) and give the trend job **1800s** (30 min) — TSMOM is a slow signal so a late fire is benign, whereas dropping the week is not. Also set `coalesce=True` explicitly (a no-op — APScheduler already defaults it).

**Rationale**: Both are defense for the *first-ever* live rebalance. The per-order commit is strictly better than before (no scenario is made worse) and the residual one-order window is inherent to any place-then-record flow. The grace bump only affects the trend job's drop-threshold; the in-handler weekday + market-open guards + natural target==current idempotency mean even a spurious late/re-fire places zero wrong orders. Opus re-review of the diff: correct and safe, no BLOCKER/HIGH.

**Consequences**: Monday's first trend rebalance is restart-safe and won't be silently skipped by a slightly-late fire. 4 new tests (per-order commit count, mid-loop order-failure durability, commit-failure rollback, scheduler grace). No WF/CPCV pipeline change → `PIPELINE_ARCHITECTURE.md` untouched. A separate follow-up tracks PEAD live-sizing fidelity (the Trader re-sizes via the generic 10%-cap `size_position`, not honoring `pm.pead_max_position_pct=0.05`).

---

## 2026-06-09 — Alpha-v5 Options Program PAUSED after reassessment: our gate is an ALPHA gate, short-vol is a RISK PREMIUM

**Context**: After two Opus-certified KILLs (OPT-3 single-name earnings IV-crush = cost-killed; OPT-4 index short-vol = VRP real + cost-robust PF 2.24/1.75 but standalone Sharpe ~0 + under-powered), the owner asked to reassess rather than keep building.

**Assessment**: The program *succeeded as research* — clean, Opus-certified harness (engine validated vs live snapshot; PIT + survivorship data, 45M bars; contract sim; CPCV adapter) and two trustworthy, theory-consistent verdicts. The decisive insight: **the significance gate (Sharpe≥0.8, %pos≥75%, path-t>2, min-fold≥−0.3, Calmar≥0.3) is an ALPHA gate** — built to validate high-Sharpe, uncorrelated, fat-tail-free equity alpha and to reject the ranker-line false positives. **Short-vol is a RISK PREMIUM** — moderate Sharpe (~0.5 unlevered), negatively skewed by construction (paid to bear the crash tail) — so it fails an alpha gate *even when legitimate*. Index short-vol's KILL is therefore *expected*, not proof it's worthless; it was measured with the wrong ruler. The correct ruler for a diversifier is **book-level contribution** (does a small, vol-targeted, regime-overlaid index short-vol sleeve improve the combined PEAD+trend book — short-vol is crisis-negative, trend crisis-positive, natural complements) — the OPT-6 allocator question never reached.

**Decision**: **PAUSE the options BUILD.** Validating "book-additive diversifier" properly needs (a) a new book-level acceptance framework (standalone significance is the wrong lens for a risk premium), (b) the tail-management overlay, and (c) MORE DATA — 4y (2022–26) is ~one vol cycle and short-vol's risk is the rare crisis 4y barely samples; tuning to a KEEP on thin data is the overfitting trap the project guards against. The validated live book (PEAD + trend) is the current value; options would be a marginal, lower-conviction add requiring significant further investment. Bank the harness + findings (permanent, cheap to revisit).

**Consequences**: options *build* paused after OPT-0..OPT-4 (all merged). **Owner disposition (2026-06-09)**: (1) **keep** the $79/mo Polygon Options data; (2) do **OPT-5 (options-data-as-signal)** as the sanctioned parting win — use the data to enhance the *validated* sleeves (implied-move/priced-in filter for PEAD, put-skew risk-off conditioner), judged on the **host sleeve's existing gate** (no options execution, no alpha-gate-vs-risk-premium mismatch); (3) **redirect near-term focus to hardening the live book** (PEAD live-sizing fidelity fix — currently sized off `generate_signal`'s swing ATR stop instead of PEAD's own; `pead_vix_conf_ref` guard; verify trend's first real paper fills Mon 2026-06-15). Revisit standalone short-vol later as a book diversifier with a risk-premium acceptance framework + more data. Reusable assets retained: `app/options/*`, `app/data/options_provider.py`, `app/backtesting/options_simulator.py`, `scripts/walkforward/options_strategy.py`, `scripts/backfill_options.py`, `validate_options_engine.py`.

---

## 2026-06-09 — Alpha-v5 OPT-4: index/ETF systematic short-vol — VRP is real + cost-robust but Sharpe-weak (KILL standalone)

**Context**: OPT-3 killed single-name earnings short-vol (cost-killed). OPT-4 tests where the VRP is documented to be positive: systematic short iron condors on SPY/QQQ/IWM (monthly ~35-DTE, 21-day hold), reusing the OPT-2 simulator + CPCV via a new `IndexShortVolStrategy` (subclasses `OptionsStrategy`; scheduled entries + realized-vol expected-move instead of earnings events). Ran raw (no overlay) first to measure the unconditional VRP, with the mandatory 1×/2× spread-stress sweep.

**Finding**: **KILL standalone, but materially better than OPT-3.** Strike sizing was decisive — short strikes at 1.0× realized-SD (≈32% breach) are structurally negative (PF 0.78, Sharpe −0.44); at the canonical **1.5× realized-SD (≈16-delta)** PF flips to **2.24 @1× and 1.75 @2×** (Sharpe +0.04, 56% positive folds). So **the index VRP is real and cost-robust** — it survives the 2× spread stress that killed single-name earnings vol, because index option spreads are ~pennies. **But it is risk-adjusted-flat** (mean Sharpe ~0, path-t ~0, residual-α t −1.25 — the crisis fat-tail eats the vol-adjusted return) and **under-powered** (7-fold low coverage on 3 ETFs/monthly), so it fails the significance/Sharpe gate. An Opus 4.8 look-ahead review **certified PF 2.24 genuine** (no leak, correct realized-vol→strike units, behavior-preserving `_select_condor_legs` refactor).

**Decision**: log the KILL (a success of the harness) and STOP parameter exploration here to avoid overfitting a thin sample (3 structures tested across OPT-3/4: 1.0×, 1.3×, 1.5× — all economically motivated, all logged). The planned refinements (regime/VIX de-risk overlay to cut the crisis tail → lift Sharpe; weekly cadence + more ETFs → power) are the path to a possible KEEP but are a larger, overfitting-prone build on thin data — deferred to an **owner checkpoint**.

**Consequences**: the options program has now shown **single-name earnings VRP = cost-killed; index VRP = real + cost-robust but Sharpe-weak**. This reframes the program: the edge exists at the index level and survives costs, but extracting gate-clearing *risk-adjusted* return needs the tail-management overlay (and more power). Reusable harness shipped (`build_index_short_condor`, `IndexShortVolStrategy`, shared `_select_condor_legs`); a new options strategy is still just a builder + runner. Not a WF/CPCV-core change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-09 — Alpha-v5 OPT-5: options-implied "priced-in" filter IMPROVES PEAD (promising lead, unconfirmed — not deployed)

**Context**: With the standalone options sleeves paused, OPT-5 uses the options data as a SIGNAL to enhance the validated PEAD sleeve (judged on PEAD's OWN CPCV gate — no options execution, so the alpha-gate-vs-risk-premium mismatch doesn't apply). The implied-move "priced-in" filter: skip PEAD entries whose realized announce-day move was within the pre-earnings IMPLIED move (realized/implied < 1.0) — i.e. the surprise was already priced by options. Required re-backfilling options for the R1K universe (60.8M bars, 100% PEAD-name coverage; the prior 47-name backfill covered only 6% of PEAD signals).

**Finding**: On the 2y options-covered window the filter **improves PEAD** — mean Sharpe **0.891 → 1.346** (+0.45), path-t **1.56 → 1.90**, Avg PF **2.09 → 2.52**, Calmar 5.4 → 8.6 (%pos 74.1% unchanged). This is the **opposite** of the earlier *price-based* priced-in filter (which hurt — "large gaps have the strongest drift") and the **first positive options signal** of the program: normalizing the reaction by the option-implied move (rather than raw % move) appears to separate genuine surprises from priced-in ones better.

**Decision**: log it as a **PROMISING LEAD, NOT a deploy.** Originally four caveats; the alpha-vs-beta one is now resolved (see Update). Remaining: (1) thin sample (7-8 folds / 2y, BEAR:1/NEUTRAL:6 — no bull coverage; DSR saturated → no selection screen); (2) single threshold (1.0) → multiplicity/overfitting risk; (3) not yet statistically significant; (4) neither arm clears the gate. The PEAD scorer hook ships **default OFF** (byte-identical committed config).

**Update (same day, alpha-vs-beta confirmed)**: fixed the EventEdge harness to emit daily_returns_dated (#422) so residual-α now computes for PEAD CPCV, and re-ran. **The filter's lift is ALPHA-LIKE, not beta**: baseline PEAD beta-hedged Sharpe **+0.035** (β=0.12, residual-α t +0.04 — pure beta, as known); filtered beta-hedged Sharpe **+0.587** (β=0.14 ~flat, residual-α t +0.65). So the filter selects PEAD trades with genuine post-earnings drift, not more market exposure — the **first sign of real (non-beta) edge enhancement in the whole options program**. BUT still **underpowered** (residual-α t 0.65 < 2 on the 2y/8-fold sample). Net: a **materially stronger lead** that now warrants the remaining confirmation — threshold robustness (0.75/1.25) + more data (4y R1K backfill, which needs a partitioned-write refactor to avoid OOM) — before any live change. Still default OFF.

**Consequences**: the program's one positive lead is logged + reproducible (`run_pead_implied_filter_cpcv.py`). Reusable: `app/data/options_signal.ImpliedMoveProvider` (PIT implied move, lazy per-symbol reads), PEAD scorer `implied_move_fn`/`min_move_vs_implied` (default OFF), `--r1k` backfill flag. Owner steer pending: invest in confirming this lead vs bank it. Not a WF/CPCV-core change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-09 — Live trading: kill the dead swing ML ranker in the live path + make the Trader's ML gate observable

**Context**: Owner reported "no trades firing." Diagnosis of the live PM→RM→Trader funnel (PostgreSQL `proposal_log`): today's 23 proposals were all `rm_status=APPROVED` but `trader_status=NULL` (Trader never recorded a decision) with confidence ≈ 0.51. Root causes: (1) the cross-sectional swing **ML ranker — concluded DEAD (2026-06-03) and frozen for *retraining* (`SWING_ENABLED=False`) — was never disabled in the LIVE proposing path**, so its 30-min `_scan_new_opportunities` rescan still produced ~30 of the last 32 live trades; it ignores both `pm.swing_selector` and `pm.min_confidence=0.55`, proposing empty-selector ~0.51 names. (2) The Trader's entry gate `ML_SCORE_THRESHOLD=0.55` then rejected them — but **silently** (DEBUG log, no `trader_status` written), the *only* entry gate that didn't surface a reason, so proposals looked "stuck at Approved." A git-archaeology found the `0.40/0.55` mismatch originated in `d728c21` (2026-05-22, "min_confidence alignment" after the LambdaRank double-normalization fix) which lowered PM `MIN_CONFIDENCE` 0.55→0.40 but left the Trader gate at 0.55 — incomplete; mooted now since the dead ranker (the only path using the 0.40 code constant) is being disabled.

**Decision**: (1) New master flag **`pm.swing_ml_live_enabled`** (agent_config, default `'false'`, fail-closed) gates BOTH live dead-ranker proposal paths — `_scan_new_opportunities` (the 30-min rescan) and the `selector=='ml_model'` fall-through in `_analyze_swing_premarket`. Default OFF aligns the LIVE system with the "ranker is dead" decision: the validated-null ranker proposes no live trades. (2) The Trader's ML-score gate now logs at INFO, writes `ProposalLog.trader_status='REJECTED_ML_SCORE'` + reason, emits an `ENTRY_REJECTED_ML_SCORE` decision, and drops the symbol — so a below-threshold rejection is never silent again. **Untouched** (verified by an Opus review + full sweep): PEAD (`pm.swing_selector='pead'`, the live book), quality_short, factor_portfolio, trend, intraday, the allocator, and all EXIT/re-eval paths (`self.model` is still used to manage existing positions, only NEW-proposal generation is gated).

**Trustworthiness**: an Opus 4.8 adversarial review certified the change correct/complete/safe (loop-mutation safe via the `list()` copy; both proposal paths gated; DB-write pattern matches the existing `MACRO_BLOCKED` gate; flag fail-closed). A second Opus full-sweep confirmed PEAD confidence (0.65–0.90, `pead_vix_conf_ref=100`) clears the 0.55 Trader gate, so PEAD is unaffected. 9 new tests.

**Consequences**: the live book is now the **validated sleeves** (PEAD + trend), not a known-null strategy; the funnel is no longer flooded with un-enterable sub-0.55 names; below-threshold rejections are visible in the dashboard/logs. Reversible by flipping `pm.swing_ml_live_enabled='true'`. **Requires a uvicorn restart** to load the PM/Trader changes (the new config key resolves to its `'false'` schema default until then). **Surfaced for owner decision (not changed here):** trend is in **shadow mode** (`pm.trend_shadow='true'` → computes but sends NO real orders) and only rebalances on `pm.trend_rebalance_weekday` (Mon) via the orchestrator process — so trend has placed no real paper orders yet; the Phase-88 swing-opportunity gate (`<0.35`) can suppress the whole swing book pre-RM (logged `SWING_ABSTAINED`); PEAD live sizing is derived from `generate_signal`'s swing ATR stop rather than PEAD's own (a live↔backtest fidelity gap); and `pead_vix_conf_ref` is one edit away (≲27) from silently pushing PEAD under the 0.55 gate in elevated VIX. Not a WF/CPCV change → `PIPELINE_ARCHITECTURE.md` untouched.

---

## 2026-06-09 — Alpha-v5 OPT-3 deep-dive + fair re-test: verdict holds (KILL) but the REASON corrected — thin/cost-killed, not structurally negative

**Context**: Before pivoting to OPT-4, the owner asked for a full Opus 4.8 deep-dive of the *entire* options build to be certain nothing could have impacted the OPT-3 outcome. Three parallel auditors covered (1) backfilled-data quality + integration, (2) simulator P&L, (3) strategy-construction fairness.

**Findings**: (1) and (2) certified the **harness/data/simulator clean** — data pristine (0 NaN/zero/neg closes across 45M bars, IV-crush empirically visible in real marks, 3.13% conservative stale marks, all 39 names fully covered, OCC strings exact, PIT verified; split-relabeled in-store discontinuities exist but are never held by the short-hold strategy — noted as a latent guard for any future long-horizon options sleeve); sim P&L sign + accounting correct, cost model mildly conservative and far too small to flip the verdict. (3) found the *first* parameterization was **handicapped** (nearest-weekly ~3-DTE expiry → max gamma/tiny vega; short strikes only 1×EM → ~40% breach; ATM strawman traded on each name's first event), so the original −1.82 understated the edge.

**Decision**: re-ran the **canonical** structure (expiry nearest ~25 DTE, short strikes 1.3×EM, strawman events skipped). Result at realistic 1× spreads: **gross-profitable but risk-adjusted-flat** — Avg PF 1.21, Calmar 0.85, residual-α t **−0.24 (≈ zero)**, mean Sharpe −1.0 with 33% positive folds (short-vol fat left tail); **collapses at 2× spread** (PF 0.82). **Verdict unchanged: KILL** (fails the 2× stress mandate + the significance/Sharpe gate), but the reason is now correct: **single-name earnings IV-crush is a real-but-too-thin premium killed by options transaction costs**, not a structurally negative trade. Three structures tested + logged (multiplicity noted); the re-parameterization was a-priori options-theory correction of objective flaws, not result-driven filter-hunting — and I stop tweaking after this canonical run regardless of outcome.

**Consequences**: the deep-dive *changed the reason, not the verdict*, and validated the build end-to-end (high confidence for all downstream options work). It **strengthens the OPT-4 pivot to index/ETF VRP**: index options spreads are ~pennies (vs 1-5% single-name) and the VRP is fatter and crisis-negative — the exact cost wall that kills single-name earnings vol is minimal there. Code: builder revised to the canonical parameterization (target-DTE expiry selection, 1.3×EM strikes, `allow_atm` gate so the ATM strawman is never traded in production).

---

## 2026-06-09 — Alpha-v5 OPT-3: earnings IV-crush KILLED (negative single-name VRP); options pipeline proven; OPT-4 pivots to index VRP

**Context**: OPT-3 wires the first end-to-end options strategy through the trusted WF/CPCV path and produces the program's first KEEP/KILL verdict (the owner checkpoint). Strategy: sell a defined-risk iron condor into each earnings event (enter T-1 close, exit T+1 close) to harvest the post-earnings IV crush, across a 39-name growth-heavy universe, 4 years, CPCV k=8/p=2, with the mandatory 1×/2× spread-stress sweep.

**Decision / finding**: **KILL.** Two economically-motivated structures were tested and logged (no filter-hunting): an ATM iron butterfly (mean Sharpe −3.86 — a strawman: 1-strike ATM wings are blown through by the earnings move) and the canonical OTM iron condor with short strikes ~1 expected-move out (mean Sharpe **−1.82 @1×, −2.52 @2×**; PF 0.59→0.47; residual-α t −2.57, beta-driven; win rate 57%). The payoff is the genuine short-vol shape — many small credit-kept wins (median trade +$12) overwhelmed by the occasional breach (worst −$848). **Economic reading: realized single-name earnings moves EXCEED the implied move on this universe, so the variance risk premium is *negative* at the single-name earnings level** — the opposite of the well-documented *index* VRP.

**Trustworthiness**: an Opus 4.8 adversarial review focused on look-ahead certified the verdict — every surface is causal (entry/exit dates strictly bracket the event; expected-move uses only past earnings; chain PIT `knowable<=entry`; bars marked by trade-date `<=d`; raw closes match unadjusted OCC strikes), P&L sign correct and golden-tested, and the negative Sharpe is a faithful consequence of a losing cost-heavy short-vol book — not an artifact. The minor caveats (liquidity-based position drops; ATM strawman on missing earnings history; current-liquid underlying universe) would *understate* an edge if anything, so none can manufacture a false KILL.

**Consequences**: (1) **The options program's data→engine→sim→adapter→CPCV pipeline is proven** — it produced a trustworthy verdict on real multi-year data (45M option bars backfilled). A KILL is a success of the harness (cf. reversal / carry). (2) **OPT-4 is reprioritized**: lead with **index/ETF systematic short-vol** (positive VRP, crisis-negative → diversifies the trend sleeve) and **cross-sectional / relative VRP** (delta-neutral long-cheap/short-rich); single-name *outright* earnings short-vol is dead. (3) The IV-crush scorer + `OptionsStrategy` adapter remain as reusable harness (a new strategy = a new position builder). Not a WF/CPCV-core change (new disposable adapter only), so `PIPELINE_ARCHITECTURE.md` is unchanged beyond OPT-2's simulator entry.

---

## 2026-06-09 — Alpha-v5 OPT-2: contract-level options simulator — mark to REAL closes, not theoretical prices

**Context**: With the engine (OPT-1a) and PIT data (OPT-1b) in place, we need to turn a sequence of option positions into a daily-MTM equity curve that the existing WF/CPCV gates can grade. The OPT-0 contract said "marks daily to the engine" — but we have the actual EOD option closes from OPT-1b.

**Decision**:
1. **Mark to REAL EOD option closes** (forward-filled on no-trade days), not theoretical engine prices. Real closes embed the actual market IV, so IV-crush is carried by the data itself (a short straddle into earnings simply reprices lower the next day) with zero synthesis — strictly more faithful than a model mark. The OPT-1a engine is for greeks/analytics in strategies, not for marking. Settlement at/after expiry uses the underlying close intrinsic. (`app/backtesting/options_simulator.py`.)
2. **Defined-risk payoff caps are automatic**: every leg is marked and settles at its own intrinsic, so a vertical caps at the strike width minus net debit — no special-case cap logic that could drift from reality.
3. **Cost = modeled spread** (% of premium × a mandatory 1×/2×/3× stress mult + per-contract fee); held-to-expiry legs pay no exit cost (assignment/expiry, no trade). Emits the same `SimResult` so every downstream gate reuses verbatim; attaches `daily_returns_dated` (for the OPT-3 FoldResult) + health flags.
4. **Opus 4.8 adversarial review** (P&L accounting / look-ahead) confirmed the MTM and look-ahead discipline correct, and drove fixes for *silent failure modes* (the dangerous class): dropped-position logging + `dropped_positions` count; `blown_up` hard-fail flag for any defined-risk book that goes ≤ 0 (was reporting a benign 0% Calmar); profit-factor cap (inf → 99, which would poison gates); rejection of unparseable contracts (was collapsing to a same-day no-op); day-1 entry-cost capture in the return series. 19 golden-path tests (hand-computed long/short/vertical-cap P&L, calendar spread, multi-day fwd-fill, intermediate short-MTM sign, qty scaling, cost-sweep monotonicity, all guards).

**Rationale**: Marking to real closes removes a whole class of model-error in the backtest (the engine's theoretical price vs the market's) and is only possible because OPT-1b gives us real per-contract closes. The silent-failure fixes matter because a defined-risk options backtest that quietly drops trades or masks a blow-up would produce a confident but false KEEP/KILL verdict.

**Consequences**: A trustworthy contract-level options P&L engine exists. `PIPELINE_ARCHITECTURE.md` §2 now lists `OptionsSimulator` (4th simulator; DAILY MTM) + a changelog entry. Unblocks OPT-3 (adapter wires `daily_returns_dated` into `FoldResult` → `run_cpcv` + significance gate + CAPM residual-α + the 2× spread-stress sweep → the program's FIRST real KEEP/KILL verdict, an owner checkpoint).

---

## 2026-06-09 — Alpha-v5 OPT-1b: options data layer — survivorship from the OPRA day files, PIT via holiday-aware knowable_date

**Context**: The pricing engine (OPT-1a) needs historical OHLCV + a contract universe to price against. Polygon Developer serves NO historical IV/greeks/OI (only the current snapshot) and NO historical NBBO — so this layer carries only PIT OHLCV + the universe; IV/greeks are computed by the engine. The two ways an options backtest silently lies are **survivorship bias** (building the universe from today's active chain drops every contract that expired worthless — the modal outcome for short premium) and **look-ahead** (using an EOD bar before it printed).

**Decision**:
1. **Survivorship by construction.** The universe is built FROM the OPRA daily flat files (`us_options_opra/day_aggs_v1`, every contract that *actually traded* that day, expired included) — not from the REST active chain. A contract enters the store the first day it prints a bar and is never removed. `fetch_contracts`/`get_current_snapshot` (REST) exist only for live/validation and are never on any historical path. (`app/data/options_provider.py`, `scripts/backfill_options.py`.)
2. **PIT via holiday-aware knowable_date.** An EOD bar trade-dated D is knowable the next *trading* day; `knowable_date = D + 1 NYSE business day` using a proper NYSE holiday calendar (observes Good Friday; not Columbus/Veterans) so it never lands on a closed session. Every historical accessor filters `knowable_date <= as_of`. Contract metadata (strike/expiry/type) is decoded from the OCC ticker; `knowable_date` for a contract is the MIN over its bars (first knowable). (`docs/reference/OPTIONS_DATA.md`.)
3. **Storage**: `data/options_bars.parquet` (long OHLCV + knowable_date) + `data/options_contracts.parquet` (derived from the bars, always consistent). Provider `PolygonOptionsProvider` implements the OPT-0 `OptionsDataProvider` contract; `polygon_s3.py` extended with `get_options_day_file` + a `dataset` param.
4. **Opus 4.8 adversarial review (look-ahead / survivorship focus) confirmed the architecture sound and drove fixes**: holiday-aware knowable_date (was weekday-only `BDay`, could stamp a holiday); datetime-resolution + dtype coercion on load (parquet round-trips ms, fresh bars are ns → concat crash; now forced to ns); coverage-start guard (an empty universe before our data window logs a DATA-GAP warning instead of masquerading as "no contracts existed"); per-day logging of dropped adjusted/non-standard roots (split-driven gaps become visible). 22 tests (OCC parse, prefix-root disambiguation, PIT no-look-ahead, survivorship incl./excl. expired, holiday knowable_date, dtype coercion, merge revision-keep, multi-underlying alignment). S3 path smoke-tested: 3 days × SPY = 19,392 bars / 8,939 contracts, 791 expired retained, PIT confirmed.

**Rationale**: Deriving the universe from traded bars is *more* survivorship-safe than the REST reference endpoint (the files are ground truth of what traded) and avoids REST pagination for history. Computing knowable_date holiday-aware (vs the SI provider's padded weekday lag) keeps the options lag exact (+1 trading day) without staleness. We backfill a focused liquid universe (index ETFs + large caps), not all of OPRA — the IV-crush/VRP strategies only need the names we trade.

**Consequences**: A multi-year, PIT, survivorship-safe options OHLCV + universe store is available behind the frozen contract. Unblocks OPT-2 (contract-level simulator marks against this data + the OPT-1a engine). Not a WF/CPCV pipeline change yet → `PIPELINE_ARCHITECTURE.md` untouched until the options simulator lands (OPT-2).

---

## 2026-06-09 — Alpha-v5 OPT-1a: options pricing/greeks engine (BS + Bjerksund-Stensland + CRR) validated vs live snapshot

**Context**: Polygon Developer serves IV/greeks only in the *current* snapshot, so all historical IV/greeks must be **computed** — the program's confidence keystone (a wrong pricer silently corrupts every options backtest). OPT-1a builds that engine and proves it against the one window with ground truth (the served-IV snapshot).

**Decision**:
1. **Engine** (`app/options/pricing_engine.py`, pure/no-I/O, implements the OPT-0 `OptionsPricingEngine` Protocol): Black-Scholes-European closed-form price + greeks; **Bjerksund-Stensland 1993** American approximation (calls direct; puts via the put-call transform P(S,K,r,b)=C(K,S,r−b,−b)); **CRR binomial** as an independent reference; bisection IV solver; American greeks via kink-aware central finite differences.
2. **Validation** (`scripts/validate_options_engine.py`, the keystone): recompute American IV from EOD close + spot + real per-underlying dividend yield + rate, compare to Polygon's served IV/delta over 15 underlyings, PASS/FAIL on the OPT-1 tolerance. **Result: PASS** — near-ATM median |IV err| 0.0072, all-contract bias +0.0068 (both < 0.010), engine-delta vs served-delta median |err| 0.0011 (greeks essentially exact). A **day-vol ≥ 10 liquidity filter** removes a +0.022 tail bias shown to be a *data-timing artifact* (snapshot pairs an option's stale last trade with the live spot) — absent in EOD-bar backtests.
3. **Adversarial review (Opus 4.8) found + fixed 3 bugs before merge**: **(CRITICAL)** for dividend-yield > rate calls the BjS h(T) term flips positive and the trigger boundary degenerates, underpricing ~95% (0.0017 vs 0.30 true) → route the strongly-negative-carry regime to the exact CRR binomial. **(HIGH)** the IV solver marched to the bracket top (3.0) for deep-ITM American prices pinned to the intrinsic floor → return `None` (vol is unrecoverable from a price = intrinsic). **(MEDIUM)** central-difference gamma spiked ~10× at the early-exercise boundary strike → one-sided 2nd difference on the smooth side. All three have regression tests (18 unit tests total: textbook BS, put-call parity, American≥European, BjS↔CRR cross-check, IV round-trip, greeks signs, contract conformance).

**Rationale**: BjS-1993 is fast and accurate in its valid regime but has a known degenerate carry regime; rather than ship a higher-order approximation, we fall back to CRR (exact in the limit) only for the rare contracts that need it — fast path stays fast, edge cases stay correct. Computing IV per-contract from its own price means we match each strike's own smile point, so the validation tests the *engine*, not a smile model.

**Consequences**: The historical IV/greeks engine is trustworthy (validated to <1 vol-pt near-ATM, exact greeks). `validate_options_engine.py` is now a repeatable nightly health gate (PASS/FAIL exit). Residual +0.0068 bias is flat-rate + crude-dividend; OPT-2 wires a real rate series. Unblocks OPT-1b (data layer) → OPT-2 (simulator). Not a WF/CPCV pipeline change yet, so `PIPELINE_ARCHITECTURE.md` is untouched until the options simulator lands (OPT-2).

---

## 2026-06-09 — Alpha-v5: Options Strategy Program launched (Polygon Developer); OPT-0 charter + spike PASS

**Context**: Free-data 3rd-sleeve candidates are exhausted (reversal/carry/estimate-revision all
eliminated — the opportunity set is fished out on free data). Owner subscribed **Polygon Options
Developer ($79/mo)** to pursue the highest-ceiling edge: the options variance risk premium.

**Decision**: Launch a phased **Options Program** (Alpha-v5) — SSOT `docs/living/OPTIONS_PROGRAM.md`.
Build a **resilient five-layer base** (data ⟂ pricing engine ⟂ simulator ⟂ pluggable strategy ⟂
reused gates/allocator/live; four frozen contracts in `app/options/contracts.py`) and explore MANY
options strategies, each validated KEEP/KILL on the SAME `run_cpcv` + significance gate + CAPM
residual-α we trust for equities, plus an options-specific **spread-stress sweep (KEEP must survive
2×)** and capacity check. Foundation first, but prove the whole pipeline end-to-end with ONE
strategy (earnings IV-crush) before building the catalog. Phases OPT-0…OPT-8; owner checkpoints at
OPT-0, OPT-3 (first verdict), OPT-8 (arm live).

**Forced data-architecture facts (Developer tier)**: Polygon serves IV/greeks/OI only in the CURRENT
snapshot → **we compute historical IV/greeks ourselves** (BS-European + Bjerksund-Stensland); **no
historical NBBO** → mark off EOD close + model/stress the spread; **no historical OI** → liquidity via
volume/notional; survivorship cured via the expired-contract universe.

**Rationale / de-risking**: the OPT-0 feasibility spike (`scripts/spike_options_iv_check.py`) proved
the confidence keystone — computed BS-European IV matches Polygon's served IV to **0.86 vol-points
median, unbiased, near-ATM** (the contracts VRP trades); all-contract bias (+0.035) is the expected
ITM/OTM + dividend gap that BjS + real dividends close in OPT-1. So computing historical IV from EOD
close is accurate enough to backtest with confidence. **Consequences**: the program's foundational
risk (computed-IV accuracy) is retired up front; OPT-1 (data + engine) green-lit. Live execution
(Alpaca options `OptionLegRequest`/mleg) is supported by the SDK but not yet wired — a later build.

---

## 2026-06-08 — Alpha-v4 P4: short-term reversal sleeve KILLED (cost-dead); 3rd-sleeve slot stays open

**Context**: Sought a 3rd uncorrelated premium after PEAD + TSMOM trend. Owner chose
short-term cross-sectional reversal (the momentum complement). Built + validated a
dollar-neutral, PIT, survivorship-safe sleeve (`app/strategy/reversal.py` + `run_reversal.py`).

**Decision**: **KILL / benchmark-only — do not live-wire.** The reversal signal is real but
weak (gross +0.40, t=1.28 @2bps) and **cost-dead**: ~159x/yr turnover → ~16%/yr cost drag →
**-0.90 Sharpe at a realistic 10bps**; adding it *drags* the book (equal-capital +1.145 →
+0.138). It IS genuinely uncorrelated (β~0.10, corr +0.13/+0.03 to PEAD/trend) — the concept
is right, the tradeable edge isn't.

**Rationale**: Opus 4.8 adversarial review verified the KILL is real, not a bug (sign / cost
single-charge / look-ahead / dollar-neutrality / liquidity-masking all correct). Short-term
reversal is the most-arbitraged anomaly; this is the expected null. **Explicitly NOT
filter-hunted** to rescue it (only-trade-in-high-VIX etc.) — that's the B5 trap on the STOP
list. **Consequences**: the harness is retained as a reusable validated null (7 tests); the
3rd-sleeve slot remains OPEN. Next candidates: options-VRP feasibility spike (needs paid IV
data — a spend decision), cross-asset carry (free data), or squeeze-conditioning (existing
SI data, but a PEAD conditioner rather than a standalone premium).

---

## 2026-06-08 — Alpha-v4 P3: live regime-aware sleeve allocator (gate-controlled, default-equal, kill-switchable)

**Context**: The live book ran two independent sleeves at static budgets (trend
`pm.trend_allocation_pct=0.40`, PEAD telemetry `pm.pead_size_mult=1.0`). The allocator
(`app/strategy/sleeve_allocator.py`) existed only as a backtest library. The Phase-3
book-level gate (`scripts/run_book_allocator.py`, margin >0.10 Sharpe AND no-worse-DD)
found on the 2-sleeve overlap: **equal +1.082 > vol +0.715 > regime +0.593** — regime is
worst. So vol/regime are not justified on the current book.

**Decision**: Wire the allocator into the LIVE book as a kill-switchable layer
(`app/live_trading/sleeve_allocator_live.py`) that **ships DISABLED** (`pm.allocator_enabled=false`)
→ byte-identical to today. When enabled it recomputes weekly (before the trend rebalance),
persists effective weights to `agent_config`, and both sleeve readers
(`effective_trend_allocation` / `effective_pead_size_mult`) consult them with a
**fixed-weight fallback** on disabled / stale / warmup / any error. Default scheme `equal`;
`vol`/`regime` are live-capable but stay OFF until `run_book_allocator.py --emit-config`
selects them (expected after a 3rd sleeve).

- **Live regime label** = the persisted, staleness-aware live score
  (`get_regime_context`) mapped **RISK_ON→BULL, RISK_CAUTION→NEUTRAL, RISK_OFF→BEAR,
  unknown→NEUTRAL** (the safe no-tilt key).
- **PEAD regime double-tilt avoided (from the Opus pre-merge review):** PM already applies
  a per-name `_regime_sizing_multiplier` to PEAD, so under `scheme=regime` the allocator
  does **not** also scale PEAD's `size_mult` (would compound the same regime bet) —
  PEAD's own per-name mult is its sole regime tilt; the allocator's regime tilt flows only
  to the TREND budget. `equal`/`vol` map PEAD normally (no regime component).
- **Gross-cap safety:** effective trend allocation is clamped to ≤0.80, and the trend
  sleeve's existing `apply_risk_gate` independently caps total (trend+PEAD) gross ≤80% on
  actual positions regardless of allocator output.

**Quality loop**: adversarial Opus 4.8 review found 1 Critical (the double-tilt, now
guarded) + doc/robustness items (now fixed); a second Opus pass verified all resolved with
no regression (SHIP).

**Known limitations (must clear before enabling `vol`/`regime`):** (1) the LIVE regime
path uses a one-shot tilt (no hysteresis/EWMA blend — those need a label series), so it
differs from the validated backtest `apply_regime_tilt` and must be re-validated before
activation; (2) the trend sleeve writes its tracker weekly, so the live per-sleeve vol
estimate is coarse until ~`pm.allocator_min_deployed_days` of history accrue (the warmup
guard keeps it in static fallback until then). **Consequences**: zero behavior change today;
infra + re-runnable gate ready to activate when the 3rd sleeve makes vol/regime earn it.

---

## 2026-06-07 — Alpha-v4 P0 gate recalibration: robustness over Sharpe level; residual-alpha-t diagnostic-first

**Context**: On N_eff≈8 a high Sharpe *level* selects for overfitting (5-LLM review:
"lower the SR≥0.80 gate — real edges live at 0.4–0.7; weight fold-consistency +
neutralized-t over level"). The significance gate already made t-stat / %pos / P5 /
worst-regime the primary criteria, but the headline bar and the missing
market-residualized check needed to land.

**Decision**:
- **Retire the legacy SR≥0.80 promotion bar.** `GATE_MODE='significance'` stays the
  default (the legacy `mean_sharpe`/0.80 path is kept only for reproducibility).
  Lower `CAPITAL_GATE_MIN_MEAN_SHARPE` 0.50→**0.45**; keep PAPER 0.35, min-fold/P5
  floors, DSR/PF/Calmar, and the worst-regime survivability floor. The Sharpe floor
  is now a materiality *backstop*, not the discriminator.
- **Add residual-alpha-t (CAPM/HAC) as a DIAGNOSTIC, not a gate — yet.** Per owner
  decision, it enters *diagnostic-first*: computed on the concatenated OOS book
  returns vs SPY, reported in `print()`/JSON + a `t<1` WARN log, and **explicitly
  excluded from gate pass/fail** until validated (it reproduces the known PEAD
  beta-driven verdict and a genuine-alpha case in tests). It graduates to a primary
  blocking criterion in a later PR — mirroring how the significance gate itself was
  rolled out behind a faithful-reproduction proof.

**Rationale**: a blocking gate on a brand-new metric over ~8 effective folds could
mis-promote/mis-retire before it's trusted; diagnostic-first de-risks that while
still surfacing the single most important robustness signal (does the edge survive
hedging out the market?). **Consequences**: every CPCV run now prints residual-α-t +
β + hedged-Sharpe; gate verdicts are unchanged this PR (proven by test). Canonical
estimator is `scripts/walkforward/attribution.capm_alpha` (shared with the PEAD
attribution script). See PIPELINE_ARCHITECTURE Gate Inventory + changelog.

---

## 2026-06-06 — Live TSMOM trend sleeve: standalone weekly rebalancer (Alpha-v4 live wiring)

**Context**: Alpha-v4 Phases 0–3 are complete. The TSMOM trend sleeve
(`app/strategy/tsmom.py`, validated standalone Sharpe +0.71, the book's crisis
diversifier) was the strongest sleeve but not live. The task: trade it live in the
paper account alongside PEAD at a simple fixed weight.

**Decisions**:
- **Standalone weekly executor, NOT a `pm.swing_selector` value.** The selector is
  mutual-exclusion (one daily stock scan producing entry signals); the trend sleeve
  is a weekly rebalance-to-target on a fixed 10-ETF basket that must run *alongside*
  PEAD. It lives in `app/live_trading/trend_sleeve.py`, fired by a daily orchestrator
  job (09:45 ET) with an in-function weekday guard (`pm.trend_rebalance_weekday`,
  live-tunable) + a fail-closed market-open check (`AlpacaClient.get_clock` — the
  weekday cron has no holiday calendar).
- **Direct Alpaca placement with a lightweight risk gate** (kill-switch, gross cap
  `trend+PEAD ≤ 80%`, fat-finger, per-name cap), NOT the PM→RM→Trader proposal queue
  (those rules are entry-signal-shaped and map poorly onto rebalance trims/sells).
- **Equal-capital 50/50**: `pm.trend_allocation_pct` default 0.70→**0.40** (trend 40%
  / PEAD 40% under the 80% gross cap — matches the Phase-3-validated equal-capital
  book, which beat vol-weight and regime-tilt).
- **PEAD dialed to telemetry** in the schema defaults too: `pm.pead_size_mult` 3.0→1.0,
  `pm.pead_max_position_pct` 0.10→0.05 (the DB values were already dialed; rebaselining
  the defaults prevents a DB reset from silently re-ramping PEAD). `test_pead_ramp_b4`
  expectations updated to match.
- **Shadow-first, dormant-by-default**: `pm.trend_enabled` default `false`,
  `pm.trend_shadow` default `true` (logs would-be orders to `decision_audit` with
  `block_reason="shadow"`, sends nothing). Owner arms via `scripts/set_trend_config.py`.
- **Trend positions tagged `selector="trend"`/`trade_type="trend"` and excluded from the
  Trader's per-tick stop/target exit loop** (`_check_exit` guard) — the weekly
  rebalancer is their sole manager; otherwise the synthetic stops the reconciler
  attaches would liquidate the sleeve mid-week.
- **Fail-closed everywhere**: kill-switch, data-fetch failure, missing core symbol
  (SPY), or NAV-fetch failure → no orders. Whole shares only (Alpaca wrapper is int-only).

**Consequences**: trend coexists with PEAD as a peer sleeve; live-vs-backtest
divergence (Alpaca vs yfinance adjustment, wall-clock vs modular rebalance) is tracked
by `app/live_trading/trend_tracker.py` (+0.71 reference, weekly rollup email). Known
limits / backlog: the gross-cap formula now lives in 3 places (trend imports the
canonical `risk_manager.GROSS_EXPOSURE_CAP`); fixed 40/40 forgoes the validated
vol-weighting + BEAR regime tilt in `sleeve_allocator.py` (deliberate "ship simple
first" — revisit when more sleeves / longer overlap earn it).

---

## 2026-06-02 — Significance-first two-tier promotion gate (replaces mean-Sharpe≥0.80)

**Context**: The promotion gate's primary discriminator was `mean_sharpe ≥ 0.80`
(swing) / `≥ 1.00` (intraday). Those thresholds were calibrated against numbers
that have since been struck as in-sample artifacts (intraday +5.14, QualityShort
+3.25). A bare mean-Sharpe threshold cannot distinguish a `+0.22 / t=0.17` noise
result from a `+0.546 / t=2.26` genuine-signal result — both are below 0.80, yet
one is statistically significant and one is pure noise. The 0.80 bar was a
frozen-WF relic: it rejected the real signal (PEAD) for the same reason it rejected
the noise, providing no actual discrimination.

**Decision**: Adopt a **significance-first two-tier** gate behind a `GATE_MODE`
flag (default `"significance"`; `"mean_sharpe"` reproduces the legacy gate exactly
for reversibility + historical re-scoring).
- Primary discriminators become statistical: path-Sharpe **t-stat** (N_eff=n_folds,
  flipped from WARN to BLOCK), sign-consistency (`pct_positive`), and the tail
  (`p5_sharpe`). Mean Sharpe is demoted to an economic-materiality FLOOR.
- **PAPER** tier (forward-validate, no capital): t≥2.0, %pos≥0.75, P5≥0.0,
  mean≥0.35, plus PF/Calmar/regime backstops.
- **CAPITAL** tier (real money): PAPER + mean≥0.50 + n_folds≥10 + (t≥2.5 OR a
  documented live-paper confirmation). The higher t-stat is a multiple-testing
  haircut (~10–15 strategy shots); n_folds≥10 is a statistical-power floor.
- A standard WF report (single point estimate, no path distribution) HARD-FAILS
  under significance — it cannot fabricate a t-stat; CPCV is required.

**Rationale**: Promotion should be gated on whether an edge is statistically real
and economically material, not on clearing an absolute Sharpe number that was set
against contaminated baselines. The two-tier split lets a genuinely-significant-
but-still-developing edge go to PAPER (forward-validate with no money at risk)
while reserving CAPITAL for results that also clear the multiple-testing haircut.

**Consequences**:
- Re-scoring every CPCV result on record (`scripts/rescore_gates.py`) promotes
  **only PEAD R1K → PAPER PASS / CAPITAL HOLD**. Every other strategy (Swing
  +0.22/t0.17, Intraday −2.80, Small/mid PEAD +0.361/t0.95/P5−1.368, QualityShort
  −0.903, Insider +0.228/t0.88) FAILs all tiers. The LEGACY(0.80) column is
  all-FAIL — confirming 0.80 never promoted any of these anyway; it just failed
  to separate the one real signal from the noise.
- PEAD is cleared to PAPER (forward validation), NOT capital — it lacks both the
  t≥2.5 haircut margin (2.26) and the n_folds≥10 power floor (8).
- `mean_sharpe` mode is a verified no-op vs pre-Phase-4 main (full legacy gate
  test corpus passes unchanged).
- No change to DSR math, N_eff=n_folds, OOS/sacred-holdout machinery, the
  simulators, or the PEAD scorer.

---

## 2026-06-02 — Significance-gate review fixes: PEAD paper PASS is a FLAGGED event-sparsity waiver, not unconditional

**Context**: An independent review of the significance-gate branch found three
blocking defects. (1) Under `GATE_MODE="significance"` a WF-only retrain hard-failed
`WalkForwardReport.gate_passed()`, and `retrain_cron.py` fed that boolean into
`record_tier3_result(gate_passed=False)`, which sets `status="RETIRED"` and rolls
back — so every scheduled WF retrain auto-retired the fresh model. The capital tier
was also unreachable (no caller ever requested `tier="capital"`). (2) The real PEAD
CPCVResult has `worst_regime_sharpe=None` due to event-sparsity (`<REGIME_MIN_OBS`
same-regime trading days — documented "not a bug"), and the backstop failed-closed
on None, so the REAL PEAD FAILED the paper gate the whole exercise was meant to pass.
(3) `rescore_gates.py` reimplemented the threshold math and hardcoded
`backstops_ok=True`, so its "PEAD PASS" was fiction, not the real gate.

**Decision**:
- **Tri-state outcome (FIX-1)**: distinguish "gate failed → retire" from "cannot
  evaluate for promotion → keep status." `GateOutcome{PROMOTE,RETIRE,INCONCLUSIVE}`;
  significance+WF → `INCONCLUSIVE` (report-only). The cron keeps the current model
  status on `INCONCLUSIVE` (no retire/rollback). Capital is reached only by an
  explicit promotion run (`--gate-tier capital`), never by the cron retrain.
- **Event-sparsity regime waiver (FIX-2)**: `worst_regime_sharpe=None` has two
  causes, now disambiguated by `CPCVResult.regime_insufficient_obs` (set from raw
  per-regime obs counts captured before the REGIME_MIN_OBS filter). For
  EVENT-SPARSITY only, the **PAPER** (zero-capital) tier waives the regime backstop
  AND flags `requires_human_review`. The **CAPITAL** tier never auto-waives (requires
  explicit `regime_waiver_approved`). A DATA-BUG None still fails closed on both.
- **Real-gate rescore (FIX-3)**: the artifact now runs the production gate.

**Rationale**: The waiver is the minimum needed to let an event-sparse strategy
reach forward-validation without opening a global fail-open. Scoping it to (a)
paper only, (b) event-sparsity only, (c) with a mandatory human-review flag keeps
the regime backstop fully enforced everywhere real capital or real regime data is at
stake. The corrected statement of the result: **PEAD R1K → PAPER PASS *with a
mandatory `requires_human_review` flag* (via the event-sparsity waiver) / CAPITAL
HOLD** — the prior "unconditional PASS" framing overstated it.

**Consequences**:
- PEAD reaches paper for forward validation but is explicitly tagged for human
  review because it was promoted without real regime data.
- A scheduled WF retrain under significance no longer auto-retires the fresh model;
  it logs INCONCLUSIVE and waits for an explicit CPCV promotion decision.
- Capital promotion of an event-sparse strategy is impossible without a documented
  `--regime-waiver-approved` human sign-off.

---

## 2026-05-23 — Adopt Opus 4.7 Four-Phase Plan

**Context**: v216 Walk-Forward gate failed (avg Sharpe -0.91, PF=0.00 every fold). Five independent LLM reviews (Claude, ChatGPT, Gemini, Grok, Deepseek) all flagged the same core issue: jumped straight to L4 (full agent stack) without validating at L1 (rank-IC) or L2 (decile spread).

**Decision**: Adopt Opus 4.7's four-phase plan:
1. WF Trustworthiness → 2. Signal Measurement → 3. Modelling → 4. Portfolio/Execution

**Rationale**: Each layer must pass independently before proceeding. Without isolating signal from execution, it's impossible to know whether PF=0.00 comes from bad features, bad labels, bad sizing, or bad simulation.

**Consequences**:
- NO retraining until Phase 2 (L2 decile spread) gate passes
- NO regime-conditional models until factor attribution confirms residual alpha
- PIT audit is the highest-risk gate: if fundamentals have look-ahead, all prior results are invalid

---

## 2026-05-23 — Fix 10 WF Simulation Bugs (PR #256)

**Context**: Opus 4.7 deep code review found 10 simulation bugs in walkforward_tier3.py and agent_simulator.py.

**Decision**: Fixed all 10 bugs:
1. MTM pricing used stale prices (off-by-one)
2. Sharpe annualization used calendar days not trading days
3. DSR formula missing sqrt(V[SR]) scaling
4. DSR N_obs used fold count not observation count
5. CPCV look-ahead: used future fold's training data for embedding
6. Force-close fired after MTM, double-counted last day P&L
7. Halt-day MTM used next day's open (look-ahead)
8. Sector ETF signal loaded same-day (look-ahead on rebalance date)
9. Short series annualization used wrong N in sqrt formula
10. profit_factor sentinel: returned 999 instead of 0 when no losses

**Consequences**: WF results are now trustworthy at the simulation level. v216 rerun gave Sharpe -0.91 (improved from -1.8+ but still gate failed).

---

## 2026-05-22 — Restore swing_v215 as Active Model

**Context**: v216 LambdaRank model trained with 18 features, 20d horizon. Walk-forward gate failed.

**Decision**: Restore v215 as the active paper-trading model while diagnostics run.

**Rationale**: v215 had better WF results than v216 post-bug-fixes. Running on broken simulation results (pre-fix) was producing misleading metrics. Running paper trading on v215 while investigating is safer than using a gate-failed model.

---

## 2026-05-20 — Adopt L/S Equity as Primary Strategy Direction

**Context**: Long-only swing strategy with ATR stops consistently fails WF gate. Opus analysis suggests the stop-loss asymmetry requires hit-rate ≥ 33% with 2:1 R:R — not achievable with IC ≈ 0.

**Decision**: Target Long/Short equity for production. Top-N long + bottom-N short, dollar-neutral.

**Rationale**: Removes the dependency on absolute return prediction (hard). L/S only requires relative ranking (easier). Eliminates directional beta. Enables full capital utilization in both bull and bear markets.

**Consequences**: Phase 4 must implement dollar-neutral construction with borrow filter.

---

## 2026-05-23 — Execute Phase 4 First If L2 Decile Sharpe >= 0.60

**Context**: Null benchmark showed random portfolio Sharpe = +0.669 vs v216 WF = -0.91 (z=-9.87). The execution layer is 9.87 sigma worse than random chance. L2 decile spread is running to determine if underlying signal exists.

**Decision (pending L2 result)**: If L2 Sharpe >= 0.60, skip Phase 3 (label redesign) and go directly to Phase 4 (execution fix: remove ATR stops, increase position count, L/S conversion).

**Rationale**: With execution destroying 1.5+ Sharpe units vs random, fixing execution is higher ROI than fixing labels. The 2021 IC = +0.023 suggests signal exists in bull regimes. The execution pathology (ATR stops + low position count) is the dominant failure mode.

**If L2 < 0.20**: No signal exists. Must rebuild features. Phase 3 before Phase 4.

---

## 2026-05-23 — Remove ATR Stops From Swing Strategy

**Context**: Null benchmark (no stops) achieves Sharpe +0.669. WF (with ATR stops) achieves -0.91.

**Decision**: The ATR stop mechanism should be disabled for initial Phase 4 testing. The stops are creating a negative feedback loop:
1. Low IC → random win rate ~50%
2. ATR stop triggers on small adverse moves, cutting many positions early  
3. Remaining positions run longer but the overall win rate < breakeven for 2:1 R:R
4. Net effect: stops increase transaction costs while not improving win rate

**Do NOT**: Add wider stops or tighter stops as a fix. The stop mechanism itself needs testing without stops first. If L2 without stops shows Sharpe > 0.60, that is the baseline.

---

## 2026-05-23 — Fold 2 Diagnosis: Opportunity Score Gate + ATR Stops (Phase 1.6)

**Context**: v216 WF Fold 2 (test: 2022-06-04..2023-05-24) had 95 trades vs 300+ in all other folds. Fold 2 covers the post-peak-inflation, aggressive-Fed-hiking period.

**Findings**:
1. Cross-sectional vol in Fold 2 = 1.04x other folds — NOT dramatically higher (test starts after the worst of the 2022 crash)
2. Symbol coverage: 769 vs 750 avg — similar, NOT a data sparsity issue
3. Primary suppressor: **opportunity score gate** (`score < 0.35 = skip`, `0.35-0.65 = cap at 2 candidates`). Model trained on 2020-2022 bull data assigns low scores to 2022 bear-market patterns → gate skips most entries
4. Secondary suppressor: ATR stops cut the few entries that pass the gate before HOLD_DAYS

**Decision**: Phase 4 isolation test must disable BOTH mechanisms:
- `--no-pm-opportunity-score` (disable opportunity score gate)
- Remove ATR stops (already decided)

**Note**: v216 WF used purge=10d not 85d. All v216 results have potential leakage and must be re-run with purge=85d post-Phase 4.

---

## 2026-05-23 — Phase 4 Before Phase 3 (Opus 4.7 Override)

**Context**: L2 decile spread returned Sharpe=0.397 (marginal, 0.20-0.60 range). Original decision tree said "Phase 3 first." Opus 4.7 reviewed all findings.

**Decision**: Run Phase 4 (execution fix) BEFORE Phase 3 (label redesign).

**Rationale**:
1. Null benchmark shows execution destroys ~1.6 Sharpe vs random. Phase 4 is a config change (1-2 days), Phase 3 is weeks.
2. Cannot measure label improvements through WF when execution layer masks signal. Phase 4 first establishes honest baseline.
3. Signal clearly exists in right regime (2021/2025 L/S Sharpe = +1.1). Short side is the structural problem, not features.
4. 2023 inversion (-1.29) is a crowded-short squeeze in narrow Mag7 rally — short-side failure, not long-side.

**Phase 4 Spec**:
- Disable opportunity score gate (`--no-pm-opportunity-score`)
- Remove ATR stops
- Position count: n=40 long, n=40 short
- Re-run v216 WF with 85d purge

**Phase 3 Spec (after Phase 4 baseline)**:
- Long-only labels: top-quintile binary (drop full cross-sectional rank)
- 10d horizon (not 20d) — doubles training samples
- Rolling 3-year window (not expanding)
- Add regime features as inputs (breadth, dispersion, VIX term structure)
- Kill sign-flipping features (per-year IC audit)
- Short side: separate model with quality overlay, NOT symmetric decile rank

**If Phase 4 WF Sharpe > +0.3**: proceed to Phase 3 with confidence.
**If Phase 4 WF Sharpe < 0**: investigate execution bug before any label work.

---

## 2026-05-24 — Opus 4.7 WF Code Audit: 10 Critical/Major Bugs Found

**Context**: After Phase 4 v2 WF (avg Sharpe +0.046, 78 trades) and L2 Sharpe=0.397, commissioned a thorough Opus 4.7 audit of walkforward_tier3.py and agent_simulator.py looking for bugs, look-ahead, and realism issues.

**Findings (prioritized)**:

1. **CRITICAL — Embargo never enforced in fold boundaries** (walkforward_tier3.py L689)
   - `raw_test_end_dt = train_end_dt + segment_days` → fold N test ends exactly where fold N+1 trains. Embargo_days was logged but had zero effect on boundary math.
   - **Fix**: `raw_test_end_dt = train_end_dt + segment_days - embargo_days`

2. **MAJOR — no_atr_stops defeated by check_exit trailing ratchet** (agent_simulator.py L1250)
   - When `no_atr_stops=True`, sentinel stop prices replaced with real trailing stops on first profitable bar, defeating the phase 4 isolation.
   - **Fix**: Only persist `new_stop` from check_exit when `not self.no_atr_stops`

3. **MAJOR — PF=999 sentinel inflates avg_profit_factor gate** (walkforward_tier3.py L269-271)
   - `avg_profit_factor` averaged PF=999 (all-wins fold) with real PFs, yanking mean far above gate threshold.
   - **Fix**: Cap individual PFs at 5.0 before averaging

4. **MAJOR — Silent trade loss when end-date data missing** (agent_simulator.py L514)
   - FORCE_CLOSE silently skipped positions with no bar data — trade never recorded, affecting trade count and equity.
   - **Fix**: Exit at entry_price with warning log when no bar data available

5. **MAJOR (deferred) — Calmar=0 "not computed" free-passes gate** (walkforward_tier3.py L292)
   - `avg_calmar == 0` was treated as "skip gate" rather than "gate fail". Ambiguous sentinel.
   - **Decision**: Document for future fix; change sentinel to NaN requires broader test updates.

6. **MAJOR (deferred) — Short buying power check uses full notional** (agent_simulator.py L889)
   - Short entries checked against cash balance using full notional (Reg-T 100%), over-rejecting shorts.
   - **Decision**: Defer; only affects short-side entries. Long-only Phase 3 is unaffected.

**Fixes implemented**: Items 1-4 committed in feat/wf-opus-audit branch.

**Consequences**: Previous WF results (all phases) used the defective embargo formula. Re-running Phase 4 v3 with corrected boundaries is required to get clean results. Embargo fix shrinks test windows by ~85 days each fold — with purge=85 and embargo=85, effective test window is 456-85=371 trading days per fold.

---

## 2026-06-03 — PEAD UI visibility: selector attribution + PEAD tracking panel

**Context**: The dashboard surfaced only "swing" and "intraday" proposals. PEAD — the sole live capital strategy — rode under the "Swing Proposals" tab, indistinguishable from swing-ranker proposals, and its rich daily scoreboard (`data/pead_tracking.db`: signals→entered→filled funnel, fill rate, gross deployed, daily/cum P&L, VIX blocks, per-overlay suppression counts) had **zero UI surface** (weekly email only). With PEAD live and currently 0-filling (price-ran / spread gates), there was no way to see *why* without querying SQLite by hand.

**Decision**:
1. **Data model** — added `selector` (VARCHAR(32), indexed) to `proposal_log`, mirroring `Trade.selector`. Chosen over deriving PEAD-ness by joining `proposal_uuid → trades.selector` so that **unfilled** PEAD proposals are attributable too (the join only covers proposals that became trades). Migration `scripts/migrations/2026_06_proposal_log_selector.py` is idempotent and backfills historical `dir_{selector}_*` batches (backfilled 271 rows: 150 quality_short, 121 pead).
2. **API** — `selector` threaded into all 3 PM `ProposalLog` persist sites; exposed on `/proposal-log` (response field + `selector` filter param) and on positions/trades responses. New `/api/dashboard/pead/tracking` wraps `pead_tracker.read_daily` with a window summary (funnel totals, fill rate, suppression counts, cumulative P&L).
3. **Frontend** — shared `SelectorBadge` across proposals/positions/trades; selector column + filter on the swing proposals table; new top-level **PEAD** tab (KPI row + signal→fill funnel + suppression breakdown + daily table) — the first UI view of the live PEAD book.

**Consequences**: PEAD is now first-class in the dashboard; the funnel/suppression view makes the live 0-fill situation diagnosable at a glance. The live PM/RM/Trader path is unchanged except the additive `selector` write (nullable, default `""`) — a server restart is needed to deploy the routes/UI but **not** for any behavior change. Built in an isolated git worktree to protect the in-flight ranker CPCV run. Not a WF/CPCV pipeline change, so `PIPELINE_ARCHITECTURE.md` is intentionally untouched.

---

## 2026-06-03 — Cross-sectional ML ranking is dead; close the ranker line, pivot to the event-driven edge family

**Context**: Alpha-v2 §3.1 hypothesized the "dead" swing ranker (+0.22, t=0.17) was merely *strangled* by a 5-position long-only book, and would show alpha if re-run **dollar-neutral, sector-neutral, high-breadth**. The first L/S run looked invalid-positive then invalid-negative; rigorous diagnosis found the book was never actually neutral (it ran ~35% net-long at 0.35 gross: the L/S rebalance was fed a one-sided **long proposal pool** of ~50 names — `_pm_score`'s `proposal_pool_size` cap + `min_confidence` floor — so the 60-long book absorbed the whole ranked set and the short leg starved; held positions were also never re-sized).

**Decision**: Fixed the validity end-to-end across 3 phases — **(1)** net-exposure observability (surface realized net beta/dollar/gross + result JSON), **(2)** dollar-neutral-at-target-gross (full-book resize each rebalance + breadth admission), **(3)** full cross-sectional scoring for the L/S arm + adequate power (k=8). On the **corrected, genuinely-neutral book** (realized net$ −0.01, gross 0.73, ~60 shorts), the decisive CPCV (N_eff=8) gave **mean Sharpe +0.14, path-t +0.18, %pos 67%, deployment-adj +0.12, DSR p 0.03** → **no cross-sectional alpha.** The long-only +0.22/+1.06 was **confirmed market beta** (neutralizing collapses it to noise).

**Rationale**: This is the *third* honest CPCV null from the cross-sectional-ML-ranking direction (swing long-only = noise; intraday v63 = cost-drag; dollar-neutral ranker = beta-only). t=0.18 is unambiguously null, not borderline — more CPCV power (purged-CV) cannot rescue a flat-zero signal, so we did not invest in it. The one validated edge (PEAD) is **event-driven, rules-based, economically grounded** — a different species from cross-sectional ranking. The data says alpha lives in the event-driven family, not in ML-ranking price/fundamental features.

**Consequences**: **The cross-sectional-ML-ranking line is closed.** The Alpha-v2 §3.3 (short-interest as a ranker feature) and Spike-B (residualized features) items are **shelved** — they were predicated on the ranker showing life. **PEAD is the sole validated edge** and now trades live (the entry-gate fix unblocked fills). Next direction (pending owner steer): pivot research to a **second event-driven edge** (analyst-revision drift / short-interest-squeeze-as-event / guidance) to diversify PEAD, and productionize PEAD (live track record + the §1.2b trend-filter). The validity-fix *infrastructure* (observability, neutral-at-gross L/S engine, full-ranking, net-exposure capture) is retained as reusable tooling even though the thesis died — its value is precisely that it prevented deploying a beta book as "alpha." See `ML_EXPERIMENT_LOG.md` (§3.1 Phase 1-3) for the run record.
