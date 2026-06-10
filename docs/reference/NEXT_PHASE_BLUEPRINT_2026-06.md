# Next-Phase Blueprint — 2026-06 (Alpha-v6)

**Inputs:** 5 independent external quant reviews of the 2026-06-10 package
([Gemini](prompts/20260610_Quant_Options_Review/responses/01_gemini.md) ·
[DeepSeek](prompts/20260610_Quant_Options_Review/responses/02_deepseek.md) ·
[Grok](prompts/20260610_Quant_Options_Review/responses/03_grok.md) ·
[ChatGPT](prompts/20260610_Quant_Options_Review/responses/04_chatgpt.md) ·
[Claude](prompts/20260610_Quant_Options_Review/responses/05_claude.md)),
reconciled against the actual codebase (gates, CPCV, EventEdge, attribution, options stack, allocator, live sleeves — verified 2026-06-10).

**Author:** Opus 4.8, acting as systematic PM. **Date:** 2026-06-10.
**Status:** This file is the SSOT for the Alpha-v6 direction. It supersedes the open-ended "choose next research direction" item in `PROJECT_STATE.md`. It does NOT supersede `QUANT_REVIEW_SYNTHESIS_2026-06.md` (Alpha-v4, complete) — it builds on it.

---

## 1. Executive synthesis

### Where the project actually stands (no flattery)

- **A top-decile research/validation platform.** PIT + survivorship by construction, per-fold retraining, purge/embargo, a forward sacred holdout (2026-11-09) enforced in code, a significance-first two-tier gate, residual-alpha attribution, an event-clustered bootstrap, mandatory 2× options-spread stress, and a genuine KILL culture (reversal, carry, SI factor, A1, XS-ML, small/mid PEAD, OPT-3 all killed honestly). This is the moat. None of the five reviewers disputed it.
- **One weak event edge.** PEAD: unbiased CPCV +0.578 but t=1.81; CAPM beta-hedged Sharpe **−0.37**. The positive Sharpe is small beta riding a bull sample. It is live at telemetry size, correctly.
- **One solid diversifier.** TSMOM trend: +0.71 standalone over 19 years including every crisis, corr +0.25 to PEAD, the stronger sleeve. First real paper rebalance Mon 2026-06-15.
- **A valuable options *information* asset — not an options *trading* edge.** 4y of local OPRA EOD bars (112.8M bars / 733 underlyings / 6.18M contracts), a validated pricing engine (computed IV accurate to ~0.7–0.9 vol-pts near-ATM), and three verdicts: single-name earnings IV-crush KILL (cost wall), index VRP real-but-Sharpe-flat (PF 2.24/1.75 @1×/2×), implied-move PEAD filter alpha-like-but-threshold-fragile.
- **Not yet a capital-grade alpha machine.** Nothing has cleared the CAPITAL tier. Live-vs-backtest fidelity has had repeated bugs (VIX block not firing, sizing cap ignored, dead ML proposing trades) — each caught, but proving the *class* of risk is live divergence, not residual leakage.

### The convergent thesis (all five reviewers, in one paragraph)

The harness has been hardened so far in the anti-inflation direction that its dominant error is now **Type II**: a t≥2.0 gate on N_eff≈6–8 folds of ≤4y data is arithmetic that a *true* Sharpe-0.5–0.7 edge fails more often than it passes (t ≈ SR·√years). Everything dying — including a confirmed-real risk premium (index VRP) — is the signature of a miscalibrated ruler, not an empty opportunity set. The fixes are convergent: **(1)** calibrate the gate with positive and negative controls and adopt a **two-track acceptance** framework (Track A: standalone alpha; Track B: book-level diversifier delta); **(2)** for event strategies, move significance from fold-Sharpes to **event-level inference** (panel of hedged event returns, two-way clustered errors) — the data has hundreds of independent announcement days, not 8 folds; **(3)** stop harvesting the options data as *strategies that pay options spreads* and harvest it as **signal executed at equity cost** — the spread wall that killed OPT-3 cannot reach an equity book; **(4)** make the live book **boring** (replay-diff, fill calibration) before any new capital; **(5)** impose **pre-registration + a research registry**, because the program's true trial count is far above any DSR constant.

### The reframed mission (Alpha-v6)

> Run a small book of robust premia — trend (broadened) as the capital base, possibly a tiny regime-gated index-VRP sleeve — accepted at the **book** level; pursue **one** concentrated genuine-alpha effort: the **earnings-event feature panel + options-conditioned PEAD v2 + options-derived cross-sectional equity signals**, accepted at the **event-panel** level; with positive-control-calibrated gates, pre-registration, the sacred holdout, and live replay-diff reconciliation as the final arbiters.

This is a sharpening of Alpha-v4 ("portfolio of uncorrelated premia"), not a reversal: Alpha-v4 built the sleeves and allocator; Alpha-v6 fixes the *measurement* (power, inference unit, acceptance tracks) and aims the options data correctly (signal-first).

---

## 2. Consensus map

Scored by how many of the 5 reviewers raised the point (explicitly or in substance), with my confidence after grounding in the code.

### STRONG CONSENSUS (act on these)

| # | Point | Raised by | Confidence | Notes |
|---|---|---|---|---|
| C1 | **Harness Type-II / power problem** — the gate kills true edges of the size findable on this data budget; under-powered KILLs were read as "no edge" | 5/5 (Claude quantified it; Gemini "hiding edge"; DeepSeek "under-powered conviction"; Grok "under-powered KILLs"; ChatGPT "underpowered") | **HIGH** | Arithmetic, not opinion. Confirmed by the gate's own record: only PEAD ever passed PAPER, and only via waiver. |
| C2 | **Two-track acceptance** — standalone-alpha gate is the wrong ruler for risk premia / diversifiers; add a book-delta gate | 5/5 (Gemini marginal-Sharpe; DeepSeek `book_level_diversifier_gate`; Grok book-level framework; ChatGPT component types; Claude Track A/B) | **HIGH** | The app's own 2026-06-09 pause decision ("alpha gate vs risk premium") already discovered this; the reviewers generalize it. |
| C3 | **Event-LEVEL inference for PEAD** — the independence unit is the announcement-day cluster, not the fold; panel of hedged event returns with two-way clustered errors (announce-day × firm) | 5/5 in substance (Claude sharpest/two-way panel; ChatGPT cluster-primary; Gemini/Grok event-clustered bootstrap; DeepSeek event-weighted regime) | **HIGH** | A quarter-cluster bootstrap already exists (`scripts/pead_significance.py`) — quarter clustering is the *over-conservative* end; day-level panel is the missing instrument. |
| C4 | **Options data as SIGNAL harvested at EQUITY cost** — expectation/crowding/uncertainty measurement, not options execution | 5/5 | **HIGH** | Empirically grounded in this repo: OPT-3 died on single-name spreads; OPT-5 (signal-only) produced the program's only alpha-like lift. |
| C5 | **Earnings-event feature panel as the research factory** — one row per event, every PIT feature a column, every future idea a regression not a new backtest | 4/5 (Claude #1, ChatGPT Phase B/C, Grok feature-pipeline, DeepSeek partially) | **HIGH** | Highest-EV single build. Increases power instead of consuming it. |
| C6 | **Continuous event score, never binary thresholds** — the OPT-5 fragility is partly a methodology artifact; re-adjudicate `realized/implied` as a continuous pre-registered feature | 5/5 (DeepSeek partially — still suggests one pre-registered threshold) | **HIGH** | 1.0 is the unique economically meaningful point; a plateau across arbitrary cuts was never the right prediction. The FRAGILE verdict stands *for the binary filter*; the continuous question is open. |
| C7 | **Specific options-as-equity-signal alphas**: CPIV (call−put IV spread, Cremers-Weinbaum), 25Δ put skew (Xing-Zhang-Zhao), O/S volume ratio, term-structure slope/kink, IV/RV richness, realized/implied reaction ratio | 5/5 (overlapping lists) | **MEDIUM-HIGH** | Academic priors are real; whether they survive 2022-26 + costs in *this* universe is exactly what Phase 4 tests. |
| C8 | **Index VRP = small book-level risk-premium diversifier behind a book gate**, not standalone alpha; do not keep parameter-mining it | 5/5 | **HIGH** | PF 2.24/1.75 is real; crisis-negative VRP + crisis-positive trend is the textbook pairing. 4y is too little for the tail — size accordingly (tail budget). |
| C9 | **Pre-registration + research registry + explicit trial accounting** | 4/5 (ChatGPT, Claude, Grok, DeepSeek-implicit) | **HIGH** | Does not exist in the repo today. DSR's constant (`N_TRIALS_TESTED=300`) cannot represent iterative human/LLM research. |
| C10 | **Live-vs-backtest reconciliation as a promotion-critical layer** (replay-diff, fill_quality, slippage calibration) | 4/5 (ChatGPT #1 priority; DeepSeek decay tracker; Grok #5; Claude first-class metric) | **HIGH** | Trackers exist (pead/trend/allocator) but no spec-replay diff, no fill-quality table, no significance-tested decay check. |
| C11 | **Gate calibration with positive + negative controls** — the gate has never been run against known-real strategies and random nulls | 2/5 explicit (Claude centrally; DeepSeek's WF-optimism test adjacent) | **HIGH** | Cheap, decisive, and the empirical answer to C1. If TSMOM-on-4y fails its own gate, the Type-II critique is proven; if it passes, recalibration is unnecessary. Run first. |
| C12 | **Options EOD-mark realism**: stale forward-filled marks smooth equity curves (inflated Sharpe/Calmar); trade-at-close circularity; EOD close OK for marks, not for execution | 4/5 (Gemini "toxic"; Claude mechanics; ChatGPT discount; Grok realism gaps) | **HIGH** | `OptionsSimulator` forward-fills untraded closes (documented). No stale-fraction metric exists. Must fix before any new options-execution verdict (Phase 6 precondition). |
| C13 | **Spread/fill calibration against live NBBO** — the modeled half-spread was never calibrated to observed quotes | 2/5 (Claude, ChatGPT) | **HIGH** | Cheapest high-impact fix; the snapshot endpoint serves NBBO live; log it nightly. |
| C14 | **Multi-factor residual alpha** — SPY-only CAPM is one afternoon away from a small ETF factor set (size/momentum/value/vol) | 2/5 (Claude, ChatGPT sector-hedge) | **HIGH** | `scripts/walkforward/attribution.py` is 79 lines, single `capm_alpha()`. Verified single-factor. |
| C15 | **Trend broadening = the boring capital-grade EV** (more legs, vol-targeting, where 19y of data gives the significance machinery actual power) | 1/5 explicit (Claude) + Grok's premise reframe implies it | **HIGH despite 1/5** | The only research direction where t≥2 is *reachable*. Low glamour, high certainty. |
| C16 | **Premise reframe** — "single-operator capital-grade standalone alpha at t≥2.5" is unreachable arithmetic; run premia as the base, alpha hunt as the satellite | 3/5 (Grok, ChatGPT, Claude) | **HIGH** | Adopted as the Alpha-v6 mission statement above. |
| C17 | **Options surface-quality layer** (stale-close filters, IV-solver flags, arbitrage sanity, corp-action guards) + **persist computed IV/greeks** (one-time engine pass; de-risks Polygon dependence) | 2/5 (ChatGPT quality layer; Claude persistence) | **HIGH** | Both prerequisites for the panel and the XS signals; the persistence also makes every panel query a join, not a 112M-row pricing run. |

### MINORITY / CONTESTED (resolved in §3)

| # | Point | For | Against | Resolution |
|---|---|---|---|---|
| X1 | **Dispersion trading** as a top-2 direction | Gemini #2, DeepSeek #1, Grok #2 | Claude: explicit anti ("both legs pay single-name spreads — the OPT-3 cost wall as a portfolio") | §3(a): feature-first, trade-maybe-never |
| X2 | **Kill CPCV on 4y data outright** | Gemini | ChatGPT/Claude: demote to robustness; Grok/DeepSeek: keep + fix | §3(b): demote for event strategies, keep for path-dependent |
| X3 | **DSR multiplicity fix** `N_eff = k(k−1)/2` (makes DSR *less* conservative) | DeepSeek | ChatGPT/Claude: DSR is theater either way | §3(c): reject the fix; demote DSR to report-only |
| X4 | **MVO / risk-parity allocator** | Gemini | App evidence: equal-capital beat vol-weight and regime-tilt on 2 sleeves; inverse-vol is fooled by PEAD's sparse vol | §3(d): rejected at n=2–3 sleeves; revisit at ≥4 |
| X5 | **Implied borrow / put-call-parity short-squeeze signal** | Gemini (unique) | Nobody else; meme-era SI factor already reversed (A2, t=−3.53) | §3(e): one panel column, not a sleeve |
| X6 | **Replace the event-sparsity waiver with `event_regime_sharpes()`** | DeepSeek (unique) | — | Adopt inside Phase 3 (good idea; entry-day regime labels over events) |
| X7 | **Delta/vega circuit breakers, liquidity impact multiplier** | DeepSeek (unique) | — | Adopt as Phase 6 guardrails (only needed when options execution goes live) |
| X8 | **BMO/AMC event-time-aware as-of model** | ChatGPT (unique) | — | Adopt in Phase 2 — materially increases usable pre-event signal without leaking |
| X9 | **Earnings long-vol selectivity** | Claude #5 (conditional) | — | Free inside the panel; build nothing until the panel hands it over |
| X10 | **Vol-of-vol / VIX calendar spreads** | DeepSeek #4 | Data limits (no SPX options in store; VIX options absent) | Decline — data-blocked, low priority |

---

## 3. Reconciliation of the real disagreements

### (a) Dispersion trading — express as a FEATURE first; only ever trade index-vs-liquid-basket; probably never trade it

Gemini/DeepSeek/Grok rank dispersion #1–2 because it is the textbook "alpha-shaped" options trade. Claude's objection is decisive **for this app**: the single-name leg pays the exact spread wall that killed OPT-3 *empirically in this codebase* (canonical earnings IV-crush: −1.02 @1×, −1.67 @2×). A basket of 30–50 single-name straddles, delta-hedged, rebalanced — at retail OPRA spreads with no NBBO history — is the OPT-3 cost structure multiplied.

**Resolution (sequenced, cheap to falsify):**
1. **Phase 2/4:** compute the *information* for free — `implied_corr_proxy = f(index IV, weighted single-name IVs)` as a daily book-level regime feature, and per-name `IV_RV_ratio` / `IV_vs_sector_richness` as cross-sectional columns. If relative-VRP mispricing is real, it shows up as an equity-signal or sizing edge at equity costs.
2. **Only if** the cross-sectional richness feature shows robust signal (Phase 4 pass), pre-register ONE execution test: index condor vs a basket restricted to the ~20 most liquid single-name option chains, 2× (and 3×) spread stress, DeepSeek's liquidity multiplier active. KILL on first failure; no parameter search.
3. Never trade broad-basket dispersion on this data. (No NBBO, no OI, EOD marks — the execution claim is unverifiable even if the signal is real.)

### (b) CPCV's role — demote, don't kill

Gemini's "CPCV on 4y is theater" overstates: the paths are correlated (the app already documents N_eff≈n_folds and corrected `unique_obs` for DSR), but CPCV still does real work — coverage maps, fold-skip honesty, OOS plumbing, and it remains the *right* harness for path-dependent trained models (swing/intraday — both dead, but the harness outlives them) and for sequencing-sensitive sleeves (trend variants).

**Resolution:** formalize what is already half-true in practice:
- **Event/rules-based strategies (PEAD family, options signals):** the **event panel + two-way clustered inference (Phase 3)** is the *significance* instrument; CPCV (full-coverage via `is_trained=False`) is the *robustness/coverage* check. Promotion requires both: panel significance AND no pathological CPCV fold (P5 floor stays).
- **Trained / path-dependent strategies:** CPCV with per-fold retraining stays primary, plus the existing `sequential_baseline.py`.
- Write this split into `PIPELINE_ARCHITECTURE.md` §7 so no future review re-litigates it.

### (c) DSR — reject DeepSeek's "fix", demote DSR to reporting

DeepSeek's `N_eff_trials = k(k−1)/2` (=15 for k=6) would make DSR *less* conservative precisely when the program's true trial count is *enormous* (hundreds of registered runs + unaccountable iterative human/LLM selection — ChatGPT's meta-overfitting point). It also doesn't touch the documented saturation defect (KL-1: p→1.0 for any Sharpe>~2 regardless of N).

**Resolution:** DSR becomes **report-only** (labeled SATURATED where applicable — already planned in KL-1). The real multiplicity defenses, in order: (1) the research registry + pre-registration (Phase 0), (2) the forward sacred holdout (exists), (3) gate thresholds calibrated empirically via positive/negative controls (Phase 0), (4) live paper as the fidelity (not edge) check. Do not spend effort on a better DSR formula.

### (d) Allocator sophistication — Gemini's MVO is rejected on the app's own evidence

The allocator already exists (`app/strategy/sleeve_allocator.py`: equal/vol/regime, inverse-vol risk parity, hysteresis; live-wired, disabled) and was *validated*: on PEAD+trend 2020-26, equal-capital (+1.082) beat static vol-weight (+0.715) beat regime-tilt (+0.593, 2.6× turnover). Inverse-vol over-weights PEAD because sparse event books look low-vol. MVO on 2–3 sleeves with ≤6y overlap estimates a covariance matrix from noise. **Resolution:** equal weights until ≥4 sleeves with ≥3y joint live/backtest overlap; then re-run the existing `run_book_allocator.py` gate. "Complexity must earn it" stands.

### (e) Implied borrow (Gemini, unique)

A real institutional signal, but: it requires accurate put-call-parity inversion off EOD closes (stale legs → garbage borrow estimates), and this app already proved the adjacent SI anomaly *reversed* 2020-26 (A2: −1.21, t=−3.53). **Resolution:** one column in the event panel / XS feature table (`pcp_borrow_proxy`, quality-filtered), evaluated like every other column. Not a sleeve, not a phase.

---

## 4. Where the reviewers were WRONG or out-of-date about THIS app

Intellectual honesty both ways — the reviews also under-credit what exists:

1. **"No sacred holdout" (Grok #4, top-5 change) — WRONG.** `SACRED_HOLDOUT_START="2026-11-09"` is a *forward* holdout enforced in code (`app/ml/retrain_config.py`; `retrain_as_of()` clamps every fold-end below it; runs reaching it raise unless explicitly bypassed). Claude correctly identified it and called it excellent. No action needed beyond protecting it.
2. **"Use event-clustered bootstrapping" as if new (Gemini, Grok) — already exists.** `scripts/pead_significance.py` implements a quarter-cluster block bootstrap (10k resamples, return-shift null, clustered-vs-iid CI comparison, Newey-West t). What's missing is the *event-level panel with two-way clustering* (Phase 3) — a power upgrade, not a from-scratch build.
3. **"Build an event harness / options adapter" — already built.** `EventEdgeStrategy` (`scripts/walkforward/event_edge.py`, emits `FoldResult` with `daily_returns_dated`), `OptionsStrategy` adapter (`scripts/walkforward/options_strategy.py`), `OptionsSimulator`, the 4-contract options stack, `ImpliedMoveProvider` — all exist and are validated. New work plugs in above the adapter.
4. **"Allocator is primitive / build risk-parity" (Gemini) — out-of-date.** Built, validated, and the sophisticated variants *lost* (§3d).
5. **"No live decay tracker" (DeepSeek) — partially exists.** `pead_tracker` (weekly realized-Sharpe-vs-+0.546 rollup), `trend_tracker` (vs +0.71), allocator_tracker. Missing: significance-tested decay and the replay-diff (Phase 1 adds them).
6. **"Extend the regime model" (Grok) — it already retrains weekly** (revived PR #439, fixed 3-class gate), and the regime *allocation* layer was tested and failed its margin. Options-derived regime features are legitimate future inputs (Phase 2 emits them) but the regime layer itself is not under-leveraged — it's correctly benched.
7. **DeepSeek's "re-run OPT-5 with a pre-registered 0.9 threshold might show a plateau" — superseded by data.** The sweep ran {0.75, 1.0, 1.25}: marginal at 0.75, spike at 1.0, *negative* at 1.25. Another binary threshold is the wrong move; the continuous re-adjudication (C6) is.
8. **Gemini's "synthetic stops underestimate gaps" — partially addressed.** The Phase-1 gapper-slippage stress (30–50bps) is exactly this test; PEAD survives 30, collapses at 50. Known and priced into the "weak satellite" verdict.
9. **Doc drift found during verification:** `CLAUDE.md`'s quick-reference table says `N_TRIALS_TESTED=250`; the code (SSOT) says **300** (updated 2026-05-31). Cosmetic, but fix the table (the prompt's app-facts also carried 250).
10. **Claude's "verify the 733-underlying backfill was built from the R1K union incl. delisted names" — legitimately open.** `backfill_options.py --r1k` claims "Russell-1000 union + index ETFs"; whether that union is the PIT union (incl. 2022-26 delistings) deserves the one-hour audit Claude asks for (folded into Phase 2 acceptance).

---

## 5. THE MULTI-PHASED BLUEPRINT

Seven phases. Phases 0–2 are foundational (cheap, fast, de-risk everything downstream). Phase 3 is the alpha centerpiece. Phases 4–6 are gated on earlier results. **Single-operator rule: never more than two phases in flight; one research phase + one infra/ops phase.**

```
P0 calibrate ruler ──┬─► P3 event panel + PEAD v2 ──► P4 XS options-signal sleeve
P1 live fidelity ────┤        (Track A)                    (Track A)
P2 options features ─┘
P5 trend broadening (Track B; parallel with P2–P4, independent)
P6 index VRP micro-sleeve (Track B; needs P0 + P1-spreads + sim-mechanics fixes)
```

---

### Phase 0 — Calibrate the ruler: gate controls + two-track acceptance + research registry

**Goal:** make every subsequent KILL/KEEP verdict trustworthy by measuring the gate's false-negative rate, splitting acceptance into alpha vs book-delta tracks, and starting trial accounting.
**Why now:** every other phase's output is judged by this instrument. Until the positive controls run, *every historical KILL has an unknown Type-II rate* (Claude), and Track B is the only honest home for index VRP and trend extensions (5/5 consensus C2). Cheapest, highest-leverage week in this document.

**Design.**
- **Positive controls** (known-real strategies, run through the production `CPCVResult.gate_passed()` on the same windows we actually use): (i) SPY buy-and-hold 2022-26; (ii) plain 12-1 cross-sectional momentum decile L/S; (iii) TSMOM restricted to 2022-26 (we *know* its 19y Sharpe is +0.71 — does a 4y window pass its own gate?); (iv) PEAD baseline (known marginal).
- **Negative controls:** 5 seeded random-signal scorers through `EventEdgeStrategy` (should fail ≥95% of the time), plus 1 deliberately-leaky scorer (uses t+1 close; should pass — proving the harness *can* detect a too-good result via the implausibility ceiling).
- **Acceptance:** an empirical operating-characteristic table (pass-rate by true-SR bucket). If positive controls with true SR 0.5–0.7 pass <50% at PAPER, recalibrate PAPER thresholds to the point where positives pass and nulls fail (pre-register the recalibration rule *before* looking: e.g. "lower PAPER_GATE_MIN_TSTAT until ≥2/3 positives pass while ≤1/5 nulls pass; floor 1.5"). If TSMOM-on-4y *passes* unchanged, the power critique is overstated — log that and leave thresholds alone (Claude's falsifiability test).
- **Two-track acceptance:**
  - **Track A (alpha):** for event/XS strategies — event-panel significance (Phase 3 instrument; until then, the existing clustered bootstrap), multi-factor residual alpha (Phase 1d), costs incl. stress, CPCV as robustness backstop. PAPER/CAPITAL tiers as today, with recalibrated thresholds.
  - **Track B (book-delta):** for risk premia / diversifiers / overlays — pre-registered improvement to the *combined book*: ΔSharpe ≥ +0.10 and ΔCalmar ≥ 0 and no worse maxDD and corr-to-book < 0.3 and a joint-tail test (candidate's worst-1% days must not coincide with the book's worst-1%) at a fixed risk budget (≤10%), net of turnover. Standalone floor: vol-targeted SR > 0.2 (DeepSeek). Track B NEVER feeds CAPITAL without explicit owner sign-off + a tail-loss budget.

**Implementation plan.**
- `scripts/walkforward/gate_calibration.py` — `run_control(control_id) -> CalibrationRow`, `run_all_controls()` emitting `docs/living/` -linked artifact `logs/gate_calibration_YYYYMMDD.json` + a printed OC table. Controls implemented as `EventEdgeStrategy`/`TSMOMStrategy` thin wrappers; random scorers seeded `1303+i`.
- `scripts/walkforward/book_gate.py` — `book_delta_gate(base_daily: pd.Series, candidate_daily: pd.Series, weights, *, registered: BookGateCriteria) -> BookGateResult` reusing `app/strategy/sleeve_allocator.build_book`; `BookGateCriteria` frozen dataclass loaded from `app/ml/retrain_config.py` new constants (`TRACKB_MIN_SHARPE_DELTA=0.10`, `TRACKB_MAX_CORR=0.30`, `TRACKB_MIN_STANDALONE_SR=0.20`, `TRACKB_MAX_RISK_BUDGET=0.10`, `TRACKB_JOINT_TAIL_PCTL=0.01`).
- **Research registry:** `data/research_registry.db` (sqlite, mirrors pead_tracker pattern) + `scripts/registry.py` (`register`, `preregister-criteria`, `record-result`, `list`). Columns: `hypothesis_id, parent_id, family, label(exploratory|confirmatory|live_confirm), features, params, universe, window, folds, cost_model, code_commit, data_hash, mechanism, acceptance_criteria, preregistered_at, run_at, result_json, decision`. Enforcement: `run_pead_*` / `run_*_cpcv` scripts gain `--hypothesis-id` (warn-only for 2 weeks, then required for any run labeled confirmatory; `preregistered_at < run_at` asserted).
- `event_regime_sharpes()` in `scripts/walkforward/regime.py` (DeepSeek X6): label each *event entry day's* regime, Sharpe per regime across events → retires the paper-only waiver for event strategies once validated.
- DSR: label report-only/SATURATED in `reports.py` (KL-1 plan); no formula change.
- Tests: `tests/test_gate_calibration.py` (null pass-rate bound, leaky-control detection), `tests/test_book_gate.py` (joint-tail math, registered-criteria immutability), `tests/test_registry.py`.

**Validation:** the phase validates the validator — exit = OC table published, thresholds either recalibrated (documented in `PIPELINE_ARCHITECTURE.md` §7 + changelog) or confirmed, Track B spec merged as §7.0-B, registry enforced.
**Effort:** **M, ~1–1.5 wk.** **Dependencies:** none. **Risks:** recalibration becomes threshold-shopping → mitigated by pre-registering the recalibration rule itself before running controls.

**🔬 P0 RESULT (2026-06-10 — harness shipped #444, full 16-control run complete):** **The "lower the t-bar" hypothesis is REFUTED.** `tsmom_4y` (t=6.72) and `tsmom_19y` (t=4.46) pass the significance CORE but fail the PAPER gate ONLY on the `worst_regime_sharpe` backstop; **3/5 TRUE zero-SR nulls cleared t≥2.0** (so lowering t* admits noise — the pre-registered rule returned `NO_ADMISSIBLE_TSTAR`); PEAD's t=3.33 ≈ a noise null (t=3.47). Type-I control is sound (0/10 nulls pass the full gate; leaky control flagged implausible). **Recast of this phase's remaining work:** do NOT touch significance thresholds. Instead prioritize **(a) two-track acceptance** (Track B book-delta — the lever that actually unblocks crisis-diversifiers like TSMOM, which die on the standalone worst-regime floor) and **(b) bringing forward event-level / cluster inference** (P3) to replace the 8-fold path t-stat, now empirically shown to be an unreliable discriminator. See `ML_EXPERIMENT_LOG.md` + `DECISIONS.md` 2026-06-10.

---

### Phase 1 — Live-book fidelity: replay-diff, fill quality, spread calibration (make the book boring)

**Goal:** before any new capital or sleeve, make PEAD+trend live behavior byte-for-byte explainable vs research, and start collecting the empirical cost data every future phase needs.
**Why now:** ChatGPT's #1 priority and the app's own bug history (VIX block, sizing cap, dead-ML proposals, trend orphan risk) say live divergence is the largest realized risk. Also: the NBBO logger must start NOW because it needs 4–6 weeks of accumulation before Phase 6 can use it. Runs **in parallel** with Phase 0 (infra vs research lanes).

**Design.** Three artifacts:
1. **StrategySpec replay-diff.** A canonical, versioned spec per live sleeve (PEAD, trend) emitting for each session: signal timestamp, eligible universe, entries/exits intended, sizes, every gating reason. Replay the same historical days through (a) the research scorer, (b) the backtest simulator, (c) the live decision path (from `decision_audit` + `proposal_log` + trackers), and diff at each stage. "Eligible for more capital" = diff report is boring for 4 consecutive weeks (only explainable diffs: fill timing, whole-share rounding).
2. **fill_quality table.** Every paper order: intended reference price, submitted limit, fill price/delay/ratio, slippage vs next-open, spread estimate, liquidity percentile, event flag. Backtest cost assumptions re-checked against the empirical distribution monthly.
3. **Options NBBO snapshot logger.** Nightly job hits `get_current_snapshot` for a fixed panel (SPY/QQQ/IWM chains + 20 liquid single names), persists bid/ask/IV/OI → after 4–6 weeks, fit half-spread% by (moneyness × DTE × underlying-class) buckets → replaces the flat spread assumption in `cost_models.py`.

**Implementation plan.**
- `app/live_trading/strategy_spec.py` — `@dataclass(frozen=True) StrategySpec` + `PEADSpec`/`TrendSpec` builders reading the same `pm.*` config the live path reads; `scripts/replay_diff.py --sleeve pead --date 2026-06-12` → `logs/replay_diff_*.json` + weekly email via `notifier.enqueue("replay_diff_weekly", …)`.
- `app/live_trading/fill_quality.py` + `data/fill_quality.db` (schema above; hooked where `pead_tracker.record_daily` already intercepts fills; trend fills via the rebalancer's order commit path hardened in #430).
- `scripts/log_options_nbbo.py` (scheduler 15:55 ET) → `data/options_spread_obs.parquet`; `scripts/fit_spread_model.py` → `data/options_spread_table.parquet`; `scripts/walkforward/cost_models.py` gains `CalibratedSpreadModel(table_path)` falling back to the current flat model when the table is absent.
- **Decay check:** extend `pead_tracker`/`trend_tracker` weekly rollups with a permutation test (live window vs same-period backtest; flag p<0.05) — DeepSeek's tracker, ~50 lines each.
- **Gate-language fix (Claude 1.2c):** in `retrain_config.py` + `PIPELINE_ARCHITECTURE.md`, re-word `CAPITAL_GATE_REQUIRE_PAPER_CONFIRMATION`: paper confirmation = *implementation fidelity* evidence only; it can never substitute for edge significance (the OR-path becomes AND-shaped: t≥2.5 required, paper confirms fidelity).
- **(d) Multi-factor residual alpha** (small, fits here): `scripts/walkforward/attribution.py` gains `factor_alpha(r_book, factors: dict[str, pd.Series], hac_lag=5)` with the default ETF set {SPY, IWM−SPY, MTUM−SPY, VLUE−SPY, VIXY} (daily, yfinance); `CPCVResult` fields `residual_alpha_multifactor_t` etc. Diagnostic first; becomes Track-A gating for capital per ChatGPT §2.2-C.
- Tests: replay-diff golden-date test (a frozen day's spec replays identically), fill_quality schema/upsert, spread-model fallback, factor_alpha vs statsmodels reference.

**Validation:** 4 consecutive boring weekly diff reports for PEAD and trend; fill_quality populated for ≥90% of paper orders; spread table fitting after data accumulates; multifactor alpha reported on every CPCV.
**Effort:** **M, ~2 wk build** (spread *calibration* completes ~wk 6 on accumulated data). **Dependencies:** none (parallel with P0). **Risks:** decision_audit granularity insufficient for full replay → mitigate by adding missing fields to the audit row (additive), not rebuilding.

---

### Phase 2 — Options feature layer: persisted greeks, surface quality, event-time-aware snapshots

**Goal:** turn the 112.8M-bar store into a queryable, quality-filtered, PIT feature asset: per-contract computed IV/greeks persisted once, a daily per-underlying feature table, and a BMO/AMC-aware event snapshot rule.
**Why now:** every downstream alpha phase (3, 4) consumes these columns; persistence makes panel queries joins instead of pricing runs and de-risks the Polygon dependence (Claude C17). Pure infra with zero research-overfitting surface — safe to build while Phase 0 calibrates.

**Design.**
- **Persisted computed greeks:** one engine pass over the store (chunked by underlying; BjS American + dividends + real rate series as in OPT-1a), output keyed (contract, date): `iv, delta, gamma, vega, theta, solver_status, stale_flag, volume`. `stale_flag` = no volume that day (mark is a forward-fill candidate).
- **Surface-quality layer (ChatGPT Gap 5):** filters applied at read time: min volume/notional, stale-run length, IV-solver failure, vertical/calendar arbitrage sanity bounds, split/corp-action guards (price-jump vs underlying check), with per-underlying daily coverage stats. Every consumer (panel, signals, simulator marks) reads through it.
- **Event-time-aware snapshot rule (ChatGPT X8):** `announcement_ts` with BMO/AMC flag from FMP/Finnhub; `pre_event_snapshot_date` = same-day close for AMC (causal: the 16:00 close precedes the report), previous close for BMO/unknown; `post_event_decision_time` = next open, explicit. Unit-tested against the knowable_date contract (the snapshot date's bar is knowable +1bd — the *decision* day is later, which the existing `knowable_asof` parameter in `ImpliedMoveProvider` already supports).
- **Daily per-underlying feature table:** `atm_iv_30d, implied_move_front, cpiv_matched_delta, skew_25d_put, term_slope_30_60, iv_rv_20d_ratio, opt_share_volume_ratio, put_call_volume_ratio, opt_volume_z, pcp_borrow_proxy` + quality/coverage flags.
- **Survivorship audit (Claude 1.2f):** verify the 733-underlying backfill list derives from the PIT R1K union incl. 2022-26 delistings; document in `OPTIONS_DATA.md`.

**Implementation plan.**
- `scripts/backfill_computed_greeks.py` (resumable, per-underlying chunks, `--workers`) → `data/options_greeks.parquet` (partitioned by underlying). Estimate: 6.18M contracts × ~18 bars avg; IV solve ~50µs ⇒ hours, not days; run overnight.
- `app/data/options_quality.py` — `QualityFilteredOptionsReader(provider, greeks_path)` with `get_quality_bars(underlying, as_of)`, `coverage_report(underlying)`.
- `app/data/event_time.py` — `EventTimeModel.pre_event_snapshot_date(symbol, announce_date) -> (date, CausalityFlag)`.
- `scripts/build_options_features.py` → `data/options_features.parquet` (date × underlying), rebuilt incrementally.
- Update `docs/reference/OPTIONS_DATA.md` (new stores §8) + `OPTIONS_PROGRAM.md` (layer diagram gains a FEATURES box above DATA).
- Tests: greeks-vs-engine spot check, quality-filter golden cases (stale run, arb violation, split), BMO/AMC causality (AMC same-day allowed, BMO same-day refused), feature-table PIT test (no column uses bars with `knowable_date > as_of`).

**Validation:** snapshot-validation re-run on the persisted greeks (near-ATM median |IV err| < 0.010 holds); coverage report ≥95% of PEAD-qualified events have pre-event implied move + skew + CPIV; survivorship audit logged.
**Effort:** **M, ~1.5–2 wk** (+1 overnight compute). **Dependencies:** none hard; parallel with P0/P1. **Risks:** IV solver failures in tails → recorded as `solver_status`, filtered, never silently defaulted; corp-action contamination → guard + spot-audit the 20 largest movers.

---

### Phase 3 — The earnings-event panel + event-level PEAD inference + PEAD v2 continuous score (THE centerpiece)

**Goal:** build the one-row-per-earnings-event research table, re-adjudicate PEAD (and the implied-move signal) at *event level* with two-way clustered errors, and ship a pre-registered continuous options-conditioned event score (PEAD v2).
**Why now:** highest-EV research move by consensus (C3/C5/C6); it can *upgrade the verdict on the existing live edge* (the same data that bootstraps p=0.19 at quarter level can be p<0.01 at day level if the edge is real — or confirm PEAD is marginal, which is equally decision-relevant), and it converts every future event idea into a column.

**Design.**
- **Panel population:** all R1K earnings events 2022-06→2026-06 with options coverage (~4y × ~1.9k events/yr ≈ 7–8k events; options-covered subset smaller), plus an equity-only extension 2019→2026 for the non-options columns (more power for the equity features; options columns NaN before 2022-06).
- **Inference instrument:** OLS/rank regression of hedged forward returns on features with **two-way cluster-robust (announcement-date × firm) standard errors** (Cameron–Gelbach–Miller), plus the existing quarter-cluster bootstrap retained as the conservative bound. Decile monotonicity checks. Robustness: leave-one-quarter-out, leave-one-sector-out, leave-top-10-events-out.
- **Three pre-registered confirmatory hypotheses (registered in Phase 0's registry BEFORE running):**
  - **H1 (PEAD base):** mean 5/10/20-day SPY-hedged event return of the current PEAD-qualified population > 0 at day×firm clustering. *This re-adjudicates the live edge.*
  - **H2 (implied-move, continuous):** the continuous `reaction_ratio = |announce move| / pre-event implied move` has a negative monotonic relationship with hedged forward drift (under-reaction proxy) on the FULL 4y panel. No thresholds; coefficient + decile monotonicity. *This settles OPT-5 properly.*
  - **H3 (PEAD v2 score):** a regularized monotonic scorecard (logistic/linear or shallow GAM bins — explicitly NOT XGBoost; ChatGPT's small-cluster warning) on a pre-registered feature list (SUE, revision momentum, announce gap vs vol, reaction_ratio, iv_runup, CPIV_pre, skew_pre, opt_volume_z, post_iv_retention) trained 2022-24, validated on 2025-26 (sacred holdout untouched), produces top-vs-bottom-decile hedged spread > 0 with day-clustered t ≥ 2 and monotonic deciles.
- **Acceptance / kill:**
  - H1 p<0.05 → PEAD upgraded from "waiver-paper" to honest Track-A paper; H1 p>0.15 → PEAD is genuinely marginal → live book becomes trend-plus-cash while the panel hunts (pre-committed; Claude's falsifiability).
  - H2/H3 pass → PEAD v2 goes to paper telemetry behind the replay-diff (Phase 1) with evidence-tiered sizing (research → paper → micro-capital); any failure → logged, parked, no threshold-hunting.
  - Additional caps: no single quarter >40% of P&L, no single name >15%, survives the gapper-slippage stress (50bps) and the fill_quality empirical distribution.

**Implementation plan.**
- `app/research/event_panel.py` + `scripts/build_event_panel.py` → `data/event_panel.parquet`. **Schema (one row per event):**
  `event_id, symbol, announce_date, announce_ts_flag(BMO|AMC|UNK), sector, mktcap_decile,`
  `sue, revision_momentum, announce_gap_pct, gap_vs_vol20, prior_qtr_drift, pead_score_v1,`
  `pre_event_implied_move, iv_runup_t10_t1, reaction_ratio, cpiv_pre, skew_25d_pre, term_kink_pre, opt_volume_z_pre, post_iv_retention_t1,`
  `fwd_ret_{1,3,5,10,20}_raw, fwd_ret_{1,3,5,10,20}_spyhedged, fwd_ret_{5,10,20}_sectorhedged,`
  `beta_60d, entry_open_next, options_coverage_flag, quality_flags`.
  All option columns via Phase 2's quality reader + event-time snapshot rule; all equity columns PIT (`knowable_date` discipline as everywhere).
- `scripts/walkforward/event_inference.py` — `twoway_cluster_ols(panel, y, X, clusters=("announce_date","symbol")) -> InferenceResult`; `decile_report(panel, feature, y)`; `loco_robustness(...)`. (statsmodels covtype or a 60-line CGM implementation; unit-test against a published reference case.)
- `scripts/run_event_panel_inference.py --hypothesis-id H1|H2|H3` — registry-enforced confirmatory runner; results appended to `ML_EXPERIMENT_LOG.md` via `/log-wf`-style skill.
- PEAD v2 scorer: `app/ml/pead_v2_scorer.py` implementing the AgentSimulator factor-scorer contract → drops into `EventEdgeStrategy` unchanged for the CPCV robustness leg; live wiring later reuses the PEAD live path (config-pinned like `pm.pead_*`).
- DeepSeek's `event_regime_sharpes()` (Phase 0) applied here → the regime waiver retired for event strategies.
- Tests: panel PIT test (every feature reproducible from as-of data), inference reference test, scorer determinism, no-options-coverage fallback.

**Validation:** Track A — panel inference primary, full-coverage CPCV (`run_pead_cpcv.py` + v2 scorer) as robustness, multi-factor residual alpha (Phase 1d) required positive for any capital path.
**Effort:** **L, ~3–4 wk.** **Dependencies:** P0 (registry + recalibrated gate), P2 (features). **Risks:** options-covered subsample too thin for H3 (mitigate: H3's equity features run on the 2019+ extension; options features evaluated on 2022+ with explicit coverage-interaction term); meta-overfitting via feature iteration (mitigate: exploratory-vs-confirmatory labels, one confirmatory run per hypothesis).

---

### Phase 4 — Options-derived cross-sectional EQUITY signals (CPIV / skew / O/S / term-slope / IV-RV)

**Goal:** test the academically-documented options-information signals as a weekly dollar-neutral L/S **equity** sleeve — options information at equity cost.
**Why now:** the consensus #2 alpha direction (C7); the dollar-neutral L/S engine (built and validated during the ranker post-mortem: neutral-at-target-gross, full-cross-section, breadth admission) and Phase 2's feature table make this a thin adapter, not a build. Sidesteps the spread wall entirely.

**Design.**
- Universe: R1K names passing the options-quality coverage filter (liquidity-filter IV inputs hard — Claude).
- Five pre-registered features (one confirmatory run each, registered together up-front; per-feature decile sorts BEFORE any multivariate combination): `cpiv_matched_delta` (long high / short low), `skew_25d_put` (short steep / long flat), `opt_share_volume_ratio` conditioned on put-heaviness (short high-put-O/S), `term_slope` around events, `iv_rv_20d_ratio` (as *equity* residual-return predictor, not vol trade).
- Weekly rebalance, dollar-neutral at target gross, equity costs (10bps + the empirical fill_quality distribution), sector/beta-neutralized returns reported alongside raw.
- **Acceptance (Track A):** day-clustered panel t ≥ 2 on the weekly L/S spread (weeks as clusters), decile monotonicity, positive multi-factor residual alpha, survives cost stress; CPCV robustness leg via a thin `EventEdgeStrategy`-style adapter. **Kill rule:** if simple decile sorts show nothing net of costs, close the line — do NOT escalate to ML combinations (this is explicitly *not* a revival of the dead XS-ML: different information set, but the same kill discipline).

**Implementation plan.**
- `app/ml/options_signal_scorers.py` — one scorer per feature reading `data/options_features.parquet` PIT.
- `scripts/run_options_xs_cpcv.py` — reuses the L/S engine + `run_cpcv`; `scripts/run_options_xs_panel.py` — weekly-spread panel inference via Phase 3's `event_inference.py` machinery (clusters = week).
- Registry entries H4a–H4e pre-registered with directions + acceptance before the first run.
- Tests: scorer PIT, neutrality invariants (net$ ≈ 0 at target gross — reuse the ranker-era observability).

**Validation:** Track A as above. Any survivor → paper telemetry behind replay-diff; book addition judged by Track B delta too (it must not just exist, it must help the book).
**Effort:** **M–L, ~2–3 wk.** **Dependencies:** P2 (hard), P0, P3's inference module (soft — can share). **Risks:** 4y window = one regime for these signals (acknowledge in the verdict; paper time becomes the extension); crowding decay post-publication (the literature effects are 2000s-2010s — expect attenuation, demand costs-net significance, accept a KILL gracefully).

---

### Phase 5 — Trend sleeve broadening (the boring, capital-grade move)

**Goal:** scale the one statistically-defensible asset: more legs, longer history, vol-targeted sizing — each addition judged on 19+ years (where t≥2 is actually reachable) and on Track B book delta.
**Why now:** highest-certainty EV in the document (C15); fully parallel with Phases 2–4 (different data, different code path); strengthens the capital base that funds the alpha hunt. Start only after Monday's first real rebalance is verified boring (Phase 1 replay-diff on trend).

**Design.** Candidate extensions, each a pre-registered Track B test on 19y: (i) broaden the ETF set (currency-hedged intl, more commodity legs, additional rates duration points); (ii) vol-targeting at sleeve level (current: inverse-vol weights; add a sleeve-level target); (iii) long-short on the existing set (short legs when below-MA *and* negative momentum — pre-registered, not fitted); (iv) lookback-blend re-weighting only if (i)–(iii) pass. Note `tsmom.py` already ensembles 21/63/126/252d lookbacks — reviewers under-credited this; the marginal EV is in (i)–(iii).
**Acceptance:** per-extension — standalone 19y Sharpe not degraded, Track B book delta vs the current 10-ETF sleeve (ΔSharpe ≥ +0.05 at equal risk, maxDD not worse, turnover-cost-net), crisis-fold attribution reported (2008/2020/2022 each).
**Implementation:** extend `app/strategy/tsmom.py` (universe/config-driven, no structural change), `scripts/run_tsmom.py --variant`, registry entries H5a–H5d; live config additive via `pm.trend_*` keys; rollout through the existing shadow → armed path with `trend_tracker` expectations updated.
**Effort:** **M, ~1–2 wk.** **Dependencies:** P0 (Track B), P1 (trend replay-diff boring). **Risks:** over-extension into exotic ETFs with short histories → require ≥15y per leg or exclude; whipsaw cost in long-short → 2bps cost model already validated, keep the cost-timing test.

---

### Phase 6 — Index VRP as a tiny book-level diversifier (behind the book gate)

**Goal:** give the one confirmed-real options return source (index VRP, PF 2.24/1.75) its honest home: a small (≤10% risk budget), regime-gated, defined-risk sleeve accepted ONLY on Track B book delta, with an explicit tail budget.
**Why last:** it needs Phase 0 (Track B exists), Phase 1's calibrated spread table (4–6 wk of NBBO accumulation), and the options-simulator mechanics fixes below. It is also the lowest-priority consensus item (real but Sharpe-flat; 4y cannot price the tail).

**Design.**
- **Simulator mechanics first (Claude's three flattering mechanics — preconditions, not optional):** (1) per-position **stale-mark fraction** on `SimResult` + a Geltner AR(1) unsmoothing check (weekly-return Sharpe comparison) — re-state OPT-4's PF under it; (2) **entry-fill convention**: select strikes off T-1, fill at T-day close or next open with doubled spread cost (kill the trade-at-close circularity); (3) **calibrated spreads** from Phase 1's fitted table replacing the flat assumption.
- **One pre-registered structure, zero parameter search:** SPY/QQQ/IWM, 30–45 DTE, ~16Δ short strikes (the canonical OPT-4 1.5×SD), defined-risk condor only, weekly cadence (power), risk-off overlay = the *existing* regime model / VIX regime (no new fitted parameters — Claude's guardrail), sized so a 2008-replay costs ≤2% NAV at the wings.
- **Acceptance (Track B):** registered book-delta criteria (Phase 0) on the joint PEAD+trend+VRP backtest AND on a forward paper period; survives 2× calibrated spread; DeepSeek's liquidity multiplier + delta/vega circuit-breaker limits wired into the RM before any live order (X7). Capital: much harder — extended live paper + owner sign-off + the tail budget codified in config.
- **Kill rule:** fails Track B → the VRP line closes for 12 months (DeepSeek's "do not re-test monthly" discipline; registry-enforced re-test date).

**Implementation:** `app/backtesting/options_simulator.py` (stale fraction, fill convention flag), `scripts/run_index_shortvol_cpcv.py` re-run under the registered spec, `scripts/walkforward/book_gate.py` consumption, RM circuit-breaker keys (`rm.options_max_book_vega_pct`, `rm.options_max_book_delta_pct`), `OPT-4b` row in `OPTIONS_PROGRAM.md`.
**Effort:** **M, ~2 wk** (after spread table lands). **Dependencies:** P0, P1 (spread data), sim mechanics. **Risks:** 4y can't price the tail → that's *why* it's tail-budgeted, tiny, and Track B; live options execution gap (Alpaca options) → shadow-first per OPT-8, fill reconciliation via fill_quality.

---

### Parallelization & single-operator load

- **Weeks 1–2:** P0 (research lane) ∥ P1 (infra lane; NBBO logger starts day 1) ∥ P2 kicked off (overnight compute).
- **Weeks 3–6:** P3 (research lane) ∥ P1 completion + P5 (infra/Track-B lane).
- **Weeks 7–10:** P4 (research lane) ∥ P6 preconditions → P6 (options lane).
- Live operations (Monday trend rebalances, PEAD telemetry) continue throughout; replay-diff makes them cheap to supervise.

---

## 6. Cross-cutting infrastructure & process

### The two-track gate spec (to be written into PIPELINE_ARCHITECTURE.md §7.0-B in Phase 0)
- **Track A (alpha):** event-panel/clustered significance (primary for event/XS) + CPCV robustness + multi-factor residual alpha + cost stress. PAPER/CAPITAL tiers retained; thresholds = post-calibration values; paper confirmation = fidelity evidence only.
- **Track B (book-delta):** pre-registered combined-book improvement at fixed risk budget, corr + joint-tail constraints, turnover-net; never auto-promotes to capital.
- Component types (ChatGPT): every registered hypothesis declares `component_type ∈ {alpha, risk_premium, diversifier, filter, tail_hedge}` → routes to the track.

### Research registry & pre-registration discipline
Schema in Phase 0. Rules: exploratory runs are unlimited but can never promote; confirmatory runs require `preregistered_at < run_at` and exactly one shot per hypothesis; re-tests require a registered cooling-off date. The registry is the program's true N_TRIALS.

### StrategySpec replay-diff contract
Phase 1. A sleeve is capital-eligible only after 4 boring weekly diffs. Every live divergence (overlay suppression, sizing clip, missed fill) must appear in the diff with a reason code — silence is a bug.

### Doc-update obligations (NO-DRIFT rule; per CLAUDE.md)
| Phase | Must update in the same PR(s) |
|---|---|
| P0 | `PIPELINE_ARCHITECTURE.md` (§7 gate changes, §7.0-B, changelog, KL-1 status), `retrain_config.py` docstrings, `MASTER_BACKLOG.md`, `DECISIONS.md` (calibration verdict + track framework), `PROJECT_STATE.md` |
| P1 | `MODEL_STATUS.md` (replay-diff status per sleeve), `PIPELINE_ARCHITECTURE.md` (cost_models change), `PROJECT_STATE.md` |
| P2 | `OPTIONS_DATA.md` (greeks/features stores, survivorship audit), `OPTIONS_PROGRAM.md` (layer diagram) |
| P3 | `ML_EXPERIMENT_LOG.md` (every H1–H3 run), `DECISIONS.md` (PEAD verdict change), `MODEL_STATUS.md` + `PROJECT_STATE.md` (if live posture changes), `PIPELINE_ARCHITECTURE.md` (event-inference module) |
| P4 | `ML_EXPERIMENT_LOG.md`, `OPTIONS_PROGRAM.md` (verdict table rows H4a–e), `DECISIONS.md` |
| P5 | `ML_EXPERIMENT_LOG.md`, `MODEL_STATUS.md` (trend sleeve config), `DECISIONS.md` |
| P6 | `OPTIONS_PROGRAM.md` (OPT-4b row), `PIPELINE_ARCHITECTURE.md` (OptionsSimulator changes), `ML_EXPERIMENT_LOG.md`, `DECISIONS.md` |
| All | `ML_EXPERIMENT_LOG.md` for any WF/CPCV/panel run; fix the stale `CLAUDE.md` N_TRIALS=250 → 300 quick-ref now |

---

## 7. Explicit KILL / DO-NOT list

- **Single-name earnings short-vol / IV-crush, any parameterization** — cost-killed twice, Opus-certified; spreads won't drop 80%.
- **Dispersion as a TRADE (broad single-name baskets)** — OPT-3's cost wall as a portfolio; feature-first only (§3a).
- **Binary threshold sweeps as a validation method** — the OPT-5 lesson; continuous features + pre-registration only.
- **Another DSR formula** — report-only; registry + holdout + calibrated gates are the real defense.
- **MVO/risk-parity allocator at n≤3 sleeves** — equal weights won on the app's own data; revisit at ≥4 sleeves.
- **XS price/fundamental ML revival; intraday 5-min ML** — three honest nulls; frozen as benchmarks, keep frozen.
- **Small/mid-cap PEAD re-runs** — weaker venue, honestly established.
- **Analyst-ratings drift (A1), short-interest factor (A2), short-term reversal, cross-asset carry** — closed with logged nulls; the registry enforces cooling-off.
- **Dealer-gamma proxies, intraday vol-arb, NBBO/quote alpha, vol-surface arbitrage, market-making** — structurally unsupported by the data (no OI / intraday / NBBO / surface history).
- **Paper-fill performance as edge evidence** — fidelity evidence only (Phase 1 language fix).
- **Single-name shorting without borrow data; LLM stock-picking; expensive alt-data** — standing STOP list carries over.
- **Re-tuning index VRP parameters before the Phase 6 registered re-test** — one registered structure, one shot.

---

## 8. Sequencing & success metrics

### 30 / 60 / 90-day shape
- **Day 30:** gate OC table published + thresholds settled (P0); registry enforced; replay-diff running on PEAD+trend with ≥2 boring weeks (P1); NBBO logger ~4 wks of data; greeks/features stores built (P2); event panel populated; **H1 (event-level PEAD) answered** — the single most important number of the quarter.
- **Day 60:** H2/H3 confirmatory runs done → PEAD v2 verdict (paper telemetry or parked); trend broadening variants tested on 19y (P5) → live config updated if Track B passes; spread table fitted; first two XS options signals (CPIV, skew) decile-sorted (P4).
- **Day 90:** remaining XS signals adjudicated; index VRP re-run under fixed mechanics + Track B verdict (P6); book = trend (broadened) + PEAD/PEAD-v2 (evidence-sized) ± VRP micro-sleeve; sacred holdout still untouched.

### Success metrics ("the next phase worked" =)
1. **Measurement:** gate OC table exists; positive controls pass / nulls fail at documented rates; every confirmatory run has a registry entry with `preregistered_at < run_at`.
2. **Inference upgrade:** PEAD has an event-level verdict (either an honest Track-A paper pass without the waiver, or a pre-committed demotion to trend-plus-cash). No strategy is ever again killed *or* kept on an 8-fold t-stat alone.
3. **Live fidelity:** 4+ consecutive boring weekly replay-diffs on both live sleeves; fill_quality coverage ≥90%; backtest cost assumptions reconciled to empirical fills.
4. **Options asset productized:** persisted greeks + quality layer + feature table consumed by ≥2 research phases; at least one options-derived signal with a clean confirmatory verdict (pass OR kill — both count).
5. **Book:** trend broadened with ≥1 Track-B-passing extension; book-level Sharpe (backtest, joint) ≥ the current PEAD+trend baseline with no worse maxDD; any VRP allocation strictly inside its tail budget.
6. **Discipline intact:** zero sacred-holdout touches; zero unregistered confirmatory runs; every verdict logged in `ML_EXPERIMENT_LOG.md` + `DECISIONS.md` same-PR.

---

*Start tomorrow: open the Phase 0 branch — `scripts/walkforward/gate_calibration.py` with the TSMOM-on-4y positive control first (it is the single cheapest experiment that can falsify or confirm the central premise of this entire document).*
