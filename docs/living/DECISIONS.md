# Decision Log
Append-only record of significant architectural and strategic decisions.
Format: `## YYYY-MM-DD — Title` then context, decision, rationale, consequences.

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
