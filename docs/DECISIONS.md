# Decision Log
Append-only record of significant architectural and strategic decisions.
Format: `## YYYY-MM-DD — Title` then context, decision, rationale, consequences.

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
