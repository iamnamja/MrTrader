# Ruler v2 — Design & Implementation Roadmap (Alpha-v7 Phase B)

**SSOT for the gate/acceptance redesign.** Strategic context: `ALPHA_V7_SYNTHESIS_AND_PLAN.md`
(Alpha-v7 = re-charter to operate a risk-premia book). This doc is the *technical* design an
implementer builds against. Authored from a repo-grounded Opus architect pass (2026-06-12).

## 0. Why (the mandate)

The 4-LLM panel's decisive critique: the promotion gate is a **Type-II / false-negative
machine** on ≤4y data. Its significance statistic — the CPCV **path-Sharpe t-stat**
`mean(path_sharpes)/(std/√n_folds)` over 15 paths reusing ~8 folds — **measures cross-fold
consistency, not significance** (a regime-homogeneous fluke gets t→∞; a real regime-heterogeneous
edge gets an inflated std → Type II). Ruler v2 replaces the significance evidence with inference
on the **OOS return series / panel** (where N is in the hundreds), and **inverts the tiers**
(paper = plausibility; capital = a Bayesian posterior that includes the live-paper track).

**Two repo realities that shape the design (verified):**
- The path-t is already `N_eff = n_folds` and **WARN-only by default** (`require_tstat_gate=False`).
  Ruler v2 *demotes it to report-only* and adds the new primary evidence — it isn't ripping out a
  load-bearing gate.
- **PBO has a data gap.** `run_cpcv` scores **one** config; Bailey/López-de-Prado PBO needs **M
  candidate configs** ranked IS-vs-OOS across the combinatorial splits. So PBO is a **standalone**
  function that consumes an externally-assembled N_configs×N_blocks matrix, and is **non-gating for
  a single sleeve** (M=1 → undefined). It only gates when a config *sweep* produced the candidate
  (the honest selection-bias case).

## 1. Module / file map

**New (pure unless noted):**
| File | Purpose |
|---|---|
| `app/research/inference.py` | The pure inference core: `hac_sharpe` (Lo 2002), `stationary_bootstrap_sr` (Politis-Romano), `pbo_cscv` (Bailey/LdP), + the canonical `multifactor_alpha` (moved here). |
| `app/research/ruler_v2.py` | Verdict orchestrator: `RulerV2Criteria.from_retrain_config()`, `paper_plausibility_verdict()`, `capital_significance_verdict()`, `RulerV2Result`. Pure (reads frozen constants). |
| `app/research/bayes_sr.py` | CAPITAL-tier Bayesian posterior `P(SR>0 | backtest + live-paper)` — normal-normal update, trial-count-tightened prior. Pure. |
| `app/research/track_b_appraisal.py` | Track-B v2: budget-invariant **appraisal ratio** + block-bootstrap CI on ΔSR; worst-regime floor waived for declared diversifiers. Coexists with legacy `book_gate.py`. Pure. |
| `app/research/live_track_record.py` | Thin I/O adapter: assemble a sleeve's live-paper daily return series from the trackers (`trend_tracker`/`pead_tracker`/`allocator_tracker`) → `pd.Series` for the Bayesian update. |

**Changed (additive, flag-gated):**
| File | Change |
|---|---|
| `app/ml/retrain_config.py` | Extend `GATE_MODE` → add `"ruler_v2"` (alongside `"significance"`/`"mean_sharpe"`). Add the Ruler-v2 constant block (SSOT). |
| `scripts/walkforward/cpcv.py` | Persist `CPCVResult.oos_returns_dated` (concatenate the existing per-fold `evaluated_fold_returns`, today discarded). Add an `elif GATE_MODE=="ruler_v2"` dispatch in `gate_passed`/`gate_detail`. |
| `scripts/walkforward/gates.py` | Mirror the `ruler_v2` dispatch → `INCONCLUSIVE` for WF-only (CPCV required). Demote PF/Calmar to report-only **inside the ruler_v2 branch only**. |
| `app/research/options_xs_ls.py` | Re-export `multifactor_alpha` from `inference.py` (canonical home), same pattern `attribution.py` used for `capm_alpha`. |

**Canonical residual-alpha:** reuse **`multifactor_alpha`** (the validated NW-HAC multi-factor
generalization of `capm_alpha`) as the gate's residual-alpha — a *premia* book needs multi-factor
residualization. Move it into `inference.py`; `capm_alpha` (SPY-only) stays for the existing CPCV
diagnostic, unchanged. **Frozen-gate integrity:** `book_gate.py` is pre-registered — never edit it
in place; Track-B v2 is a *new* module selectable by a `TRACKB_MODE` flag.

## 2. The inference core (`inference.py`) — spec

All functions PURE; explicit too-few-obs contract: below `MIN_INFERENCE_OBS` (default 60) return
the point estimate (if computable) + `gating=False` + a `reason` — never raise, never fabricate a
t-stat (mirrors `capm_alpha`'s `n<30` zero-fill).

- **`hac_sharpe(returns, *, hac_lag=HAC_SR_LAG, annualize=252) -> HACSharpeResult`** (Lo 2002).
  Autocorrelation-corrected SE of the daily Sharpe via the **Bartlett-kernel NW-HAC sandwich on a
  mean-only regression** (reuse the exact sandwich already in `multifactor_alpha` — don't re-derive
  Lo's algebra). Returns `sr_ann, se_sr_ann, t_stat, p_one_sided`. Degenerate vol (σ≤ε) →
  `t=0, p=0.5, gating=False`. **GATING:** feeds the Bayesian posterior at CAPITAL; report-only PAPER.
- **`stationary_bootstrap_sr(returns, *, n_reps=2000, mean_block_len=None, seed=0) -> BootstrapSRResult`**
  (Politis-Romano 1994). One-sided `P(SR>0)` + 95% CI. Block length: Politis-White (2004) automatic,
  fallback `ceil(T^(1/3))`. Seeded (reproducible). **GATING** at CAPITAL (P(SR>0) ≥ 0.95).
- **`pbo_cscv(is_oos_perf, *, n_splits=None) -> PBOResult`** (Bailey/LdP 2017). Input = an
  **N_configs × N_blocks** per-block OOS performance matrix; does CSCV internally (enumerate
  C(S,S/2) symmetric partitions, rank configs IS-vs-OOS, accumulate the IS-best's OOS-rank logit) →
  `pbo ∈ [0,1]`, `prob_oos_loss`. **GATING when M>1** (pbo ≤ PBO_MAX); `None`/non-gating for M=1.
  ⚠️ **Leakage:** the function can't detect a caller that fit configs on full sample — document that
  the matrix must be per-block OOS performance.
- **`multifactor_alpha(r_book, factors, hac_lag=5) -> dict`** (moved from `options_xs_ls.py`).
  **GATING PRIMARY for Track A** at CAPITAL: `t_alpha_hac ≥ RESIDUAL_ALPHA_MIN_T`.

| Statistic | PAPER | CAPITAL |
|---|---|---|
| HAC-SR t/p (Lo) | report | feeds Bayesian posterior |
| Bootstrap P(SR>0) | report | **gating** (≥0.95) |
| PBO | report | **gating** if M>1 (≤PBO_MAX) |
| Residual-α t_hac (multi-factor) | report | **gating PRIMARY** (≥2.0) |
| PF / Calmar | **report (demoted off AND)** | report |
| Point-estimate SR | **gating** (≥ plausibility floor) | gating (materiality floor) |

## 3. Gate wiring

- **`GATE_MODE="ruler_v2"`** — a third `elif` branch in `CPCVResult.gate_passed`/`gate_detail`
  delegating to `ruler_v2.evaluate(result, tier=...)`. Legacy `significance`/`mean_sharpe` branches
  **byte-for-byte untouched** → the ~175-test corpus + the recorded kill ledger keep passing. Ships
  **dark** (flag not flipped) until validated — exactly how `significance` shipped.
- **PAPER = plausibility** (NO significance t): pre-registered + economic-rationale present +
  point-SR ≥ `RULERV2_PAPER_MIN_SR` + survives 2× cost-stress + P5 not catastrophic + implausibility
  ceiling + concurrent-paper-sleeve cap.
- **CAPITAL = significance**: Bayesian posterior P(SR>0 | backtest + live paper) ≥ 0.95 AND
  residual-α t_hac ≥ 2.0 (primary) AND bootstrap P(SR>0) ≥ 0.95 AND (PBO ≤ PBO_MAX if M>1) AND a
  **POWER FLOOR** (`n_obs ≥ RULERV2_MIN_DAILY_OBS` and `n_folds ≥ 10`) — fails closed below the
  floor regardless of the posterior; returns `recommended_size ∝ posterior_mean_SR`.
- **Bayesian posterior** (`bayes_sr.posterior_sr`): prior SR~N(0, `RULERV2_PRIOR_SR_SD`²), the
  registry's **true trial count** tightens the prior toward zero (`SD/√(1+log N_trials)`) — this is
  the multiplicity defense that replaces the saturated DSR; backtest + live-paper combined as
  precision-weighted (HAC-SE) normal observations → closed-form `P(SR>0)`.
- **Track-B v2** (`track_b_appraisal`, behind `TRACKB_MODE="ruler_v2"`): replace budget-
  dependent ΔSharpe≥0.10 with the budget-invariant **appraisal ratio** (residual-α IR ≥
  `RULERV2_TRACKB_MIN_IR` via `multifactor_alpha` with factors=base-book) + block-bootstrap
  P(ΔSR>0) ≥ `RULERV2_TRACKB_MIN_PDSR`. ΔSR is measured on the **simple budget-`b` blend**
  `(1-b)·base + b·cand_vt` (NOT the allocator's `combine()` — the v2 instrument reads the
  SIGNIFICANCE of the direction of the book-Sharpe change, for which the symmetric ~1bp
  rebalance cost is noise; point estimate + bootstrap share the identical definition).
  Reuse `book_gate._vol_target_candidate`/`_series_sharpe` + the vectorized stationary
  bootstrap. **Waive worst-regime floor** for `component_type ∈ {diversifier, risk_premium}`;
  for any other component the floor gates when a worst-regime Sharpe is supplied and **fails
  closed** when it is absent (unless `regime_waiver_approved`) — mirrors the Track-A data-bug
  posture.
- **Gate-amendment recording:** each threshold change is logged as a registry row
  (`family="ruler_v2_gate_amendment"`, exploratory, `params={const: old→new}`). The kill-ledger
  re-score writes R4-compliant re-test rows (`parent_id`=killed hypothesis) so a flip is a
  *sanctioned, logged* recovery — never a silent goalpost move.

## 4. Phased build order

**Status:** Phase 1 ✅ LANDED (2026-06-12, PR #471). Phase 2 ✅ LANDED (2026-06-12, PR #472) —
`bayes_sr.py` + `ruler_v2.py` + `CPCVResult.oos_returns_dated` + the `GATE_MODE="ruler_v2"`
dispatch, all DARK; 25 new tests + 89 legacy gate tests unchanged (R5); two independent Opus
deep-dives (round-2 CRITICAL: CAPITAL was "unreachable on backtest alone" only by threshold luck →
made live-paper a STRUCTURAL gating criterion; + run_cpcv-population integration test) → SHIP.
**Key design property: CAPITAL now requires a live-paper observation by construction** — a backtest
alone can never reach capital (the posterior is `P(SR>0 | backtest AND live paper)`).
Phase 3 ✅ LANDED (2026-06-12, PR #473) — `track_b_appraisal.py` (budget-invariant appraisal IR +
block-bootstrap P(ΔSR>0)) behind `TRACKB_MODE="ruler_v2"`, DARK; legacy `book_delta_gate` untouched;
Opus deep-dive verified budget-invariance empirically + ruled out junk-sleeve gaming (2 MINOR fixed:
non-diversifier missing-regime now fails closed; doc/code reconciled) → SHIP.
Phase 5 ✅ LANDED (2026-06-12, PR #474) — `ruler_v2_rescore.py` (REPORT-ONLY kill-ledger re-score:
tabulates the PAPER-tier flip REVIVED/DEMOTED vs the recorded significance verdict). Opus deep-dive
caught a CRITICAL — the re-score MUTATED the caller's result via `requires_human_review_flag` (broke
the PURE/report-only contract) + a swallowed-exception path that could manufacture a spurious REVIVED;
fixed (deep-copy isolation + ERROR_SIG class + path-Sharpe-tstat columns so the owner sees definitional
vs substantive flips) → SHIP. **Phase B build complete; the live `GATE_MODE`/`TRACKB_MODE` flip + the
OD-1…OD-9 sign-off are the owner's calls.**

| Phase | Build | Depends | Tested by | Risk |
|---|---|---|---|---|
| **1** ✅ | **Pure inference core** (`inference.py`): hac_sharpe, stationary_bootstrap_sr, pbo_cscv, move multifactor_alpha. No wiring. | numpy/scipy only | known-answer fixtures (IID→Lo SE; AR(1)→HAC SE > IID; bootstrap 0.5 on noise→1.0 on drift; PBO 0.5 on noise, low on dominant, high on IS-selected) | **low** |
| **2** ✅ | Persist `CPCVResult.oos_returns_dated` + `ruler_v2.py` + `bayes_sr.py` behind the flag; the dispatch branches. | 1 | flag coexistence (legacy tests unchanged); tier logic on synthetic results; structural live-paper requirement for CAPITAL | medium |
| **3** ✅ | `track_b_appraisal.py` behind `TRACKB_MODE`. | 1 | budget-invariance property; diversifier-with-bad-regime passes | medium |
| **4** | PBO sweep harness (Option A) — only when a config grid is on deck. | 1 | leak signature test | low (deferred) |
| **5** ✅ | **Re-score the kill ledger** under Ruler v2 (`ruler_v2_rescore.py`, REPORT-ONLY). | 1-3 | OC-table; flips human-reviewed | medium |

## 5. Risk register

| # | Risk | Test |
|---|---|---|
| R1 | PBO IS/OOS leakage (caller fits full-sample) | inject leaked matrix → PBO≈0 falsely; document leak signature; harness must fit per-block |
| R2 | HAC lag mis-choice (too short → Type-I) | SE monotone non-decreasing in injected AR(1) ρ; lag-5/10/auto sensitivity on the live trend series |
| R3 | Bootstrap block-length degeneracy | fallback fires; CI brackets point SR; block=1 vs auto |
| R4 | **Bayesian prior mis-calibration** (over- or under-promotes — could re-create the Type-II machine) | run gate-calibration positive controls (tsmom_4y/19y) + TRUE nulls through the posterior; positives clear 0.95, nulls don't |
| R5 | Backward-compat breakage | full existing corpus at default GATE_MODE → zero diffs; `oos_returns_dated` pure-additive (default `[]`) |
| R6 | Live-paper look-ahead via tracker | assembled series ends strictly before `run_at`; use realized (not unrealized) P&L |
| R7 | Re-score double-counts the trial penalty | use `n_trials` as-of the original run, not current 300 |
| R8 | PF/Calmar demotion leaks into legacy | demotion lives only inside the ruler_v2 branch; assert significance `gate_detail` unchanged |

## 6. Open decisions (owner calls — recommended defaults; needed at Phase 2/3, NOT Phase 1)

| ID | Decision | Default |
|---|---|---|
| OD-1 | `HAC_SR_LAG` | 10 |
| OD-2 | `RULERV2_PAPER_MIN_SR` (plausibility floor) | 0.30 |
| OD-3 | `RULERV2_MAX_PAPER_SLEEVES` (concurrent cap) | 6 |
| OD-4 | `RULERV2_PRIOR_SR_SD` + trial-shrinkage form | 0.30, shrink by `1/√(1+log N_trials)` |
| OD-5 | `RULERV2_TRACKB_MIN_IR` / `..._MIN_PDSR` | 0.20 / 0.85 |
| OD-6 | `RULERV2_MIN_DAILY_OBS` (CAPITAL power floor) | 504 (~2y) |
| OD-7 | PBO scope (standalone harness vs wire into run_cpcv) | standalone now; wire later |
| OD-8 | `RESIDUAL_ALPHA_MIN_T` (CAPITAL primary) | 2.0 |
| OD-9 | Factor set residualizing a *premia* book (a trend factor risks hedging out the edge) | P4 set; trend-factor TBD |

> **Phase 1 (the pure inference core) needs NO owner decision** — its few defaults (HAC lag 10,
> 2000 bootstrap reps, T^(1/3) block fallback) are documented and tunable. The consequential
> owner calls (prior SD, plausibility floor, sleeve cap, appraisal bars) are surfaced at Phase 2/3.
