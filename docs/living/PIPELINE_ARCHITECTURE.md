# MrTrader — WF/CPCV Pipeline Architecture

**This is the single source of truth for how the walk-forward and CPCV pipeline works.**

> **Maintenance rule:** Any change to files in `scripts/walkforward/`, `app/backtesting/`, or `app/ml/` that alters behavior, data flow, gate logic, or simulator internals **must** update this document in the same PR. Do not merge without updating. The "Last verified" date in each section must be set to the PR merge date.
>
> When asking an LLM (Claude, Opus, etc.) to review or modify pipeline code, **paste the relevant sections of this document in the prompt** to prevent false assumptions. The #1 source of review errors is the LLM assuming which simulator a given path uses.

**Last full verification:** 2026-05-31 (post 13-round audit, PRs #323–327)

---

## Table of Contents
1. [Component Map](#1-component-map)
2. [Three Simulators — Know Which Is Which](#2-three-simulators)
3. [Entry Points](#3-entry-points)
4. [Data Flow — Swing WF](#4-data-flow--swing-wf)
5. [Data Flow — Intraday WF](#5-data-flow--intraday-wf)
6. [Data Flow — CPCV](#6-data-flow--cpcv)
7. [Gate Inventory](#7-gate-inventory)
8. [Fold Construction](#8-fold-construction)
9. [OOS Guard](#9-oos-guard)
10. [Regime System](#10-regime-system)
11. [Promotion Flow](#11-promotion-flow)
12. [Known Limitations & Deferred Items](#12-known-limitations--deferred-items)
13. [Feature Flags](#13-feature-flags)
14. [Changelog](#14-changelog)

---

## 1. Component Map

```
scripts/
  walkforward_tier3.py          — CLI entry point for WF + CPCV runs (2200+ lines)
  walkforward/
    engine.py                   — FoldEngine: strategy-agnostic fold construction
    cpcv.py                     — run_cpcv(): CPCV orchestration + CPCVResult
    gates.py                    — FoldResult, WalkForwardReport, all gate/metric logic
    reports.py                  — print_report() for WalkForwardReport
    oos_guard.py                — assert_model_oos(): OOS invariant enforcement
    regime.py                   — load_regime_map(), compute_regime_sharpes()
    strategies/
      swing.py                  — SwingStrategy: data fetch + per-fold run (calls AgentSimulator)
      intraday.py               — IntradayStrategy: data fetch + per-fold run (calls IntradayAgentSimulator)

app/
  backtesting/
    agent_simulator.py          — AgentSimulator: swing WF/CPCV simulator (2144 lines, DAILY MTM equity)
    intraday_agent_simulator.py — IntradayAgentSimulator: intraday WF/CPCV simulator (DAILY equity)
    strategy_simulator.py       — StrategySimulator: TIER-2 ONLY (NOT used in WF/CPCV)
    metrics.py                  — Trade, BacktestResult dataclasses
  ml/
    retrain_config.py           — ALL gate thresholds, feature flags, constants (single source of truth)
    features.py                 — Swing feature computation
    intraday_features.py        — Intraday feature computation
  notifications/
    notifier.py                 — Email notification system (enqueue → notify_watcher.py → Gmail)
```

---

## 2. Simulators

**This is the most common source of confusion. Read this before touching any backtest code.**

| Simulator | File | Used by | Equity curve | Status |
|---|---|---|---|---|
| `AgentSimulator` | `app/backtesting/agent_simulator.py` | **Swing WF + CPCV** (via `swing.py:run_fold`) | **DAILY MTM** — one equity point per calendar trading day; uses `portfolio.equity_mtm(today_closes)` | Production |
| `IntradayAgentSimulator` | `app/backtesting/intraday_agent_simulator.py` | **Intraday WF + CPCV** (via `intraday.py:run_fold`) | **DAILY** — one equity point per trading day | Production |
| `StrategySimulator` | `app/backtesting/strategy_simulator.py` | **Tier-2 standalone tool ONLY** | **ENTRY-DATE only** — one equity point per day that had a new trade entry (NOT daily MTM) | Tier-2 only; **never used in WF/CPCV** |
| `OptionsSimulator` | `app/backtesting/options_simulator.py` | **Options WF + CPCV** (via the OPT-3 adapter — pending) | **DAILY MTM** — marks each option leg to its real EOD close (fwd-filled), settles at intrinsic at expiry; one equity point per calendar trading day | Alpha-v5 OPT-2 (built; not yet wired to WF/CPCV until OPT-3) |

**Why this matters for Sharpe/n_obs:**
- `AgentSimulator`, `IntradayAgentSimulator`, and `OptionsSimulator` produce correct daily-return series → `sqrt(252)` annualization is valid.
- `StrategySimulator` produces entry-date-sampled returns → applying `sqrt(252)` is incorrect. It is only used for quick standalone sanity checks, not promotion decisions.

**SimResult** is defined in `strategy_simulator.py` and imported by all simulators. All populate it; the fields are the same. `OptionsSimulator` additionally attaches `daily_returns_dated` / `dropped_positions` / `blown_up` as result attributes (the OPT-3 adapter reads these; `blown_up` is a hard-fail flag for any defined-risk book that goes ≤ 0).

---

## 3. Entry Points

### 3a. Standard Walk-Forward (WF)
```bash
python scripts/walkforward_tier3.py --model swing  [--folds N] [--years Y] [--cpcv ...]
python scripts/walkforward_tier3.py --model intraday [--folds N] [--days D] [--cpcv ...]
```
- Calls `run_swing_walkforward()` or `run_intraday_walkforward()` in `walkforward_tier3.py`
- These functions are the **legacy monolith path** — they do NOT use `FoldEngine`; they build folds inline
- Returns `WalkForwardReport`; checks `report.gate_passed()`; optionally runs CPCV

### 3b. CPCV (Combinatorial Purged Cross-Validation)
```bash
python scripts/walkforward_tier3.py --model swing --cpcv --cpcv-k 6 --cpcv-paths 2
```
- Calls `run_cpcv(strategy, ...)` in `scripts/walkforward/cpcv.py`
- Uses `FoldEngine._build_calendar_folds()` (swing) or `FoldEngine._build_trading_day_folds()` (intraday)
- Returns `CPCVResult`
- **IMPORTANT:** CPCV scores a single **pre-trained frozen model** across C(k,paths) test windows. It is a generalization/robustness test, NOT per-fold retraining.

### 3c. FoldEngine path (modular, newer)
- Used directly by CPCV
- `FoldEngine.run()` → calls `strategy.fetch_data()` then `strategy.run_fold()` per fold
- `strategy.run_fold()` instantiates the appropriate simulator (AgentSimulator for swing, IntradayAgentSimulator for intraday)

### 3d. Weekly auto-retrain (cron)
```bash
python scripts/retrain_cron.py      # swing
python scripts/retrain_intraday.py  # intraday
```
- Calls `gate_passed()` on the resulting WalkForwardReport
- On PASS: touches `model_v{N}.gate_passed` sentinel file; updates DB record

---

## 4. Data Flow — Swing WF

```
walkforward_tier3.run_swing_walkforward()
  │
  ├─ Loads swing_v{N}.pkl (model.trained_through must be set)
  ├─ Downloads universe: pit_union("russell1000", tr_start, tr_end) via yfinance
  ├─ OOS guard: assert te_start > model.trained_through + SWING_PURGE_DAYS (85d calendar)
  │
  └─ Per fold (serial on Windows, MAX_FOLD_WORKERS=1):
       └─ AgentSimulator.run(symbols_data, start_date=te_start, end_date=te_end)
            │
            ├─ Per trading day in [te_start, te_end]:
            │    ├─ PM: features → cs_normalize → model.predict() → top-N ranked proposals
            │    ├─ Trader: generate_signal() technical filter
            │    ├─ RM: validate_* rule checks against live portfolio state
            │    ├─ Entry: fills at next-day open (simulated MOO)
            │    ├─ Exit: check_exit() ATR stop/target bar-by-bar
            │    └─ equity_by_date[day] = portfolio.equity_mtm(today_closes)   ← DAILY MTM
            │
            └─ _compute_result(accepted_trades, equity_by_date, ...)
                 ├─ equity_curve = sorted(equity_by_date.items())   ← one (date, equity) per trading day
                 ├─ daily_rets = [(eq[i]-eq[i-1])/eq[i-1] for i in 1..N]
                 ├─ sharpe_ratio = mean(daily_rets)/std(daily_rets)*sqrt(252)
                 ├─ profit_factor: from actual trade P&L (PF_NO_LOSS_SENTINEL=5.0 for all-wins)
                 └─ Returns SimResult

       FoldResult(
           sharpe = result.sharpe_ratio,
           n_obs  = len(equity_curve) - 1,   ← trading-day count (correct for DSR)
           profit_factor = result.profit_factor,
           calmar_ratio = compute_calmar(total_return, max_drawdown, years),
           regime_sharpes = compute_regime_sharpes(equity_curve, te_start, te_end, regime_map),
       )

  WalkForwardReport.gate_passed() → see Gate Inventory
```

**Universe PIT contract for swing:**
- `pit_union("russell1000", tr_start, tr_end)` — upper bound is `tr_end`, NOT `te_end`
- Names that joined the index between `tr_end` and `te_end` are excluded (no look-ahead)
- Historical symbols from DB also included to reduce survivorship bias

---

## 5. Data Flow — Intraday WF

```
walkforward_tier3.run_intraday_walkforward()
  │
  ├─ Loads intraday_v{N}.pkl
  ├─ Loads 5-min bars from Polygon cache (load_many) OR yfinance fallback (≤55d, ≤100 symbols)
  │    ⚠️  If Polygon cache is empty: silently falls back → SHORT DATA WINDOW (known gap, MEDIUM-3)
  ├─ Loads per-symbol DAILY bars (52w-position / vol-percentile features) per
  │    INTRADAY_DAILY_FEATURE_PROVIDER (default "aggregate_5min") — NOT the 5-min provider.
  │    Source history of this knob:
  │      • "alpaca"  (orig): caps at ~100 recent daily bars → 52w/vol features silently
  │                  0.5-defaulted across most of the window (see #343).
  │      • "yfinance" (#343): full multi-year history, but (a) its cache compared the
  │                  requested range against datetime.date and raised "can't compare
  │                  datetime.datetime to datetime.date" when the per-fold path passed
  │                  datetime → returned 0/703 symbols (now type-coerced to date), and
  │                  (b) can rate-limit on 150-700 symbols.
  │      • "aggregate_5min" (DEFAULT, this fix): aggregate the in-memory 5-min bars to
  │                  daily OHLCV (open=first/high=max/low=min/close=last/volume=sum). Zero
  │                  network, no rate-limit, coverage == 5-min span (cannot empty the
  │                  per-fold matrix). TRADEOFF: 5-min cache ~2yr, so windows near the
  │                  cache start get <252d backward lookback for 52w/vol (degrades to a
  │                  shorter window, NOT to 0.5 defaults). Loud WARN if coverage <50%.
  │      • "polygon": Polygon daily cache + S3 (full history) if available.
  ├─ OOS guard: assert te_start > model.trained_through + purge_days (TRADING DAYS, not calendar)
  │
  └─ Per fold:
       └─ IntradayAgentSimulator.run(symbols_data, spy_data, start_date, end_date)
            │
            ├─ Per trading day: PM → RM → entries on 5-min bars → exits (HOLD_BARS=24, ATR target/stop)
            └─ equity_by_date[day] = portfolio.equity   ← DAILY

       FoldResult (same structure as swing)
```

**Universe PIT contract for intraday:**
- `members_at("russell1000", te_start)` — uses test-start date (not train-start)
- This correctly excludes names that delisted between tr_start and te_start

---

## 6. Data Flow — CPCV

```
run_cpcv(strategy, purge_days, embargo_days, n_folds=6, n_paths=2)
  │
  ├─ strategy.fetch_data() called ONCE for the full window (data pre-loaded)
  ├─ FoldEngine._build_calendar_folds() or _build_trading_day_folds()
  ├─ OOS guard on all boundary test starts
  │
  └─ For each combo in C(n_folds, n_paths) = C(6,2) = 15 combinations:
       └─ For each test_index ti in combo:
            ├─ prior_train = [j < ti in train_indices]   ← CAUSAL ONLY (no look-ahead)
            ├─ best_train = max(prior_train)
            ├─ real_tr_end = min(all_boundaries[best_train][1], te_start - purge_days - 1)
            ├─ Overlap guard: rolling window must not overlap prior test fold + embargo
            └─ strategy.run_fold(real_tr_start, real_tr_end, te_start, te_end)
                 └─ Same AgentSimulator / IntradayAgentSimulator call as WF

  CPCVResult aggregates per-combo path_sharpes, path_profit_factors, path_calmars, path_n_obs

  CRITICAL DESIGN NOTE: CPCV uses ONE PRE-TRAINED FROZEN MODEL scored across all paths.
  The 15 path Sharpes are correlated (each trading day appears in C(5,1)=5 paths).
  pct_positive and p5_sharpe treat them as independent — they are not.
  N_eff ≈ n_folds (6), not n_combinations (15).
```

**DSR in CPCV:**
- `unique_obs = total_obs / C(n_folds-1, n_paths-1)` corrects for day-multiplicity
- `_dsr_n_obs()` returns `unique_obs` for DSR denominator
- **KNOWN LIMITATION:** DSR p saturates to 1.0 for mean Sharpe > ~2.0 regardless of N_TRIALS_TESTED (see Known Limitations #1)

---

## 7. Gate Inventory

### 7.0 — `GATE_MODE`: significance-first two-tier gate (Phase-4, DEFAULT)

`app/ml/retrain_config.py::GATE_MODE` selects the promotion gate:

- **`"significance"` (DEFAULT, owner-approved):** the new significance-first
  two-tier gate. Mean Sharpe is NO LONGER the primary discriminator — it is an
  economic-materiality floor. Promotion is gated on the path-Sharpe **t-stat**
  (N_eff=n_folds), sign-consistency (%positive), the tail (P5), with PF/Calmar/
  regime kept as backstops. Two tiers:
  - **PAPER** (forward-validate, NO capital): passes iff
    `tstat ≥ PAPER_GATE_MIN_TSTAT(2.0)` **AND** `pct_positive ≥ PAPER_GATE_MIN_PCT_POSITIVE(0.75)`
    **AND** `p5_sharpe ≥ PAPER_GATE_MIN_P5_SHARPE(0.0)` **AND**
    `mean_sharpe ≥ PAPER_GATE_MIN_MEAN_SHARPE(0.35)` **AND** PF/Calmar/regime backstops OK.
  - **CAPITAL** (real money): PAPER criteria with the capital mean floor
    `CAPITAL_GATE_MIN_MEAN_SHARPE(0.45)` **AND** `n_folds ≥ CAPITAL_GATE_MIN_N_FOLDS(10)`
    **AND** (`tstat ≥ CAPITAL_GATE_MIN_TSTAT(2.5)` **OR** a documented live-paper
    confirmation, when `CAPITAL_GATE_REQUIRE_PAPER_CONFIRMATION=True`). *(Alpha-v4 P0
    recalibration 2026-06-07: capital floor 0.50→0.45; legacy SR≥0.80 bar retired —
    robustness is primary, Sharpe level is the materiality backstop.)*
  - **Residual-alpha (CAPM/HAC) — DIAGNOSTIC, NOT GATING (Alpha-v4 P0):**
    `CPCVResult.residual_alpha_t_hac` (+`_ols`, `residual_alpha_ann`, `residual_beta`,
    `residual_sharpe`) — regresses the concatenated OOS book returns on SPY (Newey-West
    HAC) to test whether the edge survives hedging out the market (t<1 = mostly beta,
    PEAD's lesson). Reported in `print()`/JSON and logged; **excluded from gate_passed**
    this PR (graduates to a primary gate criterion later, once validated). Source:
    `scripts/walkforward/attribution.capm_alpha`.
  - **CPCV-required rule + WF-report-only (FIX-1):** a standard (non-CPCV)
    `WalkForwardReport` has only a single point estimate and NO path-Sharpe
    distribution → no t-stat. Under significance mode `WalkForwardReport.gate_passed()`
    **HARD-FAILS** with `"CPCV required for significance gate (WF has no path
    t-stat)"` — it does NOT fabricate a t-stat. BUT "cannot evaluate for promotion"
    is **distinct** from "gate failed → retire." `WalkForwardReport.gate_outcome()`
    returns a tri-state `GateOutcome` (`scripts/walkforward/gates.py`): under
    significance a WF-only run is **`INCONCLUSIVE`** (report-only), NOT `RETIRE`.
    `scripts/retrain_cron.py` reads `gate_outcome()`: on `INCONCLUSIVE` it KEEPS the
    current model status (no auto-RETIRE, no `_restore_previous` rollback, no
    `record_tier3_result(gate_passed=False)`) and logs that CPCV is required for a
    promotion decision. This prevents the prior bug where every scheduled WF retrain
    auto-retired the fresh model. Legacy `mean_sharpe` mode is unchanged: `PROMOTE`
    on pass, `RETIRE` on a real legacy fail.
  - **CAPITAL tier reachability (FIX-1):** `scripts/walkforward_tier3.py` threads a
    `--gate-tier {paper,capital}` flag (default `paper`) plus `--paper-confirmation`
    and `--regime-waiver-approved` into the CPCV gate (`_cpcv_swing_gate_ok` and the
    intraday per-fold short-circuit call `gate_passed(tier=args.gate_tier, ...)`). An
    explicit promotion run with `--gate-tier capital` is the real code path that
    evaluates the CAPITAL tier (default runs stay PAPER). The cron retrain path never
    requests capital — capital is an explicit, deliberate promotion action.
  - **Event-sparsity regime waiver (FIX-2):** PEAD is event-driven and produces
    `< REGIME_MIN_OBS(20)` same-regime trading days per bucket, so
    `worst_regime_sharpe` is legitimately `None` (documented in ML_EXPERIMENT_LOG as
    "not a bug"). `compute_regime_sharpes` now fills an `obs_counts` out-dict (raw
    per-regime counts BEFORE the REGIME_MIN_OBS filter), threaded into
    `FoldResult.regime_obs_counts`; `run_cpcv` aggregates it into
    `CPCVResult.regime_insufficient_obs` — **True** when obs WERE observed but all
    buckets fell below the floor (**event-sparsity**), **False** when no obs at all
    (**data-bug**). The gate then:
    - **PAPER tier (zero capital):** when `worst_regime_sharpe is None` AND
      `regime_insufficient_obs` → regime backstop **WAIVED** (`regime_ok=True`) AND
      `requires_human_review_flag=True` (surfaced in `gate_detail` as a
      `requires_human_review` key, ok=False). This is the ONLY way PEAD reaches paper
      PASS — a narrow, flagged, paper-only waiver, NOT a global fail-open.
    - **CAPITAL tier:** NO auto-waive. `worst_regime_sharpe=None` fails the capital
      regime backstop UNLESS an explicit human sign-off `regime_waiver_approved=True`
      (`--regime-waiver-approved`) is passed. Real capital requires real regime data
      or documented sign-off.
    - **Data-bug `None`** (`regime_insufficient_obs=False`): fails CLOSED on BOTH
      tiers (unless legacy `ALLOW_NO_REGIME_GATE` diagnostic bypass).
  - API: `CPCVResult.gate_passed(tier="paper"|"capital", paper_confirmation=False,
    regime_waiver_approved=False)` and `gate_detail(tier=...)`. The legacy
    relaxed-threshold `paper_gate` kwarg is **ignored** under significance mode (it
    relaxed Sharpe/PF, contradicting a significance-first promotion); use `tier`.

- **`"mean_sharpe"` (LEGACY reproduction):** a faithful no-op reproduction of the
  pre-Phase-4 gate — `avg_sharpe ≥ 0.80` (swing) / `≥ 1.00` (intraday) primary,
  t-stat WARN-only (`require_tstat_gate=False`), WF promotion allowed. Used for
  reversibility and historical re-scoring. Verified no-op: the entire pre-Phase-4
  gate test corpus passes unchanged under this mode (the test suite forces
  `mean_sharpe` by default via `tests/conftest.py`; new significance tests opt in
  with the `significance_gate_mode` fixture).

**Why:** 0.80/1.00 were calibrated against now-struck IN-SAMPLE numbers (intraday
+5.14, QualityShort +3.25). A bare mean-Sharpe threshold cannot separate a
`+0.22 / t=0.17` noise result from a `+0.546 / t=2.26` genuine-signal result. The
re-score artifact (`scripts/rescore_gates.py`, `python -m scripts.rescore_gates`)
shows the new gate promotes **only PEAD R1K → PAPER PASS (with a mandatory
`requires_human_review` flag, via the event-sparsity regime waiver) / CAPITAL
HOLD**; every other strategy on record FAILs all tiers, and the LEGACY(0.80) column
is all-FAIL. FIX-3: the artifact is produced by the **REAL** production gate
(`CPCVResult.gate_passed`) on reconstructed `CPCVResult`s carrying each strategy's
actual fields (incl. `worst_regime_sharpe`; PEAD = `None` + `regime_insufficient_obs
=True`) — it no longer reimplements the threshold math or hardcodes `backstops_ok=
True` (the prior version did both, which falsely showed PEAD an unconditional PASS).
PEAD's paper PASS is **conditional on the flagged waiver**, not unconditional.

| Threshold | Value | Tier |
|---|---|---|
| `PAPER_GATE_MIN_TSTAT` | 2.0 | paper |
| `PAPER_GATE_MIN_PCT_POSITIVE` | 0.75 | paper |
| `PAPER_GATE_MIN_P5_SHARPE` | 0.0 | paper (stricter than legacy MIN_FOLD −0.30) |
| `PAPER_GATE_MIN_MEAN_SHARPE` | 0.35 | paper (materiality floor) |
| `CAPITAL_GATE_MIN_TSTAT` | 2.5 | capital (multiple-testing haircut) |
| `CAPITAL_GATE_MIN_N_FOLDS` | 10 | capital (power floor) |
| `CAPITAL_GATE_MIN_MEAN_SHARPE` | 0.45 | capital (Alpha-v4 P0: 0.50→0.45; materiality backstop, not the discriminator) |
| `residual_alpha_t_hac` (CPCVResult) | — | DIAGNOSTIC ONLY (reported/logged, NOT in gate_passed) |
| `CAPITAL_GATE_REQUIRE_PAPER_CONFIRMATION` | True | capital OR-path |

### 7.0a — Gate calibration (operating characteristics) — NEW 2026-06-10 (#444)

The gate above is now **measured, not assumed.** `scripts/walkforward/gate_calibration.py`
runs known-real (positive) and known-null (negative) control strategies through this
exact gate (`run_cpcv` → `CPCVResult.gate_passed`) and reports its **false-negative
(Type-II)** and **false-positive** rates. It changes NO threshold — it emits an
operating-characteristic table plus a PURE, pre-registered recalibration
*recommendation* (structurally unable to mutate config). Two aggregates are reported:
**full-gate** pass-rate AND **significance-core** pass-rate (t-stat / %pos / P5 / mean
only), because the PF≥1.10 / Calmar backstops are **per-trade-calibrated** and mis-score
daily-return *series* controls (a genuine SR≈0.5 daily series has PF≈1.08 and fails the
PF backstop even with a perfect significance core); series rows failing only on PF/Calmar
are tagged as mapping artifacts, not Type-II evidence. **Decisive control: `tsmom_4y`** —
our +0.71/19y sleeve scored on a 4y window: if it fails its own production gate, the
Type-II thesis is confirmed and a threshold recalibration follows (separate PR). The full
control run + the resulting OC table are tracked in `ML_EXPERIMENT_LOG.md`.

### §7.0-B — Track A / Track B two-track acceptance (Alpha-v6 Phase 0, 2026-06-10, #448)

**Why.** The P0 gate-calibration run (§7.0a; `logs/gate_calibration_20260610.json`; DECISIONS
2026-06-10) showed the standalone significance gate is the wrong ruler for crisis-diversifiers:
`tsmom_4y` (t=6.72) and `tsmom_19y` (t=4.46) pass the significance CORE but fail PAPER ONLY on
the `worst_regime_sharpe` backstop — by design, a crisis-diversifier whipsaws in one regime.
Lowering the t-bar was empirically refuted (3/5 TRUE zero-SR nulls cleared t≥2.0; the
pre-registered rule returned `NO_ADMISSIBLE_TSTAR`). Acceptance is therefore split by component
type, never by threshold-shopping.

**Routing.** Every registered hypothesis declares `component_type ∈ {alpha, risk_premium,
diversifier, filter, tail_hedge}`:
- `alpha` → **Track A**: the existing standalone significance gate (`CPCVResult.gate_passed`,
  `GATE_MODE='significance'`, PAPER/CAPITAL tiers) — unchanged. (Event/XS strategies move to
  event-panel clustered inference as primary when P3 lands; CPCV becomes the robustness check.)
- `risk_premium | diversifier | tail_hedge` → **Track B**: the book-delta gate
  (`scripts/walkforward/book_gate.py::book_delta_gate`), judged on contribution to the COMBINED
  book at a small fixed risk budget, not on standalone significance.

**Track B criteria** (pre-registered 2026-06-10, frozen in `app/ml/retrain_config.py`; the gate is
a PURE function of these constants and cannot edit them). PASS iff ALL hold:

| Criterion | Constant | Bar |
|---|---|---|
| Book Sharpe Δ (with − without) | `TRACKB_MIN_SHARPE_DELTA` | ≥ +0.10 |
| Book Calmar Δ | `TRACKB_MIN_CALMAR_DELTA` | ≥ 0.0 |
| Book maxDD Δ | `TRACKB_MAX_DD_DELTA` | with-book maxDD no deeper (maxDD ≤ 0 convention) |
| Candidate corr to book (vol-targeted) | `TRACKB_MAX_CORR` | < 0.30 (one-sided; negative passes) |
| Candidate standalone vol-targeted Sharpe | `TRACKB_MIN_STANDALONE_SR` | > 0.20 |
| Standalone Sharpe plausibility | `SHARPE_IMPLAUSIBILITY_CEILING` | ≤ 3.0 (look-ahead/degenerate guard) |
| Risk budget | `TRACKB_MAX_RISK_BUDGET` | candidate blended at ≤ 10% of book risk |
| **Tail-overlap** | `TRACKB_MAX_TAIL_OVERLAP` / `TRACKB_JOINT_TAIL_PCTL` | `\|base-worst ∩ cand-worst\| / n_tail ≤ 0.30`, `n_tail=max(3, floor(1%·n_days))` — the candidate's worst days must not COINCIDE with the book's worst days |

The tail-overlap test is the blueprint's REGISTERED criterion; an earlier build used a
mean-of-tail-returns test (maskable by one lucky day + ~43% false-reject on independents) — that
unregistered divergence was replaced (overlap false-reject 0/100 in re-review).

**Mechanics.** Inner-join alignment; candidate PIT-vol-targeted to the base book's annual vol
(allocator's 60d trailing estimator, `shift(1)`, 2% floor; leverage bounded by target/floor, NO
fixed cap — preserves target-invariance of corr/SR/overlap); constant-mix blend at the risk
budget; both books built through `sleeve_allocator.combine()` (weekly rebalance, 1bp symmetric
turnover) so deltas are turnover-net and metric definitions are identical to the book harness. A
zero-variance or implausibly-high-SR candidate hard-fails.

**Promotion scope.** Track B gates PAPER-level book inclusion ONLY. It NEVER auto-promotes to
CAPITAL (owner sign-off + a codified tail-loss budget required). **Open calibration (registered):**
at a 10% budget, ΔSharpe ≥ 0.10 implicitly demands standalone SR ≈ 0.94 (corr 0) / ≈ 0.70 (corr
−0.3), so TSMOM (SR≈0.71, corr≈+0.25 to PEAD) may be structurally rejected — resolve via a
registered amendment after the first real TSMOM-vs-book run, NOT ad hoc.

The tables in §7a/§7b below describe the **legacy `mean_sharpe`** gate (active only
when `GATE_MODE="mean_sharpe"`).

### 7a. WalkForwardReport.gate_passed() — all must pass (LEGACY `mean_sharpe` mode)

| Gate | Metric | Threshold | Status | Notes |
|---|---|---|---|---|
| Avg Sharpe | `avg_sharpe` (n_obs-weighted) | ≥ 0.80 | ✅ **ACTIVE** | |
| Min fold Sharpe | `min_sharpe` | ≥ -0.30 | ✅ **ACTIVE** | |
| DSR | `deflated_sharpe_ratio(avg_sharpe, N_TRIALS_TESTED=250, total_obs)` | p > 0.95 | ⚠️ **NON-BINDING** above Sharpe ~2 | p = 1.0 for any Sharpe > 2; provides zero discrimination (see Known Limitation #1) |
| Profit factor | `avg_profit_factor` (capped at 5.0) | ≥ 1.10 | ✅ **ACTIVE** | Waived if < 2 folds have PF > 0 |
| Calmar | `avg_calmar` (geometric CAGR / max DD) | ≥ 0.30 | ⚠️ **WEAK** | No-DD folds get CAL_NO_DD_SENTINEL=+5.0 which trivially passes (see Known Limitation #3) |
| Worst regime Sharpe | `worst_regime_sharpe` | ≥ -0.50 | ✅ **ACTIVE when regime_map populated; HARD FAIL when None** (Phase 2) | `None` now fails the gate unless `ALLOW_NO_REGIME_GATE=True`. coarse3 scheme (BULL/BEAR/NEUTRAL, expanding-quantile VIX) gives enough obs/bucket; REGIME_MIN_OBS=20 floor (see Known Limitation #2 RESOLVED) |

**Paper-trade variant:** `gate_passed(paper_gate=True)` uses relaxed thresholds (Sharpe ≥ 0.50, min fold ≥ -0.40) and waives PF + Calmar gates entirely.

### 7b. CPCVResult.gate_passed() — all must pass (LEGACY `mean_sharpe` mode)

| Gate | Metric | Threshold | Status | Notes |
|---|---|---|---|---|
| Mean Sharpe | `mean_sharpe` | ≥ 0.80 | ✅ **ACTIVE** | |
| P5 Sharpe | `p5_sharpe` | ≥ -0.30 | ✅ **ACTIVE** | |
| Pct positive | `pct_positive` | ≥ 75% | ⚠️ **OVERSTATED** | Treats 15 correlated paths as independent; effective N ≈ 6 folds (see Known Limitation #4) |
| DSR | as above | p > 0.95 | ⚠️ **NON-BINDING** | Same saturation issue |
| Profit factor | `avg_profit_factor` | ≥ 1.10 | ✅ **ACTIVE** | Waived if < 2 paths |
| Calmar | `avg_calmar` | ≥ 0.30 | ⚠️ **WEAK** | Same sentinel issue as WF |
| Worst regime Sharpe | as WF | ≥ -0.50 | ✅ **ACTIVE; HARD FAIL when None** (Phase 2) | `None` fails unless `ALLOW_NO_REGIME_GATE=True` (mirrors WalkForwardReport) |
| Min active paths | `len(path_sharpes) >= 2` | | ✅ **ACTIVE** | Must have ≥ 2 paths before any gate fires |
| Path Sharpe t-stat | `path_sharpe_tstat = mean/(std/√n_folds)` | ≥ `CPCV_MIN_TSTAT=2.0` | ⏸️ **REPORTED, OFF BY DEFAULT** (Phase 3) | N_eff = n_folds (NOT n_paths). Warns when below threshold; blocks only when `require_tstat_gate=True`. Addresses KL-4 |

### 7c. What gate_passed() does NOT check (not in the boolean AND)

These exist in `gate_detail()` only — they appear in the report and inform human review but do NOT block `gate_passed()`:

| Item | Where | Implication |
|---|---|---|
| In-sample override | `in_sample_override=True` → always returns False | Always blocks promotion |
| `requires_human_review()` | Sharpe > SHARPE_IMPLAUSIBILITY_CEILING | **Not yet implemented** — planned Phase 1 |
| Deployment warning | avg capital deployed < 10% | **Not yet implemented** — planned Phase 1 |
| CPCV path t-stat | t = mean/(std/√N_eff) | **Not yet implemented** — planned Phase 3 |

### 7d. gate_passed() call sites (promotion paths)

| Location | What happens on PASS | What happens on FAIL |
|---|---|---|
| `walkforward_tier3.py:2135,2191` | Logs OK; optionally touches `.gate_passed` sentinel file + enqueues notification | Logs FAIL; does NOT promote |
| `retrain_cron.py:134,193` | Sets `gate_ok=True`; promotes model to ACTIVE in DB | Retains previous ACTIVE model |
| `retrain_intraday.py` | Marks version in DB as gate-passed | Version remains non-promoted |
| `reports.py:65` | Prints green "ALL GATES PASSED" line | Prints red "GATE NOT MET" + failed keys |

---

## 8. Fold Construction

### Calendar folds (swing)

`FoldEngine._build_calendar_folds(n_folds, start_all, end_all, total_years, train_years)`

```
segment_days = int(total_years * 365 / (n_folds + 1))

For fold_idx in 0..n_folds-1:
  train_end  = end_all - segment_days * (n_folds - fold_idx)
  test_start = train_end + purge_days + 1   ← calendar days
  test_end   = train_end + segment_days - embargo_days
  train_start = max(start_all, train_end - train_years * 365)  # if train_years set
```

**SWING_PURGE_DAYS = 85** (= FEATURE_LOOKBACK_DAYS(60) + LABEL_HORIZON_DAYS(20) + buffer(5))
Checked at import time by `_assert_purge_horizon_invariant()`.

### Trading-day folds (intraday)

`FoldEngine._build_trading_day_folds(n_folds, all_days_sorted)`

```
segment_size = len(all_days_sorted) // (n_folds + 1)

For fi in 0..n_folds-1:
  tr_end_idx   = segment_size * (fi + 1) - 1
  te_start_idx = tr_end_idx + purge_days + 1       ← trading days
  te_end_idx   = tr_end_idx + segment_size - embargo_days
```

### end_all anchor

Both fold builders use `retrain_as_of()` (last completed Friday, clamped below `SACRED_HOLDOUT_START`) as the end_all anchor. This prevents temporal multiple testing (re-running until a favourable fold boundary appears).

---

## 9. OOS Guard

`oos_guard.assert_model_oos(trained_through, fold_boundaries, purge_days, trading_day_set)`

- **Swing:** calendar-day purge (`trading_day_set=None`) — `te_start > trained_through + purge_days calendar days`
- **Intraday:** trading-day purge (`trading_day_set=all_days_sorted`) — requires `purge_days` actual trading days between `trained_through` and `te_start`; also blocks same-day overlap (`te_start == trained_through`)
- On violation: raises `OOSViolation` unless `allow_in_sample=True` (escape hatch; result marked `in_sample_override=True` and `gate_passed()` always returns False)
- Called at: `FoldEngine.run()`, `run_cpcv()`, `run_swing_walkforward()`, `run_intraday_walkforward()`

---

## 10. Regime System

### What it does
`regime.py:load_regime_map(start, end)` downloads SPY + VIX via yfinance and labels each date with a regime string. The map is pre-computed ONCE per strategy fetch (in `strategy.fetch_data()`, stored as `strategy._global_regime_map`) so VIX quantile thresholds are stable across all folds.

### Current label scheme: `legacy16`
Format: `"<vix_quartile><trend><momentum>"` → up to 16 labels
- `vix_quartile`: 1–4, computed via `pd.qcut(vix_full_window, q=4)` ← **LOOK-AHEAD** (see Known Limitation #2)
- `trend`: U = SPY > 50d MA, D = SPY ≤ 50d MA
- `momentum`: P = SPY 20d return > 0, N = SPY 20d return ≤ 0
- Examples: `"1UP"`, `"4DN"`, `"3DP"`

### Why it's effectively inactive
- 16 buckets over short test folds → most buckets get < 20 obs → dropped by `REGIME_MIN_OBS` check (currently hardcoded at 2, but insufficient)
- `compute_regime_sharpes()` returns `{}` when all buckets are too small
- `worst_regime_sharpe` returns `None` when no FoldResult has regime_sharpes populated
- `None` **silently passes** `gate_passed()` → the regime gate has never blocked a promotion

### Planned fix (Phase 2)
Coarsen to 3 buckets (BULL/BEAR/NEUTRAL), expanding quantile for VIX thresholds, `ALLOW_NO_REGIME_GATE=False` makes None a blocking failure. See `docs/living/MASTER_BACKLOG.md`.

---

## 11. Promotion Flow

```
WF run completes
  │
  ├─ report.gate_passed()?
  │    NO  → log FAIL; retain current ACTIVE model; stop
  │    YES → continue
  │
  ├─ report.requires_human_review()? [NOT YET IMPLEMENTED — planned Phase 1]
  │    YES → block auto-promotion; email human; wait
  │    NO  → continue
  │
  ├─ Touch model_v{N}.gate_passed sentinel file
  ├─ Update DB: model version = ACTIVE, gate_passed=True
  ├─ Retire old active version
  └─ notifier.enqueue("training_complete", {...}) → email kimminjae@gmail.com

CPCV run (separate from WF, no auto-promotion):
  └─ result.gate_passed() → logged; printed; not tied to model activation
     (CPCV is a validation check; the WF promotion flow drives actual model activation)
```

**Promotion sentinel files:** `app/ml/models/model_v{N}.gate_passed`
- Must exist for a version to be loaded as ACTIVE by `walkforward_tier3._load_model()`
- Created only when `gate_passed()` returns True
- The sacred holdout guard (`SACRED_HOLDOUT_START=2026-11-09`) prevents any run from reaching data after this date without `--allow-sacred-holdout`

---

## 12. Known Limitations & Deferred Items

### KL-1 — DSR gate non-binding above Sharpe ~2 (CRITICAL)
**Description:** `deflated_sharpe_ratio(sharpe=5.14, n_trials=250, n_obs=250)` returns p=1.0. The gate is a rubber stamp for any model that clears the Sharpe threshold. `N_TRIALS_TESTED` is irrelevant when Sharpe > ~2. The gate adds no protection in the regime these models operate.

**Impact:** Any model clearing avg_sharpe ≥ 0.80 will also clear DSR p > 0.95 with near-certainty. The DSR printout in reports is misleading.

**Fix status:** Planned Phase 1. Will add `SHARPE_IMPLAUSIBILITY_CEILING=3.0` and `requires_human_review()` flag. DSR retained as secondary but labeled SATURATED when p > 0.999.

### KL-2 — Regime gate silently passes (HIGH) — ✅ RESOLVED (Phase 2)
**Description:** `worst_regime_sharpe is None` → `regime_ok = True` in `gate_passed()`. The regime gate had never blocked any model promotion. Combined with 16-bucket label scheme producing too-sparse per-regime samples, the gate was effectively a no-op.

**Impact:** Models with poor performance in specific market regimes (bear markets, VIX spikes) could pass the gate while being regime-sensitive.

**Fix status:** ✅ RESOLVED Phase 2 (PR #PHASE2). `REGIME_SCHEME="coarse3"` coarsens to BULL/BEAR/NEUTRAL (3 buckets) so per-bucket obs are sufficient; `REGIME_MIN_OBS=20` floor drops too-noisy buckets; `ALLOW_NO_REGIME_GATE=False` (default) makes `worst_regime_sharpe is None` a HARD FAIL in both `WalkForwardReport.gate_passed()` and `CPCVResult.gate_passed()` (with a clear "REGIME DATA INSUFFICIENT" error log). Set `ALLOW_NO_REGIME_GATE=True` to restore legacy silent-pass for diagnostic runs.

### KL-3 — Calmar gate trivially passes via no-DD sentinel (MEDIUM)
**Description:** No-DD profitable folds receive `CAL_NO_DD_SENTINEL=+5.0`. Intraday tight-stop strategies with multiple no-DD folds trivially clear `avg_calmar ≥ 0.30`.

**Impact:** Calmar gate provides no discrimination for strategies with tight ATR stops.

**Fix status:** Planned Phase 1. Will use vol-floor drawdown estimate when max_drawdown == 0 (`USE_CALMAR_VOL_FLOOR=True`).

### KL-4 — CPCV path correlation overstates independence (HIGH)
**Description:** 15 paths from C(6,2) use the same 6 folds; each trading day appears in 5 paths. `pct_positive` and `p5_sharpe` treat them as 15 independent draws. Effective N ≈ 6 folds.

**Impact:** "92.9% positive" overstates statistical confidence. Actual effective N for significance is ~6, not 15.

**Fix status:** ⏳ PARTIALLY ADDRESSED Phase 3 (PR #PHASE3). Added `CPCVResult.path_sharpe_tstat = mean_path_sharpe / (std_path_sharpe / √N_eff)` with **N_eff = n_folds** (not n_combinations — correctly reflects that the C(k,p) paths reuse the same k folds and are correlated). Reported in `print()` and `gate_detail()` with an explicit "N_eff=n_folds, NOT n_paths" note, and WARNED when below `CPCV_MIN_TSTAT=2.0`. The `require_tstat_gate` flag (default False) makes it blocking once enabled. **Still off by default** — turn on after collecting baseline t-stat data across known-good models, so we calibrate the threshold before it can fail a promotion.

### KL-5 — No minimum data span gate (MEDIUM)
**Description:** If Polygon cache is empty, intraday silently falls back to 55-day yfinance window. CPCV on 55 days = 9-day test folds. Nothing blocks promotion on this degenerate case. No data provenance info in report output.

**Impact:** A degenerate run produces results that look identical to a properly-windowed run in the gate report.

**Fix status:** Planned Phase 1. Will raise `DataSpanError` when `len(all_days_sorted) < MIN_DATA_SPAN_TRADING_DAYS=250`.

### KL-6 — Intraday Sharpe inflated by low capital deployment (CRITICAL)
**Description:** 3% sizing × max 5 positions = ~15% of equity ever at risk. Many flat no-trade days (VIX gate, abstention filters) suppress daily return std further. Daily IR = Sharpe/√252 = 0.32 for Sharpe 5.14 — empirically implausible for a retail equity strategy. A "Sharpe 5" at 15% deployment would be roughly "Sharpe 0.75" fully invested.

**Impact:** Headline Sharpe is not comparable to any benchmark or other strategy. Gates cannot distinguish capital-deployment artifact from real edge.

**Fix status:** Planned Phase 1. Will add `avg_capital_deployed_pct` and `deployment_adjusted_sharpe` to SimResult/FoldResult, with `LOW DEPLOYMENT WARNING` in reports.

### KL-7 — VIX quantile thresholds use look-ahead (MEDIUM) — ✅ RESOLVED (Phase 2)
**Description:** `pd.qcut(vix, q=4)` in `regime.py` computed quartile boundaries over the full evaluation window including future data.

**Impact:** Regime labels for early dates were influenced by VIX values that occur after them. Mild look-ahead in regime classification.

**Fix status:** ✅ RESOLVED Phase 2. The coarse3 scheme (`_load_coarse3`) computes VIX BULL/BEAR thresholds with **expanding quantiles** (`np.percentile(vix_arr[:i+1], pctile)`) using only data up to and including day `i` — PIT-correct, no look-ahead. Days with `< REGIME_VIX_WARMUP_DAYS=60` of prior history are labeled NEUTRAL. Regression-tested by `tests/test_regime_coarse3.py::test_coarse3_no_lookahead` (label at date t is identical whether the series ends at t or extends 200 days further). The legacy16 path (`pd.qcut`) is retained only under `REGIME_SCHEME="legacy16"` for baseline comparison.

### KL-8 — StrategySimulator equity curve entry-date only (LOW, tier-2 only) — ✅ ADDRESSED (Phase 3)
**Description:** `strategy_simulator.py:StrategySimulator.run()` records equity only on trade-entry dates, not daily. Sharpe annualization is incorrect for sparse-entry strategies.

**Impact:** Tier-2 standalone tool only — NOT used in WF/CPCV. Does not affect any gate or promotion decision.

**Fix status:** ✅ ADDRESSED Phase 3 (PR #PHASE3). Added a prominent module-level **TIER-2 ONLY** banner stating WF/CPCV swing uses `AgentSimulator` (calendar-daily MTM) and this simulator's entry-date-keyed curve must never drive promotion decisions. Added a `build_daily_equity_curve=False` opt-in flag to `run()`: when True, the entry-date-keyed equity is forward-filled onto every calendar day in `[start_date, end_date]` so flat days dilute volatility correctly (lower, more honest Sharpe for sparse-entry strategies). Default False = exact legacy behaviour. Tested in `tests/test_strategy_sim_daily_curve.py`.

### KL-9 — Polygon cache may lack delisted names (MEDIUM)
**Description:** If the Polygon 5-min cache only holds currently-listed Russell 1000 names, intraday WF results have survivorship bias regardless of the `members_at(te_start)` PIT logic. The PIT logic is only as good as the cache's historical completeness.

**Impact:** Intraday WF universe may silently exclude names that delisted during the test window, slightly inflating performance.

**Fix status:** No code fix — requires data infrastructure verification. Document: run `scripts/audit_survivorship.py` before any intraday promotion decision to confirm cache completeness.

---

### KL-10 — Model without `trained_through` cannot be OOS-evaluated (CRITICAL) — ✅ RESOLVED (save-guard)
**Description:** `trained_through` is the model's training-data cutoff. The OOS guard (§9) certifies a result is out-of-sample only when `test_start > trained_through + purge`. A model whose `trained_through` is `None` has no verifiable cutoff, so any walk-forward/CPCV result on it is silently **in-sample**. Two shipped models (`intraday_v63`, `swing_v224`) predate the feature (PR #311) and so their artifacts never recorded it; `_load_model` had been injecting it from the model DB at load time, but the DB is now empty, so the value is gone. Worse, `LambdaRankModel` (swing) had no `trained_through` attribute on the class at all.

**Frozen-model CPCV cannot be OOS:** running CPCV/WF over a *frozen, already-trained* model whose cutoff is unknown can never be certified OOS — every fold's test window may overlap the (unknown) training span. Such results are diagnostic-only, never promotion-grade.

**Fix (this PR):**
1. **Save-guard (keystone):** `_assert_trained_through()` at the top of every trained-model `save()` (`PortfolioSelectorModel`, `TwoStageModel`, `ThreeStageModel`, `LambdaRankModel`, `DoubleEnsembleModel`) raises `ValueError` if `trained_through is None`, feature-flagged by `REQUIRE_TRAINED_THROUGH` (default True). A model can never again be persisted without a verifiable cutoff.
2. **Attribute on every trained class:** `self.trained_through: Optional[date] = None` added to the four classes that lacked it; carried through pickle save/load.
3. **Load-time clarity:** `_load_model` in `walkforward_tier3.py` logs a clear error (not a raise) when a loaded ML model lacks `trained_through`, surfacing the stale-artifact cause earlier; the OOS guard remains the authoritative enforcement point.
4. **Artifact-sourced, never DB-sourced:** `trained_through` is read from the pickled model object only; the DB path does not override it.

**Impact:** v63/v224 must be retrained to regain a verifiable cutoff before any promotion; their historical gate numbers were unverifiable (in-sample).

---

### KL-10b — Frozen vs. per-fold retraining (the structural OOS fix) — ✅ Phase 1 (swing) + Phase 2 (intraday) shipped

**Two WF/CPCV modes:**

1. **Frozen mode (default).** One pre-trained model is loaded and scored across every test fold. This is a *generalization test*, NOT out-of-sample: every fold's test window may overlap the model's (single) training span, and the OOS guard can only certify this when `test_start > trained_through + purge` for ALL folds — which a frozen model rarely satisfies across an expanding-window layout. Frozen runs set `is_true_walkforward=False`.

2. **Per-fold retrain mode (`--per-fold-retrain`, swing + intraday).** Inside `{Swing,Intraday}Strategy.run_fold`, a *fresh* model is trained on ONLY that fold's `[tr_start, tr_end]` window (from the already-in-memory bars — no re-fetch, no network), `trained_through` is set to `tr_end`, a per-fold OOS assertion runs (`te_start > tr_end + purge`), then the fold is simulated on `[te_start, te_end]` with that fresh model. This is genuinely out-of-sample. Per-fold runs set `is_true_walkforward=True`.

**No-leak guarantee (swing).** `ModelTrainer.build_train_matrix_for_window` slices `symbols_data` to `[train_start, train_end]` and clamps the SPY date spine to end at `train_end`. The existing worker guard `w_end_idx + FORWARD_DAYS >= len(all_dates)` then drops any window whose forward label would look past the clamped spine — so no training row can use a label beyond `train_end`. Proven by `test_build_train_matrix_no_leak`.

**No-leak guarantee (intraday) — simpler.** The intraday label is **SAME-DAY**: in `_symbol_to_rows`, `future_bars = day_bars.iloc[entry_offset:entry_offset+HOLD_BARS]` is a 24-bar (~2h) window WITHIN the entry day's own 5-min bars, and every other lookback (prior-day OHLC; per-symbol `daily_df` sliced `daily_date_arr < day`; SPY daily sliced `< day`) is strictly BACKWARD. The label therefore never crosses `train_end`, so `IntradayModelTrainer.build_train_matrix_for_window` only needs `train_days = {d : train_start <= d <= train_end}` — **no multi-day FORWARD_DAYS purge is required** (unlike swing). Proven by `test_build_train_matrix_non_empty_and_no_leak` (asserts `max(raw[:,0]) <= train_end.toordinal()`).

**Per-symbol daily bars (intraday plumbing).** `IntradayStrategy.fetch_data` loads only 5-min bars (+ SPY daily overlay), but `_symbol_to_rows` needs per-symbol DAILY bars for vol-percentile / 52w features. In per-fold mode the strategy fetches them **once** via `_ensure_daily_data()` (`IntradayModelTrainer._fetch_daily_all`, the same provider helper) on first `run_fold`, caches them on `self._daily_data`, and reuses them across all folds — the hot 5-min path is never re-fetched per fold. These DAILY bars are sourced from `INTRADAY_DAILY_FEATURE_PROVIDER` (default `"yfinance"`, full multi-year history), **independent of the 5-min `self._provider`**. Pre-#343 they came from Alpaca (~100-bar cap), silently degrading the 52w/vol features to defaults across most of the training window.

**Dedup.** CPCV's C(k,p) combinations reuse the same k folds, so `TrainWindowCache` (lifetime = one run) memoizes fitted models by `(tr_start, tr_end)` — each unique training window is retrained exactly once. The cache is retrainer-agnostic (`get(tr_start, tr_end, *fit_inputs)`), shared by both swing and intraday retrainers.

**No per-fold HPO.** `PER_FOLD_SWING_HPO_TRIALS=0` / `PER_FOLD_INTRADAY_HPO_TRIALS=0` — per-fold HPO is both prohibitively expensive and methodologically wrong (in-fold selection). Per-fold models use the frozen production hyperparameters (swing model.py params; intraday `FROZEN_HPO_PARAMS` + 3-seed XGBoost ensemble); a deterministic per-window seed (`seed_base + tr_end.toordinal() % 100000`, applied as an offset to the intraday ensemble seeds) keeps runs reproducible.

**Promotion gate.** `REQUIRE_TRUE_WF_FOR_PROMOTION` (default False during rollout) makes `gate_passed()` return False for any `is_true_walkforward=False` run regardless of metrics, in BOTH `WalkForwardReport` and `CPCVResult`. Flip to True in Phase 3 once per-fold is validated so frozen runs can no longer promote project-wide.

**Intraday feasibility — reduced universe, WF-not-full-CPCV.** Full Russell-1000 intraday per-fold is OOM-infeasible (5-min features rebuilt per training window × C(k,p) combos). The runner therefore FORCES a reduced liquidity universe when `--per-fold-retrain` + intraday: `--intraday-top-n` (default 150) keeps the top-N symbols by 20-day median dollar volume (applied once, inside `_ensure_daily_data`, shrinking both the 5-min `symbols_data` the matrix iterates and the cached daily bars). The mechanism supports both WF and CPCV, but the design recommends **reduced-universe WF with k=4**; a prominent WARNING fires when `cpcv_k > 4`. The legacy frozen intraday WF (`run_intraday_walkforward`, a bespoke loop with no per-fold path) is SKIPPED when `--per-fold-retrain` + `--cpcv` are set — only the genuine per-fold CPCV runs, and its gate result IS the intraday result (mirrors the swing short-circuit).

**Scope.** Phase 1 = SWING (daily bars). Phase 2 = INTRADAY (5-min, reduced universe). Both ship the same seam; frozen-mode behavior is UNCHANGED when `per_fold_retrain=False`.

**Real-data smoke test (guards the per-fold empty-matrix bug class).** `scripts/smoke_test_per_fold.py` runs a MINIMAL real per-fold CPCV (swing + intraday; 30 symbols, k=4, paths=2, `--as-of 2026-05-29`) and asserts the non-vacuous invariants the unit suite could not: `n_combinations > 0` (≥1 surviving path), ≥1 fold actually fit a model, `is_true_walkforward is True`, per-fold `trained_through == tr_end` for ≥1 fold, and (intraday) that the shallow daily-coverage warning did NOT fire (PR #343). It exists because PR #339 (swing regime-map crash) and PR #342 (intraday day-axis desync) BOTH shipped to prod and BOTH passed the unit suite vacuously — the tests fed mocked/synthetic frames with full coverage and asserted `len(X)>0` only on those, so the real-data "every row dropped → 0 paths" failure mode was never exercised. It is NOT in the pytest suite (too slow, needs real data/network) — run `python scripts/smoke_test_per_fold.py --model both` NIGHTLY or before merging any pipeline-touching PR (`--dry-run` validates wiring without fetching). Importing the module triggers no data fetch (lazy imports; fetch only inside `run()`).

### KL-11 — CPCV overlap guard discards ~half of all fold evaluations for rules-based scorers (HIGH) — ✅ RESOLVED (Alpha-v4 P0.1)
**Description:** The BUG-23 overlap guard (`cpcv.py`) skips a fold evaluation whenever a contiguous (rolling/expanding) **training** window would span a prior test fold in the same combo — necessary to stop a *trained* model from learning data tested elsewhere. But a rules-based scorer (`EventEdgeStrategy`/`PEADStrategy`: `model.trained_through=date.min`, nothing is fit) uses the train window **only** for PIT universe construction in `run_fold`, never for training, so the overlap cannot leak. For the expanding-window PEAD config (k=8, paths=2) the guard discarded ~50% of all fold evaluations (≈21 overlap-skips + 7 fold-0 skips of 56), biasing the surviving CPCV path distribution toward later (bull) regimes — the #1 finding of the 2026-06-06 5-LLM quant review.

**Impact:** Every absolute PEAD CPCV Sharpe was computed on a regime-biased subsample; the bias direction differed across strategy families, contaminating even cross-family relative comparisons.

**Fix status:** ✅ RESOLVED Alpha-v4 P0.1 (#TBD). New `EventEdgeStrategy.is_trained=False` class flag; `run_cpcv` resolves rules-based via that flag (else derives from `model.trained_through==date.min`; **default = treat-as-trained → guard ON**) and **bypasses the overlap guard** for rules-based runs → full, unbiased coverage. New `CPCVResult.n_overlap_bypassed` counter + completeness log. **Statistically sound:** `path_sharpe_tstat` divides by √(N_eff=n_folds), invariant to the number of (correlated) paths recovered → no fabricated significance; the `unique_obs`/DSR multiplicity formula (each day in C(k-1,p-1) paths) is exact only under full coverage, so the fix also repairs the prior partial-coverage DSR mis-calibration. The fold-0 no-causal-history skip is **retained** (justified: no lookback runway for the scorer's SMA/SUE/vol windows). Trained strategies and all MagicMock-based tests keep current behavior. Tests: `tests/test_cpcv_is_trained_coverage.py` (10). **NOTE:** this unbiases the *coverage*; it does not change PEAD's economic verdict (still real-but-underpowered) — Phase 1 re-runs PEAD on the full-coverage harness for an honest number.

---

## 13. Feature Flags

All live in `app/ml/retrain_config.py`.

| Flag | Default | Effect | Since |
|---|---|---|---|
| `RETRAIN_WEEKDAY` | -1 (disabled) | Day of week for scheduled retrains; -1 = off. **Now honored by `orchestrator._trigger_retraining`** (Alpha-v4 P0) — the daily trigger previously ignored it and retrained every night | Phase C / P0 |
| `SWING_ENABLED` | False | Enable nightly swing XS-ranker retrain. False = FROZEN (dead model: per-fold CPCV +0.22, t=0.17). Honored by `retrain_cron.run_swing` | Alpha-v4 P0 |
| `pm.allocator_enabled` (agent_config) | "false" | Live regime-aware sleeve allocator master switch. false = sleeves use static budgets (today). true = `sleeve_allocator_live.run_allocator` sets effective weights weekly; readers fall back to static on disabled/stale/warmup/error | Alpha-v4 P3 |
| `pm.allocator_scheme` (agent_config) | "equal" | Allocator weighting: equal \| vol \| regime. Keep equal until `run_book_allocator.py` gate selects vol/regime (on 2 sleeves equal wins). Under regime, PEAD's size_mult is NOT allocator-tilted (PM already tilts PEAD by regime — double-tilt guard) | Alpha-v4 P3 |
| `USE_NIS_FEATURES` | False | Include NIS macro LLM sentiment in swing training | Phase A |
| `USE_FMP_FUNDAMENTALS` | True | Load PIT FMP fundamentals (overrides EDGAR) | Phase 93 |
| `USE_REALIZED_R_LABELS` | False | Intraday: absolute R threshold labels vs cross-sectional | Phase 88 |
| `BENIGN_FILTER_ENABLED` | False | Filter non-bullish rows from training | Phase P1 |
| `INTRADAY_ENABLED` | False | Enable intraday scanning AND (Alpha-v4 P0) the nightly intraday retrain. False = FROZEN (dead: per-fold CPCV -2.80, t=-6.85). Honored by `retrain_cron.run_intraday` + `retrain_intraday.main` | Phase C / P0 |
| `REGIME_SPLIT_VIX_THRESHOLD` | 0.0 | Train separate calm/shock models; 0 = disabled | Phase B |
| `USE_CALMAR_VOL_FLOOR` | *planned* | No-DD folds: vol-floor DD instead of sentinel | Phase 1 planned |
| `REGIME_SCHEME` | `"coarse3"` | `"coarse3"` (BULL/BEAR/NEUTRAL, expanding-quantile VIX) vs `"legacy16"` | Phase 2 ✅ |
| `REGIME_MIN_OBS` | 20 | Min daily-return obs per regime bucket before its Sharpe counts | Phase 2 ✅ |
| `REGIME_VIX_WARMUP_DAYS` | 60 | Min VIX history before a day gets a non-NEUTRAL coarse3 label | Phase 2 ✅ |
| `ALLOW_NO_REGIME_GATE` | False | False = None worst_regime_sharpe HARD-FAILS gate (no silent pass) | Phase 2 ✅ |
| `CPCV_MIN_TSTAT` | 2.0 | Min path-Sharpe t-stat (N_eff=n_folds); gate via `require_tstat_gate` | Phase 3 ✅ |
| `CPCVResult.require_tstat_gate` | False | True = path_sharpe_tstat < CPCV_MIN_TSTAT blocks gate | Phase 3 ✅ |
| `ENFORCE_MIN_DATA_SPAN` | *planned* | Raise DataSpanError if data < 250 days | Phase 1 planned |
| `REQUIRE_TRAINED_THROUGH` | True | True = `save()` refuses to persist a model whose `trained_through` is None (cannot be OOS-gate-evaluated). False only for diagnostic saves. See KL-10. | save-guard |
| `PER_FOLD_RETRAIN` | False | Default per-fold-retrain mode when `--per-fold-retrain` not passed. True = WF/CPCV train a fresh model per fold (true OOS). See KL-10b. | Phase 1 |
| `REQUIRE_TRUE_WF_FOR_PROMOTION` | False | True = `gate_passed()` returns False for any non-per-fold (`is_true_walkforward=False`) run, even if metrics pass. Flip True in Phase 3. See KL-10b. | Phase 1 |
| `PER_FOLD_SWING_HPO_TRIALS` | 0 | HPO trials for per-fold swing retraining. 0 = frozen production hyperparameters (per-fold HPO is in-fold selection — keep 0). See KL-10b. | Phase 1 |
| `PER_FOLD_INTRADAY_HPO_TRIALS` | 0 | HPO trials for per-fold intraday retraining. 0 = frozen `FROZEN_HPO_PARAMS` (keep 0, same rationale as swing). See KL-10b. | Phase 2 |
| `--intraday-top-n` (CLI) | 150 | Intraday per-fold universe cap (top-N by 20d median dollar volume). Full R1000 per-fold intraday is OOM-infeasible. See KL-10b. | Phase 2 |

---

## 14. Changelog

All entries reference the PR that made the change.

| Date | PR | Change | Files |
|---|---|---|---|
| 2026-06-10 | #448 | **Alpha-v6 P0 — Track B (book-delta) acceptance gate (two-track acceptance).** New `scripts/walkforward/book_gate.py::book_delta_gate` (PURE) — judges a candidate sleeve (risk_premium / diversifier / tail_hedge) on its contribution to the COMBINED book at a ≤10% risk budget, the lever the calibration result (§7.0a) pointed at (TSMOM passes the significance CORE but fails PAPER only on the worst_regime backstop; lowering the t-bar admits noise). PASS iff all 8 pre-registered `TRACKB_*` criteria hold (Sharpe Δ≥0.10, Calmar Δ≥0, maxDD not deeper, corr<0.30 one-sided, standalone vol-targeted SR∈(0.20, `SHARPE_IMPLAUSIBILITY_CEILING`], risk budget≤10%, and the **tail-overlap** test `\|base-worst ∩ cand-worst\|/n_tail ≤ TRACKB_MAX_TAIL_OVERLAP`). Vol-targets the candidate PIT (60d trailing, shift(1), 2% floor, no fixed leverage cap → target-invariant); builds with/without books via `sleeve_allocator.combine()` (metrics identical to the book harness). Track A (significance gate) UNCHANGED; Track B is ADDITIVE (no live/gate code touched) and gates PAPER-level book inclusion ONLY (never auto-CAPITAL). Built + 2× Fable-5 adversarial review (3 MAJOR found+fixed: 5× leverage cap broke target-invariance → removed; the mean tail-test was maskable + ~43% false-reject → replaced with the REGISTERED overlap test; added an implausibility ceiling). 19 tests. New `TRACKB_*` constants in retrain_config. See §7.0-B. | `scripts/walkforward/book_gate.py`, `app/ml/retrain_config.py`, `tests/test_book_gate.py` |
| 2026-06-10 | #444 | **Alpha-v6 P0 — gate-calibration harness (measures the gate's operating characteristics; changes NO threshold).** New `scripts/walkforward/gate_calibration.py` scores known-real (positive) and known-null (negative) control strategies through the PRODUCTION gate (`run_cpcv` → `CPCVResult.gate_passed`) to MEASURE its false-negative (Type-II) and false-positive rates — the empirical test of the Alpha-v6 thesis that a t≥2.0 gate on ≈8 folds of ≤4y data over-rejects true Sharpe-0.5–0.7 edges. Positive controls: `tsmom_4y` (DECISIVE — +0.71/19y sleeve on a 4y window), `tsmom_19y`, `xmom_12_1`, `pead_baseline`, `spy_buyhold` via a PIT `SeriesReturnStrategy` adapter (per-fold test-window slice — proven leak-free to 1e-9) + the reused `run_pead_cpcv` path. Negative controls: `random_balanced_seed_1..5` (true zero-SR long/short nulls), `random_seed_1..5` (beta-loaded diagnostic), `leaky_tplus1` (look-ahead → trips `SHARPE_IMPLAUSIBILITY_CEILING`). Emits an OC table + a PURE, PRE-REGISTERED recalibration *recommendation* (structurally cannot mutate config). **Dual aggregates** (full-gate vs significance-core: t-stat/%pos/P5/mean only) separate genuine Type-II behavior from the per-trade-calibrated PF/Calmar backstop mis-scoring daily-return series controls; `run_failed` rows excluded from aggregates + rule; smoke/full artifact separation + merge-upsert; residual-alpha-t per row. **Diagnostic only — recalibration, if triggered, is a follow-up PR.** Built + 2× Fable-5 adversarial review (1 BLOCKER + 3 MAJOR found+fixed). 49 tests. See §7.0a. | `scripts/walkforward/gate_calibration.py`, `tests/test_gate_calibration.py` |
| 2026-06-10 | #439 | **Regime-model retrain: weekly cadence + fixed gate + one shared evaluator.** Adds `REGIME_RETRAIN_INTERVAL_DAYS=7` and the regime promotion gate as config (`REGIME_GATE_MACRO_F1_MIN=0.60`, `REGIME_GATE_LOG_LOSS_MAX=0.45`). Fixes a latent bug: `scripts/train_regime_model.py` read `payload["wf_auc_min"]`/`["brier_score"]` from the PICKLE, but those keys were only written to the DB row → `KeyError`; and `0.22` was a 2-class Brier cutoff wrongly applied to the 3-class cross-entropy log-loss (random baseline = log(3) ≈ 1.099 → correct threshold 0.45). `regime_training.py` now writes the gate inputs into the pickle; a single `regime_gate()` (used by BOTH the CLI and the PM) is the sole source of truth. New `PortfolioManager._retrain_regime` retrains weekly at 17:30 ET on a FILE-AGE cadence independent of `RETRAIN_WEEKDAY` (so the regime model stays current while swing/intraday retraining is frozen); gate-failed models are deleted so the filename-based loader keeps the prior passing version. NOTE: this is the regime-classifier's own training gate — separate from the WF/CPCV strategy gates in §7. 9 tests. | `app/ml/retrain_config.py`, `app/ml/regime_training.py`, `scripts/train_regime_model.py`, `app/agents/portfolio_manager.py`, `tests/test_regime_retrain.py` |
| 2026-06-09 | #415 | **Alpha-v5 OPT-2 — contract-level options simulator + spread cost model.** New `OptionsSimulator` (DAILY-MTM, marks each option leg to its real EOD close fwd-filled, settles at intrinsic at expiry using the underlying close; emits the same `SimResult`) + `OptionsSpreadCostModel` (modeled half-spread % of premium × a 1×/2×/3× stress mult + per-contract fee; no exit cost held-to-expiry). Implements the OPT-0 `OptionContractSim`/`OptionsSpreadCostModel` contracts. Marks to REAL closes (not theoretical engine prices) so IV-crush is carried by the data; defined-risk payoff caps are automatic via per-leg intrinsic settlement. **Not yet wired to WF/CPCV** — the OPT-3 adapter (next) will feed `daily_returns_dated` into `FoldResult` so `run_cpcv` + all gates apply unchanged. Opus 4.8 adversarial review confirmed MTM + look-ahead discipline correct; drove fixes for silent failure modes: dropped-position logging+count (`dropped_positions`), blow-up flag for equity ≤ 0 (`blown_up`, hard fail), profit-factor cap (inf→99), unparseable-contract rejection, day-1 entry-cost capture in the return series. 19 tests (hand-computed long/short/vertical-cap P&L, calendar spread, multi-day fwd-fill, intermediate short MTM sign, qty scaling, cost-sweep monotonicity, drop/blow-up/PF guards). | `app/backtesting/options_simulator.py`, `tests/test_options_simulator.py` |
| 2026-06-07 | #TBD | **Alpha-v4 P0 — gate recalibration + residual-alpha (CAPM/HAC) diagnostic.** (1) **Residual-alpha-t (diagnostic-first):** lifted the CAPM + Newey-West HAC alpha estimator from `pead_phase1_attribution._capm` into shared `scripts/walkforward/attribution.capm_alpha` (the script now re-exports it). `run_cpcv` concatenates each unique evaluated fold's dated OOS returns (new `FoldResult.daily_returns_dated`, populated by swing+intraday `run_fold`) and regresses them on SPY (`strategy.spy_prices`) → new `CPCVResult.residual_alpha_t_hac/_ols/_alpha_ann/_beta/_sharpe/_n`. Surfaced in `CPCVResult.print()`, the JSON dump, and a `t_HAC<1` WARN log. **DIAGNOSTIC ONLY — deliberately EXCLUDED from `gate_passed`/`gate_detail` this PR** (does the edge survive hedging out SPY? t<1 = mostly beta, PEAD's lesson); graduates to a primary gate criterion in a later PR once validated. Reproduces the beta-driven vs genuine-alpha verdicts in tests. (2) **Threshold recalibration:** `CAPITAL_GATE_MIN_MEAN_SHARPE` 0.50→0.45; legacy SR≥0.80 promotion bar RETIRED (significance mode is default; robustness — t-stat + fold-consistency + worst-regime floor — is primary, Sharpe level is the materiality backstop). PURE-ADDITIVE / backward-compatible: gate pass/fail verdicts unchanged (proven by test). Tests: `tests/test_residual_alpha_gate.py` (6). | `scripts/walkforward/attribution.py`, `scripts/walkforward/cpcv.py`, `scripts/walkforward/gates.py`, `scripts/walkforward/strategies/swing.py`, `scripts/walkforward/strategies/intraday.py`, `scripts/walkforward_tier3.py`, `app/ml/retrain_config.py`, `scripts/pead_phase1_attribution.py`, `tests/test_residual_alpha_gate.py` |
| 2026-06-07 | #TBD | **Alpha-v4 P0 — purged sequential-WF baseline for trained models.** CPCV-with-rolling-retrain is inherently low-coverage for TRAINED models (the overlap guard legitimately skips folds whose train window spans a prior test fold). The honest complement is a purged SEQUENTIAL walk-forward (expanding/rolling `[start,t]` train → `[t+purge,t+1]` test, every OOS block, zero holes), reported ALONGSIDE CPCV (2026-06-06 5-LLM review). New `scripts/walkforward/sequential_baseline.py`: `run_sequential_baseline` (thin wrapper over the already-tested `FoldEngine.run` — that IS the purged sequential machine; honours `per_fold_retrain`; reuses the strategy's `_global_regime_map`) + `print_baseline_vs_cpcv` (side-by-side avg/min/worst-regime/folds; flags an optimistic CPCV-≫-baseline gap as a low-coverage tell). Wired into `_run_cpcv_swing` / `_run_cpcv_intraday` behind a new `--sequential-baseline` CLI flag (default OFF — doubles runtime since per-fold retrain runs twice). It is a SANITY BASELINE, not a promotion path: under `GATE_MODE='significance'` a `WalkForwardReport` has no path t-stat → its gate is INCONCLUSIVE by design. Tests: `tests/test_sequential_baseline.py` (5). | `scripts/walkforward/sequential_baseline.py`, `scripts/walkforward_tier3.py`, `tests/test_sequential_baseline.py` |
| 2026-06-07 | #TBD | **Alpha-v4 P0 — fold-coverage report (gate-before-perf).** Every CPCV run now emits a year × regime coverage map of the EVALUATED test windows, surfaced BEFORE performance, so a fold set skewed toward later (bull) regimes can't masquerade as clean (the #1 finding of the 2026-06-06 5-LLM review). New `scripts/walkforward/coverage.py` (`build_fold_coverage` — pure, buckets the deduped evaluated test windows by `test_start.year` + majority regime via the strategy's pre-built `_global_regime_map`; reuses `regime.load_regime_map`, NO new VIX fetch). `run_cpcv` records each evaluated test window (deduped by fold index) and attaches `CPCVResult.coverage` / `coverage_ok` / `coverage_warnings`; `CPCVResult.print()` renders the coverage table first. **SOFT gate** — `coverage_ok=False` (too few years / regimes / no BEAR-or-high-VIX fold) flags for human review like `requires_human_review`; it does NOT block the significance gate. `_dump_cpcv_result_json` extended with `coverage`, `n_overlap_bypassed`, `path_fold_members`. Under `REGIME_SCHEME=legacy16` the labels decompose to `<vix_quartile><trend><mom>`, so the same map yields year×VIX×trend coverage. Tests: `tests/test_fold_coverage.py` (6). | `scripts/walkforward/coverage.py`, `scripts/walkforward/cpcv.py`, `scripts/walkforward_tier3.py`, `tests/test_fold_coverage.py` |
| 2026-06-07 | #TBD | **Alpha-v4 P0 — freeze dead XS-ML retrain.** The large-cap daily swing XS-ranker (per-fold CPCV +0.22, t=0.17 — noise) and the intraday 5-min ML (-2.80, t=-6.85 — cost-drag) are DEAD on the honest harness but were still retraining nightly: `orchestrator._trigger_retraining` fired `retrain_cron.py` daily and IGNORED `RETRAIN_WEEKDAY=-1` / `INTRADAY_ENABLED=False` (neither flag was enforced on that path — confirmed swing_v229 / intraday_meta_v65 at age 1d). **FIX:** new `SWING_ENABLED=False` flag + `INTRADAY_ENABLED` now also gates retrain; `retrain_cron.run_swing` / `run_intraday` early-return `True` (champion retained, no spurious exit-code-2) when frozen; `retrain_intraday.main` honors `INTRADAY_ENABLED` (with `--force` for manual diagnostics); root-cause fix — `orchestrator._trigger_retraining` now honors `RETRAIN_WEEKDAY` (skip if -1 or weekday mismatch). `regime_model` (separate `regime_training.py` path), PEAD + TSMOM (rules-based) untouched. Reversible: flip the bool. Tests: `tests/test_p0_freeze_xsml.py` (4) + `test_retrain_config.py` freeze assertion. | `app/ml/retrain_config.py`, `scripts/retrain_cron.py`, `scripts/retrain_intraday.py`, `app/orchestrator.py`, `tests/test_p0_freeze_xsml.py`, `tests/test_retrain_config.py` |
| 2026-06-06 | #TBD | **Alpha-v4 P0.1 — full CPCV coverage for rules-based scorers (`is_trained` overlap-guard bypass).** The BUG-23 overlap guard (`cpcv.py`) skips a fold when a contiguous train window spans a prior test fold — necessary for a TRAINED model (in-sample leakage) but SPURIOUS for a rules-based scorer (`EventEdgeStrategy`/`PEADStrategy`: `model.trained_through=date.min`, nothing fit; `run_fold` uses the train window ONLY for PIT universe construction, never training). For the expanding-window PEAD config (k=8,p=2) the guard discarded ~half of all fold evaluations and biased the surviving path distribution toward later (bull) regimes — the #1 finding of the 2026-06-06 5-LLM review (see KL-11). **FIX:** new `EventEdgeStrategy.is_trained=False` flag; `run_cpcv` bypasses the overlap guard for rules-based runs (resolved via the flag, else derived from `trained_through==date.min`; default = treat-as-trained → guard ON) → full, unbiased coverage. New `CPCVResult.n_overlap_bypassed` counter + completeness log. **Statistically sound:** `path_sharpe_tstat` divides by √(N_eff=n_folds) — invariant to recovered paths → no fabricated significance; the `unique_obs`/DSR multiplicity formula becomes EXACTLY correct only under full coverage (repairs prior partial-coverage DSR mis-calibration). **Backward-compatible:** trained strategies + all MagicMock tests (auto-attr `is_trained` truthy/not-False → treated as trained) unchanged; fold-0 no-causal-history skip retained (no lookback runway). Tests: `tests/test_cpcv_is_trained_coverage.py` (10) — bypass via flag/via date.min, trained still skips, explicit-flag-overrides-cutoff, coverage conservation (trained.skips == rules.skips + rules.bypassed), EventEdge/PEAD declare rules-based. Verified non-regressive: 101 CPCV/PEAD/audit/oos-guard tests pass unchanged. | `scripts/walkforward/cpcv.py`, `scripts/walkforward/event_edge.py`, `tests/test_cpcv_is_trained_coverage.py` |
| 2026-06-03 | #TBD | **Alpha-v3 A0 — reusable EventEdgeStrategy harness.** Generalized the proven PEAD CPCV adapter so any discrete-event→drift edge (analyst-revision, short-squeeze, …) runs through the SAME trusted path (per-fold `AgentSimulator` + `run_cpcv` + gate + significance/LOCO side-channels). New `scripts/walkforward/event_edge.py::EventEdgeStrategy` holds the generic `fetch_data`/`run_fold` and the run_cpcv interface (`model_type`, rules-based `trained_through=date.min`/`allow_in_sample=False`, PIT universe via `pit_index`/`pit_trade_type`, `_fold_sim_kwargs` override hook). A scorer is any `callable(day, symbols_data, vix_history=None)->[(sym,conf,dir)]`. `PEADStrategy` (run_pead_cpcv.py) refactored to a thin subclass overriding only `_fold_sim_kwargs` — **byte-identical to the committed +0.546 config** (default kwargs `{no_prefilters:True, max_hold_bars_override:None, pead_conviction_size:False}`; env levers `PEAD_MAX_HOLD_BARS`/`PEAD_CONVICTION_SIZE` + §1.1 slippage preserved). Verified non-regressive: 70 PEAD tests (cost-sensitivity/instrumentation/significance/crisis-robustness/smallmid) pass unchanged. Tests: `tests/test_event_edge_harness.py` (8 — run_cpcv interface, generic sim kwargs, configurable PIT index, PEAD byte-identity lock-in). | `scripts/walkforward/event_edge.py`, `scripts/run_pead_cpcv.py`, `tests/test_event_edge_harness.py` |
| 2026-06-03 | #TBD | **Alpha v2 §3.1 RANKER v2 — Phase-2 iteration 2: THE empty-short root cause (one-sided long proposal pool).** Phase-2's resize + breadth admission were correct but POWERLESS: the fast verification re-run still showed net dollar +0.35 / gross 0.35 (short leg ~empty, net-long). Instrumented short-funnel diagnosis pinpointed it: `_pm_score`/`_pm_score_cached` returned only the **long-only proposal pool** (top ≤`proposal_pool_size`, floored at `min_confidence`) → only ~35-50 long-attractive names ranked, but a 60-long + 60-short book needs ≥120 distinct names, so `apply_sector_cap` (longs) absorbed the ENTIRE ranked set → every short candidate was already a long → `apply_net_sector_cap` (and the Phase-2 breadth pass) correctly refused to short a held long → `capped=0`, empty short leg, net-long book (identical net-dollar across Phase-1 and Phase-2 because the short block had nothing to size). **FIX:** when `enable_shorts AND rebalance_mode`, both scorers return the FULL cross-sectional ranking (bypass the `proposal_pool_size` cap + `min_confidence` floor) so the short leg gets a genuine universe bottom distinct from the longs; longs are still re-bounded to `rebalance_target_n` (top by score, unchanged). Gated on `rebalance_mode` so the long-only SIGNAL path (no own per-name floor) is unaffected. **VERIFIED (fast k=2/yr2 diag):** ranked_syms 50→664, capped 0→60, realized net beta −0.035 (p95 0.066, clean), **net dollar +0.350→−0.041, gross 0.350→0.764** — a genuinely dollar-neutral, target-gross, ~60-short book at last. NOTE: on the now-correct neutral book the fast-config Sharpe is **−3.61 / 0% positive** — i.e. stripping the long beta reveals a strongly NEGATIVE cross-sectional signal (the +1.38 of the broken net-long book was market beta). This is an early (NOT decisive — tiny N_eff, recent window) signal the dollar-neutral ranker thesis is likely dead; the decisive call is the full-config Phase-3 run on this corrected book. Long-only / signal / live paths byte-identical (live PM does not call `AgentSimulator._pm_score`). Tests: `tests/test_ranker_phase2_neutrality.py` (+1) — full ranking when enable_shorts+rebalance_mode (all names incl. bottom), capped when long-only, and the signal-mode gate. (Pre-existing unrelated failure noted: `test_ranker_v2_spike_a.py::...admits_concentrated_tail_when_longs_elsewhere` — fails with this diff stashed; to fix before the ranker PR.) | `app/backtesting/agent_simulator.py`, `tests/test_ranker_phase2_neutrality.py` |
| 2026-06-03 | #TBD | **Alpha v2 §3.1 RANKER v2 — VALIDITY FIXES (observability + dollar-neutral-at-target-gross).** The first §3.1 control+treatment run was UNINTERPRETABLE: realized net beta/dollar/GROSS were captured but never surfaced, and the "dollar-neutral" treatment actually ran ~35% net-LONG at 0.351 gross (target 0.80) — the SPY hedge MASKED it as beta-clean (`net_beta_clean OK` while net dollar +0.351). So the earlier "treatment worse than control → thesis dead" read was INVALID (compared long-only vs a half-funded, net-long book); the dollar-neutral high-breadth thesis was never actually tested. **PHASE 1 (observability, PURE-ADDITIVE — no gate/Sharpe/PF/Calmar/`net_beta_clean` change; long-only & intraday arms untouched, gated on `net_exposure_captured`):** realized `mean_gross` was being DROPPED at the swing→cpcv seam — plumbed it end-to-end (`FoldResult.mean_gross` → `SwingStrategy.run_fold` forward → `run_cpcv` combo aggregation → `CPCVResult.path_mean_grosses`/`mean_gross`). Added a net-exposure PANEL to `CPCVResult.print()` (realized net beta mean/p95 + `net_beta_clean` lens, net dollar mean/max, net sector max, realized gross vs target — ASCII-safe for the Windows console) and a per-run result JSON (`_dump_cpcv_result_json` wired into `walkforward_tier3._run_cpcv_swing`) so arms are auditable/comparable. The fast verification re-run then REVEALED the under-gross/net-long book (mean net beta −0.008 / net dollar +0.351 / gross 0.351). **PHASE 2 (neutral-at-target-gross):** root cause — `_process_rebalance` only sized NEW adds; HELD positions kept stale entry-time sizing, so each leg drifted off budget (under-gross, net-long). New `AgentSimulator._rebalance_resize_position` resizes held names (BOTH legs) to the current per-name budget each rebalance — MTM-neutral except the tx cost on the traded share delta (blends entry on adds, releases short collateral at basis on covers; never flattens to 0 — `to_drop` owns exits; long adds respect `_effective_cash`); wired as resize passes after the long/short add loops (L/S arm only — long-only momentum path BYTE-IDENTICAL). `short_per_pos` now sizes the full realized target book regardless of `short_delta.to_add` (so the leg can be re-funded when all names are held). `apply_net_sector_cap` gains a BREADTH pass: after the net-sector greedy admit, fill the remainder by rank to `n_target` — dollar-neutrality is now enforced by per-leg SIZING (not count) and residual beta by the SPY hedge, so the short COUNT/breadth the thesis needs (IR ~ IC·√breadth) is preserved instead of being starved by the per-sector cap (was capping ~floor(0.30×60)=18/sector when the loser tail concentrates away from the longs). Opus deep-dive verified: all 4 resize branches MTM-neutral except tx; the per-position collateral invariant (short_collateral == entry×qty) holds across open→add→cover→drop; just-added names resize to delta=0 (no double-tx, `_rebalance_entry_price` mirrors the add-loop fill); resize runs BEFORE the SPY hedge + EOD capture; no sizing feedback. NO change to gate math/DSR/N_eff/OOS/sacred-holdout. Acceptance for the Phase-2 verification re-run: realized `mean_gross ≈ 0.80`, `|net dollar| ≤ 0.05`, `net_beta_clean OK`, short count ≈ target. Tests: `tests/test_ranker_observability.py` (7 — gross plumbing, print-panel gating, ASCII-safety, JSON fields) + `tests/test_ranker_phase2_neutrality.py` (11 — resize MTM-neutrality across the 4 long/short add/reduce paths, blended-basis short-collateral invariant, never-flatten, effective-cash guard, breadth admission fills to n_target / never shorts a long / respects ceiling). | `scripts/walkforward/cpcv.py`, `scripts/walkforward/gates.py`, `scripts/walkforward/strategies/swing.py`, `scripts/walkforward_tier3.py`, `app/backtesting/agent_simulator.py`, `app/strategy/portfolio_construction.py`, `tests/test_ranker_observability.py`, `tests/test_ranker_phase2_neutrality.py` |
| 2026-06-03 | #TBD | **Alpha v2 §3.1 RANKER v2 — Spike-A pre-run LAST FIX: net-beta acceptance LENS + realistic-fixture proof (two interpretation/test blockers).** Core book construction was already CONFIRMED sound; these two fixes are in the interpretation/test layer so a genuinely-neutral book is not FALSELY failed and a real failure cannot slip through. **BLOCKER 1 (net-beta lens) — `net_beta_clean` graded on the wrong statistic.** The SPY beta-hedge overlay re-sizes only on the 5-day rebalance cadence but net beta is captured DAILY, so `max_abs_net_beta` (raw daily max, incl. warmup) spikes to ~0.3-0.5 between rebalances even on a book that is beta-neutral ON AVERAGE (mean ≈ −0.01 to −0.07). The old `CPCVResult.net_beta_clean` required `|mean| ≤ 0.15 AND max_abs ≤ 0.15` → would report `net_beta_clean=False` on a neutral book (mis-firing the §9-Q4 "re-run beta-neutral" rule + the |net beta|≤0.15 acceptance), AND it diverged from the TEST which already graded warmup-trimmed p95. **FIX:** new SHARED helper `net_exposure.steady_state_net_beta(net_by_date)` computes the PERSISTENT lens — `mean_net_beta` + warmup-trimmed steady-state `p95_abs_net_beta` over the two-sided window after dropping the first `NET_BETA_WARMUP_TWO_SIDED=20` two-sided days (single source of truth for the warmup trim). `net_beta_clean` now keys on `|mean| ≤ 0.15 AND p95 ≤ 0.15`; `max_abs_net_beta` is retained as a DIAGNOSTIC only (not in the clean/accept decision). PURE-ADDITIVE p95 capture threaded `summarize_net_exposure` → `SimResult.p95_abs_net_beta` → `FoldResult.p95_abs_net_beta` → `CPCVResult.path_p95_abs_net_betas`/`p95_abs_net_beta` (combo-max, mirroring max_abs aggregation). PRODUCTION and the regression TEST (`_steady_state_net`) now call the IDENTICAL helper + window so they can never diverge again (locked by `test_production_and_test_net_beta_use_same_helper`). **BLOCKER 2 (fixture wasn't realistic) — `_realistic_r1k_fixture` did not reproduce Failure B.** The illiquid tail was sized `vol_base=800_000`×~$100 = ~$80M ADV, ABOVE the 50M short floor → 0/40 tail names filtered → never exercised the short-underfill→net-long pathology that invalidated run 2. **FIX:** illiquid sub-tail now ~250k shares ×~$100 ≈ $25M ADV (genuinely sub-50M; 40/80 short-tail names filtered); fixture grown to 240 names / 80-name 3-sector low-beta tail. New `short_realized_rescale` kwarg on `AgentSimulator` (default True = FIX-1b realized-count rescale = unchanged production path; False = pre-FIX-1b divide-by-`short_target_n` underfill) lets the test reproduce Failure B at root. Regression PROVES the cycle is broken: OLD path (per-side cap + pre-FIX-1b sizing + no hedge) → short_gross 0.20 (<0.25) AND mean net_dollar 0.15 (>0.10, net-long drift) AND production lens FAILS (mean_nb 0.26 / p95 0.46); CURRENT path (net cap + FIX-1b rescale + hedge) → mean|net_dollar| 0.02 (<0.05) AND production `net_beta_clean` lens CLEAN (mean −0.01 / p95 0.11) while raw daily max stays 0.33 (the transient the lens correctly tolerates). NO change to overlay mechanics, short P&L, tier3 wiring, the ranker/label, or the net-exposure capture semantics; long-only / signal / default paths byte-identical (`short_realized_rescale` defaults True; p95 field pure-additive). **BLOCKER 3 (short equal-weight vs long inverse-vol): DEFERRED** — non-blocking gross drift (converges to short_gross 0.39-0.40 per the diagnosis); making the short leg inverse-vol would change short P&L sizing (explicitly out of scope for this pre-run pass). Tests: `tests/test_ranker_v2_spike_a.py` (31; +4 net) — `net_beta_clean` clean on mean −0.07/p95 0.12/raw-max 0.35 (transient tolerated) and NOT clean on persistent mean 0.26; production==test same-helper lock; fixture-filters-tail-below-50M guard; OLD-reproduces-Failure-B; CURRENT-resolves-by-production-metric. Verified unchanged: 88 rebalance/portfolio/cpcv/swing + 62 gate tests. | `app/backtesting/net_exposure.py`, `app/backtesting/agent_simulator.py`, `app/backtesting/strategy_simulator.py`, `scripts/walkforward/gates.py`, `scripts/walkforward/strategies/swing.py`, `scripts/walkforward/cpcv.py`, `tests/test_ranker_v2_spike_a.py` |
| 2026-06-03 | #TBD | **Alpha v2 §3.1 RANKER v2 — RE-ARCHITECTURE: abandon the standalone driver, run both arms through tier3; NET-sector cap (Failure B fix) + SPY beta-hedge OVERLAY.** Two ~5hr Spike-A runs were invalidated: **(A)** the standalone `scripts/run_ranker_v2_cpcv.py` built a bare `SwingStrategy(...)` that DROPPED the tier3 CLI defaults producing +0.22 (ATR 0.5/1.5, `no_prefilters`, opportunity-score, earnings-blackout) → ran a different book → −1.36; **(B)** the per-side short-gross sector cap (`apply_sector_cap_shorts`, 18/sector) + `short_min_adv=50M` structurally starved the sector-concentrated, illiquid R1K short tail → realized short gross ~0.13 vs 0.40 → ~27% net long. **RETIRE DRIVER:** removed `scripts/run_ranker_v2_cpcv.py` + its driver-specific tests (`TestDriver`/`TestBaselinePin`/`TestFoldParamGuard`); both experiment arms now run through `walkforward_tier3.py::_run_cpcv_swing` (inherits the faithful defaults automatically + already wires the L/S short kwargs). KEPT: `net_exposure.py`, the `CPCVResult`/`FoldResult`/`SimResult` net-exposure fields, the short-kwarg wiring, the `long_budget`/realized-count sizing fix; the reusable tests (net-beta capture, PIT, sizing, neutrality) were MIGRATED to exercise the sim/tier3 path directly. **DESIGN:** rebalance-vs-rebalance (kills the signal-vs-rebalance confound) — CONTROL = rebalance long-only 60-name high-breadth; TREATMENT = same engine + dollar-neutral shorts (60/side, 0.40/0.40) + net-sector cap + beta overlay; only neutrality differs (the signal-mode +0.22 is a separately-reported reference). **NET-SECTOR CAP (Failure B fix):** new `apply_net_sector_cap` in `app/strategy/portfolio_construction.py` — admits shorts greedily worst-first with NO per-side per-sector count cap, skipping a short only if its admission pushes that sector's projected \|net exposure\| (= long_count − short_count, as a fraction of n_target) above the cap OR away from neutral; lets the concentrated short tail fill where the longs are not, and lets shorts OFFSET a long-over-concentrated sector (the key behavioral diff from the per-side cap). Wired in `AgentSimulator._process_rebalance` behind the new `net_sector_cap=False` kwarg (opt-in; `apply_sector_cap_shorts` per-side path is the default → every existing rebalance/L-S run byte-identical). **SPY BETA-HEDGE OVERLAY:** the existing `spy_beta_hedge` block (`agent_simulator.py`) extended — when `enable_shorts=True` it now sizes the SPY short to the RESIDUAL NET BETA of the whole book (notional-weighted Σ w_long·β_long − Σ w_short·β_short, clamped ≥0 and ≤spy_beta_cap) and KEEPS the single-name shorts (overlay, not replacement); long-only Option-A path (`enable_shorts=False`) byte-identical. `compute_book_net_exposure` gains a `hedge_keys` arg: the `__SPY_HEDGE__` overlay counts toward net BETA (β≡1) but is EXCLUDED from net_dollar/net_sector/n_short (it is a market instrument, not a stock bet) so "dollar neutrality" stays a property of the single-name book. **TIER3 WIRING:** `_run_cpcv_swing` now forwards `net_sector_cap`/`spy_beta_hedge`/`spy_beta_lookback`/`spy_hedge_max_gross`/`capture_net_exposure`/`net_beta_lookback`; new flags `--net-sector-cap`, `--capture-net-exposure`, `--net-beta-lookback` (short kwargs / rebalance flags / `--spy-beta-hedge`/`--spy-hedge-max-gross` already existed). `SwingStrategy` forwards `net_sector_cap`/`spy_beta_hedge`/`spy_beta_lookback`/`spy_hedge_max_gross`. **REGRESSION PROOF (the key gap — synthetic balanced/liquid universe kept passing while R1K failed):** new `_realistic_r1k_fixture` mimics R1K's pathology — bottom-of-rank short tail concentrated in 3 sectors with a sub-50M-ADV illiquid sub-tail and SYSTEMATICALLY LOWER beta than the longs (residual net beta after dollar-neutralizing). On THAT fixture, the OLD per-side-cap/no-hedge book leaves mean & p95 \|net beta\| ≥ 0.15 (NOT clean alpha), while the TREATMENT (net cap + beta overlay) reaches mean\|net_dollar\|<0.05 AND p95\|net beta\|<0.15 on the dollar-neutral steady-state window (the SPY overlay drives residual net beta from ~+0.20 to ~−0.01). NO change to gate math, DSR, N_eff, OOS/sacred-holdout, short P&L mechanics, the ranker/label, or the net-exposure capture semantics for non-hedge books. Self-validation anchor: the recorded 60d momentum-baseline rebalance CPCV (ML_EXPERIMENT_LOG "Momentum Baseline CPCV 2026-05-24", mean +1.306) is the engine-sanity reference; the fold-matched CONTROL arm is the in-experiment reference. Tests: `tests/test_ranker_v2_spike_a.py` (27) — short+re-arch kwarg storage/forwarding/long-only-default, dollar-neutrality, sector/inv-vol, PIT, net-beta capture (incl. hedge β≡1 + hedge excluded from net_dollar), CPCV pure-additive, BUG-1 sizing, net-sector-cap UNIT (per-side starves a concentrated tail; net cap admits it / offsets long concentration), realistic-universe regression (old fails net-beta / treatment passes both), beta-overlay reduces residual net beta. | `app/backtesting/agent_simulator.py`, `app/backtesting/net_exposure.py`, `app/strategy/portfolio_construction.py`, `scripts/walkforward/strategies/swing.py`, `scripts/walkforward_tier3.py`, `docs/living/RANKER_V2_DESIGN.md`, `tests/test_ranker_v2_spike_a.py` (removed `scripts/run_ranker_v2_cpcv.py`) |
| 2026-06-02 | #TBD | **Alpha v2 §3.1 RANKER v2 — Spike A TWO BLOCKING SIZING/FOLD BUGS (book ran ~76% net long; baseline anchor on wrong folds):** the prior Spike A run was invalidated by two root-cause defects. **BUG 1 (real sizing bug) — `long_budget` was dead code → the "dollar-neutral" book ran ~76% net long.** In `AgentSimulator._process_rebalance` (~line 1810) `split_gross_budgets` returned `long_budget, short_budget`, but `long_budget` was NEVER used — the long leg called `compute_inverse_vol_weights(..., total_equity=equity, gross_exposure_multiplier=long_mult)` / `compute_equal_weights(..., equity, long_mult)`, sizing the long book to FULL equity × regime_mult instead of `long_gross`. Realized on the run: long≈0.93×NAV, short≈0.17×NAV, net_dollar≈0.76 vs the 0.40/0.40 target (net≈0). (Masked historically because momentum runs used long_gross≈0.95≈full NAV.) **FIX 1a (the dead budget):** the L/S path (`enable_shorts=True`) now uses `long_budget` (which `split_gross_budgets` already multiplies by `long_regime_mult`) as the equity base with `gross_exposure_multiplier=1.0` → long gross = `long_gross`×NAV. **FIX 1b (short underfill + symmetric long):** the short leg reached only ~0.17 vs 0.40 because `short_min_adv`(50M)+`sector_cap`(0.30) shrink the eligible short set below `short_n=60`; dividing the short budget by `short_n` underfilled. Fix: rescale the per-name budget to the size of the REALIZED qualifying TARGET BOOK (`delta.target` / `short_delta.target` names that have a valid same-day entry price), NOT `rebalance_target_n`/`short_target_n` and NOT `len(to_add)` — weight across the full realized target book against `total_equity=budget`, then open only the NEW adds at their per-name weight (held names already carry their share at the same per-name budget). This makes each rebalance target `budget` total for the whole book so the live point-in-time gross tracks `long_gross`/`short_gross`×NAV instead of stacking each new batch onto the existing book (which had ballooned gross to ~2-4×). `short_min_adv` is NOT relaxed — liquid shorts preserved, gross hit via per-name rescaling. New helper `_rebalance_entry_price()` (mirrors the inline open/close fill logic) defines "qualifying". **Long-only paths UNTOUCHED:** the SIGNAL-mode long-only path (`rebalance_mode=False`, the +0.22 path) never enters `_process_rebalance`; the long-only REBALANCE path (`enable_shorts=False`) keeps the existing full-NAV×`long_mult` sizing byte-identical (verified: 31 rebalance tests + 210 swing/CPCV/portfolio tests pass UNCHANGED — none had encoded the buggy full-NAV-long behavior). Net-exposure summary + `SimResult` gain `mean_gross`/`last_gross` and an optional `net_exposure_by_date` (per-EOD live-book snapshots) so a POINT-IN-TIME neutrality test can inspect the live book (not trade flow). **Verified on a 130-name synthetic dollar-neutral book (60/60, 0.40/0.40):** once both legs populate, net_dollar ≈ 0 (≈+0.03 to −0.03) and gross ≈ 0.80-0.83 (creeps slightly above 0.80 over months from price drift on fixed-qty positions — realistic, positions aren't continuously resized). **BUG 2 (driver-only) — `--years` default 6 ≠ the +0.22 run's 5 → wrong fold boundaries.** `scripts/run_ranker_v2_cpcv.py` defaulted `--years=6`; the +0.22 swing v224 run inherited `walkforward_tier3.py`'s `--years` default of 5 (the logged invocation passed no `--years`). `run_cpcv` builds folds from `total_years`+`retrain_as_of()`, so 6≠5 shifted every fold window (added the 2020 COVID year) and broke the baseline self-validation (−0.83 vs +0.22). **FIX 2:** `--years` default → **5**; added a loud `_assert_fold_params` guard — the BASELINE arm asserts its `(years, cpcv_k, cpcv_paths, purge_days, embargo_days)` == `BASELINE_PIN` (now pins `years=5, cpcv_k=6, cpcv_paths=2, purge_days=85, embargo_days=85` = the tier3 CLI defaults the +0.22 run used) and raises `AssertionError` on mismatch; the dn arm only LOGS (free to sweep); SMOKE exempted. The driver parser was refactored to `_build_parser()` so the test reads the REAL default. NO change to gate math, DSR, N_eff, OOS/sacred-holdout, short P&L mechanics, the ranker/label, or the net-exposure capture from the prior entry. Tests: `tests/test_ranker_v2_spike_a.py` (33, +9) — point-in-time neutrality (`abs(net_dollar)<0.10` AND `gross≈0.80±0.10` on the live book, the guard that would have caught Bug 1), `long_budget` flows into long sizing (first-rebalance long/equity≈0.40 not full NAV), short reaches target gross via realized-count rescaling under a real ADV filter (peak short/eq>0.30, median>0.25 vs the ~0.17 underfill), `--years` default==5, baseline-arm-passes-with-pinned / raises-on-year-mismatch / raises-on-k-mismatch / dn-arm-logs-not-raises / smoke-exempt. | `app/backtesting/agent_simulator.py`, `app/backtesting/net_exposure.py`, `app/backtesting/strategy_simulator.py`, `scripts/run_ranker_v2_cpcv.py`, `docs/living/RANKER_V2_DESIGN.md`, `tests/test_ranker_v2_spike_a.py` |
| 2026-06-02 | #TBD | **Alpha v2 §3.1 RANKER v2 — Spike A BLOCKER FIXES (realized net-beta capture + +0.22 baseline pin):** the prior Spike A could not interpret a positive result — a dollar-neutral book equalizes long/short DOLLARS, giving net beta ≈ 0 only if long/short avg betas match (for a ranker they systematically don't), so a +0.5 could be leftover market beta, not alpha. The locked rule (RANKER_V2_DESIGN.md §9-Q4: dollar-neutral primary, switch to beta-neutral only if realized \|net beta\| > 0.15) REQUIRES measuring realized net beta, which was not captured. **FIX 1 (the blocker) — realized net-beta / net-dollar / net-sector capture, PURE-ADDITIVE:** new `app/backtesting/net_exposure.py` — `compute_pit_beta` (trailing-60d OLS slope of name daily returns vs SPY using ONLY bars strictly `< as_of` → NO look-ahead; reuses the spy_beta_hedge strict-`<` convention), `compute_book_net_exposure` (signed book net beta = `Σ w_long·β_long − Σ w_short·β_short`, net-dollar = `(long−short)/equity`, signed per-sector net + max\|sector\|; read-only over positions), `summarize_net_exposure`, `NET_BETA_ALPHA_THRESHOLD=0.15`. `AgentSimulator` gains `capture_net_exposure=False` (+`net_beta_lookback=60`); when True it records net exposure per EOD at the existing `deployment_by_date` seam (agent_simulator.py:~655) and surfaces mean/last/max\|net beta\|, mean/max\|net dollar\|, max\|net sector\|, `net_exposure_captured` on `SimResult`. OFF by default → long-only path BYTE-IDENTICAL (gate test `test_long_only_capture_off_is_byte_identical`: sharpe/return/trades/PF unchanged vs no-flag). `SwingStrategy` forwards the flag (defaults to `enable_shorts`) + a symbol→sector map; `FoldResult` carries the fields; `run_cpcv` aggregates them onto NEW optional `CPCVResult` fields (`path_mean_net_betas`/`path_max_abs_net_betas`/`path_mean_net_dollars`/`path_max_abs_net_dollars`/`path_max_abs_net_sectors`/`net_exposure_captured`, default empty) with `mean_net_beta`/`max_abs_net_beta`/`max_abs_net_dollar`/`max_abs_net_sector`/`net_beta_clean` properties — SAME pure-additive discipline as §1.2 `path_fold_members` (verified: default CPCVResult unchanged + 50 CPCV/rebalance tests pass). The driver reports net beta / max\|net beta\| / net dollar / max\|net sector\| and the interpretation hook `is_clean_dollar_neutral_alpha` (\|net beta\| > 0.15 → NOT clean → re-run beta-neutral). **FIX 2 — pinned + aligned the +0.22 baseline:** the +0.22 anchor (MODEL_STATUS.md / ML_EXPERIMENT_LOG.md v224 per-fold CPCV, `--as-of 2026-05-29`) was run WITHOUT `--rebalance-mode` → it was a SIGNAL-MODE long-only book capped at `MAX_OPEN_POSITIONS=5` (the 16.5% deployment + per-trade PF/Calmar artifacts confirm the 5-position signal book), NOT a 30-name rebalance book. The driver's `--baseline` arm previously ran `rebalance_mode=True, target_n=30, rebalance_days=20` — reproducing neither the mechanism nor the breadth. Pinned the exact invocation in `BASELINE_PIN`; the `--baseline` arm now runs `rebalance_mode=False, enable_shorts=False, factor_scorer=None` (native ranker, 5-pos signal mode) → genuine self-validation anchor; relabeled `baseline_long_only`→`baseline_signal_long_only_5pos`. Cost parity note: the +0.22 run used `--swing-cost-bps 5.0` default = one-way 2.5 bps/side → pass `--cost-bps 2.5`. Cost-sweep {2,5,10,15,20}bps left as a documented TODO (non-blocking; single 5bps point used). NO change to gate math, DSR, N_eff, OOS/sacred-holdout, short P&L mechanics, dollar-neutral construction, or the ranker/label. Tests: `tests/test_ranker_v2_spike_a.py` (24, +12) — PIT beta recovers a known beta, signed-net-beta hand calc (dollar-balanced book with mismatched betas → net beta 0.20 ≠ 0, proves a beta artifact is caught), net-dollar/net-sector hand calc, PIT no-look-ahead (future SPY/price move doesn't change a past day's beta), L/S sim populates the fields, long-only capture-off byte-identical, default CPCVResult unchanged + aggregation/threshold, `BASELINE_PIN` is signal-mode-5pos, baseline arm reproduces the pinned config, DN arm enables capture. | `app/backtesting/net_exposure.py`, `app/backtesting/agent_simulator.py`, `app/backtesting/strategy_simulator.py`, `scripts/walkforward/strategies/swing.py`, `scripts/walkforward/cpcv.py`, `scripts/walkforward/gates.py`, `scripts/run_ranker_v2_cpcv.py`, `docs/living/RANKER_V2_DESIGN.md`, `tests/test_ranker_v2_spike_a.py` |
| 2026-06-02 | #TBD | **Alpha v2 §3.1 RANKER v2 — Spike A: dollar-neutral high-breadth swing L/S book (book-construction-only re-test of the existing ranker; NO residualization):** the single concrete simulator-side gap was that `SwingStrategy` did NOT forward the L/S short kwargs to `AgentSimulator` (the per-fold swing strategy omitted `enable_shorts`/`long_gross`/`short_gross`/`short_target_n`/`short_min_adv`/`short_add_threshold`/`short_drop_threshold`, though the engine has a full L/S rebalance path). **NET-NEW WIRING:** added these 7 kwargs to `SwingStrategy.__init__` and forwarded them in the `AgentSimulator(...)` construction inside `run_fold`; defaults equal the `AgentSimulator` constructor defaults with `enable_shorts=False`, so **the long-only swing path is byte-identical** (confirmed: 31 existing swing tests — `test_agent_simulator_rebalance`, `test_per_fold_retrain_swing`, `test_pm_rebalance_mixin` — pass unchanged). The CLI per-fold-retrain path (`walkforward_tier3.py:_run_cpcv_swing`) now also threads the same kwargs from the EXISTING argparse flags (`--enable-shorts`, `--long-gross`, `--short-gross`, `--short-target-n`, `--short-min-adv`, `--short-add-threshold`, `--short-drop-threshold`) via `getattr(args, …, <AgentSimulator default>)` — opt-in, long-only default unchanged. **DRIVER:** new `scripts/run_ranker_v2_cpcv.py` — load-once, deterministic (`retrain_as_of()`/`--as-of`), ASCII-safe, ARTIFACTS-FIRST (writes the result JSON BEFORE printing the table, per the cp1252 lesson; `logs/ranker_v2/spike_a_<arm>_<ts>.json`), `--smoke` fast path. It reuses the HONEST per-fold-retrain OOS machinery (`run_cpcv`, `SwingFoldRetrainer`, `TrainWindowCache`, `assert_model_oos`, N_eff=n_folds, the significance gate) — does NOT fork `run_cpcv`. **Registered primary book (owner-locked):** dollar-neutral (`long_gross=0.40`+`short_gross=0.40` → `net_target=0` via `split_gross_budgets`, gross=0.80 ≤ 80% NAV), sector-capped (0.30), inverse-vol, weekly (`rebalance_days=5`), **60 names/side**; sensitivity arms (40/80 side, monthly) are simple flags. **Same model + same label as the +0.22 baseline** (`SWING_RETRAIN` lambdarank model + lambdarank label, `factor_scorer=None` → native `AgentSimulator._pm_score`) so Spike A isolates the BOOK-CONSTRUCTION effect (breadth + neutrality) from any signal change. `--baseline` arm = long-only self-validation anchor (reproduces ~+0.22). Cost = 5 bps/side headline, parameterized (`--cost-bps`). Smoke run (k=3,p=2,3y,tiny universe) executed end-to-end through the true OOS path + emitted a valid artifact. **Known limitation (carried, NOT a blocker):** per-fold realized net-beta / per-sector net exposure are not yet captured (CPCVResult exposes Sharpe/t-stat/%pos/P5/deployment); dollar-neutrality is guaranteed by construction (`net_target=0`) and verified directly in the unit tests. Borrow optimism (shorts bottom-of-rank R1K @5%/yr, `short_min_adv=50M` floor) flagged; §3.3 SI/borrow feed is the live-readiness follow-on, explicitly non-blocking. NO change to gate math, DSR, N_eff, OOS/sacred-holdout, or the simulator core. Tests: `tests/test_ranker_v2_spike_a.py` (12) — short-kwarg storage + forwarding + long-only-default-keeps-shorts-off, `split_gross_budgets` net=0, long-only-zero-shorts, dollar-neutral-book-balanced (open long vs short notional within 0.5–2.0×), inverse-vol-applied, sector-cap-threaded, no-look-ahead (inverse-vol weights ignore bars ≥ decision day), driver builds dollar-neutral + baseline arms, artifact ASCII round-trip. | `scripts/walkforward/strategies/swing.py`, `scripts/walkforward_tier3.py`, `scripts/run_ranker_v2_cpcv.py`, `docs/living/RANKER_V2_DESIGN.md`, `tests/test_ranker_v2_spike_a.py` |
| 2026-06-02 | #TBD | **Alpha v2 Phase-1 §1.2 — `run_cpcv` path-membership instrumentation (PURE-ADDITIVE) + PEAD crisis-robustness LOCO root-cause fix:** Added a new BACKWARD-COMPATIBLE field `CPCVResult.path_fold_members: list[list[int]]` (default `[]`) to `scripts/walkforward/cpcv.py`. For each SURVIVING path (1:1 aligned with `path_sharpes`, same order) it records the GLOBAL fold ids that path aggregated — the exact `fold_idx` value `run_cpcv` passes to `strategy.run_fold` (`combo_idx*len(all_boundaries)+ti+1`), captured AFTER all skip/purge/BUG-23-overlap guards, only on successful `run_fold`. `run_cpcv` is now the SINGLE SOURCE OF TRUTH for CPCV path grouping. **Pure-additive, zero behavior change:** no metric, gate, DSR, N_eff, OOS/sacred-holdout, simulator, or PEAD-scorer path is touched — confirmed by 99 existing CPCV/gate tests (`test_cpcv_*`, `test_wf3_cpcv`, `test_significance_gate`, `test_gate_*`, `test_regime_gate`, `test_data_span_gate`, `test_walkforward_dsr_n`) all passing unchanged. **Why:** the `scripts/pead_crisis_robustness.py` LOCO sub-analysis previously REIMPLEMENTED the path grouping (`_cpcv_path_membership`) and silently DIVERGED — it omitted the BUG-23 expanding-window overlap guard, so the real run (21 paths / k=8) was reconstructed as 27 paths / 48 fold-members, computing every LOCO row on the WRONG folds. FIX: deleted `_cpcv_path_membership` + `_build_boundaries` (root-cause removal — a reconstruction can re-diverge whenever cpcv.py changes); the LOCO capture run now reads `result.path_fold_members` and groups captured per-fold equity curves by THAT (identical to the real CPCV by construction). Added a LOCO SELF-CHECK (bug 2): the `(none removed)` recompute must reproduce `run_cpcv`'s OWN mean Sharpe within `LOCO_SELF_CHECK_TOL=1e-4` (same curves + same grouping → near-exact; 1e-4 allows the single 4dp rounding step in `_path_metrics`), logged loudly + flagged (`self_check_ok`) on divergence. Smoke k=4 `path_fold_members = [[14,15],[18],[23,24]]` (the real 3-path grouping, NOT the inflated reconstruction). Also corrected docstrings (`app/ml/pead_scorer.py`, harness §C) + the sub-analysis-C interpretation text: the generic regime control in the equal-weight +0.546 book is a vol/trend-gated ENTRY BLOCK (scalar<floor → block all entries) + a coarse saturating [0.75-1.25×] conviction-size tilt — NOT continuous gross vol-targeting (description-only; no behavior change). Tests: `tests/test_pead_crisis_robustness.py` (19) — replaced the vacuous `test_path_membership_matches_run_cpcv_count` with `test_cpcv_exposes_path_fold_members_aligned_with_path_sharpes`, `test_path_fold_members_is_the_real_surviving_grouping` (equals the real grouping derived from `run_cpcv`'s own `run_fold` calls; asserts <6 surviving paths so it FAILS on an inflated reconstruction), `test_loco_uses_real_membership_not_reconstruction` (asserts the reconstruction helpers are GONE + LOCO groups over the real membership), `test_loco_self_check_matches_run_cpcv_mean_sharpe`. | `scripts/walkforward/cpcv.py`, `scripts/pead_crisis_robustness.py`, `app/ml/pead_scorer.py`, `tests/test_pead_crisis_robustness.py` |
| 2026-06-02 | #TBD | **Alpha v2 Phase-1 §1.1 — PEAD cost-sensitivity sweep + slippage/cost parameterization:** Lifted `ENTRY_SLIPPAGE_PCT`/`STOP_SLIPPAGE_PCT` (and confirmed `transaction_cost_pct`) from hardcoded `AgentSimulator` module constants to constructor kwargs, defaulting to the prior values (BYTE-IDENTICAL at default — verified by `test_default_slippage_and_cost_equal_prior_constants` + identical-fill test; the validated +0.546 PEAD run is unaffected). New tool `scripts/pead_cost_sensitivity.py`: loads PEAD data ONCE then re-simulates the C(8,2) CPCV at one-way cost levels {2,5,10,20,35,50} bps (slippage folded out → `cost_bps` = honest total per-side cost on entry + every exit; round-trip ≈ 2×), plus a self-validating ANCHOR row running the exact committed config (5bps fee + 3bps entry-slip + 5bps stop-slip) which must reproduce +0.546 or it logs `HARNESS DIVERGES`. Break-even/+0.40 interpolation computed over the pure-cost rows ONLY (anchor excluded). **Result (2026-06-02, anchor +0.548 ≈ validated +0.546 → harness sound):** mean Sharpe +0.662/+0.612/+0.536/+0.402/+0.406/+0.252 at 2/5/10/20/35/50 bps; **+0.402 at 20 bps PASSES the ≥0.40 acceptance**, positive even at 50 bps; t-stat falls <2.0 and %pos 95%→57% by 20 bps (cost-robust in Sharpe, statistically fragile). Output made Windows-safe: artifacts (JSON/CSV) written BEFORE `_print_table` (wrapped in try/except) with `encoding="utf-8"`, all `print`/`logger` strings ASCII-only (`→`→`->`, `←`→`<-`, `±`→`+/-`, `§` removed) — fixes a real `UnicodeEncodeError` on Windows cp1252 that had aborted artifact persistence on the first run; `run_pead_cpcv.py` log strings de-unicoded too. NO change to cost math, the scorer, the gate, or CPCV. Tests: `tests/test_pead_cost_sensitivity.py` (13) — byte-identical default, round-trip=2×one-way, monotonic per-side cost, interp-crossing math, anchor present/flagged/excluded-from-interp, artifacts-persist-even-if-print-fails, no-non-ASCII-runtime-strings. | `app/backtesting/agent_simulator.py`, `scripts/run_pead_cpcv.py`, `scripts/pead_cost_sensitivity.py`, `tests/test_pead_cost_sensitivity.py` |
| 2026-06-02 | #TBD | **Phase-4 significance-gate review fixes (FIX-1/2/3 + coverage) — blocking defects in the two-tier gate:** **FIX-1 (cron auto-retire + capital unreachable):** under `GATE_MODE="significance"` a WF-only retrain HARD-FAILED `WalkForwardReport.gate_passed()`, and `retrain_cron.py` fed that boolean to `record_tier3_result(gate_passed=False)` → `status="RETIRED"` + `_restore_previous()`, so EVERY scheduled WF retrain auto-retired the fresh model and rolled back. Added a tri-state `GateOutcome{PROMOTE,RETIRE,INCONCLUSIVE}` enum + `WalkForwardReport.gate_outcome()`: significance+WF → `INCONCLUSIVE` (report-only). `retrain_cron.py` (swing+intraday) now reads `gate_outcome()` and on `INCONCLUSIVE` keeps current model status untouched (no retire/rollback/record-fail), logging that CPCV is required. Legacy `mean_sharpe` path unchanged (PROMOTE on pass, RETIRE on real fail). Made the CAPITAL tier reachable: `walkforward_tier3.py` threads `--gate-tier {paper,capital}` (default paper) + `--paper-confirmation` + `--regime-waiver-approved` into `_cpcv_swing_gate_ok` / intraday short-circuit `gate_passed(tier=...)`. **FIX-2 (gate BLOCKED PEAD):** PEAD's real CPCV has `worst_regime_sharpe=None` (event-sparsity: `<REGIME_MIN_OBS` same-regime days — documented "not a bug"); `_significance_backstops_ok` failed-closed on None → PEAD FAILED paper. Distinguished event-sparsity from data-bug at the SOURCE: `compute_regime_sharpes` now fills an `obs_counts` out-dict (raw per-regime counts BEFORE the REGIME_MIN_OBS filter) → `FoldResult.regime_obs_counts` → `run_cpcv` sets `CPCVResult.regime_insufficient_obs` (True = obs seen but all buckets sub-floor = event-sparsity; False = no obs = data-bug). Gate: PAPER WAIVES regime backstop for event-sparsity AND sets `requires_human_review_flag=True` (surfaced in `gate_detail`); CAPITAL never auto-waives (needs `regime_waiver_approved=True` explicit sign-off); data-bug None fails closed BOTH tiers. Narrow, flagged, paper-only waiver — NOT a global fail-open. **FIX-3 (dishonest rescore):** `scripts/rescore_gates.py` reimplemented the threshold math and hardcoded `backstops_ok=True` (falsely showed PEAD PASS). Rewrote it to construct REAL `CPCVResult`s (actual mean/tstat/%pos/P5/n_folds/PF/Calmar/`worst_regime_sharpe`; PEAD=None+`regime_insufficient_obs=True`) and call the REAL `gate_passed(tier=...)`/`gate_detail`. Output: PEAD R1K → PAPER PASS (HUMAN-REV=YES) / CAPITAL HOLD; Swing/Intraday/Small-mid/QualityShort/Insider → FAIL all tiers; LEGACY(0.80) all-FAIL. **Coverage:** `tests/test_significance_gate.py` +7 (18 total) — PEAD event-sparsity PAPER-PASS+flag/CAPITAL-FAIL, capital regime-waiver-requires-signoff, data-bug-None fails-closed-both-tiers, WF-only-INCONCLUSIVE-not-RETIRE, cron-decision-no-retire, mean_sharpe-outcome-PROMOTE/RETIRE-unchanged, capital-tier-reachable-via-`_cpcv_swing_gate_ok`. Legacy mean_sharpe corpus unchanged. | `scripts/walkforward/cpcv.py`, `scripts/walkforward/gates.py`, `scripts/walkforward/regime.py`, `scripts/walkforward/strategies/swing.py`, `scripts/run_pead_cpcv.py`, `scripts/retrain_cron.py`, `scripts/walkforward_tier3.py`, `scripts/rescore_gates.py`, `tests/test_significance_gate.py` |
| 2026-06-02 | #TBD | **Significance-first two-tier promotion gate (Phase-4) — replaces mean-Sharpe≥0.80 as the PRIMARY discriminator:** new `GATE_MODE` flag (`app/ml/retrain_config.py`, default `"significance"`; `"mean_sharpe"` = faithful legacy reproduction). Under `"significance"`, `CPCVResult.gate_passed(tier="paper"|"capital", paper_confirmation=False)` gates on path-Sharpe **t-stat** (N_eff=n_folds, flipped WARN→BLOCK), `pct_positive`, `p5_sharpe`, and a mean-Sharpe materiality FLOOR — not the 0.80 mean. **PAPER** (forward-validate, no capital): t≥2.0 AND %pos≥0.75 AND P5≥0.0 AND mean≥0.35 + PF/Calmar/regime backstops. **CAPITAL** (real money): PAPER + mean≥0.50 + n_folds≥10 + (t≥2.5 OR documented paper confirmation). A bare `WalkForwardReport` (no path t-stat) HARD-FAILS under significance with "CPCV required for significance gate (WF has no path t-stat)" — does not fabricate a t-stat. Legacy 0.80/1.00 `SWING_GATE`/`INTRADAY_GATE` thresholds kept INTACT (commented LEGACY) for the `mean_sharpe` reproduction path. New re-score artifact `scripts/rescore_gates.py` (`python -m scripts.rescore_gates`): proves the gate promotes ONLY PEAD R1K → PAPER PASS / CAPITAL HOLD; Swing/Intraday/Small-mid-PEAD/QualityShort/Insider → FAIL all tiers; LEGACY(0.80) col all-FAIL. **`mean_sharpe` no-op verified:** the entire pre-Phase-4 gate test corpus (175 tests) passes unchanged under `mean_sharpe` (forced via `tests/conftest.py` autouse fixture); new tests opt into `significance` via the `significance_gate_mode` fixture. No change to DSR math, N_eff=n_folds, OOS/sacred-holdout machinery, simulators, or PEAD scorer. Tests: `tests/test_significance_gate.py` (11) — PEAD paper-PASS/capital-HOLD, capital-fails-on-nfolds, swing FAIL-all, small/mid FAIL (t AND P5), synthetic capital PASS, capital-via-paper-confirmation OR-path, WF-only hard-fail, dispatch, mean_sharpe no-op (0.85 pass / 0.50 fail), rescore-table lock. Full suite 2482 passed / 8 skipped / 0 failed. | `app/ml/retrain_config.py`, `scripts/walkforward/cpcv.py`, `scripts/walkforward/gates.py`, `scripts/rescore_gates.py`, `tests/conftest.py`, `tests/test_significance_gate.py` |
| 2026-06-01 | #TBD | **Pre-run correctness fixes to small/mid-cap PEAD harness (PR #361 review blockers; must be bug-free before CPCV):** **FIX 1 — `delisted_haircut` was a no-op for held delisted names.** The end-of-fold FORCE_CLOSE computed `exit_price` from the name's LAST TRADED bar on-or-before `end_date`, so for a held name `exit_price` was never `None` and the `delisted_haircut=0.70` branch was DEAD CODE — a name that delisted mid-fold booked only the loss already in its last close, not the gap-to-zero a delisting implies (UNDER-penalized delistings, flattered small-cap returns). FIX: in `AgentSimulator.run()` force-close, detect "data-ended while held" — the name's last bar is materially BEFORE `end_date` (≥3 of the fold's SPY-calendar `trading_days` after the last bar had NO bar) → it survived only because no bars existed to trigger an exit, not because it traded to the boundary. For such a name apply the haircut to the LAST CLOSE (long `last_close×(1−h)`, short `last_close×(1+h)`) and log a `FORCE_CLOSE: <sym> data-ended … while held …` warning. A name that DID trade to/near the boundary keeps the full-close MTM behavior. Also fixed the existing wrong SHORT arithmetic in the `exit_price is None` fallback (`×(1−h)` → `×(1+h)`). **Strict no-op when `delisted_haircut=0.0`** (the default) → R1K large-cap `run_pead_cpcv.py` + swing/intraday `walkforward_tier3.py` (both pass 0.0) unchanged; bites ONLY where a non-zero haircut is configured (smallmid passes 0.70). **FIX 2 — eligibility-window look-ahead (minor universe-membership leak).** `run_pead_smallmid_cpcv.run_fold` selected the per-fold universe via `symbols_eligible_in_window(elig, te_start, te_end)` (union over the whole test window) → admitted names that only became liquid LATE in the fold. FIX: new `symbols_eligible_as_of(elig, te_start)` (latest eligibility snapshot on-or-before `te_start`; depends only on data ≤ te_start → PIT) matching the large-cap `run_pead_cpcv.py` "as-of te_start" convention. Tests (`tests/test_smallmid_universe.py`): `test_delisted_haircut_bites_on_data_ended_while_held` (drives the REAL force-close: held long whose data ends ~mid-fold → exit `last_close×0.30`, deep loss; control `haircut=0.0` → full last close, break-even), `test_haircut_not_applied_when_name_trades_to_fold_boundary` (name trading to boundary NOT haircut), `test_delisted_haircut_constructor_clamps`, `test_universe_selection_is_as_of_te_start_pit` (LATE-liquid name excluded, was leaked by the old window form), `test_eligible_as_of_uses_latest_snapshot_on_or_before`. NO change to R1K paths. | `app/backtesting/agent_simulator.py`, `scripts/build_smallmid_universe.py`, `scripts/run_pead_smallmid_cpcv.py`, `tests/test_smallmid_universe.py` |
| 2026-06-01 | #TBD | **Survivorship-safe small/mid-cap PEAD CPCV harness (build-only; highest-EV remaining PEAD experiment):** PEAD works in large-caps (+0.546 honest CPCV); literature says event-drift is strongest in small/mid-caps. Built a NEW harness `scripts/run_pead_smallmid_cpcv.py` cloning `run_pead_cpcv.py`'s verified honest-pipeline machinery (per-fold `n_obs`+`regime_sharpes`+`profit_factor`, OOS guard `trained_through=date.min`, CPCV C(8,2)/k=8/6yr) with FOUR bug-check fixes. **UNIVERSE (C-1 survivorship + H-1 ADV filter in ONE definition):** new `scripts/build_smallmid_universe.py` walks **Polygon grouped-daily flat files** (`PolygonProvider.get_grouped_daily` → new `PolygonS3.get_grouped_daily`, the all-tickers-per-day panel; verified live: 10,822 symbols on 2023-03-08 INCLUDING delisted SIVB+FRC) — the GOLD-STANDARD survivorship-safe candidate source (every ticker that printed a bar that day, kept to its final traded day, never retroactively dropped). Per (day,symbol) it computes trailing-20d `ADV = mean(close×volume)` over days `<=` that day (PIT — a future spike can't change today's eligibility) and keeps names in `[$2M, $50M]` (`ADV_MIN`/`ADV_MAX`; small/mid-cap liquid range), capped top-300/day by PIT-ADV rank (`MAX_NAMES_PER_DAY` — ranked on trailing ADV, not current existence → no survivorship reintroduction). Cached to `data/smallmid/{panel,eligibility}.parquet` (gitignored) so the grouped-daily walk runs ONCE. Replaces the hardcoded `pit_union("russell1000")` in `run_fold` with the PIT band-eligibility set. **PRICES:** Polygon grouped-daily (delisted-inclusive), NOT yfinance. **C-2:** `delisted_haircut=0.70` (held-through-delisting books -70%, not break-even). **H-2:** `transaction_cost_pct=0.0020` (20bps). **M-1:** per-fold guard skips names with `<MIN_HISTORY_BARS=60` bars before `te_start`. Scorer = validated long-only baseline. HONEST FLAGS: (1) shares-outstanding not survivorship-safe-available → ADV dollar-volume BAND is the size proxy (liquidity band, not strict market-cap band); (2) 300/day cap binds (>300 names/day) — PIT-rank-based so survivorship-safe. Validation (Q1-2023, 62 days): ~300 band names/day, 738 distinct; delisted small/mid-caps (COWN, ONEM, UMPQ) correctly retained to last bar; mega-caps SIVB/FRC correctly EXCLUDED (ADV>$50M). Tests: `tests/test_smallmid_universe.py` (6 + 1 skip-network). BUILD ONLY — full CPCV not run. | `app/data/polygon_s3.py`, `app/data/polygon_provider.py`, `scripts/build_smallmid_universe.py`, `scripts/run_pead_smallmid_cpcv.py`, `tests/test_smallmid_universe.py` |
| 2026-06-01 | #TBD | **PEAD conviction sizing (default OFF) — Opus #1 experiment to beat the +0.546 equal-weight ceiling:** added an OPTIONAL conviction-sizing path to `AgentSimulator` that weights each day's NEW long entries by `w_i ∝ clip(SUE_z_i, 0, 3) / realized_vol_i`, normalized so the day's new-entry gross equals the EQUAL-WEIGHT book's gross for the SAME n names (`Σ target-dollars = n × MAX_POSITION_SIZE_PCT × equity`). **Implemented in the SIMULATOR, not the scorer** — STEP-0 finding: in factor mode the sim sizes each PEAD long with a FIXED 20% circuit-breaker stop and passes `conf` only through `size_position`'s weak `conviction_multiplier` (clipped [0.75,1.25]), then clamps every name to the `MAX_POSITION_SIZE_PCT=5%` per-position cap. With `MAX_OPEN_POSITIONS=5` the 5% cap binds for nearly every name → the committed +0.546 baseline is effectively EQUAL-WEIGHT at the 5% cap. There is NO day-level gross normalization in the sim, so true conviction sizing (gross-normalized tilt) MUST be a sim sizing change, not a smarter `conf`. New params: `AgentSimulator(pead_conviction_size=False)`; `_rm_validate(..., bypass_position_cap=False)`. CONFOUND CONTROLS (all three from the task): (1) **gross-normalized** — Σ allocations == equal-weight gross; the 5% per-position cap AND `validate_position_size`/`validate_sector_concentration` are intentionally bypassed for conviction longs (`bypass_position_cap=True`) because re-applying the 5% clamp would shrink the book below the equal-weight gross and reintroduce a (downward) leverage confound + drop the most-conviction names from the entry set. (2) **PIT-safe inputs** — SUE_z standardizes the raw `fmp_surprise_1q` against an EXPANDING pool of surprises observed ONLY on days `<= entry day` (`_sue_pool` keyed by observation day; `_sue_zscore_pit` filters `d <= day`; pool is per-fold so it never spans the embargo), returning neutral z=0 with <2 prior obs; realized vol (`_realized_vol_pit`, 20d) uses closes with index date STRICTLY `< entry day`. (3) **entry set unchanged** — the conviction pre-pass only re-weights the SAME long names the equal-weight book would enter (same `long_threshold`, same VIX block, same slot order); verified at sim level (entry set identical ON vs OFF, only quantities differ). Env-gated from `run_pead_cpcv.py`: `PEAD_CONVICTION_SIZE=1` → `pead_conviction_size=True`. DEFAULT OFF → committed equal-weight +0.546 config byte-identical (regression-locked by `test_flag_off_unchanged`/`test_flag_off_no_surprise_fetch`; verified OFF trades == no-param-default trades). Tests: `tests/test_pead_conviction_sizing.py` (11) — flag-off-unchanged, sue-zscore-pit (future surprise cannot move today's z), vol-scale-pit (future/entry-day bar cannot move vol), gross-normalized (Σ == n×5%×equity), single-name==5%, higher-SUE-bigger-weight (vol held equal), clip(0,3)-bounds-tilt, insufficient-history neutral z, record-keyed-by-day. All synthetic, no network. | `app/backtesting/agent_simulator.py`, `scripts/run_pead_cpcv.py`, `tests/test_pead_conviction_sizing.py` |
| 2026-06-01 | #TBD | **PEAD earnings-quality-split lever (default OFF):** added an OPTIONAL earnings-quality gate to `PEADScorer` — the last high-EV lever to push PEAD's honest CPCV mean Sharpe (~0.55) toward the 0.80 gate. New constructor params `require_positive_revision: bool = False` and `min_analyst_momentum: float = 0.0`. When ON, a LONG signal fires only if `surprise >= long_threshold` AND the analyst-revision momentum as-of the scoring day is strictly `> min_analyst_momentum` (i.e. "EPS beat + analysts revising up" = higher-conviction drift, fewer/better trades). Short leg is unaffected. PIT-SAFE: the momentum is fetched via `get_analyst_features_at(sym, as_of)` where `as_of = scoring_day.date()` — that function windows grade records to `cutoff <= date <= as_of` (≤ scoring day only, never future). PERFORMANCE: `get_analyst_features_at` is NOT a per-(symbol,day) API call — it delegates to `get_analyst_grades_fmp(symbol)`, which fetches the symbol's FULL grade history ONCE and caches it in-process (`_grades_cache`, 24h TTL), then slices as-of in memory. So the gate adds ~1 one-time API call per Russell-1000 symbol (same pattern the existing earnings path already uses), NOT thousands of per-day calls. Env-gated from `run_pead_cpcv.py`: `PEAD_QUALITY_GATE=1` → `require_positive_revision=True`. DEFAULT OFF → committed long-only +0.546 config is byte-for-byte unchanged (regression-locked by `test_quality_gate_off_unchanged`, which also asserts the analyst provider is never called when OFF). Tests: `tests/test_pead_quality_gate.py` (10) — gate-off-unchanged, off-doesn't-call-analyst, filters-negative-revision (long suppressed ON / fires OFF), passes-positive-revision, zero-momentum-filtered (strict `>`), custom-min-momentum, short-leg-unaffected, PIT-safe (as_of == scoring day, never future), as_of-is-date-type. All stubbed/monkeypatched, no network. | `app/ml/pead_scorer.py`, `scripts/run_pead_cpcv.py`, `tests/test_pead_quality_gate.py` |
| 2026-06-01 | #TBD | **Fix PEAD CPCV instrumentation — DSR / regime-gate / profit-factor were broken, blocking honest promotion:** the first honest PEAD CPCV (+0.525 Sharpe, t=2.02) failed DSR (p=0.078) and `worst_regime_sharpe` (None) due to INSTRUMENTATION GAPS in `PEADStrategy` (`run_pead_cpcv.py`), not strategy weakness. THREE bugs, all fixed by mirroring `strategies/swing.py` exactly: (1) **`n_obs` never set** — `run_fold` built `FoldResult` without `n_obs`, so `cpcv.py` read `getattr(fold,"n_obs",0)=0` → `total_obs=0` → DSR fell back to ≈path-count (~10) instead of the true ~250 OOS trading days; at n_obs=10 the DSR p-value is unreachable at any plausible Sharpe (verified: Sharpe 0.525, N_trials 300, n_obs 10 → p=0.078 exactly). FIX: `n_obs = max(len(equity_curve)-1, 0)` (one daily-return obs per trading day). (2) **`regime_sharpes` never set** — `fetch_data` never built `self._global_regime_map` (only downloaded VIX for the scorer), so `run_fold` could not compute regime Sharpes → `worst_regime_sharpe=None` → regime gate failed CLOSED. FIX: `fetch_data` now builds the global regime map via `load_regime_map(start,end)` (mirrors swing.py:174-179), and `run_fold` calls `compute_regime_sharpes(equity_curve, te_start, te_end, regime_map=self._global_regime_map)`. (3) **`trade_returns` always `[]`** — `getattr(result,"trade_returns",[])` read a non-existent `SimResult` attr (SimResult has `profit_factor`+`trades`, NOT `trade_returns`) → `compute_profit_factor([])=0.0`, reporting PEAD's PF as 0 not its real value. FIX: `profit_factor=getattr(result,"profit_factor", compute_profit_factor([t.pnl_pct for t in trades_list]))` (AgentSimulator already computes PF with `_PF_NO_LOSS_SENTINEL`). HONEST NOTE: PEAD is event-driven (flat most days); `compute_regime_sharpes` drops regimes with `<REGIME_MIN_OBS=20` same-regime trading days, so `regime_sharpes` may still be sparse/None for PEAD even after the fix — that is a real data-sufficiency limit, not a bug. The fix ensures it populates WHEN data is sufficient and no longer FALSELY reports None due to a missing map. After merge, the definitive PEAD CPCV re-run will have a REAL DSR (should pass at Sharpe>~0.5 with ~250 n_obs) and a real (or honestly-None-due-to-sparsity) regime number. Tests: `tests/test_pead_instrumentation.py` (4) — n_obs==249 on a 250-pt curve, profit_factor reflects real `result.profit_factor` (1.4, not 0), `fetch_data` populates `_global_regime_map`, multi-regime (≥20 obs/regime) curve yields non-empty `regime_sharpes`. All stubbed/monkeypatched, no network. | `run_pead_cpcv.py`, `tests/test_pead_instrumentation.py` |
| 2026-06-01 | #TBD | **Fix #343 datetime-vs-date crash + add `aggregate_5min` daily source (new default):** PR #343 switched the intraday DAILY-bar fetch to yfinance, but on the live run the daily fetch returned **0/703 symbols** with `WARNING Daily bar fetch failed: can't compare datetime.datetime to datetime.date` — so EVERY symbol fell back to 0.5-default 52w/vol features (worse than the ~100-bar Alpaca cap it replaced). ROOT CAUSE: `_fetch_daily_all` passed `start`/`end` as `datetime` to the yfinance provider, whose cache (`cache.missing_daily_range`/`get_daily` in `app/data/cache.py`) compares the requested range against `datetime.date` (from `DatetimeIndex.date`); `date < datetime` raises `TypeError`, which the bare `except Exception` swallowed → `{}`. The Alpaca provider tolerated datetime; yfinance/its cache does not. FIX: (1) `_fetch_daily_all` coerces `daily_start`/`end` to `datetime.date` via a module-level `_to_date()` before the provider call (365d buffer preserved); (2) the bare except now logs exception TYPE+message with `exc_info` (no more silent swallow); (3) new daily source `INTRADAY_DAILY_FEATURE_PROVIDER="aggregate_5min"` (now the DEFAULT) derives daily OHLCV from the in-memory 5-min bars (`aggregate_5min_to_daily`: open=first/high=max/low=min/close=last/volume=sum) — zero network, no rate-limit, daily coverage == 5-min span (cannot empty the per-fold matrix). Per-fold path (`IntradayStrategy._ensure_daily_data`) aggregates `self.symbols_data` directly, bypassing the provider; production `train_model` passes `symbols_data` into `_fetch_daily_all`. TRADEOFF: 5-min cache ~2yr, so windows near the cache start get <252d backward lookback for 52w/vol features (degrades to a shorter window, NOT to 0.5 defaults). (4) Loud `DAILY COVERAGE LOW` WARN when <50% of symbols return daily bars (all paths). `yfinance`/`alpaca`/`polygon` remain selectable. Tests: `tests/test_intraday_daily_provider.py` extended to 11 — datetime→date coercion regression lock, aggregate OHLCV correctness, span==5-min coverage, VIX/malformed exclusion, in-memory aggregate (no network), error-logged-not-silent. | `intraday_training.py`, `retrain_config.py`, `strategies/intraday.py`, `tests/test_intraday_daily_provider.py` |
| 2026-05-31 | #343 | **Fix intraday DAILY-feature provider — Alpaca ~100-bar cap silently degraded 52w/vol features:** the intraday model's per-symbol DAILY bars (feeding `daily_vol_percentile` / 52-week-position features) were fetched via `IntradayModelTrainer._fetch_daily_all` using `self._provider`, which on this deployment is the Alpaca provider (trainer is constructed with `provider="alpaca"` both in the default `__init__` and in `IntradayStrategy._ensure_daily_data`). Alpaca returns only ~100 recent daily bars regardless of the requested start date, so across most of the 2yr training window the daily-dependent features silently fell back to their 0.5/1.0/0.0 defaults (`intraday_features.py:368`). The 5-min bars (Polygon parquet cache) were always fine — only the DAILY bars were degraded. FIX: `_fetch_daily_all` now sources daily bars from `get_provider(INTRADAY_DAILY_FEATURE_PROVIDER)` (new flag, default `"yfinance"` — full multi-year cached history; `"alpaca"` restores legacy, `"polygon"` uses the Polygon daily cache+S3), independent of the 5-min `self._provider`. Added a shallow-coverage diagnostic: if median bars/symbol < 50% of the trading-day fraction of the requested span, a WARNING fires (surfaces any future provider regression). The fix flows through BOTH the production `train_model` path and the per-fold `_ensure_daily_data` path (both call `_fetch_daily_all`). Tests: `tests/test_intraday_daily_provider.py` (5) — provider selection, 1yr buffer, shallow-coverage warn, no-warn-on-full, config override. | `intraday_training.py`, `retrain_config.py`, `strategies/intraday.py`, `tests/test_intraday_daily_provider.py` |
| 2026-05-31 | #C14-1 | **Fix intraday per-fold empty-matrix bug — data-span mismatch (2nd per-fold empty-matrix bug):** the intraday per-fold CPCV run produced 0 paths; every fold raised "per-fold-retrain: no training samples in window". ROOT CAUSE: the fold boundary day-axis comes from `IntradayStrategy.all_days_sorted` (`engine._build_trading_day_folds` → `all_boundaries[ti][0] = all_days_sorted[0]`), but the per-fold TRAIN matrix derives `train_days` from `symbols_data` (`build_train_matrix_for_window`). The two were never verified to agree; when `all_days_sorted` spanned earlier than the loaded 5-min bars, the earliest folds got `train_start` before any bar existed → `train_days = {}` → `len(train_days) < MIN_DAYS` → empty matrix. The C11-10 coverage check only ran for swing (`total_years is not None`); the intraday `total_days` path had no guard, and `build_train_matrix_for_window` emitted a generic empty matrix rather than naming the span mismatch. FIX (invariant: for every fold window, the 5-min day-axis the folds are built from never precedes the loaded bars, and daily bars cover `[first_fold_start − 1yr, last_day]`): (1) `fetch_data` now defensively CLAMPS `all_days_sorted` AND the per-symbol 5-min bars to the requested `[start, end]` (independent of `load_many`), keeping the fold day-axis and matrix-builder day-axis in lock-step (`^VIX` daily overlay left unclamped to preserve its 1yr warm-up); (2) `run_cpcv` gains an intraday (`total_years is None`) coverage guard mirroring C11-10 — raises `DataSpanError` (when `ENFORCE_MIN_DATA_SPAN`) with concrete span numbers when fold train-start precedes the 5-min bars; (3) `build_train_matrix_for_window` raises a clear span-mismatch `RuntimeError` when the window does not overlap any loaded day; (4) `_ensure_daily_data` documents/relies on the `_fetch_daily_all` −365d buffer so 52w/vol features get ~1yr daily warm-up before the first (now-clamped) fold start. Test gap closed: synthetic tests couldn't catch real-data span/coverage desync — added `test_build_train_matrix_span_mismatch_raises`, `test_fetch_data_clamps_all_days_to_window`, `test_intraday_coverage_mismatch_raises`, `test_intraday_coverage_ok_no_raise`. Frozen-mode behavior unchanged. | `strategies/intraday.py`, `cpcv.py`, `intraday_training.py`, `tests/test_per_fold_retrain_intraday.py`, `tests/test_data_span_gate.py` |
| 2026-05-31 | #PERFOLD3 | **Phase 2 — intraday per-fold retraining (true out-of-sample WF/CPCV) (KL-10b):** `--per-fold-retrain` now applies to INTRADAY too. `IntradayModelTrainer.build_train_matrix_for_window` builds a TRAIN-ONLY matrix from in-memory 5-min + per-symbol daily bars with `train_days = {d: train_start<=d<=train_end}`, no internal split; `_apply_labels` extracted from `_build_matrix_parallel` into a method so the per-fold and production paths share the exact labeling rules (realized-R / cross-sectional top-20% + cs_normalize); `_build_matrix_parallel` takes optional `train_days`/`test_days`. `fit_in_memory` reproduces the production XGBoost 3-seed ensemble (FROZEN_HPO_PARAMS, recency-decay weights, scale_pos_weight) with a per-window seed offset; LightGBM omitted intentionally (inference only blends `ensemble_models`). `IntradayFoldRetrainer` + reused `TrainWindowCache`; `IntradayStrategy.run_fold` uses the fold model + per-fold OOS guard with **TRADING-day purge** (`trading_day_set` passed, unlike swing's calendar purge); per-symbol daily bars fetched ONCE via `_ensure_daily_data` and cached (5-min never re-fetched). RISK #1 CONFIRMED leak-free: intraday label is same-day (`future_bars` within `day_bars`), all daily/prior-day lookbacks strictly backward → no FORWARD_DAYS purge needed. Feasibility: `--intraday-top-n` (default 150) forces a reduced liquidity universe; WARN when `cpcv_k>4`; legacy frozen intraday WF skipped when `--per-fold-retrain`+`--cpcv`. New flag `PER_FOLD_INTRADAY_HPO_TRIALS=0`. Frozen-mode behavior unchanged. Tests: `tests/test_per_fold_retrain_intraday.py` (8) incl. non-empty-matrix regression lock + no-leak (`max day_ordinal <= train_end`) on real synthetic 5-min/daily data. | `intraday_training.py`, `retrain_config.py`, `retrainers.py`, `strategies/intraday.py`, `walkforward_tier3.py`, `tests/test_per_fold_retrain_intraday.py` |
| 2026-05-31 | #PERFOLD2 | **Fix per-fold empty-matrix bug + horizon parity + CPCV display gate (KL-10b follow-up):** `build_train_matrix_for_window` returned an EMPTY X for every fold (CPCV reported mean Sharpe 0, n_paths=0). ROOT CAUSE: `SwingStrategy._global_regime_map` (from `load_regime_map`) is a `{date: str}` regime *label* map ("BULL"/"BEAR"/"NEUTRAL"); it was forwarded as `regime_score_map`, so the worker did `float("BULL")` → `ValueError`, swallowed by its bare `except Exception: continue` → every window dropped. FIX: the builder now validates the regime map is numeric and, if not, rebuilds the PIT composite-score map via `build_regime_score_map()` (the same helper the normal `_build_rolling_matrix` path uses). Also: per-fold path now sets `FORWARD_DAYS/STEP_DAYS/EMBARGO_WINDOWS` to `LABEL_HORIZON_DAYS` (20d) for label-scheme parity with the production swing model — `train_model()` does this but the per-fold path bypasses it; save/restore in a `try/finally` so no cross-contamination. Also: `_run_cpcv_swing`'s result was ignored, so a failed/zero-path CPCV still printed "ALL GATES PASSED"; new `_cpcv_swing_gate_ok()` propagates the CPCV gate (and zero-paths) into the overall `passed` flag. Test gap closed: the original no-leak test mocked `_windows_to_matrix` so an empty X passed vacuously — new `test_build_train_matrix_is_non_empty` runs the real spine end-to-end with the string-label map and asserts `len(X)>0`, consistent `y`/`meta`/`fnames`, finite values; `test_build_train_matrix_uses_production_horizon` and `test_cpcv_swing_gate_ok_reflects_failure` lock the other two fixes. | `training.py`, `walkforward_tier3.py`, `tests/test_per_fold_retrain_swing.py` |
| 2026-05-31 | #PERFOLD1 | **Phase 1 — swing per-fold retraining (true out-of-sample WF/CPCV) (KL-10b):** `--per-fold-retrain` (swing only) trains a fresh model inside each fold on only that fold's `[tr_start, tr_end]` window (in-memory, no re-fetch); `ModelTrainer.build_train_matrix_for_window` (date-spine clamped to `train_end` → no label leak, proven by test) + `fit_in_memory`; `SwingFoldRetrainer` + `TrainWindowCache` (dedup CPCV combos); `SwingStrategy.run_fold` uses the fold model + per-fold OOS guard when `per_fold_retrain=True`, else frozen `self.model`; `is_true_walkforward` set by engine/cpcv; `REQUIRE_TRUE_WF_FOR_PROMOTION` (default False) blocks frozen-mode promotion in both `WalkForwardReport` and `CPCVResult`; new flags `PER_FOLD_RETRAIN`/`REQUIRE_TRUE_WF_FOR_PROMOTION`/`PER_FOLD_SWING_HPO_TRIALS`. Frozen-mode behavior unchanged by default. | `training.py`, `retrain_config.py`, `retrainers.py`, `strategies/swing.py`, `engine.py`, `cpcv.py`, `gates.py`, `walkforward_tier3.py`, `tests/test_per_fold_retrain_swing.py` |
| 2026-05-31 | #KL10 | **Save-guard — refuse to persist a model without `trained_through` (KL-10):** `_assert_trained_through()` at the top of every trained-model `save()` raises `ValueError` when the cutoff is None (flag `REQUIRE_TRAINED_THROUGH`, default True); `trained_through` attribute added to `LambdaRankModel`/`TwoStageModel`/`ThreeStageModel`/`DoubleEnsembleModel` (was missing — the v224 bug); `_load_model` logs a clear stale-artifact error; `trained_through` documented as artifact-sourced, never DB-sourced. KL-10 RESOLVED. | `model.py`, `retrain_config.py`, `walkforward_tier3.py`, `tests/test_trained_through_guard.py` |
| 2026-05-31 | #PHASE3 | **Phase 3 — CPCV path t-stat + StrategySimulator tier-2 (HIGH-3):** added `CPCVResult.path_sharpe_tstat` (N_eff=n_folds, NOT n_paths) reported + warned, gateable via `require_tstat_gate` (off by default, `CPCV_MIN_TSTAT=2.0`); StrategySimulator gains a TIER-2-ONLY banner + `build_daily_equity_curve` opt-in flag (forward-fills entry-date equity onto daily calendar). KL-4 partially addressed, KL-8 addressed. | `cpcv.py`, `strategy_simulator.py`, `retrain_config.py`, `tests/test_cpcv_tstat.py`, `tests/test_strategy_sim_daily_curve.py` |
| 2026-05-31 | #PHASE2 | **Phase 2 — Regime gate overhaul (HIGH-2 + MEDIUM-4):** coarse3 BULL/BEAR/NEUTRAL labeler with expanding-quantile VIX thresholds (PIT-correct, no look-ahead); `REGIME_MIN_OBS=20` bucket floor; `worst_regime_sharpe is None` now HARD-FAILS `gate_passed()` (WF + CPCV) unless `ALLOW_NO_REGIME_GATE=True`. KL-2 + KL-7 RESOLVED. | `regime.py`, `gates.py`, `cpcv.py`, `reports.py`, `retrain_config.py`, `tests/test_regime_coarse3.py` |
| 2026-05-31 | — | Document created; reflects post-13-round-audit state | This file |
| 2026-05-30 | #327 | OOS same-day overlap guard; gate_detail n_paths consistency | `cpcv.py`, `oos_guard.py` |
| 2026-05-30 | #326 | CPCV embargo guard; intraday OOS trading-day purge; regime-gate warning | `cpcv.py`, `engine.py`, `gates.py` |
| 2026-05-29 | #325 | CPCV min-paths floor; stable regime labels; data coverage guard | `cpcv.py`, `regime.py` |
| 2026-05-29 | #324 | Swing simulator Sharpe/PF parity with intraday | `strategy_simulator.py`, `strategies/swing.py` |
| 2026-05-29 | #323 | Regime gate active in production; intraday Sharpe/PF fixes | `gates.py`, `strategies/intraday.py` |
| 2026-05-30 | #314 | Gate metric bugs: Calmar geometric, k-ratio log-equity, PF cap, retrain purge_days | `gates.py`, `retrain_config.py` |
