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

## 2. Three Simulators

**This is the most common source of confusion. Read this before touching any backtest code.**

| Simulator | File | Used by | Equity curve | Status |
|---|---|---|---|---|
| `AgentSimulator` | `app/backtesting/agent_simulator.py` | **Swing WF + CPCV** (via `swing.py:run_fold`) | **DAILY MTM** — one equity point per calendar trading day; uses `portfolio.equity_mtm(today_closes)` | Production |
| `IntradayAgentSimulator` | `app/backtesting/intraday_agent_simulator.py` | **Intraday WF + CPCV** (via `intraday.py:run_fold`) | **DAILY** — one equity point per trading day | Production |
| `StrategySimulator` | `app/backtesting/strategy_simulator.py` | **Tier-2 standalone tool ONLY** | **ENTRY-DATE only** — one equity point per day that had a new trade entry (NOT daily MTM) | Tier-2 only; **never used in WF/CPCV** |

**Why this matters for Sharpe/n_obs:**
- `AgentSimulator` and `IntradayAgentSimulator` produce correct daily-return series → `sqrt(252)` annualization is valid.
- `StrategySimulator` produces entry-date-sampled returns → applying `sqrt(252)` is incorrect. It is only used for quick standalone sanity checks, not promotion decisions.

**SimResult** is defined in `strategy_simulator.py` and imported by all three simulators. All three populate it; the fields are the same.

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

### 7a. WalkForwardReport.gate_passed() — all must pass

| Gate | Metric | Threshold | Status | Notes |
|---|---|---|---|---|
| Avg Sharpe | `avg_sharpe` (n_obs-weighted) | ≥ 0.80 | ✅ **ACTIVE** | |
| Min fold Sharpe | `min_sharpe` | ≥ -0.30 | ✅ **ACTIVE** | |
| DSR | `deflated_sharpe_ratio(avg_sharpe, N_TRIALS_TESTED=250, total_obs)` | p > 0.95 | ⚠️ **NON-BINDING** above Sharpe ~2 | p = 1.0 for any Sharpe > 2; provides zero discrimination (see Known Limitation #1) |
| Profit factor | `avg_profit_factor` (capped at 5.0) | ≥ 1.10 | ✅ **ACTIVE** | Waived if < 2 folds have PF > 0 |
| Calmar | `avg_calmar` (geometric CAGR / max DD) | ≥ 0.30 | ⚠️ **WEAK** | No-DD folds get CAL_NO_DD_SENTINEL=+5.0 which trivially passes (see Known Limitation #3) |
| Worst regime Sharpe | `worst_regime_sharpe` | ≥ -0.50 | ✅ **ACTIVE when regime_map populated; HARD FAIL when None** (Phase 2) | `None` now fails the gate unless `ALLOW_NO_REGIME_GATE=True`. coarse3 scheme (BULL/BEAR/NEUTRAL, expanding-quantile VIX) gives enough obs/bucket; REGIME_MIN_OBS=20 floor (see Known Limitation #2 RESOLVED) |

**Paper-trade variant:** `gate_passed(paper_gate=True)` uses relaxed thresholds (Sharpe ≥ 0.50, min fold ≥ -0.40) and waives PF + Calmar gates entirely.

### 7b. CPCVResult.gate_passed() — all must pass

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

---

## 13. Feature Flags

All live in `app/ml/retrain_config.py`.

| Flag | Default | Effect | Since |
|---|---|---|---|
| `RETRAIN_WEEKDAY` | -1 (disabled) | Day of week for scheduled retrains; -1 = off | Phase C |
| `USE_NIS_FEATURES` | False | Include NIS macro LLM sentiment in swing training | Phase A |
| `USE_FMP_FUNDAMENTALS` | True | Load PIT FMP fundamentals (overrides EDGAR) | Phase 93 |
| `USE_REALIZED_R_LABELS` | False | Intraday: absolute R threshold labels vs cross-sectional | Phase 88 |
| `BENIGN_FILTER_ENABLED` | False | Filter non-bullish rows from training | Phase P1 |
| `INTRADAY_ENABLED` | False | Enable scheduled intraday retrain | Phase C |
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
