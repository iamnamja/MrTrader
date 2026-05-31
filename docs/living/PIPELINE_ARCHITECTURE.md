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
| Worst regime Sharpe | `worst_regime_sharpe` | ≥ -0.50 | ⚠️ **SILENTLY INACTIVE** | Returns `None` when regime_sharpes unpopulated or insufficient obs; `None` silently passes (see Known Limitation #2) |

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
| Worst regime Sharpe | as WF | ≥ -0.50 | ⚠️ **SILENTLY INACTIVE** | `None` silently passes |
| Min active paths | `len(path_sharpes) >= 2` | | ✅ **ACTIVE** | Must have ≥ 2 paths before any gate fires |

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

### KL-2 — Regime gate silently passes (HIGH)
**Description:** `worst_regime_sharpe is None` → `regime_ok = True` in `gate_passed()`. The regime gate has never blocked any model promotion. Combined with 16-bucket label scheme producing too-sparse per-regime samples, the gate is effectively a no-op.

**Impact:** Models with poor performance in specific market regimes (bear markets, VIX spikes) can pass the gate while being regime-sensitive.

**Fix status:** Planned Phase 2. Will coarsen to 3 buckets, set `ALLOW_NO_REGIME_GATE=False`, make None a blocking failure.

### KL-3 — Calmar gate trivially passes via no-DD sentinel (MEDIUM)
**Description:** No-DD profitable folds receive `CAL_NO_DD_SENTINEL=+5.0`. Intraday tight-stop strategies with multiple no-DD folds trivially clear `avg_calmar ≥ 0.30`.

**Impact:** Calmar gate provides no discrimination for strategies with tight ATR stops.

**Fix status:** Planned Phase 1. Will use vol-floor drawdown estimate when max_drawdown == 0 (`USE_CALMAR_VOL_FLOOR=True`).

### KL-4 — CPCV path correlation overstates independence (HIGH)
**Description:** 15 paths from C(6,2) use the same 6 folds; each trading day appears in 5 paths. `pct_positive` and `p5_sharpe` treat them as 15 independent draws. Effective N ≈ 6 folds.

**Impact:** "92.9% positive" overstates statistical confidence. Actual effective N for significance is ~6, not 15.

**Fix status:** Planned Phase 3. Will add `path_sharpe_tstat = mean/(std/√n_folds)` to CPCVResult and optional `require_tstat_gate`.

### KL-5 — No minimum data span gate (MEDIUM)
**Description:** If Polygon cache is empty, intraday silently falls back to 55-day yfinance window. CPCV on 55 days = 9-day test folds. Nothing blocks promotion on this degenerate case. No data provenance info in report output.

**Impact:** A degenerate run produces results that look identical to a properly-windowed run in the gate report.

**Fix status:** Planned Phase 1. Will raise `DataSpanError` when `len(all_days_sorted) < MIN_DATA_SPAN_TRADING_DAYS=250`.

### KL-6 — Intraday Sharpe inflated by low capital deployment (CRITICAL)
**Description:** 3% sizing × max 5 positions = ~15% of equity ever at risk. Many flat no-trade days (VIX gate, abstention filters) suppress daily return std further. Daily IR = Sharpe/√252 = 0.32 for Sharpe 5.14 — empirically implausible for a retail equity strategy. A "Sharpe 5" at 15% deployment would be roughly "Sharpe 0.75" fully invested.

**Impact:** Headline Sharpe is not comparable to any benchmark or other strategy. Gates cannot distinguish capital-deployment artifact from real edge.

**Fix status:** Planned Phase 1. Will add `avg_capital_deployed_pct` and `deployment_adjusted_sharpe` to SimResult/FoldResult, with `LOW DEPLOYMENT WARNING` in reports.

### KL-7 — VIX quantile thresholds use look-ahead (MEDIUM)
**Description:** `pd.qcut(vix, q=4)` in `regime.py` computes quartile boundaries over the full evaluation window including future data.

**Impact:** Regime labels for early dates are influenced by VIX values that occur after them. Mild look-ahead in regime classification.

**Fix status:** Resolved by Phase 2 coarsening (expanding quantile for VIX thresholds in BULL/BEAR/NEUTRAL scheme).

### KL-8 — StrategySimulator equity curve entry-date only (LOW, tier-2 only)
**Description:** `strategy_simulator.py:StrategySimulator.run()` records equity only on trade-entry dates, not daily. Sharpe annualization is incorrect for sparse-entry strategies.

**Impact:** Tier-2 standalone tool only — NOT used in WF/CPCV. Does not affect any gate or promotion decision.

**Fix status:** Planned Phase 3 (low priority). Will add `build_daily_equity_curve` opt-in flag and prominent tier-2 banner.

### KL-9 — Polygon cache may lack delisted names (MEDIUM)
**Description:** If the Polygon 5-min cache only holds currently-listed Russell 1000 names, intraday WF results have survivorship bias regardless of the `members_at(te_start)` PIT logic. The PIT logic is only as good as the cache's historical completeness.

**Impact:** Intraday WF universe may silently exclude names that delisted during the test window, slightly inflating performance.

**Fix status:** No code fix — requires data infrastructure verification. Document: run `scripts/audit_survivorship.py` before any intraday promotion decision to confirm cache completeness.

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
| `REGIME_SCHEME` | *planned* | `"coarse3"` vs `"legacy16"` regime labeler | Phase 2 planned |
| `ALLOW_NO_REGIME_GATE` | *planned* | False = None worst_regime_sharpe fails gate | Phase 2 planned |
| `ENFORCE_MIN_DATA_SPAN` | *planned* | Raise DataSpanError if data < 250 days | Phase 1 planned |

---

## 14. Changelog

All entries reference the PR that made the change.

| Date | PR | Change | Files |
|---|---|---|---|
| 2026-05-31 | — | Document created; reflects post-13-round-audit state | This file |
| 2026-05-30 | #327 | OOS same-day overlap guard; gate_detail n_paths consistency | `cpcv.py`, `oos_guard.py` |
| 2026-05-30 | #326 | CPCV embargo guard; intraday OOS trading-day purge; regime-gate warning | `cpcv.py`, `engine.py`, `gates.py` |
| 2026-05-29 | #325 | CPCV min-paths floor; stable regime labels; data coverage guard | `cpcv.py`, `regime.py` |
| 2026-05-29 | #324 | Swing simulator Sharpe/PF parity with intraday | `strategy_simulator.py`, `strategies/swing.py` |
| 2026-05-29 | #323 | Regime gate active in production; intraday Sharpe/PF fixes | `gates.py`, `strategies/intraday.py` |
| 2026-05-30 | #314 | Gate metric bugs: Calmar geometric, k-ratio log-equity, PF cap, retrain purge_days | `gates.py`, `retrain_config.py` |
