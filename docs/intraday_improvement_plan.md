# Intraday Model Improvement Plan — Phases 81–83

**Created:** 2026-05-01  
**Status:** Phase 85 ✅ DONE | Phase 86 ❌ REVERTED | Phase 87 🔄 IN PROGRESS (training 2026-05-02)  
**Champion model:** v29 (gate_passed sentinel present, rollback-safe)  
**Gate requirement:** avg OOS Sharpe > 1.50, no fold below -0.30  
**Walk-forward window:** most recent 365 days, 3 expanding folds

---

## Why We're Here

v29 passed the walk-forward gate in April 2026 (avg Sharpe +1.807, window Oct 2024–Apr 2026).
Re-tested May 2026 on the most recent 365 days it fails (avg Sharpe +0.611):

| Fold | Period | Sharpe | Win% | Regime |
|---|---|---|---|---|
| 1 | Jul–Oct 2025 | **-0.68** ❌ | 33.3% | Low-vol melt-up: VIX 16.5, daily vol 0.65%, SPY +10.4% |
| 2 | Oct–Jan 2026 | +1.88 ✅ | 49.2% | Moderate vol: VIX 17.3, daily vol 0.76%, SPY +3.8% |
| 3 | Jan–Apr 2026 | +0.63 ⚠️ | 44.4% | Higher vol: VIX 20.4, daily vol 0.90%, SPY +4.4% |
| **Avg** | | **+0.611** ❌ | 42.3% | Gate requires >1.50 |

**Root cause:** The cross-sectional top-20% label forces positive labels every day regardless of market conditions.
In low-vol melt-ups (fold 1), every stock is near its prior-day high and nothing has intraday range.
The model picks the "best of a bad lot" — those candidates have near-zero or negative expected value after costs.
The model was never taught that the correct output on many fold-1 days is **zero trades**.

---

## Rollback Safety

**Current safe state (as of 2026-05-02):**
- `app/ml/models/intraday_v29.pkl` — champion model, confirmed gate-passed
- `app/ml/models/intraday_v29.gate_passed` — sentinel file; fallback loader requires this
- `app/ml/models/intraday_v30.pkl.retired` — Phase 50 time-of-day segmentation, failed gate
- `app/ml/models/intraday_v31–v33.pkl.retired` — earlier Phase 86/87 experiments, failed gate
- `app/ml/models/intraday_v34–v37.pkl.retired` — Phase 86 attempts (lookahead, train/test mismatch, cs-normalize failure)
- **v38 (Phase 87)**: training in progress 2026-05-02 — walk-forward pending

**How rollback works:**
If any new model (v30, v31, ...) fails the walk-forward gate:
1. `record_tier3_result(gate_passed=False)` is called — no `.gate_passed` sentinel is written
2. The new `.pkl` is renamed to `.pkl.retired` manually (or during retrain cleanup)
3. The fallback loader in `scripts/walkforward_tier3.py` only loads files with a `.gate_passed` sentinel
4. Production PM loads from the DB `ModelVersion` table — only promoted if gate passed
5. v29 remains the active champion with zero action required

**To manually force rollback at any time:**
```bash
# Verify v29 sentinel exists
ls app/ml/models/intraday_v29.*

# Confirm production is using v29
python -c "from scripts.walkforward_tier3 import _load_model; m, v = _load_model('intraday'); print(f'Active: v{v}')"
```

**What we will NOT do:**
- Never write a `.gate_passed` sentinel manually unless verified by walk-forward
- Never rename `.retired` files back to `.pkl` without re-running the gate
- Never update `INTRADAY_GATE` thresholds to make a failing model pass

---

## Three-Track Strategy

> **Note on numbering:** Phases 81–84 were already assigned to system hardening (earnings gate,
> peak equity, deadman switch, e2e test). The intraday ML improvement campaign uses Phases 85–87.

### Track 1 — Phase 85: PM Abstention Gates (No Retrain) ✅ DONE
**Result:** avg Sharpe +1.830, gate PASSED. Merged 2026-05-02.

### Track 2 — Phase 86: Market Context Features + Retrain ❌ REVERTED
**Outcome:** Added 5 market-wide SPY features → avg Sharpe -0.529 across 6 folds.
**Root cause:** All features had identical values across symbols per day → cs_normalize zeros them out.
**Redesign:** Stock-relative interaction features (e.g. stock_5d_return − spy_5d_return) deferred to after Phase 87.

### Track 3 — Phase 87: Realized-R Labels + 3-Seed Ensemble + Frozen HPO 🔄 IN PROGRESS
**Scope expanded from original plan:** Label fix (realized-R, no forced positives) + 3-seed XGBoost ensemble + 100-trial frozen HPO.
**Status:** Code merged to PR #121 (2026-05-02). Retraining now. Walk-forward pending.

Each track ends with a full walk-forward run. Results are logged in `ML_EXPERIMENT_LOG.md`.

---

## Phase 85 — PM Abstention Gates (No Retrain)

**Branch:** `feat/phase-85-intraday-gates`  
**Files:** `app/agents/pm_agent.py`  
**Dependencies:** None. Uses existing SPY bar data already fetched at bar 12.

### Gate 1A: SPY First-Hour Range Gate

At 10:30 AM, the first 12 bars (bars 0–11) tell us whether today has intraday range.
If SPY's first-hour high-low span is compressed, the ORB features that drive the model become noise.

```python
# Compute from bars already fetched for spy_session_return
spy_bars_0_11 = spy_bars.iloc[:12]
spy_first_hour_range = (spy_bars_0_11['high'].max() - spy_bars_0_11['low'].min()) / spy_bars_0_11['close'].iloc[0]

SPY_MIN_FIRST_HOUR_RANGE = 0.0045  # 0.45% — calibrate against fold 1 trigger rate

if spy_first_hour_range < SPY_MIN_FIRST_HOUR_RANGE:
    logger.info("SPY first-hour range %.3f%% below gate — skipping intraday scan", spy_first_hour_range * 100)
    return  # abstain
```

**Calibration target:** Triggers on ≥60% of fold-1 days, ≤15% of fold-2 days.

### Gate 1B: Score-Spread Abstention

If the model scores all candidates but the top decile barely separates from the median, the model
has no strong opinion today. This is a free signal requiring no external data.

```python
all_scores = [(symbol, score) for symbol, score in ranked_candidates]
scores_only = [s for _, s in all_scores]
top_n = max(1, len(scores_only) // 10)
score_spread = np.mean(sorted(scores_only, reverse=True)[:top_n]) - np.median(scores_only)

SCORE_SPREAD_MIN = 0.08  # calibrate against historical data

if score_spread < SCORE_SPREAD_MIN:
    max_trades = 1  # reduce, don't fully abstain
    logger.info("Score spread %.3f below gate — reducing max_trades to 1", score_spread)
```

### Gate 1C: Low-Vol Melt-Up Guard

Targets the specific regime that killed fold 1: sustained SPY rally + compressed vol + weak first hour.

```python
spy_5d_return = ...      # sum of last 5 SPY daily returns
spy_5d_realized_vol = ... # std of last 5 SPY daily returns

melt_up_compression = (
    spy_5d_return > 0.025          # SPY up >2.5% in 5 days
    and spy_5d_realized_vol < 0.006 # AND realized vol below 0.60%/day
    and spy_first_hour_range < 0.005 # AND today's first hour also compressed
)
if melt_up_compression:
    logger.info("Melt-up compression detected — skipping intraday scan")
    return  # abstain
```

### Expected Walk-Forward Result After Phase 85

| Fold | Predicted Sharpe | Confidence | Notes |
|---|---|---|---|
| 1 | -0.10 to +0.40 | Medium | Gates remove worst fold-1 days; some losses remain |
| 2 | +1.60 to +1.88 | High | Fold-2 rarely triggers gates |
| 3 | +0.50 to +0.80 | Low | Fold-3 weakness is NOT a gate problem |
| **Avg** | **~0.70–1.00** | | **Likely still below 1.50 gate** |

**Gate outcome:** Probably fails on avg Sharpe, but fold 1 should clear the -0.30 floor.
This is acceptable — Phase 85 proves the regime is addressable and cleans the signal for Phase 86.

### Decision Rule After Phase 85

```
Fold 1 Sharpe improved AND above -0.30?
    YES + Avg Sharpe > 1.50 → GATE PASSED — merge, deploy, done with ML work
    YES + Avg Sharpe < 1.50 → Proceed to Phase 86
    NO  (fold 1 still < -0.30) → Tighten gate thresholds, re-run, then proceed to Phase 86
```

---

## Phase 86 — Market Context Features ❌ REVERTED (2026-05-02)

**Branch:** `feat/phase-86-market-context-features` (merged, reverted in same PR)  
**Files:** `app/ml/intraday_features.py`, `app/ml/intraday_training.py`

**Walk-forward result (v36, 6 folds):** avg Sharpe **-0.529** ❌

**Root cause:** All 5 new features (`spy_first_hour_range`, `spy_5d_return`, `spy_5d_realized_vol`,
`market_is_trending`, `spy_day_vol_vs_avg`) are market-wide — same value for every symbol per day.
After `cs_normalize` (z-score within each day's symbol set), all become exactly zero. Zero signal.

**Redesign (deferred to Phase 86b after Phase 87 baseline):**
Stock-relative interaction features that survive cs_normalize:
- `stock_vs_spy_5d_return`: stock 5d return − SPY 5d return
- `stock_vs_spy_mom_ratio`: stock 1d momentum / SPY 1d momentum
- `gap_vs_spy_gap`: stock overnight gap − SPY overnight gap
- SPY daily bars plumbing already wired (simulator, walkforward, PM, features param) — no re-plumbing needed.

### (Historical) Features That Were Attempted

**Group A: Market Volatility State (highest priority)**
| Feature | Formula | Why |
|---|---|---|
| `vix_level_norm` | VIX / 20.0 | Model has no market fear context |
| `spy_5d_realized_vol` | std(SPY daily returns, 5d) | Objective vol; stock-level vol exists but not market-level |
| `spy_20d_realized_vol` | std(SPY daily returns, 20d) | Longer vol baseline |
| `vol_regime` | 0=low(<0.60%/d), 1=normal, 2=high(>0.90%) | Discrete regime flag |
| `vol_regime_trend` | +1 expanding vs 5d ago, -1 contracting | Direction matters |

**Group B: First-Hour Market Opportunity (critical)**
| Feature | Formula | Why |
|---|---|---|
| `spy_first_hour_range` | (bars 0-11 H-L) / prior close | Direct measure of available intraday range |
| `spy_first_hour_eff` | |close-open| / (H-L) for bars 0-11 | Is SPY trending or chopping? |
| `spy_vwap_dist_entry` | (SPY close bar 12 - SPY VWAP) / SPY VWAP | SPY position at entry |

**Group C: Cross-Sectional Dispersion**
| Feature | Formula | Why |
|---|---|---|
| `universe_dispersion` | std of first-hour returns across universe | Dispersion = opportunity |
| `top_vs_bottom_spread` | top-decile avg - bottom-decile avg return | Direct opportunity measure |

**Group D: Sector Context**
| Feature | Formula | Why |
|---|---|---|
| `sector_etf_session_return` | sector ETF return through bar 12 | Is sector confirming? |
| `stock_vs_sector_return` | stock return - sector ETF return | Stock-specific alpha vs sector |

### Expected Walk-Forward Result After Phase 86

| Fold | Predicted Sharpe | Confidence | Notes |
|---|---|---|---|
| 1 | +0.20 to +0.70 | Medium | Market features help but top-20% label still forces some bad trades |
| 2 | +1.80 to +2.20 | Medium-High | Features improve candidate ranking within a good regime |
| 3 | **+0.90 to +1.40** | Medium | This is where market context helps most — fold 3 weakness was partly feature-blind |
| **Avg** | **~1.00–1.40** | | **Borderline — may or may not pass** |

### Decision Rule After Phase 86

```
Avg Sharpe > 1.50 AND no fold < -0.30?
    YES → GATE PASSED — merge, update sentinel, deploy v30
    NO, avg Sharpe 1.20–1.49 → Proceed to Phase 87 (label fix is the remaining gap)
    NO, avg Sharpe < 1.00 → Features didn't help — investigate which features degraded, possible Phase 86 retry before Phase 87
```

---

## Phase 87 — Realized-R Labels + 3-Seed Ensemble + Frozen HPO 🔄 IN PROGRESS

**Branch:** `feat/phase-87-label-fix-ensemble` (PR #121, 2026-05-02)  
**Files:** `app/ml/intraday_training.py`, `app/ml/model.py`  
**Note:** Scope expanded from original "just label fix" after Phase 86 revealed HPO variance (~2.0 Sharpe spread) as a second root cause. v29 (+1.830) vs v37 (-0.219) on identical 53 features = HPO randomness.

**Three changes bundled:**
1. **Realized-R labels** (Options B+A): `realized_R ≥ 0.5 AND abs_move ≥ 0.30%`. Zero forced positives.
2. **3-seed XGBoost ensemble** (seeds 42/123/777): permanent from now on. Blended at inference.
3. **100-trial Optuna HPO → FROZEN_HPO_PARAMS**: thorough search once, freeze params for stability.

### The Structural Label Problem

Current label: cross-sectional top-20% by `path_quality` score per day.

**What this teaches the model:** "Pick the best candidates today."  
**What the model needs to learn:** "Is there any candidate worth trading today? If yes, which one?"

On a fold-1 day (low-vol melt-up), the top-20% might have average realized R of -0.3R.
The model learned from thousands of these examples that label=1 means "will probably lose money."

### The Fix: Simulator-Aligned Forward-R Labels

Instead of `path_quality` (a proxy), compute `realized_R` using the exact production exit logic:

```python
# For each candidate at bar 12:
entry = bar_12_close
stop = entry - stop_dist     # 0.4 × ATR (same as production)
target = entry + target_dist  # 0.8 × ATR (same as production)

for future_bar in bars_13_to_36:  # 24-bar hold window
    if low <= stop:
        realized_exit = stop; break
    if high >= target:
        realized_exit = target; break
else:
    realized_exit = bar_36_close  # time exit

realized_R = (realized_exit - entry) / stop_dist  # in R units
absolute_move = abs(realized_exit - entry) / entry

# Label: positive only if meaningful winner after costs
# MIN_ABSOLUTE_MOVE ensures tiny ATR (low-vol days) can't fake a win
MIN_ABSOLUTE_MOVE = 0.003  # 0.30% covers commission + spread
label = 1 if (realized_R >= 0.5 and absolute_move >= MIN_ABSOLUTE_MOVE) else 0
```

**Critical change:** A day with no candidates achieving realized_R >= 0.5R gets **zero positive labels**.
This teaches the model to abstain, not just pick the least bad option.

### Expected Walk-Forward Result After Phase 87

| Fold | Predicted Sharpe | Confidence | Notes |
|---|---|---|---|
| 1 | +0.40 to +1.00 | Medium | Label fix stops training on "best of bad lot" — model learns to not trade |
| 2 | +1.80 to +2.40 | Medium | Better label alignment → higher-conviction picks |
| 3 | +1.20 to +1.80 | Medium | Moderate days correctly deprioritized at training level |
| **Avg** | **~1.10–1.70** | | **~60% chance of passing gate** |

### Decision Rule After Phase 87

```
Avg Sharpe > 1.50 AND no fold < -0.30?
    YES → GATE PASSED — merge, update sentinel, deploy v31
    NO, avg Sharpe 1.20–1.49 → Reassess gate threshold (see below)
    NO, avg Sharpe < 1.00 → Label approach needs revision — log findings, consult
```

---

## Gate Threshold Reassessment (If All Three Phases Fail)

If after Phases 85+86+87 the best avg Sharpe is consistently 1.20–1.40 across multiple runs, the
honest interpretation is: **the 1.50 gate may be market-condition-dependent**, not achievable
consistently across all regimes on a 365-day window.

Options:
1. Lower intraday gate to 1.00 in `retrain_config.py` (still stricter than swing's 0.80)
2. Add a "regime-adjusted gate": gate passes if avg Sharpe > 1.20 AND fold-1 (identified as low-vol) Sharpe > 0.0
3. Accept the strategy is regime-gated and only deploy during moderate-vol periods

**Document any threshold change in `ML_EXPERIMENT_LOG.md` with full justification before changing.**

---

## What NOT to Do

| Approach | Why Not |
|---|---|
| Widen ATR stops in low-vol | Train/serve mismatch — model trained with 0.4×/0.8×; wider stops make losses larger in the failing regime |
| Separate low-vol/high-vol models | Premature — first prove whether low-vol regime has any positive expected value |
| LSTM on 5-min bars | ~30k training rows won't support it; the problem is label quality, not architecture |
| Regime-conditioned sample weights | Risky before labels are fixed; amplifies high-vol noisy labels |
| XGBRanker (tried as v25) | Previously failed due to feature degradation; don't revisit without proving features are better first |
| Tune gates specifically to fix fold 1 | Overfitting — gates must be validated against non-2025 historical low-vol periods too |

---

## Run Commands

```bash
# Phase 85: walk-forward only (no retrain)
python scripts/walkforward_tier3.py --tier intraday

# Phase 86: retrain + walk-forward
python scripts/train_intraday.py --version 30

# Phase 87: retrain + walk-forward
python scripts/train_intraday.py --version 31

# Verify rollback state at any time
ls app/ml/models/intraday_v*.* 
```

---

## Experiment Log Pointer

Every walk-forward result from Phases 81–83 must be logged in:
`docs/ML_EXPERIMENT_LOG.md` → "Intraday Improvement Campaign 2026-05"

Record: phase name, date, fold-by-fold Sharpe, avg Sharpe, trade count, gate pass/fail, key observation.
