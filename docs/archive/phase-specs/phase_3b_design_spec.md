# Phase 3b Design Spec: Swing Pre-Filter Removal

**Status:** Planning  
**Priority:** HIGH — required before swing can pass walk-forward gate  
**Estimated effort:** 2–3 weeks  
**Depends on:** Phase 1+2 complete (done), Phase 3a complete (done)

---

## Problem Statement

The swing model only scores candidates that pass RSI_DIP or EMA_CROSSOVER pre-filters. This means:
- The model learned "which RSI dips don't fail" — not "what makes a good swing trade"
- Entry candidates are pre-selected by hard rules written in 2024, before the tariff regime
- In the 2025 tariff regime, RSI_DIP catches falling knives: stocks dip due to macro, not mean-reversion setup
- Bootstrap walk-forward (2026-05-07): fold 3 (Feb 2025 → May 2026) Sharpe -1.08 in 20/20 runs

**Naive removal attempted (2026-05-06, ML_EXPERIMENT_LOG Phase 3a):** FAILED.
Avg Sharpe went from +0.422 to -0.731. Fold 3 collapsed to 3.9% win rate / -2.54 Sharpe.
Root cause: the pre-filters were acting as the only regime guards. Removing them without
replacing with ML-based regime awareness exposes the model to everything including noise.

**The correct fix** is not just removing the filters — it's giving the model full universe access
plus regime-aware training data so it learns what to enter without hard rules.

---

## Design: Phased Approach (3 Steps Must Ship Together)

### Step 1 — Full Universe Scoring (replace hard gate with features)

**Current:** `app/agents/portfolio_manager.py` calls `scan_universe()` which applies
RSI_DIP / EMA_CROSSOVER filter before sending to ML model. Only ~5-15 symbols/day pass.

**New:** Scan all 430 S&P 500 symbols daily. Compute RSI and EMA crossover as numeric
features (not as binary gates). Pass all symbols to the ML model and let it rank them.

**Key changes:**
- `app/agents/portfolio_manager.py`: remove `filter_by_prefilter()` call in swing scan path
- `app/ml/features.py`: add `rsi_dip_score` = distance below 30 RSI (0 if RSI >= 30), `ema_cross_distance` = (EMA9 - EMA20) / price
- Keep existing binary `rsi_dip` and `ema_cross` features — they become signals, not gates
- Add `is_rsi_dip` (bool) and `is_ema_cross` (bool) as model features for continuity

**Why this works:** XGBoost can learn "RSI dip in low-VIX regime = good entry; RSI dip in high-VIX tariff regime = avoid." The hard filter cannot learn this distinction.

### Step 2 — Triple-Barrier Label (replace binary label)

**Current:** `path_quality` label = 1 if stock hits +2% target within 5 days, 0 otherwise.
This is a single-sided label — it doesn't distinguish between "missed target" and "hit stop."

**New:** Triple-barrier label over 10 calendar days:
- **Upper barrier:** +2.0 × ATR from prior close (win)
- **Lower barrier:** -1.2 × ATR from prior close (loss)
- **Time barrier:** 10 calendar days (neutral/exit)
- **Label:** +1 (upper hit first), -1 (lower hit first), 0 (time — ambiguous exit)

Use **multiclass XGBoost** (objective=`multi:softprob`, num_class=3):
- P(+1) = probability of profitable exit
- P(-1) = probability of stopped-out exit
- P(0) = probability of time exit

**Entry threshold:** Enter only when P(+1) > 0.40 AND P(+1) > P(-1) + 0.10 margin.

**Why ATR-based barriers?** ATR barriers are regime-adaptive — in high-volatility regimes, the
barriers widen automatically, requiring a larger move to win but also to lose. This reduces
churn during choppy periods without needing an explicit regime gate.

**Why triple-barrier over binary?** Binary label trained on +2% absolute cannot distinguish:
- Clean setup that reaches +2% in 3 days (high quality)
- Stock that gaps up 1.8% then reverses to -3% (poor quality, same label = 0)
Triple-barrier labels the second case as -1 (stopped out), training the model to avoid it.

### Step 3 — PM Top-N Selection (replace fixed candidate list)

**Current:** PM takes all symbols that pass RSI_DIP or EMA_CROSSOVER, applies RM gates,
ranks by model score, enters top N (up to position limit).

**New:** PM takes top 10 by P(+1) from full 430-symbol daily scan, applies RM gates,
enters those passing min_probability threshold (P(+1) > 0.40).

**Position limits unchanged** — RM still enforces max positions, max sector concentration.

---

## Implementation Plan

### Phase 3b-1: Label Infrastructure (Week 1)

**Files:** `app/ml/training.py`, `app/ml/features.py`

1. Add `compute_triple_barrier_label(symbol_df, atr_mult_upper=2.0, atr_mult_lower=1.2, horizon_days=10)` to `app/ml/features.py`
   - Input: daily OHLCV DataFrame
   - Output: Series of {+1, -1, 0} labels indexed by entry date
   - ATR = 14-day ATR of prior close

2. Update `SwingModelTrainer._build_rolling_matrix()` in `app/ml/training.py`:
   - Add `label_type: str = "path_quality"` param (backward compat)
   - When `label_type="triple_barrier"`: use new label function
   - When `label_type="path_quality"`: existing behavior unchanged

3. Update `SwingModelTrainer.train()`:
   - Add `label_type` param thread-through
   - When triple-barrier: set `objective="multi:softprob"`, `num_class=3`
   - Adjust `scale_pos_weight` logic (not applicable to multiclass — use `class_weight` instead)

4. Update `app/database/models.py` `ModelVersion` perf dict to record `label_type`

5. **Tests:** `tests/test_phase_3b_triple_barrier.py`
   - `test_upper_barrier_hit()` — price crosses +2ATR on day 3, label = +1
   - `test_lower_barrier_hit()` — price drops -1.2ATR on day 2, label = -1
   - `test_time_exit()` — no barrier hit in 10 days, label = 0
   - `test_label_distribution()` — class balance is reasonable (not <5% any class)

**Deliverable:** Can train a triple-barrier swing model.

### Phase 3b-2: Full Universe Scan (Week 1-2)

**Files:** `app/agents/portfolio_manager.py`, `app/ml/features.py`

1. Add `rsi_dip_score`, `ema_cross_distance`, `is_rsi_dip`, `is_ema_cross` as numeric features in `app/ml/features.py` `compute_swing_features()`

2. In `app/agents/portfolio_manager.py` `_scan_swing_universe()`:
   - Remove call to `filter_by_prefilter()` (or make it opt-in via config flag)
   - Add config flag `swing_full_universe: bool = False` to `app/config.py`
   - When `swing_full_universe=True`: scan all 430 symbols
   - When False (default): current behavior (backward compat until WF validates)

3. Add `min_swing_probability: float = 0.40` config param — PM only enters if P(+1) exceeds this

4. **Tests:** `tests/test_phase_3b_full_universe.py`
   - `test_full_universe_scan_returns_all_symbols()` — when flag=True, no symbol filtered by RSI/EMA
   - `test_probability_threshold_gates_entry()` — entry blocked when P(+1) < 0.40
   - `test_rsi_dip_is_now_a_feature()` — `rsi_dip_score` appears in computed features

**Deliverable:** PM can scan full universe with probability threshold.

### Phase 3b-3: Training Run + Walk-Forward Validation (Week 2)

1. **Retrain:** `python scripts/train_swing.py --label-type triple_barrier --no-prefilters`
   - Use full 5yr training window, same cost model (5bps), same purge (10d)
   - Target: train on 430-symbol universe with triple-barrier labels
   - Expected training time: ~4-6hrs (5× more candidates than current ~80/day)

2. **Walk-forward:**
   - `python scripts/walkforward_tier3.py --model swing --swing-cost-bps 5 --swing-purge-days 10 --no-prefilters`
   - Gate: avg Sharpe > 0.80, no fold < -0.30, DSR p > 0.95

3. **Bootstrap validation** (20 iterations) if gate passed in Step 2
   - `python scripts/walkforward_tier3.py --model swing --bootstrap 20 --no-prefilters`
   - Gate: median Sharpe > 0.50, P5 > 0.0

### Phase 3b-4: Deployment (Week 3)

1. If walk-forward gate passed:
   - Set `swing_full_universe=True` in production config
   - Set `min_swing_probability=0.40` in config
   - Mark old model deprecated
   - Run paper trading for 14 days to validate live vs sim agreement

2. If walk-forward gate not met:
   - Analyze fold-level failures using regime diagnostic
   - Consider: (a) raise probability threshold, (b) add regime-conditional probability threshold, (c) refine ATR barrier multipliers

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| Training too slow (430 symbols × 5yr) | Use Polygon cache + parallelism (already in trainer) |
| Multiclass model harder to calibrate | Use `predict_proba` output, calibrate with isotonic regression on hold-out |
| Full universe scan too slow for live PM | Batch compute nightly for next-day; PM loads from cache |
| Triple-barrier labels imbalanced | Expected class distribution: ~20% +1, ~35% -1, ~45% 0; use `class_weight` |
| Introducing 430-symbol training breaks existing pipeline | `label_type` and `swing_full_universe` flags allow safe rollback |

---

## Success Criteria

- Walk-forward avg Sharpe > 0.80 with no-prefilter full-universe triple-barrier model
- Bootstrap median Sharpe > 0.50 (fold 3 / tariff regime no longer dominant negative)
- Live PM scan time < 30 seconds per cycle
- Paper trading live-vs-sim P&L gap < 1.5σ over 14-day validation window

---

## Dependencies / Blockers

- **Phase 3b does NOT require** Phase R5 (regime gate) — the triple-barrier label is regime-adaptive by design
- **Phase 3b benefits FROM** Phase R5 — if regime gate is deployed, fold 2 tariff period is gated out, making 3b results easier to interpret
- **Phase 4a (feature pruning)** should run BEFORE 3b training to reduce the feature set from 88 → 68, reducing overfitting risk in the full-universe setting

---

## Decision Gate

Before starting Phase 3b-3 (full training run):
- [ ] Phase 3b-1 tests passing (triple-barrier label)
- [ ] Phase 3b-2 tests passing (full universe scan)  
- [ ] Walk-forward ablation: triple-barrier label alone (no full universe) shows improvement over baseline
- [ ] Phase 4a feature pruning complete (swing 88 → 68)

Do not run the full training until label + scan changes are validated individually.
