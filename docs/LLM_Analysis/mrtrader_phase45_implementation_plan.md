# MrTrader Phase 45 — ML Model Improvement Implementation Plan

**Audience:** Claude Code (autonomous coding agent)
**Owner:** Min Kim
**Created:** 2026-04-25
**Repo:** TradingBot (existing structured backlog workflow)
**Goal:** Push v110's walk-forward Tier 3 Sharpe from current avg **−0.727** (all folds negative) to **≥ +0.8**, with no fold below −0.3.

---

## How to use this document

This plan is structured as 4 phases with numbered tasks inside each. Execute phases **in order** unless explicitly told otherwise. Phase 1 → Phase 2 → Phase 3 (with Phase 3-Parallel running alongside Phase 3).

Each phase has:
- **Why** (the reasoning — read this so you understand what to optimize)
- **Tasks** (numbered, executable units)
- **Training / tests to run**
- **Metrics to capture**
- **Decision gate** (criteria for proceeding to the next phase)
- **Things to watch for** (failure modes, what to learn for future runs)

After each phase, **stop and report** before starting the next phase. Do not auto-advance through gates.

---

## 1. Mission Context (read first)

### What we have today

`v110` is the current best model: XGBoost classifier, 400 trees, depth 4, lr=0.03, 84 OHLCV features, cross-sectional top-20% Sharpe-adjusted 5-day forward return label.

| Metric | Value |
|---|---|
| OOS AUC | 0.638–0.641 |
| Tier 3 Sharpe (full window, 2yr) | +0.34 |
| Profit factor | 1.11 |
| Win rate | 40.3% |
| Stop exits | **70%** |
| Target exits | 30% |
| Avg hold | ~1.4 bars |
| Trades | 290 |
| Walk-forward avg Sharpe (3 folds) | **−0.727** (all folds negative) |
| Top 5 features (importance) | All volatility measures |

### Why this plan now

Multiple ML and label experiments (v111–v118) have failed to break through. After independent review, the convergent diagnosis is:

> **v110 isn't a trade-entry model — it's a cross-sectional volatility opportunity ranker.** It learns to identify high-vol stocks that resolve into directional moves, but cannot predict whether the *path* to that move will avoid the 0.5× ATR stop within 1–2 days.

Three structural causes:

1. **Horizon mismatch.** Labels are 5-day forward returns; actual realized hold is ~1.4 bars. Half the label distribution describes outcomes the strategy never realizes.

2. **Vol bias in the label.** Cross-sectional Sharpe-adjusted ranking implicitly favors high-vol stocks. The top 5 features by importance are all vol measures because XGBoost found vol-rank to be the most stable, smooth gradient available.

3. **Stop/target structure is calibrated near break-even at random.** Under random-walk entries with stop=0.5×ATR and target=1.5×ATR, the gambler's ruin probability of stopping out first is `target / (target + stop) = 1.5 / 2.0 = 75%`. v110's 70% stop rate is only 5pp better than random. The structure leaves very little headroom for the alpha to translate into dollars.

This plan attacks all three causes in sequence, starting with the cheapest test (structure) and progressing to the most foundational (label) and most precise (conditional meta-model).

### Success criteria

| Criterion | Target |
|---|---|
| Walk-forward avg Sharpe (3 folds) | ≥ +0.8 |
| Worst fold Sharpe | ≥ −0.3 |
| Stop exits | ≤ 60% |
| Profit factor | ≥ 1.4 |
| Trade count over 2yr | ≥ 175 (do not trade the system into nothing) |
| Top-5 feature importance | No longer dominated by volatility features |

---

## 2. Forbidden Patterns (strict)

These have been tested and failed. **Do NOT re-attempt under any circumstances without explicit owner approval.** If a tempting refactor or experiment looks like one of these, stop and ask.

1. **DO NOT optimize for AUC.** AUC measures cross-sectional ranking, not trade-outcome accuracy. Tier 3 Sharpe is the only go/no-go metric.
2. **DO NOT use Tier 1 or Tier 2 backtest results as decision metrics.** Tier 1/2 routinely show Sharpe 2–7 while Tier 3 shows −1 to +0.34. **Tier 3 only.**
3. **DO NOT add generic technical indicators** (more RSI/MACD/Stoch variants, additional WQ alphas, more momentum lookbacks). The standard-technicals space is saturated. New features must be **entry-context** features (next-open path quality), not EOD oscillators.
4. **DO NOT use full-universe triple-barrier classification as a primary objective.** Failed in v115/v116 — model collapsed to all-positive due to class imbalance. Triple-barrier is allowed **only** as a *conditional* meta-label on a pre-screened candidate subset (see Phase 3).
5. **DO NOT tune inference-time filters on the same window used for evaluation.** Phase 34/35 lesson: full-window Sharpe +1.69, walk-forward OOS −0.727. Use **nested walk-forward** for any parameter selection.
6. **DO NOT use confidence thresholds > 0.55** without applying probability calibration (Platt or isotonic) first. v110 probabilities cluster below 0.55; raw thresholds above that produce zero trades.
7. **DO NOT prune features based on XGBoost importance.** v117 lesson: zero importance ≠ zero trading utility. Features may constrain candidate ranking at the margin even when they don't split trees.
8. **DO NOT use aggressive recency weighting** (half-life < 1 year). Will overfit to the latest regime. If recency weighting becomes necessary, use a 2–3yr half-life and clip weights at [0.35, 1.0].
9. **DO NOT swap to deep architectures** (transformers, TFT, neural ranking models) on this dataset size. 5yr × ~750 names × daily ≈ insufficient effective sample size.
10. **DO NOT ensemble XGBoost with linear models** (LR, etc.). v118 lesson: linear boundary suppresses non-linear signal, cut trade count 40%, hurt Sharpe by −0.42.
11. **DO NOT use VIX as a sample weight.** v111 lesson: redundant with VIX-as-feature, biased toward calm-market patterns, hurt Sharpe by −0.77.
12. **DO NOT overwrite v110 artifacts.** Every new model is a new version (v119, v120, v121). Keep the full lineage for attribution.
13. **DO NOT make multiple changes per retrain.** One change per version. This is the team's standing isolation rule for attribution.

---

## 3. Phase Plan

### Phase ordering rationale

- **Phase 1 (v119) first** — 0.5 day, inference-only, no retrain. Rules in or out a structural confound. If a different stop/target structure wins, all subsequent phases use the new multipliers and the same alpha translates to better dollars without any model change.
- **Phase 2 (v120) second** — fixes the foundational issue (the model is currently trained to predict the wrong thing). All subsequent ML work assumes the corrected label.
- **Phase 3 (v121) third** — the meta-model is a *conditional* filter on the primary model's candidates. It only makes sense once the primary model produces the right candidates.
- **Phase 3-Parallel** — runs alongside Phase 3 because it touches different code paths (Portfolio Manager day-level abstention vs. Risk Manager candidate-level filter) and is operationally independent.

---

## Phase 1 — v119: Stop/Target Structure A/B (inference-only)

**Estimated effort:** 0.5 day
**Retrain required:** No (uses v110 model unchanged)
**Expected Tier 3 Sharpe lift:** +0.34 → ~+0.45 on full window

### Why this phase

Under random-walk entries, `P(stop) = target / (target + stop)`. The current 0.5×/1.5× structure has a theoretical random stop rate of 75%; v110 achieves 70% — only 5pp of edge over random. Tightening the asymmetry to 0.75×/1.25× lowers the random stop rate to 62.5%, giving the same alpha much more room to translate into Sharpe. **Run this before any retrain because if it works, the new label in Phase 2 must be computed against the new multipliers.**

### Tasks

**Task 1.1 — Add a config-driven stop/target structure to the Trader execution path.**
- Locate the file(s) where stop and target multipliers are applied to ATR. Likely candidates: `src/.../trader/execution.py`, `src/.../trader/exits.py`, or the trade-config dataclass that the Trader reads.
- Add a config object `StopTargetConfig(stop_mult, target_mult, max_hold_days)` and wire the existing values (0.5, 1.5, 5) as the default.
- Confirm via grep that no hard-coded `0.5 * atr` or `1.5 * atr` remains in the execution path.

**Task 1.2 — Run the Tier 3 backtest under three configurations.**

| Config | stop_mult | target_mult | max_hold | Random P(stop) |
|---|---|---|---|---|
| Baseline | 0.5 | 1.5 | 5 | 75% |
| **A** | 0.75 | 1.25 | 3 | 62% |
| B | 0.5 | 1.0 | 3 | 67% |

Use **nested walk-forward**, not full-window. For each fold:
1. Use the train/val segment to verify config behaves sensibly (no degenerate trade behavior).
2. Freeze the config for that fold.
3. Evaluate on the OOS fold of the existing 3-fold split.

**Do NOT pick the best full-window result.** That is the Phase 34/35 mistake.

**Task 1.3 — Capture per-config metrics.** For each of the 3 configs × 3 folds = 9 backtests, record:
- Tier 3 Sharpe
- Profit factor
- Win rate
- Stop exits %
- Target exits %
- Time-stop exits %
- Trade count
- Avg hold (bars)
- Total return
- Max drawdown
- Avg winning trade R, avg losing trade R

### Decision gate

Proceed to Phase 2 with the structure that produces:
- **Walk-forward avg Sharpe** at least +0.10 better than baseline, AND
- **Trade count** ≥ 80% of baseline (similar activity), AND
- **No fold worse than −0.40**.

If neither A nor B beats baseline by ≥ +0.10, retain baseline and proceed to Phase 2. We've then learned that the structure is not the bottleneck — the alpha is — and Phase 2 / 3 carry more weight.

### Things to watch for / learn

- If A wins decisively, it strongly suggests the bottleneck was structural and the existing alpha was being squandered on a too-tight stop. Note this for future stop-calibration work.
- If trade count collapses under A or B (e.g., target hit so easily that exits cluster on day 1), check whether `max_hold_days=3` is too aggressive — try 4 or 5.
- If win rate goes up but Sharpe doesn't, you've found a regime where target wins are too small. The R:R changed (3:1 → 1.67:1) so per-trade payoff is lower; need higher hit rate to compensate.
- Capture the **distribution** of MAE (max adverse excursion) across all trades in baseline. If most stops are hit in the 0.5–0.7×ATR range, the 0.75× stop will rescue many of them.

### Reporting (after Phase 1)

Write a short note to `docs/phase45/v119_results.md`:
1. Which config won and by how much.
2. The full 3×3 metrics table.
3. The MAE distribution histogram.
4. Recommendation: which structure to lock in for Phase 2.

**Stop and wait for owner review before starting Phase 2.**

---

## Phase 2 — v120: Path-Quality 2-Day Vol-Bucketed Regression Label

**Estimated effort:** 2 days
**Retrain required:** Yes (full retrain on 5yr, 753 symbols)
**Expected Tier 3 Sharpe lift:** ~+0.45 → +0.65 (cumulative from v119)

### Why this phase

The current label has three problems compounding: 5-day horizon vs 1.4-day realized hold, Sharpe-adjusted ranking that implicitly favors high-vol stocks, and binary top-20 cutoff that throws away granularity. This phase fixes all three at once with a single new label scheme. **The label must be computed using the stop/target multipliers chosen in Phase 1.**

### Tasks

**Task 2.1 — Add new label mode `path_quality_2d_vol_bucketed_regression`.**

Locate the label construction code (search: `top.*20|forward_window|sharpe.*adjusted|scale_pos_weight`). Likely candidates:
- `src/.../training/label_builder.py`
- `src/.../ml/train_pm_model.py`
- `src/.../features/build_training_dataset.py`

**Add as a NEW label mode. Do NOT remove or modify the existing `cross_sectional_top20_5d` mode** — keep v110's label intact for regression testing.

**Task 2.2 — Implement the path-quality score function.**

```python
def compute_path_quality_score(
    entry_open: float,         # next-open price after signal
    high_d1: float, high_d2: float,
    low_d1: float, low_d2: float,
    close_d2: float,
    atr: float,                # ATR_14 at signal date
    stop_mult: float,          # from Phase 1 winning config
    target_mult: float,        # from Phase 1 winning config
) -> float:
    """
    Path-quality score: explicitly penalizes paths that would have hit the stop,
    even if terminal return ends positive. This aligns the label with what the
    Tier 3 trader actually monetizes.
    """
    mfe_2d = max(high_d1, high_d2) - entry_open       # max favorable excursion
    mae_2d = entry_open - min(low_d1, low_d2)         # max adverse excursion
    close_ret_2d = close_d2 - entry_open

    upside_capture = min(mfe_2d / (target_mult * atr), 1.0)
    stop_pressure = min(mae_2d / (stop_mult * atr), 1.0)
    close_strength = close_ret_2d / atr  # NOT capped

    score = (
        1.00 * upside_capture
      - 1.25 * stop_pressure       # asymmetric penalty — this is the key term
      + 0.25 * close_strength
    )
    return score
```

The `−1.25 × stop_pressure` term is the critical innovation. It penalizes trades that *touched the stop* even if they recovered, which terminal-return labels do not.

**Task 2.3 — Vol-bucket cross-sectional ranking.**

```python
import pandas as pd

# Per training window (date-grouped):
df["vol_bucket"] = (
    df.groupby("date")["realized_vol_20d"]
      .transform(lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop"))
)

# Rank within (date, vol_bucket). This stops vol from dominating the LABEL
# while keeping vol available as a FEATURE.
df["target"] = (
    df.groupby(["date", "vol_bucket"])["path_quality_score"]
      .rank(pct=True)
)
```

**Task 2.4 — Switch to regression objective.**

```python
xgb_params = {
    "objective": "reg:squarederror",   # was: "binary:logistic"
    "eval_metric": "rmse",
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 400,
    # Remove scale_pos_weight — not applicable for regression
}
```

At inference, rank candidates by predicted percentile (the regression target was percentile-ranked), and take top-N as before.

**Task 2.5 — Train v120.**

- Same 5yr training window, 753 symbols, 63d rolling, 1d step, embargo, time-based 75/25 split.
- Same 84 features (do NOT change the feature set in this phase — one change per retrain).
- Match the training time budget (~4 minutes on 8 workers).
- Save model artifacts as `v120_*` — do not overwrite v110.

**Task 2.6 — Run Tier 3 backtest with v120 + the structure from Phase 1.**

Capture the same metric set as Phase 1. Plus:
- **Top-15 feature importance** (we expect vol features to drop out of the top 5).
- **Distribution of predicted percentiles** for selected vs rejected candidates.
- **Walk-forward Sharpe across all 3 folds.**

### Decision gate

Proceed to Phase 3 if:
- **Walk-forward avg Sharpe** improved by ≥ +0.20 over Phase 1 result, AND
- **At least one fold** turns positive (vs all-negative for v110), AND
- **Stop exits** dropped at least 5pp from Phase 1 result, AND
- **Trade count** stayed ≥ 175 over 2yr.

If walk-forward is still all-negative, **stop and report** before Phase 3. The label fix should have moved at least one fold; if it didn't, something else is wrong (suspect: data leakage, regime shift in the test fold, or the path-quality score weights need tuning).

### Things to watch for / learn

- **Top feature importance shift.** If the top 5 are still all volatility, the vol-bucketing isn't doing enough — try residualization as a fallback (see Section 5 longer-term ideas in source plan). Capture the feature-importance delta vs v110.
- **Predicted percentile distribution.** For binary labels v110's probabilities clustered below 0.55. For regression on percentile, predictions should span a wider range. If they cluster narrowly, the model isn't differentiating — investigate before going to Phase 3.
- **AUC-equivalent metric.** Compute Spearman rank correlation between predicted percentile and realized path_quality_score. Expect 0.10–0.20 (lower than v110's AUC because the label is harder); this is a sanity check, not a gate.
- **Per-fold dispersion.** If one fold is much better than the others, note which one and what regime it covers. Regime-conditional models become a candidate for Phase 46 if dispersion is high.
- **Time-stop exits.** Track these as a separate category. With max_hold=3 from Phase 1, time-stop frequency tells us how many trades simply expired without resolution — if high, the 2-day label horizon may itself need adjusting.

### Reporting (after Phase 2)

Write `docs/phase45/v120_results.md` with:
1. Full Tier 3 metrics across all 3 walk-forward folds.
2. Top-15 feature importance (compared to v110).
3. Predicted percentile distribution histogram.
4. Spearman rank correlation (predicted vs realized score).
5. Failure modes observed (if any) and hypotheses.

**Stop and wait for owner review before starting Phase 3.**

---

## Phase 3 — v121: Early-Stopout Meta-Model (Risk Manager Filter)

**Estimated effort:** 3–4 days
**Retrain required:** Yes, but only on v120's candidate subset (~10–20K rows)
**Expected Tier 3 Sharpe lift:** ~+0.65 → +0.85–1.0 (cumulative)

### Why this phase

v120 fixed *what* we're predicting, but it still uses only EOD features. It cannot see the next-open entry context — gap, opening range, position relative to prior day's range — which is exactly the information needed to predict early stop-outs. This phase adds a second-stage classifier specifically trained to predict "given that v120 already likes this stock, will the entry stop out in days 1–2?" This is **meta-labeling** in de Prado's sense (AFML Ch. 3).

This succeeds where v115/v116 failed because:
- Trained on v120's pre-screened candidates (not full universe) → class balance is much healthier.
- Predicts a narrower question (early stopout) → simpler decision boundary.
- Acts as a Risk Manager filter, not a primary objective → keeps v120's broader signal intact.

### Tasks

**Task 3.1 — Pre-validation: do entry-context features actually predict early stopout?**

**Do this BEFORE training the full meta-model.** This is the cheapest derisking step.

```python
# Build a small validation dataset: ~5K candidates from v120's top picks,
# random sample across all 5 years.
entry_context_features = [
    "gap_at_entry_atr",                # (next_open - prior_close) / atr
    "open_location_in_prior_range",    # (next_open - prior_low) / (prior_high - prior_low)
    "prior_day_range_atr",             # (prior_high - prior_low) / atr
    "prior_day_close_location",        # (prior_close - prior_low) / (prior_high - prior_low)
    "distance_from_ema20_at_entry_atr",
    "distance_from_prior_high_atr",
    "distance_from_prior_low_atr",
    "spy_gap_at_entry",
    "spy_2d_return_prior",
    "sector_etf_gap_at_entry",
]

early_stopout = (min(low_d1, low_d2) <= entry_open - stop_mult * atr).astype(int)

# For each feature, compute univariate AUC vs early_stopout.
# Also compute mutual information.
```

If **no individual feature has AUC > 0.54** vs early_stopout, the meta-model has no signal to work with — stop and report. If at least 3 features clear AUC 0.54, proceed.

**Task 3.2 — Build the meta-training dataset.**

For every candidate that v120's Portfolio Manager would have ranked top-K (use K=10 or K=20) on each historical day, create one row containing:

- Date, symbol, v120_score (the predicted percentile)
- All 84 of v120's input features (subset OK if needed)
- The 10 entry-context features from Task 3.1
- Label: `early_stopout = 1 if min(low_d1, low_d2) <= entry_open - stop_mult * atr else 0`
- Inverted for training: `accept = 1 - early_stopout`

Expect ~10–20K rows total over 5 years.

**Task 3.3 — Train v121 meta-classifier (shallow XGBoost).**

```python
v121_params = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    "max_depth": 3,                    # SHALLOW — meta-models overfit fast
    "learning_rate": 0.05,
    "n_estimators": 150,               # also kept low
    "min_child_weight": 5,             # extra regularization
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": class_balance_ratio,  # compute from data
}
```

**Critical:** train only on the candidate subset, NOT the full universe. Use the same time-based split as v120 (75/25, embargoed). Do not let v120's training data leak into v121's training.

**Task 3.4 — Integrate v121 as a Risk Manager filter.**

In the RM rule chain, add:

```python
def stopout_meta_filter(candidate, threshold=0.50):
    """
    v121 outputs P(no_early_stopout). Reject if below threshold.
    """
    p_accept = v121.predict_proba(candidate.features)[1]
    if p_accept < threshold:
        return Reject("early_stopout_risk", p_accept=p_accept)
    return Accept(p_accept=p_accept)
```

Use as a binary reject (not a sizing dial) initially. Sizing comes later if the binary version works.

**Task 3.5 — Threshold tuning via nested walk-forward.**

Test thresholds: 0.45, 0.50, 0.55. Pick the per-fold best on val, evaluate OOS. **Do not pick the best full-window result.**

**Do NOT use thresholds > 0.55** — even after meta-model training, probability calibration may be needed. If the 0.55 threshold rejects > 50% of candidates, apply isotonic calibration on the validation set first.

**Task 3.6 — Run Tier 3 backtest with v120 + v121 filter + Phase 1 structure.**

### Decision gate

We pass the gate if:
- **Walk-forward avg Sharpe ≥ +0.8**, AND
- **No fold below −0.3**, AND
- **Stop exits ≤ 60%**, AND
- **Trade count ≥ 175** over 2yr (not over-filtered).

If trade count drops below 175, the threshold is too strict — relax to 0.45 and re-evaluate. If even at 0.45 the gate isn't met, we've learned that meta-labeling isn't sufficient and need to revisit feature engineering (Phase 46 candidate: add microstructure features at entry).

### Things to watch for / learn

- **v121 univariate feature importance.** Which entry-context features matter most? `gap_at_entry_atr` and `open_location_in_prior_range` are the leading hypotheses. If sector-ETF gap features dominate, intermarket features become the next exploration direction.
- **False negative rate.** Of the candidates v121 rejects, how many would have been winners? If > 30%, we're rejecting too aggressively — investigate whether the meta-model overfits to recent regimes.
- **Calibration curve.** Plot reliability diagram (predicted P vs realized accept rate). If miscalibrated, apply Platt or isotonic before threshold tuning.
- **Per-fold consistency.** If v121 helps on 2 folds but hurts on 1, the meta-model has a regime dependency. Note which fold and what regime.
- **Marginal contribution.** Compute counterfactual: what would Tier 3 Sharpe be with v120 alone (no v121)? The marginal Sharpe lift from v121 tells us how much value the meta-stage adds.

### Reporting (after Phase 3)

Write `docs/phase45/v121_results.md` with:
1. Pre-validation AUCs for entry-context features (from Task 3.1).
2. Final Tier 3 metrics across all 3 walk-forward folds.
3. v121 feature importance.
4. False-negative analysis.
5. Calibration curve.
6. Counterfactual: v120 alone vs v120 + v121 marginal Sharpe lift.
7. Recommendation: pass/fail gate, and if failing, what to try next.

**Stop and wait for owner review.**

---

## Phase 3-Parallel — Daily Abstention Gate (Portfolio Manager)

**Estimated effort:** 1 day
**Retrain required:** No (rule-based, no model)
**Expected Tier 3 Sharpe lift:** additive +0.10–0.20 on top of Phase 3

### Why this phase

Even with a good model, some days the cross-sectional ranking is so flat that the top picks aren't meaningfully better than the median. On those days, forcing trades into a structure that's calibrated to break even at random is negative-EV. This phase adds a daily-level abstention gate that uses the Portfolio Manager's own scores to detect "low-conviction days."

This runs **in parallel with Phase 3** because it touches the PM (day-level), not the RM (candidate-level). They don't conflict.

### Tasks

**Task P.1 — Compute daily PM-score statistics.**

In the PM, after scoring all SP-500 stocks, compute:

```python
top_5_mean = scores.nlargest(5).mean()
median_score = scores.median()
daily_edge_dispersion = top_5_mean - median_score

# Z-score the daily top-score against its own 63-day rolling distribution
top_score = scores.max()
top_score_z = (top_score - top_score_mean_63d) / top_score_std_63d
```

**Task P.2 — Add abstention rule.**

```python
# Allow new entries only when conviction is meaningful
allow_new_entries = (
    top_score_z > 0
    and daily_edge_dispersion > median_edge_dispersion_60d
    and not vix_fear_spike  # VIX > 30 still gates regardless
)
```

If `allow_new_entries == False`, skip *new* entries that day (do not close existing positions early — let exits work normally).

**Task P.3 — Backtest.**

Run Tier 3 with:
- v120 + v121 + Phase 1 structure (all from Phase 3) + the new abstention gate.

Compare:
- Tier 3 with abstention vs without.
- Distribution of trading days (% days active vs flat).

### Decision gate

Pass if:
- **Walk-forward Sharpe** improves by ≥ +0.10 over Phase 3 alone, AND
- **Trade count** stays ≥ 175 (do not abstain so much that the strategy stops trading).

### Things to watch for / learn

- **What % of days does the gate flatten the strategy?** Target: 20–40% flat days. If > 60%, the gate is too strict.
- **Sharpe lift attribution.** Is the lift from avoiding losing days, or from concentrating into winning days? Decompose by daily P&L distribution before/after.
- **Regime correlation.** Are the flat days correlated with any market regime (high VIX, low breadth, post-Fed)? If so, document — this becomes input for Phase 46 regime modeling.
- **Days when gate said "skip" but a trade would have won.** Compute the opportunity cost (trades v120+v121 would have placed but were gated). If the gate's accuracy is < 55% (i.e., it skips winners about as often as losers), tune the dispersion threshold.

### Reporting

Add a section to `docs/phase45/v121_results.md`:
1. Abstention rate (% flat days).
2. Sharpe with vs without gate.
3. Day-type analysis.
4. Opportunity cost.

---

## 4. Reporting & Logging Standards

For every backtest run during this phase, capture and persist the following to `runs/phase45/{version}/{timestamp}/`:

- `config.json` — full hyperparameter and structure config
- `metrics.json` — all Tier 3 metrics (per-fold and aggregate)
- `feature_importance.csv` — top-30 features with gain/cover/freq
- `predictions.parquet` — one row per (date, symbol, prediction) for the test set
- `trade_log.parquet` — full Tier 3 trade history with entry/exit reasons
- `mae_mfe_distribution.png` — histograms of MAE and MFE across all trades
- `walk_forward_summary.md` — human-readable summary with the per-fold Sharpe table

Every model artifact must be tagged with:
- Git SHA at training time
- Parent model version (v110, v119, v120 …)
- Label mode used
- Stop/target multipliers used
- Feature set hash

This lineage is non-optional. Without it, attribution becomes guesswork.

---

## 5. Decision Tree (after each phase)

```
Phase 1 (v119)
├── Structure A or B wins by >+0.10 Sharpe → lock new structure, proceed to Phase 2
├── Marginal improvement (<+0.10) → keep baseline structure, proceed to Phase 2
└── A and B both worse → keep baseline, proceed to Phase 2 (structure isn't bottleneck)

Phase 2 (v120)
├── Walk-forward avg lifts ≥ +0.20 AND ≥ 1 fold turns positive → proceed to Phase 3
├── Lift but all folds still negative → STOP, report, investigate label weights
└── No lift → STOP, report, investigate (data leakage? feature regression?)

Phase 3 (v121)
├── Gate met (walk-forward ≥ +0.8, no fold < −0.3) → SUCCESS, proceed to Phase 3-Parallel
├── Close to gate (≥ +0.6 avg) → run Phase 3-Parallel for additive lift, may push over
└── Far from gate (< +0.6 avg) → STOP, report, escalate to Phase 46 planning
```

---

## 6. What to Learn for Future Runs

Across all phases, capture observations on these questions. They feed Phase 46 planning regardless of outcome:

1. **How much of the lift came from structure vs label vs meta-filter vs abstention?** This tells us where to invest first in Phase 46.
2. **Which entry-context features carried v121?** This shapes future feature engineering.
3. **Are there regimes where the strategy systematically fails?** (E.g., high realized correlation, low dispersion, post-Fed days.) Regime-conditional modeling is the natural Phase 46.
4. **Does the abstention gate's flat-day rate correlate with macro regime?** If yes, an explicit regime model could replace the rule-based gate.
5. **Where does walk-forward variance come from?** If one fold dominates the variance, that fold's regime should be studied as a generalization risk.
6. **Calibration drift.** Does v121's calibration degrade across folds? If yes, periodic recalibration becomes a maintenance task.

Summarize these in a final `docs/phase45/phase45_retrospective.md` after Phase 3-Parallel completes, regardless of whether the gate is met.

---

## 7. Glossary

- **Tier 1 / Tier 2 / Tier 3 backtest:** Three levels of fidelity. Tier 1 = signal-only. Tier 2 = signal + simple execution. Tier 3 = full agent (PM/RM/Trader) on historical bars. **Tier 3 is the only metric that matters for go/no-go.**
- **Walk-forward:** 3-fold time-based out-of-sample evaluation. Currently: Fold 1 (Jan–Oct 2024), Fold 2 (Oct 2024–Jul 2025), Fold 3 (Aug 2025–Apr 2026).
- **Path-quality score:** New label scheme (Phase 2). Penalizes paths that touched the stop, even if terminal return ends positive.
- **Meta-labeling:** de Prado's two-stage approach (AFML Ch. 3). Primary model identifies candidates; secondary model filters them. Phase 3 implements this.
- **Gambler's ruin probability:** Under random-walk entries, `P(stop_first) = target_distance / (target_distance + stop_distance)`. The structural break-even rate.
- **MAE / MFE:** Maximum Adverse Excursion / Maximum Favorable Excursion. Path statistics for each trade.
- **R:R:** Reward-to-risk ratio. Current 3:1 (1.5× target / 0.5× stop). Phase 1 tests 1.67:1 (config A) and 2:1 (config B).

---

## 8. Out of Scope for Phase 45

These are deferred to Phase 46 or later, after Phase 45 lands:

- **CPCV (Combinatorial Purged Cross-Validation)** — methodology improvement on a misspecified label is wasted effort. Implement only after labels are correct.
- **Long-short construction** — meaningful Sharpe upside but requires capital structure changes. Operationally complex with $20k.
- **Regime-conditional mixture of experts** — only after the label is right.
- **Conservative recency weighting** (2–3yr half-life) — only if walk-forward still struggles after Phase 3 lands.
- **Sector-relative or beta-bucket ranking** — vol-bucketing in Phase 2 captures most of the same intuition.
- **Calibration as a permanent layer** — apply ad hoc in Phase 3 if needed; integrate properly in Phase 46.
- **Microstructure / intraday features** — beyond the 10 entry-context features added in Phase 3, deeper microstructure work is a separate research effort.

If during Phase 45 you find yourself wanting to do any of these, **stop and ask** — they require their own planning.

---

## 9. Final Notes

- **One change per retrain.** This is a hard rule. If you discover a bug that needs fixing during Phase 2, fix it as a hotfix (not as part of v120) and document the diff.
- **Preserve all artifacts.** v110 stays untouched. Every new version (v119, v120, v121) gets its own artifact directory. Lineage is required for attribution.
- **Tier 3 walk-forward is the only metric that matters for gating decisions.** Everything else (AUC, profit factor, win rate) is diagnostic, not decisional.
- **When in doubt, stop and report.** Especially before crossing a phase boundary. Better to confirm direction than to chain two large changes that turn out to be wrong.

End of plan.
