# MrTrader Phase 47 — Intraday Model Quant Analysis

## Executive Summary

Phase 47 should focus on **alignment**, not simply adding more features. The current intraday model is now directionally viable: all folds are positive after the MetaLabelModel and PM abstention gate, but the system remains below the required gate.

Current state:

- Final v22 stack: **+0.301 average Sharpe**
- Gate requirement: **average Sharpe > +0.80**, with no fold below **-0.30**
- Base model AUC: **0.5438**
- MetaLabelModel: **R² = 0.001**, correlation **0.044**
- Current trade count: roughly **175 trades per fold**
- Current win rate: roughly **51.1%**

The biggest issue is that the model is trained like a broad binary classifier, but the Portfolio Manager behaves like a **daily cross-sectional ranker**. It scores the universe and selects the top five names. Therefore, the most important Phase 47 experiment is to train and evaluate the model as a ranking system.

---

## Recommended Phase 47 Experiments

### 47A — Replace Binary Classification with Daily Ranking Objective

**Priority:** Highest

### Hypothesis

The PM does not need the model to classify every stock/day observation correctly. It needs the model to rank today’s best candidates above the rest. AUC is therefore a weak optimization target. A ranking objective should improve top-five trade selection quality even if broad AUC barely changes.

### Implementation

Keep the same:

- Feature set
- Universe
- Entry timing
- Stop/target logic
- PM abstention gate
- Walk-forward validation structure

Replace the current `XGBClassifier` with an `XGBRanker`.

Suggested configuration:

```python
XGBRanker(
    objective="rank:pairwise",
    eval_metric="ndcg@5",
    tree_method="hist",
)
```

Group training examples by `trade_date`.

Use one of the following labels:

```python
label = path_quality_cross_sectional_rank
```

or:

```python
label = daily_path_quality_percentile
```

The model should learn to rank each day’s stocks by future `path_quality`, rather than classify top 20% versus bottom 80% across the entire dataset.

Track these evaluation metrics:

```text
daily_ndcg@5
daily_top5_avg_path_quality
daily_top5_realized_pnl
walk_forward_sharpe
fold_min_sharpe
```

### Expected Impact

Estimated Sharpe improvement: **+0.20 to +0.40**.

This is the highest-leverage experiment because it directly aligns the training objective with the PM’s actual trade decision.

### Risk

Ranking models can overfit date groups if the daily universe is inconsistent or if certain market regimes dominate. Use expanding-window walk-forward only and avoid random row-level splits.

---

### 47B — Compress Stop/Target Distances While Keeping 2:1 Reward/Risk

**Priority:** Very High

### Hypothesis

The current target may be too far for a two-hour intraday trade. The existing structure uses:

```text
stop   = 0.6× prior-day range
target = 1.2× prior-day range
```

Earlier experiments showed that a 1.2× prior-day range target is often too ambitious for a two-hour window. Even with the path-quality label, the simulator may be asking the model to predict a move that is too rare.

A smaller stop/target pair may increase target-hit frequency and reduce time exits while preserving the 2:1 reward/risk profile.

### Implementation

Run a single isolated stop/target experiment:

```text
Current: stop = 0.6× prior-day range, target = 1.2× prior-day range
New:     stop = 0.4× prior-day range, target = 0.8× prior-day range
```

Regenerate labels because `path_quality` depends on `stop_dist` and `target_dist`.

Keep unchanged:

- Model architecture
- Features
- Universe
- Entry timing
- PM abstention gate

Track:

```text
target_exit_pct
stop_exit_pct
time_exit_pct
avg_win
avg_loss
win_rate
fold_sharpe
```

### Expected Impact

Estimated Sharpe improvement: **+0.15 to +0.35**.

The system may not need a major win-rate increase if more winning trades actually hit target instead of exiting by time.

### Risk

Stops may become too tight and increase noise-driven stop-outs. If stop exits rise materially, test a slightly wider pair next:

```text
stop   = 0.5× prior-day range
target = 1.0× prior-day range
```

That would mirror the successful swing model structure more closely.

---

### 47C — Add a Move-Quality and Relative-Strength Feature Pack

**Priority:** High

### Hypothesis

The current features capture momentum, volume, ORB, VWAP, RSI, and SPY context, but they may not distinguish clean institutional accumulation from noisy opening movement.

For two-hour long-only momentum trades, the model should know whether the first hour was:

- Smooth
- Persistent
- Liquid
- Stronger than the market
- Supported by volume
- Holding above VWAP

This may help most in trending folds, where the current system underperforms relative to the sideways/choppy fold.

### Implementation

Add a compact feature pack. Do not change model architecture, labels, exits, universe, or entry timing in this experiment.

Recommended raw features:

```text
trend_efficiency_60m =
    abs(close_60m_return) / sum(abs(each_5m_return))

momentum_quality_60m =
    price_momentum / realized_vol_60m

green_bar_ratio_60m =
    count(close > open) / 12

above_vwap_ratio_60m =
    count(close > vwap) / 12

vwap_slope_60m =
    slope(vwap over first 12 bars)

pullback_from_high_60m =
    (high_60m - last_close) / high_60m

range_expansion_vs_20d =
    current_60m_range / avg_60m_range_same_window_20d

relative_strength_vs_spy =
    stock_60m_return - spy_60m_return

relative_volume_x_momentum =
    volume_surge * price_momentum

gap_followthrough =
    sign(gap_pct) * price_momentum
```

Recommended cross-sectional ranks:

```text
cs_rank_relative_strength_vs_spy
cs_rank_trend_efficiency
cs_rank_momentum_quality
cs_rank_above_vwap_ratio
cs_rank_range_expansion
```

### Expected Impact

Estimated Sharpe improvement: **+0.10 to +0.25**.

This is the best feature-only experiment because it targets the likely issue: the model has weak directional discrimination in trending regimes.

### Risk

Feature rebuild cost is higher because schema changes require cache invalidation and full rebuild. The new features are also correlated with existing features, so they may add noise if the signal is weak.

Use walk-forward performance, not in-sample feature importance, as the main decision point.

---

### 47D — Shrink to a Liquidity-Focused Universe

**Priority:** Medium-High

### Hypothesis

Russell 1000 may include too many noisy names for a five-minute intraday strategy. However, jumping directly to SP-100 may reduce daily opportunity too much.

A better first test is a liquidity-filtered universe that keeps enough breadth for top-five selection while removing thin, noisy, or sparse symbols.

### Implementation

Create a static universe before training and scoring:

```text
Top 300 symbols by median 20-day dollar volume
price > $10
exclude symbols with sparse 5-minute coverage
```

Then retrain and walk-forward only on this universe.

Keep unchanged:

- Model architecture
- Features
- Labels
- Exit logic
- Entry timing
- PM abstention gate

Test top 300 liquidity before SP-500 or SP-100.

### Expected Impact

Estimated Sharpe improvement: **+0.10 to +0.25**.

The likely benefit is cleaner bar behavior, fewer poor-quality signals, and more reliable continuation.

### Risk

A smaller universe may reduce trade count and concentrate the model into crowded mega-cap behavior. If trades per fold drop too much, test top 500 liquidity instead.

---

### 47E — Replace MetaLabelModel with Calibrated Score Abstention

**Priority:** Medium

### Hypothesis

The current MetaLabelModel is trying to predict individual trade `pnl_pct`, but it has almost no predictive power. This is likely because it trains on too few trade outcomes and because individual intraday P&L is very noisy.

A simpler calibrated threshold on the base model’s score may work better.

### Implementation

Remove the `XGBRegressor` MetaLabelModel for this experiment.

Instead, calibrate the base model score using out-of-fold training predictions.

Options:

```python
CalibratedClassifierCV(method="isotonic")
```

or Platt/logistic calibration.

Then allow variable daily trade count:

```text
Take top 5 only if calibrated_prob >= threshold
Otherwise take fewer than 5
Skip day if no candidate exceeds threshold
```

Candidate thresholds:

```text
p >= 0.55
p >= 0.60
p >= 0.65
```

Also test daily confidence filters:

```text
top1_score - median_score >= threshold
```

or:

```text
top5_avg_score >= threshold
```

Tune thresholds only on the training portion of each fold.

### Expected Impact

Estimated Sharpe improvement: **+0.10 to +0.30**.

This may reduce weak trade days without relying on a noisy second-stage regressor.

### Risk

Over-abstention can create a fragile backtest with too few trades. Add a minimum trade-count guardrail:

```text
at least 100 trades per fold
```

---

## What Not to Prioritize First

### Do Not Start with 3:1 Reward/Risk

A wider target is unlikely to be the easiest path. The current 1.2× prior-day range target already appears stretched for a two-hour hold. Moving to 3:1 would likely increase time exits and reduce realized winners.

### Do Not Start with Dynamic Entry Timing

Dynamic entry timing is promising, but it introduces too many moving parts at once:

- Entry time
- Label horizon
- Feature window
- Scoring frequency
- Candidate refresh logic

It should come after the model objective, stop/target structure, and feature-quality tests.

### Do Not Over-Invest in the Current MetaLabelModel

The existing meta-model has near-zero predictive power. Until the base selection quality improves, a more complex second-stage model is unlikely to solve the problem.

---

## Recommended Run Sequence

```text
47A — XGBRanker daily ranking objective
47B — Stop/target compression: 0.4× / 0.8× prior-day range
47C — Move-quality + relative-strength feature pack
47D — Liquidity top-300 universe
47E — Calibrated score abstention replacing MetaLabelModel
```

Run each experiment independently and sequentially. Do not combine changes until each experiment has been evaluated on the same Tier 3 walk-forward framework.

---

## Decision Framework

For every experiment, record:

```text
avg_sharpe
fold_1_sharpe
fold_2_sharpe
fold_3_sharpe
min_fold_sharpe
trade_count_per_fold
win_rate
avg_win
avg_loss
target_exit_pct
stop_exit_pct
time_exit_pct
max_drawdown
```

For model-objective experiments, also record:

```text
daily_ndcg@5
top5_avg_path_quality
top5_avg_realized_pnl
score_spread_top5_vs_median
```

Promote an experiment only if it improves average Sharpe without creating a fragile fold or collapsing trade count.

Recommended promotion rule:

```text
Promote if:
  avg_sharpe improves by >= +0.15
  and min_fold_sharpe remains >= -0.30
  and trades_per_fold >= 100
```

---

## Strongest Recommendation

The best single Phase 47 experiment is:

```text
47A — XGBRanker daily ranking objective
```

The reason is simple: the PM is a cross-sectional allocator. The model should be trained as a cross-sectional ranker.

Even if broad AUC remains around 0.54, the strategy can still improve materially if the model gets better at ranking the best five names each day. That is more aligned with the actual trading decision than trying to classify every stock/day observation correctly.
