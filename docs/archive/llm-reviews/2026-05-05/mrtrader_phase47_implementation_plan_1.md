# MrTrader Phase 47 — Intraday Model Implementation Plan

**Audience:** Claude Code (autonomous coding agent)
**Owner:** Min Kim
**Created:** 2026-04-26
**Repo:** TradingBot (existing numbered-task backlog workflow)
**Goal:** Push intraday v22's walk-forward Tier 3 Sharpe from current avg **+0.301** to **≥ +0.80**, with no fold below **−0.30** and trade count ≥ 100/fold. If the gate is unreachable on OHLCV-only 5-min data, document the achievable ceiling.

---

## How to use this document

This plan is structured as 6 phases with numbered tasks inside each. Execute phases **in order**. Phase 0 is mandatory and runs before any retrain.

Each phase has:
- **Why** (read this so you understand what to optimize)
- **Tasks** (numbered, executable units)
- **Decision gate** (criteria for proceeding)
- **Things to watch for** (failure modes, lessons for future work)
- **Reporting** (what to write down before moving on)

After each phase, **stop and report** before starting the next phase. Do not auto-advance through gates.

---

## 1. Mission Context (read first)

### What we have today

`v22` is the current best intraday model:
- XGBoost classifier (binary), 42 features on 5-min OHLCV + SPY context
- Label: `path_quality` regression score → cross-sectional top-20% = label 1
- MetaLabelModel v1 (XGBRegressor) filtering trades by predicted P&L
- PM abstention gate (VIX ≥ 25 OR SPY < MA20) skips entries on regime-bad days

| Metric | Value |
|---|---|
| OOS AUC | 0.5438 |
| Walk-forward avg Sharpe (3 folds) | **+0.301** |
| Fold 1 (Oct '24 – Apr '25, trending) | +0.24 |
| Fold 2 (Apr – Oct '25, sideways) | **+0.57** |
| Fold 3 (Oct '25 – Apr '26, trending) | +0.23 |
| Win rate | 51.1% |
| Stated R:R | 2:1 (1.2× / 0.6× prior-day range) |
| Trades / fold | ~175 |
| MetaLabelModel R² | 0.001 (statistical noise) |

**Gate:** avg Sharpe ≥ +0.80, no fold below −0.30, trades ≥ 100/fold.

### Why this plan now

This plan synthesizes two independent quant reviews. The strongest single insight: **the PM is a daily cross-sectional ranker selecting top-5 names, but the model is trained as a binary classifier on top-20% across the entire dataset.** This is an objective mismatch — AUC is a weak optimization target for a top-N selection problem. Training as a ranker (XGBRanker with NDCG@5) directly aligns the loss function with the trading decision.

Three structural observations to internalize before touching code:

**1. Trending/sideways paradox.**
Fold 2 (sideways) outperforms Folds 1 & 3 (trending) for an ostensible momentum strategy. This is structurally backwards. The most likely explanation: despite the ORB framing, the model may have learned reversion-flavored patterns. The `−1.25 × stop_pressure` term in the path_quality label heavily penalizes trades that touched the stop — but trending winners almost always pull back through the stop before continuing. That term may be systematically excluding the trades that would win in trending regimes.

**2. Realized R:R is probably worse than 2:1.**
The "51% × 1.2 − 49% × 0.6" expectancy math assumes wins capture full target and losses are at full stop. With a 2-hour hold and time exits at bar 24, "wins" include time-exit positives that average +0.3× to +0.5× ATR (not +1.2×). True realized R:R is probably 1.2–1.4:1, not 2:1. **The Sharpe gap is likely wider than the +0.50 calculated.**

**3. The meta-model probably contributes nothing.**
R² = 0.001, correlation = 0.044 is statistical noise. A model with corr=0.044 cannot meaningfully filter trades; it filters approximately at random. The Stage 1 → Stage 2 Sharpe lift (−0.292 → +0.301) almost certainly came from the PM abstention gate alone. The meta-model is added complexity contributing nothing.

### Honest note on the gate

With OHLCV-only at 5-min resolution, no L2 data, no shorting, and a 2-hour hold, **+0.80 Sharpe is achievable but not guaranteed.** Realistic post-improvement range from where we sit: **+0.45 to +0.85.**

If after Phase 5 the system lands at +0.55 to +0.65, three options to weigh:
1. Lower gate to +0.60 with same robustness criteria, document rationale
2. Paper trade at half size for 60 days, let live data inform
3. Accept intraday as a smaller, lower-Sharpe sleeve; swing model (+1.18) carries the portfolio

A blended swing+intraday at +0.60 is meaningfully better than swing alone. Don't anchor on a number that may not be achievable at available data fidelity.

---

## 2. Forbidden Patterns (strict)

These have been tested and failed, or violate hard methodological principles. **Do NOT re-attempt without explicit owner approval.**

1. **DO NOT optimize for AUC.** Tier 3 walk-forward Sharpe is the only go/no-go metric.
2. **DO NOT use Tier 1 or Tier 2 backtest results as decision metrics.**
3. **DO NOT add generic technical indicators** without theoretical grounding tied to the trade horizon.
4. **DO NOT use binary ATR labels for intraday** (v19 lesson — produced 5 total trades across 3 folds).
5. **DO NOT use hard ORB or volume gates in the simulator** (v20 lesson — Fold 2 produced 0 trades). Features remain for the model to use; gates stay soft.
6. **DO NOT make multiple changes per retrain.** One isolated change per version.
7. **DO NOT overwrite v22 artifacts.** Every new model is a new version (v23, v24, …). Preserve full lineage for attribution.
8. **DO NOT skip the diagnostic dive (Phase 0).** It's free and reshapes priorities.
9. **DO NOT use confidence thresholds > 0.65** without isotonic or Platt calibration.
10. **DO NOT pursue full-universe binary triple-barrier classification.**
11. **DO NOT use random row-level CV splits.** Always expanding-window walk-forward, with embargo where applicable.
12. **DO NOT switch to deep architectures** (transformers, LSTMs, neural ranking models). 280k rows × 42 features is insufficient for those models on this signal-to-noise ratio.
13. **DO NOT combine experiments in one retrain.** Even when two changes seem complementary.
14. **DO NOT chase feature importance from XGBoost.** Zero importance ≠ zero trading utility (lesson from swing model).
15. **DO NOT increase universe back to full Russell 1000** if Phase 4 shows shrinkage helps.

---

## 3. Phase Plan

### Phase ordering rationale

| Order | Phase | Reason |
|---|---|---|
| 0 | Diagnostic dive | Free, half day, no compute. Settles three structural questions before any retrain. Skipping this means later experiments are partial guesses. |
| 1 | v23 — Drop meta + isotonic calibration | Cheapest test (no model retrain). Confirms meta-model attribution and simplifies the system before the bigger architectural change in Phase 2. Sets up clean foundation. |
| 2 | v24 — XGBRanker daily ranking objective | **Highest expected impact.** Aligns model objective with PM's actual decision. Run after meta is dropped so attribution is clean. |
| 3 | v25 — Stop/target compression to 0.4×/0.8× | Directly addresses realized R:R if diagnostic shows time-exits dominate. Relabeling required. |
| 4 | v26 — Top-300 by liquidity universe | Cleaner data per row. Dynamic dollar-volume filter avoids survivorship issues. |
| 5 | v27 — Move-quality + relative-strength features | Most expensive (cache rebuild). Run last; lowest expected ROI per hour. |
| 6 | Reassess | Decide on gate: pass to paper, lower bar, or escalate to Phase 48. |

---

## Phase 0 — Diagnostic Dive (mandatory, no compute)

**Estimated effort:** Half day of pandas analysis on existing v22 trade logs.
**Retrain required:** No.

### Why this phase

Three structural questions need answers before any retrain. Skipping the diagnostic means optimizing on incomplete information. Each cut takes minutes; together they reshape which experiments are worth running.

### Tasks

**Task 0.1 — Load v22 trade logs across all 3 folds.**
Build a single DataFrame with one row per trade containing: entry timestamp, symbol, entry price, exit price, exit reason (target/stop/time), realized P&L, MFE, MAE, ORB direction, entry direction, fold ID, VIX at entry, SPY-vs-MA20 at entry, daily PM rank position.

**Task 0.2 — Run the 7 diagnostic cuts.**

| # | Cut | Settles |
|---|---|---|
| 1 | Win rate by ORB direction (entry direction vs ORB breakout direction) | Momentum vs reversion. If win rate higher when entry *opposes* ORB direction → reversion-flavored. |
| 2 | Exit-type breakdown: % target / % stop / % time-positive / % time-negative | True realized R:R. |
| 3 | Win rate by VIX bin (<15 / 15–25 / ≥25) and by SPY-vs-MA20 (above/below) | Quantifies abstention gate's contribution. |
| 4 | Win rate stratified by `cs_rank_momentum` quintile and `orb_position` decile | Which features actually carry the edge. |
| 5 | Sharpe with PM abstention only (no meta-model) vs full stack | Meta-model attribution. |
| 6 | MFE / MAE distribution by exit type | Are winners exiting early? Are losers showing big MFE before reversing? |
| 7 | Per-fold win rate, exit mix, avg trade R | Which fold/regime has the gap. |

**Task 0.3 — Write `docs/phase47/diagnostic_report.md` with three explicit answers:**

1. **Momentum or reversion?** (cut #1)
2. **True realized R:R?** (cut #2)
3. **Meta-model marginal Sharpe?** (cut #5)

### Decision branches

- If **reversion-flavored** → v25 (stop/target compression) becomes higher priority over v24 logic; the `path_quality` weights may need reconsidering for Phase 48
- If **true R:R ≤ 1.4:1** → v25 (compression) is mandatory
- If **meta marginal contribution < +0.05 Sharpe** → v23 (drop meta) is mandatory and irrevocable
- If **one feature cluster dominates win-rate variance** → flag for v27 feature design
- If diagnostic surfaces something unexpected → **stop and report** before Phase 1

### Reporting

`docs/phase47/diagnostic_report.md` with:
1. The 7 cut tables/plots
2. Three explicit answers
3. Recommendation on whether to update Phase 1–5 ordering

**Stop and wait for owner review before starting Phase 1.**

---

## Phase 1 — v23: Drop MetaLabelModel, Replace with Isotonic Calibration

**Estimated effort:** ~30 min implementation, ~45 min walk-forward.
**Retrain required:** No (calibrator fits in seconds on validation set).

### Why this phase

R² = 0.001 says the meta-model is filtering at random. Whatever Sharpe lift came from Stage 1 → Stage 2 was the PM abstention gate. Replacing the meta-model with simple isotonic calibration of the base model gives the same filtering benefit (high-confidence selection) without the maintenance and retraining cost. **Run this before v24 so the XGBRanker switch doesn't tangle with meta-model assumptions.**

### Tasks

**Task 1.1 — Remove MetaLabelModel from the inference pipeline.**
Locate where MetaLabelModel.predict is called (likely in PM or RM). Replace with a no-op that passes base model scores through.

**Task 1.2 — Fit isotonic calibration on validation set.**

```python
from sklearn.isotonic import IsotonicRegression

# Use last 20% of training window as validation
calibrator = IsotonicRegression(out_of_bounds='clip').fit(val_probs, val_labels)
calibrated_p = calibrator.transform(test_probs)
```

**Task 1.3 — Replace meta-model gate with calibrated probability threshold.**

```python
# Threshold at 70th percentile of calibrated training probabilities
training_p70 = np.percentile(calibrator.transform(train_probs), 70)

# Entry rule
enter = (calibrated_p > training_p70) AND (PM_abstention_gate_passes)
```

**Task 1.4 — Walk-forward backtest with calibrated v23.**

Capture: Sharpe per fold, trade count per fold, win rate, exit mix, AUC (for sanity).

### Decision gate

Proceed to Phase 2 if:
- **Walk-forward avg Sharpe** stays within ±0.05 of v22 (+0.301), AND
- **Trade count** stays ≥ 100/fold

If Sharpe drops by more than 0.05, the meta-model was contributing more than diagnostic suggested. **Stop and investigate** — re-enable meta-model and proceed to Phase 2 with both stages.

### Things to watch for

- **System simplification.** One less model to maintain, retrain, debug. This is the primary win — Sharpe-neutral is success.
- **Threshold sensitivity.** If 70th percentile is too aggressive, test 60th and 80th. Tune on training portion only, never on test fold.
- **Calibration curve.** Plot reliability diagram (predicted P vs realized accept rate). Should be roughly diagonal post-calibration.

### Reporting

`docs/phase47/v23_results.md` with: per-fold metrics, calibration curve, threshold sensitivity table, recommendation on whether meta-model is permanently dropped.

**Stop and wait for owner review before starting Phase 2.**

---

## Phase 2 — v24: XGBRanker Daily Ranking Objective (HIGHEST IMPACT)

**Estimated effort:** ~20 min retrain + ~45 min walk-forward = ~65 min.
**Retrain required:** Yes (full retrain).
**Expected Sharpe lift:** +0.20 to +0.40.

### Why this phase

**This is the highest-leverage experiment in the plan.**

The Portfolio Manager is a daily cross-sectional ranker. Each day it scores all symbols, takes the top-5, and trades them. The current model is trained as a binary classifier predicting "is this row in the top 20% across all rows in the dataset?" — a fundamentally different optimization target.

Training as `XGBRanker` with `objective="rank:pairwise"` and `eval_metric="ndcg@5"` directly aligns the loss function with the trading decision. The model learns "rank today's stocks correctly, especially the top 5" rather than "classify each row."

This is similar in spirit to learning-to-rank in search/recommendations, where binary classification is known to underperform ranking objectives for top-K problems. Even if AUC barely moves, the top-5 selection quality should improve materially.

### Tasks

**Task 2.1 — Replace XGBClassifier with XGBRanker.**

```python
from xgboost import XGBRanker

ranker = XGBRanker(
    objective="rank:pairwise",
    eval_metric="ndcg@5",
    tree_method="hist",
    learning_rate=<from HPO>,
    max_depth=<from HPO>,
    n_estimators=<from HPO>,
    # Other hyperparameters from existing v22 HPO study
)
```

Reuse v22 hyperparameters as starting point. **Do not re-run Optuna in this experiment** — that's a second change.

**Task 2.2 — Group training examples by trade date.**

`XGBRanker` requires a `group` parameter — number of rows per ranking group. For daily cross-sectional ranking, the group is the trade date.

```python
# Sort training data by date, then compute group sizes
df_sorted = df_train.sort_values('trade_date')
groups = df_sorted.groupby('trade_date').size().tolist()

ranker.fit(X_train, y_train, group=groups)
```

**Task 2.3 — Use cross-sectional rank label.**

```python
# Within each trade date, rank stocks by realized path_quality
df['daily_pq_rank'] = (
    df.groupby('trade_date')['path_quality_score']
      .rank(pct=True)
)

# Ranker needs integer labels for relevance grades — bucket into 5 levels
df['rank_label'] = pd.qcut(
    df['daily_pq_rank'], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop'
).astype(int)
```

**Task 2.4 — Inference: score and select top-5 per day.**

Ranker outputs a relevance score (not a probability). Within each scoring date, take top-5 by score. Apply isotonic calibration from Phase 1 if you want a probability-like threshold for abstention; otherwise just rank-based selection.

**Task 2.5 — Track ranking-specific metrics during training.**

```python
# In addition to standard backtest metrics, capture:
daily_ndcg_at_5 = ndcg_score(y_true_per_day, y_pred_per_day, k=5)
top5_avg_path_quality = ...
top5_avg_realized_pnl = ...
score_spread_top5_vs_median = ...
```

**Task 2.6 — Walk-forward backtest with v24.**

Same 3-fold expanding-window structure. Capture full metric set + ranking metrics.

### Decision gate

Proceed to Phase 3 if:
- **Walk-forward avg Sharpe ≥ +0.45**, AND
- **No fold below −0.20**, AND
- **Trade count** ≥ 100/fold

If Sharpe lift is < +0.10, ranker isn't helping enough — investigate (group sizing wrong? label bucketing wrong?) before continuing. If Sharpe lifts but one fold collapses, regime sensitivity is concerning — note for Phase 6.

### Things to watch for

- **NDCG@5 trajectory during training.** Should be substantially higher than random (~0.5). If it plateaus near 0.5, the ranking signal is weak — model can't differentiate top-5 from rest.
- **Top-5 score spread.** If predicted scores for top-5 cluster very close to median, the model isn't producing differentiation. May need more trees or deeper trees.
- **Per-fold dispersion.** If Fold 2 (sideways) gains the most and Folds 1 & 3 (trending) gain little, the trending/sideways paradox is unresolved. Carry to v25.
- **Trade count behavior.** Ranker tends to produce slightly different daily candidate sets than classifier; trade count may shift even though selection size is fixed.

### Reporting

`docs/phase47/v24_results.md` with:
1. Per-fold Tier 3 metrics (vs v23 baseline)
2. NDCG@5 per fold, top-5 score spread distribution
3. Top-15 feature importance (compare to v22)
4. Per-fold exit mix
5. Recommendation: pass/fail gate, anything surprising

**Stop and wait for owner review before starting Phase 3.**

---

## Phase 3 — v25: Stop/Target Compression to 0.4×/0.8× Prior-Day Range

**Estimated effort:** ~20 min retrain + ~45 min walk-forward = ~65 min.
**Retrain required:** Yes (label regeneration + retrain).
**Expected Sharpe lift:** +0.15 to +0.35.

### Why this phase

If diagnostic shows time-exits dominate, the 1.2× prior-day range target is too far for a 2-hour hold. Compressing both stop and target to 0.4×/0.8× preserves the 2:1 R:R ratio but increases target-hit frequency. More target hits → higher realized R:R → directly closes the expectancy gap.

This addresses what the diagnostic likely confirmed: the system is bleeding Sharpe to time exits, not stop exits.

### Tasks

**Task 3.1 — Update simulator stop/target multipliers.**

```python
# Old:
STOP_MULT = 0.6
TARGET_MULT = 1.2

# New:
STOP_MULT = 0.4
TARGET_MULT = 0.8
```

**Task 3.2 — Regenerate path_quality labels with new multipliers.**

The `path_quality` formula depends on `stop_dist` and `target_dist`:

```python
upside_capture = min((max_high - entry) / (TARGET_MULT * prior_day_range), 1.0)
stop_pressure = min((entry - min_low) / (STOP_MULT * prior_day_range), 1.0)
close_strength = clip((final_close - entry) / (TARGET_MULT * prior_day_range), -1, 1)

path_quality = 1.0 * upside_capture - 1.25 * stop_pressure + 0.25 * close_strength
```

Recompute labels for the entire training set with new multipliers.

**Task 3.3 — Retrain v25 (XGBRanker from Phase 2 + new labels).**

Keep all other config identical to v24. Same hyperparameters, same features, same universe.

**Task 3.4 — Walk-forward backtest.**

### Decision gate

Proceed to Phase 4 if:
- **Walk-forward avg Sharpe** improves by ≥ +0.10 over v24, AND
- **Stop-hit rate** does not exceed 60% (guardrail — too tight stops will increase noise stops), AND
- **Trade count** stays ≥ 100/fold

If stop-hit rate climbs above 60%, fall back to **0.5×/1.0×** (mirroring swing model structure):

```python
STOP_MULT = 0.5
TARGET_MULT = 1.0
```

Re-run walk-forward with the fallback before deciding.

### Things to watch for

- **Exit mix shift.** Target should rise materially (was likely ~25% in v22, target ≥40% post-compression). Time-positive should fall.
- **Realized R:R.** Compute `avg_win / avg_loss` on actual trade results. Should approach 2:1 more closely than v22 did.
- **Per-fold response.** Trending folds (1 & 3) should benefit more if the time-exit hypothesis was correct — they had moves available but the target was unreachable in 2 hours.
- **Regime stability.** If Fold 2 (sideways) degrades more than Folds 1/3 improve, the structure trade-off isn't favorable.

### Reporting

`docs/phase47/v25_results.md` with: per-fold metrics, exit mix delta vs v24, realized R:R, regime-by-regime impact, recommendation.

**Stop and wait for owner review before starting Phase 4.**

---

## Phase 4 — v26: Top-300 by Liquidity Universe

**Estimated effort:** ~20 min retrain + ~45 min walk-forward = ~65 min.
**Retrain required:** Yes.
**Expected Sharpe lift:** +0.10 to +0.25.

### Why this phase

Russell 1000 contains ~500 names with materially worse intraday data quality — wider spreads, gappier 5-min bars, lower volume reliability for ORB and surge features. Top-300 by 20-day median dollar volume preserves enough breadth for daily top-5 selection while removing the noisy long tail.

**Why top-300 by liquidity, not SP-500:**
- Dynamic threshold adapts naturally to market changes
- No survivorship bias from static membership lists
- Avoids spending compute on names that get delisted or rotated
- Liquidity is the actual feature that matters for intraday execution, not index inclusion

### Tasks

**Task 4.1 — Build a daily liquidity-filtered universe.**

For each scoring date, the eligible universe is:
- Top 300 by trailing 20-day median dollar volume (close × volume)
- Price > $10 (filter out low-priced noise)
- Sparse coverage filter: ≥ 95% of expected 5-min bars present in trailing 20 days

Apply at both training time and inference time. **Important:** for training, use the universe membership *as of each historical training date*, not a single static snapshot. Avoids look-ahead bias.

**Task 4.2 — Retrain v26 with same architecture as v25 on filtered universe.**

Same XGBRanker config, same path_quality labels (with v25's compressed multipliers), same features.

**Task 4.3 — Walk-forward backtest.**

### Decision gate

Proceed to Phase 5 if:
- **Walk-forward avg Sharpe** improves by ≥ +0.05 over v25, AND
- **Trade count** stays ≥ 100/fold

If trade count drops below 100/fold, expand to top-500 by liquidity and re-run.

### Things to watch for

- **Trade count compression.** Smaller universe means fewer candidates per day. Monitor closely.
- **Per-row data quality.** Average 5-min bar count, average volume, average bid-ask proxy (high-low / midpoint) should all improve in the filtered universe.
- **Concentration risk.** Smaller universe may concentrate trades into mega-cap behavior (AAPL, MSFT, TSLA, etc.). Track sector and name concentration.
- **Calibration drift.** Probability distributions may shift on the cleaner universe — may need to refit isotonic calibrator.

### Reporting

`docs/phase47/v26_results.md` with: per-fold metrics vs v25, universe size statistics, concentration analysis, recommendation.

**Stop and wait for owner review before starting Phase 5.**

---

## Phase 5 — v27: Move-Quality + Relative-Strength Feature Pack

**Estimated effort:** ~2hr cache rebuild + ~20 min retrain + ~45 min walk-forward ≈ ~3 hours.
**Retrain required:** Yes (with feature schema bump).
**Expected Sharpe lift:** +0.10 to +0.25.

### Why this phase

Existing 42 features capture *what happened* in the first 60 minutes (range, momentum, volume, indicators) but not *how it happened* — the texture of price action that distinguishes clean institutional accumulation from noisy opening movement. For 2-hour long-only momentum trades, the model should know whether the first hour was smooth, persistent, supported by volume, and stronger than the market.

This is the most expensive experiment (cache rebuild) and has lower expected ROI per hour than Phases 2–4. Run last.

### Tasks

**Task 5.1 — Add raw features.**

```python
trend_efficiency_60m = abs(close_60m_return) / sum(abs(each_5m_return))
    # Smooth trend → near 1. Choppy → near 0.

momentum_quality_60m = price_momentum / realized_vol_60m
    # Sharpe-like ratio of 60-min movement.

green_bar_ratio_60m = count(close > open) / 12
    # Persistent buying pressure proxy.

above_vwap_ratio_60m = count(close > vwap) / 12
    # Holding above VWAP = institutional support proxy.

vwap_slope_60m = slope(vwap over first 12 bars)
    # Rising VWAP = accumulation; falling = distribution.

pullback_from_high_60m = (high_60m - last_close) / high_60m
    # How far off the high we're entering. Low = strong continuation setup.

range_expansion_vs_20d = current_60m_range / avg_60m_range_same_window_20d
    # Range expansion proxy. >1 = unusually wide. Volatility regime.

relative_strength_vs_spy = stock_60m_return - spy_60m_return
    # Outperformance vs market in opening hour.

relative_volume_x_momentum = volume_surge * price_momentum
    # Conviction interaction term. High volume * positive momentum = strong setup.

gap_followthrough = sign(gap_pct) * price_momentum
    # Did the gap continue or reverse? Positive = continuation.
```

**Task 5.2 — Add cross-sectional ranks.**

```python
cs_rank_relative_strength_vs_spy
cs_rank_trend_efficiency
cs_rank_momentum_quality
cs_rank_above_vwap_ratio
cs_rank_range_expansion
```

Computed at scoring time across the daily candidate universe.

**Task 5.3 — Bump feature schema version, invalidate SQLite cache.**

```python
# In feature schema config:
SCHEMA_VERSION = 'intraday_v3'  # was 'intraday_v2'
```

**Task 5.4 — Full feature rebuild on top-300 liquidity universe.**

~2 hours at 5-min resolution. Verify cache integrity post-build (row counts, NaN ratios, value distributions).

**Task 5.5 — Retrain v27 with same architecture as v26.**

XGBRanker, compressed stops/targets, top-300 liquidity universe, expanded feature set.

**Task 5.6 — Walk-forward backtest.**

### Decision gate

Proceed to Phase 6 (reassessment) regardless of result. This is the last planned experiment in Phase 47.

### Things to watch for

- **Feature importance migration.** Do new features displace old ones in top-15 importance? If yes, signal lift is real. If new features get zero importance, they're redundant with existing ones.
- **AUC-equivalent metrics.** Does NDCG@5 improve on validation? Does top-5 score spread widen?
- **Per-fold response.** Were trending folds the bottleneck? If so, `relative_strength_vs_spy` and `trend_efficiency` should help most there.
- **Diminishing returns flag.** If Sharpe lift is < +0.05, OHLCV ceiling has likely been reached — note for Phase 48 planning (microstructure data, fundamentals, alternative data).

### Reporting

`docs/phase47/v27_results.md` with: per-fold metrics, feature importance comparison, per-feature ablation if Sharpe lifts (which 2-3 features carried the lift?).

**Stop and wait for owner review before starting Phase 6.**

---

## Phase 6 — Reassessment & Decision

**Estimated effort:** Half day analysis + owner discussion.

### What to evaluate

After Phases 0–5 complete, write `docs/phase47/phase47_retrospective.md` with:

1. **Final walk-forward avg Sharpe** and per-fold breakdown across v22 → v23 → v24 → v25 → v26 → v27.
2. **Attribution table:** how much Sharpe lift did each phase contribute?
3. **Where the gate landed** vs the original +0.80 target.
4. **Trending vs sideways performance** — was the paradox resolved?
5. **Realized R:R** — closer to 2:1 than v22 was?
6. **Top-15 feature importance** in final v27 — what does the model actually trade on?

### Decision branches

| Outcome | Recommendation |
|---|---|
| **Avg Sharpe ≥ +0.80, no fold < −0.30** | Pass to paper trading. Document the winning configuration as the new intraday baseline. |
| **Avg Sharpe +0.60 to +0.80, no fold < −0.30** | Lower the gate to +0.60 with documented rationale (data fidelity ceiling). Pass to paper trading at half size for 60 days. |
| **Avg Sharpe +0.45 to +0.60** | Paper trade at half size for 60-day evaluation. Use live data to inform whether to push for higher gate or accept as a smaller portfolio sleeve. |
| **Avg Sharpe < +0.45** | Escalate to Phase 48 planning. Likely candidates: regime-conditional models, microstructure data sources (L2, news/event), or accepting intraday as not viable on current data fidelity. |

### Things to learn for future phases

Capture observations across all 6 phases that feed Phase 48 planning regardless of outcome:

1. **Which phase contributed the most Sharpe?** This tells us where to invest first in Phase 48 if needed.
2. **Was the trending/sideways paradox resolved?** If yes, by which change? If no, regime-conditional modeling is the natural next step.
3. **Where did walk-forward variance come from?** If one fold dominates the variance, that fold's regime should be studied as a generalization risk.
4. **Did simplifying (drop meta) help or hurt?** Lesson for system architecture going forward.
5. **Calibration stability.** Does isotonic calibration remain stable across folds, or does it drift?
6. **Feature ablation insights from v27.** Which 2–3 features actually moved the needle? Future feature engineering should prioritize that family.

---

## 4. Reporting & Logging Standards

For every backtest run during this phase, persist to `runs/phase47/{version}/{timestamp}/`:

- `config.json` — full hyperparameter and structure config
- `metrics.json` — all Tier 3 metrics (per-fold and aggregate)
- `feature_importance.csv` — top-30 features with gain/cover/freq
- `predictions.parquet` — one row per (date, symbol, prediction) for the test set
- `trade_log.parquet` — full Tier 3 trade history with entry/exit reasons
- `mae_mfe_distribution.png` — histograms of MAE and MFE across all trades
- `walk_forward_summary.md` — human-readable summary with per-fold Sharpe table
- For ranking phases: `ndcg_per_fold.json`, `top5_score_spread.png`

Every model artifact must be tagged with:
- Git SHA at training time
- Parent model version
- Label config (multipliers, formula weights)
- Universe filter
- Feature schema version
- Hyperparameter source (HPO study ID or "inherited from vXX")

This lineage is non-negotiable. Without it, attribution becomes guesswork.

---

## 5. Decision Tree (after each phase)

```
Phase 0 — Diagnostic Dive
├── All three answers as expected → proceed to Phase 1 unchanged
├── Reversion-flavored OR poor R:R → emphasize v25 over v24
└── Surprising finding → STOP and discuss with owner

Phase 1 — v23 (drop meta)
├── Sharpe stays within ±0.05 of v22 → proceed, drop meta permanently
└── Sharpe drops > 0.05 → revert, proceed with both stages to Phase 2

Phase 2 — v24 (XGBRanker)
├── Avg Sharpe ≥ +0.45 → proceed to Phase 3
├── Avg Sharpe lifts but < +0.45 → investigate ranking config, may proceed or iterate
└── Sharpe flat or worse → STOP, investigate group sizing / label bucketing

Phase 3 — v25 (compression)
├── Avg Sharpe ≥ +0.55 AND stop-hit < 60% → proceed to Phase 4
├── Stop-hit ≥ 60% → fall back to 0.5×/1.0×, re-run
└── Sharpe drops → revert, document why compression didn't help

Phase 4 — v26 (top-300 liquidity)
├── Avg Sharpe ≥ +0.60 AND trade count ≥ 100/fold → proceed to Phase 5
├── Trade count < 100/fold → expand to top-500, re-run
└── Sharpe drops → revert universe, proceed to Phase 5 on full universe

Phase 5 — v27 (features)
└── Run unconditionally; final phase before reassessment

Phase 6 — Reassessment
└── See decision branches table in Phase 6 section
```

---

## 6. Glossary

- **Tier 1/2/3 backtest:** Three levels of fidelity. Tier 1 = signal-only. Tier 2 = signal + simple execution. Tier 3 = full agent on historical bars. **Tier 3 is the only metric for go/no-go.**
- **Walk-forward:** 3-fold expanding-window time-based out-of-sample evaluation. Folds: 1 = Oct '24–Apr '25, 2 = Apr '25–Oct '25, 3 = Oct '25–Apr '26.
- **Path_quality score:** Continuous label combining upside capture, stop pressure, and close strength. Used for cross-sectional ranking labels.
- **NDCG@5:** Normalized Discounted Cumulative Gain at 5. Standard ranking metric — 1.0 = perfect ranking of top-5, 0.5 ≈ random.
- **MFE / MAE:** Maximum Favorable / Adverse Excursion. Path statistics per trade.
- **R:R:** Reward-to-risk ratio. Stated as target_distance / stop_distance.
- **PM abstention gate:** VIX ≥ 25 OR SPY < MA20 → skip new entries that day.
- **Cross-sectional rank:** Within-day rank across the daily candidate universe, normalized to [0, 1].
- **NDCG group:** In XGBRanker, the unit within which ranking happens. Here: a single trade date.

---

## 7. Out of Scope for Phase 47

These are deferred to Phase 48 or later:

- **Multi-timeframe features (15m/30m aggregations).** Dimensionality with low marginal info on 2-hour holds.
- **Regime-conditional mixture of experts.** Powerful but complex; only if Phase 47 doesn't close the gap.
- **Pre-market gap / overnight catalyst features.** Requires news/event data outside current stack.
- **Different entry timing (30m / 90m).** Probably 2nd-order. Re-evaluate after Phase 5.
- **Different model architecture (NN, LSTM, transformer).** Bottleneck is not architectural.
- **Sector-relative strength features beyond the v27 set.** Single feature was added; full re-engineer deferred.
- **Calibration as a permanent multi-fold layer.** Phase 47 fits per-fold; Phase 48 may integrate properly.
- **L2 / order book / tick data.** Different data acquisition project.
- **Shorting / pair trading.** Operationally complex with $20k account; defer until swing model is in production.

If during Phase 47 you find yourself wanting to do any of these, **stop and ask** — they require their own planning.

---

## 8. Final Notes

- **One change per retrain.** Hard rule. If you discover a bug during Phase 3, fix it as a hotfix (not as part of v25) and document the diff.
- **Preserve all artifacts.** v22 stays untouched. Every new version (v23–v27) gets its own artifact directory. Lineage required for attribution.
- **Tier 3 walk-forward is the only metric that matters for gating decisions.** AUC, profit factor, win rate are diagnostic, not decisional.
- **The diagnostic dive is mandatory.** Skipping it means optimizing on incomplete information. The 7 cuts take half a day and reshape priorities.
- **When in doubt, stop and report.** Especially before Phase 5 (the cache rebuild is a 2-hour irreversible step). Confirm direction first.
- **Don't anchor on the +0.80 gate.** It may not be achievable on OHLCV-only 5-min data. Phase 6 explicitly allows lowering the gate with documented rationale. The swing model carries the portfolio either way.

End of plan.
