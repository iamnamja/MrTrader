# MrTrader ML Improvement — LLM-Friendly Handoff Brief

## Purpose

This document is a clean, LLM-friendly version of the recommended next steps for improving the MrTrader swing trading ML model.

It is intended to be pasted into another LLM, coding agent, or autonomous development agent as implementation guidance.

---

## Core Diagnosis

The current v110 model is not truly a trade-entry model.

It is better understood as a cross-sectional volatility opportunity model.

The model can rank stocks better than random, but it is ranking the wrong object for the actual Tier 3 trading system.

### Why v110 is stuck

The current model is trained to answer this question:

> Which stocks are likely to rank in the top 20% of 5-day Sharpe-adjusted forward returns?

But the Tier 3 trader needs this question answered:

> Which stocks can be bought at the next open, avoid an early 0.5× ATR stop, and ideally reach a 1.5× ATR target before time exit?

Those are different problems.

The current label is a 5-day cross-sectional ranking label, but the actual system usually exits within roughly 1–2 bars. This creates a serious label-to-exit mismatch.

The model’s top features are volatility-driven, which suggests it has learned to select stocks with high movement potential rather than stocks with favorable entry-path quality.

### Primary failure mode

The model appears to select high-volatility names that sometimes generate strong 5-day returns, but these names also suffer large adverse movement shortly after entry.

Because the stop is only 0.5× ATR, many trades are stopped out before the 5-day signal has time to work.

### Summary diagnosis

The 70% stop-exit issue is likely caused by a combination of:

1. Label problem: 5-day forward labels do not match 1–2 day realized holding behavior.
2. Volatility bias: cross-sectional top-20% labels favor high-volatility stocks.
3. Entry timing problem: the model does not know whether the next open is a bad entry.
4. Stop structure problem: 0.5× ATR may be too tight for the selected universe and signal type.

Do not focus on improving AUC from here. Focus on improving Tier 3 path quality and reducing early stopouts.

---

# Top 3 Highest-Priority Experiments

## Experiment 1 — Replace the 5-Day Forward Label With a 2-Day Path-Quality Cross-Sectional Label

### Goal

Keep the relative cross-sectional ranking approach that worked, but make the label match how the Tier 3 trader actually exits.

Do not return to full-universe triple-barrier classification. That already failed.

Instead, create a continuous path-quality score and then rank stocks cross-sectionally.

### Current label

The current label is approximately:

```python
forward_score = sharpe_adjusted_5d_forward_return
label = 1 if forward_score in top_20_percent_cross_section else 0
```

### Proposed label

Use next-open entry and evaluate the next 2 trading days.

For each stock and signal date:

```python
entry = next_open
atr = atr_14_at_signal_date

mfe_2d = max(high_day1, high_day2) - entry
mae_2d = entry - min(low_day1, low_day2)
close_ret_2d = close_day2 - entry
```

Then compute a path-quality score:

```python
upside_capture = min(mfe_2d / (1.5 * atr), 1.0)
stop_pressure = min(mae_2d / (0.5 * atr), 1.0)
close_strength = close_ret_2d / atr

path_quality_score = (
    1.00 * upside_capture
    - 1.25 * stop_pressure
    + 0.25 * close_strength
)
```

Then rank the score cross-sectionally by date/window:

```python
label = 1 if path_quality_score in top_20_percent else 0
```

### Why this is better

This keeps the working idea:

> Relative ranking is easier than predicting absolute direction.

But it changes the target from:

> Best 5-day return.

To:

> Best near-term trade path after next-open entry.

This is much closer to the Tier 3 system.

### Suggested label mode name

```python
label_mode = "path_quality_2d_cs_top20"
```

### Where to change

Search for the existing label construction code:

```powershell
Select-String -Path .\src\**\*.py -Pattern "top.*20|forward_window|sharpe|scale_pos_weight|label"
```

Likely locations may include files such as:

```text
src/.../training/label_builder.py
src/.../ml/train_pm_model.py
src/.../features/build_training_dataset.py
```

Do not overwrite v110. Add this as a new label mode.

### Expected outcome

AUC may decline. That is acceptable.

The goal is Tier 3 improvement.

Target outcomes:

```text
Stop exits: 70% -> below 60–62%
Tier 3 Sharpe: +0.34 -> +0.60 to +0.90
Walk-forward: at least one fold turns positive
Trade count: should remain meaningfully active
```

---

## Experiment 2 — Make the Cross-Sectional Label Volatility-Neutral

### Goal

Prevent the model from learning:

> Pick the highest-volatility stock.

Instead, make it learn:

> Among stocks with similar volatility, which one has the best path-quality setup?

### Preferred implementation: rank within volatility buckets

For each date/window, create volatility buckets:

```python
vol_bucket = pd.qcut(
    df["atr_norm_or_realized_vol_20d"],
    q=5,
    labels=False,
    duplicates="drop"
)
```

Then rank the path-quality score inside each date and volatility bucket:

```python
df["rank_within_vol_bucket"] = (
    df.groupby(["date", "vol_bucket"])["path_quality_score"]
      .rank(pct=True)
)

df["label"] = (df["rank_within_vol_bucket"] >= 0.80).astype(int)
```

### Suggested label mode name

```python
label_mode = "path_quality_2d_vol_bucketed_cs_top20"
```

### Why this is better

The model can still use volatility information, but volatility no longer dominates the label.

This should reduce the chance that the model is merely learning a volatility-rank screen.

### Alternative implementation: residualize target against volatility

If bucket-ranking is not enough, residualize the raw target against ex-ante volatility features.

Example volatility features:

```python
vol_neutralization_features = [
    "realized_vol_20d",
    "atr_norm",
    "parkinson_vol",
    "vol_of_vol",
    "vol_percentile_52w",
]
```

Fit a simple cross-sectional model per date/window:

```python
target_raw = path_quality_score
target_predicted_by_vol = model.predict(vol_neutralization_features)
target_residual = target_raw - target_predicted_by_vol
```

Then label the top 20% by residual:

```python
label = 1 if target_residual in top_20_percent_cross_section else 0
```

However, start with volatility buckets first. They are easier to reason about and less fragile.

### Expected outcome

Healthy signs:

```text
Importance of realized_vol_20d declines
Importance of atr_norm declines
Directional, entry, reversal, or structure features gain importance
Stop exits decline
Tier 3 improves even if AUC declines
```

---

## Experiment 3 — Add a Candidate-Level Early-Stopout Meta-Model

### Goal

Do not ask a model to predict all stocks.

Ask a narrower question:

> Given that the PM already likes this stock, is this specific entry likely to stop out early?

This is different from the failed triple-barrier experiment.

Triple-barrier classification tried to predict absolute success/failure for the full universe.

This meta-model only trains on PM candidates that would actually be considered for trading.

### Training set

Run historical PM scoring.

For every candidate that reaches the PM candidate pool, create one training row:

```python
candidate_date
symbol
pm_score
entry_open
features_at_signal_date
entry_context_features
```

Then label early stopout:

```python
early_stopout = 1 if low_day1_or_day2 <= entry - 0.5 * atr else 0
```

Train the model to predict acceptance:

```python
accept_label = 1 if early_stopout == 0 else 0
```

### Important features

Add entry-context features that specifically capture bad next-open entry timing.

```python
gap_at_entry = (next_open / prior_close) - 1
gap_at_entry_atr = (next_open - prior_close) / atr

open_location_in_prior_range = (
    next_open - prior_low
) / max(prior_high - prior_low, small_number)

prior_day_range_atr = (prior_high - prior_low) / atr

prior_day_close_location = (
    prior_close - prior_low
) / max(prior_high - prior_low, small_number)

distance_from_ema20_at_entry = (next_open - ema20) / atr
distance_from_prior_high = (next_open - prior_high) / atr
distance_from_prior_low = (next_open - prior_low) / atr

spy_gap_at_entry = (spy_next_open / spy_prior_close) - 1
spy_2d_return_prior = spy_close / spy_close_2d_ago - 1

sector_gap_at_entry = (sector_etf_next_open / sector_etf_prior_close) - 1
```

These features are not generic technical indicators.

They answer:

> Is the next open a bad entry price?

### Suggested files

Add this as a separate component rather than modifying the PM model directly.

Possible files:

```text
src/.../ml/build_candidate_meta_dataset.py
src/.../ml/train_stopout_meta_model.py
src/.../risk/stopout_meta_filter.py
```

Then integrate into RM validation:

```python
if stopout_meta_score < accept_threshold:
    reject_candidate("early_stopout_risk")
```

### Threshold guidance

Do not overfit the threshold to full-sample Sharpe.

Start with simple thresholds:

```text
Accept if P(no_early_stopout) >= 0.45
Stricter test: >= 0.50
Very strict: >= 0.55
```

Avoid 0.60+ initially because the existing model already has tightly clustered probabilities.

### Expected outcome

Target outcomes:

```text
Trades: down 15–35%
Stop exits: 70% -> 55–62%
Win rate: modestly higher
Sharpe: higher if trade count does not collapse
```

If trade count falls by more than 50%, threshold is too strict.

---

# Longer-Term Ideas

## 1. Stop/Target Calibration With Nested Walk-Forward

The current stop/target structure may be too aggressive.

Current structure:

```text
Stop: 0.5× ATR
Target: 1.5× ATR
Risk/reward: 3:1
```

This is elegant theoretically, but the stop may be too tight for the selected high-volatility names.

Test a small grid:

```python
stop_mults = [0.5, 0.75, 1.0]
target_mults = [1.0, 1.25, 1.5, 2.0]
max_holds = [2, 3, 5]
```

Use nested walk-forward only.

For each fold:

1. Use train/validation segment to choose stop/target.
2. Freeze the selected parameters.
3. Evaluate on the OOS fold.

Do not choose the best full-sample result.

A plausible better configuration may be:

```text
Stop: 0.75× ATR
Target: 1.25× to 1.5× ATR
Max hold: 2–3 days
```

---

## 2. Add Entry-Quality Features

Do not add more generic indicators.

Add features that describe whether the next open is a good or bad entry.

Potential features:

```python
next_open_gap_atr
open_vs_prior_close_atr
open_vs_prior_high_atr
open_vs_prior_low_atr
open_location_in_prior_day_range
prior_day_close_location
prior_2d_high_break_distance_atr
prior_2d_low_distance_atr
distance_from_5d_vwap_proxy_atr
distance_from_ema20_at_entry_atr
overnight_gap_followthrough_20d
avg_gap_fill_rate_60d
```

Two especially useful features:

```python
gap_fill_rate_60d = percentage of past 60 gaps that filled intraday

gap_followthrough_rate_60d = percentage of past 60 gaps that continued in the gap direction
```

Some stocks gap and continue.

Others gap and mean-revert.

The current model likely does not distinguish these enough.

---

## 3. Sector-Relative and Beta-Relative Labels

Instead of ranking the entire SP-500 together, rank within comparable groups.

Options:

```python
sector_rank = rank target within GICS sector
beta_bucket_rank = rank target within beta quintile
```

Possible label:

```python
label = top 20% within sector
```

Or:

```python
label = top 20% within sector and not bottom 40% market-relative strength
```

Reason:

A high-volatility tech stock and a defensive utility should not be ranked as if they are directly comparable.

---

## 4. Regression Target Instead of Binary Top-20 Label

Keep the cross-sectional idea, but use a smoother target.

Instead of binary classification:

```python
label = 1 if top_20_percent else 0
```

Try XGBoost regression:

```python
objective = "reg:squarederror"
target = cross_sectional_percentile(path_quality_score)
```

This gives the model more information than a hard top-20 cutoff.

At inference, rank by predicted percentile.

This may be more stable than binary labels.

---

## 5. Regime-Conditional Model Routing

After the label is fixed, consider training specialized models:

```text
Model A: low-vol market and SPY above EMA200
Model B: high-vol market and SPY above EMA200
Model C: SPY below EMA200
```

At inference, route to the correct model based on the live market regime.

Do this only after fixing the label.

Regime-routing a bad target creates multiple bad models.

---

## 6. Conservative Recency Weighting

If 2021–2023 data is hurting current generalization, use time decay.

Do not use overly aggressive recency weighting.

Start with a 2-year or 3-year half-life:

```python
sample_weight_time = 0.5 ** (age_days / half_life_days)
```

Test:

```python
half_life_days = 504  # roughly 2 trading years
half_life_days = 756  # roughly 3 trading years
```

Cap weights:

```python
sample_weight = clip(sample_weight_time, 0.35, 1.00)
```

This gives recent data more influence without letting the most recent months dominate.

Avoid 126-day or 252-day half-life initially.

---

# Things To Avoid

## Avoid broad feature pruning

Zero XGBoost importance does not mean zero trading utility.

Previous pruning hurt Tier 3 performance.

## Avoid another generic indicator dump

The model already has RSI, MACD, Stoch, ADX, Bollinger, CCI, Williams %R, WQ alphas, pressure dynamics, VWAP distance, sector momentum, and VIX interactions.

The missing signal is likely entry-path quality, not another oscillator.

## Avoid high confidence thresholds

Do not use probability thresholds above 0.60.

The model’s probabilities cluster too tightly and this will likely kill trade count.

## Avoid full-universe triple-barrier classification for now

It matches the trade setup in theory, but prior tests showed the current feature set cannot solve this absolute classification problem.

## Avoid inference filters tuned on the same period

The Phase 34/35 filters looked excellent in-sample but collapsed out of sample.

Any new filter must be selected using nested walk-forward validation.

## Avoid jumping straight to transformers

Temporal fusion transformers or neural ranking models may overfit with only 5 years of daily data across several hundred names.

Use them later, not now.

---

# Wild Card Idea

## Add a Trade/No-Trade Abstention Layer

The system may be forcing trades when no strong opportunity exists.

Build a daily opportunity-quality model or rule layer.

Potential daily features:

```python
daily_edge_dispersion = top_5_pm_score_mean - median_pm_score

top_score_z = (
    top_score - rolling_mean_top_score_63d
) / rolling_std_top_score_63d

candidate_agreement = percentage of top candidates passing RM rules

market_noise = SPY_intraday_range_5d / SPY_ATR_20d

cross_sectional_vol_dispersion = std(stock_returns_5d across universe)
```

Only allow new entries when:

```python
top_score_z > 0
daily_edge_dispersion > rolling_50th_percentile
market_noise is not extreme
```

This is not a stock-selection model.

It is a trade/no-trade gate.

The biggest Sharpe improvement may come from trading less, not from selecting better stocks.

---

# Recommended Next Sequence

Run the next experiments in this order:

```text
v119: 2-day path-quality cross-sectional label

v120: same as v119, but top-20% ranked within volatility buckets

v121: candidate-level early-stopout meta-model added to Risk Manager
```

---

# Success Criteria

Primary criteria:

```text
Walk-forward average Sharpe improves materially
No fold below -0.3
Stop exits fall below 60–62%
```

Secondary criteria:

```text
Profit factor > 1.20
Trade count remains above roughly 175 over 2 years
Average hold remains realistic
Top feature importance is no longer dominated entirely by volatility
```

---

# Key Instruction For Future Agents

Do not optimize for AUC from here.

Optimize for Tier 3 walk-forward behavior.

The next model should rank the thing the trader actually monetizes:

> favorable next-open entry path, low early-stop risk, and enough upside excursion to justify the position.
