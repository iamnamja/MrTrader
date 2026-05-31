# MrTrader Phase 47 — Intraday Model Analysis & Recommendations

**Date:** 2026-04-26
**Author:** Quant research review (second-opinion analysis)
**Audience:** Min Kim, MrTrader system development
**Current state:** Intraday v22 at +0.301 walk-forward avg Sharpe (gate: +0.80)
**Gap to close:** +0.50 Sharpe

---

## 1. Executive Summary

The Phase 46 work was directionally correct and unblocked the system from broken (−1.16 Sharpe) to functional (+0.30 Sharpe). However, three things in the current framing of Phase 47 are underweighted, and addressing them changes the priority ordering of the next experiments:

1. **The trending-vs-sideways paradox** — Fold 2 (sideways market) outperforms Folds 1 & 3 (trending markets) for an ostensible momentum strategy. This is the single most important data point in the brief and should be diagnosed before any retrain.

2. **The expectancy math understates the gap** — The "51% × 1.2 − 49% × 0.6" calculation assumes wins = full target and losses = full stop. With a 2-hour hold and time exits, realized R:R is almost certainly worse than 2:1. Without an exit-type breakdown, the +0.50 Sharpe gap may be larger than stated.

3. **The meta-model probably contributes nothing** — R²=0.001, corr=0.044 is statistical noise. Whatever lift Stage 1 → Stage 2 provided almost certainly came from the PM abstention gate alone. Verifiable in 30 minutes.

This document recommends running a **diagnostic dive first** (no compute, half day of pandas analysis) to settle these questions, then a focused 4-experiment sequence to close the Sharpe gap.

**Recommended order of operations:**

1. **Diagnostic dive** (free, half day) — settles momentum vs reversion, true R:R, and meta-model attribution
2. **v25 — Drop meta-model, replace with isotonic calibration** (cheap, ~45 min)
3. **v23 — Universe shrink to SP-500** (1 retrain, ~65 min)
4. **v24 — Reduce stop_pressure label weight from 1.25× to 0.75×** (1 retrain, ~65 min)
5. **v26 — OHLCV microstructure proxy features** (cache rebuild ~2hr + retrain)
6. **Reassess gate** — if at +0.55 to +0.65, consider lowering the gate or paper-trading at half size

---

## 2. Three Reframings (Read First)

### 2.1 The trending-vs-sideways paradox is structurally important

| Fold | Period | Regime | Sharpe |
|---|---|---|---|
| 1 | Oct 2024 – Apr 2025 | Trending | +0.24 |
| 2 | Apr 2025 – Oct 2025 | Sideways/choppy | **+0.57** |
| 3 | Oct 2025 – Apr 2026 | Trending | +0.23 |

ORB momentum strategies are mathematically supposed to do the *opposite* — better in trending, worse in choppy. With 252 trades per fold, this is not noise. It's a structural finding the brief mentions but doesn't fully internalize.

The most likely explanation: **despite the ORB framing, the model has learned something closer to "fade extended early-session moves" than "ride the breakout."** That's a fundamentally different strategy than what's intended, and the implications cascade:

- Many of the 42 features are calibrated to capture momentum/breakout patterns. If the actual edge is reversion-flavored, a substantial fraction may be adding noise rather than signal.
- The label weighting (`−1.25 × stop_pressure`) heavily penalizes trades that touched the stop. In trending markets, real momentum trades pull back through the stop before continuing. **That term may be systematically excluding the trades that would actually win in trending regimes.**
- The 60-min feature window may be too long if the edge is reversion — by bar 12, the extension that would have been faded is already half-resolved.

**Until we understand whether v22 is doing momentum or reversion, every feature/label change is partially a guess.** The diagnostic dive in Section 3 settles this.

### 2.2 Expectancy math is more unforgiving than stated

The brief calculates:
```
Expectancy = 0.51 × 1.2 − 0.49 × 0.6 = +0.318 per unit
```

This assumes wins capture full target (1.2× ATR) and losses are exactly at stop (0.6× ATR). In practice with a 2-hour hold and time exit at bar 24:

- Some "wins" are time-exit trades that close green at bar 24 — often at +0.3× to +0.5× ATR, not +1.2× ATR
- Some "losses" are time-exit trades that close red — often at −0.2× to −0.4× ATR, not −0.6× ATR
- The realized R:R is almost certainly closer to **1.2:1 or 1.4:1**, not 2:1

If, say, 30% of "wins" are time-exits averaging +0.4× ATR, the expectancy collapses meaningfully:
```
Adjusted expectancy ≈ 0.51 × (0.7 × 1.2 + 0.3 × 0.4) − 0.49 × (0.7 × 0.6 + 0.3 × 0.3)
                    = 0.51 × 0.96 − 0.49 × 0.51
                    = 0.490 − 0.250 = +0.240 per unit
```

That's 25% smaller than stated, and the Sharpe gap to +0.80 widens correspondingly.

**Run the exit-type breakdown before changing anything.** It tells you whether the gap is "improve win rate" (push AUC) or "close trades earlier when winning" (improve exit logic).

### 2.3 The meta-model attribution is suspect

Stage 1 → Stage 2 numbers:

| Stage | Trades | Win % | Sharpe |
|---|---|---|---|
| 1 (no meta, no abstention) | 252 | 48.8% | −0.292 |
| 2 (+meta, +abstention) | 175 | 51.1% | +0.301 |

The lift was attributed to the combination of MetaLabelModel + PM abstention gate. But:

- MetaLabel R² = 0.001, correlation = 0.044 — statistically indistinguishable from random
- A model with corr=0.044 cannot meaningfully filter trades; it filters approximately at random
- Random filtering of 252 → 175 trades **does not improve Sharpe** — it just reduces sample size

**The entire +0.59 Sharpe lift almost certainly came from the PM abstention gate.** This is a 30-minute test to confirm: run Stage 1 + abstention only (no meta) and compare. If Sharpe is ≈+0.30, the meta-model is added complexity contributing nothing.

This matters because the meta-model is a maintenance burden, a retraining cost, and a source of debugging confusion. If it's not working, removing it is pure simplification.

---

## 3. Diagnostic Dive (run before any retrain)

**Estimated effort:** Half day of pandas analysis, no compute beyond loading existing trade logs.

This phase produces no model artifacts. It produces the answers needed to choose between subsequent experiments.

### 3.1 Required cuts on existing v22 trade logs

| # | Cut | What it tells you |
|---|---|---|
| 1 | Win rate by ORB direction (entry direction vs ORB breakout direction) | Confirms momentum vs reversion. If win rate is higher when entry direction *opposes* ORB direction → reversion. If aligned → momentum. |
| 2 | Exit-type breakdown: % target-hit / % stop-hit / % time-positive / % time-negative | Reveals true realized R:R. Critical for sizing the actual gap. |
| 3 | Win rate by VIX bin (< 15, 15–25, ≥ 25) and by SPY-vs-MA20 (above / below) | Quantifies the abstention gate's marginal contribution. |
| 4 | Win rate stratified by `cs_rank_momentum` quintile and `orb_position` decile | Reveals which features actually carry the edge. |
| 5 | Sharpe with PM abstention only (no meta) vs full stack | Settles the meta-model attribution question. |
| 6 | MFE / MAE distribution by exit type | Are winners exiting early? Are losers showing big MFE before reversing? |
| 7 | Per-fold win rate, exit mix, and avg trade R | Identifies which fold/regime the system fails on most. |

### 3.2 Deliverables

A single notebook or markdown report with:
- The 7 cuts above as tables/plots
- Three explicit answers:
  1. **Momentum or reversion?** (cut #1)
  2. **What's true realized R:R?** (cut #2)
  3. **Does meta-model contribute marginal Sharpe?** (cut #5)
- A short recommendation section that updates the experiment ordering below if any answer is surprising

### 3.3 Decision branches based on diagnostic

- If reversion-flavored → v24 (label weight) becomes lower priority, but a feature audit becomes higher priority
- If true R:R is ≤ 1.4:1 → exit logic redesign becomes a candidate experiment (currently not in the top 4)
- If meta-model contributes < +0.05 Sharpe → v25 becomes mandatory, not optional
- If one feature cluster dominates win-rate variance → feature selection / pruning becomes a candidate

---

## 4. Top 4 Experiments

Each is one isolated change. Do not stack changes within a single retrain.

### 4.1 v25 — Drop meta-model, replace with isotonic calibration

**Order:** Run first (cheapest, no model retrain needed)
**Effort:** ~30 min implementation, ~45 min walk-forward

**Hypothesis:**
MetaLabelModel R² = 0.001 means the meta-model is not adding signal. The Stage 1 → Stage 2 lift came from the PM abstention gate. Replacing the meta-model with simple probability calibration of the base model gives the same filtering benefit (high-confidence selection) without the complexity.

**Implementation:**
```python
from sklearn.isotonic import IsotonicRegression

# Fit on validation set (last 20% of training window)
calibrator = IsotonicRegression(out_of_bounds='clip').fit(val_probs, val_labels)
calibrated_p = calibrator.transform(test_probs)

# Threshold at 70th percentile of calibrated training probabilities
training_p70 = np.percentile(calibrator.transform(train_probs), 70)

# Entry rule
enter = (calibrated_p > training_p70) AND (PM_abstention_gate_passes)
```

No base model retrain required — calibration fits on validation set in seconds. Walk-forward only.

**Expected impact:** Neutral to mildly positive (+0.00 to +0.05 Sharpe). The real win is **simplification**: one less model to maintain, debug, and retrain.

**Risk:** Minimal. If it underperforms by >+0.05 Sharpe, revert — the meta-model is pulling its weight after all.

**What we learn:** Whether the meta-model architecture is worth investing in further, or whether the system simplifies cleanly without it.

---

### 4.2 v23 — Universe shrink to SP-500 (highest-leverage structural change)

**Order:** Run second
**Effort:** ~20 min retrain + ~45 min walk-forward = ~65 min

**Hypothesis:**
Russell 1000 contains ~500 names with materially worse intraday data quality — wider spreads, gappier 5-min bars, lower volume reliability for ORB and surge features. Below mid-cap, intraday signal-to-noise on 5-min OHLCV degrades sharply. SP-500 cuts that tail. The swing model's success on SP-100 is a strong prior that bigger-cap intraday data is meaningfully cleaner.

**Implementation:**
- Add point-in-time SP-500 membership filter at each scoring date
- **Critical:** use historical membership, not current. Static current-membership would introduce survivorship bias.
- Same training window, same features, same labels, same model config
- One retrain

**Expected impact:** +0.10 to +0.25 Sharpe.

AUC may not move much — fewer rows but cleaner ones. The lift comes from realized win rate (cleaner price action means features map more reliably to outcomes) and lower transaction-cost bleed on marginal trades.

**Risk:**
- Trade count compression. With ~500 candidates instead of ~766 and stringent RM filters, trade count may drop below 100/fold.
- Watch trade count as a guardrail metric. If it falls below 100/fold, consider relaxing RM filters or expanding to SP-500 + selected high-volume Russell 1000 names.

**What we learn:** Whether data quality vs feature engineering is the binding constraint. If shrinking universe helps materially, it de-risks all future feature work because we'll be working with cleaner inputs.

---

### 4.3 v24 — Reduce stop_pressure label weight from 1.25× to 0.75×

**Order:** Run third
**Effort:** ~20 min retrain + ~45 min walk-forward = ~65 min

**Hypothesis:**
With noisy 5-min data and a 0.6× ATR stop, normal price action will frequently touch the stop level on legitimate trades that recover. The 1.25× penalty makes the label strongly prefer trades that *don't even test the stop* — biasing toward low-vol, low-momentum candidates. This may directly explain the trending-fold underperformance: trending winners almost always test the stop on initial pullback.

**Implementation:**
```python
# Old:
score = 1.00 * upside_capture - 1.25 * stop_pressure + 0.25 * close_strength

# New:
score = 1.00 * upside_capture - 0.75 * stop_pressure + 0.25 * close_strength
```

One retrain. No feature changes, no architecture changes.

**Expected impact:** +0.10 to +0.20 Sharpe, **potentially concentrated in Folds 1 & 3** (trending), which is exactly where the gap is.

Win rate may not change much; the *distribution* of wins should shift toward larger MFE captures, raising realized R:R.

**Risk:**
- Could pick more trades that genuinely stop out → watch realized stop-hit rate as a guardrail
- If stop-hit rate climbs from 51% to >58% without a compensating increase in target hits, the change overcorrected — revert to 1.0× as a middle ground

**What we learn:** Whether label structural calibration is the bottleneck for trending-regime performance. Direct test of the trending-vs-sideways paradox hypothesis.

**Conditional on diagnostic:** If the diagnostic in Section 3 shows the model is reversion-flavored (not momentum), this experiment moves down in priority — the stop_pressure penalty is then doing what it should.

---

### 4.4 v26 — OHLCV microstructure proxy features

**Order:** Run fourth (most expensive, lowest expected ROI per hour)
**Effort:** Cache invalidation + ~2hr feature rebuild + ~20 min retrain + ~45 min walk-forward

**Hypothesis:**
The 42 existing features capture *what happened* in the first 60 minutes (range, momentum, volume, indicators) but not *how it happened* — the texture of order flow you can infer from 5-min OHLCV. Six new features try to capture this:

```python
# Conviction proxies
body_ratio_60m = mean(|close - open|) / mean(high - low)
    # How decisive each bar is — high = trending bars, low = indecisive

tick_rule_proxy = (count(close > open) - count(close < open)) / 12
    # Net buying-pressure proxy from 5-min bar directions

up_volume_ratio = sum(volume where close > open) / sum(volume)
    # Volume-on-strength — accumulation vs distribution proxy

# Texture proxies
range_expansion_velocity = slope(high - low) over 12 bars
    # Is intraday range expanding (breakout setup) or contracting (mean-reversion setup)

late_session_velocity_ratio = velocity(last 6 bars) / velocity(first 6 bars)
    # Is momentum accelerating or decaying into entry

intraday_extremes_position = (last_close - rolling_low_60m) / (rolling_high_60m - rolling_low_60m)
    # Position relative to actual extremes (not VWAP) — picks up exhaustion vs continuation
```

**Implementation:**
- Add features to the feature builder
- Bump schema version, invalidate SQLite cache
- Full rebuild for Russell 1000 at 5-min resolution (~2hr)
- One retrain

**Expected impact:** AUC 0.544 → 0.550–0.560, Sharpe lift +0.05 to +0.15. Modest but real.

**Risk:**
- Diminishing returns. With 42 existing features, marginal features tend to overlap with existing ones.
- If diagnostic shows the model is reversion-flavored, the momentum-coded features (`tick_rule_proxy`, `up_volume_ratio`) may be noise drag.
- Most expensive experiment — only run after the cheap experiments are exhausted.

**What we learn:** Whether OHLCV-derived microstructure proxies have real signal at 5-min resolution, or whether the OHLCV ceiling has been reached on this dataset.

---

## 5. What to Defer

| Idea | Why defer |
|---|---|
| Multi-timeframe features (15m/30m aggregations) | Adds dimensionality with low marginal info on 2-hour holds. The 5-min/60-min framing is appropriate to the hold horizon. |
| Regime-conditional models | Powerful but complex. Only worth doing if v23/v24 don't close most of the gap. The PM abstention gate is already doing most of the regime work implicitly. |
| LambdaRank / ranking objective swap | Cross-sectional ranks are already used at scoring time. Objective swap adds complexity for marginal gain. |
| Pre-market gap / overnight catalyst features | Real signal but requires news/event data outside the current stack. |
| Different entry timing (30m / 90m instead of 60m) | Reasonable but probably 2nd-order. Won't move things from 0.30 to 0.80 — incremental at best. Re-evaluate after the top 4 if still short of gate. |
| Different model architecture (ranker, NN, etc.) | The bottleneck is not architectural. XGBoost on 280k rows with 42 features is not where the constraint binds. |
| Sector-relative strength feature | Single feature addition is reasonable but prioritize structural changes first. |

---

## 6. Direct Answers to the 10 Questions

**Q1 (signal quality):**
Most remaining alpha in OHLCV is in *texture* features (v26 list), not more indicators. But structural changes (universe v23, label v24) probably matter more. Push AUC marginally is much harder than fixing a label weight that's miscalibrated.

**Q2 (label engineering):**
path_quality structure is fine. The weights are the issue — 1.25× stop_pressure is likely too aggressive for noisy intraday data. Weight reduction (v24) is the key change, not a new label scheme.

**Q3 (R:R vs win rate):**
Win rate is harder to move (bounded by AUC). R:R is easier to move structurally. **But the smarter answer: realized R:R is probably already much worse than 2:1** because of time exits. Investigate before changing anything (diagnostic Section 3 cut #2).

**Q4 (universe):**
Yes. SP-500 first (v23). If it works, consider SP-100 as a follow-up. Highest-leverage structural change available.

**Q5 (entry timing):**
Defer. Probably 2nd-order. Consider only if top 4 doesn't close the gap.

**Q6 (meta-model approach):**
Drop it (v25). Replace with isotonic calibration. R²=0.001 says it's contributing nothing.

**Q7 (architecture):**
Stay with XGBoost. Bottleneck is not architectural.

**Q8 (multi-timeframe):**
Defer. Marginal expected lift.

**Q9 (cross-sectional):**
Existing cross-sectional features (`cs_rank_momentum`, `cs_rank_volume`, `cs_rank_atr`) cover the important dimensions. Consider sector-relative strength at entry time as a single addition, not a full re-engineer.

**Q10 (regime-conditional):**
Powerful but defer. The PM abstention gate is already doing implicit regime work. Only worth dedicated regime modeling if v23/v24 don't close most of the gap.

---

## 7. Honest Note on the Gate

With OHLCV-only at 5-min resolution, no L2 data, no shorting, and a 2-hour hold, **+0.80 Sharpe is achievable but not guaranteed.** The realistic range from where the system sits is probably **+0.45 to +0.85** after structural improvements.

Intraday data fidelity is fundamentally lower than daily. Institutional intraday alpha typically depends on microstructure inputs that are not available in this stack.

**If after v23–v26 the system is at +0.55 to +0.65, three options merit consideration:**

1. **Lower the gate to +0.60** with the same robustness criteria (no fold below −0.30). Document why; the swing model gate of +0.80 was set on better data fidelity.

2. **Paper-trade what's there at half size** for 60 days and let live data inform whether to push for +0.80. Real spreads, real fills, real slippage will reveal whether backtest Sharpe is robust.

3. **Accept the intraday strategy as a smaller, lower-Sharpe sleeve** and let the swing model carry the portfolio. A blended portfolio of swing (+1.18) + intraday (+0.60) is meaningfully better Sharpe than swing alone, even if intraday never hits +0.80 standalone.

**Don't anchor too hard on a number that may not be achievable at the available data fidelity.** The swing model proves the approach works; the intraday version proves the approach is harder at higher frequency. That's a finding, not a failure.

---

## 8. Recommended Sequence Summary

```
Step 1: Diagnostic dive (free, half day)
        → Settle: momentum vs reversion, true R:R, meta-model attribution
        → Output: docs/phase47/diagnostic_report.md

Step 2: v25 — Drop meta, isotonic calibration (~45 min)
        → Test: does Sharpe stay ≥ +0.30 without the meta-model?
        → If yes: simplify, keep isotonic only

Step 3: v23 — Universe shrink to SP-500 (~65 min)
        → Test: does cleaner universe lift Sharpe materially?
        → Guardrail: trade count ≥ 100/fold

Step 4: v24 — Reduce stop_pressure to 0.75× (~65 min)
        → Test: does the trending-fold gap close?
        → Guardrail: realized stop-hit rate stays < 58%

Step 5: v26 — Microstructure proxy features (~3 hours total)
        → Test: do texture features add AUC and Sharpe?
        → Most expensive, lowest expected ROI per hour

Step 6: Reassess
        → If at gate (+0.80): pass to paper trading
        → If close (+0.55 to +0.65): consider lowered gate or paper at half size
        → If far (< +0.55): escalate to Phase 48 with regime-conditional modeling
```

**Total compute time across all 4 retrains: ~5 hours, sequential. Plus ~half day diagnostic and ~2hr cache rebuild for v26.**

---

## 9. Final Notes

- **Each experiment is one isolated change.** Do not stack changes within a single retrain. This is the same isolation rule used for the swing model successfully through Phase 45.

- **Trade count is a critical guardrail.** Several of these experiments could compress trade count materially. Below 100 trades per fold, walk-forward Sharpe estimates become unreliable. If trade count drops, evaluate whether the per-trade economics actually improved before declaring success.

- **Save all artifacts with full lineage.** Tag each model with: git SHA, parent version, label config, universe filter, feature schema version. This is how attribution stays clean across the next 4 retrains.

- **The diagnostic is not optional.** Skipping it and going straight to retrains means optimizing with incomplete information. A half-day of pandas analysis can change which experiments are worth running.

- **When in doubt, stop and report.** Especially before the v26 cache rebuild — that's a 2-hour irreversible step. Confirm direction first.

End of analysis.
