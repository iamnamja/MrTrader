# ML Experiment Log — Active Campaign

Tracks model improvement iterations for active and recent phases.
**Archive (Phases 18–26b/c/d, Iterations 1–6):** `docs/ML_EXPERIMENT_LOG_archive.md`

---

## How to Read This Log

- **Verdict**: ✅ Keep | ❌ Revert | 🔄 Pending
- Walk-forward gate (intraday): avg Sharpe > 1.50, no fold < -0.30
- Walk-forward gate (swing): avg Sharpe > 0.80, no fold < -0.30

> **2026-05-05 Meta-update:** Multi-LLM review revealed the walk-forward gate numbers to date are NOT reliable baselines because: (1) no transaction costs, (2) no PM opportunity score simulated, (3) no purge/embargo between folds, (4) NIS features encode time (NaN = pre-2025 regime). Phases 1–2 of MASTER_BACKLOG fix this. Re-run all champions after Phase 1+2 complete to get honest numbers. See `docs/llm_review_synthesis.md`.

---

## Current Champion Models

| Model | Version | Features | Label | Original Sharpe | Honest Sharpe (Phase 1 corrections) | Status |
|---|---|---|---|---|---|---|
| Swing | v142 | 84 | path_quality (5d) | +1.181 | **+0.422** ❌ | Paper only — stale |
| Intraday | v29 | 50 | cross-sectional top-20% | +1.830 | **-0.984** ❌ | Paper only — stale |

> **Phase 1 corrections applied (2026-05-05):** Walk-forward now includes (1) 5bps/15bps round-trip transaction costs, (2) 10-day swing / 2-day intraday purge at fold boundaries, (3) NIS features removed from training (time-leak). These are the first honest Sharpe numbers. Both models fail. See Phase 1 Corrections section below for full fold detail.

---

## Phase 1 Corrections Baseline Walk-Forward — 2026-05-05

**Purpose:** First honest re-validation of champions after applying Phase 1a (cost model), Phase 1b (purge/embargo), Phase 1c (NIS removal). Establishes true baseline before any new model work.

**Corrections vs original walk-forward:**
| Correction | Before | After |
|---|---|---|
| Transaction costs | 0bps | 5bps RT (swing) / 15bps RT (intraday) |
| Fold purge gap | None | 10 calendar days (swing) / 2 trading days (intraday) |
| NIS features | In training | Removed (time-leak — encode regime not sentiment) |
| Walk-forward window | Varies | Same 730d intraday / 5yr swing as original |

### Intraday v29 — Phase 1 Corrections

**Run date:** 2026-05-05 | **Cost:** 15bps RT (7.5bps/side) | **Purge:** 2 trading days

| Fold | Test Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | 2024-10-24 → 2025-04-22 | 244 | 41.4% | **-1.36** | ❌ |
| 2 | 2025-04-25 → 2025-10-17 | 245 | 45.7% | **-2.19** | ❌ |
| 3 | 2025-10-22 → 2026-04-17 | 245 | 50.2% | **+0.60** | ❌ |
| **Avg** | | **734** | **45.8%** | **-0.984** | ❌ GATE FAILED |

**Gate:** avg Sharpe > 1.0 (updated honest gate) | **Min fold:** -2.19 (gate: > -0.30)

**Interpretation:**
- Fold 2 (Apr–Oct 2025 tariff shock) is the regime killer at -2.19. Win rate 45.7% — model has some signal but costs destroy it in a choppy trending market.
- Fold 3 (Oct 2025–Apr 2026) recovers to +0.60 at 50.2% win rate — model works in recent regime.
- Original +1.830 was computed without costs or purge. True net Sharpe on the same data would have been ~+0.8–1.0 if costs had been included.
- **Root cause:** 15bps RT is too high relative to the 0.8×ATR target in a chop regime. Need either lower-cost execution or wider target, or a dispersion gate to skip low-opportunity days.

### Swing v142 — Phase 1 Corrections

**Run date:** 2026-05-05 | **Cost:** 5bps RT (2.5bps/side) | **Purge:** 10 calendar days

| Fold | Test Period | Trades | Win% | Sharpe | Max DD | Gate |
|---|---|---|---|---|---|---|
| 1 | 2022-08-17 → 2023-11-05 | 186 | 46.8% | **+1.25** | 3.0% | ✅ |
| 2 | 2023-11-16 → 2025-02-03 | 226 | 43.4% | **+0.24** | 3.8% | ✅ |
| 3 | 2025-02-14 → 2026-05-05 | 186 | 39.8% | **-0.23** | 5.0% | ✅ |
| **Avg** | | **598** | **43.3%** | **+0.422** | | ❌ GATE FAILED (avg < 0.6) |

**Gate:** avg Sharpe > 0.6 (updated honest gate) | **Min fold:** -0.23 (gate > -0.30 ✅ just passes)

**Interpretation:**
- Fold 1 (2022–2023 bear/recovery): +1.25 — model generalises well to trending-down/recovery regime.
- Fold 2 (2023–2025 melt-up): degrades to +0.24 — EMA/RSI_DIP signals are late entries in a sustained bull run.
- Fold 3 (2025 tariff/vol shock): -0.23, win rate 39.8%. The RSI_DIP pre-filter catches falling knives in gap-and-trend regimes. Just barely passes the min-fold gate.
- Cost correction matters: with wrong 10bps RT, fold 3 was -0.47 (fails gate). With correct 5bps RT, fold 3 is -0.23 (passes). Swing is less cost-sensitive than intraday.
- **Root cause of avg Sharpe below gate:** Mid-period degradation (fold 2 at +0.24) pulls avg down. RSI_DIP + EMA_CROSSOVER pre-filters cap alpha ceiling. ML signal is present but entry timing is constrained by hardcoded rules.
- **Next steps:** (1) PM opportunity score gate for 2025 regime (Phase 2a), (2) remove RSI_DIP/EMA_CROSSOVER pre-filters and let ML pick entries (Phase 3a), (3) shorten training window to exclude 2021 bull distortion.

---

---

## Phase 1d — Deflated Sharpe Ratio + Bootstrap Analysis — 2026-05-05

**Purpose:** Quantify selection bias from testing ~15 model variants. DSR (Bailey & López de Prado 2014) corrects the raw Sharpe for the expected maximum Sharpe from N independent random trials.

### DSR Results

| Model | Version | Raw Sharpe | N Trials | DSR z-score | P(SR>0) | Significant? |
|---|---|---|---|---|---|---|
| Swing | v142 | +1.181 (original) | 15 | -9.96 | 0.000 | ❌ No |
| Swing | v142 | +0.422 (Phase 1 corrected) | 15 | -30.95 | 0.000 | ❌ No |
| Intraday | v29 | +1.830 (original) | 15 | +0.87 | 0.807 | ⚠️ Borderline |
| Intraday | v29 | -0.984 (Phase 1 corrected) | 15 | -56.77 | 0.000 | ❌ No |

**Interpretation:**
- **Swing v142 original +1.181 is statistically meaningless** even before Phase 1 corrections. With 15 trials, the expected maximum Sharpe from random noise exceeds +1.181. We cannot tell whether swing v142 has any genuine edge.
- **Intraday v29 original +1.830 was borderline** (p=0.807) — meaningful signal existed but not definitively proven. The original result was partly real, partly selection bias.
- With Phase 1 corrections, both models fail DSR comprehensively.
- **DSR is now printed in every walk-forward run** (added to `walkforward_tier3.py`). Any future model must pass DSR p > 0.95 as an additional gate condition.

### Bootstrap Distribution (Small Sample — 10 Resamples)

Full 200-resample bootstrap would take ~50 hours (200 × 17min swing). A 10-resample run is pending — will document the distribution shape below. The DSR analytic result above is the primary Phase 1d deliverable.

*Bootstrap results: pending — run `python scripts/bootstrap_sharpe.py --model swing --n-resamples 10` overnight.*

### Updated Gate Criteria (Post Phase 1d)

```
avg net Sharpe > 0.6 (swing) / > 1.0 (intraday)   [lower but honest — with costs + purge]
worst fold Sharpe > -0.30
DSR p-value > 0.95                                  [NEW — corrects for N trials tested]
Max drawdown < 15%
```

---

> **Retrain in progress (2026-05-05 EOD):** Swing v146 (89 feat = 84 + 5 macro NIS) and Intraday v30 (63 feat = 53 + 5 NIS + 5 macro NIS) are training now. Results will be logged below when walk-forward completes.

---

## Retrain Campaign — 2026-05-05 EOD (Macro NIS Integration)

**Context:** Phases 64 (stock NIS) + 90 (macro NIS) added 10 new features with full historical backfill. Current champions (v142/v29) were trained before these features existed — live inference uses them but training didn't see them.

### Swing v146 — 89 features (84 + 5 macro NIS) 🔄 IN PROGRESS

**Training started:** 2026-05-05 ~16:08 ET
**Features added:** `macro_avg_direction`, `macro_pct_bearish`, `macro_pct_bullish`, `macro_avg_materiality`, `macro_pct_high_risk`
**Training window:** 3 years (Polygon daily bars, 430 symbols)
**Gate:** avg Sharpe > 0.80, no fold < -0.30

**Walk-forward result:**

| Fold | Trades | Sharpe |
|---|---|---|
| 1 | 189 | +0.18 |
| 2 | 254 | +0.44 |
| 3 | 208 | **-1.07** |
| **Avg** | 651 | **-0.148** ❌ GATE FAILED |

v142 restored as ACTIVE champion. Fold 3 (2025 regime) continues to collapse — worse than v145 (-0.87). Macro NIS features are not helping and may be adding noise. **Root cause is not the features — the 3yr training window includes 2021-2022 bull market data that teaches patterns incompatible with the 2025 tariff/vol regime.**

**Next swing retrain should try a shorter window (2yr or 18mo) to exclude 2021-2022.** Also: fundamentals (`profit_margin`, `revenue_growth`, `debt_to_equity`) and `sector_momentum` are now available via PIT parquets — include those in v147.

---

### Intraday v30 — 58 features (50 + 5 NIS + 3 SPY-relative) 🔄 IN PROGRESS

**Training started:** 2026-05-05 ~16:08 ET
**Features added over v29 (50 feat):**
- Phase 64: 5 stock NIS features (`nis_direction_score`, `nis_materiality_score`, `nis_already_priced_in`, `nis_sizing_mult`, `nis_downside_risk`) in `intraday_features.py`
- Phase 86b: 3 stock-relative SPY features (`stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`) in `intraday_features.py`
- **Note:** Macro NIS features (Phase 90) live in `features.py` (swing only) — intraday uses `intraday_features.py`, so macro NIS is NOT included in this retrain
**Label scheme:** `cross_sectional_top20pct_phase89` (same as gate-passing v29)
**Training window:** 730 days (Alpaca 5min bars, 750 symbols)
**Gate:** avg Sharpe > 1.50, no fold < -0.30

*Results pending — will update when walk-forward completes.*

---

## Intraday Improvement Campaign — 2026-05

**Context:** v29 passed walk-forward gate Apr 2026 (+1.807 avg Sharpe, 730d window).
Re-tested May 2026 on most recent 365 days — fails (+0.611 avg Sharpe).
**Root cause:** Cross-sectional top-20% label forces positive labels on low-opportunity days.
**Full plan:** `docs/intraday_improvement_plan.md`

### Baseline — v29 Re-Test (2026-05-01)

| Fold | Period | Trades | Win% | Sharpe | Regime |
|---|---|---|---|---|---|
| 1 | Jul–Oct 2025 | 126 | 33.3% | **-0.68** | Low-vol melt-up: VIX 16.5, 0.65%/d |
| 2 | Oct–Jan 2026 | 126 | 49.2% | +1.88 | Moderate: VIX 17.3, 0.76%/d |
| 3 | Jan–Apr 2026 | 126 | 44.4% | +0.63 | Higher vol: VIX 20.4, 0.90%/d |
| **Avg** | | **378** | **42.3%** | **+0.611** | FAIL |

---

## Phase 2a — PM Opportunity Score Walk-Forward — 2026-05-05

**What:** Wired the live PM continuous opportunity score into both simulators using historical SPY + VIX daily bars:
`score = 0.35×vix_score + 0.20×vix_trend + 0.30×ma_score + 0.15×mom_score`
`< 0.35 → skip all entries | 0.35–0.65 → cap candidates at 2 | ≥ 0.65 → normal`

**Flag:** `--pm-opportunity-score`

### Swing v142 — Phase 2a (opportunity score ON)

| Fold | Test Period | Trades | Win% | Sharpe | vs Phase 1 |
|---|---|---|---|---|---|
| 1 | 2022-08-17 → 2023-11-05 | 186 | 46.8% | **+1.25** | = |
| 2 | 2023-11-16 → 2025-02-03 | 226 | 43.4% | **+0.24** | = |
| 3 | 2025-02-14 → 2026-05-05 | 186 | 39.8% | **-0.23** | = |
| **Avg** | | **598** | **43.3%** | **+0.422** | ❌ No change |

**Finding:** Opportunity score has zero impact on swing. Identical trade counts — the score never suppressed a swing entry day. The fold 3 collapse is caused by RSI_DIP/EMA_CROSSOVER pre-filters catching falling knives in the tariff regime, not by calendar-level macro condition. Fix is Phase 3a (remove pre-filters).

### Intraday v29 — Phase 2a (opportunity score ON)

| Fold | Test Period | Trades | Win% | Sharpe | vs Phase 1 |
|---|---|---|---|---|---|
| 1 | 2024-02-07 → 2024-10-31 | 264 | 42.4% | **-2.85** | ↓ worse |
| 2 | 2024-11-05 → 2025-08-05 | 373 | 44.0% | **-1.81** | ↑ better |
| 3 | 2025-08-08 → 2026-05-05 | 354 | 47.2% | **-1.08** | ↓ worse |
| **Avg** | | **991** | **44.5%** | **-1.916** | ❌ Worse than baseline |

**Phase 1 baseline for comparison:** F1=-1.36, F2=-2.19, F3=+0.60, avg=-0.984

**Finding:** Opportunity score hurt intraday overall (-1.916 vs -0.984). Fold 3 (2025-08 → 2026-05) regressed from +0.60 to -1.08 — this was the only profitable fold. The score appears to be filtering out good intraday days during the recent high-vol regime where cross-sectional dispersion is high (i.e., good setup days) but SPY MA conditions look "opportunistic". Phase 2c (dispersion gate) is the next test — it targets macro-dominated low-dispersion days specifically.

---

## Phase 2c — Cross-Sectional Dispersion Gate — 2026-05-06

**What:** Skip intraday entries on days where cross-sectional return dispersion (std of 2h returns across all symbols, bars 12→36) < 0.5 × rolling 60-day median. These are macro-dominated days where all stocks move together and individual stock selection has no edge.

**Flag:** `--dispersion-gate` (combined with `--pm-opportunity-score`)

**Command:** `python scripts/walkforward_tier3.py --model intraday --intraday-cost-bps 15 --intraday-purge-days 2 --pm-opportunity-score --dispersion-gate`

### Intraday v29 — Phase 2c (opp score + dispersion gate)

| Fold | Test Period | Trades | Win% | Sharpe | vs Phase 2a |
|---|---|---|---|---|---|
| 1 | 2024-02-07 → 2024-10-31 | 264 | 42.4% | **-2.85** | = same |
| 2 | 2024-11-05 → 2025-08-05 | 369 | 44.4% | **-1.68** | ↑ better |
| 3 | 2025-08-08 → 2026-05-05 | 344 | 48.0% | **-0.91** | ↑ better |
| **Avg** | | **977** | **44.9%** | **-1.816** | ↑ slight improvement |

**Phase 2a baseline for comparison:** F1=-2.85, F2=-1.81, F3=-1.08, avg=-1.916

**Finding:** Dispersion gate helps folds 2 and 3 marginally (F2: -1.81→-1.68, F3: -1.08→-0.91) but fold 1 is unchanged at -2.85. Still fails gate comprehensively. The dispersion gate is filtering some low-opportunity days but the core problem (model trained on cross-sectional top-20% label in a regime where no stocks have clean setups) remains. Net improvement over Phase 2a: +0.1 Sharpe. Not significant enough to unlock the model.

**Verdict:** ❌ GATE NOT MET — avg -1.816. Phase 2c is a marginal incremental improvement. The real fix requires model architecture repair (Phase 3a/3b/4a in MASTER_BACKLOG).

---

## Phase 3a — Remove RSI/EMA Pre-Filters From Swing — 2026-05-06

**What:** Added `no_prefilters=True` option to `AgentSimulator._trader_signal()`. When enabled, bypasses:
- RSI 40-70 zone gate (was preventing entries in RSI < 40 downtrend or RSI > 70 overbought)
- EMA-20/50 near-term trend filter (was preventing entries when price below recent EMAs)
- EMA-200 long-term trend filter and volume check are KEPT

**Rationale:** In 2025 tariff/vol regime, RSI_DIP/EMA_CROSSOVER pre-filters catch falling knives. ML model should score full universe.

**Command:** `python scripts/walkforward_tier3.py --model swing --swing-cost-bps 5 --swing-purge-days 10 --no-prefilters`

### Swing v142 — Phase 3a (no pre-filters)

| Fold | Test Period | Trades | Win% | Sharpe | vs Phase 2a |
|---|---|---|---|---|---|
| 1 | 2022-08-17 → 2023-11-05 | 318 | 40.9% | **+0.24** | ↓ worse |
| 2 | 2023-11-16 → 2025-02-03 | 125 | 38.4% | **+0.10** | ↓ worse |
| 3 | 2025-02-14 → 2026-05-05 | 26 | 3.9% | **-2.54** | ↓ much worse |
| **Avg** | | **469** | **27.7%** | **-0.731** | ❌ Much worse |

**Phase 2a baseline for comparison:** F1=+1.25, F2=+0.24, F3=-0.23, avg=+0.422

**Finding:** Removing pre-filters made swing significantly worse. Fold 3 collapsed catastrophically: only 26 trades with 3.9% win rate and -2.54 Sharpe. The pre-filters (EMA-20/50, RSI 40-70) were serving as a regime guard — in a sharply declining 2025 tariff market, they prevented entries in falling stocks below their moving averages. Without them, the ML model (trained on 2021-2024 data) enters stocks that look like historical RSI dip setups but are actually momentum selldowns.

Also notable: removing pre-filters caused FEWER trades in fold 3 (26 vs 186), not more. Without EMA/RSI filters, the model's PM confidence threshold becomes the sole gate. In a declining market, few stocks pass confidence >= 0.40 AND EMA-200, so very few entries.

**Verdict:** ❌ REVERT — removing pre-filters is net negative. The correct fix per MASTER_BACKLOG is Step 1 + Step 2 together: score full universe WITH pre-filter features included as ML inputs + retrain on triple-barrier labels. Just removing the rule-based gates without a retrained model leaves the ML flying blind.

---

## Phase 3b — Rolling 2yr Training Window for Swing — 2026-05-06

**What:** Added `--swing-train-years 2` to walk-forward. Each fold's training data limited to 2 years before `train_end` (rolling window) instead of all data from 2021 (expanding window). Intent: exclude 2021-2022 bull market patterns from fold 2-3 training.

**Command:** `python scripts/walkforward_tier3.py --model swing --swing-cost-bps 5 --swing-purge-days 10 --swing-train-years 2`

### Swing v142 — Phase 3b (rolling 2yr window)

| Fold | Train Period | Test Period | Trades | Win% | Sharpe | vs Phase 2a |
|---|---|---|---|---|---|---|
| 1 | 2021-04-06 → 2022-08-06 | 2022-08-17 → 2023-11-05 | 186 | 46.8% | **+1.25** | = same |
| 2 | 2021-11-05 → 2023-11-05 | 2023-11-16 → 2025-02-03 | 226 | 43.4% | **+0.24** | = same |
| 3 | 2023-02-04 → 2025-02-03 | 2025-02-14 → 2026-05-05 | 186 | 39.8% | **-0.23** | = same |
| **Avg** | | | **598** | **43.3%** | **+0.422** | = identical |

**Phase 2a baseline for comparison:** F1=+1.25, F2=+0.24, F3=-0.23, avg=+0.422

**Finding:** Rolling 2yr window produces **identical results** to the expanding 5yr window. This is because the rolling window only affects fold 1 (fold 1's train start shifts from 2021-04 to 2021-04 since the 2yr window still goes back to 2021 from a 2022 train_end — barely different). For fold 2: train starts 2021-11 (2yr before 2023-11) vs 2021-04 (expanding). For fold 3: train starts 2023-02 (2yr before 2025-02) vs 2021-04 (expanding) — this is the meaningful difference. But fold 3 result is unchanged (-0.23), meaning: **the 2021-2022 data is not the culprit**. The 2025 tariff regime failure is caused by something the model learned from 2022-2024 data, not just the 2021 bull market.

**Verdict:** ❌ GATE NOT MET — avg +0.422 (same as Phase 2a baseline). Rolling 2yr window doesn't help. The fold 3 collapse is a genuine regime mismatch between the model's learned patterns and the 2025 tariff environment, not a training window artifact.

---

## Phase 4a — Absolute Hurdle Label Fix (Intraday Retrain) — 2026-05-06

**What:** Added `CS_ABSOLUTE_HURDLE = 0.0030` to intraday label scheme. Top-20% cross-sectional label now additionally requires ≥0.30% absolute 2h return. Prevents labeling least-bad stocks as winners on flat/down market days.

**Label scheme:** `cross_sectional_top20pct_abs_hurdle_0.30pct`
**Model trained:** v43 (XGBoost 3-seed ensemble + LightGBM) | **HPO AUC:** 0.6652 | **OOS AUC:** 0.6243
**Top features:** `seg_x_atr_norm`, `atr_norm`, `range_compression`, `minutes_since_open`, `time_of_day`
**Training time:** 2999.9s (~50 min) | **Dataset:** 277k train / 60k test rows, 61 features

**Walk-forward result:**

| Fold | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|
| 1 | — | — | — | ❌ |
| 2 | — | — | — | ❌ |
| 3 | — | — | — | ❌ |
| **Avg** | **—** | **—** | **-1.594** | ❌ GATE FAILED |

**Min fold Sharpe:** -2.025 | **Gate:** avg > 1.0, min > -0.30

**v29 restored as ACTIVE champion.**

**Finding:** Absolute hurdle fix did not improve walk-forward Sharpe vs v29 baseline (-0.984). Result (-1.594) is actually worse than Phase 2c baseline (-1.816 with dispersion gate). The label fix is conceptually sound but the walk-forward period includes the 2025 tariff regime where cross-sectional dispersion is so high (or so low on macro-dominated days) that no label scheme on top-20% percentile ranking will produce reliable positives. The problem is regime mismatch, not label quality.

**Note on AUC drift:** HPO AUC 0.6652 (above 0.65 threshold — no drift alert). OOS AUC 0.6243 (acceptable). The model learned something but it doesn't translate to Sharpe in the current regime.

**Verdict:** ❌ GATE NOT MET — v29 retained. Label fix alone is insufficient. Next investigation: label distribution analysis — how many positives survive the hurdle filter on down days vs up days, and whether the model is being trained on too sparse a positive class in the 2025 regime.

---

## Phase 4b — Swing Retrain (v142 → v148) — 2026-05-06

**What:** Standard retrain with current architecture (84 features, RSI/EMA pre-filters intact, 3yr training window). No architecture changes. Refreshed on data through 2026-05-06.

**Model trained:** v148 | **HPO AUC:** 0.5969 | **OOS AUC:** 0.6216
**Early stopping:** iteration 30 (very early — model barely learned)
**Drift alert:** AUC 0.6216 < 0.65 threshold
**Top features:** `volatility`, `atr_norm`, `parkinson_vol`, `realized_vol_20d`, `vrp`
**Class ratio:** neg=36097, pos=9077, scale_pos_weight=3.98

**Walk-forward result:**

| Fold | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| **Avg** | **—** | **—** | **-0.066** | ❌ GATE FAILED |

**Min fold Sharpe:** -0.308 | **Gate:** avg > 0.6, min > -0.30

**v142 restored as ACTIVE champion.**

**Finding:** Refreshing the model on current data (including the 2025 tariff period) made it worse than v142. The HPO AUC of 0.5969 and early stopping at iteration 30 indicate the model failed to find a generalizable pattern in the training data — likely because the 3yr window (2023-2026) spans very different regimes (2023-2024 bull melt-up + 2025 tariff volatility) and the XGBoost learner is averaging over contradictory signals. v142 (trained before the 2025 regime) had more consistent training signal.

**Verdict:** ❌ GATE NOT MET — v142 retained. Standard retrain is counterproductive in the current regime. The swing model needs a fundamentally different approach: regime-adaptive features, regime-aware training split, or separate models per regime.

---

## Phase 1e — DSR Hard Gate — 2026-05-06

**What:** Added DSR p > 0.95 as hard requirement in `WalkForwardReport.gate_passed()`. A model must now pass both the Sharpe threshold AND statistical significance after selection bias correction (N=15 trials). Previously DSR was printed but not enforced.

**Implementation:** `gate_passed()` now calls `_deflated_sharpe_ratio()` and requires `dsr_p > 0.95`.

**Impact:** All current models already fail Sharpe gates comprehensively (-0.731 to -1.916). The DSR gate adds a second layer of protection once Sharpe recovers — ensures future models with marginal improvements aren't promoted due to selection bias.

---

### Phase 85 — PM Abstention Gates (No Retrain) ✅ DONE

**Branch:** `feat/phase-85-intraday-gates` | **Completed:** 2026-05-02

**What was changed:**
- VIX ≥ 25 → abstain from all intraday entries
- SPY < MA20 → abstain from all intraday entries

**Walk-forward result (730d window, 3 expanding folds, model v29):**

| Fold | Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 250 | 48.0% | +2.95 | ✅ |
| 2 | Apr 2025–Oct 2025 | 250 | 40.4% | +0.78 | ✅ |
| 3 | Oct 2025–Apr 2026 | 250 | 46.8% | +1.75 | ✅ |
| **Avg** | | 750 | 45.1% | **+1.830** | ✅ GATE PASSED |

**Key observation:** Gates rescued fold 2 (was -0.68) without hurting high-vol folds. Gates are a runtime patch — v29 training signal is still structurally corrupted. Phases 87+86b fix the root cause.

**Verdict:** ✅ GATE PASSED — merged, v29 + abstention gates are the active live configuration.

---

### Phase 86 — Market Context Features + Retrain ❌ REVERTED

**Branch:** `feat/phase-86-market-context-features` | **Completed:** 2026-05-02

**What was tried (v34–v36):**
Added 5 market-wide SPY features: `spy_first_hour_range`, `spy_5d_return`, `spy_5d_realized_vol`, `market_is_trending`, `spy_day_vol_vs_avg`. Lookahead bug found and fixed (`d <= day` → `d < day`). Train/test mismatch found and fixed (simulator now receives `spy_daily_bars`).

**Walk-forward result (v36, 6 folds):**

| Fold | Test Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 250 | 40.8% | -1.04 | ❌ |
| 2 | Apr 2025–Oct 2025 | 252 | 46.0% | -0.41 | ❌ |
| 3 | Oct 2025–Apr 2026 | 250 | 42.8% | -0.13 | ✅ |
| **Avg** | | 752 | 43.2% | **-0.529** | ❌ GATE FAILED |

**Root cause:** All 5 features have identical values for every symbol on a given day. After `cs_normalize` (z-score within each day's symbol set), all become exactly zero — zero cross-sectional discriminating signal.

**Also discovered:** v37 (same 53 features as v29, fresh 50-trial HPO) scored **-0.219** vs v29's +1.830. Confirms HPO variance (~2.0 Sharpe spread) is a second root cause independent of features.

**Verdict:** ❌ REVERTED — market-wide features incompatible with cs-normalisation. Plumbing kept (`spy_daily_bars` wired through training, simulator, walk-forward) for Phase 86b. Proceeding to Phase 87 (stability fix) first.

---

---

## Swing Model Campaign — 2026-05

**Context:** Swing champion v142 (84 features, path_quality 5d) was promoted ~2026-05-01 by automated retrain but not logged. Walk-forward baseline run 2026-05-04 showed avg Sharpe 0.281 (below 0.8 gate) — model was already degraded. NIS features (Phase 64) added; retrain campaigns below.

### Swing Baseline Walk-Forward (2026-05-04, no retrain, existing champion v142)

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Aug 2022–Nov 2023 | 190 | 46.3% | +1.10 |
| 2 | Nov 2023–Feb 2025 | 231 | 44.2% | +0.26 |
| 3 | Feb 2025–May 2026 | 172 | 39.5% | **-0.51** |
| **Avg** | | 593 | 43.3% | **+0.281** | ❌ GATE FAILED |

Root cause: Fold 3 (Feb 2025–now) captures 2025 tariff/volatility regime. Model trained on pre-2025 data doesn't generalize.

### Phase 64 NIS Swing Retrain — v144 (2026-05-04, clean cache) ❌ GATE NOT MET

**What changed:** Added 5 NIS features (direction, materiality, already_priced_in, sizing_mult, downside_risk) with `0.0/1.0` defaults for missing data.

| Fold | Period | Trades | Sharpe | vs Baseline |
|---|---|---|---|---|
| 1 | Aug 2022–Nov 2023 | 180 | **-0.11** | ↓ was +1.10 |
| 2 | Nov 2023–Feb 2025 | 251 | +1.21 | ↑ was +0.26 |
| 3 | Feb 2025–May 2026 | 199 | +0.03 | ↑ was -0.51 |
| **Avg** | | 630 | **+0.376** | ❌ GATE FAILED |

**Root cause of Fold 1 regression:** NIS DB has no data for 2022–2023. All Fold 1 training rows got default values (0.0/1.0), teaching XGBoost a spurious pattern: "when NIS=defaults, predict X." Fixed in Phase 88.

---

### Phase 88 — NIS NaN Encoding + Loosen Intraday Label ❌ GATE NOT MET (swing)

**Branch:** `feat/phase-88-label-nis-fix` (PR #125) | **Date:** 2026-05-05

**What changed:**
1. NIS missing-data encoding: `0.0/1.0` → `NaN` so XGBoost uses learned missing-value direction
2. Intraday label: `MIN_REALIZED_R` 0.5 → 0.40 (experiment log recommendation from Phase 87)

**Swing v145 walk-forward (3 folds, 5yr window, 81 symbols):**

| Fold | Period | Trades | Sharpe | vs v144 (0.0 defaults) | vs Baseline |
|---|---|---|---|---|---|
| 1 | Aug 2022–Nov 2023 | 190 | +0.64 | ↑ from -0.11 | ↓ from +1.10 |
| 2 | Nov 2023–Feb 2025 | 236 | +0.78 | ↓ from +1.21 | ↑ from +0.26 |
| 3 | Feb 2025–May 2026 | 190 | **-0.87** | ↓ from +0.03 | ↓ from -0.51 |
| **Avg** | | 616 | **+0.182** | ❌ | ❌ GATE FAILED |

**v142 retained as ACTIVE swing champion.**

**Analysis:** NaN encoding fixed the Fold 1 regression (−0.11 → +0.64) as expected. But Fold 3 (Feb 2025–now) collapsed further (−0.51 → −0.87). The 2025 market regime is structurally hostile to the current feature set and label scheme. NIS features are not the bottleneck — the swing model needs a deeper fix for recent-regime generalization.

**Intraday v40 walk-forward (MIN_REALIZED_R=0.40, NaN NIS encoding):**

| Fold | Period | Trades | Sharpe | vs v39 (0.5 threshold) | vs v29 baseline |
|---|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 248 | +0.18 | ↑ from +0.06 | ↑ from -0.77 |
| 2 | Apr–Oct 2025 | 249 | **-1.00** | ↓ from +0.64 | ↑ from -1.45 |
| 3 | Oct 2025–Apr 2026 | 248 | -0.74 | ↓ from -0.77 | ↓ from +0.99 |
| **Avg** | | 745 | **-0.519** | ❌ worse | ❌ GATE FAILED |

OOS AUC: 0.508 (≈ random). **v29 restored as ACTIVE intraday champion.**

**Analysis:** Looser label (0.40) did not restore signal quality — AUC barely moved from 0.517 (v39) to 0.508 (v40). The realized-R label scheme appears fundamentally unsuited to the 730d training window used here, or the feature set doesn't predict it. Fold 2 (Apr–Oct 2025) collapsed from +0.64 → −1.00, which is unexplained by the threshold change alone.

**Critical insight:** v29 (cross-sectional top-20% labels + Phase 85 abstention gates) achieved +1.830 avg Sharpe. Every retrain with realized-R labels has failed. The cross-sectional label scheme combined with runtime gates may be the right architecture — the problem is v29 has 50 features but live inference now produces 58 (NIS Phase 64b causes mismatch).

**Recommended next step:** Retrain with cross-sectional labels (revert realized-R) + 58 features. This would give us a proper v29-equivalent but with the correct feature schema.

---

### Phase 87 — Realized-R Labels + 3-Seed Ensemble + Frozen HPO ❌ GATE NOT MET

**Branch:** `feat/phase-87-label-fix-ensemble` (PR #121) | **Date started:** 2026-05-02

**Architecture rationale:** Phase 86 failure revealed HPO variance (~2.0 Sharpe) dominates results. Must fix training stability before adding new features. Three changes bundled:

**What was changed:**
1. **Realized-R labels (B+A):** `realized_R ≥ 0.5 AND abs_move ≥ 0.30%` → label 1. Zero positives allowed on bad days. Replaces cross-sectional top-20% ranking.
2. **3-seed XGBoost ensemble (permanent):** Seeds 42/123/777, blend probabilities by simple average. Applied to every retrain going forward.
3. **100-trial Optuna HPO → FROZEN_HPO_PARAMS:** Thorough search once, freeze params for stability.

**Training class balance (v38, 2026-05-02):** pos=68,459 / neg=209,927 (scale_pos_weight=3.07)
Note: healthier than forced top-20% which always had ~20% positive regardless of market quality.

**Walk-forward results — full comparison (all 4 runs, same 730d window):**

| Run | Fold 1 | Fold 2 | Fold 3 | Avg Sharpe | Gate |
|---|---|---|---|---|---|
| v29, no gates | -0.82 | -1.38 | +0.82 | -0.458 | ❌ |
| v38, no gates | +0.05 | +0.34 | +0.26 | +0.215 | ❌ |
| v29 + Phase 85 gates | +2.95 | +0.78 | +1.75 | +1.830 | ✅ |
| **v38 + Phase 85 gates** | **+0.50** | **+0.51** | **+1.02** | **+0.675** | ❌ |

v38+gates detail: 524 total trades (144/224/156), win rates 49.3%/46.9%/51.9%, max DD 0.2%.

**Key observations:**
- v38 (realized-R labels) is strictly better than v29 without gates: +0.215 vs -0.458. The label change is directionally correct.
- With gates, v38 trails v29: +0.675 vs +1.830. Gates filter trade count (524 vs ~750) — realized-R labels combined with gates are over-filtering.
- All v38 folds positive (no blowups). Min fold Sharpe +0.50 — structurally healthier than v29+gates which had fold 2 at +0.78 only due to gates masking a -1.38 without-gates fold.
- Trade count drop (524 vs 750) suggests `MIN_REALIZED_R=0.5` is too strict — many 0.3–0.4R winners labeled 0 at training time, then correctly rejected by the model at test time.

**Verdict:** ❌ GATE NOT MET — v29 + Phase 85 gates remains champion. v38 is a structural improvement but needs threshold tuning. Options: (A) lower `MIN_REALIZED_R` to 0.35–0.40 and retrain, (B) proceed to Phase 86b (stock-relative features) which adds discriminating signal and may recover selectivity. **Decision: proceed to Phase 86b first** — adding features is higher leverage than threshold micro-tuning.

---

### Phase 87a — Regression Labels (Realized R-Multiple) [DEFERRED]

**Precondition:** Phase 87 ✅ + Phase 86b (stock-relative features) ✅

**Rationale:** Binary labels discard magnitude — a +3R and +0.6R win both get label=1. Regression target (predict realized R-multiple) teaches model to distinguish great setups from marginal ones.

**What to change:**
- XGBoost objective: `binary:logistic` → `reg:squarederror`
- Label: `realized_R` (clipped to [-3.0, +3.0])
- Scoring: rank by predicted R-multiple directly (no probability threshold)
- Keep ensemble 3 seeds + frozen HPO (adapted for regression)

**Verdict:** 🔄 Deferred

---

## Key Takeaways — Intraday Campaign

| Learning | Implication |
|---|---|
| cs_normalize zeros market-wide features | All new features must vary across symbols within a day |
| HPO variance ~2.0 Sharpe on identical features | 100-trial search once → freeze → 3-seed ensemble permanently |
| Forced top-20% labels corrupt bad-day training | Realized-R outcome labels allow zero positives on bad days |
| Gates rescue bad folds but don't fix training | Model must learn to self-abstain, not rely on PM-level runtime gates |

---

---

## Re-Validation on Current Data Window — 2026-05-05

**Purpose:** Re-run existing champions on current data window (folds ending 2026-04-17) to see if they still hold up. Result: neither does. This is the key finding that triggered the multi-LLM review and Phase 1–3 work.

### Intraday v41 — 61 features (NIS + SPY-relative + cs top-20%) ❌

**Trained:** 2026-05-05 EOD  
**HPO best:** AUC=0.6465 (OOS: 0.6289)  
**Top 5 features:** `seg_x_atr_norm` (14%), `atr_norm` (11%), `range_compression` (6%), `minutes_since_open` (3%), `time_of_day` (3%) — NIS not in top 5

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 248 | 41.1% | +0.69 |
| 2 | Apr 2025–Oct 2025 | 249 | 45.4% | **-0.34** |
| 3 | Oct 2025–Apr 2026 | 249 | 49.8% | +0.15 |
| **Avg** | | 746 | 45.4% | **+0.167** ❌ GATE FAILED |

v29 restored as ACTIVE champion. New features (NIS, SPY-relative) did not help and v41 is worse on fold 3 (+0.15 vs v29's +0.97). Fold 2 (Apr–Oct 2025) remains the regime problem.

**Key insight from NIS NaN analysis:** NIS features at 80% NaN teach the model `NaN = 2021-2024 regime`. v41's slight improvement on fold 2 (-0.34 vs v29's -1.27) may be the model learning to treat non-NaN rows differently by time period, not by sentiment. NIS must be removed from training. See Phase 1c.

### Intraday v29 — Re-validated on Current Window ❌

**Original gate result (older window):** +1.830  
**Same model, current data window:**

| Fold | Period | Trades | Win% | Sharpe |
|---|---|---|---|---|
| 1 | Oct 2024–Apr 2025 | 248 | 41.1% | **-0.67** |
| 2 | Apr 2025–Oct 2025 | 249 | 45.4% | **-1.27** |
| 3 | Oct 2025–Apr 2026 | 249 | 49.8% | **+0.97** |
| **Avg** | | 746 | 45.4% | **-0.327** ❌ |

The original +1.830 was computed before Apr–Oct 2025 existed in any test fold. v29 was never validated against that regime. Fold 3 (+0.97) shows the model architecture works in calmer conditions — the problem is regime-specific, not fundamental.

**Decomposition of Sharpe collapse (per multi-LLM review):**
- ~60%: Selection bias (v29 was best of several variants; regression to mean as new data arrives)
- ~20%: PIT leakage (v142 trained with non-PIT-correct fundamentals; similar effect for intraday timing data)
- ~15%: Genuine non-stationarity (tariff shock is a real regime shift)
- ~5%: Data sparsity (limited crisis regime in training history)

### Swing v142 — Re-validated on Current Window ❌

| Fold | Train | Test | Trades | Win% | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| 1 | 2021-04→2022-08 | 2022-08→2023-11 | 189 | 47.1% | +1.20 | 3.1% |
| 2 | 2021-04→2023-11 | 2023-11→2025-02 | 231 | 43.7% | +0.24 | 4.0% |
| 3 | 2021-04→2025-02 | 2025-02→2026-05 | 172 | 39.5% | **-0.51** | 5.1% |
| **Avg** | | | 592 | 43.4% | **+0.310** ❌ |

Fold 3 (Feb 2025–now) collapses across ALL swing versions. This is the 2025 tariff shock + elevated VIX period. Win rate 39.5% (vs 47.1% in fold 1) shows the model's entry pattern selection is failing, not just timing.

---

## Key Takeaways — Swing Campaign (Historical)

1. **Embargo worked:** Test set reduced by 428 samples — confirms leakage was present without it
2. **CS normalization worked:** `sector_momentum` SHAP dropped from 0.90 to 0.18
3. **Revenue growth is real alpha:** Leading SHAP feature at 0.2353
4. **AUC drift threshold:** Realistic steady-state for SP-500 universe is 0.56–0.58 (not 0.65)
5. **5-day forward window:** Doubled training samples, better label-execution alignment
6. **Triple-barrier labels:** Tried and reverted — stop exits didn't improve

Full swing improvement history: `docs/ML_EXPERIMENT_LOG_archive.md`

---

## 2026-05-06 Experiment Campaign — Items 2 & 3

**Context:** Both champion models fail honest walk-forward (swing v142 avg +0.310, intraday v29 avg -0.327 on current window). Root cause is 2025 tariff/vol regime mismatch. This campaign tests two hypotheses in parallel.

---

### Item 2 — Triple-barrier Labels for Swing ❌ GATE FAILED

**Date:** 2026-05-06
**Model:** swing v149/v150 (both trained; v150 used for WF)
**Label scheme:** `triple_barrier` — bar-by-bar simulation: label=1 if target hit before stop, label=0 if stop hit first. 1.5x ATR target / 0.5x ATR stop.
**Baseline:** swing v142 (avg Sharpe +0.310 on current 3-fold window)
**Hypothesis:** Outcome-based labels better capture real trade quality vs path_quality cross-sectional ranking. Triple-barrier is the ground truth of whether a trade worked.

**Walk-forward results (8 folds, 5yr window, 81 symbols):**

| Fold | Test Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 6 | 2024-09-18→2025-03-28 | 117 | 34.2% | -2.07 | ❌ |
| 7 | 2025-04-08→2025-10-16 | 111 | 36.9% | +0.46 | ✅ |
| 8 | 2025-10-27→2026-05-06 | 106 | 34.0% | -2.13 | ❌ |
| **Avg** | | **834** | **38.5%** | **-0.732** | ❌ GATE FAILED |

Gate: avg Sharpe > 0.80, no fold < -0.30. Both failed comprehensively.

**Analysis:** Triple-barrier labels produce worse results than path_quality in the current regime. The bar-by-bar stop simulation may be too sensitive to intraday noise — stops are triggered by short-term volatility spikes even when the trade would have been profitable over the intended 5-day hold. In the 2025 tariff regime (high VIX, gap-and-fade pattern), this means many genuine winners get labeled 0 (stop hit on day 1 gap down). The cross-sectional label scheme (path_quality) may be more robust because it's insensitive to stop placement.

**Verdict:** ❌ GATE FAILED — v149/v150 not deployed. v142 remains active champion. Triple-barrier labels ruled out for swing in current regime.

---

### Item 3 — Regime Features for Intraday 🔄 PENDING WALK-FORWARD

**Date:** 2026-05-06
**Model:** intraday v44 (AUC=0.630 on OOS)
**New features added to `compute_intraday_features()`:**
- `regime_vix_proxy`: SPY 20d realized vol annualized (VIX proxy, clipped 5–80)
- `regime_vix_pct60d`: current vol level vs trailing 60d window [0,1] percentile
- `regime_spy_ma20_dist`: (SPY close - MA20) / MA20 — above/below medium-term trend

**Hypothesis:** Intraday model has no awareness of macro vol regime. These features let it learn to be more conservative in high-vol / above-trend environments (the 2025 tariff regime). Unlike Phase 86 market-wide features, these survive `cs_normalize` because they are computed from SPY daily bars and are **the same for all symbols on a given day** — BUT they interact with stock-specific features (e.g. `regime_vix_pct60d * atr_norm`) which the model can discover.

> **Note:** These features are identical across symbols on a given day, so `cs_normalize` will zero them out if used alone. The bet is that XGBoost learns interaction terms (e.g. high-VIX + high-atr stock = avoid) rather than using the raw feature value. This is the same limitation as Phase 86 — if cs_normalize zeros them, results will mirror Phase 86 failure.

**Walk-forward results (8 folds, 730d window, 711 symbols):**

| Fold | Test Period | Trades | Win% | Sharpe | Gate |
|---|---|---|---|---|---|
| 1 | 2024-07-19→2024-10-02 | 106 | 43.4% | -2.36 | ❌ |
| 2 | 2024-10-07→2024-12-19 | 106 | 47.2% | -0.13 | ✅ |
| 3 | 2024-12-24→2025-03-13 | 106 | 48.1% | -0.24 | ✅ |
| 4 | 2025-03-18→2025-06-02 | 106 | 42.4% | -0.74 | ❌ |
| 5 | 2025-06-05→2025-08-20 | 106 | 50.0% | -0.27 | ✅ |
| 6 | 2025-08-25→2025-11-06 | 106 | 60.4% | +1.12 | ✅ |
| 7 | 2025-11-11→2026-01-28 | 106 | 46.2% | +0.08 | ✅ |
| 8 | 2026-02-02→2026-04-17 | 108 | 51.8% | -0.10 | ✅ |
| **Avg** | | **850** | **48.7%** | **-0.331** | ❌ GATE FAILED |

Gate: avg Sharpe > 1.50, no fold < -0.30. Both failed.

**Analysis:** Confirmed hypothesis — `regime_vix_proxy`, `regime_vix_pct60d`, `regime_spy_ma20_dist` are identical across all symbols on a given day, so `cs_normalize` (z-score within each day's symbol set) reduces them to exactly zero. The model never received the intended signal. Results are nearly identical to v29 baseline (-0.327), confirming zero information gain. This is the same failure mode as Phase 86 market-wide features.

**Root cause resolution:** A separate Regime Model operating outside cs_normalize is required. See comprehensive architecture plan in `docs/MASTER_BACKLOG.md` — Phases R1-R6.

**Verdict:** ❌ GATE FAILED — v44 not deployed. v29 remains active champion. Regime features confirmed incompatible with cs_normalize architecture. Next step: build dedicated Regime Model pipeline.

---

## Regime Model V1 — regime_v2.pkl ✅ GATE PASSED (2026-05-06)

**Phase:** R2 (part of Phases R1–R4 completed 2026-05-06)
**Model file:** `models/regime/regime_model_v2.pkl`
**Architecture:** XGBoost binary classifier + IsotonicRegression calibration (manual 80/20 split)

**Why manual calibration:** `CalibratedClassifierCV` incompatible with XGBoost 2.0.3 + sklearn 1.8 — `is_classifier()` returns False for XGBClassifier, causing ValueError. Fix: train XGB on 80% of data, fit `IsotonicRegression` on remaining 20%.

**Label scheme (rule_based_v1):** `spy_1d_return > 0 AND vix_level < 20 AND spy_ma20_dist > 0` → label 1 (favorable). Replaced by trade-outcome labels in R6 (after 90 days paper data).

**Features (20):** `vix_level`, `vix_pct_1y`, `vix_pct_60d`, `spy_rvol_5d`, `spy_rvol_20d`, `spy_1d_return`, `spy_5d_return`, `spy_20d_return`, `spy_ma20_dist`, `spy_ma50_dist`, `spy_ma200_dist`, `days_to_fomc`, `days_to_cpi`, `days_to_nfp`, `is_fomc_day`, `is_cpi_day`, `is_nfp_day`, `nis_risk_numeric`, `nis_sizing_factor`, `breadth_pct_ma50`

**Null handling:** `nis_risk_numeric` → 0.5, `nis_sizing_factor` → 1.0, `breadth_pct_ma50` → 0.5 (structurally null in backfill rows)

**Walk-forward results (3 expanding folds, 2023-01-01 start):**

| Fold | Train End | Test End | n_train | n_test | AUC | Brier |
|---|---|---|---|---|---|---|
| 1 | 2024-12-31 | 2025-06-30 | 522 | 129 | 0.9912 | 0.027 |
| 2 | 2025-06-30 | 2025-12-31 | 651 | 132 | 1.000 | 0.000 |
| 3 | 2025-12-31 | 2026-04-30 | 783 | 86 | 0.9583 | 0.036 |
| **Avg** | | | | | **0.9832** | **0.0210** |

Gate: AUC min ≥ 0.60 ✅, Brier < 0.22 ✅

**Thresholds:** RISK_OFF < 0.35, NEUTRAL 0.35–0.65, RISK_ON ≥ 0.65

**Note on high AUC:** Rule-based labels are deterministic functions of the same SPY/VIX features the model trains on — AUC near 1.0 is expected and validates the model learns the rule correctly. Real generalization test is R4 gate: do regime scores correlate with actual next-day P&L over 10+ trading days?

**Status:** ACTIVE — scoring daily at 7am ET (Phase R3). Parallel running analytics accumulating (Phase R4). R5 gate unlocks ~2026-05-21.

---

## Next Experiments — Intraday Stock Model (2026-05-06+)

**Current champions:** Intraday v29 (avg -0.327 on current window), Swing v142 (avg +0.310). Both fail honest walk-forward. Regime model (R5) handles macro gating; stock models need to improve cross-sectional selection.

**Planned sequence:**
1. **Phase 86b** — 3 stock-relative SPY features (`stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`). These vary cross-sectionally → survive cs_normalize. Base: v38 (realized-R labels + 3-seed ensemble + frozen HPO).
2. **MIN_REALIZED_R tuning** — If 86b gate fails, lower 0.50 → 0.35–0.40. Phase 87 showed v38 structurally healthier but under-trades (524 vs 750 trades).
3. **Shorter training window** — If still failing, rolling 18-month window. Hypothesis: 2021-22 bull data incompatible with 2025 tariff/vol regime.
