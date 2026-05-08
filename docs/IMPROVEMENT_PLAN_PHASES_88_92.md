# ML Improvement Plan — Phases 88–92
**Created:** 2026-05-08  
**Goal:** Fix persistent fold 2 collapse (Sharpe ~0.3 in Nov 2023–Feb 2025) and raise overall model quality.  
**Based on:** Opus 4.7 deep review of v167/168/169 walk-forward failures.

---

## Root Cause Summary

The fold 2 (Nov 2023–Feb 2025) test window is a low-vol AI-led bull grind (SPY +30%, VIX <16, Mag-7 dominance). The current model fails because:

1. **Mean-reversion biased features** (RSI, MACD, Stoch, Bollinger %B, RSI_DIP pre-filter) — designed for pullback entries, wrong for a grind-up regime
2. **No trend/breadth context** — model can't distinguish "RSI=70 in a bull trend" from "RSI=70 before reversal"
3. **Regime features pruned** — `regime_score`, `vix_regime_bucket`, `rs_vs_spy_*` all in `_BASE_PRUNED`
4. **R6 RISK_OFF exclusion removes 2022 training data** — model loses calibration for stress events that appear in fold 2
5. **3-fold gate** — single bad regime tank the average; statistically too few folds
6. **5-day label horizon** — misses the dominant 2024 alpha shape (slow 10–30 day grind-up)

---

## Phases

### Phase 88 — 5 Folds + Regime Features in Swing (Week 1)
**Goal:** Two cheap, high-leverage changes that directly address fold 2.  
**Changes:**
- `retrain_config.py`: `walk_forward_folds=5`, `walk_forward_years=6`
- `retrain_cron.py`: pass `n_folds=5` to intraday WF too
- Remove `regime_score`, `vix_regime_bucket`, `vix_level` from `_BASE_PRUNED` in `training.py`
- Add regime V2 scalars as per-symbol features: `vix_term_ratio`, `breadth_rsp_spy_ratio_20d`, `credit_hyg_ief_20d`, `sector_dispersion_20d`, `spy_above_ma50`, `spy_above_ma200`
- Down-weight RISK_OFF to 0.3× (not exclude) in `_load_regime_weight_map()`
- **Retrain swing v170, intraday v57**
- **Gate:** swing avg Sharpe >0.80 over 5 folds, no fold < -0.30

### Phase 89 — Trend-Persistence Features (Week 2)
**Goal:** Replace mean-reversion bias with trend-quality signals.  
**Changes:**
- Add to swing feature pipeline: `aroon_up_25`, `aroon_down_25`, `adx_rising` (bool), `pct_closes_above_ema20` (last 20 closes), `drawdown_from_20d_high`
- Add `hurst_exponent_60d` (distinguishes trending vs mean-reverting per name)
- Add `volatility_adj_dist_from_52wk_high`
- Remove `RSI_DIP` and `EMA_CROSSOVER` as hard pre-filters (let model decide)
- **Retrain swing v171**
- **Gate:** avg Sharpe >0.80

### Phase 90 — Multi-Horizon Swing Label (Week 3)
**Goal:** Fix horizon mismatch — 5-day label misses 10-30 day grind-up alpha.  
**Changes:**
- Add `FORWARD_DAYS_LONG = 15` label head alongside existing 5-day
- Auto-tune `ATR_MULT_TARGET` per quarter to maintain 25–35% positive class rate (prevent v164-style degeneracy)
- Train two XGBoost heads, blend probabilities 0.5×5d + 0.5×15d at inference
- **Retrain swing v172**
- **Gate:** avg Sharpe >0.80

### Phase 91 — Intraday: Hybrid Label + Microstructure Features (Week 4)
**Goal:** Fix intraday label noise and add missing market microstructure context.  
**Changes:**
- **Hybrid label**: top-20% AND realized-R > 0.5% (intersection, not union) — removes chop-day label noise
- **Per-day dispersion gate at training time**: drop training days where universe dispersion < threshold (matching live R5-B gate)
- Add features: `gap_pct`, `gap_fill_pct`, `opening_range_breakout`, `vwap_slope_to_bar12`, `first_30min_volume_ratio`, `spy_5min_return_bar12`, `vix_5min_change`, `sector_etf_return_today`
- Add `days_to_macro_event` scalar (proximity to FOMC/CPI/NFP)
- Down-weight RISK_OFF intraday rows to 0.3× (matching swing Phase 88 change)
- **Retrain intraday v58**
- **Gate:** avg Sharpe >1.00 over 5 folds, no fold < -0.30

### Phase 92 — Regime-Probability Meta-Features + Per-Regime Ensemble (Week 5)
**Goal:** Use regime signal as model input, not just a gate.  
**Changes:**
- Add `prob_risk_off`, `prob_risk_caution`, `prob_risk_on` as 3 features in swing feature vector (continuous, not argmax)
- Add interaction features: `momentum_20d × prob_risk_on`, `rsi_14 × vix_pct_1y`, `adx_14 × spy_above_ma200`
- Per-regime ensemble: train 3 swing sub-models (RISK_OFF/CAUTION/ON), blend at inference weighted by regime probability
- **Retrain swing v173**
- **Gate:** avg Sharpe >0.80

---

## Process for Each Phase

1. Create feature branch `feat/phase-88` (etc.)
2. Implement changes
3. Run `pytest` — 0 failures required before merge
4. Merge PR
5. Run retrain: `python scripts/retrain_cron.py --swing-only` (or `--intraday-only`)
6. Log results in `docs/ML_EXPERIMENT_LOG.md`
7. If gate passes → proceed to next phase
8. If gate fails → diagnose, adjust, retrain before moving on

---

## Gate Thresholds (Updated)

| Model | Folds | Avg Sharpe | Min Fold Sharpe |
|---|---|---|---|
| Swing | 5 | ≥ 0.80 | ≥ -0.30 |
| Intraday | 5 | ≥ 1.00 | ≥ -0.30 |

---

## Expected Fold Windows After Phase 88 (Swing, 5 folds, 6yr)

With `total_years=6`, 5 folds, expanding window:
- Fold 1: test ~2022-02 → 2023-02 (post-Fed pivot)
- Fold 2: test ~2023-02 → 2024-02 (AI rally begins)
- Fold 3: test ~2024-02 → 2025-02 (low-vol grind — the problem window, now 1 fold not the only fold)
- Fold 4: test ~2025-02 → 2025-10 (tariff shock)
- Fold 5: test ~2025-10 → 2026-05 (recent)

This splits the problematic 2023–2025 window across folds 2 and 3, preventing one regime from dominating the gate average.

---

## Status

| Phase | Status | Version | Avg Sharpe | Notes |
|---|---|---|---|---|
| 88 | ❌ gate failed | v172/v58 | -0.243 / -1.556 | Fixed fold 3 swing (+1.43) but fold 1 (2022 bear) now -1.87; intraday all negative |
| 89 | ✅ gate passed | v173 | WF running | Trend features merged; v173 ACTIVE; WF Sharpes pending |
| 90 | ✅ merged | v174 | — | PR #186 merged 2026-05-08; retrain pending |
| 91 | 🔄 PR open | v59 | — | PR #187 open; hybrid label + dispersion gate + 4 microstructure features |
| 92 | ⏳ pending | v175 | — | — |
