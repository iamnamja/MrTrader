# MrTrader — Active Phase Roadmap

**Last updated:** 2026-05-05
**Status:** Paper trading active. 8 open positions. Models loaded: swing v142, intraday v29.
**Completed phases:** `docs/phases_archive.md`

---

## Current Model State

| Model | Version | Avg Sharpe | Gate | Features | Notes |
|---|---|---|---|---|---|
| Swing | v142 | +1.181 | ✅ >0.80 | 84 | Active champion |
| Swing | v144 | ~0.28 | ❌ | 89 (+ stock NIS) | Gate failed — NIS didn't help fold 3 collapse |
| Swing | v145 | ~0.28 | ❌ | 94 (+ macro NIS) | Gate failed — macro NIS alone insufficient |
| Intraday | v29 | +1.830 | ✅ >1.50 | 50 | Active champion (cross-sectional labels) |
| Intraday | v39/v40 | negative | ❌ | 58/63 | Realized-R scheme failed (AUC ~0.51) |

**Key insight:** Cross-sectional top-20% labels + walk-forward gates is the right architecture for intraday. Realized-R labels (Phase 87/88/89 realized-R experiments) produced AUC ~0.51 — random. Reverted in Phase 89.

---

## Completed Since Last Update

### Phase 87 — Realized-R Labels ❌ (gate failed, reverted)
Tried `realized_R ≥ 0.5` label scheme. AUC ~0.51 across all folds — features don't predict realized-R outcomes with 2h holds.

### Phase 88 — NIS NaN Encoding Fix + Loosen Label Threshold ✅
- Fixed NIS default encoding: `0.0/1.0` sentinels → `float("nan")` so XGBoost uses learned missing-value direction
- Deleted stale `feature_store.db` (had 89-feature rows, inhomogeneous filter kept modal 89, discarded new rows)
- Loosened `MIN_REALIZED_R` 0.5 → 0.40 (still failed gate)

### Phase 89 — Restore Cross-Sectional Labels ✅ (PR #126)
- Reverted intraday to cross-sectional top-20% per-day label scheme (same as gate-passing v29)
- `label_scheme = "cross_sectional_top20pct_phase89"`

### Phase 64 — NIS Features in Swing Model ✅ (PR #124)
- Added 5 stock NIS features to swing: `nis_direction_score`, `nis_materiality_score`, `nis_already_priced_in`, `nis_sizing_mult`, `nis_downside_risk`
- Added NIS features to intraday as well
- All use `float("nan")` defaults (not 0.0/1.0) so XGBoost handles missing data correctly

### Phase 90 — Macro NIS Features ✅ (PR #127)
- Added `_get_macro_nis_features_pit()` — queries `MacroSignalCache` for day-level macro sentiment
- 5 macro features: `macro_avg_direction`, `macro_pct_bearish`, `macro_pct_bullish`, `macro_avg_materiality`, `macro_pct_high_risk`
- Backfill scripts written and run:
  - `scripts/backfill_macro_nis.py` — aggregated stock NIS → MacroSignalCache (251 days, `backfill_v1`)
  - `scripts/backfill_macro_nis_llm.py` — Polygon historical news → LLM classifier → MacroSignalCache (259 days, `llm_news_v1`)

### Phase 99 — Decouple Nightly Retraining ✅ (PR #123)
Retraining now runs as subprocess, not inside uvicorn executor.

### Fix — Bulk Finnhub Earnings Prefetch ✅ (PR #128, 2026-05-05)
- `prefetch()` was making 500 individual Finnhub calls on startup → instant 429 storm
- Fixed to single bulk call (Finnhub returns full calendar server-side regardless of symbol filter)

---

## Next: Retrain Both Models with Full NIS + Macro NIS Features

**When:** After market close today (2026-05-05).

Both models need retraining now that macro NIS backfill is complete. Current champions (v142/v29) were trained before Phase 90 macro features existed.

| Model | Current | Target features | Command |
|---|---|---|---|
| Swing | v142 (84 feat) | 89 feat (84 + 5 macro NIS) | `python scripts/retrain_cron.py --swing-only` |
| Intraday | v29 (50 feat) | 63 feat (53 + 5 stock NIS + 5 macro NIS) | `python scripts/retrain_cron.py --intraday-only` |

Gate thresholds: swing avg Sharpe > 0.8, intraday avg Sharpe > 1.5.

**If swing still fails gate:** Try shorter training window (3yr vs 5yr) — Fold 3 (2025) is a structurally different regime that the 2021-2022 data may be distorting.

---

## Backlog — Ordered by Expected Impact

### Phase 86b — Stock-Relative Interaction Features (Medium, 1 day)

Features that survive cs_normalize because they vary by symbol:
- `stock_vs_spy_5d_return`: stock 5d return − SPY 5d return
- `stock_vs_spy_mom_ratio`: stock 1d momentum / SPY 1d momentum
- `gap_vs_spy_gap`: stock overnight gap − SPY overnight gap

SPY daily bars plumbing already wired. Safe to implement during market hours (no live path changes).

**Files:** `app/ml/intraday_features.py`, `app/ml/intraday_training.py`

---

### Phase 88 — Dynamic Regime Gates (Medium, 1-2 days)

Replace hard VIX ≥ 25 / SPY < MA20 binary gates with graduated PM-level signals.

**88a — Sector-level abstention:** Per-sector ETF vs its own 20d MA. Allows tech stocks on strong tech day even if SPY is weak.

**88b — Continuous opportunity score:**
```python
score = 0.35*vix_score + 0.20*vix_trend + 0.30*range_score + 0.15*eff_score
# Score ≥ 0.70 → normal, 0.40–0.69 → reduced candidates, < 0.40 → skip
```

**88c — First 30-min regime signal:** Trending/choppy/quiet open → scale `max_candidates`.

All stays in PM layer (not model features — cs_normalize would zero market-wide signals).

Safe to implement during market hours (PM logic only, no model changes).

---

### Phase 89a — Historical Fundamentals Backfill (High, 3 days)

`revenue_growth` is top SHAP feature at inference but zeroed during training (`--no-fundamentals`). Model trains on price signals, scores live setups with fundamentals it never saw.

- Backfill quarterly fundamentals from SEC EDGAR XBRL, store by filing date (point-in-time safe)
- Un-prune `pe_ratio`, `pb_ratio`, `revenue_growth`, `profit_margin`, `debt_to_equity`

Safe to implement during market hours (new script + data store only).

---

### Phase 89b — Sector ETF History (Medium, 1 day)

`sector_momentum` zeroed during training, fetched live. Extend Polygon daily bar cache to 11 sector ETFs.

Safe to implement during market hours.

---

### Phase 100 — Alpaca as Single Source of Truth (High, 2-3 days)

Eliminate the DB-vs-Alpaca position state divergence at root. DB becomes append-only audit ledger; `position_store.py` always reads live state from Alpaca.

**Precondition:** Phase 99 ✅ (complete).

**Not safe during market hours** — touches live position tracking path.

---

## Training Flags Reference

```bash
# Swing retrain (always --no-fundamentals --workers 8 to avoid crash)
python scripts/retrain_cron.py --swing-only --no-fundamentals --workers 8

# Intraday retrain
python scripts/retrain_cron.py --intraday-only

# Walk-forward only (no retrain)
python scripts/walkforward_tier3.py --swing
python scripts/walkforward_tier3.py --intraday --model-version 29
```
