# MrTrader — Active Phase Roadmap

**Last updated:** 2026-05-05 (EOD update)
**Status:** Paper trading active. 5 open positions (ZS closed target_hit +6.8%). Retrains running. Models: swing v142, intraday v29.
**Completed phases:** `docs/phases_archive.md`

---

## Current Model State

| Model | Version | Avg Sharpe | Gate | Features | Notes |
|---|---|---|---|---|---|
| Swing | v142 | +1.181 | ✅ >0.80 | 84 | Active champion — pre-dates macro NIS |
| Swing | v144 | ~0.28 | ❌ | 89 (+ stock NIS) | Gate failed — NIS didn't help fold 3 collapse |
| Swing | v145 | ~0.28 | ❌ | 94 (+ macro NIS) | Gate failed — macro NIS alone insufficient |
| Swing | v146 | 🔄 training | — | 89 (84 + macro NIS) | Tonight's retrain — in progress |
| Intraday | v29 | +1.830 | ✅ >1.50 | 50 | Active champion (cross-sectional labels) |
| Intraday | v39/v40 | negative | ❌ | 58/63 | Realized-R scheme failed (AUC ~0.51) |
| Intraday | v30 | 🔄 training | — | 61 (50 + 5 NIS + 3 SPY-relative + 3 other) | Tonight's retrain — in progress |

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

### Phase 86b — Stock-Relative SPY Features ✅ (PR #129, 2026-05-05)
- Added 3 intraday features that survive cs_normalize: `stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`
- Feature count: 58 → 61. Included in tonight's intraday v30 retrain.

### Fix — Bulk Finnhub Earnings Prefetch ✅ (PR #128, 2026-05-05)
- `prefetch()` was making 500 individual Finnhub calls on startup → instant 429 storm
- Fixed to single bulk call (Finnhub returns full calendar server-side regardless of symbol filter)

### Phase 88 — Continuous Opportunity Score (Dynamic Regime Gates) ✅ (PR #130, 2026-05-05)
- Replaced hard VIX ≥ 25 / SPY < MA20 binary gates with a graduated PM-level opportunity score
- `score = 0.35*vix_score + 0.20*vix_trend + 0.30*range_score + 0.15*eff_score`
- Score ≥ 0.70 → normal sizing, 0.40–0.69 → reduced candidates, < 0.40 → skip
- All stays in PM layer (cs_normalize would zero market-wide signals if used as model features)

### Phase 89b — Sector ETF History Backfill ✅ (PR #132, 2026-05-05)
- Extended Polygon daily bar cache to 11 sector ETFs
- `sector_momentum` now populated during training (was zeroed previously)
- Also wired Phase 89a fundamentals into swing training pipeline

### Fix — Unified Proposal Log + Trade History Cleanup (Phases A/B/C) ✅ (PR #133, 2026-05-05)
- Unified `proposal_log` schema (swing + intraday in one table)
- `exit_price` captured on trade close; `exit_reason` field added
- Pending fills promoted correctly on reconcile

### Fix — Gate Calibration Persistence ✅ (PR #134, 2026-05-05)
- `decision_audit` table extended: `gate_category`, `price_at_decision`, `direction`, `outcome_fetched_at`
- New `nis_macro_snapshots` table — persists daily premarket NIS macro context so API survives restarts
- New `scan_abstentions` table — logs gate1a/gate1c SPY-level abstentions with SPY outcome backfill
- `gate_categories.py` classifies all block reasons into alpha/quality/risk/structural/scan
- EOD backfill job fetches Alpaca T+4h/T+24h prices for calibratable blocked rows
- DB migration: `python scripts/migrate_gate_calibration.py` ✅ run 2026-05-05

### Fix — bars_held Persistence Across Restarts ✅ (PR #135, 2026-05-05)
- `bars_held` was only written to DB on trade close → always 0 for active trades after restart
- `min_hold_bars=3` guard in `check_exit()` was blocking ALL exits (including target hits) post-restart
- Fix 1: on reconcile, `bars_held = max(db_value, calendar_days_since_entry)` 
- Fix 2: persist `bars_held` to DB each time it increments (once per calendar day)
- ZS position hit target $140.85 at $141.94 (+6.8%) and closed correctly after this was diagnosed

### Fix — Summary Endpoint Thread-Safety + Live P&L in Trades ✅ (PR #136, 2026-05-05)
- `_alpaca()` was called inside `asyncio.to_thread()` in the summary endpoint → intermittent blank KPIs
- Fixed: call `_alpaca()` in main async context, only thread the blocking network calls
- `_pnl_from_alpaca` timeout now caught (was causing uncaught 500 on timeout)
- Trade history endpoint now enriches ACTIVE trades with live `unrealized_pl`, `unrealized_plpc`, `current_price`

---

## Retrains In Progress — 2026-05-05 EOD

Both models are retraining now that macro NIS backfill is complete. Champions (v142/v29) were trained before Phase 90 macro features existed.

| Model | Current | New version | Target features | Status |
|---|---|---|---|---|
| Swing | v142 (84 feat) | v146 | 89 feat (84 + 5 macro NIS) | 🔄 running |
| Intraday | v29 (50 feat) | v30 | 58 feat (50 + 5 NIS + 3 SPY-relative) | 🔄 running |

Gate thresholds: swing avg Sharpe > 0.8, intraday > 1.5.

**If swing still fails gate:** Try 3yr training window — Fold 3 (2025) is a structurally different regime that 2021–2022 data may be distorting.

---

## Backlog — Ordered by Expected Impact

### Phase 86b — Stock-Relative Interaction Features ✅ (PR #129, 2026-05-05)

- `stock_vs_spy_5d_return`: stock 5d return − SPY 5d return (idiosyncratic trend)
- `stock_vs_spy_mom_ratio`: stock 1d return − SPY 1d return (idiosyncratic daily momentum)
- `gap_vs_spy_gap`: stock overnight gap − SPY gap (stock-specific gap signal)
- All vary per-symbol within each day → survive cs_normalize (unlike raw SPY values)
- Feature count: 58 → 61 intraday features
- **Walk-forward pending:** included in tonight's v30 retrain

---

### Phase 89a — Historical Fundamentals Backfill ✅ (2026-05-05)

- `scripts/backfill_fundamentals_history.py` run: 391/430 symbols, 11,285 PIT snapshots → `data/fundamentals/fundamentals_history.parquet`
- Training auto-loads the parquet and overrides `profit_margin`, `revenue_growth`, `debt_to_equity` with PIT-correct values per training window
- `pe_ratio`, `pb_ratio` still pruned (require live price at filing date — harder to backfill correctly)
- **Next retrain will include fundamentals for the first time**

### Phase 89b — Sector ETF History + sector_momentum Un-pruning ✅ (2026-05-05)

- `scripts/backfill_sector_etf_history.py` fixed (was using Alpaca IEX = 100 bars; switched to Polygon S3 = 864 bars per ETF, full 3yr)
- 11 ETFs × 864 bars → `data/sector_etf/sector_etf_history.parquet` + `data/cache/daily/XLK.parquet` etc.
- `training.py` now loads ETF history at startup and computes PIT `sector_momentum` (20d) and `sector_momentum_5d` (5d) per training window
- Removed `sector_momentum` and `sector_momentum_5d` from `PRUNED_FEATURES`
- Also updates `momentum_20d_sector_neutral`, `momentum_60d_sector_neutral`, `momentum_5d_sector_neutral` from PIT values

### NIS Backfill — Stock-Level (Backlog)

**Stock-level NIS** (`NewsSignalCache`) only has data from May 2025 onwards (430 symbols, ~1yr). Training windows go back 2-3 years — the model sees NaN for NIS in most training rows, limiting signal quality.

**Fix:** Per-symbol historical news backfill via Polygon → LLM scorer (same approach as `backfill_macro_nis_llm.py` but per-symbol).
- **Cost estimate:** ~$50-100 in LLM API calls (430 symbols × ~500 articles/year × 2yr)
- **Precondition:** Review tonight's v30 SHAP — if NIS features have low importance, deprioritize
- **Script needed:** `scripts/backfill_stock_nis_history.py` (doesn't exist yet)

---

### Phase 100 — Alpaca as Single Source of Truth (High, 2-3 days)

Eliminate DB-vs-Alpaca position state divergence at root. DB becomes append-only audit ledger; always read live state from Alpaca.

**Why:** bars_held, stop/target drift, and reconcile bugs all stem from maintaining duplicate state. Alpaca is authoritative by definition.

**Precondition:** Phase 99 ✅ (complete). **Not safe during market hours** — touches live position tracking path.

---

### Gate Calibration Tuning (Ongoing, as data accumulates)

New gate calibration infrastructure (PR #134) is live. After ~2 weeks of data:
- Review `Analytics > Gate Calibration` for gates where `verdict = recalibrate`
- Tune or remove gates that block trades the market would have rewarded
- Focus on `alpha` and `quality` gate categories (calibratable)

---

### Phase 87a — Regression Labels (Deferred)

**Precondition:** Phase 86b ✅ + stable retrain results.
Binary labels discard magnitude. Regression target (predict realized R-multiple) would teach model to distinguish great setups from marginal ones. Deferred until cross-sectional label architecture is more thoroughly explored.

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
