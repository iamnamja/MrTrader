# MrTrader — Completed Phases Archive

Phases that are fully implemented and merged. Kept for historical reference.
Active roadmap: `docs/phases_58_onwards_plan.md`

---

## Foundation (Phases 1–56) — Complete

All phases 1–56 complete prior to 2026-04-28. Walk-forward gates passed for both models.
Full specs: `memory/phases_29_onwards_spec.md`, `memory/phases_48_onwards_spec.md`

| Model | Version | Avg Sharpe | Gate |
|---|---|---|---|
| Swing | v119 | +1.181 | ✅ >0.80 |
| Intraday | v29 | +1.830 | ✅ >1.50 (with Phase 85 gates) |

---

## Paper Trading Hardening — Complete (2026-04-28)

| Phase | Name | Key File(s) |
|---|---|---|
| A | State Durability | `_reconcile_positions()` on startup; DailyState table |
| B | Swing Recovery | Swing re-proposes correctly after late restart |
| C | Proposal Hygiene | 60-min TTL, 4 PM Redis purge, one-approval-per-symbol-per-day |
| 67 | Trader Entry Quality Check | `app/strategy/entry_quality.py` |
| 68 | Live Macro/Market Gate | `app/agents/premarket.py` |
| 69 | Dynamic Stop Trailing + Partial Exits | `app/strategy/signals.py` |
| 70 | PM Re-Scoring / WITHDRAW Signal | `app/agents/portfolio_manager.py` |

---

## News Intelligence Service — Complete (2026-04-29)

| Phase | Name | Key File(s) |
|---|---|---|
| 58 | Earnings Calendar Gate (Finnhub) | `app/calendars/earnings.py` |
| 59 | Macro Calendar Awareness (NIS Tier 1) | `app/agents/premarket.py` |
| 60 | Structured NewsSignal + Haiku Scorer | `app/news/` stack |
| 61 | Decision Audit Trail | `app/database/decision_audit.py` |
| 62 | Morning NIS Digest (09:00 cache warm) | `app/agents/portfolio_manager.py` |
| 63 | News as PM Scoring Overlay | `app/agents/portfolio_manager.py` |
| 64 | NIS Backfill Script (Batch API) | `scripts/backfill_nis_history.py` |
| 71 | Correlation Gate in RM | `app/agents/risk_rules.py` |
| 72 | NIS re-check for held swing positions | `app/agents/portfolio_manager.py` |
| 73 | Overnight gap open price accuracy | `app/agents/trader.py` |
| 74 | NIS & audit dashboard API (5 endpoints) | `app/api/nis_routes.py` |

---

## Safety + Hardening — Complete (2026-05-01/02)

| Phase | Name | Key File(s) |
|---|---|---|
| 75 | Kill switch wired into all agents | `app/agents/*.py`, `app/live_trading/kill_switch.py` |
| 76 | Fix slippage measurement | `app/agents/trader.py` |
| 78 | Order lifecycle: persist pending, partial fills, reconciliation | `app/agents/trader.py` |
| 81 | Earnings gate fail-closed + Finnhub primary | `app/calendars/earnings.py` |
| 82 | Persist `_peak_equity` across restarts | `app/agents/risk_manager.py` |
| 83 | Deadman switch + external watchdog | `app/live_trading/deadman.py` |

---

## Audit Trail Hardening — Complete (2026-04-29)

| Gap | What Was Built |
|---|---|
| 1 — Feature explainability | `top_features` JSON added to `decision_audit`; PM writes top-8 feature values |
| 2 — Outcome backfill | `_run_eod_jobs()` at 16:30 ET; `retrain_cron.py` also calls post-retrain |
| 3 — Daily summary | `app/database/daily_summary.py` → `write_daily_summary()` |
| 4 — Decision audit in report | `scripts/paper_trading_report.py` |
| 5 — Day replay | `scripts/replay_day.py --date YYYY-MM-DD` |

---

## Phase 85 — Intraday PM Abstention Gates — Complete ✅ (2026-05-02)

**Result:** Walk-forward avg Sharpe **+1.830** (gate: >1.50 ✅), no fold below -0.30 ✅.

Gates in `app/agents/portfolio_manager.py`:
- VIX ≥ 25 → skip all intraday entries
- SPY < MA20 → skip all intraday entries

**Why 86+87 still needed:** Phase 85 is a runtime patch. v29 training signal is still corrupted (forced positive labels on bad days). Phases 86b+87 fix the root cause in training.

---

## Regime Model Pipeline — Complete ✅ (2026-05-06)

### Phase R1 — Backfill RegimeSnapshot History
**Gate passed:** 872 backfill rows written (≥500 required). Scored using rule-based labels: `spy_1d_return > 0 AND vix_level < 20 AND spy_ma20_dist > 0` → RISK_ON. Rows cover 2023-01-01 to 2026-04-30.

Key files: `app/database/models.py` (RegimeSnapshot table), `app/ml/regime_features.py`, `app/ml/regime_feature_builder.py`

### Phase R2 — Regime Model V1 Training (regime_v2.pkl)
**Gate passed:** AUC min 0.9583 ≥ 0.60 ✅, Brier 0.0210 < 0.22 ✅

**Architecture:** XGBoost binary classifier + IsotonicRegression calibration (manual 80/20 split — `CalibratedClassifierCV` incompatible with XGBoost 2.0.3 + sklearn 1.8). 20 features: VIX level/percentiles, SPY returns/MA distances/realized vol, FOMC/CPI/NFP calendar proximity, NIS risk/sizing, breadth_pct_ma50.

**Walk-forward results (3 expanding folds, 2023-01-01 start):**

| Fold | Train End | Test End | n_train | n_test | AUC | Brier |
|---|---|---|---|---|---|---|
| 1 | 2024-12-31 | 2025-06-30 | 522 | 129 | 0.9912 | 0.027 |
| 2 | 2025-06-30 | 2025-12-31 | 651 | 132 | 1.000 | 0.000 |
| 3 | 2025-12-31 | 2026-04-30 | 783 | 86 | 0.9583 | 0.036 |
| **Avg** | | | | | **0.9832** | **0.0210** |

Thresholds: RISK_OFF < 0.35, RISK_ON ≥ 0.65, NEUTRAL in between.

Key files: `app/ml/regime_training.py`, `app/ml/regime_model.py`, `scripts/train_regime_model.py`

### Phase R3 — Premarket Integration
Regime model scores at 7am ET daily (trigger=`premarket`). Startup catchup if no row exists and hour < 11:30 ET. Post-event re-evals scheduled via APScheduler for FOMC/CPI/NFP days. Staleness haircut: >4h old during market hours → 20% penalty + `stale=True` flag. ProposalLog writes `regime_score_at_scan`, `regime_label_at_scan`, `regime_trigger_at_scan`.

Key files: `app/agents/premarket.py`, `app/agents/portfolio_manager.py`, `app/database/models.py`

### Phase R4 — Parallel Running Analytics
Dashboard `RegimeModelWidget` (score gauge, threshold markers, label badge, auto-refresh). Weekly summary logger + daily divergence tracker in PM. API endpoints: `/regime/current`, `/regime/history`, `/regime/analytics`.

**Gate accumulation:** needs 10+ trading days + FOMC test (2026-05-07). R5 unlocks ~2026-05-21.

Key files: `app/api/routes.py`, `frontend/src/App.tsx`, `tests/test_regime_r4_analytics.py`

### Data Backfill (2026-05-06)
After R3/R4 deployment: 872 backfill RegimeSnapshot rows scored with regime_v2 (model_version=2 written to each). 268 pre-R3 proposal_log rows backfilled with nearest-date snapshot scores (0 remaining NULLs).

Key file: `scripts/backfill_proposal_regime_scores.py`

---

## Code-Grounded Critical Findings (2026-05-01) — All Resolved

| Finding | Fix | Status |
|---|---|---|
| Kill switch decorative | Phase 75 | ✅ Fixed |
| Slippage broken | Phase 76 | ✅ Fixed |
| Earnings gate fails open | Phase 81 | ✅ Fixed |
| `_peak_equity` resets on restart | Phase 82 | ✅ Fixed |
| No external watchdog | Phase 83 | ✅ Fixed |
| Pending orders lost on restart | Phase 78 | ✅ Fixed |
| MetaLabel not in live PM | Decision: keep dormant | ✅ Documented |
| Survivorship bias (universe) | Phase 79 | ⏳ Pending |
| Bar-12 never sensitivity-tested | Phase 80 | ⏳ Pending |
