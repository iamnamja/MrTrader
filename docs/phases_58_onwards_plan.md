# MrTrader — Active Phase Roadmap

**Last updated:** 2026-05-02
**Status:** Paper trading active. Safety phases all green. Phase 87 training in progress.
**Completed phases:** `docs/phases_archive.md`

---

## Current Model State

| Model | Version | Avg Sharpe | Gate | Notes |
|---|---|---|---|---|
| Swing | v119 | +1.181 | ✅ >0.80 | Active |
| Intraday | v29 | +1.830 | ✅ >1.50 | Active (+ Phase 85 gates) |
| Intraday | v38 | TBD | 🔄 | Phase 87 training 2026-05-02 |

---

## Intraday ML Improvement Sequence (Active)

### Phase 86 — Market Context Features ❌ REVERTED / REDESIGN PENDING

**Root cause of failure:** All 5 SPY features (spy_first_hour_range, spy_5d_return, etc.) are market-wide — same value for every symbol per day. After `cs_normalize` they become zero. v36 avg Sharpe: **-0.529**.

**Redesign (Phase 86b — after Phase 87):** Stock-relative interaction features that survive cs_normalize:
- `stock_vs_spy_5d_return`, `stock_vs_spy_mom_ratio`, `gap_vs_spy_gap`
- SPY daily bars plumbing already wired through simulator, walkforward, PM — no re-plumbing needed.

---

### Phase 87 — Realized-R Labels + 3-Seed Ensemble + Frozen HPO 🔄 IN PROGRESS

**Branch:** `feat/phase-87-label-fix-ensemble` (PR #121, 2026-05-02)

**Three changes bundled:**

1. **Realized-R labels (B+A):** `realized_R ≥ 0.5 AND abs_move ≥ 0.30%` → label 1. Days with no qualifying setups → all zero labels. No more forced top-20% on bad days.

2. **3-seed XGBoost ensemble (permanent):** Seeds 42/123/777, blend probabilities. Eliminates ~2.0 Sharpe HPO variance (v29 +1.830 vs v37 -0.219 on identical features).

3. **100-trial Optuna HPO → FROZEN_HPO_PARAMS:** Thorough search once, freeze params for all future retrains.

**Label formula:**
```python
realized_R = (realized_exit - entry) / stop_dist_use
absolute_move = abs(realized_exit - entry) / entry
label = 1 if (realized_R >= 0.5 and absolute_move >= 0.003) else 0
```

**Acceptance criteria:** avg Sharpe > 1.50, no fold < -0.30.

**Walk-forward:** Pending (training in progress 2026-05-02).

---

### Phase 86b — Stock-Relative Interaction Features ⏳ AFTER PHASE 87

**Precondition:** Phase 87 walk-forward result in hand.

Features that vary by symbol (survive cs_normalize):
- `stock_vs_spy_5d_return`: stock 5d return − SPY 5d return
- `stock_vs_spy_mom_ratio`: stock 1d momentum / SPY 1d momentum
- `gap_vs_spy_gap`: stock overnight gap − SPY overnight gap

**Files:** `app/ml/intraday_features.py`, `app/ml/intraday_training.py`
**Branch:** `feat/phase-86b-stock-relative-features`

---

### Phase 87a — Regression Labels (Realized R-Multiple) ⏳ AFTER PHASE 86b

**Precondition:** Phase 87 ✅ + Phase 86b ✅.

Replace binary classification with regression target (predict realized R-multiple). XGBoost objective: `reg:squarederror`. Rank by predicted R directly (no probability threshold). Label clipped to [-3.0, +3.0].

**Why deferred:** Changes the entire scoring pipeline. Do after binary classifier is stable with a baseline to compare against.

**Branch:** `feat/phase-87a-regression-labels`

---

## Phase 88 — Dynamic Regime Gates ⏳ AFTER PHASE 86b

**Precondition:** Phase 87 ✅ + Phase 86b (stock-relative features) ✅ — conceptually related, implement together.

**Problem with current Phase 85 gates:**
- VIX ≥ 25 is a hard binary cutoff — VIX 24.9 trades, VIX 25.1 doesn't. No reason that threshold is special.
- SPY < MA20 blocks the entire universe even when specific sectors are trending well independently.
- Both gates are fully on/off — no gradation for borderline days.
- All market-wide context stays in PM-level gates because cs_normalize would zero it out if baked into the model.

**What to build:**

### 88a — Sector-Level Abstention Gates

Replace the universe-wide SPY MA gate with per-sector gates:
- Map each candidate symbol to its sector ETF (XLK, XLF, XLE, XLV, XLI, etc.) using existing `_get_symbol_sector()`
- For each sector: compute sector ETF vs its own 20-day MA
- Block candidates in sectors where the sector ETF is below its MA
- Allow candidates in sectors where the sector ETF is above its MA — even if SPY overall is weak

This lets energy stocks trade on a strong oil day while tech is blocked after a sector rotation.

**Files:** `app/agents/portfolio_manager.py` — `select_intraday_instruments()`, new `_sector_allows_entry(symbol)` helper

### 88b — Continuous Opportunity Score (Replace Binary VIX Gate)

Instead of hard VIX ≥ 25 block, compute a 0–1 day opportunity score:

```python
def _intraday_opportunity_score(vix, vix_5d_avg, spy_first_hour_range, spy_first_hour_eff):
    # VIX: score drops as VIX rises, but also penalizes rapidly expanding VIX
    vix_score = max(0.0, 1.0 - (vix - 15) / 20)           # 1.0 at VIX=15, 0.0 at VIX=35
    vix_trend = max(0.0, 1.0 - (vix - vix_5d_avg) / 5)    # penalty if VIX spiking vs 5d avg
    # First-hour range: more range = more opportunity
    range_score = min(1.0, spy_first_hour_range / 0.008)   # full score at 0.80%+ first-hour range
    # First-hour efficiency: trending (not choppy)
    eff_score = spy_first_hour_eff                          # already 0–1
    return 0.35*vix_score + 0.20*vix_trend + 0.30*range_score + 0.15*eff_score
```

Use score to scale `max_candidates` rather than binary block:
- Score ≥ 0.70 → normal (max_candidates = 5)
- Score 0.40–0.69 → reduced (max_candidates = 2)
- Score < 0.40 → skip (abstain entirely)

### 88c — Intraday Regime Signal (First 30 Minutes)

SPY's first 30 minutes (bars 0–5) characterizes the day's regime:
- **Trending open:** SPY moves directionally >0.4% with efficiency >0.7 → increase max_candidates
- **Choppy open:** SPY range >0.5% but efficiency <0.4 (whipsaw) → reduce max_candidates to 1
- **Quiet open:** SPY range <0.2% → VIX-like low-opportunity signal, reduce candidates

**Files:** `app/agents/portfolio_manager.py` — `_market_regime_allows_entries()` refactored into `_compute_opportunity_score()`

**Walk-forward validation:** Run with Phase 85 gate replaced by Phase 88 dynamic gates. Gate: avg Sharpe ≥ v29+Phase85 baseline (+1.830). If equal or better with fewer blocked days → improvement confirmed.

**Important constraint:** All of 88a/b/c stay as PM-level runtime logic, not model features. Market-wide and sector-wide signals are zeroed by cs_normalize in the model — they must live in the PM layer.

**Branch:** `feat/phase-88-dynamic-regime-gates`

---

## Phase 89 — Training/Inference Data Alignment

**Motivation:** Audit (2026-05-03) revealed training data must mirror live inference data exactly. Currently several features are zeroed during training but fetched live, or vice versa. Goal: backfill historical values point-in-time so the model trains on the same signals it scores with.

### Phase 89a — Historical Fundamentals Backfill (High, 3 days)

**Problem:** `revenue_growth` is the top SHAP feature at live inference (fetched from SEC EDGAR/yfinance), but is zeroed during training (`--no-fundamentals`). Model learns on price signals only, then scores live setups using fundamental values it was never trained on.

**What to build:**
- `scripts/backfill_fundamentals_history.py` — quarterly P/E, P/B, revenue growth, profit margin, debt/equity from SEC EDGAR XBRL. Store by filing date (not report date) for point-in-time safety.
- `data/fundamentals/` parquet store keyed by `(symbol, as_of_date)`
- Update `train_model.py` to load from parquet instead of live API — remove `--no-fundamentals` flag
- Un-prune `pe_ratio`, `pb_ratio`, `revenue_growth`, `profit_margin`, `debt_to_equity` from `PRUNED_FEATURES`

**Acceptance criteria:** Walk-forward avg Sharpe improvement > 0.05 vs v119 baseline (+1.181).

**Branch:** `feat/phase-89a-historical-fundamentals`

---

### Phase 89b — Sector ETF History (Medium, 1 day)

**Problem:** `sector_momentum` and `sector_momentum_5d` are fetched live at inference (XLK, XLF, XLE etc. ETF returns) but zeroed during training. SPY daily bars plumbing already exists — extend it to sector ETFs.

**What to build:**
- Extend Polygon daily bar cache to include 11 sector ETFs (XLK, XLF, XLE, XLV, XLI, XLB, XLRE, XLU, XLP, XLY, XLC)
- Pass sector bars into training the same way `spy_daily_bars` is passed today
- Un-prune `sector_momentum`, `sector_momentum_5d` from `PRUNED_FEATURES`
- Also enables Phase 88a (sector-level abstention gates) with historical data for validation

**Acceptance criteria:** Sector momentum SHAP > 0.05 in trained model (confirms signal survives cs_normalize).

**Branch:** `feat/phase-89b-sector-etf-history`

---

### Phase 89c — Historical Options IV (Low, 2 days)

**Problem:** `options_iv_atm`, `options_put_call_ratio`, `vrp` fetched live via yfinance options chain, but training defaults them to 0.0. Options history on free tier only goes back ~2 years.

**What to build:**
- `scripts/backfill_options_iv.py` — daily ATM IV and put/call ratio from yfinance options history
- Wire into training feature computation for walk-forward folds within the 2yr window
- Alternative: drop these features entirely if backfill coverage is insufficient

**Acceptance criteria:** Either confirms IV adds signal (Sharpe improvement) or cleanly removes from feature set.

**Branch:** `feat/phase-89c-historical-options-iv`

---

## Safety + Hardening Backlog

### Phase 77 — Decision-Audit Dashboard (High, 2 days)

`decision_audit` table is populated but unread. Cannot answer "are the gates working?"

**What to build:**
- `GET /api/audit/summary`: aggregate win-rate by `block_reason`, model_score bucket vs realized 4h return
- Frontend tile + `scripts/backfill_decision_outcomes.py` (16:30 ET cron)

**Acceptance criteria:** After 2 weeks, can read "NIS sizing_multiplier 0.7–0.8 correlates with X bps lower realized return."

---

### Phase 79 — Point-in-Time Index Membership (High, 3 days)

Static universe lists (`SP_500_TICKERS`, `RUSSELL_1000_TICKERS`) reflect ~early 2026 membership. Delisted stocks 2021–2026 silently absent from training — inflates walk-forward Sharpe.

**What to build:**
- `app/data/universe_history.py` with `members_at(date)`
- `data/universe/sp500_membership.parquet`, `data/universe/russell1000_membership.parquet`
- Update `train_model.py` and `walkforward_tier3.py` to use `members_at(fold_train_start)`

**Acceptance criteria:** `members_at(date(2022,1,1))` returns names since delisted (e.g. WORK, PAGS).

---

### Phase 80 — Bar-12 Intraday Sensitivity Test (High, 1 day)

Bar 12 was chosen by intuition. Bars 9–15 have never been swept. If only bar 12 shows Sharpe > 1.5, the edge may be in-sample noise.

**What to build:** `scripts/bar_sensitivity.py --entry-offset N` → Sharpe + win-rate + trade count. Run for offsets 9–15.

**Acceptance criteria:** If 4 of 5 bars (10–14) pass Sharpe > 0.8 → robust. If only bar 12 → downgrade Sharpe expectations.

---

### Phase 84 — Integration Test: Full PM→RM→Trader Round Trip (Medium, 2 days)

No e2e test drives PM→RM→Trader→Alpaca-paper with persistent state. All "integration" tests use mocks.

**What to build:** `tests/test_e2e_round_trip.py` — drives one swing proposal end-to-end. Asserts Trade.status='ACTIVE' after fill; Trade.status='CLOSED' + pnl after stop/exit. Run on demand only (not in CI).

---

## NIS + Features Backlog

### Phase 64 — NIS as Swing Model Features ⏳ (overnight backfill first)

**Precondition:** Run `python scripts/backfill_nis_history.py --days 252 --workers 4` overnight.

**NIS features to add to swing training:**
- `nis_materiality_decayed_4h`, `nis_direction_score_3d`, `nis_article_count_4h`
- `nis_already_priced_in`, `nis_sizing_mult`
- Use point-in-time lookup only — no lookahead

**Gate:** Walk-forward avg Sharpe improvement > 0.05 vs baseline.

---

### Phase 65 — Source Expansion (after Phase 64 baseline)
- Reuters/MarketWatch RSS as second source
- Trading halts feed (NASDAQ Trader RSS) — hard gate, no LLM needed

### Phase 66 — Sonnet Escalation (after Haiku error rate measurable)
- Escalate to Sonnet for `materiality_score >= 0.70` or M&A/FDA/legal event types

---

## Deferred (After Live Trading + Calibration)

| Phase | Name | Why Deferred |
|---|---|---|
| 57 | Paper trading calibration review | Need 4+ weeks live data + all safety phases green ✅ |
| 90 | Tax + P&L Impact Review | Pre-live gate. NJ ~47% combined rate on short-term gains. Wash sale frequency from paper data. |
| 51b | Multi-scan intraday tuning | Need live data to know if 11:00/13:30 windows add alpha |
| Regime v2 | Finer-grained regime (bull/bear/chop) | Need live performance to see where model fails |
| Phase 80b | True Technical Day Trader (IntradayScalper) | Different architecture; requires L1/L2 tick data |
