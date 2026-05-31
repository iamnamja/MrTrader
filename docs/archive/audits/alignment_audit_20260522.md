# MrTrader Alignment Audit — 20260522

**Run date:** 2026-05-22T15:23:24.085087Z

**Issues:** 6 (1 CRITICAL, 4 HIGH, 1 MEDIUM)  
**OK:** 19

---

## [1.1] Feature Construction

- 🟠 **HIGH**: Training: fetch_fundamentals not explicitly False — may fetch live data
- 🟠 **HIGH**: Training: as_of_date NOT passed to engineer_features — PIT violation risk
- ✅ Training: regime_score passed ✓
- ✅ WF: fetch_fundamentals=False ✓
- ✅ WF: as_of_date passed ✓
- ✅ WF: regime_score passed ✓
- ✅ Live PM _analyze_swing_portfolio not directly inspectable: type object 'PortfolioManager' has no attribute '_analyze_swing_portfolio'

## [1.2] Label Construction

- ✅ Training labels: FORWARD_DAYS=15, STEP_DAYS=5, ATR_TARGET=1.5x, ATR_STOP=0.5x
- ✅ WF ATR_STOP_MULT=0.5 matches training ✓
- ✅ WF ATR_TARGET_MULT=1.5 matches training ✓
- ✅ ⚠ WF measures P&L of selected positions only; training labels entire universe. This is a structural misalignment (selection bias in WF).

## [1.3] Universe / Survivorship

- 🟠 **HIGH**: Training: no PIT universe filter found — may train on current S&P500 only (survivorship bias)
- ✅ WF: uses PIT universe (pit_union) ✓
- ✅ Live PM watchlist method not directly inspectable

## [1.4] Normalization

- 🟠 **HIGH**: Training calls cs_normalize on X_train before fitting model — but model.predict path also calls cs_normalize. Double-normalization?
- 🔴 **CRITICAL**: STRUCTURAL: Training cs_normalize uses N=full_universe rows (700+). WF cs_normalize uses N=symbols_with_data_on_day (varies, often < 200). Live PM cs_normalize uses N=open_positions (typically 5-20). These three distributions are incomparable — z-scores are not portable.
- ✅ Training: cs_normalize NOT in _build_rolling_matrix ✓
- ✅ WF _normalize_for_inference: uses cs_normalize fallback ✓ (TS norm if model has _ts_norm_state)

## [1.5] Inference Path

- 🟡 **MEDIUM**: Live PM min_confidence=0.55 ≠ WF min_confidence=0.4
- ✅ WF: uses model.predict_with_vix ✓
- ✅ Inference path check partial: type object 'PortfolioManager' has no attribute '_analyze_swing_portfolio'
- ✅ WF min_confidence threshold: 0.4

## [1.6] Execution

- ✅ WF: entry price uses bar open (next-day open fill) — check for P0.1 fix
- ✅ WF: stop simulation checks bar low (intrabar) ✓
- ✅ WF: borrow cost modeled for shorts ✓

