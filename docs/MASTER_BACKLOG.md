# MrTrader — Master Backlog & Roadmap

**Last updated:** 2026-05-05  
**Status:** Paper trading only. Live deployment blocked until Phase 1+2 complete.  
**Decision:** Based on multi-LLM review (Claude + ChatGPT + Gemini, 2026-05-05), we are pausing model iteration and fixing statistical foundation + simulation realism first. See `docs/llm_review_synthesis.md` for full analysis.

---

## The Core Problem (Honest Assessment)

Three independent LLM reviews + internal re-validation converged on the same diagnosis:

1. **The walk-forward measures the model, not the strategy.** PM opportunity score, RM gates, position caps, costs, and earnings blackouts are all applied live but not simulated. Model Sharpe ≠ Strategy Sharpe.
2. **The champion Sharpe numbers are inflated.** v142 +1.181 and v29 +1.830 were computed on data windows that didn't include the hard Apr–Oct 2025 regime, with no transaction costs, no PM/RM gates simulated, and no purge/embargo between folds. They are not valid production metrics.
3. **NIS features encode time, not sentiment.** With 80% NaN (pre-May 2025), XGBoost learns `NaN = 2021-2024 regime`, `non-NaN = 2025 regime`. We encoded time into the model accidentally. NIS must be removed from training until fully backfilled.
4. **No transaction cost model.** Intraday costs of 10–20bps round trip on a 0.8×ATR target (~$0.80 on a $100 stock) erase 15–25% of gross edge. We don't know if net edge is positive.
5. **Multiple-comparisons problem.** ~15 model variants tested; picking the best Sharpe inflates the result. Deflated Sharpe (Bailey & López de Prado 2014) would likely cut reported numbers by 30–50%.

**What this means:** We are several rigorous-statistics steps away from knowing if we have a real edge. The right move is not more features or more retrains — it is better measurement.

---

## PHASE 1 — Statistical Truth (This Week: Tue–Thu)

*Stop making decisions on numbers we can't trust.*

### 1a. Transaction Costs in Walk-Forward ⬜ HIGH PRIORITY
**Why:** Every Sharpe number reported to date is gross of costs. For intraday this is material.  
**What:**
- Swing: 5bps round trip (3bps half-spread + 2bps fees @ $0.005/share on ~$100 stock)
- Intraday: 15bps round trip (8bps half-spread on 5-min bars + 7bps impact/fees)
- Add `cost_model` param to `walkforward_tier3.py` — deduct per-trade cost from P&L at entry/exit
- Re-run v142 and v29 walk-forwards with costs. Report net Sharpe.  
**Deliverable:** Net Sharpe numbers we can actually trust. Expected: swing drops ~0.1–0.2, intraday drops ~0.3–0.5.  
**Files:** `scripts/walkforward_tier3.py`, `app/backtesting/intraday_agent_simulator.py`

### 1b. Purge + Embargo Between Walk-Forward Folds ⬜ HIGH PRIORITY
**Why:** Train/test boundary allows labels computed on test-window data to contaminate training (e.g., a 5-day label computed starting the last train day extends into test).  
**What:**
- Swing: 10-day purge (drop the last 10 days of training data before test window)
- Intraday: 2-day purge
- 5-day embargo after test window before next fold starts (no training rows from that gap)  
**Files:** `scripts/walkforward_tier3.py` fold boundary logic  
**Deliverable:** Cleaner fold splits. Probably drops Sharpe slightly — that's the more honest number.

### 1c. Remove NIS Features From Training Pipeline ⬜ URGENT
**Why:** ~80% NaN in training rows. XGBoost learns `NaN = 2021–2024 regime`, `non-NaN = 2025`. This is inadvertent time encoding, not sentiment signal. Destroys out-of-sample validity.  
**What:**
- Remove `nis_direction_score`, `nis_materiality_score`, `nis_already_priced_in`, `nis_sizing_mult`, `nis_downside_risk` from swing training feature set
- Remove same from intraday training feature set
- Remove `macro_avg_direction`, `macro_pct_bearish`, `macro_pct_bullish`, `macro_avg_materiality`, `macro_pct_high_risk` from swing training (same problem — 259 days only)
- **Keep NIS as a PM gate/overlay** (see Phase 3c below)  
**Files:** `app/ml/training.py` (swing feature list), `app/ml/intraday_trainer.py` (intraday feature list)  
**Deliverable:** Cleaner model without time-leak. Swing drops from ~89 → ~79 features, intraday from 61 → 56 features.

### 1d. Bootstrap Walk-Forward to Quantify Selection Bias ⬜ MEDIUM
**Why:** We've tried ~15 model variants. Picking the best Sharpe inflates the result. Need to know if the original +1.181 / +1.830 were in the tail of a noise distribution.  
**What:**
- Resample fold split dates 200× for v142 and v29 (perturb fold boundaries by ±30 days)
- Run walk-forward on each resample
- Plot distribution of resulting Sharpes
- If original result is in top 10% of distribution → confirmed selection bias  
**Deliverable:** Distribution of Sharpes per model. If median < 0.5 for swing or < 1.0 for intraday, we know the reported numbers were lucky.

### 1e. Implement Deflated Sharpe Ratio Reporting ⬜ LOW (but important)
**Why:** Raw Sharpe inflated by number of trials. Deflated Sharpe (DSR) corrects for this.  
**What:**
- Implement DSR formula: `DSR = Φ((SR - SR*) × √(T-1) / √(1 - γ₃×SR + (γ₄-1)/4 × SR²))`
- Where SR* is the expected max Sharpe from N independent trials
- Add DSR to walk-forward output alongside raw Sharpe
- New gate: DSR > 0 with p < 0.05 (in addition to existing raw Sharpe gate)  
**Files:** `scripts/walkforward_tier3.py`, `app/backtesting/walk_forward.py`  
**Reference:** Bailey & López de Prado (2014) — "The Deflated Sharpe Ratio"

---

## PHASE 2 — Simulation Realism (This Week: Wed–Fri)

*Make the walk-forward simulate what actually trades.*

### 2a. Wire PM Opportunity Score Into Walk-Forward ⬜ HIGH PRIORITY
**Why:** The opportunity score suppresses trades during high-VIX / SPY-below-trend regimes. This is the primary defense against the Apr–Oct 2025 collapse. But the walk-forward doesn't use it — meaning we're promoting models on simulated performance that includes days the live system would have sat out.  
**What:**
- In walk-forward fold simulation, load SPY/VIX daily bars for each test day
- Compute opportunity score (same formula as PM): `score = 0.35×vix_score + 0.20×vix_trend + 0.30×ma_score + 0.15×mom_score`
- If score < 0.35: skip all entries that day
- If score 0.35–0.65: cap candidates at 2
- Run v142 and v29 with score ON and score OFF — measure the delta  
**Files:** `scripts/walkforward_tier3.py`, `app/agents/portfolio_manager.py` (opportunity score logic)  
**Deliverable:** Know if the opportunity score actually rescues the bad fold. If Apr–Oct 2025 Sharpe improves significantly with score ON → it's a real defense worth keeping. If not → regime is deeper than the score catches.

### 2b. Wire Earnings Blackout Into Walk-Forward ⬜ MEDIUM
**Why:** Earnings blackouts block real trades. Simulated trades through earnings events inflate results.  
**What:** Load Finnhub earnings calendar historically. In fold simulation, skip any symbol within earnings blackout window (intraday: ±1d before / 3d after; swing: 3d before).  
**Files:** `scripts/walkforward_tier3.py`

### 2c. Cross-Sectional Dispersion Gate for Intraday ⬜ HIGH PRIORITY
**Why:** The intraday ranking model has no valid premise when all stocks move together (macro-dominated days). Cross-sectional dispersion is the key missing input. On low-dispersion days, "top 20%" winners are random noise.  
**What:**
- For each simulated intraday day, compute `std(all_symbol_2h_returns)` from the historical data
- If dispersion < 0.5×median_dispersion (low-dispersion day): skip all intraday entries
- Also add dispersion as a Branch B (global) feature in intraday model (see Phase 3b)  
**Deliverable:** Intraday model only trades when stock selection has premise. This alone may fix most of the Apr–Oct 2025 collapse — that period was high-correlation/macro-dominated.

### 2d. Live-vs-Sim P&L Reconciliation Harness ⬜ HIGH PRIORITY
**Why:** The single most informative diagnostic in the system. Every paper trade has a simulated analog in the walk-forward. The gap between simulated and actual P&L reveals execution issues, stop drift, and model decay.  
**What:**
- For every closed paper trade: log model predicted probability, simulated analog P&L (from walk-forward), actual realized P&L, exit reason
- Daily report: avg simulated P&L vs avg actual P&L, divergence alert if > 1.5σ  
**Files:** `app/api/routes.py` (new analytics endpoint), `app/database/models.py` (new `sim_pnl` column on Trade)  
**Deliverable:** Know if paper trading is performing as the simulation predicts. If large gap → we have an execution/reconciliation problem, not a model problem.

---

## PHASE 3 — Model Architecture Repair (Next Week)

*Fix the structural design flaws before adding anything new.*

### 3a. Fix cs_normalize: Branch A/B Feature Split for Intraday ⬜ HIGH
**Why:** Cross-sectional normalization zeros out any feature constant across symbols on a given day (VIX, SPY level, breadth, dispersion). The model is completely blind to absolute market state. Even though XGBoost sees `spy_session_return`, it gets zeroed after normalization because all symbols see the same SPY return.  
**What:**
- Split intraday features into:
  - **Branch A** (cross-sectional, z-scored): all stock-specific features (current behavior)
  - **Branch B** (global, NOT normalized): `vix_level`, `vix_regime_bucket`, `spy_above_ma20`, `spy_5d_return`, `market_breadth_pct`, `cs_dispersion`, `day_of_week`, `is_expiry_week`
- Both branches concatenated into single feature matrix XGBoost trains on
- Branch B features passed as-is (or simple percentile rank, not cross-sectional z-score)  
**Files:** `app/ml/intraday_trainer.py`  
**Deliverable:** Model can learn "top 20% momentum + VIX > 25 → avoid". Currently impossible.

### 3b. Remove Swing RSI_DIP / EMA_CROSSOVER Pre-Filters ⬜ HIGH (complex, plan carefully)
**Why:** All three LLMs flag this. The ML model only scores candidates that pass rule-based pre-filters. It learns "which RSI dips don't fail" — not "what makes a good swing trade." This caps the alpha ceiling at whatever the rules can pre-select.  
**What (phased approach — don't rush this):**
- **Step 1:** Score the full 430-symbol universe daily (instead of only RSI_DIP/EMA_CROSSOVER candidates). Keep pre-filters as signal features, not hard gates.
- **Step 2:** Replace binary `path_quality` label with triple-barrier label applied uniformly:
  - Upper barrier: +2.0×ATR from prior close
  - Lower barrier: -1.2×ATR from prior close  
  - Time barrier: 10 calendar days
  - Label: +1 (upper hit first), -1 (lower hit first), 0 (time/neutral)
  - Use multiclass XGBoost (predict +1/-1/0), enter only when P(+1) > threshold
- **Step 3:** PM takes top N by P(+1) from full universe daily, apply min probability threshold
**Note:** This is a 2–3 week change. Do not start until Phase 1+2 are complete.  
**Files:** `app/ml/training.py` (label computation), `app/agents/portfolio_manager.py` (scan logic)

### 3c. NIS as PM Gate Layer (Not Model Feature) ⬜ MEDIUM
**Why:** NIS has insufficient history to be a model feature. But it has real-time value as a trade filter.  
**What:**
- In PM scan (before RM gate), apply NIS overlay:
  - `nis_downside_risk > 0.7` → block new long entry for that symbol
  - `nis_direction_score < -0.3 AND nis_materiality_score > 0.5` → block long
  - `nis_already_priced_in > 0.8` → suppress (don't enter chasing a priced-in move)
  - `nis_direction_score > 0.5 AND nis_materiality_score > 0.5` → allow but note as NIS-supported
- Log NIS overlay decisions to `decision_audit` with gate_category = `alpha`
- Run NIS event study: for each NIS signal bucket, compute forward 1d/3d/5d returns net of SPY. Only after proof of value → consider backfilling.  
**Files:** `app/agents/portfolio_manager.py`

### 3d. Dynamic Position Sizing (Vol-Targeting) ⬜ HIGH
**Why:** Fixed percentage per trade ignores signal confidence, current volatility, and regime. In high-VIX regimes, the same fixed size is 2-3× more risky.  
**What:**
- Position size = `target_vol_contribution / predicted_daily_vol`
- `target_vol_contribution` = 0.5% of account equity per position (configurable)
- `predicted_daily_vol` = ATR-normalized daily vol estimate per symbol
- Apply regime multiplier from opportunity score: 0.5× at caution, 1.0× at normal
- Apply signal confidence multiplier: probability percentile → 0.5×–1.0×
- Cap: single position ≤ 2% of account (existing RM gate, keep)
- Floor: minimum position ≥ $500 (avoid micro-fills)  
**Files:** `app/agents/portfolio_manager.py` (sizing logic), `app/agents/risk_manager.py` (size validation)  
**Deliverable:** Positions automatically shrink in volatile regimes without needing a hard gate.

### 3e. Intraday Label: Add Absolute Hurdle ⬜ MEDIUM
**Why:** Top-20% cross-sectional label forces 20% positive rate even on flat/crashing days where no stock has a clean 2h return. On those days we label noise as winners and train on garbage.  
**What:**
- Add absolute hurdle: label = +1 only if `top 20% cross-sectionally AND 2h_return > 2×estimated_cost`
- Estimated cost for labeling: 15bps (round trip) = ~$0.15 on $100 stock
- Days where no symbol clears the hurdle: exclude from training entirely
- **Run ablation:** train with and without hurdle, compare walk-forward results  
**Files:** `app/ml/intraday_trainer.py` (label computation)

---

## PHASE 4 — Feature Discipline (Next Week)

*Fewer, better-validated features. Stop adding, start pruning.*

### 4a. Feature Correlation Clustering + Pruning ⬜ HIGH
**Why:** The swing model has ~79 features (after NIS removal). Many are redundant (multiple RSI variants, multiple momentum horizons, multiple stochastic oscillators). Redundant features don't add information — they add overfitting risk.  
**What:**
- Compute Spearman correlation matrix across all training rows
- Cluster features with |ρ| > 0.70 — keep only the one with highest permutation importance
- **Likely swing casualties:** `rsi_7` (keep `rsi_14`), `stochrsi_k/d` (keep `stoch_k`), `williams_r_14` (keep `rsi_14`), multiple WQ alphas (keep top 5 by permutation importance)
- **Likely intraday casualties:** redundant momentum/return variants
- Target: 35–40 swing features, 25–30 intraday features  
**Files:** New script `scripts/feature_correlation_audit.py`  
**Deliverable:** Feature reduction report showing clusters + which to keep.

### 4b. Feature Family Ablation ⬜ MEDIUM
**Why:** We've added features intuitively. We don't know which families actually contribute to worst-fold performance (the metric that matters for regime robustness).  
**What:** For each feature family, run walk-forward with that family removed. Record avg Sharpe and worst-fold Sharpe delta. Remove families that don't improve worst-fold.
- Feature families to test: technicals only, + volume, + relative momentum, + volatility, + fundamentals, + earnings, + WQ alphas (top 5), + regime interactions, + sector momentum
- Evaluate on worst-fold Sharpe, not average Sharpe  
**Deliverable:** Evidence-based feature set. Families that don't survive: removed.

### 4c. Residualize Swing Momentum Against Sector ⬜ MEDIUM (add, don't replace)
**Why:** Academic literature consistently shows sector-residualized momentum is stronger than raw momentum. We have sector ETF bars from Phase 89b — use them properly.  
**What:**
- Already have `sector_momentum`, `momentum_20d_sector_neutral` etc.
- Add: `momentum_5d_vs_sector = momentum_5d - sector_momentum_5d` (already being added Phase 89b)
- Add: `vol_vs_sector = atr_norm - sector_atr_norm` (stock-specific volatility)
- Add: `drawdown_vs_sector = stock_drawdown_20d - sector_drawdown_20d`

---

## PHASE 5 — Regime Robustness (Following Week)

*Know when to trade and when to sit out.*

### 5a. Regime Diagnostic Report ⬜ HIGH
**Why:** We know some regimes are bad. We don't know exactly which conditions break the models. This report will tell us.  
**What:** Run v142 and v29 through full simulation (with Phase 1+2 corrections). Segment every simulated trade by:
- VIX bucket: <15, 15–20, 20–25, 25–30, >30
- SPY trend: above/below MA20, MA50, MA200
- Market breadth: % of universe above MA20
- Cross-sectional dispersion: low/mid/high
- Macro event window: yes/no
- Earnings window: yes/no

For each bucket: trades, win rate, avg R, Sharpe, max drawdown, profit factor.  
**Deliverable:** Know exactly which regime each model works in. That becomes the tradable condition and shapes opportunity score tuning.

### 5b. Expand Opportunity Score with Breadth + Dispersion ⬜ HIGH
**Why:** Current score uses VIX, VIX trend, SPY MA, SPY momentum. Missing: market breadth and cross-sectional dispersion — the two most important intraday-specific inputs.  
**What (after Phase 5a diagnostic):**
- Add `breadth_score`: % of S&P 500 symbols above MA20 (clip to 0–1)
- Add `dispersion_score`: daily cross-sectional return std, normalized to 52w percentile
- New formula: `score = 0.25×vix_score + 0.15×vix_trend + 0.25×ma_score + 0.10×momentum_score + 0.15×breadth_score + 0.10×dispersion_score`
- Tune thresholds on training period only. Validate on held-out fold.  
**Files:** `app/agents/portfolio_manager.py`

### 5c. Sample Weighting by Regime in Training ⬜ MEDIUM
**Why:** 85% of 3yr training data is low-volatility bull market. Model optimizes for that regime. Adding higher weight to high-VIX training rows forces the model to learn to be cautious in chaotic conditions.  
**What:**
- In training, compute `sample_weight` per row: `base_weight × (1 + 0.5 × I(VIX > 20)) × (1 + 0.3 × I(VIX > 25))`
- Or: apply exponential time-decay weighting (recent data weighted more, because regime has shifted)
- Run ablation: with vs without sample weighting, report worst-fold improvement  
**Files:** `app/ml/training.py`, `app/ml/intraday_trainer.py`

---

## PHASE 6 — Live Readiness (Parallel with Phase 4/5)

*The plumbing before the first dollar of live capital.*

### 6a. Phase 100: Alpaca as Single Source of Truth ⬜ HIGH
**Why:** DB/Alpaca state divergence has already caused bugs (bars_held reset, ZS stuck open). For live capital, a position mismatch can cause double-entry, missed exits, or incorrect risk accounting.  
**What:**
- Remove position state from DB as authoritative source
- On every PM scan tick: read live positions from Alpaca, not DB
- DB becomes: append-only event log + audit trail + decision_audit
- Any Alpaca/DB mismatch on startup → log warning + enter safe mode (no new entries)
- Mid-session reconciliation every 30 min  
**Files:** `app/agents/trader.py`, `app/agents/portfolio_manager.py`, `app/database/models.py`  
**Precondition:** Market hours safe — do this in off-hours. Not during active trading.

### 6b. Kill Switch + Safe Mode ⬜ HIGH (before any live capital)
**Why:** If live P&L diverges significantly from expected (model decay, execution bug, data issue), we need automatic halting.  
**What:**
- Kill switch endpoint: `POST /api/kill-switch` → halts all new entries, flags active positions for exit
- Auto-trigger: if daily realized P&L < -2% of account → halt
- WebSocket disconnect → safe mode (no new intraday entries, hold existing swings)
- On restart: if Alpaca state unknown → safe mode until reconciliation passes  
**Files:** `app/agents/trader.py`, `app/api/routes.py`

### 6c. Idempotent Order IDs + Duplicate Prevention ⬜ MEDIUM
**Why:** Queue lag + restart can result in duplicate orders if client_order_id is not tracked.  
**What:**
- Generate deterministic `client_order_id` per proposal: `f"mr_{trade_type}_{symbol}_{entry_date}_{uuid4().hex[:8]}"`
- Before submitting any order: check if `client_order_id` already exists in Alpaca
- If exists: skip, log as duplicate  
**Files:** `app/agents/trader.py`

---

## DEFERRED — Do NOT Start Until Phase 1–3 Complete

The following are valid ideas but would be wasted effort before the foundation is fixed:

| Item | Why Deferred |
|---|---|
| NIS stock-level backfill (~$50-100 LLM cost) | First prove NIS has event-study value as PM gate. If it does, backfill. If not, don't spend. |
| New model families (neural nets, TabNet, LSTM) | Need clean validation framework first. Current data is too thin for sequence models. |
| Survivorship-bias-free universe | Real but minor for large-cap S&P/NASDAQ. Not a blocker for paper trading. |
| Tick-level / order book data | High value eventually. Not the bottleneck now. |
| Earnings regression labels (Phase 87a) | Deferred until triple-barrier label (Phase 3b) is validated. |
| Additional WorldQuant alphas | First prune the 13 we have down to top 5. |
| Gate calibration tuning | Need 2+ more weeks of paper trade data first. |
| Regression-based position sizing (Kelly) | Vol-targeting (Phase 3d) first. Kelly requires calibrated probabilities we don't have yet. |
| Bloomberg / institutional data sources | Phase 5+ after we know our edge. |

---

## This Week's Execution Plan (Tue–Fri)

| Day | Tasks |
|---|---|
| **Tuesday** | 1c (Remove NIS from training) + 1a (Cost model design) |
| **Wednesday** | 1a complete (cost model in walk-forward) + 2c (dispersion gate) |
| **Thursday** | 1b (purge/embargo) + 2a (opportunity score in walk-forward) |
| **Friday** | Re-run v142 + v29 walk-forwards with ALL Phase 1+2 corrections. Get honest numbers. |
| **Weekend** | 2d (live-vs-sim harness) + 3a (cs_normalize Branch A/B) design/start |

---

## Updated Walk-Forward Gate Criteria

Old gate (insufficient):
```
avg Sharpe > 0.8 (swing) / > 1.5 (intraday)
min fold Sharpe > -0.3
```

New gate (to implement as Phase 1 completes):
```
avg Sharpe (NET of costs) > 0.6 (swing) / > 1.0 (intraday)  [lower but honest]
worst fold Sharpe > -0.30
max drawdown < 8% (swing) / < 3% (intraday)
profit factor > 1.2
trade count per fold > 100 (sufficient sample size)
deflated Sharpe > 0 with p < 0.05
opportunity score applied during simulation
```

A model that passes these gates is genuinely better than one that passed the old gates.

---

## Key Learnings Preserved From Multi-LLM Review

### Structural Issues (High Confidence — All Three Agree)
- Walk-forward ≠ strategy simulation. Model Sharpe ≠ Strategy Sharpe.
- NIS NaN is a time-proxy, not a sentiment signal. Remove from training immediately.
- No transaction cost model = no meaningful Sharpe numbers for intraday.
- cs_normalize eliminates all market-wide context. Branch A/B split needed.
- RSI_DIP/EMA_CROSSOVER pre-filters cap the alpha ceiling of the swing model.
- Fixed percentage sizing is wrong. Vol-targeting is the right architecture.
- Alpaca must be the source of truth for live capital.

### Regime Problem (Consensus)
- Apr–Oct 2025 collapse is ~60% selection bias/statistical artifact, ~20% PIT leakage, ~15% genuine regime, ~5% data sparsity
- The models don't need to work in every regime. They need to know when edge is absent.
- Cross-sectional dispersion is the missing key signal: low dispersion = macro dominates = ranking has no premise = don't trade intraday.
- Opportunity score is a valid PM-layer defense but needs explicit validation (on vs off comparison).

### Label Design (Consensus)
- Swing `path_quality` (5-day binary) is too narrow for a 3–15 day swing system. Triple-barrier multiclass is better.
- Intraday cross-sectional top-20% needs an absolute hurdle filter (don't label winners on days when no stock has a clean 2h return).
- Realized-R label failed, but the failure may be implementation-specific, not conceptual. Revisit after simulation is fixed.
- Meta-labeling (train model on "given that RSI_DIP fired, should this trade be taken?") is a valid intermediate step while pre-filter removal is in progress.

### Features (Consensus)
- Too many features, not enough ablation evidence. Feature richness ≠ alpha.
- Pruning targets: multiple RSI variants, multiple momentum horizons, WorldQuant alphas (keep top 5), options data with sparse coverage.
- Missing features worth adding (after pruning): idiosyncratic volatility (Ang et al. 2006), PEAD / post-earnings drift, gross profitability (Novy-Marx 2013), dispersion of analyst forecasts, short interest.
- For intraday: RVOL by time-of-day, intraday sector-relative return, dispersion of returns, bid-ask spread proxy.

### What's Actually Good (Preserve)
- PM→RM→Trader architecture: sound. Keep it.
- Decision audit trail: production-grade. Extend it.
- PM re-score at execution time: smart pattern. Keep it.
- Walk-forward gate as hard pass/fail: correct discipline. Improve the metrics, not the concept.
- PIT-correct fundamentals migration: exactly right direction.
- 3-seed ensemble: correct principle. Keep it.
- Versioned model artifacts: keep.
- Subprocess retraining: keep.
- Earnings + macro event blackouts: standard practice, correctly implemented.

---

## Training Flags Reference (Updated)

```bash
# Swing retrain — HOLD until Phase 1+2 complete
# python scripts/retrain_cron.py --swing-only --no-fundamentals --workers 8
# Note: --no-fundamentals flag still needed to avoid crash (ProcessPoolExecutor memory)
# NIS features will be removed from training as of Phase 1c

# Intraday retrain — HOLD until Phase 1+2 complete
# python scripts/retrain_cron.py --intraday-only

# Walk-forward (current, pre-Phase 1 corrections)
python scripts/walkforward_tier3.py --model swing
python scripts/walkforward_tier3.py --model intraday --intraday-model-version 29

# Walk-forward (post-Phase 1 corrections — to be added)
python scripts/walkforward_tier3.py --model swing --cost-model --purge-days 10
python scripts/walkforward_tier3.py --model intraday --intraday-model-version 29 --cost-model --purge-days 2 --with-opportunity-score
```
