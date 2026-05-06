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

---

## PHASE R — Regime Model (Next Major Track)

**Last updated:** 2026-05-06  
**Status:** Spec complete. Not started. Prerequisite: Phase 1+2 can run in parallel.

### Why This Track Exists

Two failed experiments (Phase 86, Item 3) confirmed: `cs_normalize` zeros any market-wide feature because it is identical across all symbols on a given day. The stock selection models cannot learn macro regime context this way. A separate model is required.

**Root cause proven:** Item 3 (2026-05-06) added `regime_vix_proxy`, `regime_vix_pct60d`, `regime_spy_ma20_dist` to intraday features. Walk-forward result: avg Sharpe -0.331, indistinguishable from v29 baseline (-0.327). Features confirmed zeroed. See `ML_EXPERIMENT_LOG.md` Item 3 entry.

**Correct fix:** A **Regime Model** runs independently of cs_normalize, scores the macro environment once at premarket (and re-scores dynamically after macro events), and feeds its output into the PM pipeline as a gate and sizing scalar.

---

### Design Principles

1. **Separation of concerns.** Regime model answers "should we be trading at all today?" Stock models answer "which stocks, given we are trading." These require different feature types and different normalization.

2. **NIS macro belongs here.** The LLM-scored macro risk (FOMC/CPI/NFP pre-release assessment) is a regime signal, not a stock signal. It feeds the regime model as a feature. The binary hard block stays as an emergency fallback only.

3. **Dynamic, not static.** The regime score is not computed once at 7am and frozen. It re-evaluates after each macro event release (NFP at 8:35, FOMC at 2:05, etc.). The system tracks a schedule of re-evaluation triggers and fires them automatically.

4. **Resilient by design.** If uvicorn is down at 7am, the system catches up on restart. If a re-evaluation job fails, it retries. The `regime_snapshots` table is the single source of truth — any component reads from there, not from in-memory state.

5. **Graduated, not binary.** The output is a continuous score [0.0, 1.0], not a hard block/allow. Position sizing and ML score thresholds are scaled continuously. The hard NIS block remains only for true emergency conditions (FOMC window before release, SPY futures < -2.5%).

6. **Auditable.** Every regime score written to `regime_snapshots` includes the full feature dict, the trigger reason, and the model version. Post-hoc analysis can slice any proposal by the regime score at the time it was made.

---

### Architecture

```
INPUTS (market-wide — no cs_normalize)
┌────────────┐ ┌────────────┐ ┌─────────────────┐ ┌──────────────┐
│ VIX level  │ │ SPY daily  │ │ MacroCalendar   │ │ NIS Tier 1   │
│ + 1y hist  │ │ bars 252d  │ │ days_to_FOMC    │ │ risk_level   │
│ (yfinance) │ │ (Alpaca)   │ │ days_to_CPI/NFP │ │ sizing_factor│
└─────┬──────┘ └─────┬──────┘ └────────┬────────┘ └──────┬───────┘
      └──────────────┴─────────────────┴─────────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │   RegimeFeatureBuilder    │  app/ml/regime_features.py
                   │   build(as_of_date) -> {} │
                   └────────────┬─────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │       RegimeModel         │  app/ml/regime_model.py
                   │  XGBoost + calibration    │
                   │  regime_score [0.0, 1.0]  │
                   │  regime_label (enum)      │
                   └────────────┬─────────────┘
                                │ writes to
                   ┌────────────▼─────────────┐
                   │    regime_snapshots       │  PostgreSQL table
                   │  (audit + persistence)   │
                   └────────────┬─────────────┘
                                │ read by
        ┌───────────────────────┼──────────────────────┐
        │                       │                      │
┌───────▼───────┐   ┌───────────▼────────┐  ┌─────────▼──────────┐
│ PremarketAgent│   │ PortfolioManager   │  │ API /regime/current │
│ 7am baseline  │   │ each scan:         │  │ dashboard widget    │
│ + re-eval     │   │ gate + size scale  │  │                     │
│   schedule    │   │ + ML threshold adj │  │                     │
└───────────────┘   └────────────────────┘  └────────────────────┘
```

---

### Resiliency Model

**Problem:** Uvicorn may be down at 7am. A macro event (NFP 8:30) may fire while the system is restarting. A re-evaluation job may fail silently.

**Solution: `regime_snapshots` as source of truth + startup catchup + APScheduler re-eval jobs**

```
On uvicorn startup:
  1. Check regime_snapshots for today's 'premarket' row
  2. If missing AND time < 11:30 ET → run premarket scoring now (catchup)
  3. If missing AND time >= 11:30 ET → log warning, use NEUTRAL (0.5) as safe default
  4. Rebuild APScheduler re-eval jobs for any macro events today that haven't fired yet
     (check MacroCalendar.events_today, filter to event_time + 5min > now_et)
  5. If a re-eval should have fired while we were down (event released, no snapshot row):
     → run it immediately on startup (backfill for missed window)

APScheduler re-eval jobs (registered at startup and on premarket run):
  - For each MacroCalendar event today with time_str set:
      schedule job at event_time + 5min
      job: RegimeModel.score(trigger='post_<event_type>') → write to regime_snapshots
  - FOMC: always schedule re-eval at 2:05pm ET regardless of calendar (hardcoded)
  - If market breadth data becomes available (Phase R2): add 10:00am ET re-eval

Re-eval job behavior:
  1. Build fresh features (VIX may have moved, SPY may have gapped)
  2. Score regime model
  3. Write new row to regime_snapshots with snapshot_trigger = 'post_nfp' etc.
  4. If score changed by > 0.15 from last score: log WARNING with delta
  5. If score crosses RISK_OFF threshold (< 0.35): log CRITICAL + PM reads new score on next scan
  6. PM always reads LATEST regime_snapshots row for today (not just premarket row)
```

**Failure modes handled:**

| Failure | Response |
|---|---|
| Uvicorn down at 7am | Catchup run on restart if before 11:30 ET |
| Re-eval job throws exception | APScheduler retries 3x with 60s backoff; logs CRITICAL on final failure |
| Regime model file missing | Fall back to `_compute_opportunity_score_legacy()` with logged warning |
| DB write fails | Score cached in-memory; retry on next tick; logged to file |
| VIX data unavailable | Use last known value from `regime_snapshots`; log warning |
| MacroCalendar has no time_str | Skip re-eval schedule for that event; log warning |
| PM reads stale score (>4h old during market hours) | Log warning; apply 20% haircut to regime_score as uncertainty penalty |

---

### Database Schema

#### New table: `regime_snapshots`

```sql
CREATE TABLE regime_snapshots (
    id                  SERIAL PRIMARY KEY,
    snapshot_time       TIMESTAMP NOT NULL,          -- UTC
    snapshot_date       DATE NOT NULL,               -- ET trading date
    snapshot_trigger    VARCHAR(30) NOT NULL,        -- 'premarket' | 'post_nfp' | 'post_fomc' |
                                                     --   'post_cpi' | 'startup_catchup' |
                                                     --   'manual' | '10am_breadth'

    -- Model output
    regime_score        FLOAT NOT NULL,              -- [0.0, 1.0]
    regime_label        VARCHAR(15) NOT NULL,        -- 'RISK_OFF' | 'NEUTRAL' | 'RISK_ON'

    -- Score thresholds used (stored so threshold changes are auditable)
    risk_off_threshold  FLOAT NOT NULL DEFAULT 0.35,
    risk_on_threshold   FLOAT NOT NULL DEFAULT 0.65,

    -- Raw input features (named columns for SQL slicing)
    vix_level           FLOAT,
    vix_pct_1y          FLOAT,
    vix_pct_60d         FLOAT,
    spy_1d_return       FLOAT,
    spy_5d_return       FLOAT,
    spy_20d_return      FLOAT,
    spy_ma20_dist       FLOAT,
    spy_ma50_dist       FLOAT,
    spy_ma200_dist      FLOAT,
    spy_rvol_5d         FLOAT,
    spy_rvol_20d        FLOAT,
    days_to_fomc        INTEGER,
    days_to_cpi         INTEGER,
    days_to_nfp         INTEGER,
    is_fomc_day         BOOLEAN NOT NULL DEFAULT FALSE,
    is_cpi_day          BOOLEAN NOT NULL DEFAULT FALSE,
    is_nfp_day          BOOLEAN NOT NULL DEFAULT FALSE,
    nis_risk_level      VARCHAR(10),                 -- 'LOW' | 'MEDIUM' | 'HIGH'
    nis_sizing_factor   FLOAT,
    breadth_pct_ma50    FLOAT,                       -- NULL until Phase R2 breadth data available

    -- Model metadata
    model_version       VARCHAR(30),                 -- 'regime_v1' | NULL for backfill rows
    feature_json        JSONB,                       -- full feature dict

    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_regime_snapshot UNIQUE (snapshot_date, snapshot_trigger)
);

CREATE INDEX ix_regime_snapshots_date ON regime_snapshots (snapshot_date DESC);
CREATE INDEX ix_regime_snapshots_score ON regime_snapshots (regime_score);
```

#### New table: `regime_model_versions`

```sql
CREATE TABLE regime_model_versions (
    id              SERIAL PRIMARY KEY,
    version         VARCHAR(30) NOT NULL UNIQUE,  -- 'regime_v1'
    trained_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    training_start  DATE NOT NULL,
    training_end    DATE NOT NULL,
    label_scheme    VARCHAR(50) NOT NULL,          -- 'rule_based_v1' | 'trade_outcome_v1'
    n_training_days INTEGER,
    n_features      INTEGER,
    feature_list    JSONB,
    val_auc_fold1   FLOAT,
    val_auc_fold2   FLOAT,
    val_auc_fold3   FLOAT,
    val_auc_avg     FLOAT,
    val_brier_score FLOAT,                        -- calibration quality
    gate_passed     BOOLEAN NOT NULL DEFAULT FALSE,
    status          VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',  -- 'ACTIVE' | 'ARCHIVED'
    model_path      VARCHAR(255),
    notes           TEXT
);
```

#### Changes to existing tables

```sql
-- proposal_log (primary unified table)
ALTER TABLE proposal_log ADD COLUMN regime_score_at_scan  FLOAT;
ALTER TABLE proposal_log ADD COLUMN regime_label_at_scan  VARCHAR(15);
ALTER TABLE proposal_log ADD COLUMN regime_trigger_at_scan VARCHAR(30);  -- which snapshot row was used

-- decision_audit (agent decisions)
ALTER TABLE decision_audit ADD COLUMN regime_score_at_decision FLOAT;

-- daily_state (per-day system state)
ALTER TABLE daily_state ADD COLUMN regime_score_premarket FLOAT;
ALTER TABLE daily_state ADD COLUMN regime_label_premarket VARCHAR(15);
ALTER TABLE daily_state ADD COLUMN regime_last_updated_at TIMESTAMP;
```

#### Join patterns

```sql
-- "How did we trade in different regimes?"
SELECT
    r.regime_label,
    COUNT(p.id) as proposals,
    AVG(p.outcome_1d_pct) as avg_1d_outcome,
    AVG(t.pnl) as avg_pnl
FROM proposal_log p
JOIN regime_snapshots r
    ON r.snapshot_date = DATE(p.scan_time AT TIME ZONE 'America/New_York')
    AND r.snapshot_trigger = 'premarket'  -- use morning score
LEFT JOIN trades t ON p.trade_id = t.id
WHERE p.pm_status = 'SENT'
GROUP BY r.regime_label;

-- "What was regime score when this trade was entered?"
SELECT t.symbol, t.entry_price, t.pnl, p.regime_score_at_scan, p.regime_label_at_scan
FROM trades t
JOIN proposal_log p ON t.proposal_id = p.proposal_uuid
ORDER BY t.created_at DESC;

-- "Regime history this month"
SELECT snapshot_date, snapshot_trigger, regime_score, regime_label, vix_level, spy_5d_return
FROM regime_snapshots
WHERE snapshot_date >= CURRENT_DATE - 30
ORDER BY snapshot_date, snapshot_time;
```

---

### Backfill Plan

| Data | Source | Go-back | Notes |
|---|---|---|---|
| VIX level + history | yfinance `^VIX` | 2023-01-01 | Already used in `features.py` |
| SPY daily bars | Alpaca historical | 2023-01-01 | Already fetched; extend window |
| MacroCalendar events | `app/calendars/macro.py` | 2023-01-01 | Hardcoded — fully available |
| NIS macro risk level | `NisMacroSnapshot` table | 2025-05-01 | NULL before this → default 0.5 |
| Market breadth | Compute from intraday_cache | 2024-01-01 | 720 symbols × daily close |

Script: `scripts/backfill_regime_snapshots.py`
- Iterates trading days 2023-01-01 → today
- `model_version = NULL` (no model yet — just raw features)
- `regime_score = NULL`, `regime_label = NULL`
- Provides feature data for model training and future re-scoring
- Runtime: ~5 min for 3yr window (SPY bars cached locally)

---

### Feature Set

```python
REGIME_FEATURE_NAMES = [
    # VIX / realized vol
    "vix_level",           # spot VIX [5, 80]
    "vix_pct_1y",          # percentile vs 252-day window [0,1]
    "vix_pct_60d",         # percentile vs 60-day window [0,1] — faster signal
    "spy_rvol_5d",         # SPY 5d realized vol annualized (%)
    "spy_rvol_20d",        # SPY 20d realized vol annualized (%)

    # SPY price / trend
    "spy_1d_return",       # yesterday's daily return
    "spy_5d_return",       # 5-day rolling return
    "spy_20d_return",      # 20-day rolling return
    "spy_ma20_dist",       # (spy_close - MA20) / MA20
    "spy_ma50_dist",       # (spy_close - MA50) / MA50
    "spy_ma200_dist",      # (spy_close - MA200) / MA200 — secular context

    # Macro calendar
    "days_to_fomc",        # clipped to [0, 30]
    "days_to_cpi",         # clipped to [0, 30]
    "days_to_nfp",         # clipped to [0, 30]
    "is_fomc_day",         # 1.0 if FOMC today, else 0.0
    "is_cpi_day",          # 1.0 if CPI today, else 0.0
    "is_nfp_day",          # 1.0 if NFP today, else 0.0

    # NIS Tier 1 macro (NULL before May 2025 → use XGBoost missing-value handling)
    "nis_risk_numeric",    # HIGH=1.0, MEDIUM=0.5, LOW=0.0
    "nis_sizing_factor",   # [0.5, 1.0] recommended by NIS

    # Market breadth — Phase R2 (NULL until computed; 0.5 default)
    "breadth_pct_ma50",    # % of 720-symbol universe above MA50
]
```

**What is intentionally excluded from V1:**
- Sector ETF rotation (add in V2)
- Put/call ratio (requires CBOE data; add in V2)
- VIX term structure (requires VX futures; add in V3)
- Individual stock breadth (overlap with stock model; keep separate)

---

### Label Scheme

**V1: Rule-based (implement immediately, no training data dependency)**

```python
def label_regime_day(vix_level, spy_1d_return, spy_ma20_dist) -> int:
    """
    1 = FAVORABLE trading environment
    0 = HOSTILE — sit out
    """
    return int(
        spy_1d_return > 0.0      # market went up yesterday
        and vix_level < 20.0     # vol is calm
        and spy_ma20_dist > 0.0  # SPY above medium-term trend
    )
```

Expected positive rate: ~52% (slightly above half of trading days in a bull market). Not a class imbalance problem.

**V2: Trade-outcome labels (activate after 90 days of paper data)**

```python
# For each trading day d:
# positive_days = trades WHERE trade_type='intraday'
#                 AND DATE(closed_at AT TIME ZONE 'ET') = d
#                 AND avg(pnl_pct) > 0
# label = 1 if the day was net-profitable, 0 otherwise
```

Switch happens atomically: retrain `regime_v2` on outcome labels, validate AUC >= 0.60, promote.

---

### Dynamic Re-Evaluation Schedule

The regime model does not run once at 7am. It maintains a dynamic schedule of re-evaluation triggers built from today's MacroCalendar events:

```
7:00am ET    → baseline premarket score    (trigger='premarket')
               [schedules re-eval jobs for all today's events]

8:35am ET    → if NFP today: post-event re-score  (trigger='post_nfp')
               [NFP releases at 8:30; 5-min buffer for data propagation]

2:05pm ET    → if FOMC today: post-announcement   (trigger='post_fomc')
               [FOMC announces at 2:00pm ET]

8:35am ET    → if CPI today: post-CPI score        (trigger='post_cpi')

[startup catchup logic — see Resiliency Model above]
```

**Implementation in `app/agents/premarket.py`:**

```python
def _schedule_regime_reeval_jobs(self, scheduler: AsyncIOScheduler):
    """Register APScheduler one-shot jobs for today's macro event re-evaluations."""
    from app.calendars.macro import MacroCalendar
    cal = MacroCalendar()
    ctx = cal.get_context()
    now_et = datetime.now(ET)
    today_str = now_et.strftime("%Y-%m-%d")

    for evt in ctx.events_today:
        if evt.date_str != today_str:
            continue
        h, m = map(int, evt.time_str.split(":"))
        release_dt = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
        reeval_dt = release_dt + timedelta(minutes=5)

        if reeval_dt <= now_et:
            # Already past — check if we have a snapshot for it; if not, run now
            trigger_name = f"post_{evt.event_type.lower()}"
            existing = self._get_regime_snapshot(today_str, trigger_name)
            if existing is None:
                logger.info("Startup catchup: running missed regime re-eval for %s", trigger_name)
                asyncio.create_task(self._run_regime_reeval(trigger_name))
        else:
            # Future — schedule it
            trigger_name = f"post_{evt.event_type.lower()}"
            scheduler.add_job(
                self._run_regime_reeval,
                trigger="date",
                run_date=reeval_dt,
                args=[trigger_name],
                id=f"regime_reeval_{trigger_name}_{today_str}",
                replace_existing=True,
            )
            logger.info("Scheduled regime re-eval: %s at %s ET", trigger_name, reeval_dt.strftime("%H:%M"))
```

---

### Integration With Existing PM Gates

**During Phase R3-R4 (parallel running — no decision changes):**

```python
# In portfolio_manager.py scan path:
regime_ctx = self._get_latest_regime_snapshot()  # reads DB
scan_log["regime_score"] = regime_ctx["regime_score"] if regime_ctx else None
scan_log["regime_label"] = regime_ctx["regime_label"] if regime_ctx else None
# Write to proposal_log rows but do NOT gate on it yet
```

**During Phase R5 (regime model is primary gate):**

```python
regime_score = regime_ctx["regime_score"] if regime_ctx else 0.5

# Gate: no new entries in RISK_OFF
if regime_score < 0.35:
    self._log_abstention("regime_risk_off", regime_score)
    return

# Size scaling
size_multiplier = (
    0.5  if regime_score < 0.50 else
    0.75 if regime_score < 0.65 else
    1.0
)

# ML threshold: demand higher confidence in uncertain regimes
base_threshold = self._intraday_min_conf  # e.g. 0.55
effective_threshold = base_threshold * (1.0 + 0.4 * max(0, 0.65 - regime_score))
```

**Hard blocks that remain regardless of regime model:**
- `is_intraday_blocked()` NIS hard block (FOMC window before announcement)
- SPY futures < -2.5% at open
- Kill switch active
- RM max-positions gate

---

### Phased Implementation Plan

#### Phase R1 — Data Infrastructure (3 days)

**Goal:** Feature pipeline computable for any historical date.

**Deliverables:**
- `app/ml/regime_features.py` — `RegimeFeatureBuilder.build(as_of_date) -> dict`
  - SPY daily bars: extend current fetch window from 20d → 252d in `portfolio_manager.py`
  - VIX: use existing yfinance fetch from `features.py` L567, add 252d history
  - MacroCalendar: `days_to_fomc/cpi/nfp` computed from sorted event list
  - NIS: query `NisMacroSnapshot` table for latest entry on or before `as_of_date`
- Alembic migration: create `regime_snapshots`, `regime_model_versions` tables
- Alembic migration: add columns to `proposal_log`, `decision_audit`, `daily_state`
- `scripts/backfill_regime_snapshots.py` — populates raw features 2023-01-01 → today (model_version=NULL)

**Tests:**
- Unit test `RegimeFeatureBuilder.build(date(2025, 4, 7))` — VIX was ~23, SPY below MA20, NFP that week → manual verify
- Backfill script completes, 750+ rows in `regime_snapshots`, no NULL on named feature columns (except NIS pre-May-2025 which is expected)

**Gate:** 750+ backfilled rows, all named columns non-NULL except NIS pre-May-2025.

**Do not touch:** Any PM scan logic, any ML training code, any existing gates.

---

#### Phase R2 — Train Regime V1 (2 days)

**Goal:** Working regime model with validated walk-forward AUC.

**Deliverables:**
- `app/ml/regime_training.py` — `RegimeModelTrainer`:
  - `generate_labels(df) -> pd.Series` — V1 rule-based labels
  - `train(start, end) -> model` — XGBoost + `CalibratedClassifierCV(method='isotonic')`
  - Walk-forward: 3 expanding folds (2023-2024 / 2023-2025 / 2023-2026)
  - Saves to `app/ml/models/regime_model_v{N}.pkl`
  - Writes to `regime_model_versions` table
- `scripts/train_regime_model.py` — CLI wrapper
- `app/ml/regime_model.py` — `RegimeModel` singleton:
  - Loads model from disk at startup
  - `score(as_of_date=None, trigger='manual') -> dict` — builds features, predicts, writes to `regime_snapshots`
  - 5-minute in-memory cache (avoids re-scoring on every PM scan tick)
  - Falls back to `_compute_opportunity_score_legacy()` if model not loaded

**Tests:**
- Walk-forward AUC >= 0.60 all 3 folds
- Calibration Brier score < 0.22
- SHAP: `vix_level` and `spy_ma20_dist` must be top-3 features
- Score FOMC day (2026-05-07) → verify regime_score < 0.40

**Gate:** AUC >= 0.60 worst fold AND Brier < 0.22.

---

#### Phase R3 — Premarket Integration + Resiliency (3 days)

**Goal:** Model runs at 7am, re-evals after macro events, persists to DB, uvicorn-restart safe.

**Deliverables:**
- Modify `app/agents/premarket.py`:
  - `_run_regime_scoring(trigger: str)` — scores model, writes `regime_snapshots`
  - `_schedule_regime_reeval_jobs(scheduler)` — registers APScheduler one-shot jobs
  - `_startup_regime_catchup()` — called on init; checks for missed premarket + missed event re-evals
  - `get_regime_context() -> dict` — reads latest `regime_snapshots` row for today
- Modify `app/agents/portfolio_manager.py`:
  - Read `regime_ctx` from `premarket_intel.get_regime_context()` in each scan
  - Write `regime_score_at_scan`, `regime_label_at_scan`, `regime_trigger_at_scan` to all `proposal_log` rows
  - Write to `daily_state.regime_score_premarket` at premarket time
  - **No gate changes yet** — parallel running only
- API endpoint `GET /api/regime/current` — reads latest `regime_snapshots` for today
- API endpoint `GET /api/regime/history?days=30` — time series for dashboard

**Resiliency tests:**
- Simulate uvicorn killed at 6:55am, restarted at 9:15am → verify startup catchup runs and writes `regime_snapshots` row with `trigger='startup_catchup'`
- Simulate NFP day with uvicorn restarted at 8:45am → verify post_nfp re-eval runs immediately on startup
- Simulate DB write failure → verify in-memory cache holds score, retries on next tick
- Simulate VIX fetch failure → verify last-known VIX used from `regime_snapshots` previous row

**Gate:** 3+ consecutive trading days with regime scores in `regime_snapshots`. All `proposal_log` rows have `regime_score_at_scan` populated. No incidents in logs.

---

#### Phase R4 — Parallel Running + Evidence Collection (10+ trading days)

**Goal:** Build confidence before changing any decisions.

**Deliverables:**
- Dashboard widget (small panel on Overview page): regime score gauge, label, last updated time
- Analytics query: for each `regime_label_at_scan`, compute avg `outcome_1d_pct` from `proposal_log`
- Weekly summary in logs: "RISK_OFF days: N, avg SPY return: X%, intraday proposals sent: Y"
- Track divergence: days where regime model would have blocked but opportunity score allowed (and vice versa)

**Acceptance criteria:**
- >= 10 trading days of parallel data
- At least 1 RISK_OFF day observed and logged
- Agreement rate with current opportunity_score >= 70%
- On FOMC day (2026-05-07): verify regime_score drops below 0.40 before 2pm, rises after 2:05pm re-eval

**Gate:** 10+ trading days, FOMC test passed, no regressions in paper trading.

---

#### Phase R5 — Gate Integration (after R4 validated)

**Goal:** Regime model becomes the primary gate and size scaler.

**Deliverables:**
- Modify `portfolio_manager.py` scan gate to use `regime_score` as primary
- Rename `_compute_opportunity_score()` → `_compute_opportunity_score_legacy()` (kept as fallback)
- Implement `regime_size_multiplier(score)` in `app/strategy/position_sizer.py`
- Implement `effective_min_conf` threshold scaling
- Remove the 3 zeroed features from `intraday_features.py` (`regime_vix_proxy/pct60d/spy_ma20_dist`) atomically with next model retrain
- Update `ScanAbstentions` table writes to include `regime_score` as the abstention reason

**Gate:** 20+ trading days with regime model as primary gate. No false RISK_OFF blocks on confirmed good days. No regressions vs baseline paper P&L.

---

#### Phase R6 — Regime-Aware Stock Model Retraining (after 90 days paper data)

**Goal:** `regime_score` enters the stock model as a Branch B non-normalized feature.

**Deliverables:**
- Add `regime_score` as a pass-through feature in `compute_intraday_features()` — bypasses cs_normalize
- Retrain intraday model with `regime_score` as an explicit input (XGBoost learns interaction terms automatically)
- Walk-forward validation with Phase 1+2 corrections applied
- Gate: avg Sharpe (net of costs) > 1.0, worst fold > -0.30

**Prerequisite:** Phase 1+2 honest walk-forward must be complete before claiming this helped.

---

### What NOT to Change During R1-R4

| Component | Reason |
|---|---|
| `cs_normalize.py` | Regime model solves this at a higher level; cs_normalize is still correct for stock features |
| Stock ML models (v142, v29) | Paper trading; regime model gates when they run, not how they're trained |
| `FEATURE_NAMES` in `intraday_features.py` | Remove `regime_vix_proxy/pct60d/spy_ma20_dist` atomically with R5 retrain only |
| `_compute_opportunity_score()` in PM | Keep as fallback until R5 validated |
| `is_intraday_blocked()` hard gates | Keep all emergency circuit breakers; regime model supplements |
| `NisMacroSnapshot` / `MacroSignalCache` tables | No structural changes; NIS becomes a feature input only |
| Frontend proposal tables | Add regime widget as additive panel only; don't restructure existing pages |
| RM gates (max positions, correlation, concentration) | Unchanged throughout |

---

### Open Questions (Decide Before R2)

1. **Breadth data source:** Compute from 720-symbol `intraday_cache` (free, already have it) vs FMP API `/market-breadth` endpoint. Recommendation: intraday_cache — avoids new API dependency.

2. **Re-eval frequency cap:** If 3 macro events fire in one day (unlikely but possible), cap re-evals at 3 per day to avoid thrashing. Add `max_reevals_per_day = 3` config in `app/config.py`.

3. **Regime model retraining cadence:** Monthly automated retrain triggered by `scripts/retrain_regime_model.py` in `retrain_cron.py`. Gate: AUC >= 0.60 on last 90d held-out. Auto-promote if gate passed; alert if not.

4. **What if regime model is confidently WRONG on a day that turned out good?** Track this. After 90 days, compute: on RISK_OFF days, what % of time would we have made money if we'd traded anyway? If > 40%, the model is hurting us and needs retraining or threshold adjustment.

---

*This spec is the authoritative design document for Phase R. Update it as decisions are made. Do not start R2 until R1 gate is passed.*
