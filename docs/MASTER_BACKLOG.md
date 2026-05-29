# MrTrader — Master Backlog & Roadmap

**Last updated:** 2026-05-28
**Capital:** $100k (paper)
**Status:** LX1–LX8 complete. Root cause identified by Opus 4.7: **beta-in-the-book** — composite loads on high-beta/momentum names that get killed in vol spikes. All timing-based fixes (gates, stops, regime) ruled out. **Active: LX9-B1 (10d rebalance cadence, launched ~22:30).** Next: LX9-A (beta-neutralize feature ranking). Paper-trade gate: avg Sharpe ≥ +0.30. Opus recommends paper-trading LX1 baseline in parallel while researching.

---

## STRATEGIC DIRECTION — 2026-05-18 PIVOT

After 4 independent LLM reviews (DeepSeek, Gemini, ChatGPT, Claude/Opus 4.7) of the entire system, the active strategy is shifting from **long-only cross-sectional factor portfolio** to **directional Long/Short**.

**Why the pivot:**
- The Phase C LambdaRank campaign (9 runs) collapsed structurally in bear-market folds because cross-sectional top/bottom quintile labels in a 25% drawdown teach the model "fell least = winner" — defensive signals that contradict the momentum-quality features in non-bear regimes.
- Long-only cross-sectional ranking cannot survive bear regimes by design.
- L/S is regime-independent: shorts produce P&L when the market falls, longs when it rises. The 0.80 Sharpe gate becomes achievable.

**New strategy: directional Long/Short**
- Net exposure target: **+40% net long** (configurable via `pm.ls_net_exposure_pct`)
- Gross exposure: ~150% (e.g. 95% long + 55% short)
- Top-N: 20 longs / 15 shorts (configurable)
- WF gate stays at **avg Sharpe ≥ 0.80, min fold ≥ -0.30** (now reachable)
- Capital: $100k paper

Full synthesis lives in `docs/QUANT_REVIEW_SYNTHESIS_2026_05_18.md`.

---

## Key Architectural Decisions (2026-05-18)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Directional Long/Short, 40% net long** | Regime-independent; configurable via `pm.ls_net_exposure_pct` |
| 2 | **Factor portfolio → paper after PIT audit + L/S WF ≥ 0.80** | Already integrated (PR #224). Continuous improvement after promotion. |
| 3 | **Intraday on backburner** | Paper continues running, zero new dev. Revisit after swing validated. PDT rule change June 2026 may help. |
| 4 | **PEAD as second strategy** | FMP already has EPS surprise data (PIT-safe via `filingDate`, $0 extra cost). |
| 5 | **Equal-weight strategy allocation** | 50/50 factor + PEAD when both active; simpler than risk-parity |
| 6 | **Survivorship audit required** | `scripts/audit_survivorship.py` written; must run before any new WF |
| 7 | **Live promotion criteria** | 3 months paper, Sharpe ≥ 0.50 annualized, max DD ≤ 15%, max single position ≤ 8% NAV |
| 8 | **WF gate stays 0.80 avg Sharpe** | Achievable for L/S; was structurally unreachable for long-only ranking |

**Convergence across all 4 reviewers (treat as facts):**
- IC must be computed for every model — has never been done; most critical gap
- Long-only cross-sectional ranking is broken in bear regimes (label inversion)
- Factor decomposition required: regress factor returns on SPY + AQR MOM + QMJ; require alpha t-stat > 2
- Sector rotation as floor benchmark (11 SPDR ETFs, top-3 by 6mo momentum)

---

## Active Phase Plan

```
Phase E (DONE)        IC + survivorship scripts                  2026-05-18
Phase F (DONE)        L/S infrastructure                          2026-05-20
Phase 0-align (DONE)  10 WF simulation bugs fixed, PR #288       2026-05-27
Phase 1 (DONE)        Model rank-IC diagnostic + honest WF       2026-05-27
Phase LS0 (DONE)      L/S experiment — enable-shorts on v221     2026-05-27  FAIL (-1.31 min fold)
Phase LS1 (DONE)      swing_short_v1 design spec (Opus 4.7)      2026-05-27  DEFERRED (see below)

── ACTIVE MISSION: Find honest long-side edge (3-week timebox) ──────────────

Phase LX1 (DONE)      Experiment: equal-weight 5 IC features + B2 overlay    2026-05-27  avg Sharpe +0.557
Phase LX2 (DONE)      Experiment: v186 honest clean re-run                   2026-05-27  avg Sharpe +0.171 FAIL — XGBoost 82 features worse than equal-weight
Phase LX3 (DONE)      Experiment: Retrain XGBoost on 5 IC-validated features only        2026-05-27  avg Sharpe -2.344 FAIL — XGBoost 5 features ≪ equal-weight; ML weighting ruled out
Phase LX4 (DONE)      Experiment: Concentrated LX1 (target_n=15) + factor-stability gate  2026-05-27  avg Sharpe -0.251 FAIL — but win rate 64.7% on 1,332 trades confirms real signal edge; problem is DD (17-25%), not direction
Phase LX5 (DONE)      Experiment: Inverse-vol position sizing on LX1 (target_n=30, 20d vol lookback, 0.5x-2x cap)  2026-05-28  avg Sharpe +0.032 FAIL — inv-vol helps F1/F3 (+0.24/+0.67) but fold-2 killer remains; fold-spec mismatch vs LX1 invalidates direct comparison
Phase LX1-rb (DONE)   Re-baseline LX1 on identical folds (--as-of 2026-05-28)    2026-05-28  avg Sharpe +0.079 FAIL — original +0.557 was fold-period artifact; honest equal-weight baseline is +0.079
Phase LX6a (DONE)     Entry-only regime gate (VIX≥30→30% on new entries)              2026-05-28  avg -0.127 FAIL — WORSE than baseline; blocks recovery entries; ruled out
Phase LX6b (DONE)     Hard-exit regime gate (VIX≥30→liquidate all longs at rebalance)  2026-05-28  avg -0.103 FAIL — F2 worsened (-0.72→-0.88); exits at bottom, misses bounce; ruled out. PIVOT TRIGGERED.
Phase LX7 (DONE)        L/S: long top-20 + short bottom-20 by 5-feature composite, +40% net long  2026-05-28  avg +0.036 FAIL — short-book thesis wrong; bottom-20 composite = value/post-crash names that rally fastest; L/S ruled out
Phase LX8 (DONE)        7% per-position trailing stop on LX1 (bug-fixed as LX8b)                 2026-05-28  avg -0.207 FAIL — stop cuts winners (PF 0.957); bug found: stop fired on swing-model positions too (fixed in PR #305). Root cause confirmed: timing interventions don't fix beta exposure problem.
Phase LX9-B1 (IN PROGRESS) 10-day rebalance cycle on LX1 (test if cadence addresses F2 shock timing)  launched 2026-05-28 ~22:30  gate: F2≥-0.40 AND F1/F3 positive → run LX9-B2/A
Phase LX9-A (PENDING)   Beta-neutralize feature ranking (residualize vs trailing 1Y beta-to-SPY)    highest P(success)~45% per Opus; requires new code
Phase LX-gate         Long side honest WF Sharpe > 0.8 → UNLOCK short model

── SHORT SIDE (deferred until LX-gate passes) ───────────────────────────────

Phase SD0             Data infra: VIX3M + FINRA short interest (safe, parallel)  ~4h
Phase SD1             Rules-based short overlay (3-day, no ML)               ~3 days
Phase SD2             swing_short_v1 ML model (full 10-day build)            after LX-gate
Phase SD3             30-day paper trade gate for shorts                      3 months

── DOWNSTREAM (unchanged) ───────────────────────────────────────────────────

Phase G               PEAD strategy                               after LX-gate
Phase H               3-month paper trading gate                  after SD2
Phase I               Live $100k                                  after H passes
```

### LX-gate definition
Long side passes when ALL of:
- Honest WF avg Sharpe ≥ **0.80** (same gate as before)
- No fold < **-0.30**
- DSR p ≥ **0.95**
- Beats naive B2 (SPY>200d MA = 0.808) by ≥ **0.10 Sharpe**

### Short model deferral rationale (Opus 4.7, 2026-05-27)
Building `swing_short_v1` on a long book with DSR z=-10.4 means every future bug is unattributable (long? short? interaction?). The same research process that produced the +0.374 false-positive will embed false positives in short-model IC selection. Fix the foundation first. Full design spec is preserved in `ML_EXPERIMENT_LOG.md` and ready for implementation the moment LX-gate clears.

### Phase 1 Key Findings (2026-05-23)

v216 rank-IC@20d = **0.0012** overall (noise), BUT by-year reveals a **regime problem**:

| Year | IC@20d | t-stat | Regime |
|------|--------|--------|--------|
| 2021 | **+0.023** | **4.92** ✅ | Bull, low vol |
| 2022 | **-0.028** | **-8.73** ❌ | Bear, rate shock |
| 2023 | -0.009 | -2.11 ❌ | Mixed |
| 2024 | +0.010 | +1.98 🟡 | Bull recovery |
| 2025 | +0.017 | +4.02 🟡 | Bull |

**Interpretation:** Features work in trending/low-vol regimes. The 2022 rate-shock year **inverted** the model signal (high-momentum stocks fell hardest), destroying 5-year aggregate IC. This is a **label design problem** (LambdaRank trained cross-sectionally without regime conditioning), not a dead feature set.

**Next step (Phase 2):** Compute regime-conditional IC (filter to BENIGN regime days only). If IC > 0.02 in BENIGN regime → switch to policy-realized binary labels + regime filter in training.

---

## STRATEGIC PIVOT — 2026-05-24: Rebalancing Execution Architecture

**Context:** After Phase 4 v3 WF (all bugs fixed, Sharpe=-0.036, 7 trades/fold), Opus 4.7 analysis identified the root problem: the execution layer (RSI/EMA signal triggers) is architecturally mismatched with LambdaRank, which is a cross-sectional portfolio selection model. L3 Bridge Test confirmed alpha exists at Sharpe=0.577 when the model is used correctly (rank → pick top-N → rebalance). Fix the execution layer, not the model.

**Design decisions (Opus 4.7, 2026-05-24):**

| Decision | Detail |
|----------|--------|
| Execution mode | `REBALANCE` (new PM mode alongside existing `SIGNAL`) |
| Rebalance cadence | 20 trading days (matches 20d label horizon) |
| Target positions | N=30 at $20k, N=50 at $100k+ |
| Long/short | Long-only first; shadow-track 3 short candidate sets |
| Short deployment | $50k+ account, separate short model (not LambdaRank bottom-N) |
| Position sizing | Cascade: regime gate → inverse-vol base → NIS modulation |
| Regime exposure | Bull=100%, Neutral=70%, Bear=30% invested |
| Exit logic | 20d hold baseline + profit harvest (+12% in ≤7d) + NIS exit (<-0.4) + regime-flip forced rebalance; NO price stops |
| Score weighting | Deferred — LambdaRank scores are ordinal, not calibrated expected returns |

**Why no price stops:** 8% stop on a 20d-horizon model fires on noise (typical R1000 20d vol = 12-18%). Replace with NIS-based exits (information-driven, not price-noise-driven).

**Why no LambdaRank bottom-N as short book:** L2 L/S (0.397) < L3 long-only (0.577) proves the bottom ranking is not a good short signal. Bottom-ranked stocks are "mediocre" not "likely to fall" — and mediocre has positive drift (equity risk premium) working against shorts.

---

## Phase RA — Rebalancing Baseline ← ACTIVE NEXT

**Goal:** Replace signal-triggered execution with ranking-based rebalance. Reproduce L3 alpha (0.577) inside the WF framework. Pass criteria: WF avg Sharpe ≥ 0.50, ≥30 trades/fold.

**What to build:**
1. `app/strategy/portfolio_construction.py` — liquidity filter, sector cap (30%), hysteresis (add at rank ≤15, drop at rank ≥30), equal-weight sizing
2. PM `REBALANCE` mode — on rebalance date: score all symbols → apply constraints → compute target set → emit close/open orders
3. Config flags: `EXECUTION_MODE`, `REBALANCE_DAYS=20`, `TARGET_POSITIONS=30`, `SECTOR_CAP=0.30`
4. Attribution logging: per-trade decomposition into selection/exit/cost components
5. Regime gate (Layer 1): Bull=100%, Neutral=70%, Bear=30% gross exposure

**WF validation:** Run `walkforward_tier3.py` with rebalance mode. Expect ~11 rebalance events × ~6 rotations × 2 sides = ~130 fills/fold. Sharpe computed on daily equity curve, not per-trade.

---

## Phase RB — Sizing Overlay (after RA passes WF)

1. Inverse-volatility base weights (60d realized vol, cap 0.5×–2× equal-weight)
2. NIS modulation: NIS > +0.4 → ×1.25 size; NIS < -0.4 → exit position
3. Renormalize to regime gross target

---

## Phase RC — Exit Overlays (after RB validated)

1. Profit harvest: +12% in ≤7 days → rotate to next unheld ranked name; cap rotations at 25% of book per window
2. NIS-driven exits (already in RB; formalize as explicit exit event)
3. Regime-flip forced rebalance: bull→bear → immediate rebalance + de-risk to 30%

---

## Phase RD — Shadow Short Infrastructure (parallel to RA-RC)

Track three short candidate sets (shadow P&L only, no live trades):
1. Bottom-30 of LambdaRank (naive null)
2. Bottom-30 filtered by NIS < -0.4
3. High-short-interest + negative momentum names

Store with `is_shadow=True`, `shadow_strategy` enum. Daily P&L persistence. After 6-12 months data → decide short deployment.

---

## Phase RE — Short Deployment (deferred, $50k+)

Separate short model with different features: accruals (Sloan), leverage deterioration, dilution, high short-interest + rising days-to-cover. Deploy after Phase RD data justifies. Operationally viable at $50k+.

---

## Phase D — Factor Portfolio Integration ✅ COMPLETE (2026-05-18) — superseded by L/S pivot

- PRs #224 / #225 merged: `app/ml/factor_scorer.py` + PM routing + SPY>MA200 + VIX<30 gate
- Validated Sharpe 1.335 from `scripts/factor_portfolio_backtest.py` (monthly rebalance, equal-weight)
- AgentSimulator WF mismatch documented (ATR stops vs monthly rebalance — wrong tool, not signal failure)
- **Decision:** Factor portfolio stays as long sleeve of L/S; will be re-validated under Phase F WF after L/S infrastructure is built.

---

## Phase E — P0 Scripts ✅ COMPLETE (2026-05-18)

P0 bug audit (per LLM review synthesis):

| Item | Status |
|---|---|
| **P0.1 Entry price** | ✅ ALREADY CORRECT — `agent_simulator.py:804` uses `today_bar["open"]` |
| **P0.2 Intrabar stop** | ✅ ALREADY CORRECT — `agent_simulator.py:935-944` checks `today_low <= stop_price` / `today_high >= target_price` |
| **P0.3 Factor IC computation** | ✅ Script written: `scripts/compute_factor_ic.py` (Spearman IC, monthly rebalance dates, fwd 10d returns; pass ≥ 0.02, t-stat ≥ 2.0). Not yet run. |
| **P0.4 Survivorship audit** | ✅ Script written: `scripts/audit_survivorship.py` (checks delisted ticker coverage in daily cache). Not yet run. |

**Deliverables:**
- `scripts/compute_factor_ic.py`
- `scripts/audit_survivorship.py`
- `docs/QUANT_REVIEW_SYNTHESIS_2026_05_18.md` (synthesis + P0–P4 backlog)

---

## Phase F — Long/Short Infrastructure ✅ COMPLETE (2026-05-20)

**WF Result:** avg Sharpe 0.579, GATE FAILED (Fold 4 = -0.98 during April 2025 tariff shock). Factor IC = -0.0064 → no signal. L/S infrastructure stays as foundation for PEAD; factor portfolio deprioritized as alpha source.

**Goal:** Convert the factor portfolio + (future) PEAD into a directional L/S engine.

### F.1 — `FactorPortfolioScorer` returns shorts
- `app/ml/factor_scorer.py`: extend to return `[(symbol, confidence, direction)]` where `direction ∈ {LONG, SHORT}`
- Bottom-N composite score → SHORT candidates (with inverted score sign)
- Existing top-N → LONG candidates

### F.2 — PM proposals carry SELL_SHORT direction
- `portfolio_manager.py`: emit `Proposal.direction = SHORT` for short candidates
- New proposal type: `SELL_SHORT` (entry) and `BUY_TO_COVER` (exit)
- Routing path: factor → both long and short sleeves to RM

### F.3 — `AgentSimulator` supports short P&L
- P&L sign inverted for shorts: `pnl = (entry - exit) × qty`
- Borrow cost accrual: **0.5%/yr** (configurable) deducted daily
- Inverted stop/target: stop above entry, target below
- Update `agent_simulator.py:804/935-944` for short branch

### F.4 — `risk_rules.py` net-exposure + short-heat gates
- Net exposure gate: target ± 15% tolerance around `pm.ls_net_exposure_pct`
- Short heat: cap total short notional (e.g. 75% of NAV)
- Hard locate-availability check (Alpaca easy-to-borrow list)

### F.5 — Config keys in `agent_config.py`
```
pm.ls_net_exposure_pct       = 0.40
pm.ls_top_n_long             = 20
pm.ls_top_n_short            = 15
pm.ls_borrow_cost_annual_pct = 0.005
pm.ls_net_exposure_tolerance = 0.15
```

### F.6 — Walk-forward re-run with L/S
- 5-fold, 6-year, R1K universe with PIT membership
- **Gate:** avg Sharpe ≥ 0.80, min fold ≥ -0.30
- Block PEAD work until F.6 passes

---

## Phase G — PEAD Strategy

**Goal:** Second orthogonal strategy. Post-Earnings Announcement Drift.

### G.1 — `PEADScorer`
- New module: `app/ml/pead_scorer.py`
- Uses `fmp_provider.get_earnings_features_at(symbol, asof)` (PIT-safe, `filingDate` based)
- Features: standardized EPS surprise, revenue surprise, guidance revision direction, post-announcement gap
- Score = signed surprise z-score × confidence

### G.2 — Multi-strategy routing in PM
- PEAD has priority for symbols within ≤ 5 trading days post-earnings
- Factor fills remainder of allocation
- **Capital allocation: 50/50 equal-weight by strategy when both active**

### G.3 — `scripts/run_pead_walkforward.py`
- 5-fold, 6-year WF on R1K
- Gate: avg Sharpe ≥ 0.80, min fold ≥ -0.30 (same as factor)
- IC check: ≥ 0.02 on forward 5d returns to surprise score

---

## STRATEGIC DECISION — 2026-05-22: Stay Single-Name L/S, Fix Alignment

**Kill criterion triggered:** Model WF avg=-0.275 (folds: -0.842, -0.469, -0.624, +1.074, -0.513). 4/5 negative.

**Decision:** Do NOT pivot to ETFs. The kill criterion result is not evidence that single-name L/S has no alpha — it is evidence that the training/WF pipeline was misaligned (see root causes below). Fix the alignment first, then re-evaluate.

**Root causes identified (Opus 4.7 review, 2026-05-22):**
1. WF has always evaluated `FactorPortfolioScorer` (rules-based), never `model.predict`. Spearman=0.035 between the two — completely independent systems.
2. Training objective (triple-barrier labels) ≠ WF objective (AgentSimulator ATR-stop P&L). Different things were optimized vs evaluated.
3. Universe survivorship bias not audited.
4. Feature construction parity between training/WF/live not verified.
5. LambdaRank single-row predict = 0.0 (fixed 2026-05-22).
6. TSNormalizerState empty for LambdaRank causing all-zero inference (fixed 2026-05-22).

---

## Phase 0 — Freeze, Audit, Instrument (1-2 days) ← NEXT

**Goal:** Stop changing behavior until misalignments are measurable.

### 0.1 — Canonical contract (`app/ml/contracts.py`)
- `@dataclass FeatureRow`: symbol, asof_ts, feature_name→value, label, label_meta
- `@dataclass ScoreRow`: symbol, asof_ts, raw_score, normalized_score, rank
- `schema_hash()` — hash of feature names + dtypes + order
- Training, WF, live PM all import this. Mismatch = startup fail.

### 0.2 — Parity test harness (`tests/parity/test_training_wf_live_parity.py`)
- 5 fixed (symbol, date) pairs spanning 2019–2025
- Assert: training pipeline features == WF features == live PM features (within 1e-9)
- Assert: schema_hash identical across all three paths
- **Start by expecting failure — every diff surfaces a real bug.**

### 0.3 — Structured logging in all three paths
- After feature construction: log schema_hash, row count, NaN count
- After normalization: log normalizer ID, feature mean/std
- After model.predict: log score distribution, top-10 symbols
- **Files:** `app/ml/training.py`, `scripts/run_model_walkforward.py`, `app/agents/portfolio_manager.py`

---

## Phase 1 — Catalog All Misalignments (2 days)

**Goal:** Evidence-backed list of everything broken. No fixes yet.

### 1.1 — Feature construction audit → `docs/feature_audit.md`
For every feature: source, lookback, training code path, inference code path, PIT compliance.
- Rolling z-scores: cross-section in training vs cs_normalize on N symbols in live?
- Fundamentals: lagged by report date or filing date?
- Sector/industry: static-as-of-today (survivorship) or PIT?

### 1.2 — Label audit → `docs/label_audit.md`
Compute correlation: triple-barrier label vs AgentSimulator P&L for same trades.
**Expected:** correlation < 0.3 → training objective is provably wrong → Phase 2 mandatory.

### 1.3 — Universe/survivorship audit → `docs/universe_audit.md`
Is "Russell 1000" today's membership backfilled, or PIT? Count delisted symbols in training.
**Script:** `scripts/audit_survivorship.py` (already written)

### 1.4-1.6 — Normalization, inference path, execution audits
- cs_normalize on 5 symbols ≠ cs_normalize on 750 — live scoring is wrong
- AgentSimulator vs live: stops, slippage, sizing, rebalance cadence
- Live PM inference paths: selection (full universe) vs reeval (single symbol)

---

## Phase 2 — Replace Training Objective (3-5 days)

**Goal:** Train the model to predict what WF and live actually reward.

### 2.1 — Canonical label: forward N-day residual return vs sector
- `app/ml/labels.py` (new): `compute_residual_return_label(prices, sectors, horizon=5)`
- N matches live holding period (~5-10 days for swing)
- Continuous label (not binary triple-barrier)

### 2.2 — LambdaRank group = (date, universe), gain = return decile rank
- `app/ml/training.py`: group per date, continuous residual label
- **Pass criterion:** Training IC (rank corr predicted vs realized 5d residual) > 0.02

### 2.3 — Retrain swing_v215
- Log in `docs/ML_EXPERIMENT_LOG.md` with IC, fold Sharpes, gate result

---

## Phase 3 — Lockstep WF (2-3 days)

**Goal:** WF evaluates exactly the same code as training, on PIT-correct universe.

### 3.1 — Single feature builder (`app/ml/features.py`)
One function: `build_feature_matrix(symbols, asof_date)` → DataFrame with schema_hash.
Training, WF, live PM all call this. Delete duplicate code.

### 3.2 — PIT universe in WF
Top-1000 by trailing 60d ADV at each rebalance date. No symbol before IPO or after delisting.
**File:** `scripts/run_model_walkforward.py`

### 3.3 — WF calls model.predict on FULL universe each day
cs_normalize on K symbols ≠ cs_normalize on 750. WF must score entire eligible universe, take top-K.
**File:** `app/backtesting/agent_simulator.py` — inject `signal_fn(date, universe) -> Series[score]`

### 3.4 — Parity test green
Phase 0.2 test passes for all 5 (symbol, date) pairs across training/WF/live.

---

## Phase 4 — Honest WF Diagnosis (2 days)

Run aligned WF with v215. Required outputs regardless of pass/fail:
- **Per-fold IC:** rank corr predicted score vs realized 5d residual
- **Decile P&L:** top-decile minus bottom-decile per fold
- **Hit rate by sector and VIX bucket**
- **Attribution:** selection vs sizing vs stops

**Decision rules:**
- IC > 0.02 but P&L < gate → execution/sizing issue → Phase 5
- IC ≈ 0 → signal issue → Phase 7
- IC > 0.02 + decile P&L > SPY but full WF < SPY → stops destroying alpha → Phase 5

---

## Phase 5 — Execution Alignment (2-3 days, if IC > 0 but P&L < gate)

- Strip simulator to pure mode: equal-weight top-K long/bottom-K short, hold N days, no stops
- Sweep: ATR multiplier {1.5, 2.5, none}, holding period {3, 5, 10, 20}, K {5, 10, 20, 50}
- Find config where pure mode beats SPY; layer stops back only if they improve it
- Align live PM to winning config

---

## Phase 6 — Pre-Production Validation

- CPCV on v215 with lockstep pipeline
- 6-month strict holdout (most recent, not used in training)
- 1-week live shadow mode: log proposals without executing, verify match to WF
- **Pass:** holdout Sharpe within 1σ of CPCV median

---

## Phase 7 — Signal Engineering (if IC ≈ 0 in Phase 4)

Feature additions in priority order:
1. Earnings surprise / drift (PIT EPS vs estimate)
2. Short interest / days-to-cover (weekly, PIT)
3. Analyst revision breadth (PIT)
4. Cross-sectional residual momentum (12-1, sector-neutralized)
5. Quality: accruals, ROIC trend (PIT fundamentals)
6. Microstructure: realized vol-of-vol, Amihud illiquidity

Each addition: must improve fold IC ≥ 0.005 to stay.

---

## Sequencing Rationale

| Phase | Blocks | Why |
|-------|--------|-----|
| 0 | All | Without contracts + parity tests, every fix is unverifiable |
| 1 | 2,3 | Must know all misalignments before choosing fix order |
| 2 | 3,4 | Model must predict the right thing before WF can validate it |
| 3 | 4 | WF must call same code as training before result is meaningful |
| 4 | 5,6,7 | Diagnosis determines which branch to take |
| 5/7 | 6 | Don't validate until execution OR signal is fixed |
| 6 | Live | Don't risk capital until CPCV + holdout + shadow pass |

---

## Phase H — 3-Month Paper Trading Gate

**Live promotion requires all of:**
- 3 calendar months of continuous paper trading
- Annualized Sharpe ≥ 0.50
- Max drawdown ≤ 15%
- Max single position ≤ 8% NAV (RM-enforced)
- Live-vs-sim shortfall < 30 bps/day median
- Zero unreconciled Alpaca/DB position mismatches

---

## Phase I — Live $100k

Only after Phase H passes. Start small (e.g. $25k of $100k); scale up after 4 weeks of clean live data.

---

## Open Technical Debt (parallel, non-blocking)

| Item | Owner | Notes |
|---|---|---|
| `russell1000_membership.parquet` — 198/750 tracked | Data | Rebuild from iShares IWB monthly holdings |
| Daily price cache gap (~71 R1K symbols missing) | Data | `scripts/backfill_yfinance.py` + Polygon fallback |
| R1K fundamentals backfill (~320 R1K-only symbols) | Data | FMP — overnight job |
| StrategyContract abstraction (ChatGPT P2 suggestion) | Eng | Not blocking — refactor after L/S validated |
| Factor decomposition vs SPY + AQR MOM + QMJ | Quant | Required before live; alpha t-stat > 2 gate |
| Sector rotation floor benchmark | Quant | 11 SPDRs, top-3 by 6mo momentum, monthly rebalance |

---

## Phase Plan G-Pre → K: Foundation Hardening (2026-05-22, Opus 4.7 Reviewed)

### Background & Diagnosis

After Factor IC = -0.0064, PEAD avg 0.346, and Factor Portfolio best 0.701 (all gate FAILED), a
4-LLM synthesis + Opus 4.7 architectural review identified the root causes in descending probability:

| # | Root Cause | Est. Probability |
|---|---|---|
| 1 | **Label-evaluation mismatch**: training on triple-barrier 5d, evaluating on ATR-stop simulator Sharpe — these are different objectives | 70% |
| 2 | **Score artifact mismatch**: WF measures FactorPortfolioScorer callable, not model.predict; Spearman correlation unknown | 50% |
| 3 | **Feature pipeline asymmetry**: train and WF features computed via different code paths; never diffed | 30% |
| 4 | **Genuine no-edge**: momentum/quality/value doesn't work on R1K post-2020 at 5bps costs | 30% |
| 5 | **Survivorship/PIT leakage** | 10% |

> Note: these are not mutually exclusive — likely 2-3 are simultaneously true.

**Kill criterion (define now, before sunk-cost bias):** If after G-Pre + Phase G, SPDR rotation
beats our system by > 0.2 Sharpe → pivot sector rotation as primary strategy and deprioritize
single-name picker development.

---

### Phase G-Pre.0 — Fold Reproduction by Hand (½ day, FIRST)

Pick Fold 3 (well-behaved). Trace one stock, one trade:
- Compute features via training pipeline code path
- Compute same features via WF code path
- Feed both to scorer and model.predict
- Trace AgentSimulator P&L for that trade (entry price, stop, exit price, P&L)

If P&L cannot be reproduced within 1bp → **stop everything and fix the reproduction gap before proceeding.**

---

### Phase G-Pre.1 — Score Reconciliation (2 hours, BLOCKING)

For 20 random (date, universe) tuples sampled from WF fold windows:
```python
scorer_ranks = FactorPortfolioScorer(**SCORER_CONFIG)(day, symbols_data, vix_history)
model_ranks  = model.predict(wf_features_for_day)
spearman = scipy.stats.spearmanr(scorer_ranks, model_ranks).statistic
top10_overlap = len(set(top10_scorer) & set(top10_model))
```
**Pass criterion:** Spearman ≥ 0.85 AND top-10 overlap ≥ 7/10 on ≥ 90% of dates.

**If fails:** all prior WF numbers are measuring the scorer, not the model. Re-run all experiments
using `model.predict` rank to understand what the model actually does.

---

### Phase G-Pre.2 — Label Construction Decision (THE critical fork)

Current: triple-barrier labels (5d, +2σ/-1σ) → evaluated by AgentSimulator ATR-stop Sharpe.
These are different objective functions. The model optimizes for one thing; WF grades another.

**Option A (cleanest, ~4 hrs compute):** Replace triple-barrier with simulator-derived labels.
For each (symbol, date), forward-simulate with same ATR stops used in WF. Label = realized P&L.
This makes training and evaluation measure the same thing.

**Option B (1 day):** Replace with forward return matching typical simulator holding period.
Measure mean holding period from recent WF run logs. If ~7d, use r_{t+7} as continuous label
and train LambdaRank on those. Simpler but doesn't capture stop-out effects.

**Decision rule:** If IC of triple-barrier model on 10d forward returns stays at ≈ 0 after
G-Pre.1 reconciliation → must switch to Option A or B. Do not proceed to Phase H with broken labels.

---

### Phase G-Pre.3 — Feature Equality + PIT + Universe (parallelizable, 1 day)

**Feature equality:**
- Pickle train feature DataFrame from retrain artifacts at one fold start date
- Recompute same features via WF code path at same (symbol, date) index
- `pd.testing.assert_frame_equal(train_df, wf_df, rtol=1e-6)` on intersection
- Known failure modes: z-score window size, cross-sectional rank denominator, NaN handling

**PIT spot-check (30 min):**
```python
for (symbol, date) in random_sample_10:
    fundamentals = wf_pipeline.get(symbol, date)
    filing_date = FMP.get_filing_date(symbol, fundamentals['quarter'])
    assert filing_date <= date - timedelta(days=1)  # T-1 minimum
```
Any failure = full fundamentals stack is leaked.

**Also:** shuffle fundamentals dates by ±90 days for one fold. If Sharpe changes by < 0.05,
fundamentals were not contributing signal (consistent with IC = -0.0064).

**Universe consistency:**
```python
symmetric_diff = train_universe ^ pit_union_universe
assert len(symmetric_diff) / len(train_universe) < 0.05  # < 5%
```

**G-Pre.4:** Run `scripts/audit_survivorship.py` and `scripts/compute_factor_ic.py`.

---

### Phase G — Benchmarks (run in parallel with G-Pre, 1–2 days)

Two naive baselines. **These must be run before building anything more complex.**

**G.1 — SPDR Sector Rotation (50 LOC):**
- 11 SPDR ETFs: XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC
- Rank by 6-month total return with 1-month skip (r_{t-7m to t-1m})
- Top-3, equal-weight, monthly rebalance, 5bps costs
- 5-fold, 6-year WF, same gate (avg Sharpe ≥ 0.80)
- Also compute: equal-weight 11-SPDR portfolio as second floor
- **If top-3 rotation passes gate → it becomes primary strategy; pivot immediately**

**G.2 — Pure 12-1 Momentum Picker (30 LOC):**
- Rank R1K by 12-month return minus last 1 month
- Top-N longs, bottom-N shorts (test both long-only and L/S)
- Also test: 6-1 momentum, 3-1 momentum as variants
- 5-fold, 6-year WF
- **Purpose:** if pure 12-1 outperforms factor composite → composite weighting destroys the signal

**G.3 — Fold 4 calibration check (10 min):**
- Run equal-weight SPY (or SPY itself) through Fold 4 only
- If SPY Sharpe in Fold 4 ≈ -0.5 or worse → gate floor of -0.30 on Fold 4 is impossible without
  explicit short exposure; gate needs adjustment

---

### Phase H — Regime Allocator (1 week, only after G-Pre resolves labels)

**Do not build this until G-Pre.2 label decision is made. An allocator on a broken picker
modulates noise, not alpha.**

**Inputs (max 4 features — hard limit):**
- % of R1K universe above 200DMA (breadth) ↑ → exposure ↑
- Cross-sectional return dispersion (std dev of 20d returns) ↑ → exposure ↓
- VIX/VXV ratio (term structure, not level) ↑ → exposure ↓
- Credit spread proxy: HYG/IEF total return ratio ↑ → exposure ↑
- **Do NOT include NIS** (circular — NIS is our own constructed signal)
- SPY 50d/200d MA state can substitute breadth if breadth unavailable

**Model: logistic regression with L2, NOT a decision tree.**
Reasoning: ~15 years × 252 days = 3,800 obs, but effective ~40 regime episodes (autocorrelated).
A decision tree with 3 splits memorizes 8 leaves on 40 episodes.

**Better alternative:** continuous exposure scalar with hand-set monotonic coefficients:
```python
exposure = sigmoid(-2 * z_dispersion - 1.5 * z_vixts + 0.5 * z_breadth - 1.0 * z_credit_spread)
```
Tune ~3 coefficients only. Enforce sign constraints — reject any fit that violates direction.

**Train/freeze split:** 2007-2017 train, validate 2018-2020 (includes COVID), freeze 2021-2026.
(Not 2007-2019 freeze: COVID + 2022 rate shock are the most informative regime episodes.)

**Wiring:** Weekly rebalance of target gross exposure. Cap day-over-day change at 5pp.
Daily application via position sizing multiplier into PM.

**Overfitting safeguards:**
- Coefficient sign constraints enforced
- Max 4 features — period
- Sensitivity: drop one year, refit, compare paths — if 2020 exposure changes > 20% when dropping
  2008, model is unstable
- Smoothness penalty on day-over-day exposure changes

---

### Phase I — Retry with Allocator + Label Fix

After G-Pre.2 fixes labels and Phase H builds allocator:
- Re-run Factor Portfolio WF with new labels + allocator
- Re-run L/S WF with new labels + allocator
- Gate: avg Sharpe ≥ 0.80, min fold ≥ -0.30

---

### Phase J — PEAD Retry with Allocator

After Phase I passes (or independently, same allocator):
- PEAD had avg 0.346 — failed because Fold 4 drowned event signal
- With allocator reducing gross exposure in Fold 4, signal may survive

---

### Required WF Output Additions (apply to all future runs)

Every WF run must output these (currently missing):

1. **P&L attribution per fold:** `total = alpha - costs - stop_losses + carry`
2. **Transaction cost sensitivity:** run 5bps, 10bps, 15bps variants. If Sharpe collapses at 15bps → edge is fictitious.
3. **Bootstrap CIs on fold Sharpe:** with 5 folds × 250 days, Sharpe stderr ≈ ±0.4. Report 95% CI.
4. **Null-strategy distribution:** 100 random-pick portfolios (random N stocks, equal-weight) through same WF. Factor model must beat 80th percentile.
5. **Fold 4 SPY benchmark:** always report SPY Sharpe in Fold 4 alongside strategy Sharpe.

---

### Kill Criterion (Pre-registered 2026-05-22 BEFORE running model WF)

> If, after model-based WF with per-fold artifacts and verified feature parity:
> - Mean WF Sharpe < 0.40 AND
> - Mean WF Sharpe < (EW-all-11-SPDR mean Sharpe − 0.20) AND
> - At least 2 of 5 folds are negative
>
> Then: **STOP single-name picker development**. Pivot to ETF-level strategy
> (EW-all-11 base + drawdown overlay + regime-conditional cash tilt).
> Archive swing_v* model family with post-mortem in ML_EXPERIMENT_LOG.
>
> If 0.40 <= Sharpe < 0.80: continue with label/feature work, cap at 4 more weeks
> before second go/no-go.
>
> If Sharpe >= 0.80 with passing diagnostics: paper trading prep.

---

### Phase K — Paper Trading Gate (3 months)

Same criteria as before. Note: 3 months has stderr ~0.5 on Sharpe — treat as smoke test, not
statistical validation. Be explicit about this in reporting.

---

## Historical Phases (Archive)

Earlier phases (A diagnostics, B build, C LambdaRank campaign, R regime model, WF-A alignment, Phases 1–5 statistical truth + simulation realism, Phase 6 live readiness, data tasks D0–D5) are preserved in:

- `docs/phases_archive.md` — full completed phase history
- `docs/ML_EXPERIMENT_LOG.md` — every retrain + WF result
- `docs/ML_EXPERIMENT_LOG_archive.md` — older campaigns

**Key historical context:**
- Phase A diagnostics (2026-05-13): naive baseline +0.808 beat best ML +0.106 → confirmed ML was destroying alpha vs trivial SPY timing
- Phase C LambdaRank (9 runs, closed 2026-05-18): structural Fold 2 (bear) collapse — campaign closed
- Phase D factor portfolio (2026-05-18): integrated and paper-running; will be the long sleeve of Phase F L/S
- WF-A1/2/3 (PRs #198/199/200): TS-norm parity, PIT universe, survivorship — all merged
- Vol targeting (3d), live-vs-sim harness (2d), opportunity score (5b) — all merged, feature-flagged where appropriate

---

## Walk-Forward Gate (canonical)

```
avg Sharpe (NET of 5–15 bps costs) >= 0.80
min fold Sharpe                    >= -0.30
max drawdown                       <= 15%
profit factor                      >= 1.2
trade count per fold               >= 100
deflated Sharpe                    >  0 (p < 0.05)
IC vs forward 10d returns          >= 0.02 (t-stat >= 2.0)
opportunity / regime gate applied during simulation
PIT universe (`pit_union` with delisted seed)
```

---

## Training / WF Command Reference

```bash
# Phase E scripts (ready to run)
python scripts/compute_factor_ic.py --years 6 --horizon 10
python scripts/audit_survivorship.py --universe russell1000

# Phase F (after F.1–F.5 implemented)
python scripts/walkforward_tier3.py --model swing --strategy ls_factor \
    --net-long 0.40 --top-long 20 --top-short 15

# Phase G (after F passes)
python scripts/run_pead_walkforward.py --years 6 --folds 5
```
