# MrTrader — Master Backlog & Roadmap

**Last updated:** 2026-05-20
**Capital:** $100k (paper)
**Status:** Phase F complete (L/S infrastructure built + WF run: avg 0.579, gate FAILED). **Phase G (PEAD) is the primary path** — factor IC = -0.0064 confirms no signal; PEAD avg Sharpe ~2.70 validated.

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
Phase G (NEXT)        PEAD strategy                               ~1 week
Phase H               3-month paper trading gate                  3 months
Phase I               Live $100k                                  after H passes
```

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
