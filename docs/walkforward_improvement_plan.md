# Walk-Forward Improvement Plan

**Status:** WF-1 ✅ COMPLETE | WF-2 ✅ COMPLETE | WF-3 ✅ COMPLETE  
**Last updated:** 2026-05-07  
**Owner:** Engineering  
**Scope:** Swing + Intraday today; extensible to Day Trading and future strategies

---

## Motivation

After Phase 1 corrections (real transaction costs, purge gaps, NIS feature removal),
both champion models fail the gate:

| Model | Honest Sharpe | DSR p-value | Gate |
|---|---|---|---|
| Swing v163 | +0.358 | < 0.001 | ❌ |
| Intraday v51 | +0.529 | < 0.001 | ❌ |

The walk-forward itself has structural gaps that prevent honest model evaluation:
only 3 expanding folds, one-sided purge, single-metric gate, no regime stratification,
and no combinatorial path coverage. These gaps must be closed before model results
can be trusted as signals rather than statistical noise.

---

## Current Architecture (Baseline)

**File:** `scripts/walkforward_tier3.py` (877 lines, monolithic)

| Aspect | Current State | Gap |
|---|---|---|
| Folds | 3 expanding folds | Low statistical power; not independent |
| Purge | Pre-test only (10d swing, 2d intraday) | Post-test embargo missing |
| Gate | Sharpe + DSR | Missing: profit factor, Calmar, K-ratio |
| Regime awareness | None in fold construction | Folds may be regime-homogeneous |
| Strategy coupling | Swing/intraday hardcoded in one file | Cannot add day trading without major surgery |
| Bootstrap | ±30d date jitter | Not true combinatorial resampling |
| Metrics | Sharpe, win rate, max DD, stop rate | Missing: profit factor, Calmar, K-ratio |

---

## Phase WF-1 — Embargo + Multi-Metric Gate

**Status:** ✅ COMPLETE (2026-05-07)  
**Branch:** `feat/wf1-embargo-metrics`

### What changed

**Embargo:** Added `embargo_days` parameter (default: `purge_days`). Both sides of
every test window now have a clean gap:

```
train | purge_days | TEST | embargo_days | next_train
```

Previously only the pre-test gap existed. The post-test embargo prevents the last
rows of each test window from appearing as the first rows of the next training set,
which violates López de Prado's walk-forward protocol.

**New metrics in FoldResult:**

| Metric | Formula | Gate threshold |
|---|---|---|
| `profit_factor` | sum(wins) / sum(|losses|) | > 1.10 (at report level) |
| `calmar_ratio` | annualised_return / max_drawdown | > 0.30 (at report level) |
| `k_ratio` | slope(cum_returns) / std(annual_returns) | > 0 (directional check) |

**Updated gate (WalkForwardReport.gate_passed):**

```python
avg_sharpe >= SHARPE_GATE (0.8)
AND min_sharpe >= MIN_FOLD_SHARPE (-0.3)
AND dsr_p > 0.95
AND avg_profit_factor >= MIN_PROFIT_FACTOR (1.10)
AND avg_calmar >= MIN_CALMAR (0.30)
```

### Why these thresholds

- **Profit factor > 1.10:** A model with Sharpe 0.8 but profit factor 0.95 is losing
  on gross before costs — it's a statistical artifact. 1.10 is deliberately low to
  avoid false negatives at this stage.
- **Calmar > 0.30:** Prevents models that achieve Sharpe via concentration / lucky
  timing with large drawdowns. 0.30 = model needs $1 return per $3.33 max DD per year.
- **K-ratio > 0:** Ensures equity curve is trending upward with consistent slope,
  not lumpy (Sharpe can be high with lumpy returns).

---

## Phase WF-2 — Pluggable Engine Architecture

**Status:** ✅ COMPLETE (2026-05-07)  
**Branch:** `feat/wf2-pluggable-engine`

### What changed

Refactored monolithic `walkforward_tier3.py` into a package:

```
scripts/walkforward/
  __init__.py          ← public API: run_swing_walkforward, run_intraday_walkforward
  engine.py            ← FoldEngine: fold construction, purge, embargo (strategy-agnostic)
  gates.py             ← FoldResult, WalkForwardReport, all gate logic
  cost_models.py       ← FixedBpsCostModel, SpreadCostModel (future), ImpactCostModel (future)
  simulators.py        ← thin wrappers around AgentSimulator / IntradayAgentSimulator
  strategies/
    swing.py           ← SwingStrategy: feature fetcher, fold runner
    intraday.py        ← IntradayStrategy: feature fetcher, fold runner
    day_trading.py     ← DayTradingStrategy stub (future)
  reports.py           ← report printing, CSV export
```

`scripts/walkforward_tier3.py` retained as a thin shim that imports from the package
for full backwards compatibility. All existing tests and CLI invocations unchanged.

### Extension pattern for Day Trading

```python
from scripts.walkforward import FoldEngine
from scripts.walkforward.cost_models import SpreadCostModel

engine = FoldEngine(
    strategy=DayTradingStrategy(...),
    cost_model=SpreadCostModel(half_spread_bps=5, impact_bps=3),
    purge_days=1,
    embargo_days=1,
)
report = engine.run(n_folds=6, total_days=365)
```

No changes to `walkforward_tier3.py` CLI or existing tests required.

---

## Phase WF-3 — Combinatorial Purged K-Fold (CPCV)

**Status:** ✅ COMPLETE (2026-05-07)  
**Branch:** `feat/wf3-cpcv`

### What changed

Implemented Combinatorial Purged Cross-Validation (López de Prado, AFML Chapter 12).

**Standard 3-fold expanding WF gives:**
- 3 test periods, all right-anchored
- Single Sharpe point estimate
- Folds are NOT independent (fold 2 train ⊃ fold 1 test)

**CPCV with k=6, paths=2 gives:**
- C(6,2) = 15 independent test paths through the data
- Each path tests on 2 of 6 folds, trains on the other 4
- Reports Sharpe distribution: mean, std, P5/P95 confidence band
- Much higher statistical power; DSR uses distribution shape not just point estimate

**New CLI flag:**

```bash
python scripts/walkforward_tier3.py --model swing --cpcv --cpcv-k 6 --cpcv-paths 2
```

**New gate (CPCV mode):**

```
CPCV avg Sharpe (mean of 15 paths) > 0.8
AND CPCV P5 Sharpe > -0.3   (5th percentile of distribution, not just worst fold)
AND CPCV pct_positive >= 0.75  (≥ 75% of paths are positive Sharpe)
AND DSR p > 0.95 (using distribution shape)
AND avg_profit_factor >= 1.10
AND avg_calmar >= 0.30
```

**Why CPCV matters:**

With 3 folds, a lucky model can show avg Sharpe +0.8 on 2 good folds and 1 flat fold.
With 15 paths, luck is far harder to sustain. A genuine edge shows consistent positive
Sharpe across most paths; an overfit model shows high variance and many negative paths.

**Performance note:** CPCV with k=6 on swing requires 15 fold runs ≈ 15 × 17 min = 4h.
Recommended to run overnight. Standard 3-fold WF remains the fast development gate;
CPCV is the final promotion gate before live trading.

---

## Future Phases

### WF-4 — Regime-Stratified Testing (planned)

- Tag every trading day with regime label: VIX quartile × SPY trend × momentum
- Ensure each CPCV fold has minimum regime diversity
- Report per-regime Sharpe breakdown
- Additional gate: worst-regime Sharpe > -0.5

### WF-5 — Full Strategy Simulation Fidelity (planned)

- PM opportunity score, macro gate, earnings blackout all correctly simulated per fold
- Portfolio-level Sharpe (correlation budget, max N positions, sector limits)
- Vol-targeting position sizing to match live behavior
- After this, WF Sharpe should closely predict live paper Sharpe

### WF-6 — Live Reconciliation Bridge (planned, needs 30+ paper trading days)

- Auto-compare paper trading Sharpe vs WF predicted Sharpe for same period
- Track implementation shortfall: simulated fill vs actual Alpaca fill
- Feature drift detection: flag when live feature distribution diverges

### WF-7 — Day Trading Cost Model (future, after day trading strategy defined)

- Bid-ask spread as function of time-of-day and volume
- Market impact model for larger orders
- Latency model for direct market access

---

## Gate Summary (All Phases Applied)

| Gate | Threshold | Rationale |
|---|---|---|
| Avg Sharpe | > 0.80 | Primary performance hurdle |
| Min fold Sharpe | > -0.30 | No fold catastrophically negative |
| DSR p-value | > 0.95 | Statistical significance after N trials correction |
| Avg profit factor | > 1.10 | Gross profitability before cost optimisation |
| Avg Calmar ratio | > 0.30 | Return/drawdown quality |
| CPCV pct positive | ≥ 0.75 | Robustness across combinatorial paths |
| CPCV P5 Sharpe | > -0.30 | 5th percentile distribution gate |

A model must pass **all gates** to be promoted to paper trading.
