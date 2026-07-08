# CH4 — Ranging-Market Mean-Reversion Sleeve: PRE-REGISTRATION (frozen 2026-07-08)

**This document is the pre-registered specification of the ONE terminating CH4 search, written and
committed BEFORE the run.** It freezes the hypothesis, the exact strategy, the parameter grid (→ the
DSR trial count), the pass/kill gates, and the binding 12-month moratorium. The run PR reports the
result against THESE gates only — no goalpost-moving, no post-hoc config flipping (OPT-5 discipline).
The git commit of this file timestamps the pre-registration.

## Why this is the one sanctioned search
Per the Compound-and-Harden program (DECISIONS 2026-07-07): after CH0–CH3 (compound + harden the one
trend edge), CH4 is the SINGLE well-specified conditional family permitted before a hunting
moratorium. CH3 produced no parked-strategy candidate (both closed), so CH4 is the originally-named
**ranging-market mean-reversion sleeve**.

## Hypothesis (mechanism, not a fishing expedition)
Trend-following EARNS in trending regimes and BLEEDS in ranging/choppy ones (CH0a/CH3: BEAR SR −0.77,
NEUTRAL +0.44 with a −45% drawdown — trend struggles when there is no trend). A short-horizon
**mean-reversion** strategy — providing liquidity to short-term overreactions — should capture return
in exactly the regime where trend cannot: **ranging markets (weak trend + low realized vol)**. Gated
to that regime, it is anti-correlated to the live trend book BY CONSTRUCTION (it is only active when
trend is quiet), so if it has any standalone edge it should DIVERSIFY the book.

*Note:* the UNCONDITIONAL ETF mean-reversion was already KILLED (`etf_relative_value`, point_SR 0.026).
CH4 tests ONLY the regime-CONDITIONAL form.

## Strategy specification (frozen)
- **Universe:** `LIVE_TREND_UNIVERSE` (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP) — reuse, deep
  history to 2007, no new data.
- **MR signal (time-series, long-flat — matches the live book's long-only stance):** per name, a
  short-horizon reversal z-score `z_i[t] = −(P_i[t]/P_i[t−L] − 1) / rv_i[t]` (recent DROP → positive
  signal = buy the dip). Long only where `z_i[t] > z_enter` (oversold); size **inverse-vol**, capped
  per-name, gross-capped — the same sizing plumbing as TSMOM. PIT: signal uses data ≤ t, applied to
  the t→t+1 return (shift(1)).
- **Regime gate (the COMPLEMENT of the trend signal — reuses a non-overfit signal):** the sleeve is
  ACTIVE only on RANGING days = **trend clarity LOW** (`trend_clarity[t] < clarity_lo`, reusing
  `app/research/ch2_sizing.trend_clarity` — the CH2 signal) **AND realized book vol LOW**
  (`book_vol[t] < vol_pctl` percentile). On non-ranging (trending) days the book is **FLAT** (all
  weights 0). Regime signal is PIT (shift(1)).

## Parameter grid (frozen → DSR trial count = 8)
The PRIMARY config decides PASS; the others are pre-registered sensitivity and ALL count toward DSR
(`n_trials_registered = 8`):
| # | name | L (lookback) | z_enter | clarity_lo | vol_pctl |
|---|---|---|---|---|---|
| 1 | **mr_primary** | 5 | 1.0 | 0.40 | 0.50 |
| 2 | mr_slow | 10 | 1.0 | 0.40 | 0.50 |
| 3 | mr_deep | 5 | 1.5 | 0.40 | 0.50 |
| 4 | mr_tight_regime | 5 | 1.0 | 0.30 | 0.40 |
| 5 | mr_loose_regime | 5 | 1.0 | 0.50 | 0.60 |
| 6 | mr_slow_deep | 10 | 1.5 | 0.40 | 0.50 |
| 7 | mr_clarity_only | 5 | 1.0 | 0.40 | 1.00 (vol gate off) |
| 8 | mr_vol_only | 5 | 1.0 | 1.00 (clarity off) | 0.50 |

## Pre-registered GATES (ALL must pass, on the PRIMARY config `mr_primary`)
1. **Standalone edge:** the in-regime sleeve returns clear the uniform sleeve gate — `evaluate_sleeve`
   → CPCV Ruler-v2 **Track-A PAPER pass** (the same bar every sleeve uses), with
   `n_trials_registered = 8` charged to the DSR/Bayesian multiplicity.
2. **Off-regime Sharpe ≥ −0.10:** on the days the regime gate is OFF (trending), the sleeve's residual
   returns are flat, not negative — i.e. the edge is genuinely from the ranging regime, not leakage.
3. **Detection-LAG test (kills regime mirages):** re-run with the regime signal lagged an EXTRA 1–5
   trading days; the in-regime CPCV mean_sharpe must **survive** (retain ≥ 60% of its un-lagged value
   and stay positive). A regime edge that evaporates under a small extra lag was a look-ahead mirage.
4. **Track-B residual-alpha vs the live trend book:** the sleeve must add DIVERSIFYING alpha — Track-B
   residual-α HAC t ≥ **1.96** vs `live_trend_book_returns` (the same diversification gate carry/xsmom
   faced). Anti-correlation alone is not enough; it must earn.
5. **DSR:** `n_trials_registered = 8` (the grid) + family registration; the parametric DSR cross-check
   uses `family_trial_count()`.

**Decision rule:** PASS iff `mr_primary` clears gates 1–5. The other 7 configs are reported as
sensitivity but do NOT decide (no best-of-8 selection). If `mr_primary` fails but a sensitivity config
passes, that is a KILL (the primary was the pre-registered bet) — noted, not shipped.

## Outcome (terminating either way)
- **PASS →** a genuine second-sleeve candidate; it goes to PAPER (shadow), NOT live capital, and enters
  its own soak + the R2 data-gated sizing path. The moratorium does NOT bind (we found something).
- **KILL →** ship nothing, and the **12-MONTH HUNTING MORATORIUM BINDS** (below).

## THE 12-MONTH HUNTING MORATORIUM (binds on a CH4 KILL)
**If CH4 KILLs, no new broad edge-discovery search runs until 2027-07-08.** Between now and then the
program is CH5 only: operate + compound the hardened single trend edge (+ cash), accrue the live-forward
scorecard, and harden operations. The next edge-hunt is permitted ONLY at the CH5 pre-committed 12-month
review (2027-07-08), and only if the live trend edge has held up as expected. This moratorium is the
price of the one CH4 search — it is committed HERE, before the run, so a null result cannot rationalize
"just one more idea." (Bug-fixes, hardening, execution/data work, and re-validation of LIVE strategies
are NOT edge-hunts and remain allowed.)
