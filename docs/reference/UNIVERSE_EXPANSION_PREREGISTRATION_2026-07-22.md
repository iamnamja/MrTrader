# Universe-Expansion Study — Pre-Registration (2026-07-22)

**Frozen BEFORE the run. Owner-authorized re-validation of the EXISTING trend edge on a broader
asset set — NOT a new-edge hunt** (the same TSMOM engine + the same DUAL gate CH2 used; re-validating
the live strategy is permitted under the CH5 moratorium). Report-only; no live change without a
separate, gated flip.

## Question
Does adding one or more **genuinely orthogonal macro** ETFs to the live 10-ETF trend universe
**improve the constant-gross trend book** on the same CPCV path CH0a was frozen on — beating the
baseline mean_sharpe *significantly* AND without regressing the BEAR tail?

## Baseline (the bar)
CH0a frozen: universe `SPY,QQQ,IWM,EFA,EEM,TLT,IEF,GLD,DBC,UUP`; **CPCV mean_sharpe 0.7009**; regime
profile BULL / NEUTRAL / **BEAR** (BEAR SR is the no-regression prong). Same `BASELINE_END`, same
`tsmom_backtest(TSMOMConfig())` constant-gross book, same `evaluate_sleeve` CPCV.

## Candidates (deep 2007→2026 history confirmed for all)
Chosen for **orthogonality** to the existing legs (not more US equity — CH3 already showed correlated
equity slices don't diversify):
- **Commodity granularity** (beyond broad DBC): `USO` (oil), `UNG` (nat-gas), `DBA` (ags)
- **FX** (beyond broad-USD UUP): `FXE` (euro), `FXY` (yen)
- **Real assets**: `VNQ` (REITs), `TIP` (inflation-linked)
- **Metals**: `SLV` (silver)

Tested as: each **added individually** (11-ETF universe) + four **groups** — commodities_granular
(USO+UNG+DBA), fx (FXE+FXY), real_assets (VNQ+TIP), kitchen_sink (all 8).

## DUAL gate (identical to CH2) — PASS ⇔ all three
1. **Beats**: candidate CPCV mean_sharpe **> 0.7009** (strict).
2. **Significant**: paired stationary-block-bootstrap `P(ΔSharpe ≤ 0) < 0.05` (candidate vs baseline,
   paired on the same dates — isolates the universe change from shared market noise).
3. **BEAR no-regression**: candidate BEAR-regime Sharpe **≥ baseline BEAR − tol**.

A candidate that beats but is within the noise band (prong 2) or regresses BEAR (prong 3) is a FAIL.

## Decision rule (pre-committed)
- **No candidate PASSes** → keep the universe as-is; record the null. (Strong prior — the macro basket
  is already broad; this confirms or refutes it with evidence.)
- **A candidate PASSes** → it becomes a *promotion candidate* only; wiring the expanded universe into
  the live `pm.trend_universe` is a SEPARATE owner-gated step after a fresh out-of-sample confirmation
  (no cherry-picking the best-of-13 into production on this run alone).
- Multiplicity is disclosed: 13 evaluations (baseline + 8 individual + 4 group) charged to the DSR
  N_TRIALS as a secondary (deflated) metric; the DUAL-gate decision is on raw mean_sharpe + the paired
  significance, as in CH0a/CH2.

## Artifact
`app/research/universe_expansion.py` → `docs/reference/universe_expansion_results.json` + a printed
table. Verdict logged to `ML_EXPERIMENT_LOG.md` + a `DECISIONS.md` entry.

## Result (2026-07-22) — 0/12 PASS → KEEP the 10-ETF universe
Every candidate LOWERED CPCV mean_sharpe (best +DBA 0.6809 Δ−0.020; worst kitchen-sink-18 0.4935
Δ−0.207); none beat 0.7009, none significant (paired-bootstrap p 0.37–0.94). In-run baseline
reproduced 0.7009 exactly. Nuance: several commodity adds improved the BEAR tail (USO Δbear +0.265)
but the BULL/NEUTRAL give-up outweighed → still fail the primary prong. **No expansion.** Full table:
DECISIONS 2026-07-22 (universe-expansion study).
