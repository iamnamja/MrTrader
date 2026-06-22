# Alpha-v10 P0.4 — Credit de-risk overlay: PIT re-verdict (2026-06-22)

**Verdict: PIT-ROBUST CANDIDATE — the panel's "shrinks under PIT vol" hypothesis is REFUTED.**
Status UNCHANGED: the credit overlay remains a marginal tail-insurance **CANDIDATE, flag OFF**
(`pm.credit_governor_enabled` default false). The PIT-vol question is now closed; the binding
caveat is multiplicity (a post-hoc trigger), not vol.

## Context
The 2nd 5-LLM panel (Alpha-v10) action item (2) bundled the credit overlay with carry:
> "kill in-sample vol-matching → PIT rolling vol everywhere (the +0.17 carry dSR + the +0.064
> credit overlay likely shrink)."

P0.2 confirmed this for **carry** — its in-sample +0.17 dSR was a vol-match artifact that
collapsed to ~0.00 under PIT vol. P0.4 tests the same hypothesis for the **credit overlay**.

## Why the credit overlay is structurally different from carry
The credit overlay is a **time-varying de-risk MULTIPLIER** `m[t] ∈ [derisk_to, 1.0]` applied to
the trend book: `overlaid[t] = m[t]·base[t]` (minus a small toggle cost). Two facts make its
dSharpe independent of any vol-matching:

1. **Sharpe is scale-invariant.** A *constant* vol change (`c·base`) leaves Sharpe identical
   (`_ann_stats`: `mean/std·√252`). Only the **time-variation** of `m[t]` (the timing) can move
   it. So there is no vol-normalization step in the overlay metric to inflate — and none to remove.
   (`scripts/walkforward/sleeve_lab.py` `_overlay_report`/`_ann_stats`: `overlaid = m·base − cost`,
   `dSharpe = Sharpe(with) − Sharpe(without)`, no re-levering of either leg.)
2. **The baseline trend book is PIT.** Trailing-window signals + realized vol and `held.shift(1)`
   earning, no look-ahead (`app/strategy/tsmom.py`). (`book_vol_target` is OFF on the live trend
   book, and PIT/non-circular where used elsewhere.) The credit trigger is PIT too: the HYG/IEF
   trailing-MA signal is `.shift(1)` so close[t] governs t+1 (`credit_curve_governor.py`).

Carry's +0.17 collapsed because it *was* a vol-normalization artifact; the credit overlay's
metric is immune to that mechanism by construction.

## Result (current data, `run_credit_curve` on the live trend book, 2007-04-12 → 2026-06-12, n=4824)
Marginal to the live VIX-term governor (the live-today policy):

| metric | standalone | **marginal vs VIX governor** |
|---|---|---|
| dSharpe | +0.0937 | **+0.0639** |
| dCalmar | +0.0385 | **+0.0298** |
| dMaxDD (overall) | +0.0026 | **+0.0010** |
| dAnnRet | +0.0040 | +0.0030 |

- mean multiplier 0.909; de-risks 18.3% of days.
- per-crisis (marginal) drawdown improvement: **GFC +0.019, COVID +0.022, BEAR-2022 +0.006** → helps in all 3.
- both-halves overall-dMaxDD: **H1 +0.0224, H2 +0.0010** (both positive).
- overlay report verdict: **HELPS (shallower tail, Sharpe preserved).**

The marginal **+0.0639 ≈ the original +0.064** (and dCalmar +0.0298 ≈ +0.030). It **reproduces on
current PIT data — it did NOT shrink.** (For contrast, `curve_governor` remains NO-TAIL-BENEFIT:
marginal dSharpe −0.018 → stays off.)

## Honest caveats (unchanged from DECISIONS 2026-06-14 G1)
- **Multiplicity (the binding caveat):** the trigger `lookback=120, band=0.02` is **post-hoc** —
  the pre-registered `L=60/band=0` failed (fired ~37% of days = a slow trend filter), and
  L=120/band=0.02 (fire only on >2% deterioration, ~18% of days) was the post-hoc "principled
  fix." This is the real residual risk — selection, **not** vol.
- **Small + front-loaded tail benefit:** the marginal effect is small (+0.064 dSharpe, +0.001
  *overall* dMaxDD), and the overall-maxDD benefit is concentrated in the first half (H1 +0.022 vs
  H2 +0.001) — though it does reduce drawdown *within* all three named crisis windows.

## Decision
- **No status change.** The credit overlay stays a marginal tail-insurance **CANDIDATE, flag OFF**.
  It is not killed (it is a real, PIT-robust, all-3-crises tail hedge) and not promoted (the effect
  is small and the trigger is post-hoc).
- **The PIT-vol concern is closed** — corrected the panel's mis-attribution: the +0.064 was never a
  vol-match artifact. (`curve_governor` likewise unchanged: stays off.)

## Reproduce
`PYTHONPATH=. venv/Scripts/python scripts/run_credit_pit_reverdict.py` — runs the G1 confirmatory
harness and emits the PIT verdict. Verdict logic unit-tested in `tests/test_credit_pit_reverdict.py`.

> Cross-refs: `DECISIONS.md` 2026-06-14 (G1) + 2026-06-22 (P0.4); `ML_EXPERIMENT_LOG.md` P0.4;
> the carry analogue in `docs/reference/ALPHA_V10_SYNTHESIS_AND_PLAN.md` (P0.2).
