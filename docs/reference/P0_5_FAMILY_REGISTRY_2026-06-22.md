# Alpha-v10 P0.5 — Family-level trial counting (the program's true N_TRIALS) — 2026-06-22

**Outcome: the hardcoded "~20 families" deflation estimate is replaced by an AUDITABLE registry
of 25 enumerated families. The real burden is slightly higher than the guess (25 > 20), so the
parametric Deflated-Sharpe cross-check is now marginally MORE conservative — which leaves the GL-0
verdict UNCHANGED (BASKET_REAL; size modestly) and, if anything, reinforces "size modestly."**

## The panel's point (ChatGPT, Alpha-v10)
> "rules-based sleeves are OOS-by-construction" is FALSE at the *family-selection* level — we've
> tried ~20 sleeve families, so the multiple-testing burden is real and currently uncounted.

GL-0 (`null_zoo.py`) addressed the within-futures search with an **empirical selection-aware
max-stat null** (the primary, decisive test — it "replicates the researcher" for the 6 futures
factors). But its **parametric DSR cross-check** used a hardcoded `N≈20` placeholder for the broader
cross-asset burden the futures-only null can't see. P0.5 makes that count auditable.

## What P0.5 delivers
- **`app/research/family_registry.py`** — every distinct strategy FAMILY the program evaluated as a
  deploy candidate, each with status + one-line verdict + doc reference. A FAMILY = a distinct
  economic hypothesis (e.g. "futures carry", "PEAD"), NOT a parameter variant — within-family search
  is captured by the empirical max-stat null + the degrees-of-freedom log, not by inflating N.
- **`family_trial_count()` = 25** — the principled N for the parametric DSR (was hardcoded ~20).
- **A research degrees-of-freedom log** — discarded variants, bug-fix reruns, reviewer re-tests,
  post-hoc exclusions (the within-family burden, for transparency).
- **Wired into `null_zoo`** — the DSR cross-check now deflates at the enumerated count
  (`dsr_family`, `n_families` on `NullZooResult`); `dsr_n10/30` retained as a sensitivity band.

> **Live count grows as families are searched** — `family_registry.py` is the SSOT. Since this
> snapshot, `sector_rotation` (Option A, 2026-06-22) was added → **N_TRIALS = 26**.

## The count (25 trial families at P0.5; 27 registry entries, 2 excluded for auditability)
By status: **LIVE 3 · PAPER-CANDIDATE 5 · KILLED 14 · PARKED 3 · SCAFFOLD 2** (excluded from the
count: `cash_sleeve` = infra; `futures_book` = an ensemble of already-counted carry+xsmom).

| asset class | families |
|---|---|
| equity / ETF | etf_trend (LIVE), pead (KILLED), swing_ml_ranker (SCAFFOLD), intraday_ml (SCAFFOLD), short_interest_xs (KILLED), options_signal (KILLED), turn_of_month (KILLED), overnight (KILLED), etf_relative_value (KILLED), credit_timing (PARKED) |
| futures / rates | futures_trend (KILLED), futures_carry (PAPER), futures_xsmom (PAPER), curve_momentum (KILLED), futures_value (KILLED), futures_skewness (KILLED), basis_momentum (KILLED), cftc_cot (KILLED), rates_carry (KILLED) |
| volatility | vix_vrp (PAPER) |
| overlays | vix_crash_governor (LIVE), credit_overlay (PARKED), curve_overlay (PARKED), short_interest_overlay (KILLED) |
| crypto | crypto_trend (PAPER) |

## Why family-level (not per-backtest) is the right N
The Bailey & López de Prado DSR N is the number of strategy configurations effectively *selected
among*. Counting every backtest (hundreds, incl. variants/reruns) would over-deflate wildly; counting
families (distinct economic hypotheses) is the defensible granularity. Within-family variant search is
handled two ways: (1) the empirical max-stat null directly for the futures factors; (2) the
degrees-of-freedom log as disclosed transparency for the rest.

## Impact on the GL-0 verdict — UNCHANGED
`deflated_sharpe` is **monotone-decreasing in N** (more trials ⇒ harsher hurdle), so moving 20 → 25
can only LOWER the cross-check, never inflate it (verified in tests). The GL-0 DSR was 0.84 at N=20;
at N=25 it is slightly lower — still **borderline (<0.95)**, exactly as before. Crucially, the GL-0
**verdict is structurally independent of the DSR**: `verdict` (BASKET_REAL / CARRY_ONLY / RESIDUE)
is decided only by the empirical max-stat p-values (carry p 0.002, xsmom max-of-6 p 0.005, book p
0.002); the DSR is a parametric *cross-check* in the notes, never a hard veto. So the principled count
**confirms** the existing read: the futures book is real but modest → **size modestly**.

## Reproduce
`PYTHONPATH=. venv/Scripts/python scripts/run_family_registry.py` — prints the registry, the
N_TRIALS, the status counts, and the degrees-of-freedom log. Tests: `tests/test_family_registry.py`
(11) — registry integrity, the count, auditable exclusions, and the monotone-deflation property.

> Cross-refs: `GL0_GL1_FINDINGS_2026-06-21.md` (the empirical null + DSR), `null_zoo.py`,
> DECISIONS 2026-06-22 (P0.5), ML_EXPERIMENT_LOG 2026-06-22. Report-only — no live trading path.
