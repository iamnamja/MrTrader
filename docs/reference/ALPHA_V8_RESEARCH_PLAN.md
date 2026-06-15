# MrTrader — Alpha-v8 Research Program (Overlay & Timing track)

**Status:** ACTIVE plan (2026-06-14). Opus-architected. Execute phase-by-phase; exhaustive
pre-registered test of each before moving to the next.

**Goal:** find new alpha / risk-improvement on FREE daily US data via three sequential angles —
a **credit/curve de-risk overlay**, a **short-interest de-risk overlay**, and an **additive
long-flat timing sleeve** — each run through the existing Sleeve Lab gate.

**Why this shape (the F-series lessons):** additive *equity* sleeves on free daily US data keep
dying on one of two walls — IC ≈ 0, or (if they time broad equity) `corr ≥ 0.30` to the trend
book's SPY leg → Track-B fail (this killed the F1a calendar premia). The one thing that worked is
an **overlay** (the VIX governor) — judged on improving the *existing* book, not on being
uncorrelated. **Power** is the recurring killer (bi-monthly < daily). And the carry near-miss
proved a full-sample edge can be a sub-period artifact → a **pre-registered both-halves stability
guard is mandatory**. Independent **Opus adversarial review caught a look-ahead in every build** →
it's a required gate step.

---

## Sequencing (by EV-per-effort × power)

| Phase | Angle | Power | Friction | Prior | Order rationale |
|---|---|---|---|---|---|
| **G0** | Shared infra: overlay **marginal-stacking** API + overlay registry | — | low | — | Unblocks G1/G2 marginal eval; cheap; once |
| **G1** | Credit/curve de-risk **overlay** | **High** (daily, 2007+, ~4800 obs) | **Zero** (owned data) | High (credit/curve lead equity stress) | Best EV/effort; mirrors proven governor |
| **G2** | Short-interest de-risk **overlay** | **Low** (bi-monthly, ≤2017-12-29, ~190 obs) | Low–med (aggregation; data layer exists) | High (informed shorts) but power-starved | High prior, power-gated; depth = hard go/no-go |
| **G3** | Additive long-flat timing **sleeve** | Medium (daily) but structurally hardest | low–med | Low (corr<0.30 wall vs trend book) | Hardest Track-B wall; only if G1/G2 surface a usable orthogonal signal |
| **G4** | Synthesis + owner-gated live wiring of any winner(s) | — | — | — | Compose the live overlay stack (governor + winners) |

Each sub-phase = **one PR**, **report-only until owner promotion**, **fully pre-registered before
any confirmatory run**, **fail-safe**, **no-drift docs**.

---

## G0 — Shared infrastructure (do first)

**Gap:** `evaluate_overlay` compares `base` vs `mult×base`; it cannot answer "does credit add
*beyond* the governor?". Add the minimal **marginal** API (no new module; in `sleeve_lab.py`):

1. `compose_overlays(overlays) -> Overlay` — inner-join multiplier series, multiply elementwise,
   clamp the product to `[GLOBAL_DERISK_FLOOR, 1.0]` (default 0.25 — two 0.5 overlays = 0.25; the
   bound prevents an unintended near-flat book), warn loudly if the clamp binds.
2. `evaluate_overlay_marginal(candidate, base_book_returns, *, prior_overlays=())` — computes
   `baseline = book×compose(prior)` and `stacked = book×compose(prior+[candidate])`, runs the
   existing `_ann_stats`/crisis logic on **stacked vs baseline** → deltas are *marginal*. Reduces
   byte-exactly to `evaluate_overlay` when `prior_overlays=()` (regression-tested).
3. Overlay registry: `OVERLAY_REGISTRY`, `@register_overlay`, `build_overlay`, `list_overlays`;
   move `build_vix_term_governor` under `@register_overlay("vix_term_governor")` (behavior unchanged).

**NOT building:** an overlay optimizer / DAG / live composition logic (lives in G4 wiring).
**Tests:** compose alignment/clamp/guards; marginal-parity with `evaluate_overlay`; PIT (composed
series inherits each builder's `shift(1)`). Full pytest + flake8. **Kill:** n/a (prerequisite).

---

## G1 — Credit/curve de-risk overlay (highest EV)

**Mechanism:** credit spreads + the yield curve lead equity stress. Two daily deep signals →
as-applied `[derisk_to,1.0]` multipliers (signal close[t] → applied t+1, `.shift(1)`):
- **Credit:** HYG/IEF total-return ratio falling (20–60d) = widening spreads → de-risk.
- **Curve:** ^TNX−^IRX (10y−3m) inversion / bear-steepening = stress flag.

| Sub | Build | Eval | PR |
|---|---|---|---|
| G1a | Deep credit/curve cache: HYG/IEF→2007 (HYG inception 2007-04), ^TNX/^IRX→2002 (`fetch_yield`). Audit adjusted-close (TR) consistency + the 2018 macro-cache splice. | data audit | `g1a-credit-data` |
| G1b | `app/strategy/credit_curve_governor.py`: `credit_multiplier`/`curve_multiplier` + `live_*` (fail-safe, single-source-of-truth w/ backtest), `@register_overlay`. | `evaluate_overlay` standalone + **`evaluate_overlay_marginal` vs book×gov** | `g1b-credit-overlay` |
| G1c | Pre-registered confirmatory run + Opus adversarial review | gate verdict | registry row + verdict |

**Pre-registered acceptance (marginal: `book×gov×credit` vs `book×gov`):**
- `improves_tail` (d_max_dd>0 AND Calmar ≥ baseline) AND `sharpe_preserved` (d_sharpe ≥ −0.05).
- Crisis: improves drawdown in **≥2 of 3** {GFC_2008, COVID_2020, BEAR_2022}, degrades none.
- **MANDATORY both-halves stability:** d_max_dd>0 in BOTH halves of the overlay-active window.
- **Marginal-additivity:** must add tail benefit *beyond* the governor (not redundant — if the
  de-risk days highly overlap the governor's AND marginal d_max_dd≈0 → park).

**Exhaustive tests:** unit (multiplier math, debounce, clamp) · PIT/no-lookahead + live-vs-backtest
agreement · fail-safe (missing/stale/NaN → 1.0; flag-off → 1.0) · robustness grid (lookback ×
derisk_to × threshold — pass needs the pre-reg config AND grid majority-positive, not one lucky
cell) · crisis windows (+2011, 2018-Q4 report-only) · **Opus adversarial deep-dive** (HYG
dividend-adjust timing, TNX/HYG calendar alignment, 2018 cache splice) · full pytest+flake8.

**EV:** highest. Realistic: a *modest* marginal tail gain over the governor (credit may add
lead-time on slow-burn 2008/2022 stress the VIX governor catches late). **Kill:** marginal
d_max_dd≤0 full-sample, or both-halves fail, or pure governor-redundancy → park, keep data, → G2.

---

## G2 — Short-interest de-risk overlay (power-gated)

**Data reality:** `app/data/short_interest_provider.py` already exists (Polygon-sourced,
per-security `short_interest`, PIT `knowable_date` = settlement + ~10 bday lag, parquet cache,
paginated). Depth bottoms at **2017-12-29** → ~190 bi-monthly obs.

**G2a — DEPTH-CONFIRMATION GATE (blocking, first):** confirm a continuous bi-monthly aggregate
series 2017-12-29→today with enough distinct de-risk events across the ≤3 in-window crises
(2018-Q4, COVID-2020, 2022). **Go/no-go:** if discontinuous or all de-risk events cluster in one
crisis → **STOP** (power-starved), document, skip to G3.

**G2 build (if G2a passes):** market-level aggregate (pre-register one: broad-ETF SI, or a
survivorship-safe universe `Σ short / Σ shares|$vol`) via `aggregate_short_interest_at(as_of)`
using the provider's `knowable_date` filter. **SII = PIT-safe expanding/trailing z** (e.g. trailing
~24-obs) — NEVER full-sample z. High SII (crowded shorts) → de-risk; step-function held between
bi-monthly updates, applied only after `knowable_date`. `@register_overlay("short_interest_governor")`
+ `app/strategy/short_interest_governor.py`.

**Pre-registered acceptance** (marginal to G1+gov): `improves_tail`, `sharpe_preserved`, crisis
benefit in ≥2 of the available windows, **both-halves guard (~95 obs/half — weak power stated)**,
and a **pre-registered minimum economic drawdown reduction** (so small-n noise can't fake a pass).

**Tests:** unit (aggregation, expanding-z) · **PIT (critical):** multiplier[t] uses only rows with
`knowable_date ≤ t`; trailing-z never peeks · fail-safe (staleness threshold ~20d not 7d) ·
robustness grid · Opus deep-dive (publication-lag boundary, expanding-z leakage, aggregation
survivorship) · full pytest. **EV:** high prior / low power → likely PARK. **Kill:** G2a fail,
both-halves fail, or single-crisis artifact.

---

## G3 — Additive long-flat timing sleeve (hardest, last)

Only if G1/G2 surface a *non-VIX, non-price-momentum* signal that could be genuinely orthogonal to
the trend book's SPY leg. Convert the best signal to a long-flat SPY/broad-ETF rule (long when
not-stressed, flat when stressed), daily PIT net returns, `evaluate_sleeve` → Track-A PAPER +
Track-B.

**Pre-registered acceptance:** Track-A PAPER (point_SR≥0.30, HAC p<0.05 or regime waiver if a
declared diversifier/risk_premium) · **Track-B (the wall): IR≥0.20, P(ΔSR>0)≥0.90, corr_to_book<0.30,
standalone_vt_SR>0.20** · both-halves stability · n_obs≥504/n_folds≥10 if capital-aspiring.

**Tests:** full sleeve-lab path + cost-sensitivity + robustness grid + crisis + **Opus deep-dive** +
full pytest. **EV:** lowest (corr<0.30 wall). Likely PARK. **Kill:** corr≥0.30, IC≈0, both-halves
fail. Do NOT iterate configs to beat the wall (multiplicity trap).

---

## G4 — Live wiring of any winner (owner-gated)

Mirror `_crash_governor_multiplier` exactly (proven template): per-overlay `agent_config` flags
(default ON, reversible, no restart) + fail-safe (any error/missing/stale/thin/flag-off → 1.0) +
PIT (settled closes; SI gated on `knowable_date`). Multipliers **compose multiplicatively** in
`run_trend_rebalance`: `alloc *= gov_mult * credit_mult * si_mult`, **clamped to
[GLOBAL_DERISK_FLOOR, 1.0]** (≈0.25) so independent de-risks can't near-flatten the book; each
multiplier recorded in `decision_audit`. Shadow-first; one overlay armed at a time; owner-gated.

---

## Cross-cutting standards (EVERY phase)

1. **Pre-register (R7)** the canonical spec + acceptance criteria BEFORE the run (`preregistered_at
   < run_at`, one shot, re-test = new id + cooling-off; `scripts/registry.py`).
2. **Both-halves stability guard is mandatory + pre-registered.** Never cherry-pick a grid cell.
3. **Exhaustive test** = unit + PIT/no-lookahead + fail-safe + robustness grid + crisis windows +
   full pytest + flake8 — all green BEFORE the confirmatory run.
4. **Opus adversarial deep-dive** is a required separate gate step before any promotion.
5. **Fail-safe:** overlays only REDUCE exposure; any doubt → 1.0.
6. **One PR per sub-phase;** report-only until owner promotion; living docs updated per phase.
7. **Honest power accounting** in every pre-registration (n_obs, n_crises-in-window, effective folds).

---

## Program EV / success

- **Success:** ≥1 promoted overlay (most likely G1) that demonstrably shallows the live book's
  drawdowns with Sharpe preserved, wired live with the fail-safe governor pattern — plus clean,
  pre-registered, leak-free negatives for the rest.
- **Good even if all park:** three rigorously-closed doors prevent future re-litigation. The value
  is as much in *closing doors cleanly* as opening one — which is why the registry + both-halves
  guard + Opus review are non-negotiable.

**First actions:** G0 (marginal-stacking API) → G1a (deep credit/curve data audit) → G1b/G1c.
