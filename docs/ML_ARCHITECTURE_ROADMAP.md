# MrTrader ML Architecture Roadmap

Last updated: 2026-05-09

This document is the canonical plan for restoring statistical honesty to the
MrTrader ML stack. It supersedes ad-hoc experiment notes for any topic it
covers explicitly. Phase log entries continue to live in
`docs/ML_EXPERIMENT_LOG.md`.

---

## §1 Why this exists

After ~190 swing-model iterations and ~50 intraday iterations, the team
discovered that:

1. Multiple "honest" walk-forward numbers had been re-used across many
   experiments — the same OOS data informed dozens of decisions, so its
   statistical value was already exhausted.
2. The 5-fold expanding WF gives ONE point estimate of Sharpe per run; with
   ~200 model variants tried, selection bias on that single number is large.
3. There is no inviolable holdout. Every byte of available data has, at some
   point, been seen during development.

P0 begins by establishing one — see §2 and §9 below.

---

## §2 Guiding principles

1. **Single sacred holdout.** A fixed window of recent data
   (`SACRED_HOLDOUT_START = 2025-11-09`, ≈last 6 months) is reserved for the
   one-shot final evaluation of the eventual promotion candidate. It is
   touched ONCE, ever. A code-level hard guard (`assert_no_sacred_holdout` in
   `app/ml/retrain_config.py`) blocks any training, walk-forward, or CPCV run
   from reading data on/after this date. Bypass requires the explicit
   `allow_sacred_holdout=True` (Python) or `--allow-sacred-holdout` (CLI)
   flag, which logs a banner warning. Do not develop with the bypass set.
2. **CPCV-first measurement.** Standard 5-fold expanding WF is replaced by
   Combinatorial Purged CV (López de Prado, AFML Ch. 12) — k=6, paths=2,
   yielding C(6,2)=15 independent test paths and a Sharpe distribution
   instead of a point estimate.
3. **Decisions cite distributions, not points.** From P0 onward, gate
   thresholds are stated in terms of mean Sharpe, P5 Sharpe (the floor),
   pct-positive paths, and DSR p-value (selection-bias adjusted).
4. **Document first, code second.** Any change to a model, feature set, or
   training config requires a corresponding entry in
   `docs/ML_EXPERIMENT_LOG.md` BEFORE merge.

---

## §9 P0 results — first honest baseline (CPCV)

CPCV runs are kicked off but take ~4h each on the dev machine. The exact
commands and log paths are listed below; numbers are filled in as runs
complete.

### Reproduce

Swing v171 baseline (Tier-3 5-fold settings, but CPCV k=6 paths=2):

```bash
python scripts/walkforward_tier3.py \
  --model swing \
  --years 6 \
  --swing-train-years 6 \
  --cpcv --cpcv-k 6 --cpcv-paths 2 \
  --swing-cost-bps 5 \
  2>&1 | tee logs/p0_v171_cpcv_baseline.log
```

Intraday v51 baseline (365d window — the honest +0.529 result; gates off):

```bash
python scripts/walkforward_tier3.py \
  --model intraday \
  --days 365 \
  --cpcv --cpcv-k 6 --cpcv-paths 2 \
  --intraday-cost-bps 15 \
  --no-pm-opportunity-score \
  --no-earnings-blackout \
  --no-dispersion-gate \
  --no-macro-gate \
  2>&1 | tee logs/p0_v51_cpcv_baseline.log
```

Parse results:

```bash
python scripts/parse_cpcv_results.py logs/p0_v171_cpcv_baseline.log
python scripts/parse_cpcv_results.py logs/p0_v51_cpcv_baseline.log
```

### Results (fill in after runs complete)

| Model      | Mean Sharpe | P5 Sharpe | P95 Sharpe | % positive | DSR p | Gate |
|------------|-------------|-----------|------------|------------|-------|------|
| v171 swing | TBD         | TBD       | TBD        | TBD        | TBD   | TBD  |
| v51 intra  | TBD         | TBD       | TBD        | TBD        | TBD   | TBD  |

These numbers become the **honest baseline** every later P-phase
improvement is measured against. Replace TBD inline once each log is
parsed.

---

## P0 implementation notes (2026-05-09)

**Sacred holdout enforcement — design decisions**

- **Where the constant lives:** `app/ml/retrain_config.py` —
  `SACRED_HOLDOUT_START: str = "2025-11-09"`. Single source of truth.
- **Boundary semantics:** **inclusive** — a run whose data window ends ON
  the holdout day is rejected (the day itself is sacred). Only `end_date <
  boundary` is allowed.
- **Defense in depth:** the guard is invoked at FIVE distinct layers so a
  single missed wiring cannot leak holdout data into a training run:
  1. `ModelTrainer.train_model()` — checked against `date.today()` before
     `_fetch_data` is called.
  2. `ModelTrainer._build_rolling_matrix()` — checked against `max(all_dates)`
     of the assembled SPY/union date spine. Catches direct callers that skip
     `train_model()`.
  3. `scripts/walkforward_tier3.py` `main()` — checked at CLI entry, before
     any swing/intraday/CPCV branches run.
  4. `scripts/walkforward/cpcv.py::run_cpcv()` — checked when `end_all` is
     computed inside CPCV.
  5. `scripts/walkforward/engine.py::FoldEngine.run()` — checked when
     `end_all` is computed inside the standard WF engine.
- **Bypass mechanism:** `allow_sacred_holdout: bool` parameter (Python) and
  `--allow-sacred-holdout` flag (CLI). When True, the guard logs a banner
  WARNING (multi-line, framed by `=`*72) identifying the calling context,
  then proceeds. The bypass is plumbed through walkforward_tier3 → run_cpcv
  so the one-shot promotion run can drive end-to-end. The
  `engine.FoldEngine.run()` and `_build_rolling_matrix()` guards
  intentionally DO NOT receive the bypass from CLI plumbing — they default
  to `False` and require an explicit Python-level keyword for any internal
  caller, preventing accidental bypass from a refactor.
- **Tests:** `tests/test_p0_sacred_holdout.py` covers boundary inclusivity,
  bypass logging, input coercion (str/datetime/date), ModelTrainer
  integration with patched `date.today()`, and the walkforward_tier3 CLI
  path with patched `date.today()`.

**CPCV baseline — design decisions**

- v171 swing: same 6-year window, train_years=6, cost=5bps as the existing
  Tier-3 5-fold so the CPCV mean is directly comparable to the prior point
  estimate.
- v51 intraday: 365-day window with WF-5a gates **off**
  (`--no-pm-opportunity-score --no-earnings-blackout --no-dispersion-gate
  --no-macro-gate`) — this is the configuration that produced the honest
  +0.529 result.
- CPCV k=6, paths=2 → C(6,2)=15 paths. Per `cpcv.CPCVResult.gate_passed`,
  promotion requires mean Sharpe ≥ gate, P5 ≥ −0.30, pct_positive ≥ 75%,
  DSR p > 0.95.
- Logs go to `logs/p0_v171_cpcv_baseline.log` and
  `logs/p0_v51_cpcv_baseline.log`. `scripts/parse_cpcv_results.py` extracts
  the headline numbers (also supports `--json` for downstream tooling).

---

## §10 Phase 93 — FMP Quarterly Fundamentals (data layer)

**Status:** Infrastructure complete (2026-05-09). Backfill pending.

Replaces `data/fundamentals/fundamentals_history.parquet` (EDGAR annual)
with `data/fundamentals/fmp_fundamentals_history.parquet` (FMP quarterly).

**What it un-prunes from `_BASE_PRUNED`:** `pe_ratio`, `pb_ratio` —
computed PIT-correct as `price / (eps_diluted * 4)` and `price / bvps`
using the window-end close (subprocess-safe, no API calls in workers).

**What it newly provides:** `gross_margin`, `operating_margin`, `fcf_margin`
on a quarterly cadence (~4× more PIT snapshots than the annual EDGAR store).
Also overrides `profit_margin`, `revenue_growth` (now true YoY same-quarter),
and `debt_to_equity` with quarterly values.

**Transition policy:** EDGAR loader and parquet remain in place. FMP
overrides EDGAR where present. Toggle off via `USE_FMP_FUNDAMENTALS=False`
in `app/ml/retrain_config.py` to A/B against the EDGAR baseline.

See `app/data/fmp_fundamentals.py` for full design rationale (PIT, PE/PB
computation, transition policy).
