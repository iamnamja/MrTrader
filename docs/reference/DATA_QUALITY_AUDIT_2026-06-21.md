# Data-Quality Audit — 2026-06-21

**Trigger:** owner question — *"can we do a full review of our current historical data to see
if there are outliers in what we've saved/downloaded that might have caused issues?"* The
2026-06-18 carry hardening had already found one data artifact that flipped a sign (CL
2020-04-21 negative back-adjusted denominator), so the question is well-founded: a single bad
print can move a Sharpe enough to flip a keep/kill verdict.

**Method:** a read-only sweep over every persisted dataset, via the reusable tool
[`scripts/audit_data_quality.py`](../../scripts/audit_data_quality.py)
(`venv/Scripts/python scripts/audit_data_quality.py [--section futures|macro|finra|fundamentals|equities]`).
Checks per dataset: duplicate / non-monotonic dates, OHLC consistency (+ breach *magnitude*),
non-positive prices, return outliers (winsor-cap hit-rate, raw max move), stale/frozen runs,
calendar gaps, ratio-range `[0,1]`, and schema-specific invariants (filing-date sanity, negative
revenue, PIT duplicates).

---

## Headline

**The data behind every LIVE position and every recent keep/kill verdict is clean.** All defects
found are confined to **OFF-strategy** data (FMP fundamentals → PEAD=off + unused fundamentals
factors; the 850-symbol equity cache → swing-ML=off). **No live verdict was corrupted.**

| Dataset | Feeds | Verdict |
|---|---|---|
| Norgate futures (76-mkt universe) | carry / xsmom / book / VIX-VRP (paper) | ✅ **clean** |
| Macro VIX curve (`macro_history`) | VRP gate, regime | ✅ clean in-sample (trailing-edge NaNs self-heal) |
| FINRA short-volume | P3-5 (report-only) | ✅ **pristine** |
| Liquid ETF / large-cap prices | ETF-trend (LIVE), cash | ✅ clean |
| FMP fundamentals | PEAD (off), fundamentals factors (unused) | ⚠️ 2 defects → **FIXED** |
| 850-symbol equity daily cache | swing-ML (off) | ⚠️ known artifacts → **documented** |

---

## Findings & dispositions

### 1. Norgate futures — CLEAN
- 105 markets: **0 duplicate dates, 0 non-monotonic, 0 OHLC violations** (except CL's one
  2020-04-20 negative-price day, already guarded).
- Only **CL** has a negative unadjusted close — the exact case `true_returns()`'s
  negative-denominator guard already NaNs. Winsorization (`±50%`) touches **~0% of days**
  (max 0.02%, VX) → it neutralizes the CL/VX representation spikes without distorting anything.
- **Long stale runs** (YIB 256, ZQ 65, SO3 33, …) are all STIR/illiquid rate instruments that
  `liquid_universe()` correctly **excludes**. The one liquid market with a stale run, **HO**
  (heating oil, run=11), is entirely in **1978–79** (the contract's first thin months) — **zero**
  stale runs after 2005. No impact on modern-regime research.

### 2. Macro VIX curve — CLEAN in-sample
- 0 duplicate dates, 0 gaps. The large 1-day moves (VIX +115%, VIX3M +65%) are **real**
  spikes (Feb-2018 volmageddon, COVID), not errors. 8.6% backwardation days = real.
- The only NaNs are on the **trailing edge** (2026-05-25 holiday + the last partial row) — these
  self-heal on `load_macro_history()` (it refetches incomplete trailing rows). In-sample
  (2018→2024) history is complete. Not verdict-affecting.

### 3. FINRA short-volume — PRISTINE
- 0 dups, 0 gaps, all ratios in `[0,1]`, no NaN, positive volumes. Nothing to fix.

### 4. FMP fundamentals — TWO REAL DEFECTS → FIXED
Both fixed centrally in `load_fmp_fundamentals()` via a new pure helper
`_apply_quality_guards()` (so the on-disk parquet is corrected on read — no network refetch —
and every consumer benefits). Validated on the live parquet: **42,009 rows in → 42,009 out
(zero data loss).**

- **(a) Negative revenue (79 rows).** FMP maps a non-standard line item for some
  REITs/MLPs/partnerships (BEP, GLP, SLG, NHI, SAFE, …) → **negative "revenue"**, which poisons
  every `X/revenue` margin. `revenue < 0` is impossible → NaN the field + its same-row derived
  ratios. Genuine **zero**-revenue rows (pre-revenue biotech, 581 of them) are **left intact**
  (their margins are already NaN from backfill's div-by-zero guard).
- **(b) Non-deterministic PIT pick (72 duplicate `(symbol, as_of_date)` rows).** One filing date
  can carry multiple `period_end`s — usually a late filing **bundling several distinct quarters**
  (ADSK 2007-06-04 reports Q2+Q3+Q4), and occasionally FMP returning two near-identical
  period_ends for the **same** quarter (ADI 2017-11-22: 10-31 vs 10-28, revenue $1.54B vs
  $1.00B). The PIT consumers (`get_fundamentals_as_of`, `lookup_pit_from_index`) take the last
  row `as_of_date <= target`, so the tie-break used to depend on insertion order.
  **Fix = sort by `(symbol, as_of_date, period_end)` — do NOT drop the bundled quarters** (they
  are real history / YoY bases); the deterministic last-pick is then always the latest period
  reported as-of that filing. (An earlier draft `drop_duplicates`'d and was caught in adversarial
  review for deleting 71 legitimate quarters — corrected to sort-don't-drop.)

Covered by `tests/test_fmp_quality_guards.py` (no-data-loss, deterministic pick on both
consumers, negative→NaN, zero-preserved, purity, idempotence).

### 5. Equity daily cache (850 symbols) — KNOWN ARTIFACTS, documented
Feeds swing-ML only (off). No fix in this PR (the proper guard is the swing return pipeline's own
winsorization; **P4's equity audit will use survivorship-free Norgate, not this cache**).
- **8 symbols with a material OHLC breach (>0.5%)** — the other 116 of 124 raw "violations" are
  cosmetic sub-penny adjusted-close-vs-raw-OHLC rounding (yfinance adjusts close but not O/H/L),
  not broken bars. (The audit tool now reports breach *magnitude* to separate these.)
- **10 symbols with >200% 1-day moves** — reverse-split / bankruptcy-emergence artifacts
  (GPOR +52,648% on 2021-05-18, CHRD +25,733% on 2020-11-20). Real hazard for any naive equity
  backtest over these names; the swing pipeline winsorizes returns, and these names are not live.
- 0 non-positive closes, 0 duplicate dates.

---

## Follow-ups (discovered, NOT fixed here)

- **`app/ml/factor_scorer.py` — pre-existing PIT look-ahead.** All five scorers resolve the date
  column from `("date", "report_date", "filed_date")` — **none of which exist** (the FMP schema
  uses `as_of_date`) → `date_col=None` → the `as_of` filter is **silently skipped**, returning the
  full unfiltered frame (look-ahead). They also load via raw `pd.read_parquet`, bypassing the new
  guards. The fundamentals-factor sleeve is **parked/unused** (not live, no current verdict), so
  this changes no live result — but it means any past fundamentals-factor number from this path is
  look-ahead-contaminated and must be re-run after the fix (add `as_of_date` to the candidates +
  route through `load_fmp_fundamentals`). Tracked in MASTER_BACKLOG.
- **FMP YoY-base residual.** NaN-ing a negative-revenue row's own ratios does not retro-fix the
  *next* year's `revenue_growth_yoy` that used the bad base. Left to a backfill recompute
  (low priority; fundamentals factors unused).

---

## What this buys us

Certainty that the **futures book, the VIX-VRP sleeve, the live ETF-trend book, and the FINRA
work all rest on clean data** — the recent keep/kill verdicts are not data-artifact illusions.
And `scripts/audit_data_quality.py` is now a reusable pre-flight to re-run whenever a new dataset
is mirrored (e.g. Norgate US Stocks for P4, or the IBKR fill history for P2).
