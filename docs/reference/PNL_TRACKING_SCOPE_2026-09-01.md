# Scope — Restore sleeve P&L recording (+ backfill) — 2026-09-01

**Status: ✅ IMPLEMENTED 2026-09-01.** D1/D2/D3 all confirmed by owner (both P&L components,
daily cadence, cash sleeve included). Outcome:

| | |
|---|---|
| Days recovered | **51 per sleeve** (2026-06-17 → 2026-08-31) |
| Trend cumulative | **+$181.92** |
| Cash cumulative | **+$30.00** |
| Sum vs NAV move | +$211.92 vs +$213.33 → **residual −$1.41** (lifetime fees $4.08) |

The reconciliation criterion in §6 earned its keep — it caught **three** silent errors that all
produced plausible-looking numbers. See "What went wrong" below.

Raised during the Track A/B review: CH5's whole purpose is "let the live-forward scorecard
accrue," and the accrual columns were empty.

---

## 1. The problem

`data/trend_tracking.db → trend_daily` has **14 rows** covering 2026-06-06 → 2026-08-31 with
`realized_pnl`, `unrealized_pnl`, `daily_pnl`, `cumulative_pnl` **all NULL**.

Two distinct defects:

| # | Defect | Effect |
|---|---|---|
| A | The caller never passes P&L | every P&L column is NULL |
| B | Called only at rebalance, not daily | 14 rows in ~3 months despite the table being `trend_daily` |

## 2. Root cause

`trend_tracker.record_daily()` accepts `realized_pnl` / `unrealized_pnl` and derives
`daily_pnl` / `cumulative_pnl` from them, gated on:

```python
pnl_supplied = realized_pnl is not None or unrealized_pnl is not None
```

The only call site — `app/live_trading/trend_sleeve.py:1000` — passes four fields and omits both:

```python
trend_tracker.record_daily(
    n_positions=n_target,
    gross_deployed=_estimate_trend_gross(approved, current, live),
    turnover=sum(...),
    extra={"mode": ..., "nav": nav, ...},
)   # realized_pnl / unrealized_pnl default to None -> pnl_supplied False
```

So the tracker is fine; the wiring was never completed. Defect B follows from the call living
inside `run_trend_rebalance`, which only executes on the weekly rebalance.

## 3. The data needed already exists

`data/back_validation.db → trend_backval_daily` holds **51 DAILY rows** (2026-06-17 → 2026-08-31),
written by the 16:15 ET snapshot, with `nav`, `positions` (symbol→shares), `prices`
(symbol→close), `intended_weights`, and the governor multipliers.

Verified coverage: **322 symbol-days, 0 held-but-unpriced (0.0%)**, no empty position or price
rows. So the backfill is a pure computation over data already on disk — no vendor calls, no
reconstruction from broker history.

**Feasibility proven before scoping.** A first-cut holding-only reconstruction gave **−$346.24**
for the trend sleeve against a NAV move of **+$213.33**.

> ⚠️ **The original explanation of that ~$560 gap was WRONG and is retained here as a caution.** It
> was attributed to "SGOV income on ~$45k at ~4.5% over 2.4 months (~$430) plus trade effects" and
> called coherent. Measured, cash P&L over the window is **+$30.00** — Alpaca paper pays no
> dividends (confirmed: no DIV activities), so SGOV earns price drift only. The real gap was the
> three reconstruction errors in §9. A plausible-sounding decomposition is not a reconciliation;
> only §6's arithmetic check caught the actual problems.

## 4. Design decisions (settle these BEFORE implementing)

**D1 — Which P&L definition?** The reconstruction above uses *holding* P&L:

```
pnl_t = Σ_s  pos_{t-1}[s] × (price_t[s] − price_{t-1}[s])
```

This is the prior book marked at today's prices. It deliberately EXCLUDES trade/execution effects,
so on a rebalance day it will not equal the true economic P&L. Options:

- **(a) Holding P&L only** — clean attribution of signal vs. price, matches what the backtest
  measures, but silently omits slippage.
- **(b) Holding + realized trade P&L** — economically complete, requires joining fills from the
  executions blotter (which now carries FIFO realized P&L, PR #665).

**Recommendation: (b)**, storing the two components SEPARATELY (`unrealized_pnl` = holding,
`realized_pnl` = fills). Storing only (a) would repeat the mistake that made `back_validation`'s
drift metric hard to interpret — implementation cost invisible in the headline number.

**D2 — Daily or per-rebalance?** Move the write to the daily 16:15 snapshot alongside
`back_validation`, not the weekly rebalance. A weekly series cannot support any of the statistics
this exists to feed.

**D3 — Cash sleeve too?** `SGOV` income is ~$430 over the window — larger in magnitude than the
trend sleeve's own P&L. A trend-only scorecard would misrepresent the book. Recommend recording
the cash sleeve on the same cadence.

## 5. Work items

| # | Item | Est. |
|---|---|---|
| 1 | Move the `record_daily` call from the rebalance path to the daily 16:15 snapshot | 1h |
| 2 | Compute holding P&L from consecutive `trend_backval_daily` snapshots | 2h |
| 3 | Join realized fill P&L from the executions blotter (FIFO, PR #665) | 2h |
| 4 | `scripts/backfill_trend_pnl.py` — dry-run + apply, rebuild the 51 historical days | 2h |
| 5 | Reconciliation check: cumulative sleeve P&L + cash income ≈ Alpaca equity delta | 1h |
| 6 | Tests: reconstruction arithmetic, rebalance-day trade handling, idempotent backfill | 2h |
| 7 | Cash-sleeve equivalent (D3) | 1h |

**~1.5 days.** Moratorium-safe: this is data/observability work, explicitly permitted
("bug-fixes, hardening, execution/data work"), and changes no trading behaviour.

## 6. Validation

The backfill is only trustworthy if it ties out. Required check before accepting:

```
Σ trend holding P&L  +  Σ realized fill P&L  +  cash income  ≈  Alpaca equity delta
```

over 2026-06-17 → present, within a tolerance that accounts for fees (−$3.49 total to date). If it
does not reconcile, the reconstruction is wrong and must not be written.

## 7. Risks

- **Rebalance-day double-count.** Holding P&L uses `pos_{t-1}`; if realized fill P&L is added
  naively the trade could be counted twice. Fills must be attributed to the day they executed and
  measured against the prior mark, not the prior close.
- **Snapshot gaps.** The 16:15 job can miss a day (it did during the 2026-08-24/25 Docker outage).
  The reconstruction must skip rather than interpolate across a gap, and record that it did.
- **Backfill is retroactive history.** Write it once, idempotently, with a backup — the same
  discipline used for the trade-book repairs.

## 8. Why this matters now

The Track A decision ("continue paper trading, then reassess") depends on this data existing. At
4.7% realized vol, live P&L cannot validate the edge on any practical horizon — but it CAN measure
implementation cost, which is where the currently-measurable problem is:
`back_validation` reports **drift −1.77%/yr against a ±1.50% limit (verdict WATCH)** and
**slippage drag −0.70 bps/day**. Against an expected ~3.3%/yr gross edge, implementation is
consuming over half of it. That is the number worth watching, and it needs a working scorecard.

---

## 9. What went wrong (post-implementation)

Three errors, each silent — the output stayed plausible every time. All were caught by §6's
reconciliation, not by inspection:

**1. Row 0 absorbed all pre-window realized P&L (+$1,051.95).** `compute_daily_pnl` applied every
fill up to the first snapshot date into row 0, so the entire trading history before the window
landed on day one. Fixed with an explicit warm-up phase that builds the lot book without emitting
realized P&L.

**2. Day-0's own fills were double-counted (−$52.61).** Warming up only *strictly* before the start
date still booked fills executed ON that date — but those trades had already happened when that
day's NAV was recorded, so they were in the baseline. Traced to a 2026-06-17 QQQ sell. Fixed by
warming up through and including the first snapshot: **the first snapshot is an opening balance
sheet, not a P&L day.**

**3. The unrealized seed started at zero (−$199.33).** `daily_pnl_series` seeded `prior_unreal = 0`,
booking the inherited open position's level as day-one P&L. The trend sleeve opened the window at
−$199.33 unrealized, so the series read +$13.30 where the truth was +$212.63. Fixed by seeding from
the first marked row. **The same bug then reappeared at the DB layer**: `record_daily` re-derives
`daily`/`cumulative` from the prior DB row and treats a missing anchor as zero, so it recomputed
the wrong series even after being handed correct inputs — hence the new `*_override` parameters,
which let a caller with an already-correct series write it verbatim.

The common thread is the **level-vs-delta distinction at a boundary**. `record_daily`'s own comment
warns about it mid-series; all three failures were the same confusion at the *start* of a series,
where there is no prior row to anchor on.

## 10. Design decisions as built

- **D1 (both components)** — `realized_pnl` and `unrealized_pnl` stored separately, so
  implementation cost stays visible rather than buried in a net figure.
- **D2 (daily)** — recording moved to the 16:15 EOD job, immediately after the snapshot it marks
  against. The weekly rebalance no longer writes P&L.
- **D3 (cash sleeve)** — `cash_daily` gained the four P&L columns via the same idempotent ALTER
  pattern. Cash P&L is +$30.00 over the window, **not the ~$430 originally assumed**: Alpaca paper
  pays no dividends (confirmed — no DIV activities), so SGOV earns price drift only.
- **`cash_prices` is a SEPARATE column, not merged into `prices`.** `compute_report`'s
  `daily_rows()` iterates `for sym, px1 in p1.items()` over every key, so a cash symbol added there
  would enter the drift/tracking-error computation. It would contribute 0 today, but only because
  SGOV happens to be absent from `positions` and `intended_weights` — an accident of the current
  book, not a guarantee.

## 11. Follow-ups

- Re-baseline `back_validation` drift once several clean weeks have passed post-incident; the
  current −1.77% (verdict WATCH) is contaminated by the 2026-08-24 double-buy.
- Slippage drag of −0.70 bps/day is now measurable per sleeve — worth attacking, since against a
  ~3.3%/yr expected gross edge it consumes a large share.

---

## 12. Post-deploy findings (first live run, 2026-09-01 16:15)

The recorder fired end-to-end and the trend series continued correctly from the backfill
(`181.92 - 114.03 = 67.89`). Two issues surfaced that only a live run could expose.

### 12a. `cash_prices=0` — the cash sleeve was skipped (FIXED)

`_fetch_prices` is trend-specific and **fail-closes when SPY is absent** ("core symbol SPY
missing/short"), so the separate cash-only call returned `None` and the sleeve logged
`not marked (unpriced holdings) — P&L not written`.

Fixed by fetching the trend universe and the held cash symbols in ONE call and splitting them at
storage time. `prices` is still filtered to exactly the trend universe, so `compute_report`'s
drift metric — which iterates every key of that column — remains untouched. Verified live:
`prices=10 cash_prices=1`, and both sleeves now record.

### 12b. ⚠️ Alpaca paper pays NO dividends — the cash sleeve is systematically understated

The first cash row read `daily -$138.04`, which is impossible for a T-bill ETF on ~$49k. It was
real: SGOV closed 100.69 -> 100.41, a monthly **ex-dividend** drop, and 493 x -$0.28 = -$138.04.
The arithmetic is right; the economics are not, because no dividend is credited.

| | |
|---|---|
| SGOV ex-div drops since 2026-06-15 | -0.27, -0.29, -0.28 = **-0.84/share** |
| Price-only return (what paper shows) | **-0.119%** |
| Total return (what a live account earns) | **+0.481%** |
| Gap over ~2.5 months | **~0.60pp** (~$270 on the held size) |

**Decision: record it faithfully, do NOT correct it.** Marking cash at total-return prices would
make the sleeve series stop reconciling against the account's own NAV, and that reconciliation is
the only thing that caught the three construction errors in §9. A scorecard that quietly disagrees
with the broker is worth less than one visibly distorted in a known, quantified way.

**Implications for Track A** — the live paper record understates the book:

- The cash sleeve reads ~-3%/yr where live would be ~+4.5%/yr.
- Roughly **$270 of income is missing** from the ~2.5-month record; the headline -0.50% equity
  move would be nearer -0.23% with it.
- The distortion GROWS with the cash allocation, and cash is currently ~49% of the book.
- It vanishes entirely on real capital, so it must not be read as strategy underperformance.

Any future comparison of live-paper results against a backtest that assumes total returns has to
adjust for this, or it will systematically penalise the cash sleeve.
