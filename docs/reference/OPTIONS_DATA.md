# Options Data — Design & PIT/Survivorship Contract

**Status:** Alpha-v5 Options Program — OPT-1b (data layer)
**Author:** Opus 4.8 · **Date:** 2026-06-09
**Source:** **Polygon Options Developer** ($79/mo) — REST + S3 flat files (existing keys).

Companion to `app/data/options_provider.py`. Program SSOT: `docs/living/OPTIONS_PROGRAM.md`.
The pricing/greeks engine that consumes this data is OPT-1a (`app/options/pricing_engine.py`).

---

## 1. The hard data reality (Polygon Developer — verified)

| What | Available? | How we handle it |
|---|---|---|
| Historical OHLCV (per contract, daily) | ✅ S3 flat files `us_options_opra/day_aggs_v1` | Bulk-download day files, filter to our universe |
| Historical **IV / greeks** | ❌ current snapshot only | **Compute** (OPT-1a engine) from close + spot + r + q |
| Historical **open interest** | ❌ | Liquidity via **volume/notional**, never OI |
| Historical **NBBO / quotes** | ❌ | Mark off EOD close; **model + stress the spread** (OPT-2) |
| Contract universe incl. **expired** | ✅ day files *are* the universe (every contract that traded) | Survivorship-safe by construction (§3) |
| Current chain IV/greeks/OI snapshot | ✅ REST `/v3/snapshot/options/{u}` | **Validation + live only** — never backtest |
| Earnings dates | (not Polygon) | Existing FMP/Finnhub calendar (used in OPT-3) |

## 2. Endpoints & sources (verified)

- **Historical bars + universe (bulk):** S3 `us_options_opra/day_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz` — one file per trading day, **every OPRA contract** that printed a bar (OHLCV; key = OCC ticker). Via `app/data/polygon_s3.PolygonS3.get_options_day_file(day)`.
- **Current snapshot (IV/greeks/OI):** `GET /v3/snapshot/options/{underlying}` — served IV/greeks/OI for the live chain. Used by `fetch_current_snapshot` (validation/live).
- **Contract reference (active + expired):** `GET /v3/reference/options/contracts?underlying_ticker=&as_of=&expired=true`. Available via `fetch_contracts`, but the **historical backtest universe is derived from the day files**, not this endpoint (the files are the ground truth of what actually traded).

**OCC ticker format:** `O:{ROOT}{YYMMDD}{C|P}{strike×1000, 8 digits}` — e.g. `O:SPY260116C00500000` = SPY, exp 2026-01-16, call, strike 500.0. Decoded by `parse_occ`; the universe regex (`occ_root_pattern`) anchors `\d{6}` after the root so prefix-roots don't collide (SPY ≠ SPYG).

## 3. Survivorship (leak-killer #1)

The universe is built **FROM the daily flat files** — every contract that actually traded that day, **including contracts that have since expired worthless**. A contract enters the store the first day it prints a bar; it is never removed. So a backtest as-of a past date sees exactly the contracts that existed then, with no survivorship bias (the opposite of building the universe from today's active chain, which would silently drop every expired contract — and expired-worthless is precisely the modal outcome for short-premium strategies).

`include_expired=True` (the default for backtests) keeps expired contracts visible as-of any date; `include_expired=False` (live/snapshot) additionally requires `expiration >= as_of`.

## 4. Point-in-time (leak-killer #2)

An option's EOD bar for trade date **D** prints after the close, so it is only **knowable the next business day**:

- `knowable_date = D + OPT_BAR_LAG_BDAYS` with **`OPT_BAR_LAG_BDAYS = 1`** (mirrors short-volume's +1).

**Every historical accessor filters `knowable_date <= as_of`.** A contract is likewise only in the universe once its first bar is knowable. Covered by dedicated no-look-ahead tests (`tests/test_options_provider.py`). The current REST snapshot carries served IV/greeks/OI that do **not** exist historically — it is used only for engine validation (OPT-1a) and live execution (OPT-8), **never** in a backtest.

## 5. Storage

Parquet under `data/` (gitignored, like the other data stores):
- `data/options_bars.parquet`: `underlying, contract, date, open, high, low, close, volume, knowable_date`
- `data/options_contracts.parquet`: `underlying, contract, contract_type, strike, expiration, first_date, knowable_date` — derived from the bars (`contracts_from_bars`), so the metadata table is always consistent with what traded.

## 6. Accessors (`app/data/options_provider.PolygonOptionsProvider`) — the OPT-0 `OptionsDataProvider` contract

- `get_universe(underlying, as_of, include_expired=True) -> List[str]` — OCC tickers knowable on/before `as_of` (survivorship + PIT).
- `get_contract_bars(underlying, as_of) -> DataFrame` — PIT panel of per-contract EOD OHLCV (`knowable_date <= as_of`).
- `get_current_snapshot(underlying) -> Dict` — current chain w/ served IV/greeks/OI (validation/live only).

## 7. Backfill

`scripts/backfill_options.py` — `--underlyings`, `--r1k` (Russell-1000 union + index ETFs), `--years N` / `--start`/`--end`, `--workers N` (parallel day-file downloads), `--max-days` (debug), `--dry-run`. Defaults to a focused liquid optionable set (index ETFs + large caps) rather than all of OPRA — the earnings IV-crush (OPT-3) and VRP (OPT-4) strategies only need the names we actually trade. Re-runnable: merges + dedupes on `(contract, date)`. ASCII-safe console (Windows cp1252).

**Large/deep backfills without OOM (added #440).** A full 4y R1K run in one shot OOMs on the in-memory full-history concat (~120M+ rows). Instead: backfill the missing earlier window as a *separate, proven-size* run — `--out PATH` / `--out-contracts PATH` (target alternate files) + `--no-merge-existing` (write only the freshly-downloaded window, skipping the read+merge of the existing store) — then **`scripts/merge_options_parquet.py`** stream-merges the windows one parquet row-group at a time (peak memory = one row group), with a row-group-statistics overlap guard that aborts on overlapping date ranges.

**Current coverage (2026-06-10):** `data/options_bars.parquet` ≈ **112.8M bars** / **733 underlyings** / **~6.18M contracts**, spanning **2022-06-09 → 2026-06-08 (~4 years)** — the maximum the Polygon Developer plan serves (no data before ~mid-2022). This is the full local copy retained even if the subscription is cancelled.

Smoke-tested 2026-06-09: 3 business days × SPY = 19,392 bars / 8,939 contracts; 791 already-expired contracts retained (survivorship verified); PIT filter confirmed (the latest trade date's bars, knowable next day, are correctly excluded as-of that date).
