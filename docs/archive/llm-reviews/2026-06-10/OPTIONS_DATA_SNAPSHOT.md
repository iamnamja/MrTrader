# Options Data — Current Snapshot (2026-06-10)

The exact state of the locally-held options dataset the reviewer should assume is available.
(Companion to `OPTIONS_DATA.md`, which describes the design/contract; this is the live inventory.)

## Coverage

| | |
|---|---|
| **Daily option bars** | **~112.8M rows** |
| **Date span** | **2022-06-09 → 2026-06-08 (~4 years)** — the maximum the Polygon subscription serves |
| **Underlyings** | **733** (Russell-1000 union + index/sector ETFs: SPY, QQQ, IWM, DIA, XLE, GLD, SLV, TLT) |
| **Distinct contracts** | **~6.18M** (calls + puts, all strikes/expiries that actually traded; expired included) |
| **Granularity** | one **EOD OHLCV bar per contract per trading day** |
| **On-disk** | `data/options_bars.parquet` (~1.2 GB) + `data/options_contracts.parquet` (~47 MB) |

This is **everything we can own** — if the subscription is cancelled, this local copy persists; it
cannot be extended further back (the data simply isn't served before ~mid-2022 on this plan).

## Schema

- **bars:** `underlying, contract, date, open, high, low, close, volume, knowable_date`
- **contracts** (derived from bars): `underlying, contract, contract_type, strike, expiration, first_date, knowable_date`
- **OCC ticker:** `O:{ROOT}{YYMMDD}{C|P}{strike×1000, 8 digits}` (e.g. `O:SPY260116C00500000`).

## Point-in-time + survivorship (both enforced)

- **Survivorship-safe:** the universe is built FROM the daily flat files — every contract that
  actually printed a bar that day, **including ones since expired worthless**. A backtest as-of a
  past date sees exactly what existed then (no survivor bias — critical for short-premium, whose
  modal outcome is expire-worthless).
- **PIT:** an EOD bar for trade date *D* is only **knowable the next business day** —
  `knowable_date = D + 1 bday`. Every historical accessor filters `knowable_date <= as_of`.

## Hard data limits (verified) — these constrain what's testable

| What | Have it? | Workaround |
|---|---|---|
| Historical OHLCV per contract (daily) | ✅ | — |
| Historical **IV / greeks** | ❌ (snapshot only) | **Compute** (Black-Scholes / Bjerksund-Stensland / CRR) from close + spot + r + q |
| Historical **open interest** | ❌ | Liquidity via **volume/notional**, never OI |
| Historical **NBBO / quotes / intraday** | ❌ (EOD bars only) | Mark off EOD close; **model + stress** the bid/ask spread |
| Current-chain IV/greeks/OI snapshot | ✅ (REST) | **live + validation only — never in a backtest** |

**Source:** Polygon Options Developer ($79/mo), S3 flat files `us_options_opra/day_aggs_v1`.

> Implication for the reviewer: any options signal must be derivable from **daily OHLCV + computed
> greeks + the contract universe** — we have *no* historical IV surface, OI, or intraday/quote data.
> Strategies needing those (e.g. true dealer-gamma from OI, intraday vol-arb, quote-based execution
> alpha) are **not testable on this data** without a different/again-paid feed.
