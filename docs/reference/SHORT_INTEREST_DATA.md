# Short Interest & Short Volume Data — Design & PIT Contract

**Status:** Alpha-v3 Track A2 (short-interest / squeeze edge) — *data acquisition phase*
**Author:** Opus 4.8 · **Date:** 2026-06-03
**Source decision:** **Polygon** (existing `POLYGON_API_KEY`, on-plan) — FINRA-originated, no new credentials.

---

## 1. Why Polygon (not raw FINRA)

Empirically probed 2026-06-03 (all with our existing keys):

| Source | Short *interest* (bi-monthly) | Short *volume* (daily) | Verdict |
|---|---|---|---|
| FINRA `api.finra.org` | OAuth account required (user must create) | — | ❌ blocks autonomy |
| FINRA `cdn.finra.org` | dir listing 403 | ✅ Reg SHO `CNMSshvol*.txt` (no auth) | volume only |
| FMP (our plan) | v4 = legacy/blocked, stable = 404 | — | ❌ not on plan |
| Finnhub (our key) | 403 (premium) | — | ❌ |
| **Polygon (our key)** | **✅ `/stocks/v1/short-interest`** (to 2017) | **✅ `/stocks/v1/short-volume`** | **✅ USE THIS** |

Polygon sources both from FINRA — same regulatory data, delivered through an API we already pay for, with `days_to_cover` precomputed. No raw-file parsing, no new account.

## 2. Endpoints & schema (verified)

**Short interest** — `GET https://api.polygon.io/stocks/v1/short-interest?ticker={SYM}&sort=settlement_date.asc&limit=...&apiKey=...`
Cursor pagination via `next_url`. Fields: `settlement_date, ticker, short_interest, avg_daily_volume, days_to_cover`. Bi-monthly (≈15th + month-end settlement). History to **2017-12-29**.
⚠️ **No dissemination/publication date field** — see §3.

**Short volume** — `GET https://api.polygon.io/stocks/v1/short-volume?ticker={SYM}&sort=date.asc&...`
Fields: `date, total_volume, short_volume, exempt_volume, non_exempt_volume, short_volume_ratio` (percent, e.g. 46.7), + per-venue breakdowns. Daily, current to T-1.

## 3. Point-in-time contract (THE leak-killer)

A short-interest value is measured at a **settlement date** but is **not public until FINRA disseminates it ≈8 business days later**. Polygon returns only `settlement_date`, so we compute a **conservative publication lag ourselves** and store an explicit `knowable_date`:

- **Short interest:** `knowable_date = settlement_date + SI_PUBLICATION_LAG_BDAYS` where **`SI_PUBLICATION_LAG_BDAYS = 10`** (actual ≈8; +10 is *always ≥* actual → never optimistic, only mildly stale).
- **Short volume:** Reg SHO daily files publish next morning → `knowable_date = date + SV_PUBLICATION_LAG_BDAYS`, **`SV_PUBLICATION_LAG_BDAYS = 1`**.

**Every accessor filters `knowable_date <= as_of`.** A settlement just before `as_of` but disseminated after it is correctly excluded. This is the single most important correctness property and is covered by a dedicated no-lookahead test.

> **On "why not a feed with no delay?"** — The ~8-bday gap is a **regulatory fact, not a vendor artifact**: FINRA does not make a short-interest figure public until ~8 bdays after settlement, so the value *does not exist* before then. Any feed that lets you act on the settlement date is selling a **lookahead leak**. The only legitimate upgrade over our conservative estimate is the **exact dissemination date** per record — which we can obtain **free** from FINRA's published settlement→dissemination *schedule* (deterministic, layer it on Polygon), not by switching vendors. FMP is not an option regardless: short interest is **not on our plan** (empirically 404 on `stable/short-interest`, `shares-short`, `short-volume`; v4 legacy-blocked). Defer the exact-calendar refinement until A2 goes live (where acting ~2 days sooner matters); for the CPCV verdict, conservative lag is honestly pessimistic and never overstates.

## 4. Storage (survivorship-safe)

Parquet under `data/` (matches `macro_history.parquet`, `sector_map.parquet`):
- `data/short_interest.parquet`: `ticker, settlement_date, knowable_date, short_interest, avg_daily_volume, days_to_cover`
- `data/short_volume.parquet`: `ticker, date, knowable_date, short_volume, total_volume, exempt_volume, short_volume_ratio`

Backfill iterates the **full historical universe** (incl. delisted) — stored data is never filtered to current membership, so a backtest as-of date sees exactly what was knowable then.

## 5. Accessors (`app/data/short_interest_provider.py`) — mirror `fmp_provider.get_*_at`

- `get_short_interest_at(symbol, as_of) -> dict | None` — most recent SI row with `knowable_date <= as_of`; returns `short_interest, days_to_cover, avg_daily_volume, si_change_pct` (vs prior SI), `settlement_date, knowable_date`.
- `get_short_volume_features_at(symbol, as_of, lookback_days=20) -> dict` — trailing short-volume-ratio `mean / last / zscore` over rows with `knowable_date <= as_of`.

## 6. Squeeze-edge signal candidates (for A2 event table — built later)

- **High days-to-cover** + **SI build** (`si_change_pct` rising) → squeeze fuel.
- **Short-volume-ratio spike / z-score** → acute shorting pressure (higher frequency → more events → more statistical power, which matters for an underpowered-edge program).
- Event = crowdedness threshold crossing; drift measured forward like PEAD.

## 7. Backfill

`scripts/backfill_short_interest.py` — `--incremental`, `--workers N`, `--sv-years N` (cap daily-volume depth; SI is cheap, SV is large). ASCII-safe console (Windows cp1252).
