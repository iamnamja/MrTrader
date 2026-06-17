# Data Providers — Plans, Coverage & Usage

**What every external data source is, what plan we're on, what it covers, what we use it for, and the gotchas.** Keep this current when a plan/key/endpoint changes (NO-DRIFT rule).

**Last verified:** 2026-06-10 (FMP legacy→/stable/ migration; live endpoint checks).

---

## Quick table

| Provider | Plan / cost | Key (`settings.*` / env) | We use it for | Status |
|---|---|---|---|---|
| **FMP** (Financial Modeling Prep) | **Starter $29/mo** | `fmp_api_key` | Earnings calendar/history (PEAD), fundamentals, analyst grades, insider (Form 4), **economic calendar** | ✅ working on `/stable/` |
| **Finnhub** | **Free tier** | `finnhub_api_key` | Company news | ✅ news; ❌ economic calendar (premium-only) |
| **Polygon ("Massive")** | **Options Developer — DOWNGRADED; sub ends 2026-06-17** | `polygon_api_key` | Options OHLCV flat files (4y store → **extended to 2026-06-11, owned locally**), contract reference, current snapshot (**OI + delayed only**), `adjusted=false` aggs, short interest/volume (FINRA), stock financials/news | ⬇️ ends 06-17; REST still 200; final backfill done (re-run before 06-17 for last days) |
| **Alpaca** | **Paper (free)** | `alpaca_*` | Order execution, account/positions, live bars, market clock/calendar, easy-to-borrow, **options snapshot NBBO** (free `indicative` feed → spread logger) | ✅ paper, $100k |
| **yfinance** | Free (no key) | — | Daily + 5-min equity/ETF bars; TSMOM ETF prices; SPY; intraday daily-feature provider | ✅; survivorship + depth caveats |
| **FRED** (St. Louis Fed) | Free | (public graph JSON) | Macro time series → `macro_calendar` context + regime features | ✅ |
| **Alpha Vantage** | Free (rate-limited) | `alphavantage_api_key` | Fundamentals fallback | ✅ fallback only |
| **FINRA Reg SHO** (daily short-volume) | **Free (no key)** | — (public CDN) | Daily off-exchange short-VOLUME per NMS name → aggregate short-sell-pressure timing signal (P3-5) | ✅ CDN serves ~2019-01-02→ |

Infra (not data feeds): **Redis** (queue/cache), **SQLite/Postgres** (app DB).

---

## FMP — Starter $29/mo  ⚠️ legacy API deprecated

- **API base: `https://financialmodelingprep.com/stable`** (the post-2025 "stable" API). Modules: `app/data/fmp_provider.py` (`_BASE=/stable`), `app/data/fmp_fundamentals.py`, `app/calendars/earnings.py`, `app/news/sources/fmp_source.py`.
- **⚠️ The legacy `/api/v3` API was DEPRECATED 2025-08-31.** Any `/api/v3` call now returns `403 "Legacy Endpoint : … only available for legacy users who had valid subscriptions prior August 31, 2025."` — for a non-legacy key this is true **even for trivial endpoints like `quote`**. This (not a missing paid tier) was the root cause of the recurring econ-calendar 403 spam. **Rule: never use `/api/v3` — only `/stable/`.**
- **What the Starter plan covers (verified live on our key, 2026-06-10):**
  - `/stable/economic-calendar` → 200 (events with estimate/prior/actual/impact). Used by `fmp_source.fetch_economic_calendar` + `premarket.py`.
  - `/stable/grades` (analyst grades) → 200. Used by `get_analyst_grades_fmp` (analyst-revision scorer).
  - `/stable/earning_calendar`, fundamentals, insider (`/stable/insider-trading/search`) — same base + key; PEAD's earnings features (`get_earnings_features_at`) ride this path.
- **Endpoint name gotcha:** stable uses **hyphens** (`economic-calendar`), legacy used underscores (`economic_calendar`).
- **403 handling:** `fmp_source` disables + logs ONCE on a 401/403 (so a future genuine access gap can't spam), reset on restart.

## Finnhub — Free tier

- **API base: `https://finnhub.io/api/v1`.** Module: `app/news/sources/finnhub_source.py`. Key: `finnhub_api_key` (or `finhub_api_key`).
- **Covered (free):** company news. **NOT covered (free):** `/calendar/economic` → `403 "You don't have access to this resource."` (premium-only). So Finnhub is **not** a working economic-calendar fallback on the free tier — we rely on FMP `/stable/` instead. Same disable-once-on-401/403 discipline.

## Polygon ("Massive") — Options Developer DOWNGRADED 2026-06-12

> **⚠️ DOWNGRADED 2026-06-12.** The $79/mo Options Developer plan (bulk **S3 flat-file** historical-options access — how the 4y store was built) was dropped after Alpha-v6 concluded options are not a tradeable edge (options-as-execution + options-as-signal H4a–e all KILLED) and the **full 4y store is owned locally** (`data/options_bars.parquet`, 1.2 GB, 2022-06→2026-06 + the derived greeks store + feature table). **Polygon = "Massive"** (rebrand; API host still `api.polygon.io`; not to be confused with FMP, which serves earnings/fundamentals, never options).
> **Post-downgrade live check (2026-06-12):** the `polygon_api_key` still returns **200** on news (`/v2/reference/news`), options snapshot/contracts, stock aggs (`adjusted=false`), and short interest — so **nothing in the live system is broken** (the live news monitor degrades gracefully to empty even on a future 403). **Subscription ends 2026-06-17** — S3 access remains until then (verified working 2026-06-12). **A final backfill ran 2026-06-12 → the store now extends to 2026-06-11** (113.3M bars); re-run `backfill_options --r1k --start 2026-06-12 --end 2026-06-16` once more before 06-17 to capture the last days. After 06-17 what's gone is **bulk S3 flat-file backfills** → the store is **frozen** (no new bars). Re-subscribe + re-backfill later if a new options idea (e.g. dormant P6 index-VRP) ever needs current data.

- **API: `https://api.polygon.io` + S3 flat files.** Modules: `app/data/options_provider.py`, `polygon_provider.py`, `polygon_financials.py`, `short_interest_provider.py`, `polygon_s3.py`.
- **Covers:** historical options OHLCV via S3 flat files (`us_options_opra/day_aggs_v1` — the 4y local store, 2022-06→2026-06), contract reference (incl. expired → survivorship-safe), **current snapshot = OI + DELAYED OHLCV/last-trade only**, FINRA short interest + short volume, stock financials/news. Also `/v2/aggs/.../adjusted=false` (used by the greeks backfill for as-traded closes).
- **Hard limits (no paid tier fixes these on Developer):** **no historical IV / greeks / open interest / NBBO quotes** — AND the **current snapshot serves NO NBBO / no `last_quote` / no IV / empty greeks** either (verified 2026-06-11: `/v3/quotes/{ticker}` → **HTTP 403 NOT_AUTHORIZED**; snapshot `last_trade.timeframe="DELAYED"`). So we COMPUTE IV+greeks (BS/BjS/CRR → persisted `data/options_greeks/`); liquidity via volume/notional; and **live NBBO comes from Alpaca, not Polygon** (see below). Data starts ~mid-2022. See `docs/reference/OPTIONS_DATA.md`.

## Alpaca — Paper (free)

- Module: `app/integrations/alpaca.py`. Mode **paper**, ~$100k. Used for order execution, account/positions (`get_account`), live bars (also via the Alpaca MCP), market clock/calendar (fail-closed for holidays), easy-to-borrow list. Live equity is the source of truth (the `config.initial_capital=100000` is just a constant; `risk.peak_equity` is a separate drawdown high-water mark).
- **Options snapshot NBBO (free) — `app/data/alpaca_options.py`:** `GET data.alpaca.markets/v1beta1/options/snapshots/{underlying}` with `feed=indicative` (free tier) returns `latestQuote` **bid/ask** + sizes + `dailyBar` + IV — the live NBBO the Polygon plan does NOT serve. Used by the nightly NBBO logger (`scripts/log_options_nbbo.py`, P1c) for spread calibration. Caveat: `indicative` may be delayed/synthesized — fine for nightly spread *structure*, recorded per-row via the `feed` column. (Real-time OPRA = paid `feed=opra`.)

## yfinance — free

- Daily + 5-min OHLCV for equities/ETFs; TSMOM ETF universe; SPY; the intraday daily-feature provider (`INTRADAY_DAILY_FEATURE_PROVIDER=yfinance`, full history — fixes the Alpaca ~100-bar cap, #343).
- **Caveats:** survivorship (no delisted-name bars → flattering for any long/short or event study that needs delisted losers); depth/rate variability on some paths.

## FRED — free

- `app/macro/fred_client.py` (`fred.stlouisfed.org/graph/fredgraph.json`). Macro series feeding the **free** `macro_calendar` (context + hardcoded FOMC/NFP) used by the regime model, RM, PM. This is the always-on macro layer, independent of FMP/Finnhub economic-calendar.

## Alpha Vantage — free (rate-limited)

- `app/ml/fundamental_fetcher.py` (`_AV_BASE`). Fundamentals fallback only; rate-limited free tier — not a primary path.

---

## Economic-calendar resolution (how the macro-event layer actually works)

- **Primary:** FMP `/stable/economic-calendar` (Starter plan) — ✅ working (fixed 2026-06-10).
- **Fallback:** Finnhub `/calendar/economic` — ❌ inert on free tier (premium-only); stays as code fallback only.
- **Always-on context (independent):** FRED-backed `macro_calendar` + hardcoded FOMC/NFP — feeds regime/RM/PM regardless of the above.
- **Consumer:** the PM's `_refresh_nis_on_macro_events` re-pulls the news/macro-sentiment cache ~3 min after a CPI/NFP/FOMC release; the news `intelligence_service` uses upcoming events for macro context.

## Free alternatives considered (if a vendor calendar ever lapses)

Official release schedules — **free, authoritative, PIT-safe** (published a year ahead): Fed (FOMC dates), BLS (NFP/CPI/PPI), BEA (GDP/PCE), Census (retail sales). FRED `/releases/dates` also approximates a calendar from series we already pull. Avoid scraping (investing.com/forexfactory — fragile + ToS).

## FINRA Reg SHO daily short-volume — Free (no key)

- **Source:** `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt` — the FREE consolidated (CNMS) daily short-sale-volume file. Pipe-delimited: `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`, ~8–12k NMS names/day. Module: `app/data/finra_short_volume.py` (downloads+caches incrementally → `data/finra_short_volume.parquet`: distilled daily aggregate short-vol ratio + per-symbol).
- **Coverage gotcha:** the CDN serves **~2019-01-02 → present** (~1,875 trading days as of 2026-06-17). Pre-2019 dates return an S3 `AccessDenied` XML stub (handled → `None`); weekends/holidays 404. This is ~10× the power of the bi-monthly NYSE/Nasdaq short-INTEREST that the Alpha-v8 G2 overlay was killed on.
- **Used for:** P3-5 aggregate short-sell-pressure timing signal (`app/research/short_volume.py`). Verdict 2026-06-17: signal real but a real-but-weak near-miss (not a standalone edge) → P3-4 composite component + post-Norgate XS. The aggregate ratio is survivorship-bias-free (a market-wide sum); per-name cross-sectional use needs survivorship-free names (Norgate).
- **PIT:** the day-t ratio is knowable after t's close → a t+1 signal.
