# 04 — Data inventory: what we have, depth, cost, and gaps

*What's actually wired and usable today, with the honest depth/quality caveats — so your data
suggestions land on real constraints. (In-repo SSOT: `docs/reference/DATA_PROVIDERS.md`.)*

---

## What we have

| Source | Cost | What it gives us | Depth | Key caveats |
|---|---|---|---|---|
| **yfinance** | free | Daily + 5-min OHLCV (equities/ETFs); the TSMOM ETF universe; SPY; continuous futures (`=F`) | Daily: deep (decades). 5-min: shallow. Futures: shallow/dirty | **Survivorship bias** (no delisted names → flatters long/short & event studies). Variable depth/rate. Continuous futures have no roll schedule (carry untestable). |
| **Alpaca** | free (paper) | **Execution** (equities/ETFs/options/crypto), account/positions, live bars, market clock/calendar, easy-to-borrow, **free options NBBO snapshot** (`feed=indicative`) | live | **NO futures execution.** `indicative` NBBO may be delayed/synthesized (fine for spread structure, not HFT). $100k paper. |
| **Polygon** ("Massive") | was $79/mo, **sub ends 2026-06-17 → frozen** | Historical options OHLCV (4y local store 2022-06→2026-06, ~113M bars); contract reference (survivorship-safe incl. expired); FINRA short interest + short volume; stock financials/news | options 4y (frozen) | **No historical IV / greeks / OI / NBBO at our tier** — we *compute* IV+greeks (BS/BjS/CRR). After 06-17 the store is **frozen** (no new options bars). |
| **FMP** (Starter) | $29/mo | Earnings calendar/history (PEAD), fundamentals, analyst grades, insider (Form 4), economic calendar | varies | Must use `/stable/` only (legacy `/api/v3` deprecated → 403). |
| **FRED** | free | Macro time series → regime/context features | deep | Network-blocked in our run env (we cache to `macro_history.parquet`). |
| **`macro_history.parquet`** | derived | VIX, VIX3M, HYG, IEF, RSP, SPY (drives crash + credit overlays) | 2018+ | Local cache; 2018+ only. |
| **Finnhub** | free | Company news | — | Economic calendar is premium-only (inert on free). |
| **Alpha Vantage** | free | Fundamentals fallback | — | Rate-limited; fallback only. |

**Net data envelope:** deep free daily equity/ETF EOD + yields; shallow/dirty free futures; a frozen
4y options snapshot (no live IV/OI/NBBO history); bi-monthly short interest to 2017; cheap fundamentals
+ earnings + econ calendar; a 2018+ macro cache. **No tick data, no L2/order-book, no clean futures with
roll/carry, no live options surface history, no alt-data.**

---

## What we believe is missing (and our rough cost reads)
- **Clean futures w/ roll & carry** (e.g. **Norgate ~$270/yr**, or CSI). Unlocks futures trend + the
  *carry* premium we literally cannot test on free data. Needs a 2nd broker (IBKR) to actually trade —
  Alpaca has no futures.
- **Point-in-time fundamentals / estimates** without survivorship (Sharadar/Quandl-type, ~$ low-hundreds/
  yr). Our fundamentals path is thin and survivorship-prone.
- **Survivorship-free equity history with delistings** — critical for any honest long/short or event
  study (yfinance silently drops the losers).
- **Live/historical options surface** (IV/OI/NBBO history) — we only have a frozen 4y OHLCV snapshot and
  compute greeks; real surface data is expensive.
- **Alt-data** (positioning/COT, flows/ETF creation-redemption, sentiment, options dealer-gamma, retail
  flow). Mostly untouched; cost/quality varies wildly.
- **Crypto history** — we have Alpaca crypto execution but have barely pulled/used crypto data.

---

## The data question for the reviewer
Given a **solo/retail budget (~$100–300/yr is fine; flag institutional-only asks)**: what is the
**single highest-ROI data purchase**, and what's the cheapest way to get **survivorship-free** coverage
so our long/short and event research stops being quietly biased? Is clean futures+carry data (Norgate)
worth it, or is the higher-ROI buy elsewhere?
