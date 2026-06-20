# 04 — Data Inventory (what we own, coverage, limitations)

We want your read on whether we're under-using what we have, and what data we should buy next.

## What we have

| Source | Cost | Coverage | What it's for | Key limitation |
|---|---|---|---|---|
| **yfinance** | free | daily + 5-min equities/ETFs, deep history | ETF trend universe, SPY, equity features, intraday features | **Survivorship-biased** (current listings only); adjusted-data quirks |
| **Polygon options (flat files)** | was $79/mo; sub ended, **files OWNED locally** | **4 years** (2022-06 → 2026-06), ~112.8M bars, 733 names, ~6.18M contracts; **as-traded (adjusted=false)** | Backtests, options research | Only **4y** (≤ what the Developer plan served); **no historical IV/OI/NBBO** — we compute greeks ourselves |
| **Computed greeks store** | (derived) | full 4y store, per-contract IV/Δ/Γ/V/Θ | options-as-signal, VRP | European-warm-start American refine; as-traded inputs |
| **Alpaca options NBBO logger** | free (`indicative`) | **accruing daily since ~2026-06** (forward only) | the real spread surface for a future VRP go/no-go | Only ~weeks of history so far; needs ~months to span a test window |
| **FINRA Reg SHO daily short-volume** | free | **2019-01 → present** (~1,875 days, ~10k names/day) | aggregate short-sell-pressure timing | CDN only serves ~2019+; cross-sectional use needs survivorship-free names |
| **Norgate futures ("Silver")** | ~$297/yr | **105 markets, 30+ yr, survivorship-free**, continuous (back-adj + unadj) + full term structure | futures trend + **carry** | **EOD only** (no intraday/real-time); **license-gated** (local mirror, can't redistribute); we mirror it locally |
| **FRED macro** | free | long macro series | regime features, macro calendar | — |
| **FMP (Starter $29, /stable/ only)** | $29/mo | earnings calendar/history, fundamentals, analyst grades, insider, econ calendar | PEAD features, fundamentals | legacy /api/v3 is dead |

## Deliberately NOT bought (yet)
- **Norgate US Stocks (Platinum, ~$693/yr)** — survivorship-free US equities + delisted +
  historical index constituents. Would let us re-test the contaminated equity kills (PEAD, stat-arb)
  on clean data and run survivorship-free cross-sectional equity work. **Deferred on cost** (it
  re-tests things that already looked dead). *Is this the wrong call?*
- **Intraday/tick futures or equities** — would enable faster strategies, but our edges are slow
  (trend/carry are weekly/EOD) and we proved turnover kills (overnight). Not obviously worth it.
- **Live options NBBO history** (>$300/yr) — the missing piece for trustworthy options VRP
  backtests (our 4y frozen file + computed greeks is exploratory-only).

## The data gaps we're aware of
1. **Equities are survivorship-biased** (yfinance) — so any cross-sectional equity backtest
   (incl. the FINRA short-volume XS, PEAD redo) is contaminated until we buy Norgate stocks.
2. **Options history is short (4y) + no real NBBO** — so the VRP engine is blocked on the
   forward NBBO log maturing.
3. **Futures are EOD + license-locked** — fine for slow strategies; no real fills until IBKR.

## Questions for you on data
- Given equities + 4y options + survivorship-free futures, **what's the highest-EV next data buy**
  (Norgate stocks? live options NBBO? something else entirely — credit, FX spot, vol surface,
  alt-data)?
- Are we leaving obvious alpha on the table by NOT having survivorship-free equities?
- Is there a strategy family our existing data already supports that we simply haven't tried?
