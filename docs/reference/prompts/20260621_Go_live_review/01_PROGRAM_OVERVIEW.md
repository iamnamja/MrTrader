# 01 — Program Overview (MrTrader)

## What this is
**MrTrader** is a solo-operated, fully-systematic trading platform. In-process FastAPI
orchestrator (PM / Risk-Manager / Trader agents on an APScheduler loop), **Alpaca paper account
(~$100k)**, Postgres + Redis, React dashboard. Everything is rules-based and pre-registered; no
discretionary trading.

**Two goals, in priority order:** (1) a *resilient* book that doesn't blow up; (2) *find and
harvest real risk premia / alpha.* We would rather hold cash than deploy a fragile edge.

## What is LIVE right now (paper, on Alpaca = equities/ETFs only)
- **TSMOM trend sleeve** — 10 liquid ETFs (SPY QQQ IWM EFA EEM TLT IEF GLD DBC UUP), long-flat,
  inverse-vol, **weekly**, ~50% gross. Standalone Sharpe ≈ 0.72 (19y). **The only validated
  standalone edge we trade.**
- **VIX-term crash governor** — overlay de-risking trend to 0.5× when VIX > VIX3M. Fail-safe
  (can only reduce exposure).
- **Cash / T-bill sleeve** — parks idle settled cash (~50% of NAV) in SGOV/BIL for the risk-free
  rate; excluded from risk caps.
- **Crypto-trend live-paper tracker** — REPORT-ONLY (no capital): OOS record of a crypto-trend
  sleeve that passed paper but failed the book-improvement test.

## The asset/venue map (important for the go-live questions)
- **Alpaca** — equities/ETFs + spot crypto. Where trend + cash live today. **No futures.**
- **IBKR** — futures execution, **account applied 2026-06-21, approval imminent.** This is where
  carry / xs-momentum / VIX-VRP would run. So a live multi-premia book is **two brokers**.

## The 2-year journey (why we're here), in one breath
~7 research campaigns, each adjudicated honestly: cross-sectional equity ML (KILLED — beta),
"portfolio of uncorrelated premia" (PEAD demoted; trend kept), options VRP (paused), "is our
ruler broken?" (it was a Type-II machine → recalibrated with controls), Ruler-v2 + sleeve lab
(nothing new cleared on *free equity* data), overlays/timing (credit overlay weak; short-interest
killed for power), and the multi-engine book (cash monetized; crypto = paper-candidate; **Norgate
futures → carry + xs-momentum = a real second engine**; VIX-curve VRP reversed the VRP park).

## The honest one-paragraph state (now)
We have a **vindicated trend premium** (live, stronger post-2015), a **validated second engine**
(futures carry + xs-momentum, which improves the book at residual-α t = 2.29), and a **reversed
VRP** (VIX-curve short-vol, gated, survives crises). The free factor zoo is **exhausted** (only
xs-momentum survived of 6 tested). The next move is **not a new strategy** — it's crossing from
paper to real capital correctly: sizing, two-venue risk aggregation, an honest family-wise
multiple-testing reckoning, and a true diversification/tail check. That layer is **unbuilt**, and
that is what we want your panel to adjudicate.
