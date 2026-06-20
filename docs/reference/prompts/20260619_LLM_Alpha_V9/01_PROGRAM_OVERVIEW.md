# 01 — Program Overview (MrTrader)

## What this is
**MrTrader** is a solo-operated, fully-systematic trading platform. It runs as an in-process
FastAPI orchestrator (PM / Risk-Manager / Trader agents on an APScheduler loop) against an
**Alpaca paper account (~$100k)**, with Postgres + Redis and a React dashboard. Everything is
rules-based and pre-registered; there is no discretionary trading.

**Two goals, in priority order:** (1) a *resilient* book that doesn't blow up; (2) *find and
harvest real alpha*. We would rather hold cash than deploy a fragile edge.

## What is LIVE right now (paper)
- **TSMOM trend sleeve** — 10 liquid ETFs (SPY QQQ IWM EFA EEM TLT IEF GLD DBC UUP), long-flat,
  inverse-vol, **weekly** rebalance, ~50% gross allocation. Standalone Sharpe ≈ 0.72 over 19y.
  **This is the only validated standalone edge we run.**
- **VIX-term crash governor** — an overlay that de-risks the trend book to 0.5× when VIX > VIX3M
  (backwardation). Fail-safe (can only reduce exposure).
- **Cash / T-bill sleeve** — parks idle settled cash (~50% of NAV) into SGOV/BIL to earn the
  risk-free rate. Excluded from risk caps.
- **Crypto-trend live-paper tracker** — REPORT-ONLY (no capital): records the out-of-sample
  record of a crypto-trend sleeve that passed paper but failed the book-improvement test.

Everything else has been **turned off or never promoted** (see `02_KILL_KEEP_LEDGER.md`).

## The journey (why we are where we are)
We have run ~7 research "campaigns," each adjudicated honestly. The blunt summary:

- **Alpha-v2 — cross-sectional ML ranking (stocks):** KILLED. Three honest purged-CV nulls; the
  apparent edge was market beta, not alpha.
- **Alpha-v4 — "portfolio of uncorrelated premia":** PEAD demoted to telemetry (it's conditional
  beta, market-hedged Sharpe −0.37); **TSMOM trend kept** as the one real sleeve; a sleeve
  allocator built but "complexity must earn it" → simple fixed weights.
- **Alpha-v5 — options program (VRP):** single-name earnings IV-crush KILLED (cost); index
  short-vol = real risk-premium but Sharpe-weak and *not alpha* → paused.
- **Alpha-v6 — "is our ruler broken?":** we discovered our gate had over-corrected into a
  **Type-II / false-negative machine** (a t≥2 bar on ~8 folds of ≤4y data kills true Sharpe-0.5
  edges). Recalibrated with positive/negative controls + a two-track acceptance gate.
- **Alpha-v7 — Ruler-v2 + a "premia sleeve lab":** built a less-Type-II two-track gate and a
  uniform sleeve pipeline. Swept calendar/overnight/ETF-RV/carry sleeves on free data → **none
  cleared the bar** ("trend is the only edge on free daily US data").
- **Alpha-v8 — overlays & timing:** credit-spread de-risk overlay = a weak candidate (flag-off,
  shadowing); aggregate short-interest KILLED for power; additive timing parked.
- **Alpha-v9 — multi-engine book (current):** validate-the-validator (pipeline is faithful) →
  monetize cash → crypto trend (paper-candidate) → defined-risk VRP (pending mature options
  data) → FINRA daily short-volume (real-but-weak) → **Norgate futures: CARRY is a genuinely
  new, modern, diversifying edge** (the first real new edge since trend) — see `03`.

## The honest one-paragraph state
After two years of rigorous search, **the only thing we trade live with conviction is
multi-asset trend (via ETFs).** Almost every equity/options "signal" we tested turned out to be
beta, a risk premium misframed as alpha, or cost-killed. The one genuinely promising *new* find
is **futures carry** (just discovered on newly-purchased Norgate data) — real and diversifying,
but not yet live and with honest caveats (roll cost, no real-fill validation yet). We want a
brutal outside read on whether we're (a) finding real alpha or fooling ourselves, (b) missing
obvious strategies/data, and (c) architecturally sound.
