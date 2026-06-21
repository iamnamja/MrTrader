# 04 — The gaps (where the silo design breaks down)

A concrete inventory of what's missing, so the design targets real problems — not strawmen.

## G1 — Strategies are independent silos
Each live sleeve (trend, cash) is its own scheduled rebalancer that computes targets and trades on
its own. The validated futures sleeves would be added the same way. There is **no component that
sees all strategies at once** and decides the book.

## G2 — The risk gate doesn't cover what's live
The RiskManager's holistic checks (correlation ≤0.75 vs open, beta ≤1.30, sector ≤20%, heat ≤6%)
apply **only to the proposal-driven path (swing/PEAD/intraday) — which is currently OFF**. The
**live** sleeves (trend, cash) **bypass the RM** and use a lightweight per-order gate (fat-finger +
a shared 80% gross cap). So the sophisticated risk logic we built **does not gate the money that's
actually trading.**

## G3 — No consolidated, real-time book state
"What's open" is reconstructable (DB `trades` ∪ Alpaca positions). "What's being considered" is
**scattered** across a Redis queue, a DB table, and in-memory dicts. There is **no single object**
that represents the whole book — open + pending + aggregate exposures — at decision time. The only
book-level view is the **EOD** `risk_metrics` snapshot, which is after-the-fact.

## G4 — Sizing is per-sleeve, not joint or correlation-aware
Each sleeve vol-targets itself. Nothing allocates a **single book risk budget** across sleeves, and
nothing reduces total gross when the strategies' realized correlations rise. The book's true
volatility/drawdown is an emergent accident, not a controlled target.

## G5 — No netting / conflict resolution across strategies
If two strategies want overlapping or offsetting exposure (e.g., trend long SPY-beta + futures long
ES), they each trade independently — paying spread twice, and concentrating exposure no one is
watching. No cross-strategy netting before orders go out.

## G6 — No factor-level exposure view
Risk is checked by symbol/sector/pairwise-correlation, never by **netted factor exposure** (equity
beta, rates duration, USD, commodity, vol) across the whole book and both venues. Stacked beta is
invisible.

## G7 — No broker abstraction; cross-venue is unmodeled
Alpaca is wired directly. IBKR would be a parallel silo with ad-hoc per-sleeve routing. There is no
unified position/cash/margin view across venues, and **no single kill-switch** that can halt/flatten
both. Reconciliation ("the DB is not reality") is per-broker.

## G8 — The agents reason at the wrong granularity
PM proposes per-symbol; RM approves per-trade against a snapshot; Trader executes per-order. **No
one reasons at the book level** ("given everything open and everything being considered across both
venues, what should the whole book look like, and what's the minimal set of trades to get there?").

## G9 — Coupling & operational fragility
The scheduler + execution run **inside the FastAPI web process**. A web issue can affect trading;
trading load can affect the dashboard. State lives in three places (Postgres, Redis, in-memory
dicts) with the in-memory dicts as a fast-path that must be rebuilt on restart (handled via
reconciliation, but it's a coupling/complexity risk). The external strategy panel (and our own review)
flagged decoupling execution from the web server as mandatory before live futures capital.

## G10 — Extensibility is by-copy, not declarative
Adding a sleeve today means writing another bespoke rebalancer + wiring a scheduler job + tagging
positions + teaching the reconciler/Trader to skip them. Adding a venue means a new client + ad-hoc
routing. There is no declarative "register a strategy / register a venue" seam — so the book gets
more siloed with each addition, not more coordinated.

## The through-line
The system was **deliberately** built sleeve-by-sleeve ("validate each edge alone; don't build the
portfolio layer until ≥3-4 pass; complexity must earn it"). That discipline was correct for the
research phase. We now have enough validated streams that the **absence of a coordination layer is
itself the biggest risk** — and the imminent second venue (IBKR) forces the issue. This review is to
design that layer (or the re-architecture that subsumes it) properly, once.
