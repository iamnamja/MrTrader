# 03 — The book & strategies (what the brain must coordinate)

## Capital & venues
- Solo operator; **~$100k paper** today (Alpaca), modest scaling realistic — not institutional size.
- **Two venues** once IBKR is approved (days away):
  - **Alpaca** — equities/ETFs + spot crypto. Cash-settled, no native leverage (margin available).
  - **IBKR** — **futures** (the carry/xsmom/VRP sleeves). Margined: notional ≫ posted cash; the
    rest of the account sits in T-bills earning the risk-free rate while futures run on top.
- This is a **low-frequency book**: every strategy is **weekly** (Norgate futures data is EOD-only).
  So latency/microstructure are second-order; **sizing, netting, and risk coordination are the
  first-order problems.**

## The strategies the brain must run as one book
| Strategy | Status | Venue | Nature | Cadence | Notes |
|---|---|---|---|---|---|
| **ETF trend (TSMOM)** | **LIVE** | Alpaca | rules-based, long-flat, inverse-vol, 10 macro ETFs | weekly | the one validated standalone edge; ~50% gross; VIX-term crash governor overlay |
| **Cash / T-bills** | **LIVE** | Alpaca | sweep idle settled cash → SGOV/BIL | weekly | cash-equivalent; excluded from gross cap |
| **Futures carry** | paper | IBKR | rules-based, cross-sectional term-structure carry, 76 mkts | weekly | part of the "futures book" |
| **Futures xs-momentum** | paper | IBKR | rules-based, 12-1 cross-sectional momentum, 76 mkts | weekly | part of the "futures book" |
| **VIX-curve VRP** | paper | IBKR | rules-based short-front-VIX-future in contango, gated | weekly | short-vol; crash-gated |
| **Crypto trend** | paper (report-only) | Alpaca | rules-based TSMOM on 10 spot pairs | weekly | low-corr candidate; no capital |
| swing / intraday / PEAD (ML) | **OFF** | Alpaca | ML-ranked proposals (the PM/RM/Trader pipeline) | daily/intraday | killed; infra remains, may or may not return |

**The likely live book:** ETF-trend (Alpaca) + futures-book {carry, xsmom} (IBKR) + VIX-VRP (IBKR)
+ cash. So **3–4 risk-taking streams across 2 venues**, all weekly, all rules-based.

## Why these specifically need *coordinated* (not siloed) handling
1. **Stacked beta across venues.** ETF-trend can be long SPY/QQQ (Alpaca) while futures-xsmom/carry
   are long ES/NQ (IBKR) — that's concentrated equity beta the current design **cannot see** (the
   sleeves don't share exposure state, and the RM doesn't gate the live sleeves at all).
2. **Correlated drawdowns.** Post-2015 the streams' pairwise correlations average ~0.49 (higher than
   the headline low numbers) — the diversification benefit is real but modest, and likely **worse in
   the tail**. A book that sizes each sleeve independently mis-states total risk exactly when it
   matters.
3. **Shared risk budget.** There is one pool of capital and one drawdown tolerance. Each sleeve
   vol-targeting itself does not add up to a controlled **book** vol/drawdown.
4. **Netting / conflict.** Two sleeves can want offsetting or overlapping positions in correlated
   instruments; today they'd each trade and pay the spread, with no netting.
5. **One kill-switch, two brokers.** A risk event must be able to flatten/halt the **whole** book
   across venues from one control.

## What the brain must know at decision time (the consolidated state)
- Every **open** position across both venues (symbol, qty, notional, venue, owning strategy).
- Everything **pending / being considered** this cycle (all sleeves' intended target weights before
  any orders go out).
- **Aggregate exposures**: gross/net, by asset class, and by **factor** (equity beta, rates
  duration, USD, commodity, vol) — netted across strategies and venues.
- **Risk budget state**: book vol vs target, drawdown vs limit, margin utilization (IBKR), realized
  cross-strategy correlation.
- This consolidated object **does not exist today** — building it is step one.

## The sizing decision that's currently unmade
The brain must decide, each weekly cycle: given the validated strategies' signals, **how much risk
does each get, and how much does the book run in total** — by drawdown / vol-target / margin, *not*
Kelly, and **correlation-aware** so total gross falls when the streams converge. (The strategy panel
is separately advising on the sizing *policy*; this panel is about the *architecture* that
implements and enforces it.)
