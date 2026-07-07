# 01 — System state, architecture, and data (self-contained context)

## The trader & the account
- **Solo retail systematic trader.** Full custom stack: FastAPI backend + React dashboard, Postgres +
  Redis, an agent brain (PM / RM / Trader). ~$100k **paper** account.
- **Venues:** Alpaca (live paper equities/ETFs — the operational venue) + an IBKR paper account (futures;
  a migration to consolidate the whole book onto IBKR was built but futures-live is now gated — see 02).
- **Discipline:** shadow-first (every new order path logs what it *would* do before it can trade); a
  safety layer (DB↔broker reconciliation + a whole-book risk gate) is **live in enforce**; a state-machine
  kill-switch runs in shadow. "A missed rebalance, never a bad trade."

## The LIVE book (what actually trades — all paper)
| Sleeve | What it is | Status / evidence |
|---|---|---|
| **ETF trend** | Cross-sectional/absolute TSMOM over a liquid ETF universe (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP). Weekly (Monday) rebalance. | **The one validated edge.** Post-2015 Sharpe ~+0.77 (2020s +1.05). A risk premium, not alpha. ~50% trend budget. |
| **Cash / T-bill** | Idle capital → SGOV (+ a cash-ETF universe). Buffer-band rebalance. | LIVE; utilization sleeve, not an alpha source. |
| **Crypto trend** | Spot crypto TSMOM. | PAPER-tracked only, not live capital. |
| **VIX-term crash governor** | An overlay (see below). | LIVE overlay. |

Everything else (swing ML ranker, PEAD, intraday ML, the futures book, VRP) is **off / dormant / research**
— see the kill-list in file 02.

## The agent architecture (PM / RM / Trader)
- **PM (Portfolio Manager):** generates proposals / target exposures (weekly trend + cash; the swing ML
  ranker is dead & off; PEAD flipped off). Reads a regime context and macro-intel for entry sizing.
- **RM (Risk Manager):** correlation/beta/heat/exposure checks — **BUT the live trend & cash sleeves
  BYPASS the RM**, routing direct-to-broker behind a lightweight, now-enforced whole-book risk gate
  (gross / net-beta / notional / unmapped-symbol caps, fail-closed). So the RM's holistic checks only
  cover the (dead) proposal-driven ML path. **This is a known architectural gap.**
- **Trader:** executes, reconciles broker↔DB, manages exits/stops for the proposal path.
- **Cadence:** weekly rebalance (trend Monday, cash buffer-band); an intraday path exists but its ML is
  dead. **There is no condition-*triggered* rebalance — it's calendar-driven.**

## The REACTIVE infrastructure that ALREADY EXISTS (important — they are NOT starting from zero)
- **Regime detection:** `regime_detector` / a trained `regime_model` (used by the PM for entry sizing, by
  the Trader for exit tightening, and in the dashboard). Today it drives **sizing tweaks, not strategy
  selection.**
- **De-risk governors (all reduce-only multipliers on the trend book):**
  - **VIX-term crash governor** — HYG-independent VIX / VIX3M term-structure signal; cuts gross in
    backwardation/stress (LIVE).
  - **Credit governor** — HYG/IEF total-return ratio (credit stress) → de-risk multiplier (candidate, off).
  - **Curve governor** — 10y−3m term spread inversion → de-risk (parked, no benefit).
  - **Drawdown ladder** — account-HWM drawdown → −8/−12/−16/−20% → ×0.75/0.50/0.25/0.00 (shadow; the
    −20% rung flattens the book).
  - **Book-vol targeting / inverse-vol sizing** inside the sleeves.
- **A dormant regime-aware sleeve allocator** — CAN compute regime-tilted weekly sleeve weights; **default
  OFF** because on only 2 live sleeves, equal-weight beats vol/regime schemes.
- **Macro-intel workflow** — a graded macro-risk factor (NIS) that can size down entries / tighten exits
  on a digested-adverse macro read (flags on, paper).

**Net:** reactivity today = **de-risking only** (cut exposure in stress) + a *dormant* regime allocator.
Nothing opportunistically **rotates into a different strategy** in a different regime — because there is
only **one** return strategy to rotate between.

## Data feeds — what we HAVE
| Provider | Cost | Used for |
|---|---|---|
| **yfinance** | free | daily + 5-min equity/ETF bars, TSMOM prices, SPY (survivorship + depth caveats) |
| **FRED** | free | macro time series → macro calendar + **regime features** |
| **Norgate** (futures) | ~$297/yr | survivorship-free futures: 105 markets, continuous (back-adj + unadj) + full term structure |
| **FMP** | $29/mo | earnings calendar/history (PEAD), fundamentals, analyst grades, insider, econ calendar |
| **Finnhub** | free | company news |
| **Polygon** (options) | **downgraded, ending 2026-06-17** | options OHLCV (4y local store), OI + delayed snapshot, short interest/volume — **no live IV/NBBO history** |
| **Alpaca** | paper (free) | execution, positions, live bars, market clock, **options snapshot NBBO** (indicative) |
| **FINRA Reg SHO** | free | daily off-exchange short-VOLUME per name |
| **Alpha Vantage** | free | fundamentals fallback |

## Data we DON'T have (candidate gaps)
- **Real options *positioning* / dealer gamma / vol-of-vol** (we have VIX-curve + delayed OI only; no
  clean historical IV surface or NBBO history — Polygon options is ending).
- **Cross-asset regime signals as first-class inputs** beyond VIX/credit (real yields, the dollar, curve
  shape, cross-asset correlation/breadth).
- **Flow / order-book / microstructure** (CoT was tried and killed).
- **Alt-data** (sentiment beyond news, satellite, card, etc. — expensive, unproven for us).
- Proven caveat: **more data has repeatedly NOT produced edge** (CoT, options factors, short interest all
  killed) — data has only helped when it fed a mechanism (macro governors; PEAD's earnings data).
