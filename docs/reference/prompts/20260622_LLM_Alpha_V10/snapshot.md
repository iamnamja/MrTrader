# Snapshot — MrTrader system state (2026-06-22, for external review)

## 1. What it is
A solo, fully-systematic trading program. In-process FastAPI orchestrator (PM / RiskManager / Trader
agents + APScheduler), Postgres (truth) + Redis (ephemeral), React dashboard. Trades **Alpaca paper**
(~$100k NAV, US equities / ETFs / spot crypto). **IBKR futures account applied 2026-06-21, approval
pending.** Everything is rules-based + walk-forward-validated; no discretionary trading.

## 2. What's LIVE now
- **ETF trend sleeve** — multi-asset time-series momentum (TSMOM). Universe: SPY, QQQ, IWM, EFA, EEM,
  TLT, IEF, GLD, DBC, UUP. Signal = equal-weight ensemble of `sign(P_t / P_{t-L} − 1)` over lookbacks
  {21, 63, 126, 252} (no skip-month); inverse-realized-vol sizing to a 10% per-instrument vol target;
  long-flat (no shorts); weekly rebalance; ~2 bps/side modeled cost. **Allocation = 50% of NAV**
  (Kelly-haircut; standalone 100%-gross book is Sharpe 0.72 / ann vol 9.34% on 19.4y history → ~4.7%
  vol at the live allocation). Post-2015 trend Sharpe +0.77 (2020s +1.05) — NOT decayed.
- **Cash sleeve** — idle settled cash parked in T-bill ETFs (SGOV/BIL) for the risk-free rate; just
  enabled (first deploy 2026-06-22).
- **VIX crash governor** (LIVE overlay) — de-risks the trend book 1.0→0.5 when VIX > VIX3M (backwardation).
- Crypto trend runs in **paper-only** (not capital).

## 3. What's PAPER (validated, not yet capital — gated on IBKR)
- **Futures carry** (term-structure) — roll-honest Sharpe 0.58, post-2015 +0.89.
- **Futures cross-sectional 12-1 momentum (xsmom)** — Sharpe 0.56, corr-to-trend 0.12; survived a
  selection-aware max-of-6 null (p 0.005).
- **Futures book = equal-weight(carry, xsmom)** — Sharpe 0.67; residual-α t = 2.29 vs the live trend
  book (the "second engine"). DSR(N≈25 families) ≈ borderline → "size modestly."
- **VRP via the VIX-futures curve** (short front VIX in contango, gated by the crash governor) — Sharpe
  0.64; **DROPPED from the initial live book** per tail diagnostics (most tail-concentrating, loses 6/8
  crises).
- **Crypto trend** (TSMOM on Alpaca spot) — Sharpe 0.64, corr-to-trend 0.18; CAPITAL fail on history.

## 4. The acceptance gate (the bar everything must clear)
Two-track, walk-forward, CPCV-validated, family-wise multiple-testing corrected:
- **Track-A (standalone):** CPCV mean Sharpe, path-t-stat, %-positive, worst-path, DSR — must clear a
  paper-pass bar.
- **Track-B (marginal):** budget-invariant **residual-alpha contribution vs the existing book** —
  appraisal IR, P(ΔSR>0) ≥ ~0.875, **corr-to-book < ~0.30**, tail-overlap. *This is the real gate.* A
  strategy can be excellent standalone and still PARK if it's redundant.
- **Multiple-testing:** an auditable registry enumerates **~26 distinct strategy families searched**;
  this N feeds a deflated-Sharpe cross-check. We kill at the pre-registered sign (no flipping).

## 5. The KILL LIST (what's been tried and rejected — don't re-suggest without a new angle)
- **Equity ML cross-sectional ranker** — IC ≈ 0 across THREE independent builds; a positive-control
  proved the feature→label pipeline is faithful, so the null is the market, not a bug.
- **PEAD (earnings drift)** — real-but-beta; event-level t = −0.77 (p 0.78) → demoted, off live.
- **Short-term reversal** — real gross (+0.40) but cost-dead (159×/yr turnover, −0.90 net at 10 bps).
- **Overnight (close→open)** — real gross (+0.53) cost-killed net.
- **Short-interest XS, turn-of-month, ETF relative-value** — killed (no edge / timed-beta / reversed).
- **Options-as-signal (CPIV/skew/OI/term/IVRV)** — 5 hypotheses, all killed; single-name option spreads
  ~33% half-spread (brutal).
- **Futures factor zoo** — curve-momentum, value, skewness, basis-momentum, CFTC CoT all KILLED at the
  pre-registered sign; only carry + xsmom survived. Futures *trend* decayed post-2015 (killed).
- **Sector-ETF relative-strength rotation** (2026-06-22) — standalone CPCV Sharpe 0.86 but Track-B FAIL
  (corr 0.51 to trend) → PARKED (redundant). Lesson: gate the marginal contribution, not the standalone.
- **Credit / curve / short-interest overlays** — credit is a marginal CANDIDATE (flag off, multiplicity
  caveat); curve + short-interest killed.

## 6. The safety / risk architecture (and its TRUE status)
A "portfolio-brain" safety substrate is being built in stages (R0 minimum-viable-safety → IBKR → unified
Constructor). **Honest status:**
- **BUILT + measuring (read-only / shadow):** consolidated cross-venue book-state + netted factor
  exposure; a whole-book risk gate wired into the trend rebalance in **shadow** (logs what it *would*
  block, blocks nothing); a frozen risk-policy-v1 (vol target ~6–8%, drawdown de-gross ladder
  −8/−12/−16/−20% → 0.75/0.50/0.25/0.0, per-venue margin reserve, notional caps); daemon-decouple
  capability (default in-process).
- **BUILT but NOT WIRED to the live order path (key gap):** the new broker-vs-DB reconciliation
  (fail-closed) and the kill-switch *state machine* are imported only by tests/scripts — the live path
  still uses a startup-only reconciler + a legacy kill flag. **The drawdown ladder + vol-targeting are
  applied only in offline research, not on any live order.** So today the live book's only enforced
  book-level controls are the VIX governor + an 80% gross cap.
- **Plan:** flip the whole-book gate shadow→enforce (~2026-06-29 after a clean shadow week); wire
  reconciliation + kill-switch; then IBKR paper→tiny-live. The roadmap names the cross-venue risk
  surface + kill-switch as the **hard no-go gate before any IBKR capital** (correctly NOT yet met).

## 7. Data owned / available
- **Free / owned:** ~19y liquid ETF + large-cap daily history (survivorship-clean for ETFs); FRED macro;
  VIX-curve; FINRA short data; **Norgate FUTURES mirror** (~100+ markets incl. G10 FX 6E/6J/6B/6A/6C/6S,
  rates ZT→UB, energy CL/NG/RB/HO/BRN, metals GC/SI/HG, equity ES/NQ/RTY, ags, VX; GC to 1978).
- **Paid feeds:** FMP Starter ($29, fundamentals); Polygon Options ($79, no hist IV/OI/NBBO); Alpaca
  paper; a self-built Alpaca options-NBBO forward log (immature, ~days).
- **NOT bought (the standing decision):** **Norgate US Stocks (Platinum, ~$693/yr)** — delisting-
  inclusive / survivorship-bias-free single-name + delisted-ETF history. A "justification accumulator"
  is building: it would unlock (a) survivorship-free single-name momentum, (b) thematic/country ETF
  rotation, (c) un-biased PEAD/relative-value re-tests. Deferred until the stack of reasons clears the cost.

## 8. The honest meta-picture (from prior external + internal panels)
"The app is ahead of the alpha." Excellent process; the book is a single (trend+cash) bet with a
promising-but-paper second engine (futures carry+xsmom). The binding constraints are **capital + live
track record + not blowing up when IBKR capital is wired** — arguably more than finding a 5th sleeve.
