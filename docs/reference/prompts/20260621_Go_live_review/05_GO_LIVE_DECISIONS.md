# 05 — The go-live decisions (Theme A) — the layer we never built

We can validate sleeves; we have **never built the layer that sizes them, combines them across
brokers, and governs the book's risk.** IBKR futures execution is days away, so this is now the
binding question. Everything below is genuinely open — we want your design, not a rubber stamp.

## The book we'd actually run
- **Alpaca (equities/ETFs):** ETF-trend (LIVE, ~50% gross) + cash/T-bill sleeve (idle cash → RFR).
- **IBKR (futures):** futures-book (carry + xsmom) + VIX-VRP. **Not live yet.**
- So a live multi-premia book is **two brokers, two margin systems, two execution paths**, one
  combined risk surface. ~$100k paper notional today.

## A1 — Sizing (per-sleeve AND book-level)
Your panel said: **freeze gross; size by drawdown / vol-target / margin, NOT Kelly** (the 7.7×-Kelly
framing is dangerous). We agree in principle but have no concrete policy. Specifically:
- What **target book volatility** (or max-drawdown budget) should a solo $100k→? CTA-style book run
  at, and how do you translate that into per-sleeve risk weights?
- Equal-risk vs **risk-parity / HRP** vs inverse-vol across {trend, futures-book, VRP}? With only
  3-4 sleeves, does HRP earn its complexity, or is equal-risk-contribution the honest default?
- How do you set per-sleeve **vol targets** when one sleeve (trend) is live-proven and the others
  are paper? Should paper sleeves enter at a **fractional** risk weight that ramps with live evidence?
- Futures bring **leverage + margin** natively (notional ≫ cash). How should margin utilization
  cap the book independent of the vol target? What margin-to-equity ceiling for a solo book?

## A2 — Cross-venue risk aggregation (Alpaca + IBKR)
We have no unified risk view across two brokers. We need to design:
- A single **book-level risk surface** (aggregate gross/net, vol, drawdown, factor exposures)
  computed across both brokers daily — what's the minimal correct version?
- A **single kill-switch** that can flatten/halt *both* venues (today the kill-switch is app-level
  + per-broker). What's the right architecture so one trigger governs the whole book?
- Reconciliation discipline when "the DB is not reality" across two brokers with different
  position/cash/margin semantics.

## A3 — Forward-looking book risk (beyond the per-sleeve VIX governor)
- A **realized-correlation-spike de-gross trigger** (cheap, high-value per your panel): when
  cross-sleeve realized correlation jumps, cut gross. How would you define + calibrate it without
  overfitting (what window, what threshold, how much de-gross)?
- A **global drawdown-based de-risk** ladder on the whole book (not just the trend sleeve). What
  drawdown breakpoints → what gross reduction, and how to avoid selling the bottom?

## A4 — The promotion ladder (paper → tiny-live → scale)
This is the crux and we have only a sketch. We want an explicit ladder with **evidence thresholds**:
- **Rung 0 → 1 (paper PASS → IBKR paper):** carry/xsmom/VRP are Track-A PAPER-PASS today. What
  must a sleeve show in **IBKR paper** (real fills, real roll, ≥1 vol spike) before it earns capital?
  For how long / how many rebalances / how many rolls?
- **Rung 1 → 2 (tiny live):** at what evidence do you put *real* (small) capital on? How tiny
  (1-2 contracts/market)? What's the explicit **stop** (a tracking-error / slippage / drawdown
  breach that demotes it back to paper)?
- **Rung 2 → 3 (scale):** what live track length + what live-vs-backtest consistency check gates
  scaling? How do you avoid the classic trap of scaling right before mean-reversion?
- How should **live-paper OOS divergence** from the backtest be tested (we have an intended-vs-actual
  tracking-error instrument for trend; what's the analog for futures)?

## A5 — "What would make you NOT deploy?"
Give us the explicit **no-go list** — the conditions under which you'd keep this on paper
indefinitely rather than risk capital (e.g., "if the t=2.29 doesn't survive the null zoo," "if
the 4 premia tail-correlate above X," "if IBKR paper slippage exceeds Y% of the edge"). We want
the disqualifiers stated as hard gates, not vibes.

## Constraints / facts to design within
- Solo operator; ~$100k paper; modest capital scaling realistic, not institutional size.
- Norgate futures data is **EOD-only** (no intraday); all sleeves are **weekly**. So this is a
  low-frequency book — favorably, slippage/latency are second-order vs sizing/risk.
- We already have: a VIX-term governor, a cash sleeve, an intended-vs-actual tracking instrument,
  the Ruler-v2 gate. Reuse where sensible; tell us what's missing.
