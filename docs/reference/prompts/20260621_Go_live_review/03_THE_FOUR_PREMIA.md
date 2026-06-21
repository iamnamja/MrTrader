# 03 — The four risk premia (numbers + correlations)

All numbers net-of-modeled-cost, daily returns, our own backtests. **Challenge them.** Trend is
LIVE (Alpaca); the other three are PAPER and would run on IBKR futures.

## 1) TSMOM trend — LIVE (the validated edge)
- 10 ETFs (equity US/intl, rates, gold, broad commodity, USD); ensemble sign of trailing
  (21,63,126,252)d returns; inverse-vol; long-flat; **weekly**; per-instrument vol target 10%.
- **Standalone Sharpe ≈ 0.72** (2007-2026). **Vindicated post-2015: +0.77; 2020s +1.05.** maxDD ≈ −14%.
- Live at ~50% gross (a vol-target analysis said 25% badly under-deployed the only edge; full
  Kelly ≈ 7.7× gross, so 50% is deeply haircut). VIX governor on top.

## 2) Futures CARRY — PAPER
- Norgate, survivorship-free, **76 liquid markets**, 30+ yr. Cross-sectional term-structure carry
  `(P_front − P_next)/P_next ÷ Δt` from the nearest two contracts (scheduled expiry from the
  contract code). Weekly long-backwardated / short-contango, inverse-vol, book-vol-targeted.
- **Standalone Sharpe ≈ 0.58** (honest, after transaction-only roll cost). Positive in EVERY
  sub-period incl. **post-2015 +0.81**; signs economically correct; cost-robust. HAC p 0.0001.
- corr-to-ETF-trend ≈ **0.25**; Track-A PAPER-PASS (point-SR 0.71); **as a single sleeve its
  Track-B residual-α t ≈ 1.76** (marginal). Partly an energy/VIX bet (ex-energy ~0.54).

## 3) Futures CROSS-SECTIONAL MOMENTUM (`xsmom`) — PAPER
- Same engine/universe; 12-1 cross-sectional momentum (rank by trailing 12m-ex-1m return),
  inverse-vol, weekly.
- **Standalone Sharpe ≈ 0.56** (post-2015 0.58). corr-to-ETF-trend ≈ **0.12** (notably low — the
  *cross-sectional* selection is largely orthogonal to the *time-series* ETF trend). Track-A
  PAPER-PASS (point-SR 0.72); single-sleeve residual-α t ≈ 1.60.

## 4) The FUTURES BOOK = equal-weight(carry, xsmom) — PAPER (**the second engine**)
- corr(carry, xsmom) is low (near-orthogonal factors). Equal-risk combine.
- **Book Sharpe ≈ 0.67** (post-2015 0.83). Track-A PAPER-PASS (**point-SR 0.85 — strongest sleeve
  we've produced**).
- **Track-B vs the live ETF-trend book: residual-α t = 2.29 (SIGNIFICANT > 1.96), resid-Sharpe
  0.56, beta 0.24.** Two marginal factors (t 1.76, 1.60) combine into a book that crosses
  conventional significance. **This is the result we'd be sizing.**

## 5) VIX-curve VRP (`vix_vrp`) — PAPER (a 4th, different-risk-class premium)
- Short the front VIX future (owned Norgate VX) in **contango** (roll-down capture), flat in
  backwardation (gate: VIX < VIX3M). Vol-targeted.
- **Sharpe ≈ 0.64** (post-2015 0.53, 2020s 0.51), HAC p 0.0018, maxDD −18%. **SURVIVES the
  stress test** (Feb-2018 −4.4%, COVID −4.8% — the gate flips flat; naive short-vol loses 50-90%).
- corr-to-trend ≈ **0.46** (highest of the four — short-vol is somewhat risk-on); single-sleeve
  residual-α t ≈ 1.46. **Caveat: CPCV had no stress-regime fold → crash-survival is a manual
  event-study, must re-confirm live.**

## Correlation picture — what we have, and the gap
- We have **pairwise average correlation to trend**: carry 0.25, xsmom 0.12, VRP 0.46; and
  corr(futures-trend, carry) 0.10.
- **We do NOT yet have:** the full 4×4 (trend/carry/xsmom/VRP) correlation matrix, nor — more
  importantly — the **conditional / tail correlation in stress windows** (do they spike together
  in a crisis?). This gap is the heart of Theme C.

## The deployment question this sets up
Carry alone is marginal (t 1.76); the *book* (carry+xsmom) is significant (t 2.29); VRP is a
separate-risk-class addition (t 1.46, crash-gated). So the live multi-premia book would plausibly
be: **ETF-trend (live, Alpaca) + futures-book (IBKR) + VIX-VRP (IBKR)** — three streams across two
brokers. How to size and risk-manage that is Theme A; whether t=2.29 is *real* after multiple
testing is Theme B; whether the streams actually diversify in a crisis is Theme C.
