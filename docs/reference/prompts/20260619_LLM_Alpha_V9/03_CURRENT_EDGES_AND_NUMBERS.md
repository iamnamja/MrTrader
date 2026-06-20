# 03 — Current Edges & Numbers (the two that matter + the new find)

All numbers are net-of-modeled-cost, daily returns, our own backtests. Challenge them.

## A) TSMOM trend (LIVE) — the validated edge
- Universe: 10 ETFs (equity US/intl, rates, gold, broad commodity, USD).
- Signal: ensemble sign of trailing returns over (21, 63, 126, 252) trading days; inverse-vol
  sized; long-flat (no shorts); **weekly** rebalance; per-instrument vol target 10%.
- **Standalone Sharpe ≈ 0.72** (2007-2026, incl. all crises); crisis-positive in slow bears
  (2008), whipsawed in fast shocks (COVID). maxDD ≈ −14%.
- Live at ~50% gross (a Kelly/vol-target analysis said 25% badly under-deployed the only edge;
  full Kelly would be ~7.7× gross, so 50% is deeply haircut). VIX governor on top.

## B) Futures CARRY (NEW — the first genuinely new edge since trend; NOT yet live)
- Data: **Norgate**, survivorship-free, **76 liquid markets** (equity index, rates/bonds, FX,
  energy, metals, ags), 30+ yr history.
- Signal: **cross-sectional carry** = annualized term-structure slope `(P_front − P_next)/P_next ÷ Δt`,
  computed from the nearest two contracts (scheduled expiry from the contract code). Each week,
  long the most-backwardated / short the most-contango markets, inverse-vol sized, book-vol-targeted.
- **Standalone Sharpe ≈ 0.66**, and — unlike trend — **positive in EVERY sub-period including the
  modern regime** (2010-19 +1.00, **post-2015 +0.84-0.89**). Signs economically correct (nat-gas
  deep contango, energy backwardated). Cost-robust (≈0.62-0.65 at 2× cost).
- **Diversifies the live book:** corr to ETF-trend ≈ 0.25; **Track-B (does it improve the book):
  adding carry to the live ETF-trend book lifts Sharpe 0.72 → 0.89, dSR +0.17** — the strongest
  diversification result the program has produced.
- Official two-track gate: **Track-A PAPER-PASS** (mean-fold Sharpe 0.79, point-SR 0.81,
  HAC p≈0.000); **CAPITAL-FAIL** only on the structural blocks (needs a live-paper record +
  n_folds≥10) — same path crypto is on.

## C) Trend + carry combined (managed-futures book)
- corr(futures-trend, carry) = 0.10 (near-orthogonal factors).
- Equal-risk trend+carry book: full Sharpe ~1.0, **post-2015 +0.57**, maxDD −29% (shallower than
  either alone) — the canonical CTA result.
- **BUT full-book caveat:** layered on our *existing* ETF trend, only the CARRY half adds value.
  ETF-trend-only 0.72 → +carry **0.89** → +futures-trend **0.57** (worse, decayed+redundant) →
  all-3 0.78. **Deployment call: add carry; do NOT add futures-trend.**

## Honest caveats on the carry number (we want you to push on these)
1. **Roll cost is NOT yet in the model.** We trade the front contract and roll it (~4/yr equity-
   FX-bonds, ~12/yr energy/metals). A roll changes the *contract* not the *weight*, so our
   `|Δweight|` cost term charges nothing for it. Quantified unmodeled roll drag ≈ **1.1-1.9%/yr**
   → **Sharpe hit ≈ −0.05 to −0.15** → honest carry Sharpe ≈ **0.55-0.60**. (Fix pending.)
2. **No real fills yet.** All futures numbers are signal-level on Norgate EOD continuous series.
   The true execution test (slippage, real roll, margin) needs an IBKR paper account (planned).
3. **Survivorship:** the universe is filtered on *current* liquidity. We quantified this as
   immaterial (full 0.66 vs ≤2005-history-only 0.65; post-2015 0.89 both), and the mirror only
   has currently-listed markets (futures rarely delist) — but it's not a true point-in-time universe.
4. **In-sample vol-matching:** the combined-book / Track-B blends are vol-matched on full-sample
   std for a relative-Sharpe comparison; a PIT rolling-vol blend is modestly lower.
5. **Rebalance cadence:** weekly, confirmed by a pre-registered study (a tempting daily-carry
   "edge" was an early-period artifact; weekly beats daily in the modern era).

## What would make us deploy carry to capital
A live-paper OOS record (signal persistence) → then an IBKR-paper real-fill validation incl. ≥1
vol spike → then a small capital allocation paired with trend. We are NOT there yet.
