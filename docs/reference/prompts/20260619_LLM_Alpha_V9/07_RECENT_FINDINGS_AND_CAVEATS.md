# 07 — Recent Findings & Honest Caveats (the last ~2 weeks, Alpha-v9)

A focused log of the most recent work, so you see *how* we work and *where the bodies are buried*.

## Findings (newest first)
- **Rebalance-cadence study → keep WEEKLY for trend & carry.** A quick full-sample check hinted
  carry liked daily (0.72 vs 0.66 weekly); a pre-registered, sub-period-robust study showed the
  daily edge is an early-period artifact (in the modern era **weekly beats daily**: post-2015
  0.89>0.79, 2020s 0.62>0.44). Built a no-trade-band rebalance primitive (didn't beat weekly).
- **Futures process hardening (4-agent adversarial review).** Found + fixed 5 real bugs before
  trusting carry: a **negative-denominator sign-flip** (CL 2020 negative-oil booked a loss as a
  +23.5% gain inside the winsor band), a carry **expiry/staleness + roll-time look-ahead** (now
  uses the scheduled contract code), a dead cross-section guard (noise-trading thin days), a
  `pct_change` deprecation, and a universe-classification bug (micros escaping / real bonds cut).
  **Carry verdict survived** all fixes (point-SR 0.81, post-2015 +0.89, Track-B +0.17).
- **Futures carry = real, modern, diversifying edge** (the headline; see `03`).
- **Futures trend = decayed** (full-sample 0.83 entirely pre-2010; modern ~0; redundant with ETF trend).
- **FINRA daily short-volume = real-but-weak** (informed-short signal confirmed; overlay beats
  buy-hold; but residual-α vs SPY insignificant + sub-period unstable → not standalone).
- **Overnight vs intraday = KILL** (overnight premium real, gross Sharpe +0.53, but daily round-
  trip cost erases it: net +0.16 < 0.30 floor). Clean confirmation that turnover is the enemy.
- **Crypto trend = paper-candidate** (Sharpe 0.64, low corr, but Track-B vs trend fails; ~5y
  history → no capital; live-paper OOS tracking started).
- **Cash sleeve = LIVE** (idle ~76% of NAV was earning zero; now in T-bills).
- **Trend allocation 25%→50%** (Kelly/vol-target analysis: 25% badly under-deployed the only edge).

## Caveats / things we're explicitly unsure about (push here)
1. **Carry roll cost not yet modeled** → honest carry Sharpe is ~0.55-0.60, not 0.66 (see `03`).
2. **No real fills for futures** — IBKR paper validation not done yet.
3. **Options work is on short/biased data** — 4y frozen Polygon file + computed greeks + a
   forward NBBO log only weeks old. We KILLED options-as-signal and PARKED VRP on this data;
   **maybe those kills are data-limited, not real.**
4. **Equities are survivorship-biased** — every equity cross-sectional verdict (PEAD, XS-ML,
   short-vol XS) is on contaminated data; we haven't bought clean equity history.
5. **We trade slowly by design** (weekly/EOD). We've never seriously tried faster horizons
   (intraday, mean-reversion, microstructure) — partly conviction (turnover kills), partly data.
6. **Single live edge.** With only trend live (+ cash), the book is essentially one bet. Carry
   would be the second. Is two enough? What's the path to a genuinely diversified book?

## What we are NOT asking
- We don't want generic "use ML / use alt-data / manage risk" platitudes. We want specific,
  prioritized, implementable ideas grounded in what we have and what we've already ruled out.
- We don't want reassurance. If our edges are weak/overfit or our ruler is broken, say so plainly.
