# 06 — Diversification & tails (Theme C) — do the four premia co-crash?

Your panel's central critique was **"it's effectively a single bet."** Adding carry + xsmom + VRP
*looks* like it fixes that (low pairwise correlations). But we are worried the fix is cosmetic:
**low average correlation ≠ low tail correlation**, and all four streams may be fundamentally
"risk-on" premia that blow up together in a liquidity crisis.

## What we have
- **Pairwise average correlation to live trend:** carry 0.25, xsmom 0.12, VRP 0.46;
  corr(futures-trend, carry) 0.10.
- Each sleeve's standalone drawdown/regime behavior (trend maxDD −14%; futures book −; VRP −18%,
  gate-protected through Feb-2018 + COVID).
- A VIX-term crash governor (on trend) and a contango gate (on VRP) — both reduce exposure in
  stress, by construction.

## What we DON'T have (the honest gaps)
1. **The full conditional/tail correlation matrix.** We have not computed cross-sleeve correlation
   *conditioned on stress* (e.g., correlations in the worst 5% of equity days, or in VIX-spike
   windows). Unconditional 0.1-0.46 could mask a tail correlation that → 0.8 in a crisis.
2. **No isolated crisis fold in CPCV.** Our purged-CV folds don't cleanly isolate 2008 / Feb-2018 /
   COVID-2020 / 2022 as held-out stress tests. So the book's crisis behavior is inferred from
   manual event-studies per sleeve, not from a held-out joint stress fold.
3. **A genuine convex/defensive sleeve.** Trend is *mildly* crisis-convex (slow bears) but
   whipsaws in fast shocks; carry + xsmom are not crisis hedges; **VRP is short-vol = explicitly
   short-crisis** (gated, but still). So the book is plausibly **net short-crisis** — the opposite
   of diversified when it matters most. We have nothing deliberately long-convexity.

## The questions (Theme C)
1. **Are we actually diversified, or kidding ourselves?** Given the four premia (3 trend-ish/carry
   risk-on + 1 explicitly short-vol), is this a genuinely multi-bet book or a leveraged long-risk
   premium with extra steps? What's the tell?
2. **How would you measure it** with our data — the specific stress-conditional correlation /
   tail-dependence test (exceedance correlation? copula tail dependence? worst-N-day co-movement?)
   you'd trust, and the threshold above which you'd say "this is one bet."
3. **How do you stress-test a book whose backtest never saw an isolated crisis fold?** Synthetic
   stress (historical scenario replay across all sleeves jointly)? A deliberately crisis-weighted
   CPCV fold? Bootstrap with block-resampled crisis windows?
4. **Is there a convex/defensive sleeve we should build from the data we own** (survivorship-free
   futures incl. full term structure, VIX curve, FRED macro, equities) to offset the net-short-crisis
   tilt? Candidates we can think of: long-vol-when-cheap (the *other* side of the VRP gate),
   bond/gold trend as a flight-to-quality leg, a defensive curve trade. Which (if any) is worth it,
   and how would you test that it actually pays off in the tail rather than just bleeding carry?
5. **Does VRP belong in the book at all** given its 0.46 corr-to-trend and short-crisis nature, or
   does it concentrate the very tail risk we should be diversifying away?

## What we'd do with the answer
Decide (a) whether the 3-stream book {trend, futures-book, VRP} is the right composition or whether
VRP should be dropped / paired with a long-convexity leg, and (b) what stress-conditional risk
check to wire into the go-live risk surface (Theme A3).
