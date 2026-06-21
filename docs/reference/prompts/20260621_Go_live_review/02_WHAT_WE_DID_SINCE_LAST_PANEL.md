# 02 — What we did since your last panel (the v10 execution scorecard)

Your 5-LLM panel (2026-06-19) verdict: *"alternative risk premia, not alpha; single bet; carry
unproven; the app is ahead of the alpha."* Here is what we executed in response — so you can see
what's changed and calibrate this round. **We did almost all of it.**

## Your advice → what we did → what we found

| Your point | What we did | Result |
|---|---|---|
| **"Carry's 0.66 is a phantom until roll mechanics are modeled — but don't double-count the roll *yield*"** | Added a **transaction-only** roll cost (commission + half-spread on the contract switch, 3bps/side; round-trip per roll on \|held weight\|). Explicitly does NOT subtract the harvested roll yield. | **Honest carry Sharpe 0.66 → 0.58** (drag ~1.1%/yr). Still post-2015 0.81, HAC p 0.0001. Diversification **real but marginal** as a single sleeve (residual-α t ~1.8). |
| **"Kill in-sample vol-matching; use PIT rolling vol"** | Switched Track-B to a **budget-invariant residual-alpha** (no in-sample vol-match). | The old in-sample +0.17 carry dSR was a **vol-match artifact** (~0.00 under PIT). Residual-alpha is the honest frame. (Full PIT-vol migration of the credit overlay is the one piece still pending.) |
| **"Prove the ruler isn't leaky" (Type-I)** | Built Monte-Carlo **negative controls**. | True-null PAPER false-positive rate **23.6% floor-alone → 5.3% JOINT** with the HAC floor (leak closed to nominal); anti-correlated zero-edge null → Track-B pass-rate **5.7%** (~nominal). Gate is **Type-I controlled**. |
| **"Mine the free futures factor zoo you already own"** | Built a generic cross-sectional factor engine; tested **6 factors** (xs-momentum, curve-momentum, value, skew, basis-momentum, CFTC CoT). | **Only cross-sectional 12-1 momentum survived** → `futures_xsmom` (0.56). curve-mom / value / skew / basis-mom / CoT all KILLED at the pre-registered sign (no sign-flipping). **The free zoo is exhausted at carry + xs-momentum.** |
| **"You need a real second engine, not a single bet"** | Combined carry + xsmom into an equal-weight **`futures_book`**. | **Book Sharpe 0.67; Track-B residual-α vs the live ETF-trend book t = 2.29 (SIGNIFICANT), resid-Sharpe 0.56, beta 0.24** — vs carry-alone t 1.76. "Gate the basket" works: two marginal factors combine past significance. **This is the validated second engine.** |
| **"Re-test VRP — it was likely parked on too-short data + alpha-framing; try the VIX-futures curve"** | Built VIX-curve VRP: short the front VIX future in contango (roll-down), flat in backwardation (crash-governor gate). | **Sharpe 0.64; SURVIVES Feb-2018 (−4.4%) and COVID (−4.8%)** via the gate (naive short-vol loses 50-90%). Reverses the earlier VRP park → a **4th risk premium.** |
| **"Harden execution before live futures capital; decouple the scheduler"** | Wrote a **turnkey IBKR execution design spec** (contract master + verify-on-connect, idempotency, broker-as-source-of-truth reconciliation, immutable snapshots, kill-switch/dead-man, FastAPI decouple, replay parity). | Adapter **build starts on IBKR approval** (verify against the live API, don't build blind). |
| **"Norgate US Stocks is a close-the-question buy, after carry"** | Deferred (carry now hardened; buy is optional, ~$347 for a bounded equity audit). | Not yet bought. |
| **(bonus, this week) "could bad saved data have caused issues?"** | Ran a full **data-quality audit** of everything we've saved. | **The data behind every live position + recent verdict is CLEAN.** Fixed 2 off-strategy FMP bugs; found (documented, not live) a look-ahead in a parked fundamentals-factor scorer. |

## What we did NOT do (still open — your input wanted)
1. **Family-level / multiple-testing trial counting** (your sharpest point). Still undone. → Theme B.
2. **Capital-allocation + cross-venue sizing layer.** Never built. → Theme A.
3. **Tail / crisis co-movement of the 4 premia.** Only pairwise average corr checked. → Theme C.
4. **PIT-vol migration of the credit overlay** (likely evaporates) — minor, scheduled.

## The net
You said the bottleneck was *implementation truth, not idea generation* — and that's exactly where
we are now. The research produced a credible second engine; the open risk is whether we **size it,
combine it across two venues, and certify it (vs multiple-testing + tail risk) correctly** before
real money. That's this review.
