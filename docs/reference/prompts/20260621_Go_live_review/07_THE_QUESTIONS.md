# 07 — The questions (consolidated)

Answer in priority order; be specific, prioritized, implementable. Where you make a claim, say what
evidence would confirm/refute it — we can re-run almost anything. Lead with anything you think we've
already gotten wrong.

## Theme B — Is the edge real, or multiple-testing residue? (answer FIRST — it gates everything)
B1. Across ~20 sleeve families (file 04), does our surviving book — carry, xsmom, the **t = 2.29**
    futures-book, VRP — survive an honest **family-wise / deflated-significance** correction? What
    bar should t = 2.29 clear?
B2. The t = 2.29 came from **combining** carry (t 1.76) + xsmom (t 1.60). Is "gate the basket" a
    legitimate significance gain or a multiple-testing trick? How do we tell?
B3. Give us a concrete **null-strategy zoo** protocol for *rules-based cross-sectional futures
    factors* (randomized sign/rank? block-permuted signals? how many nulls? what statistic — White
    Reality Check / Hansen SPA / deflated Sharpe?) that we can build and run this week.
B4. How do we **prospectively** log research degrees-of-freedom so the family-wise burden stays
    counted going forward?

## Theme A — Go-live: capital sizing & risk architecture (the unbuilt layer)
A1. **Sizing:** target book vol / drawdown budget for a solo ~$100k CTA-style book → per-sleeve
    risk weights. Equal-risk-contribution vs HRP/risk-parity with only 3-4 sleeves? Should *paper*
    sleeves enter at a fractional weight that ramps with live evidence? Margin-to-equity ceiling?
A2. **Cross-venue (Alpaca + IBKR):** the minimal-correct unified book-level risk surface + a single
    kill-switch governing both brokers + reconciliation discipline.
A3. **Forward risk:** define + calibrate a realized-correlation-spike de-gross trigger and a
    global book-drawdown de-risk ladder (windows, thresholds, magnitudes — without overfitting).
A4. **Promotion ladder:** explicit rungs paper → IBKR-paper → tiny-live → scale, with the **evidence
    threshold and the demotion stop at each rung** (how long, how many rolls, ≥1 vol spike, what
    live-vs-backtest consistency check).
A5. **The no-go list:** the explicit conditions under which you'd keep this on paper indefinitely
    rather than deploy capital.

## Theme C — Do the four premia actually diversify, or co-crash?
C1. Genuinely multi-bet, or a leveraged long-risk-premium with extra steps (3 risk-on + 1
    short-vol)? What's the tell?
C2. The specific **stress-conditional correlation / tail-dependence** test you'd trust on our data,
    and the threshold above which it's "one bet."
C3. How to **stress-test a book with no isolated crisis fold** (scenario replay? crisis-weighted
    CPCV fold? block-resampled crisis bootstrap?).
C4. Is there a **convex/defensive sleeve** worth building from the data we own to offset the
    net-short-crisis tilt — and how to prove it pays in the tail rather than bleeding carry?
C5. Does **VRP** belong in the book given 0.46 corr-to-trend + short-crisis nature, or does it
    concentrate the tail risk we should diversify away?

## The closer — "If you were chairing my risk committee Monday morning"
Top 3-5 concrete moves, in order, to get from "validated paper book" to "responsibly-sized live
multi-premia book" — including the single thing you'd do FIRST and the single thing you'd refuse
to let me do.
