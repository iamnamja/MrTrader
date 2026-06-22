# Questions — External 5-LLM Review (2026-06-22, Alpha-v10)

Read `snapshot.md` first. For each block I give **the internal panel's current view** — your job is to
**contest it**, not echo it. Be brutally honest and specific (real signals, instruments, numbers,
falsification tests). Where you agree, say why; where you disagree, say what the internal panel missed.

---

## Block A — An overlooked trading method (weight: HIGH)
We trade trend (multi-asset TSMOM, live) + futures carry + xsmom (paper). We've killed the entire list
in snapshot §5. The real gate is **marginal contribution vs the existing book** (corr-to-book < ~0.30),
not standalone Sharpe.

**Internal panel's view:** the existing book is all *convergent continuation* premia; the white space is
the *reversion / convexity* quadrant the book is structurally short. Top candidates surfaced: (1)
short-horizon mean-reversion (but our red-team found we already built + shelved it as cost-dead);
(2) G10 FX *value* (mean-reverting, distinct from our commodity-heavy carry); (3) equity dispersion /
implied-correlation premium (orthogonal but needs an options-data buy); (4) re-gating crypto-trend as a
declared diversifier (corr 0.18). The red-team's punchline: **no one proposed a genuinely long-convexity
crash hedge — the one thing the book is short — and adding any new convergent "diversifier" re-violates
"one bet in a crisis."**

**A1.** What genuinely-distinct family are we missing? Rank by *expected marginal* contribution, not
standalone Sharpe. Be explicit about the economic payer.
**A2.** Is there a *long-crisis-convexity* premium/strategy that is NOT just "buy puts / long VIX"
(negative carry) — something with positive or neutral carry that still pays off in a crash? (e.g.
trend-on-defensives, FX/rates trend, gold/bond convexity, time-series-momentum tilts.) This is our
biggest gap — push hard here.
**A3.** Contest the reversion thesis: is short-horizon reversal genuinely dead for us (cost), or is there
a lower-turnover / different-instrument formulation that survives 2 bps liquid-ETF costs?
**A4.** FX value vs FX carry on G10 — which (if either) has a live pulse post-2015, and how would you
falsify it without overfitting?

## Block B — Swing equity (weight: HIGH)
We've killed the equity ML ranker (IC≈0 × 3), PEAD (event-level dead), reversal (cost), overnight,
short-interest, turn-of-month. PDT does NOT bind us (>$25k account).

**Internal panel's view:** swing equity is mostly a **sunk-cost trap**; the single opening is
**vol-managed single-name cross-sectional momentum** (Barroso/Santa-Clara — size 12-1 momentum inverse
to the momentum factor's own realized variance), which is the one construction never run (the ranker was
always constant-gross). BUT (a) it has high redundancy risk vs our trend book (also momentum), and (b)
our red-team killed the proposed "pre-screen on survivorship-biased data" because the bias *flatters*
this specific strategy (removes the crash names vol-management is meant to dodge) → false-positive-only.
It's data-gated on the $693 Norgate buy.

**B1.** Do you agree swing equity is a sunk-cost trap for this setup, or is there a credible avenue the
internal panel is too pessimistic about? Be decisive.
**B2.** Is vol-managed single-name momentum worth the Norgate buy, given it's likely redundant to our
trend sleeve? What would make it *non-redundant* (sector/beta-neutral construction)?
**B3.** Is there a *non-momentum* swing equity premium we haven't tried that wouldn't just die the way
the ranker did — and that isn't single-name-cost-dead?
**B4.** Given survivorship bias flatters most single-name tests, is there ANY valid cheap pre-screen
before spending $693, or is "buy the clean data or don't test" the honest answer?

## Block C — Better-trade what we have (weight: MEDIUM-HIGH)
Live trend at 50% NAV, ~4.7% realized vol; weekly rebalance; inverse-60d-vol sizing; no skip-month; VIX
governor live; drawdown ladder + book-vol-targeting BUILT but running shadow/research-only.

**Internal panel's view:** the highest-EV "better-trade" move is **make the existing risk machinery
load-bearing** (flip the whole-book gate to enforce; wire the drawdown ladder into the live budget) +
**harvest idle-cash RFR**. Lower-priority knobs: EWMA vol, a skip-month on the TSMOM signal, a rebalance
band. The red-team's **hard veto: do NOT turn book-vol-targeting up to 8%** — it levers the sole live
undiversified edge ~1.7× via a mechanism that's biggest right before vol spikes, just as correlated
IBKR beta is about to stack on top.

**C1.** Do you agree the vol-target-up-to-8% move is dangerous, or is under-deploying the only validated
edge (at ~4.7% vs a 6–8% policy target) the bigger sin? What vol would *you* run, and why?
**C2.** Is wiring a drawdown de-gross ladder into a single-sleeve trend book genuinely additive, or does
the VIX governor already capture most of it (correlated triggers)?
**C3.** Skip-month on the live TSMOM signal: worth the risk of touching the crown-jewel edge? Any other
sizing/timing change you'd prioritize over the internal panel's list?
**C4.** When IBKR lands and we have ≥2 live sleeves, what's the *minimum* correct way to combine them
(inverse-vol vs ERC vs covariance) given we'll have near-zero joint live history?

## Block D — Make the app stronger (weight: MEDIUM-HIGH)
See snapshot §6. The big gap our internal red-team found: reconciliation + kill-switch state machine are
**built but not wired** to the live order path; the drawdown ladder/vol-target are research-only; the
live book's only enforced controls are the VIX governor + 80% gross cap.

**Internal panel's view:** the #1 next move (over any new strategy) is **make the safety layer
load-bearing before IBKR**: wire reconciliation-before-trade (fail-closed) + the kill-switch state
machine, fix a cash-ETF instrument-mapping gap, then flip the gate to enforce. Plus IBKR-specific no-go
items: an out-of-band broker-only flatten + an external dead-man watchdog (neither exists).

**D1.** For a solo-operator, single-process, about-to-trade-futures book, what are the *catastrophic*
failure modes we're most likely underweighting? Rank by blast-radius × likelihood.
**D2.** Is "reconciliation-before-trade + kill-switch wired + gate in enforce" a sufficient hard no-go
gate before IBKR capital, or what else is mandatory (out-of-band flatten, dead-man, verify-on-connect
on futures multipliers, per-order idempotency)?
**D3.** What's the *minimum* monitoring that would stop a bad state running unnoticed overnight, without
over-building?
**D4.** Where are we over-engineering safety relative to a $100k book — what should we explicitly NOT
build yet?

## Block E — The meta-question (weight: HIGH — answer even if you skip others)
Given a ~$100k paper book, a single live bet (trend+cash), a paper second engine (futures, gated on
IBKR), and a long list of honest kills:

**E1.** Is the internal panel right that the binding constraint is **capital + live track record +
not-blowing-up**, NOT finding a 5th sleeve — i.e. should we mostly *stop hunting alpha* for the next 1–3
months and instead harden + accrue track record? Or is that complacency?
**E2.** If you could force us to do exactly THREE things in the next two weeks (IBKR still pending), what
are they — and what one thing on our list is most likely a waste of time or actively dangerous?
**E3.** What is the single most intellectually-dishonest thing in how we've framed this program — the
blind spot a desk of humans would catch that our model-panel did not?
