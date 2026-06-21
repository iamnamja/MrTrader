# 01 — Purpose & mandate

## Why this review exists
We have spent two years finding edges and validating them (one live: ETF trend; three paper:
futures carry, futures cross-sectional momentum, VIX-curve VRP). We are now crossing from
*research* to *real capital across multiple venues*. The piece that decides whether that goes well
is **not** another strategy — it's the **coordination layer**: the system that takes a set of
validated strategies and runs them as **one coherent book**, reasoning holistically about exposure,
risk, correlation, and what's open vs. what's being considered.

Today that layer is weak: strategies run as **independent silos** (see file 04). We think this is
the **most important part of the platform to get right**, and we want a deep, state-of-the-art
design — we are open to **re-architecting** if that's the correct answer.

## The mandate (the bar for your answer)
Design the **portfolio brain**: the architecture, data model, decision loop, and integration
patterns for a cohesive multi-strategy, multi-venue systematic book run by a solo operator. We want:

1. **A target architecture** — opinionated, first-principles, buildable. Replace our current
   PM/RM/Trader model if you'd design it differently.
2. **Holistic coordination, concretely** — one source of truth for book state; joint,
   correlation-aware sizing under a single risk budget; cross-strategy netting & conflict
   resolution; unified pre-trade risk; execution; reconciliation. With the algorithms and the data
   model, and an explicit answer to "how do you get diversification without a fragile optimizer."
3. **Cross-venue design** — one book over Alpaca (equities) + IBKR (futures), extensible to more.
4. **Future-proofing** — adding a strategy / venue / asset class should be a small, safe,
   declarative change.
5. **Safety & correctness** — determinism, auditability, idempotency, reconciliation, fail-closed,
   kill-switch; the failure modes you'd most fear here.
6. **LLM-in-the-loop** — whether/where LLMs belong in the live process, with the integration pattern
   and guardrails (file 06).
7. **A migration path** — from the live system we have today to the target, staged, no big-bang.

## What "good" looks like
- **Opinionated and argued**, not a menu. A strong recommendation we can push back on.
- **Concrete**: data model sketches, component responsibilities, the decision-loop sequence,
  pseudo-code where it clarifies.
- **Robust over clever**: this book is run by one person; an elegant design that's fragile or
  unauditable is worse than a boring one that's correct. Tell us where to resist sophistication.
- **Honest about trade-offs**: every choice has a cost; name it and the condition that flips it.

## What this is NOT
- Not a strategy/alpha review (that's a separate panel running in parallel).
- Not a request to validate our edges (assume the strategies are given).
- Not a green-field fantasy with no constraints — we have a live paper system and a real stack
  (file 07); the design must be reachable from here.
