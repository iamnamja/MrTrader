# 08 — The questions (consolidated, prioritized)

Answer with depth and conviction. Diagrams-in-text, data-model sketches, and pseudo-code are
welcome. Where you choose, name the trade-off and what would flip your choice.

## Tier 1 — the architecture (most important)
1. **The target design.** Give the portfolio-brain architecture end-to-end: components,
   responsibilities, data flow, the decision loop, where it runs. If you'd replace PM/RM/Trader
   (e.g., with a target-portfolio / risk-budgeting / execution-planner model), show it and justify it.
2. **The consolidated book-state model.** The single source of truth for open + pending + aggregate
   (and factor) exposures across venues — data model, where it lives, how it stays consistent with
   two brokers. (D2)
3. **Holistic sizing under one risk budget** — the concrete, robust recipe (correlation-aware but
   not fragile; the diversification-without-blowup answer). Include how paper strategies ramp into
   the live budget. (D3)
4. **Netting, conflict resolution, and the netted factor-exposure view** across strategies/venues. (D4)

## Tier 2 — cross-venue, safety, extensibility
5. **Broker abstraction + cross-venue aggregation + the single multi-broker kill-switch /
   dead-man.** (D5)
6. **Determinism, idempotency, immutable snapshots, auditability — at the book level**; and the
   failure modes you'd most fear here + their guardrails. (D7)
7. **Extensibility:** the strategy contract (register-a-strategy), the venue/asset contract, and how
   research↔live parity is kept structural. (D6)

## Tier 3 — LLMs and the path
8. **LLMs in the loop (file 06):** the LLM-IN vs LLM-OUT map, the exact integration point + bounded
   action + fallback + audit for each "IN" role, the failure modes, and a clear verdict on whether
   an LLM monitoring/anomaly/narrative layer is worth building **now** vs after the deterministic
   brain is solid.
9. **The migration path** from our live paper system to the target — staged, strangler-style, ETF
   trend never goes dark; what to build first, what to route through the brain next, how IBKR slots
   in cleanly. (D8)

## The closer — "if you were architecting this from scratch, for me"
- The **one-paragraph north-star design** for MrTrader's portfolio brain.
- The **first three things you'd build**, in order, and why.
- The **single biggest mistake** you think a solo operator makes building this, and how your design
  avoids it.
- Anything we **didn't ask** that you'd insist on — the question behind the question.
