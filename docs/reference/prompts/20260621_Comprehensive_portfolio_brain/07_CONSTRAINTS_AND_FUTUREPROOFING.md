# 07 — Constraints & the future-proofing bar

Design within these; if a constraint is wrong, challenge it explicitly.

## Hard constraints
- **Solo operator.** One person builds, runs, and debugs this. Operational simplicity and
  auditability beat sophistication. A design that needs a team to keep alive is the wrong design.
- **Capital scale:** ~$100k paper now; realistic scaling is modest, not institutional. Don't assume
  co-located infra, FIX engines, or a quant team. But don't design a toy either — it should scale
  to a few million and more strategies/venues without a rewrite.
- **Low-frequency book:** everything is weekly, EOD-driven (Norgate futures are EOD-only). No HFT
  concerns. The hard problems are coordination/sizing/risk/correctness, not latency.
- **Existing stack (reachable from here):** Python, FastAPI, APScheduler, Postgres, Redis, React
  dashboard, Alpaca (live), IBKR (incoming, `ib_insync` planned). We can change topology and add
  components; we'd prefer to evolve this stack rather than adopt an exotic one without strong reason.
- **Determinism & auditability are non-negotiable** in the sizing/execution path: every live
  decision must be reproducible from inputs and fully reconstructable after the fact.
- **Fail-closed by default:** any data/connection/clock/margin/reconciliation problem must reduce or
  halt risk, never silently increase it. (This is already how the live sleeves behave.)
- **Regulatory/practical:** PDT and margin rules on the equity side; futures margin on IBKR. Nothing
  exotic, but the design shouldn't ignore them.

## Anti-goals (things we do NOT want)
- A **fragile optimizer** that looks elegant and blows up when correlations regime-shift.
- **Over-engineering**: microservices/k8s/event-bus sprawl a solo operator can't run. Right-size it.
- A **black box**: if we can't explain why the book did what it did, it's wrong.
- **Big-bang rewrite** that takes the live book offline. Evolution, not revolution.

## The future-proofing / state-of-the-art bar
We want the design to still be the right one in 2–3 years as the book grows. Specifically:
- **Adding a strategy** = implementing a small contract (emit desired exposures + metadata) and
  registering it — no bespoke rebalancer, no scheduler surgery, no teaching the reconciler about it.
- **Adding a venue / asset class** (more futures, options, crypto, a new broker) = a contained
  adapter behind the broker/asset abstraction — the brain doesn't change.
- **Swapping the sizing policy** (equal-risk → risk-parity → regime-aware) = a strategy-pattern swap
  behind a stable interface, independently testable, with the old policy reproducible.
- **Research↔live parity** is structural: the same strategy code path backtests and trades; a replay
  mode proves the live brain reproduces the research book on golden dates.
- **Observability** is first-class: at any moment the operator can see the consolidated book, its
  factor exposures, its risk-budget utilization, and *why* the last rebalance did what it did.

## What "state of the art" means to us here
Not "uses the trendiest tech." It means: **the design a top systematic shop would recognize as
correct for a low-frequency multi-strategy book** — a clean target-portfolio / risk-budgeting core,
broker-agnostic execution, one source of truth for book state, deterministic + auditable decisions,
bounded and well-placed automation (incl. LLMs), and graceful extension to new strategies/venues —
**right-sized for a solo operator.** Tell us what that looks like, and where we should deliberately
stop short of institutional complexity.
