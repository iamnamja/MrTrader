# 05 — Design questions (the architecture)

These are the questions we most want a deep, opinionated answer to. Reason from first principles;
where you'd diverge from our current design, say so and show the better one.

## D1 — The overall architecture / decision model
- What is the right **component decomposition** for a cohesive portfolio brain? Is the
  PM→RM→Trader (propose → gate → execute) model the right abstraction, or should it be replaced by,
  e.g., a **target-portfolio model** (each strategy emits desired exposures → a portfolio
  constructor solves for the book target → an execution planner emits the minimal trade set →
  brokers)? Describe the components, their responsibilities, and the data flow.
- **Event-driven vs scheduled vs hybrid?** Today it's a 60s heartbeat + cron-like weekly jobs. For a
  weekly, low-frequency, multi-strategy book, what's the right control model? How do you make the
  weekly rebalance a single **atomic book decision** rather than several sleeves firing minutes apart?
- Where should the **"brain" run** relative to the web server? (We expect "decouple execution from
  FastAPI" — but tell us the concrete topology: one execution daemon? separate construct/risk/execute
  workers? what's the minimal robust split for a solo operator?)

## D2 — The consolidated book-state model (the source of truth)
- Design the **single source of truth** for book state: open positions + pending/intended trades +
  aggregate exposures, **across venues**, in real time. What's the data model? Where does it live
  (Postgres tables? an in-memory state actor rebuilt from an event log? event-sourcing?)?
- How do you represent **"what's being considered"** (all strategies' intended targets this cycle)
  so the brain can reason over the *proposed* book before any orders go out?
- How do you keep it **consistent with broker reality** (the "DB is not reality" problem) across two
  brokers with different position/cash/margin semantics — reconciliation design + fail-closed rules.

## D3 — Holistic sizing under one risk budget (the core)
- Given each strategy's signal/target each cycle, how should the brain set **per-strategy risk
  weights and total book gross** — by drawdown budget / vol target / margin, **correlation-aware**?
- **The robustness question we care about most:** correlation matrices are noisy and spike in
  crises. How do you capture the diversification benefit **without** a fragile optimizer that
  overfits last year's correlations and blows up? (Shrinkage? equal-risk-contribution as the robust
  default? a hard realized-correlation-spike de-gross trigger layered on a simple base allocation?
  regime-aware but bounded?) Give the concrete, defensible recipe for a solo book.
- How do **paper** strategies enter the live book — at a fractional risk weight that ramps with live
  evidence? How is that ramp encoded so it's automatic and auditable?

## D4 — Netting, conflict resolution, factor exposure
- How should the brain **net** overlapping/offsetting intended positions across strategies and
  venues before trading (avoid double-paying spread; avoid two sleeves fighting)?
- Design the **factor-exposure view** (equity beta, rates duration, USD, commodity, vol) netted
  across the whole book + both venues, and how factor limits gate the book target.
- When two strategies disagree (one wants long X, another short a correlated Y), what's the
  resolution rule — net at the book level, or honor both as independent risk takers? Argue it.

## D5 — Cross-venue / broker abstraction
- Design the **broker-abstraction layer** (Alpaca equities + IBKR futures, extensible) — the common
  interface, how venue-specific concerns (margin, contract specs, roll, settlement) are handled
  without leaking into the brain, and how positions/cash/margin are **aggregated into one book view**.
- The **single kill-switch across venues**: how does one trigger flatten/halt both brokers safely
  (idempotently, fail-closed), and what's the dead-man/heartbeat design?

## D6 — Extensibility / future-proofing
- Make **adding a strategy** a small, safe, **declarative** change (register a strategy that emits
  desired exposures + metadata: venue, asset class, risk class, capital eligibility). Sketch the
  interface/contract a strategy must satisfy.
- Make **adding a venue or asset class** (options, more futures, crypto) a contained change behind
  the broker/asset abstraction. What seams make this true?
- How do you keep the **research→live parity** (the same strategy logic backtests and trades) so a
  new sleeve's live behavior matches its validated backtest (replay/parity testing)?

## D7 — Safety, determinism, auditability
- **Idempotency & determinism:** the weekly book decision + the resulting orders must be safe to
  retry after a crash and reproducible from inputs. Design the run-id / idempotency / immutable
  snapshot scheme at the **book** level (not just per order).
- **Auditability:** every book decision should be reconstructable (inputs → targets → trades →
  fills → final book). What's the minimal event/snapshot log to guarantee that?
- **Failure modes you'd most fear** in this design, and the guardrails (fail-closed defaults,
  bounded position changes per cycle, staleness gates, reconciliation-before-trade).

## D8 — Migration from today
- We have a **live (paper) system** (file 02). Give the **staged migration**: what to build first
  (we suspect: the consolidated book-state/measurement layer), what to strangle later (the
  bypassing sleeves → routed through the brain), how to add IBKR cleanly, and how to do it **without
  a big-bang rewrite** while the ETF-trend book keeps trading.
