# Comprehensive Portfolio-Brain — LLM architecture-review pack (2026-06-21)

A **design / architecture** review (distinct from the strategy-validation panels). The subject is
the **portfolio brain**: redesigning MrTrader from independent strategy silos into one cohesive
system that reasons holistically over open positions, pending decisions, aggregate risk,
correlation, and exposure across multiple venues — and the question of **whether/where LLMs belong
in the live process.** We are explicitly open to a re-architecture.

This is, in the operator's words, **the most important part of the system** — so the pack asks for
deep, opinionated, buildable design, not a quick take.

## Files (read in order)
- `PASTE_INTO_CHAT.md` — the opening prompt (paste first)
- `01_PURPOSE_AND_MANDATE.md` — what we're asking + the bar for the answer
- `02_CURRENT_ARCHITECTURE.md` — how the live system actually works today (precise, file-level)
- `03_THE_BOOK_AND_STRATEGIES.md` — what the brain must coordinate (strategies, venues, capital)
- `04_THE_GAPS.md` — where the silo design breaks down (the problem, concretely)
- `05_DESIGN_QUESTIONS.md` — the core architecture questions (state, sizing, netting, cross-venue, safety, migration)
- `06_LLM_IN_THE_LOOP.md` — should LLMs sit inside the live process, and exactly where?
- `07_CONSTRAINTS_AND_FUTUREPROOFING.md` — hard constraints + the state-of-the-art / future-proof bar
- `08_THE_QUESTIONS.md` — consolidated, prioritized asks

## How to run it
1. Fresh chat with each LLM (ChatGPT, Claude, DeepSeek, Gemini, Grok — same panel for comparability).
2. Paste `PASTE_INTO_CHAT.md`, then attach the 8 files in order.
3. Save each response into `responses/` (e.g. `PortfolioBrain_<Model>.md`).
4. Tell me when they're in — I'll synthesize all five (one dedicated reader per response) into an
   opinionated target architecture + a staged migration plan, the same way we synthesized the
   Alpha-v10 panel.

## Design intent of this pack (why it's shaped this way)
- It is **design-first, not validation-first** — assume the strategies are given; the subject is the
  coordination architecture.
- It **invites re-architecture** — it does not ask the panel to bless the current PM/RM/Trader model.
- It is **honest about the as-built system** (file 02/04), including the uncomfortable facts (the
  live sleeves bypass the risk manager; there's no consolidated book state; no broker abstraction) —
  so the design targets real gaps.
- It makes **LLM-in-the-loop a first-class question** with our priors stated (deterministic
  execution path; LLMs bounded/around it, not inside) — and asks the panel to confirm or break that.

## Honesty contract
Candor over comfort. Tell us if our instincts (deterministic core, LLMs only around the edges,
robust-over-clever sizing) are wrong. We would rather hear "you're about to build a fragile
optimizer" or "your kill-switch design has a hole" than a polite endorsement.
