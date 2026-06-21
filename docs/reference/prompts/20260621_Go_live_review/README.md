# Go-Live Review — LLM panel kit (2026-06-21)

A focused external-panel review for the inflection from **validated paper book → responsibly-sized
live multi-premia book.** This is the follow-up to the 2026-06-19 Alpha-v10 panel (whose advice we
largely executed — see `02_WHAT_WE_DID_SINCE_LAST_PANEL.md`).

## The three themes
- **A — Go-live: capital sizing & cross-venue risk architecture** (the layer we never built).
- **B — Is the edge real, or multiple-testing residue?** (family-wise correction of the t=2.29
  second engine — the most important honesty check).
- **C — Do the four premia actually diversify, or co-crash?** (tail/crisis co-movement).

## How to run it
1. Open a fresh chat with each LLM (ChatGPT, Claude, DeepSeek, Gemini, Grok — same 5 as last time
   for comparability).
2. Paste the contents of **`PASTE_INTO_CHAT.md`** as the first message.
3. Attach (or paste) the 7 briefing files, in order:
   - `01_PROGRAM_OVERVIEW.md`
   - `02_WHAT_WE_DID_SINCE_LAST_PANEL.md`
   - `03_THE_FOUR_PREMIA.md`
   - `04_VALIDATION_AND_MULTIPLE_TESTING.md`
   - `05_GO_LIVE_DECISIONS.md`
   - `06_DIVERSIFICATION_AND_TAILS.md`
   - `07_THE_QUESTIONS.md`
4. Save each response into `responses/` (e.g. `GoLive_<Model>.md`).
5. Tell me when they're in — I'll synthesize all five with one dedicated reader per response
   (same method as the Alpha-v10 synthesis), reconcile agreements/disagreements, and turn it into
   a prioritized, implementable plan.

## Design notes (why this kit looks like it does)
- It **shows the panel what we executed** since their last review — this sharpens feedback and
  respects their prior input.
- It **leads with the honesty check (Theme B)**: if the t=2.29 second engine doesn't survive
  family-wise correction, the sizing questions (Theme A) are moot — so we ask the panel to answer
  B first.
- It is deliberately **decision-focused**, not a whole-program re-litigation — the open decisions
  are sizing, cross-venue risk, promotion gating, and tail diversification, all genuinely unbuilt.

## Honesty contract (same as last time)
Candor over comfort. We would rather hear "you're about to size a multiple-testing artifact" or
"this is a leveraged long-risk-premium pretending to be diversified" than a polite list. Every
claim should come with a refutable test we can run.
