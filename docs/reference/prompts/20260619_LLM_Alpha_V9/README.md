# LLM Alpha-v9 External Review Kit (2026-06-19)

Brutally-honest external "red team" of the whole MrTrader program now that we have
**equities + 4y options + survivorship-free futures** data. Goal: an honest assessment of our
state, strategies we should try next, and whether the architecture/harness is sound.

## How to use
1. Open a fresh chat with each LLM you want to poll (e.g. ChatGPT, Claude, Gemini, Grok,
   DeepSeek — the more independent reads the better).
2. **Attach the 7 briefing files** (everything `0X_*.md` below) — that's the "feed."
3. **Paste the contents of `PASTE_INTO_CHAT.md`** as your message. It sets the role
   (world-class hedge-fund quant dev, brutally honest) and asks the 6 questions.
4. Save each response back into this folder as `responses/<llm-name>.md` for later synthesis.

## Files to FEED the LLM (7 — under the 10 limit)
- `01_PROGRAM_OVERVIEW.md` — what it is, what's live, the journey
- `02_KILL_KEEP_LEDGER.md` — every strategy tried + verdict + why
- `03_CURRENT_EDGES_AND_NUMBERS.md` — trend (live) + carry (new) + caveats
- `04_DATA_INVENTORY.md` — data owned + gaps + what to buy next
- `05_ARCHITECTURE.md` — live system + research pipeline
- `06_VALIDATION_HARNESS.md` — the gate/CPCV/pre-registration + our own failure modes
- `07_RECENT_FINDINGS_AND_CAVEATS.md` — last ~2 weeks + where the bodies are buried

## The message to paste
- `PASTE_INTO_CHAT.md` — paste this as your chat message (it's the prompt + the 6 questions).

## Notes
- The kit is **self-contained** (synthesized for outsiders) — it deliberately does NOT rely on
  our internal living docs, so nothing is missing from context. If an LLM asks for source code or
  a specific internal doc (e.g. `PIPELINE_ARCHITECTURE.md`), you can share it on request.
- It is written to **invite criticism**, not defend the program — the caveats (roll cost,
  survivorship, short options history, single live edge, possible ruler mis-calibration) are
  stated up front on purpose.
- After collecting responses, the next step is an Opus synthesis (consensus + disagreements +
  what changes a decision), as we did for the v6/v7/v8 reviews.

## ✅ SYNTHESIS DONE (2026-06-20)
All 5 responses collected (`responses/`) + deep-read (one Opus reader each) + synthesized into
**[`docs/reference/ALPHA_V10_SYNTHESIS_AND_PLAN.md`](../../ALPHA_V10_SYNTHESIS_AND_PLAN.md)** —
the Alpha-v10 plan. Headline: we hold risk premia (not alpha), the book is a single bet, carry is
unproven on execution; next moves are FREE (futures factor zoo + CoT) + EXECUTABLE (IBKR), not a
data buy. The panel's #1 diagnostic (the "trend contradiction") was run → the live book is vindicated.
