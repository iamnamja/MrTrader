# Cover — External 5-LLM Review (2026-06-22, Alpha-v10)

**To the reviewer:** You are a world-class quantitative researcher / systematic-trading PM / quant-
platform engineer (whichever hat each question needs). I run a solo systematic-trading program,
"MrTrader." I want a **brutally honest** review — no flattery, no hedging, no participation trophies.
If an idea is weak, say so and say why. If I'm fooling myself, tell me where. I would rather hear "this
is a sunk cost, stop" than a polite list of things to try.

## What I'm asking
While I wait for IBKR futures approval (live book unchanged in the meantime), I want outside eyes on
four things, in priority order:
1. **An overlooked trading method** — given what I already trade and the long list of things I've
   already killed, what genuinely-distinct family have I missed that could clear my gates?
2. **Swing equity** — I'd like to revisit swing trading (medium-frequency, days-to-weeks). Is there a
   credible avenue, or is it a sunk-cost trap for my setup?
3. **Better-trade what I have** — sizing/timing/execution/governor/allocation improvements to the
   existing book, rather than new alpha.
4. **Make the app stronger** — robustness/risk-plumbing before I wire real IBKR capital.

## How to read the materials
- `snapshot.md` — the current system: what's live, what's paper, what I've killed and why, the
  acceptance-gate philosophy, the safety architecture status, and the data I own. **Read this first.**
- `questions.md` — the specific questions, weighted to the four areas, including the sharpest internal
  debates I've already had (I ran my own internal panel first — see below — so I want you to *contest*
  its conclusions, not just repeat them).

## Important context on method
I already ran an **internal panel** (five independent analyses that could read my actual codebase) plus
an adversarial red-team. Their synthesis is summarized in `questions.md` as "the internal panel's
current view" for each topic. **I am explicitly asking you to attack those conclusions.** You cannot see
my code, but you have decades of priors I don't — tell me where the internal panel is wrong, where it's
groupthinking, and what a desk full of people would have caught that five instances of one model
family did not.

## My standing principles (so you can hold me to them)
- **I hold risk premia, not alpha.** Trend + carry are ARP. I size by drawdown, not Kelly; I freeze gross.
- **The book is "effectively one bet in a crisis"** (post-2015 pairwise corr ~0.49; convergent premia).
- **Orthogonality without standalone return is worthless;** a great standalone Sharpe that's redundant
  to the live book (high corr-to-book) gets PARKED. The real gate is *marginal* contribution.
- **Pre-registration + multiple-testing discipline.** I kill at the pre-registered sign; no sign-flipping.
- **No data buy without a stacked justification;** no new live capital before the safety layer is real.

Be specific. Real signals, real instruments, real numbers, real falsification tests. "Try machine
learning" or "improve risk management" will be ignored.
