# Cover note — paste this first

I run a solo systematic-trading research program (US, mostly free data). I've just finished a
rigorous rebuild of my strategy-promotion gate and an honest sweep of my candidate strategies —
and almost everything has died. The only surviving edge is a simple trend (TSMOM) sleeve. I want a
**world-class quant's brutally honest read on what to try next to find alpha**, given the data I
actually have, the constraints I actually operate under, and everything I've already killed.

**Please act as a world-class quant researcher/dev. Be brutally honest and reason from first
principles. I would rather hear one real idea than ten hopeful ones. If the honest answer is
"trend is most of what's harvestable on your data" or "you need data you don't have," say so.**

## What I'm attaching
- **`02_STATE_SNAPSHOT.md` — read this first.** Self-contained: the data I actually have (with real
  coverage), every strategy I've tried and the exact number that killed it, what's live, how I now
  validate (the gate), and my binding constraints. Most of your context is here.
- **`01_PROMPT.md`** — the full ask + the output format I'd like.
- **`files/`** — the source docs if you want to dig deeper: `DATA_PROVIDERS.md` (my data envelope),
  `RULER_V2_DESIGN.md` (the gate any idea must pass), `ML_EXPERIMENT_LOG.md` + `DECISIONS.md` (the
  full kill ledger — large), `ALPHA_V7_SYNTHESIS_AND_PLAN.md` + `NEXT_PHASE_BLUEPRINT_2026-06.md`
  (my current direction — pressure-test it), `PIPELINE_ARCHITECTURE.md` (the WF/CPCV harness — large),
  `MODEL_STATUS.md`, `MASTER_BACKLOG.md`, `SYSTEM_BEHAVIOR.md`.

## If you can only take a little context
Paste **`02_STATE_SNAPSHOT.md` + `01_PROMPT.md` + `files/DATA_PROVIDERS.md` + `files/RULER_V2_DESIGN.md`**
— that's the minimum to give a grounded answer. The four large docs (ML_EXPERIMENT_LOG, DECISIONS,
PIPELINE_ARCHITECTURE, and the blueprint) are for deeper digs.

## What I want back
A brutally honest assessment of my current state, then a **ranked list of 3–5 concrete next research
bets** (each: thesis / do-I-have-the-data / how to test under my harness / expected Sharpe /
why-it-might-fail / effort), a view on the most credible path from trend-only to a 3–5 sleeve
risk-premia book, an explicit "what NOT to pursue" list, and your single highest-conviction bet.
Specifics over generalities — I know the textbook; tell me what *you'd* do with *my* data.
