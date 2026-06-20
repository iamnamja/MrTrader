You are a **world-class hedge-fund quant developer / PM** — the kind who has built and killed
real systematic strategies with real money, sat on investment committees, and reviewed other
quants' work for a living. I'm the solo operator of a systematic trading platform ("MrTrader")
and I want a **brutally honest** outside assessment. Do not flatter me, do not hedge, do not
give generic advice. If my edges are weak or overfit, if my process is fooling me, or if my
conclusions are wrong, **say so plainly and tell me why.** I would rather hear "this is mostly
beta and you're kidding yourself" than a polite list of platitudes.

I've attached 7 files giving you the full state of the program. Please read them before answering:
- **01_PROGRAM_OVERVIEW** — what it is, what's live, the 2-year journey
- **02_KILL_KEEP_LEDGER** — every strategy we've tried + our verdict + why
- **03_CURRENT_EDGES_AND_NUMBERS** — the two edges that matter (trend live; futures carry new) + numbers + caveats
- **04_DATA_INVENTORY** — what data we own (equities, 4y options, survivorship-free futures) + gaps
- **05_ARCHITECTURE** — the live system + research pipeline
- **06_VALIDATION_HARNESS** — our gate/CPCV/pre-registration, and our own known failure modes
- **07_RECENT_FINDINGS_AND_CAVEATS** — the last ~2 weeks + where the bodies are buried

## Context in one breath
After two years of rigorous, pre-registered search, the **only edge we trade live with conviction
is multi-asset trend (via ETFs, Sharpe ~0.72)**. Almost everything else (PEAD, cross-sectional
ML, options-as-signal, overnight, calendar, ETF stat-arb, rates carry, short-interest) was killed
as beta, a misframed risk premium, or cost-killed. We just bought survivorship-free **futures
data** and found what looks like a genuinely new, modern, diversifying edge — **futures carry**
(Sharpe ~0.55-0.66 after honest caveats; improves the book; not yet live). We now have
**equities + 4y options + survivorship-free futures** data. Our goal: an app that is **resilient
AND finds real alpha.** We are open to new strategies, new data/assets, and **re-running anything**
if you think our results aren't what they should be.

## What I want from you — please be specific, prioritized, and implementable
1. **Brutally honest assessment of our state.** Are we finding real alpha or fooling ourselves?
   Is "free daily US-equity directional alpha is mined out" a sound conclusion or an artifact of a
   still-miscalibrated ruler / too-short data / wrong framing? (See 06.)
2. **Our chosen edges (trend + carry):** do the results match your priors? Anything smell off,
   over-/under-stated, or overfit? **What would you re-run, and exactly how, to settle your doubts?**
   Is adding carry-to-trend the right next deployment, or are we missing a better combination?
3. **Strategies we HAVEN'T tried that we should** — grounded in the data we already have
   (equities, 4y options + computed greeks, survivorship-free futures incl. full term structure,
   FINRA short-volume, FRED macro). Prioritize by expected-edge × feasibility. Be concrete enough
   that I could build the top 2-3 (signal definition, universe, how you'd test it, kill criteria).
   Don't suggest things we already killed unless you think we killed them wrongly (say which + why).
4. **Is the architecture sound?** (live orchestrator/agents + research pipeline + two-track gate +
   pre-registration). Resilience risks? What would make this a genuinely robust multi-strategy
   book vs. effectively a single bet? Specific app/infra improvements that matter.
5. **Data / assets:** what's the single highest-EV next data buy or new asset class for us, and why?
   Are we leaving obvious alpha on the table (e.g., by not having survivorship-free equities)?
6. **If you were running this book Monday morning:** what are the top 3-5 concrete moves, in order?

Where you make a claim, say what evidence would confirm/refute it so I can go test it. I have the
data and the harness to re-run almost anything. Thank you — candor over comfort.
