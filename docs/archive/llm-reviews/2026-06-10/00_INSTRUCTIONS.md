# Read me first — instructions for the reviewer

You are a **world-class quantitative researcher/engineer** who has personally designed, built,
and run multiple production automated trading systems — several larger and more profitable than
the one described here. I want a **brutally honest, senior review**: I would rather you tell me a
strategy is noise, or that my whole premise is wrong, than have you be encouraging. Push back on
my conclusions. Think **outside the box** and draw on the **full range of trading methodologies**
— statistical arbitrage, event-driven, volatility/options, market-making/microstructure,
cross-sectional and time-series ML, alternative data, portfolio construction, execution/cost
engineering — not just the paths I've already tried.

## What this is

I run a single-operator automated **equities + ETF** trading system (paper today, with a path to
live). I've done a lot of honest validation work and killed most of what didn't survive. I now
have ~2–4 years of survivorship-safe, point-in-time **options data** and want to decide the next
phase. I'm sending this same package to **several different LLMs** and will synthesize the best
ideas across all of them — so **be specific, opinionated, and differentiated**: tell me what a
generic reviewer would miss.

## The attached files (please read in this order)

1. **`EXTERNAL_QUANT_REVIEW_PROMPT_20260610.md`** — START HERE. The primary brief: what the system
   is, the sleeves/strategies tried and their honest verdicts, the validation methodology, the new
   options data and its hard limits, and the four areas I most want critiqued. The rest are
   supporting evidence.
2. **`PIPELINE_ARCHITECTURE.md`** — the single source of truth for the backtester: walk-forward +
   CPCV engines, the simulators, the full **gate inventory** (significance-first two-tier gate,
   DSR/multiplicity penalty, purge/embargo, residual-alpha-vs-beta diagnostic, regime gating),
   known limitations, and a changelog. **This is the doc to scrutinize hardest for leakage,
   look-ahead, multiple-testing, or gates that flatter or suppress a real edge.**
3. **`MODEL_STATUS.md`** — what's actually live/paper right now, each model's version and gate
   results (e.g., PEAD's honest CPCV numbers, the regime model, the dead ML ranker).
4. **`DECISIONS.md`** — the append-only decision log: *why* things are the way they are, including
   why the cross-sectional ML line was closed, why the options program was paused, and the
   honest negative/parked results (OPT-3/4/5).
5. **`OPTIONS_DATA.md`** — the options data design + PIT/survivorship contract and the **hard data
   limits** (no historical IV/greeks/OI/NBBO — we compute greeks, mark off EOD close).
   **`OPTIONS_DATA_SNAPSHOT.md`** — the exact current inventory (~112M bars, 733 names,
   2022-06→2026-06, ~6.2M contracts) and what is / isn't testable on it. Read alongside #5.
6. **`OPTIONS_PROGRAM.md`** — the options program's plan/charter and where each sub-project (OPT-0…8)
   landed.

(If you want the full append-only experiment journal, `ML_EXPERIMENT_LOG.md` exists too, but it's
long — the above is sufficient to engage deeply.)

## What I want back

Take your time and think carefully and deeply about **what our next steps should be.** Then return
**one self-contained Markdown file I can download** (so I can feed it back and compare across
models), structured roughly as:

1. **Verdict on the validation harness** — is it sound? Where could it be (a) leaking/inflating or
   (b) *hiding* a real edge? Concrete, prioritized fixes.
2. **The 3–5 highest-expected-value research directions** given my data + constraints — ranked,
   each with the rough mechanism, *why it could be real alpha (not beta/risk-premium)*, and how
   you'd test it without fooling yourself.
3. **Best alpha-shaped uses of the options data specifically** (given the data limits).
4. **Architecture / design gaps** — what to build, remove, or re-architect for a serious operation.
5. **The first 5 things you'd change**, prioritized, with reasoning — and anything you'd flatly
   **kill**.

Assume I can implement anything you propose. Where you're uncertain, say what evidence would change
your mind. If you think the whole objective (a single-operator system chasing capital-grade alpha)
is misframed, say that too — and tell me what the right objective is.
