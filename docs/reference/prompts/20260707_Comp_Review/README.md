# Comprehensive Review — "Why no alpha, and how do we build a robust, adaptive system?" (2026-07-07)

## Why this review exists
Over ~6 weeks and ~28 evaluated strategy families we have found exactly **one** durable, live edge
(ETF trend) plus a cash sleeve. Every attempt to find a **second, uncorrelated** engine has failed a
deliberately strict gate — most recently the carry+xSMOM futures book, which passed standalone but
**failed the diversification gate on our tradeable universe** (Track-B t 2.61 → −0.20 on 16 markets).
The owner's question is not "give us more sleeve ideas" — prior panels answered that and converged on
*"you hold risk premia not alpha; stop hunting; let the live track record accrue."* The question now is
sharper and reframes the whole search:

> Are we **coming up empty because durable retail alpha is rare**, or because our **method is wrong** —
> validating on one realized history, demanding **unconditional** edge, signal-mining instead of
> mechanism-first, and running a **static weekly** system when the real edges may be **regime-conditional
> and adaptive**? What would a more robust, condition-reactive system actually require — different data,
> different validation, or a different PM/RM/Trader architecture?

## How to run the panel
1. **Assemble the context** — give each LLM, in order: `01_CONTEXT_system_and_data.md`,
   `02_evidence_killlist_and_validation.md`, `03_deep_dive_hypotheses.md`, then `PROMPT.md`.
   (01–03 are self-contained — external models cannot read the repo, so everything they need is pasted.)
2. **Run ≥5 independent models** (e.g. ChatGPT / Claude / Gemini / Grok / DeepSeek) + optionally 1–2
   **repo-grounded** agents that actually read `app/research/family_registry.py`, the validation code,
   and the roll/gate scripts (to keep the external panel honest).
3. **Save each raw response** verbatim into `responses/<model>.md` (create the subdir).
4. Claude then **synthesizes** them into a decision doc (SSOT), following the anti-patterns below.

## Anti-patterns (enforce these — prior panels drifted into cheerleading)
- **No generic idea lists.** "Try mean-reversion / options / crypto" without a *mechanism*, a *regime it
  works in*, and *why it survives our gate* is worthless — we've killed 14+ such ideas already.
- **Attack our hypotheses (file 03), don't validate them.** Each H1–H6 is a claim to *refute or sharpen*.
- **Cite mechanism + who's on the other side.** Any proposed edge must name the structural reason it
  persists and who is forced to trade against us.
- **Flag overfitting explicitly**, especially for anything regime-adaptive (labels are in-sample).
- **Answer the meta-question honestly:** is the binding constraint *ideas* or *capital + patience +
  process*? If it's the latter, say so — that is a valid, valuable answer.

## File manifest (this pack — 5 files, self-contained)
| File | Contents |
|---|---|
| `README.md` | this — purpose, run instructions, anti-patterns |
| `PROMPT.md` | the prompt to paste into each LLM: role, rules, the 6 questions, answer format |
| `01_CONTEXT_system_and_data.md` | the account, live book, PM/RM/Trader architecture, the reactive infra we ALREADY have, the data feeds we have + lack |
| `02_evidence_killlist_and_validation.md` | the ~28-family kill list (the evidence of "empty-handed") + our validation stack + its self-admitted limitations |
| `03_deep_dive_hypotheses.md` | the owner-side deep dive as **6 attackable hypotheses** (why-no-alpha, validation, regime-conditional reframe, adaptive architecture, data ROI, the meta-question) |

## Further reading (repo docs, for a repo-grounded agent only)
`docs/living/DECISIONS.md`, `docs/living/ML_EXPERIMENT_LOG.md`, `app/research/family_registry.py`,
`docs/reference/ALPHA_V10_SYNTHESIS_AND_PLAN.md`, `docs/reference/GL0_GL1_FINDINGS_2026-06-21.md`,
`docs/reference/PORTFOLIO_BRAIN_ROADMAP_2026-06-21.md`, `app/strategy/regime_detector.py` +
`app/strategy/crash_governor.py` + `app/strategy/credit_curve_governor.py` (the reactive infra).
