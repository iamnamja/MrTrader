# 06 — LLMs in the loop: should they be, and where?

A first-class question for this design. We want a concrete, opinionated answer — not "it depends."

## Where we already use LLMs (today)
- **News / Macro Intelligence (NIS):** Claude classifies macro regime (Tier-1) and per-symbol news
  materiality/sentiment (Tier-2); these feed **sizing multipliers** and some entry/exit gates.
- **Risk-reject explanations:** when the RiskManager vetoes a proposal, it can fetch a Claude
  explanation (advisory/audit, not a control).
- **Design & research:** panels like this one; synthesis of strategy reviews.
- Every call is logged to `llm_call_log` (tokens/cost/latency). LLMs are **not** in the deterministic
  sizing/execution path today.

## The question
For a state-of-the-art portfolio brain, **does it make sense to have LLM inputs in the live
process, and if so, exactly where do they sit?** We want you to map the decision/operation surface
and place LLMs where they add genuine value and **explicitly fence them out where they're dangerous.**

Consider (at least) these candidate roles — tell us which are worth it, which aren't, and why:
1. **Research / design advisor (offline):** strategy ideation, architecture review, code review,
   synthesis (what these panels do). *Likely yes — but is there more?*
2. **Narrative / event risk monitor (near-real-time):** scan news/filings/macro for regime shifts,
   tail-risk catalysts, single-name event risk the rules don't encode → feed a **bounded** de-gross
   signal or an alert. Where exactly, and how bounded?
3. **Anomaly / sanity layer over the deterministic decisions:** an LLM that *reviews* the brain's
   proposed book each cycle ("does this set of trades make sense given the state/news?") and can
   **flag/hold** (not place) — a second pair of eyes with veto-to-human, not veto-to-market.
4. **Post-trade / reconciliation analyst:** explain fills, slippage, tracking error, reconciliation
   breaks; draft the daily/weekly book commentary and the live-vs-backtest divergence narrative.
5. **Regime narration / state summarization:** turn the consolidated book state + market context
   into a human-readable situational brief for the operator each morning.
6. **Operator copilot / natural-language control:** let the operator query the book ("what's my net
   equity beta across venues?") and propose (human-confirmed) actions.
7. **In the deterministic sizing/execution path itself** (LLM decides weights/orders): we are
   **skeptical** — argue for or against, hard.

## Our priors / constraints (push back if we're wrong)
- The **sizing & execution path must stay deterministic, reproducible, and auditable.** Our instinct
  is LLMs belong **around** that path (research, monitoring, narrative, anomaly-flagging,
  post-trade), **not inside** it (no LLM directly choosing position sizes or sending orders).
- Anything an LLM influences live must be **bounded** (can only de-risk / flag / alert, never
  increase exposure or place orders autonomously), **fail-safe** (LLM unavailable/uncertain → the
  deterministic system runs unchanged), **logged**, and ideally **human-in-the-loop** for any action
  beyond de-grossing.
- Cost/latency are minor (weekly book), so the constraint is **correctness & trust**, not speed.

## What we want from you
- A clear map: **LLM-IN (and the exact integration point + guardrails)** vs **LLM-OUT (and why).**
- For each "IN" role: the **integration pattern** — input it sees, the bounded action it can take,
  the fallback if it's wrong/unavailable, how it's audited, and whether it's autonomous-bounded vs
  human-confirmed.
- The **failure modes** of LLM-in-the-loop you'd most fear (hallucinated risk-off that whipsaws the
  book; prompt-injection via news; silent drift; over-trust) and the design that prevents them.
- Whether, given all that, an LLM **monitoring/anomaly/narrative layer is worth building now**, or
  whether it's a distraction from getting the deterministic brain right first. Be honest if the
  answer is "not yet."
