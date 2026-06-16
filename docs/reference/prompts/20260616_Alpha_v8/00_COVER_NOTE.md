# Cover note — MrTrader external quant review (2026-06-16)

**Paste `01_PROMPT.md` as your chat message, and attach the rest of the files in this folder.**
(If a model can't take many attachments, attach in priority order: `02`, `03`, `04`, then `05`–`09`.)

## What this is
MrTrader is a solo-built, fully-automated systematic trading platform (US equities/ETFs via
Alpaca paper, $100k notional). Over ~6 months it has hunted for alpha across swing ML, intraday
ML, event-driven (PEAD), options-conditioned signals, and a cross-asset premia book. **The honest
result: only TREND (a 10-ETF time-series-momentum sleeve) is a validated standalone edge; almost
everything else has been rigorously killed.** We want a brutally honest, world-class-quant review:
critique the whole approach, and suggest where real alpha could still come from.

## What we want from you
A deep-dive review acting as a **world-class quant at a top systematic hedge fund** with experience
running automated trading platforms. Specifically (detailed in `01_PROMPT.md`):
1. **Critique** our design, methodology, gate/validation, and the conclusions we've drawn.
2. **Suggest many concrete, different trading-strategy ideas** we could test for alpha — be specific.
3. Tell us **what data we're missing** and what to buy, and **what model/ML approaches** are worth trying.
4. You are **free to propose a complete redesign** if you think the architecture is wrong.

## The files (≤10)
- **`01_PROMPT.md`** — the review request (paste this into the chat).
- **`02_SYSTEM_DESIGN.md`** — how the platform works: agents (Portfolio/Risk/Trader), trading flow, model training, the walk-forward/CPCV validation harness + the "Ruler-v2" promotion gate, the "Sleeve Lab". **Start here.**
- **`03_RESEARCH_HISTORY.md`** — everything we've tried and the verdicts (incl. the early-harness-bug story + re-validation), and what's arguably worth retrying.
- **`04_DATA_INVENTORY.md`** — exactly what data we have (feeds, depth, cost) and what's missing.
- **`05_PIPELINE_ARCHITECTURE.md`** — deep SSOT for the WF/CPCV pipeline + gate inventory (appendix).
- **`06_SYSTEM_BEHAVIOR.md`** — runtime PM/RM/Trader behavior detail (appendix).
- **`07_RULER_V2_DESIGN.md`** — the promotion-gate design (appendix).
- **`08_ALPHA_V7_SYNTHESIS.md`** — the *previous* (2026-06-14) 5-LLM review's synthesis + game plan (so you don't repeat it).
- **`09_ALPHA_V8_PLAN.md`** — the most recent research program (overlays + the futures-trend plan) + its results.

## Hard context to respect (so suggestions are actionable, not generic)
- **Budget/scale:** solo operator, paper at $100k, retail-grade infra. Open to a *modest* paid data subscription (~$100–300/yr proven worth it). NOT institutional budgets (no Bloomberg/tick/L2 at scale).
- **Execution:** Alpaca (US equities/ETFs/options/crypto — **NO futures**). Live futures would need a second broker (e.g. IBKR).
- **Discipline we already enforce:** pre-registration of hypotheses, CPCV with purge/embargo, transaction costs, a both-halves stability guard, and independent adversarial review. We are allergic to look-ahead and overfitting — please hold us to that standard and call out anything we've missed.
- We have already concluded **free daily US equity data is largely mined out** for additive alpha. Tell us if you disagree and why, or where the remaining edge is.
