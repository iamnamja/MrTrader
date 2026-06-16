# Deep-dive quant review request — MrTrader (Alpha-v8)

You are a **world-class quantitative researcher / portfolio manager at a top-tier systematic hedge
fund** (think Renaissance / Two Sigma / DE Shaw / AQR pedigree). You have personally built and run
fully-automated trading platforms end-to-end: signal research, validation harnesses, portfolio
construction, risk, and live execution. You have killed far more strategies than you've shipped, and
you have strong, specific opinions about what actually generates durable, capacity-aware alpha versus
what is overfit noise.

I'm a solo operator who has built a fully-automated systematic trading platform ("MrTrader") and
spent ~6 months hunting for alpha. The attached files describe the **entire system** in detail:
- `02_SYSTEM_DESIGN.md` — architecture, the three agents (Portfolio / Risk / Trader), the full
  trading flow, model training, and the walk-forward + CPCV validation harness with our promotion
  gate ("Ruler-v2").
- `03_RESEARCH_HISTORY.md` — **everything we've tried and the verdict on each** (swing ML, intraday
  ML, PEAD/earnings, options-conditioned signals, a cross-asset premia book, overlays, a futures
  POC), plus the story of early harness bugs and how we re-validated.
- `04_DATA_INVENTORY.md` — exactly what data we have, its depth and cost, and what we think is missing.
- `05`–`09` — deeper appendices (pipeline SSOT, runtime agent behavior, the gate design, and the two
  prior research-program writeups).

**Please read all of it before responding.** Then give me your honest, rigorous assessment.

## The blunt summary you're reacting to
After 6 months and many rigorous experiments, **the only validated standalone edge we found is
TREND** — a 10-ETF time-series-momentum sleeve (deep-history Sharpe ≈ 0.7). It is live (paper,
$100k, 25% allocation + cash, with a VIX-term-structure crash governor). **Essentially every
additive equity signal we tried — daily ML, intraday ML, PEAD, options signals, short-interest,
credit/curve timing, carry — was killed** by our validation gate or failed a stability guard. We've
concluded (4 times over) that **free daily US equity data is largely mined out for additive alpha.**
Tell us where we're wrong, or where the remaining edge actually is.

## What I want from you (be specific and opinionated — no generic platitudes)

### 1. Critique
- Tear apart our **methodology and validation**. Is the WF/CPCV + Ruler-v2 gate sound, or is it
  (a) still leaking look-ahead anywhere, or (b) so conservative it's a Type-II / false-negative
  machine that's killing real edges? Where exactly?
- Critique the **architecture** (three-agent PM/RM/Trader split, the Sleeve Lab, overlays vs sleeves).
- Critique our **conclusions** — especially "free daily US equity data is mined out." Do you buy it?
- Call out anything that looks like overfitting, p-hacking, survivorship/look-ahead, unrealistic
  cost/fill assumptions, or capacity self-deception.

### 2. Where is the alpha? — give us MANY concrete, distinct strategy ideas to test
This is the most important section. **List as many distinct, testable trading-strategy ideas as you
can** — we want a broad menu to prioritize and run through our harness. For each idea, briefly give:
- the **economic rationale** (why an edge should exist and persist),
- the **instruments / universe** and **horizon**,
- the **data** it needs (and whether that data is cheap/retail-accessible),
- the **signal construction** sketch,
- expected **capacity, turnover, and failure modes**,
- and a rough **priority** (quick-win vs research-heavy).

Span the space — don't just give us more trend variants. Consider e.g.: cross-asset & macro,
carry/term-structure, volatility & options structures, relative-value / stat-arb, event-driven,
seasonality/calendar, flow/positioning, intraday microstructure, crypto, alternative data, and
regime/timing overlays. **Surprise us with non-obvious ones.**

### 3. Data — what are we missing?
What datasets would most move the needle, given a solo/retail budget (a few hundred $/yr is fine;
flag anything that genuinely requires institutional spend)? What's the single highest-ROI data buy?

### 4. Models — what should we try?
Concrete ML/modeling approaches worth testing given our constraints (small-N, low-signal regime):
features, model classes, target design, ensembling, regime-conditioning, position-sizing/meta-labeling,
portfolio construction. What modeling mistakes are we likely making?

### 5. Redesign (optional but welcome)
If you think the whole approach is wrong, **propose a complete redesign.** What would *you* build
from scratch with our constraints (solo operator, retail data budget, Alpaca execution — equities/
ETFs/options/crypto, no futures without a second broker)?

## Constraints to keep your advice actionable
- Solo operator; paper at $100k notional; retail infra.
- Data budget: a *modest* paid subscription is fine (~$100–300/yr has already paid off once);
  please flag anything needing institutional budgets so I can weigh it.
- Execution = Alpaca (US equities/ETFs/options/crypto; **no futures** — that needs a 2nd broker like IBKR).
- We already enforce: hypothesis pre-registration, CPCV with purge/embargo, realistic transaction
  costs, a both-halves stability guard, and an independent adversarial review before any verdict.
  Hold us to (or above) that bar.

## Output format
1. **Verdict (1 paragraph):** is this platform on a path to durable alpha, yes/no, and the single
   biggest thing you'd change.
2. **Methodology & architecture critique** (bulleted, specific).
3. **Strategy idea menu** — the big section; as many distinct ideas as you can, in the per-idea
   format above, roughly ranked.
4. **Data gaps & top buys.**
5. **Modeling recommendations.**
6. **Redesign (if warranted).**
7. **Your top 5 things I should do next, in priority order.**

Be direct. If something we did is naive or wrong, say so plainly. I would rather hear the hard truth
now than discover it in live P&L.
