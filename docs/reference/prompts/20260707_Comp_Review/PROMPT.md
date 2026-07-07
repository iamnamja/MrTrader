# PROMPT — paste this AFTER the three context files (01, 02, 03)

## Your role
You are a **skeptical, world-class quantitative researcher and portfolio manager** advising a **solo
retail systematic trader** running a ~$100k paper account (Alpaca equities + an IBKR paper futures
account). You have just read this trader's full system state (01), their complete strategy kill-list and
validation methodology (02), and their own deep-dive hypotheses (03). Your job is **not** to encourage
them and **not** to brainstorm generic strategies. It is to give the single most useful, evidence-grounded,
intellectually honest assessment of **why they keep finding no durable second edge, and what — if anything
— would change that**, with specific attention to whether their *method* (not their ideas) is the problem.

Be adversarial with their hypotheses. Be concrete. Cite mechanisms. If the honest answer is "stop hunting
and compound the one edge you have," say so and defend it.

## The situation (grounded in files 01–02)
- **One validated live edge:** ETF trend (TSMOM), post-2015 Sharpe ~+0.77 (2020s +1.05). Plus a cash/
  T-bill sleeve. That is the entire live book (paper). It is a **risk premium, not alpha.**
- **~28 strategy families evaluated; almost all killed.** Cross-sectional ML ranker (null ×3), PEAD
  (demoted at event level), ~6 futures factors (only carry + xsmom survived, then the carry+xsmom **book
  just failed the diversification gate on the 16 tradeable IBKR markets**: Track-B t 2.61 → −0.20),
  short-interest, options-as-signal (5 factors), overnight, turn-of-month, ETF relative-value, sector
  rotation, rates carry, CoT, VRP — all killed, parked, or dropped. See the kill-list in file 02.
- **A deliberately strict gate:** purged/embargoed CPCV + Deflated Sharpe (N_trials=300) + a
  selection-aware empirical null-zoo + a **Track-B residual-alpha-vs-the-live-trend-book** test (the
  "is it a real *diversifier*" bar). Details + self-admitted limitations in file 02.
- **A static, weekly system with de-risk-only reactivity:** weekly Monday rebalance; regime detection and
  VIX/credit/curve/drawdown governors exist but only *cut* exposure — nothing rotates *into* a different
  strategy (because there is only one return strategy). Architecture in file 01.
- **The owner's vision:** a more robust system that does not just trade on a weekly schedule but is
  **reactive to current market conditions and adapts** — and a suspicion that **relying on one realized
  history (e.g. COVID's outsized impact) is the wrong way to validate.**

## The questions — answer ALL SIX, in order, with a heading each

**Q1 — Is the null result about THEM or about the MARKET?**
Given the unusually strict gate and the 28-family kill-list, are they (a) *correctly* finding that durable
retail-accessible alpha is rare; (b) *over-rejecting real-but-regime-conditional edges by demanding
UNCONDITIONAL performance*; (c) *signal-mining when they should be mechanism-first*; or some mix? Adjudicate
with evidence from the kill-list. What in the kill-list pattern most supports your view?

**Q2 — Are they validating history wrong, and what would fix it?**
Critique the validation stack (file 02). Is over-reliance on backtest-Sharpe-over-one-realized-path the
core weakness? Specify the highest-value additions — mechanism-first screening, **regime-conditional**
performance decomposition, **synthetic/bootstrapped/stress paths**, live-forward weighting — and give a
concrete example of a fragile strategy that would **pass** their current gate but **fail** a better one.

**Q3 — The regime-conditional reframe (their sharpest untested idea).**
Are they failing because they demand edges that work in ALL regimes, when the accessible edges are
**conditional** (mean-reversion/dispersion in calm-ranging, momentum in trending, defensive/carry-off in
stress)? Name the specific *conditional* strategy families a solo retail trader can actually access and
validate, the **mechanism** and **counterparty** for each, and — critically — **how to hunt/validate them
WITHOUT overfitting regime labels** (labels are in-sample; timing is hard). If this reframe is a mirage,
say why.

**Q4 — The adaptive architecture.**
They already have regime detection + reactive de-risk governors + ONE strategy. Given the constraint
*"adaptation amplifies a diverse strategy set but cannot create edge from one strategy,"* what is the
highest-EV path and in what ORDER: (a) make the single trend edge more **antifragile** via richer
condition-responsive sizing/gating; (b) build a **regime → strategy-selection** layer (only useful once
there are ≥2 uncorrelated strategies); (c) both? Then sketch the concrete redesign of the PM/RM/Trader
agents into a state-aware architecture (note: their live sleeves currently BYPASS the risk manager).

**Q5 — Data & the honest ROI.**
Added data (CoT, options factors, short interest) has repeatedly produced NO edge for them. What data, if
any, would feed a genuine **mechanism** (real options *positioning*/dealer gamma, cross-asset regime
signals, flow/microstructure) and be **accessible + affordable** to a solo retail trader — versus: the
binding constraint is not data at all, it's capital + patience + process. Take a side.

**Q6 — The brutal meta-question.**
Should they even be hunting a second edge right now? Their own prior 10-LLM panel said "compound the one
edge, harden, stop chasing a fifth sleeve." Defend or refute that verdict **in light of** the
regime-conditional/adaptive reframe (Q3/Q4) — i.e., is there a genuinely NEW reason to keep searching, or
is patience + compounding + robustness-hardening the honest highest-EV move? Give a single, unambiguous
recommendation.

## Answer format (required)
1. **One-paragraph verdict up front** — your single most important message.
2. **Q1–Q6**, each under its own heading, evidence-cited, concrete.
3. **Ranked action list** — the top 3–5 things this trader should actually DO next, in priority order,
   each with the mechanism/rationale and the expected failure mode.
4. **What you'd BET on** — if forced: does this trader find a deployable second edge in the next 6 months,
   yes/no, and what's the single highest-probability path if yes.
Keep it rigorous, not long-winded. No hedging boilerplate. If you don't know, say so.
