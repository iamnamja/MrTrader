# External Quant Review Prompt (paste into other LLMs)

> **How to use:** Paste everything below the line into a fresh LLM session. For best results,
> also attach these repo docs (they're the source of truth): `docs/living/PIPELINE_ARCHITECTURE.md`,
> `docs/living/MODEL_STATUS.md`, `docs/living/DECISIONS.md`, `docs/reference/OPTIONS_DATA.md`,
> and `docs/living/ML_EXPERIMENT_LOG.md`. If you can't attach files, the briefing below is
> self-contained enough to engage.

---

You are a **world-class quantitative developer and researcher** — you have personally designed,
built, and run multiple production automated trading systems, several of them larger and more
profitable than the one described below. You think in terms of edge, capacity, statistical
power, leakage, and execution realism, and you are ruthlessly honest: you would rather tell me a
strategy is noise than let me deploy beta as alpha. I want your candid, senior critique — not
encouragement. Where you see something wrong, say so plainly and explain why.

## What I'm asking for

Review the system below and tell me how to make it **materially better**. I care about four
things specifically, but range wider if you see something important:

1. **Models & signals on the new options data.** We now have ~4 years of survivorship-safe,
   point-in-time daily OPRA option bars for the Russell-1000 universe (details below). Short-vol
   strategies we tried look like a *risk premium, not alpha*. What genuinely *alpha-shaped*
   uses of this options data would you pursue (e.g., dispersion, skew/term-structure signals,
   vol-risk-premium-as-a-feature rather than a sleeve, gamma/dealer-positioning proxies,
   options-implied inputs into the equity event edge)? What's realistically extractable given
   our data limits (no historical IV/greeks/OI/NBBO — we compute greeks; we mark off EOD close)?

2. **Is the backtester / walk-forward actually built to find alpha — and not distort the
   output?** Scrutinize the validation methodology (described below) for leakage, look-ahead,
   multiple-testing/overfitting, gate mis-calibration, or any way the harness could be flattering
   or *suppressing* a real edge. Is CPCV being used correctly here? Are the gates measuring the
   right things? Is anything in the simulator (fills, costs, MTM, purge/embargo) biasing results?

3. **Architecture / design — now that options data is in the mix, what's missing or worth
   building?** Where is this system under-built, over-built, or structurally fragile for a
   serious quant operation? What would you add, remove, or re-architect?

4. **Overall: how would you make this better?** Prioritized. If you were dropped in as head of
   research/engineering tomorrow, what are the first 5 things you'd change, and why?

Be concrete and prioritized. Flag what you'd **kill**, what you'd **double down on**, and what
you'd **build next**. Call out anything that looks like a rookie mistake or a landmine.

---

## System briefing

**What it is.** A single-operator automated **equities + ETF** trading system (paper-trading on
Alpaca, US market hours), Python. Three cooperating agents — **Portfolio Manager** (selects/sizes),
**Risk Manager** (10-rule approval chain), **Trader** (execution) — driven by an APScheduler
orchestrator. Capital is split across **sleeves**; sleeve weights set by a (currently fixed,
gate-controlled) allocator. Goal: find and run **capital-grade alpha**, not collect beta.

**Sleeves / strategies currently live or studied:**
- **PEAD (post-earnings-announcement drift)** — the one *validated* edge and the sole live capital
  strategy. Rules-based, long-biased, ~5-day hold, entry-gated. Honest verdict from our own work:
  **real but underpowered** — event-clustered bootstrap p≈0.19, Newey-West HAC t≈1.04, realized
  Sharpe ~0.40; the per-fold CPCV t-stat (≈2.26) *overstated* significance by treating ~8
  overlapping folds as independent when the true unit is ~19 quarterly earnings clusters. We
  concluded a single event sleeve is **cluster-rate-limited** (~4 independent clusters/yr) and
  structurally can't reach standalone significance on a useful horizon.
- **TSMOM trend sleeve** — a 10-ETF inverse-vol long-flat time-series-momentum basket, weekly
  Monday rebalance; rules-based; crisis diversifier (standalone backtest +0.71 Sharpe 2007–26).
- **Regime model** (`regime_model_v5`) — XGBoost 3-class (RISK_OFF/CAUTION/ON) macro classifier
  (VIX term structure, credit spreads, breadth); feeds a live sizing multiplier. Retrains weekly.
- **Cross-sectional ML ranker (swing) — KILLED.** Long-only looked like +0.22 Sharpe but
  collapsed to noise (t≈0.18) once made genuinely dollar/sector-neutral high-breadth → it was
  **confirmed market beta**, not alpha. The whole ML-ranking line is closed. Intraday 5-min ML
  was also negative (cost-drag). We now believe alpha lives in the **event-driven family**, not
  cross-sectional ML on price/fundamental features.
- **Options program (Alpha-v5) — PAUSED.** Built the data + pricing/greeks engine + a
  contract-level options simulator, then tested: earnings IV-crush short-vol (**killed** —
  negative single-name VRP), index/ETF systematic short-vol (**killed standalone** — VRP real and
  cost-robust but Sharpe-weak/underpowered), and an options-implied "priced-in" filter for PEAD
  (improved PEAD at one threshold but a robustness sweep showed it's **threshold-fragile /
  overfit-suspect** — the lift exists only at ratio=1.0 and inverts at 1.25). Key learning we
  want challenged: *our promotion gate is an alpha gate, and short-vol is a risk premium, so it
  structurally won't clear — but maybe we're framing the options edge wrong.*

**The validation pipeline (this is the part I most want stress-tested):**
- Two engines: standard **walk-forward (WF)** and **CPCV (combinatorial purged cross-validation)**;
  daily mark-to-market equity in the simulators (swing + intraday + a contract-level options sim).
- **Leakage controls:** purge (85 calendar days swing / 2 trading days intraday) + embargo between
  folds; PIT-stamped features; survivorship-safe universes (Russell-1000 union incl. delisted);
  options data is survivorship-safe (universe built from the daily flat files that actually traded)
  and PIT (`knowable_date = trade_date + 1 business day`).
- **Significance-first two-tier gate** (`GATE_MODE`): we moved away from a raw mean-Sharpe bar.
  Diagnostics computed per run: **deflated/“Deflated Sharpe Ratio” (DSR)** with an explicit
  `N_TRIALS_TESTED≈250` multiplicity penalty; **path t-stat** with a warning that N_eff = #folds,
  not #CPCV paths (paths reuse folds and are correlated); a **residual-alpha (CAPM/HAC)**
  diagnostic — does the OOS book survive hedging out SPY? (t<1 ⇒ mostly beta); a **fold-coverage**
  year×regime map (flags bull-skewed fold sets); a **regime gate** (must not fail in the worst
  regime). For event strategies we additionally run **event-clustered block-bootstrap** and
  **Newey-West HAC** significance where the unit of independence is the *event*, not the fold/day.
- **Approximate gate thresholds** (swing / intraday): avg Sharpe ≥0.80/≥1.00 (legacy, being
  retired in favor of significance), min-fold Sharpe ≥ −0.30, DSR p>0.95, profit factor ≥1.10,
  Calmar ≥0.30. A “deployment-adjusted” Sharpe and PAPER-vs-promotion two-tier gate exist.
- **Costs/fills:** parameterized slippage (3bps entry / 5bps stop) + transaction cost (5bps);
  options sim marks each leg to its real EOD close, settles at intrinsic at expiry, and models +
  stress-tests the bid/ask spread (no historical NBBO).

**The options data we now hold locally (the new asset):**
- Source: **Polygon Options Developer** ($79/mo), S3 flat files `us_options_opra/day_aggs_v1`.
- ~**112M daily option bars**, **733 underlyings** (Russell-1000 union + index ETFs),
  **2022-06-09 → 2026-06-08 (~4 years — the maximum the subscription provides)**, ~6.2M distinct
  contracts. Survivorship-safe + PIT by construction.
- **Hard limits:** historical **IV / greeks / open-interest / NBBO are NOT available** — we
  *compute* greeks (Black-Scholes / Bjerksund-Stensland / CRR) from close + spot + r + q, mark
  off EOD close, and never use OI (liquidity via volume/notional). Current-chain IV/greeks/OI
  snapshot exists via REST but is used only for live/validation, never in backtests.

**Constraints / context:** single operator, paper-trading today with a path to live; modest
capital; no co-location/HFT ambitions; the bar for deploying real money is **statistically
credible, beta-neutral alpha that survives costs and out-of-sample**, not a backtested Sharpe.

---

## Output I want from you

1. **Verdict on the validation harness** — is it sound, and where could it be (a) leaking/inflating
   or (b) *hiding* a real edge? Concrete fixes.
2. **The 3–5 highest-expected-value research directions** given our data + constraints, ranked,
   with the rough mechanism, why it could be real alpha (not beta/risk-premium), and how you'd
   test it without fooling yourself.
3. **Best uses of the options data specifically** (alpha-shaped, given our data limits).
4. **Architecture/design gaps** — what to build, remove, or re-architect for a serious operation.
5. **The first 5 things you'd change**, prioritized, with reasoning.

Assume I can implement anything you propose. Be specific enough to act on. Where you're uncertain,
say what evidence would change your mind.
