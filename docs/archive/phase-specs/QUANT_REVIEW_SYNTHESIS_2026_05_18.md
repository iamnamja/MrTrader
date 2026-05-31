# MrTrader — Quant Review Synthesis & Prioritized Backlog

**Date:** 2026-05-18  
**Capital:** $100k paper trading  
**Source:** Synthesis of 4 independent expert reviews (DeepSeek, Gemini, ChatGPT, Claude/Opus) + Opus 4.7 analysis  
**Purpose:** Working backlog. Every item here is specific, actionable, has a pass/fail criterion.

---

## Honest Assessment — Where We Actually Stand

**What's built (the good):**
- Solid plumbing: FastAPI + Alpaca + SQLite + agent queue + walk-forward harness + feature cache + paper trading loop
- Disciplined experiment log and explicit gates — we actually fail our own gates instead of cherry-picking
- Real PIT universe handling, TSNorm fitted on train only, purge/embargo

**What's not built (the bad):**
- **Zero strategies have passed an honest walk-forward gate.** 9 LambdaRank runs failed. ~15 XGBoost runs failed. Intraday v51 below gate. Factor portfolio 1.335 Sharpe is unvalidated.
- **Honest execution simulator** — `prev_close × 1.001` and close-only stop checks invalidate every backtest number produced so far
- **Point-in-time fundamental data** — FMP quarterly with uncertain filing-date lag is the single most likely explanation for the 1.335 factor Sharpe

**Highest-confidence problems (all 4 reviewers agree):**

| Problem | Root cause | All agree? |
|---|---|---|
| Entry price `prev_close × 1.001` is wrong | Not using actual next-session open | ✅ All 4 |
| Stop checked only on daily close | Understates stop-outs by 30-60% | ✅ All 4 |
| Factor 1.335 Sharpe is suspect | Fundamental look-ahead + factor beta + period bias | ✅ All 4 |
| Factor portfolio deployed with ATR stops contradicts its design | Monthly-rebalance strategy needs 20-day hold, not 0.5× ATR stop | ✅ All 4 |
| LambdaRank structurally wrong for long-only | No "cash" decision, bear-market label inversion | ✅ All 4 |
| Too many components, none ablated | Any of NIS/regime/OpScore/BenignGate/ATR/rescore may be net negative | ✅ 3/4 |

---

## Convergence Matrix — Where Reviewers Agree vs. Disagree

### Full agreement → treat as facts

- Use actual next-session open for entries
- Check intraday low for stop hits (daily bar H/L)
- Factor 1.335 Sharpe requires honest re-validation with PIT-clean data
- Factor portfolio cannot have tight ATR stops — mismatches its validated economics
- Cross-sectional ranking labels for long-only are broken (bear market = label inversion)
- Need ablation of every system component

### Strong agreement (3/4)

- WF gate of 0.80 avg Sharpe is too strict for long-only equity → realistic gate: 0.55 avg, min fold ≥ -0.40
- ML should be a *meta-model on top of* factor candidates, not primary alpha engine
- Intraday at this cost level is marginal — 15 bps/side + no tick data is very hard

### Genuine disagreements — need test or decision

| Question | Camp A | Camp B | Recommended approach |
|---|---|---|---|
| Kill intraday? | DeepSeek/Claude: yes | Gemini: revive at $100k post-PDT (PDT rule changes June 4, 2026); ChatGPT: separate harness | Mothball — keep paper running, zero new dev investment until swing/factor validated |
| Label fix | DeepSeek/Gemini: absolute return binary (+4%/-3%) | ChatGPT: 3-class {long, skip, avoid} | Start absolute-return binary; evolve to 3-class if it shows signal |
| Next alpha track | Claude: PEAD or sector rotation | Others: fix what we have first | Both — sector rotation as floor benchmark, PEAD as parallel track |
| RISK_OFF handling | Gemini: hard zero positions | Others: size down | Hard zero in RISK_OFF for swing |

---

## Comprehensive Prioritized Backlog

### P0 — Fix before running any new model (this week)

**P0.1 — Fix walk-forward entry price** [0.5 day]
- What: Replace `prev_close × 1.001` with actual next-session open from bars data
- Why: Every WF backtest number is a lie of unknown direction
- Pass: Hand-verify 10 random historical trades match next-day open within 1 bp
- Dep: None. **Do first.**

**P0.2 — Fix stop/target simulation to use intraday H/L** [1 day]
- What: Check daily Low for stop hits, daily High for target hits. If both within same bar, conservatively assume stop hit first.
- Why: Close-only stop checks miss 30–60% of real stop-outs
- Pass: Stop-hit rate rises to 15–25% range on a 0.5× ATR stop (currently 3–5% = obviously wrong)
- Dep: None.

**P0.3 — Audit FMP fundamentals for PIT correctness** [1 day] ⭐ MOST IMPORTANT
- What: For 20 random symbol-quarter pairs in 2018–2024, confirm the value used at date T was actually filed and public by T. Use SEC EDGAR filing dates as ground truth. Add `available_date = filing_date + 1 trading day` lag if missing.
- Why: **Single most likely cause of the 1.335 Sharpe.** If FMP is using reporting-period end dates instead of filing dates, every fundamental-based backtest is contaminated with future data.
- Pass: Audit report showing max look-ahead ≤ 1 trading day for each of 20 rows. If look-ahead > 5 days for any row → critical bug, stop all fundamental-using backtests.
- Dep: None.

**P0.4 — IC analysis of every existing model** [1 day]
- What: For LambdaRank v210, factor composite, XGBoost binary, intraday v51: compute Spearman IC of model score vs. forward 5d/10d/20d realized return, per fold. Plot IC time-series.
- Why: We have never measured whether the signal actually predicts returns. Sharpe through a buggy simulator is 2 layers of indirection. IC is direct.
- Pass: IC ≥ 0.02 average across folds to justify any further ML investment. IC < 0.01 = features are noise, stop tuning.
- Dep: None (uses existing model outputs and historical return data).

---

### P1 — Truth tests — blocks all model work (weeks 1-2)

**P1.1 — Factor portfolio truth test with corrected simulator** [1 day after P0.3]
- What: Re-run factor portfolio with: (a) corrected entry/stop (P0.1+P0.2), (b) PIT-fixed fundamentals (P0.3), (c) **no ATR stops** — time-based monthly rebalance only, matching the validated design
- Why: Determine if the strategy works at all in its native execution mode
- Pass: Sharpe ≥ 0.70 after PIT fix. If drops below 0.50 → fundamentals were the alpha, not the factor scores.
- Dep: P0.1, P0.2, P0.3.

**P1.2 — Factor decomposition (AQR factor regression)** [0.5 day]
- What: Download AQR's free MOM and QMJ factor return series. Regress factor portfolio daily returns on (SPY, MOM, QMJ). Report alpha intercept, t-stat, betas, R².
- Why: Distinguish "we captured MOM+QMJ beta in a bull market" from "we have genuine alpha." A Sharpe of 1.3+ in 2019–2024 that was mostly bull-market MOM/QMJ exposure is not alpha — you can replicate it with MTUM+QUAL ETFs for zero work.
- Pass: Alpha t-stat ≥ 2.0. Below 1.5: this is factor exposure, not alpha. Still tradeable but must be framed honestly.
- Dep: P1.1.

**P1.3 — Factor portfolio component ablation** [1 day]
- What: Run factor portfolio with each component removed: (a) regime gate off, (b) Tier 2 features off, (c) momentum only, (d) quality only, (e) value/PE only, (f) random top-20 control, (g) buy-and-hold SPY benchmark, (h) buy-and-hold MTUM benchmark, (i) SPY + same regime gate benchmark
- Why: Identify what is actually contributing. If random top-20 beats the factor → scoring has no value. If SPY+gate beats everything → the regime gate *is* the alpha.
- Pass: Full factor composite must beat random top-20 by ≥ 0.2 Sharpe and beat SPY+gate by ≥ 0.15 Sharpe. Each factor sleeve (momentum, quality, value) should show positive marginal contribution.
- Dep: P1.1.

**P1.4 — Out-of-sample factor portfolio: 2007–2019** [1 day + data prep]
- What: Run factor portfolio on 2007–2019 (GFC, slow recovery, 2018 rate shock). This data was never used in development.
- Why: 2019–2024 was arguably the best possible sample for momentum+quality. Out-of-sample validation required.
- Pass: Sharpe ≥ 0.40 in 2007–2019. If negative → strategy is overfit to the recent regime.
- Dep: P1.1 + historical data availability (may need extended download).

**P1.5 — Relax WF gate to honest level** [0.5 day]
- What: Change gate to: avg Sharpe ≥ 0.55, min fold ≥ -0.40, IC ≥ 0.03, max DD ≤ 25%. Document and commit. Sanity-check: SPY buy-and-hold should NOT pass this gate.
- Why: 0.80 avg Sharpe long-only is institutional L/S territory. Holding an impossible gate means valid strategies get discarded.
- Pass: Gate is documented, SPY B&H fails it, a realistic strategy with 0.60 avg Sharpe passes it.
- Dep: None. Do this regardless of P1.1–P1.4 results.

---

### P2 — Re-specify the alphas (weeks 3-4)

**P2.1 — StrategyContract abstraction** [3-5 days]
- What: Create a config object every strategy must declare: `{strategy_id, alpha_horizon, rebalance_frequency, entry_rule, exit_rule, position_count, capital_policy, risk_policy, cost_model, data_policy}`. Both the WF simulator and live PM agent consume the same contract. A strategy cannot deploy unless it has a contract.
- Why: Eliminates the class of bug that produced the -1.43 factor result (monthly-rebalance strategy with ATR stops). Structural fix.
- Pass: Factor portfolio runs with `factor_monthly_v1` contract (no ATR stops, monthly rebalance, equal weight) AND swing trades run with `swing_atr_v1` contract (ATR stop/target) through the *same* simulator. Paper vs. backtest drift ≤ 20%.
- Dep: P0.1, P0.2.

**P2.2 — Absolute-return binary XGBoost (replacement for LambdaRank)** [2 days]
- What: Label = 1 if 10-day forward return ≥ +4% AND max intraday drawdown during that period ≤ -3% (absolute thresholds, NOT ATR-relative). Train XGBoost binary classifier with probability calibration. Entry threshold: P(1) ≥ 0.55.
- Why: Absolute threshold is regime-invariant — in a bear market, fewer stocks cross +4% → model naturally reduces positions → acts as its own regime filter. No forced ranking of "least bad" stocks.
- Pass: WF gate (P1.5 revised). IC ≥ 0.03. Must beat factor portfolio baseline.
- Dep: P0.4 (IC analysis), P1.5 (gate reset).

**P2.3 — Three-class meta-labeler on factor candidates** [3 days — only if P2.2 shows signal]
- What: After P2.2 passes, train a 3-class model {long, skip, avoid} on top of the factor portfolio's top-50 candidates. ML's job: "among names the factor model likes, which ones should we avoid or overweight?" This is a meta-filter, not primary alpha.
- Why: Much more learnable problem than primary stock selection. ML adds value as a veto/sizing overlay, not as a ranker of 750 stocks.
- Pass: Factor + meta-filter must beat factor portfolio alone by ≥ 0.20 Sharpe in WF.
- Dep: P2.2 shows signal, P1.1 factor portfolio validated.

---

### P3 — Diversify the alpha hunt (weeks 5-8)

**P3.1 — Sector rotation strategy (11 SPDR sector ETFs)** [2 days]
- What: Monthly cross-sectional 6-month momentum among XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLRE/XLB/XLU/XLC. Top-3 equal-weight, regime-gated (SPY > 200d MA). Hold 1 month. This is the floor that every other strategy must beat.
- Why: Lowest-complexity, lowest-cost, most-replicated equity strategy in academic finance. If we can't match this with our fancy system, the system has no purpose.
- Pass: WF Sharpe ≥ 0.45. Becomes the hard floor benchmark.
- Dep: P0.1, P0.2 (clean simulator).

**P3.2 — Post-Earnings Announcement Drift (PEAD)** [5 days]
- What: Long stocks with earnings surprise > +5% versus analyst consensus. Hold 5–20 days. Exit on time-stop or rescore. Event-driven, regime-independent.
- Why: Single most-replicated equity anomaly. Works because slow institutional capital cannot immediately arbitrage away earnings surprises. Orthogonal to momentum/quality.
- Pass: WF Sharpe ≥ 0.60. Cost-sensitivity check at 2× assumed costs.
- Data needed: Earnings dates + analyst consensus estimates. FMP has this; consider Sharadar for higher quality.
- Dep: P0.3 (PIT discipline mandatory for this strategy too).

**P3.3 — Honest intraday reassessment** [1 day]
- What: Rerun v51 through corrected simulator (P0.1 + P0.2). If Sharpe still ≥ 0.50, keep paper-running and log results. If drops below 0.40, decommission or mothball with zero further dev investment.
- Note: PDT rule is changing June 4, 2026 (FINRA confirmed). At $100k, intraday is no longer legally constrained by day-trade count. But 15 bps/side cost model is still brutal.
- Pass: v51 Sharpe ≥ 0.50 in corrected simulator AND paper vs. backtest drift ≤ 0.30 Sharpe over 90 days.
- Dep: P0.1, P0.2.

**P3.4 — System-wide component ablation** [4 days]
- What: Ablation of every live component in the production pipeline using fractional factorial design (~16 configs instead of 64): NIS on/off, regime gate on/off, BenignGate on/off, opportunity score on/off, ATR stops on/off, daily rescoring on/off, Kelly sizing on/off. Measure Sharpe contribution of each.
- Why: Some of these are likely net-negative. We can't know which without testing.
- Pass: Document Sharpe impact of each component. Remove any component that reduces Sharpe or increases drawdown without compensating benefit.
- Dep: P1.1 (need a baseline first).

---

### P4 — Research / longer horizon

| Item | Description | Effort |
|---|---|---|
| P4.1 | Replace XGBoost regime model with simpler Bayesian approach — less overfit, more interpretable | 3 days |
| P4.2 | Live-vs-backtest tracking dashboard: paper Sharpe rolling 60d, slippage vs. model, hit-rate drift | 3 days |
| P4.3 | Cross-asset overlay: add TLT/GLD allocation as diversifier in RISK_OFF regime | 2 days |
| P4.4 | Position sizing research: equal-weight vs. vol-targeted vs. confidence-Kelly | 2 days |
| P4.5 | Reduce universe to R1000 ex-illiquids (bottom-decile dollar volume excluded) | 1 day |
| P4.6 | CPCV (Combinatorial Purged Cross-Validation) as supplement to current WF | 3 days |
| P4.7 | Deflated Sharpe Ratio (DSR) tracking — penalize for number of trials tested | 1 day |

---

## Questions That Need Your Decision

Before proceeding, answers to these shape the work:

1. **Do you accept the relaxed WF gate (0.55 avg / -0.40 min fold / IC 0.03)?**  
   Without this, every honest long-only result will forever fail. The 0.80 gate was calibrated for L/S institutional, not long-only retail. This is a strategic call.

2. **If factor portfolio post-PIT-fix shows Sharpe 0.40–0.60 — keep or shelve?**  
   It's not the 1.335 we thought. But 0.50 Sharpe long-only is actually respectable. Decision: deploy at reduced sizing, or pivot entirely to P3.1 sector rotation?

3. **Intraday: mothball (keep paper, no dev) or decommission?**  
   Recommended: mothball until swing/factor is validated. Confirm.

4. **Are you willing to invest 5 dev-days in PEAD (P3.2)?**  
   High expected value, but needs earnings surprise data (FMP has it or Sharadar at ~$50/mo). Worth it?

5. **Capital allocation policy when multiple strategies exist:**  
   When swing + factor portfolio are both live, how do you split $100k? Equal weight by strategy? Sharpe-weighted? This needs a written policy before going live.

6. **Survivorship bias audit:**  
   The `pit_union + DB historical` approach is good but unaudited. Do you want a 1-day spot-check or accept the current approach?

---

## Brutally Honest Bottom Line

**At $100k, after costs, with retail-grade data, the realistic top-end is Sharpe 0.7–1.2 on a working long-only equity strategy.** That is approximately 8–15% annual return at 10% vol. Anything backtested above Sharpe 1.5 is almost certainly biased by look-ahead or period selection.

**Where we actually are right now:** Zero validated strategies. The 1.335 factor Sharpe is fictitious until P0.3 + P1.1 + P1.2 prove otherwise. LambdaRank is a dead end as currently specified. Intraday v51 is below gate. Nothing is ready to trade real capital.

**Most likely outcomes after P0–P1 work (probabilities):**
- 55%: Factor portfolio post-PIT-fix is Sharpe 0.40–0.70 — adequate, deploy carefully
- 25%: Factor portfolio post-PIT-fix is Sharpe < 0.30 — fundamentals *were* the alpha, pivot to P3.1 + P3.2
- 15%: Factor portfolio survives at Sharpe 0.70+ — good, deploy and stop tinkering
- 5%: Something else broken we haven't found yet

**What is genuinely worth pursuing:**
1. Factor portfolio with honest execution + PIT-clean fundamentals (if survives P1)
2. Sector rotation — the floor benchmark that must be beaten
3. PEAD — orthogonal, event-driven, academically validated
4. ML as meta-filter on top of (1)/(2)/(3) — not as primary signal

**What is likely a dead end:**
1. LambdaRank in current form — specification problem, not a tuning problem. 9 runs is proof.
2. Intraday at 15 bps/side without major architecture change
3. "More features" or "more HPO" on existing cross-sectional labels
4. Daily ATR stops on a monthly-rebalance factor strategy — conceptually incoherent

**Single most important next action:** P0.3 — Audit FMP fundamentals for PIT correctness. If FMP is using reporting-period end dates instead of filing dates, every fundamental-using backtest we have run for the past 6+ months is contaminated. This is one day of audit work and it determines whether the last 6 months of factor research is real or vapor. Do this first.

---

## Review Source Attribution

| Review | Key unique contribution |
|---|---|
| DeepSeek | Most aggressive on execution fixes; "entry price is a joke"; factor portfolio 1.335 is "impossibly good" |
| Gemini | Best explanation of *why* ATR stops kill factor premia; PDT rule change June 4, 2026 confirmed |
| ChatGPT | StrategyContract abstraction; most systematic phase structure; 3-class labels; go-live gate timeline |
| Claude/Opus | IC diagnostic as the missing test; factor decomposition against AQR MOM/QMJ; PEAD recommendation; 2007-2019 OOS test |
