# Quant Review Synthesis & Alpha-v4 Plan — 2026-06

**Inputs:** 5 independent world-class-quant reviews of `QUANT_REVIEW_PROMPT_2026-06.md`
(ChatGPT, Gemini, Grok, DeepSeek, Claude/Opus).
**Author:** Opus 4.8, acting as systematic PM. **Date:** 2026-06-06.

This is the consolidated read + a phased, EV-ranked plan. It supersedes the
ad-hoc Track-A/Track-B sequencing. Phases below are ordered by **EV / effort**,
not by glamour. **This file is the SSOT for the Alpha-v4 direction** (indexed from
`MASTER_BACKLOG.md`).

## 0. Locked decisions (2026-06-06)
Owner-confirmed after reviewing all five reports:
1. **PEAD → telemetry size NOW.** `pm.pead_size_mult` 3.0→**1.0**, `pm.pead_max_position_pct`
   0.10→**0.05** — applied live (restart-free via agent_config) on 2026-06-06. Re-size
   only if the Phase-1 neutralization test proves it real. *(Reverses last sprint's B4 ramp.)*
2. **Gates → lower bar + reweight to robustness.** Drop avg-Sharpe to ~0.45; make
   residual-alpha-t and fold-consistency the *primary* criteria; keep the worst-regime
   survivability floor (`min fold Sharpe ≥ −0.30`). (Authoritative source: `app/ml/retrain_config.py`.)
3. **Targeted re-architecture.** Keep the execution core verbatim (PM/RM/Trader, Redis,
   audit, gates, rollback, harness); rework the research/validation layer; ADD the sleeve
   + regime-allocator layer. No teardown.
4. **Free-data only until a feasibility spike justifies a purchase.** Stay on ETFs/FMP;
   authorize paid options-IV / futures data only *after* a spike shows the edge is real.

> **Housekeeping (Phase 0/1):** also update the `CONFIG_SCHEMA` *defaults* for
> `pm.pead_size_mult` (→1.0) and `pm.pead_max_position_pct` (→0.05) on a branch so a
> fresh DB / fallback reflects telemetry posture (the live DB row already overrides).

---

## 1. Cross-LLM consensus tally (what all five agree on)

| # | Claim | Agreement | My call |
|---|---|---|---|
| 1 | **Architecture is good — re-aim, don't rebuild.** PM/RM/Trader split, audit, gates, rollback are real assets. | 5/5 | **Agree.** Keep verbatim. |
| 2 | **The 52% fold-skip invalidates absolute CPCV Sharpes** (and contaminates cross-family relative rankings). Fix before anything else. | 5/5 | **Agree — highest priority.** See §3 for the surgical fix they couldn't see. |
| 3 | **PEAD is NOT proven alpha** — p≈0.19, t≈1.04, 87% P&L in up-trends = substantially conditional beta/trend. | 5/5 | **Agree.** This directly contradicts our last sprint (B4 3× ramp + B5 trend filter). See §4. |
| 4 | **Stop nightly retrain of large-cap daily XS-ML.** Freeze as benchmark; the pond is fished out for retail. | 5/5 | **Agree.** ML lives only where there's signal (events, vol, crypto), or as meta-labeling. |
| 5 | **Build a portfolio of uncorrelated premia, not one hero strategy.** 4 × SR-0.4 uncorrelated sleeves ≈ book SR 0.8. Validate at the *book* level. | 5/5 (Claude central) | **Agree — this is the single most important reframe.** |
| 6 | **Trend-following / TSMOM is the #1 new sleeve** — crisis-diversifier, rules-based, ~0 corr to PEAD, cheap data. | 5/5 | **Agree. First new sleeve.** ETF version first (no roll handling), futures later. |
| 7 | **Next-open fills on post-earnings gappers are the most-violated assumption.** Real slippage 30–50 bps, not 10. | 3/5 strong (Gemini, Claude, ChatGPT) | **Agree.** Cheap kill-test. |
| 8 | **PEAD 2.0 = genuine expectation shock**, not raw EPS surprise: SUE + revenue confirm + estimate revisions + guidance + analyst-prior inconsistency + crowding; delay entry past the open. | 4/5 | **Agree, but gate it** behind the PEAD neutralization test (§4). |
| 9 | **Highest-ceiling new edge = options VRP / earnings IV-crush**, but needs a *new* options simulator + paid IV history. | 4/5 (DeepSeek #1, Claude #2 ceiling) | **Agree it's high-ceiling; stage it as a feasibility spike, not an upfront commit.** |
| 10 | **Lower the SR≥0.80 gate** — on N_eff≈8 it selects *for* overfitting; real edges live at 0.4–0.7. Weight fold-consistency + neutralized-t over level. | 2/5 explicit (Claude sharp, DeepSeek) | **Strongly agree.** Counterintuitive and correct. |

**Where reviewers were wrong / blind (they lacked context):**
- 3/5 recommend "analyst estimate-revision drift" as a fresh idea. We **already killed A1** — but A1 was *rating upgrades/downgrades*, not the **forward consensus-EPS-revision** anomaly (Chan–Jegadeesh–Lakonishok) they mean. Different signal, better-documented. Worth **one** cheap residualized retest — not a priority.
- Several say "ditch CPCV entirely." Too blunt — see §3; CPCV is salvageable for *rules-based* strategies.
- They didn't know we **already acquired short-interest data** (A2). SI died as a standalone factor, but it's the missing conditioning variable for ChatGPT's "catalyst + crowded-short squeeze" — near-zero marginal cost to test in the existing harness.

---

## 2. My independent verdict (world-class-quant voice)

The reviews are unusually concordant, which itself is signal. Stripped down:

1. **We have been polishing a measurement instrument that is biased, and pointing
   it at the most efficient market on earth.** The graveyard is the *expected*
   outcome, not a process failure.
2. **Our one "survivor" is mostly beta + trend.** The B5 "improvement"
   (+0.546 → +0.661 by blocking SPY downtrends) is, read honestly, *proof of the
   beta contamination*, not alpha — exactly the researcher-degrees-of-freedom trap.
   We must say this plainly: **the last sprint (Track B) ramped a strategy the
   evidence says we should be shrinking.**
3. **The opportunity set, not the technique, is the wall.** The fix is to point the
   (excellent) machine at less-efficient instruments and to stop scoring sleeves in
   isolation. Diversification is the only free lunch we've been leaving on the table.

Concretely, the highest-EV path is **not** "find a better model." It is:
**(a) make the ruler honest → (b) stop over-betting the one weak edge → (c) add a
genuinely uncorrelated, crisis-positive sleeve (trend) → (d) score the book, not the
sleeve → (e) only then chase the high-ceiling/high-cost bets (PEAD 2.0, options VRP).**

---

## 3. The fold-skip resolution they couldn't see (surgical, not "ditch CPCV")

Root cause in [cpcv.py:948-973](../../scripts/walkforward/cpcv.py#L948-L973): the BUG-23 overlap
guard skips a fold whenever a contiguous training window would span a prior test
fold in the same combo. That is **correct and necessary for a trained ML model**
(prevents in-sample contamination), but it is the structural reason ~half of all
CPCV evaluations vanish — and it biases the survivors toward later/bull regimes.

**Key insight: the guard is spurious for rules-based scorers.** `PEADStrategy`
sets `trained_through=date.min` — nothing is fit, so there is *no* training-window
leakage to guard against. For event strategies, the guard is throwing away half our
coverage for a contamination that cannot occur.

**Resolution (two-track, not "ditch CPCV"):**
- **Rules-based event strategies (PEAD, PEAD 2.0, squeeze):** add a
  `strategy.is_trained` flag (False when `trained_through == date.min`); when False,
  **disable the overlap guard** → full CPCV coverage, unbiased path distribution.
  This directly repairs the #1 finding *for our actual live candidate* at near-zero cost.
- **Trained ML strategies (the swing/intraday ranker):** the guard is legitimate, so
  CPCV-with-rolling-retrain is inherently low-coverage. For these, add a
  **purged sequential walk-forward** (`[start, t]` train, `[t+purge, t+1]` test, slide
  forward, every OOS block evaluated, zero holes) as the *honest baseline*. Report both.
- **Always emit a fold-coverage report** (folds/events included by year, VIX quintile,
  SPY-trend state, regime) and require it to pass *before* looking at performance.

---

## 4. The PEAD reckoning (we must reverse part of last sprint)

The unanimous read: **do not ramp PEAD.** Our live config currently runs
`pm.pead_size_mult = 3.0` (3× ramp), `pm.pead_max_position_pct = 0.10` (10% NAV),
plus the B5 trend filter. The evidence does not support that conviction.

**Actions (cheap, decisive — Phase 1):**
1. **Dial the live ramp back to telemetry size** (`pead_size_mult → 1.0`,
   `pead_max_position_pct → 0.05`). Keep it running purely as a *systems/fills test*,
   not as a sized bet. Honest reversal of B4.
2. **Neutralization kill-test (decisive):** long positive-surprise, short the sector
   ETF (or a matched no-surprise basket). If neutralized PEAD dies, "PEAD" is mostly
   beta+trend → free the slot. If it survives, we have a thin real diversifier.
3. **Factor decomposition:** regress PEAD returns on mkt+size+value+momentum+sector;
   read the **residual-alpha t-stat** (the A1 analog was t=0.20 — a warning).
4. **Gapper-slippage stress:** re-run the backtest with 30–50 bps opening slippage on
   gappers (not 10). See what survives. Realized SR ~0.40 is likely an over-estimate.
5. **Entry-timing sensitivity:** next-open vs open+30m vs first-30m VWAP.
6. **Re-run PEAD CPCV with the guard disabled** (§3) for an unbiased number.

**Decision gate:** PEAD graduates to a *small* sized sleeve only if neutralized-PEAD
survives, residual-alpha t > 1, and it survives punitive slippage. Otherwise it stays
a benchmark and we redeploy the slot. **Paper P&L cannot resolve p≈0.19 in <6–17 yrs —
stop treating the live record as alpha evidence.**

---

## 4b. Regime handling policy — "attribute, don't amputate"

Governing question: *if a model is great in most regimes but fails in one (e.g. COVID),
can we ignore that fold and accept the model on the rest?*

**No — never drop a fold to flatter a number.** Doing so (a) re-institutionalizes the
exact fold-skip bias that is our #1 finding, (b) deletes the most informative
observation (the crisis fold is *why* we validate OOS), (c) ignores that geometric
returns are destroyed by drawdowns — "amazing then catastrophic" is often a *worse*
profile than "mediocre everywhere," and (d) is a free overfitting knob: with N_eff≈8,
the freedom to relabel one fold "abnormal" manufactures a passing Sharpe from noise,
and DSR can't catch it. (This is the B5 trap, formalized — we "improved" PEAD by
finding the SPY-downtrend filter *after* seeing which regime hurt it.)

**The legitimate kernel — model the regime, never ignore it.** A strategy is allowed
to be a *specialist*, expressed three honest ways:
1. **Conditional deployment, declared a priori.** Size down / gate off in a named
   regime, BUT: the gate is economic and chosen *before* seeing which fold it rescues;
   it is lookback-safe (live regime classifier, not hindsight); the **gated** strategy
   is evaluated across **all** folds including the crisis (which now tests whether the
   de-risk *fired in time*); the fold is never deleted from the scorecard. The B5
   difference is pure *sequence*: declare from theory → then let the crisis fold judge.
2. **A worst-regime *floor*, not a *target*.** We already have `min fold Sharpe ≥ −0.30`.
   Correct shape: don't demand a *win* in the crisis, demand it doesn't *blow up*.
   Acceptance = "great in normal regimes AND survivable in crisis."
3. **Cover the un-handleable regime at the *book* level.** Pair a risk-on specialist
   (PEAD) with a crisis-positive specialist (trend/TSMOM). A specialist is fine *iff*
   you hold its complement. This is why Phase 2 (trend) directly answers PEAD's
   COVID/down-regime weakness.

**Policy (Phase 0 gate design):** *Attribute, don't amputate.* Report per-regime
attribution; hold a survivability floor in the worst regime; allow regime-conditional
strategies only via an a-priori, lookback-safe gate scored across all folds; solve
"fails in crisis" with a complementary sleeve, not by hiding the fold.

---

## 5. The Alpha-v4 phased plan (EV / effort ranked)

### Phase 0 — Validation integrity *(Week 1; blocks everything)*
- Add `is_trained` flag; disable overlap guard for rules-based scorers (§3).
- Add purged sequential walk-forward baseline for trained models.
- Emit + gate on a fold-coverage report (year/regime/VIX/trend).
- **Recalibrate gates:** lower avg-Sharpe to ~0.45, weight fold-consistency +
  neutralized-t over level; keep DSR/PF/Calmar. (Authoritative source:
  `app/ml/retrain_config.py`.)
- **Freeze the dead swing/intraday XS-ML retrain** (stop nightly `retrain_cron`
  promotion; keep as a non-production benchmark diagnostic).
- *Exit:* every backtest carries an honest coverage map; PEAD has an unbiased CPCV #.

### Phase 1 — PEAD honest reckoning *(Week 1–2; cheap, decisive)*
- Dial back the live ramp to telemetry size (§4.1).
- Run the neutralization + factor + gapper-slippage + entry-timing tests (§4.2–4.5).
- *Exit (decision gate):* PEAD = small real diversifier **or** benchmark-only.

### Phase 2 — First uncorrelated sleeve: ETF trend / TSMOM *(Week 2–6; highest-consensus new alpha)*
- Build TSMOM on liquid ETFs (SPY/QQQ/IWM/TLT/IEF/GLD/DBC + sectors): 1/3/6/12-mo
  lookback ensemble, vol-targeted, weekly rebalance. **No futures roll yet** (ETFs first).
- Validate **as a book addition** alongside PEAD (drawdown reduction + marginal book-SR),
  not standalone. This is the crisis-diversifier that fixes PEAD's long-bias.
- *Exit:* keep if it reduces combined drawdown / left-tail beta after costs, even at modest standalone SR.

### Phase 3 — Regime-aware portfolio allocator + book-level validation *(Week 4–8; architectural unlock)*
- Add a sleeve allocator above the PM: vol-weight sleeves by marginal contribution to
  book risk. Convert the harness to score at the book level (the §1.5 reframe).
- **Regime-conditional allocation (the "right model for the market" mechanism), done safely:**
  - **Switch the ALLOCATION, never the alpha model.** Do NOT train a model per regime —
    crisis is ~15% of history, so a regime-specialist model has N_eff≈1 and is pure
    overfit. (We already can't even *measure* PEAD's per-regime Sharpe: `REGIME_MIN_OBS`
    sparsity bites — see [regime.py](../../scripts/walkforward/regime.py) `compute_regime_sharpes`.
    If you can't measure per-regime, you can't train per-regime.)
  - **Let regime-specialist behavior emerge** from a few economically-distinct sleeves,
    each validated on ALL history, with different natural habitats: PEAD = risk-on/trending-up;
    trend/TSMOM = directional incl. crisis selloffs; (later) mean-reversion = high-vol-but-rangebound;
    VRP = calm/low-vol. The allocator tilts capital toward the sleeve whose habitat we're in.
  - **Use the coarse-3 PIT regime map** (BULL/NEUTRAL/BEAR; SPY-vs-MA × expanding VIX pctile)
    + `regime_model_v5`. Keep it coarse — >2–3 regimes overfits our N. Vol alone is
    insufficient (high-vol splits into trending→trend-wins vs choppy→MR-wins); use
    direction×vol, which the coarse-3 already encodes.
  - **Continuous tilts, not binary switches** (PEAD already consumes a [0,1] regime scalar),
    plus **hysteresis + persistence** (regime must hold N days) to kill boundary thrash.
  - **Validate detection on lag & flip-cost**, not in-sample accuracy (the #1 practical
    failure: laggy classifier protects late, whipsaws at the bottom).
- **Non-negotiable validation rule:** every sleeve's *alpha* is validated across ALL
  folds/regimes (never its home-regime subset). The *allocator* is validated separately
  by walk-forward: does the regime-tilted book beat the static-equal-weight book OOS
  **after turnover costs**? If not, drop the regime layer and hold sleeves at fixed weight.
  Regime-switching must *earn* its complexity.
- *Exit:* promotion decisions are made on incremental book contribution, not standalone Sharpe.

### Phase 4 — Higher-ceiling bets *(Month 2–3; gated on Phase 0–3)*
- **4a — PEAD 2.0 (genuine expectation shock):** SUE + revenue confirm + forward
  estimate revisions + guidance + analyst-prior inconsistency + crowding (reuse the
  SI data). Built on the `EventEdgeStrategy` harness. **Only if Phase 1 neutralized
  PEAD showed life.**
- **4b — Options VRP feasibility spike (NOT a full strategy):** price out historical
  IV data (Polygon/ORATS), build a *contract-level* P&L prototype on one underlying /
  one earnings event with conservative bid/ask, validate Greeks/IV/OI. **Decision gate
  before** committing to a full options sim. Highest ceiling, real cost — never trade
  options off the equity-bar engine.
- **4c — Catalyst + crowded-short squeeze (low lift, parallel):** condition PEAD
  *longs* on high days-to-cover (the only validated use of the SI data we already
  acquired). Small opportunistic sleeve; kill if it just re-picks PEAD winners.

### Phase 5 — Optional / later
- Futures-roll upgrade to Phase 2 (micro futures: MES/MNQ/M2K/micro rates/FX/metals).
- Merger arb (reuses equity sim; clean diversifier, lower ceiling).
- One cheap **residualized** retest of forward consensus-EPS-revision drift (the
  CJL anomaly the reviewers meant — distinct from the killed A1 ratings signal).
- Crypto basis/funding harvest (structural retail edge; new infra/ops — only with appetite).

---

## 6. Explicit STOP list (all 5 reviewers)
- Large-cap daily cross-sectional XGBoost/LambdaRank with standard TA features.
- Intraday 5-min ML on retail data.
- LLM stock-picking (use LLMs only to extract *structured event features*).
- Expensive alt-data (satellite/card-spend/options-flow hype) — solo-operator money pit.
- Single-name shorting without borrow/availability data.
- "Improving" a backtest by trying filters (the B5 SPY-trend trap) — pre-declare filters on economic grounds.

---

## 7. Data acquisition priority (ranked)
1. **Roll-adjusted continuous futures** (cheap) → enables Phase 5 futures-trend upgrade.
2. **Historical options IV surfaces** (paid; Polygon/ORATS) → *only if* the 4b spike says go.
3. **Forward estimate revisions + guidance + revenue surprise** (FMP, cheap) → Phase 4a.
4. ETF daily bars (already have / free) → Phase 2.
5. **Skip:** more equity TA endpoints (same dead pond), glamorous alt-data.
