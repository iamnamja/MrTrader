# Revisit Audit — Were Our Past WF/CPCV Kills Too Conservative? (2026-07-05)

**Provenance:** multi-agent Opus 4.8 re-adjudication (run `wf_11501aab-3e7`, 30 agents, 0 errors) — one reviewer per killed/parked/scaffold family read its DECISIONS entry + WF/CPCV run logs + research code; every promising revisit got an independent adversarial refutation; an Opus lead synthesized. Read-only; no live change. Scope: 22 families (14 KILLED / 4 PARKED / 2 SCAFFOLD / 2 shelved-PAPER).

**Bottom line:** the bar was calibrated correctly — NOT a Type-II machine. 7 flagged as possibly premature; 6 refuted; **1 survives (futures_skewness) as a FREE confirmatory run, not an expected win.**

---

## Follow-up execution (2026-07-05) — all surviving threads RUN and CLOSED

The three cheap/free re-tests the audit recommended were executed (Opus agents, report-only, no retrain, no live change). **All three confirm the original verdicts — nothing resurrects.**

| Thread | Run | Result | Verdict |
|---|---|---|---|
| **futures_skewness** (commodity-only) | restricted to 30 commodity mkts, same `skew_signal`/3bps | standalone Sharpe **+0.19** (up from +0.03 full-universe; modern-stronger post15 +0.19 / 20s +0.36) BUT **residual-α t +0.08** (needs ~1.6); Sharpe < 0.30. NOT carry-adjacent (signal corr +0.07) — genuinely orthogonal, just no standalone edge | **CLOSE-AS-NULL** |
| **swing_ml_ranker** (dollar-neutral L/S, explicit SPY hedge) | already-committed RANKER-v2 "Spike A" CPCV C(8,2), bear fold incl. | CPCV mean **+0.136**, P5 (bear) **−3.46**, DSR p 0.032, t +0.18 — fails all gates, still dies in the bear regime. Removing the invalid inverted-composite hedge did NOT change the answer → the null is the **signal**, not the hedge | **CLOSE the family** |
| **curve_momentum** (free sub-period read) | `run_futures_factors` `_subs` tuple | Sharpe −0.24, post15 −0.12 (still negative), 20s +0.03 (flat), residual-α t −1.38 — no post-2015 rescue | **CONFIRMED-DEAD** |

Net: the "too conservative?" hypothesis is closed on the specific candidates too — the bar held under direct re-test, not just review.

---

# Re-Adjudication of Killed / Parked Strategy Families — Lead Quant Decision Memo

*22 families re-reviewed; 7 flagged as possible false-negatives and each put through an adversarial refutation. Date: 2026-07-05.*

---

## 1. Verdict: the bar was about right — we were NOT systematically too conservative

The review layer flagged **7 families** as potentially killed too early (`too_conservative=true`, all rated MEDIUM revisit value). Every one of those 7 was then subjected to an independent adversarial refutation. The result is decisive:

- **6 of 7 refutations upheld the original kill** (`revisit_upheld=false`): options_signal, sector_rotation, futures_value, cftc_cot, vix_vrp, crypto_trend.
- **1 survived, at LOW confidence** (`revisit_upheld=true`): futures_skewness — and even its own refuter expects the re-run to re-discover the null; it survives only as a "close the published construction cleanly" run, not a live-edge candidate.

The other 15 families were triaged LOW/NONE by the reviewer and never escalated to refutation — the reviewer did not consider them worth an adversarial round, and most carried over-determined, wrong-sign, or gross-loss kills.

So: the ruler is not a Type-II machine here. The pre-registration discipline (no sign-flipping, standalone-floor-before-orthogonality, sub-period stability guards, selection-aware nulls) is doing exactly what it was rebuilt to do. **Net conclusion: the bar was calibrated correctly. Do not loosen it.**

### Conservatism-mode hypotheses that recurred — and why each dissolved

Four modes were proposed repeatedly. Naming them with the families, and the reason the refutation dissolved each:

| Mode | Families it was invoked for | Why it didn't hold |
|---|---|---|
| **regime-conditioned** (dead full-sample / alive recently) | options_signal, cftc_cot | The "alive-recently" window IS a single non-repeating regime (2022-23 growth-crash for options; a ~6yr sub-floor 0.77 for CoT). No independent confirmation window exists; splitting an already-underpowered sample halves power and is textbook forking-paths. The attribution *was* the adjudication. |
| **redundancy-not-lack-of-edge** | sector_rotation, vix_vrp | The proposed orthogonalizing rebuild either deletes the component that produced the standalone edge (sector L/S drops the crash-governor cash filter) or invents a new unvalidated sleeve (VRP's long-vol pairing) judged on the same borderline test. Residual-α, not correlation, is the binding gate — and it's flat/weak in both. |
| **wrong-construction** | futures_value, futures_skewness | For futures_value the refuter *ran* the canonical AMP 12m-skip fix: Sharpe rose −0.24→−0.07 (full universe) / +0.01 (commodity-only), still not promotable (residual-α t=1.19). For futures_skewness the commodity-only run was *not* executed — this is the one surviving thread. |
| **insufficient-free-history** | crypto_trend (+ raised-and-dismissed for pead, intraday_ml) | Extending crypto history injects MORE of the same 2020-21 parabolic regime (not independent folds), and Alpaca remains the only executable venue so a longer backtest isn't tradeable. The live-paper OOS tracker already provides the honest forward test. |

---

## 2. Ranked REVISIT BACKLOG (survived adversarial refutation)

Only families with `revisit_upheld=true` qualify. **Exactly one does.**

| Family | Why the bar was too high | Cheapest re-test | Effort | What would change our mind (promote) |
|---|---|---|---|---|
| **futures_skewness** (futures, KILLED, Sharpe +0.03) | The kill sorted skew cross-sectionally over the **full mixed 76-market universe** (commodities + bonds + FX + equity indices). The published premium (Fernandez-Perez/Frijns/Fuertes/Miffre 2018 JFE) is **commodity-specific** via lottery-preference; pooling financials, where the mechanism doesn't operate, dilutes a real commodity edge toward the +0.03 flat. The correct economic sign was used, so this is NOT sign-mining — the actual published construction was never tested. | Add a name/metadata commodity filter to `fd.liquid_universe()` in a report-only copy of `scripts/run_futures_factors.py`; re-run the existing `skew_signal` through `xs_factor_backtest` at the same 3bps/side roll cost. Report standalone Sharpe + subperiods + **residual-α t vs the live (trend+carry) book**. Owned Norgate data, no training, minutes. | FREE | Promote only if **standalone Sharpe ≥ 0.30 AND post-2015 > 0 AND residual-α t > ~1.6** vs carry+xsmom. Base rate is low: refuter notes commodity low-skew names are carry/basis-adjacent (backwardated/high-roll-yield), so the factor most likely **re-exposes the commodity carry we already hold** rather than adding orthogonal alpha, and the post-2015 gate is hostile to a 1987-2014 anomaly in the decayed alt-beta cohort. Treat as a **clean-close confirmatory run**, not an expected win. |

That is the entire backlog. I am deliberately not padding it — inflating LOW/refute-null families into "opportunities" would misrepresent the evidence.

---

## 3. Correctly killed — do NOT revisit (the majority)

**Over-determined / gross-loss kills (no admissible construction clears the bar):**
- **pead** — negative at event-level (t=−0.77), beta-hedged Sharpe −0.37; four independent venues agree edge is absent-to-negative; only remaining retest is ~$1k/yr Norgate, low EV.
- **intraday_ml** — gross PF 0.944 (loses *before* cost), t=−6.85, 0% positive folds; the headline +5.14 was in-sample memorization. On the 5/5 STOP list.
- **short_interest_xs** — CPCV −1.21, t=−3.53 wrong-sign; classic anomaly's live regime is unobservable on free data; flipping into squeeze-long is overfitting (correctly split to a separate P4c candidate).
- **curve_momentum** — Sharpe −0.24 at pre-registered sign; even flipped (~+0.24) fails the 0.30 floor and is redundant to carry (it's trend-of-carry). Sibling basis-momentum also dead.
- **basis_momentum** — near-zero every sub-period (post-2015 +0.04), residual-α t=0.47; documented post-publication decay.
- **futures_value** — refuter *ran* the canonical AMP 12m-skip fix; still −0.07 (full) / +0.01 (commodity), residual-α t=1.19 below floor; only positive slice is a single 2021-22 commodity-spike regime obtained by stacking two post-hoc choices.

**Genuine nulls, not ruler artifacts:**
- **swing_ml_ranker** — t=0.17, DSR p=0.30 across 9+ constructions; long-only = latent beta, dies in F2 bear folds; LX9-A beta-neutral still −0.70. (A dollar-neutral L/S spread check on existing v224 scores is cheap and worth one run to formally close the family, but the prior is it lands near LX9-A's +0.031.)
- **etf_relative_value** — point_SR +0.026, p=0.46; 9-cell grid + per-pair breakdown find no cell above the 0.30 cost floor; ETFs have no survivorship gap so no data buy can rescue it.
- **turn_of_month** — HAC near-miss (p=0.097) but decisive Track-B failure: pure timed SPY beta, lowers the long-biased book's Sharpe.
- **overnight** — true cost-kill (gross +0.53 → net −0.22 at generous 1bp/side); daily round-trip is inherent to the premium; the productive form was routed into the P3-4 basket.

**Redundancy correctly diagnosed (parked/dropped, edge not lost):**
- **options_signal** — strong but inverted signs already adjudicated as a 2022-23 growth-crash regime effect; trading the inverse is forbidden sign-mining; 208 wks is underpowered by construction.
- **sector_rotation** — 0.86 standalone but corr 0.51 to the trend book (shared 12-1 momentum factor); the higher-breadth cross-asset version (futures_xsmom) is already carried as PAPER.
- **cftc_cot** — the 0.14→0.42→0.77 uptick was explicitly seen and refused as the F3/daily-carry cherry-pick trap; orthogonality is not a backdoor to additivity (P0.3 Monte Carlo); P1.4 was docs-only, so revival is a from-scratch build for a sub-floor best case.
- **rates_carry** — H1 +0.689 is duration-bull beta in a one-way QE market (residual-α measured vs SPY only, not vs bonds); H2 −0.098 modern-dead; stability guard caught exactly its intended confound.
- **credit_timing** — Track-A passes but residual-α t=1.50 (insignificant) after removing SPY beta; the productive form (G1 de-risk overlay) is preserved and PIT-re-confirmed.
- **curve_overlay** — inert marginal to the VIX governor (dSharpe −0.017); sibling credit overlay passed the same harness, proving the ruler discriminated.
- **short_interest_overlay** — wrong-sign in-window (high-SII days had *higher* SPY returns post-2017); powered FINRA re-test (P3-5, ~10× data) corroborated: real but not standalone.

**Deferred/contingent — not "revive now," but a live future thread (do not action today):**
- **vix_vrp** — correctly deferred from the *initial* book (most tail-concentrating, corr 0.46, still owes the null-zoo test). The short-contango ⊕ conditional-long-vol pairing is a legitimate **GL-4 task once the futures book is actually live-paper**, not before.
- **crypto_trend** — PARK-with-tracker; the only open thread (multi-sleeve-book diversifier at 0.18 corr) is contingent on carry actually being capitalized and OOS accruing. The forward tracker is already running (clock from 2026-06-16).
- **credit_overlay** — PARKED flag-off by design; multiplicity caveat (winning L120/band0.02 from a 12-cell grid) is real and un-disposable. Not a false-negative — it's on the scheduled shadow-then-decide path.

---

## 4. Top next actions (ordered; alongside the live-execution push)

1. **Run the futures_skewness commodity-only confirmatory backtest** (FREE, report-only, minutes). Add a commodity filter to `fd.liquid_universe()`, re-run `skew_signal` through `xs_factor_backtest` at 3bps, report standalone Sharpe + subperiods + residual-α t vs (trend+carry). This is the *only* surviving revisit — execute it once to either promote (unlikely) or formally close the published construction, then log the null and stop.

2. **Read the credit-overlay forward-shadow output and take the scheduled verdict.** The enable/park/kill review is dated ~2026-07-14 (10 days out); shadow opened 2026-06-16 and fires ~8% of days. Run `python -m scripts.shadow_credit_governor`, check marginal dSharpe/dMaxDD on genuine OOS, and be ready to decide against the written false-positive budget. FREE, no new build.

3. **Keep the primary effort on live-execution truth (Alpha-v10 P2 / IBKR).** Every refutation converges on the same conclusion — *"the app is ahead of the alpha."* The carry+xsmom futures book is already in hand and UNPROVEN only on fills/roll-cost; the highest-EV work is execution-truth to deploy it, not mining a seventh dead free factor. This should remain the top-line push; items 1-2 run in the cracks.

4. **(Optional, CHEAP) One-shot dollar-neutral L/S spread check on the existing v224 ranker scores** — pure top-vs-bottom decile spread hedged with an explicit SPY short (not the invalid inverted-composite short LX9-A used), scored on the existing CPCV folds including the F2 bear fold. No retrain. If it lands near LX9-A's +0.031, close swing_ml_ranker at the family level for good. Only do this if item 1 is done and there's idle compute — the prior is a null.

5. **Do NOT** rebuild loaders or re-slice windows for cftc_cot, options_signal, futures_value, sector_rotation, vix_vrp, or crypto_trend now. Each was adversarially closed; vix_vrp and crypto_trend re-enter only as *contingent GL-4 / multi-sleeve studies once the futures book is live-paper* — gate them on that milestone, not on spare cycles today.