# Alpha-v7 External Review — Claude (Fable 5)

**Date:** 2026-06-12
**Reviewer role:** External quant panel — skeptical, disconfirmation-first
**Materials reviewed:** Full briefing kit (00–16): `PIPELINE_ARCHITECTURE.md` (§7 Gate Inventory, §8 Fold Construction, §12 Known Limitations), `gates.py`, `cpcv.py`, `retrain_config.py`, `agent_simulator.py` (cost model), `tsmom.py`, `attribution.py`, `event_inference.py`, `registry.py`, `ML_EXPERIMENT_LOG.md`, `DECISIONS.md`, `MODEL_STATUS.md`, `DATA_PROVIDERS.md`, prior blueprint.

---

## Q1 — Is the acceptance framework sound?

**Highest-conviction point: the Alpha-v6 all-kill outcome was predetermined by geometry, not discovered by testing.** Since t ≈ SR·√years, certifying SR at t≥2 requires roughly (2/SR)² years: an SR-0.5 edge needs ~16 years, SR-0.7 needs ~8. You ran candidates on ≤4 years of data through a t≥2 gate. By construction, *nothing real and retail-sized could ever pass* — only long-history rules sleeves (TSMOM, 19y) or statistical flukes. Your program wasn't a search; it was a proof that the gate rejects everything findable in your data budget. The fix is not threshold tuning — it's changing the statistic, the unit of inference, and the tier semantics.

### 1a. The path-Sharpe t-stat is not a statistic — retire it entirely

Your calibration result (3/5 true-zero nulls clearing t≥2, with t up to 3.47) has a precise mechanism your docs don't name: **t = mean/(std/√8) rewards low cross-fold dispersion, not distance from zero.** The 15 paths reuse 8 folds, so a lucky null whose Sharpe is mildly positive *everywhere in one regime-homogeneous window* produces near-identical path Sharpes → std → 0 → t explodes. Simultaneously, a real edge with regime-heterogeneous fold Sharpes (which is what real edges look like) gets an inflated std → Type II. The statistic has both terrible size *and* terrible power because the denominator measures regime heterogeneity, not sampling error. No N_eff correction fixes this. It's the wrong estimand.

**Replace it with inference on the return series itself, where N is in the hundreds:**

| Strategy type | Primary instrument | N unit |
|---|---|---|
| Daily-return (trend, VRP, XS L/S) | HAC SR standard error (Lo 2002 / Ledoit-Wolf 2008) on concatenated OOS daily returns, plus stationary-bootstrap (Politis-Romano) p-value as the robustness twin | trading days (~1000 on 4y) |
| Event-driven | CGM two-way clustered panel — **already built, correct** | events / announce-days |
| CPCV | Demoted to robustness/coverage: report P5, %pos, fold map; compute **PBO** (below); never a gating t-stat | — |

**Compute PBO (probability of backtest overfitting).** You have full CPCV machinery and you're not computing the one statistic CPCV was designed for: López de Prado's PBO uses the combinatorial paths to estimate the probability that the IS-best configuration ranks below median OOS. It's nearly free given your existing code, it's multiplicity-aware by construction, and it's a strictly better instrument than DSR (which you've correctly demoted as saturated).

### 1b. Your tiers are inverted in stringency — the single most costly design error

You require t≥2.0 to enter **PAPER** — a tier that risks zero dollars. Paper is not a reward for significance; paper *is the data-collection mechanism* that significance-starved candidates need. Demanding significance to enter the tier whose purpose is generating more evidence guarantees the false-negative machine you measured. Invert it:

- **PAPER (plausibility tier):** pre-registered hypothesis + economic rationale + point-estimate SR ≥ 0.3 + survives 2× cost stress + P5 not catastrophic + passes implausibility ceiling. No t-stat. Cap concurrent paper sleeves (say 4) so paper itself doesn't become a multiplicity engine.
- **CAPITAL (significance tier):** Bayesian update of the backtest posterior with the live paper track record. Concretely: prior SR ~ N(0, 0.3²) (the honest distribution of true retail edges), promote when P(SR > 0 | backtest + paper) ≥ 0.95 *and* paper realized SR is within 1σ of backtest (the replay-diff discipline already supports this). Size ∝ posterior mean, not point estimate — shrinkage handles small-sample optimism continuously instead of via a binary cliff.

This answers "significant enough for paper vs. capital" directly: paper is gated on *plausibility and cost-realism*; capital is gated on *posterior probability including live evidence*. Multiplicity is handled by the registry's true trial count feeding the prior (more trials → tighter prior at zero), not by an arbitrary t haircut from 2.0 to 2.5.

### 1c. Track B: right idea, two flaws

1. **The ΔSharpe-at-budget criterion is budget-dependent — which is why you had to amend 10%→25% after a failure.** The amendment was registered, but the deeper problem is that pass/fail is a mechanical function of an arbitrary constant. Replace ΔSR≥0.10 with the budget-invariant **appraisal ratio**: regress the (vol-targeted) candidate on the base book; require residual-alpha IR ≥ 0.2 with a HAC t on the intercept. Keep corr<0.30 and the tail-overlap test (genuinely good — the overlap formulation you registered is better than the mean-of-tail version you caught and removed). Sizing then becomes a separate optimization, not part of accept/reject.
2. **Track B has no uncertainty on the delta.** A noise candidate with a lucky 6-year corr can clear +0.10. Block-bootstrap the joint return matrix and require P(ΔSR > 0) ≥ 0.85.

### 1d. Verdict on strictness

You are simultaneously too strict and too loose, in exactly the places your calibration found: too strict on real SR-0.3–0.7 material (the t-bar + paper-tier inversion + worst-regime floor applied to diversifiers), too loose in that the path-t admits consistent flukes. The two-track split fixed the worst-regime misrouting; the remaining fixes are:

1. Retire path-t for series/panel inference (1a).
2. Invert tier stringency (1b).
3. PBO + registry-fed shrinkage as the multiplicity defense.
4. Appraisal-ratio Track B with bootstrap CI (1c).

Don't touch the materiality floors (0.35/0.45) — those are fine.

---

## Q2 — Where does durable alpha come from now?

**Highest-conviction point: stop hunting alpha in US equities on free data — the search space is exhausted *for your information set*, and that's a property of the data tier, not of your effort. Become a miniature AQR, not a miniature RenTech: a 4–6-sleeve risk-premia book targeting book SR ~0.8–1.0.** Your own evidence says this: the only survivor is a risk premium, your one cost-robust options finding (index VRP, PF 2.24/1.75) is a risk premium, and your gate-calibration showed the framework's failure mode was mis-ruling diversifiers. The mission statement should change from "find durable alpha" to "assemble and operate durable premia."

Ranked by expected value per dollar and per hour for a solo operator at $100k:

### Priority 1 — Micro-futures multi-asset trend (replaces/extends the ETF sleeve). Highest certainty; do first.

The 10-ETF long-flat sleeve is a compromised expression of the one premium you've validated: no shorts, no commodities breadth, equity-beta-heavy basket, dividend-drag instruments. Micros (MES/MNQ/M2K/MYM, MGC/SIL/MCL, M6E/M6B/M6A, 2YY/10Y micro yields) give 16–24 genuinely diverse markets, native shorting, ~$1–2k margin per line — feasible at $100k. Literature and your own 19y number say a diversified L/S futures trend book runs SR 0.7–1.0 where the ETF long-flat compromise runs ~0.5–0.7.

**Critical process note: judge long-short trend on Track B (book delta + crisis convexity), not standalone 19y Sharpe — see Q3.1 on why the P5 PARK verdict used the wrong ruler.**

Data: **Norgate (~$30–60/mo)** for survivorship-free equities *and* continuous futures — the single best data dollar available to you (it also retroactively fixes the delisted-bars hole). Alpaca doesn't do futures; you'd add a small futures account (Tradovate/IBKR) — the real cost is operational, not financial.

### Priority 2 — Index VRP sleeve, 5–10% Track B budget (revive dormant P6)

Your own data: real, cost-robust (penny spreads), crisis-negative — the canonical pairing with trend. It was killed *standalone* by a significance gate that risk premia should never face; that's precisely what Track B exists for. Expression at your scale: monthly ~16Δ SPY condors via Alpaca (the NBBO logger is already accumulating the spread calibration), or defined-risk short-vol ETPs (ZIVB-style) if you don't want to re-subscribe to options data. Modest expected contribution (+0.05–0.10 book SR), but cheap and structurally diversifying.

### Priority 3 — FX carry via micro FX futures

Well-documented premium, free data (FRED rates + futures), crisis-correlated *opposite* to trend, tiny operational load once the futures pipe from Priority 1 exists. Third sleeve.

### Priority 4 — Crypto basis/funding carry (CME micro BTC/ETH futures vs spot)

Delta-neutral, documented premium, capacity-limited (which favors you), free data from exchange APIs, US-legal expression via CME micros. Medium conviction — sleeve #4, small budget, Track B.

### Priority 5 — Data purchases: mostly, don't

- **Norgate: yes** (above). Fixes survivorship and unlocks futures backtests in one $30–60/mo line.
- **I/B/E/S-class estimate revisions:** the anomaly is real and it's the one thing H3 is blocked on, but at $300–500+/mo it's not EV-positive for a $100k paper book when the equity-XS lane has just been demonstrated cost-marginal at your tier. Park; revisit if the book goes live and scales.
- **Intraday tick/L2, alt-data:** no — capacity and infrastructure you can't exploit.
- **Options re-subscription:** only if/when the VRP sleeve graduates from ETP expression to real condors and needs current chains.

### On the premise itself

Yes — the liquid US equity cross-section is efficient *relative to free EOD + $29 fundamentals*. Being small opens doors only in microcaps/illiquid corners, which your data can't support (event studies there *require* delisted bars). The honest strategic answer is the premia book plus redirecting research hours from hypothesis-hunting to execution quality, book-level vol targeting, and tail management. Your infrastructure is now over-built relative to the book — which is fine: it's the moat for operating the premia book reliably with one human.

---

## Q3 — Could the process have wrongly buried a real edge?

**Highest-conviction point: the harness didn't bury alpha — it buried *small risk premia*, systematically, because for most of the program's life every candidate faced a standalone-significance ruler that nothing with SR < 0.7 could ever clear on ≤4y. The "everything is dead" conclusion is real for alpha and an artifact for premia.** Findings, ordered by how much I'd worry:

### 3.1 — The P5 trend-broadening PARK used the wrong track (most likely genuine burial)

You judged the broadened long-short sleeve on standalone 19y Sharpe dominance (0.30 vs 0.72) — but your own data in the same log entry shows the long-short leg delivered +2.5% in 2020 (vs −6.2%) and +8.1% in 2022 (vs +0.9%). That is crisis convexity — exactly the thing your two-track doctrine says must be judged on *book contribution*, not standalone Sharpe in a sample dominated by two monster equity bulls. This verdict contradicts the framework you built three days earlier.

**Re-test:** a small (10–15% budget) long-short trend *overlay* (not replacement) through Track B with the tail-overlap and crisis-window metrics primary. Expected outcome: pass.

### 3.2 — Combination edges were structurally unfindable until last week

The ledger's "noise, not negative" pile — index VRP (SR~0, PF 2.24), xmom_12_1 (t=0.86), H4e IV/RV — are individually-insignificant SR-0.1–0.3 candidates. Three uncorrelated SR-0.25 sleeves are a book ΔSR of roughly +0.15–0.25. Every one was adjudicated standalone. Track B now exists but has run exactly once, on TSMOM.

**Re-test:** run the surviving "noise" candidates through Track B individually and pairwise against the trend book. The cheapest recovery operation available — no new data, no new code.

### 3.3 — The attribution asymmetry (be honest about the survivor)

Alpha candidates must survive CAPM hedging; the survivor is judged raw. Run `capm_alpha` on the TSMOM sleeve: long-flat on an equity-heavy basket over 19 bull-dominated years will show β ≈ 0.3–0.4 and a residual SR materially below 0.71 (estimate: 0.3–0.45). That doesn't kill it — it correctly *reframes* it as a risk-premia/timing book, which is the Q2 thesis. But the docs currently let the survivor enjoy a standard no candidate was allowed.

Also check rebalance-grid sensitivity: `np.arange(n) % 5` anchors the weekly grid on index position 0 — re-run with offsets 1–4; if Sharpe disperses more than ~±0.1 there is a timing-luck component.

### 3.4 — The H4 options-signal kills are a regime statement, not five falsifications

Four of five academic signals "significantly negative" *simultaneously*, all on a single 4-year window whose first half is the 2022 growth crash, is one finding (the window inverts vol-signal relationships), not five. The correct epistemic state is "unidentifiable on this window," not "dead." Pragmatically moot — the data is frozen and the line is closed — but do not cite H4a–e as evidence the *signals* are dead if you ever re-acquire options data with a longer window.

### 3.5 — Cost model: mostly fair, decisive in one place

5bps/side + 3bps entry + 5bps stop slippage ≈ 16bps round trip is *right* for median R1K names at your size and slightly punitive for the mega-liquid tail — the PEAD/XS-ML kills don't move at ±5bps. The one place it was decisive: **short-term reversal** (high turnover, marginal gross edge). At $100k with marketable limits in the top liquidity decile, true cost is plausibly 4–8bps round trip, not 16. Daily reversal probably still dies net — but if you want one cost-recovery re-test, that's it, gated on your *empirical* fill data once the live book has a few months of Alpaca fills (everything needed is already logged).

### 3.6 — Things checked and clean

- `tsmom.py` PIT discipline: the `held.shift(1)` / `cost.shift(1)` alignment is correct and the docstring's claimed invariant holds.
- CGM implementation: Petersen pins are the right validation; per-component Stata factors and min(G)−1 df are the standard conservative convention.
- Per-event SPY hedging: correctly conditional — avoids the static-beta-eats-timing-alpha trap.
- Survivorship direction on the PEAD kill: missing delisted losers *flatters* long-only PEAD, so the negative verdict is conservative — the kill is safe.
- KL-11 overlap-guard fix: sound.
- Registry R1–R5 integrity rules: genuinely good — better than most professional shops.

---

## Where you've fooled yourselves — summary

1. **Requiring significance to enter a zero-risk tier** (the 1b inversion) — this *manufactured* the false-negative machine you then measured.
2. **Reading the path-t's failure as "needs recalibration"** for weeks before discovering it measures consistency, not significance — and it's still nominally in the gate config.
3. **PARKing P5 on Track-A logic** three days after building Track B for exactly that component type.
4. **Calling the program "alpha exhausted"** when the gate geometry guaranteed all-kills below SR~0.7 on 4y — the search was never powered to find what realistically exists.
5. **Mild survivor's privilege on TSMOM** (no CAPM attribution, no rebalance-offset check).

---

## Bottom line — first three things I'd do Monday

1. **Rewrite the gate (one PR):** retire `path_sharpe_tstat` from all gating; primary inference = HAC SR test + stationary bootstrap on OOS daily series (events keep CGM); add PBO from the existing CPCV paths; invert tiers — paper = plausibility + cost-stress, capital = posterior P(SR>0) ≥ 0.95 with live-paper Bayesian update; Track B → appraisal ratio + bootstrap CI on the delta.
2. **Run the recovery re-tests that need zero new data:** (a) long-short trend overlay through Track B at 10–15% budget, (b) index-VRP (ETP or condor expression) through Track B at 5–10%, (c) the "noise pile" pairwise book deltas. Expectation: at least one passes and the live book goes from one sleeve to two or three.
3. **Buy Norgate, open a micro-futures pipe, and build the 16–24-market L/S trend book** as the capital-grade core — then formally re-charter the program from "find alpha" to "operate a premia book at SR ~0.8," and freeze all equity-XS/event research until the information set changes.

The uncomfortable fact that prompted this review — every candidate edge rejected — is, properly read, the system *working*: it proved your data tier contains no certifiable standalone equity alpha, which is the true state of the world. The error isn't in the kills. It's that the framework spent six weeks asking "is this alpha?" of things that were never going to be alpha, while the answerable question — "does this premia portfolio clear SR 0.8?" — went mostly unasked. Ask that one next.
