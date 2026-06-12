# Alpha-v7 — Panel Synthesis & Plan (2026-06-12)

**SSOT for the ACTIVE direction (supersedes NEXT_PHASE_BLUEPRINT_2026-06 / Alpha-v6, now COMPLETE).**
Synthesizes a 4-LLM panel (Claude/Fable-5, DeepSeek, Gemini, Grok — raw under
`prompts/20260612_Alpha_v7_Review/responses/`) reviewing the all-null Alpha-v6
outcome, then grounds their recommendations against the actual repo. The prompt + kit:
`prompts/ALPHA_V7_REVIEW_PROMPT_2026-06.md`.

---

## 0. The one-paragraph verdict

The panel **converges hard** on three things: (1) the gate is a **Type-II / false-negative
machine on ≤4y data** (t ≈ SR·√years → a real SR-0.5–0.7 edge *cannot* clear t≥2 on 4y),
so the all-kill outcome was **partly predetermined by geometry, not discovered**; (2) the
liquid-US-equity / free-data alpha search is **genuinely exhausted** — the only survivors
are *risk premia*, not alpha; (3) the right move is to **re-charter from "find alpha" to
"operate a 3–5 sleeve premia book" judged at the BOOK level (Track B)**, target book
SR ~0.8–1.0. The sharpest *disconfirming* catches (Claude): the path-Sharpe t-stat is the
**wrong estimand** (measures cross-fold consistency, not significance) and should be
retired; the **tiers are inverted** (we demand significance to enter PAPER, the zero-risk
data-collection tier); and the **P5 trend-broadening PARK used the wrong ruler** (judged a
crisis-convex long-short sleeve on standalone Sharpe in a 2-bull sample — contradicting the
Track-B doctrine we'd built days earlier). None of this changes the live book today; it
changes what we measure and what we build next.

---

## 1. Panel consensus (what all/most agree on)

| Point | Claude | DeepSeek | Gemini | Grok |
|---|:--:|:--:|:--:|:--:|
| Gate is Type-II on ≤4y (power is the binding constraint) | ✓✓ | ✓✓ | ✓ | ✓✓ |
| "Everything died" is mostly HONEST for standalone equity alpha | ✓ | ✓✓ | ~ | ✓✓ |
| Re-charter to a premia book; judge at BOOK level (Track B central) | ✓✓ | ✓ | ✓ | ✓✓ |
| Expand/scale trend (more legs: futures/FX/commodities/crypto) + book-vol target | ✓✓ | ✓✓ | — | ✓✓ |
| Index VRP as a Track-B diversifier (revive P6) | ✓✓ | — | ~ | ✓✓ |
| Residual-alpha (CAPM/factor) should be MORE central / primary | ✓ | ✓ | ✓ | ✓✓ |
| Don't buy expensive data (I/B/E/S) for a $100k paper book now | ✓ | ~ | — | ✓ |
| Worst-regime floor wrong for diversifiers (Track B fixes routing) | ✓ | ✓ | ✓ | ✓✓ |

**Gemini added little** — it largely *described* the existing system back to us (and treated
already-done work as future plans). Low new signal; not weighted below.

## 2. The sharpest unique insights (and how I adjudicate them, given the repo)

1. **Retire the path-Sharpe t-stat — it's the wrong estimand (Claude). ADOPT.**
   `t = mean/(std/√8)` over 15 paths that reuse 8 folds rewards *low cross-fold dispersion*,
   not distance from zero — so a regime-homogeneous fluke gets t→∞ (bad size) while a real
   regime-heterogeneous edge gets an inflated std (bad power). This is a *better* diagnosis
   than "needs recalibration," and it matches our own calibration data (3/5 true nulls cleared
   t≥2). **Replace with inference on the OOS return series: HAC Sharpe SE (Lo 2002) +
   stationary bootstrap** on the *concatenated daily series* (N≈1000 on 4y), not on the 15
   paths (DeepSeek's bootstrap-the-paths is weaker — the paths aren't independent). Events keep
   the CGM two-way panel (already built, panel-validated). **Add PBO** (López de Prado) from the
   CPCV paths we already generate — the multiplicity-aware statistic CPCV was designed for, and
   strictly better than the saturated DSR we already demoted.

2. **Invert the tier stringency (Claude). ADOPT — highest-leverage gate change.**
   We require t≥2 to enter PAPER, the tier that risks **zero dollars and exists to generate the
   forward evidence underpowered candidates need.** Demanding significance to enter the
   evidence-collection tier *manufactures* the false-negative machine. → **PAPER = plausibility**
   (pre-registered + economic rationale + point-estimate SR≥0.3 + survives 2× cost-stress + P5
   not catastrophic + a cap on concurrent paper sleeves so paper itself isn't a multiplicity
   engine); **CAPITAL = a Bayesian posterior** P(SR>0 | backtest + *live paper track record*) ≥
   0.95, size ∝ posterior mean (shrinkage handles small-sample optimism continuously instead of
   a binary t-cliff). **Repo note:** the live PEAD/trend trackers + the realized-Sharpe-vs-ref
   rollup are already the "self-certification clock" this Bayesian capital tier needs as input.

3. **The P5 trend-broadening PARK used the wrong track (Claude). RE-TEST — top recovery item.**
   We judged the broadened long-short sleeve on standalone 19y Sharpe dominance (0.30 vs 0.72)
   — but the same run shows the long-short leg delivered **+2.5% in 2020-COVID (vs −6.2%) and
   +8.1% in 2022 (vs +0.9%)**: that's crisis convexity, exactly what Track B exists to value, in
   a 19y sample dominated by two equity bulls. The PARK verdict contradicts our own framework.
   **This is a genuine candidate-burial. Re-pre-register a long-short trend *overlay* (10–15%
   budget) through Track B with crisis-window + tail-overlap PRIMARY** (a NEW hypothesis with
   `parent_id` → P5's one-shot is preserved). Tooling exists: `book_gate.py` + the broadened
   `tsmom.py` config I built.

4. **Combination edges were unfindable until last week (Claude). RE-TEST — cheapest recovery.**
   The "noise, not negative" pile (index VRP SR~0 / PF 2.24, xmom_12_1 t=0.86, H4e IV/RV) are
   SR-0.1–0.3 candidates each adjudicated **standalone**. Three uncorrelated SR-0.25 sleeves are
   a book ΔSR of ~+0.15–0.25. Track B exists but has run **exactly once** (TSMOM). **Run the
   survivors through Track B individually + pairwise** vs the trend book. *Repo caveat:* needs
   each candidate's daily OOS return series — partly available (gate-calibration controls, the
   OPT-4 VRP backtest); some reconstruction required, but no new data.

5. **Survivor's privilege on TSMOM (Claude). AUDIT — intellectual honesty + sizing input.**
   Alpha candidates must survive CAPM hedging; the survivor is judged raw. **Run
   `attribution.capm_alpha` + Fama-French-5 on the live sleeve** (likely β≈0.3–0.4, residual SR
   ~0.3–0.45 — which *reframes* trend as a premia/timing book, the Q2 thesis, and informs how to
   size it; it does NOT kill it). Plus a **rebalance-offset sensitivity** check — `tsmom.py`
   anchors the weekly grid on `np.arange(n) % 5 == 0`; re-run offsets 1–4 to rule out timing
   luck. (FF5 daily factors are free from Ken French — DeepSeek's good add.)

6. **Power floor + diversifier waiver + residual-alpha primary (Grok/DeepSeek). ADOPT.**
   Require a minimum data length / cluster count before any **CAPITAL** verdict (not before
   PAPER); explicitly **waive the worst-regime floor for declared diversifiers**
   (`component_type ∈ {diversifier, risk_premium}` already in the registry params); make
   **residual-alpha t_hac primary** for Track A (not just diagnostic); demote PF/Calmar to
   diagnostic (off the boolean AND).

7. **Don't (Grok, all). HOLD THE LINE.** No XS-equity-ML revival, no single-name options, no
   dispersion (the cost wall is real), no binary threshold sweeps. The H4 kills are best read as
   *one regime statement* (2022 growth-crash window inverts vol-signal relationships), not five
   independent falsifications — don't cite them as "signals dead" if options data is ever
   re-acquired with a longer window.

**Deprioritized panel suggestions (repo says low EV):** DeepSeek's swing-ranker re-tests
(2×ATR trailing stop, `exclude_today=False`) chase a **decisively dead** model (swing XS-ML
killed 5+ ways) — and `exclude_today=False` risks re-introducing look-ahead; skip. QualityShort
shorts on delisted-inclusive data is feasible (the delisted harness exists from small/mid PEAD)
but a long shot off −0.903; park.

---

## 3. The Alpha-v7 plan — re-charter: **operate a premia book**

> **Mission shift:** from *"find standalone equity alpha"* (exhausted, and now rigorously
> proven so) to *"assemble + operate a 3–5 sleeve risk-premia book at book SR ~0.8–1.0,
> adjudicated at the BOOK level."* Trend is the base; everything else is a Track-B diversifier.
> Discipline unchanged: pre-registration, one-shot R4, sacred holdout, NO-DRIFT docs.

### Phase A — Recovery re-tests (no/low new data; START NOW, parallel-safe). **Highest EV/$.**
These can recover sleeves *without* the gate rewrite (Track B already exists), turning the
live book from one sleeve into two or three.
- **A1 — Long-short trend overlay via Track B.** Re-pre-register (parent=P5) a 10–15%-budget
  L/S trend overlay; crisis-window return + tail-overlap + appraisal-ratio PRIMARY; standalone
  Sharpe is *not* the gate. Tooling: `book_gate.py` + broadened `tsmom.py`. Expected: pass.
- **A2 — Noise-pile through Track B.** Index VRP, xmom_12_1 (+ H4e if reconstructable),
  individually + pairwise vs trend. Build/locate each daily OOS series first.
- **A3 — TSMOM honesty audit.** CAPM + FF5 attribution on the live sleeve; rebalance-offset
  sensitivity (offsets 1–4). Report-only; informs sizing + the re-charter framing.

### Phase B — Ruler v2 (gate redesign). The foundational fix; do as one careful PR.
- **B1** retire `path_sharpe_tstat` from all gating.
- **B2** primary inference = HAC Sharpe SE + stationary bootstrap on the concatenated OOS
  **daily** series; events keep CGM.
- **B3** add **PBO** from the existing CPCV paths.
- **B4** **invert tiers** — PAPER = plausibility (no t-stat; SR≥0.3 point + 2× cost-stress +
  econ rationale + concurrent-sleeve cap); CAPITAL = Bayesian posterior P(SR>0 | backtest +
  live paper) ≥ 0.95, size ∝ posterior mean.
- **B5** residual-alpha (CAPM/HAC) **primary** for Track A (t_hac ≥ ~1.8).
- **B6** Track B → budget-invariant **appraisal ratio** (residual-IR ≥ 0.2) + bootstrap CI on
  ΔSR (P(ΔSR>0) ≥ 0.85); worst-regime floor **waived** for declared diversifiers.
- **B7** PF/Calmar → diagnostic; **power floor** (min days / clusters) before CAPITAL only.
- Each change pre-registered as a gate amendment in the registry; re-score the existing kill
  ledger under Ruler v2 (a kill that flips is a real recovery — and a free finding).

### Phase C — Premia base: trend expansion.
- **C1** Free-data test FIRST: add futures (/ES,/NQ,/CL,/GC,/ZB), FX (yfinance), and crypto to
  the `tsmom.py` universe (it's just a config); apply the **book-vol overlay (already built)**;
  judge on 19y + Track B. Cheap (days). *Caveat:* yfinance continuous-futures quality is poor —
  this is a screen, not the live signal.
- **C2** If C1 improves the book → buy **Norgate (~$30–60/mo)** (survivorship-free equities +
  clean continuous futures — the single best data dollar; also retroactively fixes the
  delisted-bars hole) and open a **micro-futures broker** (Tradovate/IBKR; Alpaca has no
  futures). Re-validate on Norgate before live.
- **C3** Increase the trend allocation + apply book-vol-targeting live — **gated on Phase D**.

### Phase D — Live fidelity (the capital gate; already in motion).
- **D1** Mon 2026-06-15 09:45 ET first real rebalance → **trend replay-diff + fill-quality**
  (target: 4 boring weekly diffs) BEFORE any capital scaling or new live sleeve.
- **D2** Empirical Alpaca fill costs → recalibrate the cost model (feeds A2/marginal re-tests).

### Phase E — Index VRP diversifier (Track B), AFTER the gate is solid.
- **E1** **ETP expression first** (a defined-risk short-vol ETP, e.g. ZIVB-style) — needs **no**
  options data, sidesteps the options downgrade. Pre-registered, ≤10% risk budget, ≤2%-NAV tail
  budget, regime-gated, no parameter search.
- **E2** Graduate to real condors (would need options re-subscription) ONLY if the ETP sleeve
  passes Track B and the options-sim mechanics fixes land.

### Data & guardrails
- **Buy:** Norgate, *iff* C1 proves out. **Don't buy:** I/B/E/S (not EV-positive at $100k paper),
  microstructure, alt-data. **Options:** re-subscribe only if a VRP sleeve graduates to condors.
- **Hold the line:** no XS-ML revival, single-name options, dispersion, or threshold sweeps.

---

## 4. Sequencing & first moves

```
NOW ─────────────────────────────────────────────────────────────────────►
Phase D (live fidelity) — already running; Monday's rebalance is the trigger.
Phase A (recovery re-tests) — START IMMEDIATELY; uses the existing Track B, zero new data.
Phase B (Ruler v2)         — foundational PR, in parallel with A; re-score the kill ledger.
Phase C (trend expansion)  — free screen → Norgate → live (gated on D).
Phase E (VRP via ETP)      — after B lands and the book gate is solid.
```

**The first three moves (panel + repo consensus):**
1. **Run Phase A** — the three recovery re-tests (P5 L/S overlay, noise-pile Track-B,
   TSMOM honesty audit). Cheapest path to a 2–3 sleeve book; needs no gate rewrite.
2. **Ship Ruler v2 (Phase B)** — retire path-t; HAC-SR + bootstrap + PBO; invert the tiers;
   residual-alpha primary; appraisal-ratio Track B. Then re-score the kill ledger under it.
3. **Re-charter + trend expansion screen (Phase C1)** — formally change the mission to
   "operate a premia book at book SR ~0.8," and free-screen the futures/FX/crypto trend
   universe; Norgate + a futures pipe only if it proves out.

**The honest meta-point the panel makes (and I agree with):** the all-kill outcome is the
system *working* — it proved our data tier holds no certifiable standalone equity alpha.
The error wasn't the kills; it was spending the program asking *"is this alpha?"* of things
that were never going to be alpha at this SR/sample, while the answerable question —
*"does this premia portfolio clear book SR 0.8?"* — went mostly unasked. **Alpha-v7 asks
that one.**
