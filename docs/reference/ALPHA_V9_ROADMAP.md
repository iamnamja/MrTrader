# Alpha-v9 Roadmap — synthesis of the 2026-06-16 external quant review

**Status:** ACTIVE program (supersedes Alpha-v8, which is complete).
**Created:** 2026-06-16 · **Author:** Opus 4.8 synthesis of 5 external reviews (ChatGPT, Claude Max,
DeepSeek, Gemini, Grok) + independent critical judgment.
**Inputs:** `docs/reference/prompts/20260616_Alpha_v8/responses/*.md` (the 5 raw reviews).
**Companion:** `docs/reference/ALPHA_V9_ARCHITECTURE.md` (the design + implementation plan).

> **One-line thesis.** Five independent world-class-quant reviews converged on the same verdict: our
> *process* is fund-grade and our "free daily US **equity directional** alpha is mined out" conclusion
> is **correct** — but we mistook *one quadrant's* emptiness for *the map's* emptiness. The prize is no
> longer a fourth weak uncorrelated equity sleeve. It is **(1) prove our validator can even detect a
> known edge, then (2) run a second *return engine* in a different risk class (volatility-risk-premium)
> paired with trend, (3) point the engine at a less-efficient universe (crypto), and (4) buy the one
> dataset (Norgate) that un-biases our event work and unlocks carry.**

---

## 0. How to read this

This is a **critical synthesis, not a vote count.** Where the reviewers agreed, I say so and weight it
up. Where they disagreed, I surface the disagreement and give a reasoned call. Where I think a reviewer
is wrong, I say that too. Every item carries: **why** (which reviewers + economic rationale),
**effort**, **dependency**, **owner-gate** (does this need your sign-off?), and a **falsifiable success
criterion**. Nothing here is "do it because an LLM said so."

Legend: 🟢 free + on current broker · 🟡 needs a gate/code change · 🔵 needs paid data · 🔴 needs a 2nd
broker (IBKR) · 👤 owner decision required.

---

## 1. The consensus (what all/most reviewers independently agreed on)

| # | Consensus point | Who | My call |
|---|---|---|---|
| C1 | **Process & discipline are top-decile for a solo operator**; the kills are mostly honest; trend is a real, durable edge. | all 5 | Agree. Don't over-correct. |
| C2 | **"Free daily US equity *directional* alpha is mined out" is correct** — but "we've exhausted the *retail edge envelope*" is NOT. Untouched: vol/options-*structures*, crypto, survivorship-free data. | all 5 | **Agree — this reframes the whole program.** |
| C3 | **Norgate (~$270/yr) is the #1 data buy** — fixes survivorship (re-open contaminated kills) AND unlocks futures carry. | all 5 | Agree, with sequencing (§5). |
| C4 | **Survivorship bias contaminates single-name/event kills** → relabel "killed-on-biased-data / unknown," not "dead," until re-tested on clean data. | ChatGPT, ClaudeMax, DeepSeek, Gemini, Grok | Agree. |
| C5 | **Futures trend + carry** is the top *strategic* expansion — but execution needs IBKR; research first, broker only if it passes. | all 5 | Agree. |
| C6 | **Crypto trend/momentum** — quick win: new less-efficient universe, engine exists, Alpaca executes it. | all 5 | Agree, with power-floor honesty + small sizing. |
| C7 | **Modeling: the target is wrong.** Predicting forward *returns* at IC≈0 is the *expected* result. Switch to predicting **vol/regime** (tradeable via options) and use ML for **meta-labeling/sizing** the primary signal, not signal generation. **HRP** for the multi-engine book. | ChatGPT, ClaudeMax, DeepSeek, Gemini | Strongly agree. |
| C8 | **Cash/T-bill sleeve** — ~75% idle cash should earn the risk-free rate (SGOV/BIL). Confirmed: no cash sleeve exists. | ChatGPT (lead) | Agree — easy, immediate P&L. |
| C9 | **Overlays are first-class**; the overlay-vs-sleeve split is the most sophisticated part — keep & extend. | ChatGPT, ClaudeMax, DeepSeek, Grok | Agree. |
| C10 | **Back-validate live fills vs paper** — make backtest-to-paper tracking error a first-class gate. | ChatGPT, Grok | Agree. |

---

## 2. The sharp critiques that change a decision (mostly Claude Max, the most rigorous review)

These are the high-value findings — several **qualify or overturn a load-bearing conclusion.** I rank
them by how much they should reorder our priorities.

### Ⓐ "All our bugs inflated results" is NOT safe — and it underwrites the entire "mined out" verdict. **(highest priority)**
- **The claim under attack:** our re-validation audit concluded the ~23 early-harness bugs *inflated*
  (false positives), so no real edge was wrongly killed, so "IC≈0 / mined out" is trustworthy.
- **The rebuttal (ClaudeMax, `[C: High]`):** our own changelog contains **deflationary** bugs —
  `#PERFOLD2` (regime-map type bug → **empty X for every fold → mean Sharpe 0, n_paths=0**), `#339`,
  `#342`, and a pattern of `except Exception: continue` silently dropping windows. An alignment/join/
  swallowed-exception bug produces *exactly* the "infra is sound, IC≈0" signature we're most confident
  in. "Leakage inflates" is true; "**join/alignment/PIT bugs deflate**" is *also* true, and we had both.
  Re-running the kills tests the *gate*, not whether the **feature pipeline can surface a known effect
  at all.**
- **My judgment: this is correct and it is the single most important takeaway of the whole review.** It
  is cheap to settle and it de-risks every downstream conclusion. **It must come before we act on
  "mined out" in either direction.** → Phase 0, item P0-1.

### Ⓑ The both-halves guard is an unpowered binary bolt-on. **(high)**
- ~**24% false-negative rate on a true SR-0.5 edge** (~35% at SR-0.35) from the guard *alone*, before
  the rest of Ruler-v2. It "alone killed carry." It cannot distinguish "unstable fluke" (what we want to
  catch) from "real but noisy" (what we want to keep) — it re-creates the Type-II machine Ruler-v2
  retired. **Confirmed in code:** `scripts/walkforward/sleeves.py`.
- **Fix:** replace "positive in both halves" with a *powered* stability test — test that the two
  half-Sharpes are **not significantly different** (Chow/homogeneity, or a bootstrap CI on the
  half-difference that overlaps zero) AND that the **pooled HAC-SR is significant.** "Stable" = "no
  evidence of a structural break," not "lucky enough to be positive twice." → P0-2.

### Ⓒ Track-B's `standalone_vt_SR > 0.20` floor defeats Track-B's own purpose. **(high)**
- Track-B exists to admit things *weak alone but additive*. Genuine diversifiers (carry, vol-selling,
  tail hedges) have **low or negative** standalone Sharpe — that's *why* they diversify. Requiring decent
  standalone SR **selects against exactly what Track-B should catch.** The appraisal IR already encodes
  "improves the book"; the standalone floor is redundant-to-harmful.
- **Fix:** drop / sharply lower `standalone_vt_SR` for `component_type ∈ {diversifier, risk_premium}`
  (we already waive the worst-regime floor for these — extend the logic). → P0-2.

### Ⓓ Our cluster of "robust near-misses" is real-but-weak, not null. **(high)**
- Carry: point_SR +0.314, residual-α **+2.10 (clears the 2.0 bar)**, Track-B 7/8, missed on HAC
  p=0.0998 & P(ΔSR>0)=0.886. Credit overlay: promoted candidate. PEAD: our own note says
  **"real-but-underpowered."** When mechanism-motivated candidates *repeatedly* land at p≈0.10 /
  P(ΔSR>0)≈0.88 / residual-α≈2.1, the honest Bayesian reading is "real-but-weak with ~80–88% power,"
  not "nulls." Treating an 88% posterior as a hard FAIL is a **decision-theory error for a diversifier**
  (asymmetric loss: false-accept at small size costs a few bps; true-accept buys real diversification).
- **Fix:** admit weak-but-real declared diversifiers **on probation at small size** with live-paper
  ratification; lower Track-B `P(ΔSR>0)` to ~0.75 *for small-size diversifiers only* (keep the high bar
  for core/CAPITAL sleeves). → P0-2 + P2-3.

### Ⓔ We never gate the *basket* of near-misses — only each in isolation. **(high, cheap)**
- A vol-targeted equal-risk **composite** of 4–5 mutually low-correlated weak premia can clear the gate
  *even when no single component does* — idiosyncratic noise partly cancels. We apply this logic to
  trend+diversifiers but never to the **sub-book of weak premia** we've accumulated. **Failure mode is
  informative:** if the composite *also* fails, the components were secretly correlated (the F1a "all
  just timed SPY beta" lesson) — a clean negative. → P3-4 (`risk_premia_composite`).

### Ⓕ The options cost model doesn't exist and the equity one is too optimistic for options. **(blocking for any options work)**
- Confirmed in code: costs are ~1 bps/side of *notional*. **Option spreads are 1–5%+ of *premium*** —
  our model is an *order of magnitude* off and will manufacture phantom edges. **Build a premium-%,
  IV-aware option cost model before trusting any options backtest.** Cheaper than any data buy. → P2-4.

### Ⓖ CPCV power is NOT the constraint — do not "fix" it. **(guardrail)**
- ClaudeMax credits us: we divide by `√N_eff = n_folds`, we know the C(6,2)=15 paths are correlated,
  the path-t is report-only, and Ruler-v2 scores **pooled-OOS HAC** on the concatenated daily series
  (n in the hundreds–thousands). **Do not inflate paths to manufacture significance.** Power binds only
  on the 4y options window and bi-monthly series. Keep the honest N_eff handling.

### Ⓗ Governance: `REQUIRE_TRUE_WF_FOR_PROMOTION` is still default-`False`. **(medium, governance)**
- Confirmed in `app/ml/retrain_config.py:30`. Frozen-model CPCV ≠ true per-fold-retrain walk-forward.
  Any *trained-model* promotion must prove itself with true WF. **Flip to `True` for trained paths**
  (rules-based return streams keep the non-trained path). → P0-3.

---

## 3. Where the reviewers DISAGREED — and my calls

Honest disagreement is signal. Don't paper over these.

| Topic | The disagreement | My call |
|---|---|---|
| **Options / VRP** | ClaudeMax: defined-risk **VRP is the highest-EV unexplored edge, do now.** Grok: medium, paper-only, start small. **Gemini: HALT options** — frozen 4y + indicative NBBO + BS-computed greeks = simulation bias; real options alpha needs live tick surface (>$300/yr). DeepSeek: dealer-gamma is institutionally gated (skip). | **They're partly talking past each other.** Options-*as-ML-signal* = dead (all agree). Dealer-gamma = skip (needs costly live data). **VRP via *defined-risk structures* is genuinely different and a real premium** — but Gemini's caution is valid. Verdict: **VRP is a RESEARCH track, not a confident "do now."** Treat the 4y frozen backtest as **exploratory-only**, anchor the prior on **published VRP literature**, build the premium-% cost model (Ⓕ) FIRST, and **validate primarily via live-paper on Alpaca NBBO**, not the frozen history. This reconciles all four. → P3-2. |
| **Regime-specific models** | DeepSeek: train **separate ML models per regime** (BULL/BEAR/NEUTRAL). ChatGPT: **explicitly warns against this** — separate models on small per-regime samples = hidden overfit. | **Side with ChatGPT/ClaudeMax.** Small-N per-regime models will overfit. Use regime for **sizing/risk-caps and engine on/off**, not for separate signal models. |
| **Trend allocation 25%** | ClaudeMax: **badly under-deploys** the only edge (book runs ~2.5% vol; half-Kelly on SR-0.7 is far north of 25%). Others quiet. | **Valid — but it's an owner decision.** Is this a deliberately de-risked sandbox or a real target? → P1-2 (👤): run a Kelly/vol-target analysis and decide explicitly. |
| **RL / end-to-end portfolio opt** | DeepSeek floats RL as a long shot. | **Skip.** Low-N, low-signal, non-stationary → RL overfits catastrophically. Not worth it. |
| **Both-halves: keep vs replace** | ChatGPT: keep for return-sleeves, mechanism-specific tests for overlays/tail-hedges. ClaudeMax: replace with a powered stability test everywhere. | **Compatible — converge:** replace the binary guard with a **powered stability test**, *plus* mechanism-specific tests for episodic strategies (tail hedge, carry, overlay). → P0-2. |

---

## 4. The centralized roadmap (phased, prioritized)

> **Sequencing principle:** *Validate the validator → harden/monetize what we have → fix the gate's
> diversifier-killing flaws → build new return engines on the current broker → make the one strategic
> data bet → reframe the modeling.* We do **not** fan out into 30 sleeves; we make a few high-conviction
> bets and let live evidence adjudicate. (ChatGPT's warning: a sophisticated machine with one live edge
> becomes a false-positive pressure cooker. Stay disciplined.)

### Phase 0 — Validate the validator *(do first; cheap; de-risks everything)* 🟢🟡
- **P0-1 — Positive-control the feature→label pipeline.** `[Ⓐ; ClaudeMax]` Run **12-1 cross-sectional
  momentum, 1-month short-term reversal, and low-volatility** through the *exact* swing-ML
  feature→label→CPCV path on the *exact* yfinance universe. **Success:** the pipeline reproduces the
  *published sign and rough magnitude* of ≥2 of the 3 anomalies → "IC≈0 is the market," mined-out stands,
  we've earned the right to stop mining equities. **Failure:** a well-documented anomaly *also* comes out
  IC≈0 → there is a deflationary alignment/join bug or our cost/universe kills everything → "mined out"
  is **unsafe** and we re-open the ML kills. *This is the gate on whether Phase-5 modeling work is even
  worth doing.*
- **P0-2 — Fix the two diversifier-killing gate flaws.** `[Ⓑ Ⓒ Ⓓ]` (a) Replace the binary both-halves
  guard (`scripts/walkforward/sleeves.py`) with a **powered stability test** (half-Sharpe homogeneity /
  bootstrap-CI-overlaps-zero) + pooled-HAC significance, with mechanism-specific tests for episodic
  strategies. (b) Drop/lower `standalone_vt_SR` for `{diversifier, risk_premium}`. (c) Add a
  small-size-diversifier probation path with `P(ΔSR>0) ≥ 0.75` + mandatory live-paper ratification.
  **Success:** re-run the carry F3 series through the new gate — if it now clears as a probationary
  small-size diversifier, the old guard was the problem (as predicted).
- **P0-3 — Governance: flip `REQUIRE_TRUE_WF_FOR_PROMOTION=True` for trained paths.** `[Ⓗ]` **Success:**
  no trained model can reach live without true per-fold-retrain WF; rules-based streams unaffected.

### Phase 1 — Harden & monetize what we already have 🟢👤
- **P1-1 — Explicit cash/T-bill sleeve.** `[C8; ChatGPT]` Idle capital (~75% of book) into SGOV/BIL with
  a defined risk-off liquidity policy; benchmark the whole book against **trend + T-bills**, not trend +
  zero-yield cash. **Success:** a `cash_sleeve` return stream + live policy + the book benchmark switches.
- **P1-2 — Trend allocation decision.** `[§3; ClaudeMax]` 👤 Run a Kelly / book-vol-target analysis given
  the live governor; decide explicitly: raise above 25% (governor makes it defensible) or document this
  as a deliberately de-risked sandbox. **Success:** a one-line DECISIONS entry with the chosen target +
  rationale.
- **P1-3 — Credit overlay live decision.** `[C9; ChatGPT, Grok]` 👤 Shadow the G1 credit-selective
  overlay 4–8 weeks (daily multiplier log vs the VIX-only governor), then decide enable / park / kill
  **with a written false-positive budget.** **Success:** the flag flips with a decision memo, or it's
  killed — no permanent limbo.
- **P1-4 — Live-fill back-validation as a first-class metric.** `[C10; ChatGPT, Grok]` Log expected-vs-
  actual fill, intended-vs-actual exposure, overlay-mult-applied-vs-expected, missed trades, stale-data
  events, daily per-sleeve attribution. **Success:** a weekly tracking-error report; a strategy can't
  graduate on research alone — it must show live ≈ sim.

### Phase 2 — Gate & cost-model refinement (enables the new engines) 🟡
- **P2-4 — Premium-% IV-aware option cost model.** `[Ⓕ]` **Blocking** for any options/VRP backtest.
  **Success:** option backtests cost spread as % of premium (IV-aware), not bps of notional.
- **P2-5 — Adapter, not parallel-system, discipline.** `[ChatGPT 4.3]` New asset classes (crypto,
  futures) and structures (options) enter the **existing Sleeve Lab via adapters** — no bespoke
  evaluators. (Design detail in the architecture doc.)

### Phase 3 — New return engines on the *current* broker (Alpaca) 🟢🟡
- **P3-1 — Crypto trend + cross-sectional momentum.** `[C6; all 5]` Point the existing TSMOM engine at
  BTC/ETH + a liquid Alpaca alt basket (long-flat; spot shorting limited), **hard** vol-targeted (crypto
  vol 3–5× equities). **Be honest about the short-history power floor** (Ⓖ — don't fake significance).
  **Success (falsifiable):** OOS BTC/ETH TSMOM Sharpe > 0 *and* corr-to-ETF-trend < 0.6 → a real, lowly-
  correlated new return stream. If Sharpe ≤ 0 or corr > 0.6, deprioritize.
- **P3-2 — Defined-risk index VRP (research track).** `[§3 disagreement; ClaudeMax lead, Gemini caution]`
  Sell ~10–20Δ SPY/QQQ put-spreads / short-wing condors (~30–45 DTE) **gated by the crash governor we
  already validated** (sell in VIX contango + elevated IV-rank; flat in backwardation). `component_type=
  risk_premium`, judged on **Track-A return** (not Track-B diversification — it's *short* crash risk).
  **Guardrails (non-negotiable):** premium-% cost model (P2-4) first; treat the 4y frozen backtest as
  **exploratory-only**; anchor the prior on **published VRP literature**; **promote nothing from the
  historical backtest alone — require a live-paper track on Alpaca NBBO.** **Success:** live-paper VRP
  shows a real standalone Sharpe net of realistic costs over ≥1 quarter incl. ≥1 vol spike.
- **P3-3 — Overnight vs intraday decomposition.** `[ClaudeMax #3; ChatGPT #8; Grok #3]` The equity risk
  premium largely accrues overnight (close→open); tradeable with **MOC/MOO** orders — no tick data.
  **Turnover is the whole risk** (daily → ~10%/yr cost drag). **Success:** a pre-registered net-of-cost
  test where the overnight premium clears realistic round-trip costs; else kill cleanly.
- **P3-4 — Risk-premia composite (gate the basket).** `[Ⓔ; ClaudeMax]` Equal-risk vol-targeted blend of
  {carry returns, credit-timing returns, overnight, daily-short-volume}; run *that single stream* through
  the (fixed, P0-2) Track-B. **Success:** the composite clears bars no single component did → monetize the
  near-miss pile; if it fails, the components were secretly correlated (clean negative).
- **P3-5 — Daily short-VOLUME (FINRA) overlay.** `[Ⓘ; ClaudeMax #5]` Replace the power-starved bi-monthly
  short-interest (G2, ~190 obs, killed for *power*) with **FINRA daily off-exchange short-volume**
  (~4000+ obs back to ~2009). Same idea, completely different power regime. **Success:** a clean
  pre-registered test of the short-volume-ratio z-score (aggregate timing and/or XS) clears the gate.

### Phase 4 — The strategic data bet 🔵🔴👤
- **P4-1 — Buy Norgate (~$270/yr) for research.** `[C3; all 5]` 👤 Immediate payoff: **survivorship-free
  US equities** → re-run the contaminated kills (**PEAD, F2 stat-arb, any single-name/event work**) on
  clean data before declaring them dead (C4). **Success:** PEAD/F2 re-tested on survivorship-free data;
  each gets a *final* honest verdict (real, dead, or still underpowered).
- **P4-2 — Futures trend + carry research.** `[C5; all 5]` Using Norgate continuous futures *with roll/
  carry* (the lever untestable on free data; our yfinance POC's SR +0.14 was *not* evidence against carry
  — just evidence you can't test it on dirty `=F` series). One pre-registered packet: trend-only,
  carry-only, trend+carry, asset-class risk caps, Track-B vs ETF trend, crisis analysis. **Success:** a
  pre-registered futures book clears the gate → *then and only then* P4-3.
- **P4-3 — IBKR adapter (only if P4-2 passes).** `[C5]` 👤🔴 Minimal futures-execution adapter + tiny-live
  execution validation. **Do not build before P4-2 passes.**

### Phase 5 — Modeling reframe *(gated by P0-1)* 🟡
- **P5-1 — Predict volatility/regime, not return.** `[C7]` Realized vol is far more predictable than
  return and is *directly tradeable* via options (feeds P3-2).
- **P5-2 — Meta-labeling for SIZING.** `[C7; DeepSeek, ClaudeMax]` Use the validated economic signal
  (trend, VRP, overnight) for *direction*; use ML only to predict P(trade works) → fractional sizing.
  "When does trend work?" is the ideal first meta-label.
- **P5-3 — HRP for the multi-engine book.** `[C7; ClaudeMax]` Hierarchical Risk Parity over the handful
  of sleeves — robust to noisy covariances at small N, avoids MVO blow-ups.

---

## 5. Sequencing & the single most important decision

```
P0 (validate validator)  ─────────────►  governs whether equity-ML / "mined out" work continues
   │
   ├─ P1 (cash sleeve, trend alloc, credit decision, live-fill audit)   ← do in parallel; mostly 👤
   │
   ├─ P2 (cost model + adapter discipline)  ──►  unblocks P3-2 (VRP)
   │
   ├─ P3 (crypto, VRP-research, overnight, composite, short-volume)     ← the new return engines
   │
   └─ P4 (Norgate → survivorship re-opens → futures/carry → IBKR)       ← the strategic data bet (👤 buy)
            │
            └─ P5 (modeling reframe)  ← only meaningful if P0-1 says the pipeline can detect a known edge
```

**The single most important decision (👤):** **the trend ⊕ VRP pairing as a second return engine.**
This is the review's deepest insight (ClaudeMax §6, echoed by ChatGPT/Gemini/Grok in weaker form): the
diversification we kept failing to find *between equity sleeves* (the corr<0.30 wall, hit 4×) exists
**naturally between a positive-skew engine (trend, profits in sustained crises) and a negative-skew
engine (VRP, bleeds in crashes, earns the calm).** With the governor as backstop, trend's crisis
convexity covers VRP's short-vol tail. That pairing — not a fourth weak uncorrelated sleeve — is the
most likely route from a ~0.7 Sharpe book to a ~1.0+ book. The architecture doc is built around making
this pairing safe to run.

---

## 6. The explicit "DO NOT" list (reviewer consensus + my judgment)

- ❌ Another daily US-equity XGBoost feature sweep (until P0-1 says the pipeline works *and* a new data
  source changes the premise).
- ❌ Revive intraday ML on 5-min bars (data depth + execution realism both gated).
- ❌ Re-open PEAD with threshold filters (re-open *only* on survivorship-free Norgate data, P4-1).
- ❌ Buy expensive options-surface data before live-paper VRP shows promise.
- ❌ Optimize the credit-overlay parameters again (decide enable/kill instead — P1-3).
- ❌ Build the IBKR adapter before futures research passes (P4-3 gated by P4-2).
- ❌ Separate ML models per regime (overfit; use regime for sizing only).
- ❌ Reinforcement-learning / end-to-end portfolio optimization (low-N, non-stationary → overfits).
- ❌ "Fix" CPCV by inflating paths (Ⓖ — that fabricates significance).
- ❌ Naked short vol / un-defined-risk VRP, and VIX-ETP short-vol as a *return sleeve* (catastrophic
  left tail; if ever touched, tiny + governor-gated + capped).

---

## 7. The honest fallback (ChatGPT's closing truth, which I endorse)

If, after the crypto engine, the VRP track, and the Norgate futures/carry research, **nothing passes
beyond trend**, the correct answer is not to keep mining: it is to **run trend well, size it rationally,
collect T-bill yield on idle capital, and add only overlays that improve the left tail.** A modest,
robust, low-maintenance *trend-plus-defensive-overlay-plus-cash* book is a genuinely good retail outcome
— and a far better one than a complex graveyard of false positives. The architecture in the companion
doc is designed to make *that* outcome excellent, while leaving clean, adapter-shaped room for the second
engine if it earns its place.
