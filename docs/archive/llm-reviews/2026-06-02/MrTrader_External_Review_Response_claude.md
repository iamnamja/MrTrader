# MrTrader — External Alpha Review (Response)

*Reviewer posture: world-class quant, equity statistical-arbitrage and event-driven background, has built and killed many automated books. Brief, brutal, specific. Read Section 1 first — it reframes the whole problem.*

---

## 0. Top-line verdict (the brutal version)

Your measurement-integrity work is excellent and rare — most shops never find the bugs you found, and the fact that your numbers *collapsed* after the audit is evidence the audit worked, not evidence the project failed. **Believe your low numbers.**

But you have drawn the wrong top-level conclusion from them. Your headline read is *"durable price-feature alpha is exhausted in this universe."* The correct read is **"we have been trying to measure alpha through an instrument that mathematically cannot capture it: a 5-position, long-only book."** You are not signal-starved. You are **breadth-starved and constraint-crippled**, and your validation is now good enough to tell you the painful truth that *the way you're allowed to trade* destroys the edge before the edge ever gets a chance.

Three sentences, if you read nothing else:

1. **The Fundamental Law of Active Management says your max achievable Sharpe with 5 long-only positions is structurally tiny regardless of signal quality — you are throwing away ~90% of any cross-sectional edge before costs.** Fix breadth and neutrality before concluding any signal is dead.
2. **Your single surviving edge (PEAD) is real but its entire measured advantage rests on one hand-tuned in-sample threshold (VIX>30) that sidesteps ~2–3 historical crises — that is 2–3 events of evidence, not 6–8 folds.** Stress it with leave-one-crisis-out before you trust it.
3. **Your live system is trading a different, unvalidated strategy than the one you validated** (overlays, conviction sizing, news sizing on top of clean PEAD). Either validate each overlay or strip it.

Everything below expands these.

---

## 1. The headline insight: you are measuring edge through a straw

This is the most important thing in this document and I suspect the other LLMs will under-weight it.

### 1.1 The Fundamental Law, applied to your actual constraints

Grinold–Kahn: `IR ≈ IC × √BR × TC`

- **IC** = information coefficient (cross-sectional correlation of your signal with forward returns). For a decent equity signal, IC ≈ 0.02–0.05.
- **BR** = breadth = number of *independent* bets per year.
- **TC** = transfer coefficient = how faithfully your portfolio expresses the signal (0 = you ignore the signal, 1 = perfect). This is the term everyone forgets, and it is where you are bleeding out.

Now plug in *your* book:

- **5 positions, long-only.** Clarke–de Silva–Thorley (2002) showed empirically that a **long-only constraint alone caps TC at roughly 0.4–0.5** for a typical large-cap universe, because you can't express any of the negative-alpha views (you can only *not own* a name, you can't short it). And a **5-name cap** on a 1000-name ranking means you're expressing the signal through the top 0.5% only — your effective TC is closer to **0.1–0.2**.
- So even with a genuinely good ranker (IC = 0.04) and generous breadth, your realized IR is being multiplied by ~0.15. **You are voluntarily discarding ~85% of your signal at the portfolio-construction stage.**

Your swing ranker came in at +0.22 / t=0.17. That is *exactly what the Fundamental Law predicts* a real-but-modest cross-sectional signal looks like after you strangle it with a long-only 5-position constraint. **It is not evidence the signal is dead. It is evidence the constraint is fatal.** You cannot distinguish "no signal" from "good signal, near-zero transfer coefficient" with this book design — and you've been interpreting all your nulls as the former.

**Action:** before declaring *any* cross-sectional signal dead, re-run it as a **dollar-neutral, sector-neutral, 40–80 name** book. If +0.22 long-only-5 becomes +0.7 to +1.0 dollar-neutral-60, the signal was always there. I'd bet it does.

### 1.2 You have conflated two completely different meanings of "shorting"

You concluded shorting is an "anti-edge" from QualityShort (−0.90) and inverted-long composites. **Those failures do not generalize to dollar-neutral cross-sectional shorting, and treating them as if they do is the single most expensive error in the brief.**

- **Thesis shorting** (QualityShort: "short fundamentally broken companies") — correctly an anti-edge. Beaten-down high-beta value names rally hardest on risk-on. You proved this. Good.
- **Cross-sectional shorting** ("short the *bottom decile of the same validated ranker* that you're going long the top of") — this is the entire foundation of quant equity market-neutral (AQR, Two Sigma equity stat-arb, Millennium pods). It is *not* a thesis about bad companies; it's the symmetric expression of the same ranking signal, and it's what cancels your F2 beta problem.

You diagnosed F2 perfectly ("long-only is regime-exposed by construction; top-quintile momentum flips meaning in a VIX spike") and then prescribed the wrong remedy by lumping all shorts together. **The fix for F2 is not "find a purpose-built short signal." The fix is "stop trading long-only" — short the bottom of the ranker you already have.** A long-short spread of a cross-sectional momentum/quality signal is regime-neutral by construction, which is the property you correctly said you were missing.

---

## 2. Where your validation is still fooling you

Your pipeline is genuinely strong — CPCV, purge/embargo, true per-fold retrain, DSR with conservative N_TRIALS, N_eff = n_folds. Better than 90% of shops. The remaining holes, in priority order:

### 2.1 The VIX>30 crisis block IS the edge AND IS the overfit (highest-priority issue)

You wrote it yourself: *"this block is credited with the entire edge: it lifts the P5 tail from −0.29 to +0.01."* Sit with that. **Without the VIX>30 block, PEAD has a negative tail and likely fails the gate. With it, it passes.** So your entire surviving edge reduces to a single binary rule: *don't trade PEAD in crises.*

Now ask: how many VIX>30 episodes are in your sample? Realistically ~3 (2018 Q4 brief, 2020 COVID, 2022, maybe an Aug-2024 intraday touch). **VIX>30 with that threshold value was chosen on the same data that contains those ~3 crises.** Your t=2.26 / 6–8 folds machinery does not capture this, because the crisis filter is a *structural choice*, not one of your 300 counted trials. The effective evidence for "the block works" is **~3 events**, and the threshold (30, not 28 or 33) was fit to them.

This is textbook overfitting to tail events disguised as a risk rule. The DSR cannot see it.

**Stress tests to run before you trust the +0.546:**
- **Leave-one-crisis-out:** drop each VIX>30 episode in turn; does the edge survive removing the COVID window? If COVID alone carries it, you have a 1-event edge.
- **Threshold sensitivity:** sweep the block at 25 / 28 / 30 / 33 / 35. If the edge is stable across 25–35, the rule is robust. If it's only good at exactly 30, it's fit.
- **Replace the discrete block with a regime-general control:** instead of "VIX>30 → halt," use **portfolio-level volatility targeting** (scale gross by realized/target vol) or a **trend filter** (SPY < 200dma → reduce gross). These are not fit to your specific crises, they're general, and they'll deliver most of the protection without the in-sample threshold. If PEAD survives under a *generic* regime control, I believe it. If it only survives under VIX>30-exactly, I don't.

### 2.2 Your effective breadth for PEAD is far smaller than your fold count suggests

40-day holds + earnings clustered into 4 reporting seasons/year = **massively overlapping, regime-correlated holding periods.** Within a single earnings season your 5 (or 20) positions are all exposed to the *same* macro tape over the *same* 40-day window — they are one correlated bet wearing many names, not independent bets. The number of *independent* PEAD bets in your whole sample is closer to **(years × 4 seasons) ≈ 30–40**, and they're autocorrelated on top of that.

Your N_eff = n_folds correction is the right *instinct* but it corrects for path reuse, not for **within-fold return autocorrelation from overlapping event holds.** The honest N_eff is smaller still. Net: **t=2.26 is optimistic; treat PEAD as "consistent with a small real effect we cannot statistically separate from a lucky threshold."** That's not a reason to kill it — it's a reason to size it like a hypothesis, not a conviction.

### 2.3 Your "sacred holdout" is dated in the future — you have no historical holdout

The brief says the sacred boundary is **2026-11-09**. Today is 2026-06-02. **There is no data after that boundary yet.** So your holdout set is *empty*, every CPCV run has used all available history, and there is no untouched historical block. Your "holdout" is really a *promise to forward-test in paper*, which is good discipline but is **not** an out-of-sample historical test. Be honest in your own docs: every backtest number is in-sample at the meta level (developed against all of history), and your only true OOS is the live paper forward. Either (a) this is a typo for a *past* date and you should carve out a real historical holdout (e.g. everything after 2024-06-01) and never touch it, or (b) you should stop calling it a holdout and call it "forward validation."

### 2.4 You are over-purging and starving your swing tests of power

Purge = 85 calendar days, test windows = ~62 days. **Your purge is longer than your test window.** And the justification ("60-day feature lookback + 20-day label + buffer") is half wrong: feature *lookback* looks backward and doesn't leak forward labels — it creates feature overlap, not label leakage, and the correct purge is driven by the **label horizon (20d) + embargo**, not the feature lookback. You're purging ~4× more than necessary, which (a) shreds your effective sample, (b) produces the "62-day window aggregation artifacts" you flagged in PF/Calmar, and (c) makes the t≥2.5 capital bar nearly unreachable for mechanical reasons rather than edge reasons. **Fix: purge ≈ label horizon + small embargo (~25–30 days), not 85.** This alone may move your swing power materially.

### 2.5 N_TRIALS = 300 is a large undercount (but it doesn't matter much — DSR is saturated anyway)

Every threshold sweep (surprise cutoff, hold length, VIX level, entry window, priced-in on/off), every bug-fix re-run across the 13-round audit, and every "let me try X" is a trial. True N is plausibly in the thousands. Since DSR saturates above Sharpe 2 and is non-binding here, this is mostly academic — but it reinforces 2.1: **your real multiple-testing exposure is much larger than your machinery models, and the VIX threshold is an uncounted trial.**

### 2.6 Your cost assumption is likely too optimistic *for the specific names PEAD trades*

5 bps/side is a fine *average* large-cap assumption. But **PEAD enters names within ~3 days of an earnings report — exactly when their spreads are widest and their realized volatility is highest.** Earnings-window spreads on even large caps routinely blow out to 15–40 bps, and your marketable-limit "ask + 10 bps" crosses into that. Your live tracking error vs backtest is probably *partly* this, not just the overlays. **Action:** pull realized fill slippage on your live PEAD fills vs the modeled next-open and compare to 5 bps. If live cost is 20–35 bps round-trip on earnings names, a chunk of your modest +0.55 is already gone in reality. This is concrete, testable, and potentially fatal — check it this week.

### 2.7 The live overlays mean you are not trading the thing you validated

You validated **clean 5%-equal-weight PEAD with a VIX block.** You are trading **PEAD + regime sizing + news-sentiment sizing + opportunity-score gate + macro-calendar block + conviction sizing + the full 10-rule RM chain.** None of those overlays were independently validated to *add* Sharpe to PEAD. Every unvalidated overlay on a fragile edge is, in expectation, a zero-mean perturbation with negative variance contribution — it can only widen your distribution around an already-thin mean. **Rule: never stack unvalidated discretion on your one validated edge.** Either prove each overlay improves PEAD's OOS Sharpe in the same CPCV harness (run PEAD-with-overlay vs PEAD-clean head-to-head) or strip it. Until then your live realized Sharpe is uninterpretable.

---

## 3. Are your promotion gates calibrated correctly?

**Direction is right, levels are slightly miscalibrated for your N.**

- Significance-first (t-stat over bare mean-Sharpe) is correct and a real improvement. Keep it.
- **The t≥2.5 capital bar is likely unreachable for a real-but-modest edge at N_eff ≈ 6–8 — you're right to worry.** With 6–8 folds, t≥2.5 requires mean/std ≈ 0.9–1.0 across folds, i.e. an almost crisis-proof signal. Real durable equity edges (Sharpe 0.5–1.0) will *correctly* fail this most of the time. **You are at risk of rejecting good edges, not admitting noise.** The fix is not to lower the bar — it's to **raise N**: more folds via the over-purge fix (2.4), longer history, and *more independent bets per fold* via breadth (Section 1). Raise breadth and the same edge clears t≥2.5 honestly. This is the same disease as everything else: the cure is breadth, not threshold-tuning.
- **Add one gate you're missing: a live-vs-backtest tracking gate.** Promotion to CAPITAL should require that live paper realized Sharpe is within, say, 1 std of backtested expectation over the paper window. Given 2.6/2.7, this would have flagged the overlay/cost divergence before real capital.

---

## 4. Where the alpha actually is (ranked, concrete — your Q4/Q5/Q9)

Given your constraints (US large-cap, free + modest data, $100k automated, good validation), here is where durable expected value lives, best first. **All assume the breadth + neutrality fix from Section 1.**

| Rank | Strategy | Why it's real | Data needed | Effort |
|---|---|---|---|---|
| 1 | **Analyst-revision momentum (standalone, cross-sectional)** | One of the most robust, persistent anomalies in the literature (Chan-Jegadeesh-Lakonishok). Slow diffusion of revisions = same mechanism as PEAD. You *already have* FMP upgrade/downgrade history and only use it as a PEAD sub-filter. This is your single best untapped free edge. | FMP (have it) | Low |
| 2 | **Dollar-neutral multi-factor composite** (revisions + momentum + quality + low-vol + PEAD-tilt), sector-neutral, 40–80 names/side | This is the actual job. Combining low-correlation factor premia is the only thing that reliably clears Sharpe ~1 net for a small shop. The short side is the *bottom of the same ranker*, not thesis shorts. | Short interest/borrow (acquire — see §5) | Med |
| 3 | **SUE-based PEAD** (replace raw %surprise) | Raw `(actual−est)/|est|` is noisy and explodes for tiny estimates. **SUE = surprise standardized by the name's own historical surprise std** is the academically correct PEAD signal and is cleaner. Plus combine with the drift mechanism in #1. | FMP (have it) | Low |
| 4 | **Time-series trend / regime overlay** (portfolio-level gross scaling on SPY-vs-200dma + vol-target) | Not standalone alpha — it's the *general* version of your VIX block, and it improves the Sharpe of everything above without being fit to your 3 crises. Replaces the overfit in §2.1. | Have it | Low |
| 5 | **52-week-high / fresh-high momentum** (George-Hwang) | Documented, free, low-correlation to standard momentum, mechanically simple. Worth one honest CPCV pass as a composite component. | Have it | Low |
| 6 | **Short-term reversal (1–5 day), large-cap, dollar-neutral** | Real in large-cap; the question is whether 5 bps survives the turnover. Needs breadth and tight cost modeling. Test honestly before believing or dismissing. | Have it | Med |

**Outside-the-box reframe:** notice #1, #3, and PEAD all exploit the *same* economic mechanism — **slow diffusion of public information / underreaction.** Don't treat them as separate strategies competing for the gate. **Combine them into one "information-diffusion sleeve"** (revisions + SUE-PEAD + guidance drift). Correlated-mechanism signals stacked into one book *increase the breadth of the same durable edge* rather than diluting it — which is exactly the breadth you're missing. This is probably your fastest path to a gate-clearing book.

---

## 5. Single most valuable dataset to acquire (your Q4, ranked)

For a $100k automated large-cap shop, ranked by **alpha-per-dollar**, not by sex appeal:

1. **Short interest + borrow availability/cost — ACQUIRE FIRST.** Cheap (FINRA semi-monthly short interest is near-free; borrow data is modestly priced). *Changes* in short interest, days-to-cover, and crowded-short metrics are a documented standalone large-cap anomaly (Boehmer-Jones-Zhang; Drechsler-Drechsler). Critically, **it is the precondition for the entire dollar-neutral redesign** — you cannot run a market-neutral book responsibly without knowing what's hard/expensive to borrow. Highest alpha-per-dollar by a wide margin *for your situation*.
2. **Proper PIT fundamentals** (you approximate with FMP filing dates; decent). Marginal upgrade; only worth it once #1 and the redesign are in place. Medium.
3. **Options data (IV / skew / term structure)** — **highest theoretical ceiling, wrong call for you right now.** IV-crush, earnings straddles, put-skew-as-crash-signal, and options-overlay PEAD are genuinely the richest vein — but you have *zero* options backtest infrastructure, and building a clean, PIT, survivorship-safe options harness is a 6–12 month money-and-time pit that will swallow a small shop. **Defer until you have a working dollar-neutral equity book funding the effort.** The honest version of "highest ceiling" is "not until you're ready."
4. **PIT corporate actions** (buyback/M&A/index dates) — useful but narrow; you correctly refused to build on a non-PIT feed. Acquire opportunistically.
5. **Alternative data** (cards, web, satellite) — **worst alpha-per-dollar for a small shop.** Expensive, short history (overfitting magnet), fast decay, and you'd be the last to the trade. Skip entirely at your scale.

**One-line answer: buy short-interest/borrow data, because it is cheap, it is a real edge by itself, and it unlocks the dollar-neutral fix to F2.**

---

## 6. Architecture: keep / fix / kill (your Q6)

| Component | Verdict | Reasoning |
|---|---|---|
| Three-agent PM/RM/Trader over Redis | **Keep** | Clean separation, fine engineering. Not the problem. |
| Live risk-overlay stack on the validated edge | **Fix (strip or validate)** | §2.7 — you're trading an unvalidated strategy. Validate each overlay head-to-head in CPCV or remove it. |
| 5-position cap / 5% max single | **Kill for cross-sectional/event books** | §1 — destroys the transfer coefficient and adds uncompensated idiosyncratic variance. PEAD wants 20–40 small positions, not 5 big ones; a factor book wants 40–80/side. Keep a cap, but at the *gross/sector/heat* level, not a 5-name count. |
| Long-only / long-bias default | **Kill as a default** | §1.2 — the long-only constraint is the F2 cause. Default to dollar/beta-neutral. |
| VIX>30 discrete crisis block | **Replace** | §2.1 — swap for portfolio vol-targeting + trend filter (regime-general, not crisis-fit). |
| Intraday 5-min agent path | **Kill** | §7 — you can't win microstructure without TAQ/L2 and low-latency infra. Structural cost-drag. Free up the capital allocation. |
| Regime classifier | **Keep, demote** | Useful as a gross-exposure overlay, not as a sizing oracle. |
| Sacred-holdout discipline | **Keep, fix the date** | §2.3. |

---

## 7. Triage of dead approaches (your Q7)

| Approach | Status | Verdict |
|---|---|---|
| Swing ML cross-sectional ranker | **NOT dead — mis-deployed** | §1. Resurrect dollar-neutral, sector-neutral, 40–80 names, with the over-purge fix. The ranking IC is plausibly real; long-only-5 execution killed it. **This is your highest-value resurrection.** |
| Intraday 5-min meta-model | **Genuinely dead for you** | No TAQ/L2/colocation, structural cost-drag, gross PF 0.94 *before* costs. Abandon. Stop spending the 30% capital sleeve on it. |
| QualityShort | **Dead as a thesis — but the failure is informative** | The mechanism ("broken names rally hardest risk-on") confirms shorts must be *ranker-relative*, not thesis-driven (§1.2). Don't revisit as-is. |
| Insider clusters | **Dead in large-cap (no breadth)** | Real edge concentrates in small/mid where you've (correctly) chosen not to play. Shelve. |
| Small/mid-cap PEAD | **Correctly rejected — good science** | Your survivorship-safe null is probably *right*; the literature's small-cap premium is partly survivorship/illiquidity bias. One sensitivity check: was 20 bps too punitive given marketable limits? If costs at 12–15 bps flip it, note it; otherwise leave dead. |
| Buyback announcements | **Correctly stopped** | Needs a PIT feed; refusing non-PIT was the right call. |

---

## 8. Clean-sheet recommendation (your Q8) — what a disciplined small shop should build

Strip the project to this:

**Core book — dollar-neutral, sector-neutral, multi-factor.**
- Universe: Russell 1000 (you have PIT membership — good).
- Signal: composite z-score of [revisions momentum + SUE-PEAD tilt + 12-1 momentum + quality + low-vol], sector-neutralized.
- Construction: long top ~40–80, short bottom ~40–80, gross ≤ 80% NAV (your existing cap), **net beta ≈ 0**, weekly rebalance, position sizes risk-parity-ish (inverse-vol), max single ~1.5–2%.
- This is the book that can actually clear Sharpe ~1 net and survive regimes — because it harvests low-correlation premia with real breadth and the F2 beta cancels.

**Event sleeve — information-diffusion (PEAD-family).**
- SUE-based PEAD + standalone revisions drift, 20–40 small positions, held ~30–40 days.
- Crisis control via portfolio vol-target / trend filter, **not** a fitted VIX threshold.
- This sleeve correlates low with the core book's rebalance-driven returns; run it as a satellite.

**Overlay — regime gross-scaling.** SPY<200dma and rising-realized-vol → cut gross. Validated independently as a Sharpe-improver, not bolted on by faith.

**Kill:** intraday, thesis shorts, the 5-position frame, the long-only default, the discrete VIX block, and every unvalidated live sizing overlay.

**Sequence the validation:** because the core book is dollar-neutral with high breadth, it will generate the independent-bet count you currently lack — which means it can *actually clear your t≥2.5 capital bar honestly*, unlike anything you've tested so far.

---

## 9. The uncomfortable meta-truth (read before you over-invest)

You are hunting for high-Sharpe alpha in **US large-cap equities — the single most efficient, most arbitraged market on earth — with free data and $100k.** A world-class quant's honest prior: **the durable, capacity-light edge available to you is ~0.5–1.0 net Sharpe at the book level, not 2+.** Your validation pipeline is not failing — it is *correctly reporting that the easy alpha isn't there,* which is the most valuable thing a validation pipeline can do. The shops earning 2+ Sharpe in this universe are doing it with TAQ, colocation, options, securities-lending desks, and PB-grade data you don't have and shouldn't try to fake.

So the strategic move is **not** "find the hidden high-Sharpe signal." It is:
1. **Accept PEAD is real-but-small, size it as a hypothesis, and de-overfit the crisis block.**
2. **Fix breadth + go dollar-neutral** so your *existing* signals can finally express themselves and clear the gate honestly.
3. **Harvest the information-diffusion family** (revisions + SUE-PEAD) as one combined sleeve.
4. **Defer options** until the equity book funds the infrastructure.
5. **Stop adding unvalidated overlays to fragile edges.**

A 0.8–1.2 net Sharpe dollar-neutral large-cap book on free data, built honestly, is a genuinely good result for a small shop and is *achievable* with what you have. Chasing a 2+ is how disciplined shops talk themselves back into the overfit you just spent 13 rounds escaping.

---

## 10. Prioritized next-step roadmap (do in this order)

1. **This week — kill the live divergence risk.** Pull realized PEAD fill slippage vs modeled next-open (§2.6). Run PEAD-clean vs PEAD-with-overlays head-to-head in CPCV (§2.7). If overlays don't add OOS Sharpe, strip them today.
2. **This week — stress the one edge you have.** Leave-one-crisis-out + VIX-threshold sweep on PEAD (§2.1). Replace the discrete block with vol-target/trend if it survives generically; pause live PEAD if COVID alone carries it.
3. **Next 2 weeks — fix the harness.** Correct the over-purge to ~label-horizon+embargo (§2.4). Fix or repurpose the future-dated holdout (§2.3).
4. **Next month — the big one.** Re-run the swing ranker **dollar-neutral, sector-neutral, 40–80 names/side** (§1). This is the experiment that tells you whether your "dead" ML was ever dead. I expect it wasn't.
5. **Next month — free upgrades.** Add standalone revisions momentum and SUE-PEAD; combine into the information-diffusion sleeve (§4).
6. **When budget allows — acquire short interest/borrow data** (§5) to unlock responsible dollar-neutral and a documented standalone edge.
7. **Defer indefinitely:** options infra, alt data, intraday, small-cap.

---

## 11. Direct answers to your 9 questions (index)

1. **Validation sound?** Strong, with 7 real holes — the binding ones are the VIX-block overfit (§2.1), the optimistic effective-N for overlapping event holds (§2.2), and the empty/future-dated holdout (§2.3). N_eff = n_folds is the right *instinct* but still too loose for overlapping event returns.
2. **Gate calibrated?** Direction right; t≥2.5 is unreachable for real-but-modest edges at N_eff 6–8 — fix by raising breadth, not lowering the bar (§3). Add a live-vs-backtest tracking gate.
3. **Long-only ML exhausted?** No — **mis-deployed.** F2 diagnosis correct; the fix is dollar-neutral + breadth, not abandonment (§1).
4. **Highest-EV alpha + best dataset?** Information-diffusion family + dollar-neutral multi-factor (§4); acquire **short interest/borrow** first (§5).
5. **Abandon ML ranking?** No — resurrect it dollar-neutral. Go all-in on event/structural *and* keep the (fixed) cross-sectional book; they're complementary (§7, §8).
6. **Multi-agent overlays helping or hurting?** Hurting — you're trading an unvalidated strategy (§2.7, §6).
7. **Dead-approach triage:** swing ranker = resurrect; intraday/QualityShort/insider/small-cap = stay dead; PEAD = de-overfit (§7).
8. **Redesign?** Yes — dollar-neutral multi-factor core + information-diffusion sleeve + regime gross-overlay; kill the 5-position long-only frame (§8).
9. **What I'd trade with this stack:** the §8 book — dollar-neutral large-cap multi-factor (revisions + SUE-PEAD + momentum + quality + low-vol), sector-neutral, 40–80/side, weekly rebalance, portfolio vol-targeted, with a satellite information-diffusion event sleeve. Net target ~0.8–1.2 Sharpe, which is the honest ceiling here.

---

*One closing brutal note for your synthesis pass: if another LLM tells you the answer is "add options" or "try a deep neural net on more features," it's reaching for sophistication you can't yet validate. The unglamorous truth — fix breadth, go neutral, de-overfit your one edge, combine correlated-mechanism signals — is less exciting and far more likely to actually move your honest Sharpe. Weight accordingly.*
