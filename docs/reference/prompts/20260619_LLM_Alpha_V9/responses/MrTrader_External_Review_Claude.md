# MrTrader — External Quant Review

*Brutal-honest outside assessment. Strong opinions, falsifiable claims, ranked priorities. Written for cross-LLM synthesis — every major claim has a "how to settle it" so you can go disprove me.*

---

## 0. Bottom line up front (read this if nothing else)

1. **Your process is top-decile for a solo operator and better than several funded shops I've reviewed.** The kill ledger being longer than the keep ledger, the pre-registration, the reverted goalpost-move, catching your own Type-II then Type-I — that is the real signal. I am not flattering you; calibration matters and false criticism is as useless as false praise. *But process honesty does not make conclusions correct, and you have three specific places where I think you're either wrong or one diagnostic away from being embarrassed.*

2. **Your single live edge may be a decayed edge wearing a 19-year coat.** TSMOM 0.72 over 2007–2026 is dominated by 2008 and 2022. You prominently report *carry's* post-2015 Sharpe but I cannot find ETF-trend's post-2015 Sharpe reported with the same prominence. That asymmetry is a tell. If your live 10-ETF sleeve is post-2015 < 0.4, you are deploying a decayed strategy at 50% gross on the strength of two crisis years. **This is the #1 thing to re-run and it gates your sizing decision.**

3. **Carry is your most interesting result and your most likely to be 60% an energy bet.** Cross-sectional futures carry being *stronger* in the modern era than trend is plausible (it's a real, robust factor) but the post-2015 +0.89 almost certainly leans on short-natural-gas / energy contango. Before you deploy it as a "diversifier," prove it survives dropping your top 3 contributing markets.

4. **"Free daily US-equity directional alpha is mined out" is overstated and partly unprovable on your data.** The *practical* version ("cheap, slow, price-derived equity signals don't survive cost+beta for us") is correct and well-established. The *general* version is not — you can't conclude an asset class is dead from survivorship-biased data, and you never tested the things that actually work in equities. You half-know this (caveat #4). I'd retire the strong claim.

5. **Your highest-EV next move costs $0 and isn't a data buy — it's mining the survivorship-free futures data you already own with the full factor zoo + free CFTC Commitment-of-Traders data.** You have trend (TS) and carry (XS). You're missing cross-sectional momentum, value, basis-momentum, skew, and hedging-pressure. That's the difference between "diversified-CTA-lite, two bets" and "genuine multi-factor managed-futures book."

---

## 1. Scorecard — every load-bearing claim, my verdict, how to settle it

| # | Your claim / position | My verdict | Conf. | How to settle it (you have the harness) |
|---|---|---|---|---|
| 1 | TSMOM trend is a real, robust, validated live edge | **Real but likely decayed; over-sized** | High | Report live-sleeve Sharpe by sub-period (pre-2010 / 2010–14 / 2015–19 / 2020–26), same table you made for carry |
| 2 | Trend Sharpe ≈ 0.72 is the forward expectation | **No — forward is lower** | High | Post-2015 number + out-of-sample-since-publication haircut |
| 3 | Long-flat is the right trend construction | **Questionable — you threw away half the crisis convexity** | Med | Run long-short on same 10 ETFs; compare crisis behavior + maxDD |
| 4 | Carry ≈ 0.55–0.66, modern, diversifying | **Probably real, probably energy-concentrated** | Med | PnL attribution per market; Sharpe with top-3 markets dropped; breadth count |
| 5 | Roll drag ≈ 1.1–1.9%/yr unmodeled | **Possibly double-counted — check it's *transaction* cost not roll *yield*** | Med | Confirm back-adjusted series already embeds realized roll PnL (it should) |
| 6 | Track-B +0.17 (0.72→0.89) is the investment case | **Inflated by in-sample vol-matching** | High | Re-report Track-B with PIT rolling-vol blend, not full-sample σ-match |
| 7 | "Free US-equity directional alpha is mined out" | **Overstated; partly unprovable on dirty data** | High | Can't be settled without survivorship-free equities — so retire the strong claim |
| 8 | Equity kills (PEAD, XS-ML, options-signal, overnight) are sound | **Mostly sound — but for cost/beta reasons, not survivorship** | High | These survive even on clean data; the *general* conclusion doesn't |
| 9 | Ruler-v2 is trustworthy | **Good, with two residual leaks** | Med | Two negative-control tests below |
| 10 | Adding carry is the right next deployment | **Yes — but it's the *second* CTA factor, not diversification** | High | After the breadth + PIT-vol checks pass |
| 11 | Single-process orchestrator is sound | **Fine and not your bottleneck — stop polishing it** | High | n/a — this is the wrong layer of worry |
| 12 | Buy Norgate stocks next ($693) | **No — low alpha EV; buy only to *close the question*** | Med | Free CoT data is higher EV; mine owned futures first |

---

## 2. Are you finding real alpha, or fooling yourself?

**Verdict: you're doing real research and you have one real-but-decayed edge and one promising-but-unproven candidate. You are mostly *not* fooling yourself — which is rare — but you're fooling yourself in two specific, subtle ways.**

The two ways:

**(a) Crisis-year carry on the trend Sharpe.** A 19-year window that contains 2008 and 2022 will make almost any diversified trend strategy look like a 0.7 Sharpe. The honest forward number for diversified trend, post-publication, is roughly 0.3–0.5 across the CTA industry. The SG Trend Index was roughly flat-to-down 2011–2019 and was rescued by 2022. Your ETF sleeve will have the same shape. **You are reporting an in-sample-of-the-literature number as if it's a forward expectation.** This isn't dishonesty — it's the single most common self-deception in trend-following, and you've fallen into it by not foregrounding the modern-era number the way you did for carry.

**(b) The "diversifier" framing is a softer gate, and softer gates rescue noise.** Your Track-B fix for the VRP-mis-framing problem was *correct*, but it created a structural escape hatch: the *declaration* of "this is a risk premium" is a free parameter that routes a weak result to a softer standard (appraisal IR vs the book, worst-regime waived). You can manufacture appraisal IR out of pure anti-correlation, especially with in-sample vol-matching. See §6 for the negative control that checks whether this leaks.

Everything else? Genuinely sound. The PEAD demotion is correct (it's conditional beta). The XS-ML kill is correct (long-only "edge" = beta, dollar-neutral = noise — that's the textbook result). The overnight kill is clean and exactly right (gross +0.53, net +0.16 < floor — turnover is the enemy, you proved it). You are not a person fooling himself about equity alpha. You're a person who built a good ruler and then slightly over-trusted two of its readings.

---

## 3. Trend — the hard look

**This is your most important section because trend is the only thing you trade live with conviction, and I think the conviction is partly misplaced.**

### 3.1 Report the modern-era Sharpe. This gates everything downstream.

You report carry's sub-period decomposition (2010–19 +1.00, post-2015 +0.84–0.89). You do **not** report the same table for your live ETF-trend sleeve. I want:

| Period | ETF-trend Sharpe | maxDD | % of total PnL |
|---|---|---|---|
| 2007–2009 | ? | ? | ? |
| 2010–2014 | ? | ? | ? |
| 2015–2019 | ? | ? | ? |
| 2020–2026 | ? | ? | ? |

**Falsifiable prediction:** 2007–09 and 2022 contribute a hugely disproportionate share of total PnL, and 2015–2019 is < 0.4. If I'm wrong and post-2015 is ≥ 0.6, I'll retract — trend is healthier than I think and your sizing is more defensible. If I'm right, your "validated edge" is a decayed edge and 50% gross is over-deployment.

### 3.2 The Kelly sizing logic is dangerous framing.

> "full Kelly would be ~7.7× gross, so 50% is deeply haircut"

This is a trap. Full-Kelly leverage = Sharpe/vol = 0.72/0.10 ≈ 7.2×, which is where your 7.7× comes from. **But full Kelly assumes you know the true Sharpe with certainty, and you don't — it's estimated with large error and the forward value is lower than backtest.** Citing "7.7× full Kelly" as the anchor that makes 50% feel conservative will steadily push you toward over-leverage. The correct frame: *forget Kelly multiples entirely, target a book volatility (say 10%), size to it, and remember backtest Sharpe overstates forward Sharpe by a meaningful margin.* With parameter uncertainty you want ¼-to-½ Kelly **of the forward Sharpe**, not full Kelly of the backtest Sharpe. The 25%→50% move may be fine or may be a mistake — and which one it is depends entirely on §3.1. **Do not increase gross until you've seen the post-2015 number.**

### 3.3 Long-flat throws away the diversification you're buying trend for.

The entire portfolio justification for trend is crisis convexity. That convexity lives substantially in the *short* leg — short equities in 2008, short bonds in 2022. Long-flat captures uptrends and sidesteps downtrends but cannot *profit* from sustained downtrends, and then it gets whipsawed re-entering (exactly your "whipsawed in fast shocks (COVID)" observation — that's the long-flat tax). SPY/QQQ/TLT/GLD are trivially shortable with cheap borrow, so the usual long-flat justification (frictions) doesn't apply here.

**Action:** run a long-short version on the identical 10 ETFs, identical signal. **Falsifiable prediction:** long-short shows materially better 2008 and 2022 (deeper crisis-positive) and a better Track-B vs an equity-heavy portfolio, at the cost of worse 2010–2019 (negative carry on shorts). If the crisis convexity improves, the long-short version is the more honest "diversifier" even if standalone Sharpe is similar.

### 3.4 Breadth is too thin.

10 ETFs, of which SPY/QQQ/IWM/EFA/EEM are *all* fundamentally equity beta. Your effective breadth is ~4–5 independent bets (equity, rates, gold, broad-commodity, USD). Trend's Sharpe scales with the number of *independent* trending markets — the canonical CTA result needs 50–100+ markets. You own survivorship-free Norgate futures with 76–105 markets. **Your ETF-trend and your (decayed) futures-TSMOM are the same factor measured on a thin universe vs a broad one — and you killed the broad one.** That's backwards. Re-examine whether futures-TSMOM is "decayed" or whether the *thinness-corrected, breadth-rich* trend on Norgate is actually your best trend expression, with ETFs as the liquid execution vehicle. (See §3.5.)

### 3.5 Reconcile the trend contradiction you're sitting on.

You say: futures-TSMOM is "decayed, post-2015 ~0, redundant with ETF trend (corr 0.44)" → killed. And separately: ETF-TSMOM is "the validated edge." **These cannot both be clean if they're 0.44 correlated and one is dead post-2015.** Either:
- (a) ETF-trend is *also* dead post-2015 and you haven't looked (most likely — see §3.1), or
- (b) ETF-trend has a genuine edge that broad futures-trend lacks (implausible — broad should dominate thin), or
- (c) the futures-TSMOM "decay" is an artifact of your futures universe/cost model, and trend is actually fine on both.

This contradiction is unresolved in your docs and it's load-bearing. **Run both sleeves' post-2015 Sharpe side by side.** Whichever way it resolves, you learn something that changes a live position.

---

## 4. Carry — the hard look

**This is your best new work. I want it to be real. Here's what stands between "interesting backtest" and "deployable edge."**

### 4.1 Breadth attribution — the make-or-break test.

Cross-sectional carry being +0.84–0.89 post-2015 is plausible *because* the post-2015 commodity term structure was a carry-trader's dream: 2015–16 oil crash → steep contango → great short-energy carry; nat-gas perpetual contango; 2022 backwardation. **All of which means your modern carry Sharpe may be 50–70% an energy/nat-gas bet dressed as a 76-market diversified factor.**

**Run these three and report them:**
1. Per-market PnL attribution (which markets contributed the Sharpe, ranked).
2. Sharpe with your **top-3 contributing markets removed.**
3. **Effective breadth:** average number of markets with non-trivial weight per rebalance, and the Herfindahl of weights.

**Falsifiable prediction:** dropping nat-gas + crude + one more energy/metal cuts the post-2015 Sharpe by ≥40%. If carry survives the drop with Sharpe still > 0.5, it's a genuine diversified factor and I'll be impressed — deploy it. If it collapses, it's an energy-carry sleeve: still tradeable, but **size it as one concentrated bet and stop selling the "76-market diversification" story.**

### 4.2 Roll cost — you may be double-counting (in your favor or against).

Your concern: `|Δweight|` cost term charges nothing for rolls. Correct — a roll changes the *contract*, not the *weight*. **But:** a properly difference-adjusted (back-adjusted) continuous series *already embeds the realized PnL of holding through the roll* — that's the entire point of back-adjustment. So:
- The **roll *yield*** (the return from the curve as you roll) is the *carry signal itself* and should already be in your backtested returns if you compute returns from the back-adjusted series. **Do not subtract it again** — that would double-penalize the very thing you're harvesting.
- The only genuinely unmodeled cost is the **transaction cost of the roll trades** (bid-ask on closing front + opening next, 4–12×/yr).

**So check:** is your 1.1–1.9%/yr "roll drag" the *transaction* cost of rolling, or is it the roll *yield*? If transaction cost: for liquid futures that's ~0.3–0.8%/yr, so your 1.1–1.9% is *conservative* (good — your honest Sharpe is better than you think). If you've accidentally included roll yield: you're penalizing your own signal and the true Sharpe is higher still. **Either way this needs a clean decomposition before you trust the 0.55–0.60 honest number.** I suspect you're being conservative, which is the right error to make — but know which error it is.

### 4.3 Track-B +0.17 is inflated by in-sample vol-matching.

You flag this (caveat #4, file 03). The +0.17 dSR (0.72→0.89) is the **entire investment case for carry**, and it's computed on full-sample σ-matched blends. The honest PIT rolling-vol number is "modestly lower" — *how much* lower is the whole question.

**Action:** re-report the carry Track-B with a PIT rolling-vol blend as the **headline**, demote the σ-matched number to a footnote. **Falsifiable prediction:** the big result survives (carry dSR stays > +0.10 PIT) but your *marginal* candidates evaporate — the credit/curve overlay (+0.064 σ-matched) probably goes to ~0 and should be killed, not kept-shadowing.

### 4.4 Survivorship on futures — this one's fine.

You quantified it (0.66 full vs 0.65 history-restricted; post-2015 0.89 both), futures rarely delist, and the bias is small and roughly sign-neutral for a cross-sectional factor. **Stop worrying about this one — credit to you, it's handled.** (Contrast with equities, §5, where survivorship is genuinely contaminating.)

---

## 5. "Free US-equity directional alpha is mined out" — sound or not?

**Verdict: the strong claim is overstated and partly unprovable; the practical claim is correct. Retire the strong one.**

### Why the strong claim fails:

1. **You can't conclude an asset class is dead from survivorship-biased data.** Every cross-sectional equity verdict (PEAD redo, XS-ML, short-vol XS) is on yfinance current-listings-only. Survivorship bias removes the names that went to zero — which biases cross-sectional results in hard-to-sign ways and means **a KILL on this data is not a clean KILL.** Absence of evidence on contaminated data ≠ evidence of absence. You half-know this (caveat #4).

2. **You tested the wrong things to make that claim.** What works in equities isn't price/volume signals at daily horizon — it's *fundamental* signals done well: earnings-estimate revisions, profitability/quality (Novy-Marx gross profitability), proper value with sector neutralization, post-revision drift, accruals. You tested XGBoost on price-derived features and PEAD-by-event. Of *course* the most picked-over data on Earth (daily OHLCV of current large-caps) is mined out. That's not "equity alpha is dead," it's "the cheapest, slowest, most crowded slice of it is dead **for me**."

3. **Your concentration limit destroyed your own cross-sectional test.** Your earlier notes (5-position cap killing transfer coefficient) mean your equity XS strategies were tested with a portfolio construction that *guarantees* low breadth and low information transfer. A cross-sectional equity strategy with 5 positions is not a real test of cross-sectional equity alpha.

### Why the practical claim holds:

The *specific* kills are mostly robust regardless of survivorship: PEAD is conditional beta (−0.37 market-hedged — that's a framing result, survivorship doesn't save it); options-as-signal is cost-killed at equity transaction cost (clean); overnight is round-trip-cost-killed (clean); calendar/turn-of-month is timed beta (clean). **So the honest statement is:** *"Cheap, slow, price-derived equity signals don't survive cost and beta-adjustment for us. We have NOT tested — and cannot conclude on — whether properly-neutralized fundamental equity factors on clean data work, because we have neither the clean data nor the right signals."*

That's a much better sentence and it's defensible. The current sentence will make a real equity PM roll their eyes.

---

## 6. Is your ruler sound? (the question you most want answered)

**Verdict: your ruler is good — genuinely better than most — and it has exactly two residual leaks, both fixable with negative controls you can run this week.**

### What's right (so you don't break it):
HAC Sharpe, stationary bootstrap, PBO/CSCV, CPCV with purge/embargo, pre-registration with frozen N_TRIALS, the validate-the-validator pass (label-fidelity +0.76, 3/3 anomalies recovered), the sub-period stability guard that killed rates-carry and flipped daily-carry. This is real methodology. **Do not let any reviewer (including me) talk you into loosening these.**

### Leak #1: The "diversifier waiver" lets anti-correlation rescue noise.

Track-B judges declared diversifiers on appraisal IR vs the book + block-bootstrap P(ΔSR>0), with the worst-regime backstop **waived**. The problem: **you can manufacture a positive appraisal IR out of pure anti-correlation with zero standalone edge**, especially when the blend is in-sample vol-matched. The *declaration* ("this is a risk premium, judge it on Track-B") is a free parameter — a backdoor to a softer gate.

**Negative control (run this):** construct a deliberately zero-edge return series engineered to be ~−0.2 correlated with your live book (e.g., a sign-randomized series tilted to lean against your book's worst weeks). Push it through Track-B exactly as you would a real candidate. **It must FAIL.** If it passes on appraisal IR + ΔSR alone, Track-B is leaky and you need to add a *minimum standalone floor even for diversifiers* (lower than Track-A, but non-zero) so a pure anti-correlated null can't promote.

**Falsifiable prediction:** with in-sample vol-matching on, a well-constructed anti-correlated null passes Track-B more often than it should (>5%). With PIT rolling-vol, it's harder. This is *also* why §4.3 (PIT vol-matching) matters — the two leaks compound.

### Leak #2: The PAPER tier may still be too permissive even with the HAC floor.

You closed the original 0.30-SR-floor leak (admitted ~23% of nulls) with the HAC significance floor. Good. But you should *re-measure* the joint false-positive rate of the **current** PAPER tier (0.30 floor AND HAC p<0.05 AND worst-regime backstop) on a Monte Carlo of true nulls at your typical n.

**Negative control (run this):** 10,000 zero-edge series at n≈1,500, push through the full current PAPER tier, measure pass rate. **Target: ≤5%.** If it's higher, the tier is still permissive. Report this number — it's the single most important calibration statistic for trusting your PASSes, and you should know it to two significant figures.

### On the specific question — did you wrongly kill VRP / options?

**Yes, probably — but it's data-limited mis-framing, not a ruler bug.** VRP (the equity variance premium) is one of the most robust premia in existence. You killed single-name IV-crush correctly (cost). You *parked* index VRP as "Sharpe-weak, not alpha." But:
- VRP standalone Sharpe **is** modest (~0.5–0.8) with a fat left tail — that's the premium's nature, not a defect. Judging it on an alpha floor is the exact mis-framing you already diagnosed.
- Your options data is 4y + computed greeks + a weeks-old NBBO log — genuinely too short and too biased to clear any gate. So your VRP kills/parks are **data-limited, not real.**
- **But you don't need options data to harvest the equity-index variance premium.** See §7.3 — the VIX futures term structure is a cleaner, cheaper VRP vehicle, and you're *already computing the signal* for your crash governor.

So: trust the KILLs that clear a big margin (trend exists, carry diversifies, PEAD is beta, overnight is cost-killed). **Distrust** the marginal candidates (credit overlay, FINRA composite) and **don't trust as clean** any KILL on survivorship-biased equities or <5y options data.

---

## 7. Strategies you haven't tried — prioritized by EV × feasibility

*Grounded in what you own. Ranked. Concrete enough to build the top three. I'm not re-suggesting things you killed unless flagged.*

### #1 — Complete the futures factor zoo on the Norgate data you already own. **(Highest EV, $0, data owned.)**

You have trend (TS) and carry (XS). The documented, modern, robust managed-futures factor set has **five-plus** members, and you're running two. The rest are computable from data you already mirror, with full term structure:

| Factor | Signal definition | Why it's distinct from what you have | Build effort |
|---|---|---|---|
| **Cross-sectional momentum** | Rank all 76 markets by 12-1 (or 252-21d) trailing return; long top tercile / short bottom, inverse-vol, book-vol-target | You did *time-series* TSMOM (own-trend). XS momentum (relative strength across markets) is a **different factor** with different corr to carry — Asness-Moskowitz-Pedersen "Value & Momentum Everywhere" | ~20-line sleeve decl |
| **Basis-momentum** | Boons & Prado-Tamoni (2019): momentum of the *front*-contract return minus momentum of a *back*-contract return; long high / short low | Distinct from both carry (level of basis) and momentum (price). Strong in commodities. You have term structure → you can compute it; almost nobody at your scale does | Medium |
| **Futures value** | Long-horizon (e.g. 5y) reversal / real-price level; long cheap / short rich | Negatively correlated with momentum — classic diversifier pair | ~20-line decl |
| **Commodity skewness** | Fernandez-Perez et al (2018): sort commodities by realized skewness; long low-skew / short high-skew | Orthogonal premium, distinct economic driver | ~20-line decl |
| **Hedging pressure / positioning** | **CFTC Commitment of Traders** (FREE, weekly): commercial vs non-commercial net positioning; trade with hedging pressure | A genuinely different signal (positioning, not price/curve). You don't have this data and it's free | Need CoT loader |

**Why this is #1:** (a) data owned + survivorship-free, (b) these are *less crowded* than equity factors, (c) it directly fixes your "one-bet book" problem by adding 3–5 orthogonal-ish return streams, (d) it turns "diversified-CTA-lite" into a real multi-factor managed-futures book — which is an investable thing solo operators can actually run. Build **cross-sectional momentum first** (closest to existing infra), then **basis-momentum** (your differentiated edge — full term structure is a real asset most retail quants lack).

**Kill criteria:** each factor must clear Track-A PAPER standalone AND add Track-B value over the *combined* book (not just over trend). Pre-register before the decisive run. **Falsifiable prediction:** basis-momentum and XS-value clear; skew is marginal; XS-momentum is real but partly redundant with carry+trend (corr 0.3–0.5).

### #2 — Equity-index variance risk premium via the VIX futures curve. **(High EV, data mostly owned, fixes a wrong kill.)**

You parked VRP on bad options data, but you're **already computing VIX vs VIX3M for your crash governor.** The VIX futures curve is in contango ~75–80% of the time; the systematic short-vol-via-curve trade earns that roll-down.

- **Signal:** slope of VIX futures term structure (front vs second VIX future, or VIX-future vs spot VIX). Short-vol exposure (small, sized in vega/vol units) when curve is in contango; **flat when backwardated** — which is *literally your existing governor signal* (VIX > VIX3M = de-risk).
- **Vehicle:** VIX futures directly (if in Norgate), or the VXX/VIXY/SVXY ETP complex (in your equity data), or ES/SPX option overwrite. The curve trade is cleanest.
- **Why it's not the kill you made:** single-name IV-crush was cost-killed (correct). Index VRP was judged on an *alpha* floor (the mis-framing you already diagnosed). As a **Track-B diversifier sized small under your crash governor**, it can clear — and it's a *non-trend, non-carry* return source, which is the diversification you actually lack.
- **Tail risk:** short-vol has catastrophic left tails (XIV died Feb 2018; March 2020). **This is exactly why you size it small and gate it on backwardation** — and you already built the gate. The governor that protects trend *is the same governor* that makes short-vol survivable.

**Kill criteria:** must show a positive Track-B over the trend+carry book on **PIT vol-matched** blends, and must survive a stress overlay that includes Feb 2018 + March 2020 with the governor active. **Falsifiable prediction:** standalone Sharpe ~0.5–0.8 with ugly skew; as a governor-gated Track-B sleeve it adds +0.05–0.10 dSR. If it doesn't clear PIT Track-B, kill it for real this time.

### #3 — Rates/curve relative-value via futures (steepeners/flatteners). **(Medium EV, data owned, genuinely non-trend.)**

You have the full futures term structure across rates markets (ZT/ZF/ZN/ZB / international). Curve trades (2s10s steepener/flattener, butterfly) driven by carry + momentum of the *spread* are a real RV family that is *not* directional trend or cross-asset carry. This is the kind of return source that makes a book multi-strategy rather than multi-factor-momentum.

- **Signal:** spread carry + spread momentum on rates curve pairs; mean-reversion of butterflies around fair value.
- **Why it matters:** it's the most genuinely *orthogonal* thing in this list — RV, not trend/carry — so it does the most to de-correlate your book.
- **Caveat:** lower capacity, fiddlier execution, and you'd want IBKR fills before trusting it. Build the research sleeve, don't rush to capital.

**Kill criteria:** Track-A PAPER + Track-B over the full book; sub-period stability (rates regimes change hard — pre/post-ZIRP). **Falsifiable prediction:** marginal standalone, but the lowest correlation to your existing book of anything here.

### What I'm *not* recommending (and why):
- **Re-testing equity factors** — only worth it *after* you buy clean data, and even then it's low-EV (crowded, decayed). See §8.
- **Faster/intraday equity** — your harness has a slow-strategy prior baked in (§9), you'd need a different validation rig, and you've correctly proven turnover kills your slow edges. Don't fight your own conviction without a reason.
- **Anything you killed on cost** (single-name options, overnight) — those kills are clean.

---

## 8. Data — the single highest-EV next buy

**Verdict: don't buy anything yet. Your highest-EV "acquisition" is free.**

| Option | Cost | EV of finding alpha | Verdict |
|---|---|---|---|
| **CFTC Commitment of Traders** | **FREE** | Medium — unlocks hedging-pressure factor on data you own | **Get this first** |
| Mine owned Norgate futures (factor zoo) | $0 | High — 3–5 new factors | **Do this first** |
| Norgate US Stocks (Platinum) | $693/yr | **Low** — re-tests things that looked dead, on crowded factors | **Only to close the question, not to find alpha** |
| Live options NBBO history | >$300/yr | Low — VIX-curve VRP is a cheaper path to the same premium | Skip — use VIX futures instead |
| Intraday/tick | $$ | ~0 for your slow edges | Skip |

**On Norgate stocks specifically:** the *honest* reason to buy it is not alpha — it's to convert "we think equity alpha is dead but our data is dirty, so we can't actually know" into a clean, final answer. At $693/yr that's cheap intellectual-honesty insurance *if the open question bothers you.* But the EV of *finding* a tradeable equity edge is low (crowded, decayed, and you'd be re-testing kills). **Frame the purchase honestly: "I'm buying this to stop guessing, not to find alpha." If you can live with the open question, spend $0 and mine futures.**

**Are you leaving obvious alpha on the table by not having clean equities?** No — "obvious" alpha isn't sitting in clean US equity data; it's been arbitraged. What you're leaving on the table is *certainty about your kills*, which is a different (smaller) thing. The actual untapped alpha in your reach is in the **futures factor zoo you already own** and **positioning data that's free.**

---

## 9. Architecture verdict

**Verdict: fine, over-engineered on execution, excellent on research, and NOT your bottleneck. Stop polishing it.**

### Execution side: appropriately-built, slightly ceremonial.
- **Single-process orchestrator / single point of failure:** real risk, *trivial consequence.* For weekly-rebalance paper trading, a missed rebalance means you rebalance next week — slow strategies are forgiving. **Do not build HA infrastructure for a weekly strategy.** That's effort spent at the wrong layer.
- **Three "agents" (PM/RM/Trader):** somewhat ceremonial — these are three functions, and calling them agents adds conceptual overhead. Harmless, and it makes the audit trail clean, so fine. Don't add more agent ceremony.
- **The resilience work that *will* matter** is deferred until real capital: **broker reconciliation / state-drift hardening and fill modeling.** That's where real money leaks. Build it when you fund the account, not before.

### Research side: this is the crown jewel. One structural warning.
The Sleeve-Lab + two-track gate + pre-registration + inference keystone is genuinely good architecture for finding slow alpha. **The warning you already half-asked:** *yes, the pipeline is structurally biased toward slow, low-turnover, diversifying sleeves.* Track-B rewards diversification; the worst-regime waiver favors declared premia; everything is daily-returns-based. This pipeline will systematically *find* trend/carry/premia and systematically *fail to find* faster alpha (which needs intraday data, microstructure, faster CV, real fill modeling). That's not a bug — it matches your conviction — but **know that your harness has a prior baked in that excludes a whole region of strategy space.** If you ever want faster alpha, you need a different rig; don't try to bolt it onto this one.

### What's missing for a genuinely robust multi-strategy book:
1. **A capital-allocation layer across sleeves.** You turned it off ("equal-weight beats vol/regime on 2 sleeves" — *true with 2 sleeves*). As you add the factor zoo (§7) you'll have 4–6 sleeves and this becomes real. **HRP or simple risk-parity across sleeves** will matter then. Build it when you have ≥4 sleeves, not before.
2. **Forward-looking book risk, not just caps + governor.** You have static caps and a VIX governor. You lack a covariance-based exposure model that de-grosses when *realized correlations spike* (the thing that kills diversified books — everything correlates to 1 in a crisis). A simple realized-correlation de-gross trigger is high-value and cheap.
3. **Fill/execution modeling** — the gate to real capital. Until you have it, every number is signal-level.

### The honest meta-point on architecture:
**You're worrying about the wrong layer.** The orchestrator is fine. Success or failure is determined by *whether the edges are real*, and the binding questions there are the post-2015 trend Sharpe (§3.1) and carry's breadth attribution (§4.1). **A day spent re-running those is worth a month spent on infrastructure.**

---

## 10. The destination you're actually heading toward (say it plainly)

Trend (live) + carry (pending) = the two canonical managed-futures factors. They're 0.10 correlated — good. But **two CTA factors is a *diversified CTA*, not a multi-strategy fund.** Both are momentum-adjacent, vol-sensitive, risk-on/risk-off return streams; in a multi-year trend drawdown (2011–2019 was brutal for the category) they can suffer together as a *category*. That's a fine thing to be — a small, honest, diversified CTA is a real and respectable book. **But name it correctly so you don't over-claim diversification.** The path to a *genuinely* multi-strategy book runs through the non-trend/non-carry sources in §7: VIX-curve VRP (#2) and rates RV (#3) are the things that make it more than "a CTA." The factor zoo (#1) makes the CTA *better*; the VRP and RV sleeves make the book *different*.

---

## 11. If I were running this book Monday morning — top 5 moves, in order

1. **Re-run trend's sub-period Sharpe and report post-2015 with the same prominence as carry's.** (§3.1) This gates your sizing and resolves the trend contradiction (§3.5). *Until you've seen it, freeze gross at current levels — do not act on the 25%→50% logic.* **Highest priority, ~1 day.**

2. **Carry breadth + PIT-vol diagnostics before deploying.** Per-market attribution; Sharpe with top-3 markets dropped; Track-B on PIT rolling-vol not in-sample σ-match. (§4.1, §4.3) **Deploy carry only if it survives dropping energy and PIT-Track-B is still > +0.10.** If it's an energy bet, size it as one. ~1–2 days.

3. **Run the two ruler negative controls.** Anti-correlated zero-edge null through Track-B (must fail); 10k true-nulls through the full PAPER tier (pass rate must be ≤5%). (§6) Report both numbers to 2 sig figs. **This is what lets you *trust* every future PASS.** ~1 day.

4. **Build cross-sectional momentum + basis-momentum sleeves on owned Norgate data, and write a CFTC CoT loader.** (§7.1) Pre-register before the decisive run. This is your highest-EV *research* direction and it's free. Start the week-long build. Basis-momentum is your differentiated edge — full term structure is an asset most solo quants don't have.

5. **Spend $0 on data.** (§8) Mine futures + get free CoT first. Buy Norgate stocks *only* if the unanswered equity question genuinely bothers you, and frame it as "closing the question," not "finding alpha." Use VIX futures, not options NBBO, for VRP.

---

## 12. Appendix — falsifiable claims, collected (go disprove me)

| Claim | What confirms it | What refutes it |
|---|---|---|
| Trend is decayed post-2015 | Live-sleeve 2015–19 Sharpe < 0.4; 2008+2022 = disproportionate PnL share | Post-2015 ≥ 0.6 with PnL spread across years |
| 50% gross is over-deployment | Decayed forward Sharpe + parameter uncertainty argue for less | Healthy post-2015 Sharpe + low estimation error |
| Long-flat sacrifices crisis convexity | Long-short shows better 2008/2022 + better Track-B vs equity portfolio | Long-short shows no crisis improvement |
| Carry is energy-concentrated | Dropping top-3 markets cuts post-2015 Sharpe ≥40% | Survives top-3 drop with Sharpe > 0.5 |
| Track-B +0.17 is inflated | PIT rolling-vol blend gives materially < +0.17 | PIT blend ≈ +0.17 |
| Credit overlay should be killed | PIT Track-B ≈ 0 for the +0.064 σ-matched candidate | PIT Track-B stays clearly positive |
| Track-B has an anti-correlation leak | Engineered anti-correlated null passes Track-B > 5% | Null fails Track-B reliably |
| PAPER tier still permissive | True-null pass rate through full PAPER tier > 5% | Pass rate ≤ 5% |
| Equity "mined out" is overstated | Can't be settled on dirty data → claim is unprovable, retire it | (Only clean-data testing could revisit) |
| Basis-momentum + XS-value clear the gate | Track-A PASS + Track-B over combined book | Fail standalone or add nothing over book |
| VIX-curve VRP clears as governor-gated Track-B | +0.05–0.10 dSR on PIT blends, survives Feb'18+Mar'20 stress | No Track-B value or fails stress |
| Architecture is not the bottleneck | Edge re-runs change live positions; infra changes don't | An infra fix materially changes returns |

---

*Net: you've built something real and you're mostly honest with yourself. Your live edge needs a modern-era reality check, your new edge needs a breadth check, your ruler needs two negative controls, and your highest-EV growth is the factor zoo on data you already own. Do the five Monday moves before you touch anything else.*
