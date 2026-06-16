# MrTrader — External Quant Review (Alpha-v8)

*Reviewer posture: systematic PM/researcher, multi-strat shop. I have read 01–09 including the
Ruler-v2 design, the pipeline SSOT, both prior program writeups (Alpha-v7 F-series, Alpha-v8
G-series), and the bug/decision ledgers. This review is written to be **additive** to the
2026‑06‑14 5‑LLM panel, not a re-run of it.*

**Confidence legend:** `[C: High/Med/Low]` = my confidence in the *claim*. `[FALSIFIABLE]` = a
claim you can kill with a specific test, stated inline.

---

## 0. What the prior panel already covered (so this doesn't repeat it)

The Alpha-v7 panel + your own G-series already swept, and correctly closed, almost the entire
**directional-equity-premium** space: XS-equity ML, intraday ML, large-cap PEAD, options-as-features,
ETF relative-value (F2), calendar/overnight/FOMC *as additive sleeves* (F1a/F1c), rates carry (F3,
robust near-miss), credit/curve overlay (G1, candidate), aggregate short-interest (G2, killed),
and the VIX-term crash governor (live). The panel's verdict — "trend is the only standalone edge;
build a 0.7–0.9 book around it" — is **defensible given where it looked.**

**The single most important observation in this review:** the entire search to date lives inside one
quadrant — *directional, price-based, US-equity, daily.* Three orthogonal axes have barely been
touched, and each sits outside your "mined out" conclusion:

1. **Volatility/convexity as a harvestable asset** (option *structures*, not options-as-signals). The
   prior panel deferred this as "nickels in front of a steamroller" — that critique applies to *naked*
   short vol, not to the program I describe in §3.
2. **A different, less-efficient universe — crypto.** Mentioned once ("barely touched"), never run
   through the lab. Your trend engine already exists; you have Alpaca execution.
3. **Survivorship-free data**, which silently contaminates every long/short and event study you've run
   (and therefore several of your "kills").

So my headline disagreement is narrow and specific: **"free daily US *equity* data is mined out for
additive *directional* alpha" is true and well-evidenced. "We have exhausted the retail-accessible
edge envelope" is not** — you have not seriously tested vol-premium harvesting, crypto, or anything on
survivorship-free data.

---

## 1. Verdict

**Is this on a path to durable alpha? Qualified yes — but you are about to mis-conclude "done."** The
platform, discipline, and gate are genuinely strong (top-decile for a solo operator; better than many
funded pods I've seen). But you have been mining one quadrant to exhaustion and calling the *quadrant's*
emptiness the *map's* emptiness. **The single biggest thing I'd change: stop hunting for additional
crisis-uncorrelated *equity diversifiers* (the corr<0.30 wall you keep hitting) and instead build a
second *return engine* in a different risk class — a defined-risk volatility-risk-premium book — and
pair it with trend.** Trend is a positive-skew, long-crisis engine; VRP is a negative-skew, short-crisis
engine. Run as a pair under your existing governor, they diversify *through the cycle* even though each
is individually crash-exposed. That pairing — not a fourth weak uncorrelated sleeve — is your most
likely path from a ~0.7 Sharpe book to a ~1.0+ book. `[C: Med-High]`

---

## 2. Methodology & architecture critique

I'll lead with the critiques that actually change a decision, in priority order. Several are sharp; one
or two may overturn a load-bearing conclusion.

### 2.1 "All 23 bugs inflated results; the re-validation gap is zero" — this is contradicted by your own ledger. `[C: High]` `[FALSIFIABLE]`

This is your most load-bearing methodological claim — it's the justification for trusting the **IC≈0 /
"mined out"** verdict. It is **not safe as stated**, and your own change-log proves it:

- **#PERFOLD2 (2026‑05‑31):** a string-vs-float regime-map bug meant `build_train_matrix_for_window`
  returned an **empty X for every fold → CPCV reported mean Sharpe 0, n_paths=0.** That is a *deflationary*
  bug: it manufactured a no-edge result.
- **#339 (swing regime-map crash)** and **#342 (intraday day-axis desync):** both shipped to prod, both
  passed the unit suite **vacuously**, both zeroed/contaminated results.
- The pattern across these: **bare `except Exception: continue`** swallowing errors so an entire window
  silently drops. A swallowed-exception or off-by-one in the **feature→label join** produces *exactly* the
  signature you are most confident in — "infra is sound, IC≈0, the signal isn't there."

The general claim "look-ahead inflates, so all our bugs inflated" is half-true: *leakage* bugs inflate,
but **alignment / join / PIT-membership / swallowed-exception bugs deflate**, and you demonstrably had
several. "We re-ran the kills and nothing came back" tests the *gate*; it does not test whether the
**feature pipeline can surface a known effect at all.**

**The test that settles it (do this first — it's cheap and de-risks your entire thesis):** run a
**published, robust cross-sectional anomaly through the *exact* swing-ML feature→label→CPCV pipeline on
the *exact* yfinance universe** — e.g. 12‑1 cross-sectional momentum, 1‑month short-term reversal, and
low-volatility. These have known signs and rough magnitudes.
- If the pipeline reproduces their sign/magnitude → your IC≈0 is the *market*, the "mined out"
  conclusion stands, and you've earned the right to stop. **This is the positive control your
  `gate_calibration` (tsmom_4y/19y on a *return series*) does not provide — it never exercises the
  feature-construction path.**
- If a well-documented anomaly *also* comes out at IC≈0 → there is a deflationary bug or your
  cost/universe is killing everything, and "mined out" is **unsafe**. You'd re-open the ML kills.

I'd put real money on you not having run this specific control. Until you do, "IC≈0" is "IC≈0 *or* a
silent alignment bug," and you can't tell which.

### 2.2 The both-halves guard is an unpowered binary bolt-on sitting on top of a sophisticated gate. `[C: High]`

You spent enormous, admirable effort making significance *principled* — HAC-SR (Lo), stationary
bootstrap, a trial-count-shrunk Bayesian posterior, live-paper-by-construction at CAPITAL. Then you
gate on top of it with a statistic that has **no power accounting at all**: "sign must be positive in
both halves." This guard "alone killed carry."

Quantify it. A *true* edge of annual SR 0.5, split into two ~5-year halves, has per-half SR estimation
SE ≈ 1/√5 ≈ 0.45, so P(positive in one half) ≈ Φ(0.5/0.45) ≈ 0.87, and **P(positive in BOTH) ≈ 0.76.**
You have a **~24% false-negative rate on a genuine SR‑0.5 edge** from this guard alone — *before* the
rest of the gate. For SR 0.35 it's worse (~35% FN). The guard cannot distinguish "unstable/regime-fluke"
(what you want to catch) from "real but noisy" (what you want to keep). It is the one piece of your
methodology that re-creates the Type-II machine Ruler-v2 was built to retire.

**Fix:** replace "both halves positive" with a *powered stability test* — e.g. test that the two
half-Sharpes are not *significantly different* (a Chow/homogeneity test, or a bootstrap CI on the
half-difference that overlaps zero), and separately that the *pooled* HAC-SR is significant. "Stable"
should mean "no evidence of a structural break," not "lucky enough to be positive twice."

### 2.3 Track-B's `standalone_vt_SR > 0.20` condition partially defeats Track-B's own purpose. `[C: High]`

Track-B exists to admit things that are **weak alone but improve the book.** Yet a Track-B pass requires
`standalone_vt_SR > 0.20` *and* `corr_to_book < 0.30` *and* IR≥0.20 *and* P(ΔSR>0)≥0.90 — a four-way
conjunction. The most valuable diversifiers in real books (carry, vol-selling, tail hedges) have *low
or even negative* standalone Sharpe — that's *why* they diversify; they're paid in regimes the core
isn't. Requiring decent standalone Sharpe **selects against exactly the genuine diversifiers Track-B is
supposed to catch.** The appraisal IR already encodes "improves the book"; the standalone-SR floor is
redundant at best and counter-productive at worst. **Drop or sharply lower `standalone_vt_SR` for
`component_type ∈ {diversifier, risk_premium}`.** (You already waive the worst-regime floor for these —
extend the logic.)

### 2.4 Your cluster of "robust near-misses" is diagnostic, not noise — and your loss function is wrong for diversifiers. `[C: Med-High]`

Look at what keeps happening to your *economically-motivated* candidates:
- **Carry (F3):** point_SR +0.314, residual‑α **+2.10 (clears the 2.0 bar)**, Track-B 7/8, misses on
  HAC p=0.0998 and P(ΔSR>0)=0.886.
- **Credit overlay (G1):** dSharpe +0.064, all-3-crises, both-halves stable → promoted *candidate*.
- **PEAD:** your own pipeline note (post BUG-23 full-coverage fix) calls it **"real-but-underpowered,"**
  not "no edge."

When real-mechanism candidates *repeatedly* land at p≈0.10, P(ΔSR>0)≈0.88, residual-α≈2.1 — i.e.
clustered *just* below the line — the honest Bayesian reading is **not** "these are nulls." It's "these
are real-but-weak, and my data budget gives me ~80–88% power, so I am correctly uncertain." Treating an
**88% posterior as a FAIL** is a decision-theory error for a *diversifier*, because the loss function is
asymmetric: the cost of a false-accept at small allocation is a few bps of drag; the benefit of a
true-accept is genuine book diversification. You raised Track-B P(ΔSR>0) 0.85→0.90 (OD-5) because 0.85
was "a weak 1-in-7 bar." For a *core/Capital* sleeve, fine. **For a small-allocation declared
diversifier, you have it backwards — that bar should be *lower* (~0.75), gated by (a) small size and
(b) live-paper ratification, not by a one-shot offline p-value.** Let weak-but-real diversifiers in *on
probation at small size* and let the live track record adjudicate (which is exactly the Bayesian
machinery you already built for CAPITAL).

### 2.5 You never gate the *basket* of near-misses — only each one in isolation. `[C: Med-High]` `[FALSIFIABLE]`

This is the same diversification-math argument you (correctly) make for trend+diversifiers, but you've
failed to apply it to the **sub-book of weak premia.** Carry, credit, curve, daily-short-volume,
overnight, calendar — each individually weak/near-miss. But a **vol-targeted equal-risk basket** of
4–5 *mutually* low-correlated weak premia can have a basket Sharpe and a basket-vs-book IR that clears
the gate *even when no single component does* — because their idiosyncratic noise partly cancels.
**Test:** assemble a single `risk_premia_composite` sleeve (equal-risk blend of carry + credit-timing
returns + overnight + daily-short-volume signals), and run *that* through Ruler-v2 Track-B. Gate the
composite, not the components. I'd bet the composite clears bars no single premium does. This is the
cleanest way to monetize the pile of "honest near-misses" you've accumulated.

### 2.6 Survivorship bias contaminates specific kills — those aren't "dead," they're "unknown." `[C: High]`

You acknowledge yfinance survivorship bias, then proceed to treat survivorship-contaminated kills as
settled. yfinance silently drops delisted names, which **systematically biases any long/short or
event study** — and the direction differs by strategy, so you can't even sign the bias by hand:
- **PEAD / event studies:** the worst post-earnings names (frauds, blowups → delisted) vanish, so your
  *long*-drift looks worse and your *short*-drift can't be measured at all.
- **F2 ETF stat-arb / mean-reversion:** "cheap gets cheaper → delisted" losers are gone, which *flatters*
  mean-reversion (the ones that didn't revert disappeared).
- **Any cross-sectional ML on single names:** the model is trained on a universe that mechanically
  excludes the realized left tail.

**These kills were run on biased data and should be re-labeled `killed-on-biased-data / status:unknown`,
not `dead`.** Re-open them only on survivorship-free data (§4). This is *not* a reason to distrust the
trend kill — trend lives on ETF EOD, which is essentially survivorship-clean.

### 2.7 25% allocation + 75% cash badly under-deploys your only validated edge. `[C: Med]`

Trend is your one durable edge (deep-history SR ≈ 0.7) and you run it at **25% allocation with a crash
governor**. With tail protection already in place, half-Kelly on an SR-0.7 book sits far north of 25%.
At a 10%-vol sleeve target and 25% allocation, your *book* runs ~2.5% realized vol — you are leaving the
large majority of a validated edge on the table to hold cash at the risk-free rate. At $100k paper this
is harmless for *learning*, but it means your live Sharpe and $-P&L massively understate what the edge
is worth, and it warps your intuition about "is this book good enough." Either raise the allocation
(governor makes this defensible) or be explicit that you're running a deliberately de-risked sandbox.

### 2.8 Where you should *not* second-guess yourself (credit where due). `[C: High]`

So you don't over-correct:
- **CPCV power is *not* your constraint.** I checked: you divide by `√(N_eff = n_folds)`, you *know* the
  C(6,2)=15 paths are correlated, the path-t is report-only, and Ruler-v2 scores **pooled-OOS HAC** on
  the concatenated daily series (n in the hundreds–thousands). DeepSeek was right in the prior panel:
  on deep history the significance t is large; power is binding only on the *4y options window* and the
  *bi-monthly short-interest* series. Do **not** "fix" CPCV by inflating paths — you'd be fabricating
  significance. The honest N_eff handling is a strength.
- **The Bayesian-posterior-with-mandatory-live-paper at CAPITAL is genuinely excellent** — making a
  backtest-alone *structurally unable* to reach live-money is the right design, and rare to see done.
- **The overlay-vs-sleeve distinction is the most sophisticated thing in the system** and your data
  supports it (overlays add marginal value; additive sleeves hit the wall).
- **Pre-registration + independent adversarial review** is funded-pod-grade discipline. Keep it.
- **One caveat that bites the new ideas: your 2bps/one-way cost model is fine for liquid ETFs and an
  order of magnitude too optimistic for options** (spreads are % of *premium*, often 1–5%+) and tight
  for daily-turnover ideas. Any options or daily strategy needs an IV-aware/premium-% cost model or the
  backtest is a mirage (see §3, §5).

---

## 3. Strategy idea menu (the main section)

Ranked by my estimate of **(EV × fits-your-constraints × not-already-spent)**. Each tagged **NEW** (not
meaningfully tested) / **REFRAME** (you killed a *different* version) / **DATA-GATED**. Confidence tags
refer to *whether a real premium exists*, not to whether it clears your gate net of costs (that's the
point of testing). Costs are the universal failure mode; I won't repeat it every line.

### Ranked summary

| # | Idea | Class | Universe / Horizon | New data? | Tag | EV | Priority |
|---|---|---|---|---|---|---|---|
| 1 | **Defined-risk index VRP** (put-spreads / condors, governor-sized) | Vol premium | SPY/QQQ opts / 30–45 DTE | No (have frozen 4y + Alpaca NBBO) | REFRAME | **High** | **Do now** |
| 2 | **Crypto trend + XS-momentum** | Directional / new universe | BTC/ETH + liquid alts / daily–weekly | No (free) | NEW | **High** | **Quick-win** |
| 3 | **Overnight vs intraday decomposition** | Microstructure premium | SPY/sector ETFs+large caps / 1 day | No (free OHLC) | NEW | Med-High | **Quick-win** |
| 4 | **Risk-premia composite** (basket of near-misses) | Book construction | your existing premia / daily | No | NEW | Med-High | Quick-win |
| 5 | **Daily short-VOLUME** (FINRA, not bi-monthly SI) | Flow/positioning | XS equities / days–weeks | Free download | REFRAME | Med | Medium |
| 6 | **CEF discount mean-reversion** | Behavioral RV | equity/bond CEFs / weeks–months | Cheap/free NAV | NEW | Med | Medium |
| 7 | **Earnings vol-crush** (defined-risk) | Vol/event | optionable single names / days | No (FMP dates + opts) | REFRAME | Med | Research |
| 8 | **Futures trend + CARRY** | Cross-asset premium | futures / weekly | **Norgate + IBKR** | DATA-GATED | Med-High | Strategic |
| 9 | **Index reconstitution flow** | Event-driven | adds/deletes / days | Cheap | NEW | Med | Research |
| 10 | **VIX-ETP term-structure roll** | Vol/carry | VXX/SVXY/VIXY / days | Free | REFRAME(careful) | Med | Research |
| 11 | **Merger arb** (defined-risk, small) | Event-driven | announced deals / weeks | Cheap deal data | NEW | Low-Med | Research |
| 12 | **Pre-FOMC drift as return-overlay** | Calendar+mechanism | SPY / 24h windows | Free (enumerate dates) | REFRAME | Low-Med | Research |
| 13 | **Lead-lag / intermarket** (credit→equity, $→EM) | Cross-asset | ETFs / days–weeks | Free | NEW | Low-Med | Research |
| 14 | **Cross-sectional crypto carry** (basis) | Carry | crypto / days | Free data, **exec-gated** | DATA/EXEC-GATED | Med | Research |
| 15 | **Alt-data attention** (Trends/Wiki/EDGAR hits) | Alt-data | XS equities | Free | NEW | Low | Low |

---

### The detailed cases (top ideas)

#### 1. Defined-risk index volatility-risk-premium — *your highest-EV unexplored edge* `[C: High premium exists]`
- **Economic rationale:** implied variance systematically exceeds realized variance (the **variance
  risk premium**) — among the most robust, persistent premia in all of finance, paid as compensation
  for bearing crash risk plus structural hedging demand. It is *durable* precisely because it's
  uncomfortable to hold (negative skew). The prior panel's "steamroller" objection is correct about
  **naked** short vol — and **irrelevant to defined-risk structures.**
- **Why it's a different animal than what you killed:** you tested options *as features/signals* (F4,
  H4, killed). This is options *as structures to harvest a premium* — the thing your own retry list
  flagged ("maybe the edge is in *structures* not options-as-features") and never did.
- **Instruments/horizon:** SPY (and QQQ/IWM) **put-spreads or iron condors / iron flies**, ~30–45 DTE,
  rolled. Alpaca executes US options.
- **Data:** you already have the frozen 4y Polygon options OHLCV to backtest structures, plus Alpaca
  live NBBO. (4y is one-regime-ish — supplement with the *published* VRP literature for the prior and,
  if it validates, a one-time longer SPX option history buy; do **not** let 4y be the only evidence.)
- **Signal sketch:** sell ~10–20Δ put-spreads / short-wing condors when the **VIX term structure is in
  contango** (positive roll/carry) *and* IV rank is elevated; **reduce/flat when backwardated** — i.e.
  *gate it with the crash governor you already built and validated.* Size by an explicit VRP estimate
  (front IV minus a realized-vol forecast). Hard stops / defined wings cap per-trade loss.
- **Role (important):** this is **not** a crisis-uncorrelated diversifier — it is **short crash risk**,
  so it's correlated with equities in the left tail (like carry). Classify it `component_type =
  risk_premium`, use the regime waiver, and judge it on **Track-A return** (it should clear a real
  standalone Sharpe, historically ~0.8–1.5 for sensible programs) more than on Track-B diversification.
- **Capacity/turnover/failure modes:** capacity enormous; monthly roll; failure = gap/tail events
  (defined-risk caps each trade but losses *cluster*), pin/assignment risk, and **execution cost** —
  your 2bps model will *lie* here; model costs as % of premium with an IV-aware spread or you'll
  promote a phantom.
- **Priority:** **Do now.** It's the cleanest answer to "where's the *return*," it reuses your governor,
  and it pairs structurally with trend (§7).

#### 2. Crypto trend + cross-sectional momentum — *point your existing engine at a less-efficient market* `[C: Med-High]`
- **Economic rationale:** crypto is younger, retail-dominated, 24/7, and structurally less efficient;
  **time-series and cross-sectional momentum are stronger and more persistent in crypto than in
  equities** (well-documented). You have an unfair advantage: the TSMOM engine is built and validated.
- **Instruments/horizon:** BTC, ETH (deep-ish history, real liquidity) + a liquid Alpaca-supported alt
  basket; daily–weekly TSMOM + cross-sectional momentum across the basket.
- **Data:** free OHLCV (Alpaca / free APIs). Cheap.
- **Signal sketch:** run your existing TSMOM sleeve on BTC/ETH (long/flat; spot shorting on Alpaca is
  limited, so favor long-flat); add a cross-sectional momentum rank across the alt basket (long top
  tercile). Vol-target hard — crypto vol is 3–5× equities.
- **Capacity/turnover/failure modes:** BTC/ETH capacity high, alts low (fine at $100k); moderate
  turnover; failure = **short history (your power floor will bite — be honest about n_obs)**, regime
  breaks, exchange/counterparty risk, alt survivorship, 80% drawdowns, reflexivity.
- **Priority:** **Quick-win.** Marginal engineering (mostly a data adapter + the power caveat). Even a
  modest-Sharpe crypto-trend sleeve that is *lowly correlated to your ETF trend* is a real book
  addition. This directly answers your meta-question's "(b) change asset class."

#### 3. Overnight vs intraday return decomposition — *free data, no intraday execution, genuinely non-obvious* `[C: Med-High]`
- **Economic rationale:** for many equity indices/ETFs, **essentially the entire equity risk premium
  accrues overnight (close→open); the intraday (open→close) leg is roughly flat or negative.** Driven by
  overnight risk-transfer, inventory/illiquidity effects, and the timing of news/earnings. It is *not*
  a cross-sectional daily-return prediction — so it sits **outside** your "mined out" claim.
- **Why retail-feasible:** tradeable with **MOC/MOO** orders — no tick data, no intraday execution, no
  L2. You already have free daily open/close.
- **Instruments/horizon:** SPY/QQQ/sector ETFs and large caps; 1-day hold (overnight long, intraday
  flat — or the cross-sectional version: long strong-overnight names, short weak).
- **Signal sketch:** systematically capture the overnight component where it's strongest; or use the
  overnight/intraday sign as a regime feature.
- **Capacity/turnover/failure modes:** liquid → high capacity; **turnover is HIGH (daily) → costs are
  THE risk.** 2bps × 2 sides/day ≈ ~10%/yr drag; the premium must clear that net. Auction slippage at
  open/close. The effect has decayed/varies — pre-register and test honestly.
- **Priority:** **Quick-win** (data on hand). A clean pre-registered net-of-cost test resolves it fast.

#### 4. Risk-premia composite (see §2.5) — *gate the basket, not the parts* `[C: Med-High]`
- **Rationale & construction** in §2.5. Concretely: equal-risk-weight {carry returns, credit-timing
  returns, overnight, daily-short-volume}, vol-target the blend, run *that* single stream through
  Ruler-v2 Track-B vs the live book. **Failure mode:** if the components are *secretly* correlated
  (all just timed SPY beta — the F1a lesson), the basket inherits the beta and fails Track-B honestly.
  That's a clean, informative negative. **Priority: quick-win** — it's a ~20-line `Sleeve` declaration
  given your lab.

#### 5. Daily short-VOLUME (not bi-monthly short-interest) — *the power problem disappears* `[C: Med]`
- **Economic rationale:** informed short-side flow. Your G2 short-interest used **bi-monthly** data
  (~190 obs, power-starved → killed for *power*, not for *no signal*). **FINRA publishes daily
  off-exchange short-volume by symbol, back to ~2009** — ~4000+ daily obs. Same idea, *completely
  different power regime.*
- **Instruments/horizon:** XS liquid equities; days–weeks. **Data:** FINRA daily short-volume files
  (free download).
- **Signal sketch:** short-volume-ratio z-score; aggregate (market-timing overlay) and/or cross-sectional.
- **Failure modes:** short-volume ≠ short-interest (includes MM hedging flow → noisy), survivorship,
  meme-era regime shift in the sign. **Priority: medium** — worth a clean test *specifically because the
  thing that killed your last attempt (power) is gone.*

#### 6. CEF discount mean-reversion — *different universe, genuinely orthogonal* `[C: Med]`
- **Economic rationale:** closed-end funds trade at persistent, **mean-reverting discounts/premiums to
  NAV** driven by retail sentiment and distribution-chasing — a slow behavioral signal, orthogonal to
  trend, in a corner of the market the efficient large-cap tape doesn't reach.
- **Instruments/horizon:** liquid equity/bond CEFs (Alpaca tradeable); weeks–months. **Data:** CEF
  price + NAV (CEFConnect free; cheap via APIs).
- **Signal sketch:** z-score the discount vs its own trailing history; long unusually-wide-and-narrowing
  discounts; optionally hedge the asset-class beta with the matching ETF to isolate the discount.
- **Capacity/turnover/failure modes:** low-medium capacity (fine at $100k); low turnover; failure =
  "value trap" (discounts stay wide), embedded leverage adds beta, distribution cuts, illiquidity,
  survivorship. **Priority: medium / research.**

#### 7–15 (briefer)
- **7. Earnings vol-crush (defined-risk)** `[C: Med]` — single-name IV ramps into earnings then crushes.
  Sell **defined-risk** condors/flies into rich earnings IV on names whose *realized* post-earnings move
  history is below implied; exclude binary names (biotech). You have FMP earnings dates + options. *This
  is a vol structure, distinct from the single-name-options-as-signals the panel killed.* Crowded;
  execution-sensitive; survivorship in backtest. **Research-heavy.**
- **8. Futures trend + CARRY** `[C: Med-High premium]` — the real diversifying premium you literally
  cannot test free. Norgate (~$300/yr) gives clean continuous futures **with roll/carry** *and*
  survivorship-free US equities (double payoff, §5). Execution needs IBKR. **Strategic; sequence after
  the free Alpaca-executable ideas.** Your free-yfinance POC (SR +0.14, corr +0.43) is *not* evidence
  against carry — it's evidence you can't test carry on dirty continuous series.
- **9. Index reconstitution flow** `[C: Low-Med, decayed]` — forced index-fund flows around S&P/Russell
  changes (Russell late-June especially). Real but materially arbitraged-down; honest test + survivorship
  critical. **Research.**
- **10. VIX-ETP term-structure roll** `[C: Med, dangerous]` — the VXX/SVXY/VIXY roll-yield premium,
  tradeable on Alpaca (the prior panel's deferred "Index VRP via ETP"). I'd run #1 (defined-risk options)
  *first* — ETP short-vol has the cleanest path to a -80% day. If pursued: tiny, governor-gated, capped.
- **11. Merger arb (defined-risk, small)** `[C: Low-Med]` — the deal-spread premium; retail-doable at
  $100k with cheap deal data, but capacity/labor-heavy and tail-risk on breaks. **Research.**
- **12. Pre-FOMC drift as a return-overlay** `[C: Low-Med]` — the panel killed it *as a diversifier*
  (it's timed SPY beta → fails Track-B). It may still *add return* if expressed as a sizing overlay in
  the pre-FOMC window. The "no clean FOMC date list" objection is solvable — FOMC dates are public and
  enumerable for decades. **Low priority; only as a return-timing overlay, not a sleeve.**
- **13. Lead-lag / intermarket** `[C: Low-Med]` — credit (HYG) leading equity stress, USD leading EM,
  copper/gold ratio as a cyclical gauge → short-horizon ETF timing. Some of this overlaps your governor
  and trend; test for *incremental* value only. **Research.**
- **14. Cross-sectional crypto carry (basis)** `[C: Med premium, exec-gated]` — funding/basis carry is a
  real crypto premium, but Alpaca is **spot-only** (no perps), so the clean expression is execution-gated.
  Park behind a second venue decision.
- **15. Alt-data attention** `[C: Low]` — Google Trends / Wikipedia views / EDGAR access-log hits as
  retail-attention features. Free, noisy, low-ROI. Mention for completeness; **low priority.**

**Span check:** cross-asset/macro (8,13), carry/term-structure (8,10,14), vol/options structures
(1,7,10), relative-value/stat-arb (6), event-driven (7,9,11,12), seasonality/calendar (12), flow/
positioning (5,9), microstructure (3), crypto (2,14), alt-data (15), regime/timing overlays (4,13).
The deliberately non-obvious ones: **3 (overnight), 5 (daily short-volume vs your killed bi-monthly),
6 (CEF discounts), and the reframing of 1 as a *return engine* rather than a diversifier.**

---

## 4. Data gaps & top buys

**Single highest-ROI buy: Norgate (~$270–330/yr).** `[C: High]` It is the only ~$300 subscription that
fixes your **two** biggest data gaps at once:
1. **Survivorship-free US equities with delistings** — removes the silent bias from *every* long/short
   and event study you've run, and lets you honestly **re-open the contaminated kills** (PEAD, F2,
   short-side work) instead of leaving them mislabeled "dead" (§2.6). This payoff accrues even with
   Alpaca-only execution.
2. **Clean continuous futures with roll/carry** — unlocks the **carry premium that is untestable on
   free data** and proper futures-trend breadth (idea #8).

The catch is futures *execution* needs IBKR. So the sequencing is: **buy Norgate now for research**
(survivorship fix helps your equity work immediately), and **add IBKR only once a futures/carry strategy
validates** in the lab.

**Why options-surface history is *lower* priority than you might think:** you already have a frozen 4y
options OHLCV store, which is enough to backtest VRP *structures* (idea #1). A full IV/OI/NBBO history is
expensive and not the binding constraint right now — defer it until VRP proves out, then consider a
one-time longer SPX-option-history purchase.

**Free wins you're not fully using** `[C: High]`:
- **FINRA daily short-volume** (idea #5) — free, daily, deep; kills the power problem that sank G2.
- **CFTC COT positioning** — free; informs equity/commodity-ETF timing even without futures execution.
- **FOMC date list / EDGAR / CEFConnect NAV** — all free and unlock specific ideas above.

**The cost caveat is itself a "data" gap:** your 2bps/one-way model is not realistic for options or
daily strategies. Build an **IV-aware / premium-% option cost model** before trusting any options
backtest — this is cheaper than any data buy and prevents the most likely false-positive in the new
menu.

---

## 5. Modeling recommendations

Your **core modeling mistake is target choice, not model choice.** `[C: High]` You used GBTs to predict
**forward returns** on low-signal daily data and got IC≈0 — which is the *expected* result; ML extracts
structure, it cannot manufacture signal where SNR≈0. The fixes are about *what you predict* and *how you
use the prediction*, not about a fancier learner:

1. **Predict volatility / regime, not return.** `[C: High]` Realized vol is far more predictable
   (strong autocorrelation) than return, and it's **directly tradeable via options** (feeds idea #1).
   Switching the target from "will it go up" to "how much will it move / what regime are we in" is the
   single highest-leverage modeling change.
2. **Meta-labeling (López de Prado).** `[C: High]` Stop asking ML to *generate* trades. Use a primary
   *economic* signal for direction (trend, VRP, overnight) and use ML **only to size/filter** it —
   predict P(this trade works) → fractional bet sizing. Your validated TSMOM is the ideal primary signal
   to meta-label ("when does trend work?"). This is where ML earns its keep in low-signal regimes.
3. **Regime-conditioning.** A simple 2–3 state HMM (or your existing vol/credit governor logic) to
   switch engines on/off. You're already doing a primitive version with overlays — formalize it.
4. **Ensemble across sleeves, not within one weak model.** Diversification across *signals* beats
   squeezing one IC≈0 model.
5. **Portfolio construction: Hierarchical Risk Parity (HRP)** for the multi-engine book. `[C: Med-High]`
   With a handful of sleeves and noisy covariances, HRP is far more robust out-of-sample than
   mean-variance and avoids the estimation-error blowups MVO is prone to at small N.
6. **Position sizing:** fractional-Kelly / vol-target (you do this) *driven by* the meta-label
   probability, with a book-level vol target — and **raise the trend allocation** (§2.7).
7. **The control that underwrites all of the above:** the §2.1 positive-control. Before trusting *any*
   "no edge from ML" claim, prove the feature→label pipeline can reproduce a known anomaly. If it can't,
   the modeling discussion is moot until the pipeline is fixed.

---

## 6. Redesign — a *reframe*, not a teardown

The architecture is good; don't rebuild it. **Reframe the objective.** You've been hunting *additive,
crisis-uncorrelated equity diversifiers* and hitting the **corr<0.30 wall** four times. That wall is
real and you should stop charging it. Instead, operate a small book of **2–3 return engines across risk
classes, under a shared tail governor:**

- **Engine 1 — Trend** (have it): ETFs + crypto (idea #2) + later futures (idea #8). *Positive skew,
  long-crisis* (profits in sustained moves; the canonical crisis-robust engine).
- **Engine 2 — Volatility risk premium** (idea #1): defined-risk index structures, governor-gated.
  *Negative skew, short-crisis* (bleeds in crashes; pays the steady VRP in calm).
- **Engine 3 — One genuine orthogonal diversifier *if* one survives:** the risk-premia composite (#4),
  or carry/credit on a probationary accept (§2.4).
- **Shared:** the VIX-term + credit governor as **portfolio-level** tail control; allocate via **HRP**;
  size each engine by its own vol to a book vol target.

**Why this beats "add a fourth weak uncorrelated equity sleeve":** the diversification you keep failing
to find *between equity sleeves* exists naturally **between a positive-skew engine and a negative-skew
engine.** Trend tends to *profit* in the sustained crises where VRP *bleeds*; VRP harvests the calm grind
where trend *whipsaws*. Their P&Ls are diversifying **through the cycle** even though each is
individually crash-exposed *in its own direction*. That internal hedge — trend's crisis convexity
covering VRP's short-vol tail, with the governor as backstop — is the real edge. It's roughly what
multi-strat shops actually run (trend + carry/vol-selling), and it's the most likely route from your
~0.7 book to a ~1.0+ book. `[C: Med-High]` `[FALSIFIABLE: backtest the trend⊕VRP blend; the claim
predicts the blend's drawdowns are materially shallower than either engine's and the blended Sharpe
exceeds both — if trend and a governor-gated VRP book are positively correlated in the left tail, I'm
wrong.]`

---

## 7. Top 5 things to do next (priority order)

1. **Positive-control the feature→label pipeline.** `[Do first — cheap, decisive]` Run 12‑1 momentum +
   1‑month reversal + low-vol through the *exact* swing-ML pipeline on the *exact* yfinance universe. If
   it can't reproduce known effects, your "IC≈0 / mined out" verdict is partly a deflationary bug, not
   the market (§2.1). This de-risks every downstream conclusion and your own change-log says it's a real
   risk.

2. **Stand up a defined-risk index VRP sleeve as a *second return engine*.** `component_type=risk_premium`,
   backtest on the frozen 4y options **with a premium-% cost model**, validate the prior against the VRP
   literature, **size with the existing governor**, judge on Track-A return, and pair with trend (§3.1,
   §6). This is your highest-EV unexplored edge and it fits Alpaca.

3. **Point the trend engine at crypto.** `[Quick-win]` BTC/ETH + a liquid alt basket, long-flat,
   hard-vol-targeted, honest about the short-history power floor (§3.2). Marginal engineering for a
   genuinely new, lowly-correlated return stream — and it directly tests "change asset class."

4. **Fix the gate's two diversifier-killing flaws and monetize your near-misses.** (a) Replace the binary
   both-halves guard with a powered stability test (§2.2); (b) for declared diversifiers, drop the
   `standalone_vt_SR>0.20` floor and lower Track-B P(ΔSR>0) to ~0.75 with small-size + live-paper
   ratification (§2.3–2.4); (c) build and gate a **risk-premia composite** of carry+credit+overnight+
   daily-short-volume as one sleeve (§2.5). Your pile of "honest near-misses" is probably a book
   contributor you're throwing away.

5. **Buy Norgate (~$300/yr) now; defer IBKR.** `[Strategic]` Use it immediately for survivorship-free
   equities — **re-run PEAD and F2 stat-arb on clean data before declaring them dead** (§2.6) — and to
   finally test the **carry** premium properly (#8). Add IBKR only when a futures strategy validates.
   *And, separately: raise the trend allocation above 25% (§2.7) — the governor makes it defensible.*

---

## Falsifiable-claims recap (for your cross-LLM synthesis)

| Claim | How to kill it |
|---|---|
| IC≈0 may be a deflationary pipeline bug, not the market | Reproduce 12‑1 momentum / reversal / low-vol through the *feature pipeline*; if signs/magnitudes match published, I'm wrong and "mined out" stands |
| Both-halves guard has ~24% FN on a true SR‑0.5 edge | Simulate known SR‑0.5 series through the guard; measure kill rate |
| A composite of weak premia clears Track-B though no single one does | Assemble the equal-risk basket; gate it; if it fails Track-B too, the components were secretly correlated and I'm wrong |
| VRP + trend diversify through the cycle (skew hedge) | Backtest the blend; if trend & governor-gated VRP are positively correlated in the left tail, I'm wrong |
| Norgate is the single highest-ROI buy | If your event/LS kills are robust on survivorship-free data *and* carry still fails properly-tested, the buy was low-ROI |
| Crypto trend is a real, lowly-correlated return stream | Run TSMOM on BTC/ETH OOS; if Sharpe ≤ 0 or corr-to-ETF-trend > 0.6, deprioritize |

**Bottom line:** the system and discipline are excellent and most of your *equity-directional* kills are
honest. But you've mapped one quadrant and called it the world. Run the §2.1 control to make sure IC≈0 is
real; then build the second return engine (VRP) and the second universe (crypto) you haven't touched, fix
the two gate flaws that are quietly killing your diversifiers, and buy the one dataset (Norgate) that
un-biases your event work and unlocks carry. The realistic prize isn't "more trend" — it's **trend ⊕ a
governor-protected VRP book ⊕ one surviving diversifier**, which is a genuinely fund-grade retail outcome.
