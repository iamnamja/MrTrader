# MrTrader — External Quant Research Review

**Reviewer role:** world-class quantitative researcher / quant dev (overfitting-allergic).
**Date:** 2026-06-14.
**Question answered:** Given the data you actually have, the constraints you operate under, and everything you've already killed — what are the best next research steps to find alpha, and where does testable, undiscovered edge plausibly still live?

> **Basis of this review.** Grounded on `01_PROMPT.md` (incl. its "hard truths" digest of the snapshot), `DATA_PROVIDERS.md`, and `ALPHA_V7_SYNTHESIS_AND_PLAN.md`. The primary `02_STATE_SNAPSHOT.md`, the gate spec `RULER_V2_DESIGN.md`, and the full kill ledger (`ML_EXPERIMENT_LOG.md` / `DECISIONS.md`) were **not** available to this reviewer. Where a claim depends on the snapshot's exact numbers or the ledger's per-verdict detail, it is flagged. Two specific facts that would change the conclusions if checked: (1) the exact ticker list of the live 10-ETF TSMOM sleeve, and (2) the history depth of FINRA short-interest available via Polygon.

---

## §1 — Honest assessment

**The read is correct, and it was proven correctly.** On a free-daily, liquid-US-instrument envelope, trend is most of the harvestable systematic edge, and the standalone-alpha kills are honest negatives, not gate artifacts. The dispositive move was re-scoring candidates under *best-case* gate settings (worst-regime waiver, Bayesian posterior replacing the saturated DSR) and watching them still fail. That closes off "just lower the bar." It should not be relitigated.

**The first-principles reason this had to happen** — the lens to carry into every future bet — is the **fundamental law: IR ≈ IC × √breadth.** Free daily data on Russell-1000-liquid names is the single most-arbitraged information set in existence. Your *IC* there is ≈ 0 by construction; you are fishing the most-efficient pond with the same rod as every pod and every index fund.

- Cross-sectional equity ML died because there was no IC to find.
- PEAD died at `t = −0.77` (event level) — note that is a *negative point estimate*, not merely insignificant — because the drift has been arbitraged out of liquid large-caps.

Neither is bad luck. The only available lever is **breadth**, and trend is the canonical breadth play — many bets, across many periods and (when expanded) many assets. **Every survivor from here will be a *premium harvested with enough breadth to validate*, not a *precision alpha*.** The Alpha-v7 re-charter from "find alpha" to "operate a premia book judged at book level" is correct because it stops asking the unanswerable question.

The **path-Sharpe-t-stat catch is genuinely sharp and correct**: `mean/(std/√8)` over 15 paths that recycle 8 folds rewards low cross-fold *dispersion*, not distance from zero — bad size and bad power simultaneously. Retire it. That was the wrong estimand, not a recalibration.

### Where the program is wrong, or under-using an asset

1. **You own FINRA short interest and barely use it.** Aggregate short interest is, per Rapach–Ringgenberg–Tu (2016), arguably the single strongest known predictor of aggregate market returns — and it is a *positioning* signal, maximally orthogonal to a *price* signal like trend. Prior framing treated short data as a shorting-*enabler* (the dollar-neutral fix); the high-value use is an aggregate market-timing diversifier. It appears nowhere in the Alpha-v7 phases. That is a gap.
2. **"Trend expansion" is being conflated with "needs futures," and the bad yfinance futures proxy is a trap.** Continuous-contract construction (roll / back-adjust method) materially moves trend backtests; a dirty proxy misleads the screen in *either* direction. But the *validation* of cross-asset trend is fully doable today on clean long-history ETFs already in hand. Futures are needed for *live execution* (cost, capital efficiency, 24h), not for the test.
3. **The public VIX complex is a distinct, free, ~15y asset** from the frozen single-name options store. The Phase E1 ETP idea is close, but `^VIX` / `^VIX3M` term structure is an under-credited free regime engine that does not die on 6/17.
4. **The PEAD kill is right for large-caps, but the anomaly is not dead — it moved to small/micro-caps,** which survivorship-free history (Norgate) would unlock. That is a *second* payoff for the Norgate purchase already contemplated, currently uncredited.

### Methodology blind spot to name

Inverting the tiers (PAPER = plausibility, no significance floor) is the correct fix for Type-II, but it converts the paper tier into a multiplicity / Type-I engine **unless the concurrent-sleeve cap is strict and pre-registered.** You traded one error for the other. Make the cap real and small, or you will "recover" SR-0.1 noise that happened to have low cross-fold dispersion — the exact failure the path-t already produced.

### The meta-point, brutally

You've built an F1 pit crew (CPCV, HAC-SR, PBO, Bayesian posterior, Track A/B) for a bicycle-grade edge. The rigor is *correct* — it killed your false positives. But **the marginal hour is now worth far more on breadth and data than on more referee-polishing.** Reorder accordingly: front-load trend breadth and the short-interest asset; let the gate redesign ride alongside, not first.

---

## §2 — Ranked next research bets

Ranked by `expected_alpha × testability_with_your_data × not-already-dead`. **Four bets — deliberately not padded to five.**

| # | Bet | Track | Expected SR | Testable on data you have *now*? | Effort | Confidence it clears the gate |
|---|---|---|---|---|---|---|
| 1 | Cross-asset trend breadth (validate on clean ETFs) | A (anchor) | 0.8–1.0 book / ~0.5–0.7 standalone legs | **Yes, fully** | 2–4 days | High |
| 2 | Aggregate short-interest market-timing overlay | B (diversifier) | ~0.3–0.5 standalone; +0.05–0.15 book ΔSR | Yes (history-depth check) | 3–5 days | Medium |
| 3 | VIX term-structure book overlay (+ gated VRP sleeve) | B / governor | overlay: +0.1–0.2 book SR | **Yes, fully** | 2–3 days overlay | Medium-High (overlay) |
| 4 | Cross-asset carry (rates + FX, from FRED + ETFs) | B (diversifier) | ~0.3–0.6 standalone, fair-weather | Partly (skip commodity leg) | ~1 week | Medium-Low |

---

### Bet 1 — Cross-asset trend breadth *(the anchor; do this first)*

- **Thesis.** Trend/TSMOM is the longest-evidenced, most-robust systematic premium across asset classes (Moskowitz–Ooi–Pedersen 2012; AQR "A Century of Evidence"). It pays because someone must provide liquidity to hedgers and forced sellers, and because of slow under-/over-reaction. The edge does not live in any one market — it lives in *breadth across uncorrelated markets*, the exact fundamental-law lever your data permits. The current sleeve (10 US ETFs) is almost certainly equity-concentrated; adding bonds, gold, broad commodities, FX, and intl equity multiplies independent bets **and** adds the crisis convexity that diversifies the equity beta in the book.
- **Data — do you have it?** **Yes, for validation, today.** yfinance has clean 15–19y daily history on `GLD, SLV, TLT, IEF, SHY, UUP/FXE/FXY, DBC/DBA/USO/BNO, VNQ, EEM, EFA`, etc. Crypto is only ~2–3y (`GBTC/IBIT`) — exclude or heavily down-weight. **Do not validate on yfinance continuous futures** — the proxy is too dirty to trust the screen. Buy Norgate (~$30–60/mo) + open a micro-futures account (Tradovate/IBKR — Alpaca has no futures) **only for live execution, after** the ETF validation passes.
- **Test under your harness.** Config change to `tsmom.py`: add the cross-asset ETF universe, run the existing vol-scaled TSMOM with the book-vol overlay, judge on the full 19y via Track B — appraisal ratio vs equity beta, ΔSR vs the current sleeve, crisis-window convexity in 2008/2020/2022. **This is the one place the power constraint dissolves**: ~12–15 assets × 19y × weekly ≈ thousands of effective bets, so ~8-fold CPCV is more than adequate. Re-run rebalance offsets 1–4 to rule out the `np.arange(n) % 5 == 0` timing-luck artifact.
- **Expected Sharpe.** Standalone cross-asset legs ~0.5–0.7 each; a clean own-built diversified trend program historically ~0.8–1.1 pre-cost (~0.7–0.9 net of realistic ETF slippage). Book anchor, Track-A-grade — clears significance on 19y.
- **Pre-mortem.** (a) If the 10 ETFs are *already* cross-asset, marginal benefit collapses and this becomes "add futures markets," which needs Norgate to validate well — so **check the ticker list first.** (b) Crisis convexity is sample-dependent: strong in 2008/2020/2022, whippy in range markets (2011–12, 2015, 2023). (c) Turnover/slippage on thinner ETFs can shave 0.1–0.2 off gross. (d) Several commodity/FX ETFs launched ~2006–09, shortening the true common window.
- **Effort.** 2–4 days to validate (config + Track-B run). Live is weeks more (Norgate + futures account + re-validate), gated on Phase D fidelity.

---

### Bet 2 — Aggregate short-interest market-timing overlay *(the under-used asset you already own)*

- **Thesis.** Short sellers are reluctant, informed traders; their aggregate positioning leaks slow-moving information about future market returns. Rapach–Ringgenberg–Tu (2016) find the aggregate short interest index is among the strongest single OOS predictors of aggregate equity returns. Crucially, it is a *positioning/sentiment* signal — structurally uncorrelated with price-trend — the most genuinely trend-diversifying idea in the pack.
- **Data — do you have it?** **Yes**, with a caveat: FINRA short interest + short volume via Polygon REST survives the 6/17 downgrade (REST still 200). **Check the history depth Polygon serves** — if short, FINRA's bi-monthly short-interest file is public back to ~2005 and freely downloadable; backfill from source. So: have it, pending a one-hour depth check.
- **Test under your harness.** Build the aggregate SI index (detrended, equal- and value-weighted across the R1K/market), sign-test/regress next-period market excess return, then express as a long/flat or gross-scaling overlay on SPY or the whole book. Judge via **Track B** (book ΔSR, drawdown reduction), **not** Track A. Power caveat, stated plainly: ~24 obs/yr, heavily autocorrelated → low effective N; it will **not** clear a standalone t≥2 — which is exactly why it belongs in the appraisal/ΔSR tier as a diversifier. **Point-in-time hygiene is the whole ballgame** — use FINRA settlement+dissemination dates, never the as-of date, or you bake in look-ahead.
- **Expected Sharpe.** Standalone ~0.3–0.5 (most value in drawdown avoidance); as a book overlay, +0.05–0.15 ΔSR with meaningful tail reduction. Track-B diversifier.
- **Pre-mortem.** (a) Public since 2016 → crowding/decay; live OOS may sit below paper. (b) Reporting lag makes real-time implementation fiddly — PIT errors inflate it spuriously. (c) Low breadth → underpowered; lives or dies on the Track-B diversification bar, not its own t-stat.
- **Effort.** 3–5 days.

---

### Bet 3 — VIX term-structure book overlay (+ a gated VRP sleeve) *(cheapest book improvement; sharpens Phase E)*

- **Thesis.** The slope of the VIX term structure (`^VIX` vs `^VIX3M`, or VIX vs trailing realized) is a durable regime signal — backwardation = stress/de-risk, contango = calm/risk-on. As a *de-risking overlay* it cuts the book's left tail; as a *timing gate* on a short-vol (VRP) sleeve it harvests the variance risk premium only when it is being paid and stands aside during a crash, when naive short-vol detonates. The VRP exists because investors chronically overpay for crash insurance; the term structure tells you when that overpayment is rich vs. when the steamroller is already moving.
- **Data — do you have it?** **Yes, free**: `^VIX` and `^VIX3M` on yfinance (~15y), VIX-futures proxies via `VXX/VIXY/SVXY/ZIVB`. Does **not** depend on the frozen single-name options store — key, since that store dies 6/17.
- **Test under your harness.** Two uses, ranked. **(a) Overlay (do this):** scale book gross down when the term structure inverts / VIX exceeds realized by a threshold; judge book ΔSR and drawdown reduction (book-level, not alpha). **(b) Sleeve (small, last):** a defined-risk short-vol ETP (the ZIVB-style E1), gated ON contango + below a VIX threshold, ≤10% risk budget, ≤2% NAV tail, no parameter search; judge via Track B. ~15y of VIX data gives decent power for the regime signal.
- **Expected Sharpe.** Overlay: not a standalone SR, but historically +0.1–0.2 to a risk-asset book's Sharpe, mostly via 2008/2020 drawdown avoidance. Gated sleeve: standalone can look ~0.8–1.2 but with a brutal left tail *even gated* — treat with suspicion.
- **Pre-mortem.** (a) Short-vol is nickels in front of a steamroller; a 2018-Volmageddon / March-2020 gap hands back years in days, and the gate reduces but does not remove this. (b) The overlay can de-risk at the bottom and hurt in V-recoveries. (c) VRP has compressed as the trade crowded. **The overlay half (a) is high-EV/low-risk; the short-vol sleeve (b) is genuinely dangerous — small and last, or skip.**
- **Effort.** Overlay ~2–3 days; gated sleeve ~1 week incl. tail modeling.

---

### Bet 4 — Cross-asset carry from FRED + ETFs *(the classic trend complement; honest lower rank)*

- **Thesis.** Carry and trend are the two pillars of cross-asset systematic investing and are famously complementary — carry harvests in calm, trend protects in trends/crises (Koijen–Moskowitz–Pedersen–Vrugt, "Carry"). Carry pays because it bears the risk of sharp reversals (carry crashes), which someone must.
- **Data — do you have it?** **Partly.** Rates/bond carry = Treasury curve slope, **fully available from FRED** (decades — `DGS2/DGS10/DGS30`, full curve). FX carry = rate differentials (FRED has foreign short rates) + FX spot (yfinance majors / FX ETFs). Equity carry ≈ dividend yield (have it). **Commodity carry needs clean futures term structure — you do not have it; skip that leg.**
- **Test under your harness.** Rates: time-series carry on duration (lengthen when the curve is steep / positive carry, vol-scaled). FX: cross-sectional G10 carry — but clean shorting needs FX ETFs/pairs, so a long-flat or paired version. Judge on 19y+ (FRED gives far longer) via Track B; the *point* is low/negative correlation to trend. Power is fine on rates (decades); FX cross-section is thin (G10 → low breadth).
- **Expected Sharpe.** Rates carry ~0.3–0.5; FX carry ~0.4–0.6 historically with a fat left tail. As diversifiers, modest ΔSR with low trend-correlation.
- **Pre-mortem — the serious one:** carry crashes are violently correlated with the equity beta you are trying to diversify (FX carry blew up in 2008 *with* everything). The diversification is fair-weather and fails exactly when you need it. Rates carry has been a one-way duration bet for 40 years and 2022 shows the regime can break; post-ZIRP rate compression killed FX carry for a decade. **Carry is a real but fair-weather diversifier — weight it small, and never let the backtest's average correlation hide its crisis correlation.**
- **Effort.** ~1 week (rates ~3 days off FRED; FX messier ~4–5 days; commodity leg skipped).

---

## §3 — The book view: trend-only → 3–5 sleeves

Credible architecture, in build order:

1. **Anchor — cross-asset trend (Bet 1), maximized.** You cannot build a multi-sleeve book before the one sleeve that works is at full breadth. Target ~0.8 standalone. This also *fixes the denominator* — you cannot measure what diversifies the trend book until the trend book is final.
2. **Orthogonal diversifier — aggregate short interest (Bet 2).** A positioning signal is maximally orthogonal to a price signal; best shot at a sleeve whose correlation to trend stays low *and* does not spike in crises (unlike carry/short-vol).
3. **Complementary premium — rates/FX carry (Bet 4), sized small.** Harvests in the calm regimes where trend whipsaws — priced for its crisis correlation, not its average.
4. **Governor (not a sleeve) — VIX term-structure overlay (Bet 3a)** on top of the whole book, cutting gross in stress.
5. **Optional, last, dangerous — gated short-vol ETP (Bet 3b)**, ≤10% budget, only after the overlay and gate are proven.

Tie together with the book-vol-targeting overlay already built.

**Brutal honesty on the target:** book SR 0.8–1.0 is achievable *if* the trend anchor hits ~0.8 standalone and the diversifiers are genuinely uncorrelated — the diversification math does the rest. But **1.0 is optimistic net-of-cost for a solo retail-data book; 0.7–0.9 is the defensible target, and that is already excellent.** The trap: every diversifier's correlation to trend (and to each other) *rises in a crisis* — carry and short-vol both blow up with equities. So the *realized* book SR in a stress event will be materially worse than the historical average implies. **Size for the crisis correlation, not the average one.**

---

## §4 — What NOT to pursue

- **XS equity ML ranking** — dead, correctly. No IC on free daily data (the fundamental-law reason). Do not revive.
- **Single-name options (signal *or* execution)** — dead. Cost/NBBO wall is real; store freezes 6/17. The H4a–e negatives are best read as *one 2022-regime statement*, not five falsifications — but that does not resurrect them on 4y.
- **Index VRP via condors / dispersion** — cost wall + frozen data. The *only* VRP expression worth touching is the gated ETP overlay (Bet 3b), small and last.
- **Large-cap PEAD / earnings-event equity sleeves** — correctly dead in the R1K (`t = −0.77` is consistent with "drift arbitraged out of liquid names"). Do not re-test on yfinance. *Small-cap PEAD is a different bet that only becomes testable if you buy Norgate for survivorship-free small-caps* — contingent future option, not now.
- **I/B/E/S, microstructure, alt-data** — not EV-positive at $100k paper. Hold the line.
- **Binary threshold sweeps** (the implied-move-filter lesson) — threshold-fragile by construction.
- **Commodity carry** — un-testable without clean futures curves. Skip until futures data is wired.
- **More gate machinery beyond the path-t fix.** Do retire `path_sharpe_tstat` and add PBO — genuine fixes. But the broader Ruler v2 redesign is *lower marginal EV than data/breadth*. Don't let referee-polishing crowd out recruiting players. **This is the one real disagreement with the current sequencing: front-load Phase C (trend breadth) and the short-interest asset; let Phase B ride alongside, not first.**

---

## §5 — The one bet

**Cross-asset trend breadth, validated on clean long-history ETFs (Bet 1).**

Because it is the only bet that is simultaneously: (a) maximally testable on data in hand *today* with **no power problem** — 12–15 assets × 19y is the single place the binding constraint dissolves; (b) the actual *anchor* of the book you want — you cannot assemble a 3–5 sleeve book without first maximizing the one sleeve that works; (c) the highest expected contribution to book SR; and (d) the prerequisite that makes every diversifier's ΔSR *measurable*. It also resolves the parked P5 question — does broadening trend help? — but does it *right*: clean ETFs, Track B, cross-asset (not just more equity ETFs), crisis-window primary.

**The conditional:** if the ticker list shows the 10 ETFs are *already* cross-asset, the one bet pivots to **Bet 2 (aggregate short interest)** — because then trend breadth needs Norgate to extend meaningfully, and the highest-EV thing achievable with data-in-hand becomes the most orthogonal diversifier already owned.

---

## Bottom line

The kills are honest and the re-charter is right: on free daily US data the IC is structurally ≈ 0, so breadth is the only lever and trend is the canonical breadth play. The realistic prize is a **0.7–0.9 book SR** from a maximized cross-asset trend anchor plus 2–3 genuinely orthogonal diversifiers — not a home run, and that is fine. The two under-used assets are both already inside the envelope: **cross-asset trend breadth (testable today, no power problem) and aggregate short interest (the most trend-orthogonal signal owned).** Spend the next month there, not on more gate engineering. Buy Norgate only if the ETF trend screen proves out — and credit it for the small-cap-PEAD door it also opens.
