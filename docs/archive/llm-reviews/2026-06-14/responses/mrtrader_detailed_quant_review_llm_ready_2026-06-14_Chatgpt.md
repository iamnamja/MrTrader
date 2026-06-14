# MrTrader External Quant Review — Alpha Search After the Kill Sweep

## 1. Brutally honest assessment

Your read is mostly correct: on the current data envelope, **the only genuinely validated edge is the 10-ETF TSMOM sleeve**, and the rest of the current research estate is either dead, parked for good reasons, or untestable with the remaining data. That is not a failure of the platform; it is the expected outcome of finally applying a real ruler to a free-data, solo-operator, US-centric alpha search.

The strongest fact in the pack is not “trend Sharpe +0.71.” It is this combination:

- the trend sleeve has 19 years of daily ETF data behind it;
- most other candidates have 2–4 years of usable data;
- the old path-Sharpe statistic was not measuring what you wanted;
- Ruler v2 was rebuilt to be less Type-II, and the re-run still failed the available candidates on real SR/significance, with the diversifier regime waiver available.

That means the next alpha is unlikely to be hiding in “one more model variant.” The remaining viable search space is narrow:

1. **Deep-history ETF premia/overlays** that can use 2005–2026 daily data.
2. **Event-panel conditional structures** that use event-level inference, not 8-fold path t-stat inference.
3. **New data / new tradable markets** if you want more than one or two incremental sleeves.

The uncomfortable truth: a 3–5 sleeve book at SR 0.8–1.0 is possible only if you stop defining every sleeve as standalone alpha. But even as a premia book, **3–5 sleeves from your current free data is optimistic**. Trend is real. One low-turnover ETF relative-value/carry-like sleeve may be real. One crash/risk-off overlay may improve utility. A conditional event sleeve is a long shot. Beyond that, you probably need better data or broader markets.

Where you may be wrong:

- **You may have killed “carry” too crudely.** A trailing ETF distribution-yield sort is not proper carry. Bond carry/curve rolldown and credit carry can be built from FRED + ETF returns. That does not mean it will work, but the previous carry kill should not be read as a full carry-family kill.
- **You may be underusing ETF relative-value structure.** High-turnover single-name reversal is dead. That does not kill low-turnover ETF pair/spread mean reversion across economically linked exposures.
- **You may be over-reading the options signal deaths as universal.** The 2022–2026 options sample is a regime statement, not a universal proof. But because the options store is frozen, that nuance is not actionable now.
- **You may still be too focused on finding “alpha” when the best next dollar of effort may be book construction around the one real edge.** This is not glamorous, but it is more likely to improve deployed outcomes than another cross-sectional model.

## 2. Ranked next research bets

### Bet 1 — Low-turnover ETF relative-value / spread mean-reversion sleeve

**Rank:** #1  
**Type:** Track-B diversifier / possible standalone weak alpha  
**Expected standalone Sharpe:** 0.25–0.45 if real; assume 0.0 until proven  
**Expected book impact:** +0.05 to +0.15 book Sharpe if uncorrelated to TSMOM  
**Effort:** 5–8 dev days

#### Thesis

Your live TSMOM sleeve profits from persistent directional trends. The natural complement is not daily single-name reversal; it is **slow relative-value mean reversion between economically related liquid ETF exposures**. Examples:

- QQQ vs SPY: growth/mega-cap tech relative overextension.
- IWM vs SPY: small/large relative cycle.
- EFA vs EEM: developed/emerging relative cycle.
- HYG vs IEF or LQD vs IEF: credit risk vs duration.
- TLT vs IEF: duration-curve relative move.
- GLD vs UUP: dollar/real-asset relative shock.

Who pays? Flows, benchmark reallocations, ETF crowding, and slow-moving institutional rebalance cycles. This is not a “secret alpha”; it is a low-turnover liquidity-provision/risk-transfer sleeve. But it is exactly the kind of thing that can survive in a small retail-sized book if costs are low and turnover is controlled.

#### Data required

You mostly have it. The deep substrate is free daily ETF history from yfinance, which is the same substrate that validated TSMOM. You may need to add a handful of ETF tickers already contemplated in P5, such as HYG, LQD, SHY, SLV, VGK, EWJ, plus perhaps sector ETFs if you want later expansion. No options, no fundamentals, no intraday data.

#### How to test

Pre-register one fixed design. Do not sweep thresholds.

- Universe: fixed ETF pairs grouped by economic rationale.
- Signal: rolling log-spread z-score using a 252-day lookback.
- Entry: weekly rebalance; long undervalued leg / short overvalued leg when `|z| > 1.5`; partial or full exit when `|z| < 0.5`.
- Sizing: pair vol target, equal risk across pairs, gross cap.
- Costs: 2 bps base, 4 bps stress; include ETF short borrow proxy or forbid hard-to-borrow ETFs.
- Window: 2007–2026 for pairs requiring HYG/LQD; 2005–2026 where available.
- Gate: first Track-A Ruler v2 for standalone sanity; then Track-B v2 versus the live trend sleeve, using appraisal IR, block-bootstrap `P(ΔSR>0)`, correlation, and tail-overlap.
- PBO: only if you test multiple z-score thresholds or many pair sets; ideally avoid this by freezing pairs and thresholds.

#### Why it might fail

ETF spreads can trend for years. “Cheap” can get cheaper. The pair set may just be hidden equity beta, credit beta, or duration beta. QQQ/SPY and IWM/SPY may be too crowded and too structurally regime-dependent. The sleeve can also fail by being positive only in 2008 and flat afterward. Kill it if the residual alpha is not there or if Track-B improvement disappears outside crisis windows.

#### My read

This is the cleanest next bet because it is **not already dead**, uses the only powered dataset you have, trades slowly, and is naturally diversifying to directional trend. If this fails, that is highly informative: it means the deep ETF substrate is basically trend-only for you.

---

### Bet 2 — Fast-crash / risk-off overlay for trend’s known blind spot

**Rank:** #2  
**Type:** Track-B risk overlay, not standalone alpha  
**Expected standalone Sharpe:** -0.10 to +0.20  
**Expected book impact:** may improve max drawdown/tail overlap more than Sharpe  
**Effort:** 4–7 dev days

#### Thesis

Your TSMOM sleeve handles slow bears better than fast V-shaped shocks. That is exactly what trend systems do: they need time to rotate. The sleeve was negative in COVID and Q4 2018-type whipsaws, even though it helped in slower bear markets. So the relevant question is not “can I predict crashes?” It is “can I cheaply buy reaction speed for the first 5–20 trading days of an equity shock?”

This is an insurance/risk-transfer sleeve. It should be judged as a book overlay, not as standalone alpha.

#### Data required

You have most of it:

- ETF daily history for SPY, TLT, IEF, GLD, UUP, SHY.
- VIX can be sourced via yfinance/FRED, but your cached macro history is thinner, so use price-based triggers first and VIX only as a secondary confirmation.
- VIXY/VIXM can be tested from inception, but I would avoid them initially because VIX ETP data length, roll decay, and product changes create a measurement mess.

#### How to test

Freeze a simple, non-optimized overlay:

- Trigger: SPY 3-day return below a fixed percentile, SPY below 200-day moving average, and/or realized-vol shock above a fixed percentile. Use only one pre-registered trigger family.
- Action: temporarily allocate a small sleeve budget to SHY/IEF/TLT/GLD/UUP or reduce equity-risk notional in the trend book.
- Hold: fixed 5, 10, or 20 trading days. Pick one. Do not sweep.
- Test: 2005–2026 daily. Evaluate as Track-B against current TSMOM, with primary metrics = maxDD reduction, left-tail overlap, crisis-window returns, and bootstrap `P(ΔSR>0)`.
- Important: if Ruler v2 rejects this because standalone SR is low, treat it as a **risk overlay**, not as a promoted alpha sleeve.

#### Why it might fail

Most crash signals fire after the move has already happened. Defensive ETFs may fail in inflation/rate shocks like 2022. VIX ETPs are negative-carry traps. The overlay may reduce drawdown at the cost of too much long-run return. If it helps only in COVID but hurts across 2005–2026, kill it.

#### My read

This is a pragmatic hedge for the actual weakness of your live sleeve. It is not “finding alpha,” but it may add more real book value than a weak third alpha sleeve.

---

### Bet 3 — Proper bond/credit carry and curve sleeve using FRED + ETFs

**Rank:** #3  
**Type:** Track-B risk premium  
**Expected standalone Sharpe:** 0.20–0.40 if real  
**Expected book impact:** +0.03 to +0.10, but likely regime-dependent  
**Effort:** 1–2 weeks

#### Thesis

The previous ETF carry screen was too narrow: trailing distribution yield is not proper carry. For bonds and credit, carry is about yield, curve rolldown, spread compensation, and expected excess return after financing/rate risk. FRED can give you yield curves and credit spreads for long histories. You can express the signals with ETFs.

Possible sub-sleeves:

- Treasury curve carry: SHY/IEF/TLT allocation based on yield curve level/slope and realized volatility.
- Credit carry: HYG/LQD vs IEF/SHY based on option-adjusted spreads and spread momentum.
- Defensive cash carry: risk-free/cash proxy when carry is bad and trend is weak.

Who pays? Investors who need liquidity or insurance during tightening/default-risk regimes; carry buyers are paid until inflation/default shocks arrive.

#### Data required

You need to extend the macro/FRED layer beyond the current cached `macro_history` fields. This is still free and should be lightweight. Useful series include Treasury yields, term spreads, credit OAS, Fed funds, CPI/inflation proxies, and unemployment/claims trend. ETF prices come from yfinance.

#### How to test

Do not run a large macro feature search.

- Build only two pre-registered sleeves:
  1. Treasury carry/curve sleeve.
  2. Credit spread carry sleeve.
- Monthly rebalance, slow signals, low turnover.
- Use 2007–2026 where HYG/LQD exist; longer where Treasury ETFs exist.
- Compare against naive bond trend and against simply adding IEF/TLT to TSMOM.
- Track-B against trend, with multi-factor residual alpha and correlation to duration/equity carefully reported.

#### Why it might fail

It may just be duration beta. It may look good only in the 2009–2020 falling-rate regime and get destroyed in 2022. Credit carry can be short-crash-risk in disguise, which is the opposite of what you need next to trend. Kill it if 2022 or rising-rate subperiods dominate the left tail.

#### My read

This is worth testing because the prior carry kill did not fully adjudicate proper carry. But I would not expect miracles. If it passes, size it modestly.

---

### Bet 4 — Event-panel residual scorecard, but only as a continuous, pre-registered panel test

**Rank:** #4  
**Type:** Track-A candidate, likely long shot  
**Expected standalone Sharpe:** 0.00–0.30; maybe higher if a true interaction exists  
**Expected book impact:** uncertain; likely low correlation to trend if real  
**Effort:** 1–2 weeks

#### Thesis

PEAD’s unconditional level is dead. But event data still has one thing your fold-based paths do not: **many independent events**. The only event idea I would still allow is not “PEAD v3” and not a threshold filter. It is a small, continuous, pre-registered residual-return scorecard built on the event panel.

Candidate mechanism:

- Underreaction exists only when surprise/gap/revision/quality/crowding align.
- Overreaction/fade exists when the announcement move is extreme relative to fundamentals and pre-event implied move.
- Crowded/high-short-interest names have different post-event behavior than normal names.

#### Data required

You have a 2019–2026 event panel with thousands of events, plus options features from 2022–2026 for a subset, plus short-interest history. You do **not** have forward estimate-revision history, which is the feature I would most want. That is a real hole.

#### How to test

One design only:

- Outcome: 5/10/20-day SPY-hedged and sector-hedged event returns.
- Model: linear or monotonic scorecard, not XGBoost.
- Features: pre-registered, maximum 6–8 variables. Include interaction terms only if economically specified in advance.
- Split: train 2019–2023, validate 2024–2026; or train 2019–2024, validate 2025–2026.
- Inference: two-way clustered by announcement date and firm. Do not use path-Sharpe.
- Trading rule: top/bottom decile only after the scorecard is frozen, not during feature discovery.

#### Why it might fail

It probably fails because H1/H2 already showed PEAD and implied-move reaction ratio do not carry the effect, H3 is blocked on revisions, and options coverage is only four years. The remaining effect may be too small, too conditional, or pure data-mining. Kill it if the validation deciles are non-monotone or the two-way clustered t-stat is weak.

#### My read

This is the only event research I would permit. It is not my top bet because the most economically important event feature — PIT estimate revision history — is missing.

---

## 3. Path from trend-only to a realistic 3–5 sleeve book

The credible path is not “find four alphas.” It is:

1. **Anchor:** Keep TSMOM as the core sleeve. Audit it with FF5 / multi-factor residuals and rebalance-offset sensitivity, then size from evidence rather than hope.
2. **Add one deep-history diversifier:** ETF relative-value / spread mean reversion if it passes Track-B. This is the best candidate for a true second sleeve.
3. **Add one risk overlay:** fast-crash/risk-off overlay if it improves left tail without too much return drag. Treat it as risk management, not alpha.
4. **Optionally add one carry sleeve:** only if proper bond/credit carry survives 2022 and is not just duration beta.
5. **Keep event panel as optional alpha R&D:** only promote if event-level inference is clean.

My realistic expectation:

- With current data, a robust **2-sleeve book** is plausible: TSMOM + one ETF relative-value/carry/overlay sleeve.
- A **3-sleeve book** is possible but not likely.
- A **5-sleeve book** is not realistic without new data or new markets.

If the target is book SR 0.8–1.0, the most honest route is probably not more equity alpha. It is better trend sizing, one diversifier, one overlay, and strict live/backtest fidelity.

## 4. What NOT to pursue

Do not spend more time on:

1. **Cross-sectional equity ML rankers.** You killed long-only, beta-neutral, and intraday variants. The signal is beta/noise/cost drag.
2. **Intraday 5-minute ML.** Two years of data is too thin and the gross edge is below costs.
3. **PEAD level / PEAD threshold filters.** Event-level PEAD is negative after hedging; implied-move thresholding was fragile.
4. **Options-as-equity-signal decile sorts on the frozen 2022–2026 window.** H4a–H4e were not borderline; several were significantly negative.
5. **Index VRP using the current frozen options store.** It may be a real long-run premium, but your 2022–2026 implementation is negative under realistic stress and no longer has a forward data stream.
6. **Single-name options execution.** The spread wall is real.
7. **Dispersion trading.** It combines the single-name options cost wall with a more complex multi-leg structure. Not a solo/free-data priority.
8. **Short-interest standalone factors.** The tested factor is an anti-edge in the current era.
9. **Daily single-name reversal/stat-arb.** The signal may exist gross, but turnover kills it.
10. **Large macro timing models with many thresholds.** With a few crisis observations, these will overfit quickly.
11. **More broadening of the current TSMOM sleeve by random ETF additions.** P5 already failed; complexity must earn it.

## 5. The one bet

If I had to pick one research bet for the next 2–4 weeks, I would build the **low-turnover ETF relative-value / spread mean-reversion sleeve**.

Why this one:

- It uses the only powered dataset you have: 19 years of daily ETF history.
- It is not the same as the killed daily single-name reversal.
- It is not an options/data-fee dependency.
- It is naturally diversifying to trend.
- It can be specified cleanly without a huge sweep.
- It is buildable by one developer in about a week.
- If it fails, the negative result is valuable: it tells you the deep ETF substrate is basically trend-only.

The exact first implementation:

1. Freeze 6–8 economically justified ETF pairs.
2. Use rolling 252-day log-spread z-scores.
3. Weekly rebalance.
4. Enter at `|z| > 1.5`, exit at `|z| < 0.5`.
5. Vol-target each pair; equal-risk across active pairs.
6. Use 2 bps base and 4 bps stress.
7. Run 2007–2026 CPCV and Track-B versus the live TSMOM sleeve.
8. Kill unless it adds book value outside one crisis window and keeps correlation to trend low.

## Bottom line

You are not one clever feature away from a multi-alpha machine. Your platform is now strong enough to show that most obvious retail-accessible US equity alpha is gone, too small, too costly, or untestable. The next real work is to mine the **deep ETF history** for one slow diversifier, build a **risk-off overlay** around trend’s fast-crash weakness, and only then revisit events with clean panel inference. If those fail, stop pretending the answer is inside the current free-data envelope: keep trend, improve execution/fidelity/sizing, and either accept a simpler book or acquire genuinely new data/markets.
