# MrTrader — Brutally Honest Quant / PM Review

**Date:** 2026-06-19  
**Reviewer stance:** top-down hedge-fund quant PM / developer review; blunt by design.  
**Source files reviewed:** `01_PROGRAM_OVERVIEW.md`, `02_KILL_KEEP_LEDGER.md`, `03_CURRENT_EDGES_AND_NUMBERS.md`, `04_DATA_INVENTORY.md`, `05_ARCHITECTURE.md`, `06_VALIDATION_HARNESS.md`, `07_RECENT_FINDINGS_AND_CAVEATS.md`.

---

## 0. Executive verdict

You are not obviously fooling yourself in the usual solo-quant way. Your process is much better than the median independent systematic trader: you pre-register, you kill ideas, you separate paper from capital, you have discovered and fixed validator failures, and you are not pretending every backtest is an edge.

But the hard verdict is this:

**MrTrader is not yet an alpha platform. It is a small, rules-based alternative-risk-premia platform with one live premium, one promising new premium, a cash optimization sleeve, and a research harness that is finally becoming useful.**

That is not an insult. It is the correct state for a solo operator with mostly daily, low-cost data. But you should stop using language that implies you are harvesting unique alpha. Trend is not alpha. Futures carry is probably not alpha. Cash in T-bills is not alpha. These are tradable premia. If you size them sensibly and survive, that can still be a good book.

The most important conclusions:

1. **Trend at Sharpe ~0.7 over 2007-2026 is plausible and consistent with priors, but it is not a proprietary discovery.** It is a known premium. The implementation may be robust, but the edge is public and capacity-rich. Your job is not to “prove alpha”; your job is to implement it without overfitting overlays, over-sizing it, or letting execution/state bugs kill you.

2. **Futures carry at honest Sharpe ~0.55-0.60 is plausible and worth prioritizing.** The result matches my priors for a broad, liquid, cross-sectional futures term-structure strategy. The fact that it improves the trend book is exactly what I would expect. The result does not smell obviously fake after the bug fixes, but it is not deployable until roll cost, contract selection, and live futures execution are validated.

3. **Your conclusion that “free daily US-equity directional alpha is mined out” is directionally right but overstated.** The more precise conclusion is: *your yfinance/current-listing/free-daily-equity feature-mining program has not found a deployable directional equity edge after costs, and further mining in that box is low EV.* That is not the same as proving daily US equity alpha is gone. You cannot make the stronger claim without survivorship-free equities, delisted returns, point-in-time fundamentals/constituents, and a cleaner event-study stack.

4. **I trust many of your KILLs, but not all of them.** I trust the broad lesson that high-turnover single-name/options/equity signals are cost- and data-fragile. I do *not* fully trust the PEAD, cross-sectional equity, FINRA cross-sectional, or options-as-signal kills because the data limitations are material. They may still be dead, but the current evidence is not clean enough to close the case forever.

5. **The architecture is good for paper trading and low-frequency ETF execution, but not yet robust enough for live multi-asset futures capital.** A single-process FastAPI/APScheduler orchestrator is acceptable while the account is small and execution is slow. Before adding futures, you need stronger process isolation, idempotent order handling, broker-as-source-of-truth reconciliation, margin/collateral controls, contract-roll state, and immutable signal snapshots.

6. **The highest-EV “next spend” is not another shiny dataset. It is making futures carry executable.** Fix the carry cost model, run IBKR paper with real contract rolls, and build a production-grade futures execution/reconciliation path. After that, buy survivorship-free US equities because it is cheap and it will either reopen or finally bury your contaminated equity research.

7. **The single biggest strategic risk is that you keep building an app around a two-premium book and call it diversified.** Trend + carry is a good start, but it is still mostly a slow macro/risk-premia book. It can have multi-year droughts. You need at least one of: cleaner equity-event alpha, an investable volatility-risk-premium sleeve, a more isolated futures calendar-spread/carry implementation, or a robust risk overlay that materially reduces left-tail without curve-fitting.

---

## 1. Are you finding real alpha or fooling yourself?

### My answer

You are finding **real tradable premia**, not much evidence of proprietary alpha.

That distinction matters because it changes how you should validate, size, and talk about the book.

A proprietary alpha should be judged on residual return after known factors, with high scrutiny around leakage, crowding, capacity, and decay. A risk premium should be judged on economic logic, implementability, drawdown tolerance, financing/margin, crash behavior, and whether it improves the total portfolio.

Most of your surviving edges fall into the second bucket:

| Sleeve | What it really is | My confidence | Comment |
|---|---:|---:|---|
| ETF TSMOM trend | Public trend-following / crisis-premium / convex defensive asset allocation | High | Plausible and useful, not proprietary alpha. |
| VIX crash governor | Risk overlay / conditional de-risking | Medium | Useful if fail-safe and simple; easy to overfit. |
| Cash/T-bill sleeve | Collateral/cash management | Very high | Correct operational hygiene, not a strategy edge. |
| Futures carry | Public futures term-structure carry / alternative risk premium | Medium-high after fixes | Most promising next sleeve. Needs execution validation. |
| Crypto trend | Public trend premium in noisy asset class | Low-medium | Track-B fail is the right reason not to allocate yet. |
| FINRA short volume | Weak information/risk-state signal | Low-medium | More likely overlay/composite input than standalone sleeve. |
| Index VRP | Short-vol risk premium | Medium conceptually, low with current data | Parked, not killed. Needs executable quotes and strict tail design. |

The good news: you are not forcing dead alphas into production. That alone puts you ahead of many professional teams.

The bad news: the current system is still mostly a wrapper around known premia. The app may be sophisticated, but the edge content is simple. That is fine if you accept it and keep costs/complexity low. It is dangerous if the app complexity creates false confidence.

### The uncomfortable truth

A two-year solo research program ending with trend, cash, and futures carry is not surprising. It is exactly what I would expect when the inputs are mostly free daily data and the operator has become honest enough to kill spurious results.

That does not mean the work failed. It means the work learned the correct lesson: **most easy equity “alpha” is beta, cost leakage, survivorship, or look-ahead.**

---

## 2. Is “free daily US-equity directional alpha is mined out” a sound conclusion?

### Short verdict

**Directionally yes. Literally no.**

The correct statement is:

> “With current-listing yfinance-style daily equity data, generic directional feature mining is low EV and has not produced a deployable edge after proper controls. I should stop spending most of my time there.”

The incorrect statement is:

> “Daily US equity alpha is mined out.”

That stronger statement requires cleaner data and a narrower definition of alpha.

### Why your conclusion is directionally right

For a solo operator using daily bars, no borrow data, no point-in-time institutional fundamentals, no true historical constituents, no robust transaction-cost model for small-cap liquidity, and no unique alternative data, generic US equity alpha is a terrible hunting ground.

The reasons are structural:

- **Competition is brutal.** Daily equity signals are crowded by quant funds, stat-arb desks, pods, execution algos, and retail factor products.
- **Data is dirty.** Survivorship, corporate actions, delistings, stale adjusted prices, bad event timestamps, and universe leakage can easily dominate weak effects.
- **Costs are understated.** Shorting, borrow, spreads, market impact, liquidity filters, and rebalancing frictions matter more than most backtests admit.
- **Long-only equity “alpha” is usually beta timing.** Your PEAD and XS-ML conclusions are consistent with this.
- **Weak true effects require huge breadth.** If you do not have clean breadth, you will either miss them or overfit them.

So yes: I would stop allocating major research time to generic free daily equity directional alpha.

### Why the conclusion is not fully proven

You do not have survivorship-free US equities. That contaminates any cross-sectional conclusion. The contamination can go in both directions:

- It can **inflate** performance by excluding delisted losers.
- It can **hide** event effects if the usable universe is distorted toward survivors and current liquid names.
- It can **break historical constituent logic**, especially for PEAD, small-cap signals, and short-volume work.

Your equity KILLs are useful evidence, not final truth.

### What would settle it

Buy survivorship-free equities and run one clean, tightly scoped equity audit. Do not reopen endless equity ML research. Run the audit as a burial-or-resurrection exercise.

Minimum audit:

1. Use survivorship-free daily prices, delisted returns, historical liquidity, and historical index/sector membership.
2. Re-run only three families:
   - PEAD/event drift.
   - FINRA short-volume cross-section / overlay.
   - Simple robust cross-sectional factors: 12-1 momentum, 1-month reversal, value/quality if point-in-time fundamentals are available, liquidity/volume shock.
3. Require dollar-neutral or beta-neutral returns, real liquidity filters, and modern sub-period stability.
4. Cap total configurations up front. Do not let this become Alpha-v10 feature-mining hell.

Kill the entire equity program again if:

- No family has post-2015 net Sharpe > 0.35 after realistic costs.
- No family has residual alpha t-stat near 2 under HAC / block bootstrap.
- Results concentrate in microcaps you cannot trade.
- Track-B versus trend+carry is negligible or unstable.

If one family survives, it will probably be a low-turnover, event/risk-state overlay, not a clean standalone stock-picking engine.

---

## 3. Trend sleeve review

### Does the result match priors?

Yes. A diversified long-flat ETF time-series momentum strategy with equities, rates, gold, broad commodities, and USD producing Sharpe ~0.7 over 2007-2026 is plausible.

It does not smell obviously fake. The universe is simple, the rebalance frequency is slow, the signal windows are standard, and the crisis profile is what I would expect: helpful in slow bear markets, vulnerable in fast reversals.

### What I like

- The universe is liquid and implementable.
- Weekly rebalance is sane.
- Long-flat avoids shorting complexity and borrow/financing surprises.
- Inverse-vol sizing is reasonable.
- You are not pretending it is a 2-Sharpe edge.
- You increased allocation from 25% to 50%, which is probably correct if this is the only validated premium.

### What I do not like

#### 1. Calling it “alpha” is sloppy

Trend is a public alternative risk premium. Treating it as proprietary alpha will lead to wrong expectations and wrong validation standards.

#### 2. The 2007 start date is short and regime-specific

A 2007-2026 ETF backtest includes:

- a major equity crisis,
- a multi-decade bond bull market tail-end,
- post-GFC QE,
- COVID crash/rebound,
- inflation/rate shock,
- a strong US mega-cap regime.

It is useful, but it is not enough to infer a stable long-run Sharpe. You should extend the conceptual test with futures, indices, mutual fund proxies, or synthetic asset-class histories where possible.

#### 3. Full Kelly at 7.7x should be thrown in the trash

Do not let a computed full-Kelly number enter serious discussion. For a public, time-varying, crash-exposed premium estimated over 19 years, full Kelly is a fantasy. Your 50% gross may be conservative, but the “7.7x full Kelly” anchor is dangerous even if you haircut it.

Use drawdown, margin, live tracking error, and business/personal tolerance constraints instead.

#### 4. VIX governor may be right, but it is an overfit magnet

A fail-safe overlay that only reduces exposure is acceptable. But VIX/VIX3M has limited history, and crash filters are easy to tune post hoc.

The right question is not “does it improve Sharpe?” It is:

- Does it reduce left-tail in pre-registered crises?
- Does it avoid excessive false positives in calm markets?
- Does it help after costs and taxes?
- Does it reduce exposure before or during stress without relying on same-day impossible fills?

### What I would re-run for trend

1. **Leave-one-asset-out / leave-one-sector-out robustness**
   - Remove each ETF one at a time.
   - Remove all equities, all rates, all commodities/FX alternatives one sleeve at a time.
   - Confirm the edge is not mostly TLT + equity crash timing.

2. **Synthetic pre-ETF extension**
   - Use futures/index proxies to push the backtest earlier than 2007 where possible.
   - Compare parameter-free variants to your ETF implementation.

3. **Execution timing test**
   - Signal at close, trade next open.
   - Signal at prior close, trade VWAP-like proxy.
   - Include ETF spreads and realistic missed-fill assumptions.

4. **Parameter fragility grid**
   - Do not optimize. Run coarse grids:
     - lookbacks: 1/3/6/12 months, 3/6/12, 1/6/12, 1/3/12;
     - rebalance: weekly, biweekly, monthly;
     - vol windows: 20, 60, 120 days.
   - You do not need the best config. You need a plateau.

5. **Realized live/paper attribution**
   - Decompose live slippage into signal timing, ETF spread, partial fills, config drift, and broker state mismatch.

### Kill criteria for trend

I would not kill trend unless the re-runs show one of the following:

- The Sharpe collapses below ~0.35 under simple parameter variants.
- The post-2015 result is entirely one asset or one crisis.
- Next-open execution materially destroys the edge.
- The VIX governor only works because of impossible same-day assumptions.

Otherwise, trend remains the core sleeve.

---

## 4. Futures carry review

### Does the result match priors?

Yes. Broad futures carry with 30+ years of data, many markets, term-structure signals, weekly rebalance, inverse-vol sizing, and cross-sectional long/short construction producing honest Sharpe ~0.55-0.60 is plausible.

It is also exactly the kind of edge that should diversify ETF trend. A correlation around 0.25 to ETF trend and a book Sharpe lift from ~0.72 to ~0.89 are believable.

This is the most important new result in your program.

### What I like

- The economic intuition is real: term structure embeds financing, storage, convenience yield, hedging pressure, scarcity, and risk premia.
- Norgate futures data is a much better research substrate than yfinance equities.
- 76 liquid markets gives breadth.
- Weekly rebalance is sensible.
- The post-bug-fix survival is encouraging.
- Positive modern performance matters; carry is not just a pre-2010 artifact in your runs.
- Track-B improvement versus trend is the right deployment lens.

### What still smells risky

#### 1. Roll cost is not a small footnote

You already know this, but I will state it harder: **without modeled roll execution, the current carry Sharpe is not a final number.**

A missing 1.1-1.9% annual drag is large for a Sharpe ~0.6 strategy. It can be the difference between a good sleeve and a marginal one, especially after commissions, slippage, contract selection constraints, and margin/cash drag.

The fact that the strategy is still plausible after a -0.05 to -0.15 Sharpe haircut is encouraging. But do not deploy until this is built into the decisive run.

#### 2. Front-next carry can be too crude

Using nearest two contracts is defensible as a first pass, but it can be distorted by:

- first-notice/expiry effects,
- commodity seasonality,
- front-contract squeezes,
- stale or illiquid contracts,
- negative/near-zero denominator issues,
- contract-specific delivery optionality,
- curve kinks unrelated to persistent carry.

You fixed one CL negative-price bug. That is good. But the existence of that bug tells me the curve engine needs more adversarial tests before capital.

#### 3. Current-liquidity universe is not fully point-in-time

Futures rarely delist in the same way equities do, so I am less worried than I would be in stocks. But current liquidity filters can still introduce survivorship/capacity bias. Your <=2005-history check helps, but I would not call the issue closed.

#### 4. Full-sample vol matching should be removed from all decisive book claims

For research comparison, fine. For promotion, no. Use point-in-time rolling vol estimates and fixed ex-ante allocation rules.

#### 5. “Carry + futures trend” needs careful interpretation

You say futures trend is decayed and redundant with ETF trend. I believe that. Do not allocate to decayed futures trend just because it creates a canonical CTA narrative. If it helps only as a crisis hedge, model it as a conditional protective overlay, not as a return-seeking sleeve.

### What I would re-run for carry before deployment

This is the decisive checklist.

#### A. Cost and execution realism

1. Add explicit roll cost by contract family:
   - commissions,
   - half-spread / spread estimates,
   - slippage by asset class,
   - roll-window crossing cost,
   - contract switch cost even when target weight is unchanged.
2. Force signal date / execution date separation:
   - signal from EOD data available after close,
   - trade next session using executable contract.
3. Add contract multipliers, tick values, minimum contract granularity, and margin constraints.
4. Use realistic cash collateral accounting.
5. Simulate roll schedules using actual last trade / first notice / liquidity roll rules, not just continuous-series convenience.

#### B. Term-structure robustness

Run all of these as pre-registered robustness checks, not optimization:

| Variant | Purpose |
|---|---|
| Front-next slope | Your base case. |
| Second-third slope | Tests whether front expiry effects are driving results. |
| Front-third annualized slope | Tests curve shape stability. |
| Log-price slope instead of ratio slope | Reduces denominator pathologies. |
| Skip front in energy/agriculture | Tests commodity front squeeze/seasonality dependence. |
| Sector-neutral carry ranks | Tests whether one sector dominates. |
| Sector leave-one-out | Tests concentration. |
| Long-only backwardation only | Tests whether short contango leg is doing all work / has crash risk. |
| Short-only contango only | Separates legs and borrow/margin pain. |
| Carry excluding energy | Critical because energy often dominates term-structure carry. |
| Carry excluding commodities | Tests whether financial futures carry contributes at all. |

#### C. Attribution

Decompose returns into:

- spot move,
- roll yield / curve carry,
- collateral return,
- cross-sectional selection,
- sector allocation,
- volatility targeting effect,
- rebalance effect,
- short leg contribution,
- crisis periods.

If “carry” profits mostly from spot trend or one sector, it is not dead, but you need to know what you own.

#### D. Out-of-sample / forward test

IBKR paper is not statistically meaningful for Sharpe validation over a few months. It is meaningful for implementation validation.

Use live paper to answer:

- Did we trade the intended contract?
- Did the roll happen on schedule?
- Did the order size match margin and contract granularity constraints?
- Did slippage match modeled assumptions?
- Did broker state reconcile to internal state every day?
- Did cash/margin behave as expected?
- Did a volatile day break anything?

Do not wait for a statistically significant live Sharpe. You will never get that in a reasonable time. The backtest plus economic prior validates the premium; live paper validates the plumbing.

### My deployment verdict on carry

**Yes, adding carry to trend is the right next deployment path.**

But sequence matters:

1. Finish roll-cost model.
2. Re-run decisive carry tests with PIT rolling vol and no full-sample vol matching.
3. Run term-structure robustness grid.
4. Build IBKR paper execution and reconciliation.
5. Paper trade through at least one full roll cycle in multiple asset classes.
6. Start live tiny, with strict notional/margin caps.

If honest Sharpe remains >0.45, Track-B remains positive, and implementation slippage is within modeled bands, I would deploy a small allocation.

---

## 5. Validation harness review

### Overall verdict

Your validation harness is directionally sound and far better than “backtest Sharpe and pray.” But it still has three conceptual weaknesses:

1. **Human-selected rule families are not truly OOS-by-construction.**
2. **Family-level multiple testing is still hard to measure.**
3. **Risk premia and alpha need different acceptance logic.**

You know pieces of this already. The danger is becoming proud of the harness and letting it become a new source of false confidence.

### What is strong

- Pre-registration ledger.
- Two-track gate separating standalone stream from book improvement.
- HAC Sharpe and stationary bootstrap.
- PBO / CSCV when multiple configs are tested.
- Structural live-paper requirement before capital.
- Acknowledgment of prior Type-I and Type-II failures.
- Positive/negative controls.
- Adversarial review culture.
- Sub-period stability guard.

This is a serious process.

### What needs correction

#### 1. “Rules-based sleeves are OOS-by-construction” is not fully true

A fixed rule can be OOS in the mechanical sense that it is not refit per fold. But the human selected the rule family after observing history, literature, prior runs, adjacent failures, and known market behavior.

That means the *family selection* is in-sample even if the parameters are frozen.

Pre-registration helps. It does not erase the fact that you chose which hypotheses to pre-register based on everything you already knew.

What to do:

- Track trial count at the **strategy-family level**, not just parameter/config level.
- Maintain a “research degrees of freedom” log: discarded variants, informal notebooks, bug-fix reruns, reviewer suggestions, and post-hoc exclusions.
- Use stronger priors for heavily mined families and weaker priors for economically grounded, externally documented premia.

#### 2. PAPER tier may still be too permissive for crowded families

Point Sharpe >= 0.30 + HAC p<0.05 sounds reasonable, but with enough strategy families it will still pass noise. You need empirical false-positive calibration by family.

Suggested fix:

- Build a null strategy zoo:
  - random signals with same turnover,
  - permuted cross-sectional ranks,
  - lag-shifted labels,
  - synthetic AR(1)/GARCH returns,
  - random sector-neutral long/short sleeves,
  - randomized event dates.
- Run them through the full gate.
- Estimate empirical pass rates by asset class, turnover, and sample length.
- Adjust PAPER thresholds by family.

#### 3. CAPITAL tier should test expected utility, not just P(SR>0)

P(SR>0) is too weak. A strategy with 70% probability of SR>0 but 30% probability of painful tail loss may be a bad allocation.

Add posterior tests like:

- P(SR > 0.30 after costs) or P(SR > strategy-specific hurdle).
- P(maxDD > allowed drawdown).
- P(cost drag > 50% of gross edge).
- P(Track-B improvement > material threshold).
- Expected log/CRRA utility under cost and volatility uncertainty.

#### 4. Sub-period stability guard is useful but can overkill real premia

For alphas, sub-period stability is an excellent overfit guard.

For risk premia, requiring every half/era to look good can be too harsh. Real risk premia can have decade-long pain. The better test is:

- Does the bad period match the economic risk of the premium?
- Is the drawdown survivable at intended sizing?
- Is the premium compensated over full cycles?
- Does it diversify the rest of the book when it is not paid?

Rates carry being pre-2016 only is a good kill. But be careful applying the same standard to VRP or carry, where left-tail/drought behavior is part of the bargain.

#### 5. Track-B can reject useful options

Track-B versus the current book is good, but the current book is small and path-dependent. A strategy can fail Track-B today and be useful once another sleeve is added, or if it improves skew/drawdown rather than Sharpe.

For diversifiers and hedges, add metrics beyond ΔSharpe:

- Δexpected shortfall,
- Δmax drawdown,
- crisis-period contribution,
- correlation in worst 5% book days,
- margin/cash usage,
- convexity/concavity profile.

---

## 6. Which KILLs I trust and which I would reopen

### High-confidence KILLs / keeps

| Strategy | My verdict | Reason |
|---|---|---|
| Overnight vs intraday premium | Trust KILL | Gross exists, net dies. Daily round-trip cost is fatal. |
| Single-name earnings IV crush | Trust KILL for now | Single-name option spreads and event risk are brutal. Not your best arena. |
| ETF relative-value stat-arb | Trust KILL | ETF RV without intraday execution/microstructure edge is unlikely. |
| Rates carry via IEF ETF | Trust KILL | Pre-2016 artifact is plausible and correctly caught. |
| Sleeve allocator | Trust OFF | With 1-2 sleeves, allocator complexity is not earned. |
| Futures trend as standalone | Mostly trust not adding | Decay/redundancy with ETF trend is believable. Keep only as crisis diagnostic/overlay candidate. |

### Medium-confidence KILLs / parks

| Strategy | My verdict | Why not final |
|---|---|---|
| PEAD | Reopen only after clean equities | Survivorship/current universe and event timestamp quality matter. But expect modest result at best. |
| Cross-sectional ML swing ranker | Do not reopen generic ML; reopen only as clean factor audit | Current result likely right, but yfinance universe makes final burial imperfect. |
| FINRA daily short-volume | Reframe as overlay/composite | “Real but weak” is exactly the type of signal that may help de-risk or condition another sleeve. |
| Options-as-signal | Do not trust the kill as universal | Four years, no NBBO/OI, computed greeks, and 2022-2026 regime are not enough. But do not spend heavily here yet. |
| Index VRP / short-vol | Park, not kill | It is a risk premium, not alpha. Needs executable data and strict tail design. |
| Crypto trend | Keep paper only | Track-B fail is decisive for capital now. Revisit only with longer live OOS or if book composition changes. |

### What I would not reopen

Do not reopen broad “let’s try more equity ML features” research. That is where solo quant programs go to die.

If you reopen equities, it should be a single, clean, pre-registered audit after buying survivorship-free data.

---

## 7. Highest-EV strategies to test with existing data

Below are the strategy families I would prioritize. I am ranking by expected edge × feasibility × fit with your current data and infrastructure.

### Priority table

| Rank | Strategy family | Data needed | Expected edge | Feasibility | Why it belongs near the top |
|---:|---|---|---:|---:|---|
| 1 | Futures carry production-hardening + curve variants | Existing Norgate futures | High | High | Highest probability of becoming a real second sleeve. |
| 2 | Commodity calendar-spread / curve carry isolation | Existing Norgate full term structure | Medium-high | Medium | Uses your best data; may isolate carry from spot beta. |
| 3 | Carry × trend agreement / disagreement filter | Existing Norgate futures | Medium | High | Simple way to reduce carry crash/left-tail if robust. |
| 4 | Cross-sectional futures momentum, sector-neutral | Existing Norgate futures | Medium | High | Different from TSMOM; could diversify carry. |
| 5 | FINRA short-volume as equity/beta risk overlay | FINRA + SPY/ETF data; better with Norgate stocks | Medium-low | High | You already found information content; use it correctly. |
| 6 | Index/ETF defined-risk VRP | Polygon options + forward NBBO log | Medium concept, low current confidence | Medium | Potential third premium, but data not mature. |
| 7 | Trend breadth / cross-asset risk-state overlay | Existing ETF/futures/FRED/VIX data | Medium-low | High | Risk control, not alpha; useful if it reduces left-tail without overfit. |
| 8 | Clean equity event/factor audit | Requires Norgate stocks | Unknown | Medium | Not with current yfinance. Do once after data buy. |

---

## 8. Top strategy build cards

### Strategy 1 — Futures carry 2.0: robust production version

This is less “new idea” and more “do the obvious highest-EV thing correctly.” It should be first because you already have a plausible edge.

#### Hypothesis

A broad, diversified futures term-structure carry premium remains positive after explicit contract-roll costs, realistic execution lag, PIT rolling-vol targeting, sector concentration controls, and contract-level implementation constraints.

#### Universe

- Liquid futures from Norgate.
- Exclude contracts that fail minimum history/liquidity/price sanity checks.
- Group by sector: equity index, rates/bonds, FX, energy, metals, agriculture/softs.
- Pre-register inclusion/exclusion rules.

#### Signal

Base signal:

```text
carry_i,t = annualized_log_slope(front_contract, next_contract)
```

Use log slope rather than raw ratio as a robustness variant:

```text
carry_i,t = (log(P_front) - log(P_next)) / days_between_expiries * 365
```

Positive = backwardation / long preference. Negative = contango / short preference.

Robustness variants:

- front-next,
- second-third,
- front-third,
- skip-front for energy/agriculture,
- sector-neutral ranks,
- long-only backwardation,
- short-only contango,
- leave-one-sector-out.

#### Portfolio construction

- Weekly rebalance.
- Cross-sectional rank: long top carry, short bottom carry.
- Inverse-vol weights using PIT rolling vol.
- Sector cap and market cap/notional cap.
- Book vol target, but use only PIT rolling volatility.
- Contract granularity simulation.
- Cash collateral modeled.

#### Costs

- Commission per contract.
- Half-spread/slippage by asset class.
- Contract roll cost even if target notional unchanged.
- Execution next session after signal.
- Stress cost at 1x, 2x, 3x assumptions.

#### Tests

- Full period, post-2010, post-2015, 2020s.
- Leave-one-sector-out.
- Long leg / short leg attribution.
- Crisis contribution.
- Track-B vs ETF trend.
- Track-B vs ETF trend + cash.
- Track-B vs ETF trend + VIX governor.
- Cost sensitivity.
- Roll schedule sensitivity.

#### Promotion criteria

Promote to small live capital only if:

- Honest post-cost Sharpe > 0.45 full and > 0.35 post-2015.
- Track-B ΔSharpe or appraisal improvement remains positive under PIT vol blending.
- No single sector contributes >50% of total P&L.
- 2x cost remains viable.
- IBKR paper confirms roll/execution/slippage assumptions through at least one full multi-asset roll cycle.
- Worst historical drawdown is tolerable at intended allocation.

#### Kill criteria

Kill or shrink if:

- Edge disappears after explicit roll costs.
- Performance is mostly energy-only or short-contango-only.
- Post-2015 Sharpe falls below ~0.25.
- Implementation slippage exceeds 50% of gross edge.
- Margin/collateral usage makes the sleeve operationally fragile.

---

### Strategy 2 — Commodity calendar-spread / curve carry isolation

This is the most interesting genuinely under-used strategy in your current data. You own full futures term structures. Outright carry mixes spot beta, curve shape, collateral, and trend. Calendar spreads can isolate term-structure risk more directly, especially in commodities.

#### Hypothesis

Commodity term-structure premia are more cleanly harvested through calendar-spread positions than through outright front-contract long/short portfolios, producing lower correlation to trend and lower spot beta while preserving roll/carry return.

#### Universe

Start with liquid commodity futures only:

- energy,
- metals,
- grains,
- softs,
- livestock if data quality/liquidity is acceptable.

Exclude financial futures initially because calendar-spread mechanics differ and may need DV01/rate modeling.

#### Signal

For each commodity, compute curve slope between adjacent liquid maturities:

```text
slope_k,t = (log(P_k,t) - log(P_k+1,t)) / days_between_expiry(k,k+1) * 365
```

Also compute curve z-score versus that market’s own trailing history:

```text
z_slope_i,t = (slope_i,t - rolling_mean_i) / rolling_std_i
```

Two possible pre-registered variants:

1. **Carry-following variant:** trade in direction of backwardation/contango to earn rolldown.
2. **Curve-normalization variant:** fade extreme curve dislocations when slope z-score is extreme.

Do not test both endlessly. Pick one primary and one robustness variant.

#### Portfolio construction

For each commodity:

- Construct a calendar spread using contract k and k+1.
- Dollar-notional or volatility-match legs.
- Roll both legs before first notice/liquidity cutoff.
- Size spreads by realized spread volatility, not outright volatility.
- Equal risk across commodities, with sector caps.

Example for carry-following:

- If curve is backwardated, position to benefit from near contract richness / rolldown depending on precise spread return definition.
- If curve is contango, take opposite or avoid if short-side/roll mechanics are too expensive.

Be very careful with sign conventions. Calendar-spread P&L sign bugs are common.

#### Tests

- Spread return built from actual contracts, not continuous approximations only.
- Next-day execution.
- Roll rules explicit.
- Seasonality controls:
  - month-of-year fixed effects,
  - contract-month families,
  - exclude delivery squeeze windows.
- By sector and by commodity attribution.
- Stress periods: 2008, 2014-2016 oil collapse, 2020 negative oil, 2022 commodity shock.
- Correlation to ETF trend, outright futures carry, and equity beta.

#### Promotion criteria

- Standalone post-cost Sharpe > 0.4.
- Correlation to ETF trend < 0.3 and to outright carry < 0.7.
- Post-2015 positive.
- No single commodity dominates.
- Drawdown smaller than outright carry at comparable vol.
- Execution complexity manageable at your account size.

#### Kill criteria

- Returns vanish when excluding energy.
- Returns depend on front-month delivery squeeze windows.
- Calendar-spread slippage overwhelms edge.
- Contract granularity makes sizing impossible in a small account.

#### Why this is high EV

Because it exploits the best thing you now own: futures term structure. It is not another yfinance stock-feature lottery ticket. It is economically coherent and adjacent to your strongest new result.

---

### Strategy 3 — Carry × trend agreement filter

This is not “add futures trend as a sleeve.” It is a risk-control variant for carry.

#### Hypothesis

Futures carry performs better, or has less left-tail risk, when the carry direction is not fighting the market’s own medium-term price trend. A simple agreement filter can improve drawdown and Track-B without overfitting.

#### Universe

Same as futures carry.

#### Signal

For each market:

```text
carry_direction_i,t = sign(carry_i,t)
trend_direction_i,t = sign(total_return_i,t over 6 or 12 months)
```

Primary pre-registered rule:

```text
if carry_direction == trend_direction:
    weight = base_carry_weight
else:
    weight = 0.5 * base_carry_weight   # or 0, but choose one before test
```

Do not optimize the lookback. Pick 12-month excluding most recent week/month, with 6-month as robustness.

Alternative:

```text
weight = base_carry_weight * clip(z_trend_strength, 0.5, 1.5)
```

But I prefer the binary/simple version first.

#### Tests

- Compare base carry vs filtered carry.
- Focus on drawdown, skew, worst 5% days, and Track-B, not just Sharpe.
- Test by sector.
- Test post-2015 and 2020s.
- Count turnover changes.
- Run cost sensitivity.

#### Promotion criteria

- Improves expected shortfall or maxDD by at least 10-15% at similar return.
- Does not reduce post-2015 Sharpe materially.
- Does not add significant turnover.
- Works across sectors or at least is not entirely one sector.

#### Kill criteria

- Improvement is only full-sample/pre-2010.
- Improvement comes from avoiding one historical event only.
- Sharpe improves but expected shortfall worsens.
- Rule becomes parameter-sensitive.

#### Why this belongs

Carry can be a “picking up premium until regime break” strategy. A simple trend agreement filter is one of the few overlays with economic rationale and low degrees of freedom.

---

### Strategy 4 — Cross-sectional futures momentum, sector-neutral

You killed futures TSMOM as decayed/redundant. That does not fully kill cross-sectional futures momentum.

#### Hypothesis

Relative momentum across futures markets, especially sector-neutral and low-turnover, can produce a return stream distinct from time-series trend and carry.

#### Universe

Same liquid futures universe.

#### Signal

```text
mom_i,t = total_return_i,t over 12 months excluding most recent 1 month
```

Rank within sector first, then combine sectors. This prevents “all equity indices long / all rates short” macro beta from dominating.

#### Portfolio

- Long top quantile, short bottom quantile within sectors.
- Equal risk by market and sector.
- Monthly or weekly rebalance; prefer monthly if turnover lower.
- Vol target with PIT vol.

#### Tests

- Full, post-2010, post-2015, 2020s.
- Compare to futures TSMOM, ETF trend, and carry.
- Residualize against carry and ETF trend.
- Cost sensitivity.
- Sector-neutral vs global rank.
- Long/short leg attribution.

#### Promotion criteria

- Post-cost Sharpe > 0.35.
- Correlation to ETF trend < 0.5 and to carry < 0.5.
- Positive post-2015.
- Track-B positive versus trend+carry.

#### Kill criteria

- It is just another version of trend.
- It dies post-2015.
- It requires weekly turnover but only has marginal gross edge.

#### My prior

Lower than carry. Worth testing because it is cheap and uses good data, but I would not expect a miracle.

---

### Strategy 5 — FINRA short-volume as a risk overlay, not standalone alpha

You already found “real but weak.” That is exactly the profile of a risk-state input.

#### Hypothesis

Aggregate or sector-level abnormal short-volume pressure contains information about near-term equity downside risk, but not enough to stand alone as a stock-picking strategy. It may improve equity/trend exposure timing or crash governor behavior.

#### Universe

- Start aggregate market: SPY / QQQ / IWM risk overlay.
- Then sector ETFs if aggregate works.
- Do not do single-name cross-section until survivorship-free equities are available.

#### Signal

For each day:

```text
short_pressure_t = zscore(aggregate_short_volume_ratio_t over trailing 252 days)
```

Where:

```text
short_volume_ratio = short_volume / total_volume
```

Potential variants:

- aggregate all names,
- liquidity-weighted aggregate,
- top 1000 liquid names,
- sector aggregates.

Pick one primary; avoid variant explosion.

#### Overlay rule

Example:

```text
if short_pressure_z > 1.5:
    equity_risk_multiplier = 0.5
else:
    equity_risk_multiplier = 1.0
```

Apply only to equity ETF trend sleeve exposure, not the entire book.

#### Tests

- Does it reduce SPY/QQQ/IWM left-tail next 1-10 days?
- Does it improve trend sleeve expected shortfall?
- Does it add beyond VIX/VIX3M governor?
- Does it work post-2020?
- Does it avoid excessive false positives?

#### Promotion criteria

- Improves equity sleeve expected shortfall or maxDD materially.
- Does not reduce annual return by more than the drawdown benefit justifies.
- Adds value beyond VIX governor.
- No look-ahead from FINRA publication timing.

#### Kill criteria

- It is redundant with VIX/VIX3M.
- It reduces return without reducing left-tail.
- It only works in one crisis.

#### My prior

Modest. But the cost is low because you already have the data and signal work.

---

### Strategy 6 — Index/ETF defined-risk VRP

Do not run this as “alpha.” Run it as a small, explicitly crash-risk-budgeted premium.

#### Hypothesis

Broad index/ETF option implied volatility is overpriced versus realized volatility often enough to support a defined-risk short-vol sleeve, but only when spreads, tail loss, and execution are tightly controlled.

#### Universe

Start only with the most liquid underlyings:

- SPY,
- QQQ,
- IWM,
- maybe TLT/GLD later.

Do not start with single names.

#### Instruments

Prefer defined-risk structures:

- put credit spreads,
- iron condors,
- delta-capped short strangles with hard stop/hedge rules only if execution is trustworthy.

Initial candidate:

- 30-45 DTE put spread,
- short delta around 20-30,
- long wing defining max loss,
- enter only when IV-RV spread and term structure exceed threshold,
- no entry during VIX backwardation / crash governor risk-off.

#### Data issue

Your historical options data is not enough for promotion. Four years with computed greeks and no historical NBBO/OI is exploratory. Forward NBBO logging is the right path, but it needs time.

#### Tests

- Use conservative fill assumptions: enter near ask for shorts / bid for longs, exit pessimistically.
- Stress vol shocks.
- Include early assignment/exercise logic where relevant.
- Separate 2022, 2023, 2024, 2025, 2026.
- Measure tail loss, not just Sharpe/profit factor.
- Track-B versus trend+carry.

#### Promotion criteria

- Positive after conservative executable fills.
- Defined max loss acceptable at intended sizing.
- Does not blow up in 2022-style rate/vol markets or COVID-like vol expansion proxy scenarios.
- Forward NBBO paper confirms fill assumptions.
- Adds to book after left-tail penalty, not just Sharpe.

#### Kill criteria

- Edge disappears at conservative fills.
- Tail losses dominate small gains.
- Signal requires single-name options.
- Requires daily management you cannot execute reliably.

#### My prior

Conceptually real, but easy to underestimate tails and costs. This is not the next deployment after carry. It is a later third-premium candidate.

---

## 9. Data / asset recommendation

### The highest-EV next spend

**First spend: make futures carry executable.**

This may not be a “data buy,” but it is the highest-EV capital/research spend:

- IBKR paper/live futures plumbing.
- Contract metadata and roll calendar hardening.
- Realistic slippage/commission model.
- Margin/collateral accounting.
- Broker reconciliation.
- Contract-level order preview and kill switches.

You have a candidate edge. The bottleneck is no longer idea generation. It is implementation truth.

### The highest-EV actual data buy

**Buy survivorship-free US equities next, but for a bounded audit, not because I expect a gold mine.**

Norgate US Stocks / equivalent clean survivorship-free equity data is the right next buy because it is relatively cheap and resolves a major epistemic hole.

What it gives you:

- Delisted names.
- Historical constituents.
- Cleaner cross-sectional tests.
- Re-test PEAD and FINRA cross-section honestly.
- Ability to stop arguing with yourself about yfinance contamination.

What it probably does *not* give you:

- A magical standalone stock alpha.
- A reason to restart generic ML feature mining.
- A high-turnover stat-arb program.

Use it to answer: “Was my equity research dead because the idea was dead, or because the data was invalid?”

### What not to buy yet

#### Do not buy intraday/tick data yet

You have repeatedly learned that turnover kills you. You do not currently have the execution stack, market microstructure edge, or operational capacity to compete intraday. Tick data will create a giant research surface and very little deployable edge.

#### Do not buy expensive options history yet unless the VRP design is frozen

Historical NBBO/options quote data can be useful, but only after you pre-register a tight index/ETF VRP design. Otherwise, it becomes an expensive overfit playground.

Let your forward NBBO logger mature. Use it to calibrate fills. If the strategy still looks promising, then buy deeper options history.

#### Do not buy alt-data

Alt-data is almost certainly low EV for you right now. It is expensive, messy, crowded, and requires domain-specific cleaning. Your current bottleneck is not lack of exotic data; it is turning your best existing data into robust deployable premia.

---

## 10. Architecture review

### Overall verdict

Your architecture is appropriate for paper trading and slow ETF sleeves. It is not yet robust enough for live multi-asset futures trading without upgrades.

The design is not crazy. A single-process FastAPI app with APScheduler, agents, Postgres, Redis, React dashboard, and config-gated sleeves is fine for low-frequency paper trading. But the failure modes change once you add futures:

- contract rolls,
- margin,
- exchange hours,
- contract multipliers,
- partial fills,
- stale market data,
- order rejection,
- broker outages,
- state drift,
- liquidation risk,
- multiple currencies/asset classes if expanded,
- cash collateral interaction.

### What is good

- Modular sleeves.
- PM/RM/Trader separation.
- DB-driven config flags.
- Decision audit trail.
- Reconciler.
- Tests and CI discipline.
- Pre-registration and registry.
- Research/live separation conceptually.

### What I would improve before futures capital

#### 1. Separate the scheduler from the web server

Do not let the live trading scheduler live inside the same process as the API/dashboard indefinitely.

Target architecture:

- API/dashboard process: read-only monitoring and controlled config writes.
- Scheduler/strategy process: creates signal snapshots and target portfolios.
- Execution worker: consumes approved orders and places broker orders.
- Reconciliation worker: broker/account state reconciliation.
- Notification worker: alerts.

For small scale, these can still be simple processes/containers. You do not need Kubernetes theater. You do need failure isolation.

#### 2. Make every trading action idempotent

Every rebalance should have a unique run ID:

```text
strategy_id + signal_date + rebalance_timestamp + config_hash + code_version
```

Orders should be safe to retry. If the process crashes mid-run, restarting should not duplicate trades.

#### 3. Broker state must be source of truth before trading

Before any order placement:

- Pull broker positions.
- Pull open orders.
- Pull cash/margin.
- Compare with internal DB.
- Block trading if mismatch exceeds threshold.
- Require manual or automated reconciliation.

Your DB is not reality. The broker is reality.

#### 4. Immutable signal snapshots

For every live decision, store:

- raw input data version/hash,
- signal values,
- target weights,
- risk adjustments,
- config hash,
- code commit hash,
- generated orders,
- broker responses,
- final positions.

You should be able to reconstruct any position months later.

#### 5. Futures-specific risk layer

Add risk checks that are futures-aware:

- contract multiplier sanity,
- tick value sanity,
- margin requirement,
- available liquidity/cash buffer,
- notional exposure by asset class,
- stress loss by market move,
- exchange holiday/session rules,
- roll-window exposure limits,
- maximum contracts per market,
- correlated sector exposure.

#### 6. Dead-man switch and heartbeat

For live futures:

- Daily heartbeat expected.
- If no heartbeat, no new trades.
- If reconciliation fails, no new trades.
- If market data stale, no new trades.
- If margin buffer below threshold, reduce risk or block adds.
- If broker API errors persist, stop.

#### 7. Backtest/live parity tests

For every sleeve:

- Same signal code or identical generated signal artifact in research and live.
- Golden test dates with expected signals/weights/orders.
- Replay mode: feed historical data into live engine and confirm it reproduces research outputs.

This matters more than adding more tests broadly.

### What not to overbuild

Do not build a complex allocator yet. With trend + carry, a fixed-risk allocation is better than an optimizer. Optimizers will mostly fit noise and create turnover.

The architecture should make simple strategies safe, not make fragile strategies look professional.

---

## 11. What makes this a real multi-strategy book?

Right now, it is not one.

Trend + cash is one premium plus collateral. Trend + carry becomes two premia. Better, but still not a robust multi-strategy book.

A robust book needs return streams that differ across:

- economic driver,
- asset class,
- crisis behavior,
- holding period,
- cost structure,
- data dependency,
- crowding risk,
- liquidity/margin profile.

### Current book after carry

| Sleeve | Driver | Crisis behavior | Concern |
|---|---|---|---|
| ETF trend | Trend / crisis convexity | Good in slow bear, whipsaw in fast shock | Public, regime-dependent. |
| Futures carry | Term-structure / hedging pressure / convenience yield | Can be exposed to commodity/risk unwind | Needs roll/margin execution validation. |
| Cash/T-bills | Risk-free rate | Stabilizer | Not alpha. |
| VIX governor | Crash de-risking | Helpful if timely | Overfit/redundancy risk. |

This is a reasonable starter alternative-premia book. It is not enough to declare strategic diversification.

### What the third real sleeve should probably be

Best candidates:

1. **Index/ETF defined-risk VRP** if executable and tail-budgeted.
2. **Commodity calendar-spread carry** if distinct from outright carry.
3. **Clean equity event/factor sleeve** if survivorship-free audit revives something.
4. **Risk-state overlay** if it materially reduces tail without killing return.

My preferred path:

```text
Trend -> Futures carry -> Commodity spread/curve carry OR defined-risk index VRP -> clean equity audit only if data bought
```

---

## 12. Monday-morning action plan

If I ran this book on Monday morning, I would do the following in order.

### Move 1 — Freeze new idea generation until carry is made executable

Stop opening new research branches for a few weeks. The highest-EV task is converting carry from “promising backtest” to “small live-ready sleeve.”

Deliverables:

- Explicit roll-cost model.
- PIT rolling-vol blend.
- Contract-level execution simulation.
- Sector/curve robustness report.
- IBKR paper plan.

Evidence that confirms the move:

- Carry remains honest Sharpe >0.45 after all costs.
- Track-B remains positive versus trend.
- No single sector/contract dominates.

Evidence that refutes it:

- Roll/slippage/costs erase the edge.
- Edge is mostly front-month energy artifact.
- Implementation complexity is too high for small account size.

### Move 2 — Build futures execution and reconciliation as production infrastructure

Do not bolt futures onto the Alpaca-style ETF executor. Futures need their own contract/margin/roll logic.

Deliverables:

- IBKR paper account connected.
- Contract master and roll calendar.
- Order preview with contract multipliers and margin.
- Reconciliation worker.
- Kill switch and heartbeat.
- Immutable signal/order snapshots.

Evidence that confirms:

- Paper trades match intended contract/notional.
- Roll cycle completes without state drift.
- Slippage matches model.
- Margin/cash accounting is correct.

Evidence that refutes:

- Contract granularity makes intended sizing impossible.
- Broker/API issues create unacceptable operational risk.
- Slippage materially exceeds assumptions.

### Move 3 — Run a futures curve research sprint, not an equity sprint

Use the Norgate term structure harder before buying more data.

Test only:

1. Commodity calendar-spread carry.
2. Carry × trend agreement filter.
3. Cross-sectional futures momentum sector-neutral.

Deliverables:

- One pre-registration per strategy.
- One decisive run each.
- One short kill/keep memo each.

Evidence that confirms:

- At least one strategy has post-cost Sharpe >0.35 and Track-B positive versus trend+carry.

Evidence that refutes:

- All results collapse post-2015 or after costs.
- Results are redundant with carry.

### Move 4 — Buy survivorship-free equities only after the carry sprint

Buy clean equities to settle the contaminated research question. Do not let it derail carry deployment.

Deliverables:

- Clean PEAD audit.
- Clean FINRA cross-sectional/overlay audit.
- Simple factor audit.

Evidence that confirms:

- A low-turnover equity/event family survives post-2015 with residual alpha and Track-B improvement.

Evidence that refutes:

- Clean data confirms prior nulls.
- Edge exists only in untradable microcaps.
- Costs/beta explain returns.

### Move 5 — Keep options VRP parked until NBBO evidence matures

Do not declare options dead, but do not prioritize them over carry.

Deliverables:

- Continue forward NBBO logging.
- Freeze one defined-risk index VRP design.
- Paper fill model based on actual quotes.

Evidence that confirms:

- Conservative executable fills leave enough premium.
- Tail loss at defined risk is acceptable.
- Track-B improves trend+carry after drawdown penalty.

Evidence that refutes:

- Spread capture eats edge.
- Tail events dominate.
- Requires active management beyond your ops capacity.

---

## 13. Specific claims and how to test them

| Claim | My belief | Confirming evidence | Refuting evidence |
|---|---:|---|---|
| ETF trend is real enough to trade | High | Robust across parameters, assets, execution timing, and extended proxies | Collapse under next-open execution or dependence on one asset/crisis |
| ETF trend is alpha | Low | Significant residual after broad trend/risk-premia factors | Explained by public trend/risk-premia exposures |
| Futures carry is real | Medium-high | Survives roll costs, sector leave-outs, front-skip variants, PIT vol | Dies after roll/slippage, dominated by one sector/front artifact |
| Carry should be next deployment | High if execution tests pass | Track-B remains positive and IBKR paper validates implementation | Contract/margin/slippage make it untradeable at account size |
| Futures trend should be added | Low | Adds crisis convexity without hurting modern Track-B | Continues to degrade trend+carry book |
| US equity daily alpha is mined out | Directionally medium-high, literally unproven | Clean Norgate equity audit confirms PEAD/XS nulls | Clean data revives low-turnover event/factor effect |
| Options-as-signal is dead | Low-medium | Longer executable quote history confirms negative net returns | Clean NBBO/OI data shows robust effect in liquid index/ETF options |
| Index VRP is investable for you | Medium-low now | Conservative defined-risk backtest + forward NBBO fills + tail budget pass | Tail/costs dominate or ops too active |
| FINRA short-volume is useful | Medium as overlay, low standalone | Improves expected shortfall beyond VIX governor | Redundant with VIX/market vol and costs return |
| Architecture is live-futures-ready | Low-medium | Separate execution/recon, idempotent orders, roll/margin controls pass paper | State drift, contract errors, margin surprises, API fragility |

---

## 14. Research rules I would impose going forward

### Rule 1 — No more broad equity feature mining without clean data

This is the biggest time sink. Stop it until survivorship-free equities are bought.

### Rule 2 — Every strategy must be labeled before testing

Use one of:

- alpha,
- risk premium,
- hedge/overlay,
- cash/collateral optimization,
- implementation improvement.

Each class gets a different gate.

### Rule 3 — Promotion requires implementation truth, not just statistical truth

For each sleeve, ask:

- Can I actually trade this at my account size?
- Do I know the contract/instrument I will hold?
- Are costs modeled where they occur?
- Is broker state reconciled?
- What breaks on a volatile day?

### Rule 4 — Family-level trial accounting

Count strategy families, not just parameter variants. If you test 20 futures curve ideas and one passes, that is not the same as one pre-registered economic hypothesis passing.

### Rule 5 — No optimizer until there are at least three real sleeves

Fixed risk budgets first. Optimizers later, if ever.

### Rule 6 — Any overlay must pass a “painfully simple” benchmark

Compare against:

- static de-risking,
- volatility targeting,
- simple VIX threshold,
- simple trend filter,
- cash allocation.

If a complex overlay cannot beat simple rules robustly, kill it.

### Rule 7 — Treat live paper correctly

Live paper validates plumbing, slippage, data availability, order handling, and operational behavior. It does not validate Sharpe over short periods.

Do not require live paper to prove the premium statistically. That will either delay good strategies forever or create false comfort from tiny samples.

---

## 15. Final blunt assessment

You are not kidding yourself in the most dangerous way. You have killed enough ideas to show discipline. You found trend because trend is real. You found futures carry after buying better data because better data and better asset-class structure are where your real opportunity was. That is the right lesson.

But you are still at risk of kidding yourself in a subtler way: thinking that a rigorous harness plus many killed ideas means the remaining book is more special than it is.

The current book is not special. It is a sensible, small, slow alternative-premia implementation. That can be valuable if it is robust, cheap, and honestly sized.

The next step is not “find ten new alphas.” The next step is:

1. Turn carry into a real executable sleeve.
2. Use the futures term structure more deeply.
3. Buy clean equities only to resolve contaminated conclusions.
4. Keep options VRP on a slow, data-maturing track.
5. Harden the live architecture before futures capital.

If carry survives the next round, MrTrader becomes a legitimate two-premium book: trend + carry + cash, with a path to a third sleeve. That is not a hedge fund yet. But it is no longer a toy.

If carry fails after roll/execution realism, then the honest answer is harsher: you have one public premium and a research platform. In that case, reduce ambition, keep trading trend/cash conservatively, and spend your effort acquiring cleaner data or moving into asset classes where daily signals are not completely over-mined.

Candor over comfort: **the app is ahead of the alpha.** Your job now is to keep the app from fooling you while you turn the one promising new premium into something actually tradable.

