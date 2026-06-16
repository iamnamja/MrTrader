# MrTrader Alpha-v8 — Brutally Honest Quant Review

**Date:** 2026-06-16  
**Reviewer stance:** world-class systematic PM / quant researcher perspective  
**Input packet reviewed:** `01_PROMPT.md` through `09_ALPHA_V8_PLAN.md`  
**Output purpose:** feed this back into the main LLM / VS Code agent as an external critique and next-step plan

---

## 1. Verdict

MrTrader is **not currently on a path to discovering many proprietary alpha signals from free daily US equity data**. It **is** on a credible path to becoming a small, disciplined, retail-scale **risk-premia and overlay engine** whose durable core is trend, cash/T-bills, and maybe one or two defensive/timing overlays. The biggest change I would make is this: **stop treating the platform as an alpha-discovery factory for single-name US equities, and re-charter it as a tradable multi-asset premia book with explicit data-acquisition gates.** Your process is now much better than the data you are feeding it. Further “cleverness” on the same free equity bars is mostly research theater. The next real lever is **new tradable breadth**: clean futures data plus a futures-capable broker if the research passes; crypto spot because Alpaca can execute it; and a carefully live-papered options-structure sleeve if you accept that historical options data is underpowered.

My blunt read: your kills are probably honest. Trend survived because it is real, economically grounded, and has enough history and breadth. Most of the rest died because the signal is not there at your data horizon, not because your gate is stupid. The danger now is not under-searching; it is continuing to spend months polishing the referee while playing in a field with no edge left.

---

## 2. What you got right

### 2.1 The research process is unusually good for a solo system

You have moved well past the normal retail-algo failure mode. Most solo trading systems die from some combination of look-ahead, survivorship bias, optimizing on a single backtest, under-modeled costs, and emotional attachment to a story. You have done the opposite:

- pre-registration before confirmatory runs;
- CPCV / purge / embargo discipline;
- corrected early harness bugs and re-ran the suspect kills;
- a both-halves stability guard;
- separate Track-A standalone and Track-B book-delta thinking;
- independent adversarial review before promotion;
- feature flags and owner-gated activation;
- a uniform Sleeve Lab instead of one-off notebooks.

That is a serious research culture. It is more institutional than most small funds.

### 2.2 The main empirical conclusion is probably right

Your conclusion that **free daily US equity data is largely mined out for additive large-cap alpha** is correct enough to be operationally decisive. I would phrase it more narrowly:

> “Currently-listed, survivorship-prone, free daily US equity bars plus common price/volume/fundamental features are not sufficient to produce reliable additive alpha after realistic costs and stability gates.”

That statement is hard to dispute. It does not mean no US equity alpha exists. It means **the alpha left requires something you do not currently have**: cleaner point-in-time data, broader/deeper universe with delistings, better event data, flow/positioning data, microstructure data, or a structural execution advantage.

### 2.3 Killing PEAD was mature

Many retail/systematic researchers keep PEAD alive forever because the academic story is plausible and the in-sample variants can be made to look good. Your event-level inference demotion is exactly the right kind of kill. If an earnings-event strategy does not survive event-panel inference and does not deliver clean post-cost book value, stop reopening it with threshold filters.

### 2.4 Treating overlays separately from sleeves is correct

This is a real architectural insight. A de-risking overlay should not be forced through the same standalone-return lens as an additive sleeve. It is allowed to give up return if it improves the left tail, provided the rule is pre-registered, point-in-time, and measured marginally against the already-overlaid book.

The VIX-term governor and credit-selective overlay are the right class of idea for your current data envelope: not “alpha” in the pure sense, but **risk-shaping signals** that can make the one real edge more livable.

---

## 3. Methodology and validation critique

This section is intentionally critical. Most of your setup is good. These are the places I would still worry.

### 3.1 The gate may still be too binary for weak but useful premia

Ruler-v2 is much better than the old Sharpe/DSR gate. However, you still have a tendency to make promotion decisions as binary pass/fail verdicts on noisy components. That is healthy for alpha claims, but too rigid for portfolio construction.

A weak sleeve with SR 0.25, correlation 0.05 to trend, low crisis co-crash, and a strong economic rationale may be economically useful at 5–10% risk even if it misses a strict Track-A test. Conversely, a sleeve with SR 0.55 and 0.55 crisis beta to the trend book may be useless.

**Recommendation:** Keep the promotion gate for “standalone alpha” claims, but add an **expected utility / Bayesian allocation layer** for small risk-premia satellites. The question should not always be “did it pass?” It should sometimes be:

- What is the posterior distribution of marginal book Sharpe?
- What is the posterior distribution of max drawdown contribution?
- What size would make the worst credible outcome tolerable?
- Does it earn a tiny allocation, not a binary live/dead verdict?

This matters especially for rates carry, which was a robust near-miss. A system that treats that exactly the same as an IC=0 equity ML failure is throwing away information.

### 3.2 The both-halves guard is good, but it can kill regime premia

The both-halves guard is excellent against p-hacking. It is less appropriate for strategies whose payoff is explicitly episodic: tail hedges, crash governors, volatility premia, carry crashes, and some event premia.

For those, the right question is not always “positive in both halves.” It is:

- Did it behave according to the economic mechanism in the regimes where it is supposed to pay?
- Was the bleed consistent with the expected insurance cost?
- Did it avoid catastrophic co-crash with the book?
- Are the crisis windows sufficiently independent to justify the inference?

For example, a tail hedge can be negative in one half and still be rational if it pays in the crash half and has acceptable carry cost. A carry strategy can be positive in both halves and still be unacceptable if it loads on the same left-tail as equities.

**Recommendation:** Keep both-halves for return-seeking sleeves. For overlays / tail hedges / carry, replace it with a **mechanism-specific stability test**:

- tail hedge: bleed budget + crisis convexity + no hidden beta;
- carry: drawdown clustering + crisis covariance + recovery time;
- overlay: marginal drawdown reduction in multiple stress windows + false-positive opportunity cost;
- trend: whipsaw regime tolerance + crisis participation.

### 3.3 Track-B’s raw correlation threshold can be too crude

The `corr < 0.30` rule is directionally right, but raw full-sample correlation is not the best gate. A strategy can have moderate average correlation and still help if it is negatively correlated in the book’s left tail. Another strategy can have low average correlation but crash exactly when the book crashes.

You already partly address this with tail-overlap and crisis windows. I would go further and demote raw correlation from hard gate to one diagnostic among:

- conditional beta on book’s worst 5% days;
- marginal expected shortfall contribution;
- drawdown-overlap ratio;
- crash-window covariance;
- time-varying correlation during stress;
- “does it need liquidity at the same time the book needs liquidity?”

This is especially important if you test crypto or options. Full-sample correlations will understate crisis correlation until they suddenly do not.

### 3.4 You need to flip “true walk-forward required for promotion” to hard-on

The pipeline doc says you have per-fold retrain support and a `REQUIRE_TRUE_WF_FOR_PROMOTION` flag, but it is documented as default false during rollout. If that is still true in code, it is a live governance problem.

Frozen-model CPCV is useful as a robustness test, but it is not the same thing as a true out-of-sample walk-forward retrain. You already know this. The system should enforce it.

**Action:** Set `REQUIRE_TRUE_WF_FOR_PROMOTION=True` for all trained-model promotion paths. Rules-based return streams can use the appropriate non-trained path, but any ML model must prove itself with true per-fold retraining.

### 3.5 Your biggest remaining bias is not look-ahead; it is survivorship and tradability

The yfinance equity path remains structurally compromised for long/short and event work because delisted losers vanish. This can both overstate returns and distort universe formation. You have historical DB membership logic, but the price data itself remains the limiting object.

This does not invalidate your “no edge” conclusion; if anything, survivorship bias usually flatters long/short equity results. But it means you should stop spending time on fine distinctions in single-name equity research until you have survivorship-free data.

**Practical rule:** No new single-name equity long/short, event, or cross-sectional strategy should be allowed to claim anything stronger than “exploratory” without survivorship-free prices and point-in-time membership.

### 3.6 Paper trading is not execution validation

Alpaca paper is useful for workflow, state, order logic, and reconciliation. It is not sufficient for slippage, queue position, partial fills, options spreads, borrow availability, or overnight liquidity.

This matters because your surviving edge is low-frequency ETF trend, where paper/live slippage is probably fine, but your future candidates may not be. Options structures, 24/5 overnight trading, crypto, and intraday rules all have execution details that paper trading will not capture.

**Action:** When a candidate survives research, use a tiny real-money “execution validation” phase before trusting fills. For $100k notional paper, even $1k–$5k live notional can reveal spread/fill behavior without turning the project into a capital-risk exercise. If you do not want real money yet, keep these strategies in shadow and do not mark them as execution-validated.

### 3.7 The platform may be overbuilt relative to the number of live edges

The PM/RM/Trader split is clean. The risk is that the architecture encourages you to keep filling slots: swing selector, intraday selector, PEAD selector, options path, event path, etc. A sophisticated machine with one live strategy can accidentally become a pressure cooker for false positives.

**Recommendation:** Treat every inactive path as technical debt unless it supports a live or near-live sleeve. Freeze or remove dead scanning/retraining code. Keep the Sleeve Lab. Keep the live trend sleeve. Keep overlay support. Archive the rest until data changes.

---

## 4. Architecture critique

### 4.1 Keep the three-agent abstraction, but make execution state more deterministic

The PM/RM/Trader design is fine. What matters for live reliability is not the agent metaphor; it is deterministic state transitions and idempotency.

I would want the live engine to have:

- a single append-only decision ledger;
- deterministic target-position snapshots by sleeve;
- explicit order intent IDs;
- idempotent order submission and cancellation;
- broker-position reconciliation before and after every rebalance;
- a replay tool: “given this data snapshot and config, reproduce the target orders exactly.”

If this already exists, good. If not, it is higher priority than more research ideas.

### 4.2 Separate research code from live code more aggressively

The current system has feature flags and registries, but the docs show many historical paths still wired or feature-flagged. The safest architecture is:

- **research producers** generate validated return streams and target weights;
- **promotion artifacts** are immutable and versioned;
- **live runtime** only knows about approved sleeve manifests, not research scripts;
- a dead strategy cannot accidentally wake up because a config flag was stale.

This is especially important because you have had multiple historical harness bugs. Your process caught them, but live runtime should be narrower than research runtime.

### 4.3 Sleeve Lab is the right center of gravity

Do not abandon Sleeve Lab. It is the strongest architectural component in the platform. Every future strategy should be a small declaration producing a return series / target-weight series / overlay multiplier, with the shared harness doing the rest.

The danger is “special path creep.” Options, crypto, futures, and 24/5 trading will tempt you to create bespoke evaluators. Resist that. Add adapters, not parallel systems.

### 4.4 You need an explicit cash sleeve

Right now, “trend at 25%, rest cash” is an implicit allocation. In a 2026 rate environment, cash/T-bill treatment is not a trivial implementation detail. You need a formal **cash/T-bill sleeve** that defines whether idle capital sits in broker cash, SGOV/BIL/SHV, Treasury bills, or something else.

This is not alpha, but it is return. For a system whose only live risky sleeve is 25% notional, idle capital treatment may dominate realized P&L.

### 4.5 Portfolio construction should become more central than signal discovery

You already have enough evidence that signal discovery is hard. The next alpha-equivalent gains will likely come from:

- better sizing of trend;
- better cash treatment;
- risk targeting;
- avoiding unnecessary de-risk false positives;
- robust rebalancing cadence;
- overlay composition;
- avoiding live/backtest drift.

This is unsexy, but it is where a solo system can still win.

---

## 5. Do I buy “free daily US equity data is mined out”?

Yes, with caveats.

I buy that **free daily data on large, currently listed US equities is mined out for your setup**. You are not going to XGBoost your way into a persistent daily-stock-selection edge using price/volume/fundamentals that every platform has. If a signal is visible in yfinance daily bars, liquid enough for Alpaca, cheap to trade, and simple enough to discover with six months of systematic search, it has probably been competed down or is too weak for your gate.

I do **not** buy the broader statement that “US equities have no retail-accessible edge.” The remaining possibilities are just outside your current envelope:

1. **Data edge:** survivorship-free small/mid-cap data, point-in-time fundamentals, estimates/revisions, short/borrow data, corporate actions, ETF flows, insider clusters.
2. **Structural edge:** options risk premia, overnight liquidity provision, tax/calendar effects, event-risk premia.
3. **Execution edge:** market-on-close/overnight/extended-hours behavior, spreads, liquidity provision. This is hard for retail and dangerous without data.
4. **Constraint edge:** things institutions avoid because capacity is too small, operationally annoying, or mandate-constrained.

But the honest point is this: most of those require data you do not have or execution realism you cannot easily model. Therefore, the right conclusion is not “search harder.” It is **change the data / asset class / structure**.

---

## 6. Strategy idea menu — ranked and concrete

Below is a broad menu. I am intentionally separating **high-priority actionable ideas** from **research-heavy / low-priority ideas**. Some ideas are not “alpha” in the pure sense, but they can improve the book.

### Priority 1 — Clean futures trend + carry research, even before live futures trading

**Rationale:** Trend works for you. The most natural extension is not more ETFs; it is more liquid macro breadth and proper futures carry/roll yield. Futures are where time-series momentum and carry are structurally cleaner: equity index, rates, FX, commodities. ETFs are an imperfect proxy and cannot express many carry premia.

**Instruments / universe:** Liquid global futures: equity index, rates, bonds, FX, commodities. Research first; live only later via IBKR or another futures-capable broker.

**Horizon:** Daily to weekly signals; 1–12 month trend; monthly carry/roll.

**Data needed:** Clean individual futures contracts, roll schedules, continuous series, settlement prices. Norgate Futures is the obvious retail-budget candidate.

**Signal construction:**

- Trend: 1/3/6/12-month return signs, vol-targeted per market.
- Carry: roll yield / term structure slope using front vs next contracts, normalized by volatility.
- Combined: `score = z_trend + 0.5*z_carry`, with asset-class risk caps.
- Portfolio: equal risk contribution by asset class, not equal contract count.
- Track-B vs current ETF trend book, then standalone macro book.

**Capacity / turnover / failure modes:** Huge capacity at institutional scale, but operationally more complex. Failure modes: bad roll logic, contaminated continuous contracts, ignoring contract multipliers, crisis correlation in carry, margin/liquidation handling.

**Priority:** **Highest data-buy priority.** This is the first paid-data bet I would make if you are willing to eventually open a futures broker account.

---

### Priority 2 — Crypto spot trend / cross-sectional momentum sleeve

**Rationale:** Crypto remains less institutionally efficient than large-cap US equities, trades 24/7, and Alpaca can execute crypto. Trend and momentum tend to travel across asset classes. You do not need to believe in crypto fundamentals; you need behavioral flow, reflexivity, and volatility clustering.

**Instruments / universe:** BTC, ETH, and the largest/liquid crypto assets available through Alpaca. Do not include illiquid meme coins unless the venue, fees, and spreads are explicitly modeled.

**Horizon:** 1 week to 6 months. Avoid intraday unless you have excellent spread/fee modeling.

**Data needed:** Alpaca crypto history if sufficient; otherwise free exchange OHLCV via public APIs for research. Funding/open interest from perp venues can be used as conditioning data but not as directly tradable carry if Alpaca only supports spot.

**Signal construction:**

- TSMOM: 21/63/126-day returns, vol-targeted.
- Cross-sectional momentum: rank top liquid coins, long top N, flat bottom N unless shorting available and realistic.
- Crash filter: BTC below 200d MA or realized vol spike → reduce gross.
- Optional conditioning: perp funding extreme positive + trend deterioration = de-risk.

**Capacity / turnover / failure modes:** Good capacity for BTC/ETH, worse for smaller coins. Failure modes: exchange-specific data mismatches, weekend gaps, spread/fee drag, regime collapse, crypto-specific tail risk, correlation to NASDAQ in stress.

**Priority:** **High.** It is one of the few new asset classes you can trade now without a new broker.

---

### Priority 3 — Make idle capital productive: cash/T-bill sleeve

**Rationale:** This is not alpha, but it is economically meaningful. If 75% of the book is cash because trend is at 25% allocation, cash treatment materially changes realized return and drawdown.

**Instruments / universe:** SGOV, BIL, SHV, short Treasury ETF, broker sweep cash, or Treasury bills depending on operational simplicity.

**Horizon:** Persistent allocation, with risk-off overrides only for liquidity.

**Data needed:** Free EOD data and current yield estimates.

**Signal construction:**

- Define cash benchmark explicitly.
- Hold T-bill ETF for idle risk budget unless upcoming rebalances require cash.
- Treat it as a sleeve with near-zero risk, not a strategy alpha.
- Model bid/ask and tax implications separately.

**Capacity / turnover / failure modes:** Very high capacity. Main risks: treating duration/credit risk as cash, tax drag, liquidity during extreme stress, broker margin treatment.

**Priority:** **Very high operational priority.** This is the easiest way to improve book economics without pretending to find alpha.

---

### Priority 4 — Trend book portfolio-construction upgrade

**Rationale:** The edge you have is trend. Improve how you harvest it before adding low-conviction satellites. A 10-ETF TSMOM sleeve at 25% allocation is conservative; the biggest performance gain may be sizing and construction, not a new signal.

**Instruments / universe:** Current 10 ETFs plus a small carefully justified ETF expansion: equities, bonds, gold, broad commodities, dollar, international equities, credit. Do not randomly broaden.

**Horizon:** Weekly or monthly rebalance.

**Data needed:** Existing ETF EOD data.

**Signal construction:**

- Compare current fixed 10% vol target to realized portfolio-level target vol.
- Add trend-strength weighting rather than binary long/short.
- Add signal ensemble: 1/3/6/12 month trend, moving-average slope, breakout.
- Use asset-class risk caps.
- Add cash/T-bill carry explicitly.
- Test weekly vs monthly rebalance and no-trade bands.

**Capacity / turnover / failure modes:** Very high capacity. Failure modes: overfitting lookbacks, hidden equity beta, too much bond duration, false precision in vol targeting.

**Priority:** **High.** Do not call this new alpha; call it improving the one real alpha.

---

### Priority 5 — Credit overlay live-shadow and activation decision

**Rationale:** The credit-selective overlay is the one Alpha-v8 winner. It is modest, post-hoc enough to deserve humility, but economically plausible: credit often leads equity stress.

**Instruments / universe:** It scales the existing trend book. Uses HYG/IEF or similar credit/rates proxy.

**Horizon:** Daily signal, applied to weekly or next rebalance exposure.

**Data needed:** Existing macro/ETF cache, with live refresh and fail-safe.

**Signal construction:** Use the pre-registered L120 / 2% band / derisk-to-0.5 configuration, but treat it as a shadow/live-paper overlay before any real scaling.

**Capacity / turnover / failure modes:** Capacity is the trend book’s capacity. Failure modes: false de-risking during recoveries, redundancy with VIX governor, post-hoc parameter selection, dividend-adjustment or timing mismatch.

**Priority:** **High as an overlay, not as alpha.** I would shadow it immediately and activate only with a documented false-positive budget.

---

### Priority 6 — Rates carry fresh confirmation

**Rationale:** Your rates carry result was the strongest additive near-miss: low correlation, positive residual alpha, robust grid behavior, but just missed the formal bars. That is not the same category as dead equity ML. It deserves one clean, pre-registered retest.

**Instruments / universe:** Treasury ETFs: SHY, IEI, IEF, TLT, EDV; possibly TYD/TMF excluded unless leverage is deliberately modeled.

**Horizon:** Monthly signal, slow turnover.

**Data needed:** FRED yield curve, ETF total return EOD.

**Signal construction:**

- Estimate expected roll-down/carry by duration bucket using yield curve slope and ETF duration proxy.
- Long duration when carry/roll is favorable and trend is not strongly adverse.
- Flat or short duration when curve/carry adverse.
- Penalize equity-crisis covariance and inflation-shock regimes.

**Capacity / turnover / failure modes:** High capacity. Failure modes: 2022-style inflation shock, overlapping with trend’s bond sleeve, ETF duration drift, FRED/yfinance calendar alignment.

**Priority:** **High-medium.** One retest only. Do not grid-search it to death.

---

### Priority 7 — Defined-risk index option premium sleeve, live-paper first

**Rationale:** Options alpha is more likely in **structure and risk transfer** than in using options data as features for stock signals. Retail can access SPY/QQQ/IWM options. But historical options data is short and expensive, so backtest-only confidence should be low.

**Instruments / universe:** SPY, QQQ, IWM options. Avoid single-name options initially.

**Horizon:** 7–45 DTE structures.

**Data needed:** Existing frozen Polygon options history for exploratory backtests; Alpaca live options NBBO snapshot for live-paper; ideally live greeks/IV/OI if available later.

**Signal construction ideas:**

- Put credit spreads only when trend positive, VIX term structure not inverted, credit overlay not stressed.
- Iron condors only in low realized-vol / falling-vol regimes, but with strict tail stop.
- Put-spread replacement for equity exposure: use defined-risk call spreads instead of ETF long in high-vol regimes.
- Tail hedge: buy cheap put spreads when trend breadth collapses but VIX has not yet inverted.

**Capacity / turnover / failure modes:** Capacity in index options is excellent. Failure modes: tail loss, spread fill optimism, early assignment, expiration pin risk, IV surface data quality, paper fills not representative.

**Priority:** **Medium-high, but only as a live-paper experimental track.** Do not promote from 4y historical backtest alone.

---

### Priority 8 — Overnight / 24/5 ETF risk-transfer strategies

**Rationale:** Alpaca now supports 24/5 trading for eligible US NMS securities. Overnight markets have lower liquidity and wider spreads, which is a risk, but the very existence of constrained liquidity may create small structural premia in broad ETFs.

**Instruments / universe:** SPY, QQQ, IWM, TLT, GLD, maybe sector ETFs. Avoid single names initially.

**Horizon:** Overnight / premarket / postmarket to regular session open/close.

**Data needed:** Alpaca extended-hours data and execution logs. Historical extended-hours data may be limited; this may need forward collection.

**Signal construction:**

- News-free overnight mean reversion after large after-hours ETF move.
- Follow-through after macro events only when spread/liquidity conditions pass.
- Limit-order-only liquidity provision at conservative prices.
- Do not trade when quoted spread exceeds a pre-set threshold.

**Capacity / turnover / failure modes:** Small capacity; execution-sensitive. Failure modes: wide spreads, stale quotes, corporate action eligibility changes, overnight liquidity disappearance.

**Priority:** **Medium.** This is a forward-data collection project, not a backtest-first project.

---

### Priority 9 — ETF volume-shock reversal / continuation

**Rationale:** Liquid ETFs sometimes overreact to forced flows, hedging, or macro shocks. You failed slow ETF relative value, but an event-style ETF shock strategy is a different object.

**Instruments / universe:** Broad ETFs and sector ETFs only: SPY, QQQ, IWM, XLF, XLK, XLE, XLU, XLP, TLT, HYG, LQD, GLD, UUP, USO/DBA/DBC if liquidity and structure acceptable.

**Horizon:** 1–10 trading days.

**Data needed:** Free ETF OHLCV; adjusted close; volume history.

**Signal construction:**

- Identify 3–5 sigma price move plus 3–5 sigma volume shock.
- Classify by trend regime: continuation in strong trend, reversal in range-bound regime.
- Require cross-ETF confirmation: e.g., XLE shock vs oil ETF, HYG shock vs SPY.
- Use tight risk budget and no intraday assumptions.

**Capacity / turnover / failure modes:** Good ETF capacity, moderate turnover. Failure modes: catching falling knives, crisis-period correlation, overfitting event definitions.

**Priority:** **Medium.** One pre-registered design only.

---

### Priority 10 — Macro nowcast overlay using public data only

**Rationale:** Macro variables are too slow for stock selection but can help risk-budgeting. The goal is not to predict SPY daily returns; it is to avoid allocating the same risk in very different macro regimes.

**Instruments / universe:** Applies to trend book and bond/rates sleeves.

**Horizon:** Monthly.

**Data needed:** FRED: unemployment, claims, CPI, industrial production, credit spreads, yield curve, financial conditions, dollar, commodities. Use release dates / vintages where possible.

**Signal construction:**

- Build 3-state macro regime: growth up/down, inflation up/down, financial stress up/down.
- Use only data knowable after release lag.
- Map regimes to risk caps, not directional trades.
- Example: inflation shock + rising yields → cap bond duration trend exposure.

**Capacity / turnover / failure modes:** Very high capacity. Failure modes: vintage/release-date leakage, overfitted regime maps, false de-risking.

**Priority:** **Medium.** Useful for risk, not likely standalone alpha.

---

### Priority 11 — CFTC COT positioning overlay for futures research

**Rationale:** Positioning extremes can condition trend/carry. It is free and economically meaningful. It is most useful once you have futures data, but can be explored as a macro ETF overlay.

**Instruments / universe:** Futures markets or ETF proxies: currencies, commodities, equity index, rates.

**Horizon:** Weekly to monthly.

**Data needed:** CFTC Commitment of Traders data, futures/ETF prices.

**Signal construction:**

- Commercial vs non-commercial positioning z-scores.
- Fade crowded spec positioning only when trend weakens.
- Use as de-risk overlay, not standalone short signal.

**Capacity / turnover / failure modes:** High capacity. Failure modes: reporting lag, structural changes in participant categories, crowding can persist.

**Priority:** **Medium, especially after Norgate futures.**

---

### Priority 12 — Index reconstitution / corporate-action event basket

**Rationale:** Index additions/deletions and corporate actions create forced-flow effects. However, many are crowded, require clean event calendars, and capacity is limited.

**Instruments / universe:** Large-cap and mid-cap equities around S&P/Russell reconstitution; ETFs for broad flow.

**Horizon:** Days to weeks around event dates.

**Data needed:** Historical index constituent changes, announcement/effective dates, survivorship-free prices. Norgate US stocks or another constituent-history source would help.

**Signal construction:**

- Pre-register one event family only, e.g., Russell reconstitution addition/deletion effect.
- Use announcement date and effective date separately.
- Test liquidity, spread, and gap behavior.

**Capacity / turnover / failure modes:** Moderate to low. Crowded; high implementation shortfall. Failure modes: date leakage, event calendar survivorship, execution at bad prices.

**Priority:** **Low-medium unless you buy survivorship-free constituent data.**

---

### Priority 13 — Tax-loss / January effect basket

**Rationale:** Seasonal flows around year-end tax-loss selling can create rebound in beaten-down names. This is one of the few seasonal equity effects with plausible behavioral and institutional flow rationale.

**Instruments / universe:** Liquid small/mid-cap US stocks, maybe sector ETFs if single-name data unavailable.

**Horizon:** December to January / early February.

**Data needed:** Survivorship-free equity data with delistings; corporate actions; liquidity filters.

**Signal construction:**

- Rank stocks by YTD underperformance through mid/late December.
- Exclude distress, low price, pending delisting, extreme illiquidity.
- Buy basket late December, exit January strength.
- Market/sector neutral variant if borrow data available.

**Capacity / turnover / failure modes:** Limited capacity, high event concentration. Failure modes: losers keep losing, delisting bias, tax regime changes, January effect crowding.

**Priority:** **Low-medium.** Not worth doing without survivorship-free data.

---

### Priority 14 — Insider cluster / buyback-event drift

**Rationale:** Insider purchases and buyback announcements can signal undervaluation or management confidence. Effects are slow and noisy but can be more structural than price-only signals.

**Instruments / universe:** Liquid US equities.

**Horizon:** 1–6 months.

**Data needed:** Form 4 insider data, buyback announcements, clean event timestamps, survivorship-free prices. FMP may help but must be audited.

**Signal construction:**

- Use only open-market insider buys, cluster by multiple insiders, exclude option grants/sales.
- Combine with quality/liquidity filters.
- Hold diversified basket with market/sector neutralization.

**Capacity / turnover / failure modes:** Moderate capacity. Failure modes: data timestamp issues, sparse events, many false positives, survivorship bias.

**Priority:** **Low-medium.** Better than generic analyst drift, but needs clean data.

---

### Priority 15 — Analyst revision diffusion, only if you buy proper estimates data

**Rationale:** Estimate revisions can diffuse slowly through prices. This is a known institutional factor, but retail-grade versions are usually stale or incomplete.

**Instruments / universe:** Liquid US equities.

**Horizon:** 1–3 months.

**Data needed:** Point-in-time analyst estimates/revisions. This is usually not cheap at good quality.

**Signal construction:**

- Revision breadth and magnitude, sector-neutral.
- Combine with earnings surprise and post-event drift.
- Penalize crowded/high-valuation names.

**Capacity / turnover / failure modes:** Moderate. Failure modes: timestamp leakage, vendor revisions backfilled, crowding, overlap with momentum/quality.

**Priority:** **Low unless you find affordable PIT estimates.**

---

### Priority 16 — ETF “carry” proxies outside rates

**Rationale:** Carry exists beyond rates: credit spread carry, currency interest differentials, commodity roll yield. ETFs are imperfect but can sometimes proxy the exposure.

**Instruments / universe:** HYG/LQD/IEF, UUP/FXE/FXY/FXA, DBC/commodity ETFs, GLD, TLT.

**Horizon:** Monthly.

**Data needed:** ETF total returns, FRED yields, FX rates, commodity futures curves if available.

**Signal construction:**

- Credit carry: HYG yield spread proxy minus duration hedge, with stress cap.
- FX carry: long high-yield currency ETF proxies vs low-yield, if ETFs are liquid enough.
- Commodity carry: do not use ETF returns as clean roll yield; needs futures curves.

**Capacity / turnover / failure modes:** ETF capacity good; signal quality uneven. Failure modes: proxies do not equal carry, crisis co-crash, ETF structural decay.

**Priority:** **Medium-low.** Futures data is the better way.

---

### Priority 17 — Trend-following with drawdown-sensitive convex sizing

**Rationale:** Trend has positive skew in crises but bleeds in whipsaw. Sizing rules can improve utility even if they do not improve raw Sharpe.

**Instruments / universe:** Current trend ETF book.

**Horizon:** Weekly/monthly.

**Data needed:** Existing ETF data.

**Signal construction:**

- Reduce risk after realized whipsaw / trend dispersion collapse.
- Increase risk when trend breadth is broad and realized vol stable.
- Do not use recent P&L alone; that creates pro-cyclical de-risking.
- Pre-register utility metric: drawdown-adjusted return, not just Sharpe.

**Capacity / turnover / failure modes:** High. Failure modes: overfitting risk controls, reducing exposure before the trend payoff.

**Priority:** **Medium.** Good for book utility, not standalone alpha.

---

### Priority 18 — Cross-asset “safe-haven rotation” overlay

**Rationale:** During equity stress, the best defensive asset changes: bonds, gold, dollar, cash. Trend handles some of this, but a slow safe-haven selector may improve crisis behavior.

**Instruments / universe:** TLT/IEF/SHY, GLD, UUP, SGOV, maybe defensive equity sectors.

**Horizon:** Weekly/monthly.

**Data needed:** Existing ETF data.

**Signal construction:**

- When equity trend/stress overlay de-risks, allocate some reduced risk to the strongest safe-haven asset by trend/carry.
- Do not short equities simultaneously unless already in trend book.
- Evaluate against pure cash de-risk baseline.

**Capacity / turnover / failure modes:** High. Failure modes: 2022-style bond/equity co-crash, gold/dollar reversals, hidden duration risk.

**Priority:** **Medium-low.** Potentially useful, but avoid creating a second trend book with a new name.

---

### Priority 19 — Broad equity volatility targeting / volatility switching

**Rationale:** Volatility targeting has a mechanical risk-control benefit and can sometimes improve geometric returns. It is not alpha, but it affects realized book behavior.

**Instruments / universe:** Applies to trend book and cash allocation.

**Horizon:** Daily/weekly risk estimate.

**Data needed:** Existing returns.

**Signal construction:**

- Target total book volatility rather than sleeve notional.
- Use slow EWMA vol with caps/floors.
- Add no-trade bands to avoid turnover.
- Evaluate with transaction costs and drawdown.

**Capacity / turnover / failure modes:** High. Failure modes: selling after volatility spike, missing rebound, false comfort from backward-looking vol.

**Priority:** **Medium.** Important for live sizing.

---

### Priority 20 — Sector ETF momentum / rotation

**Rationale:** Sector momentum is more persistent than many single-name signals and avoids survivorship issues. But it overlaps heavily with broad equity trend.

**Instruments / universe:** SPDR sector ETFs and industry ETFs.

**Horizon:** 1–6 months.

**Data needed:** Free ETF EOD.

**Signal construction:**

- Long top sector ETFs by 3/6/12-month momentum, flat or underweight bottom.
- Beta-neutralize to SPY if possible.
- Track-B vs trend book to avoid just adding equity beta.

**Capacity / turnover / failure modes:** High capacity. Failure modes: high correlation to SPY/trend, crowding, sector reclassification.

**Priority:** **Medium-low.** Worth one clean test if not already covered by trend ETF universe.

---

### Priority 21 — Low-vol / quality ETF premia using ETFs, not stocks

**Rationale:** Single-name factor data is biased without survivorship-free history. Factor ETFs provide tradable proxies with clean histories, though short histories and fees are limitations.

**Instruments / universe:** USMV, SPLV, QUAL, MTUM, VLUE, SIZE, RPV, etc.

**Horizon:** Monthly/quarterly.

**Data needed:** ETF EOD, inception-aware.

**Signal construction:**

- Relative momentum/carry among factor ETFs.
- Defensive factor rotation: low-vol/quality during stress, momentum/value during expansion.
- Compare to SPY and trend book after fees.

**Capacity / turnover / failure modes:** High ETF capacity. Failure modes: short histories, factor crowding, ETF construction changes, correlation to equity beta.

**Priority:** **Low-medium.** Good as allocation research, unlikely to be standalone alpha.

---

### Priority 22 — Market breadth thrust / deterioration overlay

**Rationale:** Breadth deterioration can precede equity drawdowns; breadth thrust can confirm rebounds. This is a classic timing overlay, not a stock-selection edge.

**Instruments / universe:** SPY / trend book exposure.

**Horizon:** Days to weeks.

**Data needed:** Survivorship-free index constituent breadth ideally; without it, use ETF proxies or current universe with caution.

**Signal construction:**

- Percentage of stocks above 50d/200d MA.
- Advance/decline proxies if available.
- Use expanding PIT constituent membership; otherwise mark exploratory only.
- De-risk when price index holds up but breadth collapses.

**Capacity / turnover / failure modes:** High capacity. Failure modes: survivorship bias, false warnings, overlap with price trend and credit overlay.

**Priority:** **Low-medium unless you have clean constituent data.**

---

### Priority 23 — Short-borrow / hard-to-borrow risk avoidance

**Rationale:** Borrow constraints and shorting demand can contain information. But it is more reliable as a risk-control input than as an alpha source.

**Instruments / universe:** Single-name equities, if you revive long/short strategies.

**Horizon:** Daily to weekly.

**Data needed:** Alpaca easy-to-borrow live flags; historical borrow data is likely unavailable or expensive.

**Signal construction:**

- Do not short names that are not easy-to-borrow or have deteriorating borrow availability.
- Use live-only tracker to collect borrow-state transitions and subsequent returns.
- Potential squeeze-avoidance overlay, not return signal.

**Capacity / turnover / failure modes:** Depends on single-name strategy. Failure modes: no historical depth, live-only slow learning, borrow changes after signal.

**Priority:** **Low unless you revive shorting.**

---

### Priority 24 — Single-name post-event gap fade / continuation, only with survivorship-free data

**Rationale:** Large gaps after earnings, guidance, legal/regulatory events, or index news can under- or overreact. Your PEAD failed, but gap behavior can be conditional and shorter horizon.

**Instruments / universe:** Liquid US equities.

**Horizon:** 1–10 days.

**Data needed:** Clean event timestamps, survivorship-free EOD/intraday data, split-adjusted gaps.

**Signal construction:**

- Classify event gap by size, volume, pre-event trend, and market regime.
- Continuation for surprise + volume + strong breadth; fade for non-news liquidity gaps.
- Use event-level clustered inference.

**Capacity / turnover / failure modes:** Moderate. Failure modes: event timestamp errors, spread/gap fills, delisting bias, too many filters.

**Priority:** **Low until data improves.**

---

### Priority 25 — Intraday microstructure strategies

**Rationale:** Intraday edges exist, but retail-grade 5-minute bars and paper fills are not enough. You already learned this.

**Instruments / universe:** Highly liquid ETFs only if revisited.

**Horizon:** Minutes to hours.

**Data needed:** Tick / quote / order book / auction imbalance data; realistic fill model.

**Signal construction:**

- Opening range liquidity imbalance.
- VWAP reversion.
- Market-on-close imbalance response.
- ETF/underlying lead-lag.

**Capacity / turnover / failure modes:** Execution dominates. Failure modes: spread/queue position, latency, false fills, insufficient data depth.

**Priority:** **Very low under current constraints.** Do not go back here without better data and live execution validation.

---

### Priority 26 — News / LLM sentiment

**Rationale:** Language signals can work in institutional setups with timestamped feeds and event taxonomies. Retail news APIs are delayed, noisy, and survivorship-prone.

**Instruments / universe:** Single names or sectors.

**Horizon:** Hours to days.

**Data needed:** Timestamped news with point-in-time availability, entity linking, historical archive. Free sources are usually not sufficient.

**Signal construction:**

- Event classifier: guidance raise/cut, regulatory approval, management change, litigation, M&A.
- Trade only high-confidence structured events.
- Use event-panel clustered inference.

**Capacity / turnover / failure modes:** Moderate but data-limited. Failure modes: timestamp leakage, duplicated news, low precision, already priced by the time retail API sees it.

**Priority:** **Low.** Good software project, weak alpha project unless you buy data.

---

### Priority 27 — Small-cap liquidity premium / reversal

**Rationale:** Small-cap effects may remain because capacity is limited and trading costs are real. But they require clean survivorship-free data and realistic execution.

**Instruments / universe:** Liquid small-cap stocks; perhaps IWM constituents.

**Horizon:** 1 week to 3 months.

**Data needed:** Survivorship-free prices, delistings, corporate actions, liquidity history.

**Signal construction:**

- Monthly reversal after extreme underperformance, filtered by liquidity and distress.
- Momentum with skip-month, sector-neutral.
- Avoid penny/biotech/event-landmine names.

**Capacity / turnover / failure modes:** Limited capacity; costs matter. Failure modes: delisting losses, liquidity gaps, borrow constraints, data bias.

**Priority:** **Low unless you buy survivorship-free equity data.**

---

### Priority 28 — Pairs / stat-arb revival with proper universe, not ETFs

**Rationale:** ETF relative value failed because the obvious ETF pairs are efficient and low spread. Single-name stat-arb can work, but it is data/execution-heavy.

**Instruments / universe:** Large liquid pairs within sectors.

**Horizon:** Days to weeks.

**Data needed:** Survivorship-free prices, borrow, corporate actions, ideally intraday quotes.

**Signal construction:**

- Sector-neutral residual mean reversion from robust PCA/factor model.
- Trade residual z-score with stop and catalyst exclusions.
- Strict cost and borrow modeling.

**Capacity / turnover / failure modes:** Medium-low capacity. Failure modes: crowded, regime breaks, earnings gaps, borrow, hidden factor exposure.

**Priority:** **Very low under current constraints.** This is not a solo/free-data edge.

---

### Priority 29 — Volatility ETP strategies

**Rationale:** VIX futures term structure has persistent risk premia, but retail products are dangerous and structurally decaying. This can be profitable, but the left tail is brutal.

**Instruments / universe:** VIXY, UVXY, SVIX/SVXY, VIX options/futures if available elsewhere.

**Horizon:** Days to weeks.

**Data needed:** VIX futures curves ideally; ETP history; product methodology changes.

**Signal construction:**

- Short vol only in contango, trend-positive, low stress, with hard tail budget.
- Or use VIX ETPs only as a hedge trigger, not a return sleeve.

**Capacity / turnover / failure modes:** Good capacity, catastrophic tail. Failure modes: volmageddon, product termination, borrow/locate, gap risk.

**Priority:** **Very low / skip for now.** If tested, cap at tiny risk and treat as hazardous.

---

### Priority 30 — Synthetic “all-weather” ETF premia book

**Rationale:** Instead of hunting alpha, assemble known premia: trend, carry proxy, defensive cash, gold, bonds, equity beta, credit, and volatility control. The edge is disciplined allocation.

**Instruments / universe:** SPY/QQQ/IWM, EFA/EEM, TLT/IEF/SHY/SGOV, GLD, DBC, UUP, HYG/LQD.

**Horizon:** Monthly.

**Data needed:** Free ETF EOD.

**Signal construction:**

- Equal-risk allocation to sleeves, not assets.
- Trend overlay per asset.
- Carry proxy where clean.
- Vol target total book.
- Stress overlays reduce risky sleeve exposure, not all exposure.

**Capacity / turnover / failure modes:** High. Failure modes: mostly repackaged beta, overfitting allocation rules, 2022 multi-asset selloff.

**Priority:** **Medium as a portfolio redesign, not alpha discovery.**

---

## 7. Data gaps and top buys

### 7.1 Single highest-ROI data buy

If you are willing to eventually add a futures-capable broker, the single highest-ROI buy is:

> **Norgate Futures data first; IBKR only if the research earns it.**

Why: it directly extends the only proven edge, trend, into the asset class where trend and carry are structurally strongest. It also lets you test the missing carry/roll-yield premium properly, which free yfinance continuous futures cannot do.

Current public Norgate pages list the Futures package at **USD 148.50 for 6 months or USD 270 for 12 months**, covering around 100 futures markets, with 30+ year individual contract daily history, end-of-day updates, and pre-built continuous contracts. That is exactly in your stated budget. The important point is not the exact price; it is that this is a small enough spend to justify one clean research phase.

### 7.2 Second-highest ROI, if you insist on equity research

If you want to keep doing single-name equity/event research, you need survivorship-free equity data. Norgate’s US Stocks Platinum package is explicitly positioned as backtesting-suitable and includes delisted securities and historical index constituents. I did not rely on a dynamic price from the page, but the feature set is the relevant point: this is the type of data required before any serious long/short or event study.

However, I would not buy survivorship-free US equity data unless you have a specific research plan that needs it. Buying it just to rerun every dead ML idea is probably low ROI.

Best equity-data use cases if purchased:

- small-cap / tax-loss / January effects;
- Russell reconstitution;
- delisting-aware momentum/reversal;
- clean event panels;
- breadth overlays with real historical constituents.

### 7.3 Do not buy expensive options surface data yet

Your options history is frozen and incomplete, but that does not automatically mean “buy options data.” Good historical options surface data with NBBO, IV, greeks, OI, and corporate action handling can get expensive quickly. Worse, even good data will not solve the biggest problem: options structures are tail-sensitive and require execution validation.

Recommended approach:

1. Use existing frozen data for exploratory sanity only.
2. Start a live-paper options structure tracker.
3. Promote nothing from historical options backtests alone.
4. Buy better options data only after live-paper suggests a structure is worth deeper research.

### 7.4 Free / cheap data worth adding before another paid equity source

Add these before you spend more on US stock alpha:

- **CFTC COT:** weekly positioning for futures/macro overlays.
- **Treasury / FRED:** already partly used; improve release-date discipline.
- **Crypto spot OHLCV:** exchange public APIs for broader history than Alpaca if needed.
- **Perp funding / open interest:** not directly tradable through Alpaca, but useful as crypto conditioning data.
- **SEC EDGAR structured events:** insider buys, buybacks, offerings, 8-K event categories, but only if you build a clean event timestamp layer.
- **Alpaca extended-hours / 24/5 execution logs:** collect your own forward dataset.

### 7.5 Data I would avoid for now

- Retail news sentiment feeds unless you can verify timestamp quality.
- Cheap fundamentals without point-in-time semantics for confirmatory ML.
- Options greeks computed from bad marks and then treated as truth.
- Intraday minute/tick data unless you have a realistic execution simulator.
- Alternative data subscriptions that do not map to a pre-registered, tradable strategy.

### 7.6 Broker / execution data gap

Alpaca is fine for equities/ETFs/options/crypto, but it is not enough if the next frontier is futures. Do not open IBKR just because futures sound attractive. Do it only after futures research clears a pre-registered gate.

The decision tree should be:

1. Buy Norgate Futures data.
2. Test futures trend + carry in Sleeve Lab.
3. If it passes, build minimal IBKR adapter for futures.
4. Paper / tiny-live futures execution validation.
5. Only then allocate.

---

## 8. Modeling recommendations

### 8.1 Stop defaulting to XGBoost for daily equity alpha

Tree ensembles are not magic. In low-signal, non-stationary financial data, they can become extremely good at learning microstructure of the training set and extremely bad at producing stable edge.

For daily equity cross-sectional work, your default should be:

- linear / ridge / elastic net;
- monotonic transformations of economic features;
- rank-based signals;
- hierarchical Bayesian shrinkage;
- simple ensembles of pre-registered signals;
- then maybe GBT as a challenger, not the champion.

If a signal does not survive a linear model or a simple rank rule, I would be skeptical that XGBoost found durable economics.

### 8.2 Model expected return and uncertainty, not classification accuracy

Your historical labels include top-quintile classification, path quality, and triple-barrier variants. These can be useful, but they often optimize the wrong thing.

A trading model needs:

- expected return net of costs;
- uncertainty around that expectation;
- expected drawdown / tail contribution;
- turnover and capacity;
- correlation to existing book.

For most sleeves, a continuous return target with robust loss and cost-aware ranking is better than binary classification. A classifier with AUC 0.60 can still be useless if the positive class does not map to enough net return after costs.

### 8.3 Use Bayesian shrinkage for sleeve allocation

For a small book, estimate error dominates. Do not let point Sharpe drive allocation. Use a Bayesian posterior over sleeve Sharpe / covariance and size by conservative posterior mean or expected utility.

A practical rule:

- start with prior SR = 0 for all new sleeves;
- shrink heavily by number of tried hypotheses;
- cap initial risk budget at 5–10%;
- increase only after live-paper / live evidence;
- never allocate purely because backtest point SR is high.

### 8.4 Use regime models only for sizing, not signal discovery

Regime classifiers are seductive. Most become hidden overfit. Use regimes to modulate risk caps, not to flip signals aggressively.

Good uses:

- cap duration exposure in inflation shock regime;
- reduce short-vol exposure in stress regime;
- choose cash vs bond safe haven;
- interpret strategy diagnostics.

Bad uses:

- train totally separate models per regime with small samples;
- grid-search regime thresholds;
- use future-known macro revisions;
- allow regime model to rescue a dead signal.

### 8.5 Add “negative controls” to every new data source

You already have a gate-calibration mindset. Extend it to data:

- shuffled signal dates;
- lagged/unlagged signal comparison;
- impossible signal test that should fail;
- random universe control;
- transaction-cost stress;
- inverse signal sanity check.

If the unlagged version is spectacular and the properly lagged version is marginal, you probably found timing leakage, not alpha.

### 8.6 For options, model payoff distributions directly

Do not use options features to predict stock direction. That path already failed and is economically weak. If you revisit options, model the option structure’s payoff distribution:

- expected premium capture;
- tail loss conditional on gap;
- IV vs realized vol;
- skew regime;
- assignment/expiry behavior;
- spread fill model;
- margin and max loss.

ML, if used, should predict **when not to sell premium**, not which single-name option has alpha.

### 8.7 Use conformal / quantile methods for risk bounds

For small samples, a point forecast is less useful than a calibrated prediction interval. Conformal or quantile regression methods can help define position size and stop conditions based on uncertainty.

Applications:

- trend sleeve volatility forecast interval;
- crypto position sizing;
- options structure loss distribution;
- event-strategy expected P&L distribution.

### 8.8 Add live “backtest-to-paper tracking error” metrics as first-class gates

You already track some live behavior. Make it formal:

- expected vs actual fill price;
- intended vs actual exposure;
- missed trades;
- stale data events;
- overlay multiplier applied vs expected;
- daily return attribution by sleeve;
- live-paper returns vs simulated returns using same day’s official close.

A strategy should not graduate to live capital merely because research passed. It should also demonstrate that live implementation matches the research assumptions.

---

## 9. Redesign I would build from scratch

If I were rebuilding MrTrader under your constraints, I would not build a stock-picking ML platform. I would build a **small systematic premia operating system**.

### 9.1 Core design

**Layer 1 — Data lake**

- Immutable raw data snapshots.
- Point-in-time calendars and release timestamps.
- Data version hashes in every research result.
- Survivorship-free datasets only for single-name confirmatory work.

**Layer 2 — Sleeve declarations**

Every strategy is one of:

- return sleeve;
- overlay multiplier;
- execution strategy;
- cash sleeve;
- hedge sleeve.

Each sleeve declares:

- economic rationale;
- data dependencies;
- signal timestamp convention;
- rebalance schedule;
- turnover/cost model;
- expected failure modes;
- acceptance criteria.

**Layer 3 — Research harness**

- One Sleeve Lab.
- One event-panel inference path.
- One options-structure simulator path, but adapter-compatible.
- No bespoke run scripts except wrappers.

**Layer 4 — Portfolio allocator**

- Risk budgets, not capital budgets.
- Conservative Bayesian shrinkage.
- Total book vol target.
- Tail/correlation constraints.
- Explicit cash/T-bill sleeve.

**Layer 5 — Live executor**

- Deterministic target positions.
- Idempotent orders.
- Broker reconciliation.
- Shadow mode.
- Tiny-live validation mode.
- Full audit ledger.

### 9.2 Target book under current constraints

Initial target book:

1. **Trend ETF sleeve** — core, risk-targeted, 40–70% of risky budget depending on confidence.
2. **Cash/T-bill sleeve** — all unused capital, explicit and benchmarked.
3. **VIX-term governor** — live overlay.
4. **Credit governor** — shadow then owner decision.
5. **Rates carry** — one retest; tiny allocation only if posterior supports it.
6. **Crypto trend** — new Alpaca-tradable satellite.
7. **Options defined-risk premium** — live-paper only until execution is proven.
8. **Futures trend/carry** — research after Norgate; live only after IBKR adapter.

That is the whole platform. Everything else is archived until data changes.

### 9.3 Research budget allocation

I would allocate your next 100 research hours roughly as:

- 25 hours: live reliability / audit / cash sleeve / backtest-to-paper tracking;
- 25 hours: Norgate futures trend+carry proof-of-concept;
- 20 hours: crypto trend/momentum sleeve;
- 15 hours: trend portfolio construction upgrade;
- 10 hours: rates carry retest;
- 5 hours: options live-paper tracker design.

I would allocate **zero** hours to new XGBoost equity ranking, 5-minute intraday ML, or PEAD variants.

---

## 10. Concrete next 5 actions

### 1. Freeze the dead equity ML / intraday / PEAD paths as research archives

Do not let them retrain. Do not let them consume attention. Keep the code only as historical evidence and infrastructure reference. The active platform should not look like it is waiting for those selectors to come back.

**Definition of done:** no scheduled retraining; no live config path can accidentally activate them; docs say “archived until new data source.”

### 2. Promote cash/T-bill handling to a first-class sleeve

This is the most immediately monetizable improvement. Define idle capital policy and benchmark the entire book against trend + T-bills, not trend + zero-yield cash.

**Definition of done:** a `cash_sleeve` / `tbill_sleeve` with return series, live target policy, and reporting.

### 3. Shadow the credit-selective overlay and decide with a false-positive budget

You already did the research. Now make an owner decision. Do not keep it in limbo forever.

**Definition of done:** 4–8 weeks of shadow/live-paper audit, daily multiplier log, comparison to VIX-only governor, decision memo: enable, park, or kill.

### 4. Buy or trial Norgate Futures data and run one futures trend+carry research program

This is the highest-ROI way to answer whether the platform can move beyond ETF trend. Do not start by building an IBKR adapter. Start with research.

**Definition of done:** one pre-registered futures research packet: trend-only, carry-only, trend+carry, asset-class risk caps, Track-B vs ETF trend, crisis analysis, no live trading decision until complete.

### 5. Build Alpaca-tradable crypto trend as the next new sleeve

This is the cleanest new tradable asset class inside your current broker. Keep it simple and brutal.

**Definition of done:** BTC/ETH/top-liquid crypto spot TSMOM / cross-sectional momentum, daily/weekly rebalance, conservative fees/spread model, crash filter, Track-A/Track-B, shadow execution.

---

## 11. What not to do next

Do not do these unless a new data source or execution capability changes the premise:

- Do not run another daily US equity XGBoost feature sweep.
- Do not revive intraday ML with 5-minute bars.
- Do not re-open PEAD with threshold filters.
- Do not buy expensive options data before a live-paper structure shows promise.
- Do not optimize the credit overlay parameters again.
- Do not add new gates unless a specific failure mode requires it.
- Do not build an IBKR futures execution adapter before futures research passes.
- Do not let the system’s complexity imply that more sleeves must exist.

---

## 12. Final hard truth

Your platform has probably already extracted the biggest lesson available from six months of honest research: **the edge is not in your model zoo; it is in the intersection of real economic premia, enough breadth/history, and executable instruments.** Trend qualifies. Most free-data equity ideas do not.

That is not failure. It is exactly what a good research process is supposed to reveal. The danger now is refusing to accept the map because it shows a small island.

The correct posture is:

1. harvest trend properly;
2. make cash productive;
3. add only overlays that improve the left tail;
4. buy one dataset that opens a genuinely new premia family;
5. test crypto because you can actually trade it;
6. require live execution evidence for anything options/overnight/intraday;
7. stop mining the same exhausted free equity bars.

If, after Norgate futures and crypto, nothing passes beyond trend, the honest answer is: **run trend well, size it rationally, collect T-bill yield on idle capital, and stop pretending a solo free-data platform should behave like a diversified systematic hedge fund.** A modest, robust, low-maintenance trend-plus-defensive-overlay book is a much better outcome than a complex graveyard of false positives.

---

## 13. External source notes checked while preparing this review

These are not the basis of the quant conclusions; they were checked only for current data/execution constraints.

- Norgate Futures package: public page lists around 100 futures markets and subscription pricing of USD 148.50 for 6 months / USD 270 for 12 months. Source: https://norgatedata.com/futurespackage.php
- Norgate subscription page: US Stocks Platinum includes daily price history back to 1990, delisted securities, current fundamentals, historical index constituents, and “suitable for backtesting.” Source: https://norgatedata.com/subscribe/subscribe.php
- Alpaca public site: markets its API around stocks, ETFs, options, and crypto. Source: https://alpaca.markets/
- Alpaca 24/5 article: says eligible US NMS securities can trade nearly around the clock, with overnight session 8:00 PM–4:00 AM ET, limit orders only, and warns liquidity can be lower and spreads wider outside regular hours. Source: https://alpaca.markets/learn/how-to-trade-us-stocks-24_5-overnight-with-python-and-alpaca

