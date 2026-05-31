# MrTrader Strategy Review — Independent Quant Assessment

**Prepared for:** Min / MrTrader project  
**Review date:** 2026-05-19  
**Input reviewed:** `LLM_STRATEGY_REVIEW_PROMPT.md`  
**Role assumed:** skeptical senior quant / systematic trading reviewer

---

## 1. Executive conclusion

My blunt assessment: **do not treat the current walk-forward results as evidence that the strategy is production-ready.** PEAD may be a real anomaly, and a quality-deterioration short screen may have some signal value, but the reported MrTrader results are not merely “too good.” They are **internally inconsistent with the described sizing, trade counts, max drawdowns, and mechanics.**

The main issue is not that PEAD Sharpe 8.1 is hard to believe. The stronger issue is this:

- PEAD reports **11x to 38x annual fold returns** with roughly **300 trades/year**, roughly **5% max position size**, and max drawdown around **2%**.
- QualityShort reports **3x to 17x annual fold returns** with only **29–61 trades/year**, win rate as low as **21%**, and max drawdown as low as **0.01%**.
- With the stated 5% position cap, these return figures imply average per-trade returns that are economically absurd for Russell 1000 daily equity trades.
- Therefore, I would prioritize **simulator/equity-curve accounting audit** over further factor research.

My estimated diagnosis, before seeing the actual code:

| Suspected explanation | Probability |
|---|---:|
| Equity curve / P&L / return calculation bug or double counting | 45–60% |
| Position sizing or cash reservation bug allowing effective hidden leverage | 20–30% |
| Look-ahead / timestamp / data availability leakage | 15–25% |
| Survivorship and multiple-testing inflation, but no core bug | 10–20% |
| Results are broadly real at paper scale | <5% |

The most important conclusion: **Phase I should not be “paper trading the winning PEAD + QualityShort combo.” Phase I should be “forensic validation of the backtest engine and data availability model.”**

---

## 2. Core red flags

### 2.1 PEAD total returns are mechanically inconsistent with described sizing

The prompt states that PEAD uses a 5% max position cap, about 300 trades/year, next-day open entry, 5-day hold, and reported annual fold returns of 11.90x to 38.61x.

A quick sanity check:

If each trade is sized at 5% of equity and the fold returns 38x in 300 trades, then the average per-trade notional return required is approximately:

```text
(1 + 0.05 * r)^300 = 38
r ≈ 24% per trade
```

That means the average PEAD trade would need to earn roughly **24% over five trading days** on the underlying position, not just on the allocated capital. That is not plausible for a Russell 1000 post-earnings drift strategy entered after the announcement gap. Even if some earnings names move sharply, the *average* winning signal cannot look like that across ~300 trades/year without being one of the most valuable short-horizon equity anomalies ever discovered.

A more plausible PEAD implementation might earn something like 0.5%–2.0% average gross return on notional over several days before costs if the signal is real and well-selected. At a 5% position size and 300 trades/year:

```text
300 trades * 5% capital allocation * 1% trade return ≈ 15% annual simple return
300 trades * 5% capital allocation * 2% trade return ≈ 30% annual simple return
```

Those are already attractive. They are nowhere near **1,100%–3,800% annual return**.

This strongly suggests at least one of the following:

1. Trade returns are being applied to the whole portfolio rather than the position notional.
2. Closed-trade P&L is being counted multiple times.
3. Daily returns are being constructed from cumulative trade P&L incorrectly.
4. Position sizing uses equity but does not actually reserve cash/notional, allowing hidden leverage.
5. Simulated positions are duplicated across agents or signal paths.
6. Equity is being compounded at the trade level in a way that ignores open exposure constraints.
7. Sharpe is calculated from a non-portfolio return series.

### 2.2 QualityShort returns are even more suspicious

QualityShort reports 14x–17x fold returns in some years with only about 60 trades and win rates around 21%–30%.

If each short trade is capped at 5% of equity, then 14x over 60 trades requires:

```text
(1 + 0.05 * r)^60 = 14
r ≈ 90% per trade
```

For a short, the maximum possible unlevered notional return is close to 100% if the stock goes to zero. So the simulator is implying that the average QualityShort trade is close to a total collapse, despite most folds showing win rates below 30%. That is mechanically incompatible with normal short equity trading.

This points to a likely bug in short P&L accounting or total-return reconstruction. Specific short-side bugs to check:

- Short return calculated as `(entry - exit) / exit` instead of `(entry - exit) / entry`, which explodes when price falls.
- Short proceeds incorrectly treated as free deployable capital without corresponding liability.
- Borrow cost sign or accrual logic wrong.
- Unrealized short P&L included repeatedly in daily returns.
- Trade-level returns compounded as if each trade used 100% of equity.
- Max drawdown calculated on a different series than total return.

### 2.3 Max drawdowns are not credible

PEAD fold max drawdowns of ~1.7%–2.8% paired with 11x–38x annual returns are not credible. QualityShort max drawdowns of **0.01%–0.69%** are even less credible.

If a strategy can make 10x+ per year, it has enormous mark-to-market movement unless it has hidden arbitrage economics, which PEAD and fundamental shorts do not. A short book of deteriorating, levered companies should experience violent rallies, borrow pressure, squeeze events, and gap risk. A 0.01% drawdown year while returning 14x is not a realistic portfolio outcome.

This is a major reason I believe the equity curve is not representing actual portfolio equity.

---

## 3. Go / no-go view

### Current status

**No-go for real capital. Conditional go for instrumented paper/shadow trading only.**

I would not run this as an actual money strategy until the backtest engine passes a full accounting audit. Paper trading is acceptable only if the goal is to test:

- signal availability,
- API latency,
- fill/slippage differences,
- shortability/ETB rejection rates,
- live-vs-sim replay accuracy,
- order lifecycle reliability.

It should not be framed as “validating Sharpe.” Three months is too short to statistically validate annualized Sharpe.

### Gate before any serious deployment

Before treating any Sharpe as meaningful, I would require all of the following to pass:

1. Independent daily equity reconstruction from raw trades, fills, and positions.
2. Proof that daily returns are computed only as `equity_t / equity_{t-1} - 1`.
3. Proof that cash, short proceeds, margin, open liabilities, borrow costs, and transaction costs reconcile daily.
4. Trade-level P&L bounded by position notional where appropriate.
5. Exposure report showing long gross, short gross, net exposure, and concurrent positions by day.
6. Re-run with PIT universe and delisted securities.
7. Re-run with realistic open/close slippage and shortability constraints.
8. Entry-delay tests to determine whether the PEAD signal survives realistic data latency.
9. Parameter stability tests rather than a single selected hold/threshold configuration.
10. A paper/live replay harness that compares actual paper fills to the backtest fill assumption for the same trade date.

---

## 4. Answers to the 26 reviewer questions

## Data integrity

### 1. Is Yahoo Finance acceptable for Russell 1000 daily OHLCV Sharpe validation?

**For exploration: yes. For validating a Sharpe 8 strategy: no.**

Yahoo Finance via yfinance is fine for early research and broad sanity checks, especially on liquid large caps. But it is not acceptable as the final evidence source for a strategy whose result is already implausible. The required validation dataset should include:

- delisted securities,
- split/dividend adjusted prices,
- point-in-time ticker changes,
- historical index membership,
- ideally raw and adjusted OHLCV,
- corporate action metadata,
- timestamp/audit trail for fundamentals and earnings data.

Recommended options:

- **Norgate**: practical retail/pro-sumer choice for survivorship-bias-free historical equities and index constituents.
- **Polygon / Databento / Tiingo / Intrinio / Nasdaq Data Link**: useful depending on budget and required fields.
- **CRSP / Compustat / IBES**: institutional benchmark if accessible.

For this project, the minimum upgrade should be: **Norgate or equivalent survivorship-bias-free equity data with delisted names**, plus a separate audit of FMP fundamentals.

### 2. Is FMP `earnings-surprises` truly PIT-safe in this usage?

**Not proven. Treat it as not PIT-safe until you validate `available_at`, not just `report_date`.**

FMP documentation describes the earnings-surprises endpoint as returning a list of earnings announcements with announcement date, estimated EPS, actual EPS, and surprise. That supports the idea that the `date` field is the announcement date, not necessarily a later filing date. However, for a live strategy, the key field is not just `date`; it is:

```text
When did this record become available through the API?
```

A backtest using `report_date <= as_of` can still leak if the data provider populated the surprise field later but backfilled it to the announcement date.

Required validation:

- During the next earnings cycle, poll FMP every 5–15 minutes and store raw snapshots.
- Add `fmp_observed_at_utc` to every earnings record.
- In backtest mode, simulate availability using `observed_at`, not `report_date`.
- Separate before-open, during-market, and after-close earnings.
- Do not allow a premarket signal unless the data was observable before the trading decision time.

### 3. Are fundamental fields most-recent-quarter or TTM?

Based on the prompt, they appear to be **most-recent-quarter fields**, with YoY revenue growth computed against the same fiscal quarter prior year. That is fine as long as:

- the financial statement row is selected using a true `available_at` date;
- `filingDate` is reliable and not backfilled;
- fallback logic is conservative;
- quarterly and TTM variants are clearly separated.

For QualityShort, most-recent-quarter deterioration can be useful, but it is also noisy. A better short quality model should test both:

- most recent quarter deterioration;
- TTM deterioration;
- trailing 4-quarter margin/revenue trend;
- revision changes;
- accruals/cash-flow quality;
- liquidity and debt maturity pressure.

### 4. Realistic latency between earnings release and FMP surprise availability

Unknown without direct measurement. The prompt assumes hours. FMP’s published cycle-time page says the earnings calendar updates every 3 hours, and the FMP FAQ says financial data can often be gathered through an 8-K within 24 hours of an earnings report, with the 10-Q taking longer.

For a PEAD strategy entering next-day open, this may be acceptable for after-close earnings if the data is available overnight. It may be problematic for premarket earnings if the strategy assumes immediate recognition but FMP updates after the open.

The correct approach is empirical:

```text
For every earnings event:
  company_release_time
  FMP_first_seen_time
  model_signal_time
  first_tradeable_time
  actual_order_time
```

Then compare performance under:

- same-day close entry,
- next-day open entry,
- next-day close entry,
- one additional day delay,
- FMP-first-seen-time entry.

### 5. Hidden look-ahead from `filingDate` vs `acceptedDate` fallback?

Yes, possible. The danger is not just the fallback itself, but whether fallback usage is material.

Potential issues:

- `filingDate` may represent local date, not timestamp.
- `acceptedDate` may be later than filing date and safer, but if missing, `period_end + 45d` is a synthetic assumption.
- `period_end + 45d` can be too early for delayed filings and too late for 8-K preliminary data.
- If restated data is pulled later and assigned to old filing dates, the model can see revised fundamentals unavailable at the time.

Required audit:

```text
For all FMP fundamental rows:
  % using filingDate
  % using acceptedDate
  % using period_end + 45d fallback
  performance contribution by each bucket
  compare values to SEC filing values for random samples
  identify restatements and amended filings
```

If a large share of profitable trades comes from fallback rows, the result is not trustworthy.

---

## Walk-forward validity

### 6. Is 5 folds x 1-year adequate for ~300 trades/year?

Superficially, 1,500 PEAD trades is a reasonable sample count. But the effective sample size is smaller because PEAD trades are clustered around earnings seasons and correlated by:

- sector,
- market regime,
- earnings cycle,
- macro volatility,
- liquidity regime,
- reporting calendar.

Five annual folds are not enough when the strategy has been selected from many configurations. The fold design is useful, but insufficient.

Add:

- rolling 6-month and 3-month walk-forward windows;
- anchored vs expanding vs rolling training windows;
- fold-start jitter by 1–3 months;
- event-cluster bootstrap by earnings week, not individual trade;
- sector-cluster bootstrap;
- year-by-year and quarter-by-quarter attribution.

### 7. Does `purge_days = 10` adequately prevent leakage for 5-day PEAD?

For a pure 5-day forward-return label, 10 calendar days is probably adequate. However, this assumes there are no overlapping features that reach into the future, no cross-sectional ranks using future universe membership, and no data availability backfills.

The bigger issue is not purge length. It is:

- static current Russell 1000 universe;
- FMP data availability timestamps;
- parameter selection across many trials;
- potential equity accounting error.

### 8. Is Sharpe 8.1 a bug, overfit, or real effect?

Most likely: **bug or accounting error first, then overfit/data leakage second, real effect distant third.**

A Sharpe 8 PEAD strategy that compounds 11x–38x per year after costs with tiny drawdowns is not consistent with known PEAD economics. If such a strategy existed at Russell 1000 scale with next-day open fills, it would be heavily exploited by institutional event-driven and stat-arb desks.

Highest-probability bug categories:

1. Daily return series is not portfolio-equity-based.
2. Position notional is not correctly applied to trade returns.
3. Cash is not reserved against positions.
4. Multiple open positions are counted incorrectly.
5. Short P&L formula is wrong.
6. Max drawdown is computed from a different or smoothed series.
7. Trade P&L is re-counted across hold days.
8. Exit logic applies target/stop/close in a way that creates impossible fills.
9. Folds include overlap or repeated trades.
10. Survivorship and timestamp leakage inflate the base signal.

### 9. Are 2022 fold Sharpes suspicious?

Yes. Fold 2 being the best fold for both PEAD and QualityShort is not impossible, but it increases suspicion.

For QualityShort, 2022 was a favorable environment for weak-balance-sheet shorts. A strong short result in 2022 is plausible. But the reported combination of high return and near-zero drawdown is not.

For PEAD, 2022 being the best fold is more ambiguous. Earnings surprises may have generated stronger dispersion in volatile markets, but PEAD should also face wider spreads, gap risk, and risk-off correlation. If costs and open slippage are under-modeled, 2022 may look artificially good.

Required attribution:

- Long PEAD vs short PEAD separately.
- Beat vs miss separately.
- Sector and market-cap buckets.
- Earnings before-open vs after-close.
- Open-to-close vs close-to-close components.
- Contribution from top 10 trades.
- Distribution of trade returns, not just averages.

### 10. Multiple-testing correction

Use DSR with **N_trials greater than or equal to every meaningful strategy variant tested**, not just the current formal configs.

The prompt mentions ~17 Phase H/H+ configs, plus earlier Phase F/G iterations. I would set `N_trials` to at least:

```text
max(50, all parameter combinations, all discarded notebooks, all LLM-suggested variants that influenced design)
```

That may sound harsh, but the actual research process includes informal data snooping: every time a human or LLM observes a result and changes the next test, that is part of the search.

Use:

- Deflated Sharpe Ratio;
- Probability of Backtest Overfitting (PBO);
- Combinatorially Symmetric Cross-Validation (CSCV);
- White’s Reality Check or Hansen SPA if feasible;
- simple holdout set untouched by model selection.

---

## PEAD specific

### 11. What explains a 5–10x higher PEAD Sharpe?

Ranked by likelihood:

1. **Portfolio accounting bug** — most likely.
2. **Position sizing/cash/leverage bug** — very likely.
3. **FMP timestamp leakage / backfilled surprise data** — possible.
4. **Survivorship bias from current Russell 1000 list** — meaningful but unlikely to explain Sharpe 8 alone.
5. **Open fill assumptions too favorable** — meaningful but unlikely to explain 38x returns alone.
6. **Parameter/data snooping** — meaningful.
7. **Genuine anomaly at $100k scale** — possible only after the first six are disproven.

The return magnitude is the key. Costs and survivorship could turn a Sharpe 1.5 into Sharpe 3 in a biased backtest. They do not normally turn it into Sharpe 8 with 38x annual return and 2% drawdown.

### 12. Are 11x–38x fold returns plausible?

No, not under the stated sizing and trade count.

With 300 trades/year and 5% sizing, a 38x fold requires roughly 24% average return per trade on notional. That is an impossible average for next-day-open PEAD across Russell 1000.

Audit tasks:

```text
For each PEAD fold:
  export trade_id, symbol, direction, signal_date, entry_date, exit_date
  entry_price, exit_price, shares, notional_at_entry
  realized_pnl_dollars
  realized_pnl_pct_on_notional
  realized_pnl_pct_on_equity
  fees, slippage, borrow
  account_equity_before_trade
  account_equity_after_trade
```

Then verify:

```text
sum(realized_pnl_dollars) roughly reconciles to final_equity - starting_equity
each trade contribution <= position_weight * plausible underlying return
daily equity equals cash + market value of positions - short liabilities
```

If the trade ledger cannot explain the equity curve, the backtest is invalid.

### 13. Is 5 bps/side realistic for PEAD entering next-day open?

For liquid mega-caps: maybe. For the full Russell 1000, especially mid-cap earnings movers at the open: probably too optimistic.

I would test:

| Scenario | Entry slippage | Exit slippage | Round-trip friction |
|---|---:|---:|---:|
| Optimistic | 5 bps | 5 bps | 10 bps |
| Base | 15 bps | 15 bps | 30 bps |
| Conservative | 30 bps | 20 bps | 50 bps |
| Stress | 50 bps | 50 bps | 100 bps |

Also test worse slippage for:

- low ADV names,
- high overnight gap names,
- high spread names,
- premarket earnings,
- post-announcement volatility spikes.

At $100k, market impact is small, but spread/open-auction slippage is still real.

### 14. What happens if PEAD entry is delayed one more day?

This is a critical test. A real PEAD effect should decay but not disappear instantly if the strategy is genuinely exploiting delayed assimilation. If the Sharpe collapses from 8 to near zero when entry is delayed by one day, that points to either:

- timing leakage,
- unrealistic open fill,
- exploiting the immediate post-announcement gap rather than drift,
- data timestamp issue.

Run at least:

```text
D0 close after announcement
D1 open
D1 VWAP/close
D2 open
D2 close
D3 open
```

For each, evaluate:

- long beats separately;
- short misses separately;
- before-open vs after-close announcements;
- market-cap/liquidity buckets;
- earnings surprise deciles.

---

## QualityShort specific

### 15. Does QualityShort equity curve reconstruct correctly?

Probably not. The reported win rate, return, and drawdown combination is the strongest evidence.

With 60 trades/year, 5% sizing, 14x return, and 21% win rate, either the strategy found repeated near-bankruptcies before they collapsed, or the accounting is wrong. In a current Russell 1000 universe, repeated near-zero shorts are especially unlikely because true collapses and delistings are underrepresented.

Audit short P&L specifically:

```text
long pnl = shares * (exit - entry)
short pnl = shares_short * (entry - exit)
short return on notional = (entry - exit) / entry
max profit on a short before fees cannot exceed entry notional
short liability = shares_short * current_price
cash includes short-sale proceeds but liability offsets it
borrow cost = borrow_rate * short_market_value / day_count
```

Also check whether a falling short price creates excess buying power that is immediately redeployed in a way that creates unintended leverage.

### 16. Realistic borrow-cost distribution for QualityShort names

The assumed 0.5% annual borrow cost is not realistic for the type of names QualityShort is likely to select.

A realistic model should include:

| Bucket | Approx. annual borrow | Notes |
|---|---:|---|
| Easy-to-borrow liquid large caps | 0%–1% | Alpaca currently supports ETB only and may charge 0% borrow on ETB shares |
| Moderately constrained | 1%–5% | realistic for some weaker names |
| Hard-to-borrow | 5%–25% | common for crowded deteriorating shorts |
| Extreme / meme / distress | 25%–100%+ | can be unavailable or uneconomic |

Important Alpaca-specific point: Alpaca says it currently supports opening shorts only in easy-to-borrow securities, with hard-to-borrow not yet available. Therefore the live paper strategy may simply be unable to short many of the exact names QualityShort wants.

So the right live constraint is not “borrow cost = 10%.” It is:

```text
if not shortable or not easy_to_borrow: trade rejected
if ETB changes to HTB before open: order canceled
```

Backtests should include a proxy shortability filter, and paper trading must log all short rejects.

### 17. Exposure to short squeezes

High. QualityShort is structurally exposed to short squeezes because it selects weak, levered, negative-margin, negative-surprise names — exactly the type of stock that may have high short interest and convex rally risk.

The current backtest likely underrepresents squeeze risk because:

- current Russell 1000 membership excludes many historical disasters;
- current membership may exclude names that were squeezed, delisted, acquired, or demoted;
- borrow constraints are not modeled;
- hard-to-borrow and threshold security behavior is not modeled;
- 20% circuit breaker may not execute at 20% during gaps;
- short sale restriction effects are ignored;
- crowded short exposure is not included.

Stress tests:

```text
Inject +20%, +40%, +80% one-day gap events into short book.
Replay Jan 2021 meme-stock dates against any names selected then.
Add short-interest and borrow-fee proxies.
Disallow shorts that fail ETB/shortable screen.
Cap aggregate short exposure by squeeze risk.
```

---

## Portfolio construction

### 18. Expected concurrent position count and MAX_OPEN_POSITIONS

Back-of-envelope:

- PEAD: ~300 trades/year, 5-day hold → ~6 average concurrent positions.
- But earnings are clustered. During peak weeks, PEAD could exceed 20 concurrent names.
- QualityShort: ~50–60 trades/year, 20-day hold → ~4–5 average concurrent shorts.
- Factor long book: proposed 20 names.
- Combined average: roughly 30 positions.
- Combined peak: potentially 45–70 positions during earnings season if not actively constrained.

`MAX_OPEN_POSITIONS = 40` may be enough on average but not enough for earnings clusters. More importantly, a raw count cap is inferior to risk budgeting.

Use risk budgets:

| Leg | Suggested initial gross cap | Name cap | Notes |
|---|---:|---:|---|
| PEAD long | 20% | 3%–5% | event risk |
| PEAD short | 15% | 2%–4% | gap/squeeze risk |
| QualityShort | 15% | 2%–3% | only ETB; hard squeeze cap |
| Factor long | 40% | 3%–5% | regime-gated |
| Cash/reserve | 10%+ | n/a | operational buffer |

Total gross should start below 75% until simulator reliability is proven.

### 19. Could PEAD-short and QualityShort double short same name?

Yes, unless there is explicit de-duplication. This must be fixed.

Rules:

```text
one net position per symbol
strategy sleeves can vote, but execution layer nets exposure
symbol-level max gross exposure
directional conflict resolver
duplicate short resolver
same-day signal priority
```

Example:

- PEAD miss says short 5%.
- QualityShort says short 5%.
- Execution should not blindly short 10%.
- It should either cap at 5%, or allow a higher cap only under explicit multi-signal agreement with risk budget approval.

### 20. Should PEAD also be regime-gated?

Not necessarily with the same SPY MA200 rule, but PEAD needs some regime awareness.

PEAD longs and PEAD shorts should be modeled separately. It may be true that:

- PEAD longs work best in normal/up regimes;
- PEAD shorts work better in stressed/down regimes;
- miss reactions are more volatile and borrow-constrained;
- beats in bear markets fade quickly.

Recommended regime tests:

```text
SPY above/below MA200
SPY above/below MA50
VIX < 20, 20–30, 30–40, >40
credit spread proxy if available
sector-level regime
earnings season volatility regime
```

Do not add a regime gate because it sounds prudent. Add it only if it is stable across fold-start jitter and parameter perturbation.

---

## Risk management

### 21. Does inconsistent Sharpe gate calibration matter?

For the current top picks, no mechanically, because the reported Sharpes are far above either gate. But strategically, yes. It suggests the gating framework is not mature.

A meaningful research gate should include:

- minimum number of trades;
- max contribution from top 5 trades;
- max drawdown sanity;
- turnover and cost sensitivity;
- exposure constraints;
- parameter stability;
- probability of backtest overfitting;
- live implementability;
- data-source independence;
- slippage and borrow stress survival.

A Sharpe gate alone is not adequate, especially when Sharpe may be computed from a flawed return series.

### 22. Is 20% circuit-breaker stop appropriate for 5-day PEAD?

For PEAD, 20% is not really a risk-management stop. It is a disaster stop. Because earnings/event gaps can jump through it, it should not be modeled as a guaranteed exit at exactly 20%.

I would use:

- no intraday stop in the base event-driven research model;
- close-based or next-open risk exit;
- volatility-scaled stop for operational risk only;
- gap-aware fill model;
- per-symbol event risk cap;
- portfolio-level daily loss stop.

For a 5-day PEAD hold, the main risk control should be position sizing and diversification, not a 20% stop.

### 23. Borrow cost estimate for QualityShort

A flat 0.5% annual borrow cost is too low for the intended short universe. But with Alpaca, the more important issue is that hard-to-borrow shorts may be unavailable.

For modeling:

- ETB-only simulation: 0%–1% borrow, but many trades rejected.
- Institutional simulation: dynamic borrow 1%–25%, with locate failures.
- Stress simulation: 10% flat, 25% flat, 50% for crowded names.
- Rejection simulation: reject 10%, 25%, and 50% of QS candidates randomly and by risk score.

Also model short-sale constraints:

- no fill if not shortable;
- no fill if not easy-to-borrow in Alpaca;
- canceled order if ETB becomes HTB;
- potential inability to maintain borrow;
- forced buy-in scenario.

---

## Paper trading plan

### 24. Is 3 months sufficient to validate Sharpe?

No. Three calendar months is roughly 60–65 trading days. Even using 90 daily observations, the confidence interval around an annualized Sharpe estimate is extremely wide.

Approximate annualized Sharpe standard error under a simple iid assumption:

```text
SE(annualized Sharpe) ≈ sqrt(252 / N_daily)
```

So:

```text
N = 63 trading days -> SE ≈ 2.0
N = 90 trading days -> SE ≈ 1.7
```

A 95% interval is roughly ±3.3 to ±4.0 Sharpe points. That means a 3-month observed Sharpe of 0.5 tells you almost nothing statistically. It could easily come from a zero-edge strategy.

Use paper trading to validate implementation, not Sharpe.

Paper metrics that matter more:

- signal generated when expected;
- FMP data available before trade decision;
- no lookahead in live logs;
- fill price vs next-open assumption;
- realized slippage by symbol/liquidity bucket;
- rejected short rate;
- order cancellation rate;
- live trade ledger reconciles exactly;
- paper P&L matches independent replay;
- factor exposures match intended budgets.

### 25. What paper Sharpe should we expect after live/sim degradation?

If the backtest were real, 30%–70% degradation from PEAD Sharpe 8 would still leave an extremely high live Sharpe. That itself is suspicious.

A more realistic expectation after fixing accounting, costs, survivorship, and data timing:

| Strategy | Plausible true paper/live Sharpe range after fixes |
|---|---:|
| PEAD long/short | 0.3–1.5 |
| PEAD long only | 0.4–1.2 |
| PEAD short only | 0.0–1.0, more borrow/squeeze risk |
| QualityShort | -0.2–1.0 unless borrow/shortability handled well |
| Factor long regime-gated | 0.3–1.2 depending on market regime |
| Combined book | 0.5–1.5 if truly diversified and costs controlled |

If a 3-month paper run prints Sharpe 4+, I would not celebrate. I would first look for continued accounting/fill issues or lucky clustering.

### 26. Recommended kill-switch criteria

Use several kill switches. Do not rely on one Sharpe trigger.

#### Operational kill switches

Stop trading immediately if:

- daily equity reconciliation fails by more than $1 or 1 bp;
- any trade executes without a valid signal snapshot;
- any signal uses data without `observed_at <= decision_time`;
- any short order is submitted for non-shortable or non-ETB stock;
- duplicate same-direction exposure breaches symbol cap;
- live position count or gross exposure exceeds configured budget;
- order status is unknown for more than a defined window.

#### Slippage/fill kill switches

Pause strategy if:

- average round-trip slippage exceeds 50 bps over 20 trades;
- any single fill is worse than 150 bps from modeled price without explanation;
- PEAD next-open fills are systematically worse than backtest by >20 bps;
- paper reject/cancel rate exceeds 20% for QualityShort.

#### Risk kill switches

Initial paper risk limits:

- daily loss > 2% of equity: pause new entries for 1 trading day;
- weekly loss > 4%: pause and review;
- peak-to-trough drawdown > 6%: halt strategy pending review;
- single-name loss > 1% of account: block that symbol and review;
- short squeeze event > 20% adverse gap: disable short sleeve pending review;
- cumulative slippage + fees > 30% of gross alpha estimate: pause.

#### Statistical/process kill switches

Over first 3 months, do not kill solely on Sharpe. Kill based on process failures:

- live hit rate directionally opposite of backtest by >2 standard errors;
- average PEAD trade return after costs below 0 across 100+ trades;
- long and short PEAD both fail separately;
- top 5 trades account for more than 80% of P&L;
- live rejects materially change portfolio composition vs sim.

---

## 5. Additional validation tests I would run immediately

## Phase 0 — Forensic accounting audit

This is the top priority.

### Test 0.1: Daily accounting invariant

For each day:

```text
equity = cash + long_market_value - short_market_value + accrued_income - accrued_costs
daily_return = equity_t / equity_t_minus_1 - 1
```

Recompute this independently from the simulator output. No strategy research matters until this passes.

### Test 0.2: Trade ledger replay

Create a standalone script that reads only the exported trade ledger and price data and reconstructs the full daily equity curve. It should not call the strategy code. If the replay does not match the backtest, the simulator is broken.

### Test 0.3: P&L bound checks

For every closed trade:

```text
long_trade_return_on_notional ≈ (exit - entry) / entry
short_trade_return_on_notional ≈ (entry - exit) / entry
```

Flag:

- long returns below -100% or absurd positive returns;
- short returns greater than 100% before fees;
- trade contribution to equity larger than position weight times notional return;
- any trade with P&L inconsistent with shares * price movement.

### Test 0.4: Exposure/cash reservation audit

For every day, export:

```text
gross_long
gross_short
net_exposure
cash
buying_power
number_open_positions
max_single_name_weight
sector_weights
strategy_sleeve_weights
```

Check that total exposure never exceeds intended limits.

### Test 0.5: Drawdown recomputation

Recompute max drawdown from the daily equity curve in a separate script. If QualityShort still shows 0.01% drawdown with 14x return, inspect the equity curve manually.

---

## Phase 1 — Data and timestamp audit

### Test 1.1: PIT universe rebuild

Re-run using PIT index constituents and delisted names. Minimum acceptable proxy:

- historical Russell 1000 constituents if available;
- otherwise S&P 500 / Russell 3000 with PIT membership as a sanity comparison;
- include delisted securities.

### Test 1.2: FMP snapshot logging

Start logging raw FMP responses with:

```text
request_time_utc
endpoint
symbol
raw_payload_hash
earnings_date
actual_eps
estimated_eps
surprise
```

Backtest should use the earliest observed timestamp, not just reported date.

### Test 1.3: FMP vs SEC/press release sample audit

For 100 random earnings events:

- company release timestamp;
- FMP date;
- FMP first observed timestamp;
- before-open / after-close flag;
- next tradable open;
- backtest signal date.

No leakage should be possible.

### Test 1.4: Corporate action and bad-bar audit

Flag all trades where:

- split occurred within ±10 days;
- dividend ex-date occurred during hold;
- OHLC values have high/low inconsistencies;
- open-to-close move exceeds 30%;
- volume is zero or missing;
- price gaps are extreme.

---

## Phase 2 — Cost, slippage, and shortability stress

### PEAD cost stress

Run the exact current PEAD configuration under these cost models:

| Model | Entry | Exit | Other |
|---|---:|---:|---|
| Current | 5 bps | 5 bps | no spread |
| Base | 15 bps | 15 bps | no impact |
| Conservative | 30 bps | 20 bps | gap-aware |
| Stress | 50 bps | 50 bps | no fill for extreme gap names |

A legitimate edge should not collapse completely under the base case.

### QualityShort borrow stress

Run:

- 0.5% flat;
- 5% flat;
- 10% flat;
- 25% for high-risk names;
- reject 25% non-ETB candidates;
- reject 50% most distressed candidates.

If QS only works when every distressed short is borrowable at 0.5%, it is not implementable.

---

## Phase 3 — Robustness and anti-overfit tests

### Parameter stability

PEAD:

```text
surprise threshold: 2%, 5%, 7.5%, 10%
hold: 1, 3, 5, 7, 10 days
entry delay: D1 open, D1 close, D2 open, D2 close
long only / short only / both
market cap buckets
sector-neutral vs unconstrained
```

QualityShort:

```text
flags_required: 1, 2, 3
max_shorts: 5, 10, 15, 20
hold: 5, 10, 20, 40 days
shortability filter on/off
borrow assumptions
VIX gates
exclude high short-interest/meme-risk names
```

The strategy should degrade smoothly. It should not depend on exactly one parameter tuple.

### Placebo tests

Run:

- shuffled earnings surprise signs;
- surprise dates shifted forward/backward by 5–20 days;
- random Russell 1000 symbols matched by sector and market cap;
- fake earnings calendar with same monthly clustering;
- prior-quarter surprise applied to current quarter.

If placebo Sharpe remains high, the simulator or universe is leaking.

### Fold-start jitter

Shift fold start dates by:

```text
1 month, 2 months, 3 months, 6 months
```

If results collapse, the annual folds are lucky.

### Top-trade concentration

Report:

- % P&L from top 5 trades;
- % P&L from top 10 trades;
- % P&L from top 5 symbols;
- % P&L from best earnings week;
- % P&L from 2022.

If top 5 trades dominate, the Sharpe is not stable.

---

## Phase 4 — Paper trading design

### Shadow mode first

Before live paper orders, run two weeks of shadow mode:

- generate signals;
- log decision times;
- record intended orders;
- record theoretical fill prices;
- compare to actual market opens/closes;
- log shortability/ETB status;
- do not submit orders.

### Then instrumented paper mode

Start with reduced risk:

- PEAD max 1%–2% per name;
- QualityShort max 1% per name;
- combined gross cap 25%–40%;
- no factor long book until PEAD/QS accounting is validated;
- no duplicate symbols;
- no non-ETB shorts.

Paper trading objective:

```text
Validate implementation, not Sharpe.
```

Required daily outputs:

- signal file;
- orders file;
- fills file;
- positions file;
- equity file;
- reconciliation file;
- slippage report;
- rejected short report;
- live-vs-sim replay.

---

## 6. Proposed next-step roadmap

## Stage A — Stop and audit the backtest engine

Deliverables:

1. `daily_equity.csv`
2. `positions_by_day.csv`
3. `trades.csv`
4. `orders.csv`
5. `fills.csv`
6. `cash_ledger.csv`
7. `recomputed_equity_audit.py`
8. `short_pnl_unit_tests.py`
9. `drawdown_recalc.py`

Acceptance criteria:

- Independent replay matches simulator equity within tolerance.
- No impossible short returns.
- No hidden leverage unless explicitly intended.
- Max drawdown recomputation matches report.
- Trade contributions reconcile to total return.

## Stage B — Re-run current strategies after accounting fixes

Do not change alpha logic yet. Re-run exactly:

- PEAD hold 5;
- QualityShort flags 2;
- current costs;
- current universe.

Compare old vs new metrics. If Sharpe drops from 8 to normal levels, the issue was accounting.

## Stage C — Add realistic frictions

Re-run with:

- 30 bps PEAD round trip;
- D2 entry delay;
- ETB-only shorts;
- 10% borrow stress;
- PIT/delisted universe if available.

Acceptance criteria for continued research:

- PEAD Sharpe > 1.0 in base realistic case;
- QS Sharpe > 0.5 in ETB/borrow-constrained case;
- max drawdown and returns plausible;
- no single fold or top trades dominate.

## Stage D — Research improvements only after validation

Once accounting is clean, improve the alpha:

PEAD improvements:

- surprise standardized by analyst dispersion;
- earnings time-of-day handling;
- abnormal volume and gap filters;
- post-announcement drift confirmation;
- sector-relative earnings surprise;
- exclude high-spread/low-liquidity names;
- separate beat and miss models;
- text/news sentiment from earnings call or press release.

QualityShort improvements:

- short-interest / days-to-cover filter;
- borrow availability/rate model;
- avoid meme/squeeze risk;
- accruals and cash-flow quality;
- analyst revision deterioration;
- debt maturity / refinancing risk;
- sector-relative deterioration;
- catalyst timing.

Portfolio improvements:

- sleeve-level risk budgets;
- volatility targeting;
- sector caps;
- beta and factor exposure control;
- de-duplication and netting;
- execution-aware sizing.

---

## 7. Probability estimate

The requested probability is: probability that proposed paper trading deployment achieves Sharpe > 0.50 over a 3-month live paper period.

I would split this into two definitions.

### Observed 3-month paper Sharpe > 0.50

Because 3-month Sharpe is extremely noisy, even a weak or zero-edge strategy can print above 0.50 by chance. So the probability of *observed* paper Sharpe > 0.50 is not the same as the probability the strategy is good.

My estimate:

```text
Observed 3-month paper Sharpe > 0.50: 40%–55%
```

This is not an endorsement. It reflects noise.

### True live process Sharpe > 0.50 after correcting implementation issues

Given the current red flags, my estimate is lower:

```text
True live/paper process Sharpe > 0.50 without major fixes: 25%–40%
True live/paper process Sharpe > 0.50 after accounting/data/cost fixes: 40%–60%
True live/paper process Sharpe > 1.50 after fixes: 15%–30%
True live/paper process Sharpe near current WF result: <5%
```

My best single estimate for the current proposed deployment, without further audit:

```text
35% probability that the strategy has a true live/paper Sharpe > 0.50.
45% probability that a 3-month observed paper Sharpe prints > 0.50 due to noise/luck even if the true edge is weak.
```

---

## 8. The most important thing to tell the next LLM

Do not let the next reviewer spend most of their time proposing new alpha features. The immediate question is not “how do we improve PEAD?” It is:

```text
Can the reported equity curve be mechanically reconciled to the described trades, position sizes, fills, costs, and daily account equity?
```

Until the answer is yes, every Sharpe number is suspect.

The key challenge to give the next LLM:

> Given 300 PEAD trades/year, 5% max position size, and 38x annual return, calculate the implied average trade return. Given 60 QualityShort trades/year, 5% max position size, 14x annual return, and 21% win rate, calculate whether the return is even feasible. Then identify the most likely accounting bugs.

That will force the analysis into the right place.

---

## 9. Concise final recommendation

**Do not proceed to normal Phase I paper deployment as currently framed.**

Proceed with:

1. forensic backtest accounting audit;
2. trade-ledger replay;
3. short P&L unit tests;
4. PIT/timestamp audit;
5. realistic cost and borrow stress;
6. shadow-mode live signal/fill logging;
7. only then reduced-risk paper trading.

If the Sharpe survives those tests above ~1.0–1.5, then this is worth continuing. If it collapses, that is still a win: you avoided building live infrastructure around a broken backtest.

---

## 10. Source notes used in this review

The review primarily relies on the uploaded MrTrader strategy prompt. I also checked current/public documentation and literature references for:

- Deflated Sharpe Ratio and multiple-testing adjustment: Bailey & López de Prado, “The Deflated Sharpe Ratio.”
- Sharpe estimation uncertainty: Andrew Lo, “The Statistics of Sharpe Ratios.”
- FMP earnings-surprise and earnings calendar documentation.
- FMP FAQ/cycle-time notes on earnings calendar refresh and financial statement data collection timing.
- Alpaca documentation/support notes on easy-to-borrow vs hard-to-borrow shorting constraints.
- SEC Regulation SHO locate requirement.
- Norgate description of survivorship-bias-free data and historical index constituents.

