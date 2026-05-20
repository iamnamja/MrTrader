# MrTrader Strategy Review — Quant Expert Analysis Request

## Reviewer Instructions

You are a senior quantitative researcher at a top-tier hedge fund (think Citadel, Two Sigma, Renaissance) with 15+ years of systematic trading experience across statistical arbitrage, event-driven, and factor investing. You have personally run hundreds of walk-forward studies, have seen many strategies die in production, and you are paid to be skeptical.

Your job in this review is **not** to be encouraging. It is to find every flaw, methodological weakness, false assumption, latent bias, or hidden risk in the work below. We are about to deploy real (paper) capital and the worst outcome is a strategy that looks great in research but degrades in production. Be precise, be technical, and use specific references to the code/data described.

Specifically:
1. Identify any methodological issue in the walk-forward that could inflate Sharpe.
2. Identify any look-ahead, survivorship, or selection bias.
3. Critique the economic plausibility of the reported Sharpe ratios (PEAD avg 8.1 is extraordinary — is it real?).
4. Critique the realism of execution assumptions vs live trading.
5. Recommend concrete additional validation tests we should run before going live.
6. Give a calibrated probability estimate that the proposed paper trading deployment will achieve Sharpe > 0.50 over a 3-month live paper period.

When you are uncertain, say so. When you can compute an order of magnitude estimate (e.g., expected slippage on 300 trades/yr in Russell 1000), do.

---

## System Overview

**MrTrader** is a systematic equity trading system written in Python.

- **Capital:** $100,000 paper account (Alpaca paper broker).
- **Universe:** Russell 1000 constituents (`RUSSELL_1000_TICKERS` constant in `app/utils/constants.py`). Static membership list — composition is fixed at build time, **not** rebalanced as the index reconstitutes. This is the survivorship-bias / index-membership-bias concern (see Limitations).
- **Data:**
  - **Price/volume (OHLCV):** Yahoo Finance via `yfinance`, daily bars.
  - **Fundamentals & earnings:** Financial Modeling Prep (FMP) API, stored to a local parquet (`app/data/fmp_fundamentals.py`). PIT key is `filingDate` (with fallback to `acceptedDate`, or `period_end + 45d` if both missing).
  - **VIX:** loaded from `^VIX` or `VIX` symbol via Yahoo.
  - **Sector map:** static `SECTOR_MAP` constant.
- **Architecture:** PM / RM / Trader agents (async queue). For backtesting and walk-forward, `AgentSimulator` (`app/backtesting/agent_simulator.py`) runs PM scoring + RM rule validation + Trader signal day-by-day on historical bars.
- **Execution model in WF:** entries fill at next-day **open**; exits fill at intrabar stop/target/close.
- **Transaction cost:** `TRANSACTION_COST = 5 bps per side` (`TX_COST_PCT = TRANSACTION_COST`) is applied on both entry and exit. Borrow cost for shorts is 0.5%/year flat, charged daily as `entry_price * qty * 0.005 / 252`.
- **No market impact, no spread, no partial fills, no halts, no borrow availability constraint** in the walk-forward.

---

## Strategy 1: PEAD — Post-Earnings Announcement Drift, 5-Day Hold

**File:** `app/ml/pead_scorer.py` (class `PEADScorer`).

### Signal generation

For each trading day `as_of`:
1. For every symbol in the universe, look up cached FMP earnings features via `get_earnings_features_at(sym, as_of)`. This returns `fmp_surprise_1q` (EPS surprise as a fraction, e.g. `0.08` = 8% beat) and `fmp_days_since_earnings`.
2. Filter to only symbols whose earnings report is `<= MAX_DAYS_AFTER_EARNINGS = 3` calendar days old.
3. Long if `surprise >= +0.05` (5% beat). Short if `surprise <= -0.05` (5% miss).
4. Confidence mapping: `conf = clip(0.65 + (|surprise| − 0.05) * 2.0, 0.65, 0.90)`. So a 5% surprise → 0.65 conf; a 20%+ surprise → 0.90 conf.
5. Sort by `|conf|` descending, return list of `(sym, conf, direction)`.

### Entry, hold, exit

- Entry fill: **next day's open** (NOT the post-earnings-release open — the day after PEADScorer fires).
- Hold cap: `max_hold_bars_override` set to **5** in WF runner. Exits fire on stop, target, max-hold, or force-close at last bar.
- Long stop/target (factor-portfolio mode, `factor_scorer is not None`): `stop_price = entry_price * 0.80` (20% circuit breaker), `target_price = entry_price * 2.0` (effectively never hit). So in practice the long PEAD trade exits on **max-hold = 5 bars** or 20% drawdown.
- Short stop/target: `stop_price = entry_price * 1.20`, `target_price = entry_price * 0.50`. Same — usually max-hold.
- Position sizing: `size_position(account_equity, available_cash, entry_price, stop_price, ml_score=|conf|)` then capped by `MAX_POSITION_SIZE_PCT = 5%`.
- PEAD scorer **does not** apply a VIX or SPY regime gate at the scorer level (the simulator-level regime gate is **skipped** when `factor_scorer is not None`). So PEAD trades fire in any regime including 2022 bear.

### PIT safety

`get_earnings_features_at` is asserted PIT-safe; only earnings reports with `report_date <= as_of` are visible. The FMP `as_of_date` for fundamentals uses `filingDate` (the date the 10-Q/K hit EDGAR), not `period_end`. For the earnings-surprise endpoint specifically, however, the date used is the **earnings report date** itself; this is what most PEAD papers use, but it ignores the gap between press release and 10-Q filing (immaterial here since the press release is when surprise is announced).

### Reported WF result (5 folds, 2021-05-31 → 2026-05-19)

| Fold | Test Start | Test End | Sharpe | Trades | Win Rate | Max DD | Total Return |
|------|-----------|----------|--------|--------|----------|--------|--------------|
| 1    | 2021-05-31 | 2022-05-20 | 8.427 | 275 | 58.9% | 1.74% | 26.46x |
| 2    | 2022-05-31 | 2023-05-20 | 8.787 | 332 | 60.5% | 2.48% | 38.61x |
| 3    | 2023-05-31 | 2024-05-19 | 7.197 | 304 | 58.2% | 2.79% | 11.90x |
| 4    | 2024-05-30 | 2025-05-19 | 8.058 | 290 | 53.1% | 2.05% | 32.20x |
| 5    | 2025-05-30 | 2026-05-19 | 8.077 | 303 | 60.7% | 1.78% | 38.26x |
| **avg** |  |  | **8.109** | 301 | 58.2% | 2.17% | — |

**The total-return figures (11x–38x in a single fold) are an obvious tell that something is wrong, or that the strategy is unconstrained by realistic capacity / sizing.** This is a primary concern we want you to focus on.

---

## Strategy 2: QualityShort — Fundamental Deterioration Shorts (shorts-only mode)

**File:** `app/ml/short_scorers.py` (class `QualityShortScorer`, `legs_mode="shorts_only"`).

### Short candidate filter

For each name in the universe, on `as_of`:
1. Look up the latest FMP fundamentals row with `as_of_date <= as_of` (PIT-safe via filingDate).
2. Count "deterioration flags":
   - `operating_margin <= 0` (most recent quarter)
   - `revenue_growth_yoy <= 0` (YoY, same fiscal quarter prior year)
   - `debt_to_equity >= 1.5`
   - Negative earnings surprise: `fmp_surprise_1q <= -0.05` AND `fmp_days_since_earnings <= 90`
3. Require `flags >= flags_required = 2` (the chosen configuration). Other configs tested 1 and 3.
4. Rank surviving candidates by **most-negative composite factor score** (ascending) — the "most broken" first.
5. Take top `max_shorts = 15`.

### Regime gates (scorer-level)

- `VIX >= 40` → return `[]` (no trades at all).
- Long leg disabled if `SPY < SPY.MA200` OR `VIX >= 30`. Shorts are **allowed in all regimes** except `VIX >= 40`. In `shorts_only` mode the long leg is disabled entirely regardless of regime.

### Entry / exit mechanics

Same factor-portfolio mode in `AgentSimulator`: entry at next day's open, exit on max-hold (`MAX_OPEN_POSITIONS * 4 = 5 * 4 = 20 bars` default, unless overridden), 20% upside circuit breaker. Borrow cost 0.5%/yr.

### Reported WF result (A2_QS_shorts_only)

| Fold | Sharpe | Trades | Win Rate | Max DD | Total Return |
|------|--------|--------|----------|--------|--------------|
| 1 | 3.958 | 29 | 44.8% | 0.03% | 3.18x |
| 2 | 6.773 | 61 | 21.3% | 0.01% | 14.16x |
| 3 | 6.545 | 60 | 26.7% | 0.59% | 14.73x |
| 4 | 6.398 | 61 | 26.2% | 0.35% | 16.69x |
| 5 | 6.093 | 55 | 29.1% | 0.69% | 11.86x |
| **avg** | **5.953** | 53 | 29.6% | 0.33% | — |

**Win rate of 21–45% with these total returns is also unusual** — implies a small number of large winning shorts dominate. Almost-zero max drawdown is suspicious given short books historically experience violent reversals.

---

## Strategy 3: Factor Long Book

**File:** `app/ml/factor_scorer.py` (`compute_composite_score`, `FactorPortfolioScorer`).

### Composite formula (cross-sectional z-scored, winsorised at ±3σ, mean of z-scores)

- Tier 1 (full weight):
  - `momentum_252d_ex1m` × **2.0** (the dominant factor)
  - `price_to_52w_high` × 1.0
  - `profit_margin` × 1.0
  - `operating_margin` × 1.0
  - `pe_ratio` × −1.0
- Tier 2 (× 0.5 each, on by default):
  - `price_to_52w_low`, `volume_trend (20d/60d)`, `range_expansion (ATR5/ATR20)`, `gross_margin`, `revenue_growth_yoy`

### Regime gate

`regime_gate_ok`: SPY > SPY.MA(200) AND VIX < 30. If gate fails, the factor scorer returns `[]` — no longs at all (sits in cash). This is intentional: bottom-N momentum shorts blew up in 2022 bear-market rallies, so the strategy sits out bear markets entirely.

### Position cadence / management

`FactorPortfolioScorer` is scored daily in WF but in practice positions are held until max-hold bars. The simulator uses `MAX_OPEN_POSITIONS = 5` from `RiskLimits` default — but a 20-position factor book is being proposed for Phase I (this is one of the **inconsistencies** the reviewer should flag).

---

## Walk-Forward Methodology

**File:** `scripts/walkforward_tier3.py`.

- **Folds:** 5.
- **Total span:** 6 years (2020-05 → 2026-05, with first segment used as initial train).
- **Purge:** `purge_days = 10` between train_end and test_start.
- **Embargo:** `embargo_days = 10` (defaults to purge_days) after test_end before next fold's train.
- **Segment construction:** `segment_days = int(total_years * 365 / (n_folds + 1))` — so each test window is ~365 days.
- **Universe:** static `RUSSELL_1000_TICKERS` (no point-in-time index membership; current constituents only).
- **Price data:** Yahoo Finance — no adjustment for delistings, no inclusion of names that dropped out of the index.
- **Gate (research mode):** `SHARPE_GATE = 0.8` avg, `MIN_FOLD_SHARPE = -0.3`. (Paper-gate: 0.50 avg, −0.40 min.)
- **N_TRIALS_TESTED** is used to compute a Deflated Sharpe Ratio (DSR) — but the count of trials tracked in `app/ml/retrain_config.py` may not include every L/S configuration tested (Phase H tested 4, Phase H+ tested 13; total trial count for DSR may be undercounted, which inflates DSR p-value).

### Specific concern: trial counting and multiple testing

Phase H tested 4 configurations (A_QualityShort, B_MeanReversionShort, C_SectorRelative, D_Combined). Phase H+ tested 13 more (A1–A4 QualityShort variants, B1–B4 MeanReversion variants, C, D1, D2, E_ABCombined, F_AnalystRev, G_PEAD_hold5). Plus the earlier Phase F and G configs. We pick the top 2 of ~17+ tested. DSR with `N=17` produces a much weaker significance test than `N=1`.

---

## Complete Walk-Forward Results

### Phase H (initial L/S research, 4 configs)

| Config | avg Sharpe | min fold | F2 (2022 bear) | Verdict |
|--------|-----------|----------|----------------|---------|
| A_QualityShort (both legs) | 3.255 | 2.043 | 5.145 | PASS |
| B_MeanReversionShort (both legs) | 3.061 | 2.148 | 3.573 | PASS |
| C_SectorRelative | 2.112 | 0.697 | 3.027 | PASS |
| D_Combined (PEAD + factor + QS) | 3.138 | 2.800 | 2.998 | PASS |

### Phase H+ (13 configs)

| Config | avg Sharpe | min fold | F2 | Verdict |
|--------|-----------|----------|-----|---------|
| A1_QS_longs_only | 1.888 | -1.306 | 1.736 | FAIL |
| **A2_QS_shorts_only** | **5.953** | **3.958** | **6.773** | **PASS** |
| A3_QS_flags1_shorts10 | 3.405 | 2.364 | 4.400 | PASS |
| A4_QS_flags3_shorts20 | 3.518 | 2.561 | 4.554 | PASS |
| B1_MR_longs_only | 1.888 | -1.306 | 1.736 | FAIL |
| B2_MR_shorts_only | 5.371 | 4.381 | 4.381 | PASS |
| B3_MR_aggressive | 2.967 | 2.114 | 3.573 | PASS |
| B4_MR_selective | 0.000 | 0.000 | 0.000 | FAIL (no trades) |
| E_ABCombined | 3.265 | 2.186 | 4.557 | PASS |
| **G_PEAD_hold5** | **8.109** | **7.197** | **8.787** | **PASS** |
| D1_concentrated | 0.000 | 0.000 | 0.000 | FAIL (no trades) |
| D2_broad | 3.059 | 2.836 | 3.038 | PASS |
| F_AnalystRev | 2.569 | 1.877 | 1.877 | PASS |

The two configs marked bold are the proposed Phase I production combination.

---

## Key Architectural Decisions

1. **5-day PEAD hold:** chosen as the academically standard PEAD drift window. Tested only at hold = 5 in the WF runner that produced the headline 8.109 Sharpe; no sensitivity sweep on hold = {1, 3, 5, 10}.
2. **flags_required = 2 of 4 for QualityShort:** chosen because flags=1 produced a wider (noisier) candidate pool and flags=3 produced too few trades. A2 (the winning config) explicitly uses default flags_required = 2.
3. **SPY MA200 for regime:** simple, robust; the same rule disabled the long leg in 2022. We have **not** tested SPY > MA50 or VIX-based regime gates with comparable rigour.
4. **FMP for earnings:** FMP `earnings-surprises` endpoint provides quarterly EPS surprises. We treat the surprise date as PIT-safe; FMP typically populates this within hours of the earnings release.
5. **Russell 1000 universe:** chosen for liquidity. But the constituent list is **the current list as of build time** — no historical reconstitution.
6. **0.5%/year flat borrow cost:** borrow rates for hard-to-borrow names (which QualityShort explicitly targets) can be 5%–50%+ annualised.
7. **MAX_OPEN_POSITIONS = 5 (default RiskLimits):** this is for the live trader. The factor portfolio uses top-N = 20 (and `MAX_OPEN_POSITIONS` cap is overridden in the scorer's intended use). For PEAD with 300 trades/year at 5-day hold, the implied average concurrent open positions is ~6 — manageable. But during earnings-season clusters it could spike to 20+.
8. **5 bps per side transaction cost:** for Russell 1000 names this is roughly accurate for a $100k account in Alpaca's commission-free model with average spreads, but does not include slippage on aggressive orders.

---

## Known Limitations and Risks

We list these candidly so you can confirm or extend them.

### Data
- **Yahoo Finance** is unaudited, has occasional bad ticks, and has been known to have unadjusted splits/dividends on small caps. For Russell 1000 the quality is generally OK but not institution-grade.
- **FMP** PIT safety depends on `filingDate` being set correctly. We have not audited what fraction of rows fall back to `acceptedDate` or `period_end + 45d`.
- **Survivorship bias:** the universe is current Russell 1000 — companies that went bankrupt, were acquired, or fell out of the index over 2021–2026 are missing. For a long-only momentum book, this **inflates** results because winners stayed in and losers exited. For a QualityShort book, this likely **biases against** us in real terms (the worst shorts went to zero and aren't in the data) — but it also removes the most dangerous short squeezes from the sample.

### Methodology
- **No transaction costs above 5 bps/side and no slippage.** PEAD trades enter at the next-day open — this is precisely the time of day when liquidity is poor and effective spreads can be 10–30 bps on R1000 mid-caps. Realistic round-trip cost is likely 20–40 bps total, vs the 10 bps we model.
- **No market impact.** With $100k notional and 5% position caps ($5k positions), market impact in R1000 is negligible, so this assumption is fine at this scale. At larger scale, PEAD capacity is well-known to be limited.
- **No partial fills, no halts.** PEAD signals fire on stocks that just announced — halts and gap moves are real.
- **Borrow cost is a flat 0.5%/yr.** QualityShort targets levered, declining, negative-margin names — these are often on the "hard-to-borrow" list at 5–25%/yr.
- **No borrow availability constraint.** We assume any name can be shorted at any size, which is unrealistic for QualityShort candidates.
- **Multiple testing.** ~17 configs tested in Phase H/H+ alone, before that Phases F and G iterations. The 8.109 Sharpe for PEAD hold-5 is the **best** of many trials — the DSR adjustment we apply may not fully discount this.
- **Universe-conditioning bias.** The flags (operating_margin ≤ 0, etc.) are computed only on current R1000 names. A name that meets QualityShort flags **and** subsequently delisted is invisible to us.

### Strategy-specific
- **PEAD anomaly decay:** academic literature documents substantial PEAD decay since 2000. Sharpe 8 is far above any reported PEAD academic result (typical post-cost PEAD Sharpe is 0.5–1.5).
- **Earnings gap risk for PEAD:** we enter the day after the surprise; we don't enter into the gap itself. But the bars that follow can still gap on guidance changes, analyst reactions, etc.
- **Short squeeze risk for QualityShort:** by construction we are short the names with the highest debt/equity and worst fundamentals. These overlap heavily with the universe of high short-interest, meme-vulnerable names (GME 2021, BBBY 2022, AMC, etc.).
- **Regime gate lag:** SPY MA200 is a slow gate — by the time SPY drops through MA200, the bear market is well underway and we've already taken losses on long positions.

---

## Walk-Forward vs Reality: Specific Concerns

| Concern | WF assumption | Likely live reality |
|---------|---------------|---------------------|
| Entry fill | Next-day open, no slippage | 5–20 bps slippage on open auction for mid-cap PEAD names |
| Exit on max-hold | Mark to close | Close auction or VWAP execution, similar slippage |
| Borrow rate (QS) | 0.5%/yr | 2%–25%/yr for the kind of name QualityShort selects |
| Borrow availability | Always available | ~10–30% of QualityShort candidates may be locate-restricted on a given day |
| FMP data freshness | Assumed available on filing date | Real-time API can lag the press release by minutes-hours; rate limits on free tier |
| Yahoo Finance gaps | None modelled | Occasional missing bars / bad ticks for stocks around corporate actions |
| Earnings halt / pre-market gap | Ignored | PEAD names sometimes gap 10%+ overnight; our "next day open" already absorbs part of the drift |
| Concurrent position limit | Implicit (we don't hit MAX_OPEN_POSITIONS = 40 in proposed config) | Earnings clusters in Jan/Apr/Jul/Oct could exceed 40 simultaneously open PEAD positions |
| Position sizing | `size_position` uses stop distance | Live PM applies same logic, but conf-to-size mapping may differ subtly |
| Regime gate | Computed at end of bar | Live PM evaluates intraday; off-by-one-day differences possible |

---

## Questions for the Quant Reviewer

Please address these explicitly.

### Data integrity
1. Is Yahoo Finance acceptable for Russell 1000 daily OHLCV for a Sharpe-validation study, or do we need a paid source (e.g., Polygon, Norgate, CRSP)?
2. Is FMP `earnings-surprises` data truly PIT-safe in our usage? Specifically, does the `date` field on that endpoint represent the announcement date (PIT-safe) or a populated-at date (potentially look-ahead)?
3. For `operating_margin`, `revenue_growth_yoy`, `debt_to_equity` from FMP quarterly statements: are these most-recent-quarter or TTM in our parquet? (Code suggests **most-recent-quarter**, with growth computed against same fiscal quarter prior year.)
4. What is the realistic latency between earnings press release and FMP API populating the surprise field?
5. Could there be hidden look-ahead from `filingDate` vs `acceptedDate` fallback when filingDate is missing?

### Walk-forward validity
6. Is 5 folds × 1-year test window adequate for a strategy doing 300 trades/year? (≈1500 trades total in OOS — sounds OK on the surface, but folds 4 and 5 partially overlap with prior train data via the embargo boundary if the embargo is too short.)
7. Does `purge_days = 10` adequately prevent leakage given the 5-day PEAD label? (We argue yes, but please confirm.)
8. The reported Sharpe of 8.1 is ~5σ above plausible. Is this a code bug (e.g., daily-returns scaling, ann factor mistake, double-counting), a parameter overfit, or a real effect at the simulated capital scale where capacity is unbounded?
9. Are the fold-2 (2022 bear) Sharpes of 8.787 (PEAD) and 6.773 (QS) suspicious — they are **higher** than other folds, which is the opposite of what we'd expect for momentum-driven strategies?
10. With ~17 configs tested, what is the appropriate multiple-testing correction (DSR with N_trials ≥ 17, or harsher)?

### PEAD specific
11. PEAD academic literature reports Sharpes of 0.5–1.5 post-cost since 2000. What might explain a 5–10× higher Sharpe in our setup? Candidates: (a) we're not paying realistic costs, (b) overfitting to fold dates, (c) survivorship in the universe, (d) PIT-leakage we haven't identified, (e) actual return-magnitude inflation from compound geometric returns on small positions.
12. The total returns per fold (11x–38x) imply daily compounding of a ~5% position into many concurrent winners. Is this plausible, or is the equity curve being constructed in a way that double-counts?
13. Is 5 bps/side cost realistic for PEAD names entering at next-day open?
14. What happens to PEAD signal performance if we delay entry by 1 more day (does the drift survive a slower entry)?

### Quality Short specific
15. The QualityShort win rate of 21–45% with high Sharpe — does the equity curve reconstruct correctly with the borrow cost model? Is borrow being capitalised correctly?
16. What's a realistic borrow-cost distribution for stocks meeting our flag criteria (op margin ≤ 0, D/E ≥ 1.5, negative surprise)?
17. How exposed is QualityShort to short squeezes? In Fold 2 (2022 bear), the strategy was the **easiest** environment for shorts and Sharpe was highest. In Fold 1 (2021–22 incl. meme-stock era), Sharpe is lowest (3.958). Is the strategy really robust to a repeat of Jan 2021 meme dynamics, or are squeezes underrepresented in fold-2 because the universe filter happens to exclude them?

### Portfolio construction
18. PEAD generates ~300 trades/year at 5-day hold → average ~6 concurrent positions, but earnings clusters could spike well above 20. Proposed Phase I uses PEAD + QualityShort + Factor longs simultaneously. What is the expected concurrent position count, and is `MAX_OPEN_POSITIONS = 40` sufficient? Should risk be re-budgeted across legs?
19. Could PEAD-short and QualityShort short the same name on the same day, doubling exposure? Is there any de-duplication?
20. The factor-long book's regime gate (SPY > MA200) sits the longs in cash during bear markets, while PEAD trades continue. Is this the right allocation, or should PEAD also be regime-gated?

### Risk management
21. Walk-forward gate is `min_sharpe ≥ −0.3` for L/S but the factor portfolio originally used `-1.0`. Inconsistency — does this matter for our top-2 picks (both blow past the threshold by orders of magnitude, but is the gate calibration meaningful at all)?
22. The 20% circuit-breaker stop on a 5-day PEAD trade: too wide? Too narrow? In practice exits happen on max-hold, but in a fat-tail event a 20% gap is bigger than typical 1-day moves.
23. Borrow cost 0.5%/yr — please give an order-of-magnitude estimate of true cost for the QualityShort universe.

### Paper trading plan
24. Is 3 months sufficient to validate Sharpe statistically, given the trade frequencies (PEAD ~1.2 trades/day; QualityShort ~0.2 trades/day)? What confidence interval can we expect on Sharpe from N≈90 trading days?
25. Given typical 30–50% live/sim Sharpe degradation, what Sharpe should we expect to see in paper for it to be plausibly the same strategy as the WF result?
26. What kill-switch criteria would you recommend? (Cumulative drawdown, weekly Sharpe trigger, "max consecutive losses", etc.)

---

## Self-Critique (from the author, Claude Opus 4.7 acting as senior quant)

### What I believe is genuinely strong
- **PEAD is a real anomaly.** The economic mechanism (delayed information assimilation, attention-constrained investors) is documented in 30+ years of literature.
- **QualityShort uses fundamental flags, not pure momentum.** This is a structurally different short signal than "bottom-N by momentum" — which failed catastrophically in 2022 — and should reverse less violently in bear-market rallies.
- **The regime gate on the long book is honest.** It sits the longs in cash in 2022 rather than pretending the long book "should" trade through the bear market.
- **Fold 2 (2022 bear) is the strongest test.** That both strategies posted their best Sharpes in 2022 is unusual but consistent with: PEAD beats/misses widen in volatile regimes, and QualityShort gets paid by macroeconomic pressure on weak balance sheets.
- **PIT safety is taken seriously** in the FMP integration (filingDate is the correct anchor).

### What concerns me most
- **Sharpe 8.1 is implausible by any historical reference.** No multi-year, post-cost PEAD strategy has ever published a Sharpe near 8. This strongly suggests one or more of: (a) cost model is too generous, (b) total-return inflation from geometric compounding of small-position winners is being mis-Sharpe'd, (c) the universe/PIT setup has subtle look-ahead, (d) survivorship bias.
- **QualityShort max drawdowns of 0.01%–0.69%.** A genuine short book of 15 levered, declining names cannot plausibly have max DD < 1% over a full year — this is a red flag. Likely the equity curve is being driven by a few enormous wins that mask the volatility, OR the position sizing is so small that the equity curve barely moves.
- **17+ trials, two winners.** Selection bias on the strategy-selection step is enormous.
- **The total returns are unphysical at $100k.** 38x in a year means $100k → $3.8M. Either the simulator is treating position sizing geometrically without realistic capacity constraints, or the strategy genuinely produces this — which would be the most exploited trade on Earth.

### What additional tests I would run before going live
1. **Re-run PEAD WF with 30 bps round-trip cost** and 10 bps slippage at open. If Sharpe drops to <2, the original number is cost-fragile.
2. **Re-run with hold = {1, 3, 5, 7, 10} bars** — sensitivity to the single hyperparameter we already chose.
3. **Re-construct the universe point-in-time** from a delisted-inclusive source (e.g., Polygon delisting endpoint or Norgate) and rerun. If Sharpe collapses, survivorship was a major driver.
4. **Audit the equity-curve construction.** Specifically verify the daily-return series used in Sharpe is portfolio-equity-based (not trade-pnl-summed) and not double-counting borrow rebates.
5. **Stress test QualityShort with realistic borrow costs** — apply 10%/yr flat instead of 0.5%/yr and see what the Sharpe degradation looks like.
6. **Audit positions on the worst PEAD days** — manually trace ~10 sample trades through the simulator to confirm entry/exit logic matches PEAD academic methodology.
7. **Out-of-time validation:** hold out a clean ~6 months (e.g., 2026-05 to 2026-11) and don't touch it until the strategy is live. The current fold 5 ends at 2026-05-19 — only ~today.
8. **Cross-validate fold assignment:** the fold dates landing on 2022-05/2023-05 etc. may be coincidentally favourable; re-run with different rolling start dates.

### Probability estimate

**Probability that the proposed deployment achieves live-paper Sharpe > 0.50 over a 3-month period: ~55–65%.**

Justification:
- PEAD is a real anomaly; even with 70–80% Sharpe degradation from 8.1 → 1.5–2.5, that comfortably exceeds 0.50.
- However, my confidence is low because the headline numbers are so far outside the historical reference distribution that I cannot rule out a methodological bug that would zero out the strategy entirely.
- QualityShort is more concerning — the near-zero drawdowns suggest the equity curve may not be measuring what we think it is.
- 3 months × ~250 PEAD trades = adequate sample size, but a single bad earnings week could blow through the apparent edge.
- If I had to bet real money: 60% chance live Sharpe > 0.50, 30% chance live Sharpe between 0 and 0.50, 10% chance live Sharpe < 0 (the strategy is broken or the headline is a bug).

The single most important action before going live is to **audit the equity-curve construction and re-run with realistic costs**. If the Sharpe survives that with > 1.5, the deployment is sound. If it does not, we are in front-running our own backtest bugs.

---

*End of document. Reviewer: please return your assessment in the same structural order as the Questions section, with explicit reasoning and quantitative estimates where possible.*
