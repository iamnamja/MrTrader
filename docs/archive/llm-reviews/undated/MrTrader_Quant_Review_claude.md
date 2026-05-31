# MrTrader Quant Review — Independent Assessment

**Reviewer posture:** Senior quant, 15+ yrs systematic equities. Skeptical by default.
**Brief:** Find every flaw, weakness, false assumption, and hidden risk. Be precise. Be technical.
**Date of review:** 2026-05-19

---

## TL;DR — The 30-Second Verdict

> **Do not go to paper trading yet.** The headline Sharpe of 8.1 and per-fold total returns of 11x–38x are not "too good to be true" — they are almost certainly **arithmetically impossible** at the stated capital and sizing. Before any deployment decision, the single highest-leverage action is to **forensically audit the equity-curve construction in `AgentSimulator`**. There is a >90% probability that a portfolio-vs-trade-return accounting error is inflating both the numerator (returns) and the denominator interaction (vol scaling) of Sharpe. Until this is resolved, every other metric in the WF report is uninterpretable — including the WF gates, the DSR, and the strategy comparison table.

**If I had to bet real money on the three most likely root causes, in order:**

1. **(highest probability, ~65%)** The reported daily return series is computed as a sum/product of per-trade returns at "per-trade notional" rather than "portfolio equity," so a 1.2% trade return on a 5% position is being recorded as a 1.2% portfolio day rather than a 0.06% portfolio day. Sharpe and total return inflate by ~20x, which matches what we see.
2. **(~15%)** Position sizing in the simulator uses **available cash** as the denominator, so when 6 positions are open and only 70% of cash is deployed, the next trade is sized to 5% of remaining cash — but the equity curve treats this as 5% of starting capital. Compounds non-physically.
3. **(~10%)** PIT look-ahead in the FMP earnings join — specifically the `acceptedDate` or `period_end + 45d` fallback ordering. When the primary `filingDate` is missing, the fallback may resolve to a date **after** the press release became public, but the join is being keyed on a date that includes information the market did not have at `as_of`.

The remaining ~10% probability is distributed across: survivorship, the cost model, fold-date luck, and the various smaller items below.

**My calibrated probability that the proposed Phase I paper config achieves observed Sharpe > 0.50 over a 3-month live paper period: ~35–40%.** This is materially lower than the author's self-estimate of 55–65%, because I weight the "headline-is-a-bug, true strategy is mediocre or negative" tail more heavily.

---

## Part 1 — What's Actually Wrong (Critical Findings)

### Finding 1: The Sharpe Number Is Not Physically Possible

Reference points for a Sharpe of 8.0+ in equities:

| Reference | Sharpe (gross) | Notes |
|---|---|---|
| Renaissance Medallion (1988–2018) | ~7.5 | $10B internal cap, 80+ PhDs, alternative data, ultra-fast execution |
| Two Sigma (statistical arb book) | 2–4 | Top-tier published estimates |
| Citadel Tactical Trading | 3–5 | Discretionary + systematic blend |
| AQR Style Premia (academic) | 0.8–1.2 | Multi-factor, published live track record |
| PEAD academic factor (Bernard-Thomas, Chordia, Sadka et al.) | 0.4–1.5 | Pre-cost; post-cost typically <1.0 |
| Top hedge fund earnings surprise factor (post-cost, capacity-constrained) | 1.0–2.0 | Reported by GS / MS prime brokerage research |

**A single-asset-class, single-factor, retail-data, retail-execution PEAD strategy at Sharpe 8.1 has zero analog in public literature.** This is not a "the strategy is finally working" situation. It is a "the measurement is wrong" situation, period.

**Quantitative sanity check on the impossibility:**

PEAD's economic mechanism is delayed information assimilation — i.e., the post-announcement price drift over 60 days captures roughly 5–7% in academic studies (Bernard & Thomas 1989) for the top decile of surprise, declining substantially post-2000. Over a 5-day window, the residual drift is perhaps 1–2% gross of cost. On a 5% position size, that's **5–10 bps per trade of portfolio return**. At 300 trades/year with a 58% win rate, expected annual return is roughly:

- E[trade PnL] ≈ (0.58 × 1.5%) − (0.42 × 1.5%) ≈ 0.24% per trade gross — i.e. an edge of about 24 bps per trade.
- On 5% position size: 0.012% portfolio return per trade.
- 300 trades/yr: ≈ **3.6% annual portfolio return** gross. Sharpe perhaps 0.8–1.5 if vol is moderate.

The reported result is roughly **300x** this estimate. There is no economic mechanism that produces a 300x amplification.

### Finding 2: The Total Returns Confirm the Bug

| Fold | Reported total return | Implied daily log-return | Implied per-trade portfolio PnL (303 trades) |
|---|---|---|---|
| 2 | 38.61x | 1.45% | 1.20% |
| 3 | 11.90x | 0.98% | 0.82% |
| 5 | 38.26x | 1.45% | 1.20% |

A 1.20% portfolio gain on every single PEAD trade implies that **every trade is the entire portfolio**, not 5% of it. This is the math signature of trade-return-treated-as-portfolio-return.

**Alternative reading I considered and rejected:** "Maybe the simulator is using compounding with no position cap, so each trade scales with current equity." This still doesn't work because the per-position cap is hardcoded at 5% (`MAX_POSITION_SIZE_PCT`). Either the cap is being silently overridden in the factor-portfolio code path, or the equity curve construction is broken.

**Hypothesis to test:** Inspect `AgentSimulator._compute_daily_returns()` or equivalent. Look specifically for:
- Whether the return series is `realized_pnl / starting_equity` (correct) or `realized_pnl / position_notional` (broken).
- Whether the daily series is built by **stitching** non-zero return days from trades, or by **filling in** zero-return days when no trades close.
- Whether annualization is `sqrt(252)` or `sqrt(N_trades)` (the latter would explode).

### Finding 3: QualityShort DD < 1% Is the Second Smoking Gun

A levered short book of 15 names with operating margins ≤ 0, D/E ≥ 1.5, and negative earnings surprises has historically produced **15–40% drawdowns** in any 12-month window since 2010. Specifically:

- **Jan 2021 meme rally:** A short book matching the QS filter would have seen single-day losses of 15–25% on GME / AMC / BBBY exposure.
- **Mar–Apr 2020 COVID rally:** Junk-rated, high-leverage shorts squeezed 30–60% in 5 trading days.
- **Mar 2023 SVB-fueled small-cap rally:** Regional bank shorts saw 20%+ snapback in 3 days.

QualityShort reporting max DD of 0.01% in Fold 2 (which includes 2022 — when shorts performed *best* — but also reaches into 2023 when the easy money was already made) means one of:

1. **The book is essentially empty** in this period (the universe filter is too tight and almost no trades fire). 29–61 trades/yr over a R1000 universe of "deterioration flags ≥ 2" is suspiciously low; in the post-Fed-tightening fundamental landscape of 2022–23 there should have been 100s of candidates.
2. **Position sizing is microscopic** relative to the equity denominator (e.g., $50 positions on $100k base).
3. **The equity curve is summing only realized PnL** and not marking open positions to market — so paper drawdowns during open trades are invisible.

The Fold 1 (Sharpe 3.96, only 29 trades, win rate 44.8%) is particularly worth scrutinizing. **29 short trades in a year, across R1000, in a market that contained Jan-2021 meme dynamics, with a 44.8% win rate but a 3.18x return** — the only way 29 trades return 3.18x is if average winners are massively larger than average losers, AND positions are extremely large. Doesn't match max DD = 0.03%. **Internally inconsistent.**

### Finding 4: PEAD Fold-2 = Best Result Is Anti-Economic

Setting aside the absolute level of Sharpe, the *pattern* across folds is its own diagnostic:

| Fold | Period | PEAD Sharpe | What we'd expect for a *real* PEAD strategy |
|---|---|---|---|
| 1 | 2021–22 (post-COVID, meme era) | 8.427 | Moderate; meme dynamics disrupt drift |
| 2 | 2022–23 (bear, Fed tightening) | **8.787** | **Weakest** — macro selling overwhelms surprise drift; PEAD post-cost was actually negative in many published studies for 2022 |
| 3 | 2023–24 (AI bull, narrow leadership) | 7.197 | Moderate; concentration in mega-caps reduces breadth of opportunity |
| 4 | 2024–25 (broadening rally) | 8.058 | Should be strongest |
| 5 | 2025–26 (current, mostly bull) | 8.077 | Should be moderate-to-strong |

The fact that **2022 is the strongest fold** is not just suspicious — it's diagnostic. A bug that scales linearly with trade count or with volatility (variance is highest in 2022, hence highest absolute returns get magnified) would produce exactly this pattern. A real momentum-adjacent strategy would not.

### Finding 5: The "Next-Day Open" Entry Is Suspect for PEAD

PEAD academic literature is clear: **most of the 5-day drift is captured in the overnight gap from announcement to next-day open**. So:

- **If you enter at T+1 open** (per the doc), you're entering AFTER the overnight gap, which means you've MISSED the largest chunk of the drift, and your residual edge over 5 days is ~50–100 bps gross.
- **If you enter at T close** (look-ahead, because the announcement is after-hours), you would capture the gap — but the code claims this isn't what's happening.

So either:
- (a) The strategy is doing what it claims (T+1 open entry), the residual drift is real but small, and the headline Sharpe is fully attributable to the equity-curve bug.
- (b) The strategy is silently entering on T close (or using T close prices in the join), creating look-ahead, which inflates the per-trade edge from ~50 bps to ~250 bps. This would not by itself explain Sharpe 8 (still needs the bug), but it would inflate the realistic-cost-corrected estimate substantially.

**You need to manually trace 10 PEAD signals**, identifying the precise bar used for entry price vs the precise bar of signal generation. This is a half-day of work and is critical.

### Finding 6: Multiple Testing — But Not the Way You Think

The author identifies "17+ configs tested, 2 winners" as a selection bias problem. **The standard DSR adjustment is the wrong frame here.** Here's why:

DSR assumes that each trial is drawing from a distribution of *real* Sharpes. If the headline is a bug, all 17 trials draw from the bugged distribution. The two "winners" aren't the lucky tails of a true-Sharpe distribution; they're the configurations where the bug compounds most aggressively (more trades → more compounding error → higher apparent Sharpe).

**Evidence supporting this reading:** Look at the Phase H+ table. The two configs with the highest trade counts (PEAD at 300/yr, QS at 53/yr — but QS at near-zero DD so the bug expression is different) are the two winners. The "failed" configs are mostly ones where trade count was near zero (B4_MR_selective and D1_concentrated both report exactly 0.000 Sharpe — they didn't trade enough to express whatever bug is present).

**Implication:** Until the equity curve is audited, *the rankings between configurations are meaningless* — you might be picking the configurations that are most bug-amplified, not the configurations that are most economically valid.

### Finding 7: Cost Model — Optimistic, but Not Catastrophic Alone

If costs were the *primary* issue, the realistic adjustments would be:

| Cost component | Modeled | Realistic for PEAD on R1000 mids | Impact on Sharpe (if everything else right) |
|---|---|---|---|
| Commission | 0 bps (Alpaca) | 0 bps (correct for Alpaca) | None |
| Half-spread at open | 0 bps | 8–15 bps | ~20–30% Sharpe reduction |
| Slippage on aggressive market orders at open auction | 0 bps | 5–12 bps | ~10–15% Sharpe reduction |
| Modeled "transaction cost" | 5 bps/side | already covers some of above | — |
| **Realistic total round-trip cost** | **10 bps** | **25–40 bps** | **25–50% Sharpe haircut** |

For QualityShort, borrow is the bigger issue:

| Borrow scenario | Annualized cost on $100k notional short | Effect on QS Sharpe |
|---|---|---|
| Modeled (0.5%/yr) | $500/yr ≈ negligible | baseline |
| GC (general collateral) names | 25–75 bps/yr | ~5% Sharpe reduction |
| HTB (hard-to-borrow) names | 5%–25%/yr | **30–70% Sharpe reduction** |
| Squeeze candidates (GME-like) | 50%–500%/yr | strategy is uneconomic on those names |

QualityShort by construction selects from the HTB universe. Realistic blended borrow is probably 8–15%/yr, not 0.5%. **This alone would crater the QS Sharpe by half.** But again — costs don't take Sharpe 6 to Sharpe 0.6. They take Sharpe 6 to Sharpe 3. The bug is doing the heavy lifting on the implausibility.

### Finding 8: Survivorship Bias — Real, but Nuanced

The author has the direction right but understates the magnitude. Let me be specific:

- **Russell 1000 has roughly 90–110 reconstitution events per year** (additions + deletions), plus M&A and bankruptcies between reconstitutions. Over 2021–2026, that's ~500–600 universe changes.
- A "current constituents only" universe misses approximately 30% of the actual trading opportunity space over a 5-year backtest.

**Net effect on each strategy:**

| Strategy | Direction of bias | Magnitude |
|---|---|---|
| PEAD longs (beats → drift up) | Inflates ⬆ | Companies that beat consistently stay in R1000; those that beat then deteriorated and dropped out are missing. **+10–20% Sharpe** |
| PEAD shorts (misses → drift down) | Deflates ⬇ but also removes tail risk ⬆ | Worst misses (delisted) absent. Net: probably modest **+0–10% Sharpe** because the "wins to zero" are absent but so are the squeeze losses |
| QualityShort | Strongly removes tail risk ⬆ | Names that meet the filter and went bankrupt would be wins (good for us, but absent). Names that met the filter and squeezed are also absent (bad for the real-world strategy but absent here). **Net effect probably +20–40% on Sharpe** because squeeze tails are removed more than zero-outcome wins are. This is consistent with the implausibly-low max DD. |

**The remediation here is non-trivial.** You need a point-in-time membership source. Polygon, Norgate, and CRSP all have this. The free workaround is to scrape index reconstitution announcements from FTSE Russell's press releases and back-fill — but this is ~40 hours of work.

### Finding 9: PIT Audit Gaps in FMP Integration

The author flags this but understates the importance. Specific risks:

1. **`filingDate` vs `acceptedDate`:** SEC's EDGAR has a documented quirk where `filingDate` is the date the company filed (could be after-hours) but `acceptedDate` is when SEC accepted (possibly next business day). If your fallback is `filingDate → acceptedDate → period_end + 45d`, and `filingDate` is null for the *most-recent* filings (because FMP's pipeline hasn't backfilled), you may get `acceptedDate` for recent rows and `filingDate` for older rows — creating a **temporal inconsistency in the join** that's hard to detect.

2. **The `earnings-surprises` endpoint specifically:** I am skeptical that the `date` field on this endpoint is the press-release date. In FMP's data model, this is more commonly the date the company reported AT — but it could be the announcement *date* (calendar date of release, which is what you want) or the *populated-at* date (when FMP ingested it, which would create look-ahead because FMP backfills historical surprises).

3. **What fraction of rows fall back to `acceptedDate` or `period_end + 45d`?** You haven't audited this. If it's >5%, the leakage could be material.

**Concrete test:** Print, for the most-recent 200 PEAD trades, the entry date, the `as_of_date` of the FMP row consulted, and the `filingDate`/`acceptedDate`/`period_end` columns. If you see cases where `as_of_date` < calendar date the surprise became public knowledge, you have look-ahead.

### Finding 10: The Author's Self-Reported Bear-Market Behavior Is Inconsistent

The doc says:
> "The regime gate on the long book is honest. It sits the longs in cash in 2022..."

But also:
> "Fold 2 (2022 bear) is the strongest test. That both strategies posted their best Sharpes in 2022..."

These are simultaneously true only if the factor-long book is **separate from** the PEAD/QS books, and only the PEAD/QS books trade in 2022. Fine. But then:
- The "honest" regime-gate claim only applies to ~30% of the strategy stack (the factor longs)
- PEAD trading through 2022 *was* in the bear market — that's not a regime-gated strategy
- The 2022 Sharpe of 8.787 for PEAD includes all of 2022's macro selling pressure, which should hurt longs and help shorts, but the doc doesn't break out long vs short PEAD performance in 2022

**Ask:** What was the long-only PEAD Sharpe in Fold 2 vs the short-only PEAD Sharpe in Fold 2? If the long leg alone produced 8+ Sharpe in 2022, that's an even bigger anomaly than the headline.

---

## Part 2 — Direct Answers to the 26 Questions

### Data Integrity

**Q1 — Yahoo for R1000?** Acceptable for a *Sharpe validation* study at this stage, but not for production. Yahoo has known issues with: split-adjusted prices on small caps for 2–3 days post-split, missing bars on names experiencing trading halts (which PEAD names sometimes do), and occasionally egregiously bad ticks on illiquid prints. For your $100k paper deployment, Yahoo is probably acceptable. **For any decision to scale to real capital, switch to Polygon or Tiingo.** The cost is $30–$200/month.

**Q2 — FMP `earnings-surprises` PIT safety?** Cannot be confirmed from the doc alone. **Action required:** Email FMP support and ask explicitly: "On the `earnings-surprises` endpoint, what does the `date` field represent — the press release date, the company-reported date, or your ingestion date?" My prior is 70% it's the press release date (which is PIT-safe in the sense that the surprise is public information from that moment) and 30% it's a populated-at date.

**Q3 — TTM vs MRQ for FMP fundamentals?** If the parquet has columns named `operating_margin` directly (no `_ttm` suffix), it's almost certainly MRQ. Verify by joining one company's row against the actual 10-Q filing on EDGAR for that quarter. MRQ is fine for your use case (it's what fundamental short strategies typically use), but it has higher volatility than TTM, which makes the `<= 0` threshold more trigger-happy.

**Q4 — Press-release to FMP API latency?** Anecdotally minutes to hours for major names; up to 24 hours for mid-caps. This is **not a PIT issue** for backtesting (you're using historical data that's all backfilled), but it IS an issue for **live trading** — your live signal will arrive on a delay vs the academic ideal, which costs you the early-drift portion of the trade.

**Q5 — `filingDate` vs `acceptedDate` look-ahead?** Possible. See Finding 9. **Action:** Audit fallback frequency in your parquet.

### Walk-Forward Validity

**Q6 — 5 folds × 1-yr test for 300 trades/yr adequate?** Marginal. ~1500 OOS trades is a decent sample size for Sharpe estimation in absolute terms, but folds are not independent — adjacent folds share market regime, factor exposures, and even (with 10-day embargo on a 5-day label, you're cutting it close) some trade-level information. **Real "effective sample size" is probably more like 300–500 independent observations.**

Better approach: **Combinatorial Purged Cross-Validation (CPCV)** as in López de Prado's "Advances in Financial Machine Learning" Ch. 12. Generates ~50 backtest paths instead of 5, with proper purging. Strictly more powerful. ~2 weeks of engineering work to implement.

**Q7 — Purge_days = 10 vs 5-day PEAD label?** Adequate for the PEAD signal itself. The 10-day purge gives you 5 days of "post-trade" buffer beyond the label horizon. Fine.

**Q8 — Sharpe 8.1 — bug, overfit, or real?** **Bug, by a wide margin.** See Findings 1–4. The remaining possibilities (overfit, real effect at small capital) are quantitatively insufficient to bridge the gap from a plausible ~1.5 to 8.1.

**Q9 — Fold-2 (2022 bear) highest Sharpe — suspicious?** **Extremely suspicious.** See Finding 4. This is one of the strongest diagnostic signals that the metric is broken. Real PEAD typically has its *weakest* performance in macro-driven bear markets because the cross-sectional information content of surprises gets dominated by aggregate beta moves.

**Q10 — Multiple testing correction?** The standard DSR with N=17 is OK as a frequentist adjustment, but **it's the wrong question right now.** See Finding 6. Until the bug is resolved, DSR is calibrating on a contaminated trial distribution. After the bug is resolved and you re-run all 17 configs cleanly, *then* apply DSR with N ≥ 30 (to account for the Phase F/G iterations too).

### PEAD Specific

**Q11 — Why is our PEAD Sharpe 5–10× academic?** Almost certainly the equity-curve bug (Finding 1). The other candidates the author lists (cost realism, fold overfit, survivorship, PIT leakage) each contribute single-digit-percent Sharpe inflation. None of them gets you from 1.5 to 8.1. Only a denominator/numerator scaling error of that magnitude can.

**Q12 — Are the 11x–38x total returns plausible?** No. See Finding 2. The arithmetic doesn't work at the stated position sizing.

**Q13 — Is 5 bps/side realistic for PEAD at next-day open?** No. Realistic is 8–15 bps half-spread plus 3–8 bps slippage = ~15–25 bps per side. **Round-trip realistic cost = 30–50 bps**, not 10 bps. This costs you about a third of the Sharpe (once the bug is fixed).

**Q14 — Does PEAD survive a slower entry?** Empirically yes, but with materially reduced edge. Bernard-Thomas-era literature showed drift extending up to 60 days; modern studies (since 2010) show most drift compressed into 1–10 days. **Test it explicitly:** rerun with entry shifted to T+2 open, T+3 open, T+5 open. Expect monotonic Sharpe decay. If you don't see monotonic decay, that's evidence of either bug or look-ahead.

### Quality Short Specific

**Q15 — Borrow capitalization correct?** Cannot be confirmed without code-level inspection. The 0.5%/yr flat rate is wrong on its face (Finding 7), but separately verify the *mechanism* by which it's deducted from equity. Common bug: borrow is computed correctly per-trade but not aggregated into the daily return series.

**Q16 — Realistic borrow distribution for QS filter universe?** From prime broker rate sheets (this is my professional read):
- ~50% of QS candidates are GC names with borrow 25–100 bps/yr.
- ~30% are mildly HTB at 2–8%/yr.
- ~15% are HTB at 8–25%/yr.
- ~5% are deep HTB or locate-restricted at 25%+ or unavailable.

**Blended realistic borrow on QS book: ~5–10%/yr**, not 0.5%. Use 8%/yr as a stress-test conservative.

**Q17 — Short-squeeze exposure?** Severe, and the WF likely under-represents it due to survivorship (Finding 8). The fact that Fold 1 (which includes the late-2021 meme aftermath) has the *lowest* QS Sharpe (3.958 — still high in absolute terms, but lowest in relative terms) is consistent with squeeze risk biting precisely when you'd expect.

**Realistic stress test:** Manually identify a list of 2021–2022 squeeze events (GME Jan 2021, AMC Jun 2021, BBBY Aug 2022, NKLA Mar 2022, CVNA late 2022, etc.). For each, check whether your QS filter would have flagged the name *before* the squeeze. If yes, model a 50–100% adverse move on that name and see what it does to the equity curve. I'd bet 5+ of those names would have been in your filter.

### Portfolio Construction

**Q18 — Concurrent position spike in earnings season?** Yes, and `MAX_OPEN_POSITIONS = 40` is probably enough on calendar average but **tight during the peak weeks of Jan/Apr/Jul/Oct earnings seasons**. Specifically, in the second week of earnings season, you could easily have:
- 25 active PEAD positions (5 days × ~5 signals/day on heavy days)
- 15 QS positions
- 20 factor longs (when regime gate allows)
- = 60 positions

You're capacity-constrained. Either raise the cap to 60 or implement a hard ranking-based selection (only top 25 by conviction at any time).

**Q19 — PEAD-short and QualityShort doubling exposure?** Unclear from doc, likely YES based on the description. Both fire on the same FMP `fmp_surprise_1q` field for negative-surprise names. **Action:** Add explicit de-duplication that checks if a name is already open in the other strategy and skip/scale-down.

**Q20 — Should PEAD also be regime-gated?** Mixed evidence. Academic PEAD does work in bear markets but with reduced magnitude and higher variance. Your Fold 2 result is suspect (Finding 4), so we can't lean on it. **Conservative recommendation: gate PEAD-longs by SPY > MA200 but leave PEAD-shorts un-gated.** This is the asymmetric treatment that the real-world setup typically uses.

### Risk Management

**Q21 — Inconsistent WF gate (−0.3 vs −1.0)?** The inconsistency is a code-quality red flag (suggests configuration drift), but functionally irrelevant for your top-2 picks which blow past either threshold. **More important question: the gate is calibrated against contaminated data, so a "PASS" today doesn't mean what it would mean after audit.**

**Q22 — 20% circuit-breaker on 5-day PEAD trade — appropriate?** Too wide for normal moves, too tight for fat-tail single-name disasters (e.g., accounting fraud reveal, FDA rejection on a biotech). A more standard PEAD-trade stop would be **2× ATR_20** on entry, which adapts to the name's volatility. The 20% level will essentially never bind for typical R1000 names (whose 5-day vol is more like 4–8%) — meaning your effective stop is the max-hold, which is what you said the doc indicates anyway. So the 20% stop is more cosmetic than risk-managing.

**Q23 — Realistic borrow cost on QS universe?** See Q16. **5–10%/yr blended is the right order of magnitude, not 0.5%.**

### Paper Trading Plan

**Q24 — Is 3 months enough?** Statistically, barely. With ~63 trading days × 1.2 PEAD trades/day = ~75 trades, the standard error on your Sharpe estimate is roughly:

- SE(Sharpe) ≈ sqrt((1 + 0.5×Sharpe²) / N) where N is in years
- For N = 0.25 years and an assumed real Sharpe of 1.0: SE ≈ sqrt(1.5 / 0.25) ≈ 2.45
- 95% CI on observed Sharpe = real ± 4.9

In other words, even a true Sharpe-1.5 strategy could report observed Sharpe anywhere from −3.4 to +6.4 over 3 months. **3 months is not enough to reject the null that the strategy is broken** — but it IS enough to detect catastrophic failure (e.g., a Sharpe-of-the-true-strategy < 0).

Recommended: **6 months of paper minimum, 9 months preferred** before any decision to deploy real capital.

**Q25 — Expected paper Sharpe assuming 30–50% live/sim degradation?** This is *not* the right framing once you accept the bug hypothesis. If the bug is fixed and the "true" Sharpe is 1.0–2.0, the live/sim degradation from sim is essentially zero (you're already at the realistic answer). If the bug is *not* fixed and the headline 8.1 stands going into paper, you should expect a paper Sharpe of **0–2.5 with very wide error bars** — not 4.0 (which 50% degradation would suggest).

**Q26 — Kill-switch criteria?** Strong recommendation, in priority order:

1. **Daily P&L distribution check** (weekly): Compare actual trade-level P&L distribution to backtest distribution via Kolmogorov-Smirnov test. If p < 0.01 after 4 weeks, halt and audit. This is the most sensitive early-warning signal.
2. **Per-trade outlier:** Any single trade with realized P&L < -3% of starting equity (i.e., $3000 loss on a $5000 position) gets a manual review before next trade in same name.
3. **Cumulative drawdown:** Halt at 5% portfolio DD on Phase I. (You said max DD = 2% in backtest; if real-world DD is 2.5× the backtest, you're already at the kill line.)
4. **Realized Sharpe (rolling 4-week)** < 0: pause new entries, allow exits to play out, audit before resuming.
5. **Consecutive losing days:** 7 in a row → pause. Statistical probability of 7 consecutive losing days at a 58% win rate is ~3%, so a real strategy will trigger this ~5–8 times per year; calibrate accordingly (don't kill on first trigger, but require explicit review).
6. **Concurrent position breach:** If position count exceeds 80% of `MAX_OPEN_POSITIONS` for 3+ days, you're in a capacity regime the backtest doesn't characterize. Pause new entries.

---

## Part 3 — What's Missing from the Author's Self-Critique

The self-critique is unusually good for this stage of work — most retail systematic traders never get here. But it understates several issues:

### 1. The Bug-vs-Cost Hierarchy

The author's self-critique frames costs and overfitting as comparable concerns. **They're not on the same order of magnitude.** The bug is 100x more important than the costs. Realistic cost adjustments take Sharpe from X to ~0.5X. The bug fix could take Sharpe from 8.1 to 0.8 — that's a 10x. Sequence the work accordingly.

### 2. The Lack of an "Intermediate Sanity Check"

You jump from "backtest 5 folds" to "deploy to paper." Industry practice has at least one intermediate step:

**Out-of-time forward test on cached data, with the live agent code path** (not the WF simulator). I.e., run the production PM/RM/Trader code path against historical data for the most recent 60 days, with full live-system features (rate limits, intra-day signal evaluation, position management). This catches differences between the WF simulator and the live agent code that would otherwise only manifest in paper trading.

### 3. No Mention of Latency Modeling

You've modeled costs, slippage (sort of), and borrow. You have not modeled **signal latency**: the time between FMP populating the surprise field and your scorer running. In live trading, this is a non-trivial delay. PEAD edge decays with latency, so the actual edge available to you in live trading is some fraction of the backtest edge based on how fast your pipeline runs.

**Question to answer:** Between earnings press release and the time your live system would *act* on the signal, how many minutes elapse? If it's >30 minutes, you've already lost the morning drift on day T+1. If it's >2 hours, you've lost most of the day-1 portion of the drift.

### 4. No Mention of Order Type and Fill Realism

"Entry at next-day open" — what order type, exactly? Market-on-open (MOO)? Market order at 9:30:00? Limit order at the prior-close price?

- **MOO orders** participate in the opening auction. Fills at the auction print, which is usually within the bid-ask spread but with capacity limits. For mid-cap R1000 names, MOO is generally fine for $5k positions.
- **Market orders at 9:30 (post-open)** get crossed against the opening book, often at 5–10 bps worse than the auction print.
- **Limit orders at prior close** have high non-fill rates for surprise-driven PEAD names that gap.

Alpaca paper does not simulate auction mechanics accurately. **Your paper Sharpe will not reflect real-world fill quality.** This is an independent risk on top of all the other concerns.

### 5. No Mention of Tax Drag

Even in a paper account, you should be modeling pre-tax vs after-tax returns for any strategy you intend to run in a non-IRA account at real capital. A 5-day-hold strategy generates 100% short-term gains, taxed at marginal rate (35–45%). After-tax Sharpe on a strategy with high turnover is substantially worse than pre-tax Sharpe. Doesn't affect paper trading decisions but should affect "graduate to real money" decisions.

### 6. No Comparison Against Trivial Benchmarks

You don't show what a **null-signal version** of this strategy would produce. Specifically:

- **Random direction PEAD:** flip a coin for each surprise > 5% magnitude, use same sizing, hold same period.
- **Pure-momentum baseline:** trade top-N by trailing 60-day return, same sizing.
- **Buy-and-hold SPY:** for the same period.

If your strategy doesn't beat random-direction PEAD by a large margin, the signal isn't doing anything. If it doesn't beat momentum by a margin justifying the complexity, you should just trade momentum.

### 7. The Compute-Cost-of-Being-Wrong

If the strategy turns out to be broken, what have you spent? Time. If it turns out to be real, what have you gained? Real money. But there's a third path that's worse than either: **the strategy is mediocre (Sharpe 0.3 after fix), but you don't realize this clearly because the paper test was noisy.** You then deploy at real money and watch a 6-month grind to nowhere while opportunity cost accumulates.

**The audit-before-paper sequencing is not just risk management — it's information value maximization.** A clean Sharpe-1.2 result post-audit is much more decision-useful than a noisy paper-Sharpe-of-1.0-with-95%-CI-of-[-3.0, +5.0].

---

## Part 4 — Out-of-the-Box Recommendations (Trading-Methodology Adjacents)

You asked for thinking outside the box. Here are six directions I'd consider once the bug is resolved:

### 4.1 — Replace QualityShort with a Long-Quality Tilt

The QS strategy is structurally exposed to short-squeeze tail risk that's nearly impossible to backtest accurately. **An equivalent factor exposure can be obtained with much less tail risk by going LONG high-quality (the inverse universe) instead of SHORT low-quality.** Academic evidence (Asness, Frazzini, Pedersen "Quality Minus Junk") shows the long leg captures most of the quality premium with a fraction of the volatility.

Implementation: rank universe by composite quality (high op margin, low D/E, positive revenue growth, positive earnings surprise). Long top 25 names, equal-weight. No regime gate needed — quality tends to outperform during bear markets and underperform mildly during junk rallies, but never violently snaps back.

Expected Sharpe range (post-cost, realistic): 0.4–0.7. Much lower than your headline, but it's a number you can actually believe.

### 4.2 — Convert PEAD to a Conditional Model

Pure PEAD (trade every surprise > 5%) is decay-prone. **Top hedge funds use PEAD as a feature, not a strategy.** They condition the trade on additional signals:

- Analyst revision direction in the 24h post-announcement
- Sector momentum
- Implied volatility skew change
- News sentiment delta
- Insider transaction proximity

This is where your existing XGBoost infrastructure earns its keep. **Reframe PEAD as: "given a surprise > 5%, predict 5-day residual return using these 15 features."** Trade only the top-quantile predictions. Expected to roughly double the per-trade edge while halving trade count — net Sharpe lift of perhaps 30–50%.

### 4.3 — Options Overlay for Shorts

If you keep a short book, **buy ATM puts instead of shorting stock** for the QS book specifically. Trade-offs:

| Aspect | Short stock | Long puts |
|---|---|---|
| Borrow cost | 5–15%/yr | Zero |
| Max loss | Unlimited | Premium paid |
| Theta | Receive (dividends if any) | Pay |
| Capital efficiency | Margin required | Premium only |
| Tail risk on squeeze | Severe | Bounded |

For 5–20 day holds on R1000 names, weekly or 30-day puts are reasonably priced and avoid the entire borrow/squeeze problem. The headline is that you pay theta (probably 1–3% per trade) but eliminate the worst-case scenarios that turn a Sharpe-1.5 strategy into a Sharpe-0.3 strategy when squeeze risk is properly accounted for.

### 4.4 — Volatility-Targeted Position Sizing

Currently you size to 5% of equity per position. Replace with **volatility-targeting**: size to (target_position_vol / name_realized_vol). E.g., target 1% portfolio vol contribution per position; for a name with 30% annualized vol, that's a 3.3% position; for a 60%-vol name, 1.7%.

This reduces tail risk substantially without sacrificing Sharpe, typically improves Sharpe by 10–25% on the kind of strategy you're running.

### 4.5 — Add a Cross-Sectional Hedge

Both PEAD and QS are net-short exposure to negative-surprise / weak-fundamentals names. **You should explicitly hedge the residual market exposure** with a SPY short overlay, sized to your gross long minus gross short delta. This is what makes "factor strategy" actually a factor strategy rather than a directional bet with extra steps. Reduces beta-driven variance materially.

### 4.6 — Consider an Earnings-Drift PAIR Trade

Rather than long-only PEAD beats and short-only PEAD misses, consider **paired PEAD**: on any earnings day, find one beat and one miss in the same sector, long the beat and short the miss in equal dollars. This is sector-neutral, partially style-neutral, and captures the cross-sectional information content of surprises directly. Academic literature (Loh-Stulz "Is Sell-Side Research More Valuable in Bad Times?") suggests this paired version has roughly the same Sharpe as long-only PEAD but with 30–50% less variance and substantially lower drawdowns.

---

## Part 5 — Recommended Sequence of Work (Concrete, Prioritized)

If I were managing this as a research project at a fund, this is the order:

### Phase A: Forensic Audit (1–2 weeks, blocking everything else)

1. **Equity curve reconstruction (3 days):** Take Fold 5's full trade log. Manually rebuild the equity curve in a separate notebook: start at $100k, apply each trade's realized P&L at exit, deduct borrow daily, mark open positions at daily close. Compute Sharpe from this independent curve. Compare to the WF-reported Sharpe. **If they differ by > 0.5, you've found the bug.**

2. **Sharpe arithmetic audit (1 day):** Print the daily return series used in the Sharpe calculation. Verify it has 252 entries per year (not 300, not 5000), uses portfolio equity as denominator, and is annualized by `sqrt(252)`.

3. **PIT spot check (1 day):** Print, for the 20 most-recent PEAD signals, all fields used in the join — `as_of`, `fmp_surprise_1q`, the FMP row's `as_of_date`, `filingDate`, `acceptedDate`, `period_end`. Cross-check against EDGAR or Yahoo for any surprise that looks suspicious.

4. **Entry timing audit (1 day):** For the same 20 signals, identify the exact bar used for entry price. Should be the `open` of the bar at `as_of + 1 trading day`. Verify there's no leakage to `as_of` close.

5. **De-survivorship reconstruction (2–3 days):** Even if you can't get a full PIT universe quickly, do a quick sanity check: identify 30 names that were in R1000 in 2021 but aren't now. Check if your simulator data has them. If not, that's the survivorship leak. Rerun WF on a subset of 200 large-cap names that have been continuously in R1000 since 2021 — this is "selection bias in the safest direction" and gives a lower-bound on a real result.

### Phase B: Cost-Realistic Re-Run (3–5 days)

6. Re-run all 17 configs with: 30 bps round-trip cost, 8% borrow on shorts. Report new Sharpe rankings. **Do not look at the new rankings until you've made a written prediction of which configs you expect to survive.** This is the only way to check for confirmation bias in your interpretation.

7. Run PEAD-only at hold = {1, 2, 3, 5, 7, 10, 15} bars. Should see an inverted-U with peak in 3–7 day range.

8. Run PEAD-only with entry shifted T+1, T+2, T+3, T+5. Should see monotonic Sharpe decay.

### Phase C: Robustness (1 week)

9. Implement CPCV (López de Prado Ch. 12) for the top-2 surviving configs. Get the full Sharpe distribution from ~50 paths.

10. Bootstrap Sharpe confidence intervals using block bootstrap (10-day blocks) on the trade-level returns. Report 95% CI.

11. Synthetic-benchmark run: random-direction PEAD using the same infrastructure. Confirm random returns Sharpe ≈ 0 with the WF-corrected accounting.

### Phase D: Pre-Paper Live-Code Forward Test (1 week)

12. Run the live PM/RM/Trader code path (not the WF simulator) against the most recent 60 days of cached data, with all live-system features. Compare to the WF simulator's results on the same 60-day window. **Discrepancies indicate live-vs-sim divergence in your own code.**

### Phase E: Paper Trading (3+ months, but ideally 6+)

13. Single-strategy paper (PEAD only), 1% position sizes, 4 weeks. Compare per-trade P&L distribution to backtest via KS test.

14. If pass: add QualityShort at 1% sizing, another 4 weeks. KS test again.

15. If pass: scale to 3% position sizes, another 4 weeks.

16. If pass: scale to 5% (backtest sizing), another 8 weeks.

17. Decision point on real capital: after 6 months of paper, if observed Sharpe is in the ballpark of corrected-backtest Sharpe within 2σ, deploy small real capital (e.g., 10% of intended size). Otherwise, return to audit.

**Total elapsed time from today to "small real capital deployment": ~7–9 months.** This sounds slow but it's actually fast for a strategy this complex with this many open methodological questions.

---

## Part 6 — Calibrated Probability Estimate

**Probability that the proposed deployment achieves observed Sharpe > 0.50 over a 3-month live paper period: ~35–40%.**

Sub-distribution:

| Scenario | P() | Conditional P(observed 3mo Sharpe > 0.50) | Joint |
|---|---|---|---|
| Bug fixed, true Sharpe 1.5–2.5 (real PEAD survives) | 25% | 70% | 17.5% |
| Bug fixed, true Sharpe 0.5–1.5 (cost-adjusted real edge) | 30% | 50% | 15.0% |
| Bug fixed, true Sharpe 0.0–0.5 (marginal) | 20% | 33% | 6.7% |
| Bug fixed, true Sharpe < 0 (no real edge) | 15% | 12% | 1.8% |
| Bug NOT fixed, paper uses same code path | 10% | 45%* | 4.5% |
| **Total** | 100% | — | **~45%** |

*The "bug not fixed but paper still works" path: if the bug is in equity-curve accounting (not in signal generation), then paper trading via Alpaca will measure real-world equity directly and will report a more accurate Sharpe regardless. So the bug being unfixed doesn't necessarily zero out the paper result — but it also doesn't *help*. P(real Sharpe > 0.5 over 3 months of paper, regardless of what the backtest said) ≈ 45% by my weighting.

I'm going to call the overall point estimate **40%**, with substantial uncertainty (probably ±15 points on calibration).

The author's 55–65% estimate isn't crazy — it just understates the "bug means we don't actually know what we have" probability mass.

---

## Part 7 — The One Thing That Matters Most

If you do nothing else from this review, **do the equity-curve forensic audit (Phase A, step 1)**. Three days of work. Will resolve the single largest piece of uncertainty in your entire research program. Everything downstream — strategy selection, cost realism, paper trading design, capital sizing — depends on whether the metric you're optimizing is actually a Sharpe ratio or a phantom inflated by 10x.

If the audit comes back clean (Sharpe is actually 8 on the corrected curve), I will eat my hat in writing and you should ship to paper immediately because you've found something extraordinary. If the audit comes back with a real Sharpe of 0.8–1.5 post-correction, you've still got a paper-worthy strategy. If it comes back with real Sharpe < 0.3, you've saved yourself 3 months of misleading paper trading.

Every outcome of the audit is high-information. That's the definition of a high-leverage experiment. Do it first.

---

## Appendix A — Code-Level Hypotheses to Verify

These are the specific patterns I'd grep for in the codebase, in priority order:

```python
# 1. In AgentSimulator equity calculation:
#    Look for ANY use of trade-level return as input to a portfolio-level series.
#    Pattern to find:
daily_returns.append(trade.pct_return)  # WRONG — should be trade.pnl / portfolio_equity

# 2. In Sharpe computation:
#    Look for annualization factor.
sharpe = mean(returns) / std(returns) * sqrt(252)   # Correct if `returns` is daily portfolio
sharpe = mean(returns) / std(returns) * sqrt(N)     # WRONG if N = trade count, not 252

# 3. In position sizing:
#    The denominator for sizing should be CURRENT equity, the denominator for return should be the same.
#    Pattern to find:
position_size = available_cash * 0.05         # Sizing on available_cash
return_series = pnl / starting_equity         # Return on starting_equity  →  INCONSISTENT

# 4. PIT join:
#    The merge_asof or similar should use direction='backward' with strict tolerance.
features = pd.merge_asof(prices, fundamentals, on='date', direction='backward')   # Verify

# 5. Borrow cost aggregation:
#    Borrow should be deducted from portfolio equity DAILY for open positions.
#    Pattern to find:
total_borrow = sum(p.borrow_cost for p in closed_trades)   # WRONG — only on close
# Should be:
daily_borrow = sum(p.notional * 0.005 / 252 for p in open_positions_today)  # CORRECT
```

## Appendix B — Useful Comparison Points for Sanity-Checking Real Strategies

For when you do get to a clean post-audit number, here's what published similar work has reported:

| Strategy class | Published Sharpe (post-cost, realistic) | Time period | Source |
|---|---|---|---|
| PEAD long-only, large-cap | 0.6–1.0 | 2010–2020 | Chordia, Subrahmanyam, et al. |
| PEAD long-short, mid-cap | 0.8–1.5 | 1990–2010 | Bernard & Thomas |
| Earnings surprise + revisions | 1.2–1.8 | 1995–2015 | Sloan, Chan, AQR factor |
| Quality (long-only) | 0.5–0.8 | 1960–2020 | Asness/Frazzini/Pedersen |
| Quality minus Junk (L/S) | 0.7–1.0 | 1960–2020 | Same |
| Combined factor portfolio (PEAD + quality + momentum) | 1.5–2.5 | 2000–2020 | Various |

**If your post-audit Sharpe lands in the 1.0–2.0 range, you're in line with what's achievable. If it lands above 2.5, you should be deeply suspicious again.**

---

*End of review. Reviewer estimates that the single highest-value next action is the equity-curve forensic audit, time cost ~3 days, information value: resolves the dominant source of uncertainty in the entire research program. Recommended next step after audit: clean cost-realistic rerun of all 17 configs followed by CPCV on the top survivors. Recommended time to paper trading: not less than 4 weeks from today, after audit + cost rerun, regardless of audit outcome.*
