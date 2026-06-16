# Independent Quant Review — Alpha-v6 Post-Mortem & Next Steps

**Reviewer:** World-class quantitative researcher (solo operator perspective)
**Date:** 2026-06-12
**System:** MrTrader — $100k paper book, TSMOM trend sleeve (10-ETF, 25% NAV) live, all other strategies killed.

## Q1 — Gate parameters: sound in principle, misapplied to short-sample strategies

Your significance-first gate is conceptually correct, and the calibration study is the right way to measure Type-II error. The fundamental issue is not the threshold values themselves, but the **mismatch between the statistical power of your data (≤4 years / ≈8 independent fold-equivalents) and the effect size needed to clear t≥2.0**.

### The power problem in numbers
For a true Sharpe of 0.5 and N_eff = 8 folds each with 250 trading days, the standard error of the mean Sharpe is about 0.035, giving t≈14 — that suggests you should clear t≥2 easily. The fact that your PEAD (SR=0.55) had t=2.26 and a pure noise null had t=3.47 shows that **the effective N_obs is not the number of days, but the number of *active* days**. For event-sparse strategies, daily returns are mostly zero, so the variance of the mean Sharpe is driven by the few event days. Your event-level two-way clustered inference (p=0.78) correctly captured this.

### Specific changes I recommend

| Gate component | Current | Problem | Recommended change |
|----------------|---------|---------|---------------------|
| Minimum data length | None | Underpowered strategies fail t-test even with real signal. | **Track A paper**: require ≥10y daily *non-zero* returns OR ≥200 independent event clusters. Otherwise, candidate must pass **Track B** and is capped at 25% risk budget. |
| t-stat threshold | 2.0 (fixed) | Underpowered strategies get low t. | Replace with **bootstrap p-value** on mean Sharpe across CPCV paths. Resample the 15 paths (with replacement) 10,000 times; p = proportion(bootstrap mean ≤ 0). This automatically handles correlation. |
| PF and Calmar backstops | Mandatory | For high-win-rate, low-avg-winner strategies (e.g., PEAD 95% win rate, PF=1.54), these are misleading. | **Make diagnostic only** — report them but do not block the gate. Remove from the boolean AND. |
| Worst-regime Sharpe | -0.5, event-sparsity waiver | Works well. | Keep as is. For **CAPITAL tier**, require at least 20 trading days in each regime bucket (already in REGIME_MIN_OBS). |
| Track B (book-delta) | Well designed | Correctly passed TSMOM. | Add a **minimum candidate history** (e.g., 5 years) before a Track B evaluation, to avoid overfitting to a single regime. |
| DSR | Report-only | Correct — DSR saturates to p=1.0 for Sharpe >2, giving no discrimination. | Do not resurrect. Use the bootstrap p-value instead. |
| Mean Sharpe floor (Paper) | 0.35 | Too high for underpowered strategies (point estimate is noisy). | Lower to **0.25** for strategies with <5 years history or <200 event clusters. |

**Actionable next step**: Add a function `strategy_has_sufficient_power()` that returns True if daily return series length ≥2520 OR independent event clusters ≥200. For strategies that fail this check, use the bootstrap p-value (p<0.05) and skip PF/Calmar backstops. Keep the PF/Calmar backstops for strategies with long histories (e.g., TSMOM).

---

## Q2 — What is next? The honest answer: stop hunting alpha, operate the trend sleeve

You have killed *every* signal that required stock selection or event prediction using free data. This is not failure — it is a highly informative negative result. The liquid US equity market is efficient, especially for the long-only, large-cap universe you tested. The one thing that works is **trend-following on diversified multi-asset ETFs**, a well-known risk premium that works across centuries.

### Priority 1: Enhance the TSMOM sleeve (no new data, highest EV)

- **Increase allocation** — you are at 25% of NAV. Backtest shows Sharpe >0.7 up to 100% (with vol-targeting). Use your capital; raise to 50–75%.
- **Add uncorrelated trend assets** — free data exists for **crypto** (CoinGecko API, daily), **forex** (yfinance: EURUSD=X, etc.), and **futures** (yfinance: /ES, /CL, /GC). These have near-zero correlation to equities. A simple 20-day momentum basket of 10–15 such assets will likely add 0.2–0.3 Sharpe with very low correlation. Test in a few days using your existing `tsmom.py` (just change the universe).
- **Portfolio-level vol targeting** — you already have inverse-vol weighting per asset. Add a scaling that adjusts gross exposure so that the trailing 60d vol of the *portfolio* equals a target (e.g., 15% annualized). This will keep risk constant and improve Sharpe during low-vol periods.
- **Reduce costs** — Alpaca's $0 commission + typical 1–2bps slippage is lower than your 5bps assumption. Re-run TSMOM with 1bps; Sharpe likely >0.8.

### Priority 2: If you insist on hunting alpha, buy **I/B/E/S estimate revisions**

The one untested, academically robust, and plausibly tradeable signal with your infrastructure is **forward EPS estimate revision momentum** (Chan, Jegadeesh, Lakonishok 1996). It is distinct from analyst ratings (which you killed) because it uses the *change in consensus forward EPS*. It works cross-sectionally and survives costs in large-cap US.

- **Data source**: Interactive Brokers offers delayed I/B/E/S for ~$10–20/month with 10+ years history. Requires a funded IB account. Alternatively, Zacks Estimates via Quandl ~$300–500/month.
- **Implementation**: Use your `EventEdgeStrategy` — treat each quarterly earnings announcement as an event; feature = revision in consensus EPS from 90 days before to the day before the announcement; label = forward 5d sector-neutral return. Run CPCV on 10y of data.
- **Kill early**: If pre-cost IC <0.015 after 3 years of data, abandon.

### Priority 3: Low-frequency macro / carry (lower priority)

- **Cross-asset carry** (FX carry) — free FRED interest rates + yfinance FX. Documented Sharpe ~0.5–0.8, low correlation. But would require new broker (forex).
- You already do trend. That is enough.

**Bottom line for Q2**: Do not spend another month on stock-specific ML or event strategies with free data. Expand the trend universe, increase allocation, and consider estimate revisions as the only paid data worth buying.

---

## Q3 — Process flaws that could have buried a real edge

Yes, your process is biased toward false negatives for certain strategy types. I see three systematic reasons you may have wrongly dismissed real signals.

### 1. Survivorship bias works *against* short and event-driven strategies — and you may have over-corrected

Your use of PIT index membership and delisted-name inclusion via Polygon (for small-mid PEAD) is good. However, for **short-only** or **long-short** strategies, missing delisted names is *catastrophic*: the most profitable shorts (stocks that go to zero) are omitted, making the strategy look worse than reality. Your QualityShort (fundamental deterioration) short-only had Sharpe -0.903, but that was run on a survivorship-biased universe (yfinance). You never re-ran it on the Polygon delisted-inclusive flat files.

**Action**: Modify `scripts/run_qualityshort_cpcv.py` to accept `--delisted-haircut 1.0` and use `polygon_provider` for prices (via `PolygonProvider.get_grouped_daily` or your `symbols_data` built from those files). Re-run for 2019–2026 with realistic costs (20bps round-trip) and the delisted haircut. If the Sharpe becomes >0.5 after costs, you have a real short alpha.

### 2. ATR stops are too tight and kill winners — you never tested them on the high-win-rate composite

Your Phase-4 isolation test (removing stops) improved Sharpe, but you only tested it on the full XGBoost model (which had low IC). You never tested it on the **5-feature equal-weight composite** that had 64% win rate but negative Sharpe due to drawdowns. The LX8b test with a 7% flat stop made it worse because the stop was static, not volatility-adjusted.

**Action**: Implement a **trailing stop at 2× ATR** (from the peak) in `_process_exits` behind a flag `--rebalance-trailing-atr-stop`. Test on the LX1 composite (5-feature equal-weight) with 20-day rebalance, no entry gate, realistic costs (2bps). Expected Sharpe could be >0.6.

### 3. Hedging out SPY beta may have destroyed factor premia that are not beta

Your CAPM residual-alpha diagnostic is a good first cut, but many well-known factors (size, value, quality) have low or even negative market beta during certain periods. Hedging with SPY alone can remove a significant part of their return.

**Action**: Add a script `scripts/factor_attribution.py` that downloads the **Fama-French 5-factor daily factors** (from Ken French's data library — free), aligns them with your strategy's daily OOS returns (from CPCV concatenated series), and runs a regression with Newey-West standard errors. If the intercept t-stat > 2 after controlling for market, size, value, profitability, and investment, the edge is not explained by known risk factors.

### Minor concerns (already partially addressed)

- **CPCV path correlation** — you correctly use N_eff=n_folds for the t-stat. The bootstrap p-value (recommended in Q1) would give you more power by using the full distribution of mean Sharpe across paths.
- **Entry decision lag** — your `_process_entries` uses `exclude_today=True` for feature computation, which is one day more conservative than live (a real trader could use today's close to decide tomorrow's open). For event strategies, this lag could kill a real intra-event edge. Add an option `--entry-lookahead` that sets `exclude_today=False` (for decision) while keeping the fill at next open. Re-run PEAD with that; the t-stat might improve.

---

## Bottom line — if I took over the system tomorrow, my first three actions

1. **Expand and scale the TSMOM sleeve**
   - Increase allocation to 50–75% of NAV.
   - Add 5 futures (/ES, /NQ, /CL, /GC, /ZB) and 3 crypto (BTC, ETH) to the trend universe using free data.
   - Implement portfolio-level vol targeting (15% annualized).
   - Re-run 19y backtest; if Sharpe >0.7, switch the live book.
   **Expected gain**: +0.2 Sharpe, lower drawdowns, negligible research cost.

2. **Re-run QualityShort shorts-only on delisted-inclusive Polygon data**
   - Use `--delisted-haircut 1.0` and realistic costs (20bps round-trip).
   - If post-cost Sharpe >0.5, add it as a 10% allocation (short hedge) in the live book.
   - If not, drop it forever.

3. **Implement the bootstrap p-value for CPCV**
   - Replace the t-stat gate with a bootstrap p-value (p<0.05) for underpowered strategies (<5y data or <200 event clusters).
   - Re-evaluate the top 3 strategies that had high win rate but low Sharpe (5-feature composite, PEAD long-only, short-interest factor) using a 10-year window (where available) and drop PF/Calmar from the gate.
   - If none pass p<0.05, accept that **the only real edge is trend**.

**If after these three steps nothing new passes, stop researching.** Run the trend sleeve, focus on execution and risk management, and consider scaling to real capital. A solo operator with a $100k book cannot beat the market's collective intelligence; the only rational strategy is to harvest well-known risk premia with low costs. You have already proven that works.
