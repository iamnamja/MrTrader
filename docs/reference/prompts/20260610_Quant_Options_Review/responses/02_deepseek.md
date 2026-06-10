# External Quant Review - MrTrader Options & Alpha Pipeline (DeepSeek)

## Executive Summary

You have built one of the most honest and self-aware retail quant setups I have seen. The kill discipline, the event-sparsity waiver flag, the regime-gate hardening, the per-fold retraining - all of this is production-grade thinking. Your biggest risk is no longer leakage or overfitting; it is **under-powered conviction** and **framing the wrong thing as alpha**. You are correctly killing strategies that fail a significance gate, but you may be killing *portfolio* value because your gate is designed for high-Sharpe, low-skew equity alpha - not for diversifying risk premiums or low-signal-to-noise event edges. The options data is a diamond, but you are using it mostly to re-test short-vol. I will tell you where the real alpha-shaped opportunities are, where your harness still flatters or hides edges, and what I would change tomorrow.

---

## 1. Verdict on the validation harness - sound but with critical blind spots

Your harness is **trustworthy for its stated purpose** (detecting high-Sharpe, beta-neutral, low-skew alpha). However, it has three structural blind spots that matter for options and event-driven strategies.

### What you do right (better than most funds)
- **PIT + survivorship** by construction from OPRA daily files - gold standard.
- **Per-fold retraining** for ML models - true OOS.
- **Regime coverage map** printed before performance - forces honesty.
- **`requires_human_review` flag** for event-sparse waivers - disciplined.
- **2x spread stress** - essential for options costs.

### Blind spot 1: CPCV path correlation is still under-penalised in DSR
You correctly use `n_folds` for the path t-stat (good). But the DSR multiplicity adjustment (`unique_obs = total_obs / C(k-1,p-1)`) still assumes test days across paths are independent. They are not. A single trading day appears in `C(k-1,p-1)` paths. The effective number of *independent* trials is closer to `n_folds` (not `n_paths`), but DSR's `N_TRIALS_TESTED=250` is a fixed guess, not derived from fold structure. For a 6-fold, 2-path CPCV, the true multiplicity is much lower than 250 - your DSR p-values are **too conservative** (inflated, making it harder to pass). But for strategies with mean Sharpe ~0.5, DSR p ~0.08 as you saw - that's not "no edge", that's "under-powered" (see below).

**Fix:** Replace fixed `N_TRIALS_TESTED` with a data-derived estimate:
`N_eff_trials = n_folds * (n_folds-1) / 2` for the number of distinct train/test splits. Then compute DSR via the standard formula (Marcos Lopez de Prado). This gives an honest, less conservative multiplicity penalty.

### Blind spot 2: You have no direct test for *skew* or *tail* robustness
Your regime gate looks at worst-regime Sharpe, but a strategy can have Sharpe ~0 (short-vol) yet still be a valuable diversifier because its left tail is offset by the rest of the book. Your gate kills it. That is correct for a *standalone alpha* sleeve, but wrong for a *portfolio* diversifier. You need a **second, parallel acceptance framework** for risk premiums (low Sharpe, negative skew, low correlation).

**Fix:** Add `book_level_diversifier_gate()` that checks:
- Correlation to existing book < 0.3
- Worst 1% drawdown does not coincide with existing book's worst 1% (joint tail test)
- Sharpe > 0.2 (not 0.8) after vol-targeting to 10% annual vol.

### Blind spot 3: The event-sparsity waiver for PEAD's regime gate is a *symptom*, not a fix
You correctly flagged that PEAD has `<20` same-regime trading days. But instead of waiving the gate, you should **re-design the regime test for event strategies**. A regime label on a *day* where no trade exists is meaningless. For PEAD, measure regime exposure by *event-weighted* returns: for each earnings event, label the regime of the entry day only, then compute Sharpe per regime across events (not days). That yields sufficient observations (PEAD has ~500 events over 4 years).

**Fix:** New function `event_regime_sharpes()` in `regime.py`, used only for strategies with `is_event_driven=True`. Then you never need the waiver.

---

## 2. The 3-5 highest-EV research directions (ranked)

Given your data (daily OHLCV + computed greeks, 4y, R1K, no OI/NBBO) and constraints (EOD marks, single operator), here is where I would put money.

### #1 - Dispersion trading on the index (SPX/SPY vs components) - **alpha-shaped, not risk premium**
You already have index short-vol (VRP) but it's a risk premium. **Dispersion** is different: long index options (e.g., SPY straddle) and short a basket of single-name options (or vice versa). The edge comes from mispricing of correlation. When single-name implied vols are too high relative to index implied vol (correlation too low), you short single-names and buy index. This is a **relative value trade** - market-neutral, beta-neutral, and historically has positive skew (pays in crises). It is true alpha because it exploits a structural segmentation between index and single-name option markets.

**Testable on your data:**
- Compute daily index implied vol (from SPY options, your engine).
- Compute average single-name implied vol for the top 50 R1K names (equal-weight).
- Signal = index_iv - avg_single_iv. When too high (correlation cheap), short index vol, buy single-name vol (or the reverse).
- Execute weekly, delta-hedged with SPY futures or underlying.
- You need only EOD closes + computed IV (you have it). No OI required.

**Why it could be real:** This is a structural anomaly documented in academic literature (e.g., "dispersion trading" by Carr & Wu). It survives because of institutional constraints (index options used for hedging, single-name for speculation). Your data is sufficient for a first pass.

### #2 - Cross-sectional vol carry (long cheap IV, short rich IV, delta-hedged) - **alpha-shaped**
Forget the short-vol sleeve. Instead, rank all R1K options by `implied_vol / realized_vol` (IV/RV ratio). Long the cheapest decile (IV << RV), short the richest decile (IV >> RV), delta-hedge each leg daily with the underlying. This is **vol carry** - you get paid for the mean reversion of IV/RV. It is market-neutral, has near-zero beta, and positive expected return. Unlike outright short-vol, it is not a risk premium (it is a relative mispricing).

**Testable on your data:**
- For each stock, compute trailing 20d realized vol from daily returns.
- For each option (closest to 30DTE, delta ~0.50), compute IV (your engine).
- Daily signal = IV/RV for the ATM option.
- Long the lowest 10%, short the highest 10%, each position sized to 1% vol contribution.
- Rebalance weekly. You need only EOD closes + computed IV (you have it).

**Why this is better than short-vol:** It is not a pure crash risk premium. It works in low-vol and high-vol regimes because it is relative. It has higher Sharpe (historical ~0.8-1.0) and lower drawdowns. And you can test it without OI or intraday data.

### #3 - Implied skew change as a signal for equity returns (options-as-signal) - **already partially validated**
Your OPT-5 implied-move filter for PEAD is promising but threshold-fragile. The more robust version: use the **change in put skew** (e.g., 25-delta put IV minus 50-delta put IV) over the 5 days before earnings. A steepening skew (puts becoming more expensive) predicts negative post-earnings drift. This is a genuine signal because options market makers hedge skew changes, causing directional pressure.

**Testable on your data:**
- For each earnings event, compute put skew 5 days before and 1 day before (using your engine on the nearest monthly option).
- Signal = skew_change (positive = puts more expensive).
- Trade: short the stock when skew_change > 0.5 standard deviations.
- Hold 5 days. This is a pure equity signal derived from options, no options execution needed.

**Why it could be real:** It is a different form of "informed trading" - options traders anticipate the move and skew prices accordingly. Your 2022-2026 data includes enough events to test.

### #4 - Vol-of-vol carry using VIX term structure (VIX futures not required) - **low data requirement**
You have VIX index daily values. You can compute a proxy for vol-of-vol using the difference between VIX and a 20-day realized vol of VIX. When VIX is high and vol-of-vol is low, short VIX via options (e.g., sell VIX calls). But that requires VIX options (which you may not have). However, you can trade vol-of-vol via *calendar spreads on SPX options* - long 30DTE, short 60DTE, vega-neutral. The signal is the term structure slope change.

**Testable:** Compute VIX term structure proxy as `VIX_30d - VIX_60d` (using your engine on SPX options). When the slope is too steep (short-term fear > long-term), sell the steepness via calendar spread. This is a genuine relative value trade, not outright short-vol.

### #5 - (Lowest EV, but complementary) Double down on index short-vol but with a vol-targeting tail overlay
You correctly found that index short-vol has real VRP (PF 2.24) but Sharpe ~0 because of the left tail. Instead of killing it, **vol-target it to 10% annual vol** and add a put-spend overlay: spend 10% of expected premium on 5% OTM puts on SPX. That transforms the return distribution from negatively skewed to slightly positive skew. Then re-evaluate at the *book* level (not standalone). You have all the data to backtest this.

---

## 3. Best alpha-shaped uses of the options data (given your limits)

Your data limits (no historical IV/OI/NBBO) are severe but not fatal. The highest-EV uses are those that require only **daily OHLCV + computed greeks + contract universe**. Here is what I would build, in order:

1. **Dispersion trading (index vs single-names)** - as above. Requires only daily closes and computed IV per option. You have both.
2. **Cross-sectional IV/RV carry** - as above. Requires daily underlying returns and computed ATM IV. You have both.
3. **Implied skew change as equity signal** - as above. Requires computed IV for two strikes (25-delta and 50-delta). Your engine gives you that.
4. **Put/call ratio from volume** - you have daily volume per contract. Compute put/call ratio (volume-weighted) for each stock. Use it as a sentiment signal for the underlying equity (high put volume -> bullish reversal). No OI needed.
5. **Options-implied earnings move vs realized** - your OPT-5 filter but with a **pre-registered threshold** (0.9, not 1.0) and tested on the full 4y R1K. The threshold fragility you found is real, but a single pre-registered threshold (e.g., 0.85) might still show a plateau. Run that.

What I would **not** pursue given your data:
- Dealer gamma positioning (needs OI).
- Intraday volatility arbitrage (needs quotes).
- True market-making (needs NBBO).
- Volatility surface arbitrage (needs full surface historical - you have only one strike per option, not a continuous surface).

---

## 4. Architecture / design gaps

### Gap 1 - No systematic overfitting detector beyond DSR
DSR is weak. Add a **walk-forward optimism test**: For each fold, compare the model's Sharpe on its own test set to a simple baseline (e.g., previous fold's model on current test). If the new model is not better than the previous fold's model (t-test on paired differences), flag it. This catches overfitting that DSR misses.

### Gap 2 - Options simulator lacks market impact / liquidity model
You stress spreads, but you ignore that a large position (e.g., 100 contracts) will move the market. For index options (SPY, QQQ) this is fine, but for single-name options (e.g., a small-cap stock), 50 contracts could be 10% of daily volume. Add a **liquidity multiplier**:
- `execution_cost = spread * (1 + volume_ratio * impact_factor)`
- `volume_ratio = order_contracts / avg_daily_volume(option)`
- If `volume_ratio > 0.05`, reject the trade (illiquid). This will kill many single-name options trades - and that is honest.

### Gap 3 - No live-to-backtest decay tracker
You have a shadow-mode trend sleeve, but no automated tracker that compares live paper Sharpe to backtest Sharpe with statistical significance. Build a **decay detector**:
- Monthly, compute live Sharpe (paper) vs backtest Sharpe (same period) with a permutation test.
- If live is worse than backtest at p < 0.05, flag for review.
- This catches data drift, broker differences, and execution slippage.

### Gap 4 - Single operator - need automated "circuit breakers" for options
Options have non-linear risk. You have a kill switch, but you need **delta- and vega-based circuit breakers**:
- If portfolio delta > 0.5 x equity, reduce exposure.
- If portfolio vega > 5% of equity, hedge with VIX futures (if available) or reduce.
- These should be in the Risk Manager, with hard limits that cannot be overridden without code change.

### Gap 5 - No serial correlation test for event strategies
PEAD's events are clustered in time (earnings seasons). Your CPCV purge+embargo partially addresses this, but you should add a **Ljung-Box test** on the daily returns of the event book. If significant serial correlation exists (p < 0.05), the effective sample size is smaller than `n_obs`. Then adjust your t-stat using the Newey-West standard error (you already compute it in residual-alpha - apply it to raw Sharpe too).

---

## 5. The first 5 things I would change (prioritised)

### #1 - Build the dispersion trading backtest (index vs single-names)
This is your highest expected value move. It uses only data you already have, it is genuinely alpha-shaped (not a risk premium), and it can be tested within 2 weeks. I would make this the sole focus of the options program for the next month. If it works (CPCV mean Sharpe > 0.5, t > 2, positive skew), I would paper trade it alongside PEAD.

### #2 - Fix the DSR multiplicity penalty
Replace fixed `N_TRIALS_TESTED=250` with a data-derived `n_effective_trials = n_folds * (n_folds-1) / 2`. This will make your DSR p-values less conservative and will correctly classify the PEAD CPCV (which likely passes at p > 0.95 with the corrected formula). Then re-evaluate the index short-vol CPCV - it might still fail, but at least the test is honest.

### #3 - Add the event-regime Sharpe test (no more waiver)
Implement `event_regime_sharpes()` and use it for PEAD and any future event strategy. Then remove the manual waiver flag. This will give you a true regime gate for event strategies without spurious failures.

### #4 - Run the cross-sectional IV/RV carry on a small universe (e.g., top 100 R1K)
This is a one-week side project. Use your existing `OptionsStrategy` adapter, build a `VolCarryStrategy` that:
- Each week, for each stock, compute ATM IV (closest to 30DTE) and 20d RV.
- Long bottom 10% IV/RV, short top 10%, delta-hedge daily with underlying.
- Run CPCV (k=6, paths=2, 4y). If mean Sharpe > 0.4 and path t > 1.5, escalate to a full research project. If not, kill it quickly (your strength).

### #5 - Kill the stand-alone index short-vol sleeve in the allocator (but keep the data)
You already decided it's KILL standalone. But do not delete the code. Instead, **relegate it to a "risk premium library"** that can be added to the book only through the new `book_level_diversifier_gate()` (see #2 in the architecture gaps). Then set a reminder to re-evaluate in 12 months when you have more data (4 years is one vol cycle - you need a crisis to see if the overlay works). This prevents you from re-testing it every month and overfitting.

---

## What you are doing right (do not change)

- **Per-fold retraining** - essential for ML.
- **PIT + survivorship from OPRA files** - best in class.
- **Event-sparsity flag** - honest and disciplined.
- **Regime coverage table printed before performance** - forces you to look at the data distribution.
- **Killing strategies that fail** - this is the #1 discipline that separates successful quants from pretenders.

## What I would kill immediately

- **The `requires_human_review` waiver for regime gate.** Replace it with `event_regime_sharpes()` (above). The waiver is a band-aid, not a fix.
- **The implied-move filter at threshold = 1.0** from any live consideration. The threshold fragility you found is damning. If you re-run it, pre-register a threshold (e.g., 0.9) and do not peek.
- **Any plan to trade single-name earnings IV-crush** - you correctly killed it. Do not revisit unless options spreads drop by 80% (they won't).

## Final, most important observation

You are trying to find **standalone alpha** in a world where most edges are **small, diversifying, and low-Sharpe**. Your gate is calibrated for a high-Sharpe equity trend following or market-neutral factor book. But real portfolio construction is about combining many low-Sharpe, low-correlation strategies.

You have one such candidate: **index short-vol with a tail overlay** (VRP real, PF 2.24, but Sharpe ~0). By your current gate, it is dead. By a book-level diversifier gate, it might be alive.

My recommendation: **Build the book-level diversifier gate first** (Correlation < 0.3, joint tail test, Sharpe > 0.2). Then re-evaluate index short-vol. If it passes *that* gate, deploy it at 5% of risk budget. If it fails, kill it forever. But do not kill it because it fails a standalone alpha gate - that is using the wrong ruler.

You have built a world-class validation harness. Now use it to measure the right thing.

---

*If you implement only one thing from this review, make it the dispersion trading backtest. That is where the real, untapped, alpha-shaped options edge lives.*
