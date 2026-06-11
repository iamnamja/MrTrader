# External Quant Review — MrTrader System (2026-06-10)

**Author:** Senior Quant (multiple production L/S vol, event, and index systems; >$2B AUM peak). Ruthlessly honest lens: edge must survive costs, multiplicity, regime stress, and true OOS power. Single-operator constraint respected.

## 1. Verdict on the Validation Harness

**Overall soundness: B+ (strong for a solo shop, but several structural leaks and calibration issues that can both inflate and hide edges).** The CPCV + purge/embargo + PIT + survivorship design is materially better than 90% of retail quant setups. The significance-first two-tier gate (t-stat, %pos, P5, with mean-Sharpe as floor) is a good evolution away from raw Sharpe worship. Residual-alpha (CAPM/HAC) diagnostic is excellent and correctly flagged PEAD as mostly beta in some windows.

**Critical issues (can inflate false positives or suppress real edges):**

- **CPCV path correlation under-counted.** N_eff ≈ n_folds (correct call) but path t-stat and %pos still treat correlated paths as independent. On short-vol or event strategies with clustered realizations, this overstates significance. Fix: bootstrap at the *cluster* level (earnings quarters or vol regimes) for DSR/p-values, not just path-level. Your event-clustered bootstrap for PEAD is the right direction — generalize it.
- **Regime gate fragility on sparse strategies.** The event-sparsity waiver for PEAD is pragmatic but dangerous precedent. Worst-regime=None should force human review + longer holdout, not auto-waive to PAPER. Real crises are rare; 4y data barely samples one vol cycle.
- **Options simulator realism gaps.** EOD close marking + stressed spreads (2× mandatory) is good, but no intraday gamma/theta path dependency and no historical vol surface means IV-crush and dynamic hedging edges are approximated, not measured. American exercise + dividend handling validated well (OPT-1a), but tail events (earnings gaps, pin risk) are likely understated. Your stress-mult survives cost reality for index but not single-name — correct.
- **Multiple-testing not fully penalized.** N_TRIALS≈250 in DSR is a start, but your research log shows many more implicit trials (parameter sweeps, OPT-3/4/5 variants, threshold fragility). Pre-register strategy specs before running CPCV. The implied-move filter fragility (spike at 1.0, inversion at 1.25) is textbook post-hoc overfitting — you correctly parked it.
- **Power & coverage.** Many verdicts on thin samples (8 folds, ~2y options window, monthly cadence). Under-powered KILLs (e.g., index short-vol) may hide a book-level diversifier. Positive: honest nulls on reversal/carry/ranker show the harness *can* kill.

**Net:** Harness is honest enough to trust KILLs (earnings IV-crush, XS-ML ranker) but can suppress low-Sharpe, negatively-skewed risk premia that improve portfolio SR. It is calibrated for "pure alpha" not "portfolio contributor." No fatal look-ahead (PIT + knowable_date strong), but simulation lacks full microstructure realism for options execution.

## 2. 3–5 Highest-EV Research Directions (Ranked)

Given constraints (4y options data, no hist IV/OI/NBBO, single-op, event-driven bias, PEAD+trend core):

1. **Options-implied signals as *features* into existing equity event edge (highest EV).**  
   Mechanism: Implied move, put/call skew, term-structure slope, and realized-vs-implied volatility ratio as real-time filters or conviction multipliers for PEAD (or new analyst-revision / guidance events). Your OPT-5 was a good start but too binary/threshold-fragile.  
   Why alpha (not beta/risk-premium): These are *information* proxies for informed flow and uncertainty resolution. Dealer positioning / gamma exposure inferred from skew + volume (not OI) can predict post-event drift or reversal. Cross-sectional relative richness (compute synthetic IV surfaces via your engine) gives cheap/expensive vol signals orthogonal to price momentum.  
   Test: Augment PEAD scorer with continuous features (no hard thresholds). CPCV on full 4y R1K with regime interaction. Pre-register 2-3 features. Expect modest Sharpe lift + better P5 if it works. Low capacity cost (no options execution).

2. **Dispersion trading (index vs single-stock vol) — relative VRP.**  
   Mechanism: Long ATM/OTM straddle or strangle baskets on cheap single-names (low IV rank) + short index (SPX/NDX) using your computed greeks + EOD marks. Delta-hedge daily or use gamma scalping approximations. Weekly/monthly rebalance with regime overlay (VIX term structure from your regime model).  
   Why alpha: Classic stat-arb on mispricing between implied correlation and realized. Your data supports cross-sectional IV rank (compute surfaces per underlying). Index spreads are pennies; single-name cost drag mitigated by selection. Complements your index short-vol learnings.  
   Test: OPT-4b extension. Stress 2-3× spreads. Book-level contribution (add small sleeve to PEAD+trend). Kill if no residual alpha after beta-hedging.

3. **Skew/term-structure as crash-hedge timing or regime enhancer.**  
   Mechanism: Put-skew steepness or VIX futures basis as inputs to regime model or dynamic tail-hedge allocator (buy OTM index puts when skew signals fear). Or short OTM puts in low-skew regimes.  
   Why real: Skew is a forward-looking risk aversion signal, often leading equity drawdowns. Your computed greeks + volume allow proxy without hist surface. Natural fit for crisis-diversifier gap in trend sleeve.  
   Test: Extend regime model with options features. Forward-test book Sharpe/Calmar improvement.

4. **Options liquidity / volume as equity signal (microstructure proxy).**  
   Mechanism: Abnormal option volume/notional (your only liquidity signal) + IV skew as predictor of informed equity flow for event or swing entries.  
   Test: As additive feature to PEAD or new short-interest-squeeze event strategy.

5. **Tail-hedge sleeve (OPT-7) using cheap OTM puts, dynamically sized.**  
   Lower priority — only after 1-2 above confirm. Pure cost center unless timed well via skew.

**Kill/double-down:** Kill further single-name earnings short-vol. Double-down on event-driven family (PEAD core + extensions). Cross-sectional ML is correctly dead.

## 3. Best Alpha-Shaped Uses of the Options Data (Given Limits)

- **Computed IV surfaces + relative value (cross-sectional VRP / dispersion)** — strongest. Your pricing engine (BS/BjS/CRR validated) lets you rank rich/cheap vol without hist IV.
- **Implied-move / skew as *information* features** (not execution) — already partially validated; refine continuously.
- **Volume surges + greeks for flow proxies** (informed trading signals).
- **Dynamic cost modeling** for any future options execution (stress spreads decisive).

**Not feasible reliably:** True dealer-gamma from missing OI, intraday vol-arb (EOD only), pure market-making. Avoid high-frequency options trading.

## 4. Architecture / Design Gaps

- **Portfolio construction too static.** Fixed sleeves + simple allocator. Missing full risk-parity / vol-targeting / covariance-based optimization at book level. Regime model is good but under-leveraged for dynamic weights.
- **No sacred holdout.** CPCV is strong but no final untouched period post all research. Risk of meta-overfitting.
- **Execution realism lagging.** Marketable limits for PEAD good, but no adverse selection modeling, no dark-pool / algo routing simulation. Options live path (Alpaca) will be painful without better spread/liquidity forecasting.
- **Observability strong but fragmented.** PEAD tracker excellent; unify with trend and potential options sleeves.
- **Data engineering:** Options backfill solid (PIT/survivorship). Equity side needs similar rigor for fundamentals (analyst revisions already used — expand).
- **Over-reliance on rules-based.** Good for PEAD, but hybrid (simple ML on top of events) under-explored.

**Fragility:** Single-operator → kill-switch and test/prod isolation hardening (your recent decisions) are critical and well-executed. But ops burden grows with every sleeve.

## 5. First 5 Things I Would Change (Prioritized)

1. **Implement book-level evaluation framework immediately (before any new sleeve).** Standalone gates are wrong ruler for diversifiers/risk-premia. Run joint CPCV on PEAD+trend + candidate options sleeve. Optimize for combined SR, Calmar, max-DD, and residual alpha. This reframes index short-vol from "KILL" to "potential 0.2-0.4 sleeve contributor."
2. **Pre-register and limit experiments.** One pre-registered spec per major direction (e.g., dispersion params). Track total trials explicitly in DSR. Stop threshold-hunting.
3. **Extend options data pipeline for relative IV surfaces.** Script to compute ATM/OTM IV ranks and skew per underlying daily (using your engine). Feed as features to PEAD/regime. Highest ROI on existing data.
4. **Add a true out-of-sample sacred holdout (last 6-9 months untouched).** Run final promotion only after all tuning on earlier data. Your 4y window allows it.
5. **Harden live execution & reconciliation.** Shadow *everything* longer. Implement daily P&L attribution vs backtest expectation (you have trackers — surface discrepancies aggressively). For options, model realistic fill rates based on volume/notional.

**Flat kill:** Any further standalone short-vol without book context. XS price-feature ML.

**Overall premise challenge:** Single-operator chasing capital-grade *pure alpha* in US equities is extremely hard in 2026 (competition, costs, data arms race). Your PEAD+trend base is respectable. Shift objective slightly toward **robust, diversified, risk-premium harvesting with alpha overlays** — easier to scale, better Sharpe via diversification. Options data is a real asset here, but primarily as signal enhancer + cheap index vol tool, not revolutionary standalone edge.

The harness works. The data is clean. The honesty in killing things is rare and valuable. Focus on integration and book-level metrics next — not more isolated strategy hunts. Prioritize 1-3 above for material improvement within 4-6 weeks.

**End of review.** Questions on any section welcome.