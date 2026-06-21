# RISK COMMITTEE REVIEW — MrTrader Multi-Premia Book (2026-06-21)

**To:** Solo Operator  
**From:** Grok (as CRO / Quant PM, ex-systematic hedge fund)  
**Subject:** Brutally honest gate on go-live. We have improved since v10, but **do not deploy capital yet**. The second engine is plausible but unproven under family-wise scrutiny; diversification is cosmetic without tail tests; the risk layer is still missing.  

This is implementable. Everything I recommend you can code this week.

## Executive Summary (Lead with what you got wrong)
- **Positive:** You executed well. Trend is real (live track strengthens it). Futures book (carry + xsmom) is the best marginal addition you've found. VRP reversal is smart. Ruler-v2 Type-I control is solid. Data audit clean. IBKR prep good.
- **What you still got wrong / too optimistic:**
  1. **Family-wise multiple testing is the killer.** t=2.29 on the basket is likely inflated. "Gate the basket" after mining 20 families is classic data-snooping. You cannot trust it without the null zoo.
  2. **Diversification is overstated.** Pairwise ~0.1-0.46 hides likely tail dependence. The book is net long risk-on + short-vol (gated). Still effectively one bet in liquidity crises.
  3. **Sizing instinct dangerous.** Any Kelly reference (even haircut) for a solo book is reckless. Freeze at conservative vol-target.
  4. **Promotion ladder too loose.** Paper PASS ≠ ready for even 1 contract.

**Verdict:** Keep on paper. Build the missing layers (B null zoo first, then A risk surface, then C tails). Re-convene after 4-6 weeks of those runs. No capital until t=2.29 survives deflated test **and** tail corr <0.6 in stress.

## Theme B — Is the edge real? (Priority 1 — gates everything)
**B1/B2:** No, I do not fully trust t=2.29 yet. Combining two marginal singles (t=1.76 + 1.60) into significance is a legitimate gain *if* the factors are pre-registered and orthogonal, but here they come from the same family-level search. This is multiple-testing residue until proven otherwise. Bonferroni on 20 families would require t≈3.5+; your survivors likely fail that.

**B3. Concrete null-strategy zoo protocol (build this week):**
- **Universe:** Same 76 futures markets, weekly rebalance.
- **Null generation (rules-based, no model retrain):**
  - 5000+ nulls: Randomize sign of the actual carry/xsmom signals per instrument per rebalance (block-permuted within 3-month blocks to preserve autocorrelation).
  - Additional 2000: Random rank permutations of the 12-1 momentum ranks (preserves cross-sectional distribution).
  - Block bootstrap the returns themselves under null (Politis-Romano, preserve vol clustering).
- **Statistic:** Deflated Sharpe (Bailey et al.) + Hansen SPA (Superior Predictive Ability) test on the full zoo vs your survivors. Also compute empirical family-wise p-value: % of null *books* (equal-weight carry-null + xsmom-null) that achieve residual-α t ≥2.29 vs your live trend book.
- **Target:** Your futures book must have deflated t > 2.0 (or empirical p < 0.05 family-wise). If it drops below, kill or haircut 50%.
- **Evidence to refute:** Run it. If >10% of null books beat your t=2.29, it's residue.
- **Prospective logging:** Git commit every variant with a `research_log.md` entry: "Tested X variants, discarded Y for Z reason, degrees-of-freedom penalty +1". Use a simple counter per family.

This is doable in Python/Pandas in <1 week. Prioritize this over everything.

**B4:** Done via the log above + pre-register every family before testing.

## Theme A — Go-live decisions (the unbuilt layer)
**A1. Sizing:**
- **Target book vol:** 8-10% annualized for $100k solo CTA-style (maxDD budget ~15-20%). Conservative; trend alone at 50% gross already hits decent vol.
- **Per-sleeve:** Inverse-vol risk parity (simple, no HRP needed for 3-4 sleeves). Paper sleeves start at 30-50% of target risk weight, ramp +10% per 6 months clean live track.
- **Margin:** Cap IBKR margin-to-equity at 30% (futures leverage is the real risk). Vol-target overrides but never exceed margin cap.
- **No Kelly.** Freeze gross; use drawdown-based scaling only.

**A2. Cross-venue:**
- **Unified risk surface:** Daily Python ETL pulling positions/cash/margin from both brokers' APIs → single Postgres view. Compute book vol (exponentially weighted, 21d), gross, net, factor betas (to SPX/VIX/equity).
- **Single kill-switch:** Central FastAPI endpoint that triggers flatten on both via their SDKs. Dead-man timer.
- **Reconciliation:** Hourly snapshot compare; alert on >0.5% NAV drift.

**A3. Forward risk:**
- **Corr-spike de-gross:** 21d rolling pairwise corr > 0.6 (from unconditional baseline) → de-gross 30%. Calibrate on historical stress windows only (no look-ahead).
- **Book DD ladder:** -5% → 0.75× gross; -10% → 0.5×; -15% → cash. Linear, no discretion.

**A4. Promotion ladder:**
- **Rung 0→1 (IBKR paper):** 3 months, ≥12 rolls, ≥1 vol spike, slippage < 30% of modeled edge, tracking error < backtest vol.
- **Rung 1→2 (tiny live):** 6 months paper on IBKR + residual-α consistency (t>1.5 live vs paper), max 1-2 contracts per market. Demote on >2× expected DD or slippage >50% edge.
- **Rung 2→3 (scale):** 12 months live, live SR within 0.3 of backtest, no demotion triggers.

**A5. No-go list (hard gates):**
- t=2.29 fails deflated test.
- Stress tail corr >0.65 any pair.
- IBKR paper slippage >40% of edge.
- Book maxDD in synthetic 2008 replay >35%.
- Any sleeve fails live-paper tracking for 3 consecutive months.

## Theme C — Diversification & tails
**C1:** Kidding yourselves on full diversification. It's 3 risk-on trend/carry/mom + gated short-vol. Net short crisis. Tell: high loading on equity risk premium + short VIX in normal times.

**C2. Tail test:**
- Exceedance correlation (top 5% worst equity days or VIX>30 windows).
- Threshold: If conditional corr >0.7, it's one bet. Use copula if you want fancy (but exceedance sufficient).

**C3. Stress test:**
- Joint historical scenario replay (2008, 2018, 2020, 2022) across all sleeves using actual futures prices.
- Crisis-weighted CPCV: force 30% of folds to be stress periods.
- Block bootstrap with oversampled crisis blocks.

**C4. Convex sleeve?** Yes — build a simple **long bond/gold trend** or **long VIX when cheap** (opposite gate). Test via same Ruler + tail contribution (negative beta in crises). Worth it if it reduces book tail DD > bleed in normal times.

**C5:** Drop VRP for now or pair 1:1 with long-convex. Its 0.46 corr + short nature concentrates tail risk.

## If I were chairing your risk committee Monday morning
**Top 5 moves, in order:**
1. **FIRST:** Build & run the null zoo (Theme B3). This gates deployment. Do not pass Go without it.
2. Implement unified risk surface + kill-switch (A2).
3. Run full joint stress scenarios (C3) + conditional corrs.
4. Code the promotion ladder and no-go gates into the orchestrator.
5. **Refuse to let you do:** Deploy any IBKR capital before null zoo + 3 months clean IBKR paper. No exceptions.

Re-convene after these. You're close — closer than most solo ops — but the jump to real money requires these proofs. Cash is the best position until then.