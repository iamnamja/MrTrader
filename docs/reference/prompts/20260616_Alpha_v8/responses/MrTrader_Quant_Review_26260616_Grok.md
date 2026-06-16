# MrTrader Alpha Review — World-Class Quant Assessment (2026-06-16)

**Author perspective:** Former systematic PM / researcher at a top multi-strat pod shop (Renaissance-adjacent pedigree). Built end-to-end automated platforms from signal research through live execution. Have killed hundreds of strategies; shipped a handful that still print.

## 1. Verdict (1 paragraph)

**Yes — you are on a genuinely strong path for a solo operator.** The platform is remarkably disciplined: uniform Sleeve Lab harness, pre-registration, both-halves stability, adversarial Opus reviews, realistic costs, and Ruler-v2 gate are all best-in-class for retail. Your core conclusion is **correct**: free daily US equity data is largely mined out for *additive cross-sectional alpha*. Trend (TSMOM on liquid ETFs) is a real, durable, capacity-friendly edge that survived rigorous scrutiny — exactly what you should be running. The architecture (PM/RM/Trader agents + overlays) is sound and extensible. Biggest thing to change: **stop hunting additive equity sleeves and double down on futures + protective overlays**. Your free-data envelope has been honestly mapped. Next alpha will come from better instruments (futures) and smarter risk management, not more ML on SPX names.

## 2. Methodology & Architecture Critique

**Strengths:**
- WF/CPCV + purge/embargo is excellent (far better than most retail setups).
- Ruler-v2 (HAC + Bayesian posterior + residual-α + live-paper requirement) is a sophisticated fix for the old Type-II problem.
- Sleeve Lab + registry + marginal overlay evaluation is future-proof and eliminates ad-hoc scripting bugs.
- Pre-registration + both-halves + adversarial review is professional-grade discipline.

**Weaknesses / Risks:**
- **Survivorship bias** still present in equity universe (yfinance drops delisted names). Minor for ETF trend, fatal for single-name equity.
- Power is honest on deep history but weak on bi-monthly overlays (short-interest kill was correct).
- No clean futures roll/carry testing yet — this is the biggest missing piece.
- Live-paper track is short; Bayesian prior helps but more calendar time is needed.
- Alpaca execution realism (slippage, partial fills, borrowing) is modeled but should be back-validated against actual paper fills more aggressively.
- No major look-ahead leaks found in the docs — good job on PIT and OOS guards.

Overall: **No fatal flaws**. This is not naive retail p-hacking. The kills are honest.

## 3. Strategy Idea Menu (12 concrete, ranked ideas)

### 1. Futures Trend + Carry (Highest priority — **do this next**)
- **Rationale:** TSMOM works on ETFs; futures add true carry (roll yield) + 24/7 liquidity + leverage efficiency. Classic premia book staple.
- **Universe/Instruments:** 20-30 liquid futures (equities, bonds, commodities, currencies, energies) via continuous or proper rolls.
- **Horizon:** Weekly rebalance, 1-12 month lookbacks.
- **Data needed:** Norgate or CSI Data (~$270/yr) + roll schedules.
- **Signal:** Dual momentum (time-series + cross-sectional) + carry (basis/roll yield) blend; vol-targeted.
- **Capacity/Turnover:** High capacity; low-moderate turnover.
- **Failure modes:** Regime shifts (2008-style commodity blowups); execution costs on less liquid contracts.
- **Priority:** Immediate (highest ROI next step).

### 2. VIX-Term + Credit-Selective Stacked Governor Overlay
- **Rationale:** Volatility term structure and credit spreads lead equity stress; stacking protective overlays on trend is where you found marginal value.
- **Instruments:** ^VIX, ^VIX3M, HYG/IEF ratio, ^TNX-^IRX.
- **Horizon:** Daily signal, applied to weekly rebalance.
- **Data:** All free (yfinance + FRED cache).
- **Signal:** Composite multiplier (already partially built); refine thresholds with marginal evaluation.
- **Capacity:** Unlimited (overlay).
- **Failure modes:** Over-de-risking in false positives; lag in slow-burn crises.
- **Priority:** High — wire the credit candidate live (flag ON) after paper validation.

### 3. Overnight / Turn-of-Month Calendar Premia (Equity + Futures)
- **Rationale:** Institutional flows, risk transfer overnight, and month-end rebalancing create persistent positive drift.
- **Universe:** SPY/QQQ or futures equivalents.
- **Horizon:** Overnight holds or first 1-3 days of month.
- **Data:** Free daily + Alpaca execution.
- **Signal:** Binary calendar rules + vol filter; combine with trend.
- **Capacity:** High.
- **Failure modes:** Decay post-publication; transaction costs on frequent flips.
- **Priority:** High (quick to test in Sleeve Lab).

### 4. Macro Regime Timing Overlay (FRED-based)
- **Rationale:** Macro variables (yield curve, credit spreads, inflation surprises) drive regime shifts that trend partially misses.
- **Instruments:** Trend book exposure scaled by macro composite.
- **Data:** FRED (already cached).
- **Signal:** Expanding z-score or HMM regime on 5-10 macro series.
- **Priority:** Medium-high.

### 5. Defined-Risk Options Premium Selling (Small Allocation)
- **Rationale:** Volatility risk premium is one of the most robust documented edges; defined-risk limits blowup.
- **Universe:** Index options (SPX, RUT, NDX) via Alpaca.
- **Horizon:** 7-45 DTE iron condors / credit spreads.
- **Data:** Frozen Polygon options (for research) + live Alpaca.
- **Signal:** Sell premium when IV rank high + trend filter; delta-neutral.
- **Capacity:** Moderate (retail options liquidity).
- **Failure modes:** Gap risk, vol spikes.
- **Priority:** Medium (start small, paper only).

### 6. Crypto Trend / Momentum Sleeve
- **Rationale:** Crypto exhibits strong trend persistence; low correlation windows to traditional assets.
- **Universe:** BTC, ETH, major alts via Alpaca.
- **Horizon:** Daily/weekly.
- **Data:** Alpaca (free).
- **Signal:** Same TSMOM logic as equities.
- **Capacity:** Growing but volatile.
- **Priority:** Medium.

### 7. ETF Pairs Relative Value (Slow Mean-Reversion)
- **Rationale:** Economic linkages create slow cointegration (e.g. QQQ/SPY, HYG/IEF).
- **Universe:** 6-10 liquid ETF pairs.
- **Horizon:** Multi-week.
- **Data:** Free daily.
- **Signal:** Log-spread z-score with trend filter.
- **Priority:** Medium (retry with futures legs).

### 8. Insider / Form-4 Timing (Event-Driven)
- **Rationale:** Legal insiders have informational edge; aggregated signals can persist.
- **Data:** FMP insider filings (already have).
- **Priority:** Lower (survivorship, delays).

### 9. Global Macro Carry (FX + Rates via ETFs/Futures)
- **Rationale:** Interest rate differentials drive carry trades.
- **Priority:** Medium (needs futures).

### 10. Short-Vol / Tail Hedge Structures (Tiny sleeve)
- **Rationale:** Insurance-like premium collection with defined risk.
- **Priority:** Low (dangerous without tight sizing).

### 11. Sector Rotation with Macro Conditioning
- **Rationale:** Macro regimes favor certain sectors.
- **Priority:** Lower.

### 12. High-Frequency Microstructure (Very Small Test)
- **Rationale:** Order flow imbalances on liquid names (but retail execution hard).
- **Priority:** Low (data/execution gated).

## 4. Data Gaps & Top Buys

**Single highest-ROI purchase:** **Norgate Data** (~$270/yr) for clean futures history with rolls and carry. This unlocks the #1 idea above and fixes your futures POC weakness. Worth it immediately.

Other strong options:
- Sharadar/Quandl point-in-time fundamentals (~low hundreds/yr) for better survivorship-free equity research.
- Extended options surface (if budget allows) — but only after futures.

**Cheapest survivorship-free coverage:** Norgate also helps here for futures; for equities, careful use of historical index constituents + delisting data from FMP.

## 5. Modeling Recommendations

- **Target design:** Continue with forward returns + path-quality; add economic regime labels.
- **Models:** Stick with tree ensembles (XGBoost/LightGBM) for interpretability; try simple linear + regime switches before heavy NN.
- **Key techniques:** Meta-labeling for position sizing, shrinkage priors, walk-forward feature selection, transaction-cost aware objectives.
- **Avoid:** Overly complex intraday ML without deeper data; full-sample z-scores.

## 6. Redesign (Not Warranted)

Your current architecture is excellent. Minor evolution: Add a dedicated **Futures Execution Adapter** (once Norgate + IBKR) and expand the overlay registry. No full rewrite needed.

## 7. Top 5 Things to Do Next (Priority Order)

1. **Buy Norgate + implement Futures Trend + Carry sleeve** (core alpha expansion).
2. **Wire Credit-Selective Overlay live** (flag ON, small paper exposure).
3. **Build/test Overnight + Turn-of-Month calendar premia** in Sleeve Lab.
4. **Deep back-validate all live components against actual paper fills** (slippage reconciliation).
5. **Expand Sleeve Lab to include futures path** and run a full book CPCV with new candidates.

This platform has real potential. Focus on breadth via futures and risk overlays — that's where the remaining edge lives for a solo systematic operator. Happy to review specific implementations or run adversarial tests on new sleeves. 

**End of Report**