# 08 — External Quant Review: MrTrader Program Assessment

**Date**: June 2026  
**Reviewer**: World-class hedge fund quant / PM (external perspective)  
**Scope**: Brutal honesty on all provided docs (01-07). No fluff.

## Executive Summary
Your program is one of the most disciplined solo systematic efforts I've seen. Pre-registration, two-track gate, adversarial reviews, and honest caveats show real intellectual honesty. You're not fooling yourself on most kills.

**Core verdict**:
- **Real alpha**: Yes, but thin. ETF TSMOM trend is legitimate (matches priors). Futures carry is promising and likely real, but numbers are optimistic pre-roll/execution.
- **Equity directional alpha mined out**: Sound conclusion on free/survivorship-biased data. Not ruler artifact.
- **State**: Effectively a single-bet book with cash. Carry addition is the right next step if hardened.
- **Risks**: Concentration, execution gaps, data biases. Architecture mostly sound for solo but needs resilience hardening.
- **Overall**: Strong process. Weak book diversification. Prioritize execution over new ideas.

## 1. Assessment of Current Edges
**TSMOM Trend (LIVE)**:
- Matches priors: Classic multi-asset trend-following works, Sharpe ~0.7 plausible on 19y. Weekly rebalance and inverse-vol sensible. Crisis behavior expected.
- No major red flags. VIX governor good overlay.

**Futures Carry (NEW)**:
- Plausible. Term-structure carry is a known premia (commodity literature). Modern persistence (+0.89 post-2015) is encouraging vs decayed trend.
- **Caveats I push**:
  - Roll cost: Your 1.1-1.9% estimate seems reasonable; honest SR ~0.55 is more believable. Re-run with explicit roll simulation (contract switching + slippage).
  - Survivorship/universe: Minor issue per your test, but confirm with point-in-time liquidity filter.
  - Execution: Critical. Futures rolls have real costs/slippage. Do IBKR paper immediately.
- Track-B improvement real if numbers hold. Good diversifier.

**Combined**: Trend + Carry could be solid CTA core (~0.9 Sharpe). Do **not** add decayed futures trend.

**Equity conclusion**: "Free daily US-equity directional alpha mined out" — correct. Your kills (PEAD, XS-ML, etc.) align with post-2010 quant experience. Beta misframing common.

## 2. Validation Harness (Ruler-v2)
**Do I trust verdicts?** Mostly yes.
- Two-track gate is sophisticated and addresses prior Type-I/II errors.
- Pre-registration + sub-period stability + controls = strong anti-p-hacking.
- CPCV/purge good. Bayesian + HAC solid.
- **Weaknesses**:
  - PAPER tier SR≥0.30 still permissive for noisy short histories.
  - Power issues on short data (options 4y) likely killed real-but-weak premia.
  - In-sample vol-matching in Track-B: switch to PIT rolling for purity.
- **Re-runs to settle doubts**:
  - Re-test VRP/index short-vol on full options + NBBO: extend forward log to 6+ months, compute realistic transaction costs (bid-ask + slippage).
  - Equity kills on Norgate stocks (see Data section).
  - Carry: full PIT roll simulation + vol-target overlay.

Ruler is not broken; it's appropriately conservative for solo capital.

## 3. Architecture
**Single-process orchestrator**: Acceptable for solo paper/live at $100k scale. Resilience risks real (crash = downtime, state drift).
- **Improvements**:
  - Decouple execution: Separate Trader process/microservice with Redis queue. Heartbeat + auto-restart.
  - Broker reconciliation: More frequent (every 5min) + alert on drift > threshold.
  - Kill-switch: Hard broker-level (not just app).
  - Monitoring: Add Prometheus/Grafana for metrics, not just React.
  - Backups: Daily DB snapshots + offsite.

**Sleeve-Lab + pre-reg**: Excellent backbone. Not biasing toward slow sleeves — your turnover proof is correct.

**Missing for robust book**:
- Regime detection / dynamic allocation (beyond fixed weights).
- Position-level VaR/stress (beyond per-name caps).
- Capital allocation (Kelly across sleeves with correlation matrix).
- Drawdown-based de-risk (global, not just VIX).

## 4. Data Inventory & Gaps
**Highest-EV next buy**: **Norgate US Stocks (Platinum)**. Survivorship bias contaminates all equity work. Re-test PEAD/short-volume/XS on clean data. Cost justified now that you have futures working.

Alternatives: Mature options NBBO log first for VRP.

**Strategies on existing data** (top prioritized, concrete):

1. **Futures Ensemble (Carry + Trend + MOM variants)**  
   - Universe: Your 76 Norgate.  
   - Signal: Meta-ensemble (equal-risk or Bayesian avg of carry, TSMOM, breakout, carry-zscore).  
   - Test: Sleeve-lab PIT WF/CPCV, Track-B vs current.  
   - Kill if no dSR improvement or modern SR<0.4.  
   - Feasibility: High (reuse engines).

2. **Macro-Regime Overlay on Trend/Carry**  
   - Features: FRED (yield curve, inflation, recession prob).  
   - Signal: Reduce exposure in high-regime-risk (e.g., inverted curve).  
   - Test: Overlay on combined book.  
   - Expected: Tail protection, low turnover.

3. **Options VRP Redux (once NBBO matures)**  
   - Defined-risk index puts (not single-name).  
   - Realistic costs from logger.  
   - If SR>0.4 net + Track-B positive → allocate small.

**Not suggesting**: Intraday (turnover kills proven), more equity XS-ML (likely dead).

## 5. Top 3-5 Concrete Moves (Monday Morning)
1. **Harden Carry**: Implement explicit roll cost/slippage sim. Run IBKR paper tracking parallel to signal. If survives 1-2 months → small live allocation (10-20% risk).
2. **Buy Norgate Stocks**: Re-test equity kills + FINRA XS on clean data.
3. **Resilience**: Implement process decoupling + enhanced reconciliation/alerts.
4. **Book**: Add carry to live (post-hardening). Increase trend to 50% as planned. Build simple ensemble.
5. **Monitor**: Weekly regime review. Pre-register next campaign (macro + options).

## Final Candor
You're doing it right — rigorous, honest, iterative. Edges are real but book is fragile (one main bet). Execution/roll proof for carry is the make-or-break. Diversify via futures ensemble + data buy. Not "beta kidding yourself" on trend/carry. Process beats 95% of retail quants. Push hard on the next 2-3 months validation.

**Evidence needed**:
- Carry post-roll SR >0.5 + live-paper persistence.
- Norgate equity re-tests confirming kills.
- No major drift in reconciliation.

Feed this back or ask for code sketches on any item. 

---
*End of Review*