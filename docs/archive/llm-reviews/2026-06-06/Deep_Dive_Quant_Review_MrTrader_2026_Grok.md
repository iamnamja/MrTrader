# Deep-Dive Quant Review: MrTrader Automated Trading System

**Date:** June 2026  
**Reviewer:** Grok (world-class systematic quant PM perspective)

## Executive Summary (Brutally Honest)
Your setup is impressive for a solo operator: clean architecture, thoughtful risk gates, honest self-critique, and a proper (if flawed) validation harness. You've correctly killed the usual retail traps (overfit cross-sectional ML, pure technicals, beta-as-alpha).

**Core issue:** Mostly weak, long-biased, crisis-vulnerable drifts in a hyper-competitive equity market. PEAD is the survivor for good economic reasons, but stats (p≈0.19 clustered bootstrap, HAC t≈1.04) indicate noise with bull-market help. Paper-trading it is okay for data, but don't scale yet.

Cross-sectional ML/factors died because public signals in liquid large-caps are arbitraged or collapse to beta. CPCV bias toward recent regimes inflates results.

**Alpha for your constraints** (solo, $100k paper, equity-bars, cheap data, rules preference): Discrete event-driven + time-series/relative value edges. Focus on capacity, robustness, crisis diversification.

**Ranked Recommendations (by EV):**
1. Enhanced PEAD + NLP/text (high EV)
2. Time-series momentum + regime (medium-high)
3. Relative value / short-biased (medium)
4. Options/vol event plays (if sim upgraded)

## 1. Method Critique
CPCV + purge/embargo + Deflated SR + event bootstrap + sacred holdout is excellent—rare for solo setups.

**52% fold-skip bias is damaging**: Selects recent bull periods, optimistic Sharpes, under-samples crises. Use for relative triage, not absolute belief. Survivorship in your research is real.

**Gates**: Sensible but add:
- Stricter regime-stratified floors
- Full-sample CAPM/FF5 alpha t-stats
- Stronger event clustering
- Conservative purge

Research process strong on audit/observability. Nightly retrain with rollback good. Improve with more crash stress tests and implementation realism.

## 2. Brutal Triage of PEAD
**Worth paper-trading?** Yes, small scale (10-20% alloc) to build live record. Not core proven alpha yet.

**To prove/kill:**
- 2+ years live/paper with costs
- Variants: SUE + gap filters + transcript NLP + attention proxies
- Kill if live SR <0.4 or excessive bear drawdown
- Capacity good for your size

Rational to continue cautiously.

## 3. Where's the Alpha? (Top 2-4 Sources)
Prioritize rules/simple, events, time-series, low-cost data, crisis-hedging.

**1. Enhanced Event-Driven Drifts (PEAD family)**
- Signal: EPS surprise + revisions + guidance tone + news volume
- Data: FMP, EDGAR, NewsAPI/Finnhub + transcript NLP (Anthropic)
- Validation: Your harness + bootstrap
- Corr to current PEAD: High but diversifies with shorts/negatives
- Why: Persistent underreaction; fits your style perfectly

**2. Time-Series Momentum (TSMOM) + Regime**
- Signal: Absolute momentum (1-12mo) long/flatten/short; regime-gated
- Data: Your price bars; extend to ETFs
- Validation: CPCV with regime strat
- Why: More robust than CS; literature support; diversification

**3. Relative Value / Pairs / Short Bias**
- Signal: Cointegration or short weak vs long strong (with short data filters)
- Data: Short interest + fundamentals
- Why: Adds crisis hedge; fixes past L/S failures with better gates

**4. Options / Vol Event Plays (Upgrade sim)**
- Signal: Earnings vol + post-event directional/vol crush
- Data: Polygon options
- Why: Natural hedge; higher potential Sharpe

**Avoid:** Pure CS ML on public data, HFT, expensive alt data initially.

## 4. Missing Data (High ROI)
- Earnings transcripts + NLP sentiment/uncertainty
- Estimate revisions + guidance history
- Options IV/chains/order flow
- Aggregated news/social attention
- More macro (FRED/COT)

## 5. Models
CS ML not dead but correctly killed here. Shift to:
- Time-series per-name + regime features
- Rules + light ensembles
- Heavy regime conditioning
- Event-specific trees/logistic

## 6. Redesign Sketch
**Keep:** Multi-agent, Redis, audit, vol-targeting, risk gates.

**Change:**
- Add Event Scanner agent
- Ensemble PM (PEAD + TSMOM + relative)
- Broaden to ETFs/futures for TSMOM
- Upgrade sim for options
- Multi-strat risk parity portfolio
- More live incubation + crash stress

You're disciplined—strong position. Next: PEAD enhancements + TSMOM prototype. Build live track record. Compound small robust edges.

*End of Review*