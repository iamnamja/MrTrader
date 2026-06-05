# MrTrader — "What Should We Trade Next?" Briefing (2026-06-04)

> **Purpose:** a self-contained brief to paste into other LLMs. We want **brutally honest**
> critique + **out-of-the-box** ideas for the *next* strategy to research. Please don't be
> polite — tell us what's a waste of time. Think beyond US-equity cross-sectional signals.

---

## 0. TL;DR (the honest state)

A solo quant runs an automated **multi-agent paper-trading** system on Alpaca (~$100k paper,
Russell-1000 universe, daily-bar cadence). Over ~2 months of disciplined, leak-free research we
**killed almost everything we tried** and found exactly **one** edge that survives honest
validation: **PEAD** (post-earnings-announcement drift). PEAD is *real but underpowered*
(event-bootstrap p≈0.19 — **not** significant at conventional levels), long-biased, and
crisis-vulnerable. It's now live in paper. **We need a genuinely different return source** — and
we're skeptical of our own past optimism, so we want hard external pushback.

---

## 1. System architecture (what exists)

- **Live loop (paper):** Portfolio Manager (builds proposals) → Risk Manager (hard risk gates) →
  Redis queue → Trader (marketable-limit execution on Alpaca paper). ~$100k, Russell-1000.
- **Backtest/validation:** custom event-driven simulator (`AgentSimulator`, daily mark-to-market,
  long/short, ATR stops, vol-targeting). **Equity-bars only** — no options Greeks, no futures roll,
  no intraday microstructure modeling.
- **Statistical rigor (this is the system's main asset):**
  - **CPCV** (Combinatorial Purged Cross-Validation), k=8, purge+embargo, per-fold retraining.
  - **N_eff = #folds (8), not #paths** (paths reuse folds → correlated). Path-Sharpe t-stat gate.
  - **Deflated Sharpe Ratio**, **deployment-adjusted Sharpe**, **event-clustered bootstrap**
    significance, OOS purge guards, a **sacred holdout** (2026-11-09+).
  - Promotion gates: avg Sharpe ≥0.80, min/P5 fold ≥−0.30/≥0.0, DSR p>0.95, PF≥1.10, Calmar≥0.30.
  - A reusable **EventEdgeStrategy** harness: a "scorer" = `(day, data) → [(symbol, conf, dir)]`,
    run through the same CPCV + gate + bootstrap. Adding an event edge = write a scorer.
  - A **CAPM beta-isolation** cross-check (`analyst_beta_check.py`) for unmasking beta-as-alpha.
- **Known, important flaw:** **~52% of CPCV fold-evaluations are skipped** (a rolling-window
  overlap guard), which **biases the realized fold distribution toward recent (bull-market)
  regimes** → **all our CPCV numbers are probably optimistic.** We treat absolute Sharpes with
  suspicion and rely on *relative* comparisons on the same biased sample.

---

## 2. The graveyard — everything we tried, with verdicts

| Strategy | Method | Verdict | Why it died |
|---|---|---|---|
| Swing ML ranker | LambdaRank/XGBoost cross-sectional rank of ~1000 stocks on price+fundamental+technical features | ❌ **DEAD** | No out-of-sample IC; collapses in bear folds (label inversion); long-only "edge" was market **beta** |
| Intraday ML | XGBoost meta-model on 5-min features | ❌ **DEAD** | CPCV ≈ −2.8 Sharpe; **transaction-cost drag** dominates any signal |
| Dollar-neutral high-breadth ranker | L/S, 60/60, sector-neutral, SPY beta-hedged | ❌ **DEAD** | On a *verified* neutral book: Sharpe +0.14, t=0.18. The long-only +1.06 was **pure beta** |
| Factor portfolio | momentum+quality composite, long-only | ❌ superseded | cross-sectional; same family as the dead ranker |
| Insider-buying cluster | Form-4 cluster-buy event signal | ❌ **FAIL/weak** | no robust drift |
| **A1: Analyst up/downgrade drift** | event = rating change → forward drift | ❌ **DEAD** | CPCV looked *best-in-campaign* (+0.894, t=2.85) but was a **fold-skip artifact** — dollar-neutral L/S collapsed to t=1.24 and a full-window CAPM regression gave **alpha t=0.20** (noise) |
| **A2: Short-interest factor** | dollar-neutral long-low-DTC / short-high-days-to-cover (Boehmer/Asquith anomaly) | ❌ **DEAD** | CPCV **−1.213, t=−3.53** — the anomaly **reversed** in the 2020-26 meme/retail era (high-SI names squeezed *up*) |
| Small/mid-cap PEAD | PEAD on a smaller-cap universe | ❌ rejected | doesn't survive survivorship + cost modeling |
| **PEAD (large-cap)** | event = EPS surprise → enter next open, hold ~weeks | ✅ **THE edge** | see §3 |

**The pattern:** every attempt to extract alpha from **cross-sectional ML / single-factor
ranking** of US equities came back **noise or beta**. The only thing that worked is an
**economically-grounded, rules-based, discrete-event → drift** mechanism. That's the prior we'd
weight heavily when judging new ideas.

---

## 3. The one survivor — PEAD (and why we're not celebrating)

- **What it is:** long stocks with a positive earnings surprise, enter at the next open, hold
  ~weeks (drift). Long-only, equal-weight, rules-based. Economically grounded (sluggish reaction
  to earnings news), F2-immune (not fit to specific crisis dates).
- **Backtest:** CPCV mean Sharpe **+0.546**. Recently improved to **+0.661** by replacing a crude
  VIX>30 crisis block with an **SPY<200d trend filter** (it's an up-trend harvester).
- **Brutally honest caveats:**
  - **Underpowered / NOT statistically significant:** event-clustered bootstrap **p≈0.19**, HAC
    t≈1.04. The CPCV t=2.26 was *optimistic* (fold-skip bias). We are explicitly treating PEAD as
    a **small diversifier, never a capital centerpiece** until a multi-year *live* track record
    self-certifies it.
  - **Long-biased + crisis-vulnerable:** ~**87% of P&L comes in up-trends**; realized Sharpe ~0.40.
  - **No meaningful live track record yet** (just ramped in paper).
- **Live status:** running in paper, sized aggressively (3× ramp, capped 10%/name, configurable),
  with the trend filter + a realized-Sharpe self-certification clock + ADV-participation/slippage
  capacity instrumentation.

---

## 4. Datasets — what we have vs. can get

**Have (API keys + data in hand):**
- **Alpaca** — equities + **options** trading (paper), real-time/historical equity bars.
- **Polygon** (+ S3 flatfiles) — equity daily/intraday bars, **short interest** (bi-monthly, to
  2017) + **short volume** (daily Reg SHO, to 2024), reference data. *Options data likely on plan
  (unconfirmed).*
- **FMP `/stable`** — EPS/earnings surprise history (PIT via filingDate), analyst grades
  (up/down/maintain), insider (Form-4) trades, institutional 13F, company fundamentals.
- **yfinance** — daily bars, ^VIX, SPY (free, used for backtests).
- **FRED** — macro series. **Finnhub**, **NewsAPI**, **AlphaVantage** — keys held (news/sentiment,
  some premium endpoints gated).
- Backfilled: short interest/volume (540k rows, PIT-safe), regime labels, sector map.

**Can get cheaply/free:**
- **FINRA Reg SHO daily short volume** (free flat files), FINRA short-interest API (free OAuth).
- More FMP endpoints (analyst **price targets**, **estimate revisions**, **guidance**, economic
  calendar) — some gated by plan tier.
- **Options chains/IV** via Polygon (confirm plan) — needed for any vol strategy; *our simulator
  can't price options today.*
- **Crypto** (Alpaca crypto / Polygon crypto) — 24/7, funding rates via exchanges.
- News/8-K/SEC EDGAR filings (free) for NLP event extraction.

**Do NOT have:** futures data/execution, FX spot, tick/order-book microstructure, paid alt-data
(options flow, satellite, card spend), a margin/short-borrow-cost model beyond approximations.

---

## 5. Brutally honest limitations (read before suggesting)

1. **We may be systematically over-optimistic** — the 52% fold-skip biases CPCV toward recent bull
   regimes. Our one "edge" is *not* statistically significant. Assume our backtests overstate.
2. **Narrow search so far:** US equities, daily bars, free fundamental/event data, cross-sectional
   or single-event signals. We have **not** touched: options/vol, futures, FX, crypto, intraday
   microstructure, pairs/stat-arb, cross-asset, or real alt-data.
3. **Simulator constraints:** equity-bars long/short only. **No options Greeks, no futures roll,
   no intraday fills.** Anything needing those requires a *new backtest stack* (real cost).
4. **Capital/execution reality:** one ~$100k paper account, retail marketable-limit fills,
   approximate costs (~10 bps) + slippage. No prime broker, no cheap leverage, no securities
   lending revenue.
5. **Operator bandwidth:** solo, automated; prefers rules-based + economically-grounded over
   fragile ML; values *not fooling ourselves* over chasing the highest backtest number.
6. **The thing we keep re-learning:** cross-sectional ML ranking of equities is dead for us;
   genuinely *different return sources* with strong documented priors are more promising than
   another speculative equity factor.

---

## 6. Candidate directions (our current shortlist — please critique + extend)

Framed by: **prior strength × feasibility on our stack × diversification vs PEAD** (PEAD is
long-biased/crisis-vulnerable, so a complement that pays off in PEAD's drawdowns is worth more
than its standalone Sharpe).

- **Time-series momentum / trend-following** (own each asset when *its own* trend is up; diversified
  ETF basket: equity/bond/gold/commodity/FX ETFs). Decades of OOS evidence; **crisis-positive**
  (best PEAD diversifier); **cheap to test in our existing harness** (ETFs trade like stocks).
  *Our current top pick.*
- **Merger arbitrage** (capture announced-deal spread; market-neutral, defined-risk, deal-break
  tail). Event-driven like PEAD but a different P&L model; needs deal data + a spread/break engine.
- **Options volatility-risk-premium** (sell rich implied vol). Highest standalone EV + persistent,
  but **crisis-negative** (correlates with PEAD's pain) and needs a **new options backtest stack**.
- **Crypto carry/momentum** (perp funding-rate carry; TS-momentum). Different asset class, less
  efficient, 24/7; execution/borrow + venue risk.
- **Pairs / statistical arbitrage** (cointegration, mean-reversion, market-neutral). Crowded/hard,
  but genuinely different from momentum.
- **More event edges** (spinoff drift, buyback announcements, guidance-revision drift, index
  reconstitution, IPO lockup expiry, 8-K NLP). Reuses our engine; weaker priors than the above.
- **Seasonality/calendar** (turn-of-month, FOMC-day drift, overnight-vs-intraday return anomaly).
  Cheap to test; small/fragile.
- **PEAD + complement portfolio** (the real prize): even a mediocre *uncorrelated* second sleeve
  could lift the combined Sharpe more than chasing a high solo number.

---

## 7. What we want from you (the external LLM)

1. **Brutal triage:** of §6, what's a waste of time for a solo retail-paper operator, and why?
2. **Out-of-the-box:** what are we *not* seeing? Return sources with strong priors that fit our
   constraints (equity-bars sim, free/cheap data, ~$100k, rules-based, want crisis-diversification).
3. **For your top 2-3 picks:** the specific signal, the data needed (from §4), how to validate it
   *honestly* on our CPCV harness (or what new harness it needs), and its likely correlation to a
   long-biased earnings-drift book.
4. **Devil's advocate on PEAD:** given p≈0.19, is paper-trading it even worth it, or is it noise
   we've fallen in love with?

*(Internal refs, not needed externally: `docs/living/ML_EXPERIMENT_LOG.md`,
`docs/living/PIPELINE_ARCHITECTURE.md`, `docs/living/PROJECT_STATE.md`.)*
