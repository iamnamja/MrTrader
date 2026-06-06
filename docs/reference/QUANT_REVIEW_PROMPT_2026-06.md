# Deep-Dive Quant Review — MrTrader Automated Trading System

> **YOU ARE:** a world-class quantitative researcher / PM from a top systematic hedge fund
> (think Renaissance / Two Sigma / DE Shaw / AQR), with deep experience building and running
> **automated, multi-agent trading platforms**. You've seen hundreds of strategies live and die.
>
> **YOUR JOB:** Read this dossier and deliver a **brutally honest** review. Critique the design,
> the research process, and the statistics. Then tell us **where the alpha actually is** for an
> operator with our constraints. **You are free to recommend a complete redesign**, to say our one
> "edge" is noise, to tell us what **data we're missing**, and to propose **specific models /
> return sources** to try. Do not be diplomatic — we want the review a skeptical IC would give.
>
> **CONTEXT:** Solo operator, fully automated, currently **paper-trading** (Alpaca, ~$100k),
> goal = find robust, deployable alpha and eventually trade real capital. We value *not fooling
> ourselves* over chasing the prettiest backtest. We've already killed most of what we tried; we
> want genuinely new directions + a critique of our method.

---

## 1. What the system is

A fully-automated, multi-agent equities trading platform:
- **Universe:** Russell-1000 (~750 liquid US names after filtering).
- **Cadence:** daily-bar "swing" horizon (hold days–weeks) + a (currently dormant) 5-min intraday horizon.
- **Capital:** ~$100k Alpaca **paper** account; retail marketable-limit execution; ~10 bps cost assumption + measured slippage.
- **Stack:** Python; XGBoost/LambdaRank models + rules-based "scorers"; Redis pub/sub between agents; SQLite for state/audit; FastAPI + a React dashboard.

---

## 2. Live architecture — the three agents & the trade flow

A 60-second heartbeat drives a daily schedule (all times ET):

```
09:00  Premarket routine (regime read, universe refresh)
09:30  PORTFOLIO MANAGER: run swing selector (daily bars) -> proposals
09:45  (intraday selector — currently dormant: intraday model is dead)
09:50  Send swing proposals downstream (after open volatility settles)
11:15-13:00  Lunch block (no new entries)
16:30  EOD jobs: backfill decision outcomes, write daily summary, PEAD realized-Sharpe row,
       Friday weekly rollup
17:00  Retrain swing model on latest daily data (gate-checked; auto-rollback on fail)
```

**Agent 1 — Portfolio Manager (PM):** the alpha layer. Each day it runs a configurable
**"selector"** over the universe to produce ranked trade proposals (symbol, direction, confidence,
size, stop/target). Selectors are pluggable:
- `ml_model` — XGBoost/LambdaRank cross-sectional rank (now considered dead, see §6)
- `factor_portfolio` — momentum+quality composite (dead family)
- **`pead`** — rules-based post-earnings-drift scorer (**the only live edge**)
- `quality_short`, `pead_quality_short` — short/combined variants
- (research) `analyst_revision`, `short_interest_factor` — both killed
PM sizes positions via **vol-targeting** (target 0.5%/name equity-vol, ATR-based) bounded by a
per-name cap and a strategy budget; applies a **regime-sizing multiplier** (RISK_OFF → 0.3×) and,
for PEAD, an **SPY<200d trend filter** (block new entries in downtrends) + a configurable
"aggressive paper ramp" multiplier. Emits proposals to Redis.

**Agent 2 — Risk Manager (RM):** independent veto layer. Subscribes to proposals; approves/rejects
each against **10 hard gates** (config-tunable, no restart): buying-power, **per-name size cap**
(PEAD gets a higher ceiling), **sector concentration**, **correlation risk**, **daily-loss limit**,
**account drawdown**, **max open positions**, **portfolio heat** (sum of per-position risk),
**net exposure**, **short notional**. Approved → Redis queue; rejected → logged with reason. A
kill-switch suppresses all trading on breach.

**Agent 3 — Trader:** execution layer. Pulls approved proposals; re-checks an **entry-quality
gate** (spread / adverse-move / price-run thresholds, looser for PEAD's post-earnings gappers);
routes **marketable-limit** orders (crosses the spread ~10 bps for fills that track the backtest's
next-open assumption); records intended-vs-filled **slippage_bps** + ADV participation. Manages
exits (ATR stops/targets, max-hold-days, EOD limit expiry).

**Audit/observability:** every decision (and every *gated* decision) is logged to a
`decision_audit` table with backfilled forward outcomes (incl. counterfactuals for blocked
trades), a daily risk-metrics summary, and a PEAD-specific tracker (signals → entered → filled →
realized/unrealized P&L → rolling realized Sharpe vs backtest).

---

## 3. Data points we have

**Price/market:** Alpaca + Polygon (daily + 5-min bars, S3 flatfiles), yfinance (daily, ^VIX, SPY).

**Engineered features (per name, point-in-time):** RSI, MACD, EMAs + price-vs-EMA position, price
changes, 52-week high/low ratio, volume ratio + trend, trend flags, annualized volatility,
**momentum (5/20/60/252d)**, **ATR(14) normalized**, Bollinger %B, plus regime/breadth/dispersion
context.

**Fundamental/event (FMP `/stable`, PIT via filingDate):** quarterly EPS **earnings surprise**,
**analyst grades** (up/down/maintain), **insider (Form-4)** trades, **institutional 13F**, company
fundamentals.

**Short data (Polygon/FINRA, PIT, backfilled):** bi-monthly **short interest + days-to-cover**
(to 2017), daily **short volume** (Reg SHO, to 2024).

**Macro:** FRED series + a "macro NIS" (net-impulse) feature set, point-in-time.

**Regime:** a regime classifier (VIX/trend/breadth) producing RISK_ON / CAUTION / RISK_OFF labels
used for sizing.

**Keys held (not all fully utilized):** Alpaca (incl. **options trading**), Polygon (+S3),
FMP, FRED, Finnhub, NewsAPI, AlphaVantage, Anthropic.

---

## 4. Model training

- **Swing model (XGBoost classifier / LambdaRank ranker):** trained on the engineered features
  above with triple-barrier / forward-return labels; HPO via Optuna; **per-fold retraining** in
  walk-forward. Nightly `retrain_cron` retrains, runs the WF gate, and **auto-rolls-back to the
  previous champion if the gate fails** (it has been failing — the model is not deployable).
- **Intraday meta-model:** XGBoost on 5-min features + a scaler; **dead** (cost-drag).
- **Regime classifier:** separate model for RISK_ON/OFF labels (sizing input).
- **Rules-based scorers** (PEAD, analyst, short-interest, quality-short): **not trained** — they're
  economically-motivated rules with a few config thresholds, run through the same validation harness.
- **Guards:** an **OOS purge guard** refuses to promote a model whose test folds lack the required
  purge gap vs its training cutoff; a **sacred holdout** (2026-11-09+) is never touched.

---

## 5. Backtesting & validation (the system's main asset — and its main flaw)

**Simulators:** a custom event-driven `AgentSimulator` (daily mark-to-market, long/short, ATR
stops, vol-targeting, regime sizing, optional SPY beta-hedge) is the backtest engine. **It is
equity-bars only** — no options Greeks, no futures roll, no intraday fills.

**Walk-forward + CPCV:** we validate with **Combinatorial Purged Cross-Validation** (López de
Prado style):
- k=8 fold-groups, choose-2 test combinations → 21 paths, **purge + embargo** around each test
  window, **per-fold retraining** (no leakage of future into the model).
- We score significance honestly: **N_eff = #folds (8), NOT #paths (21)** (paths reuse folds →
  correlated); a **path-Sharpe t-stat** gate; **Deflated Sharpe Ratio**; **deployment-adjusted
  Sharpe** (penalizes idle cash); and for event strategies an **event-clustered bootstrap** that
  resamples *events*, not days.
- **Promotion gates:** avg Sharpe ≥0.80, min/P5 fold ≥−0.30 / ≥0.0, DSR p>0.95, PF≥1.10,
  Calmar≥0.30, worst-regime-Sharpe floor.

**⚠️ THE KNOWN FLAW (please weigh this heavily):** ~**52% of CPCV fold-evaluations are skipped**
by a rolling-window overlap guard, and the surviving distribution is **biased toward recent
(bull-market) regimes**. We believe **all our CPCV Sharpes are optimistic**, and we lean on
*relative* comparisons on the same biased sample rather than absolute levels. We'd value a view on
whether this invalidates our conclusions.

---

## 6. What we've tried — the graveyard (with verdicts)

| Strategy | Method | Verdict | Why it died |
|---|---|---|---|
| Swing ML ranker | XGBoost/LambdaRank cross-sectional rank on price+fundamental+technical features | ❌ DEAD | no OOS IC; bear-fold label inversion; long-only "edge" was market **beta** |
| Intraday ML | XGBoost meta on 5-min features | ❌ DEAD | CPCV ≈ −2.8; **cost-drag** dominates |
| Dollar-neutral high-breadth L/S ranker | 60/60 sector-neutral, SPY beta-hedged | ❌ DEAD | verified-neutral book: Sharpe +0.14, **t=0.18**; long-only was pure beta |
| Factor portfolio | momentum+quality composite, long-only | ❌ superseded | same dead cross-sectional family |
| Insider-buying cluster | Form-4 cluster-buy events | ❌ weak | no robust drift |
| Analyst up/downgrade drift | rating-change event → forward drift | ❌ DEAD | best-*looking* (CPCV +0.894, t=2.85) but a **fold-skip artifact** — dollar-neutral L/S collapsed to t=1.24, full-window **CAPM alpha t=0.20** (noise) |
| Short-interest factor | dollar-neutral long-low / short-high days-to-cover (Boehmer/Asquith) | ❌ DEAD | CPCV **−1.21, t=−3.53** — anomaly **reversed** in the meme era (high-SI squeezed up) |
| Small/mid-cap PEAD | PEAD on smaller caps | ❌ rejected | fails survivorship + cost modeling |
| **PEAD (large-cap)** | EPS-surprise event → enter next open → hold ~weeks, long-only | ✅ **the one survivor** | see §7 |

**Pattern:** every cross-sectional ML / single-factor ranking of equities → **noise or beta**.
The only thing that worked is an **economically-grounded, rules-based, discrete-event → drift**
mechanism. (Is this a real lesson, or survivorship bias in our own research? Tell us.)

---

## 7. The one survivor — PEAD (and why we're skeptical of it)

- Long positive-EPS-surprise names, enter next open, hold ~weeks; long-only, equal-weight, rules-based.
- CPCV mean Sharpe **+0.546**, improved to **+0.661** by swapping a VIX>30 block for an SPY<200d
  trend filter (it's an up-trend drift harvester: ~**87% of P&L in up-trends**, realized SR ~0.40).
- **Brutally honest:** event-clustered bootstrap **p≈0.19** (NOT significant), HAC t≈1.04; the
  CPCV t=2.26 was optimistic (fold-skip). **Long-biased, crisis-vulnerable, no live track record
  yet.** We're treating it as a small diversifier and running it in paper to build a real record —
  *not* as proven alpha. **Is paper-trading a p≈0.19 signal rational, or are we in love with noise?**

---

## 8. Constraints & what we explicitly have NOT tried

- **Have NOT touched:** options/volatility strategies, futures, FX, **crypto**, intraday
  microstructure / order-flow, pairs/stat-arb, cross-asset carry, real alternative data
  (options flow, satellite, card-spend), NLP on filings/news.
- **Simulator can't model:** options Greeks, futures roll, intraday fills → any of those needs a
  **new backtest stack** (real cost; tell us if it's worth it).
- **Operator reality:** solo, automated, ~$100k paper, retail fills, no prime broker, no cheap
  leverage, no securities-lending revenue, limited engineering bandwidth.
- **Can get cheaply:** FINRA short data (free), more FMP endpoints (price targets, **estimate
  revisions**, guidance), Polygon **options chains/IV** (confirm plan), crypto data, EDGAR filings.

---

## 9. What we want from you (be specific)

1. **Method critique:** Is our CPCV/significance approach sound? How damaging is the 52% fold-skip
   bias? Are our gates sensible? What would you change about the research process itself?
2. **Brutal triage of PEAD:** given p≈0.19, is running it worth it? What would *prove* or *kill* it?
3. **Where's the alpha for us?** Given our constraints (equity-bars sim, free/cheap data, ~$100k,
   rules-based preference, want crisis-diversification vs a long-biased earnings-drift book), name
   the **2–4 return sources you'd actually pursue** and why. Be concrete: the signal, the data
   needed (from §3/§8), how to validate it honestly (our harness or a new one), expected capacity,
   and likely correlation to PEAD.
4. **Missing data:** what data (free or paid-but-worth-it) would most change our odds of finding alpha?
5. **Models:** any modeling approach we're dismissing too fast (e.g., is cross-sectional ML truly
   dead, or did we do it wrong)? Time-series vs cross-sectional? Ensembles? Regime-conditioning?
6. **Redesign:** if you'd tear down and rebuild differently (different universe, asset class,
   horizon, or architecture), say so and sketch it.

Please rank your recommendations by expected value for *our* situation, and flag anything that's a
glamorous-but-wrong waste of time. Assume we'll act on the consensus across several reviews.
