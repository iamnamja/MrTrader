# MrTrader — External Alpha Review Brief

*A self-contained briefing for an external reviewer. You have no access to our code or repository — everything you need is in this document.*

---

## 0. How to read this document

You are being asked to act as a **world-class quantitative researcher at a top hedge fund** with deep experience building and running fully automated equity trading platforms. We have built a small automated paper-trading system, validated it adversarially, and found that almost everything we tried does **not** work. We want your honest, skeptical, specific critique — and your best ideas for where real, durable alpha actually lives given our constraints.

We are not looking for encouragement. We are looking for hard truths and concrete direction. Brutal is welcome.

**Acronyms** are expanded on first use. **All performance numbers are real** and pulled from our honest validation runs (after we fixed a large number of measurement bugs — see Section 4.3, which is essential context for why our numbers are low and why we now believe them).

---

## 1. Executive summary & the ask

**What MrTrader is.** A fully automated, multi-agent **paper-trading** platform. Capital: **$100,000 of simulated (paper) money** via the Alpaca paper-trading broker. Universe: **US equities, roughly the Russell 1000 (large-cap) names** (~1000 liquid large-caps). It is long-biased (shorting is possible but capital-constrained and, in our tests, an anti-edge — see Section 5). It was built to **discover and trade durable alpha**, with a validation pipeline designed to be paranoid about overfitting and look-ahead bias.

**What is actually live.** Exactly **one** validated edge: **PEAD** (Post-Earnings-Announcement Drift, defined in Section 5). It cleared our forward-validation ("paper") promotion tier and is now **switched on and trading in paper** (it scans the universe each premarket, generates signals, and routes them through the risk/execution chain; on a representative day it generated 2 signals that were correctly filtered out by spread/price-run risk gates, i.e. it trades only when a qualifying name passes every gate). Everything else we tried is dead or unvalidated.

**The honest headline:** after correcting roughly six inflated/in-sample performance numbers and seven latent simulation bugs, our two flagship machine-learning (ML) strategies (a long-only cross-sectional swing ranker, and an intraday 5-minute meta-model) both have **no out-of-sample edge**. The only thing that survived honest testing is a modest, statistically fragile, well-documented academic anomaly (PEAD), at roughly half the Sharpe of what the literature suggests is achievable.

### The ask to you (the reviewer)

**(a) Critique** the methodology, architecture, and data. Be skeptical and specific. Where are we fooling ourselves? Where is our validation too loose — or too strict?

**(b) Find alpha we're missing.** Tell us where, given our universe and budget, the highest-expected-value alpha actually is.

**(c) You are explicitly invited to propose a complete redesign.** Name missing datasets we should acquire (free or paid), and recommend model classes / strategy types worth trying.

**(d) Triage our dead approaches.** Tell us what is genuinely exhausted versus what we abandoned prematurely and should revisit.

We want your unvarnished professional judgment.

---

## 2. System architecture (runtime)

The live system is **three asynchronous agents** plus supporting machinery. Agents communicate over a **Redis-backed message queue** (no HTTP between them).

```
[Portfolio Manager] → proposals → [Risk Manager] → approved → [Trader] → [Alpaca paper broker]
        ↑                                                          |
        └──────────────── re-evaluation requests ──────────────────┘
```

### 2.1 The three agents

- **Portfolio Manager (PM).** Generates the day's **target positions** from the active strategy/model. Premarket it pulls an intelligence pass (earnings calendar, regulatory 8-K filings, news sentiment), then runs the active selector (currently the PEAD selector when enabled; otherwise a swing ML ranker or an intraday model) to produce ranked buy/sell proposals. It also re-scores open positions intraday and can request exits/extensions.

- **Risk Manager (RM).** The **sole gatekeeper** for execution. Every proposal runs a multi-rule risk chain (Section 2.3). Approved proposals go to the Trader; rejected ones are logged with reasoning and dropped. The RM also enforces a **kill switch** (rejects all new proposals when active) and budget caps.

- **Trader.** Executes approved proposals on the Alpaca paper broker and monitors the position lifecycle (stops, max-hold, exits). Handles order routing (Section 2.4).

### 2.2 Supporting machinery

- **Regime classifier.** Labels each day BULL / BEAR / NEUTRAL using VIX (the CBOE volatility index) level relative to expanding-window percentile thresholds, combined with SPY (S&P 500 ETF) position relative to its 50-day and 200-day moving averages. Used both as a live sizing overlay and in backtest stratification. (VIX = market-implied 30-day S&P 500 volatility; a fear gauge.)
- **Kill switch + circuit breaker.** Hard stops on new trading under stress / operator command.
- **Scheduler.** Runs premarket analysis, selection at fixed ET times, intraday position reviews every 30 minutes, and end-of-day jobs.

### 2.3 The Risk Manager rule chain (real limits)

Every proposal must pass **all** of these (defaults shown; some are DB-configurable):

| # | Rule | Limit |
|---|---|---|
| 1 | Buying power | Must have cash; shorts require 150% of notional (Reg-T margin) |
| 2 | Max single position size | **5%** of account value |
| 3 | Sector concentration | **20%** of account value per sector (net, direction-aware) |
| 3b | Correlation risk | Reject if 30-day return correlation with an existing ≥5%-of-book position **> 0.75** (sign-adjusted: a short vs a correlated long is treated as a hedge) |
| 4 | Daily loss limit | Block new trades if today's realized loss ≥ **2%** of account |
| 5 | Account drawdown | Block new trades if drawdown from peak ≥ **5%** |
| 6 | Open position count | Hard cap **5** simultaneous positions (intraday sub-cap: 3) |
| 7 | Portfolio heat | Total risk across positions ≤ **6%** |
| 8 | Dynamic stop-loss | Volatility-adjusted (ATR-scaled) stop; base 2% at "normal" volatility |
| 9 | Net-exposure gate (L/S mode) | Keep net long exposure near a target band (default 40% ± 15% of NAV) |
| 10 | Short-notional cap | Total short notional ≤ **75%** of NAV |

Additional **budget caps**: gross exposure never exceeds **80%** of the account; the swing book is allotted ~**70%** and intraday ~**30%** of capital (when both are active).

### 2.4 Execution / costs

- **Order routing.** Standard swing entries use a below-ask limit; PEAD entries (when live) use a **marketable limit** that crosses the spread (long = ask + 10 bps, short = bid − 10 bps) so live fills track the backtest's next-open assumption and avoid the adverse-selection trap of below-ask limits (which fill only the names that *don't* run).
- **Backtest cost assumptions.** Large-cap (Russell 1000) backtests assume **5 basis points (bps)** per side transaction cost. The small/mid-cap experiment (Section 5) used **20 bps** plus a delisting haircut. (1 bp = 0.01%.)
- **Position lifecycle.** Entries fill at next-day open (simulated market-on-open). Exits via ATR-based stop/target, plus a max-hold (PEAD ≈ 40 trading days; intraday ≈ 24 five-minute bars).

### 2.5 Honest caveat: the live agents add risk overlays the backtest does not have

The live PEAD path deliberately keeps the live risk overlays **on top of** the validated config: regime-based sizing multipliers, news-sentiment sizing, an opportunity-score gate, a macro-calendar block, the full 10-rule RM chain, and the live position-sizing logic (not a clean 5% equal weight). **This causes real tracking error versus the clean backtest.** We instrument it (per-day tracking artifact, weekly realized-Sharpe-vs-expectation email) rather than pretend it doesn't exist — but you should assume the live system's realized performance can diverge meaningfully from the backtested edge.

---

## 3. Data inventory

### 3.1 What we have

| Source | What | Frequency | History | Point-in-time (PIT)? | Survivorship-safe? |
|---|---|---|---|---|---|
| yfinance (Yahoo) | Daily OHLCV bars | Daily | ~Multi-year | Adjusted closes; OK for daily | **No** — silently drops delisted tickers |
| Polygon (cache + S3 flat files) | 5-minute intraday bars | 5-min | ~2 years (cache) | OK | Partial (cache completeness dependent) |
| **Polygon grouped-daily flat files** | All-tickers-per-day daily panel (every ticker that printed a bar) | Daily | Multi-year | OK | **Yes** — delisted-inclusive; gold-standard survivorship-safe candidate source. Validated: ~10,800 symbols on a sample day including names that later delisted (SIVB, FRC, etc.) |
| Alpaca | Daily bars (paper broker) | Daily | Capped ~100 recent bars on our plan | OK for recent | n/a |
| FMP (Financial Modeling Prep) | Quarterly fundamentals, EPS surprises, analyst upgrade/downgrade history, institutional (13F) holdings changes, insider (Form 4) transactions | Event/quarterly | Earnings ~back to 2016; fundamentals ~10y | **Yes** — we act on `filingDate` (public date), never `transactionDate` | Mostly |
| EDGAR | Annual fundamentals (fallback) | Annual | Long | Yes | Mostly |
| Macro series | VIX, VIX3M (3-month VIX), HYG (high-yield credit ETF), IEF (Treasury ETF), RSP (equal-weight S&P), SPY | Daily | Multi-year | Yes | Yes (indices/ETFs) |
| Sector ETFs | 11 SPDR sector ETFs | Daily | Multi-year | Yes | Yes |

**Universe membership.** We maintain point-in-time Russell 1000 membership so backtests only include names that were in the index *as of* the relevant date (no look-ahead on index adds). Historical/delisted symbols from our DB are also included to reduce survivorship bias on the large-cap side.

**Earnings-surprise detail (the PEAD input):** FMP gives reported EPS vs consensus estimate per quarter, with report dates; we compute surprise as `(actual − estimate) / |estimate|`. PIT-enforced: for a window ending date D, only reports with report date ≤ D are visible.

### 3.2 What is MISSING (not on our current free / low-cost plan)

- **Options data** — implied volatility (IV), skew, term structure, IV-crush around earnings. None.
- **Short interest / borrow availability / borrow cost.** None.
- **Intraday order-book / microstructure / Level 2 / trade-and-quote (TAQ).** None.
- **Reliable PIT corporate-action feed** — buyback announcements, M&A, index add/delete dates with reliable public-knowledge timestamps. We could **not** source a trustworthy PIT buyback-announcement feed on our plan (so we refused to build on it).
- **Alternative data** — credit-card / transaction panels, web traffic, satellite, app downloads, supply-chain, sentiment beyond basic news.
- **Fundamental point-in-time vendor data** (e.g. Compustat PIT) — we approximate PIT with FMP/EDGAR filing dates, which is decent but not institutional-grade.

**We invite you to prioritize:** of the missing datasets, which one or two would most change our expected alpha per dollar of cost?

---

## 4. Model training & the validation pipeline

This is where our credibility lives — and where we most want your scrutiny.

### 4.1 Model types tried

- **Swing (multi-day) cross-sectional ranking model.** Gradient-boosted trees (XGBoost), including a **LambdaRank** (learning-to-rank) objective, trained to rank Russell 1000 names cross-sectionally on each day. Feature families: price/technical (momentum at multiple horizons, 52-week position, volatility regime, range expansion, volume trend, a few WorldQuant-style alphas), fundamental/quality (profit margin, operating margin, gross margin, P/E, revenue growth — PIT via FMP), and a couple of sector-neutralized momentum features. A feature-IC audit reduced ~69 raw features to ~14–17 with stable positive information coefficient (IC). Label: top-quintile cross-sectional forward return over a 20-trading-day horizon.
- **Intraday 5-minute meta-model.** XGBoost ensemble on 5-minute bars + per-symbol daily features (52-week position, vol percentile) + macro/regime overlays. Same-day labels (forward window within the entry day). Reduced liquidity universe for feasibility.
- **Regime classifier.** Rules + VIX/SPY thresholds → BULL/BEAR/NEUTRAL.
- **PEAD event scorer.** Rules-based (no learned model): long on EPS surprise > +5%, enter within ~3 calendar days of the report, hold ~40 trading days, with a VIX-based crisis block (Section 5).

### 4.2 The validation methodology

A reviewer will judge us primarily on this, so it is described in full.

- **Combinatorial Purged Cross-Validation (CPCV).** Instead of a single train/test split, we split the timeline into *k* blocks and evaluate every combination of held-out test blocks — giving multiple "paths" through history. Following López de Prado:
  - **Purge:** training observations whose label horizon overlaps the test window are removed. Swing purge = **85 calendar days** (= 60-day max feature lookback + 20-day label horizon + buffer; enforced by an import-time invariant). Intraday purge = **2 trading days**.
  - **Embargo:** a gap after each test block so post-test leakage can't feed the next training block.
- **Per-fold retraining (true out-of-sample).** Each fold trains a **fresh model on only its own causal past** `[train_start, train_end]`, sets the model's `trained_through = train_end`, then evaluates on `[test_start, test_end]` strictly after `train_end + purge`. This is the keystone fix (see 4.3) — earlier "CPCV" scored a single **frozen** model whose training window overlapped the test folds, which is in-sample by construction.
- **OOS guard.** A hard assertion at multiple layers: `test_start > trained_through + purge`. A model with no recorded training cutoff is *refused* (it cannot be certified out-of-sample). Violations raise rather than silently pass.
- **Sacred holdout.** A permanent date boundary (currently **2026-11-09**); all data on/after it is reserved for a single, final, one-shot promotion-candidate evaluation. Touching it during development raises an error. This prevents iterative tuning from leaking into the final test.
- **Deflated Sharpe Ratio (DSR).** Bailey & López de Prado's multiple-testing correction: it deflates an observed Sharpe for the number of strategy variants tried (we track this as **N_TRIALS_TESTED = 300**, deliberately conservative — higher N makes DSR harder to pass) and for the number of observations. **Known weakness:** DSR p-value *saturates* to ~1.0 for any Sharpe above ~2 at our observation counts, so it provides essentially zero discrimination in that regime — we flag it as "saturated / non-binding" when that happens.
- **Path t-statistic with N_eff = number of folds (not number of paths).** The C(k, p) CPCV paths reuse the same *k* folds and are strongly correlated (each trading day appears in multiple paths). Treating the ~15–28 paths as independent would overstate significance by roughly √(paths/folds). We therefore compute the path-Sharpe t-stat as `mean / (std / √N_eff)` with **N_eff = n_folds** (typically 6–8). This is the single most important honesty fix in our significance math.

### 4.3 The measurement-integrity history (read this — it explains why our numbers are low)

We ran a **13+ round adversarial code audit** plus a deep bug-fix campaign. It converted roughly **six inflated / in-sample numbers into honest ones** and fixed roughly **seven latent simulation bugs**. The headline findings:

- **The "+5.14 Sharpe" intraday CPCV result was in-sample memorization, not edge.** It scored a single frozen model whose 730-day training window fully contained all the CPCV test folds. Per-fold Sharpes of 2.05 / 9.64 / 8.16 are the signature of memorization. **Struck from the record.** The swing model had the identical disease.
- **A structural insight:** frozen-model CPCV (one pre-trained model scored across folds) is *structurally incapable* of being out-of-sample when the model is trained on the full window. The only honest fix is true per-fold retraining (now implemented, Section 4.2).
- **Other fixed bugs:** delisting P&L booked as zero (blowups counted as break-even); survivorship leakage in the small-cap universe; a regime-label-vs-score type crash that silently dropped every training window; a profit-factor sentinel that inflated the gate; an embargo that was logged but never applied to fold boundaries; Sharpe annualized on calendar instead of trading days; a no-drawdown Calmar sentinel that trivially passed the gate.

**Bottom line:** early "successes" were measurement artifacts. We do not trust any number produced before this campaign. We *do* trust the current (low) numbers precisely because the pipeline that produced them is now adversarially hardened. **Please stress-test this claim** — if our validation still has holes, that is the most valuable thing you can tell us.

### 4.4 The promotion gate (significance-first, two-tier)

We recently **replaced** a primary gate of "mean Sharpe ≥ 0.80" (swing) / "≥ 1.00" (intraday). That threshold was a relic: it had been calibrated against the now-struck inflated numbers, and a bare mean-Sharpe bar cannot distinguish a `+0.22 / t=0.17` noise result from a `+0.546 / t=2.26` genuine-signal result (both are below 0.80). The new gate is **significance-first**, with mean Sharpe demoted to an economic-materiality floor:

| Tier | Purpose | Criteria (all must hold) |
|---|---|---|
| **PAPER** | Forward-validate; **no capital** | Path t-stat ≥ **2.0**, % of paths positive ≥ **0.75**, 5th-percentile path Sharpe (P5) ≥ **0.0**, mean Sharpe ≥ **0.35**, plus profit-factor / Calmar / regime backstops |
| **CAPITAL** | Real money | All PAPER criteria **plus** mean Sharpe ≥ **0.50**, n_folds ≥ **10** (power floor), and (t-stat ≥ **2.5** OR a documented live-paper confirmation) |

- A standard single-split walk-forward (no path distribution → no t-stat) **cannot** be promoted under this gate; CPCV is required.
- **Event-sparsity regime waiver.** PEAD is event-driven and produces too few same-regime trading days to compute a per-regime Sharpe, so the regime backstop is legitimately undefined. For the **PAPER** tier only (zero capital), and only when this is genuine event-sparsity (not a data bug), the regime backstop is **waived but flagged for mandatory human review**. The CAPITAL tier never auto-waives — it requires real regime data or an explicit documented sign-off. This narrow, flagged waiver is the *only* way PEAD reaches PAPER.

Re-scoring every result on record through this gate promotes **only PEAD → PAPER PASS (flagged) / CAPITAL HOLD**. Everything else fails all tiers. **We invite critique of these thresholds** — are they sound, too strict, too lax?

---

## 5. The honest scoreboard — everything we tried

All numbers are from honest, post-audit CPCV runs unless marked. "t" = path t-stat with N_eff = n_folds. "%pos" = fraction of paths positive. "P5" = 5th-percentile path Sharpe. "PF" = profit factor.

| Strategy | Mean Sharpe | t-stat | %pos | P5 | PF / Calmar | Verdict |
|---|---|---|---|---|---|---|
| **Swing long-only cross-sectional ranker** (true per-fold OOS) | **+0.22** | **0.17** | 50% | −3.97 | 1.26 / 6.5† | **DEAD — noise** |
| **Intraday 5-min meta-model** (true per-fold OOS) | **−2.80** | **−6.85** | 0% | — | 0.94 / — | **DEAD — cost-drag** |
| Intraday (frozen-model, struck) | +5.14 | — | — | — | — | **In-sample memorization — struck** |
| **PEAD long-only** (R1K, validated) | **+0.546** | **+2.26** | **95%** | **+0.009** | 1.54 / 0.77 | **REAL — live-wired, paper-gate-ready** |
| Small/mid-cap PEAD (survivorship-safe, 20 bps) | +0.361 | +0.95 | 76% | −1.368 | 1.78 / 1.45 | **REJECTED — weaker venue** |
| QualityShort (short deteriorating fundamentals) | **−0.903** | −3.19 | 24% | −1.99 | 0.63 / −0.70 | **Anti-edge** |
| Insider-cluster buying | +0.228 | +0.88 | 76% | −0.95 | 1.38 / 0.83 | **Weak — fails edge bar** |
| Buyback announcements | — | — | — | — | — | **Untestable (no PIT feed)** |

† The swing PF/Calmar "OK" values are short-window aggregation artifacts (62-day test windows from purge=embargo=85), not evidence of edge — ignore them. The verdict is set by mean +0.22 / t=0.17.

### 5.1 Swing long-only cross-sectional ranking — DEAD

Nine distinct labeled experiments ("LX1"–"LX9") plus the production architecture all failed. The honest equal-weight baseline was +0.079; a beta-neutralized variant was +0.031; the production LambdaRank's first *true* OOS CPCV was **+0.22, t=0.17 (statistically zero)**. ML weighting was *worse* than equal weight (an 82-feature XGBoost variant came in below equal-weight; a 5-feature one was −2.34).

**The structural failure ("F2").** Cross-sectional ranking is regime-neutral *only if you trade the long-short spread*. Long-only execution is regime-exposed by construction: the long book always carries net equity beta into bear / VIX-spike folds (notably an August-2024 VIX spike), where "top-quintile momentum" flips meaning from "fastest riser" to "fell least." A beta-residualized + diversified variant *still* posted ≈ −0.70 in that fold. Our read: long-only price-feature ML in this universe is **structurally exhausted**, not a tuning miss. (Note: every audited oddity in the swing run — e.g. a fold-skip biased toward friendly 2023–24 bull windows — biases the +0.22 *upward*, so a cleaner run would be lower.)

### 5.2 Intraday 5-minute meta-model — DEAD (cost-drag)

Honest per-fold OOS: **−2.80, t=−6.85, 0% positive, gross PF 0.94** (below break-even *before* even getting to net). The gross edge sits below the transaction-cost hurdle. The earlier "+5.14" was the frozen-model memorization artifact described in 4.3.

### 5.3 PEAD — the one real edge

**PEAD = Post-Earnings-Announcement Drift:** the long-documented tendency for a stock's price to *continue drifting* in the direction of an earnings surprise for weeks after the announcement, because the market under-reacts to the news (Ball & Brown 1968; Bernard & Thomas 1989).

**Our validated config:** long-only; enter on EPS surprise ≥ +5% within ~3 calendar days of the report; **hold ~40 trading days**; a hard **VIX > 30 crisis block** (no new entries — this block is credited with the entire edge: it lifts the P5 tail from −0.29 to +0.01); priced-in filter **off** (we deliberately keep large announcement-day-gap names, because drift is strongest there in our data). Result: **mean +0.546, t = 2.26, 95% positive, P5 +0.009, PF 1.54, Calmar 0.77.** This clears our PAPER tier (with the flagged regime waiver) and is **now live and trading in paper**.

**Four enhancement levers, all of which FAILED to beat the +0.546 baseline:**

| Lever | Result | Why it failed |
|---|---|---|
| Shorter hold (15d) | +0.411, t=1.19 | Cuts the drift short — drift wants the full ~8-week hold |
| Long-short (add short leg on negative surprise) | +0.456, t=2.61 | Lower mean but tighter distribution (a more *robust*, lower-return variant — interesting for deployment, not for the mean gate) |
| Earnings-quality filter (require positive analyst revisions) | +0.449, t=1.02 | Higher per-trade quality but far fewer trades → statistical power collapses |
| Conviction sizing (tilt by standardized surprise / vol) | +0.515, t=2.03 | Concentrated downside in bad folds without lifting the median |

**Conclusion:** PEAD's honest ceiling on this universe is ~0.55 — consistent with the academic ~0.5–0.7 range for vanilla large-cap PEAD after costs. Config tuning cannot move it.

### 5.4 QualityShort — anti-edge

Shorting fundamentally deteriorating names (negative margins, falling revenue, high leverage): **−0.903, t=−3.19** (significantly *negative*). A pre-honest-pipeline run had shown +5.95 — another inflated number deflated. Mechanism: "broken" names are beaten-down high-beta value stocks that **rally hardest on any risk-on bounce**, so systematically shorting them bleeds.

### 5.5 Insider-cluster buying — weak

Long on clusters of open-market insider purchases (Form 4): **+0.228, t=0.88, 76% positive.** Statistically indistinguishable from zero. Structural reason: large-cap insiders rarely make open-market purchases, so clusters are rare in the Russell 1000 (only ~24–35 trades per ~9-month fold, 52% of fold evaluations empty). The documented insider edge concentrates in small/mid-caps.

### 5.6 Buyback announcements — untestable

We could not source a trustworthy PIT buyback-announcement feed on our data plan, and refused to build on a non-PIT feed (it would re-introduce exactly the look-ahead we spent the audit eliminating). Clean stop.

### 5.7 Small/mid-cap PEAD — REJECTED

The literature says event-drift is *stronger* in small/mid-caps (slower information diffusion, less analyst coverage, less arbitrage). We built a properly **survivorship-safe** harness on Polygon grouped-daily flat files (delisted-inclusive — 8,755 delisted/halted names retained), a PIT liquidity-band universe (trailing-20-day average dollar volume in $2M–$50M), realistic **20 bps** costs, and a delisting haircut. Result: **+0.361, t=0.95, P5 −1.368** — *weaker* than large-cap PEAD and statistically a coin flip. After an adversarial oddity review (ruling out a symbology bug and confirming the CPCV math), the verdict was a **real failure, not under-testing**. **A meaningful negative result:** the apparent small-cap event premium in the literature may partly be survivorship/illiquidity bias that an honest harness strips out.

### 5.8 Meta-conclusion

- **Long-only price-feature ML appears exhausted** in this universe (convergent across swing and intraday horizons).
- **The only durable edge found is event-driven** (PEAD), and it is modest and statistically fragile.
- **Free-data event diversification is exhausted** (insider weak, buyback no-data, small-cap weaker).

---

## 6. Constraints, known limitations, and current hypotheses

### Constraints

- **Data:** free / low-cost only (see Section 3.2 for the gaps).
- **Universe:** ~Russell 1000 large-caps. Small/mid-cap expansion was tested and rejected.
- **Direction:** long-biased. Shorting needs margin and has been an anti-edge in every test so far.
- **Capital:** ~$100k paper notional. Position cap 5%, max 5 open positions, gross ≤ 80%.
- **Statistical power:** effective N ≈ 6–8 folds. This is *low* — it is hard to clear a t ≥ 2.5 capital bar with so few effective observations.
- **DSR saturation:** DSR provides no discrimination above Sharpe ~2; we rely on the path t-stat / %pos / P5 instead.
- **PEAD ceiling:** ~0.5–0.7 academic ceiling for vanilla large-cap PEAD; we sit at the low end (~0.55).

### Open ideas considered but NOT validated

- True **dollar-neutral long/short** (net beta ≈ 0) with a *purpose-built* short signal (never an inverted long composite — that failure mode is well-established in our tests).
- **Options-based PEAD** (IV-crush / earnings volatility / short verticals) — highest theoretical ceiling, but we have **no options data or options backtest infrastructure**.
- **Mean reversion** (untested honestly).
- **Analyst-revision momentum** as a standalone signal (currently only a PEAD sub-filter).
- **Sector rotation** (11 SPDR ETFs, top-3 by 6-month momentum) — cheap benchmark floor, not yet run honestly.
- **Portfolio-level regime switch** to cash/Treasuries as an overlay (not standalone alpha).
- **Paid alternative data.**

---

## 7. Explicit questions for the reviewer

1. **Is our validation sound, or are we still fooling ourselves?** Specifically: CPCV with purge/embargo, true per-fold retraining, N_eff = n_folds for the t-stat, DSR (acknowledging saturation), and the sacred holdout. Where are the remaining holes? Is N_eff = n_folds the right correction, or too conservative / too loose?
2. **Is the significance-first two-tier gate (t ≥ 2.0 / %pos ≥ 0.75 / P5 ≥ 0.0 / mean ≥ 0.35 for paper; t ≥ 2.5 / n_folds ≥ 10 / mean ≥ 0.50 for capital) calibrated correctly?** With effective N ≈ 6–8, is a t ≥ 2.5 capital bar even reachable for a real-but-modest edge? Are we rejecting good edges or admitting noise?
3. **Is long-only cross-sectional ML genuinely exhausted, or did we miss a feature class, horizon, or sub-universe?** Is the "F2" long-only-beta diagnosis correct, and is the only fix to go dollar-neutral?
4. **Where is the highest-expected-value alpha for a small automated shop with our universe and data — and what is the single most valuable dataset to acquire** (free or paid)? Rank options data, short interest, PIT corporate actions, alternative data, and PIT fundamentals by expected alpha-per-dollar.
5. **Should we abandon ML ranking entirely and go all-in on event/structural edges** (PEAD-family, options overlays, dollar-neutral L/S)? Or is there a salvageable ML angle we're not seeing?
6. **Is the multi-agent live-overlay architecture helping or hurting?** The live risk overlays cause deliberate tracking error versus the backtest. For a system whose only edge is a fragile +0.55-Sharpe anomaly, are these overlays prudent risk control or are they quietly destroying the edge?
7. **Of our dead approaches, which deserve a second look** versus truly exhausted? (Intraday cost-drag, QualityShort anti-edge, insider clusters, small/mid-cap PEAD, the swing ranker.)
8. **Would you redesign the whole approach — and if so, how?** A clean-sheet recommendation for what a disciplined small shop should actually build with this stack is explicitly welcome.
9. **What would *you* trade with this stack** (~$100k, large-cap US equities, free data + a modest budget, automated execution, the validation pipeline above)? Concrete strategies, not categories.

---

*Provenance: generated 2026-06-02 from the system's internal living documentation (pipeline architecture, model status, experiment log, decision log, and source configuration). All performance figures are real and post-audit. This document is self-contained and intended for external review.*
