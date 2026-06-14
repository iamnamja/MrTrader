# MrTrader — State of the Research (Brutally Honest Snapshot)

**For external world-class quant reviewers. Written cold, no hype. As of 2026-06-14.**

This document is self-contained. Every number is pulled from the live repo (parquet row counts, the research registry DB, the experiment log). Where a result died, the killing statistic is named. Read Section 6 last; read the one-line takeaway at the very bottom first if you only have 30 seconds.

---

## 1. TL;DR (5 lines)

1. **MrTrader** is a single-developer systematic trading research system (US equities/ETFs, paper-traded on Alpaca), with an honest CPCV/walk-forward validation harness as its centerpiece.
2. **The one live edge is time-series momentum (TSMOM): a 10-ETF, long-flat, inverse-vol, weekly-rebalanced trend sleeve, standalone Sharpe +0.71 over 19 years.** It runs at 25% of book + cash. Nothing else is live.
3. **The brutal truth: everything else we have tried has died** — cross-sectional ML ranking (swing + intraday), PEAD post-earnings drift (was live, now demoted), options-as-signal (5 hypotheses, all significantly negative), index short-vol/VRP (real premium, Sharpe-negative under stress), and trend-broadening (weaker than the simple sleeve). All killed or parked on the honest harness.
4. **The hard-won lesson: with only ~8 CPCV folds over ≤4 years of options data, a t≥2 gate is a Type-II machine** — it rejects genuine Sharpe-0.4–0.7 edges. We rebuilt the gate (Ruler v2) to fix this, then re-ran every candidate through it — and *still* nothing new passed. So the failures are real, not measurement artifacts.
5. **The question for you: given free-data-only constraints (our options feed is being cancelled 2026-06-17), an ~8-fold CPCV power ceiling, and a US-centric universe — where does testable, undiscovered alpha plausibly still live?**

---

## 2. Data we actually have

Pulled from `data/` (row counts and date spans are real, read off the parquet/DB files today).

| Dataset | What it is | Coverage | Rich / thin / gone | Free / paid |
|---|---|---|---|---|
| `price_cache/*.parquet` (679 files) | Daily equity/ETF OHLCV (yfinance) | **2005-01 → 2026-05**, ~5,371 days/symbol (e.g. SPY) | **RICH** (19y deep) | FREE (yfinance) |
| TSMOM ETF universe | SPY,QQQ,IWM,EFA,EEM,TLT,IEF,GLD,DBC,UUP daily | 2005→2026 (subset of above) | **RICH** — the live edge runs on this | FREE |
| `macro_history.parquet` | spy/vix/vix3m/hyg/ief/rsp daily | 2018-01 → 2026-05, 2,102 rows | THIN window (8y) but clean | FREE (FRED/yfinance) |
| `event_panel.parquet` | R1K earnings-event panel: SUE, gaps, drift, PEAD score, VIX/regime, options-pre-event | **2019-01 → 2026-06**, 21,330 events / **9,774 PEAD-qualified**, 787 symbols, 45 cols | RICH for event studies | FREE (FMP $29 earnings + yfinance) |
| `short_interest.parquet` | FINRA bi-monthly SI + days-to-cover (PIT `knowable_date`) | 2017-12 → 2026-05, 140,286 rows | RICH-ish (semi-monthly) | PAID (Polygon, ending) |
| `short_volume.parquet` | Daily FINRA short-volume ratio (PIT) | **2024-02 → 2026-06** only, 403,103 rows | THIN (2y) | PAID (Polygon, ending) |
| `intraday/*.parquet` (720 files) | 5-min equity bars | **2024-04 → 2026-04** only (~2y), ~38k bars/symbol | THIN (2y) — too short for CPCV | FREE (yfinance) + Polygon cache |
| `smallmid/panel.parquet` | Survivorship-safe small/mid daily (delisted retained) | 14.5M rows, eligibility table 454k rows | RICH but the PEAD venue it served was rejected | PAID (Polygon flat files, ending) |
| **`options_bars.parquet`** | Historical options OHLCV (Polygon S3 flat files) | **2022-06-09 → 2026-06-11**, **113.3M bars**, 733 underlyings | RICH in volume, **GOING FROZEN** | PAID — **sub ends 2026-06-17** |
| `options_contracts.parquet` | Contract reference incl. expired (survivorship-safe) | 2022-06 → 2026-06, **6.20M contracts** | RICH | PAID (ending) |
| `options_greeks/` (733 underlyings) | **COMPUTED** IV/delta/gamma/vega/theta (BS/CRR; Polygon serves none) | matches options_bars (113.3M bars, 5.2 GB) | RICH but **derived, not vendor** | computed from PAID bars |
| `options_features.parquet` | Daily PIT options-signal features: CPIV, 25Δ skew, term slope, IV/RV, O-S volume | 2022-06 → 2026-06, **583,701 rows** | RICH (~4y) but the signals it serves are DEAD | derived |
| `options_spread_obs.parquet` | Live NBBO spread snapshots (Alpaca free `indicative` feed) | **2026-06-11 → 2026-06-12 only** (just lit) | **THIN — basically empty** (2 days) | FREE (Alpaca) |
| `research_registry.db` | Pre-registration / N_TRIALS ledger | 11 registered hypotheses | the audit trail | — |
| fundamentals/ (FMP, AlphaVantage) | Quarterly financials, analyst grades, insider | sparse | THIN | PAID $29 (FMP) + free fallback |

**What is GONE / going:** the Polygon Options Developer plan ($79/mo) is **cancelled effective 2026-06-17**. After that we own a **frozen** 4-year options store (no new bars), and Polygon never served historical IV/greeks/OI/NBBO anyway — we computed those ourselves. The live NBBO logger (Alpaca) has **2 days** of data. So forward options work is effectively limited to a frozen 2022–2026 window.

**What is free and deep:** yfinance daily ETF/equity bars back to **2005** (19 years, the substrate of the only live edge) and FRED macro. These cost nothing and are the only genuinely long-history assets we have.

---

## 3. What we've tried & how it died

Every strategy line the project has run, with the killing number. Verdicts cross-checked against `ML_EXPERIMENT_LOG.md`, `DECISIONS.md`, `MODEL_STATUS.md`, and the `research_registry.db` decision column.

| Strategy | Verdict | The killing number | Why it died |
|---|---|---|---|
| **Swing XS-ML ranker** (long-only cross-sectional, LambdaRank, price/technical features) | ❌ DEAD (frozen) | True per-fold CPCV **mean +0.22, t=0.17**, 50% folds positive, DSR p=0.30 | First *honest* OOS = noise. Long-only beta gets destroyed in VIX spikes (the "F2" structural loss). 9 LX variants + beta-neutral all failed. Exhausted. |
| **Intraday 5-min meta-model** | ❌ DEAD (frozen) | True per-fold CPCV **mean −2.80, t=−6.85**, 0% positive, gross PF 0.94 | Gross edge is below the cost hurdle. The earlier +5.14 was **in-sample memorization** (training window contained all test folds) — struck from record. |
| **Dollar-neutral L/S ranker** (the capital hope after long-only died) | ❌ DEAD | CPCV **Sharpe +0.14, t=0.18**, on a genuinely net-flat book (net$ −0.01) | The long-only +0.22 was pure market beta. Removing beta removes the "edge." Cross-sectional ML ranking declared exhausted. |
| **PEAD** (post-earnings announcement drift, long-only, VIX/trend crisis block) | 🔻 **DEMOTED** (was live for ~2 weeks) | **Event-level** re-adjudication: 10d SPY-hedged mean **−8.3bp, t=−0.77**, one-sided p=0.78, on 9,774-qualified R1K events, two-way (date×firm) cluster-robust | The CPCV +0.546/t=2.26 that got it to paper was an **8-fold path-t illusion** (≈ a noise null's t). At the event level, market-hedged, it makes no money. Corroborates the earlier CAPM hedged Sharpe of −0.37. |
| Small/mid-cap PEAD | ❌ REJECTED | mean +0.361, **t=0.95** (N_eff=8), P5 −1.368 | Weaker than R1K, survivorship-safe with real costs. The "event edges are bigger in small caps" folklore did not survive honest modeling. |
| Analyst up/downgrade drift (2nd event edge) | ❌ NULL | Best-in-campaign +0.894/t=2.85 was a **52% fold-skip artifact**; neutralized +0.342/t=1.24, CAPM α t=0.20 | Survivorship/fold-selection inflation. Noise once cleaned. |
| Short-interest factor (Boehmer/Asquith) | ❌ ANTI-EDGE | dollar-neutral **−1.213, t=−3.53** | The classic SI anomaly *reversed* in the meme-stock era. |
| **OPT-3** single-name earnings IV-crush (options-as-execution) | ❌ KILL | cost-killed (spread > crush) | Premium real, transaction costs eat it. |
| **OPT-4** index short-vol / VRP | ❌ KILL | VRP real (PF 2.24/1.75) but Sharpe-weak; **2026-06-14 spread-stress CPCV re-run: mean Sharpe −0.207 (1×) / −0.426 (2×)**, negative 2022–2026 | A risk premium, not alpha; negative under realistic spread stress. Can't clear a standalone SR>0.20 backstop. |
| **OPT-5** implied-move filter for PEAD | ⏸️ PARKED (fragile) | Lift exists only at ratio=1.0, *inverts* at 1.25 | Threshold-fragile / overfit-suspect. Continuous re-test (H2) → t=−1.21, not confirmed. |
| **H4a–H4e** options-as-equity-signal (CPIV, 25Δ skew, put O/S, term slope, IV/RV) | ❌ **ALL 5 KILL** | Decile L/S net-of-cost t-stats: CPIV **−2.70**, skew **−4.10**, put-O/S **−4.43**, term-slope **−2.83**, IV/RV −0.12 | Academic signs do **not** hold 2022–26 at equity cost (significantly *negative*); the inverse isn't tradeable either (post-hoc sign-mining, growth-crash regime). Options confirmed as a data asset, not an equity signal. |
| **P5** trend-broadening (more assets/sleeves on the trend base) | ⏸️ PARKED | Broadened Sharpe 0.30/t=1.31 vs live 0.72/t=3.18; **Ruler-v2 re-score: OOS SR 0.123, HAC p 0.30** | Genuinely weaker than the simple 10-ETF sleeve. Failed on *significance*, with the regime backstop waived — not a Type-II. |

**Registry ledger** (`data/research_registry.db`, the true N_TRIALS): 11 pre-registered hypotheses. Decisions: 6× `kill` (all 5 H4 options + ...), 4× `park` (TSMOM-vs-PEAD Track-B ×2, H2 implied-move, P5 trend-broaden), H1-PEAD and H3 recorded via the event-inference path (PEAD demoted; H3 BLOCKED on revision data we don't have).

**Net: the only surviving edge in the entire program is TSMOM trend.**

---

## 4. What's live + the harness/gate (the bar a new idea must clear)

### The live book
- **Trend-only: 10-ETF TSMOM, +0.71 standalone Sharpe over 19y, at 25% of book + cash.** Universe: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP. Long-flat, inverse-vol weighting, weekly Monday rebalance, direct Alpaca paper placement.
- **Swing + intraday retrain: DISABLED** (frozen as non-production benchmarks; both proven null).
- **PEAD: flipped OFF** (demoted at event level).
- **Models on disk:** `swing_v229`, `intraday_meta_v65` (both dormant/dead), `regime_model_v6` (the BULL/BEAR/NEUTRAL regime classifier, still active for context/sizing, separate AUC gate).

### Implemented strategy code (`app/strategy/`)
Real, existing modules: `tsmom.py` (the live edge), `mean_reversion.py`, `reversal.py`, `signals.py`, `sleeve_allocator.py` (equal/vol/regime; ships disabled — equal wins on 2 sleeves), `regime_detector.py`, `position_sizer.py`, `portfolio_construction.py`, `portfolio_heat.py`, `earnings_filter.py`, `entry_quality.py`, `benign_gate.py`.
**There is NO carry sleeve and NO VRP sleeve beyond the killed index short-vol.** Mean-reversion/reversal exist as code but are not validated, not live.

### How we now validate — **Ruler v2** (LIVE since 2026-06-13)
A new idea must clear a two-tier gate (built specifically to *stop* over-rejecting real edges):
- **PAPER tier:** plausibility (point-SR floor ≥0.30, implausibility ceiling, non-catastrophic worst-regime) **+ a light one-sided HAC-SR significance floor (HAC p < 0.05 on the pooled OOS book series)** + a sleeve cap. PF/Calmar are demoted to report-only. Declared **diversifiers/risk-premia get a worst-regime waiver** (so a crisis-positive sleeve isn't killed for losing in a bull regime).
- **CAPITAL tier:** Bayesian posterior **P(SR>0) ≥ 0.95** (replaces the saturated DSR; registry trial-count shrinks a mean-zero prior) + **live-paper is a STRUCTURAL requirement** (the posterior must include a live-paper observation — a clean backtest alone is provably insufficient) + multi-factor residual-α t + stationary-bootstrap + PBO (if M>1) + a hard power floor.

### The harness
- **CPCV** (~8 folds across k,p combinatorial paths; N_eff = n_folds), purge + embargo (85 calendar days swing / 2 trading days intraday).
- **Windows:** ~19y for trend, ~4y for anything options-dependent.
- **Inference core** (`app/research/inference.py`): Lo-2002 HAC Sharpe t, Politis-Romano stationary bootstrap, Bailey/López-de-Prado PBO via CSCV, multi-factor α.

**The empirical takeaway you should internalize before proposing:** we ran the two data-complete candidates (P5 broadened trend, index VRP) back through the *new, less-Type-II* gate on 2026-06-14. **Both still failed — on real significance/SR with the regime backstop waived.** The gate is now working correctly; the failures are honest negatives. A genuinely new edge therefore needs **new data or new strategy code**, not a re-run of what we have.

---

## 5. The binding constraints (please respect these)

1. **Free-data-only going forward.** The $79 options feed ends 2026-06-17 → frozen 2022–2026 options store, no new bars, no vendor IV/greeks/NBBO ever (we compute greeks; live NBBO = Alpaca free, 2 days deep). Paid feeds we keep: FMP Starter **$29 only on `/stable/`** (earnings/fundamentals/analyst). Everything else (yfinance, FRED, Alpaca paper, AlphaVantage) is free.
2. **~8-fold CPCV power is the binding constraint, not the t-bar.** Calibration proved 3/5 *true zero-SR nulls* clear t≥2.0 on 8 folds, and PEAD's t=3.33 ≈ a noise null's 3.47. Conversely a real Sharpe 0.4–0.7 edge over ≤4y can't reliably clear t≥2. **Realistic edges live at SR ~0.4–0.7, and statistical power is what kills them.** Long history (the 19y free ETF data) is the only escape from this.
3. **US-centric.** US equities + US-listed ETFs (which do give global/asset-class exposure: EFA, EEM, TLT, GLD, DBC, UUP). No direct international/futures/FX feeds.
4. **The Type-II / power lesson is now structural** (it's why Ruler v2 exists). Proposals that require many independent bets or short windows to reach significance are not testable here.
5. **Single quant-dev bandwidth.** One person. Proposals should be buildable + testable in days-to-weeks, not require a data-engineering team or six-figure data spend.

---

## 6. The genuine open question

Given all of the above — **where does undiscovered alpha plausibly still live that we can actually TEST with what we have?**

Concretely, the assets that are *both* deep-history-free *and* under-exploited here:
- **19-year free daily ETF history** across equities/bonds/gold/commodities/FX (the only data with the power to clear an 8-fold gate). TSMOM works on it. What *else* does — carry/term-structure across that ETF set? Cross-asset trend/value combinations? Trend at other speeds/breadths that *aren't* just a weaker copy of the live sleeve (P5 already failed that)?
- **A 9,774-event PIT R1K earnings panel + frozen 4y options surface** — PEAD's *level* is dead, but is there a *conditional* event structure (regime, dispersion, cross-sectional) that survives, given options as a *conditioning* variable rather than a standalone signal (which is dead)?
- **FRED macro + regime labels** as an allocation/timing layer over the one thing that works, rather than as a standalone signal.

We are explicitly NOT looking for: another long-only cross-sectional ML ranker, another options-as-equity-signal decile sort, another short-interest factor, or a threshold-swept filter — all four families are dead with named numbers above.

---

**The single most important fact to internalize before proposing anything:**

> After running every candidate through a deliberately less-strict, rebuilt gate, the ONLY edge that survives is a 19-year free-ETF trend sleeve (Sharpe +0.71) — so any proposal must either exploit deep free history (to beat the ~8-fold power ceiling) or be testable on the frozen data we already own; a clever idea that needs new paid data or thousands of independent bets cannot be validated here.
