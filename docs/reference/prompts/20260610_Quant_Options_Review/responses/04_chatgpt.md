# External Quant Review - MrTrader Alpha-v5 (ChatGPT)

**Reviewer stance:** senior quant / quant-engineering review, based on the 2026-06-10 file package.
**Scope:** document-level review of the architecture, validation methodology, live/paper state, and options-data roadmap. I did **not** inspect source code beyond the design evidence summarized in the attached docs, so code-level correctness is not guaranteed.
**Output intent:** one self-contained file suitable for feeding to other LLMs or using as a next-phase planning artifact.

---

## 0. Executive verdict

The blunt version: **you do not yet have a capital-grade automated trading system. You have a much-better-than-retail research and validation platform, one weak-but-real event edge, a trend diversifier, and a valuable new options-derived information set.** That is still progress. The danger is that the system's engineering maturity can make the research look more mature than the statistical evidence actually is.

My top-level read:

1. **The validation harness is unusually honest for a single-operator system, but it is still not the right final judge for sparse event alpha.** CPCV, purge/embargo, PIT data, daily MTM, per-fold retraining, and the negative-result discipline are real strengths. But for PEAD-like strategies, the true independent unit is not a fold or a daily return; it is an earnings/event cluster. CPCV should be treated as a coverage and robustness tool, not the main significance engine.

2. **The options data is more valuable as an equity/event signal layer than as an executable options strategy layer.** With 112.8M EOD option bars across 733 underlyings from 2022-06-09 to 2026-06-08, survivorship-safe and PIT, you have a legitimate signal research asset. But the hard limits - no historical NBBO, no OI, no intraday, no historical vendor IV/greeks - mean you should not overclaim dealer-gamma, flow direction, quote alpha, or precise executable vol-arb. Use this data to infer expectations, crowding, uncertainty, skew, and event repricing.

3. **Your options conclusions are directionally right but slightly misframed.** Single-name earnings short-vol should stay dead. Index short-vol is not dead as a return source; it is dead as standalone alpha. It belongs in a small, tail-budgeted, book-level risk-premium sleeve only if it improves the combined PEAD + trend book under a pre-registered overlay. Do not keep tuning it until it passes an alpha gate.

4. **Your best next research path is not "more options strategies." It is an options-informed event-relative-value engine.** PEAD is your only demonstrated edge family. Use the options surface to separate "true unpriced surprise" from "expected move already priced," but do it as a continuous, pre-registered event score - not another fragile threshold sweep.

5. **The largest landmine is research meta-overfitting.** You have done many iterations, many LLM reviews, many bug fixes after seeing results, multiple thresholds, multiple gates, and multiple strategy families. That is normal research, but the effective number of trials is far above any simple `N_TRIALS_TESTED=250` constant. You need an experiment registry and alpha-budgeting discipline before the next phase, not after.

My recommended strategic pivot:

> **Stop trying to prove a standalone "capital-grade alpha sleeve" from thin data. Build a controlled, beta-aware event trading program where PEAD is the seed edge, options data is the expectation/crowding layer, trend is the crash diversifier, and live paper reconciliation is the final arbiter.**

---

## 1. What I would trust today vs. what I would not

### I would trust

**1. The negative verdicts on long-only price/fundamental ML and intraday 5-minute ML.**
The docs show the earlier ML results were contaminated by frozen-model in-sample evaluation and that the corrected per-fold OOS results collapsed: swing ranker around +0.22 Sharpe with t-stat 0.17, intraday around -2.80 Sharpe with t-stat -6.85. That is not a live-trading candidate. Do not spend more cycles trying to rescue it with clever filters.

**2. The broad negative verdict on single-name earnings IV-crush short-vol.**
Even after the canonical re-parameterization, the economics look like a thin premium eaten by single-name options costs and event tail risk. The conclusion "not worth standalone execution" is right.

**3. The statement that index VRP is real but not standalone alpha.**
The index condor result with PF around 2.24 at 1x spread and 1.75 at 2x spread is economically meaningful, but the Sharpe is essentially flat and the tail is the product. That belongs in portfolio construction, not in an alpha promotion gate.

**4. The options data architecture direction.**
Building the contract universe from actual OPRA day files, retaining expired contracts, and using a +1 business day `knowable_date` is the right default bias: conservative, survivorship-safe, and hard to accidentally leak.

**5. The current "PEAD is real but underpowered" conclusion.**
This is the most important nuance. PEAD is not fake, but it is not statistically overwhelming either. Treat it as a weak edge with a plausible behavioral mechanism, not as a solved strategy.

### I would not yet trust

**1. Any final conclusion from a single thresholded options filter.**
The implied-move filter improving only at realized/implied < 1.0 and inverting at 1.25 is the exact signature of a threshold artifact. The right next version should be a continuous event score with monotonicity checks and pre-registered bins, not a binary cutoff.

**2. Any strategy whose live path is not byte-for-byte behaviorally reconciled to the backtest path.**
Your own docs show multiple live/backtest fidelity bugs: VIX block not firing live, PEAD sizing cap ignored, trend rebalance orphan risk, dead ML still proposing trades. These are not embarrassing; they are normal. But they prove that your architecture needs a stronger shadow/reconciliation layer before real capital.

**3. Any options execution backtest that relies on EOD close as if it were tradeable.**
Marking to real closes is good for MTM. It is not enough for executable options alpha. Without NBBO, EOD quote, or intraday data, options execution results should carry a heavy discount unless they are index options with deep liquidity and conservative spread stress.

**4. DSR as meaningful protection.**
Your own pipeline doc notes DSR saturation. I would keep it in reports but stop treating it as a serious research-safety mechanism. The real defense is pre-registration, cluster-aware inference, live paper, and strict experiment accounting.

**5. A regime ML classifier as a source of edge.**
Use it as a defensive sizing/risk overlay only. Macro regime classifiers are notoriously unstable out-of-sample, especially with small samples and many regime definitions. Do not let it become a hidden optimizer that rescues weak strategies in backtests.

---

## 2. Verdict on the validation harness

### 2.1 What is sound
- **PIT and survivorship discipline.** The options universe built from daily files rather than today's active chain is the correct approach.
- **Daily mark-to-market equity curves.**
- **Per-fold retraining for ML.**
- **CPCV path-correlation awareness.** Effective N closer to the number of folds than the number of paths.
- **Rules-based overlap-guard fix.**
- **Residual-alpha diagnostic.** CAPM/HAC residual alpha is exactly the right question after the ranker beta episode.
- **Negative-result culture.**

### 2.2 Where it can still inflate results

#### A. Research meta-overfitting is not solved by CPCV
CPCV protects a single research run from some leakage and fold luck. It does not protect the *research program* from repeated human/LLM iteration. The true number of implicit trials is not cleanly represented by a constant like 250.

Concrete fix: a mandatory `research_registry.yaml` / DB table. Every run gets: hypothesis ID; parent hypothesis ID; strategy family; pre-registered features; pre-registered thresholds or explicit "exploratory" tag; universe; sample window; folds; cost model; code commit hash; data snapshot hash; expected mechanism; acceptance criteria before running; result; decision: kill / park / promote-to-paper / live / exploratory-only.

Separate results into: **Exploratory** (can inspire ideas, cannot promote); **Confirmatory** (pre-registered, eligible for paper/live gates); **Live-paper confirmation** (only forward data after the decision timestamp).

#### B. CPCV is the wrong primary significance unit for event strategies
For PEAD and earnings options signals, the unit of independence is the earnings date cluster / calendar quarter / sector-event cluster. If the system gets ~19 quarterly earnings clusters, the honest statistical power is about 19 clusters, not hundreds of daily returns.

Concrete fix: for every event strategy, make the primary report table cluster-based (mean event return clustered by announcement date/week; t-stat two-way clustered by calendar cluster and symbol, or block bootstrap; robustness via leave-one-quarter-out, leave-one-sector-out, leave-top-10-events-out). Hard warning: "For event strategies, CPCV t-stat is not promotion-grade unless corroborated by event-cluster bootstrap."

#### C. Residual alpha should gate capital (not just be diagnostic)
For real capital on equity/event sleeves: residual beta estimated and reported; residual alpha positive after SPY hedge and sector ETF hedge; event-cluster bootstrap median residual alpha > 0; no single quarter contributes more than a fixed percentage of residual P&L. For PEAD specifically, require a sector-neutral or SPY-hedged report beside the long-only report.

#### D. Live/backtest mismatch is a bigger risk than remaining statistical leakage
Build a **Strategy Replay Contract**: for each validated strategy, a canonical `StrategySpec` producing signal timestamp; eligible universe; entry decision; intended order; intended quantity; intended risk cap; intended exit rule; all gating reasons. Run the same historical days through: research scorer; backtest simulator; paper shadow live path; actual order-generation path. The output should diff at every stage. A live strategy is not eligible for real capital until the diff report is boring.

#### E. Cost/fill assumptions need empirical calibration
Create a `fill_quality` table for every paper/live order (signal timestamp; intended reference price; open/mid/last/ask; submitted limit; fill price; fill delay; fill ratio; realized slippage vs next-open; spread estimate; liquidity percentile; earnings/event flag). Then backtest PEAD and trend under the empirical fill distribution, not just static bps. For options: EOD close OK for marks, NOT for execution unless harsh spread/stale filters; any option strategy that earns its edge inside the spread should be killed immediately.

### 2.3 Where the harness may hide a real edge

#### A. Date-level `knowable_date` can be overly conservative for event options signals
The +1 business day lag is safe generically but for AMC earnings the option close at 16:00 on the report date can be a causal pre-announcement implied-move observation if the announcement occurs after the close. Add an event-time-aware as-of model (announcement_timestamp with BMO/AMC flag; pre_event_option_snapshot_date AMC=same-day close if ordering confirmed, BMO=previous trading day close; post_event_decision_time = next open or close, explicitly modeled).

#### B. Binary gates may discard weak but additive edges
Separate **alpha sleeves** from **book components**. Different component types (standalone alpha; risk premium; diversifier; filter/conditioner; tail hedge) should not all pass the same gate.

#### C. Sparse event strategies should not be forced to prove too much too early
Use evidence-tiered sizing: Research pass -> Paper telemetry (no capital, full live-path reconciliation) -> Micro-capital telemetry (tiny capital, hard loss cap) -> Capital sleeve (after enough forward events, residual alpha positive, live slippage matches assumptions).

---

## 3. Ranked highest-EV research directions

### Rank 1 - Options-informed PEAD v2: continuous event surprise score
**This is the best next project.** Replace the failed binary implied-move filter with a continuous event score. Candidate features (causal under event-time-aware snapshot): signed surprise ratio (signed_announce_move / pre_event_implied_move); absolute surprise ratio; event-vol richness; pre-event IV run-up (T-10 to T-1); term-structure event kink; skew / risk reversal; option volume attention/crowding (z-scored vs baseline; call it attention not direction - volume is unsigned); post-event vol repricing.

Model form: do NOT start with XGBoost. Start with a monotonic scorecard / logistic-or-linear with regularization / shallow GAM bins (small number of event clusters - a flexible model will hallucinate edge).

Test design: R1K earnings events with option coverage; entry next session open/close (pre-registered); hedge report raw, SPY-hedged, sector-ETF-hedged; validation leave-one-quarter/year/sector-out, leave-largest-10-events-out, event-cluster bootstrap, decile monotonicity. Acceptance: monotonic/stable top-vs-bottom relationship; no single threshold; positive residual alpha after hedging; improvement not concentrated in one quarter or mega-cap cluster; live paper confirms fills.

### Rank 2 - Event-relative-value PEAD: beta/sector-neutral event pairs
Long stocks with positive, underpriced, options-confirmed surprises; short sector ETF/SPY/matched peer basket. Isolates post-event drift while stripping market and sector exposure at the trade level. Start simple (fixed-beta SPY/sector hedge), attribute P&L into stock residual / hedge / sector residual / gap slippage. Do not resurrect the dead generic ranker.

### Rank 3 - Options volume/crowding as an event and short-horizon equity signal
Even without OI or direction, option volume by strike/expiry/moneyness carries attention/disagreement/hedging-pressure information. Best use cases: PEAD conditioner (avoid crowded events); gap-risk filter (reduce size when put-wing volume/skew indicates downside tail risk); post-event unwind. Normalize per-symbol; bucket into deciles; report earnings vs non-earnings separately; always sector/beta-adjust.

### Rank 4 - Cross-sectional implied-volatility richness/cheapness as an equity signal
Use the computed IV surface to ask whether the options market prices a stock as unusually risky vs its own realized risk, sector peers, upcoming events. Evaluate as an equity residual-return predictor, not an option P&L strategy. Keep interpretable and low-dimensional (it can easily become another weak cross-sectional ML line).

### Rank 5 - Index VRP as a small book-level risk-premium sleeve, not alpha
Use one fixed pre-registered structure (SPY/QQQ/IWM, 30-45 DTE, ~16 delta, defined-risk condor only, no naked, no single-name, minimal theory-driven risk-off overlay). Evaluate at the combined-book level (improves utility/Sharpe by a pre-registered margin; does not worsen max DD; survives 2x spread; explicit tail loss budget). Caution: four years is too little for short-vol tail estimation.

---

## 4. Best alpha-shaped uses of the options data
4.1 Pre-event implied expectation vs realized event surprise (highest value; continuous score, decile ranking, monotonicity, event-cluster inference, no magic threshold). 4.2 Event-vol term-structure kink. 4.3 Skew and tail asymmetry as risk conditioners (use for sizing/conditioning before direction). 4.4 Option-volume attention proxies (attention/crowding, not dealer positioning). 4.5 Implied-vol richness vs realized-vol forecast (keep interpretable).

**What NOT to claim/build:** dealer gamma/positioning (no OI); intraday vol-arb (no intraday); quote/mid execution alpha (no NBBO); market-making; precise option flow direction (volume unsigned); single-name short-premium production strategies.

---

## 5. Architecture and design gaps
Gap 1 - Research governance is behind engineering maturity (build experiment registry, pre-registration templates, exploratory vs confirmatory labels, hashes, trial accounting). Gap 2 - StrategySpec unification between research and live (pure versioned spec; golden-date replay tests; backtest-vs-live diff artifacts). Gap 3 - Portfolio risk engine should be more central (gross/net, beta, sector, factor, event-cluster, liquidity exposure; drawdown by sleeve; stress scenarios). Gap 4 - Paper/live telemetry should be promotion-critical (fill rate; slippage vs assumption; live signal/size diff vs research; P&L attribution; overlay suppression counts; missed trades; rejections). Gap 5 - Options surface needs a quality layer (stale-close filter; min volume/notional; IV-solver failure flags; arbitrage sanity; split/corp-action guards; fallback when sparse). Gap 6 - Simplify the live path for validated rules-based sleeves (spec emits target positions; risk layer clips; trader executes; every modification logged - do not let PM/RM/Trader each contain independent strategy logic). Gap 7 - Data horizon insufficient for crash-tail of short-vol (four years is fine for signal research, not for permanent short-vol allocation).

---

## 6. The first five things I would change
1. **Freeze strategy sprawl and make PEAD + trend paper-live reconciliation the immediate priority** (do not add another sleeve until live fidelity is boring). 2. **Rebuild PEAD research around event-cluster inference and options-informed continuous scoring.** 3. **Make residual-alpha and cluster-based evidence primary capital gates for event strategies.** 4. **Reframe the options program as signal-first, risk-premium-second, execution-last.** 5. **Build the research registry before the next serious sweep.**

---

## 7. Kill / park / double down / build next
**Kill:** generic long-only XS-ML; retail intraday 5-min ML; single-name earnings short-vol execution; any options strategy requiring NBBO/OI/signed-flow/intraday; threshold-spike filters; DSR as a serious green light. **Park:** index short-vol (tiny book-level RP, no more parameter mining); relative VRP with options execution; regime allocation schemes. **Double down:** PEAD as seed edge; trend as diversifier; options data as expectation/crowding measurement; validation honesty + negative-result logging; paper/live telemetry. **Build next:** event-time-aware options snapshot layer; options surface-quality + feature layer; PEAD v2 continuous event score; event-cluster validation report; StrategySpec replay/diff framework; research registry; portfolio-level risk attribution.

---

## 8. Implementation blueprint for the next 30-60 days
**Phase A - Production fidelity first** (make current paper system boring and measurable; daily signal/size/order/fill diff reports; lock dead ML proposal paths off; EOD P&L reconciles to broker). **Phase B - Options signal foundation** (surface-quality filters; event-time-aware pre-event snapshot; standardized features ATM implied move / event kink / IV percentile / skew percentile / option-volume z-scores / call-put notional ratio / term-structure slope; coverage report; BMO/AMC causality tests). **Phase C - PEAD v2 confirmatory test** (pre-register feature set, score formula, hold periods, hedge method, acceptance; run baseline / +score / +hedge / +score+hedge; report event-cluster bootstrap, robustness, monotonicity, residual alpha, trade-count impact). **Phase D - Book-level integration** (combine PEAD + trend + PEAD v2; equal vs fixed risk weights; simple allocator; tiny index VRP under fixed overlay).

---

## 9. Acceptance criteria
**PEAD v2 / event alpha** - Paper-eligible: event-cluster bootstrap median residual alpha > 0; cluster t directionally positive; decile/bucket monotonicity; no single-threshold dependency; SPY/sector-hedged positive; no single quarter > 40% of P&L; no single name > 15% of P&L; fill stress does not erase. Capital-eligible: all paper + live paper confirms fill/slippage + >=1 forward earnings season + hard loss cap first + residual alpha remains positive. **Index VRP** - fixed pre-registered structure + risk-off overlay; survives 2x spread; improves combined book; explicit tail budget; capital much harder (longer data or extended live paper; defined-risk only; tiny allocation). **Options-derived equity signals** - features pass coverage/quality; signal survives sector/beta neutralization; no reliance on stale prints; monotonicity; pre-registered confirmatory run.

---

## 11. Final answer to the premise
Your premise - "single-operator system chasing capital-grade, beta-neutral alpha" - is too ambitious if interpreted literally. The better premise: **Build a small, rigorously validated, event-driven trading program with explicit beta control, conservative sizing, strong live reconciliation, and a willingness to monetize weak but plausible edges only after paper/live evidence accumulates.**

Final recommendation: 1. Run PEAD + trend in paper with full reconciliation. 2. Build PEAD v2 as an options-informed continuous event score. 3. Make cluster-aware residual alpha the capital gate. 4. Keep index VRP parked as a tiny book-level risk premium, not alpha. 5. Add research registry / pre-registration before the next sweep. Do those before building another model family.
