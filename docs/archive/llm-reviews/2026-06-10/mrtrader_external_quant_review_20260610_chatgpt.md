# External Quant Review — MrTrader Alpha-v5

**Reviewer stance:** senior quant / quant-engineering review, based on the 2026-06-10 file package.  
**Scope:** document-level review of the architecture, validation methodology, live/paper state, and options-data roadmap. I did **not** inspect source code beyond the design evidence summarized in the attached docs, so code-level correctness is not guaranteed.  
**Output intent:** one self-contained file suitable for feeding to other LLMs or using as a next-phase planning artifact.

---

## 0. Executive verdict

The blunt version: **you do not yet have a capital-grade automated trading system. You have a much-better-than-retail research and validation platform, one weak-but-real event edge, a trend diversifier, and a valuable new options-derived information set.** That is still progress. The danger is that the system’s engineering maturity can make the research look more mature than the statistical evidence actually is.

My top-level read:

1. **The validation harness is unusually honest for a single-operator system, but it is still not the right final judge for sparse event alpha.** CPCV, purge/embargo, PIT data, daily MTM, per-fold retraining, and the negative-result discipline are real strengths. But for PEAD-like strategies, the true independent unit is not a fold or a daily return; it is an earnings/event cluster. CPCV should be treated as a coverage and robustness tool, not the main significance engine.

2. **The options data is more valuable as an equity/event signal layer than as an executable options strategy layer.** With 112.8M EOD option bars across 733 underlyings from 2022-06-09 to 2026-06-08, survivorship-safe and PIT, you have a legitimate signal research asset. But the hard limits — no historical NBBO, no OI, no intraday, no historical vendor IV/greeks — mean you should not overclaim dealer-gamma, flow direction, quote alpha, or precise executable vol-arb. Use this data to infer expectations, crowding, uncertainty, skew, and event repricing.

3. **Your options conclusions are directionally right but slightly misframed.** Single-name earnings short-vol should stay dead. Index short-vol is not dead as a return source; it is dead as standalone alpha. It belongs in a small, tail-budgeted, book-level risk-premium sleeve only if it improves the combined PEAD + trend book under a pre-registered overlay. Do not keep tuning it until it passes an alpha gate.

4. **Your best next research path is not “more options strategies.” It is an options-informed event-relative-value engine.** PEAD is your only demonstrated edge family. Use the options surface to separate “true unpriced surprise” from “expected move already priced,” but do it as a continuous, pre-registered event score — not another fragile threshold sweep.

5. **The largest landmine is research meta-overfitting.** You have done many iterations, many LLM reviews, many bug fixes after seeing results, multiple thresholds, multiple gates, and multiple strategy families. That is normal research, but the effective number of trials is far above any simple `N_TRIALS_TESTED=250` constant. You need an experiment registry and alpha-budgeting discipline before the next phase, not after.

My recommended strategic pivot:

> **Stop trying to prove a standalone “capital-grade alpha sleeve” from thin data. Build a controlled, beta-aware event trading program where PEAD is the seed edge, options data is the expectation/crowding layer, trend is the crash diversifier, and live paper reconciliation is the final arbiter.**

---

## 1. What I would trust today vs. what I would not

### I would trust

**1. The negative verdicts on long-only price/fundamental ML and intraday 5-minute ML.**  
The docs show the earlier ML results were contaminated by frozen-model in-sample evaluation and that the corrected per-fold OOS results collapsed: swing ranker around +0.22 Sharpe with t-stat 0.17, intraday around -2.80 Sharpe with t-stat -6.85. That is not a live-trading candidate. Do not spend more cycles trying to rescue it with clever filters.

**2. The broad negative verdict on single-name earnings IV-crush short-vol.**  
Even after the canonical re-parameterization, the economics look like a thin premium eaten by single-name options costs and event tail risk. The conclusion “not worth standalone execution” is right.

**3. The statement that index VRP is real but not standalone alpha.**  
The index condor result with PF around 2.24 at 1x spread and 1.75 at 2x spread is economically meaningful, but the Sharpe is essentially flat and the tail is the product. That belongs in portfolio construction, not in an alpha promotion gate.

**4. The options data architecture direction.**  
Building the contract universe from actual OPRA day files, retaining expired contracts, and using a +1 business day `knowable_date` is the right default bias: conservative, survivorship-safe, and hard to accidentally leak.

**5. The current “PEAD is real but underpowered” conclusion.**  
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

The harness has several features I would expect from an institutional research process and rarely see in single-operator systems:

- **PIT and survivorship discipline.** The options universe built from daily files rather than today’s active chain is the correct approach. For short-premium tests, this is not optional; otherwise you silently delete the most important losing/expired contracts.
- **Daily mark-to-market equity curves.** This is necessary for meaningful Sharpe, drawdown, and exposure analysis. The docs correctly distinguish daily-MTM promotion paths from entry-date-only tier-2 tools.
- **Per-fold retraining for ML.** This fixed the biggest original invalidity: frozen models scored on folds inside their own training window.
- **CPCV path-correlation awareness.** The docs explicitly state that CPCV paths are correlated and that effective N is closer to the number of folds than the number of paths.
- **Rules-based overlap-guard fix.** Bypassing trained-model overlap rules for rules-based event scorers was the right correction. Otherwise you were biasing the PEAD fold set for no leakage benefit.
- **Residual-alpha diagnostic.** CAPM/HAC residual alpha is exactly the right question after the ranker beta episode: “Does this survive hedging SPY?”
- **Negative-result culture.** This is underrated. You killed multiple appealing ideas. That discipline is a core edge if you preserve it.

### 2.2 Where it can still inflate results

#### A. Research meta-overfitting is not solved by CPCV

CPCV protects a single research run from some leakage and fold luck. It does not protect the *research program* from repeated human/LLM iteration.

Your effective search space includes:

- strategy families tried;
- feature families tried;
- thresholds and robustness sweeps;
- universe definitions;
- hold periods;
- cost assumptions;
- regime overlays;
- live/backtest bug fixes after seeing results;
- gate recalibrations;
- “parked but not killed” ideas that may later be revived;
- prompts sent to multiple LLMs that influence the next iteration.

That means the true number of implicit trials is not cleanly represented by a constant like 250. This does not mean the research is bad. It means the system needs **experiment accounting**.

Concrete fix:

Create a mandatory `research_registry.yaml` or database table. Every run gets:

- hypothesis ID;
- parent hypothesis ID;
- strategy family;
- pre-registered features;
- pre-registered thresholds or explicit “exploratory” tag;
- universe;
- sample window;
- folds;
- cost model;
- code commit hash;
- data snapshot hash;
- expected mechanism;
- acceptance criteria before running;
- result;
- decision: kill / park / promote-to-paper / live / exploratory-only.

Then separate results into:

- **Exploratory:** can inspire ideas, cannot promote.
- **Confirmatory:** pre-registered and eligible for paper/live gates.
- **Live-paper confirmation:** only forward data after the decision timestamp.

No more “this looks promising; let’s sweep it once more” unless that sweep is labeled exploratory and cannot promote.

#### B. CPCV is the wrong primary significance unit for event strategies

For PEAD and earnings options signals, the unit of independence is not daily return and not CPCV path. It is closer to:

- earnings date cluster;
- calendar quarter;
- sector-event cluster;
- firm-event pair with dependence across adjacent events.

If the system gets 19 quarterly earnings clusters, the honest statistical power is about 19 clusters, not hundreds of daily returns. The documents already acknowledge this, but the gate architecture still risks letting CPCV path metrics dominate the story.

Concrete fix:

For every event strategy, make the primary report table cluster-based:

| Metric | Primary unit |
|---|---|
| Mean event return | firm-event, clustered by announcement date/week |
| t-stat | two-way clustered by calendar cluster and symbol, or block bootstrap |
| Residual alpha | daily book returns, but block bootstrapped by event cluster |
| Drawdown | daily MTM path |
| Robustness | leave-one-quarter-out, leave-one-sector-out, leave-top-10-events-out |
| CPCV | coverage/robustness only, not final inference |

I would add a hard warning:

> “For event strategies, CPCV t-stat is not promotion-grade unless corroborated by event-cluster bootstrap.”

#### C. Residual alpha is still diagnostic, but it should gate capital

The system learned the hard way that beta can masquerade as alpha. Yet residual-alpha is currently diagnostic-only in the significance gate. For paper, that is fine. For real capital, it should not be optional.

Concrete capital gate for equity/event sleeves:

- residual beta estimated and reported;
- residual alpha positive after SPY hedge and sector ETF hedge;
- event-cluster bootstrap median residual alpha > 0;
- residual alpha t-stat hurdle can be modest at paper stage, but must be non-trivial for capital;
- no single quarter contributes more than a fixed percentage of residual P&L.

For PEAD specifically, I would require a sector-neutral or SPY-hedged report beside the long-only report. Long-only PEAD may still be tradable, but you need to know exactly how much you are being paid for event drift vs. equity beta.

#### D. Live/backtest mismatch is a bigger risk than remaining statistical leakage

The docs show repeated production-fidelity bugs. That tells me the system’s biggest practical risk is no longer “obvious look-ahead.” It is “validated strategy mutates when it passes through PM/RM/Trader/live config.”

Concrete fix:

Build a **Strategy Replay Contract**:

For each validated strategy, define a canonical `StrategySpec` that produces:

- signal timestamp;
- eligible universe;
- entry decision;
- intended order;
- intended quantity;
- intended risk cap;
- intended exit rule;
- all gating reasons.

Then run the same historical days through:

1. research scorer;
2. backtest simulator;
3. paper shadow live path;
4. actual order-generation path.

The output should diff at every stage. A live strategy is not eligible for real capital until the diff report is boring.

The multi-agent PM/RM/Trader design is fine for observability, but it is dangerous if each layer can independently reinterpret the strategy.

#### E. Cost/fill assumptions need empirical calibration

Equity costs of 3bps/5bps/5bps may be acceptable for liquid ETFs and large caps, but event names around earnings can gap, widen, and reject clean open assumptions. PEAD entries as marketable limits at ask + 10bps may track next-open in spirit, but the live fill distribution must become the source of truth.

Concrete fix:

Create a `fill_quality` table for every paper/live order:

- signal timestamp;
- intended reference price;
- open/mid/last/ask reference if available;
- submitted limit;
- fill price;
- fill delay;
- fill ratio;
- realized slippage vs next-open assumption;
- spread estimate;
- liquidity percentile;
- earnings/event flag.

Then backtest PEAD and trend under the empirical fill distribution, not just static bps.

For options, require separate treatment:

- EOD close is acceptable for marks.
- EOD close is **not** enough for execution unless you apply harsh spread/stale filters.
- Any option strategy that earns its edge inside the spread should be killed immediately.

### 2.3 Where the harness may hide a real edge

#### A. Date-level `knowable_date` can be overly conservative for event options signals

The +1 business day lag is safe for generic EOD option bars. But earnings are timestamped BMO/AMC. A blunt date-level lag can suppress legitimate pre-event information.

Example:

- For an AMC earnings report, the option close at 16:00 on the report date can be a causal pre-announcement implied-move observation if the announcement occurs after the close.
- For a BMO report, the prior day’s close is the causal pre-announcement observation.

If the system always treats report-date option close as knowable only the next day, it may be too conservative for AMC events. If it uses report-date close for BMO, it leaks.

Concrete fix:

Add an event-time-aware as-of model:

- `announcement_timestamp` with BMO/AMC certainty flag;
- `pre_event_option_snapshot_date`:
  - AMC: same trading day close may be valid if decision happens after the close and before announcement only if timestamp confirms that ordering;
  - BMO: previous trading day close;
  - unknown: previous trading day close only;
- `post_event_decision_time`:
  - next regular-session open or next close, explicitly modeled.

This may recover signal power without loosening PIT discipline.

#### B. Binary gates may discard weak but additive edges

Your current process is excellent at killing standalone noise. It may also kill genuinely additive weak edges.

This matters for:

- PEAD conditioners;
- index VRP;
- trend overlays;
- tail hedges;
- risk reducers.

Concrete fix:

Separate **alpha sleeves** from **book components**:

| Component type | Evaluation lens |
|---|---|
| Standalone alpha | residual alpha, cluster significance, beta/sector neutrality |
| Risk premium | expected return per tail budget, contribution to portfolio utility, crash behavior |
| Diversifier | drawdown reduction, crisis convexity, correlation stability |
| Filter/conditioner | host-sleeve improvement, trade-count impact, monotonicity, robustness |
| Tail hedge | crisis payoff, carry drag budget, convexity under gap scenarios |

Do not ask all five component types to pass the same gate.

#### C. Sparse event strategies should not be forced to prove too much too early

If PEAD has only ~4 independent earnings clusters per year, demanding institutional statistical certainty before any small live deployment is impractical. The correct response is not to lower standards blindly; it is to scale exposure by evidence quality and use forward paper/live confirmation.

Concrete fix:

Use evidence-tiered sizing:

- **Research pass:** pre-registered, positive mechanism, no obvious leak.
- **Paper telemetry:** no capital, full live-path reconciliation.
- **Micro-capital telemetry:** tiny capital, hard loss cap, only after live/path diff passes.
- **Capital sleeve:** after enough forward events, residual alpha remains positive, and live slippage matches assumptions.

This is more realistic than “wait for p<0.05 on 4 years of sparse events.”

---

## 3. Ranked highest-EV research directions

### Rank 1 — Options-informed PEAD v2: continuous event surprise score

**This is the best next project.** It uses the only demonstrated edge family and the new data in the most plausible way.

#### Mechanism

PEAD exists because markets underreact to genuine information surprises. Options markets encode the pre-event distribution. A raw earnings reaction is not enough; a 5% move may be huge for one stock and fully expected for another. The right question is:

> “Was the realized event shock large, directional, and underpriced relative to the pre-event option-implied distribution?”

The failed binary implied-move filter was pointing in the right direction but implemented in the wrong form. Replace it with a continuous event score.

#### Candidate features

Use only features that are causal under the event-time-aware snapshot model:

1. **Signed surprise ratio**  
   `signed_announce_move / pre_event_implied_move`

2. **Absolute surprise ratio**  
   `abs(announce_move) / pre_event_implied_move`

3. **Event-vol richness**  
   Pre-event straddle implied move vs realized pre-event volatility and sector peers.

4. **Pre-event IV run-up**  
   Short-dated ATM IV percentile change from T-10 to T-1.

5. **Term-structure event kink**  
   Event-expiry IV minus adjacent non-event expiry IV, normalized by historical event kinks for that name.

6. **Skew / directional fear**  
   Put skew, call skew, risk reversal, and skew percentile. Do not overfit exact deltas; use robust buckets like 20–35 delta or moneyness bands.

7. **Option volume attention/crowding**  
   Call/put notional volume by moneyness and expiry, z-scored vs the same name’s trailing baseline. Since you do not know trade direction, call it attention/crowding, not buying/selling.

8. **Post-event vol repricing**  
   Post-event IV crush or residual IV relative to expected crush. High residual uncertainty after a genuine positive surprise may predict continued drift or reversal depending on sign; test, do not assume.

#### Model form

Do **not** start with XGBoost. Start with:

- a monotonic scorecard;
- logistic/linear model with regularization;
- or shallow GAM-style bins.

Reason: you have a small number of event clusters. A flexible model will hallucinate edge.

#### Test design

- Universe: R1K earnings events with option coverage.
- Entry: next regular-session open or close after announcement, pre-registered.
- Hedge report: raw, SPY-hedged, sector-ETF-hedged.
- Validation:
  - leave-one-quarter-out;
  - leave-one-year-out if enough events;
  - leave-one-sector-out;
  - leave-largest-10-events-out;
  - event-cluster bootstrap by earnings week/quarter;
  - decile monotonicity of score vs forward residual return.
- Acceptance:
  - monotonic or at least stable top-vs-bottom event score relationship;
  - no single threshold required;
  - positive residual alpha after hedging;
  - improvement not concentrated in one quarter or mega-cap cluster;
  - live paper confirms fill/slippage.

#### Why this could be real alpha

This is not just selling vol or buying beta. It is conditioning a documented behavioral anomaly — underreaction to earnings information — on the market’s own ex-ante expectation. The options surface is not the trade; it is the measurement device.

---

### Rank 2 — Event-relative-value PEAD: beta/sector-neutral event pairs

#### Mechanism

Your current PEAD is long-biased and partly beta-like. Instead of trying to make a generic cross-sectional ranker work, make the cross-section event-specific:

- long stocks with positive, underpriced, options-confirmed surprises;
- short sector ETF, SPY, or matched peer basket;
- possibly short weak/negative event names only if borrow/liquidity constraints are manageable.

The idea is to isolate post-event drift while stripping market and sector exposure at the trade level.

#### Why this could be alpha

Generic cross-sectional ML failed because it was mostly equity beta. Event-relative-value is different: the ranking universe is names with fresh information shocks, and the hedge is designed to remove broad exposure.

#### Test design

Start simple:

1. For each event, compute the options-informed surprise score.
2. If positive and high-score, go long the stock.
3. Hedge with a fixed beta estimate using SPY and/or sector ETF.
4. Hold 3, 5, 10 trading days in a pre-registered grid.
5. Attribute P&L into stock residual, hedge P&L, sector residual, and gap/open slippage.

Then test matched peer baskets only after the ETF hedge baseline works.

Acceptance:

- residual alpha positive after hedge;
- lower drawdown than long-only PEAD;
- not materially worse fill quality;
- event-cluster t-stat improves or drawdown-adjusted utility improves.

#### What not to do

Do not resurrect the dead generic ranker and call it “event-enhanced ML.” Keep the model event-local and interpretable.

---

### Rank 3 — Options volume/crowding as an event and short-horizon equity signal

#### Mechanism

Even without OI or trade direction, option volume by strike/expiry/moneyness carries information:

- attention;
- disagreement;
- speculative crowding;
- hedging pressure;
- event anticipation;
- post-event unwind.

You cannot say “dealers are short gamma” without OI. But you can say “short-dated OTM call volume exploded relative to this name’s baseline before earnings.” That may predict post-event drift, reversal, or gap risk.

#### Feature examples

For each underlying and date:

- total option notional volume z-score;
- call/put notional ratio z-score;
- short-dated volume share vs long-dated volume share;
- OTM call wing activity percentile;
- OTM put wing activity percentile;
- near-ATM straddle volume percentile;
- volume-weighted moneyness;
- volume-weighted DTE;
- volume-weighted computed gamma/delta exposure **as a flow proxy**, not dealer positioning.

#### Best use cases

- PEAD conditioner: avoid crowded “everyone already positioned” events or favor events where options attention confirms underreaction.
- Gap-risk filter: reduce size when put-wing volume/skew indicates unusual downside tail risk.
- Post-event unwind: test whether extreme pre-event call crowding reverses after a positive gap.

#### Test design

- Use only data knowable before the entry decision.
- Normalize per-symbol; raw option volume is structurally different across AAPL vs a smaller R1K stock.
- Bucket features into deciles; look for monotonicity.
- Report results separately for earnings vs non-earnings days.
- Always sector/beta-adjust.

---

### Rank 4 — Cross-sectional implied-volatility richness/cheapness as an equity signal

#### Mechanism

The computed IV surface can be used to ask:

> “Is the options market pricing this stock as unusually risky relative to its own realized risk, sector peers, and upcoming events?”

This can create equity signals even if trading the options directly is not attractive.

Potential hypotheses:

1. **High put-skew / high IV with no realized deterioration** may indicate excessive fear and predict equity rebound.
2. **Cheap implied move before an event** may identify underpriced surprise potential.
3. **Rich implied move before an event** may identify crowded events where drift is weaker after the announcement.
4. **Term-structure inversion in single-name vol** may identify unresolved information risk and change optimal PEAD holding period.

#### Test design

- Build daily 30D ATM IV estimate and skew/term features for liquid names.
- Compare to trailing realized vol, sector median IV, and stock-specific percentile.
- Exclude earnings windows first, then run earnings-specific tests separately.
- Evaluate as an equity residual-return predictor, not as an option P&L strategy.
- Require stability across sectors and years.

#### Caution

This is lower EV than PEAD v2 because cross-sectional IV predictors can easily become another weak cross-sectional ML line. Keep it interpretable and low-dimensional.

---

### Rank 5 — Index VRP as a small book-level risk-premium sleeve, not alpha

#### Mechanism

Index short-vol is paid for bearing crash/convexity risk. It naturally complements trend if trend actually performs in crises. The right question is not “does index short-vol pass an alpha gate?” It is:

> “Does a small, mechanically defined, tail-budgeted index VRP sleeve improve the PEAD + trend portfolio after realistic spread stress and drawdown constraints?”

#### Pre-registered structure

Use one fixed structure, not a tuning lab:

- underlyings: SPY, QQQ, IWM only at first;
- DTE: 30–45;
- short strikes: approximately 16 delta / 1.5x realized-SD equivalent;
- defined-risk condor only;
- no naked short options;
- no single-name short-vol;
- no trade when simple risk-off rules trigger.

Risk-off rules should be minimal and theory-driven. Example:

- no new short-vol if VIX term structure is inverted;
- no new short-vol if SPY is below 200-day MA and realized vol is rising;
- reduce or skip around known macro shock windows if already in your macro calendar.

Do not tune five overlays on four years of data.

#### Acceptance

Evaluate at the combined-book level:

- improves expected utility or Sharpe by a pre-registered margin;
- does not worsen max drawdown beyond a fixed threshold;
- survives 2x spread stress;
- has explicit tail loss budget;
- does not rely on the one best post-2022 regime segment.

#### Caution

Four years is too little for short-vol tail estimation. If you want to take this seriously, get longer SPY/QQQ/IWM option history or use it only as a tiny live-paper exercise with hard loss caps.

---

## 4. Best alpha-shaped uses of the options data

Given the data limits, these are the best uses, ranked from most to least attractive.

### 4.1 Pre-event implied expectation vs realized event surprise

This is the highest-value use. The options market tells you what magnitude of move was expected. PEAD should be conditioned on whether the actual event was large relative to that expectation.

Do this as:

- continuous score;
- decile ranking;
- monotonicity check;
- event-cluster inference;
- no single magic threshold.

### 4.2 Event-vol term-structure kink

For earnings, compare event-expiry IV to adjacent expiries. A large event kink means the market expects a discrete move. The relation between kink size and post-event drift is exactly the kind of question your data can answer.

Potential interpretations:

- Large kink + large positive surprise: maybe genuine surprise despite high expected move.
- Large kink + move within expectation: likely priced-in, weaker drift.
- Small kink + large realized move: underpriced information shock, stronger drift candidate.

### 4.3 Skew and tail asymmetry as risk conditioners

Put skew and call skew can help size or skip trades:

- extreme put skew before earnings may warn of downside information risk;
- extreme call skew may indicate speculative upside crowding;
- skew normalization by symbol and sector is essential.

Use this for sizing/conditioning before using it for direction.

### 4.4 Option-volume attention proxies

You do not have OI or trade direction. Still, abnormal option volume is useful.

Best use:

- attention/crowding signal;
- event risk flag;
- post-event unwind predictor;
- not dealer positioning.

### 4.5 Implied-vol richness vs realized-vol forecast

Compute IV minus expected realized volatility using simple, pre-registered RV forecasts. Use it to condition equity trades and identify fear/crowding.

Caution: this can quickly become a generic cross-sectional factor. Keep it interpretable.

### 4.6 What not to claim or build from this data

Do not build or market these as robust with the current data:

- true dealer gamma / dealer positioning — no OI;
- intraday vol-arb — no intraday options data;
- quote/mid execution alpha — no historical NBBO;
- market-making/microstructure strategies — EOD OHLCV is the wrong data;
- precise option flow direction — volume is not signed;
- single-name short-premium production strategies — costs and tails dominate.

---

## 5. Architecture and design gaps

### Gap 1 — Research governance is behind engineering maturity

The codebase appears to have good tests, docs, and hardening. The research process now needs the same rigor.

Build:

- experiment registry;
- pre-registration templates;
- exploratory vs confirmatory run labels;
- commit/data hashes;
- trial accounting;
- automatic “cannot promote exploratory result” enforcement;
- final decision log tied to run artifacts.

This will do more for real alpha discovery than another model family.

### Gap 2 — StrategySpec unification between research and live

Validated strategies should be pure, versioned specifications. The live PM/RM/Trader path should execute the spec, not reinterpret it.

Build:

- `StrategySpec` schema;
- canonical signal and sizing function;
- golden-date replay tests;
- backtest-vs-live diff artifacts;
- order-intent vs order-fill reconciliation.

The recent live-path bugs are symptoms of this missing contract.

### Gap 3 — Portfolio risk engine should be more central

Right now the system has sleeves and risk rules, but the next phase needs portfolio-level risk attribution:

- gross and net exposure;
- beta;
- sector;
- factor exposure;
- event-cluster exposure;
- liquidity exposure;
- option delta/gamma/vega if options are live;
- drawdown contribution by sleeve;
- expected shortfall or stress scenarios.

The portfolio manager should know when PEAD, trend, and short-vol are all implicitly the same macro bet.

### Gap 4 — Paper/live telemetry should be promotion-critical

You have trackers, but the next step is to make them gates:

- fill rate;
- slippage vs assumption;
- live signal diff vs research signal;
- live size diff vs research size;
- P&L attribution vs expected;
- overlay suppression counts;
- missed trade count;
- broker/order rejection count.

A strategy should not move from paper to capital until telemetry matches the research assumptions within tolerance.

### Gap 5 — Options surface construction needs a quality layer

If options data becomes a signal engine, build a robust surface-quality module:

- stale close filter;
- min volume/notional filter;
- crossed/bad price guards if inferable;
- implied-vol solver failure flags;
- arbitrage sanity checks across strikes/maturities;
- split/corporate-action guards;
- surface interpolation rules;
- fallback behavior when surface is sparse.

Do not let a noisy computed IV from a stale contract drive an event score.

### Gap 6 — The system is overbuilt in some live-agent areas and underbuilt in research attribution

The three-agent design is useful for explainability, but for validated rules-based strategies it can introduce avoidable mismatch. Conversely, attribution needs more depth.

I would simplify the live path for validated sleeves:

- PEAD strategy spec emits target positions;
- trend strategy spec emits target ETF weights;
- risk layer clips or rejects;
- trader executes;
- every modification is logged.

Do not let PM, RM, and Trader each contain independent strategy logic.

### Gap 7 — Data horizon is insufficient for some strategy classes

Four years of options data is fine for signal research and event conditioning. It is not enough to estimate crash-tail behavior for short-vol. Do not let a four-year index option backtest decide a permanent short-vol allocation.

If short-vol remains a serious candidate, get longer data or keep it tiny.

---

## 6. The first five things I would change

### 1. Freeze strategy sprawl and make PEAD + trend paper-live reconciliation the immediate priority

Do not add another sleeve until the current validated sleeves are live-path boring.

Deliverables:

- PEAD activated in paper at telemetry size;
- trend activated in shadow, then paper;
- daily backtest-vs-live signal diff;
- sizing diff;
- fill diff;
- overlay suppression report;
- weekly realized-vs-expected report.

Why first: your recent bugs show that live fidelity is the biggest practical gap. A weak edge with perfect execution is better than a stronger backtest with mutated production behavior.

### 2. Rebuild PEAD research around event-cluster inference and options-informed continuous scoring

This is the main research build.

Deliverables:

- event-time-aware option snapshot logic;
- continuous surprise score;
- options-surface quality module;
- cluster bootstrap report;
- leave-quarter/sector/year robustness;
- sector/SPY-hedged residual alpha;
- decile monotonicity report.

Why second: this is the highest-probability path to actual alpha because it builds on your only real edge.

### 3. Make residual-alpha and cluster-based evidence primary capital gates for event strategies

CPCV remains useful, but it should not be the final judge for events.

Deliverables:

- event strategy report template;
- residual-alpha gate for capital;
- cluster bootstrap gate for paper/capital;
- “single quarter dominance” warning;
- “top N trades contribution” warning.

Why third: it prevents beta and cluster luck from sneaking through.

### 4. Reframe the options program as signal-first, risk-premium-second, execution-last

New ordering:

1. options data as PEAD/event signal;
2. options data as risk conditioner;
3. index VRP as small book-level risk premium;
4. options execution only after stronger evidence and better execution data.

Kill or park:

- single-name earnings short-vol;
- threshold-only PEAD filters;
- dealer-gamma claims;
- quote-sensitive strategies.

Why fourth: your data is excellent for expectations and poor for microstructure execution.

### 5. Build the research registry before the next serious sweep

Deliverables:

- `research_registry` artifact;
- confirmatory run enforcement;
- automatic source/config hash capture;
- exploratory result watermark;
- result comparison dashboard.

Why fifth: this directly attacks the meta-overfitting problem that CPCV does not solve.

---

## 7. What I would kill, park, double down on, and build next

### Kill

- **Generic long-only cross-sectional ML on price/fundamental features.** You already proved it is beta/noise.
- **Retail intraday 5-minute ML.** Negative after costs; structurally difficult without microstructure edge.
- **Single-name earnings short-vol execution.** Cost/tail profile is not attractive.
- **Any options strategy requiring NBBO, OI, signed flow, or intraday data.** You do not have the data.
- **Threshold-spike filters.** If the edge only exists at exactly one threshold and flips nearby, it is not a deployable edge.
- **DSR as a serious green light.** Keep it as a diagnostic only.

### Park

- **Index short-vol.** Park as a possible tiny book-level risk premium. Do not continue parameter mining.
- **Cross-sectional/relative VRP with options execution.** Interesting, but only after the signal layer proves value and the surface-quality module exists.
- **Regime allocation schemes.** Equal beating vol/regime on two sleeves is a warning. Revisit only after there are at least three genuinely different components.

### Double down

- **PEAD as the seed edge.** It is weak, but it is the only real signal family.
- **Trend as a diversifier, not alpha.** It may be the most valuable portfolio component during crashes.
- **Options data as expectation/crowding measurement.** This is the best use of the new dataset.
- **Validation honesty and negative-result logging.** This is one of the strongest parts of the project.
- **Paper/live telemetry.** This is what will tell you whether the system is tradable.

### Build next

1. Event-time-aware options snapshot layer.
2. Options surface-quality and feature layer.
3. PEAD v2 continuous event score.
4. Event-cluster validation report.
5. StrategySpec replay/diff framework.
6. Research registry.
7. Portfolio-level risk attribution.

---

## 8. Specific implementation blueprint for the next 30–60 days

### Phase A — Production fidelity first

**Goal:** make current paper system boring and measurable.

Tasks:

- Activate PEAD in paper at telemetry size if not already active.
- Run trend in shadow for at least one rebalance, then paper if diff is clean.
- Generate daily signal/size/order/fill diff reports.
- Lock dead ML proposal paths off and remove any accidental dependencies.
- Confirm no strategy-specific sizing is overridden downstream.

Exit criteria:

- no unexplained proposal-to-order gaps;
- no sizing drift vs spec;
- fills within pre-defined slippage tolerance;
- EOD P&L tracker reconciles to broker positions.

### Phase B — Options signal foundation

**Goal:** make options features trustworthy before testing alpha.

Tasks:

- Build surface-quality filters.
- Build event-time-aware pre-event snapshot logic.
- Build standardized features:
  - ATM implied move;
  - event kink;
  - IV percentile;
  - skew percentile;
  - option-volume z-scores;
  - call/put notional ratio;
  - term-structure slope.
- Produce feature coverage report by symbol, sector, date, and event type.

Exit criteria:

- feature coverage high enough for R1K PEAD names;
- no stale/illiquid contract drives critical features;
- BMO/AMC causality verified by tests.

### Phase C — PEAD v2 confirmatory test

**Goal:** test whether options information improves event residual alpha.

Pre-register:

- feature set;
- score formula or model type;
- hold periods;
- hedge method;
- acceptance metrics.

Run:

- baseline PEAD;
- PEAD + continuous score top buckets;
- PEAD + hedge;
- PEAD + score + hedge.

Report:

- event-cluster bootstrap;
- sector/year/quarter robustness;
- decile monotonicity;
- residual alpha;
- trade-count impact;
- live implementability.

Exit criteria:

- score improves residual alpha or drawdown-adjusted return;
- improvement is not a single-threshold artifact;
- no one quarter or sector dominates;
- realistic fills do not erase it.

### Phase D — Book-level integration

Only after Phase C:

- combine PEAD + trend + PEAD v2 variant;
- evaluate equal vs fixed risk weights;
- keep allocator simple;
- test whether index VRP at tiny weight improves utility under fixed overlay.

Exit criteria:

- combined book improves after costs;
- no hidden beta concentration;
- max drawdown and tail behavior acceptable;
- live paper confirms assumptions.

---

## 9. Concrete acceptance criteria I would use

### For PEAD v2 / event alpha

Paper-eligible:

- event-cluster bootstrap median residual alpha > 0;
- cluster t-stat directionally positive, even if not institutionally conclusive;
- decile or bucket monotonicity visible;
- no single threshold dependency;
- SPY/sector-hedged result positive;
- no single quarter contributes more than 40% of P&L;
- no single name contributes more than 15% of P&L;
- fill stress does not erase the edge.

Capital-eligible:

- all paper criteria;
- live paper confirms fill/slippage and signal frequency;
- at least one forward earnings season confirms directionally;
- hard loss cap and telemetry sizing first;
- residual alpha remains positive after hedging.

### For index VRP

Paper-eligible:

- fixed pre-registered structure;
- fixed pre-registered risk-off overlay;
- survives 2x spread;
- improves combined book utility or drawdown profile;
- explicit tail loss budget.

Capital-eligible:

- much harder: either longer data or extended live paper;
- no size increase based solely on 2022–2026 results;
- defined-risk only;
- tiny allocation relative to trend/PEAD.

### For options-derived equity signals

Paper-eligible:

- features pass coverage and quality checks;
- signal remains after sector/beta neutralization;
- no reliance on stale/illiquid option prints;
- monotonicity across feature buckets;
- pre-registered confirmatory run.

---

## 10. Rookie mistakes you have mostly avoided — and the ones still nearby

### Mostly avoided

- survivorship bias in options universe;
- obvious EOD look-ahead;
- active-chain-only options history;
- in-sample frozen-model validation after the audit;
- treating gross Sharpe as sufficient;
- hiding negative results;
- assuming ML must work because it is ML.

### Still nearby

- letting a complex gate create false confidence on sparse data;
- letting repeated LLM/human iteration become unaccounted multiple testing;
- mistaking risk-premium carry for alpha;
- letting live PM/RM/Trader behavior drift from research spec;
- treating EOD option close as executable;
- calling volume-derived proxies “dealer positioning” without OI;
- reviving dead ideas because a new filter creates a temporary bump.

---

## 11. Final answer to the premise

Your premise — “single-operator system chasing capital-grade, beta-neutral alpha” — is too ambitious if interpreted literally. A single operator with modest capital, EOD/daily data, and four years of options history is unlikely to manufacture clean institutional-grade standalone alpha across many sleeves.

The better premise is:

> **Build a small, rigorously validated, event-driven trading program with explicit beta control, conservative sizing, strong live reconciliation, and a willingness to monetize weak but plausible edges only after paper/live evidence accumulates.**

That is achievable. It is also more likely to survive contact with real markets.

The highest-EV next move is not another broad strategy search. It is to turn PEAD into a better event-relative-value engine using options-implied expectations, while hardening the live path and tracking real slippage. If that fails, you will have learned something important: the current data stack may be better suited to disciplined semi-systematic trading and research than fully automated capital deployment. If it works, you will have a real foundation: modest event alpha, trend diversification, and an options-informed risk/expectations layer.

My final recommendation:

1. **Run PEAD + trend in paper with full reconciliation.**
2. **Build PEAD v2 as an options-informed continuous event score.**
3. **Make cluster-aware residual alpha the capital gate.**
4. **Keep index VRP parked as a tiny book-level risk premium, not alpha.**
5. **Add research registry / pre-registration before the next sweep.**

Do those before building another model family.
