MrTrader System Review — Critical Assessment
Reviewer: Independent LLM critique against MrTrader briefing dated 2026-05-05
Posture: Direct, critical, hedge-fund-production lens. Optimized for surfacing what's wrong, not validating what exists.
Intended use: Hand to other LLMs (GPT, Gemini, etc.) for cross-review and consensus-building on next steps.

EXECUTIVE SUMMARY — THE FIVE THINGS THAT MATTER MOST

The Sharpe collapse from "older window" to "current data" on both champions is not a regime story — it's almost certainly statistical artifact. v142 went +1.181 → +0.310. v29 went +1.830 → -0.327. When you add ~6–12 months of fresh data and the result halves or flips sign on the same model, the original number was over-fit to the test window or selection-biased, not genuinely out-of-sample. This needs to be the lead diagnostic, not "tariff regime."
HPO variance of ~2.0 Sharpe on identical features is the smoking gun. If hyperparameter seeds alone move the result by 2 Sharpe units, the underlying signal is dominated by noise, not feature alpha. The "frozen HPO" workaround masks this — it doesn't fix it. You almost certainly do not have a deployable edge yet; you have a noisy signal that occasionally produces favorable backtests.
The multiple-comparisons problem has not been addressed at all. At least 10–15 model variants (v119, v142, v144, v145, v146, v29, v34–v41) have been tested. Picking the best by Sharpe and reporting that Sharpe is the textbook case of selection bias. The deflated Sharpe ratio (Bailey & López de Prado, 2014) on this trial count would likely deflate +1.181 to something near zero. This is existential for the credibility of every result quoted.
Walk-forward methodology has structural flaws beyond the regime issue. Three expanding folds, no purge/embargo, no combinatorial CV, gates not simulated, and — most importantly — no transaction cost model. Until trades are simulated net of slippage and fees, no Sharpe number is meaningful. On 5-min intraday bars at retail size, realistic slippage can be 5–15 bps per side, which on a 0.8×ATR target erodes most of the gross edge.
The system is sophisticated in the wrong places. Effort has gone into 84-feature swing models, 22-gate RM, agent architecture, HPO. Effort has not gone into: deflated Sharpe / multiple-testing correction, transaction cost modeling, label leakage audit, paper-vs-sim P&L reconciliation, feature correlation analysis, stationarity testing. Production-grade quant funds spend 70%+ of effort on the second list, not the first.

Honest framing: Despite real engineering polish, this system is several rigorous-statistics steps away from deployment. The path forward is not more features or more models — it is fewer, validated more carefully.

A. CRITICAL ASSESSMENT
What a quant at a serious systematic fund would flag in 60 seconds
ObservationSeverityWhy it mattersSharpe collapse on champion models when re-evaluatedCriticalIndicates the original number was not OOS in any meaningful senseHPO variance > target Sharpe gateCriticalThe gate is below the noise floor — passing it is luckNo deflated Sharpe / multiple-testing correctionCriticalAll reported Sharpes inflated by selection across ~15 variantsNo transaction cost / slippage model in walk-forwardCriticalGross Sharpe ≠ net Sharpe; on intraday this gap is large3 expanding folds with no purge/embargoHighNot enough independent test windows; cross-fold label leakage possibleWalk-forward doesn't apply PM gatesHighReported Sharpe is for unfiltered model; live performance differs84 features on ~18-month fold windowsHighCurse of dimensionality regardless of XGBoost regularizationTrain/serve skew via PRUNED_FEATURESHighFeatures available at inference but not training is a category errorChampion (v142) trained when fundamentals were NOT PIT-correctHighThe +1.181 has look-ahead bias; +0.310 may be the unbiased numberNo live-vs-sim P&L reconciliationHighThe single most informative diagnostic, missingClass imbalance fix via scale_pos_weight onlyMediumDoesn't address regime-specific imbalanceWorldQuant alphas mixed with rule-based entry filtersMediumConceptually inconsistent; WQ designed for cross-sectional rankingCross-sectional top-20% label on flat daysMediumForces 20% positive rate even when no symbol has true edge0.4×ATR stop on 5-min barsMediumLikely whipsaw-prone; check stop-out distributionNo probabilistic framing of model uncertaintyMediumXGBoost gives points; no calibrated probability of edge
Design decisions most likely limiting alpha

Mixing rule-based pre-filters (RSI_DIP, EMA_CROSSOVER) with ML scoring on top. The ML never learns to select entries — only to score what the rules already pre-filtered. That caps the information the model can extract to "is this rule-based candidate good?" rather than "is this symbol set up to move?" If RSI 45–58 is arbitrary, it bounds the alpha ceiling.
cs_normalize on intraday is doing real damage. By forcing every feature cross-sectional within day, you've eliminated all market-context information from the model — VIX, SPY level, macro state, time of week, day of week, week of month. None survive cs_normalize because they're constant across symbols on a given day. The interaction features (seg_x_atr_norm, seg_x_high_dist) recover this only crudely.
Single XGBoost across regimes. 2021 (zero-rate momentum), 2022 (rate shock), 2023 (recovery), 2024 (AI bull), 2025 (tariff/vol) are not the same data-generating process. Forcing one model to fit all of them is asking it to learn a useless average. Either regime-condition, sample-weight recent data, or accept that the model is a "median regime" model and gate hard at the PM layer.
Label horizons are arbitrary and don't reflect the trade. Swing label = "did target hit in 5 days?" but swing trades are "3–15 days." Mismatch. Intraday label = "rank in top 20% over 2 hours?" but execution is binary.
No regime detection inside the model. Opportunity score sits at PM layer as a gate, not as a model feature in any structured way. A more honest design: predict P(target hit) conditional on observed regime, with regime as a top-level feature.

Is the label design appropriate?
Swing — not really.

"Path quality" should mean MFE/MAE ratio over the hold or time-weighted return. The current binary label loses all this.
Better: train multi-objective with three labels:

Sign: sign of return at exit (binary)
Magnitude: R-multiple realized (regression)
Path: MFE/MAE ratio (regression)


Use magnitude prediction for sizing, sign for entry, path for stop optimization.

Intraday — fundamentally limited by the cross-sectional setup.

Top-20% rank labels guarantee 20% positive rate. On flat days, you label noise as winners.
Filter training days by minimum cross-sectional dispersion. If best symbol that day < 0.3 ATR, drop the day.
The realized-R failure (AUC 0.51) is buried as a footnote but is critical: features predict relative ranking but contain no information about magnitude. That's a low-quality alpha. Sizing has to be agnostic, and Sharpe has a hard ceiling.

Is the walk-forward methodology sound?
No.
IssueFix3 folds is too few10+ rolling folds, or combinatorial purged k-foldNo purge between train/testPurge 5 days swing, 1 day intradayNo embargo after test windowAdd 5-day embargoExpanding window lets distant past dominateCompare expanding vs rolling; rolling often better when DGP shiftsWalk-forward doesn't include execution layerAdd slippage, fees, partial fills; simulate gatesSingle time path testedUse combinatorial purged CV (López de Prado) for many backtest pathsHPO inside walk-forward not describedIf per-fold, leaking. If once before, "frozen" params overfit a specific split
Gate thresholds (avg Sharpe > 0.8 / 1.5, min fold > -0.3) aren't unreasonable in absolute terms, but they're below the noise floor given HPO variance. They should be deflated Sharpe > 0 with p < 0.05, not raw Sharpe > some number.

B. THE REGIME PROBLEM
Is Apr–Oct 2025 collapse a model, feature, label, or data problem?
Probably mostly a methodology problem masquerading as a regime problem. Suspected contributors in order of magnitude:

Selection bias (60%). The model that "passed" originally was the lucky tail of a distribution of variants. New data exposes the regression to mean.
Look-ahead in original training (20%). v142 was trained when fundamentals were not PIT-correct. The original +1.181 likely had subtle leakage. Current eval is closer to truth.
Genuine non-stationarity (15%). Tariff/vol shock is real, but markets have shocks regularly. A robust system should weather this.
Insufficient training data for the regime (5%). 2025 conditions have limited historical analog.

Test directly:

Re-train v142 with PIT-correct fundamentals on the original window. If +1.181 drops, you've quantified leakage.
Bootstrap walk-forward: resample fold splits 100×, plot Sharpe distribution. If +1.181 is in top 5% tail, that's selection bias.

Approaches to make models more regime-robust
ApproachProsConsRecommendationRegime-conditional models (separate per VIX bucket)Clean conceptuallySplits already-thin data; doubles overfit riskSkipRegime as a feature with interactionsAlready partialDoesn't help if regime distribution shiftsKeep, expandSample weighting by regime similarityFocuses learning on relevant historySubjective weight choicesTryOnline learning / continuous retrainingAdapts to new regimesHard to validate; chases noiseSkip for nowEnsembling models trained on different regimesRobust averagingUnclear how to weight liveTryHard regime gate at PM layer (current)Simple, conservativeWastes capital in regimes you could tradeKeep but tightenATR-normalized featuresRemoves regime amplitudeDoesn't address structural shiftsAlready partialDomain randomizationRobust by constructionHard to do well; introduces artifactsSkip
Recommended path: Keep PM opportunity score as a hard gate. Inside the model, add a regime feature (VIX bucket, SPY trend, term structure) and let XGBoost learn regime-specific decision rules via tree splits. Don't fragment data with separate models.
The deeper question: does this strategy class have edge in the 2025 regime at all? Honest answer might be no. A 2:1 R/R momentum strategy in tariff-shock chop is structurally disadvantaged. Trading less is a valid strategy.

C. FEATURE SUGGESTIONS
Likely-noise features (audit or remove)

Multiple RSI variants (rsi_14, rsi_7, stochrsi_k, stochrsi_d, williams_r_14, stoch_k) — variants of the same idea. Keep one.
Multiple momentum at same horizon family — collapse to 2–3.
WorldQuant alphas as a block — 13 features that may be mutually redundant. Permutation importance test; keep top 3–5.
options_* with sparse coverage — if <80% coverage, model is learning "is this symbol covered" not "what does options market signal."
insider_score — without clear definition, suspect; insider data has well-known filing-vs-effective-date issues.
NIS at 80% NaN — model learns "is news available" not "news quality." See section below.

Likely-missing features
Swing — established academic factors not present:
FactorSourceWhy usefulQuality (gross profitability/assets)Novy-Marx (2013)Robust factor; not in current fundamentalsInvestment / asset growthFama-French 5-factorKnown negative predictorEarnings revisionsManyComplementary to surpriseShort interest / days-to-coverNYSE/NasdaqSqueeze risk + crowded-trade signalIdiosyncratic volatilityAng et al. (2006)Robust low-IVOL anomalyDispersion of analyst forecastsDiether et al. (2002)High dispersion → lower returnsAccruals / earnings qualitySloan (1996)Classic anomalyPEAD — post-earnings driftBernard-Thomas (1989)Most robust earnings-related anomalyBeta-anomaly / low-volFrazzini-PedersenBeta-orthogonal featureTerm structure of vol (VIX9D vs VIX vs VIX3M)—Stress vs complacency, more than VIX levelCredit spreads (HY OAS, IG OAS)FRED, ICEMacro risk signal not currently capturedDollar index (DXY) momentum—Macro context that survives cs_normalize as global
Intraday — given cs_normalize constraint:
FeatureRationaleOrder flow imbalance / signed volumeCross-sectional; genuine intraday alphaTrade size distributionBlock prints vs retailBid-ask spread (live)Wide spreads → less reliable signals; gate or weightEffective tick volatilityTrue microstructure measureLead-lag with sector ETF (intraday)Intraday pairs/sector relativeIntraday seasonality (DoW × ToD)Survives cs_normalize partiallyClosing auction imbalance proxyIf accessible from PolygonShort-term reversal at open (5-min)Heston/Sadka effect at intraday scale
How to think about NIS given sparsity
NIS is not yet a feature, it's a hypothesis. With 80% NaN, no model can validate it. Three options:

Postpone. Pull NIS out entirely until backfill complete. Saves wasted retrains.
Restrict scope. Train and evaluate only on post-May-2025. Less power, but a clean test.
Use NIS as a gate, not a feature. At PM layer, suppress entries when NIS_downside_risk is high. Avoids encoding problem; uses NIS as designed.

Recommended: Do (3) immediately, then (1) until backfill, then full integration.

D. ARCHITECTURE SUGGESTIONS
PM → RM → Trader async architecture
Sound at system level. Specific concerns:
ComponentAssessmentRecommendationAsync queue between agentsGood decouplingAdd explicit timeout + dead-letter handlingPM re-scoring at execution timeExcellentDocument score-decay logic; ensure re-score uses same feature snapshot22 sequential RM gatesThorough but rigidGate categories with parallel evaluation within categoryDecision audit trailExcellentAdd gate-pass-rate dashboards; calibrate quarterly from real outcomesSQLite as primary state storeFine for scalePhase 100 (Alpaca SOT) is correct directionSubprocess retrainingGood isolationAdd health checks; ensure failed retrain doesn't promote bad model
Missing pieces:

Slippage/fee model in execution simulation. Add this before any other change.
Position correlation tracker (realized correlation, not just sector).
Drawdown-conditional sizing (graduated reduction, not just hard limit).
Kill switch on live vs sim P&L divergence (N-sigma over rolling window).

Walk-forward gates
Current: avg Sharpe > 0.8 (swing), > 1.5 (intraday), min fold > -0.3.
Issues: below HPO noise floor (meaningless), Sharpe high-variance in thin data, no significance testing.
Better gates:

Deflated Sharpe ratio > 0 with p < 0.05
Bootstrap CI on Sharpe; lower bound > 0
Sharpe stability: SD of fold-Sharpes < some fraction of mean
Information ratio vs benchmark (long-only SPY, or random entry on same universe)
Calmar (return/MaxDD) > 1.0

Combine swing and intraday into one model?
No. Keep separate. Different horizons, features, microstructure. Unify at the portfolio layer (sizing, correlation, gross exposure), not the signal layer.
Is XGBoost the right algorithm?
Probably yes for this scale. But:
AlternativeWhen winsWhen losesLightGBMFaster on large dataSame family pitfallsCatBoostBetter with categoricalsFew categoricals hereLinear (Lasso, ElasticNet)Interpretability, stable coefficientsLoses non-linearityNeural nets (MLP, TabNet)When >>1M rows + complex interactionsMassive overkill; harder to validateSequence models (LSTM, Transformer)Bar-by-bar trajectory learningDramatic overfitting risk; needs more dataMulti-family ensembleRobust if signals uncorrelatedNeed uncorrelated edges first
Recommendation: Stay XGBoost/LightGBM. Add a Lasso/ElasticNet in parallel as sanity check — if linear disagrees with XGBoost on top features, you may be learning interactions that aren't real.
Position sizing
Major gap not addressed in the briefing. What sizing is used? Fixed fraction? Equal weight? Vol-targeted?
Recommendations:

Volatility targeting: Position size ∝ 1/ATR or 1/realized_vol; equal vol contribution per position.
Kelly fraction (capped): 0.25× full Kelly, conservative.
Confidence weighting: XGBoost probability × edge size — but requires calibrated probabilities (not currently demonstrated).
Drawdown scaling: size = base × (1 − DD/MaxAcceptableDD)².

Don't deploy live without explicit sizing logic. "Fixed percentage" is not a strategy.

E. ROADMAP — TOP 5 PRIORITIES
In impact-to-effort order:
Priority 1: Statistical hygiene foundation (1–2 weeks)
Before any new model work.

Implement deflated Sharpe ratio (Bailey & López de Prado 2014) on all reported results.
Audit total trial count across variants; quantify effective number of independent tests.
Bootstrap each champion's walk-forward 1000× with reshuffled fold splits.
Re-train v142 on PIT-correct fundamentals on the original data window to quantify leakage.
Output: Truthful Sharpe estimate per champion. Almost certainly lower than reported.

Priority 2: Transaction cost model + execution sim (1 week)

Slippage: half-spread + impact based on size/ADTV.
Fees: $0.005/share Alpaca floor + SEC/FINRA.
Re-run all walk-forwards net of costs.
Output: Net Sharpe gates that reflect what the strategy would actually earn.

Priority 3: Combinatorial purged k-fold CV (1 week)

Replace 3-fold expanding with López de Prado's combinatorial purged k-fold (e.g., N=10, k=2).
Implement purge (5 days swing, 1 day intraday) and embargo.
Many backtest paths per model → real distribution of OOS Sharpe.
Output: Distribution of Sharpe per model, not point estimates.

Priority 4: Live-vs-sim P&L reconciliation harness (3–5 days)

For every paper trade, log: model signal, predicted edge, realized P&L, simulated P&L from walk-forward.
Daily report on divergence between sim and live.
Output: The single most important diagnostic in the system.

Priority 5: Feature reduction + correlation audit (3 days)

Cluster features by Spearman correlation; keep one representative per cluster.
Permutation importance on validated CV folds.
Target: ~25–30 features per model, not 84.
Output: Less overfit, faster training, more interpretable.

Things NOT to prioritize until 1–5 are done:

Adding new features
Trying new model families
Backfilling NIS
Building new model variants
Live capital deployment

Highest-marginal-value data sources

Polygon options data (full chain, IV surface) — forward-looking risk pricing.
Tick-level trade data with side classification — for intraday order flow.
High-quality earnings revisions (S&P Capital IQ or equivalent).
CFTC COT data — index futures positioning, macro regime feature.
Real-time news with millisecond timestamps — current NIS is post-hoc.

What would need to be true for a fund to use this?
RequirementCurrent stateGapDeflated Sharpe > 1.0 net of costsUnknownCritical5+ years of OOS paper P&LNone5 yearsCapacity modelNoneBuildRisk decomposition (factor, beta, sector)PartialNeeds portfolio-level analyticsOperational maturity (alerts, kill switches, audit)StrongAlready production-gradeReproducible builds (model versioning, data lineage)PartialAdd data-version pinningInstitutional execution (TWAP/VWAP, smart routing)Alpaca paperSignificant gap for >$1M AUMIndependent validationNoneHard for solo dev
Operational layer is genuinely strong. Signal layer is the gap. Don't conflate them.
Risks the developer may not be aware of

Correlation between swing and intraday models. Overlapping universe + shared features (NIS, fundamentals proxies) means P&Ls may be correlated. Budget caps are not risk diversification if signals are correlated.
Survivorship bias in symbol universe. "S&P 500 + NASDAQ 100 today" is not the same set as 3 years ago.
Overnight gap risk in swing. ATR-based stops are intraday concepts; gap-down through stop is unmodeled.
Earnings concentration. Multiple positions reporting in same window is portfolio-level risk even with per-symbol blackout.
Liquidation cost in stress. ADTV gate at single-position level doesn't model book flatten in one day.
Model staleness during retraining. Hours-old model = signal latency in fast-moving regimes.
API single-point-of-failure. Polygon, Finnhub, FMP, Alpaca — single vendor each. Outage handling?
Backtest contamination from manual iteration. Every re-run after seeing prior results and tweaking is the deepest form of leakage; essentially impossible to undo without resetting the holdout.


F. WHAT'S ACTUALLY GOOD
Preserve and build on:

Decision audit trail with gate categories — genuinely production-grade.
PM re-scoring at execution time — sophisticated pattern, catches stale signals.
22-gate sequential RM with category labeling — structure is correct.
Recognizing fold-3 collapse and not hiding it — intellectual honesty is rare.
Walk-forward gates as hard pass/fail — gate-based promotion discipline is correct.
Versioned model artifacts (v29, v142, etc.) — reproducibility infrastructure sound.
Subprocess retraining decoupled from API server — correct architecture.
PIT-correct fundamentals migration — recognizing and fixing this is exactly right.
NIS sparsity awareness — knowing data isn't ready beats pretending it is.
3-seed ensemble — reduces seed variance; principle right.
ProcessPoolExecutor with worker cap — pragmatic engineering.
HPO frozen after search — given the deeper variance problem, freezing at least prevents drift across retrains.
Strategy budget caps + gross exposure cap — right structure.
Earnings + macro event blackouts — standard practice, correctly implemented at gate layer.
Documentation depth of this briefing — many systems can't be reviewed because they're undocumented.


SUMMARY OF THE CRITIQUE'S CORE THESIS
The MrTrader system has excellent operational engineering and weak statistical foundations. The next phase should be almost entirely on the second, not the first.

Stop testing new model variants until deflated Sharpe and bootstrap CIs are computed on existing variants.
Add transaction costs to walk-forward before any other change.
Reconcile live paper P&L against walk-forward sim P&L — this single diagnostic will tell more than 10 new feature additions.
Accept that "+1.181" and "+1.830" were almost certainly inflated by selection and data-pipeline bugs, and that the true edge of the current system may be near zero.
That last point is not a failure — it's the normal state of systematic strategy R&D. Most strategies don't have edge. Knowing that early is the most valuable output.

The honest path to "production-grade hedge fund standard" runs through less — fewer features, fewer variants, fewer optimistic Sharpes — and through more rigor — deflated stats, embargoed CV, cost models, live reconciliation. The current trajectory of "more features, more versions, more retrains" is moving away from that goal.
Final framing: Engineering quality is 7/10. Statistical rigor is 3/10. The gap between these two scores is the work plan.

INSTRUCTIONS FOR DOWNSTREAM LLM REVIEWERS
If you are an LLM (GPT, Gemini, etc.) cross-reviewing this critique, please specifically address:

Do you agree the Sharpe collapse is more selection bias than regime? What evidence would change your view?
Is HPO variance of 2.0 Sharpe a death sentence for the system, or recoverable? If recoverable, how?
What did this review miss? Specific items, not generalities.
Is the priority ordering (stats hygiene → costs → CV → live-vs-sim → feature reduction) correct? What would you reorder?
What specific statistical tests should run on existing data before any new modeling? Be concrete (test name, formula, threshold).
For the 2025 regime collapse: is this a "the strategy doesn't have edge here, sit out" story or a "build a better model" story? Defend your answer.
What is the realistic Sharpe ceiling for a retail-capital ML-equity strategy with this data and feature universe? Numerical estimate with reasoning.
What is missing from the operational/risk side that this review treated as adequate?

Disagreement is welcome. Flag specific claims you'd push back on.