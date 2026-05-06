# MrTrader — Multi-LLM Review Synthesis

**Date:** 2026-05-05  
**Reviewers:** Claude (Anthropic), ChatGPT-4o (OpenAI), Gemini (Google)  
**Source prompt:** `docs/llm_review_prompt.md`  
**Decision output:** `docs/MASTER_BACKLOG.md`

This document preserves the key insights from all three independent reviews, synthesized against the author's knowledge of the actual codebase. Organized by confidence level, not by source.

---

## Tier 1 — High Confidence (All Three Agree)

These conclusions are almost certainly correct. Act on them immediately.

### Walk-forward measures the model, not the strategy
The tier-3 walk-forward runs ML model signal in isolation. It does not simulate:
- PM opportunity score (would have suppressed many trades in Apr–Oct 2025)
- RM gates (22 checks: position caps, earnings blackouts, budget limits)
- Transaction costs (5–20bps round trip depending on strategy)
- Position caps and portfolio-level constraints

Model Sharpe ≠ Strategy Sharpe. The promotion gate currently passes models on a metric that doesn't reflect what actually trades.

**Fix:** Wire PM opportunity score, earnings blackouts, position caps, and cost model into walk-forward fold simulation. Phase 2 of MASTER_BACKLOG.

### NIS time-leak is destroying model validity (Gemini — sharpest insight)
NIS stock-level data is only available from May 2025. With ~80% NaN in training rows, XGBoost doesn't learn sentiment quality — it learns:
- `NaN = 2021–2024 (low-vol bull) regime`
- `non-NaN = 2025 (tariff/vol) regime`

We accidentally encoded *time* into the model. This is a subtle form of look-ahead bias. It may explain why v41 (with NIS) performed slightly better on fold 2 (Apr–Oct 2025) than v29 — the model learned "non-NaN = treat this differently."

**Fix:** Remove NIS features from training entirely until full backfill complete. Use NIS as a PM gate overlay only. Phase 1c.

### No transaction cost model means no valid Sharpe for intraday
For intraday on 5-min bars with 0.8×ATR targets:
- ATR on a $100 stock ≈ $1.00 → target = $0.80
- Round-trip cost estimate: 15bps × $100 = $0.15
- Cost = 19% of the gross target

On a 40% win rate strategy, this cost meaningfully erodes edge. We don't know if net Sharpe is positive.

**Fix:** Add cost model to walk-forward. Phase 1a.

### RSI_DIP/EMA_CROSSOVER pre-filters cap the alpha ceiling
The ML model only scores symbols that pass rule-based pre-filters. It learns "which RSI dip candidates don't fail" — not "what makes a good swing trade." The alpha ceiling is bounded by what the rules can pre-select.

Remove the pre-filters. Let XGBoost score the full universe. Let the model discover the entry pattern. This requires redesigning the label simultaneously (triple-barrier, applied uniformly to full universe). Phase 3b.

### cs_normalize eliminates all market-wide context from intraday model
Z-score normalizing features cross-sectionally per day forces any feature constant across symbols to zero. VIX, SPY level, SPY trend, market breadth, macro state, day of week — all zero after normalization.

The model identifies the best stock *relative to peers* but cannot know if today is a +3% melt-up or -4% crash. This is a fundamental design flaw.

**Fix:** Split features into Branch A (cross-sectional, z-scored) and Branch B (global, NOT normalized). Both branches concatenate into feature matrix. Phase 3a.

### Dynamic position sizing is missing
Fixed percentage per trade ignores: signal confidence, regime volatility, portfolio context. In high-VIX regimes, the same fixed size is 2–3× more risky in dollar-volatility terms.

**Fix:** Volatility-targeted sizing: position size = `target_vol_contribution / predicted_daily_vol` × regime_multiplier × confidence_multiplier. Phase 3d.

### Alpaca must be the source of truth for live capital
DB/Alpaca state divergence has already caused production bugs (bars_held reset, ZS stuck open post-restart). For live capital, state mismatches can cause double-entry, missed exits, or incorrect risk calculations. Phase 6a (Phase 100).

---

## Tier 2 — High Confidence (Two of Three Agree)

### Selection bias inflates the reported Sharpe numbers (Claude's strongest point)
~15 model variants tested. Picking the best Sharpe by walk-forward and reporting that Sharpe is textbook selection bias. The Deflated Sharpe Ratio (Bailey & López de Prado 2014) corrects for number of trials.

Estimated impact: the +1.181 (swing) and +1.830 (intraday) gate-passing Sharpes are likely 30–50% inflated by selection across variants. The true out-of-sample expectation is lower.

v142 was also trained when fundamentals were NOT PIT-correct. The original +1.181 has look-ahead bias. The current +0.310 (on PIT-correct data, current window) is closer to the truth.

**Fix:** Implement DSR on all walk-forward outputs. New gate includes DSR > 0 with p < 0.05. Phase 1e.

### The Sharpe collapse is mostly statistical artifact, partly regime
The conventional explanation for v142 (+1.181 → +0.310) and v29 (+1.830 → -0.327) is the Apr–Oct 2025 tariff/vol regime. The more accurate decomposition:
- ~60%: Selection bias (these were the best of 15 variants — regression to mean as more data arrives)
- ~20%: PIT leakage (v142 trained with non-PIT-correct fundamentals)
- ~15%: Genuine regime non-stationarity (tariff shock is real)
- ~5%: Data sparsity (limited crisis regimes in training history)

This doesn't mean the regime problem isn't real — it is. But it means adding more features is unlikely to fix it, because the problem is partly methodological, not purely signal quality.

### Three folds with no purge/embargo are not enough
The current walk-forward has structural flaws:
- 3 expanding folds → too few independent test windows for statistical confidence
- No purge between train/test → labels computed on boundary rows can span both sides
- No embargo → the test window immediately follows training, no buffer
- HPO frozen params overfit the original data window, not each fold's window

**Fix:** Add purge (10-day swing, 2-day intraday) and embargo (5-day). Eventually: combinatorial purged k-fold (López de Prado) for distribution of Sharpes rather than point estimates. Phase 1b.

### Cross-sectional dispersion is the missing key intraday input
On macro-dominated days (high correlation, low dispersion), the cross-sectional ranking model has no valid premise. The "top 20%" are randomly determined by noise. The Apr–Oct 2025 collapse is largely a dispersion problem: during that period, stocks moved together, not independently.

**Fix:** (1) Dispersion gate: if daily cs_dispersion < 0.5×median → skip all intraday entries. (2) Dispersion as Branch B feature so model learns the relationship. Phase 2c + 3a.

---

## Tier 3 — Medium Confidence (One Source, Context-Dependent)

### Label design mismatch (all three, independently)

**Swing:** Binary `path_quality` (target hit in 5 days) is too narrow for a 3–15 day swing system. Rewards fast pops, discards slow-developing trades, doesn't match live exit policy.
- Better: Triple-barrier multiclass (+1/0/-1 = upper/time/lower barrier hit first) on 10-day window
- Or: Meta-label (given that RSI_DIP fired, should this trade be taken?) as transitional step while pre-filter removal is in progress

**Intraday:** Top-20% cross-sectional label is sound in principle but dangerous for long-only execution on flat days. Forcing 20% positive rate when no stock has a clean 2h return trains on noise.
- Better: Top-20% cross-sectionally AND 2h return > estimated cost buffer
- Days where no symbol clears the hurdle: exclude from training

**What was tried:** Realized-R label failed (AUC ~0.51). The failure may be implementation-specific (the simulation of actual stop/target/time exits wasn't realistic). Worth one more attempt after simulation is fixed.

### Feature redundancy is high (Claude + ChatGPT)
Many features measure similar underlying phenomena. Redundancy doesn't add information — it adds overfitting risk and noise. Target: 35–40 swing features (from ~79), 25–30 intraday (from ~56 after NIS removal).

Pruning candidates:
- Multiple RSI variants (rsi_7, stochrsi_k, stochrsi_d, williams_r_14 vs rsi_14): keep rsi_14 only
- Multiple momentum horizons at similar windows: collapse to 5d, 60d, 252d
- WorldQuant alphas (13 in current model): run permutation importance, keep top 5
- options_* with sparse coverage: if <80% symbol coverage, model learns "is options data available" not the signal

### Academic factors missing from swing model (Claude)
Well-established in literature, not in current feature set:
- **Gross profitability / assets** (Novy-Marx 2013): robust quality factor
- **Investment / asset growth** (Fama-French 5-factor): known negative predictor
- **Post-earnings announcement drift** (Bernard-Thomas 1989): most robust earnings anomaly
- **Idiosyncratic volatility** (Ang et al. 2006): low-IVOL anomaly
- **Dispersion of analyst forecasts** (Diether et al. 2002): high dispersion → lower returns
- **Accruals** (Sloan 1996): earnings quality signal
- **Short interest / days-to-cover**: crowding + squeeze risk
- **VIX term structure** (VIX9D/VIX3M ratio): forward stress vs. complacency

### Intraday features missing (ChatGPT + Claude)
Given cs_normalize constraint, focus on cross-sectional features that vary per-symbol:
- **RVOL by time-of-day**: not just volume surge vs. rolling avg, but vs. expected volume at that minute-of-day
- **Intraday sector-relative return at bar 12**: stock return since open minus sector ETF return since open
- **Bid-ask spread proxy**: (high - low) / close on 5-min bar as liquidity signal
- **Short-term reversal at open**: 5-min reversal pattern (Heston/Sadka effect)
- **Intraday breadth at bar 12**: % of universe positive since open (Branch B global feature)

---

## Tier 4 — Valid But Lower Priority

These are real issues but less urgent than Phases 1–3.

### Survivorship bias in symbol universe
"S&P 500 + NASDAQ 100 today" is not the same set as 3 years ago. Companies dropped from the index typically declined significantly before removal — including them in historical training would lower apparent win rate. The bias inflates results.

Assessment: For large-cap universe, this bias is real but smaller than for small-caps. Survivorship-biased results are still directionally informative. Fix eventually via Polygon's historical constituent data; not an immediate blocker.

### Hidden common factor exposure
Swing and intraday models share universe (top 430 symbols), share some features (fundamentals, NIS), and both tend to be long. Budget caps don't prevent correlated P&L if the signals themselves are correlated.

Assessment: Important for live risk management but less critical while paper trading. Address in Phase 5 regime diagnostics.

### Overnight gap risk in swing
ATR-based stops assume intraday execution. A gap-down through the stop price results in a worse fill than modeled. This is unmodeled execution risk.

Assessment: Add gap-through-stop scenario to walk-forward simulation. Lower priority but real.

### HPO variance floor
The 2.0 Sharpe HPO variance problem means frozen params selected on one data window may not transfer. The 3-seed ensemble reduces but doesn't eliminate seed variance.

Assessment: The combinatorial purged k-fold (Phase 1b extended) and the deflated Sharpe (Phase 1e) address this more fundamentally than re-running HPO. Don't re-run HPO until those are implemented.

---

## What's Genuinely Good — Don't Change

All three reviewers independently flagged these as sound:

| Component | Assessment |
|---|---|
| PM→RM→Trader architecture | Production-grade separation of concerns. Keep. |
| Decision audit trail with gate categories | Genuinely institutional-grade. Extend, don't replace. |
| PM re-score at execution time | Smart pattern; catches stale signals in queue. Keep. |
| Walk-forward as hard pass/fail gate | Correct discipline. Improve the metrics, not the concept. |
| PIT-correct fundamentals migration | Exactly the right direction. Continue. |
| 3-seed ensemble | Reduces seed variance. Correct principle. Keep. |
| Subprocess retraining decoupled from API | Correct architecture. Keep. |
| Versioned model artifacts | Reproducibility infrastructure is sound. Keep. |
| Earnings + macro event blackouts | Standard practice, correctly implemented. Keep. |
| Opportunity score (graduated, not binary) | Exactly how discretionary macro overlays work in systematic funds. Keep and improve. |
| NIS sparsity awareness | Knowing the data isn't ready beats pretending it is. Right call. |
| ProcessPoolExecutor with worker cap | Pragmatic engineering. Keep. |

---

## The One-Sentence Summary

> MrTrader has production-grade operational engineering and a statistical foundation that cannot yet prove it has a real edge — fix the measurement before fixing the model.

---

## Appendix: Specific Statistical Tests to Run on Existing Data

Before any new model work, run these tests on v142 and v29:

1. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): `DSR = Φ((SR - SR*) × √(T-1) / √(1-γ₃×SR + (γ₄-1)/4×SR²))` where SR* = E[max Sharpe across N trials]
2. **Bootstrap fold resampling**: Perturb fold boundaries ±30 days, 200 iterations. If original Sharpe is in top 10% of distribution → selection bias confirmed.
3. **Regime-stratified Sharpe**: Report Sharpe by VIX bucket, SPY trend, breadth. Know the tradable conditions.
4. **Cost-adjusted Sharpe**: Deduct transaction costs from each simulated trade. Report net Sharpe.
5. **Opportunity-score-filtered Sharpe**: Re-run simulation applying opportunity score as a trade filter. Report improvement vs unfiltered.
6. **Feature permutation importance on worst-fold only**: Features that improve worst-fold performance matter; features that improve average Sharpe at cost of worst-fold don't.
