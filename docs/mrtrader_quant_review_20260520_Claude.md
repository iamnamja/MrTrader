# MrTrader — Independent Quant Review
**Reviewer perspective:** Buy-side quant lead, 15+ years L/S equity, multi-manager platform background
**Date:** 2026-05-20
**Mandate:** Brutally honest. Push back on assumptions. Cross-methodology critique.

---

## 0. EXECUTIVE VERDICT (read this first)

**The architecture is competent. The strategy is not yet defensible. Promote nothing live until the items below are fixed.**

The system shows above-average engineering discipline — direction-aware audits, PIT-safe data, immutable order ledger, startup reconciliation, DB-backed config. That's the top 10% of retail-built bots. **That's not the bar.**

The bar is whether this system has identifiable, decomposable, repeatable alpha that survives realistic costs. On the evidence presented, **it does not yet**, and several elements of the current evaluation framework are actively misleading you about that.

### Top 5 things I would do this week (in order):

1. **STOP all new strategy work. Fix walk-forward P0.1 and P0.2 bugs first.** Every Sharpe number in your head right now is contaminated. You cannot make a single strategic decision until WF is honest.
2. **Raise the live promotion Sharpe bar from 0.50 to ≥1.0.** A 0.5 Sharpe L/S system is a slightly more elaborate way of holding SPY. You will lose money to costs.
3. **Run alpha decomposition.** PEAD-only, QualityShort-only, combined, and ablate every multiplier (NIS, regime, vol-targeting). I will bet a steak dinner that 1–2 of the multipliers are negative-alpha.
4. **Decide whether PEAD+QualityShort is actually two strategies or one.** They are almost certainly correlated. If correlated, you are over-allocated to the same factor under two names.
5. **Reduce scope before adding complexity.** $20k can't carry 7 positions × 2 strategies × overlay stack. Strip it to one strategy, 3–5 positions, then earn the right to expand.

---

## 1. WHAT YOU GOT RIGHT (so the criticism lands fairly)

| Item | Why it matters |
|---|---|
| PIT-safe earnings features (`get_earnings_features_at`) | 80% of retail bots fail here silently |
| Immutable Order ledger + `recompute_partial_pnl` | Most retail bots can't survive a restart with open partials; this can |
| Direction-aware sector/correlation/heat math after the audit | Many "L/S" systems silently double-count shorts |
| ProposalLog UUID lineage (Proposal → Trade → Order) | This is institutional-grade traceability |
| DB-backed config with TTL cache | Right pattern for live tuning |
| Startup reconciler with ghost cleanup | Production-grade thinking |
| Compliance (PDT, T+1, wash sale, halt list) | Most people forget these until they get burned |

Keep all of this. The criticism that follows is about the layer above — the **strategy and evaluation framework**, not the plumbing.

---

## 2. CRITICAL ISSUES (these invalidate current evidence)

### 2.1 Walk-forward bugs P0.1 / P0.2 — these are not "P0" they are showstoppers

You explicitly documented:
- Entry simulation uses `prev_close × 1.001` instead of next-session open
- Stop-out check uses daily close, not intraday low

**Translation:** Every backtest number you have is wrong in a *systematically optimistic direction*.

- Entry bias: you're modeling entry at near-perfect prices; live execution will be 20–80 bps worse on average for swings, multiples of that on intraday.
- Stop-out bias: intraday low-touch is where stops fire. Using daily close vastly undercounts losers. Real Sharpe is materially lower than reported.

**My estimate:** True WF Sharpe is **20–50% below** what your current numbers say. Combined with the ~15% survivorship bias you've already identified, your real out-of-sample expectancy may be **half** of what your dashboards report.

You have framed these as "P0 bugs in backlog." They are not bugs — **they are the bugs that make every other decision you make wrong.** Fix them this week. Re-run everything. Then come back to strategy.

### 2.2 No deflated Sharpe / multiple-testing correction

You've stated:
- 9 failed LambdaRank runs
- 4 active swing selector variants in code (`pead_quality_short`, `pead`, `quality_short`, `factor_portfolio`)
- Default XGBoost selector also exists
- Multiple sizing overlays (regime, vol, NIS, confidence) each independently tunable

That is **at least 13 strategy trials** plus an unknown number of hyperparameter trials per ML model. The probability that *at least one* of these shows a spuriously high Sharpe by chance is enormous.

**Required:** Implement deflated Sharpe ratio (Bailey & López de Prado, 2014). Apply it to whichever variant you promote. If deflated Sharpe < 0, the variant is overfitted regardless of raw Sharpe.

### 2.3 Live promotion criteria are too soft

| Criterion | Current | My Recommendation | Why |
|---|---|---|---|
| Sharpe | ≥ 0.50 | ≥ 1.0 (net of all costs) | 0.5 = SPY unmanaged; you're paying complexity for no edge |
| Max DD | ≤ 15% | ≤ 8% (paper); ≤ 12% (live) | 15% on $20k = $3k drawdown; psychologically untradeable |
| Sortino | not measured | ≥ 1.5 | Sharpe penalizes upside vol; Sortino doesn't |
| Information Ratio vs SPY | not measured | ≥ 0.6 | Proves you have α, not just β |
| Calmar (CAGR / MaxDD) | not measured | ≥ 0.5 | Catches strategies that work until they don't |
| Single-strategy P&L share | not capped | < 50% of total | Forces diversification check |
| Live-sim slippage shortfall | < 30 bps/day median | < 20 bps/day **p75** | Median hides tail; p75 is the honest measure |
| Minimum trade count | not specified | ≥ 60 closed trades | Below this, t-stat on returns is meaningless |

A 3-month paper trade with Sharpe 0.5 and 25 trades tells you almost nothing statistically. You need either longer paper trading or higher trade frequency.

---

## 3. STRATEGIC CONCERNS

### 3.1 PEAD + QualityShort: this is likely one strategy wearing two hats

You describe them as "complementary." I think they are **correlated**.

- PEAD short fires on: EPS miss > 5%
- QualityShort fires on: declining margins, declining revenue, high PE

**Companies that miss earnings by >5% almost always have deteriorating fundamentals.** The Venn diagram overlap is large. You're not diversifying signal sources — you're concentrating on "broken fundamentals" with two different triggers.

**Tests to run:**
1. Compute Pearson correlation of daily signal scores between PEAD and QualityShort across the WF universe. If > 0.4, they are not independent.
2. Compute hit-rate overlap: of all QualityShort firings, what % also had PEAD short within ±10 days? If > 25%, you're double-trading the same thesis.
3. Run each strategy in isolation. Compare ICs. If they're additive vs. redundant will be obvious.

**If correlated:** Drop one (pick the one with cleaner microstructure — probably PEAD because it's event-driven and time-bounded) and replace the slot with something *actually* uncorrelated — e.g., cross-sectional mean-reversion, pairs trading, or volatility carry.

### 3.2 The 40% net long target has no documented thesis

Why 40%? Not 0% (market-neutral), not 100% (long-biased)?

If you ask the question this way, you'll typically find one of three honest answers:
- **(a)** "We back-tested several net exposures and 40% had the best Sharpe." → Overfit to one regime. Don't anchor on it.
- **(b)** "It's a compromise between long-biased equity premium and short hedging benefit." → Defensible but should be derived from a Black-Litterman or Markowitz-style optimization, not chosen.
- **(c)** "Honestly, it felt right." → That's the answer 90% of the time, and it means you have a free parameter undefended.

**Recommendation:** Either derive net exposure from a model (e.g., target portfolio beta ≈ 0.3 → solve for net long given current factor exposures), or **start at 0% net (market-neutral) and only deviate when you have explicit conviction**. Net long is a directional bet; size it deliberately.

### 3.3 $20k capital is structurally suboptimal for this design

Hard math:
- 7 positions × $2,857 average = bare minimum for diversification
- Per-position 2% risk = $400 stop budget = ~14% adverse move tolerance
- PDT zone ($25k threshold) is right there — one bad week and intraday becomes impossible
- Short borrow on the small-caps where shorting alpha actually lives is 5–15% annualized, not 0.5% — your sim is wildly understating costs for any signal that points at high-short-interest names
- 100 round-trip trades/year × $5 all-in cost (commission + spread + impact) = $500 = 2.5% drag

For $20k you have three structurally better choices:
1. **Concentrate.** 3–5 positions max, higher conviction per name, no L/S complexity until $50k+.
2. **Options-based defined-risk.** A $20k account running PEAD via short verticals or iron condors has 5–10× the capital efficiency of direct equity L/S, plus you collect the post-earnings IV crush which is a real edge separate from PEAD direction.
3. **Single-strategy paper-to-live at $20k, then scale.** Pick the cleanest alpha source (probably PEAD long-only on positive surprises with a hard 5-day exit), prove it works, *then* layer.

The current design is built for a $250k–$2M account. You're operating at 1/10th of that. Capital efficiency is going to dominate strategy edge.

---

## 4. ARCHITECTURAL CONCERNS

### 4.1 The PM/RM/Trader agent split is overengineered for the problem

I get the appeal — it mirrors a real desk and it's clean SAFe-style separation. But for a single-process bot running 7 positions:

- Redis adds latency you don't need and a failure mode (queue death, ordering glitches) you'll spend time debugging
- The async boundary between PM and RM means rejection feedback loops are slow — PM doesn't know in real time why proposals are dying, so it keeps proposing
- You can't easily reason about "what was the system state when this trade was approved?" across three agents and two queues without elaborate tracing

**Counter-argument:** the agent split *does* prepare you for follow-the-sun / multi-strategy / multi-manager scale, which aligns with your day-job thinking. If that's the long-term vision, keep it. But know that you're paying a complexity tax now for an option you may not exercise.

**Mitigation if you keep it:** add a single end-to-end trace ID flowing through every log line, every Redis message, every DB write. You have UUID per proposal — push it further so a single grep on the UUID reconstructs the entire trade lifecycle including all multiplier values applied.

### 4.2 Multiplier stacking is a silent risk

A proposal's final size is the product of:
- `confidence_scalar` (0.5×–2.0×)
- `regime_sizing_mult` (0.3×–1.0×)
- `vol_targeting_mult` (continuous)
- NIS `sizing_multiplier` (0.3×–1.5×)
- Possibly `macro_sizing_factor` separately
- Eventually clipped by `MAX_POSITION_SIZE_PCT`

That's **5 multiplicative stages**, several of which are themselves derived from ML/LLM outputs. The product distribution is wide. Even if each multiplier has *zero* alpha individually, their compounded variance increases your sizing noise without adding signal.

**Action:** For each multiplier, run an ablation — replace it with `1.0` constant in WF and see whether Sharpe goes up or down. My prior: 1–2 of these are negative-alpha overlays. The honest answer is probably "regime gate and vol-targeting help; the others are noise or worse."

### 4.3 11-rule sequential fail-fast risk validation has a regime problem

In benign markets the fail-fast structure is fine. In stressed markets you get **rejection cascades** — most proposals fail at gate 4 (daily loss), and you accumulate stale unfilled proposals while your edge bleeds away. You also can't tell whether you would have approved a proposal "but for" gate N — useful information.

**Better pattern:** Two-tier validation:
- Hard gates (PDT, halt, buying power, regulatory) → binary reject
- Soft gates (sector concentration, correlation, beta) → continuous penalty that reduces size rather than rejecting

This also gives you a continuous "risk budget utilization" metric to monitor, instead of binary "rejected for reason X."

### 4.4 Partial exit + breakeven stop is retail folklore, not quant practice

The 50% at 1×ATR + breakeven move is a popular retail pattern. It feels good (lock in profit, can't lose). It's **mathematically inferior** to either:
- Trailing stop only (let winners run, no early scaling)
- Volatility-based pyramid-add (add to winners as they prove the thesis)

Reason: cutting position size at the moment your thesis is *most validated* (you're up 1×ATR) reduces compounding on the biggest winners, where strategy P&L lives. The "breakeven stop" is functionally a tighter trailing stop, which will get whipsawed out at higher cost than a wider ATR-based trail.

**Test:** Run WF with three exit configurations:
1. Current: 50% partial + breakeven
2. Trailing stop only (1.5× ATR from highest/lowest)
3. Pyramid add (add 25% size at +1×ATR, full trail thereafter)

I expect #2 or #3 to dominate #1 by 10–30% on net Sharpe in the long leg.

---

## 5. NIS (News Intelligence) — SPECIFIC CRITIQUES

### 5.1 LLM-based news scoring is not a stable signal source

You are using Claude Haiku to score news. This is fast and cheap but:
- LLM scoring is **not deterministic** (even at temp=0, different prompts/contexts give different scores)
- It has **no statistical track record** — there's no Sharpe attribution for NIS-influenced trades vs. NIS-ignored trades
- It is **un-backtestable** because Haiku 2024 ≠ Haiku 2026 — your historical NIS scores are conditional on the model version at time of scoring

**Replace or supplement with:**
- **FinBERT** (or domain-tuned BERT) sentiment — deterministic, backtestable, well-studied
- **News volume z-score** — simple, robust, surprisingly predictive
- **Trigger keyword count** (Loughran-McDonald financial dictionary) — boring but works
- **News source diversity** (single-source vs. multi-source coverage) — proxies for materiality

Use Haiku for **rationale generation** (human-readable explanation) but not for the **score** that drives sizing.

### 5.2 NIS direction-blindness on shorts is real and material

You correctly identified this. The fix is straightforward:

```
Effective_sentiment_for_proposal = direction_score × (+1 if BUY else -1)
```

If `effective_sentiment < threshold` → block.
If `effective_sentiment > threshold` → confirming signal → consider *upsizing*.

But the deeper issue is **NIS may not add alpha at all**. Before fixing the direction logic, *prove NIS is positive-alpha*. If ablation shows it's neutral or negative, fix the direction logic in case you later turn it back on, but disable it in production.

### 5.3 The post-event refresh logic (3–8 min after release) is a microstructure trap

You re-fetch Tier 1 macro context 3 minutes after each event release. That's the worst possible window — initial reactions are noise, real direction emerges over 15–60 minutes for most data releases. A 3-min snapshot is more likely to capture a spike than a signal.

**Better:**
- Block all new entries for 15 minutes post-release (you may already do this via `block_new_entries`)
- Re-fetch at T+30 minutes for actual sentiment
- Compare T+0 expected vs T+30 actual market reaction → use the *surprise residual*, not the headline

---

## 6. CROSS-METHODOLOGY IDEAS — things you're not doing that you should consider

These are framed as "would I run this myself" not "Min should definitely do this." Pick what fits your time budget.

### 6.1 Options overlay (high-value for $20k accounts)

PEAD is one of the highest-IV-crush events on the calendar. The volatility risk premium around earnings is **larger and more consistent** than the directional drift you're trading.

- **Iron condor** on PEAD names with elevated IV → collect premium, defined risk, capital-efficient
- **Short put vertical** on positive-surprise PEAD names (synthetic long with downside cap)
- **Short call vertical** on negative-surprise PEAD names (synthetic short, no borrow cost, no PDT exposure)

For $20k, options-structured PEAD is probably 3–5× more capital-efficient than your current equity L/S, *and* you sidestep PDT and borrow cost entirely.

### 6.2 Pairs trading / statistical arbitrage

Genuinely uncorrelated with PEAD. Pick cointegrated S&P 100 pairs (e.g., XOM/CVX, KO/PEP, MA/V), trade z-score reversion. Low capital intensity, market-neutral, well-documented edge. This is what I'd add as a second sleeve before doubling down on fundamentals-driven shorts.

### 6.3 Cross-asset features (free Sharpe)

You're equity-only. The following inputs have well-documented predictive power for equity factor returns and are free:
- 2y/10y Treasury slope (regime indicator)
- HY-IG credit spread (risk appetite)
- DXY (FX regime)
- VIX term structure (contango/backwardation)
- Gold/Silver ratio
- WTI/Brent ratio

Add 3–5 of these as features to your XGBoost. Expected lift: small but real, and you're not paying for it.

### 6.4 Meta-labeling (López de Prado)

Instead of: "predict whether stock goes up tomorrow" → "given that strategy X said BUY, predict whether this specific trade will be profitable."

Meta-model takes as input: primary signal, regime, microstructure, news, fundamentals. Output: probability the *primary signal* will succeed for *this specific trade*. Only trade above some threshold.

This is exactly what your `confidence` and `min_confidence` *should* be doing but probably isn't. A proper meta-model on top of PEAD typically improves hit-rate by 5–15% and Sharpe by 20–40%.

### 6.5 Earnings call transcript embeddings

Free data via Seeking Alpha / company IR. Cheap-to-run via OpenAI/Anthropic embeddings (or local SBERT). Predictive features:
- Management tone shift quarter-over-quarter (Δ embedding similarity)
- Q&A evasion patterns (Loughran-McDonald uncertainty markers spike)
- Forward guidance language changes

Earnings call sentiment is **far more predictive** than headline news sentiment. This is a real edge available to anyone who builds the pipeline.

### 6.6 Insider transaction features (Form 4)

Free from SEC EDGAR. Cluster buying by multiple insiders within 30 days is one of the strongest documented anomalies in academic literature (Cohen, Malloy, Pomorski 2012). Easy add-on signal for the long side.

### 6.7 Short interest changes (vs. levels)

Short interest level is noise; *Δ short interest over 30 days* is signal. Sharp increases in SI flag deteriorating sentiment that often precedes price weakness. Combine with your QualityShort signal.

### 6.8 Bayesian / Kelly sizing instead of linear confidence scaling

Your `confidence_scalar` is linear in confidence. That's wrong. The relationship between predicted probability and optimal bet size is **fractional Kelly** — concave and bounded. Linear over-sizes high-confidence trades and under-sizes medium-confidence ones.

Replace with: `bet_size = clip(0.5 × Kelly_optimal, 0, max_position_pct)` where Kelly is computed from estimated edge and odds. The 0.5 fraction is the standard "half-Kelly" haircut for parameter uncertainty.

### 6.9 Regime model upgrade

Your current regime gate is `SPY > SMA200 AND VIX < 30`. Two binary indicators. That's a 4-state machine, three of which suppress entries entirely.

Better: Hidden Markov Model on (SPY return, VIX, term structure, credit spread). Outputs continuous regime probabilities. Use these to *tilt* factor exposures (e.g., momentum-tilted in trend regime, quality-tilted in stress regime) rather than to gate entries on/off.

This is one of the highest-Sharpe single improvements you can make.

### 6.10 Bracket OCO orders at the broker

Right now your stops are client-side polled. If your bot crashes, you have unstopped positions. Move stops to Alpaca-native bracket orders (OCO entry + stop + target). Latency goes from "next tick" to "Alpaca matching engine." Resilience goes from "depends on your VPS uptime" to "broker-enforced."

---

## 7. PRIORITIZED ACTION PLAN

### This week (no new strategy work allowed)
1. Fix P0.1: next-session open entry simulation
2. Fix P0.2: intraday low/high for stop-out check
3. Re-run all walk-forward backtests
4. Compute deflated Sharpe ratio on each variant
5. Compute factor IC (your existing backlog item P0.3)
6. Run PEAD-only vs QualityShort-only correlation analysis

### Next 2 weeks (decisions, not code)
7. Decide whether `pead_quality_short` is one strategy or two (based on correlation analysis)
8. Ablate every multiplier (NIS, regime, vol-targeting, confidence-scalar) and keep only the ones that add WF Sharpe
9. Decide net exposure target with explicit rationale (or move to market-neutral as default)
10. Raise live promotion criteria per Section 2.3

### Next month
11. Replace LLM news scoring with FinBERT + volume + source features for the *score*; keep Haiku for *rationale*
12. Fix NIS direction-blindness for shorts
13. Replace 50% partial + breakeven with ATR-based trailing stop; A/B test
14. Migrate stops to Alpaca-native bracket OCO orders
15. Add cross-asset features (2y/10y, HY-IG, DXY, VIX term structure, gold)

### Next quarter (only after the above)
16. Pilot options overlay for PEAD names (paper, separate sleeve)
17. Add pairs trading sleeve as an uncorrelated second strategy
18. Build meta-labeling model on top of primary signals
19. Add insider buying + Δshort interest features
20. Upgrade regime detection to HMM

---

## 8. QUESTIONS YOU SHOULD ANSWER (to yourself, honestly)

1. **What is the *one* edge this system has** that you can explain in one sentence and defend with data? If the answer is "the combination of PEAD + Quality + NIS + regime + vol-targeting," you don't have an edge yet — you have a stack of overlays whose individual contributions are unmeasured.

2. **What is the realistic worst-case 12-month outcome** if you deploy this live tomorrow with $20k? If the answer involves "drawdowns might be uncomfortable but I'd still believe in the system," good. If it involves "I don't really know," do not deploy.

3. **What would convince you the system doesn't work?** Pre-register failure criteria *now*. If you can't articulate them, you'll rationalize losses forever (this is the single most common reason retail quant systems die slowly).

4. **Are you trading because it's profitable, or because it's interesting?** Both are valid — but they imply different acceptable Sharpe levels, different time investment, and different success metrics. The system you've designed is consistent with "interesting." That's fine if you own it. Don't promote it to live with "profitable" expectations unless the numbers actually clear that bar.

5. **What's the opportunity cost of one more month on MrTrader vs. the muni platform or the home purchase?** I'd argue this in your shoes: MrTrader is leverage on AI-agent skills that matter at MarketAxess. As R&D it's worth significant time. As a profit center on $20k, it's marginal. Frame it accordingly and don't let "but I'm so close to live" pressure you into deploying an under-tested system.

---

## 9. WHAT I WOULD STEAL FROM YOUR DESIGN

Genuine credit — most retail quant systems don't have these:
- ProposalLog UUID lineage end-to-end
- `recompute_partial_pnl` from immutable Order ledger
- Direction-aware audit campaign (this is rare even at small hedge funds)
- DB-backed live-editable config with TTL
- Startup reconciliation with ghost cleanup
- 11-point validation as a *concept* (even if I'd restructure it)

If you build the same plumbing discipline into a smaller, more focused strategy, you'll have a real system.

---

## 10. ONE-LINE SUMMARY

**You have built a beautiful engine. You have not yet proven the car can drive. Fix the speedometer (P0 bugs), pick one race (one strategy), then race it.**

---

*End of review. Feed this verbatim to the other LLMs. The interesting test will be whether they reach the same conclusion on (a) PEAD/QualityShort correlation, (b) the Sharpe target, and (c) the multiplier stack. If two of three independent reviews flag the same items, treat them as confirmed.*
