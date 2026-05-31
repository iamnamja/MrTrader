# MrTrader: Synthesis of Four LLM Reviews — Phased Action Plan
*Opus 4.7 senior quant synthesis, 2026-05-20. Supersedes MASTER_BACKLOG strategic prioritization for the next 6 months.*

---

## 0. TL;DR Verdict

MrTrader is a **well-engineered software harness wrapped around a research process that does not yet meet institutional standards of evidence.** All four reviewers agree on this.

The engineering quality is materially better than typical retail systems. The research quality — what we actually believe about edge, and why — is materially worse than the engineering would suggest.

**The single most important fact:** None of the published walk-forward Sharpes are currently trustworthy. P0.1 (entry at `prev_close×1.001` instead of next-session open) and P0.2 (stops on daily close, not intraday extremes) are not minor bugs — they systematically inflate winners and suppress losers most aggressively on PEAD, because PEAD's alpha lives in the overnight gap that P0.1 silently captures for free.

Plausible Sharpe deflation combining both bugs + 15% survivorship + deflation from ≥13 selector variants: **0.4×–0.6× of reported numbers.** The swing model reporting WF Sharpe near 0.8 is, with high probability, sub-0.5 in reality.

**Nothing — not a model retrain, not a new feature, not a new sleeve — should be promoted to paper or live until P0.1/P0.2/P0.3 and survivorship-labeled WF reruns are complete.**

---

## 1. Where All Four Reviewers Converge (Near-Certain Issues)

### 1.1 Walk-forward integrity is broken (P0.1, P0.2) — SHOWSTOPPER
All four flag this with similar magnitude estimates (Sharpe collapse ~0.3–0.5 absolute). Gemini is most pointed: for PEAD specifically, P0.1 does not just inflate Sharpe — it **manufactures** the entire alpha, because PEAD edge is concentrated in the post-announcement overnight gap which `prev_close×1.001` captures for free. Every strategic decision downstream is contaminated until fixed.

### 1.2 Factor IC has never been computed — SHOWSTOPPER
Factor weights in QualityShortScorer and the swing feature stack are currently opinions, not evidence. IC must be broken down by **horizon (1/3/5/10/20d), regime, sector, cap bucket, long-vs-short separately**. Be psychologically prepared to delete factors that do not carry their weight. Non-negotiable before any feature pruning or selector consolidation.

### 1.3 Survivorship bias must be a labeled axis on every experiment
The 15% estimate is plausible but coarse. Every walk-forward result must carry a `universe_mode` tag (current-constituents vs. point-in-time-with-delistings) and both must be reported side by side. The habit of comparing across runs without checking the universe is a silent, persistent bias source.

### 1.4 Paper trading is software QA, not performance proof
Unanimous. Alpaca paper does not model: market impact, borrow availability, borrow rate dispersion, queue position, partial fills under stress, opening auction dynamics. A 60-day paper Sharpe of 1.2 means the software works — it tells you essentially nothing about live edge. Promotion criteria must shift to "economic P&L after modeled costs on point-in-time data with deflated Sharpe."

### 1.5 NIS `action_policy = "block_entry"` is directionally wrong for shorts
Negative news on a quality-short candidate is confirmation, not invalidation. Real economic cost every day this remains open. Single-file fix.

### 1.6 The five-stage multiplier stack is unvalidated
`confidence_scalar × regime_mult × vol_targeting_mult × NIS_sizing_mult × macro_sizing_factor` — 5 multiplicative stages, none individually proven Sharpe-accretive. Prior: **1–2 of these are net negative alpha**. Ablation is mandatory. Default for any unvalidated multiplier: 1.0.

### 1.7 PEAD exit logic is wrong for the strategy
PEAD is an event strategy with finite alpha decay (~3–5 trading days post-announcement). ATR trailing stops and `max_hold_days=5` as documentation rather than enforcement means positions are sometimes held into noise unrelated to the original catalyst. Correct exit: **hard T+5 close exit + gap-invalidation stop + no extension absent new catalyst.**

### 1.8 Partial exit at 1×ATR + breakeven stop is retail folklore
Taking 50% off at 1×ATR and moving remainder to breakeven reduces variance of small wins and converts large wins into breakeven trades. Dominated by either a properly calibrated trailing stop on the full position, or pyramid-add on confirmation.

### 1.9 Net exposure must be a hard gate, not a config target
40% net long is a configuration aspiration rather than an enforced constraint. If longs fill and shorts do not (borrow unavailable, high spread), net could be 80% long. In a –3% SPY day, that is –2.4% portfolio loss exceeding the 2% daily limit in one hour.

### 1.10 The promotion bar is too low
A 0.50 paper Sharpe gate is below SPY on a risk-adjusted basis after costs. The bar should be: deflated Sharpe ≥ 0.8 per sleeve on point-in-time data after modeled borrow/impact/commission, ≥ 60 trading days, with per-sleeve attribution showing the sleeve is not duplicating an existing one.

---

## 2. Where Reviewers Diverge — Adjudication

### 2.1 Net-exposure gate: RM (Gemini) or PM (DeepSeek)?
**Verdict: PM, with RM as backstop.** DeepSeek is architecturally correct — RM sees proposals one at a time and cannot reason about a basket. PM is the only component with full prospective state. But RM should retain a defensive net-exposure check that rejects any individual proposal that would push the simulated post-fill net outside the hard band. Standard two-layer pattern: PM optimizes, RM enforces invariants.

### 2.2 FinBERT vs Haiku for NIS scoring (Claude)
**Verdict: Claude is correct in principle, but Phase 3, not Phase 0.** The immediate move is to log every Haiku call's input/output to disk — required to build the evaluation dataset. FinBERT migration follows once evaluation shows the direction of fit.

### 2.3 Agent split: over-engineered? (Claude)
**Verdict: Claude is right that it is over-engineered for 7 positions, but wrong that it should be collapsed.** The cost is already paid. Collapsing adds risk without solving the actual problem, which is insufficient logic, not too much plumbing. Keep the architecture; fix the logic gaps.

### 2.4 Options overlay for PEAD (Claude)
**Verdict: Defer to Phase 4.** Correct that $20k is structurally cramped and PEAD has natural option-overlay characteristics. But options introduce multi-leg execution, IV modeling, and Alpaca API complexity. Do not touch until equity sleeve is validated.

### 2.5 Kelly sizing vs target portfolio optimizer
**Verdict: Both, in sequence.** Kelly with a hard cap (25%-Kelly, max 5% NAV) is the right per-position sizing primitive once you have edge estimates. A portfolio optimizer (mean-variance with sector and gross-exposure constraints) is the right cross-position construction primitive. Sequence: Kelly first (simpler, removes unvalidated `confidence_scalar`), optimizer second.

### 2.6 PEAD + QualityShort correlation (Claude)
**Verdict: Claude is almost certainly right, and this is more important than the reviews convey.** Companies that miss EPS ≥5% almost by construction overlap with companies showing fundamental deterioration. The `pead_quality_short` selector is likely **one strategy with two scores** — apparent diversification is illusory, position concentration risk is higher than gross-exposure numbers suggest. Must be measured before Phase 1 selector work begins.

### 2.7 HMM regime detection (Claude)
**Verdict: Reject.** HMMs on regime over-fit to training windows and switch regimes too slowly for live trading. The v5 regime model (F1=0.728, weekly cadence) is acceptable. The real regime work should be factor weighting by regime, not a regime model rewrite.

### 2.8 OFI RL agent (DeepSeek)
**Verdict: Reject for ≥12 months.** Requires tick data, microstructure expertise, co-located infrastructure. Alpaca offers none of these.

---

## 3. What the Reviewers Missed or Underweighted

### 3.1 PDT regulation is binding on the capital constraint
$20k is at the PDT threshold. A single bad week drops the account below $25k, constraining intraday exits. Need an account-level circuit breaker that halts new entries if equity drops below $26k.

### 3.2 Walk-forward gate values are arbitrary
Swing > 0.8, intraday > 1.5 — these appear to be set by what prior iterations achieved, not by coherent theory of required edge given costs, capacity, and capital. Gates should be derived from annual turnover, modeled all-in transaction cost, target information ratio, and deflation factor for multiple testing. Honest derivation will likely raise the swing gate to ~1.0 and intraday gate to ~2.0.

### 3.3 No live-vs-WF drift detection
There is no automated comparison of realized live trade economics (slippage, fill rate, hold duration, win rate) against WF simulator assumptions for the same trades. This is the highest-value diagnostic an L/S book can have and it is missing.

### 3.4 The 11-rule RM has no time-in-trade decay
Existing positions age. A 30-day-old quality-short is treated identically to a 1-day-old one for risk checks. Thesis-driven strategies need time-in-trade penalties.

### 3.5 Per-sleeve P&L attribution is invisible
The unified P&L blob makes it impossible to tell which sleeve is paying for which. Every "the system made money this month" claim is uninterpretable until this is fixed.

---

## 4. Brutally Honest State Assessment

| Layer | Grade | Notes |
|---|---|---|
| Engineering | B+ | Better than 90% of retail. Async architecture, RM, WF harness, NIS all real. |
| Research | D+ | WF bugs invalidate published numbers. Factor IC never computed. Multiplier stack unvalidated. Selectors plausibly redundant. Promotion bar below SPY. |
| Operations | C | Auto-retrain disabled (Phase C). No drift detector. No per-sleeve attribution. Paper trading mistaken for performance proof. |
| Capital fit | D | $20k hostile to L/S equity + quality-short sleeve. Borrow costs will eat 1–3% annualized for names QualityShortScorer surfaces. Commission drag 2–3% at current turnover. |

**Net**: A research project with the appearance of being closer to production than it is. The risk is not that the system loses money — it is that published numbers cause an overconfident scaling decision before underlying edge is real. Stop publishing Sharpes until P0.1/P0.2/P0.3 are closed.

---

## 5. Phased Action Plan

### Phase 0 — Stop the Bleeding (Weeks 1–3)
**Nothing in Phases 1–4 starts until Phase 0 is complete. Halt model promotions during this phase.**

| # | Action | Why it matters | Files affected | Effort |
|---|---|---|---|---|
| 0.1 | Fix P0.1: entry priced at next-session open + half-spread slippage | Restores WF integrity; PEAD numbers will likely collapse | `agent_simulator.py` | 2–3 days |
| 0.2 | Fix P0.2: stop-out check uses intraday high/low, not daily close | Stops fire when they actually would; current code hides losers | `agent_simulator.py` | 1–2 days |
| 0.3 | Run factor IC: per horizon × regime × sector × cap × side | Turns factor weights from opinion to evidence | `scripts/compute_factor_ic.py` + new reports | 3–5 days |
| 0.4 | Survivorship label on every WF run (`universe_mode` flag + report header) | Eliminates silent ~15% bias; makes runs comparable | WF harness | 1 day |
| 0.5 | Re-run all WF backtests with 0.1+0.2+0.4; archive old numbers as "pre-fix" | Resets evidentiary baseline | WF harness + ML log | 2 days runtime |
| 0.6 | Fix NIS direction bug: `signal_alignment = NIS_direction × (+1 if BUY else -1)`; block if alignment=−1 AND materiality > 0.7 | Real cost every day this is open | `news_intelligence_service.py`, RM | 0.5 day |
| 0.7 | PDT circuit breaker: halt new entries if equity < $26k | Prevents rule violation that bricks the account | `risk_manager.py` Rule 0 | 0.5 day |
| 0.8 | Log every Haiku NIS call (input, output, latency, timestamp, forward returns) to DB | Required for Phase 1 NIS evaluation dataset | `news_intelligence_service.py`, DB migration | 0.5 day |
| 0.9 | Rewrite promotion bar: deflated Sharpe ≥ 0.8 per sleeve, after modeled costs, point-in-time data, ≥ 60 trading days | Stops promoting noise | Promotion gate code + policy docs | 0.5 day |

**Phase 0 exit criteria:** All WF numbers regenerated with pre/post diff archived. NIS short bug closed. Haiku call log live. PDT circuit breaker live.

---

### Phase 1 — Strategy Validation (Weeks 4–9)
Now that measurement is honest, find out what (if anything) actually has edge.

| # | Action | Why | Files | Effort | Deps |
|---|---|---|---|---|---|
| 1.1 | Sleeve isolation: separate WF reports for PEAD-long, PEAD-short, QualityShort, factor-long. No combined numbers. | Reveals which sleeve is paying for which | New `backtest/sleeve_runner.py` | 1 week | 0.5 |
| 1.2 | PEAD/QualityShort overlap audit: rank-correlation of scores on shared candidate universe | Tests whether they are one strategy or two | `analytics/selector_overlap.py` | 2 days | 1.1 |
| 1.3 | Multiplier ablation grid: each of 5 multipliers set to 1.0 in isolation; WF Sharpe compared | Identifies which multipliers are net-negative; default losers to 1.0 | `backtest/multiplier_ablation.py` | 1 week (mostly runtime) | 0.5 |
| 1.4 | NIS evaluation dataset: label logged Haiku scores against forward 1d/5d returns; compute hit rate, AUC | Tests whether NIS adds value at all before allowing size-up | `analytics/nis_eval.py` | 1 week | 0.8 (need 2wks of logs) |
| 1.5 | PEAD exit redesign: hard T+5 close exit + gap-invalidation stop (next-day open re-crosses announcement close → exit). No ATR trailing for PEAD. | Aligns exit with alpha-decay window | `pead_scorer.py`, `trader.py` | 3–4 days | 1.1 |
| 1.6 | QualityShort exit redesign: wider stops (2.5–3×ATR), longer max hold (20–40 days), borrow-rate-aware sizing | Stops stale-short shakeout; models borrow cost | `short_scorers.py`, `trader.py` | 1 week | 0.5 |
| 1.7 | Replace `confidence_scalar` with 25%-Kelly (capped at 5% NAV) using rolling 6-month per-sleeve win rate and avg win/loss | Removes one unvalidated multiplier; replaces with theory-grounded sizing | `portfolio_manager.py` | 4–5 days | 1.1, 1.3 |
| 1.8 | Position-size language audit: rename configs to distinguish `risk_dollars_pct` vs `notional_pct`; enforce `qty = risk_dollars / stop_distance` | Removes latent inconsistency flagged by ChatGPT | Config + sizing code paths | 2–3 days | None |
| 1.9 | Replace partial-exit + breakeven with Chandelier trailing stop (3×ATR) on full position; A/B test both in WF | Fixes retail folklore exit | `trader.py` | 3 days + WF runtime | 0.5 |

**Phase 1 exit criteria:** Per-sleeve deflated Sharpes published. Overlap matrix published. Multiplier ablation report published. NIS eval report with explicit keep/kill recommendation. Exit redesigns merged with WF improvement quantified.

---

### Phase 2 — Portfolio Construction (Weeks 10–15)

| # | Action | Why | Files | Effort | Deps |
|---|---|---|---|---|---|
| 2.1 | Net-exposure hard gate in PM (simulate fills; reject if net breaches ±55%/±25% band) + RM backstop Rule 0d | Closes the biggest L/S risk gap | `portfolio_manager.py`, `risk_manager.py` | 1 week | Phase 1 stable |
| 2.2 | Gross AND net sector exposure limits (sector_gross ≤ 35%, sector_net ≤ 20%) | Net sector = 0 can still mean massive gross crowding | `risk_manager.py` | 3 days | 2.1 |
| 2.3 | Split RM 11-rule sequence into hard gates (binary reject) and soft gates (continuous size penalty that reduces rather than rejects) | Removes cliff-edge rejections; captures alpha that currently gets hard-blocked | `risk_manager.py` refactor | 1 week | 2.1 |
| 2.4 | Document thesis for 40% net long (or reset to 0% market-neutral as default until derived from optimization) | Currently a number with no provenance | Config + analysis doc | 1 week analysis | 1.1 sleeve numbers |
| 2.5 | Regime-conditioned factor weighting: separate weight vectors per regime label, learned from Phase 0.3 IC by regime | Higher-leverage use of regime model than current multiplicative size adjustment | `factor_scorer.py` + regime-weight config | 1 week | 0.3 |
| 2.6 | Simple L/S portfolio optimizer (mean-variance with sector + gross constraints, shrinkage covariance) replacing top-N selection | Strictly dominates top-N ranking | New `portfolio/optimizer.py`; PM integration | 2 weeks | 2.1, 2.2 |
| 2.7 | Per-sleeve P&L attribution in DB + dashboard | Currently impossible to tell which sleeve makes money | DB schema + FastAPI endpoint | 1 week | 1.1 |

**Phase 2 exit criteria:** Net exposure enforced and held within band for 4 consecutive weeks of paper. Sector gross+net caps enforced. Per-sleeve attribution live and reviewed weekly.

---

### Phase 3 — Production Readiness (Weeks 16–20)

| # | Action | Why | Files | Effort | Deps |
|---|---|---|---|---|---|
| 3.1 | Broker-side bracket OCO orders for all entries (stop + take-profit attached at order placement) | Client-polled stops fail when the bot crashes; broker-side is resilient | `trader.py`, Alpaca API | 1 week | None |
| 3.2 | Limit orders pegged to NBBO mid (or TWAP for >0.5% ADV trades) replacing market orders | Market orders into small-cap momentum are toxic (Gemini) | `trader.py` execution layer | 1 week | 3.1 |
| 3.3 | Live-vs-WF drift detector: nightly comparison of realized fill price, slippage, hold duration, win rate vs WF assumptions for the same trades | Highest-value diagnostic for an L/S book; currently absent | New `monitoring/wf_drift.py` + alerting | 1 week | 2.7 |
| 3.4 | Per-symbol slippage model: log assumed-vs-realized per (symbol, time-of-day, side, size bucket); feed back into limit pricing and WF execution model | Closes the loop on transaction cost realism | DB schema + `execution_model.py` update | 1 week | 3.3 |
| 3.5 | Real borrow rate ingestion (IBKR static daily file or Alpaca borrow API) + WF cost model using real rates | 0.5% flat is unrealistic; quality-short names can cost 20%+ annualized | New `data/borrow/` ingest + WF update | 1–2 weeks | 1.6 |
| 3.6 | Time-in-trade penalty in RM for thesis-driven positions | A 30-day-old quality-short carries different risk than a 1-day-old one | `risk_manager.py` | 2 days | 2.3 soft-gate split |
| 3.7 | Migrate NIS scoring from Haiku to FinBERT (deterministic, backtestable); keep Haiku for rationale generation only | Reproducibility + cost + eliminates evaluation bias | `news_intelligence_service.py`, new model artifact | 2 weeks | 1.4 NIS eval shows fit direction |
| 3.8 | Reintroduce auto-retrain (RETRAIN_WEEKDAY) with promotion gated on Phase 0.9 deflated criteria | Phase C diagnostic mode was always temporary | `retrain_config.py` | 2 days | All Phase 0–2 |
| 3.9 | Multiple-testing register: every selector, feature change, overlay gets a row; deflated Sharpe applied automatically | Stops silent multiple-testing from re-accumulating | `docs/ML_EXPERIMENT_LOG.md` template + helper script | 3 days | None (start now) |

**Phase 3 exit criteria:** Bracket orders live ≥30 days with zero client-side stop incidents. Drift detector running with weekly reports. Real borrow data ingested. Deflated Sharpe register live.

---

### Phase 4 — Expansion (Month 6+)
Only after Phases 0–3 complete and ≥60 trading days of paper meet the deflated promotion bar.

| # | Action | Complexity |
|---|---|---|
| 4.1 | Options overlay for PEAD (defined-risk verticals, IV-crush aware) | 4–6 weeks |
| 4.2 | Capital scaling decision: raise to $30k+ to exit PDT zone and improve borrow/commission drag | Decision, not build |
| 4.3 | Additional sleeves: market-neutral pairs, sector-relative PEAD, intraday mean-reversion (only if independently validated) | 4–8 weeks per sleeve |
| 4.4 | SQLite → Postgres migration at >25 concurrent positions | 2 weeks |
| 4.5 | Dynamic macro-tier-1 strategy switching (HIGH risk → defensive; LOW risk → aggressive 60% net long) | 3 weeks |

**Explicitly deferred / rejected:**
- HMM regime model (over-fits, switches too slowly for live use)
- OFI RL agent (wrong venue, wrong data infrastructure)
- Agent architecture collapse (cost paid, benefit modest)

---

## 6. What Must Be True Six Months From Now

If Phases 0–3 are executed honestly:
- Every published Sharpe is on point-in-time data, with intraday-aware stops, deflated for multiple testing, after modeled costs
- Each sleeve has its own attribution and its own promotion record
- The five-multiplier stack is gone, replaced by fractional Kelly per position and portfolio optimizer cross-position
- PEAD has event-aware exits; QualityShort has borrow-aware sizing
- NIS overlay is either proven value-adding (remains on FinBERT) or removed
- Net exposure is enforced, not aspired to
- A drift detector runs nightly flagging any divergence between WF assumptions and live realization

**If the system is making money in paper but none of the above are true, the money is noise and should not be scaled. That is the single most important sentence in this document.**
