# MrTrader — LLM Quant Review Synthesis (2026-05-19)

**Reviewers:** Gemini, Claude, ChatGPT, DeepSeek (4 independent senior-quant reviews of WF results)
**Synthesizer:** Internal code-audit + adjudication, 2026-05-19
**Supersedes:** previous 2026-05-05 multi-LLM synthesis (archived earlier in this file historically).
**Verdict TL;DR:** All four reviewers correctly identified the headline (Sharpe 8.1, 11x-38x fold returns, QualityShort DD 0.01%) as arithmetically impossible. **Code audit confirms two critical equity-accounting bugs in `AgentSimulator` that together fully explain the headline.** Do not paper trade until fixed. Updated probability of observed live paper Sharpe > 0.50 over 3 months: **~25-35%** (slightly below the median reviewer estimate, because the bugs are now confirmed real).

---

## The Four Reviewers

- **Gemini** (`MrTrader_Quant_Review_Report_gemini.md`) — Strongest tone. "0% probability of success in current state." Frames everything as a structural simulator failure. Highest-priority fix: MTM bug + cash management. Demands Norgate data and realistic 15bps/side costs.
- **Claude** (`MrTrader_Quant_Review_claude.md`) — Most quantitatively rigorous. Decomposes the inflation into a ~20x factor (per-trade-return treated as portfolio-return) plus secondary issues. Probability 35-40%. Best academic benchmarking table. Best phased action plan (Phases A-E over 7-9 months).
- **ChatGPT** (`MrTrader_Independent_Quant_Strategy_Review_chatgpt.md`) — Most precise on the arithmetic impossibility. Specifically computes that 38x in 300 trades at 5% size requires ~24% per trade. Best taxonomy of candidate bugs (7 hypotheses). Probability <5% real edge at headline scale; ~35% real edge survives audit.
- **DeepSeek** (`MrTrader_Quant_Review_deepseek.md`) — Most operationally tactical. Gives concrete code snippets for the audit, sensitivity tests, kill-switch numbers. Probability 30-40%. 3-4 weeks of fixes.

---

## Consensus Findings (all 4 agree) — Code-Verified

| # | Consensus claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Sharpe 8.1 is implausible — must be a bug | **CONFIRMED BUG** | Equity series is artificially smooth (Q2 below) → Sharpe inflated |
| 2 | 11x-38x total returns inconsistent with 5% sizing | **CONFIRMED BUG (new finding via audit)** | Q4 below: opening a short *increases* portfolio equity by the short notional |
| 3 | QualityShort 0.01% max DD with 21% WR is impossible | **CONFIRMED BUG** | Q2+Q3: drawdown computed from equity series that does not mark-to-market open positions |
| 4 | Borrow 0.5%/yr is too low for QS universe | **CONFIRMED PARAMETER ERROR** | `agent_simulator.py` line 959: `borrow_cost = pos.entry_price * pos.quantity * 0.005 / 252` hardcoded |
| 5 | Survivorship bias from static R1000 is material | **PARTIALLY MITIGATED** | WF-A2/A3 added `pit_union("russell1000", ...)` and DB historical names. FMP/feature side may still miss delisted names. |

---

## Disagreements and Adjudication

| Topic | Gemini | Claude | ChatGPT | DeepSeek | Adjudication |
|---|---|---|---|---|---|
| Probability strategy works (>0.5 Sharpe live, 3mo paper) | 0% | 35-40% | 35% | 30-40% | **25-35%.** Gemini overconfident on 0% (paper Alpaca measures real equity, bug doesn't carry forward). The other three are clustered. After audit reveals the bug, true-edge probability is somewhat lower than their priors because the bug contaminates all 17 ranked configurations. |
| Severity / time-to-paper | Halt indefinitely | 4+ weeks after audit | Audit, then full Phase A-E | 3-4 weeks of fixes | **4-8 weeks minimum.** Audit + MTM rewrite + cost recalibration + re-run all 17 configs. CPCV deferred to post-audit. |
| Cost realism vs bug priority | Bug = primary | Bug 10x larger than costs | Bug = primary; cost is ~30% Sharpe haircut | Borrow is ~20x off; bug + borrow together are fatal | **Bug-first.** Costs are a 30-50% Sharpe haircut; the bug is a 10x inflation. |
| FMP look-ahead severity | Huge risk | Possible but not dominant | Possible | Audit needed | **Medium.** Code filters `date <= as_of` with no fallback used in `get_earnings_features_at`. Risk depends on FMP `date` semantics. Email FMP. |

---

## Code Audit Findings

### Q1 — Daily return series construction
**Location:** `app/backtesting/agent_simulator.py` lines 1086-1093.

```python
daily_rets = [(eq_vals[i] - eq_vals[i-1]) / max(eq_vals[i-1], 1e-9)
              for i in range(1, len(eq_vals))]
ret_series = daily_rets if len(daily_rets) >= 2 else [t.pnl_pct for t in accepted_trades]
```

**Verdict: STRUCTURALLY CORRECT.** Returns are computed from the portfolio `equity_by_date` series, not from per-trade pct. Denominator is portfolio equity. This is the **right shape** — but the input series is contaminated (see Q2).

**Edge case bug:** When `len(daily_rets) < 2`, falls back to `t.pnl_pct` (per-trade pct return) and feeds that into Sharpe annualized by `sqrt(252)`. For a fold with very few trading days that would massively inflate Sharpe. Unlikely to bind the headline configs but worth fixing defensively.

### Q2 — MTM tracking for open positions
**Location:** lines 86-92 in `_PortfolioState`.

```python
@property
def position_market_value(self) -> float:
    return sum(p.entry_price * p.quantity for p in self.positions.values())
```

**CRITICAL BUG — CONFIRMED.** Open positions are valued at **entry price**, never at today's close. Consequences:

- Unrealized losses on open positions are **invisible** to the equity curve until the trade closes (via stop/target/max_hold).
- Daily P&L variance is artificially low → Sharpe inflated.
- Drawdowns occur only on bad closes, never on mark-to-market squeeze. This is exactly the "QualityShort 0.01% DD" smoking gun.

**Fix:** PMV must use today's close (longs) or `2*entry - today_close` (shorts), so equity reflects current marks. Or equivalently, replace `entry_price * quantity` with `today_close * quantity` for longs and an explicit short-liability accumulator for shorts.

### Q3 — Max drawdown computation
Computed from `eq_vals` (the equity-by-date series). Method is **standard and correct**, but the input series is bugged per Q2 — drawdown only registers on bad closes. With tight stops at 0.5×ATR and a 1.5×ATR target, closing losses are bounded so DD prints as ~0%.

### Q4 — Cash reservation on entry, and the short-equity bug
**Location:** lines 906-911.

```python
if not is_short:
    portfolio.cash -= trade_cost + tx_cost
else:
    portfolio.cash += trade_cost - tx_cost   # receive proceeds
    portfolio.cash -= trade_cost              # post margin = notional
```

Net cash delta for a short: `-tx_cost`. Cash logic in isolation is fine. **But combined with the `position_market_value` definition** (which sums all positions including shorts at `+entry_price * qty`):

- **Opening a short ADDS short_notional to reported equity.** PMV grows by `+entry*qty` (treating the short notional as if it were a long asset) while cash only drops by tx_cost.
- This is the dominant mechanism explaining the 11x-38x fold returns. Every short position open INCREASES headline equity.
- A 15-position QualityShort book sized at 5% each adds **~75% of starting capital to ghost equity** the moment the book is filled.

**This is a second critical bug, complementary to Q2.** Together they explain the entire headline.

**Fix:** (a) make PMV signed by direction: `+entry*qty` for longs, `-entry*qty` for shorts; or (b) treat short positions with an explicit `liability` accumulator subtracted from equity. The current accounting effectively double-counts short proceeds — once into cash (via the receive-proceeds line) and again as positive PMV.

### Q5 — Short P&L formula on close
**Location:** lines 1034-1040.

```python
gross_pnl = (pos.entry_price - exit_price) * pos.quantity
portfolio.cash += pos.entry_price * pos.quantity + gross_pnl - tx_cost
```

**CORRECT.** Profit = (entry - exit) * qty for shorts. Cash at close = margin returned + realized P&L − tx_cost. This actually rebalances cash correctly on close, which is why realized P&L flows correctly. The open-position window is the buggy part (Q2+Q4).

### Q6 — Sharpe annualization
Computed via `StrategySimulator._sharpe(ret_series, 252)`. Standard form: `mean/std * sqrt(252)`. **CORRECT method.** Inflation comes from the input series (Q2/Q4), not from the annualization factor.

### Q7 — FMP PIT safety
**Location:** `app/data/fmp_provider.py` lines 92-129.

```python
past = [r for r in records
        if r["date"] and r["surprise_pct"] is not None
        and datetime.strptime(r["date"], "%Y-%m-%d").date() <= as_of]
```

- Filters strictly by `date <= as_of`. **No `filingDate`/`acceptedDate` fallback in this function** — that risk only applies if the FMP `date` field itself encodes after-population information.
- **Unknown risk:** FMP docs do not specify whether `date` is press-release date (PIT-safe) or populated-at date.
- **Action:** Email FMP to confirm semantics. Spot-check 20 random earnings rows against EDGAR / news wire timestamps.
- **No look-ahead from this code path was found,** but the risk is not zero.

---

## Probability Estimate (post-code-audit)

| Scenario | P | Notes |
|---|---|---|
| Bugs fixed, true Sharpe 1.5-2.5 | ~15% | "Real PEAD + Quality + factor" stack; aligned with academic post-cost benchmarks |
| Bugs fixed, true Sharpe 0.5-1.5 | ~30% | Cost-corrected, partial signal survives |
| Bugs fixed, true Sharpe 0.0-0.5 | ~30% | Marginal — signal but not investable at retail |
| Bugs fixed, true Sharpe < 0 | ~20% | No real edge after honest accounting |
| Bug not fixed before paper (Alpaca measures real equity anyway) | ~5% | Paper would correctly report a much worse number |

**Joint probability of observed paper Sharpe > 0.50 over 3 months, given Phase H/H+ configs:** **25-35%.**

---

## Prioritized Action Plan

| # | Action | File / Location | Expected Sharpe impact |
|---|---|---|---|
| 1 | **Fix `_PortfolioState.position_market_value` to mark to today's close, with sign by direction** | `app/backtesting/agent_simulator.py` lines 86-92 | Removes dominant inflation. Sharpe 8.1 → 1.0-2.5 range |
| 2 | **Fix short entry accounting** (don't add short notional as positive PMV; net it as a liability) | same file, lines 906-911 + 86-92 | Removes ghost-equity bump on short opens |
| 3 | **Recompute daily return series including unrealized MTM** so Sharpe denominator reflects real vol | same file, lines 1086-1093 | Sharpe approaches realistic 0.5-2.0 |
| 4 | **Replace flat 0.5%/yr borrow with regime-aware borrow** (5-10% blended baseline; 25%+ for HTB) | line 959 | -30 to -70% on QS Sharpe |
| 5 | **Raise round-trip cost from 10 bps to 30 bps** for next-day-open R1000 mid-caps | `transaction_cost_pct` defaults | -25 to -50% Sharpe haircut |
| 6 | **Audit FMP `date` semantics** (email FMP, spot-check 20 earnings rows vs EDGAR) | `app/data/fmp_provider.py` | Confirms or invalidates PIT safety |
| 7 | **Re-run all 17 configs after #1-#5** with written predictions before reading results | `scripts/walkforward_tier3.py` | New honest baseline |
| 8 | **Run CPCV (López de Prado Ch 12)** on the top 2 surviving configs | `scripts/walkforward/cpcv.py` (exists) | Honest distribution |
| 9 | **Replace fixed-pct sizing with vol-targeted sizing** (target 1% position vol contribution) | `app/strategy/position_sizer.py` | +10 to +25% Sharpe; reduces tail risk |
| 10 | **Add de-duplication between PEAD-short and QualityShort** | `app/backtesting/agent_simulator.py` and live PM | Prevents double exposure |
| 11 | **Edge case:** drop the `t.pnl_pct` Sharpe fallback at line 1090; set Sharpe=NaN instead | `agent_simulator.py` line 1090 | Defensive |

**Total estimated effort:** 3-5 days for #1-#3 (the critical bugs) + 1-2 weeks for #4-#7 (cost realism + re-run) + 2+ weeks for #8-#10.

---

## What Is Likely Still Real

1. **PEAD is a real anomaly.** Academic post-cost Sharpe is 0.6-1.5. With cost realism, regime gating, and the existing ML signal, expect 0.5-1.5 — not 8.
2. **Quality factor (long leg) is a real, durable premium** (Asness/Frazzini/Pedersen QMJ). Sustains 0.5-0.8 Sharpe over decades.
3. **Combined PEAD + quality long + sensible shorts may produce post-cost Sharpe 1.0-1.8.** Investable for retail, but nothing close to the headline.
4. **The ML feature engineering (TS norm, regime gates, factor scorer) is well-engineered.** The bugs are purely in the equity-accounting layer, not in the signal generation.

The QualityShort side is most at risk: a 21% win-rate book of HTB names with realistic borrow + squeeze exposure may have Sharpe near zero or negative after honest accounting. Consider switching to long-quality-only or options-overlay shorts (Claude review §4.1, §4.3).

---

## Final Recommendation

**No-Go for paper trading as configured.** The simulator has two confirmed equity-accounting bugs (MTM never marked; short opens inflate equity). Until those are fixed, every WF result — including the 17 Phase H+ configs — is contaminated and rankings cannot be trusted.

**Go-to-paper conditions (all required):**
- [ ] Fix PMV MTM bug (#1) and short-equity bug (#2). Reproduce the equity curve in an independent notebook and verify Sharpe drop.
- [ ] Re-run all Phase H+ configs with realistic costs (#4, #5) and the corrected simulator.
- [ ] At least one config survives with Sharpe > 1.0 across all folds and DSR p > 0.95 with N=50.
- [ ] FMP PIT semantics confirmed (#6).
- [ ] CPCV (#8) confirms top survivor has 75%+ positive paths.

Then deploy paper at 1% sizing for 4 weeks, KS-test trade distribution, scale up over 6 months. Real-capital decision no earlier than late 2026 / early 2027.

---

*End of synthesis. Single highest-value next action: fix #1+#2. Three days of work; resolves the dominant source of uncertainty in the entire research program.*
