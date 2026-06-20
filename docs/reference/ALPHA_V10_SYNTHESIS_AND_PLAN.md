# Alpha-v10 — External Panel Synthesis & Implementation Plan (2026-06-20)

**Inputs:** 5 independent "brutally honest world-class quant" reviews (ChatGPT, Claude, DeepSeek,
Gemini, Grok) of the full program — kit + raw responses in
`docs/reference/prompts/20260619_LLM_Alpha_V9/`. Synthesized by Opus 4.8 after deep-reading each
(one dedicated reader per response) + running the one diagnostic the panel flagged as decisive.

---

## Part I — The verdict, unvarnished

**All five converge on the same uncomfortable truth: we do not have alpha — we have a small book
of alternative risk premia (ARP), run with an excellent process, but it is effectively a single
bet (trend + cash), and our one new candidate (carry) is not yet proven.** ChatGPT's line captures
it: **"the app is ahead of the alpha."** Gemini: operate as **the CTA you've mathematically proven
yourself to be.** Grades clustered B/B+. Nobody thinks we're frauds; everybody thinks we've slightly
over-trusted two readings and that the bottleneck is now **implementation truth, not idea generation.**

This is not a failure verdict. It says: stop hunting, harden what's real, label it correctly, size
it honestly, and diversify deliberately.

---

## Part II — Consensus / disagreements / my calls

### Unanimous (or near-unanimous) consensus
1. **Carry's 0.66 Sharpe is a "phantom number" until roll mechanics are modeled.** (All 5; Gemini
   "cardinal sin.") **Critical nuance (Claude + DeepSeek): in commodities the roll YIELD *is* the
   carry signal — so a naive roll-cost subtraction risks DOUBLE-PENALIZING the very premium we
   harvest.** We must decompose: charge the *transaction* cost of rolling (commission + half-spread
   + slippage on the contract switch), NOT the roll yield. The honest number is somewhere ~0.55-0.60
   but must be *derived*, not asserted.
2. **Kill in-sample vol-matching in all decision claims** — re-report Track-B / combined-book on PIT
   rolling vol. The +0.17 carry dSR and especially the +0.064 credit-overlay likely shrink. (All 5.)
3. **The book is a single bet** (trend + cash). Carry is the right second sleeve *iff hardened*. We
   need a real path to ≥3-4 low-correlation sleeves. (All 5.)
4. **Trend/carry are ARP, not alpha** — mislabeling drives wrong validation + sizing standards.
   Relabel in the docs and size accordingly. (Gemini + ChatGPT explicit; others implicit.)
5. **"Free daily US-equity directional alpha is mined out" is overstated** — unprovable on
   survivorship-biased data. Retire the *strong universal* claim; keep the narrow defensible one
   ("cheap/slow/price-derived equity signals don't survive cost+beta *for us*"). (4 of 5.)
6. **The Kelly 7.7× framing is dangerous** — freeze gross; size by drawdown / margin / tracking-error,
   not Kelly. (Claude + ChatGPT strong.)
7. **Don't add decayed futures-trend as a return sleeve** (redundant with ETF trend). (All 5.)
8. **Execution/architecture must harden before live futures capital** — decouple the scheduler from
   the web server, broker-as-source-of-truth, idempotent orders, immutable snapshots, recon worker,
   kill switch. (4 of 5; Claude dissents — see below.)

### Disagreements → my adjudication
- **Buy Norgate US Stocks ($693)?** DeepSeek/Grok = yes (#1 buy); ChatGPT = yes but *bounded + after
  carry*; Claude = low-EV "buy only to stop guessing"; Gemini = no (academic). **My call: a "close
  the question" buy, not an alpha buy. Do it AFTER carry is hardened, for ONE bounded pre-registered
  equity audit (PEAD, FINRA-XS, simple factors) — burial-or-resurrection, not a restart of ML mining.**
- **Highest-EV next move?** Claude (sharpest) = **FREE: mine the futures factor zoo we already own +
  add CFTC CoT** before any buy. ChatGPT/Gemini = make carry *executable* (IBKR) is "the spend."
  DeepSeek/Grok = Norgate stocks. **My call: do BOTH free things first (factor zoo + CoT, $0), in
  parallel with the carry-executable build — those are the two highest-EV tracks; data buys come after.**
- **Re-test VRP/options?** 4 of 5 say it was likely *parked/killed on too-short data + alpha-framing*
  → re-test. Claude's cheap idea: **VRP via the VIX-futures curve** (short vol gated by our existing
  crash-governor signal), not options NBBO. Gemini alone defends the equity-direction kills as sound.
  **My call: re-test VRP cheaply via the VIX-futures curve as a *risk premium* (Track-B), now; the
  options-NBBO path waits for the forward log to mature (months). Re-frame, don't re-mine.**
- **Architecture urgency?** Claude = "not the bottleneck, stop polishing." Others = harden before
  futures capital. **My call: it's fine for current ETF *paper*; the hardening becomes mandatory at
  the IBKR/futures-capital step — sequence it WITH the futures execution build, not before.**

### The three sharpest individual insights
- **Claude — the "trend contradiction":** we report carry's post-2015 decomposition but never
  trend's; if ETF-trend were also dead post-2015, both our live conviction AND the futures-trend kill
  would be wrong. **We ran it (Part III): the live book is VINDICATED.**
- **ChatGPT — family-level trial counting:** "rules-based sleeves are OOS-by-construction" is false at
  the *family-selection* level; we've tried ~20 sleeve families, so the multiple-testing burden is
  real and currently uncounted. Build a null-strategy zoo to calibrate per-family false-positive rates.
- **The roll-cost double-count trap** (Claude + DeepSeek): confirm 1.1-1.9%/yr is *transaction* cost,
  not accidentally subtracting harvested roll *yield*.

---

## Part III — What we already settled (the decisive diagnostic)

The panel's top-priority ~1-day check, run 2026-06-20:

| period | live ETF-trend Sharpe | futures-trend Sharpe |
|---|---|---|
| pre-2015 | +0.66 | +1.09 |
| **post-2015** | **+0.77** | **+0.01** |
| 2020-2026 | +1.05 | +0.25 |

**Result: the contradiction dissolves and the live book is vindicated.** Our live ETF-trend is *not*
a pre-2010 relic — it is *stronger* post-2015 (+0.77, and +1.05 in the 2020s); the 10 liquid macro
ETFs (equity/bond/gold/commodity/USD) trended well through COVID + the 2022 rates/commodity moves.
The broad 76-market futures-trend genuinely decayed (+0.01 post-2015). **Both decisions hold:** keep
ETF-trend live; the futures-trend kill is correct. (This also softens — but does not eliminate — the
"freeze gross" caution: post-2015 Sharpe 0.77 is robust, so 50% is defensible, but we still size by
drawdown, not Kelly.) Book maxDD is the true vol-targeted −14%.

---

## Part IV — The Alpha-v10 plan (phased, prioritized)

Cross-cutting principles adopted from the panel (apply throughout):
- **Relabel:** trend/carry/cash are **ARP, not alpha** — in docs, language, and sizing.
- **PIT vol everywhere** in decision claims (no in-sample vol-matching).
- **Family-level trial counting** + a research-degrees-of-freedom log.
- **Size by drawdown/TE, not Kelly;** freeze gross at current levels pending live evidence.
- **The bottleneck is execution truth, not ideas** — weight effort accordingly.

### Phase 0 — Trust the numbers (FREE, days; BLOCKS any deploy/sizing change)
- **0.1 ✅ Trend sub-period diagnostic** — DONE (Part III): live trend vindicated; futures-trend kill confirmed.
- **0.2 ✅ DONE 2026-06-20 — carry honesty pass.** Added a TRANSACTION-only roll cost
  (`app/research/futures_roll.py` + `CarryConfig.roll_cost_bps`, 3bps/side; round-trip per roll on
  |held weight| — does NOT subtract the roll yield, no double-count) + switched Track-B to the
  budget-invariant residual-alpha. **Honest carry Sharpe 0.66→0.58** (drag ~1.1%/yr; still HAC
  p 0.0001, post-2015 0.81, Track-A PAPER-PASS point_SR 0.71). **Diversification REAL but MARGINAL:**
  the old in-sample +0.17 dSR was a vol-match artifact (~0.00 under PIT); residual-α t~1.8 /
  resid-Sharpe 0.43 → "probably helps," not a slam-dunk. Partly an energy/VIX bet (ex-energy ~0.54).
  → carry stays a PAPER-candidate; measured paper-deploy, not urgent capital. See DECISIONS 2026-06-20 (P0.2).
- **0.3 ✅ DONE 2026-06-20 — ruler negative controls → CLEAN.** `app/research/ruler_controls.py` +
  `scripts/run_ruler_controls.py`. True-null PAPER FP rate: floor-alone **23.6%** → JOINT with the HAC
  floor **5.3%** (n=1500 & 3000) — the known point-SR-floor leak is closed to nominal. Anti-correlated
  zero-edge null → Track-B residual-alpha pass-rate **5.7%** (~size) — not gamed by anti-correlation.
  The gate's Type-I error is controlled on both tiers; PASS/FAIL is trustworthy at face value. (Also
  retroactively validates the P0.2 switch to residual-alpha Track-B.) 3 tests. See DECISIONS 2026-06-20 (P0.3).
- **0.4 PIT-vol migration + credit-overlay re-verdict.** Replace in-sample vol-matching in Track-B /
  combined-book; re-report carry dSR; **the +0.064 credit overlay likely evaporates → kill or keep on
  the honest number.**
- **0.5 Family-level trial counting** in the registry + a degrees-of-freedom log (discarded variants,
  bug-fix reruns, reviewer suggestions, post-hoc exclusions).

### Phase 1 — Mine what we already own (FREE; the futures factor zoo + positioning)
All on the owned Norgate mirror; each pre-registered, through the (PIT-vol, family-counted) gate,
judged on Track-B over the *combined* book:
- **1.1 CFTC Commitment-of-Traders loader** (free weekly) → **hedging-pressure / positioning** factor
  (commercial vs non-commercial net) — the one genuinely *new* signal we don't have.
- **1.2 Futures factor sleeves** (~20-line declarations each): **XS-momentum (12-1, sector-neutral)**,
  **basis-momentum** (Boons & Prado-Tamoni — our differentiated edge given full term structure),
  **calendar/curve-spread carry** (isolate the premium from spot beta), **curve-momentum** (trend of
  the carry signal), **futures value** (~5y reversal), **commodity skewness**.
- **1.3 Futures multi-factor book.** Equal-risk / HRP ensemble of carry + the survivors; Track-B vs
  the live book. **Target: a genuinely multi-factor CTA sleeve, not a single carry bet.**

### Phase 2 — Make the futures book EXECUTABLE (the real bottleneck)
- **2.1 Roll-cost model** baked into the futures engine (per-market transaction cost; from 0.2).
- **2.2 IBKR paper account + futures execution adapter** (this is P4-3): contract master, roll
  calendar, margin-aware order preview, **broker-as-source-of-truth reconciliation**, idempotent
  orders (run-id keyed), immutable signal snapshots, kill switch, heartbeat/dead-man, replay/parity
  tests. (IBKR has a free paper account + supports futures, which Alpaca does not.)
- **2.3 Decouple the orchestrator** — execution as a standalone daemon (systemd/process), FastAPI
  read-only into Postgres, scheduler out of the web server. **Required before live futures capital.**
- **2.4 Live-paper the futures book on IBKR** → accrue the real-fill OOS record CAPITAL needs.

### Phase 3 — Re-test VRP properly (cheap first)
- **3.1 VRP via the VIX-futures curve** — short vol (small, vega-sized) gated by our existing
  crash-governor signal (contango = on, backwardation = flat); judged as a **risk premium on Track-B**,
  not an alpha floor. Owned VIX futures / VXX-family. Must survive Feb-2018 + Mar-2020 stress.
- **3.2 Defined-risk index VRP (options)** — only after the forward NBBO log matures (~months);
  freeze one design (30-45 DTE put-spreads / condors) now; build a quote-based fill model.

### Phase 4 — Close the equity question (bounded; AFTER carry is hardened)
- **4.1 Buy Norgate US Stocks (Platinum, ~$693)** → ONE bounded, pre-registered **equity audit**:
  PEAD redo, FINRA short-volume XS, simple factors (vol-managed 12-1 momentum, quality, value, 1-mo
  reversal) on survivorship-free data. **Burial-or-resurrection, NOT a restart of ML feature mining.**
  Kill the whole equity program again, for good, if nothing clears post-2015 net SR > 0.35 + residual
  t ≈ 2 + positive Track-B.

### Phase 5 — Book resilience (as sleeves accrue)
- Capital allocation across sleeves (HRP / risk-parity) once ≥4 sleeves exist.
- **Forward-looking book risk:** a realized-correlation-spike de-gross trigger (cheap, high-value).
- Global drawdown-based de-risk (beyond the VIX governor).
- Utility-based CAPITAL gate (P(SR>hurdle after costs), P(maxDD>limit), tails) — not just P(SR>0).

---

## Part V — How to proceed (recommendation)

**Sequencing.** Phases 0 → 1 → 2 run partly in parallel; 3 is cheap-now (3.1) + later (3.2); 4 waits
for carry; 5 grows with the book. Concretely, the recommended order of the next work:

1. **Phase 0.2 + 0.3 + 0.4 first** (days, free, no new code risk): make the carry number honest, prove
   the ruler isn't leaky, kill in-sample vol-matching. These *gate* every deployment decision — we
   should not size carry or kill the credit overlay until they're done.
2. **Phase 1 in parallel** (free): CoT loader + the futures factor zoo — this is where the *next* real
   diversification comes from, and it's $0.
3. **Phase 2 when you open IBKR** — the execution-truth build is the gate to ever trading futures with
   capital; it's the single biggest determinant of whether carry is real money or a backtest.
4. **Phase 3.1** (VIX-curve VRP) as a cheap, high-information re-test of a possibly-wrongly-parked premium.
5. **Phase 4** only after carry is hardened — the $693 equity buy to close the question for good.

**The honest bottom line for the operator:** we have a vindicated trend premium, a promising-but-
unproven carry premium, and an excellent but idle research machine. The win for the next quarter is
*not* a new strategy — it's (a) making carry honestly costed + executable, (b) mining the 3-5 free
futures factors we already own into a genuinely multi-factor book, and (c) labeling and sizing the
whole thing as the CTA risk-premia book it is. Resilience + correct sizing first; new alpha second.

**Suggested immediate next step:** execute **Phase 0.2 (carry honesty pass)** — it's the highest-value,
fully-autonomous, decision-gating piece, and it directly answers the panel's loudest criticism.
