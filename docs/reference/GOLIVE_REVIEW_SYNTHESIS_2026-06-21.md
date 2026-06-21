# Go-Live Review — External Panel Synthesis (2026-06-21)

**Inputs:** 5 independent "brutally honest risk-committee" reviews (ChatGPT, Claude, DeepSeek,
Gemini, Grok) — kit + raw responses in `docs/reference/prompts/20260621_Go_live_review/`.
Synthesized by Opus 4.8 after a dedicated deep-read of each response (one reader per response).
Three themes: **A** go-live sizing & cross-venue risk architecture; **B** is the t=2.29 second
engine real or multiple-testing residue; **C** do the four premia diversify or co-crash.

---

## Part I — The verdict, unvarnished

**All five converge, with unusual force, on one message: DO NOT deploy capital yet — not because
the research is bad, but because the layer you're about to deploy *into* (selection-honesty +
book-level risk governance + tail proof) does not yet exist.** DeepSeek put it best: *"the missing
layer is not alpha — it is risk governance."* Grades/labels: ChatGPT "yellow, not green"; Gemini
"CONDITIONAL HOLD — probably residue"; Grok "do not deploy, keep on paper"; DeepSeek "you are not
yet ready"; Claude "one edge you can defend, one factor probably real, one selection-inflated, one
I'd bench."

This is **not** a failure verdict. Every panelist credited the progress (carry honestly de-rated,
in-sample vol-match removed, Type-I negative controls, pre-registered-sign discipline killing
value/skew/CoT, the data audit). The message is: **three gates stand between us and capital, and
they are all buildable in weeks, mostly for free.**

The three gates, in priority order (unanimous):
1. **B — the selection-aware null-strategy zoo.** Is t=2.29 real after the ~20-family search? Until
   this runs, the "second engine" is unproven. *Gates everything else.*
2. **C — the tail / crisis co-movement test.** The book is almost certainly **net short-crisis**;
   prove (or disprove) real diversification before sizing. Decides VRP's fate.
3. **A — the cross-venue risk surface + single kill-switch.** A hard **operational no-go**: no real
   capital on a two-broker book you cannot halt and reconcile from one place.

---

## Part II — Consensus, disagreements, and my calls

### Theme B — Is the edge real? (the gate)

**Unanimous (5/5):** t=2.29 is **not trustworthy yet** — it is the Track-B residual-α of the
equal-weight futures book, uncorrected for the ~20 sleeve families tried over 2 years. A raw t>1.96
is meaningless after that search. **Run a null-strategy zoo before any capital.** All five rate this
the #1 move, buildable in <1 week.

**The deflated bar — where they land (the spread is the story):**
| Reviewer | Bonferroni-20 ref | Their effective bar | Prior on t=2.29 surviving |
|---|---|---|---|
| ChatGPT | t≈2.8 | empirical max-stat; eff-6≈2.4, eff-4 might pass | "uncomfortable middle" |
| Claude | t≈2.81 | DSR>0.95 @ N_eff=20; xsmom max-of-6 p≈0.29 | **~50/50, hinges on carry-alone** |
| DeepSeek | t≈2.58 (eff-10) | ~2.3–2.5 | "I expect it to survive" |
| Gemini | t≈3.02 | empirical ~2.8–3.1 | **"probably residue / DOA"** |
| Grok | t≈3.5+ | deflated t>2.0 OR empirical p<0.05 | "likely inflated" |

→ **My call:** genuinely uncertain — the optimists (DeepSeek, Claude-on-carry) and pessimists
(Gemini, Grok) disagree precisely *because the empirical test hasn't been run*. The likely outcome,
reading across all five: **carry survives its single-factor null; xsmom is marginal (it's a max-of-6
statistic — family-adjusted p≈0.29, not 0.055); the basket lands high-1s to low-2s** — probably
clearing a *correlated / effective-N* bar but probably failing a strict Bonferroni-20. **We must
build for the carry-only fallback** (Claude's branch): if only carry clears, size carry-only and
drop/de-weight xsmom; if neither, stay single-sleeve.

**"Gate the basket" — the one real disagreement (3 vs 2):**
- *Legitimate IF pre-registered* (ChatGPT, Claude, DeepSeek): equal-weight is a zero-free-parameter
  ex-ante combination; the math is honest. Claude's decisive framing: **the combination is
  mechanically honest (t_basket ≈ (t₁+t₂)/√(2(1+ρ)) ≈ 2.38 matches the observed 2.29); the
  contamination is the xsmom *input*, not the combination.**
- *A trick as practiced* (Gemini, Grok): combining two factors that fail the family-wise bar
  individually, post-hoc, is dredging.
→ **My call:** both are right about different things. The combination is legitimate *as math*; its
legitimacy *as evidence* hinges on two facts the null zoo + the research log will settle: (1) was
`equal-weight(carry, xsmom)` pre-registered before the basket t was seen, and (2) does the xsmom
input survive its **own** max-of-6 null. Grok operationalizes the resolution best: build **null
*books*** (equal-weight carry-null + xsmom-null) and measure the fraction that beat t=2.29 vs the
live trend book — **>~5–10% ⇒ residue.**

**Null-zoo design — strong consensus on the recipe** (synthesized from all five):
- **Null the signals, not returns, by BLOCK-PERMUTATION** (not Gaussian) — permute carry's sign /
  xsmom's 12-1 rank within randomly-chosen 2–13-week blocks → preserves autocorrelation so HAC SEs
  stay valid (DeepSeek, Grok, Claude converge).
- **Push each null through the ENTIRE real pipeline** — same 76-market universe, weekly rebalance,
  inverse-vol sizing, roll-cost model, Track-A gate, basket construction, **and the Track-B
  residual-α-vs-trend regression.** ChatGPT's principle: **"the null must replicate the *researcher*,
  not just the strategy"** — include the dead families (curve-mom/value/skew/basis/CoT) in the
  search the null reproduces.
- **Statistic:** max residual-α t / **Deflated Sharpe (Bailey-LdP)** + **Hansen SPA** (preferred
  over White's Reality Check because several of the 6 factors were clearly dead → SPA down-weights
  them). Report: xsmom percentile vs single-factor null; xsmom vs max-of-6; basket DSR @ N_eff=20
  (band 10–30); empirical family-wise p (fraction of null books ≥ 2.29).
- **N replications:** 1,000 (min) → 5,000–10,000 (preferred); ChatGPT wants the full tiered FWER.
- **Decision (tiered, ChatGPT's frame, which I adopt):** empirical FWER p ≤ 0.10 → tiny-live
  plumbing only; ≤ 0.05 → scalable; ≤ 0.025 + subperiod-stable → core engine. p > 0.10 → paper only.

**Plus (Claude B5, important):** the carry/xsmom pipeline is the **newest** code and has **not** had
the paranoid look-ahead audit the 2-year equity pipeline got — and we just found a look-ahead bug in
a parked scorer (the data-quality audit). Before trusting carry (the keystone): verify PIT-known
expiry only, no future-price leak in contract stitching, and **PIT rolling vol (not full-sample) in
the inverse-vol sizing.** Break-it-then-confirm-clean.

### Theme A — Sizing & risk architecture

**Unanimous calls (no meaningful dissent):**
- **Equal-Risk-Contribution / equal-vol is the sizing default. NO HRP / optimizer for 3–4 sleeves**
  — every reviewer, emphatically ("mathematical masturbation," "optimize noise," "doesn't earn its
  complexity"). Add conservative covariance shrinkage (ChatGPT: use the *most conservative* of
  long-history / rolling / stress-window estimates — the one that yields the highest book vol).
- **Delete Kelly from the vocabulary.** All five. Size by drawdown / vol-target / margin only. The
  "7.7×-Kelly, so 50% is deeply haircut" framing in our own docs is the single instinct several
  reviewers most want extinguished.
- **Fractional, evidence-gated entry for paper sleeves** — they do **not** enter at full ERC weight.
  Consensus ~**0.25–0.5× weight until live-proven**, ramping with evidence. ChatGPT's explicit
  ladder: futures-book 0.20 → 0.25 (post-zoo) → 0.40–0.50 (post tiny-live) → 0.75 (12mo live); VRP
  0.05 → 0.15 **max**.
- **Cross-venue unified risk surface + single kill-switch is a HARD no-go gate** before any IBKR
  capital. Daily (or 5-min) pull of positions/cash/margin from both brokers → one stitched book:
  aggregate gross/net, **netted factor exposures** (equity-β$, rates DV01, USD, commodity, short-vol
  vega — Claude: "the single most important output"), margin-to-equity. **Brokers are source of
  truth; the DB is intent/cache; halt on drift** (>0.1–0.5% NAV). Decouple it from the FastAPI web
  process (Gemini's "Risk Daemon"; ChatGPT's out-of-band flatten script; Claude's dead-man's switch).

**Numbers — consensus ranges:**
- **Target book vol:** ~**8–12%** steady-state, anchored to a **max-DD budget of ~15–20%**.
  *Outlier:* ChatGPT argues **3–8%** ("never size a drawdown you haven't lived through with real
  money"). → **My call:** ChatGPT's *principle* governs — we have **zero** live multi-strategy track,
  so **start low (~5–6% vol) and target ~8% steady-state**, hard-capped at a **−15% to −20%** book
  drawdown. Raise only with live evidence.
- **Margin-to-equity ceiling:** maintenance margin ≤ **~20–25% of NAV** (hard stop ~35%), with a
  stress-margin sub-limit; **always run the LOWER of {vol-target gross, margin-cap gross}.**
- **Global drawdown ladder (from HWM):** consensus ~ **−8/−10% → cut 25%; −12/−15% → cut 50%;
  −16/−20% → flat/halt.** −20% is the universal kill line. Asymmetric re-risk with hysteresis (don't
  re-risk on a bounce; ChatGPT: restore one rung only after ~20 days with no new low).
- **Correlation-spike de-gross:** best single metric (Claude) = **absorption ratio / first-PC share
  (Kritzman-Li)**; or 21-63d avg pairwise corr vs 252d baseline. Cut 25–30% at corr>0.6 / abs-ratio
  >90th pct; 50% at corr>0.7 + drawdown. **Round, un-optimized thresholds; require DD/vol/VIX
  confirmation — "a circuit breaker, not a trading signal"** (ChatGPT). Several call this the
  highest value-per-effort item in Theme A.

**Kill-switch — sub-disagreement (auto-flatten vs halt-only):** DeepSeek + Claude lean *halt new
orders + resting protective orders at the broker; full flatten is a separate (human) decision*
(auto-flattening a futures book at market in a crisis can be worse than holding). ChatGPT/Gemini/Grok
describe a `flatten_all()`. → **My call:** adopt **ChatGPT's state machine** —
`RUN / HALT_NEW / REDUCE_ONLY / FLATTEN_NON_CORE / FLATTEN_ALL / MANUAL_LOCK` — with the **dead-man
trigger auto-setting HALT_NEW**, resting protective orders at IBKR, and **full flatten reserved for
the extreme tier / operator confirmation.** Best of both.

**Promotion ladder — strong structural consensus** (numbers are the consensus center):
- **Rung 0 → 1 (paper PASS → IBKR paper):** ~**2–3 months**, ≥**10 weekly rebalances**, ≥**2 roll
  cycles** (≥1 clean VX roll if VRP), ≥**1 vol spike (VIX>25)**, **slippage <30% of modeled edge**
  (the 3bps/side roll assumption must hold), **intended-vs-actual daily PnL corr >0.9–0.95**, zero
  contract-mapping / reconciliation errors. Treat this as an **execution-system test, not a capital
  decision.**
- **Rung 1 → 2 (tiny live):** **1–2 contracts/market**, sleeve ≤**10% of risk budget**, futures
  margin <5–10% equity. **Demotion stops:** wrong contract traded; live-vs-paper-shadow daily corr
  <0.6; slippage >50% of edge for 2 months; TE >40% of expected sleeve vol; sleeve DD >1.5× backtest;
  margin breach.
- **Rung 2 → 3 (scale):** **≥12 months** live; scale **≤25%/quarter (≤2×/6mo)**, **by process
  fidelity, NOT live t-stat** (you won't have the data) and **never on a winning streak**
  (own-momentum-crash). Require ≥1 stress event handled correctly.
- **Two-instrument decomposition (ChatGPT + Claude):** run a paper-shadow alongside live per sleeve →
  (a) live vs paper-shadow = execution TE; (b) paper-shadow vs backtest = alpha decay. Tells you
  whether to fix the adapter or kill the sleeve. (We already have the intended-vs-actual instrument
  for trend; extend it to futures.)

**No-go list (union):** t=2.29 fails the null zoo; stress-conditional correlation >0.6–0.7; IBKR
paper slippage >30–40% of edge; no unified risk surface + cross-venue kill-switch + dead-man; margin
>25% NAV; VRP worsens book ES or sized >10% naked; book DD worse than trend-alone in crises, or you
can't reproduce a held-out crisis.

### Theme C — Diversification / tails

**Unanimous: the book is NOT genuinely diversified — it is net short-crisis, "effectively one bet."**
Framings: Gemini *"a synthetic short-put on global liquidity"*; Claude *"a leveraged long-risk-
premium with extra steps... short fast-crisis with a slow-crisis hedge from trend"*; ChatGPT *"a
long-risk-premia portfolio with some trend convexity, not all-weather."* The unconditional
correlations (0.12–0.49) are **"dangerously misleading."**

**Test — unanimous: exceedance / stress-conditional correlation.** Conditional correlation among the
sleeves on the **worst-5% equity days / VIX>30 / VIX-spike windows**. "One bet" if stress-conditional
corr **>0.6–0.7** (vs ~0.25 unconditional). Claude adds the sharpest single diagnostic: **book
down-market beta vs up-market beta — if down-β > up-β, the book has negative convexity / a hidden
long-risk bet ("the tell").** Backups: lower-tail dependence (copula λ_L), absorption ratio, ES
contribution per sleeve.

**Stress-test without a crisis fold — consensus 3 methods:** (1) **joint historical scenario
replay** through named crises (2008, 2011, 2015, Feb-2018, Q4-2018, Mar-2020, 2022) — *replay the
rules, not static weights*, so the gates fire on PIT data; (2) **leave-one-crisis-out / crisis-
weighted CPCV fold** — re-derive every data-driven param (vol, **gate thresholds**, weights) on
pre-crisis data, then evaluate the held-out crisis (directly attacks the VRP-gate-fitting concern);
(3) **crisis-block-oversampled bootstrap** — resample joint sleeve return *vectors* (preserves tail
correlation). ChatGPT adds a **synthetic correlation shock** (force all corr→0.8, vol 2–3×).

**VRP — strong consensus: do NOT run it as a 4th equal/naked sleeve.** 3 of 5 (DeepSeek, Gemini,
Grok) say **drop it from the initial live book**; 2 (ChatGPT, Claude) say keep only **capped (5–10%,
15% max) and never naked.** Reasons (shared): highest corr-to-trend (0.46), explicitly short-crisis,
lowest conviction (t=1.46), most selection bites (tried, parked, re-tried), and its crash-survival is
a **fitted event study** (the gate was calibrated with Feb-2018 + COVID in view). → **My call:
DROP VRP from the initial live book.** It may return **only** as Claude's **integrated vol sleeve**
— VRP paired with **conditional-long-vol** (long VX when cheap) as ONE sleeve, judged on **CVaR-per-
bleed (tail-risk reduction), not standalone Sharpe** — and only after a leave-COVID/leave-Feb2018-out
gate re-confirmation.

**Convex/defensive sleeve — consensus YES, and it comes from TREND on bonds/gold/FX, NOT options.**
All five converge: the most realistic crisis hedge from our data is a **defensive / crisis-alpha
trend sleeve on Treasuries (TLT/IEF) + gold (GLD) + USD/FX** — Claude notes a *can-go-short* trend
leg beats a static long (which fails in a 2022-style inflationary crisis where bonds *and* equities
fall). **Unanimous warning: do NOT buy structural long-VIX options as a Sharpe sleeve** (negative
carry bleeds the book). Conditional-long-vol (the VRP gate's opposite) is the *insurance* pairing,
not a standalone.

---

## Part III — The plan (prioritized, mostly free, weeks not months)

Sequencing reflects the unanimous "Monday morning" lists, which were remarkably aligned.

### GL-0 — The selection-aware null-strategy zoo *(FIRST; gates all capital; ~1 week; free)*
Build the null zoo per the consensus recipe: block-permuted carry-sign / xsmom-rank nulls → the
**full pipeline incl. Track-B + basket construction** → **Deflated Sharpe + Hansen SPA** + the
empirical **"null books"** p-value (fraction of null baskets beating t=2.29 vs live trend). Report
the deflated-t bar and the tiered decision (p≤0.10 / 0.05 / 0.025). **Branch logic:** basket real →
proceed; carry-only clears → size carry-only, drop xsmom; neither → single-sleeve. **Bundle Claude's
B5 look-ahead audit** of the carry/xsmom pipeline (PIT expiry, no stitch leak, **PIT rolling vol**).
*This is P0.5 (family-level trial counting) from the Alpha-v10 plan, finally built — and the single
highest-value thing we can do.*

### GL-1 — The tail / diversification test *(parallel with GL-0; ~days; free)*
Exceedance correlation + **book down-β vs up-β** + joint historical crisis scenario replay (rules,
not weights) + leave-one-crisis-out fold. Output: is the book one bet (stress corr >0.6–0.7?), does
VRP worsen the tail, and is a convex sleeve required. **Decides VRP in/out and the defensive-sleeve
go/no-go.**

### GL-2 — The cross-venue risk surface + single kill-switch *(hard infra no-go gate; before any IBKR capital)*
The unified daily cross-broker book state (open + pending + **netted factor exposures** + margin),
brokers-as-truth reconciliation with halt-on-drift, the state-machine kill-switch
(RUN/HALT/REDUCE_ONLY/FLATTEN tiers) with a dead-man trigger, decoupled from FastAPI. **This is the
measurement-first step of the Portfolio-Brain pack** — the two reviews dovetail here; build it once.

### GL-3 — Codify sizing & the promotion ladder *(after GL-0/GL-1 verdicts)*
ERC + conservative shrinkage + the **probationary-bucket fractional ramp**; the **lower-of
{vol-target, margin-cap} gross**; the **drawdown de-risk ladder** + the **absorption-ratio
correlation-spike trigger**; and the **promotion ladder with explicit demotion stops** + the
two-instrument (live / paper-shadow / backtest) decomposition. Delete Kelly from the docs.

### GL-4 — Reformulate the vol/convexity exposure
**Drop VRP** from the initial live book. If GL-1 says a convex sleeve is needed, build the
**defensive trend sleeve** (bonds/gold/FX, can-go-short) and/or the **VRP⊕conditional-long-vol**
integrated sleeve judged on CVaR-per-bleed. No long-VIX-options Sharpe sleeve.

### Then — the execution path (only after GL-0/1/2 pass)
IBKR paper (execution-system test, the P2 spec) → Rung-1 evidence → tiny-live (1–2 contracts) →
scale by process fidelity. The futures book reaches capital **only** if it survived GL-0, the tail
test, and a clean IBKR-paper record.

---

## Part IV — How to proceed (recommendation)

**The honest bottom line:** the panel did not move the *strategies* — it moved the *order of
operations*. We were treating the futures book as "validated (t=2.29)" and the next step as
execution plumbing. All five say: **t=2.29 is unproven until the selection-aware null zoo runs; the
book is net short-crisis until the tail test proves otherwise; and no capital touches a two-broker
book without a unified risk surface + kill-switch.** None of these needs a data buy or new strategy
— they are days-to-weeks of work we already have the tools for, and they *gate* the IBKR build that
was the prior "next step."

**The most likely outcome (my read):** carry survives, xsmom is marginal, VRP comes out of the
initial book, the live book becomes **ETF-trend + a carry-led futures sleeve (+ maybe a defensive
trend sleeve)** sized by ERC at ~8% vol — a smaller, more honest book than the 4-premia version, but
one we can actually stand behind in a crisis.

**Suggested immediate next step:** build **GL-0 (the null zoo)** — it is the single gate every
reviewer put first, it directly answers "is the second engine real," and it is fully autonomous and
free. GL-1 (tail test) runs in parallel. GL-2 (risk surface) is the bridge to the Portfolio-Brain
review and the prerequisite for ever funding IBKR.

> Cross-references: this synthesis feeds the **Portfolio-Brain** architecture review
> (`docs/reference/prompts/20260621_Comprehensive_portfolio_brain/`) — GL-2's risk surface is that
> pack's measurement layer. The look-ahead audit (GL-0/B5) ties to the 2026-06-21 data-quality
> audit's `factor_scorer` finding. Raw responses: `docs/reference/prompts/20260621_Go_live_review/responses/`.
