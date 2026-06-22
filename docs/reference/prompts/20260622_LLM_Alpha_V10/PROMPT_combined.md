# MrTrader — External Review (2026-06-22, Alpha-v10)

> **HOW TO USE THIS FILE:** paste this entire document into a fresh chat with each model
> (ChatGPT / Claude / DeepSeek / Gemini / Grok). It is self-contained — no attachments needed.
> It has four parts: (1) the brief + brutal-honesty mandate, (2) a system snapshot, (3) the questions,
> (4) a reference appendix (exact gate thresholds, CPCV geometry, and the full family registry — consult
> only if a question needs a precise number).
>
> **OUTPUT FORMAT (please follow):** Write your full response as a single self-contained **Markdown
> (`.md`) file and give me a download link** to it, so I can save it and bring it back. If your interface
> cannot produce a downloadable file/link, instead output the entire response inside **one Markdown code
> block** that I can copy in a single click. Title the file `mrtrader_review_<yourname>_2026-06-22.md`
> (e.g. `mrtrader_review_chatgpt_2026-06-22.md`).

---

# Part 1 — The brief

**To the reviewer:** You are a world-class quantitative researcher / systematic-trading PM / quant-
platform engineer (whichever hat each question needs). I run a solo systematic-trading program,
"MrTrader." I want a **brutally honest** review — no flattery, no hedging, no participation trophies.
If an idea is weak, say so and say why. If I'm fooling myself, tell me where. I would rather hear "this
is a sunk cost, stop" than a polite list of things to try.

## What I'm asking
While I wait for IBKR futures approval (live book unchanged in the meantime), I want outside eyes on
four things, in priority order:
1. **An overlooked trading method** — given what I already trade and the long list of things I've
   already killed, what genuinely-distinct family have I missed that could clear my gates?
2. **Swing equity** — I'd like to revisit swing trading (medium-frequency, days-to-weeks). Is there a
   credible avenue, or is it a sunk-cost trap for my setup?
3. **Better-trade what I have** — sizing/timing/execution/governor/allocation improvements to the
   existing book, rather than new alpha.
4. **Make the app stronger** — robustness/risk-plumbing before I wire real IBKR capital.

## Important context on method
I already ran an **internal panel** (five independent analyses that could read my actual codebase) plus
an adversarial red-team. Their synthesis is summarized in Part 3 as "the internal panel's current view"
for each topic. **I am explicitly asking you to attack those conclusions, not repeat them.** You cannot
see my code, but you have decades of priors I don't — tell me where the internal panel is wrong, where
it's groupthinking, and what a desk full of people would have caught that five instances of one model
family did not.

## My standing principles (so you can hold me to them)
- **I hold risk premia, not alpha.** Trend + carry are ARP. I size by drawdown, not Kelly; I freeze gross.
- **The book is "effectively one bet in a crisis"** (post-2015 pairwise corr ~0.49; convergent premia).
- **Orthogonality without standalone return is worthless;** a great standalone Sharpe that's redundant
  to the live book (high corr-to-book) gets PARKED. The real gate is *marginal* contribution.
- **Pre-registration + multiple-testing discipline.** I kill at the pre-registered sign; no sign-flipping.
- **No data buy without a stacked justification;** no new live capital before the safety layer is real.

Be specific. Real signals, real instruments, real numbers, real falsification tests. "Try machine
learning" or "improve risk management" will be ignored.

---

# Part 2 — System snapshot

## 1. What it is
A solo, fully-systematic trading program. In-process FastAPI orchestrator (PM / RiskManager / Trader
agents + APScheduler), Postgres (truth) + Redis (ephemeral), React dashboard. Trades **Alpaca paper**
(~$100k NAV, US equities / ETFs / spot crypto). **IBKR futures account applied 2026-06-21, approval
pending.** Everything is rules-based + walk-forward-validated; no discretionary trading.

## 2. What's LIVE now
- **ETF trend sleeve** — multi-asset time-series momentum (TSMOM). Universe: SPY, QQQ, IWM, EFA, EEM,
  TLT, IEF, GLD, DBC, UUP. Signal = equal-weight ensemble of `sign(P_t / P_{t-L} − 1)` over lookbacks
  {21, 63, 126, 252} (no skip-month); inverse-realized-vol sizing to a 10% per-instrument vol target;
  long-flat (no shorts); weekly rebalance; ~2 bps/side modeled cost. **Allocation = 50% of NAV**
  (Kelly-haircut; standalone 100%-gross book is Sharpe 0.72 / ann vol 9.34% on 19.4y history → ~4.7%
  vol at the live allocation). Post-2015 trend Sharpe +0.77 (2020s +1.05) — NOT decayed.
- **Cash sleeve** — idle settled cash parked in T-bill ETFs (SGOV/BIL) for the risk-free rate; just
  enabled (first deploy 2026-06-22).
- **VIX crash governor** (LIVE overlay) — de-risks the trend book 1.0→0.5 when VIX > VIX3M (backwardation).
- Crypto trend runs in **paper-only** (not capital).

## 3. What's PAPER (validated, not yet capital — gated on IBKR)
- **Futures carry** (term-structure) — roll-honest Sharpe 0.58, post-2015 +0.89.
- **Futures cross-sectional 12-1 momentum (xsmom)** — Sharpe 0.56, corr-to-trend 0.12; survived a
  selection-aware max-of-6 null (p 0.005).
- **Futures book = equal-weight(carry, xsmom)** — Sharpe 0.67; residual-α t = 2.29 vs the live trend
  book (the "second engine"). DSR(N≈25 families) ≈ borderline → "size modestly."
- **VRP via the VIX-futures curve** (short front VIX in contango, gated by the crash governor) — Sharpe
  0.64; **DROPPED from the initial live book** per tail diagnostics (most tail-concentrating, loses 6/8
  crises).
- **Crypto trend** (TSMOM on Alpaca spot) — Sharpe 0.64, corr-to-trend 0.18; CAPITAL fail on history.

## 4. The acceptance gate (the bar everything must clear)
Two-track, walk-forward, CPCV-validated, family-wise multiple-testing corrected:
- **Track-A (standalone):** CPCV mean Sharpe, path-t-stat, %-positive, worst-path, DSR — must clear a
  paper-pass bar.
- **Track-B (marginal):** budget-invariant **residual-alpha contribution vs the existing book** —
  appraisal IR, P(ΔSR>0) ≥ ~0.875, **corr-to-book < ~0.30**, tail-overlap. *This is the real gate.* A
  strategy can be excellent standalone and still PARK if it's redundant.
- **Multiple-testing:** an auditable registry enumerates **~26 distinct strategy families searched**;
  this N feeds a deflated-Sharpe cross-check. We kill at the pre-registered sign (no flipping).

## 5. The KILL LIST (what's been tried and rejected — don't re-suggest without a new angle)
- **Equity ML cross-sectional ranker** — IC ≈ 0 across THREE independent builds; a positive-control
  proved the feature→label pipeline is faithful, so the null is the market, not a bug.
- **PEAD (earnings drift)** — real-but-beta; event-level t = −0.77 (p 0.78) → demoted, off live.
- **Short-term reversal** — real gross (+0.40) but cost-dead (159×/yr turnover, −0.90 net at 10 bps).
- **Overnight (close→open)** — real gross (+0.53) cost-killed net.
- **Short-interest XS, turn-of-month, ETF relative-value** — killed (no edge / timed-beta / reversed).
- **Options-as-signal (CPIV/skew/OI/term/IVRV)** — 5 hypotheses, all killed; single-name option spreads
  ~33% half-spread (brutal).
- **Futures factor zoo** — curve-momentum, value, skewness, basis-momentum, CFTC CoT all KILLED at the
  pre-registered sign; only carry + xsmom survived. Futures *trend* decayed post-2015 (killed).
- **Sector-ETF relative-strength rotation** (2026-06-22) — standalone CPCV Sharpe 0.86 but Track-B FAIL
  (corr 0.51 to trend) → PARKED (redundant). Lesson: gate the marginal contribution, not the standalone.
- **Credit / curve / short-interest overlays** — credit is a marginal CANDIDATE (flag off, multiplicity
  caveat); curve + short-interest killed.

## 6. The safety / risk architecture (and its TRUE status)
A "portfolio-brain" safety substrate is being built in stages (R0 minimum-viable-safety → IBKR → unified
Constructor). **Honest status:**
- **BUILT + measuring (read-only / shadow):** consolidated cross-venue book-state + netted factor
  exposure; a whole-book risk gate wired into the trend rebalance in **shadow** (logs what it *would*
  block, blocks nothing); a frozen risk-policy-v1 (vol target ~6–8%, drawdown de-gross ladder
  −8/−12/−16/−20% → 0.75/0.50/0.25/0.0, per-venue margin reserve, notional caps); daemon-decouple
  capability (default in-process).
- **BUILT but NOT WIRED to the live order path (key gap):** the new broker-vs-DB reconciliation
  (fail-closed) and the kill-switch *state machine* are imported only by tests/scripts — the live path
  still uses a startup-only reconciler + a legacy kill flag. **The drawdown ladder + vol-targeting are
  applied only in offline research, not on any live order.** So today the live book's only enforced
  book-level controls are the VIX governor + an 80% gross cap.
- **Plan:** flip the whole-book gate shadow→enforce (~2026-06-29 after a clean shadow week); wire
  reconciliation + kill-switch; then IBKR paper→tiny-live. The roadmap names the cross-venue risk
  surface + kill-switch as the **hard no-go gate before any IBKR capital** (correctly NOT yet met).

## 7. Data owned / available
- **Free / owned:** ~19y liquid ETF + large-cap daily history (survivorship-clean for ETFs); FRED macro;
  VIX-curve; FINRA short data; **Norgate FUTURES mirror** (~100+ markets incl. G10 FX 6E/6J/6B/6A/6C/6S,
  rates ZT→UB, energy CL/NG/RB/HO/BRN, metals GC/SI/HG, equity ES/NQ/RTY, ags, VX; GC to 1978).
- **Paid feeds:** FMP Starter ($29, fundamentals); Polygon Options ($79, no hist IV/OI/NBBO); Alpaca
  paper; a self-built Alpaca options-NBBO forward log (immature, ~days).
- **NOT bought (the standing decision):** **Norgate US Stocks (Platinum, ~$693/yr)** — delisting-
  inclusive / survivorship-bias-free single-name + delisted-ETF history. A "justification accumulator"
  is building: it would unlock (a) survivorship-free single-name momentum, (b) thematic/country ETF
  rotation, (c) un-biased PEAD/relative-value re-tests. Deferred until the stack of reasons clears the cost.

## 8. The honest meta-picture (from prior external + internal panels)
"The app is ahead of the alpha." Excellent process; the book is a single (trend+cash) bet with a
promising-but-paper second engine (futures carry+xsmom). The binding constraints are **capital + live
track record + not blowing up when IBKR capital is wired** — arguably more than finding a 5th sleeve.

---

# Part 3 — Questions

For each block I give **the internal panel's current view** — your job is to **contest it**, not echo it.
Be brutally honest and specific (real signals, instruments, numbers, falsification tests). Where you
agree, say why; where you disagree, say what the internal panel missed.

## Block A — An overlooked trading method (weight: HIGH)
**Internal panel's view:** the existing book is all *convergent continuation* premia; the white space is
the *reversion / convexity* quadrant the book is structurally short. Candidates surfaced: (1)
short-horizon mean-reversion (but we found we already built + shelved it as cost-dead); (2) G10 FX
*value* (mean-reverting, distinct from our commodity-heavy carry); (3) equity dispersion /
implied-correlation premium (orthogonal but needs an options-data buy); (4) re-gating crypto-trend as a
diversifier (corr 0.18). Red-team punchline: **no one proposed a genuinely long-convexity crash hedge —
the one thing the book is short — and adding any new convergent "diversifier" re-violates "one bet in a
crisis."**

- **A1.** What genuinely-distinct family are we missing? Rank by *expected marginal* contribution, not
  standalone Sharpe. Be explicit about the economic payer.
- **A2.** Is there a *long-crisis-convexity* premium/strategy that is NOT just "buy puts / long VIX"
  (negative carry) — something with positive or neutral carry that still pays off in a crash? (trend-on-
  defensives, FX/rates trend, gold/bond convexity, TSMOM tilts.) This is our biggest gap — push hard.
- **A3.** Contest the reversion thesis: is short-horizon reversal genuinely dead for us (cost), or is
  there a lower-turnover / different-instrument formulation that survives 2 bps liquid-ETF costs?
- **A4.** FX value vs FX carry on G10 — which (if either) has a live pulse post-2015, and how would you
  falsify it without overfitting?

## Block B — Swing equity (weight: HIGH)
**Internal panel's view:** swing equity is mostly a **sunk-cost trap**; the single opening is
**vol-managed single-name cross-sectional momentum** (Barroso/Santa-Clara — size 12-1 momentum inverse
to the momentum factor's own realized variance), the one construction never run (the ranker was always
constant-gross). BUT (a) high redundancy risk vs our trend book (also momentum), and (b) the red-team
killed the proposed "pre-screen on survivorship-biased data" because the bias *flatters* this specific
strategy (removes the crash names vol-management dodges) → false-positive-only. Data-gated on the $693
Norgate buy. PDT does NOT bind us (>$25k account).

- **B1.** Do you agree swing equity is a sunk-cost trap for this setup, or is the internal panel too
  pessimistic? Be decisive.
- **B2.** Is vol-managed single-name momentum worth the Norgate buy, given it's likely redundant to our
  trend sleeve? What would make it *non-redundant* (sector/beta-neutral construction)?
- **B3.** Is there a *non-momentum* swing equity premium we haven't tried that wouldn't just die the way
  the ranker did — and isn't single-name-cost-dead?
- **B4.** Given survivorship bias flatters most single-name tests, is there ANY valid cheap pre-screen
  before spending $693, or is "buy the clean data or don't test" the honest answer?

## Block C — Better-trade what we have (weight: MEDIUM-HIGH)
**Internal panel's view:** highest-EV "better-trade" move is **make the existing risk machinery
load-bearing** (flip the whole-book gate to enforce; wire the drawdown ladder into the live budget) +
**harvest idle-cash RFR**. Lower-priority knobs: EWMA vol, a skip-month on the TSMOM signal, a rebalance
band. Red-team **hard veto: do NOT turn book-vol-targeting up to 8%** — it levers the sole live
undiversified edge ~1.7× via a mechanism biggest right before vol spikes, just as correlated IBKR beta
is about to stack on top.

- **C1.** Do you agree the vol-target-up-to-8% move is dangerous, or is under-deploying the only
  validated edge (~4.7% vs a 6–8% policy target) the bigger sin? What vol would *you* run, and why?
- **C2.** Is wiring a drawdown de-gross ladder into a single-sleeve trend book genuinely additive, or
  does the VIX governor already capture most of it (correlated triggers)?
- **C3.** Skip-month on the live TSMOM signal: worth the risk of touching the crown-jewel edge? Any
  other sizing/timing change you'd prioritize over the internal panel's list?
- **C4.** When IBKR lands and we have ≥2 live sleeves, what's the *minimum* correct way to combine them
  (inverse-vol vs ERC vs covariance) given near-zero joint live history?

## Block D — Make the app stronger (weight: MEDIUM-HIGH)
**Internal panel's view:** the #1 next move (over any new strategy) is **make the safety layer
load-bearing before IBKR**: wire reconciliation-before-trade (fail-closed) + the kill-switch state
machine, fix a cash-ETF instrument-mapping gap, then flip the gate to enforce. Plus IBKR-specific no-go
items: an out-of-band broker-only flatten + an external dead-man watchdog (neither exists). Today the
live book's only enforced controls are the VIX governor + 80% gross cap.

- **D1.** For a solo-operator, single-process, about-to-trade-futures book, what catastrophic failure
  modes are we most likely underweighting? Rank by blast-radius × likelihood.
- **D2.** Is "reconciliation-before-trade + kill-switch wired + gate in enforce" a sufficient hard no-go
  gate before IBKR capital, or what else is mandatory (out-of-band flatten, dead-man, verify-on-connect
  on futures multipliers, per-order idempotency)?
- **D3.** What's the *minimum* monitoring that would stop a bad state running unnoticed overnight,
  without over-building?
- **D4.** Where are we over-engineering safety relative to a $100k book — what should we explicitly NOT
  build yet?

## Block E — The meta-question (weight: HIGH — answer even if you skip others)
- **E1.** Is the internal panel right that the binding constraint is **capital + live track record +
  not-blowing-up**, NOT finding a 5th sleeve — i.e. should we mostly *stop hunting alpha* for 1–3 months
  and instead harden + accrue track record? Or is that complacency?
- **E2.** If you could force us to do exactly THREE things in the next two weeks (IBKR still pending),
  what are they — and what one thing on our list is most likely a waste of time or actively dangerous?
- **E3.** What is the single most intellectually-dishonest thing in how we've framed this program — the
  blind spot a desk of humans would catch that our model-panel did not?

---

# Part 4 — Reference appendix (consult only if a question needs a precise number)

## A1. Validation geometry
- **CPCV (full run):** `n_folds = 8`, `n_paths = 2`, `purge_days = 10`, `embargo_days = 10`. Each sleeve's
  return series is run through this same CPCV; the path Sharpe distribution gives mean-SR, path-t-stat,
  %-positive, and worst-path (p5).
- **Deflated Sharpe (DSR):** computed against `N_TRIALS_TESTED = 300` for the ML/model retrain path; for
  premia families it is cross-checked against the **enumerated family count = 26** (see A4). DSR is a
  cross-check, never a sole veto — the empirical selection-aware max-stat null is the primary test.
- **Sacred holdout:** 2026-11-09 (never touched in research).

## A2. Track-B v2 acceptance thresholds (the real gate for any new sleeve — budget-invariant)
| Criterion | Threshold | Meaning |
|---|---|---|
| `appraisal_ir` (residual-α IR) | **≥ 0.20** | quality of the marginal stream vs the book, allocation-invariant |
| `P(ΔSR > 0)` (full pass) | **≥ 0.90** | bootstrap prob. the book's Sharpe rises by adding the sleeve |
| `P(ΔSR > 0)` (probation→PAPER only) | **≥ 0.75** | smaller-size probation path; requires live-paper ratification |
| `corr_to_book` | **< 0.30** | candidate's corr to the existing book (sector-rotation failed here at 0.51) |
| `min_standalone_sr` | **> 0.20** | candidate's own vol-targeted Sharpe (waived for declared `diversifier`) |
| `max_risk_budget` | **≤ 0.25** | candidate blended at ≤ this fraction of book risk |
| tail-overlap (`joint_tail_pctl` 0.01) | **≤ 0.30** | overlap of the two series' worst days (co-crash test) |
| prior SR sd (DSR shrink) | 0.30 | prior SR ~ N(0, sd²), shrunk by 1/√(1+log N_trials) |

## A3. ML / model retrain gates (the swing-ML / ranker path — distinct from the premia Track-A above)
| Gate | Swing | Intraday |
|---|---|---|
| Avg Sharpe | ≥ 0.80 | ≥ 1.00 |
| Min fold Sharpe | ≥ −0.30 | ≥ −0.30 |
| DSR p-value | > 0.95 | > 0.95 |
| Profit factor | ≥ 1.10 | ≥ 1.10 |
| Calmar | ≥ 0.30 | ≥ 0.30 |
| Purge | 85 calendar days | 2 trading days |

## A4. The full strategy-family registry (26 trial families + 2 excluded; status · one-line verdict)
**By status: LIVE 3 · PAPER-CANDIDATE 5 · KILLED 14 · PARKED 4 · SCAFFOLD 2** (excluded from the count:
cash = infra; futures_book = carry+xsmom ensemble).

| Family | Asset | Status | Verdict |
|---|---|---|---|
| etf_trend | equity ETF | LIVE | the only validated free-daily edge; post-2015 +0.77 |
| vix_crash_governor | overlay | LIVE | modest tail help; first positive overlay; live |
| cash_sleeve* | cash | LIVE | capital-preservation infra (not a search trial) |
| futures_carry | futures | PAPER | roll-honest Sharpe 0.58, post-2015 +0.89; survives GL-0 |
| futures_xsmom | futures | PAPER | Sharpe 0.56, corr-to-trend 0.12; max-of-6 p=0.005 |
| futures_book* | futures | PAPER | equal-weight ensemble of carry+xsmom (not a separate trial) |
| vix_vrp | volatility | PAPER | Sharpe 0.64; survives crashes; DROPPED per GL-1 (tail-conc.) |
| crypto_trend | crypto | PAPER | Sharpe 0.64, corr-to-trend 0.18; CAPITAL fail (history) |
| pead | equity | KILLED | event-level t=−0.77 (p=0.78) → demoted, off live |
| swing_ml_ranker | equity | SCAFFOLD | confirmed NULL (IC~0 annually); flag off, dormant |
| intraday_ml | equity | SCAFFOLD | cost/slippage unmodeled; deprioritized, never live |
| short_interest_xs | equity | KILLED | CPCV −1.21 Sharpe; meme-era reversal flip → no edge |
| options_signal | equity options | KILLED | H4a-e all t≪−2 → no tradeable equity edge (5 factors) |
| turn_of_month | equity ETF | KILLED | miss HAC + zero diversification (timed SPY beta) |
| overnight | equity ETF | KILLED | gross +0.53 → net +0.16/−0.22; cost-killed |
| etf_relative_value | equity ETF | KILLED | orthogonal but point_SR 0.026, p 0.46 → zero edge |
| credit_timing | equity ETF | PARKED | Track-A pass but corr 0.52 to beta → not diversifying |
| sector_rotation | equity ETF | PARKED | standalone CPCV SR 0.86 but Track-B FAIL vs trend (corr 0.51) |
| futures_trend | futures | KILLED | real historically, DECAYED post-2015 (+0.02); redundant |
| curve_momentum | futures | KILLED | Sharpe −0.24; killed at pre-registered sign (no flip) |
| futures_value | futures | KILLED | Sharpe −0.24; killed at pre-registered sign |
| futures_skewness | futures | KILLED | Sharpe +0.03; insufficient edge |
| basis_momentum | futures | KILLED | Sharpe −0.10, residual-α t 0.47; orthogonal, no edge |
| cftc_cot | futures | KILLED | Sharpe +0.06, residual-α t 0.27; orthogonal, no edge |
| rates_carry | rates | KILLED | config-robust but time-unstable (post-2016 dead) |
| credit_overlay | overlay | PARKED | marginal +0.064 dSharpe, PIT-confirmed; flag OFF (multiplicity) |
| curve_overlay | overlay | PARKED | no tail benefit (dSharpe −0.018) → off |
| short_interest_overlay | overlay | KILLED | uniformly Sharpe-negative marginal to governor |

`*` = in the registry for auditability but EXCLUDED from the 26-family trial count (cash = infrastructure;
futures_book = an ensemble of already-counted carry + xsmom).

> Within-family search burden (transparency): the futures factor zoo screened 6 free factors → only
> xsmom survived (killed at the pre-registered sign, no flipping); credit overlay's pre-registered
> L60/band0 FAILED → post-hoc L120/band0.02 selected (multiplicity disclosed); rates-carry passed a
> 12-cell grid but died on a fresh pre-registered stability run; swing/intraday ML ran 9+ training
> iterations to a confirmed null.
