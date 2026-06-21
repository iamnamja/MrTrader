# Alpha-v10 GL-0 / GL-1 / R0.1 — Findings (2026-06-21, autonomous session)

Three pieces of the Go-Live plan, built while IBKR approval pends. All are **report-only / pure
data artifacts** — they touch no live trading code. Each: Opus design → implement → adversarial
deep-dive review → fix/iterate → tests → document.

- **GL-0** — selection-aware null-strategy zoo + carry/xsmom look-ahead audit → is the t≈2.6 futures
  book real, carry-only, or multiple-testing residue? (`app/research/null_zoo.py`)
- **GL-1** — tail / co-crash diagnostics → is the book "one bet", does VRP belong, is a defensive
  sleeve needed? (`app/research/tail_diagnostics.py`)
- **R0.1** — the frozen v1 risk policy artifact the future whole-book risk gate will read
  (`app/live_trading/risk_policy.py`) + this/ADR documentation.

---

## GL-0 — null-strategy zoo + look-ahead audit

### Method (faithful to the panel's "the null must replicate the researcher")
Randomised signals are driven through the **identical** pipeline (same 76-market universe, the same
`carry_backtest` engine + 3bps/side roll cost, the same 50/50 basket combine, and the same Track-B
residual-alpha-vs-trend statistic — `multifactor_alpha(...)["t_alpha_hac"]`, **verified
byte-identical to the canonical `appraise_track_b`**). The null = per-date **cross-sectional
permutation among present markets** (preserves each date's present-set + cross-sectional distribution
+ the return panel; destroys only the signal→market alignment).

Tests: (1) **null-books p** — null(carry)+null(xsmom) 50/50 → Track-B t; empirical P(t_null ≥ t_obs).
(2) **xs-momentum max-of-6** — the best of the SIX distinct factor families we searched, each
permuted (the selection bar). (3) **carry single-factor** — an independent permutation (carry had a
strong prior, not selected from 6). (4) **Deflated Sharpe** (Bailey-LdP) on the residual at
N∈{10,20,30} — the parametric trial-count cross-check. Verdict: BASKET_REAL / CARRY_ONLY / RESIDUE.

### Look-ahead audit (Claude B5) — PIT-CLEAN ✅
Perturbing the future (×5 the last 120 days) leaves every past backtest return **byte-identical**
(carry engine, xsmom engine, and the momentum signal all future-blind; max past-diff 0.00). The
carry *signal* PIT-correctness was established separately by the 2026-06-18 scheduled-expiry
hardening. **No look-ahead in the carry/xsmom pipeline.**

### Observed (current, verified == canonical gate)
Track-B residual-α t vs the live ETF-trend book (window 2007-02-16 → 2026-06-12, 4861d):
**book 2.61, carry 2.03, xs-momentum 2.22** (book residual-Sharpe 0.63). *(Higher than the docs'
2.29/1.76/1.60 from 2026-06-20 — drifted up with more data + the carry hardening; the zoo tests the
CURRENT numbers.)*

### Results (n=400 null replications, seed=0; 0/400 degenerate)
| test | observed t | empirical p | 95th-pct null t |
|---|---|---|---|
| **null-books** (basket, primary) | 2.61 | **0.002** | 0.10 (99th 0.52) |
| null-books (circular-shift cross-check) | 2.61 | 0.002 | — |
| **carry** single-factor null | 2.03 | **0.002** | 0.51 |
| **xs-momentum** max-of-6 (distinct factors) | 2.22 | **0.005** | 0.94 |
| Deflated Sharpe on residual | — | N10 0.903 / **N20 0.840** / N30 0.798 | bar 0.95 |

### Verdict — **BASKET_REAL** (size carry + xs-momentum; modestly)
The futures book is **real, not multiple-testing residue.** Against the panel's primary requested
test — the empirical selection-aware max-stat null ("the null replicates the researcher") — the
observed t's (2.0–2.6) crush the null distributions (95th-pct 0.1–0.9; empirical p 0.002–0.005).
**Both carry AND xs-momentum survive their own selection nulls** — xs-momentum clears even the
best-of-6-distinct-factors bar (2.22 vs 95th-pct 0.94). This **reverses the pessimistic predictions**
(Gemini "probably residue", Grok "likely inflated") and confirms the Claude/DeepSeek "genuine but
modest" read: the empirical deflation is decisive, while the **parametric DSR(N=20)=0.84 is
borderline (<0.95)** — reflecting a genuine-but-modest residual Sharpe (0.63 ann), not a failure
(Bonferroni-style parametric bars over-correct for these correlated factors; the empirical max-stat
null is the right test and it clears).

**Deploy implication:** the second engine clears GL-0 → both carry and xs-momentum register at
`live_fraction > 0` for R1 (no carry-only fallback needed). But the borderline DSR + modest residual
Sharpe say **size it modestly** — enter tiny-live and scale by process fidelity, exactly the
promotion ladder the Go-Live plan specifies. PIT-clean, so no look-ahead caveat.

---

## GL-1 — tail / co-crash diagnostics

### Method
Exceedance (stress-conditional) correlation among {trend, carry, xsmom, vrp} on the worst-q% SPY
days (full-history AND post-2015 — the worse window drives the verdict); down-vs-up beta vs SPY (the
"negative convexity" tell) with a bootstrap CI; named-crisis cumulative-return replay; a
weight-honest book-delta (with-VRP vs without-VRP) crisis comparison. Window 2007-04-02 → 2026-06-12
(4831 aligned days).

### Results
| metric | value |
|---|---|
| unconditional avg pairwise corr | **0.28** |
| SPY-worst-5% stress corr (full) | **0.30** |
| SPY-worst-5% stress corr (post-2015) | **0.40** |
| core book (ex-VRP) down/up beta asymmetry | **+0.08** [90% CI +0.04, +0.13] |
| VRP down/up asymmetry | **+0.20** (worst of the four) |
| VRP standalone crisis losses | **6 of 8** |
| VRP deepens the book's crisis loss (book-delta) | **4 of 8** |

Crisis replay (cumulative %): trend is the crisis hedge (GFC **+7.0**, EuroAug-2011 +0.8, 2022 +0.9,
Mar-2023 +2.3; modestly negative in the fast shocks), while carry/xsmom are the convergent legs
(COVID **−20.9 / −13.0**, Q4-2018 −16.7 / −10.2). VRP mixed (GFC +9.8 but Q4-2018 −15.0).

### Verdicts
1. **NOT "one bet."** Even the modern-regime stress correlation (0.40 post-2015) sits well below the
   0.60 bar — the rigorous tail test **refines down** the earlier rough "~0.49 / effectively one
   bet" worry. The book is genuinely diversified because **trend is divergent (crisis-convex) and
   offsets the convergent carry/xsmom.**
2. **Drop VRP from the initial live book** (or pair it with conditional-long-vol). It is the most
   tail-concentrating sleeve (asymmetry +0.20, loses standalone in 6/8 crises); at inverse-vol book
   weight its *marginal* impact is borderline (deepens 4/8) — so "drop / pair," not "catastrophic."
   Consistent with the Go-Live synthesis.
3. **Defensive (bond/gold/FX trend) sleeve: BORDERLINE — recommended, not mandatory.** The core
   book is reliably net-negatively-convex (asymmetry CI lower bound +0.04 > 0) but not reliably
   above the 0.10 action bar — trend's convexity largely covers it. Recommended if/when convergent
   exposure is scaled.

### Honesty notes (from the deep-dive review, addressed in the code)
- The verdict uses the **worse of full / post-2015** stress correlation (the panel flagged the
  modern regime as more correlated — it is, 0.40 vs 0.30, but still diversified).
- The VRP verdict leads with the **weight-honest book-delta** crisis comparison; the with/without
  asymmetry leg is flagged as confounded (inverse-vol over-weights low-vol VRP).
- The defensive verdict uses a **bootstrap CI**, not a hard point-vs-0.10 (which would be noise).
- Book-conditioned exceedance correlation is reported but flagged **biased-low (collider)** and is
  NOT used in the verdict.

---

## R0.1 — risk policy v1 (the artifact the whole-book gate will read)

`app/live_trading/risk_policy.py` — a frozen, pure, conservative v1 policy (controls nothing yet;
safe in shadow). Launch values (drawdown-anchored, **never Kelly**): **book vol ~6% launch / 8%
steady / 10% hard cap; −20% max-drawdown kill line**; a drawdown de-gross ladder
(−8%→0.75, −12%→0.50, −16%→0.25, −20%→flat) with 20-day re-risk hysteresis; stress-correlation
de-gross at 0.60/0.70; **per-venue IBKR margin ≤25% NAV + a ≥10%-NAV IBKR cash reserve** (Alpaca
cash cannot fund IBKR margin — the Go-Live margin-call guard); net-equity-beta ≤1.0, gross-ex-cash
≤80% NAV; **absolute notional caps** (single instrument ≤25% NAV, book ≤3× NAV) as the dumb
backstop; and the fractional paper-ramp steps (0 → 0.25 → 0.50 → 1.0, human-confirmed). Every change
bumps `POLICY_VERSION` + a DECISIONS entry.

---

## Tests
`tests/test_null_zoo.py` (11) · `tests/test_tail_diagnostics.py` (5) · `tests/test_risk_policy.py`
(5). All green; flake8 clean on `app/`.

## What this gates
GL-0 decides which futures strategies register at `live_fraction > 0` for R1 (the carry-only
fallback is built in). GL-1 decides VRP-out + the defensive-sleeve flag for R1. R0.1 is the policy
the R0.4 whole-book risk gate enforces. None changes the live book (trend 50% + cash; crypto paper).
