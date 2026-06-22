# Option A-v2 — thematic / country dual-momentum rotation (SCOPE) — 2026-06-22

**Status: SCOPE ONLY (not built). Data-gated. Lower priority than the IBKR/futures execution track.**
Follows Option A (GICS sector rotation, 2026-06-22), which was *real standalone* (CPCV Sharpe 0.86)
but **redundant to the live trend book** (Track-B corr 0.51, dSR +0.072) → PARKED. A-v2 is the one
rotation variant with a credible path to being **non-redundant**, because it changes the *universe*,
not the signal.

---

## 1. The hypothesis (and why it's different from Option A)

Option A failed Track-B not because relative-strength rotation is a bad methodology, but because its
universe — the 11 GICS sector ETFs — lives **entirely inside the equity bucket the live trend sleeve
already owns** (`SPY,QQQ,IWM` within the 10-ETF multi-asset trend basket). Carving SPY into 11
sectors and holding the top-4 still leaves you ~fully long the same equity beta, de-risking on the
same broad-market signal → corr 0.51.

**A-v2's claim:** relative-strength rotation adds orthogonal information *only* in proportion to the
**cross-sectional dispersion** of its universe. Move from broad → GICS → **thematic/industry +
country** and pairwise correlations fall sharply:

| Universe tier | Typical pairwise corr | Cross-sectional dispersion | Tested? |
|---|---|---|---|
| Broad / multi-asset (live trend) | 0.9+ within equity | low | live |
| GICS sectors (Option A) | ~0.7–0.9 | medium | **redundant** (corr 0.51) |
| Thematic / industry (SMH, XBI, KRE, XOP, GDX…) | ~0.5–0.8 | **high** | **never** |
| Country / region (EWZ, FXI, INDA, EWJ…) | ~0.3–0.6 | **highest** | **never** |

A semis-vs-utilities or Brazil-vs-Japan spread carries momentum information that broad cross-asset
trend **structurally cannot see**. That dispersion is the entire prize.

> **Distinctness target:** corr-to-trend materially below the 0.51 we saw for GICS (pre-register a
> threshold, e.g. < 0.35), AND a Track-B residual-α that clears the bar Option A missed.

---

## 2. The crux: survivorship bias makes this a DATA problem, not a code problem

GICS rotation was safe on free (yfinance) data because the 11 SPDR sector ETFs **never delisted** —
the survivor set *is* the full set. **Thematic and country ETFs delist constantly** (launch-and-die
is the norm), so the survivor set is a biased, optimistic sample. Backtesting only today's survivors
would systematically inflate Sharpe — exactly the failure mode the program is built to avoid.

**Concrete example — the test that free data silently breaks:** `RSX` (VanEck Russia ETF) was a
standard country-rotation constituent until it was **halted and delisted in 2022** (Russia sanctions;
holders ≈ wiped). A survivor-only backtest never sees RSX, never takes the −100%/halt, and reports a
country-rotation Sharpe that is a **mirage**. Any honest A-v2 result *must* carry RSX (and every
other delisted thematic/country ETF) with its real return-to-delisting.

→ **A-v2 is data-gated on a delisting-inclusive (survivorship-bias-free) ETF history.**

### Data-source options
| Source | Delisting-inclusive ETFs? | Cost | Notes |
|---|---|---|---|
| **Norgate Data (US Stocks, Platinum)** | **Yes** (includes delisted) | **$693/yr** | **The recommended path. Same purchase already on the table for Option B (single-stock momentum) — one buy unlocks BOTH.** |
| CRSP | Yes (gold standard) | academic/expensive | overkill for our scale |
| yfinance / free | **No** (survivors only) | free | **structurally invalid for this exact test** — the reason GICS was safe and thematics are not |
| Reconstruct delisting roster from ETF.com / issuer data | partial | manual/messy | usable as a *cross-check* on Norgate's roster, not a primary source |

**Synergy to flag for the data decision:** the $693 Norgate Platinum buy serves **both** Option B
(delisting-inclusive single-stock momentum — the bigger prize) **and** A-v2. Do not buy data for
A-v2 alone; bundle it with the Option B decision.

---

## 3. Candidate universe (PIT-eligible, delisting-inclusive)

Two buckets, ~30–50 names total. Final list pre-registered before any run.

**Thematic / industry (US):** SMH/SOXX (semis), XBI/IBB (biotech), KRE (regional banks), XOP (E&P),
XHB/ITB (homebuilders), IGV (software), CIBR/HACK (cyber), TAN (solar), URA (uranium), JETS
(airlines), GDX/GDXJ (gold miners), XME (metals & mining), XRT (retail), KIE (insurance), ARKK
(innovation), and similar.

**Country / region:** EWZ (Brazil), EWJ (Japan), FXI/MCHI (China), INDA (India), EWG (Germany), EWW
(Mexico), EWT (Taiwan), EWY (Korea), EWU (UK), EWC (Canada), **plus delisted names (RSX/Russia, etc.)
carried to their real termination.**

**Eligibility rules (the new ENGINE work — PIT):**
- A name enters the universe only after `inception_date + warmup buffer` AND a **PIT minimum-liquidity
  gate** (AUM and/or 21-day ADV above a floor, evaluated as-of, never with future data).
- A delisting name is **carried with its actual return path to termination**, then removed — never
  silently dropped.
- Ragged history is handled by the **fold-in union book** already in `multistrat_eval` (Phase A); the
  **common-window CPCV is the verdict**, the union is corroboration.

---

## 4. Methodology — ~80% engine reuse

The signal engine is universe-agnostic and already built: `app/research/etf_rotation.py`
(`rotation_target_weights`, `rotation_backtest`) — 12-1 PIT relative-strength, top-K, inverse-vol with
vol floor, **Antonacci dual-momentum cash filter** (the absolute+relative combination), cost-charged,
`held.shift(1)`/`cost.shift(1)`.

**What changes for A-v2:**
- **Universe loader (NEW):** survivorship-aware, PIT-eligible, delisting-inclusive — the main new
  work. Reads Norgate (delisted incl.), applies the eligibility rules above, emits a clean
  point-in-time price panel with NaNs only for genuinely-not-yet-existing names.
- **Config:** larger universe (~40); `top_k` larger (e.g. 8–10 of 40); `cost_bps` raised (10–15 for
  thematics — 5 is optimistic for narrow ETFs); optional turnover cap.
- **Dual-momentum stays ON** — in a crash thematics all fall together, so the absolute-momentum
  cash filter carries the downside; the dispersion alpha is an up-market phenomenon.

---

## 5. Validation gates (pre-registered — OPT-5 discipline)

Pre-register **before** the first run: lookback=252, skip=21, `top_k`, weight scheme, `cost_bps`,
dual-momentum on, liquidity floor, the distinctness threshold, and the kill criteria below.

1. **Track-A** — standalone CPCV gate (paper-pass bar).
2. **Track-B** — **the real bar**: budget-invariant residual-α marginal contribution vs the **live
   multi-asset trend book** (the exact gate Option A failed). This is the promotion decision.
3. **Distinctness** — corr-to-trend < pre-registered threshold (target < 0.35; GICS was 0.51).
4. **Survivorship A/B (the headline diagnostic)** — run the *identical* backtest on (a) survivor-only
   free data and (b) Norgate delisting-inclusive data. **The gap = the survivorship premium.** If the
   edge exists only in the survivor data, it was a mirage → KILL. This A/B is a deliverable in its own
   right (it quantifies the bias the whole program worries about).
5. **Sample-adequacy** — many thematics are post-2015/2018; report effective common-window length and
   flag if DSR/CPCV geometry is too thin to trust (don't over-claim on a 6-year sample).

---

## 6. Decision tree (what each outcome means)

```
Track-B residual-α t < ~1.5            → KILL (no marginal edge; same as Option A)
corr-to-trend > ~0.45                  → PARK (still redundant despite the finer universe)
edge present in survivors-only, gone
   under delisting-inclusive data      → KILL (survivorship mirage)
Track-B PASS + corr < 0.35 + survives
   the survivorship A/B                → PAPER-CANDIDATE (pair with trend; capital later)
```

---

## 7. Phasing, effort, priority

| Phase | Work | Gate | Cost |
|---|---|---|---|
| **A2-0** (free, now) | Assemble candidate list + inception dates + **build the delisted-ETF roster**; estimate dispersion/corr on *survivor* data only to **size the prize** (explicitly biased-optimistic) | — | free |
| **A2-1** (data-gated) | Buy Norgate Platinum (**shared with Option B**) → build the survivorship-aware PIT universe loader | Norgate purchase | $693 (shared) |
| **A2-2** | Run the pre-registered gates (Track-A + Track-B + **survivorship A/B**) | — | free |
| **A2-3** | Verdict per the decision tree → park / kill / paper-candidate; log to `family_registry` (N_TRIALS +1), DECISIONS, ML_EXPERIMENT_LOG | — | free |

**Effort:** engine ~80% reuse; the genuinely new work is the **survivorship-aware universe loader +
PIT eligibility/delisting logic**. Medium.

**Priority:** **below** the IBKR/futures execution track (that's the live-capital path) and below
Option B (bigger prize, same data). A-v2 is research, data-gated, and *only* worth the data buy as
part of the Option B decision. **A2-0 (the free roster + prize-sizing) can be done anytime** and is
the right next step *if* we want to pursue rotation further — it tells us whether the dispersion prize
is even large enough to justify the loader work, before spending a dollar.

---

## 8. Honest caveats (why this is NOT a slam-dunk)
- **Still long equity beta.** Thematics co-crash; the dual-momentum filter, not the rotation, carries
  the downside. A-v2 diversifies *within* equity-momentum, it is not a new return engine.
- **Short, regime-poor history.** Most thematics post-date the GFC; a clean cross-regime CPCV may be
  impossible → the verdict could be "promising but unprovable on available history."
- **Cost realism.** Narrow ETFs have wider spreads; if the edge dies at 10–15 bps it was never real.
- **Capacity is irrelevant at our size** (a plus), but ADV/AUM still matters for the *eligibility*
  filter (don't backtest names that were untradeably thin at the time).

> Cross-refs: `etf_rotation.py` (the reusable engine), `multistrat_eval.py` (Track-B + union book),
> `family_registry.py` (the trial count), DECISIONS 2026-06-22 (Option A), and the Norgate/Option B
> data discussion in `ALPHA_V10_SYNTHESIS_AND_PLAN.md` §P4. Report-only — no live trading path.
