# Alpha-v7 Research Synthesis & Game Plan — 2026-06-14

**SSOT for the NEXT research direction.** Synthesizes the 5-LLM external panel (Opus 4.8, ChatGPT,
DeepSeek, Gemini, Grok) requested after the Ruler-v2 go-live + the honest candidate sweep (only
trend survived). Inputs archived at `docs/archive/llm-reviews/2026-06-14/`. Written by the owner-quant
after a deep careful read of all five, weighted against what we actually know about this system.

---

## 0. TL;DR

- **The panel is unanimous on the big picture:** the kills are honest, the gate works, **trend is the
  only surviving edge**, and the right next move is a **3–5 sleeve risk-premia book around trend, judged
  at book level (Track B), targeting a realistic book SR ~0.7–0.9** (NOT a 1.0+ home run).
- **Every new bet must live on DEEP FREE HISTORY** (19y yfinance ETFs, multi-decade FRED macro) — the
  only place our gate has the statistical power to confirm a true SR-0.4–0.7 edge. The 4y frozen options
  store is for *conditioning at most*, never standalone signals or execution.
- **The owner's decision: invest first in a future-proof "Sleeve Lab" foundation** (Phase F0), then run a
  small number of orthogonal, deep-history, owned-data premia through it — ordered by
  *(orthogonality-to-trend × power × not-already-spent × cheap)*. The highest-EV new bets are **structural/
  calendar/overnight premia + a crash/VIX-term governor (F1)** and **slow ETF relative-value (F2)**; carry
  (F3) and options-conditioned events (F4) are real-but-lower. Trend-via-futures, aggregate short-interest
  timing, and an index-VRP sleeve are **deferred** (data/infra-gated or dangerous).
- **Honest expectation:** likely a 2–3 sleeve book. Maybe only trend + one diversifier survives — and that
  is still a win. If nothing new passes on free data, the answer is operational excellence + a deliberate
  data-acquisition decision (Norgate futures), not more model variants.

---

## 1. Where all five agree (accept as settled)

1. **The kills are honest, not gate artifacts.** Re-scoring under best-case Ruler-v2 settings (regime
   waiver, Bayesian posterior replacing the saturated DSR) and still failing closes off "just lower the
   bar." Trend (10-ETF TSMOM, +0.71/19y) is the one validated edge.
2. **Re-charter to a premia BOOK** judged at book level (Track B), realistic **book SR ~0.7–0.9**. Not a
   home run; diversification math, not precision alpha, does the work.
3. **Deep free history is the only powered sandbox.** New ideas must be testable on 19y ETF / decades of
   FRED. (See §2.1 on the power debate.)
4. **The frozen 4y options store** is at most a *conditioning* input (event interactions), never standalone
   signals or execution. It freezes ~2026-06-17.
5. **Diversifiers must be genuinely uncorrelated — especially in crises.** Carry and short-vol both blow up
   *with* equities; size for the crisis correlation, never the average. (Opus, ChatGPT, Gemini all stress this.)
6. **Do NOT revisit** (unanimous): XS-equity ML ranking, intraday ML, single-name options (signal or
   execution), large-cap PEAD, frozen-options decile sorts, binary threshold sweeps, the cross-sectional
   short-interest factor, dispersion, paid data without a free proof-of-concept, random trend-broadening.

## 2. Where they diverge — and the owner's adjudication

### 2.1 Is POWER the binding constraint? (DeepSeek says no; Gemini/Grok/Opus say yes)
**Both are right at different scopes — and the distinction sets our whole strategy.** DeepSeek is correct
that on a *daily series with long history* the significance t is huge (SR 0.5 over 1000d → t≈16); power is
NOT binding there. Gemini/Grok are correct that the *old path-Sharpe t (N_eff=8 folds)* and the *4y options
window* are genuinely underpowered. **Resolution:** the path-t was the wrong, underpowered estimand — now
retired; Ruler-v2 scores the *pooled-OOS HAC* (n = thousands of days), which is well-powered on deep history.
→ **Implication:** the candidates that died on 19y data (XS-ML, PEAD) died on **zero edge**, not power.
So every new bet must be **(a) on deep history AND (b) backed by a real economic mechanism** — not a search.

### 2.2 Retest carry, or is it dead? (4 say retest properly; Gemini says dead)
**Worth a proper retest, eyes open — but it is NOT the #1.** The prior kill used *distribution yield*, which
is not carry; proper carry = curve roll-down / term structure / rate differentials (FRED + ETFs). BUT:
(a) it is crisis-correlated / fair-weather (Opus, ChatGPT both flag carry crashes *with* equities — the
opposite of what we need next to trend); (b) it likely **overlaps** what TSMOM already harvests (trend goes
long bonds/gold when they trend); (c) free-data carry proxies are crude (DeepSeek's own pre-mortem). →
**Medium priority, sized small, judged on CRISIS correlation, not the average.**

### 2.3 The #1 new sleeve (trend-breadth / carry / calendar / relative-value all nominated)
Owner's rubric: rank by *(orthogonality-to-trend × power-on-free-data × not-already-spent × cheap/owned-data)*.
- **Trend breadth (Opus #1) — largely SPENT.** Our live 10 ETFs are *already* cross-asset (equities, intl,
  bonds, gold, commodity, USD), and P5 already tested broadening (+HYG/LQD/SHY/SLV/VGK/EWJ + L/S) and
  failed. Genuine extension needs *futures* (Norgate + a futures account) — a data/infra spend. Opus's own
  conditional ("if the 10 are already cross-asset, pivot") fires here. **→ Defer to a futures-gated phase.**
- **Structural / calendar / overnight premia (Gemini) — highest-EV NEW bet.** Time-fixed (FOMC,
  turn-of-month) and overnight-gap premia are *maximally orthogonal* to a price-trend signal, *structural*
  (institutional flows / overnight risk transfer), *deep-history & high-event-count* (→ power), *liquid-ETF*
  (no spread wall), owned data, and **not already tried.**
- **Slow ETF relative-value (ChatGPT) — strong #2.** Slow log-spread mean-reversion between economically
  linked liquid ETFs (QQQ/SPY, IWM/SPY, HYG/IEF, TLT/IEF, GLD/UUP). Orthogonal (mean-reversion vs trend),
  deep-history, **not** the killed high-turnover single-name reversal. Risk: hidden beta / "cheap gets
  cheaper" / only-works-in-2008.
- **Aggregate short-interest market-timing (Opus #2) — real but data-constrained.** The *aggregate* SI index
  (Rapach-Ringgenberg-Tu) is a different object from the dead *cross-sectional* SI factor — a positioning
  signal, maximally trend-orthogonal. BUT our SI data is ~9y, paid, and ending; using it well needs a FINRA
  public backfill (to ~2005) and it's low-breadth (Track-B only). **→ Promising but gated on a data-backfill
  task; not the first move.**

### 2.4 A book-level GOVERNOR, not a sleeve (Opus #3a, ChatGPT #2)
A **VIX-term-structure / fast-crash de-risking overlay** (free `^VIX`/`^VIX3M`, ~15y) that cuts book gross
in stress is high-EV/low-risk and improves Calmar/left-tail. It is an *overlay on the whole book*, not a
sleeve. **Do it alongside F1.** (The short-vol *sleeve* it could gate is dangerous — deferred, see §5.)

## 3. The architectural insight (the future-proof investment the owner is prioritizing)

Every strategy so far has been a **bespoke per-idea script** (`run_*_cpcv.py`, and the one-off
`run_trend_broaden_rulerv2.py` written during the sweep). That is the bug-prone, non-scalable pattern that
will throttle a multi-sleeve book. **The future-proof move — "best, not fast" — is to consolidate the
research→gate→book flow into a uniform, hardened, tested "Sleeve Lab" before piling on more sleeves.**

**A "sleeve" becomes a small uniform declaration:**
- a **daily-return producer** (returns + dates; reuses the proven `gate_calibration.SeriesReturnStrategy`
  + `run_cpcv` path so OOS is genuine, leak-free, regime-mapped),
- a **`component_type`** (`alpha` | `diversifier` | `risk_premium` | `overlay`),
- a **pre-registered acceptance** row in the research registry (R4 one-shot).

**One tested harness then runs ANY sleeve through:** CPCV → Ruler-v2 **Track-A** (standalone) + **Track-B**
(appraisal IR + block-bootstrap P(ΔSR>0) vs the *current book*) → `sleeve_allocator` book-delta → a uniform
**sleeve report** (the same shape for every idea). The pieces already exist (SeriesReturnStrategy, run_cpcv,
ruler_v2, track_b_appraisal, sleeve_allocator) — the work is to **unify + harden + test** them into one
`sleeve_lab` module + a sleeve registry, retiring the bespoke scripts.

**Why this is the right first investment:** it makes every subsequent bet (F1–F4) a small, uniform,
*hardened* declaration instead of a fresh bespoke script; it removes the per-script bug surface; it makes
the book trivially extensible to *any* future idea; and it forces every sleeve through the identical,
audited gate. This is precisely the "future-proof, flexible to new ideas, hardened against bugs" the owner
asked for — and it is cheap because 80% of the substrate already exists and is tested.

## 4. The game plan (phased; pre-registered; kill-fast)

| Phase | Focus | Why / source | Effort |
|---|---|---|---|
| **F0** | **Sleeve Lab foundation** — unify the sleeve research→Ruler-v2(A+B)→allocator→report pipeline into one tested module + a sleeve registry; retire bespoke `run_*` scripts. | The future-proof substrate (§3). | 1–2 wks |
| **F1** | **Structural premia + crash governor** — turn-of-month / FOMC / overnight-gap premia (sleeves) + a VIX-term de-risking overlay (governor). Most orthogonal, most powered, cheapest, owned data. | Gemini #1/#2, Opus #3a, ChatGPT #2 | 1–2 wks |
| **F2** | **Slow ETF relative-value** — pre-registered log-spread mean-reversion across ~6–8 economically-linked ETF pairs; slow, vol-targeted, low-turnover. | ChatGPT #1 | ~1 wk |
| **F3** | **Carry done right (small)** — rates/curve roll-down (FRED, decades) + FX rate-diff; skip commodity (no clean futures). Judge on CRISIS correlation; expect overlap with trend. | DeepSeek #1, Grok #1, Opus #4, ChatGPT #3 | 1–2 wks |
| **F4** | **Options-conditioned event interaction (long shot)** — continuous, pre-registered interaction regression on the owned event panel × frozen options features (NOT a threshold filter, NOT path-t; use `event_inference` two-way CGM). | DeepSeek #2, Grok #2, ChatGPT #4 | 1–2 wks |
| **F5** | **Book assembly + live fidelity** — when ≥2 sleeves pass Track-B, run the `sleeve_allocator` book CPCV; replay-diff live-vs-backtest before any scaling. | all | ongoing |

**Each sleeve:** ONE pre-registered design (no sweeps — the OPT-5 lesson), CPCV on the deepest window the
data allows, Track-B vs the *current* book (appraisal IR ≥ 0.20, P(ΔSR>0) ≥ 0.90, corr < 0.30, tail-overlap
ok), kill-fast if it doesn't add book value *outside* a single crisis window. Build → independent Opus
deep-dive → fix → tests → no-drift docs → merge (the standing discipline).

### Deferred (data/infra-gated or dangerous — revisit only when a precondition is met)
- **Cross-asset trend via futures** — needs Norgate (~$30–60/mo) + a futures account (Alpaca has none).
  Buy ONLY if an ETF-validated cross-asset screen justifies the spend (and it doubles as the small-cap-PEAD
  data door — a second payoff).
- **Aggregate short-interest market timing** — needs a FINRA public short-interest backfill to ~2005; then
  a Track-B overlay. Promising (most trend-orthogonal owned signal) but gated on the backfill.
- **Index VRP via ETP** — genuinely dangerous (nickels-in-front-of-a-steamroller); only as a tiny, gated,
  defined-risk overlay AFTER the VIX-term governor is proven, ≤10% budget, ≤2% NAV tail. Last, or skip.

## 5. What we will NOT pursue (consensus kill list — do not re-tread)
XS-equity ML ranking · intraday 5-min ML · single-name options (signal OR execution) · dispersion ·
large-cap PEAD revival / PEAD threshold filters · frozen-options decile sorts (H4) · the cross-sectional
short-interest factor (reversed in the meme era) · analyst-drift / insider clustering (noise) · binary
threshold sweeps as validation · random trend-broadening (P5 failed) · cross-asset carry's **commodity** leg
(no clean futures) · single-name VRP/condors · paid data without a free proof-of-concept · more gate
machinery beyond what's shipped (the rigor is sufficient; the marginal hour is now worth more on
breadth/data than on referee-polishing — Opus's sharpest meta-point).

## 6. Honest bottom line
On free daily US data the cross-sectional IC is ≈ 0, so **breadth is the only lever and trend is the
canonical breadth play.** The realistic prize is a **0.7–0.9 book SR** from trend + 1–2 genuinely
orthogonal diversifiers — not a home run, and that is a fully respectable outcome for a solo, free-data
book. The single most valuable thing we can build is **not another model — it is the Sleeve Lab** that makes
testing the next ten ideas cheap, uniform, and hardened, plus the two cheapest orthogonal premia (structural
+ relative-value) and a crash governor. If those fail too, the system will have *honestly* mapped the edge
of its data envelope — at which point the decision is a deliberate data-acquisition bet (Norgate futures),
not more searching. **The next move: Phase F0 (Sleeve Lab), then F1.**

---

*Panel inputs (archived):* `docs/archive/llm-reviews/2026-06-14/` — `01_PROMPT.md`, `02_STATE_SNAPSHOT.md`,
the data/gate docs, and `responses/` (Opus 4.8, ChatGPT, DeepSeek, Gemini, Grok).
