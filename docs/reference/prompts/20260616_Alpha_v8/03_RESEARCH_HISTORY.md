# 03 — Research history: what we've tried and the verdicts

*The honest ledger. Full journal: `docs/living/ML_EXPERIMENT_LOG.md` (in-repo). This is the curated
summary an external reviewer needs.*

Legend: **LIVE** = trading paper capital · **CANDIDATE** = passed gate, owner-gated off ·
**PARK** = inconclusive/insufficient power, may revisit · **KILLED** = failed gate or stability guard.

---

## A. The one thing that worked

### TREND — 10-ETF time-series momentum  → **LIVE**
- **What:** TSMOM on ~10 liquid ETFs (equity/bond/commodity/intl). Lookbacks (21, 63, 126, 252),
  60-day vol-target 0.10, weekly rebalance, longs+shorts allowed, book-level vol target.
- **Result:** the **only** validated standalone edge. Deep-history Sharpe **≈ 0.7** (positive in both
  halves, survives CPCV + Ruler-v2). Now live at 25% allocation.
- **Why it survives:** trend/TSMOM is a well-documented, economically-grounded premium with deep free
  history (ETF EOD) — enough breadth × history to clear the power floor that kills everything else.

---

## B. Additive equity-signal attempts — all KILLED

### Swing ML (daily features → forward returns)  → **KILLED (×4)**
- GBT models over daily price/volume/technical features (+ optional fundamentals, macro context).
- **Verdict:** out-of-sample **IC ≈ 0**. No additive equity alpha from free daily US data. Re-confirmed
  on multiple independent rebuilds. Infra is sound; the signal isn't there. This is the single most
  repeated finding and the basis for our "free daily US equity data is mined out" conclusion.

### Intraday ML (5-min bars)  → **KILLED / STOP-listed**
- Intraday simulators + 5-min Polygon bars (~2y depth). Multiple intraday signal families explored.
- **Verdict:** no edge that clears the (higher) intraday gate; further work gated by both **data depth**
  (~2y) and **execution realism** (fills/slippage at intraday frequency on retail infra). Recommended
  NOT to pivot here without better data + execution modeling.

### PEAD / earnings-event panel (H1)  → **DEMOTED**
- Post-earnings-announcement-drift, event-level inference on an earnings-event panel.
- **Verdict (commit #456):** H1 = **PEAD DEMOTED** — the drift didn't survive event-level inference at
  our bar after costs. Code remains but contributes no live capital.

### Options-conditioned signals (F4)  → **mostly nothing**
- Signals conditioned on the options surface (4y frozen Polygon options snapshot): skew/IV/flow proxies
  as features or event filters.
- **Verdict:** no clean additive edge survived; options data depth/quality (frozen 4y, no live NBBO/OI
  history at retail) is a binding limitation.

### ETF relative-value / stat-arb (F2)  → **KILLED**
- Cross-sectional relative value among ETFs.
- **Verdict:** failed the correlation/standalone-return bar; nothing additive over trend.

### Carry (F3)  → **KILLED by the both-halves guard**
- Cross-asset carry sleeve.
- **Verdict:** looked plausible in aggregate but was **not positive in both halves** of the sample →
  killed by the mandatory stability guard. (Carry is the real lever in futures, but it's untestable on
  free data — see futures POC below.)

### Short-interest timing overlay (G2)  → **KILLED**
- Aggregate short-interest index (PIT trailing-z from bi-monthly short-interest data to 2017) used as a
  de-risking timing signal.
- **Verdict:** no marginal value on the trend book; killed.

---

## C. Overlays (timing/de-risking the trend book) — where marginal value showed up

### Crash governor — VIX term-structure (F1b)  → **LIVE**
- Cuts trend gross exposure when the VIX term structure inverts (VIX/VIX3M stress), floored at 25%.
- **Verdict:** marginal improvement to the trend book's drawdown/Sharpe; **live (flag ON)**. (Was
  caught & fixed for a look-ahead bug — it now reads settled closes only.)

### Credit-selective overlay (G1)  → **CANDIDATE (flag-OFF, owner-gated)**
- De-risk trend when credit spreads / curve signal stress (`CreditGovernorConfig(lookback=120,
  band=0.02, derisk_to=0.5)`).
- **Verdict:** passed the marginal-stacking gate as a CANDIDATE; **wired but flag-OFF** pending an owner
  decision on whether to enable.

### Credit *timing returns* as a sleeve (G3)  → **PARK**
- Tried to turn the credit signal into a standalone return stream.
- **Verdict:** insufficient power / inconclusive → parked.

---

## D. The futures-trend POC (free data)  → **weak; carry untestable free**
- Motivation: trend works; futures add carry + breadth. Ran a $0 yfinance continuous-futures (`=F`)
  trend POC before considering paid data (Norgate, ~$270/yr).
- **Verdict:** weak — SR **≈ +0.14**, correlation to existing book **≈ +0.43** (too correlated, too
  weak). Winsorization showed the weakness is *sustained*, not data spikes. The real futures lever —
  **carry/roll-yield** — is **untestable on free yfinance data** (dirty continuous series, no roll
  schedule). Conclusion: to pursue futures seriously we'd need Norgate-quality data + a second broker
  (IBKR; Alpaca has no futures).

---

## E. The early-harness-bug audit (important methodological context)
A natural worry: many swing/intraday "no edge" verdicts came early, when the harness had bugs — did we
**false-negative** a real edge?
- We found ~**23 bugs** in the early harness over time.
- **Crucially, the bugs INFLATED results (false positives), not deflated them** (look-ahead/leakage
  makes strategies look *better*, not worse). So the risk was accepting noise, not rejecting signal.
- We re-ran the suspicious kills on the corrected harness: **the re-validation gap is ZERO** — no edge
  that we killed came back as real. The early gate's real flaw was the opposite (Type-II over-killing),
  which Ruler-v2 addressed. **Net: our kills are honest; if anything we were too generous early.**

---

## F. What's arguably worth re-trying (our own list — challenge it)
- **Trend, deeper & broader:** more instruments/asset classes (needs paid futures data; carry).
- **Overlays generally:** the credit overlay is a candidate; more *timing/de-risking* overlays on the
  trend book (vs. additive sleeves) is where marginal value has appeared.
- **Options structures (not signals):** maybe the edge is in *structures* (defined-risk premium
  selling, hedged carry) rather than options-as-features — we mostly tried the latter.
- **Crypto:** Alpaca supports it; we've barely touched it; trend/carry may travel there.
- **Anything that needs data we don't have** — tell us what to buy.

**The meta-question for you:** given that only trend survived 6 months of rigorous search on free
daily US data, is the right move to (a) go buy better data, (b) change asset class, (c) change horizon/
frequency, (d) change the *type* of edge (structure/vol/event vs. directional), or (e) accept that one
good trend book + protective overlays IS the realistic retail outcome? We want your honest take.
