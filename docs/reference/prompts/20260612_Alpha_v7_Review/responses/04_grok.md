# Alpha-v7 Independent Quant Review — Grok

## Q1: Gate parameters — mostly sound but the Type-II diagnosis is real and the power floor is the binding constraint.

Your significance-first two-tier gate (t ≥ 2.0/2.5, %pos ≥ 75%, P5 ≥ 0, mean ≥ 0.35/0.45 + backstops; Track B book-delta) is a strong improvement over the old mean-Sharpe ≥ 0.80/1.00 relic. The calibration run was decisive: on ≤4y/≈8-fold data the 8-fold path t-stat is not a reliable discriminator (3/5 true nulls cleared t≥2 by chance; TSMOM's t=6.72 died only on worst-regime). Lowering t* would admit noise. DSR report-only is correct — it saturates. The registry + pre-registration + one-shot R4 is excellent multiplicity defense.

Critiques / changes I'd make immediately:

- **Power is the real problem.** With N_eff≈6–8 a true SR 0.5–0.7 often fails t≥2 (t ≈ SR·√years). Require minimum 4 years / 12+ effective independent clusters (event strategies) or 10+ folds before any CAPITAL verdict. PAPER can be looser (t≥1.65, 3+ years) for forward validation. Your H1 PEAD demotion at event-panel level was the right call — the harness finally told the truth.

- **Worst-regime backstop is too strict for diversifiers.** Track B already recognizes this; make it explicitly waived for any candidate that declares component_type="diversifier" or "risk_premium" in the registry (with mandatory tail-overlap / joint-maxDD checks). Crisis-positive sleeves should have a bad regime.

- **Residual alpha t (CAPM/HAC) should be primary (not just diagnostic) for Track A**, with t_hac ≥ 1.8 required. PEAD's lesson is exactly this. Beta riding the bull is not alpha.

- **Deployment-adjusted Sharpe and low-deployment warning are good** — enforce avg deployment ≥ 30% or the result is report-only.

- **Event strategies get their own lane:** panel two-way CGM primary + CPCV robustness leg. Daily-fold t is the wrong instrument (your H1 confirmed this).

Overall: keep the framework, raise the data/power floor, make residual-alpha and Track-B more central. You're not too loose; you're appropriately strict on power-starved small samples. The "everything died" outcome is more likely honest than a process flaw.

## Q2: Next — stop equity-signal hunting; run the trend book boringly and expand into futures / vol-premium harvesting at equity cost.

You've exhausted the free US-equity opportunity set for cross-sectional / short-horizon edges. The data is too noisy, too crowded, too short-history for retail quant scale. The survivor (TSMOM) is the right base: multi-asset trend is one of the few things with 20+ year evidence that survives honest costs and regime tests.

Prioritized roadmap (solo operator, $100k paper, cheap data):

1. **Make the trend sleeve boring and scalable (2–4 weeks, highest EV).** Add more legs (currency-hedged intl ETFs, more commodities, rates duration), proper vol-targeting at sleeve level, and live replay-diff + fill-quality reconciliation. Verify 4+ boring weekly diffs before any capital. This is your capital base.

2. **Index VRP as tiny book diversifier (Track B, behind book gate).** Your OPT-4 PF was real. Use the frozen options store + Alpaca NBBO logger for calibrated spreads. One pre-registered defined-risk structure (30–45 DTE, 16Δ short strikes, weekly, regime-gated), ≤10% risk budget, tail-budgeted (≤2% NAV 2008-replay cost). No parameter search. Kill on first Track-B fail.

3. **Futures trend / carry expansion.** Futures give real shorting, better liquidity, lower costs, longer history. Start with equity index + bond + commodity futures (via broker API or cheap data). This is the natural extension of your surviving sleeve.

4. **Data buys only if specific:** I/B/E/S or Zacks estimate revisions only if you want to revive a forward-revision momentum sleeve (independent of killed PEAD). Otherwise skip — your free FMP + yfinance is sufficient for the above. No microstructure / alt-data at this scale.

Do NOT: another XS equity ML revival, single-name options, dispersion trading (Claude was right — cost wall), or binary threshold sweeps. The opportunity set for liquid US equity alpha at retail scale is thin. Run premia as base, hunt satellite edges with registry discipline.

Bottom line for Q2: Your premise was partly wrong — durable alpha for this setup is broad trend + vol-premium harvesting, not concentrated equity alpha. Accept it and execute.

## Q3: Process flaws that could have buried edges — yes, but your recent rigor probably didn't.

Your harness is among the best I've seen for a solo shop (registry + pre-reg + sacred holdout + CPCV + event-panel CGM + residual alpha + Track B). The "everything died" is mostly honest. But specific flaws that could still bite:

- **Survivorship bias in yfinance-heavy paths** (no delisted bars) flatters long-only / event studies. You patched with PIT membership for some, but uneven. Re-test any survivor with Polygon delisted-inclusive data.

- **Over-purging / short windows for event strategies** — your 85d purge + 8 folds starves power. Event-panel CGM was the correct fix (you did it for PEAD).

- **Beta hedging too aggressively** can kill real exposure premia. Residual alpha t is good; make it primary.

- **Cost modeling** — your 5–20bps is realistic for $100k, but verify with Alpaca fill data. If an edge dies only at 15bps+, it might be marginal but keepable at your size.

- **The deeper risk:** demanding standalone significance from things that only work in a book (diversifiers). Your Track B addresses this well — lean into it.

You probably didn't bury a real equity edge; the data says the liquid US equity set is efficient for your style. The process over-killed some marginal things (via power issues), but the big verdicts (PEAD demotion, options signals dead) look correct.

My first three moves if handed this system tomorrow:

1. Stabilize / broaden the TSMOM sleeve + live fidelity (replay-diff, NBBO spreads) — make the book boring.
2. Implement one pre-registered Track-B index VRP micro-sleeve behind the book gate.
3. Registry-enforce everything going forward; run no new equity-signal hypothesis without event-panel CGM + residual alpha.

Your discipline (pre-reg, one-shot, honest kills) is rare and excellent. The "everything died" outcome is probably truth, not artifact. Run the trend book, harvest the one real options asset (VRP as diversifier), and stop grinding equity signals. The edge was in accepting that.
