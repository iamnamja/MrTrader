# Walk-Forward Review Synthesis — 2026-05-23
*Synthesized by Opus 4.7 from 5 independent LLM reviews*

## Raw Reviews
- `raw/MrTrader_WF_Brutal_Review_claude.md` — Claude
- `raw/MrTrader_WF_Deep_Dive_Review_Grok.md` — Grok
- `raw/MrTrader_WF_Quant_Review_2026-05-23_chatgpt.md` — ChatGPT
- `raw/MrTrader_Institutional_Review_Prompt_Gemini.md` — Gemini
- `raw/deepseek_text_20260523_3da0a3.md` — Deepseek

---

## Consensus Findings (4+ of 5 reviewers agreed)

### 1. Diagnostic Ladder Was Never Used
Every reviewer flagged the same core failure: the system jumped straight to full agent stack (L4) without ever validating at signal level (L1/L2). PF=0.00 on every fold could have been diagnosed in days with a simple decile backtest that instead took months of agent/execution debugging.

### 2. PIT Compliance Is The Highest-Risk Assumption
All five reviewers flagged `pe_ratio`, `gross_margin`, and `revenue_growth` as high look-ahead risk. SEC filings are released 45-90 days after period end. If these features use announcement-date values, in-sample IC is meaningfully contaminated. OOS IC ≈ 0 is the expected symptom.

### 3. Stop-Loss Asymmetry Makes L4 Nearly Impossible With IC ≈ 0
With a 2:1 reward-risk ratio, you need ≥33% hit rate to break even. With IC ≈ 0, you get ~50% random noise minus costs and slippage. The stops aren't creating alpha — they're destroying it by increasing transaction frequency without improving win rate.

### 4. 2022 Signal Inversion Is Structural, Not Regime
IC@20d = -0.028 in 2022 (t=-8.73) is too strong and sustained to be noise. The features that predict positive returns in bull markets (momentum, quality) actively predicted negative returns in the 2022 drawdown. This is not something regime-conditioning fixes — it's a fundamental feature construction problem.

### 5. Purge Gap Is Too Short
10d + 10d = 20d total purge/embargo. With 60d feature lookback + 20d label horizon, the minimum defensible purge is 85d. Short purge overfits to correlated samples and inflates in-sample IC.

---

## Split Opinions

### Regime-Conditional Models
- **For (2/5)**: ChatGPT and Gemini suggested regime-conditional retraining to handle the 2022 inversion.
- **Against (3/5, including Opus synthesis)**: Claude, Grok, and Deepseek all noted this is HARKing — fitting a new hypothesis to the observed pattern. 2024 was also a bull year but IC was only +0.010 vs +0.023 in 2021. If the regime explanation were correct, 2024 would have similar IC to 2021. It doesn't. The regime hypothesis doesn't hold.

### Feature Rebuild vs Signal Measurement First
- **Feature rebuild (2/5)**: Deepseek and Gemini recommended immediately pivoting to new features (microstructure, options-derived).
- **Signal measure first (3/5)**: Claude, ChatGPT, and Grok recommended L2 decile spread as the first step — to confirm whether the current features have ANY signal before rebuilding.

**Opus decision**: Measure first. L2 takes 2 days. Feature rebuild takes 3+ weeks. If L2 shows Sharpe 0.40, that's buildable. If L2 < 0.20, THEN rebuild.

---

## Key Recommendations Adopted

1. **L2 decile spread as signal gate** (Sharpe threshold: 0.60 = signal exists, <0.20 = stop)
2. **PIT audit before ANY retraining** — if this fails, all results are contaminated
3. **Fix purge to 85d** — structural correctness regardless of signal outcome
4. **Policy-realized labels** (Phase 3) — training labels should match what the strategy actually does
5. **Null benchmark** (100-seed random portfolio) — gate requires 2σ outperformance

## Key Recommendations NOT Adopted

1. **Regime-conditional retraining** — HARKing. Rejected by Opus.
2. **Immediate feature pivot** — premature without L2 measurement.
3. **LightGBM over XGBRanker** — not enough evidence to prefer one over the other until L2 passes.
