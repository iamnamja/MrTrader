# Alpha-v7 Independent Quant Review — Gemini

> Synthesis note (added at archival): Gemini's response largely *describes the existing
> system back* (and treats already-completed Alpha-v6 work as future plans), so it carries
> low new signal relative to the other three. Preserved verbatim for the record.

## Q1: Gate Parameters, Thresholds, and Multiplicity
The program's gate mechanics have historically been too rigid, leading to a critical evaluation of how edges are accepted and how trial multiplicity is managed.

* **The Type II Power Problem:** The previous gate suffered from a severe Type II error rate, effectively killing true edges because it evaluated risk premia using standalone alpha standards. The gate's mathematical calibration demanded a $t \ge 2.0$ on a small number of folds, which is an unrealistic threshold for true edges with a Sharpe ratio of 0.5 to 0.7.
* **Two-Track Acceptance:** To fix this, the blueprint establishes a two-track framework. Track A is reserved for standalone alpha (event/cross-sectional strategies), while Track B evaluates book-level diversifiers and risk premia based on their combined contribution to the portfolio (e.g., $\Delta$Sharpe, joint-tail risk) rather than standalone metrics.
* **Research Registry and Multiplicity:** Instead of relying solely on deflated Sharpe ratios to manage multiplicity, the program now enforces a strict SQLite-backed research registry. This registry legally binds the research process by requiring pre-registration of acceptance criteria before a run executes (Rule 2) and strictly limiting tests to one shot per hypothesis (Rule 4).

## Q2: Roadmap and Data Strategy (What's Next)
The roadmap shifts away from direct options trading and leans heavily into feature engineering and live-book fidelity.

* **Options as Equity Signal:** Options data will no longer be traded directly via options spreads. Empirical evidence showed that the spread wall killed previous options strategies. Instead, options data will be harvested purely as a cross-sectional signal executed at equity costs.
* **The Earnings-Event Panel:** The centerpiece of the new alpha hunt (Phase 3) is a unified earnings-event feature panel. This creates a one-row-per-event research table to re-adjudicate PEAD and develop a continuous options-conditioned score, eliminating fragile binary thresholds.
* **Live Fidelity over New Capital:** Before any new capital is deployed, Phase 1 mandates making the live book "boring". This requires a four-week clean run of a replay-diff report and comprehensive fill-quality tracking to ensure live execution perfectly matches backtest expectations.

## Q3: Did the Process Bury an Edge?
Yes, the overarching consensus is that the previous process mathematically obscured true edges due to improper inference scaling and accounting artifacts.

* **Incorrect Inference Unit:** For event strategies, evaluating significance at the CPCV fold level buried edges. The true unit of independence is the announcement-day cluster.
* **Two-Way Clustered Inference:** To accurately measure edge, the pipeline now utilizes a dedicated event-level inference module. This applies two-way Cameron-Gelbach-Miller cluster-robust standard errors (clustered by announcement date and symbol) to the panel of hedged forward returns.
* **Simulator Accounting Fixes:** The historical simulation engine was vulnerable to artificial Sharpe inflation. The tier 3 agent simulator now explicitly subtracts short collateral from cash when calculating mark-to-market equity. Without this correction, opening short positions would artificially inflate the equity curve because the cash proceeds were recognized without the corresponding liability.
