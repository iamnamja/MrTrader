# 09 — Architecture Design: The Portfolio Brain

## The North Star Design
The `PM -> RM -> Trader` abstraction is a relic of discretionary trading desks and the wrong model for a systematic, low-frequency book. A state-of-the-art systematic architecture operates as a mathematical pipeline: `Signals -> Targets -> State -> Trades`. The new North Star is an atomic, event-driven pipeline decoupled from the web server. It treats strategies purely as dumb signal generators, builds a single consolidated state of the world, applies top-down portfolio construction (sizing and netting), and delegates execution to stateless broker adapters. Everything runs deterministically, logged to a single `cycle_id`, allowing a solo operator to mathematically prove why any trade was placed.

---

### D1 — The Target Architecture

The pipeline replaces the PM/RM/Trader model with four distinct, strictly sequential phases. 

* **1. The Signal Generators (formerly Strategies):** Strategies do not size themselves and do not trade. They simply emit a `DesiredExposure` vector (e.g., "I want 0.5 risk units of SPY").
* **2. The State Aggregator (The Brain's Eyes):** Reconciles reality before any logic runs. It queries all `BrokerAdapters` to construct the `BookState` (Current Positions, Cash, Margin).
* **3. The Portfolio Constructor (The Brain's Core):** Takes `BookState` and `DesiredExposures`. It applies the global risk budget, scales the strategy weights, nets overlapping exposures, and outputs the `TargetBook` (the exact nominal positions we want).
* **4. The Execution Planner & Router:** Diffs the `TargetBook` against `BookState` to generate a minimal list of required `Orders`. It routes these to the correct `BrokerAdapter`.

**Topology & Execution:** Decouple this from the FastAPI web server immediately. Run a separate `portfolio_daemon` process. Since this is a weekly book, it should be a hybrid event-driven cron: APScheduler wakes the daemon up, the daemon generates a unique `cycle_id`, runs the atomic sequence `Generators -> Aggregator -> Constructor -> Planner`, writes the results to Postgres, and goes back to sleep.

---

### D2 — Consolidated Book State (Single Source of Truth)

The "DB is not reality" problem is solved by making the broker the indisputable source of truth for positions and cash, while the DB serves as the source of truth for *intent* and *history*. 

* **The State Object:** The `PortfolioState` object is instantiated fresh every cycle in memory. It contains `venues` (Alpaca, IBKR), `total_equity`, `margin_used`, and a unified list of `Position` objects mapped to common identifiers.
* **"What's being considered":** This is no longer scattered across Redis. It is encapsulated entirely within the `cycle_id` execution context as the `TargetBook`. If the pipeline halts before execution, nothing is pending. 
* **Reconciliation Rules:** On wake-up, the `StateAggregator` queries `db.get_expected_positions()` and `broker.get_actual_positions()`. If $\Delta_{pos} \neq 0$, the system fails closed, alerts the operator, and halts the cycle. You must manually intervene. Do not auto-reconcile live capital.

---

### D3 — Holistic Sizing & Risk (Diversification without Fragility)

Optimizers that rely on covariance matrix inversion (like Markowitz) are dangerously fragile to correlation spikes during crises. For a solo operator, simplicity is survival.

* **The Robust Recipe:** Use Inverse Volatility weighted by a structural correlation penalty, governed by a hard realized-correlation cap. 
* **Step 1 (Base Weighting):** Size strategies inversely proportional to their realized volatility, scaling up to the total book target vol.
* **Step 2 (The De-Gross Trigger):** Calculate the trailing 20-day realized pairwise correlation of the live streams. If the average correlation $\rho_{realized}$ crosses a threshold (e.g., 0.65), apply a scalar multiplier $M_{degross} \in (0, 1]$ to the total book leverage. You do not re-optimize; you simply shrink the gross exposure. 
* **Paper to Live Ramp:** Add a `live_weight` column to the `strategies` table. Paper strategies run at $0.0$. When promoting to live, the operator updates this to $0.25$, then $0.50$, then $1.0$. The Portfolio Constructor automatically multiplies the strategy's target exposure by this fractional weight.

---

### D4 — Netting, Conflicts, and Factor Views

When strategies disagree, honor the net, not the independent risk takers. Paying spread twice to hold overlapping exposure is institutional waste.

* **The Netting Rule:** The Portfolio Constructor aggregates all `DesiredExposures` by mapped underlying instrument. If Strategy A wants +2 ES contracts and Strategy B wants -1 ES contract, the `TargetBook` demands +1 ES contract. 
* **Factor Exposure Gate:** Map every instrument to its beta, duration, and USD exposure. Before generating orders, the Constructor calculates the total netted factor load. 
* If the aggregate equity beta exceeds the system limit, the Constructor applies a pro-rata shrinkage to all beta-positive targets until the book is within limits. 

---

### D5 — Broker Abstraction & The Kill Switch

You are adding IBKR futures alongside Alpaca equities. This requires a strict abstraction layer.

* **The Interface:** Create a standard Python `Protocol` or Abstract Base Class: `BaseBrokerAdapter`. It requires exactly three methods: `get_positions()`, `submit_orders(orders)`, and `liquidate_all()`.
* **Venue Routing:** The `ExecutionPlanner` maps asset classes to venues. Equities $\rightarrow$ AlpacaAdapter. Futures $\rightarrow$ IBKRAdapter.
* **The Dead-Man / Kill Switch:** A standalone, decoupled script `kill_switch.py`. When triggered (via dashboard or CLI), it sets a Redis flag `EMERGENCY_HALT=1`. It then iterates through all registered `BrokerAdapters` and concurrently fires `adapter.liquidate_all()`. The main pipeline checks `EMERGENCY_HALT` at the start of every phase and aborts if true.

---

### D6 — Extensibility & Parity

Adding a strategy or venue must be a configuration change, not a structural one.

* **Declarative Strategy Registry:** Use a decorator pattern. `@register_strategy(name="futures_carry", asset_class="futures", venue="ibkr")`. The daemon auto-discovers these at startup. 
* **Research vs. Live Parity:** The live system and the backtester must import the exact same `generate_signals()` function from the strategy module. The only difference is the data injected: the backtester feeds historical dataframes, while the live daemon feeds the live `BookState`.

---

### D7 — Safety & Determinism

For a solo operator, auditability is more important than speed. 

* **The `cycle_id` Schema:** Every time the pipeline runs, it generates a UUID. Every database insert (`state_snapshots`, `target_exposures`, `orders`, `fills`) includes this `cycle_id`. 
* **Idempotency:** If the pipeline crashes midway, restarting it does not duplicate orders. The `ExecutionPlanner` sees the `TargetBook`, compares it to the fresh `BookState` (which reflects any already-executed fills), and only issues orders for the delta.
* **Failure Modes Feared:** * *Stale Data:* Guardrail = Check timestamp of last tick/EOD close. If > threshold, abort.
    * *Sizing Math Error (Fat Finger):* Guardrail = Hard-coded global constraints in the `ExecutionPlanner` (e.g., `MAX_NOTIONAL_ORDER_SIZE = $50,000`). This overrides any output from the Constructor.

---

### D8 — LLMs in the Loop

LLMs are brilliant at unstructured synthesis and terrible at arithmetic. Keep them strictly out of the deterministic execution path. 

| Role | Status | Integration & Guardrails |
| :--- | :--- | :--- |
| **Sizing / Execution** | **LLM-OUT** | Absolute no-fly zone. Determinism and reproducibility are mathematically impossible with an LLM. |
| **Narrative Risk Monitor** | **LLM-IN** | Bounded alert only. Scans weekend macro/news. If it detects a "regime break," it can flag the dashboard and pause the cron. It *cannot* autonomously liquidate or place orders. |
| **Post-Trade Analyst** | **LLM-IN** | Offline cron job. Takes the JSON logs of the weekly `cycle_id` and the broker execution report to write a human-readable summary of slippage and intended vs actual book. |

**Verdict:** Do not build the narrative/anomaly layer yet. It is a distraction. Build the deterministic pipeline first. Once you have a unified JSON state object, plugging an LLM into it for post-trade analysis is trivial.

---

### D9 — The Migration Path (Strangler Fig)

Do not execute a big-bang rewrite while ETF-trend is live.

1.  **Phase 1: The Observer (Weeks 1-2).** Build the `StateAggregator` and the `BrokerAdapters` (Alpaca first, IBKR second). Have them run alongside the current setup, logging the consolidated state to the DB. Ensure reconciliation matches reality.
2.  **Phase 2: The Pass-Through (Weeks 3-4).** Build the `ExecutionPlanner`. Point ETF-trend's output to the `TargetBook` schema. The new pipeline handles ETF-trend routing to Alpaca, while bypassing the complex Portfolio Constructor (pass-through sizing). 
3.  **Phase 3: The Brain (Weeks 5-6).** Build the `PortfolioConstructor` with the global risk limits and factor mapping. Ramp up the paper futures strategies via the $w_{live}$ multiplier. 
4.  **Phase 4: Cleanup.** Delete the old PM/RM/Trader code and the FastAPI scheduler loops.

---

### The Closer

**The North Star:** A purely deterministic, stateless execution pipeline that maps decoupled strategy signals into a unified portfolio state, applies rigid top-down risk and netting limits, and executes via uniform broker interfaces, with every action mathematically traceable to a single execution cycle.

**The First Three Things to Build:**
1.  The `BaseBrokerAdapter` interface and the specific Alpaca/IBKR implementations. 
2.  The `StateAggregator` to prove you can pull and unify positions/cash accurately across both venues.
3.  The `TargetBook` schema and the `ExecutionPlanner` to diff state against target. 

**The Single Biggest Mistake:** Trying to shoehorn multi-venue logic into the existing `Trader` agent while keeping the in-memory fast-paths. It guarantees a state-desync that will eventually liquidate the wrong asset or double-buy a position. The design above avoids this by enforcing a stateless, query-reality-first loop.