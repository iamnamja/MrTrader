"""
Frozen interface contracts for the options program (Alpha-v5) — OPT-0.

These are the stable seams that make the program resilient: a new options strategy is a
new scorer + a run_*.py, and NOTHING below the strategy adapter changes. Implementations
arrive in later phases (pricing engine OPT-1a, data provider OPT-1b, simulator OPT-2,
strategy adapter OPT-3). Defining them as Protocols up front locks the design and lets
each layer be built + tested in isolation.

Layering (each ⟂ the next):
    [DATA] ⟂ [PRICING ENGINE] ⟂ [SIMULATOR+COST] ⟂ [STRATEGY SPEC] ⟂ [REUSED gates/allocator/live]
     durable      durable             durable           disposable        already built

See docs/living/OPTIONS_PROGRAM.md for the full charter, confidence plan, and verdict table.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

OptionKind = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


# ─────────────────────────────────────────────────────────────────────────────
# Contract 1 — OptionsDataProvider (PIT, as-of). Mirrors short_interest_provider:
# every historical accessor is point-in-time (filters knowable_date <= as_of). The
# CURRENT snapshot is used ONLY for engine validation + live execution (it carries
# Polygon-served IV/greeks/OI that are NOT available historically).
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class OptionsDataProvider(Protocol):
    def get_universe(self, underlying: str, as_of: date,
                     include_expired: bool = True) -> List[str]:
        """OCC contract tickers (e.g. 'O:SPY260630C00720000') listed as of `as_of`.
        include_expired=True is the survivorship cure (uses the expired contract universe)."""
        ...

    def get_contract_bars(self, underlying: str, as_of: date):
        """PIT panel of per-contract EOD OHLCV for `underlying`, all rows with
        knowable_date <= as_of. Columns include (contract, date, open/high/low/close/
        volume, knowable_date). No historical IV/greeks/OI (computed by the engine)."""
        ...

    def get_current_snapshot(self, underlying: str) -> Dict[str, Dict[str, Any]]:
        """CURRENT chain snapshot {contract: {implied_volatility, greeks, open_interest,
        day_close, underlying_price, ...}}. Validation/live ONLY — never backtest."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Contract 2 — OptionsPricingEngine (PURE, no I/O). Computes the historical IV +
# greeks Polygon doesn't serve. BS-European fast path + Bjerksund-Stensland for
# American exercise; OPT-0 spike showed BS near-ATM IV matches served IV to <1
# vol-point, so American+dividends mainly tighten the ITM/OTM tails.
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class OptionsPricingEngine(Protocol):
    def price(self, S: float, K: float, T: float, r: float, q: float, sigma: float,
              kind: OptionKind, style: ExerciseStyle = "american") -> float:
        """Theoretical option price."""
        ...

    def implied_vol(self, price: float, S: float, K: float, T: float, r: float, q: float,
                    kind: OptionKind, style: ExerciseStyle = "american") -> Optional[float]:
        """Implied vol backed out of `price`; None if outside a sane bracket."""
        ...

    def greeks(self, S: float, K: float, T: float, r: float, q: float, sigma: float,
               kind: OptionKind, style: ExerciseStyle = "american") -> Dict[str, float]:
        """{'delta','gamma','theta','vega','rho'}."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Contract 3 — OptionsSpreadCostModel + OptionContractSim. No historical NBBO, so
# the sim marks off EOD close and the cost model imposes a MODELED spread (% of
# premium) with a mandatory stress multiplier (KEEP must survive 2x). The sim emits
# the SAME SimResult shape as the equity sims so all downstream gates reuse verbatim.
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class OptionsSpreadCostModel(Protocol):
    def entry_exit_cost(self, premium: float, spread_mult: float = 1.0) -> float:
        """One-way cost for opening/closing a leg at `premium` (% of premium × mult
        + per-contract fee). No exit cost when held to expiry/assignment."""
        ...


@runtime_checkable
class OptionContractSim(Protocol):
    def run(self, positions_spec: Any, start: date, end: date,
            spread_mult: float = 1.0):
        """Backtest a sequence of (defined-risk) option positions. Marks daily to the
        engine, handles IV-crush (carried by the IV data), payoff caps, expiry/
        assignment, ex-div early exercise. Returns a SimResult
        (app/backtesting/strategy_simulator.SimResult) with a daily equity curve."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Contract 4 — OptionsStrategy adapter. The ONLY disposable layer. Must duck-type
# EXACTLY to scripts/walkforward/event_edge.EventEdgeStrategy so run_cpcv / FoldEngine
# drive it unchanged and the significance gate + CAPM residual-α + fold-coverage all
# apply. A new strategy = a new scorer + run_*.py; nothing below this changes.
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class OptionsStrategy(Protocol):
    model_type: str
    is_trained: bool          # False -> rules-based; CPCV overlap guard bypassed (full coverage)
    per_fold_retrain: bool

    def fetch_data(self, start: date, end: date) -> None:
        """Populate self.symbols_data / all_days_sorted / spy_prices for the window."""
        ...

    def run_fold(self, fold_idx: int, n_folds: int, tr_start: date, tr_end: date,
                 te_start: date, te_end: date):
        """Run the options simulator over the test window; return a FoldResult
        (scripts/walkforward/gates.FoldResult) with daily_returns_dated populated."""
        ...
