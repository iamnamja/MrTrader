"""
risk_policy.py — Alpha-v10 R0.1: the pre-registered, frozen RISK POLICY v1 for the portfolio brain.

This is the single source of the book-level risk constants the governed sizing path will read
(the R0.4 whole-book risk gate; later the R2 Constructor inherits them). It is a PURE data artifact
— it controls nothing on its own and is safe to land in shadow. Values are deliberately CONSERVATIVE
because we have ZERO live multi-strategy track record (per the Go-Live synthesis: launch low, raise
only on live evidence; size by drawdown / vol / margin, NEVER Kelly).

Sources: GOLIVE_REVIEW_SYNTHESIS_2026-06-21.md (sizing verdict) +
PORTFOLIO_BRAIN_ROADMAP_2026-06-21.md (Part III + R0.1). Any change here is a deliberate,
versioned policy decision — bump `POLICY_VERSION` and log it in DECISIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

POLICY_VERSION = "v1.0-2026-06-21"


@dataclass(frozen=True)
class RiskPolicy:
    """Frozen book-level risk policy. Conservative launch values; widen only with live evidence."""
    version: str = POLICY_VERSION

    # --- book volatility / drawdown budget (drawdown-anchored, NOT Kelly) ---
    book_vol_target_launch: float = 0.06        # ~6% annualized at launch (no live multi-strat track)
    book_vol_target_steady: float = 0.08        # ~8% steady-state, raised only on live evidence
    book_vol_hard_cap: float = 0.10             # never run the book above ~10% ann vol
    max_drawdown_budget: float = 0.20           # -20% from HWM = the kill line

    # --- global drawdown de-risk ladder (gross multiplier by drawdown-from-HWM) ---
    # applied as: gross *= ladder_multiplier(current_drawdown); asymmetric re-risk with hysteresis
    drawdown_ladder: Dict[float, float] = field(default_factory=lambda: {
        -0.08: 0.75,    # -8%  -> cut gross 25%
        -0.12: 0.50,    # -12% -> cut gross 50%
        -0.16: 0.25,    # -16% -> cut gross 75%
        -0.20: 0.00,    # -20% -> flat / halt (the kill line)
    })
    rerisk_hysteresis_days: int = 20            # restore one rung only after ~20 days with no new low

    # --- realized-correlation / absorption de-gross (stress-conditional, NOT 63d average) ---
    corr_spike_degross_at: float = 0.60         # stress-conditional pairwise corr > 0.60 -> cut 25%
    corr_spike_degross_strong: float = 0.70     # > 0.70 (+ drawdown) -> cut 50%

    # --- per-venue margin ceilings (futures; per venue, never netted across brokers) ---
    margin_to_equity_target: float = 0.20       # maintenance margin <= ~20% NAV target
    margin_to_equity_hard: float = 0.25         # hard stop (cut futures gross clear of this)
    ibkr_min_cash_reserve_frac: float = 0.10    # reserve >= 10% NAV in cash AT IBKR (Alpaca cash
    #                                             cannot fund IBKR margin — different brokers)

    # --- hard book-level factor caps (netted across venues) ---
    max_net_equity_beta: float = 1.00           # |net equity beta| in SPY-equivalents
    max_gross_ex_cash: float = 0.80             # gross risk notional ex-cash <= 80% NAV (launch)

    # --- ABSOLUTE notional sanity caps (the dumb backstop, live from the first live phase) ---
    max_single_instrument_notional_frac: float = 0.25   # any one instrument <= 25% NAV notional
    max_book_notional_frac: float = 3.00        # total book notional <= 3x NAV (futures are margined)

    # --- per-strategy fractional paper-ramp (live_fraction steps; the step to real money is
    #     human-confirmed — the one irreversible move) ---
    ramp_steps: tuple = (0.0, 0.25, 0.50, 1.0)
    ramp_min_clean_weeks: int = 8               # >= 8 clean weekly runs before the next ramp step
    ramp_max_slippage_frac_of_edge: float = 0.30  # slippage must stay < 30% of modeled edge

    def ladder_multiplier(self, drawdown_from_hwm: float) -> float:
        """Gross multiplier for a given drawdown (<= 0). Picks the deepest breached rung."""
        mult = 1.0
        # iterate least-negative first so a DEEPER breached rung overwrites a shallower one ->
        # the deepest breached rung's (smallest) multiplier wins.
        for thresh in sorted(self.drawdown_ladder, reverse=True):
            if drawdown_from_hwm <= thresh:
                mult = self.drawdown_ladder[thresh]
        return mult

    def to_dict(self) -> dict:
        return asdict(self)


# the canonical singleton (import this)
RISK_POLICY_V1 = RiskPolicy()
