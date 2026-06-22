"""
family_registry.py - Alpha-v10 P0.5: the strategy-family registry (the program's true N_TRIALS).

The 5-LLM panel's family-level-trial-counting point (ChatGPT): "rules-based sleeves are
OOS-by-construction" is FALSE at the *family-selection* level - across the program we've searched
~two dozen distinct strategy families, so the multiple-testing burden is real and was, until now,
an UNCOUNTED hardcoded estimate ("~20") feeding the GL-0 deflated-Sharpe cross-check.

This module replaces that guess with an AUDITABLE enumeration: every distinct strategy FAMILY the
program evaluated as a deploy candidate, with its status + one-line verdict + doc reference. A
FAMILY is a distinct economic hypothesis / signal construction (e.g. "futures carry", "PEAD"), NOT a
parameter variant (lookback 60 vs 120) - within-family search is captured by the empirical
max-stat null + the degrees-of-freedom log below, not by inflating the family count.

`family_trial_count()` is the principled N for the parametric Deflated Sharpe (Bailey & López de
Prado) cross-check in `null_zoo`. The PRIMARY GL-0 test remains the empirical selection-aware
max-stat null (which replicates the researcher's within-futures search directly); this family count
represents the BROADER cross-asset burden that the futures-only empirical null cannot see.

Report-only / research-integrity - touches no live trading path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# ── Status vocabulary ─────────────────────────────────────────────────────────
LIVE = "LIVE"                  # trading the live book now
PAPER = "PAPER_CANDIDATE"      # passed paper gate; not yet capital
KILLED = "KILLED"              # evaluated and rejected
PARKED = "PARKED"             # built + evaluated, held off (candidate flag / no-benefit)
SCAFFOLD = "SCAFFOLD"          # heavily tested, confirmed null, code dormant
_VALID_STATUS = {LIVE, PAPER, KILLED, PARKED, SCAFFOLD}


@dataclass(frozen=True)
class Family:
    id: str
    name: str
    asset_class: str
    status: str
    verdict: str
    doc_ref: str
    counts_as_trial: bool = True   # False for infra (cash) / ensembles (book) - not a search trial


# ── The registry - every distinct family searched as a deploy candidate ───────
# Sourced from ML_EXPERIMENT_LOG / DECISIONS / ALPHA_V{7,8,9,10} plans / GL0 findings (2026-06-22).
FAMILIES: List[Family] = [
    # ── Equities / ETF ──
    Family("etf_trend", "ETF trend (TSMOM)", "equity_etf", LIVE,
           "the only validated free-daily edge; post-2015 +0.77", "DECISIONS 2026-06-16 P1-2"),
    Family("pead", "PEAD earnings drift", "equity", KILLED,
           "event-level t=-0.77 (p=0.78) -> demoted, flipped off live", "DECISIONS 2026-06-12 H1"),
    Family("swing_ml_ranker", "Swing cross-sectional ML ranker", "equity", SCAFFOLD,
           "confirmed NULL (IC~0 annually); flag off, dormant", "PROJECT_STATE 2026-06-09 #418"),
    Family("intraday_ml", "Intraday ML (5-min)", "equity", SCAFFOLD,
           "cost/slippage unmodeled; deprioritized, never live", "DECISIONS 2026-05-18"),
    Family("short_interest_xs", "Short-interest days-to-cover (XS)", "equity", KILLED,
           "CPCV -1.21 Sharpe; meme-era reversal flip -> no edge", "DECISIONS 2026-06-03 A2"),
    Family("options_signal", "Options-as-signal (CPIV/skew/OI/term/IVRV)", "equity_options", KILLED,
           "H4a-e all t<<-2 -> no tradeable equity edge (5 factors)", "DECISIONS 2026-06-12 H4"),
    Family("turn_of_month", "Turn-of-month", "equity_etf", KILLED,
           "miss HAC + zero diversification (timed SPY beta)", "DECISIONS 2026-06-14 F1a"),
    Family("overnight", "Overnight close->open", "equity_etf", KILLED,
           "gross +0.53 -> net +0.16/-0.22; cost-killed", "DECISIONS 2026-06-16 P3-3"),
    Family("etf_relative_value", "ETF relative-value (log-spread MR)", "equity_etf", KILLED,
           "orthogonal but point_SR 0.026, p 0.46 -> zero edge", "DECISIONS 2026-06-14 F2"),
    Family("credit_timing", "Credit-timing (HYG/IEF on SPY)", "equity_etf", PARKED,
           "Track-A pass but corr 0.52 to beta -> not diversifying", "DECISIONS 2026-06-14 G3"),
    Family("sector_rotation", "Sector-ETF relative-strength rotation", "equity_etf", PARKED,
           "standalone PAPER-PASS (CPCV SR 0.86) but Track-B FAIL vs trend (corr 0.51) -> redundant",
           "DECISIONS 2026-06-22 Option A"),
    # ── Futures ──
    Family("futures_trend", "Futures trend (cross-asset TSMOM)", "futures", KILLED,
           "real historically, DECAYED post-2015 (+0.02); redundant", "DECISIONS 2026-06-18 P4-2"),
    Family("futures_carry", "Futures carry (term-structure)", "futures", PAPER,
           "roll-honest Sharpe 0.58, post-2015 +0.89; survives GL-0", "DECISIONS 2026-06-20 P0.2"),
    Family("futures_xsmom", "Futures cross-sectional momentum (12-1)", "futures", PAPER,
           "Sharpe 0.56, corr-to-trend 0.12; max-of-6 p=0.005", "DECISIONS 2026-06-20 P1.2"),
    Family("curve_momentum", "Futures curve-momentum", "futures", KILLED,
           "Sharpe -0.24; killed at pre-registered sign (no flip)", "DECISIONS 2026-06-20 P1.2"),
    Family("futures_value", "Futures value (5y reversal)", "futures", KILLED,
           "Sharpe -0.24; killed at pre-registered sign", "DECISIONS 2026-06-20 P1.2"),
    Family("futures_skewness", "Futures skewness", "futures", KILLED,
           "Sharpe +0.03; insufficient edge", "DECISIONS 2026-06-20 P1.2"),
    Family("basis_momentum", "Futures basis-momentum (2nd-nearby)", "futures", KILLED,
           "Sharpe -0.10, residual-alpha t 0.47; orthogonal, no edge", "DECISIONS 2026-06-20 P1.4"),
    Family("cftc_cot", "CFTC CoT hedging-pressure", "futures", KILLED,
           "Sharpe +0.06, residual-alpha t 0.27; orthogonal, no edge", "DECISIONS 2026-06-20 P1.4"),
    Family("rates_carry", "Rates carry (IEF duration)", "rates", KILLED,
           "config-robust but time-unstable (post-2016 dead)", "DECISIONS 2026-06-14 F3"),
    # ── Volatility ──
    Family("vix_vrp", "VRP via VIX-futures curve", "volatility", PAPER,
           "Sharpe 0.64; survives crashes; DROPPED per GL-1 (tail-conc.)", "DECISIONS 2026-06-20 P3.1"),
    # ── Overlays (de-risk timing candidates) ──
    Family("vix_crash_governor", "VIX-term crash governor", "overlay", LIVE,
           "modest tail help; first positive overlay; live", "DECISIONS 2026-06-14 F1b"),
    Family("credit_overlay", "Credit de-risk overlay (HYG/IEF)", "overlay", PARKED,
           "marginal +0.064 dSharpe, PIT-confirmed; flag OFF (multiplicity)", "DECISIONS 2026-06-22 P0.4"),
    Family("curve_overlay", "Curve-inversion de-risk overlay", "overlay", PARKED,
           "no tail benefit (dSharpe -0.018) -> off", "DECISIONS 2026-06-14 G1"),
    Family("short_interest_overlay", "Short-interest de-risk overlay", "overlay", KILLED,
           "uniformly Sharpe-negative marginal to governor", "DECISIONS 2026-06-15 G2"),
    # ── Crypto ──
    Family("crypto_trend", "Crypto trend (TSMOM, spot)", "crypto", PAPER,
           "Sharpe 0.64, corr-to-trend 0.18; CAPITAL fail (history)", "DECISIONS 2026-06-16 P3-1"),
    # ── Shown for auditability but EXCLUDED from the trial count (counts_as_trial=False) ──
    # Not search trials: cash is infrastructure (not an alpha hypothesis); futures_book is an
    # ENSEMBLE of carry+xsmom (already counted) - counting it would double-count the search.
    Family("cash_sleeve", "Cash / T-bill sleeve", "cash", LIVE,
           "capital-preservation infra, not an alpha-search trial", "DECISIONS 2026-06-16 P1-1",
           counts_as_trial=False),
    Family("futures_book", "Futures book (carry + xsmom ensemble)", "futures", PAPER,
           "equal-weight ensemble of two already-counted families (not a separate trial)",
           "DECISIONS 2026-06-20 P1.3", counts_as_trial=False),
]


# ── Research degrees-of-freedom log ───────────────────────────────────────────
# The within-family search burden (variants, bug-fix reruns, reviewer re-tests, post-hoc
# exclusions) - transparency on WHY the family count, not the raw backtest count, is the right N.
DEGREES_OF_FREEDOM: List[str] = [
    "futures factor zoo: 6 free factors screened (xsmom/curve-mom/value/skew/basis-mom/CoT) -> "
    "only xsmom survived; killed at the PRE-REGISTERED sign (no sign-flipping = OPT-5 discipline).",
    "credit overlay: pre-registered L=60/band=0 FAILED -> post-hoc L=120/band=0.02 selected "
    "(multiplicity disclosed; the binding caveat, not vol - P0.4).",
    "rates carry: 12-cell post-hoc grid (all cells +residual-alpha) -> killed on a fresh pre-registered "
    "sub-period stability run (config-robust but time-unstable).",
    "futures carry: 6 bug-fix reruns in the P4-2 hardening pass + the P0.2 roll-cost/PIT-vol "
    "honesty pass (the +0.17 in-sample dSR was a vol-match artifact, ~0 under PIT).",
    "options-as-signal: 5 pre-registered decile-sort hypotheses, all killed at the pre-reg sign.",
    "swing/intraday ML: 9+ training iterations (LX1-LX9 / v88-v94); confirmed regime-null.",
    "VRP: parked on short SPX options (alpha-framing) -> REVIVED via the VIX-futures curve (a "
    "different signal) -> then dropped per GL-1 tail diagnostics.",
    "PEAD: live size 3.0x -> 1.0x telemetry -> event-level demote (p=0.78); never high-conviction "
    "capital.",
]


# ── Accessors ─────────────────────────────────────────────────────────────────
def family_trial_count() -> int:
    """The principled multiple-testing N: distinct families searched as deploy candidates.
    This is the program's true N_TRIALS for the parametric Deflated-Sharpe cross-check -
    auditable (every family enumerated above), replacing the prior hardcoded ~20."""
    return sum(1 for f in FAMILIES if f.counts_as_trial)


def count_by_status() -> dict:
    out: dict = {}
    for f in FAMILIES:
        out[f.status] = out.get(f.status, 0) + 1
    return out


def families_by_status(status: str) -> List[Family]:
    return [f for f in FAMILIES if f.status == status]
