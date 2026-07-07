"""
per_name_gate.py — CH1 (Compound-and-Harden): the PER-NAME risk gate for the live trend book
(shadow-first).

The live trend+cash order path already passes the WHOLE-BOOK gate (gross / net-equity-beta /
single+book notional) + reconciliation-before-trade in enforce — but it never ran the agent
RiskManager's PER-NAME checks (correlation / heat / concentration). Those checks only ever covered
the dead proposal-driven ML/swing path (`RiskManager._validate_trade`), so the live book had no
per-name safety net. CH1 closes that gap the SAME way the whole-book gate did: a book-level,
fail-safe, shadow-first gate wired onto `run_trend_rebalance`.

Design notes (why this is NOT a reuse of the order-level RM functions):
- The RM's per-name checks are ORDER-level (one candidate vs the book) and its
  sector/factor-concentration check maps EVERY trend ETF to "UNKNOWN" (the SECTOR_MAP has only
  single stocks) → it would breach every week and fail-CLOSE the whole book in enforce. So CH1
  evaluates the PROPOSED BOOK as a whole, on metrics that make sense for an all-ETF universe.
- All three metrics are computed from data ALREADY in scope at the wiring point — the target
  book (`intended_weights`, NAV fractions) + the price panel the sleeve already fetched — so the
  gate does NO new I/O and cannot fail on a data-feed outage.

The three checks (book-level):
- CORRELATION (the primary, genuinely-new protection): the NAV-weighted average SIGNED pairwise
  correlation of the target book. Low/negative = genuinely diversified (equities + bonds/gold/FX
  hedges); high positive = the diversifiers have dropped out or correlations have spiked in a
  crisis and the "10-name book" is really ONE bet. A COARSE extreme-concentration backstop
  thresholded HIGH (BOOK_CORR_GATE_AT, above the structural ~0.85 equity-ETF correlation) — the
  nuanced continuous correlation RESPONSE is CH2b, not this gate. The threshold is PROVISIONAL;
  the gate's real shadow-mode job is to RECORD the weekly book-corr so enforce can be calibrated
  from the observed distribution (see the BOOK_CORR_GATE_* note below).
- CONCENTRATION: any single name's target weight above the risk-policy per-name notional cap
  (defense-in-depth — the sleeve already clips to `max_position_pct` and the whole-book gate caps
  single-instrument notional; this catches a regression in either).
- HEAT: proposed-book open-risk = Σ(weight)·FALLBACK_RISK_PCT (trend carries no stops, so the
  established 2%-of-notional fallback is used) vs MAX_PORTFOLIO_HEAT_PCT. For the stopless trend
  book this is a LOOSE bound (≈ gross × 2%) that rarely binds — kept for parity with the RM heat
  check + as a backstop if gross ever balloons.

Rollout mirrors the whole-book gate: `pm.per_name_gate_mode` = 'shadow' (default, logs + emails
what it WOULD hold, blocks nothing) / 'enforce' (a breach HOLDS the rebalance, fail-closed) / 'off'
(not evaluated). `shadow_per_name_gate` NEVER raises — any internal error returns allow=True so a
gate bug can't disrupt a live rebalance; the caller acts on the verdict only in enforce mode.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from app.live_trading.risk_policy import RISK_POLICY_V1, RiskPolicy
from app.strategy.portfolio_heat import FALLBACK_RISK_PCT, MAX_PORTFOLIO_HEAT_PCT

log = logging.getLogger(__name__)

SHADOW = "shadow"
ENFORCE = "enforce"
OFF = "off"

# CH1 correlation-gate thresholds — DELIBERATELY HIGH and PROVISIONAL.
# The trend equity ETFs (SPY/QQQ/IWM/EFA/EEM) are STRUCTURALLY ~0.85 correlated over any long
# window, and a normal long-only risk-on trend book drops the bond/gold/FX diversifiers (they
# trend-flat → excluded), so its weighted-avg book correlation legitimately sits ~0.75-0.90 EVERY
# equity-led week. So this gate is a COARSE "the 10-name book has collapsed to genuinely ONE bet"
# backstop (diversifiers gone / crisis correlation spike), NOT the nuanced correlation RESPONSE —
# that is CH2b's continuous correlation-regime degross. The thresholds sit ABOVE risk_policy's
# stress-conditional degross levels (0.60/0.70), which are for normally-diversified assets whose
# corr SPIKES — a different metric than this full-window book average.
# PROVISIONAL: the real job in shadow is to RECORD `weighted_avg_book_corr` weekly; the enforce
# threshold must be calibrated from that observed distribution BEFORE any flip to enforce (a
# threshold that trips most weeks = a permanent HOLD, the failure this gate exists to avoid).
BOOK_CORR_GATE_AT = 0.90
BOOK_CORR_GATE_STRONG = 0.95


@dataclass(frozen=True)
class PerNameGateVerdict:
    allow: bool
    mode: str
    breaches: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def weighted_avg_book_corr(weights: Dict[str, float], corr) -> Optional[float]:
    """NAV-weighted average of the SIGNED off-diagonal pairwise correlations over the target book.

    num = Σ_{i<j} w_i·w_j·ρ_ij ; den = Σ_{i<j} w_i·w_j ; return num/den (in [-1, 1]).
    High positive ⇒ the book behaves like a single bet; low/negative ⇒ genuinely diversified.
    Returns None when fewer than 2 target names have a defined pairwise correlation (nothing to
    concentrate). Pure; tolerant of missing symbols / NaN correlations (those pairs are skipped).
    """
    names = [s for s, w in weights.items() if w and float(w) > 0]
    num = 0.0
    den = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            try:
                c = float(corr.at[a, b])
            except (KeyError, TypeError, ValueError):
                continue
            if c != c:  # NaN
                continue
            wa, wb = float(weights[a]), float(weights[b])
            num += wa * wb * c
            den += wa * wb
    return (num / den) if den > 0 else None


def evaluate(target_weights: Dict[str, float], corr,
             policy: RiskPolicy = RISK_POLICY_V1,
             per_name_cap: Optional[float] = None) -> PerNameGateVerdict:
    """Pure: check a PROPOSED per-name book (target weights = NAV fractions) + its realized
    correlation matrix against the per-name risk metrics. `mode` is set by the caller.

    per_name_cap: the concentration ceiling. Defaults to the risk-policy single-instrument cap
    (0.25); the caller passes the sleeve's configured `max_position_pct` so the EFFECTIVE cap is
    max(policy, configured) — this way raising `pm.trend_max_position_pct` above 0.25 can NOT make
    the gate breach on every top name (a permanent enforce-HOLD). Since intended_weights are
    already clipped to the configured cap, the concentration check is a defense-in-depth CLIP-
    INTEGRITY guard: it fires only if a weight exceeds the cap it should already have been clipped
    to (a bug), never in normal operation."""
    breaches: List[str] = []
    details: Dict[str, float] = {}

    # --- concentration: any single name above the per-name notional cap ---
    weights_pos = {s: float(w) for s, w in (target_weights or {}).items() if w and float(w) > 0}
    max_w = max(weights_pos.values(), default=0.0)
    details["max_name_weight"] = max_w
    cap = policy.max_single_instrument_notional_frac
    if per_name_cap is not None:
        cap = max(cap, float(per_name_cap))
    for s, w in sorted(weights_pos.items()):
        if w > cap + 1e-9:
            breaches.append(f"per_name_concentration {s} {w:.1%} > cap {cap:.0%}")

    # --- correlation: has the whole target book collapsed to genuinely ONE bet? ---
    wavg = weighted_avg_book_corr(weights_pos, corr) if corr is not None else None
    details["weighted_avg_book_corr"] = wavg if wavg is not None else float("nan")
    if wavg is not None and wavg > BOOK_CORR_GATE_AT + 1e-9:
        sev = "STRONG " if wavg > BOOK_CORR_GATE_STRONG else ""
        breaches.append(f"{sev}book_correlation {wavg:.2f} > {BOOK_CORR_GATE_AT:.2f} "
                        f"(target book behaving like one bet)")

    # --- heat: proposed-book open risk (stopless trend -> FALLBACK_RISK_PCT per name) ---
    gross = sum(weights_pos.values())
    heat = gross * FALLBACK_RISK_PCT
    details["portfolio_heat_frac"] = heat
    if heat > MAX_PORTFOLIO_HEAT_PCT + 1e-9:
        breaches.append(f"portfolio_heat {heat:.1%} > cap {MAX_PORTFOLIO_HEAT_PCT:.0%}")

    return PerNameGateVerdict(allow=not breaches, mode="", breaches=breaches, details=details)


def _corr_from_prices(prices, names: List[str]):
    """Realized pairwise correlation of daily returns over the fetched price panel, for the target
    names present as columns. Returns a DataFrame (possibly empty if <2 usable names). `prices` is
    the sleeve's already-fetched close panel (DataFrame: dates × symbols)."""
    cols = [n for n in names if n in getattr(prices, "columns", [])]
    if len(cols) < 2:
        import pandas as pd
        return pd.DataFrame()
    return prices[cols].pct_change().corr()


def shadow_per_name_gate(target_weights: Dict[str, float], prices, *,
                         mode: str = SHADOW, label: str = "",
                         policy: RiskPolicy = RISK_POLICY_V1,
                         per_name_cap: Optional[float] = None,
                         notifier=None) -> PerNameGateVerdict:
    """FAIL-SAFE per-name gate for a sleeve to call before placing. Builds the correlation matrix
    from the already-fetched price panel, evaluates the proposed book, logs (+ emails on a breach),
    and returns the verdict. NEVER raises — any error → allow=True so a gate bug can't disrupt a
    live rebalance. The CALLER acts on the verdict only in ENFORCE mode. `per_name_cap` = the
    sleeve's configured per-name cap (see `evaluate`)."""
    try:
        corr = _corr_from_prices(prices, list((target_weights or {}).keys()))
        v = evaluate(target_weights, corr, policy, per_name_cap=per_name_cap)
        v = PerNameGateVerdict(allow=v.allow, mode=mode, breaches=v.breaches, details=v.details)
        if v.breaches:
            log.warning("[per-name-gate:%s mode=%s] WOULD-BLOCK%s: %s | details=%s",
                        label, mode, ("" if mode == ENFORCE else " (shadow: not blocking)"),
                        "; ".join(v.breaches), v.details)
            if notifier is not None:
                try:
                    notifier.enqueue("per_name_gate_breach", {
                        "label": label, "mode": mode, "breaches": v.breaches,
                        "details": v.details})
                except Exception:
                    log.debug("per-name-gate: notify failed", exc_info=True)
        else:
            log.info("[per-name-gate:%s mode=%s] OK: %s", label, mode, v.details)
        return v
    except Exception as exc:  # noqa: BLE001 — the gate must NEVER break a live rebalance
        # Fail-SAFE: proceed (the per-order + whole-book + 80% gross caps still bind). Make a wedged
        # gate VISIBLE (H8 pattern) so it gets fixed rather than silently not running.
        log.warning("[per-name-gate:%s] evaluation error (fail-safe allow): %s", label, exc,
                    exc_info=True)
        if notifier is not None:
            try:
                notifier.enqueue("gate_error", {
                    "gate": "per_name", "label": label, "mode": mode, "error": str(exc)},
                    dedup_key=f"gate_error:per_name:{label}")
            except Exception:
                log.debug("per-name-gate: gate_error notify failed", exc_info=True)
        return PerNameGateVerdict(allow=True, mode=mode, error=str(exc))
