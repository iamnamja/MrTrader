"""
sleeve_lab.py — the SLEEVE LAB (Alpha-v7 F0): one uniform, hardened pipeline for
taking any candidate return-stream from idea -> Ruler-v2 verdict, retiring the
bespoke per-idea `run_*_cpcv` scripts.

WHY THIS EXISTS
---------------
Every sleeve we have tested (PEAD, options-XS, index short-vol, trend-broaden) was a
hand-written `run_*_cpcv.py` that re-implemented the SAME plumbing — fetch returns ->
wrap in SeriesReturnStrategy -> run_cpcv -> set component_type -> ruler_v2.evaluate ->
print. Each re-implementation is a fresh bug surface and a fresh chance to wire the gate
slightly differently. The 2026-06-14 research panel's plan calls for a 3-5 sleeve premia
book; running that on a pile of mismatched one-offs is how silent inconsistencies (and
look-ahead) creep in.

The Sleeve Lab makes a sleeve a DECLARATION, not a script. You give it:
  1. a daily PIT net-return series (the strategy's output),
  2. a `component_type` (alpha | diversifier | risk_premium),
  3. (optionally) a pre-registered trial count + registration id (R7),
and `evaluate_sleeve` runs the IDENTICAL, audited path every time:
  SeriesReturnStrategy -> run_cpcv (proven FULL_* constants) -> Ruler-v2 Track-A
  (PAPER + CAPITAL) -> [optional] Track-B book-delta appraisal -> a uniform SleeveReport.

It changes NOTHING about the gate semantics — it composes the existing proven pieces
(scripts.walkforward.gate_calibration.SeriesReturnStrategy, scripts.walkforward.cpcv,
app.research.ruler_v2, scripts.walkforward.track_b_appraisal, app.strategy.sleeve_allocator)
behind one tested surface. Promotion to paper/capital remains an explicit owner decision;
this module only produces REPORT-ONLY verdicts (like the Track-B park pattern).

REGISTRY
--------
`@register_sleeve("name")` decorates a builder `(...) -> Sleeve`; `build_sleeve("name", **kw)`
constructs it; `list_sleeves()` enumerates. New sleeves register here instead of spawning a
new top-level script, so the book is assembled from a single source of truth.

KNOWN LIMITATIONS (honest, by design — surfaced as loud warnings at runtime)
----------------------------------------------------------------------------
- CAPITAL residual-α is SPY-only CAPM here. In the SeriesReturnStrategy/run_cpcv path,
  result.residual_alpha_t_hac is populated by the single-factor SPY CAPM diagnostic (and
  only when `spy_prices` is supplied) — NOT yet the multi-factor harvested-premium-excluded
  residual-α (ruler_v2.residual_alpha_t / RULERV2_HARVESTED_FACTOR). Wiring a real factor
  panel is a follow-up; until then CAPITAL residual-α is SPY-CAPM, and is None (fails
  closed) without spy_prices. CAPITAL is in any case structurally unreachable on a backtest
  alone (it requires a live-paper observation), so this is latent, but stated plainly.
- CAPITAL power floor needs n_folds >= RULERV2_MIN_N_FOLDS (10); the default lab geometry
  is FULL_N_FOLDS=8 (parity with the proven calibration controls). A capital-aspiring run
  must pass n_folds>=10. evaluate_sleeve warns when n_folds is below the floor.
- OVERLAYS are not supported (see VALID_COMPONENT_TYPES note) — an overlay modifies the book
  rather than adding a return stream; it gets a dedicated path in F1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Allowed component types (mirror retrain_config / ruler_v2 vocabulary) ──────────
# NOTE: "overlay" is deliberately EXCLUDED. An overlay (e.g. a VIX-term crash governor)
# MODIFIES the book (scales exposure / de-risks in a regime) rather than contributing an
# additive return stream — the additive Track-A significance + Track-B blend model here is
# the WRONG instrument for it and would emit a confidently-wrong verdict. Overlays get a
# dedicated book-with-vs-without evaluation path (planned with F1's crash governor); until
# then a Sleeve cannot be declared "overlay" (fail-loud beats silently-wrong).
VALID_COMPONENT_TYPES = frozenset({"alpha", "diversifier", "risk_premium"})


# ──────────────────────────────────────────────────────────────────────────────────
# Sleeve declaration
# ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class Sleeve:
    """A candidate return-stream declared for evaluation through the uniform gate.

    `returns` is the strategy's daily PIT NET (post-cost) return series, indexed by
    trade date. `component_type` drives the Ruler-v2 regime waiver (declared
    diversifiers / risk_premia are exempt from the worst-regime backstop — failing a
    crisis is their purpose). `n_trials_registered` is the R7 pre-registered family
    trial count for the Bayesian multiplicity shrinkage (None -> the conservative
    N_TRIALS_TESTED fallback). `spy_prices` (optional) feeds the single-factor residual-α
    diagnostic when a multi-factor panel is unavailable.
    """
    label: str
    component_type: str
    returns: pd.Series
    spy_prices: Optional[pd.Series] = None
    n_trials_registered: Optional[int] = None
    registration_id: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.label or not isinstance(self.label, str):
            raise ValueError("Sleeve.label must be a non-empty string")
        ct = (self.component_type or "").lower()
        if ct not in VALID_COMPONENT_TYPES:
            raise ValueError(
                f"Sleeve.component_type {self.component_type!r} not in {sorted(VALID_COMPONENT_TYPES)}"
            )
        self.component_type = ct
        self.returns = _clean_return_series(self.returns, name=f"{self.label} returns")
        if self.spy_prices is not None:
            self.spy_prices = _clean_price_series(self.spy_prices, name=f"{self.label} spy_prices")
        if self.n_trials_registered is not None:
            self.n_trials_registered = int(self.n_trials_registered)
            if self.n_trials_registered < 1:
                raise ValueError("Sleeve.n_trials_registered must be >= 1 when provided")

    @property
    def regime_waived_by_type(self) -> bool:
        """True iff this component_type is policy-exempt from the worst-regime floor."""
        from app.ml.retrain_config import RULERV2_REGIME_WAIVED_TYPES
        return self.component_type in RULERV2_REGIME_WAIVED_TYPES


def _clean_return_series(s: pd.Series, *, name: str) -> pd.Series:
    """Coerce to a sorted, de-duplicated, NaN-free daily return Series with a
    DatetimeIndex. Fails LOUD on the things that silently corrupt autocorrelation-
    sensitive stats downstream (non-datetime index, duplicate dates, all-NaN)."""
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(s).__name__}")
    s = s.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"{name} index is not datetime-convertible: {e}") from e
    s = s[~s.index.isna()]
    s = s.dropna()
    if s.empty:
        raise ValueError(f"{name} is empty after dropna")
    s = s.sort_index()
    if s.index.has_duplicates:
        # Keep first observation per date (mirror run_cpcv's OOS de-dup convention).
        s = s[~s.index.duplicated(keep="first")]
    s = s.astype(float)
    if not np.isfinite(s.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values after cleaning")
    return s


def _clean_price_series(s: pd.Series, *, name: str) -> pd.Series:
    s = _clean_return_series(s, name=name)  # same structural cleaning
    if (s <= 0).any():
        raise ValueError(f"{name} contains non-positive prices")
    return s


# ──────────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────────
SLEEVE_REGISTRY: Dict[str, Callable[..., Sleeve]] = {}


def register_sleeve(name: str) -> Callable[[Callable[..., Sleeve]], Callable[..., Sleeve]]:
    """Decorator: register a builder `(...) -> Sleeve` under `name`. New sleeves
    register here instead of becoming a new top-level `run_*_cpcv.py` script."""
    def deco(fn: Callable[..., Sleeve]) -> Callable[..., Sleeve]:
        key = name.lower()
        if key in SLEEVE_REGISTRY:
            raise ValueError(f"sleeve {name!r} already registered")
        SLEEVE_REGISTRY[key] = fn
        return fn
    return deco


def build_sleeve(name: str, **kwargs) -> Sleeve:
    key = (name or "").lower()
    if key not in SLEEVE_REGISTRY:
        raise KeyError(f"unknown sleeve {name!r}; registered: {sorted(SLEEVE_REGISTRY)}")
    sleeve = SLEEVE_REGISTRY[key](**kwargs)
    if not isinstance(sleeve, Sleeve):
        raise TypeError(f"builder {name!r} returned {type(sleeve).__name__}, expected Sleeve")
    return sleeve


def list_sleeves() -> List[str]:
    return sorted(SLEEVE_REGISTRY)


# ──────────────────────────────────────────────────────────────────────────────────
# Uniform report
# ──────────────────────────────────────────────────────────────────────────────────
def _obs(detail: dict, key: str, default=float("nan")):
    """First (observed) element of a (observed, ok) detail entry, or `default`."""
    v = detail.get(key)
    return v[0] if isinstance(v, tuple) and v else default


def failed_criteria(detail: dict) -> List[str]:
    """The GATING criteria that failed (informational keys excluded) — mirrors the
    reducer in ruler_v2.gate_passed so the report can never disagree with the verdict."""
    from app.research.ruler_v2 import INFORMATIONAL_KEYS
    return [k for k, v in detail.items()
            if isinstance(v, tuple) and len(v) == 2 and not v[1] and k not in INFORMATIONAL_KEYS]


@dataclass
class SleeveReport:
    """The uniform verdict for one sleeve. Track-A (standalone PAPER/CAPITAL) is always
    present; Track-B (book-delta appraisal) is present only when a base book was supplied."""
    label: str
    component_type: str
    n_obs: int
    n_folds: int
    window_start: str
    window_end: str
    mean_sharpe: float
    path_sharpe_tstat: float
    # Track-A standalone
    paper_passed: bool
    capital_passed: bool
    paper_detail: dict
    capital_detail: dict
    paper_failed: List[str]
    capital_failed: List[str]
    point_sr: float
    hac_p_one_sided: float
    worst_regime_sharpe: Optional[float]
    residual_alpha_t_hac: Optional[float]
    regime_waiver_approved: bool
    # Track-B (optional book-delta appraisal)
    track_b: Optional[object] = None  # TrackBAppraisalResult | None
    # underlying CPCV result (kept for deep inspection; not serialized)
    result: Optional[object] = field(default=None, repr=False)

    @property
    def verdict(self) -> str:
        a = "PAPER-PASS" if self.paper_passed else "PAPER-FAIL"
        c = "CAPITAL-PASS" if self.capital_passed else "CAPITAL-FAIL"
        if self.track_b is not None:
            b = f"TRACK-B-{'PASS' if self.track_b.passed else 'FAIL'}"
            return f"{a} / {c} / {b}"
        return f"{a} / {c}"

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "component_type": self.component_type,
            "n_obs": self.n_obs,
            "n_folds": self.n_folds,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "mean_sharpe": self.mean_sharpe,
            "path_sharpe_tstat": self.path_sharpe_tstat,
            "paper_passed": self.paper_passed,
            "capital_passed": self.capital_passed,
            "paper_failed": list(self.paper_failed),
            "capital_failed": list(self.capital_failed),
            "point_sr": self.point_sr,
            "hac_p_one_sided": self.hac_p_one_sided,
            "worst_regime_sharpe": self.worst_regime_sharpe,
            "residual_alpha_t_hac": self.residual_alpha_t_hac,
            "regime_waiver_approved": self.regime_waiver_approved,
            "verdict": self.verdict,
        }
        if self.track_b is not None:
            d["track_b"] = self.track_b.to_dict()
        return d


# ──────────────────────────────────────────────────────────────────────────────────
# The evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────────
def evaluate_sleeve(
    sleeve: Sleeve,
    *,
    base_book_returns: Optional[pd.Series] = None,
    regime_map: Optional[dict] = None,
    regime_waiver_approved: Optional[bool] = None,
    n_folds: Optional[int] = None,
    n_paths: Optional[int] = None,
    purge_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
    total_years: Optional[int] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> SleeveReport:
    """Run one sleeve through the uniform gate and return a SleeveReport.

    `base_book_returns` — if supplied, additionally runs the Track-B book-delta
    appraisal (candidate=sleeve vs base=book). `regime_waiver_approved` defaults to
    True for declared diversifiers/risk_premia (policy) and False otherwise; the gate
    independently re-checks the component_type, so over-approving an `alpha` is inert.
    `total_years=None` uses the FULL series window (deep history — the powered sandbox).
    """
    from scripts.walkforward.gate_calibration import (
        SeriesReturnStrategy, FULL_N_FOLDS, FULL_N_PATHS,
        FULL_PURGE_DAYS, FULL_EMBARGO_DAYS,
    )
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.regime import load_regime_map
    from app.research import ruler_v2

    n_folds = FULL_N_FOLDS if n_folds is None else int(n_folds)
    n_paths = FULL_N_PATHS if n_paths is None else int(n_paths)
    purge_days = FULL_PURGE_DAYS if purge_days is None else int(purge_days)
    embargo_days = FULL_EMBARGO_DAYS if embargo_days is None else int(embargo_days)
    if regime_waiver_approved is None:
        regime_waiver_approved = sleeve.regime_waived_by_type

    rets = sleeve.returns
    start, end = rets.index.min().date(), rets.index.max().date()
    if regime_map is None:
        regime_map = load_regime_map(start, end)

    strat = SeriesReturnStrategy(sleeve.label, rets, spy_prices=sleeve.spy_prices,
                                 regime_map=regime_map)
    result = run_cpcv(strategy=strat, purge_days=purge_days, embargo_days=embargo_days,
                      n_folds=n_folds, n_paths=n_paths, total_years=total_years)
    # Stamp the declared metadata the gate reads off the result.
    result.component_type = sleeve.component_type
    if sleeve.n_trials_registered is not None:
        result.n_trials_registered = sleeve.n_trials_registered

    # ── Hardening warnings (loud, non-fatal): surface the conditions under which a
    #    verdict is weakly-powered or a tier is structurally unreachable, so a thin or
    #    under-specified run is never mistaken for a clean one. ───────────────────────
    from app.ml.retrain_config import RULERV2_MIN_DAILY_OBS, RULERV2_MIN_N_FOLDS
    n_oos = len(result.oos_returns_dated)
    if n_oos < RULERV2_MIN_DAILY_OBS:
        log.warning("[sleeve_lab] %s: only %d pooled-OOS obs (< power floor %d) - HAC/"
                    "bootstrap inference is weakly powered; treat PAPER as tentative.",
                    sleeve.label, n_oos, RULERV2_MIN_DAILY_OBS)
    if n_folds < RULERV2_MIN_N_FOLDS:
        log.warning("[sleeve_lab] %s: n_folds=%d < CAPITAL power floor %d - CAPITAL is "
                    "UNREACHABLE by the power gate at this geometry (pass n_folds>=%d for "
                    "a capital-aspiring run).",
                    sleeve.label, n_folds, RULERV2_MIN_N_FOLDS, RULERV2_MIN_N_FOLDS)
    if sleeve.spy_prices is None:
        # SPY-CAPM residual-α (the only residual-α wired on this path) needs spy_prices;
        # without it result.residual_alpha_t_hac is None -> CAPITAL's residual-α criterion
        # fails closed regardless of merit. See module KNOWN LIMITATIONS.
        log.warning("[sleeve_lab] %s: no spy_prices - CAPITAL residual-alpha t is None "
                    "(fails closed). Supply spy_prices for the SPY-CAPM residual-alpha.",
                    sleeve.label)

    # ── Track-A: standalone PAPER + CAPITAL ──────────────────────────────────────
    paper_detail = ruler_v2.evaluate(result, tier="paper",
                                     component_type=sleeve.component_type,
                                     regime_waiver_approved=regime_waiver_approved)
    paper_passed = ruler_v2.gate_passed(result, tier="paper",
                                        component_type=sleeve.component_type,
                                        regime_waiver_approved=regime_waiver_approved)
    capital_detail = ruler_v2.evaluate(result, tier="capital",
                                       component_type=sleeve.component_type,
                                       regime_waiver_approved=regime_waiver_approved)
    capital_passed = ruler_v2.gate_passed(result, tier="capital",
                                          component_type=sleeve.component_type,
                                          regime_waiver_approved=regime_waiver_approved)

    # ── Track-B: book-delta appraisal (optional) ─────────────────────────────────
    track_b = None
    if base_book_returns is not None:
        from scripts.walkforward.track_b_appraisal import (
            appraise_track_b, TrackBAppraisalCriteria,
        )
        base = _clean_return_series(base_book_returns, name="base_book_returns")
        # `worst_regime_sharpe` here is the CANDIDATE's standalone CPCV worst-regime SR —
        # this matches Track-B's contract: it is the candidate's regime SAFETY backstop
        # (waived for diversifiers/risk_premia), while the candidate's BOOK CONTRIBUTION is
        # measured separately by appraisal-IR / P(dSR>0) / tail-overlap. The Track-A waiver
        # is reused; for non-waived types Track-B re-derives its own posture from it.
        track_b = appraise_track_b(
            base, rets, component_type=sleeve.component_type,
            criteria=TrackBAppraisalCriteria.from_retrain_config(),
            worst_regime_sharpe=result.worst_regime_sharpe,
            regime_waiver_approved=regime_waiver_approved,
            n_boot=n_boot, seed=seed, candidate_label=sleeve.label)

    return SleeveReport(
        label=sleeve.label,
        component_type=sleeve.component_type,
        n_obs=len(result.oos_returns_dated),
        n_folds=result.n_folds,
        window_start=str(start),
        window_end=str(end),
        mean_sharpe=result.mean_sharpe,
        path_sharpe_tstat=result.path_sharpe_tstat,
        paper_passed=paper_passed,
        capital_passed=capital_passed,
        paper_detail=paper_detail,
        capital_detail=capital_detail,
        paper_failed=failed_criteria(paper_detail),
        capital_failed=failed_criteria(capital_detail),
        point_sr=_obs(paper_detail, "point_sr_floor"),
        hac_p_one_sided=_obs(paper_detail, "hac_significance"),
        worst_regime_sharpe=result.worst_regime_sharpe,
        residual_alpha_t_hac=result.residual_alpha_t_hac,
        regime_waiver_approved=regime_waiver_approved,
        track_b=track_b,
        result=result,
    )


# ──────────────────────────────────────────────────────────────────────────────────
# Book assembly (F5) — combine accepted sleeves into a book via the proven allocator
# ──────────────────────────────────────────────────────────────────────────────────
def assemble_book(sleeves: List[Sleeve], *, scheme: str = "vol",
                  regime_labels: Optional[pd.Series] = None, cfg=None):
    """Combine multiple sleeves' return streams into one book via the proven
    sleeve_allocator. Aligns the streams on their COMMON dates (inner join — a book
    return needs every sleeve present that day). Returns a BookResult."""
    from app.strategy.sleeve_allocator import build_book, AllocatorConfig
    if len(sleeves) < 1:
        raise ValueError("assemble_book needs at least one sleeve")
    frame = pd.DataFrame({s.label: s.returns for s in sleeves})
    n_union = len(frame)
    frame = frame.dropna(how="any")
    if frame.empty:
        raise ValueError("sleeves share no common dates - cannot assemble a book")
    if n_union and len(frame) < 0.5 * n_union:
        # The inner-join keeps only dates where EVERY sleeve is present, so one short
        # sleeve silently truncates the whole book window. Warn loudly rather than
        # report a book Sharpe computed over an unexpectedly short window.
        log.warning("[sleeve_lab] assemble_book: inner-join kept %d/%d dates (%.0f%%) - "
                    "a short sleeve is truncating the book window; per-sleeve coverage: %s",
                    len(frame), n_union, 100.0 * len(frame) / n_union,
                    {s.label: int(s.returns.notna().sum()) for s in sleeves})
    cfg = cfg or AllocatorConfig()
    return build_book(frame, scheme=scheme, regime_labels=regime_labels, cfg=cfg)


# ──────────────────────────────────────────────────────────────────────────────────
# Overlays (F1b) — a book-MODIFYING signal (e.g. a crash governor), NOT an additive
# sleeve. Evaluated book-WITH vs book-WITHOUT, on tail metrics, not Track-A significance.
# ──────────────────────────────────────────────────────────────────────────────────
# Default crisis windows for the with/without tail comparison (ISO date ranges).
DEFAULT_CRISIS_WINDOWS = {
    "GFC_2008": ("2008-09-01", "2009-03-31"),
    "COVID_2020": ("2020-02-15", "2020-04-30"),
    "BEAR_2022": ("2022-01-01", "2022-10-31"),
}


@dataclass
class Overlay:
    """A book-modifying exposure overlay. `multiplier` is the AS-APPLIED daily exposure
    scalar (typically in [0, 1]) — i.e. already lagged by the builder so that
    `multiplier[t]` is knowable before day t and scales the book's day-t return. The
    builder owns the PIT lag (e.g. signal at close[t-1] -> multiplier[t])."""
    label: str
    multiplier: pd.Series
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Overlay.label must be non-empty")
        m = _clean_return_series(self.multiplier, name=f"{self.label} multiplier")
        if (m < 0).any():
            raise ValueError(f"{self.label}: multiplier has negative exposure")
        self.multiplier = m


def _ann_stats(r: pd.Series, ann: int = 252) -> dict:
    """Annualized Sharpe / return / vol / maxDD / Calmar for a daily return series."""
    r = r.dropna()
    if len(r) < 2:
        return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "max_dd": 0.0, "calmar": 0.0}
    mu, sd = float(r.mean()), float(r.std())
    sharpe = float(mu / sd * np.sqrt(ann)) if sd > 0 else 0.0
    growth = float((1.0 + r).prod())
    years = len(r) / ann
    ann_ret = float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    calmar = float(ann_ret / abs(max_dd)) if (max_dd < 0 and np.isfinite(ann_ret)) else 0.0
    return {"sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": float(sd * np.sqrt(ann)),
            "max_dd": max_dd, "calmar": calmar}


@dataclass
class OverlayReport:
    label: str
    n_days: int
    window_start: str
    window_end: str
    mean_multiplier: float
    derisk_fraction: float          # fraction of days with multiplier < 1
    without: dict                   # base book stats
    with_: dict                     # overlaid book stats
    d_sharpe: float
    d_max_dd: float                 # with - without; >0 => shallower drawdown (better)
    d_calmar: float
    d_ann_ret: float
    crisis: dict                    # {window: {without_dd, with_dd, dd_improve}}
    improves_tail: bool
    sharpe_preserved: bool

    @property
    def verdict(self) -> str:
        if self.improves_tail and self.sharpe_preserved:
            return "HELPS (shallower tail, Sharpe preserved)"
        if self.improves_tail and not self.sharpe_preserved:
            return "TRADES RETURN FOR TAIL (Sharpe hit)"
        return "NO TAIL BENEFIT"

    def to_dict(self) -> dict:
        return {
            "label": self.label, "n_days": self.n_days,
            "window_start": self.window_start, "window_end": self.window_end,
            "mean_multiplier": self.mean_multiplier, "derisk_fraction": self.derisk_fraction,
            "without": self.without, "with": self.with_,
            "d_sharpe": self.d_sharpe, "d_max_dd": self.d_max_dd,
            "d_calmar": self.d_calmar, "d_ann_ret": self.d_ann_ret,
            "crisis": self.crisis, "improves_tail": self.improves_tail,
            "sharpe_preserved": self.sharpe_preserved, "verdict": self.verdict,
        }


def evaluate_overlay(overlay: Overlay, base_book_returns: pd.Series, *,
                     crisis_windows: Optional[dict] = None, toggle_cost_bps: float = 1.0,
                     sharpe_tolerance: float = 0.05) -> OverlayReport:
    """Evaluate a book-modifying overlay by comparing the book WITH vs WITHOUT it on the
    governor-active (overlapping) window. The overlaid return is
    `multiplier[t] * base[t]` minus a small toggle cost on |Δmultiplier| (changing
    exposure trades notional). Reports Sharpe/Calmar/maxDD deltas + per-crisis drawdown.
    A crash governor is judged on TAIL improvement with Sharpe roughly preserved — NOT on
    standalone significance (which is why it is not a Sleeve)."""
    base = _clean_return_series(base_book_returns, name="base_book_returns")
    mult = overlay.multiplier
    aligned = pd.concat([base.rename("base"), mult.rename("mult")], axis=1,
                        join="inner").dropna()
    if len(aligned) < 2:
        raise ValueError(f"{overlay.label}: overlay and book share <2 common dates")
    n_union = len(base)
    if n_union and len(aligned) < 0.5 * n_union:
        log.warning("[sleeve_lab] evaluate_overlay %s: governor-active window is %d/%d book "
                    "days (%.0f%%) - the overlay's data is shorter than the book history.",
                    overlay.label, len(aligned), n_union, 100.0 * len(aligned) / n_union)
    b = aligned["base"]
    m = aligned["mult"]
    toggle_cost = m.diff().abs().fillna(0.0) * (toggle_cost_bps / 1e4)
    overlaid = (m * b - toggle_cost).rename("with")

    without = _ann_stats(b)
    with_ = _ann_stats(overlaid)
    # maxDD are negative; with - without > 0 means the overlaid book drew down LESS.
    d_max_dd = with_["max_dd"] - without["max_dd"]
    d_sharpe = with_["sharpe"] - without["sharpe"]

    crisis = {}
    for name, (lo, hi) in (crisis_windows or DEFAULT_CRISIS_WINDOWS).items():
        seg = aligned.loc[(aligned.index >= pd.Timestamp(lo)) & (aligned.index <= pd.Timestamp(hi))]
        if len(seg) < 5:
            continue
        sb = seg["base"]
        sov = seg["mult"] * seg["base"]
        wo_dd = _ann_stats(sb)["max_dd"]
        wi_dd = _ann_stats(sov)["max_dd"]
        crisis[name] = {"without_dd": wo_dd, "with_dd": wi_dd, "dd_improve": wi_dd - wo_dd}

    improves_tail = (d_max_dd > 0) and (with_["calmar"] >= without["calmar"])
    sharpe_preserved = d_sharpe >= -abs(sharpe_tolerance)

    return OverlayReport(
        label=overlay.label, n_days=len(aligned),
        window_start=str(aligned.index[0].date()), window_end=str(aligned.index[-1].date()),
        mean_multiplier=float(m.mean()), derisk_fraction=float((m < 1.0).mean()),
        without=without, with_=with_,
        d_sharpe=d_sharpe, d_max_dd=d_max_dd,
        d_calmar=with_["calmar"] - without["calmar"], d_ann_ret=with_["ann_ret"] - without["ann_ret"],
        crisis=crisis, improves_tail=improves_tail, sharpe_preserved=sharpe_preserved)


def format_overlay_report(rep: OverlayReport) -> str:
    """Console-safe with/without summary for an overlay (e.g. crash governor)."""
    L = []
    bar = "=" * 78
    L.append(bar)
    L.append(f"  OVERLAY: {rep.label}")
    L.append(bar)
    L.append(f"  window {rep.window_start} -> {rep.window_end}   n_days={rep.n_days}  "
             f"mean_mult={_fmt(rep.mean_multiplier, 3)}  derisk_days={rep.derisk_fraction:.1%}")
    L.append("  " + "-" * 74)
    L.append(f"  {'metric':12} {'without':>12} {'with':>12} {'delta':>12}")
    for k, lab in (("sharpe", "Sharpe"), ("ann_ret", "AnnRet"), ("ann_vol", "AnnVol"),
                   ("max_dd", "MaxDD"), ("calmar", "Calmar")):
        L.append(f"  {lab:12} {rep.without[k]:>12.3f} {rep.with_[k]:>12.3f} "
                 f"{rep.with_[k]-rep.without[k]:>+12.3f}")
    if rep.crisis:
        L.append("  " + "-" * 74)
        L.append("  crisis maxDD (without -> with, +improve):")
        for name, c in rep.crisis.items():
            L.append(f"    {name:12} {c['without_dd']:>+8.3f} -> {c['with_dd']:>+8.3f}  "
                     f"(improve {c['dd_improve']:>+.3f})")
    L.append("  " + "-" * 74)
    L.append(f"  VERDICT: {rep.verdict}")
    L.append(bar)
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────────────────────
# ASCII reporting (no non-ASCII glyphs — Windows cp1252 console safe)
# ──────────────────────────────────────────────────────────────────────────────────
def _fmt(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    try:
        if isinstance(x, float) and not np.isfinite(x):
            return "n/a"
        return f"{x:+.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def format_sleeve_report(rep: SleeveReport) -> str:
    """A uniform, console-safe block summarizing the sleeve verdict."""
    L = []
    bar = "=" * 78
    L.append(bar)
    L.append(f"  SLEEVE: {rep.label}  [{rep.component_type}]")
    L.append(bar)
    L.append(f"  window {rep.window_start} -> {rep.window_end}   n_obs={rep.n_obs}  "
             f"n_folds={rep.n_folds}")
    L.append(f"  mean_sharpe={_fmt(rep.mean_sharpe)}  path_t={_fmt(rep.path_sharpe_tstat, 2)}  "
             f"point_SR={_fmt(rep.point_sr)}  hac_p(1s)={_fmt(rep.hac_p_one_sided, 4)}")
    L.append(f"  worst_regime_SR={_fmt(rep.worst_regime_sharpe)}  "
             f"residual_a_t={_fmt(rep.residual_alpha_t_hac, 2)}  "
             f"regime_waived={rep.regime_waiver_approved}")
    L.append("  " + "-" * 74)
    L.append(f"  TRACK-A PAPER  : {'PASS' if rep.paper_passed else 'FAIL'}"
             + (f"   failed={rep.paper_failed}" if rep.paper_failed else ""))
    L.append(f"  TRACK-A CAPITAL: {'PASS' if rep.capital_passed else 'FAIL'}"
             + (f"   failed={rep.capital_failed}" if rep.capital_failed else ""))
    if rep.track_b is not None:
        tb = rep.track_b
        L.append("  " + "-" * 74)
        L.append(f"  TRACK-B BOOK   : {'PASS' if tb.passed else 'FAIL'}  ({tb.verdict})")
        L.append(f"     appraisal_IR={_fmt(tb.appraisal_ir)}  dSR={_fmt(tb.delta_sr_point)}  "
                 f"P(dSR>0)={_fmt(tb.p_delta_sr_gt_0, 3)}  corr={_fmt(tb.corr_to_book, 3)}")
        if tb.failed_criteria:
            L.append(f"     failed={tb.failed_criteria}")
    L.append("  " + "-" * 74)
    L.append(f"  VERDICT: {rep.verdict}")
    L.append(bar)
    return "\n".join(L)
