"""
multistrat_eval.py — Phase A: the unified combined-book walk-forward evaluator.

Answers "how does this SET of strategies perform AS ONE BOOK, under the same WF rigor as a single
sleeve?" — so adding a strategy is a one-call holistic check. It WIRES TOGETHER the existing pieces
(no reinvention):

  • assemble the book   — sleeve_allocator.build_book (PIT inverse-vol / equal / regime, weekly-rebal,
                          turnover-costed). The book is itself a return series.
  • A2 combined-book WF — run that book return series through the SAME CPCV the single-sleeve gate
                          uses (SeriesReturnStrategy + run_cpcv) → a trustworthy BOOK-level verdict
                          (mean Sharpe, path-t, %positive, worst path) + a family-level deflated Sharpe.
  • A3 risk governor    — optionally apply the risk-policy-v1 drawdown de-gross LADDER to the book
                          (PIT) and re-run the WF, so we SEE the governor's effect holistically. (This
                          is the return-level part of the live whole-book gate; the notional/beta caps
                          are position-level and stay in the live R0.5 gate.)
  • A4 attribution      — per-sleeve standalone Sharpe + avg weight + leave-one-out Track-B marginal
                          (appraise_track_b: candidate vs the book of the OTHERS) + GL-1 cross-strategy
                          tail diagnostics (exceedance corr / down-up beta / co-crash).
  • A5 ragged history   — the COMMON-window book (inner-join, apples-to-apples → the CPCV verdict) AND
                          a fold-in UNION book (each sleeve contributes from its own start, weights
                          renormalised over the present sleeves) reported descriptively.
  • P0.5                — the book's parametric deflation uses the enumerated family_trial_count().

Report-only / research — no live trading path. Phase B (covariance/ERC sizing) and Phase C
(research↔live replay parity) are data-gated (see HOLISTIC_MULTISTRAT_STATE_AND_SCOPE doc).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── results ───────────────────────────────────────────────────────────────────
@dataclass
class SleeveContribution:
    label: str
    avg_weight: float
    standalone_sharpe: float
    track_b_appraisal_ir: float        # leave-one-out residual-α info ratio (book-hedged Sharpe)
    track_b_t_alpha_hac: float         # NW-HAC t of the leave-one-out residual alpha
    track_b_passed: Optional[bool]     # None when it's the only sleeve (no "others" to hedge against)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BookWF:
    """Book-level walk-forward summary (from a CPCVResult on the book return series)."""
    mean_sharpe: float
    path_sharpe_tstat: float
    pct_positive: float
    p5_sharpe: float
    calmar: float
    n_obs: int
    n_folds: int
    dsr_family_p: float                # parametric deflated-Sharpe p at the family trial count
    paper_passed: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MultiStratReport:
    window_start: str
    window_end: str
    n_days: int
    scheme: str
    n_families: int
    book_raw: BookWF
    book_governed: Optional[BookWF]    # None unless apply_governor=True
    sleeves: List[SleeveContribution] = field(default_factory=list)
    tail: Optional[dict] = None
    union: Optional[dict] = None       # fold-in (ragged-history) descriptive summary
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("book_raw", "book_governed", "sleeves")}
        d["book_raw"] = self.book_raw.to_dict()
        d["book_governed"] = self.book_governed.to_dict() if self.book_governed else None
        d["sleeves"] = [s.to_dict() for s in self.sleeves]
        return d


# ── helpers ───────────────────────────────────────────────────────────────────
def _clean(s: pd.Series, name: str = "returns") -> pd.Series:
    s = pd.Series(s).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _sharpe(r: pd.Series, ann: int = 252) -> float:
    r = _clean(r)
    sd = float(r.std())
    return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0


def assemble_common_book(sleeve_returns: Dict[str, pd.Series], *, scheme: str = "vol", cfg=None):
    """Combine sleeve return streams into one book on their COMMON (inner-join) dates via the
    proven sleeve_allocator. Returns the BookResult (has .returns, .weights, .sharpe, ...)."""
    from app.strategy.sleeve_allocator import build_book, AllocatorConfig
    frame = pd.DataFrame({k: _clean(v) for k, v in sleeve_returns.items()}).dropna(how="any")
    if frame.empty:
        raise ValueError("sleeves share no common dates — cannot assemble a book")
    return build_book(frame, scheme=scheme, cfg=cfg or AllocatorConfig())


def assemble_union_book(sleeve_returns: Dict[str, pd.Series], *, vol_lookback: int = 60,
                        rebalance_days: int = 5, ann: int = 252, cost_bps: float = 1.0) -> pd.Series:
    """Fold-in (ragged-history) book: every sleeve contributes from its OWN start; inverse-vol
    weights are renormalised each day over the sleeves PRESENT that day. Uses ALL data (vs the
    common-window inner-join). Descriptive only — the CPCV verdict is on the common-window book."""
    frame = pd.DataFrame({k: _clean(v) for k, v in sleeve_returns.items()}).sort_index()
    if frame.empty:
        raise ValueError("no sleeve data for the union book")
    rv = frame.rolling(vol_lookback, min_periods=max(vol_lookback // 2, 10)).std() * np.sqrt(ann)
    raw_w = 1.0 / rv                                   # NaN where a sleeve is absent or vol unknown
    w = raw_w.div(raw_w.sum(axis=1, skipna=True), axis=0)   # renormalise over present sleeves
    n = len(frame)
    is_rebal = (np.arange(n) % rebalance_days == 0)
    held = w.where(pd.Series(is_rebal, index=frame.index), other=np.nan).ffill().fillna(0.0)
    gross = (held.shift(1) * frame.fillna(0.0)).sum(axis=1)
    dw = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = dw * (cost_bps / 1e4)
    net = (gross - cost.shift(1)).dropna()
    # drop the leading warmup run (all sleeves still in vol-lookback → held all-zero → 0 returns)
    # so n_days / Sharpe aren't diluted by no-position days. cost matches the common book (1bps).
    return net.loc[net.ne(0).cumsum() > 0].rename("union_book")


def apply_drawdown_ladder(book_returns: pd.Series, policy=None) -> pd.Series:
    """Apply the risk-policy-v1 drawdown de-gross LADDER to a book return stream (PIT): the gross
    multiplier for day t is set from the drawdown known at the END of t-1 (shift(1)), so it never
    peeks. De-grosses as the book draws down; restores as it recovers."""
    from app.live_trading.risk_policy import RISK_POLICY_V1
    policy = policy or RISK_POLICY_V1
    r = _clean(book_returns)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0)
    mult = dd.apply(policy.ladder_multiplier).shift(1).fillna(1.0)
    return (r * mult).rename("governed_book")


def book_walkforward(book_returns: pd.Series, *, spy_prices: Optional[pd.Series] = None,
                     regime_map: Optional[dict] = None, n_folds: Optional[int] = None,
                     n_paths: Optional[int] = None, purge_days: Optional[int] = None,
                     embargo_days: Optional[int] = None, total_years: Optional[int] = None,
                     n_families: Optional[int] = None, component_type: str = "risk_premium",
                     regime_waiver_approved: bool = True, label: str = "book") -> BookWF:
    """A2 — run the BOOK return series through the SAME CPCV as a single sleeve (the book IS a
    return series), then summarise. The parametric deflated-Sharpe uses the enumerated family
    trial count (P0.5)."""
    from scripts.walkforward.gate_calibration import (
        SeriesReturnStrategy, FULL_N_FOLDS, FULL_N_PATHS, FULL_PURGE_DAYS, FULL_EMBARGO_DAYS)
    from scripts.walkforward.cpcv import run_cpcv
    from scripts.walkforward.gates import deflated_sharpe_ratio
    from app.research import ruler_v2
    from app.research.family_registry import family_trial_count

    r = _clean(book_returns)
    n_folds = FULL_N_FOLDS if n_folds is None else int(n_folds)
    n_paths = FULL_N_PATHS if n_paths is None else int(n_paths)
    purge_days = FULL_PURGE_DAYS if purge_days is None else int(purge_days)
    embargo_days = FULL_EMBARGO_DAYS if embargo_days is None else int(embargo_days)
    if n_families is None:
        n_families = family_trial_count()

    if regime_map is None:
        from scripts.walkforward.regime import load_regime_map
        regime_map = load_regime_map(r.index.min().date(), r.index.max().date())

    strat = SeriesReturnStrategy(label, r, spy_prices=spy_prices, regime_map=regime_map)
    res = run_cpcv(strategy=strat, purge_days=purge_days, embargo_days=embargo_days,
                   n_folds=n_folds, n_paths=n_paths, total_years=total_years)
    res.component_type = component_type
    _, dsr_p = deflated_sharpe_ratio(res.mean_sharpe, n_families, res._dsr_n_obs())
    paper_passed = ruler_v2.gate_passed(res, tier="paper", component_type=component_type,
                                        regime_waiver_approved=regime_waiver_approved)
    return BookWF(
        mean_sharpe=res.mean_sharpe, path_sharpe_tstat=res.path_sharpe_tstat,
        pct_positive=res.pct_positive, p5_sharpe=res.p5_sharpe, calmar=res.avg_calmar,
        n_obs=len(res.oos_returns_dated), n_folds=res.n_folds,
        dsr_family_p=float(dsr_p), paper_passed=bool(paper_passed))


def per_sleeve_contributions(sleeve_returns: Dict[str, pd.Series], *, scheme: str = "vol",
                             component_types: Optional[Dict[str, str]] = None,
                             n_boot: int = 2000, seed: int = 0) -> List[SleeveContribution]:
    """A4 — per-sleeve attribution: standalone Sharpe + avg book weight + LEAVE-ONE-OUT Track-B
    (does this sleeve add residual-α on top of the book of the OTHERS?). NOTE: avg_weight is the
    sleeve's mean weight in the FULL book (a descriptive attribution), NOT in the others-only base
    the Track-B is appraised against."""
    from scripts.walkforward.track_b_appraisal import appraise_track_b, TrackBAppraisalCriteria
    component_types = component_types or {}
    labels = list(sleeve_returns)
    out: List[SleeveContribution] = []

    # avg weights from the common-window book (one assemble; reused for all sleeves)
    try:
        book = assemble_common_book(sleeve_returns, scheme=scheme)
        avg_w = {c: float(book.weights[c].mean()) for c in book.weights.columns}
    except Exception:  # noqa: BLE001
        avg_w = {}

    for lab in labels:
        cand = _clean(sleeve_returns[lab])
        others = {k: v for k, v in sleeve_returns.items() if k != lab}
        tb = None
        if others:
            try:
                base = assemble_common_book(others, scheme=scheme).returns
                tb = appraise_track_b(
                    _clean(base), cand,
                    component_type=component_types.get(lab, "risk_premium"),
                    criteria=TrackBAppraisalCriteria.from_retrain_config(),
                    regime_waiver_approved=True, n_boot=n_boot, seed=seed, candidate_label=lab)
            except Exception as exc:  # noqa: BLE001
                log.warning("per_sleeve Track-B failed for %s: %s", lab, exc)
        out.append(SleeveContribution(
            label=lab, avg_weight=avg_w.get(lab, float("nan")),
            standalone_sharpe=_sharpe(cand),
            track_b_appraisal_ir=float(tb.appraisal_ir) if tb else float("nan"),
            track_b_t_alpha_hac=float(tb.t_alpha_hac) if tb else float("nan"),
            track_b_passed=(bool(tb.passed) if tb else None)))
    return out


def run_multistrat_eval(sleeve_returns: Dict[str, pd.Series], *, spy: Optional[pd.Series] = None,
                        spy_prices: Optional[pd.Series] = None, scheme: str = "vol",
                        component_types: Optional[Dict[str, str]] = None,
                        apply_governor: bool = True, run_tail: bool = True,
                        n_boot: int = 2000, seed: int = 0,
                        cpcv_kw: Optional[dict] = None) -> MultiStratReport:
    """The Phase A entry point. Given {label: daily-return-series}, evaluate them AS ONE BOOK:
    combined-book WF (raw + governed), per-sleeve attribution + Track-B, GL-1 tail, and a
    fold-in union book. `cpcv_kw` overrides the CPCV GEOMETRY ONLY (n_folds/n_paths/purge_days/
    embargo_days/total_years/regime_map); reserved keys (n_families/spy_prices/label) are stripped
    so they can't collide with the args this function sets."""
    if len(sleeve_returns) < 1:
        raise ValueError("run_multistrat_eval needs >=1 sleeve")
    cpcv_kw = dict(cpcv_kw or {})
    for _reserved in ("n_families", "spy_prices", "label"):
        cpcv_kw.pop(_reserved, None)        # geometry only — these are set by this function
    from app.research.family_registry import family_trial_count
    n_families = family_trial_count()

    book = assemble_common_book(sleeve_returns, scheme=scheme)
    book_r = book.returns
    notes: List[str] = []

    book_raw = book_walkforward(book_r, spy_prices=spy_prices, n_families=n_families, **cpcv_kw)

    book_governed = None
    if apply_governor:
        governed = apply_drawdown_ladder(book_r)
        book_governed = book_walkforward(governed, spy_prices=spy_prices, n_families=n_families,
                                         label="book_governed", **cpcv_kw)
        notes.append("governed = risk-policy-v1 drawdown de-gross ladder applied to the book (PIT). "
                     "Return-level governor only; notional/beta caps are the live R0.5 gate.")

    sleeves = per_sleeve_contributions(sleeve_returns, scheme=scheme,
                                       component_types=component_types, n_boot=n_boot, seed=seed)

    tail = None
    if run_tail and spy is not None and len(sleeve_returns) >= 2:
        try:
            from app.research.tail_diagnostics import run_tail_diagnostics
            frame = pd.DataFrame({k: _clean(v) for k, v in sleeve_returns.items()}).dropna(how="any")
            td = run_tail_diagnostics(frame, _clean(spy))
            # extract the scalar/dict fields (skip crisis_table — a DataFrame, not summary-grade)
            tail = {
                "uncond_corr_avg": td.uncond_corr_avg,
                "spy_exceedance": td.spy_exceedance,
                "spy_exceedance_post2015": td.spy_exceedance_post2015,
                "core_asymmetry_ci": td.core_asymmetry_ci,
                "one_bet": td.one_bet,
                "vrp_crises_worse": td.vrp_crises_worse,
                "vrp_crises_total": td.vrp_crises_total,
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("tail diagnostics failed: %s", exc)

    union = None
    try:
        ub = assemble_union_book(sleeve_returns)
        common_n = len(book_r)
        union = {"n_days": int(len(ub)), "sharpe": _sharpe(ub),
                 "window_start": str(ub.index.min().date()), "window_end": str(ub.index.max().date()),
                 "common_window_n_days": common_n,
                 "uses_extra_days": int(len(ub) - common_n)}
        if len(ub) > common_n:
            notes.append(f"union (fold-in) book spans {len(ub)} days vs {common_n} common — "
                         f"{len(ub) - common_n} extra days used (descriptive; CPCV is on the common book).")
    except Exception as exc:  # noqa: BLE001
        log.warning("union book failed: %s", exc)

    return MultiStratReport(
        window_start=str(book_r.index.min().date()), window_end=str(book_r.index.max().date()),
        n_days=len(book_r), scheme=scheme, n_families=n_families,
        book_raw=book_raw, book_governed=book_governed, sleeves=sleeves, tail=tail,
        union=union, notes=notes)
