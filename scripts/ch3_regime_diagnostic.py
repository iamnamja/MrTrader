"""CH3 (Compound-and-Harden) — regime-conditional DIAGNOSTIC (read-only).

Three questions, decomposed by the frozen regime labels (BULL / NEUTRAL / BEAR — the same taxonomy
CH0a is profiled on and CH2 gated against):

  Q1 — When does the LIVE constant-gross trend edge WORK vs WHIPSAW? (per-regime Sharpe / drawdown /
       Calmar / hit-rate / contribution). Extends the CH0a regime profile; combined with CH2's kill
       of conditional sizing, tells us the regime structure is real but not tradeable via gross sizing.

  Q2 — Are the two PARKED collinear strategies (`sector_rotation`, `credit_timing` — the reason they
       were parked is ~0.5 unconditional corr to trend; CH3 recomputes it over the 2008-2026 common
       window) CONDITIONAL diversifiers — is there a regime where they DECORRELATE from trend AND
       stay active/non-losing there? That combination is the ONLY thing that could rescue a keeper: a
       regime-gated version would become a CH4-style pre-registered candidate. Decorrelation alone is
       NOT enough (a long-flat strategy sitting in cash in BEAR decorrelates mechanically). If neither
       holds in any regime, the question is CLOSED.

  Q3 — The CH2c flag: the plain VIX crash governor (LIVE, default ON) does NOT beat constant-gross on
       trend-book Sharpe (−0.0024) and worsens the BEAR-regime Sharpe. But its actual MANDATE is
       crisis TAIL-INSURANCE / drawdown reduction. Does it earn its keep on the RIGHT objective —
       does it reduce max-drawdown / crisis-window losses enough to justify the Sharpe drag?

READ-ONLY: produces a report + a versioned artifact (`docs/reference/ch3_regime_diagnostic.json`).
No live change; no signal search. All regime slicing is PIT (labels ffill'd from CH0a's map).
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone

import pandas as pd

from scripts.ch0_baseline import BASELINE_END, standalone_metrics

log = logging.getLogger(__name__)
ARTIFACT = "docs/reference/ch3_regime_diagnostic.json"
_MIN_REGIME_DAYS = 20   # below this a per-regime stat is not reported (thin bucket)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _aligned_labels(index) -> pd.Series:
    """PIT regime label per date, ffill'd onto `index` (same idiom as CH0a's regime_conditional_sharpe);
    days before the first regime label bucket as UNLABELED rather than being dropped."""
    from scripts.walkforward.regime import load_regime_map
    idx = pd.DatetimeIndex(index)
    rmap = load_regime_map(idx.min().date(), idx.max().date())
    labels = pd.Series({pd.Timestamp(d): v for d, v in rmap.items()}).sort_index()
    return labels.reindex(idx, method="ffill").where(lambda s: s.notna(), "UNLABELED")


# ──────────────────────────────────────────────────────────────────────────────────
# Q1 — the trend edge by regime
# ──────────────────────────────────────────────────────────────────────────────────
def regime_conditional_profile(rets: pd.Series) -> dict:
    """Per-regime return profile of a daily series: Sharpe / ann_return / ann_vol / hit-rate /
    summed-return contribution + day counts (via ch0_baseline.standalone_metrics).

    CAVEAT: `max_drawdown` / `calmar` are computed WITHIN each regime's NON-CONTIGUOUS days
    (interleaved other-regime days, including recoveries, are removed), so they OVERSTATE drawdown
    vs anything the book realized and are NOT realized drawdowns — they are order-dependent
    artifacts, kept only as a rough within-regime shape indicator. The order-INDEPENDENT stats
    (sharpe, ann_return, ann_vol, hit_rate, contribution) are valid; decision-grade drawdown
    evidence is the FULL-SAMPLE number + contiguous crisis windows (see Q3)."""
    r = rets.dropna()
    aligned = _aligned_labels(r.index)
    out: dict = {}
    for label, idx in r.groupby(aligned).groups.items():
        rr = r.loc[idx]
        m = standalone_metrics(rr)   # {} if < 21 obs
        out[str(label)] = {
            "n_days": int(len(rr)), "frac_days": round(len(rr) / len(r), 3),
            "hit_rate": round(float((rr > 0).mean()), 3),
            "contribution_sum_ret": round(float(rr.sum()), 4),
            "sharpe": m.get("sharpe"), "ann_return": m.get("ann_return"),
            "ann_vol": m.get("ann_vol"),
            "max_drawdown_artifact": m.get("max_drawdown"),   # concatenation artifact (see caveat)
            "calmar_artifact": m.get("calmar"),
        }
    return out


def run_q1() -> dict:
    from scripts.walkforward.sleeves import live_trend_book_returns
    trend = live_trend_book_returns(end=BASELINE_END)
    return {
        "question": "Q1 - when does the constant-gross trend edge work vs whipsaw, by regime?",
        "regime_profile": regime_conditional_profile(trend),
        "note": "Confirms + deepens the CH0a profile (BULL/NEUTRAL earn, BEAR bleeds). Combined with "
                "CH2 (conditional gross sizing on these regimes does NOT beat static), the regime "
                "structure is REAL but not tradeable via reactive gross sizing.",
    }


# ──────────────────────────────────────────────────────────────────────────────────
# Q2 — conditional collinearity of the parked strategies
# ──────────────────────────────────────────────────────────────────────────────────
def regime_conditional_corr(frame: pd.DataFrame, base_col: str) -> dict:
    """Per-regime Pearson correlation of every non-base column to `base_col`, over the (already
    inner-joined) frame. 'ALL' = the unconditional reference. Regimes with < _MIN_REGIME_DAYS
    report corr=None (a thin bucket is not correlation-estimable)."""
    aligned = _aligned_labels(frame.index)
    others = [c for c in frame.columns if c != base_col]
    out = {"ALL": {"n_days": int(len(frame)),
                   "corr_to_base": {c: round(float(frame[base_col].corr(frame[c])), 3)
                                    for c in others}}}
    for label, idx in frame.groupby(aligned).groups.items():
        sub = frame.loc[idx]
        if len(sub) < _MIN_REGIME_DAYS:
            out[str(label)] = {"n_days": int(len(sub)), "corr_to_base": None,
                               "note": "too few days to estimate correlation"}
            continue
        out[str(label)] = {"n_days": int(len(sub)),
                           "corr_to_base": {c: round(float(sub[base_col].corr(sub[c])), 3)
                                            for c in others}}
    return out


def _conditional_diversifier_verdict(corr: dict, name: str, parked_profile: dict,
                                     full_ann_vol: float | None, *,
                                     diversify_threshold: float = 0.30,
                                     active_vol_frac: float = 0.30,
                                     bleed_floor: float = -0.05) -> dict:
    """Is `name` a conditional diversifier — a CH4 candidate? Requires BOTH:
      (a) its corr-to-trend drops below `diversify_threshold` in some regime, AND
      (b) it is ACTUALLY ACTIVE + non-losing in that regime (standalone ann_vol >= active_vol_frac
          of its full-sample vol, and ann_return >= bleed_floor).
    (b) is the guard against 'uncorrelated-because-DEAD': the parked strategies are long-flat
    (dual-momentum / go-to-cash-in-stress), so a strategy sitting in cash in BEAR has ~0 variance
    and its correlation collapses MECHANICALLY — decorrelation alone is not diversifying VALUE.
    Even a passing candidate is only a PRE-REGISTERED lead — it must still clear the full CH4 gate."""
    uncond = corr["ALL"]["corr_to_base"].get(name)
    by_regime = {lab: v["corr_to_base"][name] for lab, v in corr.items()
                 if lab != "ALL" and v.get("corr_to_base") and name in v["corr_to_base"]}
    lowest_regime = min(by_regime, key=by_regime.get) if by_regime else None
    lowest = by_regime.get(lowest_regime) if lowest_regime else None
    decorrelates = lowest is not None and lowest < diversify_threshold

    prof = parked_profile.get(lowest_regime, {}) if lowest_regime else {}
    r_vol, r_ret, r_sharpe = prof.get("ann_vol"), prof.get("ann_return"), prof.get("sharpe")
    active = bool(r_vol is not None and full_ann_vol and r_vol >= active_vol_frac * full_ann_vol)
    not_bleeding = bool(r_ret is not None and r_ret >= bleed_floor)
    is_candidate = bool(decorrelates and active and not_bleeding)

    if not decorrelates:
        why = "collinear across all regimes - question CLOSED"
    elif not active:
        why = (f"decorrelates in {lowest_regime} (corr {lowest}) but is FLAT there (ann_vol {r_vol} "
               f"< {active_vol_frac:.0%} of its {full_ann_vol}) - uncorrelated-because-dead, NOT a "
               f"diversifier - CLOSED")
    elif not not_bleeding:
        why = (f"decorrelates in {lowest_regime} (corr {lowest}) but LOSES there (ann_return {r_ret}"
               f" < {bleed_floor}) - CLOSED")
    else:
        why = (f"decorrelates in {lowest_regime} (corr {lowest}) AND is active + non-losing there "
               f"(ann_vol {r_vol}, ann_return {r_ret}) - a pre-registered CH4 candidate")
    return {
        "unconditional_corr": uncond, "by_regime": by_regime,
        "lowest_regime": lowest_regime, "lowest_corr": lowest,
        "regime_standalone": {"ann_vol": r_vol, "ann_return": r_ret, "sharpe": r_sharpe},
        "full_ann_vol": round(float(full_ann_vol), 4) if full_ann_vol else None,
        "conditional_diversifier": is_candidate,
        "verdict": (f"CH4 CANDIDATE - {why}" if is_candidate else f"NOT a candidate - {why}"),
    }


def run_q2() -> dict:
    from scripts.walkforward.sleeve_lab import build_sleeve
    from scripts.walkforward.sleeves import live_trend_book_returns
    trend = live_trend_book_returns(end=BASELINE_END).rename("trend")
    series = [trend]
    built = []
    for name in ("sector_rotation", "credit_timing"):
        try:
            series.append(build_sleeve(name).returns.rename(name))
            built.append(name)
        except Exception:  # noqa: BLE001
            log.exception("ch3 Q2: failed to build %s (skipped)", name)
    frame = pd.concat(series, axis=1).dropna(how="any")
    corr = regime_conditional_corr(frame, "trend")
    parked_profiles = {name: regime_conditional_profile(frame[name]) for name in built}
    full_vols = {name: standalone_metrics(frame[name].dropna()).get("ann_vol") for name in built}
    verdicts = {name: _conditional_diversifier_verdict(corr, name, parked_profiles[name],
                                                       full_vols[name]) for name in built}
    return {
        "question": "Q2 - are the parked collinear strategies CONDITIONAL diversifiers by regime?",
        "coverage": "Common window 2008-2026 (inner-join of trend + both parked). sector_rotation "
                    "ranks whatever SPDR sectors are rankable each day (the 9 original SPDRs date to "
                    "~2000; XLRE/XLC join 2015/2018 but are NOT required), so its BEAR bucket "
                    "(n=1218) INCLUDES GFC-2008. credit_timing (HYG/IEF) covers 2007+.",
        "method_note": "A candidate must (a) decorrelate to < 0.30 in some regime AND (b) be ACTIVE "
                       "+ non-losing there (standalone ann_vol >= 30% of its full-sample vol, "
                       "ann_return >= -5%) - the guard against 'uncorrelated-because-flat' (a "
                       "long-flat strategy in cash during BEAR decorrelates mechanically without "
                       "diversifying). Decorrelation is necessary NOT sufficient; a pass is a "
                       "pre-registered lead that must still clear the full CH4 gate (Track-B + "
                       "detection-lag + DSR). Per-regime max_drawdown/calmar are concatenation "
                       "artifacts (non-adjacent days) - not used here.",
        "common_window": [str(frame.index.min().date()), str(frame.index.max().date())],
        "n_common_days": int(len(frame)),
        "regime_conditional_corr": corr,
        "parked_regime_profiles": parked_profiles,
        "verdicts": verdicts,
    }


# ──────────────────────────────────────────────────────────────────────────────────
# Q3 — governor tail-vs-Sharpe (the CH2c flag) on the RIGHT objective
# ──────────────────────────────────────────────────────────────────────────────────
def run_q3() -> dict:
    from scripts.walkforward.sleeve_lab import DEFAULT_CRISIS_WINDOWS, evaluate_overlay
    from scripts.walkforward.sleeves import build_vix_term_governor, live_trend_book_returns
    trend = live_trend_book_returns(end=BASELINE_END)
    overlay = build_vix_term_governor()   # plain live governor: VixTermGovernorConfig() defaults
    rep = evaluate_overlay(overlay, trend, crisis_windows=DEFAULT_CRISIS_WINDOWS)

    # Regime-slice the with/without book (the governor's SIZING effect; the ~1bp toggle cost is
    # negligible and omitted here). d_max_dd > 0 = the governor made the regime's drawdown shallower.
    m = overlay.multiplier.reindex(trend.index).fillna(1.0)
    with_book = (trend * m).dropna()
    without = trend.reindex(with_book.index)
    aligned = _aligned_labels(with_book.index)
    per_regime: dict = {}
    for label, idx in with_book.groupby(aligned).groups.items():
        w, wo = with_book.loc[idx], without.loc[idx]
        sw, swo = standalone_metrics(w), standalone_metrics(wo)
        if not sw or not swo:
            per_regime[str(label)] = {"n_days": int(len(w)), "note": "too few days"}
            continue
        per_regime[str(label)] = {
            "n_days": int(len(w)),
            "d_sharpe": round(sw["sharpe"] - swo["sharpe"], 4),          # order-independent, valid
            # d_max_dd here is a DIFFERENCE of two concatenation-artifact drawdowns (non-adjacent
            # regime days) - directionally suggestive, NOT a realized-drawdown delta. Decision-grade
            # drawdown evidence is d_max_dd_full + the contiguous crisis windows below.
            "d_max_dd_artifact": round(sw["max_drawdown"] - swo["max_drawdown"], 4),
        }
    bear = per_regime.get("BEAR", {})
    return {
        "question": "Q3 - does the live plain VIX crash governor earn its keep on its MANDATE "
                    "(drawdown / crisis tail), given it costs trend-book Sharpe (CH2c -0.0024)?",
        "overlay_report": rep.to_dict(),
        "per_regime": per_regime,
        "per_regime_caveat": "d_sharpe is valid (order-independent); d_max_dd_artifact is a difference "
                             "of within-regime concatenated (non-adjacent) drawdowns - directional "
                             "only. The DECISION rests on the full-sample d_max_dd + contiguous crises.",
        "interpretation": {
            "reduces_full_sample_maxdd": bool(rep.d_max_dd > 0),          # CONTIGUOUS - decision-grade
            "d_max_dd_full": rep.d_max_dd, "d_sharpe_full": rep.d_sharpe,
            "crisis_dd_improve": {k: round(v.get("dd_improve", 0.0), 4)
                                  for k, v in rep.to_dict()["crisis"].items()},   # contiguous
            "bear_d_sharpe": bear.get("d_sharpe"),
            "overlay_verdict": rep.verdict,
            "read": "Decision-grade evidence = the FULL-SAMPLE d_max_dd (contiguous) + the contiguous "
                    "crisis-window dd_improve. If those show the governor makes crisis drawdowns "
                    "SHALLOWER it EARNS its keep on its tail-insurance mandate DESPITE the small "
                    "Sharpe drag (expected for insurance). If it does NOT reduce crisis drawdown it "
                    "is a pure drag. This is the CH3 adjudication of the CH2c flag - it does NOT "
                    "change the live governor.",
        },
    }


def _print_q(block: dict) -> None:
    print(f"  {block['question']}")
    if "regime_profile" in block:
        for lab, m in block["regime_profile"].items():
            print(f"    {lab:<10} n={m['n_days']:<5} SR {m['sharpe']}  annRet {m['ann_return']}  "
                  f"hit {m['hit_rate']}  contrib {m['contribution_sum_ret']}")
    if "verdicts" in block:
        print(f"    common window {block['common_window']} ({block['n_common_days']} days)")
        for name, v in block["verdicts"].items():
            print(f"    {name:<16} uncond {v['unconditional_corr']}  by_regime {v['by_regime']}")
            print(f"      standalone@{v['lowest_regime']} {v['regime_standalone']} -> {v['verdict']}")
    if "interpretation" in block:
        i = block["interpretation"]
        print(f"    overlay: {i['overlay_verdict']} | full d_maxDD {i['d_max_dd_full']} "
              f"d_sharpe {i['d_sharpe_full']} | crisis dd_improve {i['crisis_dd_improve']}")


def build_report() -> dict:
    return {
        "artifact": "CH3 - regime-conditional diagnostic (read-only; no live change)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "baseline_end": str(BASELINE_END),
        "q1_trend_by_regime": run_q1(),
        "q2_conditional_collinearity": run_q2(),
        "q3_governor_tail": run_q3(),
    }


def main() -> int:
    report = build_report()
    with open(ARTIFACT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"CH3 regime diagnostic -> {ARTIFACT}")
    for key in ("q1_trend_by_regime", "q2_conditional_collinearity", "q3_governor_tail"):
        _print_q(report[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
