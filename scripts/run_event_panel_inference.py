"""
run_event_panel_inference.py — the event-LEVEL confirmatory runner (Phase 3).

Adjudicates the pre-registered event hypotheses on data/event_panel.parquet
with the two-way (announce_date x firm) cluster-robust instrument
(scripts/walkforward/event_inference.py). The H1 run —

    python scripts/run_event_panel_inference.py \
        --hypothesis-id H1-PEAD-EVENTLEVEL-20260611

— is the ONE confirmatory shot (registry R4) that re-adjudicates the LIVE
PEAD edge. begin_run() FAILS FAST before the panel is even loaded on every
registry violation (unregistered id, not preregistered, already-recorded R4,
R2 ordering), so an accidental re-run can never burn anything.

WHAT IS DECIDED vs WHAT IS REPORTED (frozen in the registry BEFORE this code
ran — scripts/preregister_event_hypotheses.py):
  DECIDES : the PRIMARY one-sided p of the intercept-only two-way test on
            fwd_ret_10_spyhedged over the PEAD-qualified, complete-forward-
            window population.
              p < 0.05  -> GRADUATE  (honest Track-A paper; waiver retired)
              p > 0.15  -> DEMOTE    (live book becomes trend-plus-cash)
              else      -> INCONCLUSIVE (telemetry size; no posture change)
  REPORTED (never re-deciding): 5d/20d horizons, the quarter-cluster block
            bootstrap (the conservative bound, seed 1303 / 10k resamples),
            beta-adjusted hedged returns (decision 3: H1 decides on the UNIT
            hedge), the trend-gated live-B5 slice (decision 1), the
            announce+2-open day-1-momentum gap (decision 2), LOCO
            quarter/sector/top10, decile monotonicity vs pead_score_v1, the
            concentration caps (quarter<=40% / name<=15%) and the 50bps
            gapper-slippage stress (the two-way test on y - 0.0050).

Population: pead_qualified AND no excluding quality bits (incomplete 20d
forward window / missing SPY hedge / unhealed suspect bar series) — see
event_panel.QF_EXCLUDE_FROM_INFERENCE.

ONE-SHOT PROTECTION: a real --hypothesis-id additionally passes a CONFIRMATORY
COVERAGE GATE before any inference runs (default-panel path unless
--allow-nonstandard-panel, event/population floors, 2019->2026 announce span,
>0 qualified events every year 2019..2025). These are coverage SANITY guards
— NOT part of the frozen statistical spec — so a typo'd --panel, the smoke
panel, or an FMP-rate-limit-thinned build can never consume the one-shot.
The panel file's SHA256 + coverage summary are recorded in result_json, so
the registry verdict pins the exact panel it came from (the registry row's
data_hash is fixed null at registration — PR1).

Artifacts are written to logs/ FIRST (durable), then the ASCII report prints;
the registry result and the phase_complete email are best-effort afterwards.
Exit codes: 0 graduate / 2 demote / 3 inconclusive / 1 build-or-input error.

Usage:
    python scripts/run_event_panel_inference.py --exploratory \
        --panel data/event_panel_smoke.parquet          # dev smoke (unrecorded)
    python scripts/run_event_panel_inference.py \
        --hypothesis-id H1-PEAD-EVENTLEVEL-20260611     # THE one-shot
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

logger = logging.getLogger("event_panel_inference")

DEFAULT_PANEL = ROOT / "data" / "event_panel.parquet"
LOG_DIR = ROOT / "logs"

PRIMARY_Y = "fwd_ret_10_spyhedged"          # decision 3: UNIT hedge decides
HORIZONS = (5, 10, 20)
P_GRADUATE = 0.05                            # frozen H1 decision rule
P_DEMOTE = 0.15
QUARTER_PNL_CAP = 0.40                       # frozen caps
NAME_PNL_CAP = 0.15
GAPPER_STRESS = 0.0050                       # 50bps per-event slippage stress
N_RESAMPLES = 10_000                         # quarter-cluster bootstrap (frozen)
BOOTSTRAP_SEED = 1303

# ── Confirmatory coverage SANITY guards (NOT part of the frozen statistical
# spec above — the test, horizons and verdict thresholds are untouched).
# They only protect the one-shot from executing on the WRONG population:
# a typo'd --panel, the ~52-event smoke panel, or a silently thinned build.
MIN_PANEL_EVENTS = 3000          # full R1K 2019->2026 build is far larger
MIN_POPULATION = 400             # qualified+clean events the test runs on
COVERAGE_MIN_YEAR = 2019         # announce span must reach back to 2019...
COVERAGE_MAX_YEAR = 2026         # ...and forward into 2026
QUALIFIED_YEARS_REQUIRED = tuple(range(2019, 2026))   # >0 qualified each year

VERDICTS = ("GRADUATE", "DEMOTE", "INCONCLUSIVE")


def verdict_from_p(p: float) -> str:
    """The frozen H1 decision rule on the PRIMARY one-sided p."""
    if p < P_GRADUATE:
        return "GRADUATE"
    if p > P_DEMOTE:
        return "DEMOTE"
    return "INCONCLUSIVE"


def concentration_caps(pop, y: str) -> dict:
    """Frozen caps: no single quarter > 40% of P&L, no single name > 15%.

    'P&L' here is the sum of hedged event returns (equal-weight event book).
    Shares are only meaningful against a POSITIVE total — a non-positive
    total fails the caps by definition (and H1 itself would already be dead).
    """
    import pandas as pd
    from scripts.pead_significance import cluster_key

    total = float(pop[y].sum())
    if total <= 0:
        return {"total_pnl": total, "max_quarter_share": None, "max_quarter": None,
                "max_name_share": None, "max_name": None,
                "quarter_cap_ok": False, "name_cap_ok": False,
                "note": "non-positive total P&L — shares undefined, caps FAIL"}
    quarters = pop["announce_date"].map(lambda d: cluster_key(pd.Timestamp(d).date()))
    q_share = pop.groupby(quarters)[y].sum() / total
    n_share = pop.groupby(pop["symbol"])[y].sum() / total
    return {
        "total_pnl": total,
        "max_quarter_share": float(q_share.max()),
        "max_quarter": str(q_share.idxmax()),
        "max_name_share": float(n_share.max()),
        "max_name": str(n_share.idxmax()),
        "quarter_cap_ok": bool(q_share.max() <= QUARTER_PNL_CAP),
        "name_cap_ok": bool(n_share.max() <= NAME_PNL_CAP),
    }


def select_population(panel) -> "tuple":
    """The pre-registered H1 population: PEAD-qualified events with a complete
    forward window and a SPY hedge (event_panel.QF_EXCLUDE_FROM_INFERENCE).
    Returns (population, exclusion_accounting_dict)."""
    from app.research.event_panel import QF_EXCLUDE_FROM_INFERENCE

    n_all = len(panel)
    qualified = panel[panel["pead_qualified"].astype(bool)]
    clean = qualified[
        (qualified["quality_flags"].astype(int) & QF_EXCLUDE_FROM_INFERENCE) == 0
    ]
    pop = clean[clean[PRIMARY_Y].notna()]
    return pop, {
        "panel_events": n_all,
        "qualified": int(len(qualified)),
        "excluded_quality": int(len(qualified) - len(clean)),
        "excluded_missing_primary_y": int(len(clean) - len(pop)),
        "population": int(len(pop)),
    }


def _sha256_file(path: Path) -> str:
    """SHA256 of the panel file — pins the exact bytes the verdict came from
    (the registry row's data_hash is fixed null at registration, so the hash
    lives in result_json where the recorded verdict is auditable)."""
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_coverage(panel) -> dict:
    """Coverage summary recorded alongside the verdict: total events, the H1
    population size, announce-year span, per-year qualified counts."""
    import pandas as pd
    years = pd.to_datetime(panel["announce_date"]).dt.year
    qualified = panel["pead_qualified"].astype(bool)
    pop, _ = select_population(panel)
    return {
        "panel_events": int(len(panel)),
        "population": int(len(pop)),
        "min_year": int(years.min()),
        "max_year": int(years.max()),
        "per_year_qualified": {
            int(y): int((qualified & (years == y)).sum())
            for y in sorted(years.unique())
        },
    }


def enforce_confirmatory_coverage(panel, panel_path: Path,
                                  allow_nonstandard_panel: bool = False) -> dict:
    """The confirmatory coverage gate (sanity guards, NOT the frozen stat
    spec): a real --hypothesis-id consumes a one-shot, so refuse to run on
    anything but the full default panel with non-degenerate coverage. Returns
    the coverage summary on success; raises ValueError listing every failure."""
    if Path(panel_path).resolve() != DEFAULT_PANEL.resolve() \
            and not allow_nonstandard_panel:
        raise ValueError(
            f"confirmatory runs must use the default full panel "
            f"({DEFAULT_PANEL}) — got {panel_path}. A smoke/alt panel would "
            f"consume the one-shot on the wrong population; pass "
            f"--allow-nonstandard-panel only if this is deliberate."
        )
    cov = panel_coverage(panel)
    problems = []
    if cov["panel_events"] < MIN_PANEL_EVENTS:
        problems.append(f"panel has {cov['panel_events']} events "
                        f"(floor {MIN_PANEL_EVENTS}) — smoke/thinned build?")
    if cov["population"] < MIN_POPULATION:
        problems.append(f"population is {cov['population']} events "
                        f"(floor {MIN_POPULATION})")
    if cov["min_year"] > COVERAGE_MIN_YEAR or cov["max_year"] < COVERAGE_MAX_YEAR:
        problems.append(f"announce span {cov['min_year']}->{cov['max_year']} "
                        f"does not cover {COVERAGE_MIN_YEAR}->"
                        f"{COVERAGE_MAX_YEAR}")
    degenerate = [y for y in QUALIFIED_YEARS_REQUIRED
                  if cov["per_year_qualified"].get(y, 0) <= 0]
    if degenerate:
        problems.append(f"zero qualified events in year(s) {degenerate} — "
                        f"silently thinned build?")
    if problems:
        raise ValueError(
            "confirmatory coverage gate REFUSED (one-shot protection; "
            "coverage sanity guards, not the frozen stat spec): "
            + "; ".join(problems))
    return cov


def run_inference(panel, n_resamples: int = N_RESAMPLES,
                  seed: int = BOOTSTRAP_SEED) -> dict:
    """The full pre-registered analysis on an event panel (pure; tested)."""
    import numpy as np
    import pandas as pd
    from app.ml.retrain_config import SACRED_HOLDOUT_START
    from scripts.pead_significance import cluster_key, event_clustered_bootstrap
    from scripts.walkforward.event_inference import (
        decile_report, loco_robustness, twoway_cluster_ols,
    )

    # Sacred-holdout assertion (the builder guarantees it; re-assert anyway).
    holdout = date.fromisoformat(SACRED_HOLDOUT_START)
    ann_dates = pd.to_datetime(panel["announce_date"]).dt.date
    assert (ann_dates < holdout).all(), "panel reaches the sacred holdout"

    pop, accounting = select_population(panel)
    if len(pop) < 30:
        raise ValueError(f"population too thin for inference ({len(pop)} events)")

    def _twoway(sub, y):
        return twoway_cluster_ols(sub, y=y, clusters=("announce_date", "symbol"))

    # ── PRIMARY + secondary horizons (intercept-only mean test) ────────────
    horizons = {}
    for h in HORIZONS:
        ycol = f"fwd_ret_{h}_spyhedged"
        res = _twoway(pop, ycol)
        horizons[f"{h}d"] = res.summary_dict()
    primary = horizons["10d"]
    p_primary = float(primary["p_one_sided"][0])

    # ── Quarter-cluster block bootstrap — the CONSERVATIVE bound ───────────
    rets = pop[PRIMARY_Y].astype(float).tolist()
    clusters = [cluster_key(pd.Timestamp(d).date()) for d in pop["announce_date"]]
    boot = event_clustered_bootstrap(rets, clusters, n_resamples, seed)

    # ── Reported robustness (never re-deciding) ────────────────────────────
    # Beta-adjusted hedge (decision 3: reported ALONGSIDE, unit hedge decides).
    # spy window leg == raw - unit-hedged, so betaadj = raw - beta*(raw-hedged).
    sub_beta = pop[pop["beta_60d"].notna()].copy()
    beta_adjusted = None
    if len(sub_beta) >= 30:
        spy_leg = sub_beta["fwd_ret_10_raw"] - sub_beta[PRIMARY_Y]
        sub_beta["_betaadj_10"] = sub_beta["fwd_ret_10_raw"] - \
            sub_beta["beta_60d"] * spy_leg
        beta_adjusted = _twoway(sub_beta, "_betaadj_10").summary_dict()

    # Trend-gated live-B5 slice (decision 1: REPORTED, non-deciding).
    trend_slice = None
    flags = pop["spy_below_200d"]
    known = flags.notna()
    is_below = flags.fillna(False).astype(bool)  # fillna: nullable boolean dtype
    up_pop = pop[known & ~is_below]
    if len(up_pop) >= 30:
        trend_slice = _twoway(up_pop, PRIMARY_Y).summary_dict()
        trend_slice["n_blocked_by_trend_gate"] = int((known & is_below).sum())

    # Day-1 momentum the live book forfeits (decision 2: reported ONCE).
    gap = (pop["entry_open_next2"] / pop["entry_open_next"] - 1.0).dropna()
    day1_gap = {"n": int(len(gap)), "mean": float(gap.mean()),
                "median": float(gap.median())}
    if len(gap) >= 30:
        gap_panel = pop.loc[gap.index, ["announce_date", "symbol"]].assign(_gap=gap)
        day1_gap["twoway"] = _twoway(gap_panel, "_gap").summary_dict()

    def _infer(sub):
        return _twoway(sub, PRIMARY_Y)

    loco = {
        unit: loco_robustness(pop, PRIMARY_Y, unit, _infer)
        for unit in ("quarter", "sector", "top10")
    }
    try:
        deciles = decile_report(pop, "pead_score_v1", PRIMARY_Y,
                                n=min(10, max(2, len(pop) // 10)))
    except ValueError as exc:
        deciles = {"error": str(exc)}

    # ── Frozen caps + gapper-slippage stress ───────────────────────────────
    caps = concentration_caps(pop, PRIMARY_Y)
    stress_panel = pop[["announce_date", "symbol"]].assign(
        _stressed=pop[PRIMARY_Y] - GAPPER_STRESS)
    stress = _twoway(stress_panel, "_stressed").summary_dict()
    stress["stress_bps"] = GAPPER_STRESS * 1e4

    verdict = verdict_from_p(p_primary)
    return {
        "primary_y": PRIMARY_Y,
        "decision_rule": {"graduate_p_lt": P_GRADUATE, "demote_p_gt": P_DEMOTE},
        "population": accounting,
        "horizons": horizons,
        "primary_p_one_sided": p_primary,
        "primary_mean": float(primary["coef"][0]),
        "primary_t": float(primary["tstat"][0]),
        "quarter_bootstrap_conservative": boot,
        "reported_not_deciding": {
            "beta_adjusted_10d": beta_adjusted,
            "trend_gated_slice_10d": trend_slice,
            "day1_momentum_gap": day1_gap,
            "loco": loco,
            "deciles_pead_score_v1": deciles,
        },
        "caps": caps,
        "gapper_stress_50bps": stress,
        "verdict": verdict,
        "n_obs": int(np.asarray(primary["n_obs"]).item()),
    }


def _print_report(out: dict, hypothesis_id) -> None:
    bar = "=" * 96
    sub = "-" * 96
    print()
    print(bar)
    print(f"EVENT-PANEL INFERENCE  (two-way announce_date x firm clustered)"
          f"   hypothesis={hypothesis_id or 'EXPLORATORY'}")
    print(bar)
    acc = out["population"]
    print(f"  population: {acc['population']} events "
          f"({acc['qualified']} qualified of {acc['panel_events']} panel events; "
          f"{acc['excluded_quality']} excluded on quality bits, "
          f"{acc['excluded_missing_primary_y']} missing primary y)")
    print(sub)
    print(f"  {'horizon':>8} | {'mean':>9} | {'se':>8} | {'t':>7} | "
          f"{'p(1s)':>8} | {'clusters d x f':>14} | psd_fixed")
    for h in HORIZONS:
        r = out["horizons"][f"{h}d"]
        nc = r["n_clusters"]
        days = nc.get("announce_date", "?")
        firms = nc.get("symbol", "?")
        print(f"  {h:>7}d | {r['coef'][0]*1e4:>+8.1f}bp | "
              f"{r['se'][0]*1e4:>6.1f}bp | {r['tstat'][0]:>+7.2f} | "
              f"{r['p_one_sided'][0]:>8.4f} | {days:>6} x {firms:<5} | "
              f"{r['psd_fixed']}")
    print(sub)
    b = out["quarter_bootstrap_conservative"]
    print(f"  quarter-cluster bootstrap (conservative bound, seed {BOOTSTRAP_SEED}): "
          f"p={b['p_value']:.4f}  CI=[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]  "
          f"({b['n_clusters']} quarters)")
    rep = out["reported_not_deciding"]
    if rep["beta_adjusted_10d"]:
        r = rep["beta_adjusted_10d"]
        print(f"  beta-adjusted 10d (reported): mean={r['coef'][0]*1e4:+.1f}bp "
              f"t={r['tstat'][0]:+.2f} p(1s)={r['p_one_sided'][0]:.4f}")
    if rep["trend_gated_slice_10d"]:
        r = rep["trend_gated_slice_10d"]
        print(f"  trend-gated (live-B5) slice (reported): n={r['n_obs']} "
              f"mean={r['coef'][0]*1e4:+.1f}bp t={r['tstat'][0]:+.2f} "
              f"(blocked: {r['n_blocked_by_trend_gate']})")
    g = rep["day1_momentum_gap"]
    print(f"  day-1 momentum forfeited by announce+2 entry (reported once): "
          f"mean={g['mean']*1e4:+.1f}bp median={g['median']*1e4:+.1f}bp n={g['n']}")
    dec = rep["deciles_pead_score_v1"]
    if "error" not in dec:
        print(f"  deciles vs pead_score_v1: monotone={dec['is_monotone']} "
              f"rho={dec['spearman_rho']:+.2f} "
              f"top-bottom={dec['top_minus_bottom']*1e4:+.1f}bp")
    for unit, rows in rep["loco"].items():
        ps = [r["p_one_sided"] for r in rows if r.get("p_one_sided") is not None]
        if ps:
            print(f"  LOCO {unit:<8}: worst p(1s)={max(ps):.4f} over {len(rows)} cuts")
    caps = out["caps"]
    print(sub)
    if caps.get("max_quarter_share") is not None:
        print(f"  caps: max quarter share={caps['max_quarter_share']:.1%} "
              f"({caps['max_quarter']}, cap {QUARTER_PNL_CAP:.0%}) "
              f"{'OK' if caps['quarter_cap_ok'] else 'BREACH'}; "
              f"max name share={caps['max_name_share']:.1%} "
              f"({caps['max_name']}, cap {NAME_PNL_CAP:.0%}) "
              f"{'OK' if caps['name_cap_ok'] else 'BREACH'}")
    else:
        print(f"  caps: {caps.get('note')}")
    s = out["gapper_stress_50bps"]
    print(f"  50bps gapper stress: mean={s['coef'][0]*1e4:+.1f}bp "
          f"t={s['tstat'][0]:+.2f} p(1s)={s['p_one_sided'][0]:.4f}")
    print(bar)
    print(f"  PRIMARY 10d one-sided p = {out['primary_p_one_sided']:.4f}  "
          f"->  VERDICT: {out['verdict']}")
    print(f"  (frozen rule: p<{P_GRADUATE} GRADUATE / p>{P_DEMOTE} DEMOTE / "
          f"else INCONCLUSIVE; promotion remains owner-gated)")
    print(bar)
    print()


def main(argv=None) -> int:
    from scripts.walkforward.registry_enforcement import add_arguments, begin_run

    ap = argparse.ArgumentParser(
        description="Event-panel inference (two-way clustered) — H1/H2/H3 runner")
    add_arguments(ap)  # --hypothesis-id / --exploratory
    ap.add_argument("--panel", default=str(DEFAULT_PANEL),
                    help=f"event panel parquet ({DEFAULT_PANEL.relative_to(ROOT)})")
    ap.add_argument("--allow-nonstandard-panel", action="store_true",
                    help="confirmatory runs only: deliberately override the "
                         "default-panel path check (the coverage floors still "
                         "apply — this never bypasses them)")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES,
                    help=f"quarter-bootstrap resamples ({N_RESAMPLES}; frozen "
                         f"for the confirmatory run)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [event_inference] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )
    try:
        if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Registry enforcement FAILS FAST — before the panel is even loaded (R2/R4).
    run = begin_run(args.hypothesis_id, exploratory=args.exploratory)

    import pandas as pd
    panel_path = Path(args.panel)
    if not panel_path.exists():
        logger.error("panel not found: %s (build it with "
                     "scripts/build_event_panel.py)", panel_path)
        return 1
    panel = pd.read_parquet(panel_path)
    logger.info("panel loaded: %s (%d events)", panel_path, len(panel))

    # Pin the exact panel the verdict comes from + coverage; under a real
    # hypothesis id the coverage GATE must pass BEFORE any inference runs
    # (one-shot protection — record() is never reached on refusal).
    panel_sha256 = _sha256_file(panel_path)
    if args.hypothesis_id is not None:
        try:
            coverage = enforce_confirmatory_coverage(
                panel, panel_path,
                allow_nonstandard_panel=args.allow_nonstandard_panel)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("confirmatory coverage gate OK: %s", coverage)
    else:
        coverage = panel_coverage(panel)

    try:
        out = run_inference(panel, n_resamples=args.n_resamples)
    except (ValueError, AssertionError) as exc:
        logger.error("inference failed: %s", exc)
        return 1
    out["panel_path"] = str(panel_path)
    out["panel_sha256"] = panel_sha256
    out["panel_coverage"] = coverage
    out["hypothesis_id"] = args.hypothesis_id
    out["exploratory"] = bool(args.exploratory)
    out["n_resamples"] = int(args.n_resamples)

    # Artifacts FIRST (durable), then the guarded ASCII print.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    artifact = LOG_DIR / f"event_panel_inference_{stamp}.json"
    artifact.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    out["artifact"] = str(artifact)
    logger.info("artifact written: %s", artifact)
    try:
        _print_report(out, args.hypothesis_id)
    except Exception as exc:  # pragma: no cover - defensive console guard
        logger.warning("report print failed (%s: %s); artifact already at %s",
                       type(exc).__name__, exc, artifact)

    # Best-effort registry record. decision=None: promotion is owner-gated,
    # never auto-decided from one run's metrics (the verdict is in the result).
    if run is not None:
        run.record({
            "verdict": out["verdict"],
            "primary_y": PRIMARY_Y,
            "primary_p_one_sided": out["primary_p_one_sided"],
            "primary_mean": out["primary_mean"],
            "primary_t": out["primary_t"],
            "n_obs": out["n_obs"],
            "n_clusters": out["horizons"]["10d"]["n_clusters"],
            "bootstrap_p": out["quarter_bootstrap_conservative"]["p_value"],
            "caps": out["caps"],
            "gapper_stress_p": out["gapper_stress_50bps"]["p_one_sided"][0],
            "artifact": str(artifact),
            "panel_path": str(panel_path),
            "panel_sha256": panel_sha256,
            "panel_coverage": coverage,
        }, decision=None)

    try:
        from app.notifications import notifier
        notifier.enqueue("phase_complete", {
            "phase": "P3 event-panel inference"
                     + (f" ({args.hypothesis_id})" if args.hypothesis_id else
                        " (exploratory)"),
            "tasks_done": f"two-way clustered inference on {out['n_obs']} events",
            "outcome": f"VERDICT {out['verdict']} — primary 10d one-sided "
                       f"p={out['primary_p_one_sided']:.4f}",
            "next_phase": "owner adjudication per frozen H1 decision rule",
            "notes": f"artifact: {artifact}",
        })
    except Exception as exc:
        logger.warning("notifier enqueue failed: %s", exc)

    return {"GRADUATE": 0, "DEMOTE": 2, "INCONCLUSIVE": 3}[out["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
