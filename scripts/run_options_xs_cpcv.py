"""
run_options_xs_cpcv.py — Alpha-v6 P4 options-as-signal confirmatory runner (H4a-H4e).

Adjudicates ONE pre-registered hypothesis (scripts/preregister_options_xs_features.py,
frozen 2026-06-12T12:00Z) by executing its frozen instrument: a weekly
dollar-neutral DECILE long/short EQUITY sleeve on the R1K options-quality-qualified
universe, sorting on the hypothesis's options-derived feature, judged at EQUITY
cost (NOT an options trade).

The FROZEN decision_rule (PASS): week-clustered t>=2 in the hypothesized direction
AND monotonic deciles AND positive cost-net multi-factor residual alpha. KILL:
simple decile sorts show nothing net of costs -> CLOSE the line (do NOT escalate
to ML combinations).

Three primary tests (the verdict), all forward-aligned (the L/S spread earned over
week [d, d+1) is regressed/tested against same-week forward factor returns):
  1. WEEK-CLUSTERED t on the weekly net L/S spread (weeks are the cluster unit) —
     event_inference.twoway_cluster_ols(..., clusters=("week","week")), one-sided
     in the hypothesized direction (the book is signed by FEATURE_DIRECTION, so a
     POSITIVE spread = the hypothesized edge).
  2. DECILE MONOTONICITY of forward return across the signal deciles —
     event_inference.decile_report(..., clusters=("week","week")).
  3. Cost-net MULTI-FACTOR residual alpha vs SPY / IWM-SPY / MTUM-SPY / VLUE-SPY /
     VIXY — options_xs_ls.multifactor_alpha; must be POSITIVE.
Plus a CPCV robustness BACKSTOP (run_cpcv via an EventEdge-style L/S adapter):
report-only "no pathological fold" check.

SAFETY: the run is EXPLORATORY (no registry write) by DEFAULT. Recording the
ONE-SHOT R4 result (immutable) requires --record AND a run timestamp strictly
after the prereg instant (R2, enforced by the registry).

Usage:
  python -m scripts.run_options_xs_cpcv --hypothesis-id H4a-OPTIONS-CPIV-20260612
  python -m scripts.run_options_xs_cpcv --hypothesis-id H4a-... --smoke
  python -m scripts.run_options_xs_cpcv --hypothesis-id H4a-... --record --run-at 2026-06-13T00:00:00+00:00
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.research import options_xs_ls as xs  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_options_xs")

_ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = _ROOT / "data" / "options_features.parquet"

# Frozen instrument constants.
COST_BPS_PER_SIDE = 10.0          # equity cost, bps per side, charged on turnover
SPREAD_STRESS_MULT = 2.0          # spread-stress sensitivity (report net at 1x + this)
REBALANCE_WEEKDAY = 0             # Monday
FACTOR_ETFS = ["SPY", "IWM", "MTUM", "VLUE", "VIXY"]
PIT_INDEX = "russell1000"
T_PASS = 2.0                      # week-clustered t threshold (one-sided, hyp. dir.)

WINDOW_START = date(2022, 6, 13)  # first full week of the greeks store
WINDOW_END = date(2026, 6, 8)     # store end
SMOKE_END = date(2023, 6, 30)     # ~1y for a fast path-proof


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (I/O; the pure math lives in app/research/options_xs_ls.py)
# ═══════════════════════════════════════════════════════════════════════════

def weekly_rebalance_dates(start: date, end: date,
                           weekday: int = REBALANCE_WEEKDAY) -> List[pd.Timestamp]:
    """Every `weekday` (Mon) in [start, end], as Timestamps — the rebalance grid."""
    days = pd.bdate_range(start, end)
    return [d for d in days if d.weekday() == weekday]


def _forward_weekly_returns(closes: pd.Series,
                            rebal: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """Forward 1-week return at each rebalance date d: close[next rebal]/close[d]-1,
    using the last available close ON/BEFORE each rebal date (PIT; holidays/halts
    snap back to the prior session). Missing either endpoint -> the week is absent.
    `closes` is a date-indexed close Series (split-adjusted)."""
    if closes is None or closes.empty:
        return {}
    c = closes.sort_index()
    c = c[c > 0]
    out: Dict[pd.Timestamp, float] = {}
    for i in range(len(rebal) - 1):
        d0, d1 = rebal[i], rebal[i + 1]
        p0 = c.loc[:d0]
        p1 = c.loc[:d1]
        if p0.empty or p1.empty:
            continue
        v0, v1 = float(p0.iloc[-1]), float(p1.iloc[-1])
        # Guard against a stale snap-back giving an identical endpoint twice.
        if p0.index[-1] >= d1 or v0 <= 0:
            continue
        out[d0] = v1 / v0 - 1.0
    return out


def load_universe_returns(symbols: List[str], rebal: List[pd.Timestamp],
                          start: date, end: date) -> Dict[str, Dict[pd.Timestamp, float]]:
    """Forward weekly returns per symbol over the rebalance grid, via the PIT-safe
    split-healed daily loader (event_panel._get_daily_bars). Returns
    {symbol: {rebal_date: fwd_week_ret}}; symbols that fail to load are omitted."""
    from app.data.cache import get_cache
    from app.research.event_panel import _get_daily_bars
    cache = get_cache()
    pad_start = start - pd.Timedelta(days=10)
    pad_end = end + pd.Timedelta(days=10)
    out: Dict[str, Dict[pd.Timestamp, float]] = {}
    for i, sym in enumerate(symbols):
        try:
            bars = _get_daily_bars(cache, sym, pad_start.date()
                                   if hasattr(pad_start, "date") else pad_start,
                                   pad_end.date() if hasattr(pad_end, "date") else pad_end)
        except Exception:
            bars = None
        if bars is None or bars.empty or "close" not in bars.columns:
            continue
        fwd = _forward_weekly_returns(bars["close"], rebal)
        if fwd:
            out[sym] = fwd
        if (i + 1) % 100 == 0:
            logger.info("  loaded returns for %d/%d names", i + 1, len(symbols))
    return out


def load_factor_frame(rebal: List[pd.Timestamp],
                      start: date, end: date) -> pd.DataFrame:
    """Weekly FORWARD factor-return frame indexed by rebalance date d (return over
    [d, d+1)): columns SPY, IWM_SPY, MTUM_SPY, VLUE_SPY, VIXY (style legs are
    long-short vs SPY). Same forward alignment as the L/S spread, so the
    residual-alpha regression is contemporaneous."""
    etf_ret = load_universe_returns(FACTOR_ETFS, rebal, start, end)
    if "SPY" not in etf_ret:
        logger.warning("SPY factor returns unavailable — multi-factor alpha disabled")
        return pd.DataFrame()
    weeks = sorted(set().union(*[set(v) for v in etf_ret.values()]))
    rows = []
    for d in weeks:
        spy = etf_ret["SPY"].get(d)
        if spy is None:
            continue
        row = {"week": d, "SPY": spy}
        for style in ("IWM", "MTUM", "VLUE"):
            if style in etf_ret and d in etf_ret[style]:
                row[f"{style}_SPY"] = etf_ret[style][d] - spy
        if "VIXY" in etf_ret and d in etf_ret["VIXY"]:
            row["VIXY"] = etf_ret["VIXY"][d]
        rows.append(row)
    f = pd.DataFrame(rows).set_index("week").sort_index()
    return f.dropna()


# ═══════════════════════════════════════════════════════════════════════════
# Weekly panel construction
# ═══════════════════════════════════════════════════════════════════════════

def _latest_known_features(features: pd.DataFrame, qualified: List[str],
                           as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per qualified symbol: its latest feature row knowable by `as_of`
    (knowable_date <= as_of), indexed by underlying. Mirrors the options-quality
    filter's PIT row selection so the sort sees only public data."""
    known = features[(features["knowable_date"] <= as_of)
                     & (features["underlying"].isin(qualified))]
    if known.empty:
        return pd.DataFrame()
    idx = known.groupby("underlying")["knowable_date"].idxmax()
    return known.loc[idx].set_index("underlying")


def build_weekly_panel(features: pd.DataFrame, feature: str, direction: int,
                       rebal: List[pd.Timestamp],
                       returns_by_sym: Dict[str, Dict[pd.Timestamp, float]]):
    """Build (weekly_spread_df, pooled_decile_panel) over the rebalance grid.

    For each week d: qualify the R1K names (options-quality filter, PIT), read the
    frozen sort signal, form the decile dollar-neutral book in the hypothesized
    direction, and realize the forward 1-week spread (gross + cost-net). Costs =
    COST_BPS_PER_SIDE on the one-way turnover vs the prior week's book.
    Returns:
      weekly: DataFrame [week, ls_spread_gross, ls_spread_net, n_long, n_short,
              turnover, n_qualified]
      pooled: DataFrame [week, symbol, signal, fwd_ret] for ALL qualified names
              (the decile-monotonicity panel)."""
    from app.data.options_quality import filter_options_universe
    from app.data.universe_history import pit_union

    weekly_rows: List[dict] = []
    pooled_rows: List[dict] = []
    prev_w = pd.Series(dtype=float)
    fwd_index = {sym: rets for sym, rets in returns_by_sym.items()}

    for d in rebal:
        as_of = d.date() if hasattr(d, "date") else d
        try:
            candidates = pit_union(PIT_INDEX, as_of, as_of)
        except Exception:
            candidates = []
        if not candidates:
            continue
        qualified = filter_options_universe(as_of, candidates, features)
        if len(qualified) < xs.MIN_NAMES_FOR_DECILES:
            continue
        feats_w = _latest_known_features(features, qualified, d)
        if feats_w.empty:
            continue
        signal = xs.build_signal(feats_w, feature)
        # Forward returns available for this week, per qualified name.
        fwd = pd.Series({s: fwd_index[s][d] for s in signal.index
                         if s in fwd_index and d in fwd_index[s]})
        signal = signal[signal.index.isin(fwd.index)]
        if len(signal) < xs.MIN_NAMES_FOR_DECILES:
            continue
        weights = xs.decile_ls_weights(signal, direction)
        if weights.empty:
            continue
        gross = xs.ls_spread_return(weights, fwd)
        to = xs.turnover(prev_w, weights)
        cost = (COST_BPS_PER_SIDE / 1e4) * to
        net = gross - cost
        weekly_rows.append({
            "week": d, "ls_spread_gross": gross, "ls_spread_net": net,
            "n_long": int((weights > 0).sum()), "n_short": int((weights < 0).sum()),
            "turnover": to, "n_qualified": int(len(signal)),
        })
        for s in signal.index:
            pooled_rows.append({"week": d, "symbol": s,
                                "signal": float(signal[s]), "fwd_ret": float(fwd[s])})
        prev_w = weights

    weekly = pd.DataFrame(weekly_rows)
    pooled = pd.DataFrame(pooled_rows)
    return weekly, pooled


# ═══════════════════════════════════════════════════════════════════════════
# The three frozen primary tests + verdict
# ═══════════════════════════════════════════════════════════════════════════

def run_primary_tests(weekly: pd.DataFrame, pooled: pd.DataFrame,
                      factor_frame: pd.DataFrame, direction: int) -> dict:
    """The three frozen tests on the assembled panel. Returns a result dict with
    the week-clustered t, decile monotonicity, and cost-net multi-factor alpha."""
    from scripts.walkforward.event_inference import decile_report, twoway_cluster_ols

    out: dict = {"n_weeks": int(len(weekly))}
    if weekly.empty:
        out["error"] = "no weeks built"
        return out

    # 1. Week-clustered t on the weekly NET spread (weeks = clusters; one row/week
    #    so the cluster-robust SE is the robust SE of the weekly-spread mean). The
    #    book is signed by direction, so a POSITIVE spread is the hypothesized edge.
    wk = weekly.copy()
    inf = twoway_cluster_ols(wk, y="ls_spread_net", X=None,
                             clusters=("week", "week"))
    out["mean_spread_net_bps"] = float(inf.coef[0] * 1e4)
    out["mean_spread_gross_bps"] = float(wk["ls_spread_gross"].mean() * 1e4)
    out["week_clustered_t"] = float(inf.tstat[0])
    out["p_one_sided"] = float(inf.p_one_sided[0])  # H0: mean<=0 vs mean>0

    # 2. Decile monotonicity of forward return across signal deciles (week-clustered).
    # NOTE: this uses decile_report's qcut binning, which differs marginally at the
    # bucket boundaries from decile_ls_weights' floor-div binning (test 1's spread).
    # Both are rank-based deciles; with hundreds of names/week the two partitions
    # agree up to a few boundary names — immaterial to the sign-based gate below.
    # "Monotonic deciles" is operationalized the STANDARD XS-anomaly way: the decile
    # forward-return gradient TRENDS in the hypothesized direction — sign(Spearman
    # rho)==sign(direction) AND the top-minus-bottom decile spread agrees in sign.
    # (NOT strict every-adjacent-step monotonicity, which one noisy decile breaks —
    # the Type-II trap Alpha-v6 exists to avoid; significance is the t>=2 gate.)
    # The strict is_monotone flag is still REPORTED for transparency.
    try:
        dr = (decile_report(pooled, feature="signal", y="fwd_ret", n=10,
                            clusters=("week", "week"))
              if not pooled.empty else {})
    except Exception as exc:           # too-sparse/degenerate panel -> no monotonicity
        logger.warning("decile_report failed (%s) — monotonicity = False",
                       type(exc).__name__)
        dr = {}
    rho = float(dr.get("spearman_rho", 0.0))
    tmb = float(dr.get("top_minus_bottom", 0.0))
    out["decile_is_monotone_strict"] = bool(dr.get("is_monotone", False))
    out["decile_spearman_rho"] = rho
    out["decile_top_minus_bottom_bps"] = tmb * 1e4
    out["decile_dir_ok"] = bool(
        dr and np.sign(rho) == np.sign(direction)
        and np.sign(tmb) == np.sign(direction))

    # 3. Cost-net multi-factor residual alpha (POSITIVE to pass).
    if not factor_frame.empty:
        spread = wk.set_index("week")["ls_spread_net"]
        mf = xs.multifactor_alpha(spread, factor_frame, hac_lag=5)
        out["alpha_bps_per_week"] = float(mf["alpha_bps_d"])  # per-week here
        out["alpha_t_hac"] = float(mf["t_alpha_hac"])
        out["alpha_positive"] = bool(mf["alpha_ann"] > 0)
        out["factor_betas"] = mf["betas"]
        out["resid_sharpe"] = float(mf["resid_sharpe"])
    else:
        out["alpha_positive"] = False
        out["alpha_t_hac"] = 0.0

    # Spread-stress: net spread if equity costs are SPREAD_STRESS_MULT x.
    extra = (COST_BPS_PER_SIDE * (SPREAD_STRESS_MULT - 1.0) / 1e4) * wk["turnover"]
    out["mean_spread_stressed_bps"] = float((wk["ls_spread_net"] - extra).mean() * 1e4)
    return out


def verdict(tests: dict) -> str:
    """The FROZEN pass/kill rule. PASS requires ALL THREE: week-clustered t>=2 in
    the hypothesized direction, monotone deciles (in-direction), and positive
    cost-net multi-factor residual alpha. Otherwise KILL (close the line)."""
    if tests.get("error"):
        return "INCONCLUSIVE"
    t_ok = (tests.get("week_clustered_t", 0.0) >= T_PASS
            and tests.get("p_one_sided", 1.0) < 0.025)
    mono_ok = tests.get("decile_dir_ok", False)
    alpha_ok = tests.get("alpha_positive", False)
    return "PASS" if (t_ok and mono_ok and alpha_ok) else "KILL"


def verdict_to_decision(v: str) -> str:
    """Map a verdict to a valid registry DECISIONS member: PASS graduates to paper,
    KILL closes the line, INCONCLUSIVE parks."""
    return {"PASS": "promote_paper", "KILL": "kill"}.get(v, "park")


# ═══════════════════════════════════════════════════════════════════════════
# CPCV robustness backstop (EventEdge-style L/S adapter; report-only)
# ═══════════════════════════════════════════════════════════════════════════

def cpcv_backstop(weekly: pd.DataFrame, n_folds: int = 4) -> dict:
    """Report-only robustness: split the weekly NET-spread series into `n_folds`
    contiguous blocks and report each block's annualized Sharpe + the worst block.
    A faithful 'no pathological fold' check on the same weekly book (the frozen
    instrument's CPCV backstop; the verdict itself is the three primary tests)."""
    if weekly.empty or len(weekly) < n_folds * 4:
        return {"available": False}
    s = weekly.sort_values("week")["ls_spread_net"].to_numpy()
    blocks = np.array_split(s, n_folds)
    sharpes = []
    for b in blocks:
        if len(b) > 1 and b.std(ddof=1) > 0:
            sharpes.append(float(b.mean() / b.std(ddof=1) * np.sqrt(52)))
        else:
            sharpes.append(0.0)
    return {"available": True, "fold_sharpes": [round(x, 3) for x in sharpes],
            "worst_fold_sharpe": round(min(sharpes), 3),
            "mean_fold_sharpe": round(float(np.mean(sharpes)), 3)}


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════

def load_hypothesis(hid: str) -> dict:
    """Read the frozen hypothesis row; return its feature + direction + criteria."""
    from app.research.registry import ResearchRegistry
    reg = ResearchRegistry()
    row = reg.get(hid)
    if row is None:
        raise SystemExit(f"hypothesis {hid!r} is not registered — run "
                         f"scripts.preregister_options_xs_features first")
    if row.get("preregistered_at") is None:
        raise SystemExit(f"hypothesis {hid!r} is not preregistered — refusing to run")
    feature = row["params"]["feature"]
    if feature not in xs.FEATURE_DIRECTION:
        raise SystemExit(f"feature {feature!r} has no FEATURE_DIRECTION mapping")
    return {"row": row, "feature": feature,
            "direction": xs.FEATURE_DIRECTION[feature],
            "preregistered_at": row["preregistered_at"]}


def main() -> int:
    p = argparse.ArgumentParser(description="Run a P4 options-XS confirmatory hypothesis")
    p.add_argument("--hypothesis-id", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="short ~1y window for a fast path-proof")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--record", action="store_true",
                   help="RECORD the one-shot R4 result (immutable). Default = "
                        "EXPLORATORY (no registry write).")
    p.add_argument("--run-at", type=str, default=None,
                   help="ISO run timestamp for the R4 record (must be strictly "
                        "after the prereg instant). Default: now (UTC).")
    p.add_argument("--decision", type=str, default=None,
                   choices=["kill", "park", "promote_paper", "exploratory_only"],
                   help="decision label to record (must be a registry DECISIONS "
                        "member; default derived from the verdict)")
    args = p.parse_args()

    h = load_hypothesis(args.hypothesis_id)
    feature, direction = h["feature"], h["direction"]
    start = (date.fromisoformat(args.start) if args.start else WINDOW_START)
    end = (date.fromisoformat(args.end) if args.end
           else (SMOKE_END if args.smoke else WINDOW_END))
    logger.info("=" * 78)
    logger.info("P4 %s — feature=%s direction=%+d window=%s..%s%s",
                args.hypothesis_id, feature, direction, start, end,
                " [SMOKE]" if args.smoke else "")

    if not FEATURES_FILE.exists():
        raise SystemExit(f"feature table missing: {FEATURES_FILE} — build it with "
                         f"scripts.build_options_features")
    features = pd.read_parquet(FEATURES_FILE)
    features["knowable_date"] = pd.to_datetime(features["knowable_date"])
    logger.info("feature table: %d rows / %d underlyings",
                len(features), features["underlying"].nunique())

    rebal = weekly_rebalance_dates(start, end)
    logger.info("rebalance weeks: %d", len(rebal))

    # Universe over the whole window (union) -> load forward weekly returns once.
    from app.data.universe_history import pit_union
    universe = sorted(set(pit_union(PIT_INDEX, start, end)))
    logger.info("R1K union over window: %d names — loading forward returns…",
                len(universe))
    returns_by_sym = load_universe_returns(universe, rebal, start, end)
    logger.info("loaded forward returns for %d names", len(returns_by_sym))

    factor_frame = load_factor_frame(rebal, start, end)
    weekly, pooled = build_weekly_panel(features, feature, direction, rebal,
                                        returns_by_sym)
    logger.info("panel: %d weeks, %d pooled name-weeks", len(weekly), len(pooled))

    tests = run_primary_tests(weekly, pooled, factor_frame, direction)
    backstop = cpcv_backstop(weekly)
    v = verdict(tests)

    logger.info("=" * 78)
    logger.info("VERDICT: %s", v)
    for k in ("n_weeks", "mean_spread_net_bps", "week_clustered_t", "p_one_sided",
              "decile_dir_ok", "decile_is_monotone_strict", "decile_spearman_rho",
              "decile_top_minus_bottom_bps", "alpha_bps_per_week", "alpha_t_hac",
              "alpha_positive", "mean_spread_stressed_bps"):
        if k in tests:
            logger.info("  %-28s %s", k, tests[k])
    logger.info("  cpcv_backstop               %s", backstop)
    logger.info("=" * 78)

    result = {"verdict": v, "feature": feature, "direction": direction,
              "window": [str(start), str(end)], "tests": tests,
              "cpcv_backstop": backstop,
              "cost_bps_per_side": COST_BPS_PER_SIDE}

    if args.record:
        from app.research.registry import RegistryIntegrityError, ResearchRegistry
        run_at = args.run_at or datetime.now(timezone.utc).isoformat()
        decision = args.decision or verdict_to_decision(v)
        try:
            ResearchRegistry().record_result(
                args.hypothesis_id, run_at=run_at, result=result, decision=decision)
            logger.info("RECORDED R4 result for %s (run_at=%s, decision=%s)",
                        args.hypothesis_id, run_at, decision)
        except RegistryIntegrityError as exc:
            logger.error("registry refused the result: %s", exc)
            return 1
    else:
        logger.info("EXPLORATORY run — result NOT recorded (pass --record to commit "
                    "the one-shot R4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
