"""
run_pead_v2_hypotheses.py — Alpha-v6 P3 event-conditioned confirmatory runner for
H2 (implied-move reaction_ratio, continuous) and H3 (PEAD v2 options scorecard).

Both adjudicate on the options-enriched event panel (data/event_panel.parquet, the
OPTION_COLUMNS now populated by scripts/enrich_event_panel_options.py). EXPLORATORY
by default; --record gates the ONE-SHOT R4 write (run_at must post-date the
hypothesis's prereg instant, enforced by the registry).

H2 (H2-IMPLIEDMOVE-CONTINUOUS-20260611) — settle OPT-5 PROPERLY as a CONTINUOUS
feature (no binary thresholds, the OPT-5 trap is forbidden): two-way clustered OLS
coefficient of SPY-hedged forward drift on reaction_ratio + decile monotonicity.
CONFIRMED iff coefficient NEGATIVE AND day-clustered t <= -2 AND monotonic deciles
(under-reaction proxy: small realized vs implied move -> more residual drift).
PRIMARY horizon = 10d.

H3 (H3-PEADV2-SCORECARD-20260611) — a regularized monotonic scorecard over 9
FROZEN features; train 2022-24 / validate 2025-26; top-minus-bottom decile hedged
spread t>=2 + monotone on validation. NOTE: H3's frozen feature list includes
`revision_momentum`, which is 100% NULL in the panel (forward-estimate-revision
data is DATA-BLOCKED — see the P4-estrev backlog finding). H3 therefore cannot be
adjudicated as pre-registered; this runner DETECTS the missing frozen feature and
REFUSES to run/record (never consume the one-shot on an un-runnable test) unless
--force-partial is given for an EXPLORATORY (never recordable) 8-feature view.

Usage:
  python -m scripts.run_pead_v2_hypotheses --hypothesis-id H2-IMPLIEDMOVE-CONTINUOUS-20260611
  python -m scripts.run_pead_v2_hypotheses --hypothesis-id H2-... --record
  python -m scripts.run_pead_v2_hypotheses --hypothesis-id H3-PEADV2-SCORECARD-20260611
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pead_v2")

_ROOT = Path(__file__).resolve().parent.parent
PANEL_FILE = _ROOT / "data" / "event_panel.parquet"

HORIZONS = (5, 10, 20)
PRIMARY_H = 10
T_CONFIRM = 2.0

H3_FEATURES = ["sue", "revision_momentum", "gap_vs_vol20", "reaction_ratio",
               "iv_runup_t10_t1", "cpiv_pre", "skew_25d_pre", "opt_volume_z_pre",
               "post_iv_retention_t1"]


# ═══════════════════════════════════════════════════════════════════════════
# H2 — reaction_ratio continuous
# ═══════════════════════════════════════════════════════════════════════════

def run_h2(panel: pd.DataFrame) -> dict:
    """Two-way clustered OLS of SPY-hedged forward drift on reaction_ratio +
    decile monotonicity, at horizons {5,10,20} (primary 10). CONFIRMED iff the
    PRIMARY coefficient is NEGATIVE with day-clustered t<=-2 AND monotone deciles
    in the negative direction."""
    from scripts.walkforward.event_inference import decile_report, twoway_cluster_ols

    out: dict = {"feature": "reaction_ratio", "by_horizon": {}}
    base = panel[panel["reaction_ratio"].notna()].copy()
    for h in HORIZONS:
        y = f"fwd_ret_{h}_spyhedged"
        sub = base[base[y].notna()].copy()
        if len(sub) < 100:
            out["by_horizon"][h] = {"n": int(len(sub)), "error": "too few"}
            continue
        inf = twoway_cluster_ols(sub, y=y, X=["reaction_ratio"],
                                 clusters=("announce_date", "symbol"))
        # coef[1] is the reaction_ratio slope (index 0 is the intercept).
        coef = float(inf.coef[1])
        t = float(inf.tstat[1])
        dr = decile_report(sub, feature="reaction_ratio", y=y, n=10,
                           clusters=("announce_date", "symbol"))
        out["by_horizon"][h] = {
            "n": int(len(sub)), "coef": coef, "coef_t": t,
            "decile_rho": float(dr.get("spearman_rho", 0.0)),
            "decile_monotone_strict": bool(dr.get("is_monotone", False)),
            "decile_top_minus_bottom_bps": float(dr.get("top_minus_bottom", 0.0) * 1e4),
        }
    prim = out["by_horizon"].get(PRIMARY_H, {})
    # CONFIRMED: negative slope, t<=-2, and the decile gradient trends negative.
    coef_neg = prim.get("coef", 0.0) < 0
    t_sig = prim.get("coef_t", 0.0) <= -T_CONFIRM
    mono_neg = prim.get("decile_rho", 0.0) < 0
    out["verdict"] = ("CONFIRMED" if (coef_neg and t_sig and mono_neg)
                      else "NOT_CONFIRMED")
    out["primary_horizon"] = PRIMARY_H
    return out


# ═══════════════════════════════════════════════════════════════════════════
# H3 — scorecard (data-blocked detection)
# ═══════════════════════════════════════════════════════════════════════════

def h3_blocking_features(panel: pd.DataFrame) -> list:
    """Frozen H3 features that are ENTIRELY NULL in the panel (un-runnable)."""
    return [c for c in H3_FEATURES
            if c not in panel.columns or panel[c].notna().sum() == 0]


def run_h3_partial(panel: pd.DataFrame) -> dict:
    """EXPLORATORY-ONLY 8-feature scorecard view (drops the data-blocked
    revision_momentum). NEVER recordable — the frozen feature list is immutable,
    so this is a research peek, not the pre-registered test. Regularized (ridge)
    monotonic linear scorecard: standardize on train (2022-24), fit, score
    validation (2025-26), decile the score, report top-minus-bottom hedged spread."""
    from sklearn.linear_model import Ridge

    from scripts.walkforward.event_inference import decile_report

    feats = [c for c in H3_FEATURES if c != "revision_momentum"]
    y = f"fwd_ret_{PRIMARY_H}_spyhedged"
    d = panel.dropna(subset=feats + [y]).copy()
    d["announce_date"] = pd.to_datetime(d["announce_date"])
    train = d[d["announce_date"] < "2025-01-01"]
    val = d[d["announce_date"] >= "2025-01-01"].copy()
    if len(train) < 200 or len(val) < 200:
        return {"available": False, "n_train": int(len(train)), "n_val": int(len(val))}
    mu, sd = train[feats].mean(), train[feats].std().replace(0, 1.0)
    Xtr = (train[feats] - mu) / sd
    Xval = (val[feats] - mu) / sd
    model = Ridge(alpha=10.0).fit(Xtr, train[y])
    val["score"] = model.predict(Xval)
    dr = decile_report(val, feature="score", y=y, n=10,
                       clusters=("announce_date", "symbol"))
    return {"available": True, "n_train": int(len(train)), "n_val": int(len(val)),
            "val_top_minus_bottom_bps": float(dr.get("top_minus_bottom", 0.0) * 1e4),
            "val_decile_rho": float(dr.get("spearman_rho", 0.0)),
            "val_monotone_strict": bool(dr.get("is_monotone", False)),
            "note": "EXPLORATORY 8-feature view (revision_momentum data-blocked); "
                    "NOT the pre-registered test, NOT recordable."}


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run H2/H3 event-conditioned hypotheses")
    ap.add_argument("--hypothesis-id", required=True)
    ap.add_argument("--panel", default=str(PANEL_FILE))
    ap.add_argument("--record", action="store_true",
                    help="record the one-shot R4 (immutable)")
    ap.add_argument("--run-at", default=None)
    ap.add_argument("--force-partial", action="store_true",
                    help="H3 only: run the EXPLORATORY 8-feature view despite the "
                         "data-blocked frozen feature (never recordable)")
    args = ap.parse_args(argv)

    from app.research.registry import RegistryIntegrityError, ResearchRegistry
    reg = ResearchRegistry()
    row = reg.get(args.hypothesis_id)
    if row is None or row.get("preregistered_at") is None:
        raise SystemExit(f"{args.hypothesis_id} not registered/preregistered")
    panel = pd.read_parquet(args.panel)
    logger.info("panel: %d events", len(panel))

    hid = args.hypothesis_id
    if hid.startswith("H2-"):
        result = run_h2(panel)
        v = result["verdict"]
        prim = result["by_horizon"].get(PRIMARY_H, {})
        logger.info("H2 VERDICT: %s | 10d coef=%.5f t=%.2f rho=%.3f n=%d", v,
                    prim.get("coef", float("nan")), prim.get("coef_t", float("nan")),
                    prim.get("decile_rho", float("nan")), prim.get("n", 0))
        decision = "promote_paper" if v == "CONFIRMED" else "park"
    elif hid.startswith("H3-"):
        blocked = h3_blocking_features(panel)
        if blocked:
            logger.warning("H3 BLOCKED: frozen feature(s) entirely null in panel: %s",
                           blocked)
            logger.warning("H3 cannot be adjudicated as pre-registered "
                           "(forward-estimate-revision data is DATA-BLOCKED). "
                           "Refusing to consume the one-shot R4.")
            if args.force_partial:
                logger.info("partial view: %s", run_h3_partial(panel))
            if args.record:
                logger.error("REFUSING to record H3: %s is data-blocked.", blocked)
            return 2
        return 0  # (would run the full scorecard if ever unblocked)
    else:
        raise SystemExit(f"unknown hypothesis family: {hid}")

    if args.record:
        run_at = args.run_at or datetime.now(timezone.utc).isoformat()
        try:
            reg.record_result(hid, run_at=run_at, result=result, decision=decision)
            logger.info("RECORDED R4 for %s (run_at=%s, decision=%s)",
                        hid, run_at, decision)
        except RegistryIntegrityError as exc:
            logger.error("registry refused: %s", exc)
            return 1
    else:
        logger.info("EXPLORATORY — not recorded (pass --record to commit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
