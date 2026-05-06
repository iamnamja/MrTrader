"""Backfill regime_score_at_scan / regime_label_at_scan / regime_trigger_at_scan
on existing proposal_log rows that pre-date Phase R3.

Step 1: Score each backfill regime_snapshot row that has NULL regime_score
        using the trained regime model.
Step 2: Join proposal_log rows to nearest regime_snapshot by date and copy scores.

Idempotent — safe to re-run.
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _score_snapshot(snap, xgb_model, iso_model, feature_names) -> tuple[float, str] | None:
    """Score a single snapshot using the loaded model. Returns (score, label) or None."""
    feats = [getattr(snap, f, None) for f in feature_names]
    if any(f is None for f in feats):
        # Fill neutral defaults for structurally-null features
        feat_dict = {}
        for fname in feature_names:
            v = getattr(snap, fname, None)
            if v is None:
                if fname == "nis_risk_numeric":
                    v = 0.5
                elif fname == "nis_sizing_factor":
                    v = 1.0
                elif fname == "breadth_pct_ma50":
                    v = 0.5
                else:
                    return None  # can't score with missing core feature
            feat_dict[fname] = v
        feats = [feat_dict[f] for f in feature_names]

    import numpy as np
    X = np.array([feats])
    raw = xgb_model.predict_proba(X)[0][1]
    score = float(iso_model.predict([raw])[0])
    label = "RISK_OFF" if score < 0.35 else ("RISK_ON" if score >= 0.65 else "NEUTRAL")
    return round(score, 4), label


def main() -> int:
    from app.database.session import init_db, get_session
    from app.database.models import ProposalLog, RegimeSnapshot
    from app.ml.regime_features import REGIME_FEATURE_NAMES

    init_db()

    # Load the trained regime model
    import pickle
    from app.ml.regime_model import MODEL_DIR
    candidates = sorted(MODEL_DIR.glob("regime_model_v*.pkl"))
    if not candidates:
        logger.error("No trained regime model found — run train_regime_model.py first")
        return 1
    model_path = candidates[-1]
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    xgb_model = payload["xgb_model"]
    iso_model = payload["iso_model"]
    feature_names = payload["feature_names"]
    model_version_int = payload.get('version')  # integer (e.g. 2)
    model_version = f"regime_v{model_version_int}" if model_version_int is not None else "regime_v?"
    logger.info("Loaded regime model: %s from %s", model_version, model_path.name)

    with get_session() as session:
        # Step 1: Score backfill snapshots that have NULL regime_score
        unscored_snaps = (
            session.query(RegimeSnapshot)
            .filter(
                RegimeSnapshot.snapshot_trigger == "backfill",
                RegimeSnapshot.regime_score.is_(None),
            )
            .all()
        )
        logger.info("Unscored backfill snapshots: %d", len(unscored_snaps))

        snap_scored = 0
        snap_skipped = 0
        for snap in unscored_snaps:
            result = _score_snapshot(snap, xgb_model, iso_model, feature_names)
            if result is None:
                snap_skipped += 1
                continue
            score, label = result
            snap.regime_score = score
            snap.regime_label = label
            if model_version_int is not None:
                snap.model_version = model_version_int
            snap_scored += 1

        session.commit()
        logger.info("Backfill snapshots scored: %d  skipped: %d", snap_scored, snap_skipped)

        # Step 1b: Backfill model_version on already-scored backfill snapshots that have NULL model_version
        if model_version_int is not None:
            mv_null_snaps = (
                session.query(RegimeSnapshot)
                .filter(
                    RegimeSnapshot.snapshot_trigger == "backfill",
                    RegimeSnapshot.regime_score.isnot(None),
                    RegimeSnapshot.model_version.is_(None),
                )
                .all()
            )
            mv_updated = 0
            for snap in mv_null_snaps:
                snap.model_version = model_version_int
                mv_updated += 1
            session.commit()
            logger.info("Backfill snapshots model_version patched: %d", mv_updated)

        # Reload snap_by_date with now-scored snapshots
        snaps = (
            session.query(RegimeSnapshot)
            .filter(
                RegimeSnapshot.snapshot_trigger == "backfill",
                RegimeSnapshot.regime_score.isnot(None),
            )
            .all()
        )
        snap_by_date = {s.snapshot_date: s for s in snaps}
        sorted_dates = sorted(snap_by_date.keys())
        logger.info("Usable backfill snapshots with scores: %d", len(snap_by_date))

        if not snap_by_date:
            logger.error("No scored snapshots available — cannot backfill proposal_log")
            return 1

        # Step 2: Backfill proposal_log rows
        null_rows = (
            session.query(ProposalLog)
            .filter(ProposalLog.regime_score_at_scan.is_(None))
            .all()
        )
        logger.info("ProposalLog rows with NULL regime_score: %d", len(null_rows))

        updated = 0
        skipped = 0

        for row in null_rows:
            if row.proposed_at is None:
                skipped += 1
                continue

            proposal_date = row.proposed_at.date()
            snap = snap_by_date.get(proposal_date)
            if snap is None:
                prior_dates = [d for d in sorted_dates if d <= proposal_date]
                if not prior_dates:
                    skipped += 1
                    continue
                snap = snap_by_date[prior_dates[-1]]

            row.regime_score_at_scan = snap.regime_score
            row.regime_label_at_scan = snap.regime_label
            row.regime_trigger_at_scan = "backfill"
            updated += 1

        session.commit()
        logger.info("ProposalLog backfill: updated=%d  skipped=%d", updated, skipped)

        remaining = (
            session.query(ProposalLog)
            .filter(ProposalLog.regime_score_at_scan.is_(None))
            .count()
        )
        logger.info("Remaining NULL regime_score_at_scan rows: %d", remaining)

    return 0


if __name__ == "__main__":
    sys.exit(main())
