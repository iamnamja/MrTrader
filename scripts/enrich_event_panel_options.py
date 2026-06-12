"""
enrich_event_panel_options.py — populate the event panel's OPTION_COLUMNS from the
daily options feature table via the PIT event-time join (app/research/
event_options_join.py). Prerequisite for H2 (reaction_ratio) and H3 (the
options-conditioned scorecard).

The equity columns (SUE, hedged forward returns, qualification) are UNTOUCHED —
this only fills the np.nan OPTION_COLUMNS placeholders the panel builder left and
sets options_coverage_flag. Atomic in-place rewrite of data/event_panel.parquet.

Usage:
  python -m scripts.enrich_event_panel_options            # enrich in place
  python -m scripts.enrich_event_panel_options --dry-run  # report coverage only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402

from app.research.event_options_join import compute_event_option_features  # noqa: E402
from app.research.event_panel import OPTION_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_event_panel")

_ROOT = Path(__file__).resolve().parent.parent
PANEL_FILE = _ROOT / "data" / "event_panel.parquet"
FEATURES_FILE = _ROOT / "data" / "options_features.parquet"

# Feature columns the join reads (a thin slice keeps the per-symbol groups small).
# knowable_date gates the PIT pre-event snapshot (chain date + 1 NYSE session).
_FEAT_COLS = ["date", "knowable_date", "atm_iv_30d", "implied_move_front",
              "cpiv_matched_delta", "skew_25d_put", "term_slope_30_60",
              "opt_volume_z"]


def enrich(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `panel` with OPTION_COLUMNS + options_coverage_flag filled
    from `features` (the daily options feature table) at each event's pre-event
    snapshot. Symbols absent from the feature table stay NaN / coverage False."""
    feats = features[[c for c in _FEAT_COLS if c in features.columns]].copy()
    feats["date"] = pd.to_datetime(feats["date"])
    by_sym = {u: g for u, g in
              features.assign(date=pd.to_datetime(features["date"]))[
                  ["underlying"] + _FEAT_COLS].groupby("underlying")}

    out = panel.copy()
    results = []
    for sym, ann, gap in zip(out["symbol"], out["announce_date"],
                             out["announce_gap_pct"]):
        rows = by_sym.get(sym)
        results.append(compute_event_option_features(ann, gap, rows))
    res = pd.DataFrame(results, index=out.index)
    for col in OPTION_COLUMNS:
        out[col] = res[col]
    out["options_coverage_flag"] = res["options_coverage_flag"].astype(bool)
    return out


def _coverage(panel: pd.DataFrame) -> None:
    n = len(panel)
    cov = int(panel["options_coverage_flag"].sum())
    logger.info("options coverage: %d/%d events (%.1f%%)", cov, n, 100.0 * cov / n)
    for col in OPTION_COLUMNS:
        nn = int(panel[col].notna().sum())
        logger.info("  %-24s non-null %6d (%.1f%%)", col, nn, 100.0 * nn / n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Enrich the event panel with options "
                                             "pre-event features")
    ap.add_argument("--panel", default=str(PANEL_FILE))
    ap.add_argument("--features", default=str(FEATURES_FILE))
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + report coverage but do NOT rewrite the panel")
    args = ap.parse_args(argv)

    panel_path, feat_path = Path(args.panel), Path(args.features)
    if not panel_path.exists() or not feat_path.exists():
        logger.error("missing panel (%s) or features (%s)", panel_path, feat_path)
        return 1
    panel = pd.read_parquet(panel_path)
    features = pd.read_parquet(feat_path)
    logger.info("panel: %d events; feature table: %d rows / %d underlyings",
                len(panel), len(features), features["underlying"].nunique())

    enriched = enrich(panel, features)
    _coverage(enriched)

    if args.dry_run:
        logger.info("dry-run — panel NOT rewritten")
        return 0
    tmp = panel_path.parent / f".{panel_path.name}.tmp"
    enriched.to_parquet(tmp, index=False)
    os.replace(tmp, panel_path)
    logger.info("rewrote %s with options-pre-event features", panel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
