"""P2-4 — calibrate the option spread cost surface from the live NBBO observation log.

Reads data/options_spread_obs.parquet (accumulated nightly by the P1c fuse), buckets the
observed relative spread by (contract_type, moneyness, DTE), and writes the calibrated
artifact to data/options_spread_model.json for the OPT-2 simulator's
CalibratedOptionsSpreadCostModel. Re-run as the NBBO log accumulates — the surface sharpens
automatically with more data.

  python -m scripts.calibrate_option_spreads [--obs PATH] [--out PATH] [--as-of YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
from datetime import date

from app.options.spread_model import (
    OBS_PARQUET, MODEL_PATH, calibrate_from_parquet,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default=OBS_PARQUET, help="NBBO observation parquet")
    ap.add_argument("--out", default=MODEL_PATH, help="output calibrated-model JSON")
    ap.add_argument("--as-of", default=None, help="PIT cutoff YYYY-MM-DD (default: all data)")
    a = ap.parse_args()
    as_of = date.fromisoformat(a.as_of) if a.as_of else None

    model = calibrate_from_parquet(a.obs, as_of=as_of)
    print(model.summary())
    if model.n_obs == 0:
        print("\nWARNING: 0 usable observations — wrote a global-default model. The NBBO log "
              "may be empty/young; re-run after more accumulation.")
    model.save(a.out)
    print(f"\nWrote calibrated spread model -> {a.out}")


if __name__ == "__main__":
    main()
