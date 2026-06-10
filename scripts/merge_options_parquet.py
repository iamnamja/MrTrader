"""Stream-merge options parquet files with BOUNDED memory.

Combining the existing ~2y options store with a separately-backfilled earlier window into a
single 4y file would, done naively (`pd.concat` of both → one `to_parquet`), hold ~120M+ rows
in RAM and OOM (the exact reason `backfill_options.py --r1k` warns to keep one run to ~2y).

This merges the BARS by streaming one parquet row-group at a time through a single
`pyarrow.ParquetWriter` — peak memory is one row group, not the whole dataset. The input date
windows must be DISJOINT (verified cheaply from row-group statistics; we error on overlap rather
than silently emit duplicate (contract, date) rows). CONTRACTS are small, so they are merged in
memory with a dedup that keeps the EARLIEST first_date/knowable_date per contract (survivorship +
PIT correct). Both outputs are written atomically (tmp + os.replace).

Usage
-----
    python scripts/merge_options_parquet.py \
        --bars data/options_bars_early.parquet data/options_bars.parquet \
        --out-bars data/options_bars.parquet \
        --contracts data/options_contracts_early.parquet data/options_contracts.parquet \
        --out-contracts data/options_contracts.parquet
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_options")


def _date_bounds(path: str, col: str = "date"):
    """(min, max) of `col` from row-group statistics only — no full-column load."""
    pf = pq.ParquetFile(path)
    idx = pf.schema_arrow.get_field_index(col)
    if idx < 0:
        return None, None
    mins, maxs = [], []
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(idx).statistics
        if st is not None and st.has_min_max:
            mins.append(st.min)
            maxs.append(st.max)
    if not mins:
        return None, None
    return min(mins), max(maxs)


def _assert_disjoint(paths: list[str]) -> None:
    bounds = []
    for p in paths:
        lo, hi = _date_bounds(p)
        bounds.append((lo, hi, p))
        logger.info("  %s: date %s → %s", p, lo, hi)
    known = [b for b in bounds if b[0] is not None]
    known.sort(key=lambda b: b[0])
    for (lo1, hi1, p1), (lo2, hi2, p2) in zip(known, known[1:]):
        if hi1 is not None and lo2 is not None and hi1 >= lo2:
            raise SystemExit(
                f"ABORT: date ranges overlap between {p1} (…{hi1}) and {p2} ({lo2}…). "
                "Merging overlapping windows would create duplicate (contract,date) rows. "
                "Backfill the EARLIER window to end one day before the existing store starts."
            )


def merge_bars(inputs: list[str], out: str) -> None:
    """Stream every input's row groups into one parquet (bounded memory). Inputs must be
    schema-compatible (all produced by backfill_options.py) and date-disjoint."""
    _assert_disjoint(inputs)
    tmp = str(out) + ".tmp"
    writer = None
    total = 0
    try:
        for path in inputs:
            pf = pq.ParquetFile(path)
            for i in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(i)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table)
                total += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, out)  # atomic; inputs/readers are closed by now
    logger.info("wrote %s (%d bar rows, streamed from %d files)", out, total, len(inputs))


def merge_contracts(inputs: list[str], out: str) -> None:
    """Concat the (small) contract tables and keep the EARLIEST first_date/knowable_date per
    contract — survivorship + PIT correct when the same contract traded in multiple windows."""
    frames = [pd.read_parquet(p) for p in inputs if Path(p).exists()]
    if not frames:
        raise SystemExit("no contract inputs found")
    df = pd.concat(frames, ignore_index=True)
    agg = {
        "contract_type": "first",
        "strike": "first",
        "expiration": "first",
        "first_date": "min",
        "knowable_date": "min",
    }
    merged = (df.sort_values(["contract", "first_date"])
                .groupby(["underlying", "contract"], as_index=False)
                .agg(agg))
    tmp = str(out) + ".tmp"
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    logger.info("wrote %s (%d contracts, deduped from %d rows)", out, len(merged), len(df))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream-merge options bars + contracts parquet files")
    ap.add_argument("--bars", nargs="+", required=True, help="Input bars parquet files (date-disjoint)")
    ap.add_argument("--out-bars", required=True)
    ap.add_argument("--contracts", nargs="+", required=True, help="Input contracts parquet files")
    ap.add_argument("--out-contracts", required=True)
    args = ap.parse_args()
    logger.info("merging %d bars files → %s", len(args.bars), args.out_bars)
    merge_bars(args.bars, args.out_bars)
    logger.info("merging %d contracts files → %s", len(args.contracts), args.out_contracts)
    merge_contracts(args.contracts, args.out_contracts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
