"""Refresh data/macro/macro_history.parquet from yfinance.

Incremental: only fetches dates after the latest stored row. If the file
doesn't exist, initialises from 2018-01-01 (~6 years of history for training).

Usage:
    python scripts/update_macro_history.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("update_macro_history")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from app.data.macro_history import update_macro_history, MACRO_PATH, TICKER_COLUMNS

    df = update_macro_history()

    if df is None or df.empty:
        logger.warning("Macro history is empty after update")
        return

    print()
    print(f"File:      {MACRO_PATH}")
    print(f"Rows:      {len(df)}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Columns:   {df.columns.tolist()}")
    print()
    print("Coverage per ticker (non-null counts):")
    for col in TICKER_COLUMNS.values():
        if col in df.columns:
            non_null = int(df[col].notna().sum())
            print(f"  {col:8s}: {non_null} / {len(df)}")

    # Quick gap check
    import pandas as pd
    dates = pd.to_datetime(df["date"])
    gaps = dates.diff().dt.days
    big_gaps = gaps[gaps > 5]
    if len(big_gaps) > 0:
        print(f"\nNote: {len(big_gaps)} gap(s) > 5 days (expected around holidays/weekends)")


if __name__ == "__main__":
    main()
