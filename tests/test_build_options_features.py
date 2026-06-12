"""
Tests for scripts/build_options_features.py — the feature-table build driver.

Focus: the assemble step concats EVERY part on disk (a single-name refresh must
not silently truncate the full table), the resume filter, and atomic output.
"""
from __future__ import annotations

import pandas as pd

from app.data.options_features import FEATURE_COLS
from scripts.build_options_features import (
    assemble_final, pending_underlyings,
)


def _part(underlying, dates):
    """A minimal valid feature part (FEATURE_COLS) for one underlying."""
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        rows.append({
            "underlying": underlying, "date": d,
            "knowable_date": d + pd.offsets.BDay(1),
            "atm_iv_30d": 0.3, "implied_move_front": 0.05,
            "cpiv_matched_delta": 0.01, "skew_25d_put": 0.03,
            "term_slope_30_60": 0.02, "iv_rv_20d_ratio": 1.1,
            "opt_share_volume_ratio": 2.0, "put_call_volume_ratio": 1.0,
            "opt_volume_z": 0.0, "total_opt_volume": 5000.0,
            "n_valid_contracts": 30, "coverage_flags": 0,
        })
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def _write_part(parts_dir, underlying, dates):
    parts_dir.mkdir(parents=True, exist_ok=True)
    _part(underlying, dates).to_parquet(parts_dir / f"{underlying}.parquet",
                                        index=False)


def test_assemble_includes_all_parts_not_just_a_subset(tmp_path):
    # Three names already have parts on disk (a prior full build). A subsequent
    # single-name refresh must still assemble ALL THREE — never truncate to one.
    parts = tmp_path / "parts"
    for u in ("AAA", "BBB", "CCC"):
        _write_part(parts, u, ["2024-03-01", "2024-03-04"])
    out = tmp_path / "features.parquet"
    n = assemble_final(parts, out)
    full = pd.read_parquet(out)
    assert n == 6
    assert set(full["underlying"].unique()) == {"AAA", "BBB", "CCC"}
    # Sorted by (date, underlying); columns exactly FEATURE_COLS.
    assert list(full.columns) == FEATURE_COLS
    assert full["date"].is_monotonic_increasing


def test_assemble_skips_tmp_staging_files(tmp_path):
    parts = tmp_path / "parts"
    _write_part(parts, "AAA", ["2024-03-01"])
    # A leftover atomic-write staging file must be ignored, not concatenated.
    (parts / ".BBB.parquet.tmp").write_bytes(b"garbage")
    out = tmp_path / "features.parquet"
    n = assemble_final(parts, out)
    assert n == 1
    assert set(pd.read_parquet(out)["underlying"].unique()) == {"AAA"}


def test_assemble_no_parts_writes_nothing(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    out = tmp_path / "features.parquet"
    assert assemble_final(parts, out) == 0
    assert not out.exists()


def test_pending_underlyings_resume_and_force(tmp_path):
    parts = tmp_path / "parts"
    _write_part(parts, "AAA", ["2024-03-01"])
    # AAA has a part -> skipped on resume; BBB is pending.
    assert pending_underlyings(["AAA", "BBB"], parts, force=False) == ["BBB"]
    # --force rebuilds everything.
    assert pending_underlyings(["AAA", "BBB"], parts, force=True) == ["AAA", "BBB"]
