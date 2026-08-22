"""Numeric model-version selection (2026-08-22).

The bug these guard against was silent and lived in production for six weeks: with 40 regime
models on disk, ``sorted(glob("regime_model_v*.pkl"))[-1]`` returns ``regime_model_v9.pkl``
because "v9" sorts after "v40" as text. The live scorer loaded v9; the retrain-interval
guard aged v9 and therefore retrained daily instead of weekly. No error was ever raised.

Every case below uses double-digit-vs-single-digit versions, since that is precisely where
lexical and numeric ordering diverge.
"""

import pytest

from app.ml.model_versioning import (
    latest_version,
    latest_versioned_file,
    next_version,
    parse_version,
    versioned_files,
)


def make_models(tmp_path, prefix, versions):
    for v in versions:
        (tmp_path / f"{prefix}_v{v}.pkl").write_bytes(b"x")
    return tmp_path


class TestParseVersion:
    def test_parses_trailing_version(self, tmp_path):
        assert parse_version(tmp_path / "regime_model_v40.pkl") == 40

    def test_prefix_containing_v_does_not_confuse(self, tmp_path):
        # `swing_meta_v10.pkl` must read as 10, not as something from the "meta" segment.
        assert parse_version(tmp_path / "swing_meta_v10.pkl") == 10

    def test_non_numeric_returns_none(self, tmp_path):
        assert parse_version(tmp_path / "regime_model_vBACKUP.pkl") is None


class TestLatestBeatsLexical:
    def test_v40_beats_v9(self, tmp_path):
        """THE regression: lexical sort picks v9, numeric picks v40."""
        make_models(tmp_path, "regime_model", range(1, 41))
        assert sorted(tmp_path.glob("regime_model_v*.pkl"))[-1].name == "regime_model_v9.pkl"
        assert latest_versioned_file(tmp_path, "regime_model").name == "regime_model_v40.pkl"

    def test_swing_v223_beats_v99(self, tmp_path):
        make_models(tmp_path, "swing", [99, 100, 222, 223])
        assert latest_versioned_file(tmp_path, "swing").name == "swing_v223.pkl"
        assert latest_version(tmp_path, "swing") == 223

    def test_empty_dir_returns_none_and_zero(self, tmp_path):
        assert latest_versioned_file(tmp_path, "regime_model") is None
        assert latest_version(tmp_path, "regime_model") == 0

    def test_malformed_file_is_skipped_not_fatal(self, tmp_path):
        make_models(tmp_path, "regime_model", [1, 2])
        (tmp_path / "regime_model_vBACKUP.pkl").write_bytes(b"x")
        assert latest_versioned_file(tmp_path, "regime_model").name == "regime_model_v2.pkl"

    def test_prefix_isolation(self, tmp_path):
        """A different prefix must not leak in."""
        make_models(tmp_path, "swing", [5])
        make_models(tmp_path, "intraday", [63])
        assert latest_version(tmp_path, "swing") == 5
        assert latest_version(tmp_path, "intraday") == 63

    def test_files_sorted_ascending_numerically(self, tmp_path):
        make_models(tmp_path, "m", [2, 10, 1, 20])
        assert [p.name for p in versioned_files(tmp_path, "m")] == [
            "m_v1.pkl", "m_v2.pkl", "m_v10.pkl", "m_v20.pkl"
        ]


class TestNextVersion:
    def test_next_is_max_plus_one(self, tmp_path):
        make_models(tmp_path, "regime_model", range(1, 41))
        assert next_version(tmp_path, "regime_model") == 41

    def test_next_on_empty_is_one(self, tmp_path):
        assert next_version(tmp_path, "regime_model") == 1

    def test_max_based_survives_a_deleted_version(self, tmp_path):
        """len(files)+1 would return 40 here and OVERWRITE the live v40."""
        make_models(tmp_path, "regime_model", [v for v in range(1, 41) if v != 7])
        assert len(list(tmp_path.glob("regime_model_v*.pkl"))) == 39
        assert next_version(tmp_path, "regime_model") == 41


class TestRealModelDirectory:
    def test_live_regime_dir_resolves_to_highest(self):
        """Against the real artifacts: must not pick v9."""
        from pathlib import Path
        d = Path("app/ml/models")
        if not d.exists() or not any(d.glob("regime_model_v*.pkl")):
            pytest.skip("no regime models on disk")
        latest = latest_versioned_file(d, "regime_model")
        assert parse_version(latest) == latest_version(d, "regime_model")
        assert parse_version(latest) >= 40
