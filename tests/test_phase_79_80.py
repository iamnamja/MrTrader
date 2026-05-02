"""
Tests for Phase 79 (Point-in-Time Index Membership) and Phase 80 (Bar Sensitivity).
"""
import pytest
from datetime import date
from unittest.mock import patch
import pandas as pd
import io


# ─── Phase 79: Point-in-Time Universe ────────────────────────────────────────

class TestMembersAt:
    """Tests for app.data.universe_history.members_at()."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from app.data.universe_history import invalidate_cache
        invalidate_cache()
        yield
        invalidate_cache()

    def _make_parquet(self, rows: list[dict]) -> bytes:
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        return buf.read()

    def _patch_parquet(self, index: str, rows: list[dict]):
        """Patch _load_membership to return a DataFrame from rows."""
        import app.data.universe_history as uh
        df = pd.DataFrame(rows)
        df["added"] = pd.to_datetime(df["added"])
        df["removed"] = pd.to_datetime(df["removed"], errors="coerce")
        return patch.object(uh, "_load_membership", return_value=df)

    def test_ticker_present_after_add_date(self):
        rows = [{"ticker": "WORK", "added": "2019-09-26", "removed": None}]
        from app.data.universe_history import members_at, invalidate_cache, _load_membership
        with self._patch_parquet("sp500", rows):
            result = members_at("sp500", date(2020, 1, 1))
        assert "WORK" in result

    def test_ticker_absent_before_add_date(self):
        rows = [{"ticker": "TSLA", "added": "2020-12-21", "removed": None}]
        with self._patch_parquet("sp500", rows):
            from app.data.universe_history import members_at
            result = members_at("sp500", date(2020, 6, 1))
        assert "TSLA" not in result

    def test_ticker_absent_after_remove_date(self):
        rows = [{"ticker": "WORK", "added": "2019-09-26", "removed": "2021-07-22"}]
        with self._patch_parquet("sp500", rows):
            from app.data.universe_history import members_at
            result = members_at("sp500", date(2022, 1, 1))
        assert "WORK" not in result

    def test_ticker_present_on_add_date(self):
        rows = [{"ticker": "TSLA", "added": "2020-12-21", "removed": None}]
        with self._patch_parquet("sp500", rows):
            from app.data.universe_history import members_at
            result = members_at("sp500", date(2020, 12, 21))
        assert "TSLA" in result

    def test_ticker_absent_on_remove_date(self):
        # removed is the first day it's OUT
        rows = [{"ticker": "WORK", "added": "2019-09-26", "removed": "2021-07-22"}]
        with self._patch_parquet("sp500", rows):
            from app.data.universe_history import members_at
            result = members_at("sp500", date(2021, 7, 22))
        assert "WORK" not in result

    def test_multiple_tickers_mixed(self):
        rows = [
            {"ticker": "AAPL", "added": "1982-11-30", "removed": None},
            {"ticker": "TSLA", "added": "2020-12-21", "removed": None},
            {"ticker": "WORK", "added": "2019-09-26", "removed": "2021-07-22"},
        ]
        with self._patch_parquet("sp500", rows):
            from app.data.universe_history import members_at
            result = members_at("sp500", date(2021, 1, 1))
        assert "AAPL" in result
        assert "TSLA" in result
        assert "WORK" in result

    def test_sivb_removed_after_march_2023(self):
        rows = [{"ticker": "SIVB", "added": "2018-03-19", "removed": "2023-03-10"}]
        with self._patch_parquet("sp500", rows):
            from app.data.universe_history import members_at
            pre = members_at("sp500", date(2023, 3, 1))
            post = members_at("sp500", date(2023, 3, 15))
        assert "SIVB" in pre
        assert "SIVB" not in post

    def test_fallback_when_no_parquet(self):
        """When no parquet, members_at falls back to static constant."""
        import app.data.universe_history as uh
        with patch.object(uh, "_load_membership", return_value=None):
            from app.data.universe_history import members_at
            result = members_at("sp500", date(2023, 1, 1))
        assert len(result) > 0  # static list non-empty

    def test_real_parquet_files_exist(self):
        """Integration: verify seeded parquet files are loadable."""
        from app.data.universe_history import members_at
        result = members_at("sp500", date(2021, 1, 1))
        assert "AAPL" in result
        assert "MSFT" in result

    def test_work_in_2021_not_2022_real_parquet(self):
        """WORK (Slack) removed 2021-07-22 — spot-check real parquet."""
        from app.data.universe_history import members_at
        assert "WORK" in members_at("sp500", date(2021, 1, 1))
        assert "WORK" not in members_at("sp500", date(2022, 1, 1))

    def test_tsla_not_in_sp500_pre_dec_2020(self):
        """TSLA added 2020-12-21 — should not appear in mid-2020."""
        from app.data.universe_history import members_at
        assert "TSLA" not in members_at("sp500", date(2020, 6, 1))
        assert "TSLA" in members_at("sp500", date(2021, 1, 1))


# ─── Phase 80: Bar Sensitivity Sweep ─────────────────────────────────────────

class TestBarSensitivity:
    """Basic smoke tests for the bar sensitivity sweep script."""

    def test_bar_sensitivity_script_exists(self):
        from pathlib import Path
        script = Path("scripts/bar_sensitivity.py")
        assert script.exists(), "scripts/bar_sensitivity.py not found"

    def test_bar_sensitivity_imports(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "bar_sensitivity", Path("scripts/bar_sensitivity.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Just check it parses; don't execute __main__ block
        spec.loader.exec_module(mod)

    def test_sensitivity_range_includes_baseline(self):
        """The sweep must include bar offset 12 (current baseline)."""
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "bar_sensitivity", Path("scripts/bar_sensitivity.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "SWEEP_OFFSETS") or hasattr(mod, "DEFAULT_OFFSETS")
        offsets = getattr(mod, "SWEEP_OFFSETS", getattr(mod, "DEFAULT_OFFSETS", []))
        assert 12 in offsets, f"Baseline bar 12 missing from sweep offsets: {offsets}"
