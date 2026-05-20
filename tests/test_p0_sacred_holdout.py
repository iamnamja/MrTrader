"""
P0 — Sacred holdout enforcement tests.

Verifies:
  * SACRED_HOLDOUT_START is defined and parseable
  * The guard fires (RuntimeError) for end_date >= holdout
  * The guard passes for end_date < holdout
  * The guard boundary is INCLUSIVE (the holdout date itself is rejected)
  * The bypass flag works AND logs a prominent warning banner
  * walkforward_tier3 main() guard fires when "today" is past the holdout
  * ModelTrainer.train_model rejects/accepts based on date.today()
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from app.ml import retrain_config
from app.ml.retrain_config import (
    SACRED_HOLDOUT_START,
    assert_no_sacred_holdout,
    _parse_sacred_holdout_start,
)


# ── 1. Constant defined / parseable ──────────────────────────────────────────

def test_sacred_holdout_constant_defined():
    assert isinstance(SACRED_HOLDOUT_START, str)
    parsed = _parse_sacred_holdout_start()
    assert isinstance(parsed, date)
    # Anchored to the value the roadmap committed to
    # 2026-11-09: reset after 2025-11-09 boundary was exhausted by v182–v185 retrains
    assert SACRED_HOLDOUT_START == "2026-11-09"
    assert parsed == date(2026, 11, 9)


# ── 2. Boundary is inclusive ─────────────────────────────────────────────────

def test_holdout_boundary_is_exclusive_for_safe_dates():
    """Data ENDING the day before the holdout is allowed."""
    boundary = _parse_sacred_holdout_start()
    safe_day = boundary - timedelta(days=1)
    # No exception
    assert_no_sacred_holdout(safe_day, context="test")


def test_holdout_boundary_is_inclusive_for_holdout_day():
    """Data ENDING ON the holdout date itself is rejected — that day is sacred."""
    boundary = _parse_sacred_holdout_start()
    with pytest.raises(RuntimeError, match="SACRED HOLDOUT VIOLATION"):
        assert_no_sacred_holdout(boundary, context="test")


# ── 3. Guard fires for any date past holdout ─────────────────────────────────

def test_guard_rejects_date_after_holdout():
    boundary = _parse_sacred_holdout_start()
    with pytest.raises(RuntimeError, match="SACRED HOLDOUT VIOLATION"):
        assert_no_sacred_holdout(boundary + timedelta(days=180), context="test")


def test_guard_accepts_date_well_before_holdout():
    boundary = _parse_sacred_holdout_start()
    assert_no_sacred_holdout(boundary - timedelta(days=365), context="test")


# ── 4. Bypass flag ───────────────────────────────────────────────────────────

def test_allow_flag_bypasses_guard(caplog):
    boundary = _parse_sacred_holdout_start()
    with caplog.at_level(logging.WARNING, logger="app.ml.retrain_config"):
        assert_no_sacred_holdout(
            boundary + timedelta(days=10),
            allow_sacred_holdout=True,
            context="promotion-run",
        )
    text = caplog.text
    assert "SACRED HOLDOUT BYPASS" in text
    assert "promotion-run" in text


def test_bypass_off_by_default():
    boundary = _parse_sacred_holdout_start()
    with pytest.raises(RuntimeError):
        assert_no_sacred_holdout(boundary + timedelta(days=1))


# ── 5. Input coercion (str / datetime / pandas Timestamp) ───────────────────

def test_guard_accepts_iso_string():
    # 2026-01-01 is now before the 2026-11-09 holdout → safe
    assert_no_sacred_holdout("2026-01-01")
    # 2026-11-09 and beyond must be rejected
    with pytest.raises(RuntimeError):
        assert_no_sacred_holdout("2026-11-09")
    assert_no_sacred_holdout("2024-01-01")


def test_guard_accepts_datetime():
    # 2026-01-01 is now before the 2026-11-09 holdout → safe
    assert_no_sacred_holdout(datetime(2026, 1, 1, 12, 0, 0))
    with pytest.raises(RuntimeError):
        assert_no_sacred_holdout(datetime(2026, 11, 9, 0, 0, 0))


def test_guard_accepts_none_silently():
    # No end_date supplied → nothing to check.
    assert_no_sacred_holdout(None)


# ── 6. ModelTrainer integration ──────────────────────────────────────────────

class _FakeDate(date):
    """Subclass so date.today() can be patched without breaking date(...) calls."""
    pass


def _patch_date_today(monkeypatch, today_value: date):
    """Patch date.today() inside app.ml.training to return today_value."""
    import app.ml.training as training_mod

    class _D(date):
        @classmethod
        def today(cls):
            return today_value

    monkeypatch.setattr(training_mod, "date", _D)


def test_training_rejects_data_after_holdout(monkeypatch):
    """ModelTrainer.train_model fails fast when today >= SACRED_HOLDOUT_START."""
    from app.ml.training import ModelTrainer

    boundary = _parse_sacred_holdout_start()
    _patch_date_today(monkeypatch, boundary + timedelta(days=10))

    trainer = ModelTrainer()
    with pytest.raises(RuntimeError, match="SACRED HOLDOUT VIOLATION"):
        trainer.train_model(symbols=["AAPL"], years=1)


def test_training_allows_data_before_holdout(monkeypatch):
    """ModelTrainer.train_model proceeds past the guard when today < holdout.

    We don't run the full pipeline (slow). We patch _fetch_data to short-circuit
    AFTER the guard runs — so the test verifies only that the guard does NOT
    fire in the safe case.
    """
    from app.ml.training import ModelTrainer

    boundary = _parse_sacred_holdout_start()
    _patch_date_today(monkeypatch, boundary - timedelta(days=30))

    trainer = ModelTrainer()
    # Make _fetch_data raise a sentinel so we know we passed the guard
    sentinel = RuntimeError("PASSED_GUARD_OK")
    monkeypatch.setattr(trainer, "_fetch_data", lambda *a, **kw: (_ for _ in ()).throw(sentinel))

    with pytest.raises(RuntimeError, match="PASSED_GUARD_OK"):
        trainer.train_model(symbols=["AAPL"], years=1)


def test_training_bypass_flag_works(monkeypatch, caplog):
    """allow_sacred_holdout=True bypasses guard in train_model + logs warning."""
    from app.ml.training import ModelTrainer

    boundary = _parse_sacred_holdout_start()
    _patch_date_today(monkeypatch, boundary + timedelta(days=30))

    trainer = ModelTrainer()
    sentinel = RuntimeError("BYPASS_OK")
    monkeypatch.setattr(trainer, "_fetch_data", lambda *a, **kw: (_ for _ in ()).throw(sentinel))

    with caplog.at_level(logging.WARNING, logger="app.ml.retrain_config"):
        with pytest.raises(RuntimeError, match="BYPASS_OK"):
            trainer.train_model(symbols=["AAPL"], years=1, allow_sacred_holdout=True)

    assert "SACRED HOLDOUT BYPASS" in caplog.text


# ── 7. walkforward_tier3 CLI guard ───────────────────────────────────────────

def test_wf_script_rejects_data_after_holdout(monkeypatch):
    """walkforward_tier3 main() must hard-fail when today >= holdout (no bypass)."""
    import scripts.walkforward_tier3 as wf3

    boundary = _parse_sacred_holdout_start()

    class _D(date):
        @classmethod
        def today(cls):
            return boundary + timedelta(days=5)

    monkeypatch.setattr(wf3, "date", _D)
    # Patch sys.argv to a minimal invocation
    test_argv = ["walkforward_tier3.py", "--model", "swing", "--folds", "1", "--years", "1"]
    monkeypatch.setattr("sys.argv", test_argv)

    with pytest.raises(RuntimeError, match="SACRED HOLDOUT VIOLATION"):
        wf3.main()


def test_wf_script_allows_bypass_after_holdout(monkeypatch):
    """walkforward_tier3 main() with --allow-sacred-holdout passes the guard."""
    import scripts.walkforward_tier3 as wf3

    boundary = _parse_sacred_holdout_start()

    class _D(date):
        @classmethod
        def today(cls):
            return boundary + timedelta(days=5)

    monkeypatch.setattr(wf3, "date", _D)
    test_argv = [
        "walkforward_tier3.py", "--model", "swing", "--folds", "1", "--years", "1",
        "--allow-sacred-holdout",
        "--no-earnings-blackout", "--no-macro-gate",  # skip network calls in CI
    ]
    monkeypatch.setattr("sys.argv", test_argv)

    # Mock the heavy data fetch so the test exits immediately after the guard check
    # without crashing the xdist worker process via OOM / training load.
    def _fast_fail(*args, **kwargs):
        raise SystemExit(0)

    monkeypatch.setattr(wf3, "run_swing_walkforward", _fast_fail, raising=False)
    monkeypatch.setattr(wf3, "run_intraday_walkforward", _fast_fail, raising=False)
    monkeypatch.setattr(wf3, "_run_swing_wf", _fast_fail, raising=False)
    monkeypatch.setattr(wf3, "_run_intraday_wf", _fast_fail, raising=False)

    # We expect the run to fail eventually (no model, no data), but NOT due to
    # the sacred-holdout guard. Catch any exception and assert it's not the
    # holdout one.
    try:
        wf3.main()
    except RuntimeError as e:
        assert "SACRED HOLDOUT VIOLATION" not in str(e)
    except SystemExit:
        pass
    except Exception:
        # Any other error is fine — we only care that the guard didn't fire.
        pass
