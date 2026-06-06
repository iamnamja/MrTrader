"""
Regression guard for the pytest-xdist 'database is locked' CI flake fix.

The recurring flake was ~4 xdist workers (within a CI shard, sharing the runner
filesystem) contending on the same hardcoded SQLite files (feature_store.db, and
the bare universe_history read of it; pead_tracking.db; notifications.db). The fix:
each DB path is env-overridable and conftest points them at a PER-WORKER temp dir
(set at conftest import, before any app module is imported).

These tests assert the isolation is actually active during the test session — so
if someone removes the conftest block or breaks an env-override, CI catches it
here instead of via a flaky 'database is locked' on an unrelated PR.
"""
import os
from pathlib import Path


def _norm(p) -> str:
    return str(p).replace("\\", "/")


def test_feature_store_db_is_isolated_not_the_real_file():
    from app.ml.feature_store import _DEFAULT_DB
    p = _norm(_DEFAULT_DB)
    assert "mrtrader_test_sqlite" in p, f"feature store DB not isolated: {p}"
    assert not p.endswith("app/ml/models/feature_store.db"), \
        f"tests must NOT touch the real feature store: {p}"


def test_pead_tracking_db_is_isolated():
    from app.live_trading.pead_tracker import DB_PATH
    assert "mrtrader_test_sqlite" in _norm(DB_PATH), f"pead tracking DB not isolated: {DB_PATH}"


def test_notifications_db_is_isolated():
    from app.notifications.notifier import DB_PATH
    assert "mrtrader_test_sqlite" in _norm(DB_PATH), f"notifications DB not isolated: {DB_PATH}"


def test_env_overrides_are_set():
    for var in ("MRTRADER_FEATURE_STORE_DB", "MRTRADER_PEAD_TRACKING_DB",
                "MRTRADER_NOTIFICATIONS_DB"):
        assert os.environ.get(var), f"{var} should be set by conftest"
        assert Path(os.environ[var]).parent.exists(), f"{var} dir should exist"


def test_per_worker_path_under_xdist():
    """Under xdist the path is keyed on the worker id; standalone it falls back to
    'gw_main'. Either way the file lives under the isolated test dir."""
    fs = _norm(os.environ["MRTRADER_FEATURE_STORE_DB"])
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw_main")
    assert f"mrtrader_test_sqlite/{worker}/" in fs, \
        f"feature store path {fs} not keyed on worker {worker}"
