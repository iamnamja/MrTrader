"""
Test-log isolation (live-system health-check item 2).

pytest used to write into the live ops log (logs/mrtrader_<date>.log) — the same
file the running paper server writes to — producing hundreds of fake "Orchestrator
started" banners and test-only errors that made the live log unusable for ops.

The fix: _DailyFileHandler detects pytest (PYTEST_CURRENT_TEST in env, or pytest
in sys.modules) and switches the file prefix to "test_mrtrader_". Production
behavior is unchanged (the env var is never set outside pytest).

These tests assert that, under pytest, the handler targets the test prefix and
NOT the live mrtrader_<date>.log.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from app.main import _DailyFileHandler


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def test_prefix_is_test_under_pytest():
    """Under pytest the daily-file prefix must be the isolated test prefix."""
    assert _DailyFileHandler._prefix() == "test_mrtrader_"


def test_handler_does_not_open_live_log(tmp_path):
    """A handler constructed under pytest must NOT open logs/mrtrader_<date>.log."""
    handler = _DailyFileHandler(log_dir=str(tmp_path))
    try:
        live_path = tmp_path / f"mrtrader_{_today()}.log"
        test_path = tmp_path / f"test_mrtrader_{_today()}.log"
        assert not live_path.exists(), "handler opened the LIVE ops log under pytest"
        assert test_path.exists(), "handler did not open the isolated test log"
    finally:
        handler.close()


def test_emit_writes_only_to_test_log(tmp_path):
    """A log record routed through the handler lands in the test log, not the live log."""
    handler = _DailyFileHandler(log_dir=str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord(
            name="mrtrader.test", level=logging.INFO, pathname=__file__,
            lineno=1, msg="Orchestrator started — isolation probe", args=(), exc_info=None,
        )
        handler.emit(record)
        live_path = tmp_path / f"mrtrader_{_today()}.log"
        test_path = tmp_path / f"test_mrtrader_{_today()}.log"
        assert not live_path.exists()
        assert "isolation probe" in test_path.read_text(encoding="utf-8")
    finally:
        handler.close()


def test_no_active_daily_handler_targets_live_log():
    """No _DailyFileHandler currently installed on the root logger may target the live log.

    Guards against a regression where _configure_logging() (invoked via the
    FastAPI TestClient lifespan in other tests) installs a handler pointing at
    mrtrader_<date>.log during a pytest run.
    """
    live_name = f"mrtrader_{_today()}.log"
    for h in logging.getLogger().handlers:
        if isinstance(h, _DailyFileHandler):
            f = getattr(h, "_file", None)
            if f is not None and getattr(f, "name", None):
                assert Path(f.name).name != live_name, (
                    "A _DailyFileHandler is writing to the LIVE ops log under pytest"
                )
