"""Tests for the shared test-mode detector (app.utils.runtime.is_test_mode).

This is the single authoritative check that gates test-vs-production behaviour (log
routing, kill-switch persistence, the email-drainer subprocess). The key property —
the one whose absence let a test app-boot leak into the live ops log — is that it is
correct ACROSS a process boundary: an env var is inherited by spawned children, the
runtime signals (sys.modules / PYTEST_CURRENT_TEST) are not.
"""
import sys

from app.utils.runtime import is_test_mode


def test_true_under_normal_pytest():
    """In-process pytest: detected (env var set by conftest, and/or runtime signals)."""
    assert is_test_mode() is True


def test_env_var_alone_detects_subprocess(monkeypatch):
    """The fix: simulate a spawned subprocess — no `pytest` in sys.modules, no
    PYTEST_CURRENT_TEST — but the inherited MRTRADER_TEST_MODE env var must still flag it.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.setenv("MRTRADER_TEST_MODE", "1")
    assert is_test_mode() is True


def test_runtime_signal_alone_detects_inprocess(monkeypatch):
    """Belt-and-suspenders: even with the env var unset, an in-process pytest run is
    still caught by the runtime fallback (`pytest` in sys.modules)."""
    monkeypatch.delenv("MRTRADER_TEST_MODE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # `pytest` remains in sys.modules here (we are mid-test) → fallback catches it.
    assert is_test_mode() is True


def test_false_when_no_signal_at_all(monkeypatch):
    """A genuine production process: no env var, no runtime signals → not test mode."""
    monkeypatch.delenv("MRTRADER_TEST_MODE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    assert is_test_mode() is False


def test_kill_switch_guard_uses_shared_detector(monkeypatch):
    """The kill-switch persist/audit guard must follow the shared detector — including the
    subprocess case (no runtime signals, env var only). Previously it keyed on
    os.environ['_'] and would have returned False here, persisting to the real DB."""
    from app.live_trading.kill_switch import _running_under_pytest
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.setenv("MRTRADER_TEST_MODE", "1")
    assert _running_under_pytest() is True
