"""Reliability regression: KillSwitch.load_state() must never coerce a non-bool
persisted value to active.

Background: on 2026-06-10 the live log showed "Kill switch restored as ACTIVE from
persisted state" at startup, yet the kill switch had never been activated (config
value was a real bool False, no activation audit event ever). Root cause: a startup
path ran with a MOCKED config store whose get_config returned a MagicMock, and the old
`self._active = bool(val)` turned that truthy MagicMock into True. The same bug would
fire in production for any non-bool value (e.g. a legacy/corrupted string "false",
which `bool()` also evaluates True), spuriously HALTING live trading on startup.

The fix: strict isinstance(val, bool) — a genuine activation always persists a real
JSON bool, so a clean True is never lost; any non-bool is malformed and treated as
INACTIVE (refuse to halt on garbage).
"""
from unittest.mock import MagicMock, patch

import pytest

from app.live_trading.kill_switch import KillSwitch, _CFG_KS_ACTIVE


def _load_with_config(value):
    """Run KillSwitch.load_state() with get_config patched to return `value`."""
    ks = KillSwitch()
    db = MagicMock()
    with patch("app.live_trading.kill_switch.get_session", return_value=db), \
         patch("app.database.config_store.get_config", return_value=value) as gc:
        ret = ks.load_state()
    return ks, ret, gc


def test_magicmock_value_does_not_activate():
    """THE regression: a MagicMock from a mocked config store must NOT activate.

    Pre-fix `bool(MagicMock())` is True → false "restored as ACTIVE". Post-fix it is
    a non-bool → treated as inactive.
    """
    ks, ret, gc = _load_with_config(MagicMock())
    assert ks.is_active is False
    assert ret is True  # a value was present (load succeeded), just not a valid-True
    assert gc.call_count == 1 and gc.call_args[0][1] == _CFG_KS_ACTIVE


def test_string_false_does_not_activate():
    """A legacy/corrupted string 'false' must not become True via bool('false')."""
    ks, _, _ = _load_with_config("false")
    assert ks.is_active is False


def test_string_true_does_not_activate_either():
    """Even a truthy-looking string must not activate — only a real JSON bool does.
    A genuine activation persists bool True, never a string, so this loses nothing."""
    ks, _, _ = _load_with_config("true")
    assert ks.is_active is False


def test_genuine_bool_true_activates():
    """A real persisted bool True (the only thing activate() ever writes) activates."""
    ks, ret, _ = _load_with_config(True)
    assert ks.is_active is True
    assert ret is True


def test_genuine_bool_false_stays_inactive():
    ks, ret, _ = _load_with_config(False)
    assert ks.is_active is False
    assert ret is True


def test_missing_row_stays_inactive():
    """No persisted row (None) → default inactive, returns False (nothing restored)."""
    ks, ret, _ = _load_with_config(None)
    assert ks.is_active is False
    assert ret is False
