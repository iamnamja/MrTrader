"""
Regression: the test_client fixture must not leak its patched `get_session` mocks.

Root cause (proven): test_client patches `app.database.session.get_session` and then
`app.analytics.{signal_attribution,drawdown_analyzer}.get_session` in one `with` block.
If an analytics module is first imported WHILE session.get_session is already patched,
its `from app.database.session import get_session` binds to the mock, which
unittest.mock captures as the "original" and restores to on exit — leaking the mock into
every subsequent test in the worker. conftest now pre-imports those modules so their
name binds to the real function before any patching. These two tests run in order: the
first exercises test_client, the second asserts no mock leaked.
"""
from __future__ import annotations

import unittest.mock as _m


def test_a_exercises_test_client(test_client):
    r = test_client.get("/api/config")
    assert r.status_code == 200


def test_b_get_session_not_leaked_as_mock():
    import app.analytics.signal_attribution as sa
    import app.analytics.drawdown_analyzer as da
    import app.database.session as sess
    assert not isinstance(sa.get_session, _m.NonCallableMock), \
        "signal_attribution.get_session leaked as a mock from a prior test_client test"
    assert not isinstance(da.get_session, _m.NonCallableMock), \
        "drawdown_analyzer.get_session leaked as a mock from a prior test_client test"
    assert not isinstance(sess.get_session, _m.NonCallableMock), \
        "app.database.session.get_session leaked as a mock"
