"""Runtime-environment helpers shared across the app.

Deliberately dependency-free (stdlib only) so it is import-safe from very early code
paths — e.g. logging configuration in app.main._DailyFileHandler, which runs before
most of the app is imported.
"""
import os
import sys


def is_test_mode() -> bool:
    """True iff this process is running under — or was spawned by — the pytest session.

    This is the single, authoritative test-mode detector. It exists because the app has
    several places that MUST behave differently under test (route logs to a separate
    file; never persist the kill switch or write audit rows; never spawn the live email
    drainer). Each used to roll its own check, and the weakest ones leaked test output
    into production resources.

    PRIMARY signal — ``MRTRADER_TEST_MODE``, force-set by tests/conftest.py at import.
    Unlike the runtime signals below, an env var is **inherited by spawned child
    processes**, so an app boot in a pytest-spawned subprocess (Windows ``spawn`` starts a
    fresh interpreter with no ``pytest`` in ``sys.modules`` and possibly no
    ``PYTEST_CURRENT_TEST``) is still correctly detected. It is the only signal that is
    correct on **both** sides of a process boundary — relying on in-memory/per-test state
    was the root cause of a test app-boot leaking into the live ops log.

    FALLBACK signals — belt-and-suspenders for any in-process pytest context where the env
    var is somehow unset: ``PYTEST_CURRENT_TEST`` is set by pytest for the duration of
    every test; ``"pytest" in sys.modules`` covers collection / fixture-setup before the
    first test starts.

    Production is unaffected: conftest never runs, so none of these signals are present
    and the function returns False.
    """
    if os.environ.get("MRTRADER_TEST_MODE") == "1":
        return True
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
