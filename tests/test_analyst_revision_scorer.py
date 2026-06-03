"""Unit tests for AnalystRevisionScorer (Alpha-v3 A1).

Mock the FMP grades feed; verify event-recency gating, PIT correctness, the
net-momentum confirmation filter, long/short, and the VIX crisis block.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from app.ml import analyst_revision_scorer as ars
from app.ml.analyst_revision_scorer import AnalystRevisionScorer

AS_OF = date(2024, 6, 14)


def _bars():
    idx = pd.date_range(end=pd.Timestamp(AS_OF), periods=30, freq="B")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1e6}, index=idx)


def _sd(extra_vix=None):
    sd = {"AAPL": _bars()}
    if extra_vix is not None:
        vidx = pd.date_range(end=pd.Timestamp(AS_OF), periods=5, freq="B")
        sd["^VIX"] = pd.DataFrame({"close": extra_vix}, index=vidx)
    return sd


def _grade(days_before, action, company="X"):
    return {"date": (AS_OF - timedelta(days=days_before)).isoformat(),
            "action": action, "gradingCompany": company,
            "previousGrade": "Hold", "newGrade": "Buy"}


def _patch(monkeypatch, grades):
    import app.data.fmp_provider as fp
    monkeypatch.setattr(fp, "get_analyst_grades_fmp", lambda sym: grades)


def test_long_on_recent_upgrade_with_confirmation(monkeypatch):
    _patch(monkeypatch, [_grade(3, "upgrade"), _grade(20, "upgrade")])
    out = AnalystRevisionScorer(max_days_after=5, min_net_momentum=1.0)(AS_OF, _sd())
    assert out == [("AAPL", pytest.approx((2 + 1) / 4), "long")]  # net=2 -> conf 0.75


def test_no_signal_when_event_too_old(monkeypatch):
    _patch(monkeypatch, [_grade(10, "upgrade")])
    assert AnalystRevisionScorer(max_days_after=5)(AS_OF, _sd()) == []


def test_no_signal_on_announcement_day(monkeypatch):
    _patch(monkeypatch, [_grade(0, "upgrade")])  # same-day jump, not drift
    assert AnalystRevisionScorer(max_days_after=5)(AS_OF, _sd()) == []


def test_pit_excludes_future_grade(monkeypatch):
    # an upgrade dated AFTER as_of must not be seen
    future = {"date": (AS_OF + timedelta(days=2)).isoformat(), "action": "upgrade"}
    _patch(monkeypatch, [future, _grade(20, "downgrade")])
    # most recent knowable event is the 20d-old downgrade -> too old, no long leak
    assert AnalystRevisionScorer(max_days_after=5)(AS_OF, _sd()) == []


def test_net_momentum_filters_contrarian_upgrade(monkeypatch):
    # a lone fresh upgrade against a stream of downgrades -> net negative -> no long
    grades = [_grade(3, "upgrade")] + [_grade(d, "downgrade") for d in (8, 12, 18)]
    _patch(monkeypatch, grades)
    assert AnalystRevisionScorer(max_days_after=5, min_net_momentum=1.0)(AS_OF, _sd()) == []


def test_short_leg_only_when_enabled(monkeypatch):
    _patch(monkeypatch, [_grade(2, "downgrade"), _grade(15, "downgrade")])
    long_only = AnalystRevisionScorer(max_days_after=5, long_short=False)(AS_OF, _sd())
    assert long_only == []  # downgrade ignored when long-only
    ls = AnalystRevisionScorer(max_days_after=5, long_short=True)(AS_OF, _sd())
    assert ls == [("AAPL", pytest.approx(-(2 + 1) / 4), "short")]  # net=-2


def test_vix_crisis_blocks_all(monkeypatch):
    _patch(monkeypatch, [_grade(2, "upgrade"), _grade(10, "upgrade")])
    out = AnalystRevisionScorer(max_days_after=5, vix_block_all=30.0)(AS_OF, _sd(extra_vix=45.0))
    assert out == []


def test_maintain_only_history_no_signal(monkeypatch):
    _patch(monkeypatch, [_grade(2, "maintain"), _grade(5, "maintain")])
    assert AnalystRevisionScorer(max_days_after=5)(AS_OF, _sd()) == []


def test_skips_synthetic_symbols(monkeypatch):
    _patch(monkeypatch, [_grade(2, "upgrade"), _grade(10, "upgrade")])
    sd = {"SPY": _bars(), "^VIX": _bars()}
    assert AnalystRevisionScorer(max_days_after=5)(AS_OF, sd) == []
