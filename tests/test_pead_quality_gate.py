"""
Tests for the earnings-quality-split lever on PEADScorer.

The quality gate (require_positive_revision=True) requires a long signal to be
not just an EPS beat but a beat WITH positive analyst-revision momentum as-of
the scoring day. This filters EPS beats down to "beat + analysts revising up" =
higher-conviction drift.

Covers:
1. test_quality_gate_off_unchanged  — gate OFF → output identical to baseline
   (regression lock on the committed +0.546 long-only config).
2. test_quality_gate_filters_negative_revision — EPS beat + NEGATIVE revision →
   gate ON suppresses the long; gate OFF still fires it.
3. test_quality_gate_pit_safe — the analyst lookup is called with as_of == the
   scoring day (data <= day only), never future.

All stubbed/monkeypatched — no network.
"""
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scorer(**kwargs):
    from app.ml.pead_scorer import PEADScorer
    return PEADScorer(**kwargs)


def _dummy_df():
    idx = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame({"close": np.linspace(100, 110, 10),
                         "volume": [1e6] * 10}, index=idx)


def _run(scorer, feats_by_sym: dict, analyst_by_sym: dict | None = None, day=None):
    """Run scorer with earnings (and optionally analyst) features mocked per-symbol."""
    if day is None:
        day = date(2024, 3, 15)
    symbols_data = {sym: _dummy_df() for sym in feats_by_sym}

    def _fake_earnings(sym, as_of):
        return feats_by_sym.get(sym)

    def _fake_analyst(sym, as_of):
        analyst = (analyst_by_sym or {}).get(sym, {})
        # Mirror the real return shape from get_analyst_features_at.
        return {
            "fmp_analyst_upgrades_30d": float(analyst.get("upgrades", 0.0)),
            "fmp_analyst_downgrades_30d": float(analyst.get("downgrades", 0.0)),
            "fmp_analyst_momentum_30d": float(analyst.get("momentum", 0.0)),
        }

    with patch("app.data.fmp_provider.get_earnings_features_at", side_effect=_fake_earnings), \
         patch("app.data.fmp_provider.get_analyst_features_at", side_effect=_fake_analyst):
        return scorer(day, symbols_data)


# ── Test 1: gate OFF leaves committed config unchanged ──────────────────────────

class TestQualityGateOff:
    def test_quality_gate_off_unchanged(self):
        """With require_positive_revision=False, output matches the baseline scorer
        regardless of analyst data (regression lock on +0.546 config)."""
        feats = {
            "AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1},
            "NVDA": {"fmp_surprise_1q": 0.30, "fmp_days_since_earnings": 2},
            "TSLA": {"fmp_surprise_1q": 0.06, "fmp_days_since_earnings": 1},
        }
        # Strongly negative revisions everywhere — would be filtered if gate ON.
        analyst = {s: {"momentum": -3.0} for s in feats}

        baseline = _make_scorer(require_positive_revision=False)
        result_off = _run(baseline, feats, analyst)

        # Reference: same scorer, no analyst mock at all (pure baseline path).
        ref = _make_scorer()
        with patch("app.data.fmp_provider.get_earnings_features_at",
                   side_effect=lambda sym, as_of: feats.get(sym)):
            result_ref = ref(date(2024, 3, 15), {s: _dummy_df() for s in feats})

        assert result_off == result_ref
        assert {s for s, _, _ in result_off} == {"AAPL", "NVDA", "TSLA"}

    def test_quality_gate_off_does_not_call_analyst(self):
        """Gate OFF must not even import/call the analyst provider."""
        feats = {"AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}}
        scorer = _make_scorer(require_positive_revision=False)
        with patch("app.data.fmp_provider.get_earnings_features_at",
                   side_effect=lambda sym, as_of: feats.get(sym)), \
             patch("app.data.fmp_provider.get_analyst_features_at") as m_analyst:
            scorer(date(2024, 3, 15), {"AAPL": _dummy_df()})
        m_analyst.assert_not_called()


# ── Test 2: gate filters out negative-revision beats ────────────────────────────

class TestQualityGateFilters:
    def test_quality_gate_filters_negative_revision(self):
        """EPS beat + NEGATIVE analyst revision → no long with gate ON, long with gate OFF."""
        feats = {"AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}}
        analyst = {"AAPL": {"upgrades": 0, "downgrades": 2, "momentum": -2.0}}

        scorer_on = _make_scorer(require_positive_revision=True)
        result_on = _run(scorer_on, feats, analyst)
        assert result_on == [], "Gate ON must suppress a beat with negative revision"

        scorer_off = _make_scorer(require_positive_revision=False)
        result_off = _run(scorer_off, feats, analyst)
        assert len(result_off) == 1
        assert result_off[0][2] == "long", "Gate OFF must still fire the long"

    def test_quality_gate_passes_positive_revision(self):
        """EPS beat + POSITIVE revision → long fires even with gate ON."""
        feats = {"AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}}
        analyst = {"AAPL": {"upgrades": 3, "downgrades": 0, "momentum": 3.0}}
        scorer_on = _make_scorer(require_positive_revision=True)
        result_on = _run(scorer_on, feats, analyst)
        assert len(result_on) == 1
        assert result_on[0][2] == "long"

    def test_quality_gate_zero_momentum_filtered(self):
        """Zero momentum is NOT strictly positive (> min_analyst_momentum=0) → filtered."""
        feats = {"AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}}
        analyst = {"AAPL": {"momentum": 0.0}}
        scorer_on = _make_scorer(require_positive_revision=True)
        result_on = _run(scorer_on, feats, analyst)
        assert result_on == [], "Flat revision (momentum=0) must be filtered out"

    def test_quality_gate_custom_min_momentum(self):
        """min_analyst_momentum=2 requires momentum strictly > 2."""
        feats = {
            "AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1},
            "NVDA": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1},
        }
        analyst = {"AAPL": {"momentum": 2.0}, "NVDA": {"momentum": 3.0}}
        scorer = _make_scorer(require_positive_revision=True, min_analyst_momentum=2.0)
        result = _run(scorer, feats, analyst)
        syms = {s for s, _, _ in result}
        assert syms == {"NVDA"}, "Only momentum strictly > 2 survives"

    def test_quality_gate_does_not_affect_short_leg(self):
        """Quality gate applies to longs only; short leg is unaffected."""
        feats = {"TSLA": {"fmp_surprise_1q": -0.10, "fmp_days_since_earnings": 1}}
        analyst = {"TSLA": {"momentum": -3.0}}
        scorer = _make_scorer(require_positive_revision=True, long_short=True)
        result = _run(scorer, feats, analyst)
        assert len(result) == 1
        assert result[0][2] == "short", "Short leg must not be gated by the revision filter"


# ── Test 3: PIT safety ──────────────────────────────────────────────────────────

class TestQualityGatePITSafe:
    def test_quality_gate_pit_safe(self):
        """The analyst lookup must be called with as_of == the scoring day (data <= day)."""
        feats = {"AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}}
        scoring_day = date(2024, 3, 15)
        seen_as_of = {}

        def _fake_analyst(sym, as_of):
            seen_as_of[sym] = as_of
            return {"fmp_analyst_momentum_30d": 3.0}

        scorer = _make_scorer(require_positive_revision=True)
        with patch("app.data.fmp_provider.get_earnings_features_at",
                   side_effect=lambda sym, as_of: feats.get(sym)), \
             patch("app.data.fmp_provider.get_analyst_features_at",
                   side_effect=_fake_analyst):
            scorer(scoring_day, {"AAPL": _dummy_df()})

        assert "AAPL" in seen_as_of, "analyst lookup must be performed when gate ON"
        as_of = seen_as_of["AAPL"]
        # as_of must equal the scoring day — never a future date.
        assert as_of == scoring_day, f"PIT violation: as_of={as_of} != scoring day {scoring_day}"
        assert as_of <= scoring_day

    def test_quality_gate_as_of_is_date_type(self):
        """as_of passed to the analyst provider is a datetime.date (matches earnings path)."""
        feats = {"AAPL": {"fmp_surprise_1q": 0.10, "fmp_days_since_earnings": 1}}
        captured = {}

        def _fake_analyst(sym, as_of):
            captured["as_of"] = as_of
            return {"fmp_analyst_momentum_30d": 1.0}

        scorer = _make_scorer(require_positive_revision=True)
        with patch("app.data.fmp_provider.get_earnings_features_at",
                   side_effect=lambda sym, as_of: feats.get(sym)), \
             patch("app.data.fmp_provider.get_analyst_features_at",
                   side_effect=_fake_analyst):
            scorer(pd.Timestamp("2024-03-15"), {"AAPL": _dummy_df()})

        assert isinstance(captured["as_of"], date)
        assert captured["as_of"] == date(2024, 3, 15)
