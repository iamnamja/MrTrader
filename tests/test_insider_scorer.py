"""
Tests for the InsiderClusterScorer (candidate second edge).

Covers:
  - PIT-safety: filings with filing_date > as_of must NOT influence the signal.
  - Cluster logic: >=2 distinct buyers OR a single large buy fires; otherwise not.
  - Long-only: scorer never emits a "short" direction.
  - Freshness: stale clusters (beyond max_days_after_filing) do not fire.

All tests monkeypatch the FMP fetcher so they are hermetic (no network).
"""
from datetime import date

import pandas as pd
import pytest

import app.data.fmp_provider as fmp
from app.ml.insider_scorer import InsiderClusterScorer


def _trade(filing_date, name, shares=1000.0, price=10.0, txn_date=None):
    return {
        "filing_date": filing_date,
        "transaction_date": txn_date or filing_date,
        "reporting_name": name.upper(),
        "type_of_owner": "director",
        "shares": shares,
        "price": price,
        "notional": shares * price,
    }


@pytest.fixture(autouse=True)
def _clear_insider_cache():
    fmp._insider_cache.clear()
    yield
    fmp._insider_cache.clear()


def _patch(monkeypatch, records):
    monkeypatch.setattr(fmp, "get_insider_trades_fmp", lambda sym: records)


# ── get_insider_features_at: cluster logic & PIT ──────────────────────────────

def test_two_distinct_buyers_is_cluster(monkeypatch):
    recs = [
        _trade("2024-01-10", "ALICE"),
        _trade("2024-01-15", "BOB"),
    ]
    _patch(monkeypatch, recs)
    f = fmp.get_insider_features_at("XYZ", date(2024, 1, 16))
    assert f["insider_distinct_buyers"] == 2.0
    assert f["insider_is_cluster"] == 1.0


def test_single_small_buyer_not_cluster(monkeypatch):
    recs = [_trade("2024-01-10", "ALICE", shares=100.0, price=5.0)]  # $500
    _patch(monkeypatch, recs)
    f = fmp.get_insider_features_at("XYZ", date(2024, 1, 16))
    assert f["insider_distinct_buyers"] == 1.0
    assert f["insider_is_cluster"] == 0.0


def test_single_large_buy_is_cluster(monkeypatch):
    # One buyer but >= $1M notional → fires on the size criterion.
    recs = [_trade("2024-01-10", "ALICE", shares=100_000.0, price=20.0)]  # $2M
    _patch(monkeypatch, recs)
    f = fmp.get_insider_features_at("XYZ", date(2024, 1, 16))
    assert f["insider_max_notional"] >= fmp.INSIDER_LARGE_BUY_NOTIONAL
    assert f["insider_is_cluster"] == 1.0


def test_pit_excludes_future_filings(monkeypatch):
    # Two buyers but BOB's filing is AFTER as_of → must be invisible (PIT).
    recs = [
        _trade("2024-01-10", "ALICE"),
        _trade("2024-01-20", "BOB"),  # future relative to as_of
    ]
    _patch(monkeypatch, recs)
    f = fmp.get_insider_features_at("XYZ", date(2024, 1, 15))
    assert f["insider_distinct_buyers"] == 1.0
    assert f["insider_is_cluster"] == 0.0


def test_window_excludes_old_filings(monkeypatch):
    # ALICE bought long ago (outside 30d window); only BOB is recent → no cluster.
    recs = [
        _trade("2023-11-01", "ALICE"),
        _trade("2024-01-15", "BOB"),
    ]
    _patch(monkeypatch, recs)
    f = fmp.get_insider_features_at("XYZ", date(2024, 1, 16))
    assert f["insider_distinct_buyers"] == 1.0
    assert f["insider_is_cluster"] == 0.0


def test_no_purchase_history_returns_none(monkeypatch):
    _patch(monkeypatch, [])
    assert fmp.get_insider_features_at("XYZ", date(2024, 1, 16)) is None


# ── Scorer: long-only, freshness, PIT propagation ─────────────────────────────

def _df(start, end):
    idx = pd.date_range(start, end)
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx
    )


def test_scorer_emits_long_on_cluster(monkeypatch):
    recs = [_trade("2024-01-10", "ALICE"), _trade("2024-01-12", "BOB")]
    _patch(monkeypatch, recs)
    sc = InsiderClusterScorer()
    out = sc(pd.Timestamp("2024-01-13"), {"XYZ": _df("2023-12-01", "2024-02-01")}, None)
    assert len(out) == 1
    sym, conf, direction = out[0]
    assert sym == "XYZ"
    assert direction == "long"
    assert conf > 0


def test_scorer_long_only_never_short(monkeypatch):
    recs = [_trade("2024-01-10", "ALICE"), _trade("2024-01-12", "BOB")]
    _patch(monkeypatch, recs)
    sc = InsiderClusterScorer()
    out = sc(pd.Timestamp("2024-01-13"), {"XYZ": _df("2023-12-01", "2024-02-01")}, None)
    assert all(d == "long" for _, _, d in out)


def test_scorer_no_signal_without_cluster(monkeypatch):
    recs = [_trade("2024-01-10", "ALICE", shares=100.0, price=1.0)]  # tiny single buy
    _patch(monkeypatch, recs)
    sc = InsiderClusterScorer()
    out = sc(pd.Timestamp("2024-01-13"), {"XYZ": _df("2023-12-01", "2024-02-01")}, None)
    assert out == []


def test_scorer_freshness_stale_cluster_skipped(monkeypatch):
    # Cluster exists but latest filing is > max_days_after_filing before as_of.
    recs = [_trade("2024-01-01", "ALICE"), _trade("2024-01-02", "BOB")]
    _patch(monkeypatch, recs)
    sc = InsiderClusterScorer(max_days_after_filing=5)
    # as_of is 20 days after the last filing → too stale.
    out = sc(pd.Timestamp("2024-01-22"), {"XYZ": _df("2023-12-01", "2024-02-01")}, None)
    assert out == []


def test_scorer_pit_future_filing_no_signal(monkeypatch):
    # BOB files in the future; only ALICE visible → no cluster → no signal.
    recs = [_trade("2024-01-10", "ALICE"), _trade("2024-01-20", "BOB")]
    _patch(monkeypatch, recs)
    sc = InsiderClusterScorer()
    out = sc(pd.Timestamp("2024-01-12"), {"XYZ": _df("2023-12-01", "2024-02-01")}, None)
    assert out == []


def test_scorer_vix_crisis_blocks(monkeypatch):
    recs = [_trade("2024-01-10", "ALICE"), _trade("2024-01-12", "BOB")]
    _patch(monkeypatch, recs)
    sc = InsiderClusterScorer(vix_block_all=30.0)
    vix = pd.DataFrame({"close": 40.0}, index=pd.date_range("2024-01-01", "2024-01-20"))
    out = sc(pd.Timestamp("2024-01-13"),
             {"XYZ": _df("2023-12-01", "2024-02-01"), "^VIX": vix}, None)
    assert out == []
