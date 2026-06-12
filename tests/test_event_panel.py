"""
Tests for app/research/event_panel.py — the earnings-event panel (PR3).

Coverage:
  - schema: frozen column set/order, one row per (symbol, announce_date),
    UNK announce_ts_flag, all-NaN reserved option columns;
  - PIT reproducibility: validate_panel_pit passes on fixture inputs AND
    compute_event_features is invariant to truncation at announce_date
    (the hard no-look-ahead proof);
  - qualification PARITY: the panel's pead_qualified set == the signal set of
    a real PEADScorer oracle (the COMMITTED +0.546 config) evaluated on
    identical fixture inputs at each event's entry day;
  - forward-window/quality flags (QF_INCOMPLETE_FWD20, QF_NO_SECTOR_HEDGE);
  - suspect-bars threading (QF_SUSPECT_BARS): a persistent split-cliff the
    refetch cannot heal marks the symbol suspect, ALL its events carry the
    flag and are excluded from the H1 population; healed symbols are not;
  - entry/forward-return arithmetic against hand-computed values;
  - sue_z_pit expanding strictly-prior standardization;
  - sacred-holdout exclusion (window-crossing events dropped; an end date at
    or past the holdout refuses to build);
  - membership_fn exclusion (PIT R1K discipline).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.research.event_panel import (
    FWD_HORIZONS,
    OPTION_COLUMNS,
    QF_EXCLUDE_FROM_INFERENCE,
    QF_INCOMPLETE_FWD20,
    QF_NO_SECTOR_HEDGE,
    QF_NO_VOL20,
    QF_SUSPECT_BARS,
    SECTOR_HEDGE_HORIZONS,
    assemble_panel,
    compute_event_features,
    parse_earnings_rows,
    qualify_event,
    validate_panel_pit,
)

# ── deterministic fixture market ───────────────────────────────────────────────

IDX = pd.bdate_range("2023-01-02", "2024-12-31")


def _make_bars(seed: int, base: float = 100.0, index=IDX) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, len(index))
    close = base * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, len(index)))
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * 1.005,
        "low": np.minimum(open_, close) * 0.995,
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, len(index)).astype(float),
    }, index=index)


def _vix_df() -> pd.DataFrame:
    # Calm 18 everywhere, crisis 35 during late-Jul/early-Aug 2024.
    close = pd.Series(18.0, index=IDX)
    close.loc["2024-07-25":"2024-08-05"] = 35.0
    return pd.DataFrame({"close": close})


def _rec(d: str, sue) -> dict:
    return {"date": d, "epsActual": 1.0, "epsEstimated": 0.9, "surprise_pct": sue}


EARNINGS = {
    # AAA: qualified / sue-below / vix-blocked
    "AAA": [_rec("2024-08-01", 0.20), _rec("2024-05-01", 0.10),
            _rec("2024-02-02", 0.04)],
    # BBB: qualified / incomplete-forward-window
    "BBB": [_rec("2024-12-20", 0.30), _rec("2024-05-01", 0.06)],
    # CCC: long-only reject / late-first-session / qualified
    "CCC": [_rec("2024-08-20", 0.07), _rec("2024-06-03", 0.10),
            _rec("2024-05-02", -0.10)],
}
SECTORS = {"AAA": "Technology", "BBB": "Energy", "CCC": "UNKNOWN"}


@pytest.fixture(scope="module")
def market():
    bars = {
        "AAA": _make_bars(1),
        "BBB": _make_bars(2, base=50.0),
        # CCC is missing 2024-06-04..06-06 -> the 2024-06-03 event's first
        # session is 4 calendar days out (the late-first-session case).
        "CCC": _make_bars(3, base=20.0).drop(
            index=pd.bdate_range("2024-06-04", "2024-06-06")),
    }
    spy = _make_bars(4, base=400.0)
    etfs = {"XLK": _make_bars(5, base=150.0), "XLE": _make_bars(6, base=80.0)}
    return {
        "bars": bars, "earnings": EARNINGS, "spy_bars": spy,
        "vix_close": _vix_df()["close"], "sector_etf_bars": etfs,
        "sectors": SECTORS,
    }


@pytest.fixture(scope="module")
def panel(market):
    return assemble_panel(
        bars=market["bars"], earnings=market["earnings"],
        spy_bars=market["spy_bars"], vix_close=market["vix_close"],
        sector_etf_bars=market["sector_etf_bars"], sectors=market["sectors"],
        start=date(2024, 1, 1), end=date(2024, 12, 31),
        holdout_start=date(2026, 11, 9),
    )


# ─────────────────────────────────────────────────────────────────── schema

def test_schema_columns_and_uniqueness(panel):
    expected = [
        "event_id", "symbol", "announce_date", "announce_ts_flag", "sector",
        "mktcap_decile", "sue", "sue_z_pit", "revision_momentum",
        "announce_gap_pct", "gap_vs_vol20", "prior_qtr_drift", "pead_score_v1",
        "vix_at_signal", "spy_below_200d", "pead_qualified", "qual_reason",
        *OPTION_COLUMNS,
        *[f"fwd_ret_{h}_raw" for h in FWD_HORIZONS],
        *[f"fwd_ret_{h}_spyhedged" for h in FWD_HORIZONS],
        *[f"fwd_ret_{h}_sectorhedged" for h in SECTOR_HEDGE_HORIZONS],
        "beta_60d", "dollar_vol20", "entry_open_next", "entry_open_next2",
        "cal_days_to_entry", "options_coverage_flag", "quality_flags",
    ]
    assert list(panel.columns) == expected
    assert len(panel) == 8  # every fixture event lands exactly once
    assert not panel.duplicated(subset=["symbol", "announce_date"]).any()
    assert (panel["announce_ts_flag"] == "UNK").all()          # decision 4
    assert (~panel["options_coverage_flag"]).all()
    for col in OPTION_COLUMNS:                                  # reserved for P2
        assert panel[col].isna().all()
    assert (panel["event_id"] ==
            panel["symbol"] + "|" + panel["announce_date"].astype(str)).all()


# ──────────────────────────────────────────────── PIT: no look-ahead, ever

def test_validate_panel_pit_passes(panel, market):
    res = validate_panel_pit(
        panel, bars=market["bars"], spy_bars=market["spy_bars"],
        vix_close=market["vix_close"], earnings=market["earnings"],
        sample_n=len(panel),
    )
    assert res["ok"], res["mismatches"]
    assert res["checked"] == len(panel)


def test_features_invariant_to_future_truncation(market):
    """The hard anti-leak proof: feature values computed from FULL history
    equal those computed from history truncated at announce_date."""
    ann = pd.Timestamp("2024-05-01")
    bars = market["bars"]["AAA"]
    spy = market["spy_bars"]["close"]
    vix = market["vix_close"]
    full = compute_event_features(bars, spy, vix, ann)
    trunc = compute_event_features(
        bars.loc[:ann], spy.loc[:ann], vix.loc[:ann], ann)
    for k in ("announce_gap_pct", "gap_vs_vol20", "prior_qtr_drift",
              "beta_60d", "dollar_vol20", "vix_at_signal"):
        assert full[k] == pytest.approx(trunc[k], rel=1e-12), k
    assert full["spy_below_200d"] == trunc["spy_below_200d"]


def test_features_empty_pre_window_no_crash(market):
    """Regression: a name whose history starts AT/AFTER the announce date has an
    empty pre-announce slice. pandas pct_change() raises on an empty Series
    ("argmax of an empty sequence"), which crashed the full R1K build. Must
    return all-NaN features + the relevant quality bits, never raise."""
    ann = pd.Timestamp("2024-05-01")
    bars = market["bars"]["AAA"]
    spy = market["spy_bars"]["close"]
    vix = market["vix_close"]
    # Only bars ON/AFTER the announce -> pre window is empty.
    bars_after = bars[bars.index >= ann]
    feats = compute_event_features(bars_after, spy, vix, ann)  # must not raise
    assert np.isnan(feats["vol20"])
    assert np.isnan(feats["prior_qtr_drift"])
    assert np.isnan(feats["beta_60d"])
    assert feats["quality_bits"] & QF_NO_VOL20
    # Fully-empty bars likewise must not raise.
    empty = bars.iloc[0:0]
    feats2 = compute_event_features(empty, spy, vix, ann)
    assert np.isnan(feats2["vol20"]) and np.isnan(feats2["beta_60d"])


# ─────────────────────────────── qualification PARITY vs the live PEADScorer

def _oracle_earnings_features(records_by_sym):
    """Fixture stand-in for fmp_provider.get_earnings_features_at, replicating
    its PIT filter on the SAME records the panel was built from."""
    def fake(symbol: str, as_of: date):
        recs = [
            r for r in records_by_sym.get(symbol, [])
            if r["surprise_pct"] is not None
            and date.fromisoformat(r["date"]) <= as_of
        ]
        if not recs:
            return None
        recs.sort(key=lambda r: r["date"], reverse=True)
        last = date.fromisoformat(recs[0]["date"])
        return {
            "fmp_surprise_1q": float(recs[0]["surprise_pct"]),
            "fmp_surprise_2q_avg": float(
                sum(r["surprise_pct"] for r in recs[:2]) / min(len(recs), 2)),
            "fmp_days_since_earnings": float((as_of - last).days),
        }
    return fake


def test_qualification_parity_with_pead_scorer_oracle(panel, market, monkeypatch):
    """Panel pead_qualified set == the signal set of the COMMITTED +0.546
    PEADScorer (long-only, sue>=0.05, days 1..3, VIX>30 block) evaluated at
    each event's entry day on PIT-truncated fixture data."""
    from app.ml.pead_scorer import PEADScorer

    monkeypatch.setattr(
        "app.data.fmp_provider.get_earnings_features_at",
        _oracle_earnings_features(market["earnings"]),
    )
    # The committed config — byte-identical to run_pead_cpcv.build_pead_scorer
    # defaults (long-only path).
    scorer = PEADScorer(
        long_threshold=0.05, short_threshold=-0.05, long_short=False,
        vix_block_all=30.0, vix_block_short=100.0, vix_conf_ref=100.0,
        max_announce_day_move=1.0, require_positive_revision=False,
        min_analyst_momentum=0.0,
    )
    vix_df = pd.DataFrame({"close": market["vix_close"]})

    panel_qualified = set(panel.loc[panel["pead_qualified"], "event_id"])
    oracle_qualified = set()
    for _, row in panel.iterrows():
        sym = row["symbol"]
        ann = pd.Timestamp(row["announce_date"])
        bars = market["bars"][sym]
        fwd = bars.loc[bars.index > ann]
        if len(fwd) == 0:
            continue
        entry_day = fwd.index[0]
        # PIT-at-entry data: bars/VIX knowable when the announce+1 open trades
        # (i.e. through the announce-day close).
        symbols_data = {
            sym: bars.loc[:ann],
            "SPY": market["spy_bars"].loc[:ann],
            "^VIX": vix_df.loc[:ann],
        }
        signals = scorer(entry_day, symbols_data)
        if any(s[0] == sym and s[2] == "long" for s in signals):
            oracle_qualified.add(row["event_id"])

    assert panel_qualified == oracle_qualified
    # The fixture exercises every rejection path — make sure the panel agrees.
    reasons = dict(zip(panel["event_id"], panel["qual_reason"]))
    assert reasons["AAA|2024-02-02"] == "sue_below_long_threshold"
    assert reasons["AAA|2024-08-01"] == "vix_block"
    assert reasons["CCC|2024-05-02"] == "sue_below_long_threshold"
    assert reasons["CCC|2024-06-03"] == "first_session_gt_max_cal_days"
    assert reasons["AAA|2024-05-01"] == "ok"
    assert reasons["BBB|2024-05-01"] == "ok"
    assert reasons["CCC|2024-08-20"] == "ok"
    assert reasons["BBB|2024-12-20"] == "ok"  # qualified; flagged incomplete-fwd


# ───────────────────────────────────────── forward returns + quality flags

def test_entry_and_forward_return_arithmetic(panel, market):
    row = panel.set_index("event_id").loc["AAA|2024-05-01"]
    bars = market["bars"]["AAA"]
    spy = market["spy_bars"]
    fwd = bars.loc[bars.index > pd.Timestamp("2024-05-01")]
    s1 = fwd.index[0]
    assert row["entry_open_next"] == pytest.approx(float(fwd["open"].iloc[0]))
    assert row["entry_open_next2"] == pytest.approx(float(fwd["open"].iloc[1]))
    assert row["cal_days_to_entry"] == 1
    raw10 = float(fwd["close"].iloc[9]) / float(fwd["open"].iloc[0]) - 1.0
    assert row["fwd_ret_10_raw"] == pytest.approx(raw10, rel=1e-12)
    spy_leg = float(spy.at[fwd.index[9], "close"]) / float(spy.at[s1, "open"]) - 1.0
    assert row["fwd_ret_10_spyhedged"] == pytest.approx(raw10 - spy_leg, rel=1e-12)
    etf = market["sector_etf_bars"]["XLK"]
    etf_leg = float(etf.at[fwd.index[9], "close"]) / float(etf.at[s1, "open"]) - 1.0
    assert row["fwd_ret_10_sectorhedged"] == pytest.approx(raw10 - etf_leg, rel=1e-12)
    assert row["quality_flags"] == 0


def test_incomplete_forward_window_flagged(panel):
    row = panel.set_index("event_id").loc["BBB|2024-12-20"]
    assert np.isnan(row["fwd_ret_20_raw"])
    assert row["quality_flags"] & QF_INCOMPLETE_FWD20
    # Earlier horizons that DO fit are still populated.
    assert np.isfinite(row["fwd_ret_5_raw"])


def test_unknown_sector_flags_no_sector_hedge(panel):
    row = panel.set_index("event_id").loc["CCC|2024-08-20"]
    assert row["quality_flags"] & QF_NO_SECTOR_HEDGE
    assert np.isnan(row["fwd_ret_10_sectorhedged"])
    assert np.isfinite(row["fwd_ret_10_spyhedged"])  # SPY hedge unaffected


# ──────────────────────────────────────────────────────────────── sue_z_pit

def test_sue_z_pit_expanding_strictly_prior(panel):
    p = panel.sort_values(["announce_date", "symbol"]).reset_index(drop=True)
    # First announce date (2024-02-02): pool empty -> z = 0.
    assert p.loc[0, "sue_z_pit"] == 0.0
    # A later event standardizes against ALL strictly-earlier sues.
    later = p[p["event_id"] == "CCC|2024-08-20"].iloc[0]
    pool = p.loc[p["announce_date"] < date(2024, 8, 20), "sue"].to_numpy()
    expect = (later["sue"] - pool.mean()) / pool.std(ddof=1)
    assert later["sue_z_pit"] == pytest.approx(expect, rel=1e-9)
    assert (panel["pead_score_v1"] == panel["sue_z_pit"]).all()


# ─────────────────────────────────────────────────────────── sacred holdout

def test_sacred_holdout_drops_crossing_events(market):
    # Pretend the holdout starts 2024-07-01: June events' 20d forward windows
    # cross it -> dropped; early-May events complete before it -> kept.
    p = assemble_panel(
        bars=market["bars"], earnings=market["earnings"],
        spy_bars=market["spy_bars"], vix_close=market["vix_close"],
        sector_etf_bars=market["sector_etf_bars"], sectors=market["sectors"],
        start=date(2024, 1, 1), end=date(2024, 6, 30),
        holdout_start=date(2024, 7, 1),
    )
    assert "CCC|2024-06-03" not in set(p["event_id"])      # window crosses
    assert "AAA|2024-05-01" in set(p["event_id"])          # completes before
    assert (pd.to_datetime(p["announce_date"]).dt.date < date(2024, 7, 1)).all()


def test_build_refuses_to_reach_holdout(market):
    with pytest.raises(AssertionError, match="sacred holdout"):
        assemble_panel(
            bars=market["bars"], earnings=market["earnings"],
            spy_bars=market["spy_bars"], vix_close=market["vix_close"],
            sector_etf_bars=market["sector_etf_bars"], sectors=market["sectors"],
            start=date(2024, 1, 1), end=date(2024, 12, 31),
            holdout_start=date(2024, 7, 1),
        )


# ─────────────────────────────────────────────────────── membership (PIT R1K)

def test_membership_fn_excludes_non_members(market):
    p = assemble_panel(
        bars=market["bars"], earnings=market["earnings"],
        spy_bars=market["spy_bars"], vix_close=market["vix_close"],
        sector_etf_bars=market["sector_etf_bars"], sectors=market["sectors"],
        start=date(2024, 1, 1), end=date(2024, 12, 31),
        membership_fn=lambda sym, d: not (sym == "BBB" and d < date(2024, 6, 1)),
        holdout_start=date(2026, 11, 9),
    )
    assert "BBB|2024-05-01" not in set(p["event_id"])
    assert "BBB|2024-12-20" in set(p["event_id"])


# ─────────────────────────────────────────────── unit helpers (pure functions)

def test_qualify_event_paths():
    assert qualify_event(0.10, 1, 18.0, 100.0) == (True, "ok")
    assert qualify_event(0.04, 1, 18.0, 100.0)[1] == "sue_below_long_threshold"
    assert qualify_event(0.10, 4, 18.0, 100.0)[1] == "first_session_gt_max_cal_days"
    assert qualify_event(0.10, 1, 30.5, 100.0)[1] == "vix_block"
    assert qualify_event(0.10, 1, 30.0, 100.0) == (True, "ok")   # block is STRICT >
    assert qualify_event(0.10, 1, np.nan, 100.0) == (True, "ok")  # no VIX: fail open
    assert qualify_event(0.10, 1, 18.0, np.nan)[1] == "no_entry_bar"
    assert qualify_event(0.05, 1, 18.0, 100.0) == (True, "ok")   # threshold is >=


def test_split_artifact_guard_heals_cache(tmp_path, monkeypatch):
    """A >45% overnight cliff (un-readjusted split rows in the incremental
    cache) triggers ONE fresh full-window refetch that heals the cache."""
    from app.data.cache import DataCache
    import app.research.event_panel as ep

    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    half = len(idx) // 2
    corrupt = _make_bars(7, base=400.0, index=idx)
    corrupt.iloc[:half] = corrupt.iloc[:half] * 4.0   # pre-split rows unadjusted
    clean = _make_bars(7, base=100.0, index=idx)

    cache = DataCache(cache_dir=tmp_path)
    cache.put_daily("XYZ", corrupt)
    monkeypatch.setattr(ep, "_fetch_daily_yf", lambda sym, s, e: clean.copy())

    assert len(ep._split_artifact_dates(corrupt)) >= 1
    suspect: set = set()
    df = ep._get_daily_bars(cache, "XYZ", idx[0].date(), idx[-1].date(),
                            suspect_out=suspect)
    assert ep._split_artifact_dates(df) == []          # healed
    assert suspect == set()                            # healed -> NOT suspect
    assert df["close"].iloc[0] == pytest.approx(clean["close"].iloc[0])
    # ... and the heal is persisted on disk, not just in memory.
    df2 = DataCache(cache_dir=tmp_path).get_daily("XYZ", idx[0].date(), idx[-1].date())
    assert ep._split_artifact_dates(df2) == []


def test_calendar_guard_removes_bogus_holiday_bars(tmp_path, monkeypatch):
    """Bars on non-US-session dates (foreign-exchange holiday rows — observed
    on AAPL: Presidents-Day bar at ~0.66x) are healed off the DISK cache, not
    just dropped in memory (put_daily merge alone cannot delete them)."""
    from app.data.cache import DataCache
    import app.research.event_panel as ep

    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    calendar = idx.drop(pd.Timestamp("2024-02-19"))     # Presidents Day closed
    clean = _make_bars(9, base=100.0, index=calendar)
    corrupt = pd.concat([
        clean,
        clean.iloc[[30]].set_index(pd.DatetimeIndex([pd.Timestamp("2024-02-19")]))
        * 0.66,                                          # bogus foreign bar
    ]).sort_index()

    cache = DataCache(cache_dir=tmp_path)
    cache.put_daily("HOL", corrupt)
    monkeypatch.setattr(ep, "_fetch_daily_yf", lambda sym, s, e: clean.copy())

    df = ep._get_daily_bars(cache, "HOL", idx[0].date(), idx[-1].date(),
                            calendar=calendar)
    assert pd.Timestamp("2024-02-19") not in df.index
    df2 = DataCache(cache_dir=tmp_path).get_daily(
        "HOL", idx[0].date(), idx[-1].date())
    assert pd.Timestamp("2024-02-19") not in df2.index   # healed on disk too


def test_split_artifact_guard_keeps_genuine_moves(tmp_path, monkeypatch):
    """If FRESH data shows the same extreme move it is genuine — kept."""
    from app.data.cache import DataCache
    import app.research.event_panel as ep

    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    crash = _make_bars(8, base=50.0, index=idx)
    crash.iloc[len(idx) // 2:] = crash.iloc[len(idx) // 2:] * 0.4  # real -60% day

    cache = DataCache(cache_dir=tmp_path)
    cache.put_daily("CRSH", crash)
    monkeypatch.setattr(ep, "_fetch_daily_yf", lambda sym, s, e: crash.copy())

    df = ep._get_daily_bars(cache, "CRSH", idx[0].date(), idx[-1].date())
    assert len(ep._split_artifact_dates(df)) == 1      # the genuine move stays
    # parquet roundtrip drops the index freq attribute — compare values only.
    pd.testing.assert_frame_equal(df, crash, check_freq=False)


def test_failed_refetch_marks_symbol_suspect(tmp_path, monkeypatch):
    """When the >45%-cliff refetch FAILS (delisted/renamed names yfinance
    cannot serve) the suspect series is KEPT but the symbol is recorded in
    suspect_out — never a silent keep (F1)."""
    from app.data.cache import DataCache
    import app.research.event_panel as ep

    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    half = len(idx) // 2
    corrupt = _make_bars(7, base=400.0, index=idx)
    corrupt.iloc[:half] = corrupt.iloc[:half] * 4.0   # pre-split rows unadjusted

    cache = DataCache(cache_dir=tmp_path)
    cache.put_daily("DEAD", corrupt)
    monkeypatch.setattr(ep, "_fetch_daily_yf", lambda sym, s, e: None)
    monkeypatch.setattr(ep.time, "sleep", lambda s: None)  # skip retry waits

    suspect: set = set()
    df = ep._get_daily_bars(cache, "DEAD", idx[0].date(), idx[-1].date(),
                            suspect_out=suspect)
    assert suspect == {"DEAD"}
    assert df is not None and len(df) == len(idx)      # kept, flagged downstream


def test_insufficient_refetch_marks_symbol_suspect(tmp_path, monkeypatch):
    """A refetch that returns too few rows (<90% of the cached series) also
    leaves the suspect series in place -> suspect_out records the symbol."""
    from app.data.cache import DataCache
    import app.research.event_panel as ep

    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    half = len(idx) // 2
    corrupt = _make_bars(7, base=400.0, index=idx)
    corrupt.iloc[:half] = corrupt.iloc[:half] * 4.0
    thin = _make_bars(7, base=100.0, index=idx[: len(idx) // 4])  # way short

    cache = DataCache(cache_dir=tmp_path)
    cache.put_daily("THIN", corrupt)
    monkeypatch.setattr(ep, "_fetch_daily_yf", lambda sym, s, e: thin.copy())

    suspect: set = set()
    ep._get_daily_bars(cache, "THIN", idx[0].date(), idx[-1].date(),
                       suspect_out=suspect)
    assert suspect == {"THIN"}


def test_suspect_symbol_events_flagged_and_excluded(market):
    """ALL events of a persistent-suspect symbol carry QF_SUSPECT_BARS
    (conservative blanket flag) and drop out of the H1 population via
    select_population; other symbols are untouched."""
    from scripts.run_event_panel_inference import select_population

    p = assemble_panel(
        bars=market["bars"], earnings=market["earnings"],
        spy_bars=market["spy_bars"], vix_close=market["vix_close"],
        sector_etf_bars=market["sector_etf_bars"], sectors=market["sectors"],
        start=date(2024, 1, 1), end=date(2024, 12, 31),
        holdout_start=date(2026, 11, 9),
        suspect_symbols={"AAA"},
    )
    aaa = p[p["symbol"] == "AAA"]
    rest = p[p["symbol"] != "AAA"]
    assert len(aaa) > 0
    assert (aaa["quality_flags"] & QF_SUSPECT_BARS != 0).all()
    assert (rest["quality_flags"] & QF_SUSPECT_BARS == 0).all()
    # The bit is an inference-excluding bit (like the incomplete-fwd flag)...
    assert QF_SUSPECT_BARS & QF_EXCLUDE_FROM_INFERENCE
    # ...so the qualified AAA event is OUT of the H1 population but still IN
    # the panel for audit.
    pop, _ = select_population(p)
    assert not (pop["symbol"] == "AAA").any()
    assert "AAA|2024-05-01" in set(p["event_id"])


def test_parse_earnings_rows_reuses_clip_formula():
    rows = [
        {"date": "2024-05-01", "epsActual": 2.0, "epsEstimated": 1.0},   # +100%
        {"date": "2024-02-01", "epsActual": 0.5, "epsEstimated": -1.0},  # clip -1..1
        {"date": "2024-08-01", "epsActual": None, "epsEstimated": 1.0},  # future
        {"date": "2023-11-01", "epsActual": 1.0, "epsEstimated": 0.0001},  # |est| tiny
    ]
    recs = parse_earnings_rows(rows)
    assert [r["date"] for r in recs] == ["2024-05-01", "2024-02-01", "2023-11-01"]
    assert recs[0]["surprise_pct"] == pytest.approx(1.0)   # clipped at +1
    assert recs[1]["surprise_pct"] == pytest.approx(1.0)   # (0.5-(-1))/1 = 1.5 -> 1.0
    assert recs[2]["surprise_pct"] is None                 # est below 0.001 floor
