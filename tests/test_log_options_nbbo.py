"""Nightly options NBBO logger (scripts/log_options_nbbo.py) — FUSE A.

Pure snapshot->row flattening, spread/moneyness math, no-bid/no-ask drops,
append/dedup idempotence on a tmp parquet, and the end-to-end run with the Alpaca
client mocked out (no network).
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.log_options_nbbo import (
    MAX_DTE, OBS_COLS, PANEL, _is_trading_day, append_observations,
    flatten_snapshots, run_nbbo_logging,
)

OBS_DATE = date(2026, 6, 11)


def _snap(bid, ask, *, bs=10, as_=12, iv=None, close=None, vol=None,
          ts="2026-06-11T19:59:59Z"):
    s = {"latestQuote": {"bp": bid, "ap": ask, "bs": bs, "as": as_, "t": ts},
         "dailyBar": {"c": close, "v": vol}}
    if iv is not None:
        s["impliedVolatility"] = iv
    return s


class TestFlattenSnapshots:
    def test_field_mapping_and_math(self):
        snaps = {"SPY260717C00600000": _snap(1.0, 1.5, iv=0.21, close=1.2, vol=345)}
        rows, dropped = flatten_snapshots("SPY", snaps, OBS_DATE,
                                          underlying_price=600.0)
        assert dropped == 0 and len(rows) == 1
        r = rows[0]
        assert r["contract"] == "O:SPY260717C00600000"   # store-joinable OCC form
        assert r["underlying"] == "SPY"
        assert r["contract_type"] == "call"
        assert r["strike"] == 600.0
        assert r["dte"] == (date(2026, 7, 17) - OBS_DATE).days
        assert r["mid"] == pytest.approx(1.25)
        assert r["spread_pct"] == pytest.approx(0.5 / 1.25)
        assert r["moneyness"] == pytest.approx(1.0)
        assert r["bid_size"] == 10 and r["ask_size"] == 12
        assert r["iv"] == pytest.approx(0.21)
        assert r["day_close"] == pytest.approx(1.2)
        assert r["day_volume"] == 345
        assert r["feed"] == "indicative"
        assert set(r) == set(OBS_COLS)

    def test_crossed_quote_dropped_locked_kept(self):
        snaps = {
            "SPY260717C00600000": _snap(1.5, 1.0),    # crossed (bid > ask): dropped
            "SPY260717P00600000": _snap(1.0, 1.0),    # locked (bid == ask): kept
        }
        rows, dropped = flatten_snapshots("SPY", snaps, OBS_DATE,
                                          underlying_price=600.0)
        assert len(rows) == 1 and dropped == 1
        assert rows[0]["contract"] == "O:SPY260717P00600000"
        assert rows[0]["spread_pct"] == 0.0            # never negative in the store

    def test_no_bid_or_no_ask_dropped_and_counted(self):
        snaps = {
            "SPY260717C00600000": _snap(0.0, 1.5),    # no bid
            "SPY260717P00600000": _snap(1.0, None),   # no ask
            "SPY260717C00610000": _snap(1.0, 1.2),    # kept
        }
        rows, dropped = flatten_snapshots("SPY", snaps, OBS_DATE,
                                          underlying_price=600.0)
        assert len(rows) == 1 and dropped == 2

    def test_dte_window_and_malformed_keys_dropped(self):
        far_exp = "SPY270115C00600000"                 # ~7 months out > MAX_DTE
        expired = "SPY260605C00600000"                 # before obs_date
        snaps = {far_exp: _snap(1, 1.2), expired: _snap(1, 1.2),
                 "GARBAGE": _snap(1, 1.2)}
        rows, dropped = flatten_snapshots("SPY", snaps, OBS_DATE,
                                          underlying_price=600.0, max_dte=MAX_DTE)
        assert rows == [] and dropped == 3

    def test_missing_underlying_price_gives_nan_moneyness(self):
        rows, _ = flatten_snapshots("SPY", {"SPY260717C00600000": _snap(1, 1.2)},
                                    OBS_DATE, underlying_price=None)
        import math
        assert math.isnan(rows[0]["moneyness"])
        assert math.isnan(rows[0]["underlying_price"])


class TestAppendDedup:
    def _df(self, bid=1.0):
        rows, _ = flatten_snapshots("SPY", {"SPY260717C00600000": _snap(bid, bid + 0.2)},
                                    OBS_DATE, underlying_price=600.0)
        return pd.DataFrame(rows, columns=OBS_COLS)

    def test_append_then_rerun_is_idempotent(self, tmp_path):
        out = tmp_path / "obs.parquet"
        assert append_observations(self._df(), out) == 1
        assert append_observations(self._df(), out) == 1   # same (contract, obs_date)
        assert len(pd.read_parquet(out)) == 1

    def test_rerun_same_day_keeps_last(self, tmp_path):
        out = tmp_path / "obs.parquet"
        append_observations(self._df(bid=1.0), out)
        append_observations(self._df(bid=2.0), out)
        df = pd.read_parquet(out)
        assert len(df) == 1
        assert df["bid"].iloc[0] == pytest.approx(2.0)

    def test_distinct_days_accumulate(self, tmp_path):
        out = tmp_path / "obs.parquet"
        append_observations(self._df(), out)
        rows, _ = flatten_snapshots("SPY", {"SPY260717C00600000": _snap(1, 1.2)},
                                    date(2026, 6, 12), underlying_price=600.0)
        append_observations(pd.DataFrame(rows, columns=OBS_COLS), out)
        assert len(pd.read_parquet(out)) == 2


class TestFrozenPanel:
    def test_panel_sanity(self):
        assert PANEL[:3] == ["SPY", "QQQ", "IWM"]
        assert len(PANEL) >= 20
        assert len(set(PANEL)) == len(PANEL)
        assert all(u == u.upper() for u in PANEL)


class TestRunNbboLogging:
    def test_end_to_end_with_mocked_client(self, tmp_path):
        out = tmp_path / "obs.parquet"
        snaps = {"SPY260717C00600000": _snap(1.0, 1.2),
                 "SPY260717P00600000": _snap(0.0, 1.2)}   # one dropped (no bid)
        with patch("scripts.log_options_nbbo.fetch_option_snapshots",
                   return_value=snaps) as fos, \
             patch("scripts.log_options_nbbo.fetch_latest_underlying_prices",
                   return_value={"SPY": 600.0}):
            s1 = run_nbbo_logging(["SPY"], out_path=out, obs_date=OBS_DATE)
            s2 = run_nbbo_logging(["SPY"], out_path=out, obs_date=OBS_DATE)  # idempotent
        assert fos.call_count == 2
        assert s1["status"] == "ok"
        assert s1["rows_written"] == 1
        assert s1["rows_dropped_no_quote"] == 1
        assert s2["store_rows_total"] == 1                 # dedup held
        df = pd.read_parquet(out)
        assert list(df.columns) == OBS_COLS
        assert df["feed"].iloc[0] == "indicative"          # quality caveat auditable

    def test_total_failure_reports_no_data(self, tmp_path):
        with patch("scripts.log_options_nbbo.fetch_option_snapshots",
                   side_effect=RuntimeError("feed down")), \
             patch("scripts.log_options_nbbo.fetch_latest_underlying_prices",
                   return_value={}):
            # obs_date pinned to a known trading day so the calendar gate never
            # turns this into "skipped" when the suite runs on a weekend/holiday.
            s = run_nbbo_logging(["SPY"], out_path=tmp_path / "obs.parquet",
                                 obs_date=OBS_DATE)
        assert s["status"] == "no_data"
        assert s["failed_underlyings"] == ["SPY"]
        assert not (tmp_path / "obs.parquet").exists()


class TestTradingDayGate:
    def test_calendar_knows_weekends_and_nyse_holidays(self):
        assert _is_trading_day(date(2026, 6, 11))          # Thursday session
        assert not _is_trading_day(date(2026, 6, 13))      # Saturday
        assert not _is_trading_day(date(2026, 6, 19))      # Juneteenth
        assert not _is_trading_day(date(2026, 7, 3))       # July 4th observed
        assert not _is_trading_day(date(2026, 11, 26))     # Thanksgiving

    def test_holiday_skips_without_fetching(self, tmp_path):
        # The 15:55 weekday schedule fires on market holidays — the gate must skip
        # (prior-session quotes under a phantom obs_date) and write NOTHING.
        with patch("scripts.log_options_nbbo.fetch_option_snapshots") as fos:
            s = run_nbbo_logging(["SPY"], out_path=tmp_path / "obs.parquet",
                                 obs_date=date(2026, 11, 26))
        assert s["status"] == "skipped"
        assert s["reason"] == "not_a_trading_day"
        assert fos.call_count == 0
        assert not (tmp_path / "obs.parquet").exists()

    def test_force_overrides_calendar(self, tmp_path):
        snaps = {"SPY261218C00600000": _snap(1.0, 1.2)}   # expires after the obs_date
        with patch("scripts.log_options_nbbo.fetch_option_snapshots",
                   return_value=snaps), \
             patch("scripts.log_options_nbbo.fetch_latest_underlying_prices",
                   return_value={"SPY": 600.0}):
            s = run_nbbo_logging(["SPY"], out_path=tmp_path / "obs.parquet",
                                 obs_date=date(2026, 11, 26), force=True)
        assert s["status"] == "ok" and s["rows_written"] == 1

    def test_calendar_check_failure_proceeds_fail_safe(self, tmp_path):
        # A broken calendar must NOT silently kill data collection — a missed
        # calibration day is worse than a flagged one.
        snaps = {"SPY260717C00600000": _snap(1.0, 1.2)}
        with patch("scripts.log_options_nbbo._is_trading_day",
                   side_effect=RuntimeError("calendar exploded")), \
             patch("scripts.log_options_nbbo.fetch_option_snapshots",
                   return_value=snaps), \
             patch("scripts.log_options_nbbo.fetch_latest_underlying_prices",
                   return_value={"SPY": 600.0}):
            s = run_nbbo_logging(["SPY"], out_path=tmp_path / "obs.parquet",
                                 obs_date=OBS_DATE)
        assert s["status"] == "ok" and s["rows_written"] == 1
