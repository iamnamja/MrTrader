"""Greeks backfill (scripts/backfill_computed_greeks.py) — FUSE B.

The load-bearing claims: the warm-start American IV solver recovers a known sigma
WITHOUT ever touching the CRR-500 fallback (the 31 ms/solve path the R2 spike ruled
out), the solver_status mapping is explicit on every None-path, and the per-underlying
pipeline (synthetic bars -> hive part file) computes T/greeks/stale_flag correctly.
No network: rate/dividend/close inputs are passed in directly.
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from app.options.pricing_engine import ENGINE, american_price
from scripts.backfill_computed_greeks import (
    GREEKS_COLS, SOLVER_STATUSES, _q_asof, fetch_raw_closes,
    pending_underlyings, process_underlying, solve_iv_american,
)

S, K, R = 100.0, 100.0, 0.045
T = 165 / 365.0
SIGMA = 0.25


class TestWarmStartSolver:
    def test_recovers_known_sigma_put(self):
        price = american_price(S, K, T, R, 0.0, SIGMA, "put")
        iv, status = solve_iv_american(price, S, K, T, R, 0.0, "put")
        assert status == "ok"
        assert iv == pytest.approx(SIGMA, abs=1e-3)

    def test_recovers_known_sigma_call_with_dividends(self):
        price = american_price(S, K, T, R, 0.02, SIGMA, "call")
        iv, status = solve_iv_american(price, S, K, T, R, 0.02, "call")
        assert status == "ok"
        assert iv == pytest.approx(SIGMA, abs=1e-3)

    def test_never_calls_crr(self, monkeypatch):
        # THE R2 regression guard: the naive solver's sigma->1e-4 probe routes q<r puts
        # into the CRR-500 fallback (~31 ms/solve -> ~122h wall). The warm bracket must
        # keep every american_price call out of the degenerate regime entirely.
        def _boom(*a, **kw):
            raise AssertionError("CRR fallback fired — warm bracket regressed")
        monkeypatch.setattr("app.options.pricing_engine.crr_price", _boom)
        for kind in ("put", "call"):
            for sig in (0.12, 0.30, 0.80):
                price = american_price(S, K, T, R, 0.0, sig, kind)
                iv, status = solve_iv_american(price, S, K, T, R, 0.0, kind)
                assert status == "ok"
                assert iv == pytest.approx(sig, abs=1e-3)

    def test_agrees_with_naive_engine_solve(self):
        # Calls with q<r never hit the CRR-degenerate regime, so the naive engine path
        # is safe to use as the reference here.
        price = american_price(S, K, T, R, 0.0, SIGMA, "call")
        naive = ENGINE.implied_vol(price, S, K, T, R, 0.0, "call", style="american")
        warm, status = solve_iv_american(price, S, K, T, R, 0.0, "call")
        assert status == "ok"
        assert warm == pytest.approx(naive, abs=1e-4)

    # ── solver_status mapping (never silently default) ───────────────────────

    def test_below_intrinsic(self):
        iv, status = solve_iv_american(10.0, 80.0, 100.0, T, R, 0.0, "put")
        assert (iv, status) == (None, "below_intrinsic")

    def test_nonpositive_price_maps_below_intrinsic(self):
        assert solve_iv_american(0.0, S, K, T, R, 0.0, "call")[1] == "below_intrinsic"
        assert solve_iv_american(None, S, K, T, R, 0.0, "call")[1] == "below_intrinsic"

    def test_pinned_at_intrinsic_floor(self):
        # Deep-ITM American put priced exactly at intrinsic: vol is unidentifiable
        # (the engine's flat-f(lo) case) — must NOT fall through to a European IV.
        iv, status = solve_iv_american(30.0, 70.0, 100.0, 0.5, R, 0.0, "put")
        assert (iv, status) == (None, "pinned")

    def test_out_of_bracket(self):
        # Price above the sigma=3.0 ceiling for an OTM call -> not attainable.
        iv, status = solve_iv_american(99.0, S, 150.0, 0.05, R, 0.0, "call")
        assert (iv, status) == (None, "out_of_bracket")

    def test_expired(self):
        assert solve_iv_american(1.0, S, K, 0.0, R, 0.0, "call")[1] == "expired"

    def test_european_fallback_labeled_not_ok(self):
        # Short-dated low-IV ATM put with r > q: at the warm bracket's low edge the
        # BjS american_price blows up toward S, so f(lo) and f(hi) share a sign (the
        # non-straddling branch) and the solver returns the EUROPEAN IV. That answer
        # is biased (true American IV <= ivE) and MUST be labeled european_fallback —
        # never "ok" — so downstream can filter the contamination.
        iv, status = solve_iv_american(0.10, 100.0, 100.0, 0.05, 0.05, 0.0, "put")
        assert status == "european_fallback"
        assert iv is not None and math.isfinite(iv) and iv > 0
        g = ENGINE.greeks(100.0, 100.0, 0.05, 0.05, 0.0, iv, "put", style="american")
        assert all(math.isfinite(g[k]) for k in ("delta", "gamma", "vega", "theta"))

    def test_european_fallback_in_status_vocabulary(self):
        assert "european_fallback" in SOLVER_STATUSES


class TestDividendYieldAsOf:
    SCHED = [(date(2025, 3, 10), 0.010), (date(2025, 6, 10), 0.012)]
    EX = [d for d, _ in SCHED]

    def test_q_zero_fallback_before_first_ex_date_and_empty_schedule(self):
        assert _q_asof(self.SCHED, self.EX, date(2025, 1, 1)) == 0.0
        assert _q_asof([], [], date(2025, 1, 1)) == 0.0

    def test_step_function_picks_most_recent_ex_date(self):
        assert _q_asof(self.SCHED, self.EX, date(2025, 3, 10)) == 0.010
        assert _q_asof(self.SCHED, self.EX, date(2025, 5, 1)) == 0.010
        assert _q_asof(self.SCHED, self.EX, date(2025, 7, 1)) == 0.012


class TestAsTradedCloses:
    """BLOCKER regression: the S fed to the solver must be AS-TRADED (Polygon
    ``adjusted=false``), matching the store's unadjusted OCC strikes.

    The yfinance trap this guards against: ``auto_adjust=False`` only skips DIVIDEND
    adjustment — closes stay SPLIT-adjusted (NVDA 2023-06-01: 39.77 vs the store's
    ~394 strikes; as-traded is 397.70), so S would be wrong by the split ratio for
    every pre-split bar of every split name.
    """

    T_MS = int(pd.Timestamp("2023-06-01 04:00", tz="UTC").value // 1_000_000)

    class _Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    def _patch_polygon(self, monkeypatch, results):
        calls = {}

        def fake_get(url, params=None, timeout=None):
            calls["url"], calls["params"] = url, dict(params or {})
            return self._Resp({"results": results})

        monkeypatch.setattr("scripts.backfill_computed_greeks.requests.get", fake_get)
        from app.config import settings
        monkeypatch.setattr(settings, "polygon_api_key", "test-key", raising=False)
        return calls

    def test_requests_unadjusted_and_returns_as_traded_close(self, monkeypatch):
        calls = self._patch_polygon(monkeypatch, [{"t": self.T_MS, "c": 397.70}])
        closes = fetch_raw_closes("NVDA")
        assert "/v2/aggs/ticker/NVDA/range/1/day/" in calls["url"]
        assert calls["params"]["adjusted"] == "false"        # TRULY unadjusted
        assert closes == {date(2023, 6, 1): pytest.approx(397.70)}   # NOT 39.77

    def test_solver_receives_as_traded_S(self, monkeypatch, tmp_path):
        # End-to-end: the close the worker hands the solver for a PRE-SPLIT date is
        # the as-traded one (same scale as the ~394 strike), never split-adjusted.
        d = date(2023, 6, 1)
        self._patch_polygon(monkeypatch, [{"t": self.T_MS, "c": 397.70}])
        closes = fetch_raw_closes("NVDA")
        contract = "O:NVDA230616C00394000"          # as-traded strike scale
        bars = pd.DataFrame([{
            "underlying": "NVDA", "contract": contract, "date": pd.Timestamp(d),
            "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0, "volume": 10.0,
            "knowable_date": pd.Timestamp(d) + pd.Timedelta(days=1),
        }])
        meta = pd.DataFrame([{
            "underlying": "NVDA", "contract": contract, "contract_type": "call",
            "strike": 394.0, "expiration": pd.Timestamp(date(2023, 6, 16)),
            "first_date": pd.Timestamp(d), "knowable_date": pd.Timestamp(d),
        }])
        bars_path, meta_path = tmp_path / "bars.parquet", tmp_path / "contracts.parquet"
        bars.to_parquet(bars_path, index=False)
        meta.to_parquet(meta_path, index=False)
        summary = process_underlying("NVDA", str(bars_path), str(meta_path),
                                     str(tmp_path / "greeks"), rates={d: R},
                                     div_schedule=[], closes=closes)
        out = pd.read_parquet(tmp_path / "greeks" / "underlying=NVDA" / "part-0.parquet")
        assert out["underlying_close"].iloc[0] == pytest.approx(397.70)
        assert summary["statuses"].get("no_underlying", 0) == 0


class TestResumeSkip:
    def test_pending_skips_existing_parts_unless_forced(self, tmp_path):
        done = tmp_path / "underlying=SPY"
        done.mkdir()
        (done / "part-0.parquet").touch()
        assert pending_underlyings(["SPY", "QQQ"], tmp_path, force=False) == ["QQQ"]
        assert pending_underlyings(["SPY", "QQQ"], tmp_path, force=True) == ["SPY", "QQQ"]


class TestProcessUnderlying:
    D = date(2026, 1, 5)
    EXP = date(2026, 6, 19)            # T = 165/365 from D

    def _write_store(self, tmp_path):
        put_px = american_price(S, K, T, R, 0.0, SIGMA, "put")
        call_px = american_price(S, K, T, R, 0.0, SIGMA, "call")
        rows = [
            # (contract, date, close, volume) — statuses exercised one each
            ("O:TEST260619P00100000", self.D, put_px, 100.0),       # ok
            ("O:TEST260619C00100000", self.D, call_px, 0.0),        # ok + stale (vol=0)
            ("O:TEST260619P00150000", self.D, 10.0, 50.0),          # below_intrinsic
            ("O:TEST260619P00200000", self.D, 100.0, 50.0),         # pinned (=intrinsic)
            ("O:TEST260619C00100000", self.EXP, 1.0, 10.0),         # expired (T=0)
            ("O:TEST260619C00100000", date(2026, 1, 6), 5.0, 10.0),  # no_underlying
        ]
        bars = pd.DataFrame([{
            "underlying": "TEST", "contract": c, "date": pd.Timestamp(d),
            "open": px, "high": px, "low": px, "close": px, "volume": v,
            "knowable_date": pd.Timestamp(d) + pd.Timedelta(days=1),
        } for c, d, px, v in rows])
        meta = pd.DataFrame([{
            "underlying": "TEST", "contract": c,
            "contract_type": "put" if "P00" in c else "call",
            "strike": float(c[-8:]) / 1000.0, "expiration": pd.Timestamp(self.EXP),
            "first_date": pd.Timestamp(self.D), "knowable_date": pd.Timestamp(self.D),
        } for c in bars["contract"].unique()])
        bars_path = tmp_path / "bars.parquet"
        meta_path = tmp_path / "contracts.parquet"
        bars.to_parquet(bars_path, index=False)
        meta.to_parquet(meta_path, index=False)
        return bars_path, meta_path

    def _run(self, tmp_path):
        bars_path, meta_path = self._write_store(tmp_path)
        out_dir = tmp_path / "greeks"
        summary = process_underlying(
            "TEST", str(bars_path), str(meta_path), str(out_dir),
            rates={self.D: R, self.EXP: R, date(2026, 1, 6): R},
            div_schedule=[], closes={self.D: S, self.EXP: S})
        out = pd.read_parquet(out_dir / "underlying=TEST" / "part-0.parquet")
        return summary, out

    def test_status_histogram_and_schema(self, tmp_path):
        summary, out = self._run(tmp_path)
        assert summary["rows"] == 6
        assert summary["statuses"] == {"ok": 2, "below_intrinsic": 1, "pinned": 1,
                                       "expired": 1, "no_underlying": 1}
        assert list(out.columns) == GREEKS_COLS   # underlying lives in the hive dir
        assert set(out["solver_status"]) == {"ok", "below_intrinsic", "pinned",
                                             "expired", "no_underlying"}

    def test_iv_recovery_and_greeks_match_engine(self, tmp_path):
        _, out = self._run(tmp_path)
        ok = out[out["solver_status"] == "ok"].set_index("contract")
        put = ok.loc["O:TEST260619P00100000"]
        assert put["iv"] == pytest.approx(SIGMA, abs=1e-3)
        g = ENGINE.greeks(S, K, T, R, 0.0, float(put["iv"]), "put", style="american")
        assert put["delta"] == pytest.approx(g["delta"], abs=1e-9)
        assert put["vega"] == pytest.approx(g["vega"], abs=1e-9)
        assert put["delta"] < 0 < put["gamma"]

    def test_stale_flag_and_nan_greeks_on_failed_solves(self, tmp_path):
        _, out = self._run(tmp_path)
        by_c = out.set_index(["contract", "date"])
        assert bool(by_c.loc[("O:TEST260619C00100000", pd.Timestamp(self.D)), "stale_flag"])
        assert not bool(by_c.loc[("O:TEST260619P00100000", pd.Timestamp(self.D)), "stale_flag"])
        failed = out[out["solver_status"] != "ok"]
        assert failed["iv"].isna().all() and failed["delta"].isna().all()

    def test_T_from_calendar_days_over_365(self, tmp_path):
        # The recovered IV is only exact if T = (expiration - date).days / 365 — a
        # wrong day-count would shift the put IV by far more than the 1e-3 tolerance.
        _, out = self._run(tmp_path)
        put = out[(out["contract"] == "O:TEST260619P00100000")
                  & (out["solver_status"] == "ok")].iloc[0]
        assert (pd.Timestamp(self.EXP) - pd.Timestamp(self.D)).days / 365.0 == pytest.approx(T)
        assert put["iv"] == pytest.approx(SIGMA, abs=1e-3)

    def test_expired_row_has_T_zero_semantics(self, tmp_path):
        _, out = self._run(tmp_path)
        exp_row = out[out["date"] == pd.Timestamp(self.EXP)].iloc[0]
        assert exp_row["solver_status"] == "expired"
        assert math.isnan(exp_row["iv"])

    def test_crashed_tmp_file_does_not_break_dataset_reads(self, tmp_path):
        # A mid-write crash leaves the tmp file behind. pyarrow dataset discovery only
        # ignores '.'/'_'-prefixed files, so the tmp name MUST be dot-prefixed — a bare
        # part-0.parquet.tmp would make pd.read_parquet(root) raise until hand-cleaned.
        summary, _ = self._run(tmp_path)
        out_dir = tmp_path / "greeks"
        (out_dir / "underlying=TEST" / ".part-0.parquet.tmp").write_bytes(b"garbage")
        df = pd.read_parquet(out_dir)
        assert len(df) == summary["rows"]
