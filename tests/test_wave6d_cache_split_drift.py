"""Alpha-v10 audit Wave 6d — daily cache must not mix split/dividend adjustment bases.

The daily cache is append-only + dedup-by-date, but bars are stored split/dividend-ADJUSTED. When a
new split/dividend re-scales the provider's whole back-history, a blind merge mixes adjustment bases
(drift in returns/vol/momentum). Fix: on put_daily, if the overlap between cached and fresh bars
shows close prices differing beyond _DAILY_READJUST_TOL, discard the stale-basis history and keep
only the freshly-adjusted bars (the missing-range re-fetch rebuilds the rest on the current basis).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from app.data.cache import DataCache


def _bars(dates, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": [1000] * len(closes)}, index=idx)


def _cache(tmp_path):
    return DataCache(cache_dir=str(tmp_path))


_OLD = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
_NEW = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]   # 3-date overlap (05/06/07)


def test_normal_same_basis_update_merges(tmp_path):
    c = _cache(tmp_path)
    c.put_daily("AAPL", _bars(_OLD, [100.0, 101.0, 102.0, 103.0]))
    # overlap at the SAME prices + one new day -> clean append, full history kept
    c.put_daily("AAPL", _bars(_NEW, [101.0, 102.0, 103.0, 104.0]))
    out = c.get_daily("AAPL", date(2026, 1, 1), date(2026, 1, 9))
    assert len(out) == 5                          # 5 distinct days retained (history preserved)
    assert out["close"].iloc[0] == 100.0          # oldest bar kept


def test_split_readjustment_discards_stale_history(tmp_path):
    c = _cache(tmp_path)
    c.put_daily("AAPL", _bars(_OLD, [200.0, 202.0, 204.0, 206.0]))   # pre-split adjusted prices
    # a 2:1 split re-scales the WHOLE history; the fresh fetch reports the overlap at ~HALF price
    c.put_daily("AAPL", _bars(_NEW, [101.0, 102.0, 103.0, 104.0]))   # ~50% lower on overlap
    out = c.get_daily("AAPL", date(2026, 1, 1), date(2026, 1, 9))
    # stale-basis bars (200/202/204/206) must be GONE; only the freshly-adjusted basis remains
    assert set(round(x, 1) for x in out["close"]) == {101.0, 102.0, 103.0, 104.0}
    assert 200.0 not in list(out["close"])        # no mixed-basis drift persisted


def test_small_dividend_does_not_trigger_discard(tmp_path):
    c = _cache(tmp_path)
    c.put_daily("AAPL", _bars(_OLD, [100.0, 101.0, 102.0, 103.0]))
    # a small regular dividend back-adjusts recent overlap by < 1% -> NOT a re-adjustment -> keep
    c.put_daily("AAPL", _bars(_NEW, [100.3, 101.3, 102.3, 103.3]))   # ~0.3% diff on overlap
    out = c.get_daily("AAPL", date(2026, 1, 1), date(2026, 1, 9))
    assert len(out) == 5                          # history preserved (no spurious discard)
