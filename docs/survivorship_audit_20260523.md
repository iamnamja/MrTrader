# Survivorship Bias Audit — 2026-05-23

## Summary

- Total symbols in daily cache: **824**
- Known delisted names checked: **24**
- Found in cache: **14 (58%)**
- Missing from cache: **10**
- Verdict: **ACCEPTABLE**

> Note: This audit uses a partial list of S&P 500 deletions 2019–2024.
> For a definitive audit, use CRSP or Compustat membership history.

## Delisted Names — Cache Coverage

| Symbol | In Cache | First Date | Last Date | N Rows |
|--------|----------|------------|-----------|--------|
| HTZ | ❌ MISSING | — | — | — |
| GE | ✅ | 2019-01-02 | 2026-05-22 | 1858 |
| DISCA | ✅ | 2021-01-04 | 2022-04-08 | 320 |
| XLNX | ✅ | 2019-01-02 | 2022-02-11 | 786 |
| ATVI | ❌ MISSING | — | — | — |
| TWTR | ✅ | 2019-01-02 | 2022-10-27 | 964 |
| CTXS | ✅ | 2019-01-02 | 2022-09-29 | 944 |
| VIAC | ❌ MISSING | — | — | — |
| FB | ❌ MISSING | — | — | — |
| BRTX | ❌ MISSING | — | — | — |
| KSU | ✅ | 2019-01-02 | 2021-12-13 | 744 |
| INFO | ❌ MISSING | — | — | — |
| CERN | ✅ | 2019-01-02 | 2022-06-07 | 865 |
| SIVB | ✅ | 2019-01-02 | 2023-03-09 | 1054 |
| FRC | ✅ | 2021-03-15 | 2023-04-28 | 536 |
| SBNY | ❌ MISSING | — | — | — |
| LB | ❌ MISSING | — | — | — |
| PBCT | ✅ | 2019-01-02 | 2022-04-01 | 820 |
| NLOK | ✅ | 2021-01-04 | 2022-11-07 | 466 |
| VRSN | ✅ | 2019-01-02 | 2026-05-22 | 1858 |
| MXIM | ❌ MISSING | — | — | — |
| ALXN | ❌ MISSING | — | — | — |
| RTN | ✅ | 2019-01-02 | 2020-04-02 | 316 |
| UTX | ✅ | 2019-01-02 | 2020-04-02 | 316 |

## Active Survivors — Cache Coverage (Control Group)

| Symbol | First Date | Last Date | N Rows |
|--------|------------|-----------|--------|
| AAPL | 2019-01-02 | 2026-05-22 | 1871 |
| MSFT | 2019-01-02 | 2026-05-22 | 1858 |
| AMZN | 2019-01-02 | 2026-05-22 | 1858 |
| GOOGL | 2019-01-02 | 2026-05-22 | 1858 |
| META | 2021-06-30 | 2026-05-22 | 1140 |
| NVDA | 2019-01-02 | 2026-05-22 | 1858 |
| TSLA | 2019-01-02 | 2026-05-22 | 1858 |
| JPM | 2019-01-02 | 2026-05-22 | 1858 |
| JNJ | 2019-01-02 | 2026-05-22 | 1858 |
| V | 2019-01-02 | 2026-05-22 | 1858 |
| UNH | 2019-01-02 | 2026-05-22 | 1858 |
| HD | 2019-01-02 | 2026-05-22 | 1858 |
| PG | 2019-01-02 | 2026-05-22 | 1858 |
| MA | 2019-01-02 | 2026-05-22 | 1858 |
| BAC | 2019-01-02 | 2026-05-22 | 1858 |

## Interpretation

- If delisted names **are present** in the cache with data through their delisting date: good.
- If delisted names are **absent**: the backtest universe is survivors-only,
  inflating factor scores (losers excluded) and overstating WF Sharpe.

## Recommended Action

- If bias detected: augment cache with historical Polygon S3 data for delisted tickers.
  The `polygon_s3.py` provider already supports this.
- If coverage is adequate: mark audit passed; no action needed.