# Survivorship Bias Audit — 2026-05-18

## Summary

- Total symbols in daily cache: **816**
- Known delisted names checked: **24**
- Found in cache: **12 (50%)**
- Missing from cache: **12**
- Verdict: **ACCEPTABLE**

> Note: This audit uses a partial list of S&P 500 deletions 2019–2024.
> For a definitive audit, use CRSP or Compustat membership history.

## Delisted Names — Cache Coverage

| Symbol | In Cache | First Date | Last Date | N Rows |
|--------|----------|------------|-----------|--------|
| HTZ | ❌ MISSING | — | — | — |
| GE | ✅ | 2019-01-02 | 2026-05-18 | 1854 |
| DISCA | ❌ MISSING | — | — | — |
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
| NLOK | ❌ MISSING | — | — | — |
| VRSN | ✅ | 2019-01-02 | 2026-05-18 | 1854 |
| MXIM | ❌ MISSING | — | — | — |
| ALXN | ❌ MISSING | — | — | — |
| RTN | ✅ | 2019-01-02 | 2020-04-02 | 316 |
| UTX | ✅ | 2019-01-02 | 2020-04-02 | 316 |

## Active Survivors — Cache Coverage (Control Group)

| Symbol | First Date | Last Date | N Rows |
|--------|------------|-----------|--------|
| AAPL | 2019-01-02 | 2026-05-18 | 1867 |
| MSFT | 2019-01-02 | 2026-05-18 | 1854 |
| AMZN | 2019-01-02 | 2026-05-18 | 1854 |
| GOOGL | 2019-01-02 | 2026-05-18 | 1854 |
| META | 2021-06-30 | 2026-05-18 | 1136 |
| NVDA | 2019-01-02 | 2026-05-18 | 1854 |
| TSLA | 2019-01-02 | 2026-05-18 | 1854 |
| JPM | 2019-01-02 | 2026-05-18 | 1854 |
| JNJ | 2019-01-02 | 2026-05-18 | 1854 |
| V | 2019-01-02 | 2026-05-18 | 1854 |
| UNH | 2019-01-02 | 2026-05-18 | 1854 |
| HD | 2019-01-02 | 2026-05-18 | 1854 |
| PG | 2019-01-02 | 2026-05-18 | 1854 |
| MA | 2019-01-02 | 2026-05-18 | 1854 |
| BAC | 2019-01-02 | 2026-05-18 | 1854 |

## Interpretation

- If delisted names **are present** in the cache with data through their delisting date: good.
- If delisted names are **absent**: the backtest universe is survivors-only,
  inflating factor scores (losers excluded) and overstating WF Sharpe.

## Recommended Action

- If bias detected: augment cache with historical Polygon S3 data for delisted tickers.
  The `polygon_s3.py` provider already supports this.
- If coverage is adequate: mark audit passed; no action needed.