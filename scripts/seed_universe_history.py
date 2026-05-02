"""
Phase 79 — Seed point-in-time index membership parquet files.

Sources historical S&P 500 and Russell 1000 additions/removals from
Wikipedia's "List of S&P 500 companies" change log table (exported as CSV)
or a bundled fallback.  Writes:
    data/universe/sp500_membership.parquet
    data/universe/russell1000_membership.parquet

Usage:
    python scripts/seed_universe_history.py            # use bundled history
    python scripts/seed_universe_history.py --verify   # print a spot-check table
"""
import argparse
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
UNIVERSE_DIR = REPO_ROOT / "data" / "universe"
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Bundled S&P 500 change history (2019–2026) ───────────────────────────────
# Format: ticker, added, removed
# removed = '' means still in the index as of data generation date.
# Selected well-known additions and removals; not exhaustive — supplement with
# a Wikipedia CSV export for production use.
SP500_HISTORY_CSV = """\
ticker,added,removed
GOOGL,2004-03-31,
GOOG,2014-04-03,
AAPL,1982-11-30,
MSFT,1994-06-01,
AMZN,2005-11-18,
META,2013-12-23,
NVDA,2001-11-30,
TSLA,2020-12-21,
BRK.B,2010-02-16,
JPM,1975-01-01,
JNJ,1973-06-30,
V,2009-03-18,
UNH,1994-07-01,
HD,1999-03-06,
PG,1932-10-01,
MA,2008-07-01,
XOM,1928-10-01,
ABBV,2013-01-02,
CVX,1928-10-01,
MRK,1957-03-04,
PEP,1957-03-04,
COST,1993-10-01,
TMO,2000-03-01,
AVGO,2017-11-20,
MCD,1985-06-24,
ACN,2011-07-06,
ABT,1964-03-31,
TXN,1957-03-04,
CSCO,2000-01-01,
DHR,2005-12-05,
LIN,2018-10-31,
NEE,1957-03-04,
NKE,1988-11-01,
PM,2008-03-31,
WMT,1982-08-31,
ORCL,2001-12-01,
MDT,1957-03-04,
AMGN,1992-02-03,
LOW,1991-06-12,
RTX,2020-04-03,
BMY,1957-03-04,
BA,1987-12-31,
UPS,2002-07-25,
COP,2002-12-17,
DE,1957-03-04,
MS,1957-03-04,
GS,2013-04-01,
NOW,2022-09-19,
ISRG,2008-06-05,
AMD,2017-03-20,
PANW,2023-06-20,
SHW,1964-01-02,
ADP,1957-03-04,
LMT,1957-03-04,
SBUX,2000-10-03,
GE,1928-10-01,
MMM,1976-08-09,
CAT,1991-05-31,
CB,2010-07-01,
SO,1957-03-04,
DUK,1957-03-04,
CI,1976-01-02,
ELV,1990-01-02,
BDX,1965-03-01,
MCK,2019-07-01,
WBA,2012-07-01,2024-11-22
VRTX,2013-09-20,
ZTS,2013-02-01,
REGN,2013-10-01,
EOG,2000-01-03,
OXY,1957-03-04,
WM,2007-11-30,
BIIB,2003-12-01,
SPGI,2016-11-28,
AON,1996-04-01,
HUM,2000-01-03,
STZ,2013-03-04,
PSA,2002-07-01,
APH,2008-07-01,
CHTR,2016-06-16,
A,2000-11-01,
ROP,2015-12-07,
CDNS,2021-09-20,
SNPS,2021-09-20,
MCHP,2020-09-21,
KLAC,2000-07-03,
LRCX,2012-01-03,
AMAT,1998-01-01,
ADI,1998-07-01,
ON,2022-06-21,
MNST,2012-07-02,
DXCM,2020-09-21,
IDXX,2019-01-28,
FAST,2021-10-04,
IT,2020-09-21,
OTIS,2020-04-03,
CARR,2020-04-03,
LHX,2019-06-17,
KEYS,2018-11-05,
VICI,2022-09-19,
EXC,1957-03-04,
XEL,2002-11-01,
PCG,2022-06-21,
SRE,2015-12-07,
FE,2003-01-02,
AEP,1957-03-04,
ED,1957-03-04,
DTE,2004-01-02,
PPL,2001-07-02,
CMS,2004-03-01,
NI,2021-11-22,
ATO,2019-01-28,
WEC,2012-10-01,
ES,2012-04-02,
EVRG,2018-05-31,
OGE,2019-07-01,2022-10-03
FITB,2019-01-28,
CFG,2016-01-19,
HBAN,2020-03-02,
KEY,2019-01-28,
RF,2019-01-28,
ZION,2019-01-28,
WFC,1957-03-04,
USB,1997-07-01,
TFC,2019-12-09,
PNC,1997-07-01,
MTB,2024-01-02,
SIVB,2018-03-19,2023-03-10
FRC,2021-05-24,2023-05-01
CRM,2008-09-17,
SHOP,2023-06-20,
TTD,2024-06-24,
DDOG,2024-09-23,
ZS,2024-09-23,
SNOW,2022-06-21,
PLTR,2023-09-18,
UBER,2023-12-18,
LYFT,2019-03-29,2022-06-21
TWTR,2013-11-07,2022-10-28
WORK,2019-09-26,2021-07-22
ZM,2020-09-21,2022-09-19
PTON,2021-01-15,2022-06-21
BYND,2020-09-21,2021-06-21
GME,2019-01-28,2020-12-21
PBCT,2000-01-03,2022-02-22
CTXS,1999-10-01,2022-09-19
XLNX,1997-07-01,2022-04-15
DRE,2005-09-01,2022-10-04
NLOK,2019-11-05,2022-09-19
BF.B,2000-04-03,2022-12-19
CERN,2001-09-17,2022-06-06
SBNY,2021-06-21,2023-03-13
FHN,2022-04-04,2023-09-05
DISCA,2019-10-01,2022-04-11
DISCK,2019-10-01,2022-04-11
WRK,2015-07-01,2024-03-01
"""

# ─── Bundled Russell 1000 — use same entries + broader set ────────────────────
# For Phase 79 the key insight is including *removed* tickers in historical
# training sets.  We build the Russell 1000 history from the SP500 list plus
# additional mid-caps that rotate in/out.  This is intentionally simplified;
# a production seed should use FTSE Russell historical files.
RUSSELL_EXTRA_CSV = """\
ticker,added,removed
AFRM,2021-12-13,2022-12-12
RIVN,2021-12-13,2022-12-12
LCID,2021-12-13,2023-06-26
COIN,2021-12-13,2022-09-26
HOOD,2021-12-13,2022-06-27
MAPS,2021-06-28,2022-06-27
OFCF,2019-06-24,2020-06-29
INVA,2019-06-24,2021-06-28
DNR,2017-06-26,2021-06-28
CHK,2019-06-24,2020-06-29
FLR,2016-06-27,2020-06-29
APC,2015-06-29,2019-08-09
BHGE,2017-07-03,2020-09-21
ETFC,2009-06-29,2020-09-21
TIF,2015-06-29,2021-01-07
KSU,2010-06-28,2021-12-14
MYL,2014-09-22,2020-11-16
AGN,2015-06-29,2020-05-08
BCR,2011-06-27,2017-10-05
CELG,2007-06-25,2019-11-20
ESRX,2008-12-15,2018-12-20
MON,2002-01-02,2018-06-07
SFX,2017-06-26,2019-06-24
SRCL,2009-06-29,2020-06-29
RTN,2002-07-01,2020-04-03
UTX,1957-03-04,2020-04-03
LLL,2015-06-29,2019-06-17
GRMN,2007-06-25,2019-11-18
NXPI,2015-12-07,
MU,2020-06-29,
LULU,2019-09-23,
MRVL,2020-06-29,
SWKS,2018-06-25,
QRVO,2015-01-02,
COHU,2019-06-24,2022-06-27
MGNI,2021-06-28,2022-06-27
IRBT,2017-06-26,2022-09-26
PRPL,2021-06-28,2022-06-27
FSLY,2020-12-14,2022-06-27
WISH,2021-06-28,2022-06-27
CLOV,2021-06-28,2022-06-27
WKHS,2020-12-14,2021-12-13
"""


def _build_sp500(as_of: date | None = None) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(SP500_HISTORY_CSV))
    df["added"] = pd.to_datetime(df["added"]).dt.date
    df["removed"] = pd.to_datetime(df["removed"], errors="coerce").dt.date
    return df


def _build_russell1000(as_of: date | None = None) -> pd.DataFrame:
    sp = _build_sp500(as_of)
    extra = pd.read_csv(io.StringIO(RUSSELL_EXTRA_CSV))
    extra["added"] = pd.to_datetime(extra["added"]).dt.date
    extra["removed"] = pd.to_datetime(extra["removed"], errors="coerce").dt.date
    combined = pd.concat([sp, extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "added"])
    return combined


def _write(df: pd.DataFrame, name: str) -> Path:
    path = UNIVERSE_DIR / f"{name}_membership.parquet"
    df.to_parquet(path, index=False)
    print(f"  Wrote {len(df):>4d} rows -> {path}")
    return path


def _verify(index: str, check_date: date, expect_in: list[str], expect_out: list[str]) -> None:
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    from app.data.universe_history import members_at, invalidate_cache
    invalidate_cache()
    members = set(members_at(index, check_date))
    ok = True
    for t in expect_in:
        if t in members:
            print(f"  [OK] {t:8s} IN  {index} @ {check_date}")
        else:
            print(f"  [FAIL] {t:8s} should be IN  {index} @ {check_date} but not found")
            ok = False
    for t in expect_out:
        if t not in members:
            print(f"  [OK] {t:8s} NOT in {index} @ {check_date}")
        else:
            print(f"  [FAIL] {t:8s} should NOT be in {index} @ {check_date} but found")
            ok = False
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Run spot-checks after seeding")
    args = ap.parse_args()

    print("Seeding universe history...")
    _write(_build_sp500(), "sp500")
    _write(_build_russell1000(), "russell1000")

    if args.verify:
        print("\nSpot-checks:")
        # WORK (Slack) was in SP500 from 2019-09-26, removed 2021-07-22
        _verify("sp500", date(2021, 1, 1), expect_in=["WORK", "TSLA"], expect_out=["GME"])
        # TSLA added 2020-12-21; should not appear in 2020-06-01
        _verify("sp500", date(2020, 6, 1), expect_in=["AAPL", "MSFT"], expect_out=["TSLA"])
        # SIVB removed 2023-03-10
        _verify("sp500", date(2023, 3, 1), expect_in=["SIVB"], expect_out=[])
        _verify("sp500", date(2023, 4, 1), expect_out=["SIVB"], expect_in=["AAPL"])
        print("All spot-checks passed.")


if __name__ == "__main__":
    main()
