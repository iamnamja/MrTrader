"""
Static reference data: S&P 100/500 tickers, sector mappings, market hours.
"""

# ─── S&P 500 Tickers ─────────────────────────────────────────────────────────
# Approx 480 liquid S&P 500 members (as of early 2026); BRK.B/BRK-B for yfinance
SP_500_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN", "IBM", "INTC", "QCOM",
    "TXN", "AMD", "AMAT", "MU", "LRCX", "KLAC", "ADI", "MCHP", "CDNS", "SNPS",
    "ANSS", "FTNT", "PANW", "CRWD", "ZS", "DDOG", "NOW", "SNOW", "TEAM", "ADSK",
    "INTU", "MSCI", "FSLR", "HPQ", "HPE", "WDC", "STX", "NTAP", "JNPR", "CSCO",
    "ANET", "NET", "OKTA", "SPLK", "WDAY", "ROP", "PAYC", "EPAM", "FFIV", "VRSN",
    # Communication Services
    "GOOG", "GOOGL", "META", "NFLX", "DIS", "CHTR", "VZ", "T", "TMUS",
    "WBD", "PARA", "OMC", "IPG", "NWSA", "LYV", "EA", "TTWO", "MTCH",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG",
    "CMG", "ORLY", "AZO", "ROST", "TJX", "DHI", "LEN", "PHM", "NVR",
    "F", "GM", "APTV", "LKQ", "GPC", "BBY", "DRI", "YUM", "HLT", "MAR",
    "CCL", "RCL", "NCLH", "MGM", "WYNN", "LVS", "VFC", "PVH", "HAS", "MHK",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ",
    "EL", "CL", "CHD", "CLX", "SJM", "CAG", "GIS", "K", "CPB", "HRL",
    "TSN", "KHC", "MKC", "INGR", "SYY", "KR", "ACI", "BJ",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SPGI",
    "MCO", "ICE", "CME", "CBOE", "NDAQ", "SCHW", "USB", "PNC", "TFC",
    "COF", "DFS", "SYF", "ALLY", "RF", "CFG", "HBAN", "KEY", "MTB", "CMA",
    "FRC", "SIVB", "ZION", "FHN", "FITB", "NTRS", "STT", "BK", "TROW",
    "AFL", "MET", "PRU", "AIG", "TRV", "CB", "ALL", "PGR", "HIG", "WRB",
    "RE", "EG", "ACGL", "RNR", "L", "LNC", "UNM", "GL", "FG",
    "AMG", "BEN", "IVZ", "APAM", "VCTR", "SEIC", "WEX", "FIS", "FI", "GPN",
    "V", "MA", "PYPL", "SQ", "AFRM",
    # Health Care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "MDT", "SYK", "BSX", "EW", "ZBH", "BAX", "BDX", "ISRG",
    "IDXX", "IQV", "CRL", "MTD", "WAT", "A", "HOLX", "TECH", "NEOG",
    "HCA", "THC", "UHS", "CNC", "MOH", "HUM", "ELV", "CI",
    "BIIB", "REGN", "VRTX", "MRNA", "ILMN", "ALNY", "BMRN", "INCY",
    "EXAS", "DXCM", "PODD", "TNDM", "PHG",
    "PFE", "AZN", "NVO", "RHHBY", "SNY", "GSK",
    # Industrials
    "CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE",
    "EMR", "ETN", "PH", "ROK", "IR", "DOV", "AME", "XYL", "FTV", "GNRC",
    "ITW", "SWK", "FAST", "GWW", "MSC", "WSO", "TT", "LII", "CARR", "OTIS",
    "FDX", "CHRW", "EXPD", "XPO", "SAIA", "ODFL", "JBHT", "KNX", "WERN",
    "DAL", "UAL", "LUV", "AAL", "ALK", "HA", "JBLU",
    "CSX", "NSC", "UNP", "CP", "CNI",
    "WM", "RSG", "CLH", "SRCL", "GFL",
    "J", "PWR", "ACM", "MTZ", "MYR", "PRIM",
    "CTAS", "ADP", "PAYX", "MAN", "RHI", "KELYA",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "MPC", "PSX", "VLO",
    "HES", "DVN", "FANG", "APA", "HAL", "BKR", "FTI", "NOV", "RIG",
    "KMI", "WMB", "OKE", "ET", "EPD", "MMP", "PAA",
    # Materials
    "LIN", "APD", "ECL", "NEM", "FCX", "NUE", "STLD", "RS", "CMC",
    "ALB", "SQM", "LTHM", "MP", "LAC",
    "DD", "DOW", "LYB", "CE", "EMN", "ASH",
    "IP", "PKG", "WRK", "SEE", "SON",
    "MLM", "VMC", "SUM", "EXP", "USCR",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "WELL", "VTR", "PEAK", "HR",
    "O", "NNN", "STOR", "ADC", "EPRT",
    "SPG", "MAC", "CBL", "SKT",
    "EXR", "CUBE", "PSA", "LSI",
    "EQR", "AVB", "ESS", "MAA", "UDR", "CPT",
    "BXP", "VNO", "SLG", "KRC", "DEI", "CLI",
    "DLR", "QTS", "COR",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "ED", "PPL",
    "XEL", "WEC", "ES", "ETR", "FE", "EIX", "CMS", "LNT", "EVRG", "NI",
    "AWK", "SJW", "YORW", "MSEX",
    "AES", "NRG", "VST", "CWEN",
]

# ─── S&P 100 Tickers ─────────────────────────────────────────────────────────
# Note: BRK.B → BRK-B for yfinance compatibility
SP_100_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN", "IBM", "INTC", "QCOM", "TXN", "AMD",
    # Communication Services
    "GOOG", "META", "NFLX", "DIS", "CHTR", "VZ", "T",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SPGI",
    # Health Care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS",
    # Industrials
    "CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Materials
    "LIN", "APD", "ECL", "NEM",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX",
    # Utilities
    "NEE", "DUK", "SO",
]

# ─── Sector Map ───────────────────────────────────────────────────────────────
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology",
    "ACN": "Technology", "IBM": "Technology", "INTC": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "AMD": "Technology",
    # Communication Services
    "GOOG": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CHTR": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples",
    # Financials
    "BRK-B": "Financial Services", "JPM": "Financial Services",
    "BAC": "Financial Services", "WFC": "Financial Services",
    "GS": "Financial Services", "MS": "Financial Services",
    "BLK": "Financial Services", "C": "Financial Services",
    "AXP": "Financial Services", "SPGI": "Financial Services",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "TMO": "Health Care",
    "ABT": "Health Care", "DHR": "Health Care", "BMY": "Health Care",
    "AMGN": "Health Care", "GILD": "Health Care", "CVS": "Health Care",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "GE": "Industrials", "MMM": "Industrials", "DE": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "NEM": "Materials",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
}

SECTOR_LIST = sorted(set(SECTOR_MAP.values()))

# ─── Market Hours (Eastern) ───────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
