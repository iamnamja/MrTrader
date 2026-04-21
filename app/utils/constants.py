"""
Static reference data: S&P 100/500 tickers, sector mappings, market hours.
"""

# ─── S&P 500 Tickers ─────────────────────────────────────────────────────────
# Approx 480 liquid S&P 500 members (as of early 2026); BRK-B excluded (invalid on Alpaca)
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
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SPGI",
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

# ─── Russell 1000 Extra Tickers ──────────────────────────────────────────────
# Mid/large-cap Russell 1000 members not already in SP_500_TICKERS (as of 2026).
# yfinance will silently skip any delisted or unresolvable symbols.
RUSSELL_1000_EXTRA = [
    # Technology / Software
    "UBER", "ABNB", "DASH", "RBLX", "COIN", "HOOD", "SOFI", "AFRM",
    "HUBS", "TWLO", "DOCN", "GTLB", "MDB", "DDOG", "ZI", "BILL",
    "PCTY", "SMAR", "APPN", "COUP", "JAMF", "FROG", "ESTC", "ELASTIC",
    "PSTG", "NTDOY", "GRAB", "SE", "RELY", "FLYW", "EFX", "VRSK",
    "LDOS", "SAIC", "CACI", "BAH", "CSIQ", "RUN", "ENPH", "SEDG",
    "APPS", "IRBT", "SMCI", "ONTO", "CRUS", "WOLF", "AMBA", "SLAB",
    "FORM", "ACLS", "MKSI", "COHU", "ICHR", "UCTT",
    # Consumer / Retail
    "DUOL", "PINS", "SNAP", "LYFT", "DKNG", "PENN", "CHWY", "W",
    "ETSY", "POSHMARK", "REAL", "RENT", "PRTY", "BYND", "APRN",
    "ELF", "ULTA", "SFM", "GO", "FOUR", "FWRG", "JACK", "CAKE",
    "TXRH", "EAT", "DIN", "BLMN", "SHAK", "RRGB", "NDLS",
    "BOOT", "CROX", "SKX", "GOOS", "PVH", "LULU", "VFC",
    "WSM", "RH", "ARHAUS", "HIBB", "CATO", "ANF", "AEO", "GPS",
    "CHICO", "EXPR", "JOANN", "TLYS",
    # Financials / Fintech
    "SQ", "AFRM", "UPST", "LC", "OPEN", "CURO", "WRLD", "RM",
    "CACC", "DT", "TREE", "RATE", "HMST", "COLB", "PACW", "WAL",
    "BOKF", "IBOC", "CVBF", "TCBK", "HOPE", "NBTB", "TBNK",
    "OFG", "BRKL", "DCOM", "EGBN", "WASH", "NYCB", "TBK",
    "WSFS", "UMBF", "FFIN", "CBTX", "HTLF", "PPBI", "BANR",
    "SFNC", "FBMS", "CATY", "HAFC", "BCAL", "PFBC",
    # Health Care
    "DOCS", "ACAD", "HALO", "INSM", "ITCI", "SAVA", "RCKT", "BEAM",
    "EDIT", "NTLA", "CRSP", "FATE", "BLUE", "SGMO", "RARE", "FOLD",
    "IONS", "SRPT", "PTCT", "NBIX", "ARWR", "MDGL", "RYTM",
    "AXSM", "INVA", "PCRX", "PAHC", "LNTH", "DVAX", "VNDA",
    "QDEL", "ONEM", "OCDX", "MMSI", "NVCR", "GKOS", "ICAD",
    "OMCL", "NTRA", "NVST", "HSIC", "PDCO", "XRAY",
    # Industrials
    "GTLS", "GNSS", "FSS", "ROAD", "MATX", "ARCB", "MRTN",
    "HUBG", "ECHO", "RADX", "RXO", "GXO", "DSVL", "LSTR",
    "MWA", "BMI", "RBC", "WIRE", "AIN", "TREX", "FBHS",
    "AZEK", "PGTI", "PATK", "DOOR", "UFPI", "UFP", "LPX",
    "BECN", "IBP", "BLDR", "GMS", "SITE", "POOL", "LESL",
    "SSD", "NCS", "AWI", "TILE", "APOG", "LYTS",
    # Energy
    "AR", "CTRA", "SM", "MTDR", "ESTE", "BATL", "TALO",
    "PDCE", "VTLE", "CLR", "MRO", "PR", "GPOR", "SWN",
    "RRC", "CNX", "CHRD", "CIVI", "DEN", "ERF",
    "HESM", "ENLC", "DKL", "MPLX", "TRGP", "GLP", "NGL",
    # Materials
    "CC", "HUN", "OLN", "RPM", "FUL", "H", "KWR",
    "TROX", "TPC", "HWKN", "BCPC", "GEF", "SLVM", "BERY",
    "ATI", "CENX", "AMG", "CSTM", "KALU", "MTRN", "SCCO",
    "MP", "NOVT", "ITRI", "REXNORD", "GNRC",
    # Real Estate
    "IRT", "NHI", "LTC", "SBRA", "GMRE", "GOOD", "LAND",
    "IIPR", "SAFE", "PINE", "ALEX", "ROIC", "KITE", "RPAI",
    "NXRT", "APTS", "CLPR", "AIRC", "NRE", "JBGS",
    "NFIN", "EQC", "PDM", "HIW", "CIO", "OFC",
    "COLD", "FR", "EGP", "STAG", "REXR", "ILPT",
    # Utilities
    "SPWR", "NOVA", "RUN", "ARRY", "STEM", "BE",
    "ORA", "GPRE", "CLNE", "TPVG", "HASI", "CWEN",
    "BEP", "BEPC", "NEP", "NRUC",
]

RUSSELL_1000_TICKERS = list(dict.fromkeys(SP_500_TICKERS + RUSSELL_1000_EXTRA))

# ─── S&P 100 Tickers ─────────────────────────────────────────────────────────
# Note: BRK-B excluded from universe (invalid Alpaca symbol)
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
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SPGI",
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
# Covers all SP_500_TICKERS; unknown symbols default to "Unknown" via .get()
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology",
    "ACN": "Technology", "IBM": "Technology", "INTC": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "AMD": "Technology",
    "AMAT": "Technology", "MU": "Technology", "LRCX": "Technology",
    "KLAC": "Technology", "ADI": "Technology", "MCHP": "Technology",
    "CDNS": "Technology", "SNPS": "Technology", "ANSS": "Technology",
    "FTNT": "Technology", "PANW": "Technology", "CRWD": "Technology",
    "ZS": "Technology", "DDOG": "Technology", "NOW": "Technology",
    "SNOW": "Technology", "TEAM": "Technology", "ADSK": "Technology",
    "INTU": "Technology", "MSCI": "Technology", "FSLR": "Technology",
    "HPQ": "Technology", "HPE": "Technology", "WDC": "Technology",
    "STX": "Technology", "NTAP": "Technology", "JNPR": "Technology",
    "CSCO": "Technology", "ANET": "Technology", "NET": "Technology",
    "OKTA": "Technology", "SPLK": "Technology", "WDAY": "Technology",
    "ROP": "Technology", "PAYC": "Technology", "EPAM": "Technology",
    "FFIV": "Technology", "VRSN": "Technology",
    # Communication Services
    "GOOG": "Communication Services", "GOOGL": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CHTR": "Communication Services",
    "VZ": "Communication Services", "T": "Communication Services",
    "TMUS": "Communication Services", "WBD": "Communication Services",
    "PARA": "Communication Services", "OMC": "Communication Services",
    "IPG": "Communication Services", "NWSA": "Communication Services",
    "LYV": "Communication Services", "EA": "Communication Services",
    "TTWO": "Communication Services", "MTCH": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "DHI": "Consumer Discretionary", "LEN": "Consumer Discretionary",
    "PHM": "Consumer Discretionary", "NVR": "Consumer Discretionary",
    "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "APTV": "Consumer Discretionary", "LKQ": "Consumer Discretionary",
    "GPC": "Consumer Discretionary", "BBY": "Consumer Discretionary",
    "DRI": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    "HLT": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    "CCL": "Consumer Discretionary", "RCL": "Consumer Discretionary",
    "NCLH": "Consumer Discretionary", "MGM": "Consumer Discretionary",
    "WYNN": "Consumer Discretionary", "LVS": "Consumer Discretionary",
    "VFC": "Consumer Discretionary", "PVH": "Consumer Discretionary",
    "HAS": "Consumer Discretionary", "MHK": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples",
    "EL": "Consumer Staples", "CL": "Consumer Staples", "CHD": "Consumer Staples",
    "CLX": "Consumer Staples", "SJM": "Consumer Staples", "CAG": "Consumer Staples",
    "GIS": "Consumer Staples", "K": "Consumer Staples", "CPB": "Consumer Staples",
    "HRL": "Consumer Staples", "TSN": "Consumer Staples", "KHC": "Consumer Staples",
    "MKC": "Consumer Staples", "INGR": "Consumer Staples", "SYY": "Consumer Staples",
    "KR": "Consumer Staples", "ACI": "Consumer Staples", "BJ": "Consumer Staples",
    # Financial Services
    "JPM": "Financial Services", "BAC": "Financial Services",
    "WFC": "Financial Services", "GS": "Financial Services",
    "MS": "Financial Services", "BLK": "Financial Services",
    "C": "Financial Services", "AXP": "Financial Services",
    "SPGI": "Financial Services", "MCO": "Financial Services",
    "ICE": "Financial Services", "CME": "Financial Services",
    "CBOE": "Financial Services", "NDAQ": "Financial Services",
    "SCHW": "Financial Services", "USB": "Financial Services",
    "PNC": "Financial Services", "TFC": "Financial Services",
    "COF": "Financial Services", "DFS": "Financial Services",
    "SYF": "Financial Services", "ALLY": "Financial Services",
    "RF": "Financial Services", "CFG": "Financial Services",
    "HBAN": "Financial Services", "KEY": "Financial Services",
    "MTB": "Financial Services", "CMA": "Financial Services",
    "FRC": "Financial Services", "SIVB": "Financial Services",
    "ZION": "Financial Services", "FHN": "Financial Services",
    "FITB": "Financial Services", "NTRS": "Financial Services",
    "STT": "Financial Services", "BK": "Financial Services",
    "TROW": "Financial Services", "AFL": "Financial Services",
    "MET": "Financial Services", "PRU": "Financial Services",
    "AIG": "Financial Services", "TRV": "Financial Services",
    "CB": "Financial Services", "ALL": "Financial Services",
    "PGR": "Financial Services", "HIG": "Financial Services",
    "WRB": "Financial Services", "RE": "Financial Services",
    "EG": "Financial Services", "ACGL": "Financial Services",
    "RNR": "Financial Services", "L": "Financial Services",
    "LNC": "Financial Services", "UNM": "Financial Services",
    "GL": "Financial Services", "FG": "Financial Services",
    "AMG": "Financial Services", "BEN": "Financial Services",
    "IVZ": "Financial Services", "APAM": "Financial Services",
    "VCTR": "Financial Services", "SEIC": "Financial Services",
    "WEX": "Financial Services", "FIS": "Financial Services",
    "FI": "Financial Services", "GPN": "Financial Services",
    "V": "Financial Services", "MA": "Financial Services",
    "PYPL": "Financial Services", "SQ": "Financial Services",
    "AFRM": "Financial Services",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "TMO": "Health Care",
    "ABT": "Health Care", "DHR": "Health Care", "BMY": "Health Care",
    "AMGN": "Health Care", "GILD": "Health Care", "CVS": "Health Care",
    "MDT": "Health Care", "SYK": "Health Care", "BSX": "Health Care",
    "EW": "Health Care", "ZBH": "Health Care", "BAX": "Health Care",
    "BDX": "Health Care", "ISRG": "Health Care", "IDXX": "Health Care",
    "IQV": "Health Care", "CRL": "Health Care", "MTD": "Health Care",
    "WAT": "Health Care", "A": "Health Care", "HOLX": "Health Care",
    "TECH": "Health Care", "NEOG": "Health Care", "HCA": "Health Care",
    "THC": "Health Care", "UHS": "Health Care", "CNC": "Health Care",
    "MOH": "Health Care", "HUM": "Health Care", "ELV": "Health Care",
    "CI": "Health Care", "BIIB": "Health Care", "REGN": "Health Care",
    "VRTX": "Health Care", "MRNA": "Health Care", "ILMN": "Health Care",
    "ALNY": "Health Care", "BMRN": "Health Care", "INCY": "Health Care",
    "EXAS": "Health Care", "DXCM": "Health Care", "PODD": "Health Care",
    "TNDM": "Health Care", "PHG": "Health Care", "PFE": "Health Care",
    "AZN": "Health Care", "NVO": "Health Care", "RHHBY": "Health Care",
    "SNY": "Health Care", "GSK": "Health Care",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "GE": "Industrials", "MMM": "Industrials", "DE": "Industrials",
    "EMR": "Industrials", "ETN": "Industrials", "PH": "Industrials",
    "ROK": "Industrials", "IR": "Industrials", "DOV": "Industrials",
    "AME": "Industrials", "XYL": "Industrials", "FTV": "Industrials",
    "GNRC": "Industrials", "ITW": "Industrials", "SWK": "Industrials",
    "FAST": "Industrials", "GWW": "Industrials", "MSC": "Industrials",
    "WSO": "Industrials", "TT": "Industrials", "LII": "Industrials",
    "CARR": "Industrials", "OTIS": "Industrials", "FDX": "Industrials",
    "CHRW": "Industrials", "EXPD": "Industrials", "XPO": "Industrials",
    "SAIA": "Industrials", "ODFL": "Industrials", "JBHT": "Industrials",
    "KNX": "Industrials", "WERN": "Industrials", "DAL": "Industrials",
    "UAL": "Industrials", "LUV": "Industrials", "AAL": "Industrials",
    "ALK": "Industrials", "HA": "Industrials", "JBLU": "Industrials",
    "CSX": "Industrials", "NSC": "Industrials", "UNP": "Industrials",
    "CP": "Industrials", "CNI": "Industrials", "WM": "Industrials",
    "RSG": "Industrials", "CLH": "Industrials", "SRCL": "Industrials",
    "GFL": "Industrials", "J": "Industrials", "PWR": "Industrials",
    "ACM": "Industrials", "MTZ": "Industrials", "MYR": "Industrials",
    "PRIM": "Industrials", "CTAS": "Industrials", "ADP": "Industrials",
    "PAYX": "Industrials", "MAN": "Industrials", "RHI": "Industrials",
    "KELYA": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PXD": "Energy", "OXY": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "HES": "Energy", "DVN": "Energy",
    "FANG": "Energy", "APA": "Energy", "HAL": "Energy", "BKR": "Energy",
    "FTI": "Energy", "NOV": "Energy", "RIG": "Energy", "KMI": "Energy",
    "WMB": "Energy", "OKE": "Energy", "ET": "Energy", "EPD": "Energy",
    "MMP": "Energy", "PAA": "Energy",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "NUE": "Materials",
    "STLD": "Materials", "RS": "Materials", "CMC": "Materials",
    "ALB": "Materials", "SQM": "Materials", "LTHM": "Materials",
    "MP": "Materials", "LAC": "Materials", "DD": "Materials",
    "DOW": "Materials", "LYB": "Materials", "CE": "Materials",
    "EMN": "Materials", "ASH": "Materials", "IP": "Materials",
    "PKG": "Materials", "WRK": "Materials", "SEE": "Materials",
    "SON": "Materials", "MLM": "Materials", "VMC": "Materials",
    "SUM": "Materials", "EXP": "Materials", "USCR": "Materials",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "WELL": "Real Estate", "VTR": "Real Estate",
    "PEAK": "Real Estate", "HR": "Real Estate", "O": "Real Estate",
    "NNN": "Real Estate", "STOR": "Real Estate", "ADC": "Real Estate",
    "EPRT": "Real Estate", "SPG": "Real Estate", "MAC": "Real Estate",
    "CBL": "Real Estate", "SKT": "Real Estate", "EXR": "Real Estate",
    "CUBE": "Real Estate", "PSA": "Real Estate", "LSI": "Real Estate",
    "EQR": "Real Estate", "AVB": "Real Estate", "ESS": "Real Estate",
    "MAA": "Real Estate", "UDR": "Real Estate", "CPT": "Real Estate",
    "BXP": "Real Estate", "VNO": "Real Estate", "SLG": "Real Estate",
    "KRC": "Real Estate", "DEI": "Real Estate", "CLI": "Real Estate",
    "DLR": "Real Estate", "QTS": "Real Estate", "COR": "Real Estate",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    "SRE": "Utilities", "PCG": "Utilities", "ED": "Utilities",
    "PPL": "Utilities", "XEL": "Utilities", "WEC": "Utilities",
    "ES": "Utilities", "ETR": "Utilities", "FE": "Utilities",
    "EIX": "Utilities", "CMS": "Utilities", "LNT": "Utilities",
    "EVRG": "Utilities", "NI": "Utilities", "AWK": "Utilities",
    "SJW": "Utilities", "YORW": "Utilities", "MSEX": "Utilities",
    "AES": "Utilities", "NRG": "Utilities", "VST": "Utilities",
    "CWEN": "Utilities",
}

SECTOR_LIST = sorted(set(SECTOR_MAP.values()))

# ─── Market Hours (Eastern) ───────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
