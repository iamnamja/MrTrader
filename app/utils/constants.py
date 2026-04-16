"""
Static reference data: S&P 100 tickers, sector mappings, market hours.
"""

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
