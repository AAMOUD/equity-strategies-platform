"""
Platform configuration
"""

# Available strategies
STRATEGIES = [
    'Buy-Write',
    'Enhanced Collar',
    'Forward-Start',
    'Vol-Target',
    'ExpOU-Collar'
]

# Default backtest parameters
DEFAULT_PARAMS = {
    'Buy-Write': {
        'maturity_days': 21,
        'strike_offset': 0.02,
        'transaction_cost': 0.001
    },
    'Enhanced Collar': {
        'maturity_days': 63,
        'k1_pct': 0.95,
        'k2_pct': 0.80,
        'kf_pct': 1.02,
        'transaction_cost': 0.0001
    },
    'Forward-Start': {
        'forward_start_days': 21,
        'maturity_days': 63,
        'strike_mult': 1.02,
        'transaction_cost': 0.001
    },
    'Vol-Target': {
        'target_vol': 0.10,
        'lookback_days': 21,
        'rebalance_freq': 5,
        'transaction_cost': 0.001
    },
    'ExpOU-Collar': {
        'maturity_days': 63,
        'k1_pct': 0.95,
        'k2_pct': 0.80,
        'kf_pct': 1.02,
        'transaction_cost': 0.0001
    }
}

# Available assets
ASSETS = {
    'S&P 500': '^GSPC',
    'Nasdaq 100': '^NDX',
    'Russell 2000': '^RUT',
    'Dow Jones': '^DJI',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'Nikkei 225': '^N225',
    'Euro Stoxx 50': '^STOXX50E',
    'Hang Seng': '^HSI',
    'CAC 40': '^FCHI',
    'S&P 500 Equal Weight': '^SPXEW',
    'S&P 400 MidCap': '^MID',
    'S&P 500 Low Volatility': '^SPLV',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOGL',
    'Meta': 'META',
    'NVIDIA': 'NVDA',
    'Alphabet (Class C)': 'GOOG',
    'Berkshire Hathaway B': 'BRK-B',
    'JPMorgan Chase': 'JPM',
    'Johnson & Johnson': 'JNJ',
    'Procter & Gamble': 'PG',
    'Visa': 'V',
    'Mastercard': 'MA',
    'Coca-Cola': 'KO',
    'PepsiCo': 'PEP',
    'Exxon Mobil': 'XOM',
    'Chevron': 'CVX',
    'Walmart': 'WMT',
    'Costco': 'COST',
    'Netflix': 'NFLX',
    'Adobe': 'ADBE',
    'Salesforce': 'CRM',
    'Intel': 'INTC',
    'AMD': 'AMD',
    'Cisco': 'CSCO',
    'Pfizer': 'PFE',
    'UnitedHealth': 'UNH',
    'Boeing': 'BA',
    'Gold': 'GC=F',
    'Oil (WTI)': 'CL=F',
    'US Dollar Index': 'DX-Y.NYB',
    'US 10Y Yield': '^TNX',
}

# Data sources
DATA_SOURCES = {
    'price_ticker': 'Asset ticker symbol',
    'vix_ticker': '^VIX',
    'rf_ticker': '^IRX'
}

# Default date range (in days)
DEFAULT_BACKTEST_DAYS = 365 * 5

# Explicit slider bounds per strategy param: (min, max, step)
PARAM_BOUNDS = {
    'Buy-Write': {
        'maturity_days':    (5,    63,    1),
        'strike_offset':    (0.0,  0.20,  0.01),
        'transaction_cost': (0.0,  0.05,  0.0001),
    },
    'Enhanced Collar': {
        'maturity_days':    (21,   252,   1),
        'k1_pct':           (0.50, 1.50,  0.01),
        'k2_pct':           (0.50, 1.50,  0.01),
        'kf_pct':           (0.50, 1.50,  0.01),
        'transaction_cost': (0.0,  0.05,  0.0001),
    },
    'Forward-Start': {
        'forward_start_days': (5,   63,   1),
        'maturity_days':      (21,  252,  1),
        'strike_mult':        (1.0, 1.20, 0.01),
        'transaction_cost':   (0.0, 0.05, 0.0001),
    },
    'Vol-Target': {
        'target_vol':       (0.05, 0.40, 0.01),
        'lookback_days':    (5,    63,   1),
        'rebalance_freq':   (1,    21,   1),
        'transaction_cost': (0.0,  0.05, 0.0001),
    },
    'ExpOU-Collar': {
        'maturity_days':    (21,   252,   1),
        'k1_pct':           (0.50, 1.50,  0.01),
        'k2_pct':           (0.50, 1.50,  0.01),
        'kf_pct':           (0.50, 1.50,  0.01),
        'transaction_cost': (0.0,  0.05,  0.0001),
    },
}
