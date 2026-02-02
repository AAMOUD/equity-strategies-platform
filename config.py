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
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOGL',
}

# Data sources
DATA_SOURCES = {
    'price_ticker': 'Asset ticker symbol',
    'vix_ticker': '^VIX',
    'rf_ticker': '^IRX'
}

# Default date range (in days)
DEFAULT_BACKTEST_DAYS = 365 * 5  # 5 years
