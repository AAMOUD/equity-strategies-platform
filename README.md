# Equity Strategies Backtest Platform

Professional backtesting platform for equity options strategies with advanced analytics and interactive visualizations.

## Features

- **5 Production-Ready Strategies**
  - Buy-Write: Covered call income generation
  - Enhanced Collar: Downside protection with put spreads
  - Forward-Start: Forward-start call options (Rubinstein 1991)
  - Vol-Target: Dynamic volatility targeting
  - ExpOU-Collar: Exponential OU volatility model

- **Professional Analytics**
  - Real-time NAV tracking and performance metrics
  - Interactive charts with Plotly
  - Drawdown analysis and risk assessment
  - Monthly returns distribution
  - Benchmark comparison (Buy & Hold)

- **Market Data Integration**
  - Yahoo Finance integration via yfinance
  - S&P 500, Nasdaq 100, Russell 2000, major stocks
  - VIX volatility data
  - Risk-free rates (13-week T-Bill)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd equity-strategies-platform

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Strategy Configuration

Each strategy has configurable parameters:

**Buy-Write:**
- Maturity days (option expiration)
- Strike offset (OTM percentage)
- Transaction costs

**Enhanced Collar:**
- Put spread strikes (K1, K2)
- Call strike (forward-start)
- Maturity period

**Forward-Start:**
- Forward start delay
- Maturity days
- Strike multiplier

**Vol-Target:**
- Target volatility level
- Lookback window
- Rebalance frequency

**ExpOU-Collar:**
- Calibrated volatility parameters
- Put spread configuration
- Call strike settings

## Performance Metrics

- **Annualized Return**: Compound annual growth rate
- **Annualized Volatility**: Risk measure (standard deviation)
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of positive return periods

## Technical Stack

- **Frontend**: Streamlit 1.28.0+
- **Data**: yfinance, pandas, numpy
- **Options Pricing**: Black-Scholes, scipy
- **Visualization**: Plotly
- **Python**: 3.8+

## Project Structure

```
equity-strategies-platform/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
├── backtest/
│   └── engine.py         # Backtesting orchestration
├── strategies/
│   ├── base.py           # Abstract strategy class
│   ├── buy_write.py      # Buy-Write strategy
│   ├── enhanced_collar.py # Enhanced Collar
│   ├── forward_start.py  # Forward-Start calls
│   ├── vol_target.py     # Volatility targeting
│   └── expou_collar.py   # ExpOU collar
└── utils/
    └── data.py           # Market data fetching
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
