# Equity Options Strategies Backtesting Platform

A Streamlit backtesting platform for equity options strategies. The technically distinguishing parts are: put spread pricing under an exponential Ornstein-Uhlenbeck volatility model calibrated on historical VIX autocorrelation, forward-start call pricing via the Rubinstein (1991) formula with the strike fixed at the forward date spot, and a dynamic volatility-targeting allocator with half-life smoothing on exposure. All other strategies use standard Black-Scholes.

Live demo: [equity-strategies-platform.streamlit.app](https://equity-strategies-platform.streamlit.app/)

## Architecture

`utils/data.py` fetches close prices, VIX (^VIX), and the annualised 13-week T-Bill rate (^IRX) from Yahoo Finance and aligns them on common trading dates. `backtest/engine.py` is a stateless router that instantiates the requested strategy class and wraps the result with a buy-and-hold benchmark. Each strategy in `strategies/` inherits `BaseStrategy` and implements `run_backtest(price_data, vix_data, rf_data, params) -> (nav_series, metrics)`. `app.py` handles parameter input, caches data fetches for one hour, and renders charts via Plotly.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Strategies

**Buy-Write** — Sells an OTM call at each roll date and settles at expiry. Premium accrues interest to maturity: P&L contribution at expiry = `(C·e^{rT} − max(S_T − K, 0)) / S_0`.

**Enhanced Collar** — Long OTM put spread (K1, K2 as fractions of spot) funded by a short OTM call (K_f > S_0). Collar cost is debited at the roll date; put spread payoff and call settlement are credited at expiry.

**Forward-Start** — Prices a forward-starting call using Rubinstein (1991). The strike `K = κ · S_{t_0 + τ}` is fixed at the forward date using the realised spot, not the spot at inception. The Rubinstein formula prices the option as of today given that the strike will be set proportionally at `τ`.

**Vol-Target** — Adjusts equity exposure to `w = target_vol / σ̂` where `σ̂` is the rolling realised vol over a configurable lookback. Exposure is capped at 1.5× and half-life smoothed each rebalance to limit turnover during vol spikes.

**ExpOU-Collar** — Same collar structure as Enhanced Collar, but the put spread is priced using an effective volatility derived from an exponential OU process fitted to `log(VIX²)`. The calibration matches the autocorrelation structure of `log(VIX²)` to estimate the mean-reversion rate α and noise parameter k, giving an effective vol of `σ_eff = m · exp(k² / 4α)`.

## Custom Strategies

Select "Custom (User Code)" and paste a function with the signature:

```python
def run_strategy(price_data, vix_data, rf_data, params):
    # price_data, vix_data, rf_data: pd.Series aligned on trading dates
    # rf_data is the annualised rate (e.g. 0.05 for 5%)
    # return a pd.Series of NAV values indexed by date
```

Execution is sandboxed: only `pd`, `np`, and basic Python builtins are available. `import` is not accessible.

## Limitations

- **Flat vol surface.** Black-Scholes pricing ignores the volatility smile and term structure. All options at a given expiry use the same implied vol (VIX level).
- **VIX as implied vol proxy.** VIX reflects 30-day SPX implied vol; it is not the correct implied vol for arbitrary assets or maturities.
- **No bid-ask spread.** Transaction costs are a flat percentage of NAV, not a function of option liquidity or moneyness.
- **European exercise only.** Option payoffs are computed at the roll date expiry; early exercise and path-dependent behaviour are not modelled.
- **ExpOU calibration stability.** The OU parameters are refitted at each roll date on a trailing year of VIX data. In low-liquidity or regime-change periods the calibration may be unreliable; the fallback is `α=0.1, k=0.2, m=0.15`.
