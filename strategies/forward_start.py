"""Forward-Start Call Strategy Implementation"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from .base import BaseStrategy


class ForwardStartStrategy(BaseStrategy):
    """Income generation using forward-start call options."""
    
    def __init__(self):
        super().__init__("Forward-Start")
        self.default_params = {
            'forward_start_days': 21,
            'maturity_days': 63,
            'strike_mult': 1.02,
            'transaction_cost': 0.001
        }
    
    @staticmethod
    def price_forward_start_call(S0, KF, T0, T, r, sigma):
        """Price forward-start call using Rubinstein (1991) formula."""
        tau = T - T0
        if tau <= 0:
            raise ValueError("T must be greater than T0")
        
        d1 = (np.log(1/KF) + (0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        
        bs_price = norm.cdf(d1) - KF * np.exp(-r * tau) * norm.cdf(d2)
        return S0 * bs_price
    
    def run_backtest(self, price_data, vix_data, rf_data, params):
        df = pd.concat([price_data, vix_data, rf_data], axis=1).dropna()
        df.columns = ['S', 'VIX', 'Rf']
        
        forward_days = params.get('forward_start_days', self.default_params['forward_start_days'])
        maturity_days = params.get('maturity_days', self.default_params['maturity_days'])
        strike_mult = params.get('strike_mult', self.default_params['strike_mult'])
        tx_cost = params.get('transaction_cost', self.default_params['transaction_cost'])
        
        roll_dates = df.iloc[::maturity_days].index
        
        nav = 100.0
        nav_series = pd.Series(index=df.index, dtype='float64')
        nav_series.iloc[0] = nav
        
        for i in range(len(roll_dates) - 1):
            t0 = roll_dates[i]
            t1 = roll_dates[i + 1]
            
            S0 = df.loc[t0, 'S']
            r = df.loc[t0, 'Rf']
            sigma = df.loc[t0, 'VIX'] / 100
            
            T0 = forward_days / 252
            T = maturity_days / 252
            
            try:
                premium = self.price_forward_start_call(S0, strike_mult, T0, T, r, sigma)
            except Exception as e:
                logging.warning(f"[ForwardStart] pricing failed at {t0}: {e}")
                premium = 0
            
            period_prices = df.loc[t0:t1, 'S']
            fwd_idx = min(forward_days, len(period_prices) - 1)
            S_forward = period_prices.iloc[fwd_idx]
            strike = strike_mult * S_forward

            for j in range(1, len(period_prices)):
                date = period_prices.index[j]
                S_prev = period_prices.iloc[j-1]
                S_curr = period_prices.iloc[j]

                ret_stock = (S_curr - S_prev) / S_prev

                if date == t1:
                    payoff_call = max(S_curr - strike, 0)
                    ret_call = (premium - payoff_call) / S0
                    total_ret = ret_stock + ret_call - tx_cost
                else:
                    total_ret = ret_stock
                
                nav *= (1 + total_ret)
                nav_series.loc[date] = nav
        
        nav_series = nav_series.ffill().fillna(100.0)
        
        self.results = nav_series
        self.metrics = self.calculate_metrics(nav_series, df['Rf'])
        
        return nav_series, self.metrics
