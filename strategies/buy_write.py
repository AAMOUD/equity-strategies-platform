"""Buy-Write Strategy Implementation"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .base import BaseStrategy


class BuyWriteStrategy(BaseStrategy):
    """Income generation by selling call options against long stock positions."""
    
    def __init__(self):
        super().__init__("Buy-Write")
        self.default_params = {
            'maturity_days': 21,
            'strike_offset': 0.02,
            'transaction_cost': 0.001
        }
    
    @staticmethod
    def bs_call(S, K, T, r, sigma, q=0.0):
        """Black-Scholes call pricing."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return np.exp(-q*T)*S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def run_backtest(self, price_data, vix_data, rf_data, params):
        """Execute buy-write strategy backtest."""
        df = pd.concat([price_data, vix_data, rf_data], axis=1).dropna()
        df.columns = ['S', 'VIX', 'Rf']
        
        maturity_days = params.get('maturity_days', self.default_params['maturity_days'])
        strike_offset = params.get('strike_offset', self.default_params['strike_offset'])
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
            T = (t1 - t0).days / 252
            strike = S0 * (1 + strike_offset)
            
            premium = self.bs_call(S0, strike, T, r, sigma)
            
            period_prices = df.loc[t0:t1, 'S']
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
        self.metrics = self.calculate_metrics(nav_series)
        
        return nav_series, self.metrics
