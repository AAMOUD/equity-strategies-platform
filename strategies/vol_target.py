"""Volatility Targeting Strategy Implementation"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class VolTargetStrategy(BaseStrategy):
    """Dynamically adjust exposure to target a specific volatility level."""
    
    def __init__(self):
        super().__init__("Vol-Target")
        self.default_params = {
            'target_vol': 0.10,
            'lookback_days': 21,
            'rebalance_freq': 5,
            'transaction_cost': 0.001
        }
    
    def run_backtest(self, price_data, vix_data, rf_data, params):
        """Execute volatility targeting strategy backtest."""
        df = pd.concat([price_data, vix_data, rf_data], axis=1).dropna()
        df.columns = ['S', 'VIX', 'Rf']
        
        target_vol = params.get('target_vol', self.default_params['target_vol'])
        lookback = params.get('lookback_days', self.default_params['lookback_days'])
        rebal_freq = params.get('rebalance_freq', self.default_params['rebalance_freq'])
        tx_cost = params.get('transaction_cost', self.default_params['transaction_cost'])
        
        nav = 100.0
        nav_series = pd.Series(index=df.index, dtype='float64')
        nav_series.iloc[0] = nav
        
        exposure = 1.0
        
        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i-1]
            
            S_prev = df.loc[prev_date, 'S']
            S_curr = df.loc[date, 'S']
            ret_stock = (S_curr - S_prev) / S_prev
            
            nav *= (1 + exposure * ret_stock)
            
            if i % rebal_freq == 0 and i >= lookback:
                returns_window = df['S'].iloc[i-lookback:i].pct_change().dropna()
                realized_vol = returns_window.std() * np.sqrt(252)
                
                if realized_vol > 0.01:
                    new_exposure = target_vol / realized_vol
                    new_exposure = np.clip(new_exposure, 0.0, 2.0)
                    
                    if abs(new_exposure - exposure) > 0.05:
                        nav *= (1 - tx_cost * abs(new_exposure - exposure))
                        exposure = new_exposure
            
            nav_series.loc[date] = nav
        
        nav_series = nav_series.ffill().fillna(100.0)
        
        self.results = nav_series
        self.metrics = self.calculate_metrics(nav_series)
        
        return nav_series, self.metrics
