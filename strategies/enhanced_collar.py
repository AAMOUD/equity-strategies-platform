"""Enhanced Collar Strategy Implementation"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from .base import BaseStrategy


class EnhancedCollarStrategy(BaseStrategy):
    """Downside protection with put spread and income from short calls."""
    
    def __init__(self):
        super().__init__("Enhanced Collar")
        self.default_params = {
            'maturity_days': 63,
            'k1_pct': 0.95,
            'k2_pct': 0.80,
            'kf_pct': 1.02,
            'transaction_cost': 0.0001
        }
    
    @staticmethod
    def bs_put(S, K, T, r, sigma, q=0.0):
        """Black-Scholes put option pricing."""
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    
    @staticmethod
    def bs_call(S, K, T, r, sigma, q=0.0):
        """Black-Scholes call option pricing."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return np.exp(-q*T)*S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def run_backtest(self, price_data, vix_data, rf_data, params):
        df = pd.concat([price_data, vix_data, rf_data], axis=1).dropna()
        df.columns = ['S', 'VIX', 'Rf']

        maturity_days = params.get('maturity_days', self.default_params['maturity_days'])
        k1_pct = params.get('k1_pct', self.default_params['k1_pct'])
        k2_pct = params.get('k2_pct', self.default_params['k2_pct'])
        kf_pct = params.get('kf_pct', self.default_params['kf_pct'])
        tx_cost = params.get('transaction_cost', self.default_params['transaction_cost'])

        T = maturity_days / 252

        nav_series = pd.Series(index=df.index, dtype='float64')
        nav = 100.0
        nav_series.iloc[0] = nav

        roll_dates = df.iloc[::maturity_days].index

        for i in range(len(roll_dates) - 1):
            t0 = roll_dates[i]
            t1 = roll_dates[i + 1]

            S0 = df.loc[t0, 'S']
            r = df.loc[t0, 'Rf']
            sigma = df.loc[t0, 'VIX'] / 100

            K1 = k1_pct * S0
            K2 = k2_pct * S0
            strike_call = kf_pct * S0

            try:
                put_spread_cost = self.bs_put(S0, K1, T, r, sigma) - self.bs_put(S0, K2, T, r, sigma)
                call_premium = self.bs_call(S0, strike_call, T, r, sigma)
                net_cost = max(put_spread_cost - call_premium, 0)
                nav -= net_cost
                nav *= (1 - tx_cost)
                has_position = True
            except Exception as e:
                logging.warning(f"[EnhancedCollar] pricing failed at {t0}: {e}")
                has_position = False

            period_prices = df.loc[t0:t1, 'S']
            for j in range(1, len(period_prices)):
                date = period_prices.index[j]
                S_prev = period_prices.iloc[j - 1]
                S_curr = period_prices.iloc[j]

                ret_stock = (S_curr - S_prev) / S_prev

                if date == t1 and has_position:
                    put_spread_payoff = max(K1 - S_curr, 0) - max(K2 - S_curr, 0)
                    call_payoff = max(S_curr - strike_call, 0)
                    ret_options = (put_spread_payoff - call_payoff) / S0
                    total_ret = ret_stock + ret_options - tx_cost
                else:
                    total_ret = ret_stock

                nav *= (1 + total_ret)
                nav_series.loc[date] = nav

        nav_series = nav_series.ffill().fillna(100.0)

        self.results = nav_series
        self.metrics = self.calculate_metrics(nav_series, df['Rf'])

        return nav_series, self.metrics
