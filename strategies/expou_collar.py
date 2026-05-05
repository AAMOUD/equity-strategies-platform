"""Exponential OU Collar Strategy Implementation"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from .base import BaseStrategy


class ExpOUCollarStrategy(BaseStrategy):
    """Enhanced collar using exponential OU volatility model for put spread pricing."""
    
    def __init__(self):
        super().__init__("ExpOU-Collar")
        self.default_params = {
            'maturity_days': 63,
            'k1_pct': 0.95,
            'k2_pct': 0.80,
            'kf_pct': 1.02,
            'transaction_cost': 0.0001
        }
    
    @staticmethod
    def bs_put(S, K, T, r, sigma, q=0.0):
        """Black-Scholes put pricing."""
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    @staticmethod
    def bs_call(S, K, T, r, sigma, q=0.0):
        """Black-Scholes call pricing."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return np.exp(-q*T)*S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def calibrate_expOU_from_VIX(vix_series, lags=20):
        """Calibrate exponential OU model from VIX data."""
        try:
            log_vix2 = np.log((vix_series / 100)**2).dropna()
            if len(log_vix2) < lags:
                return {'alpha': 0.1, 'k': 0.2, 'm': 0.15}
            
            autocorrs = [log_vix2.autocorr(lag) for lag in range(1, lags + 1)]
            
            def ou_autocorr(alpha, lag):
                return np.exp(-alpha * lag)
            
            def cost(params):
                alpha = params[0]
                return sum((ou_autocorr(alpha, lag) - autocorrs[lag - 1])**2 
                          for lag in range(1, lags + 1))
            
            res = minimize(cost, x0=[0.1], bounds=[(0.001, 2.0)])
            alpha = res.x[0]
            k = np.sqrt(2 * alpha * np.var(log_vix2))
            m = np.exp(np.mean(log_vix2))
            
            return {'alpha': alpha, 'k': k, 'm': m}
        except:
            return {'alpha': 0.1, 'k': 0.2, 'm': 0.15}
    
    @staticmethod
    def price_put_spread_expOU(S, K1, K2, T, r, m, alpha, k):
        """Price put spread using exponential OU model."""
        if alpha <= 0 or m <= 0 or k <= 0:
            sigma_eff = 0.2
        else:
            sigma_eff = m * np.exp(k**2 / (4 * alpha))
        
        P1 = ExpOUCollarStrategy.bs_put(S, K1, T, r, sigma_eff)
        P2 = ExpOUCollarStrategy.bs_put(S, K2, T, r, sigma_eff)
        return P2 - P1
    
    def run_backtest(self, price_data, vix_data, rf_data, params):
        """Execute ExpOU collar strategy backtest."""
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
        
        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i-1]
            
            S0 = df.loc[prev_date, 'S']
            ST = df.loc[date, 'S']
            r = df.loc[prev_date, 'Rf']
            sigma = df.loc[prev_date, 'VIX'] / 100
            
            ret_stock = (ST - S0) / S0
            nav *= (1 + ret_stock)
            
            if date in roll_dates and date != df.index[0]:
                try:
                    vix_window = df.loc[:date, 'VIX'].tail(252)
                    calib = self.calibrate_expOU_from_VIX(vix_window)
                    
                    K1 = k1_pct * ST
                    K2 = k2_pct * ST
                    put_spread_cost = self.price_put_spread_expOU(
                        ST, K1, K2, T, r, calib['m'], calib['alpha'], calib['k']
                    )
                    
                    strike_call = kf_pct * ST
                    call_premium = self.bs_call(ST, strike_call, T, r, sigma)
                    
                    net_cost = max(put_spread_cost - call_premium, 0)
                    nav -= net_cost
                    nav *= (1 - tx_cost)
                
                except:
                    pass
            
            nav_series.loc[date] = nav
        
        nav_series = nav_series.ffill().fillna(100.0)
        
        self.results = nav_series
        self.metrics = self.calculate_metrics(nav_series)
        
        return nav_series, self.metrics
