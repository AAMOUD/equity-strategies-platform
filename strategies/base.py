"""Base Strategy Class"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):

    def __init__(self, name):
        self.name = name
        self.results = None
        self.metrics = None
        self.greeks_history = {}

    @abstractmethod
    def run_backtest(self, price_data, vix_data, rf_data, params):
        pass

    @staticmethod
    def option_transaction_cost(vega, spread_factor=0.10):
        """
        Bid-ask spread proportional to option vega.
        spread_factor: fraction of vega as proxy for spread (0.05-0.15 for liquid index options).
        """
        return spread_factor * abs(vega)

    @staticmethod
    def compute_greeks(bs_func, S, K, T, r, sigma, bump=0.01):
        """Bump-and-reprice Greeks. bs_func: callable(S, K, T, r, sigma) -> price."""
        if T <= 0 or sigma <= 0:
            return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan}

        price     = bs_func(S, K, T, r, sigma)
        price_up  = bs_func(S * (1 + bump), K, T, r, sigma)
        price_dn  = bs_func(S * (1 - bump), K, T, r, sigma)
        price_vup = bs_func(S, K, T, r, sigma + 0.01)
        price_tup = bs_func(S, K, T - 1/252, r, sigma)

        delta = (price_up - price_dn) / (2 * bump * S)
        gamma = (price_up - 2 * price + price_dn) / (bump * S) ** 2
        vega  = (price_vup - price) / 0.01
        theta = (price_tup - price) / (1/252)

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

    @staticmethod
    def drawdown_series(nav_series):
        running_max = nav_series.expanding().max()
        return (nav_series - running_max) / running_max

    @staticmethod
    def calculate_metrics(nav_series, rf_data=None):
        returns = nav_series.pct_change(fill_method=None).dropna()

        if len(returns) == 0:
            return {
                'Annualized Return': np.nan,
                'Annualized Volatility': np.nan,
                'Sharpe Ratio': np.nan,
                'Max Drawdown': np.nan,
                'Final NAV': nav_series.iloc[-1] if len(nav_series) else np.nan,
                'Total Return': np.nan,
            }

        ann_return = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)

        if rf_data is not None:
            rf_ann = float(rf_data.reindex(nav_series.index).ffill().mean())
        else:
            rf_ann = 0.0

        sharpe = (ann_return - rf_ann) / ann_vol if ann_vol > 0 else np.nan

        drawdown = BaseStrategy.drawdown_series(nav_series)
        mdd = drawdown.min()

        return {
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': mdd,
            'Final NAV': nav_series.iloc[-1],
            'Total Return': (nav_series.iloc[-1] / nav_series.iloc[0]) - 1,
        }
