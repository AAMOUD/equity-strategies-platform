"""Base Strategy Class"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):

    def __init__(self, name):
        self.name = name
        self.results = None
        self.metrics = None

    @abstractmethod
    def run_backtest(self, price_data, vix_data, rf_data, params):
        pass

    def calculate_metrics(self, nav_series, rf_data=None):
        returns = nav_series.pct_change(fill_method=None).dropna()

        ann_return = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)

        if rf_data is not None:
            rf_ann = float(rf_data.reindex(nav_series.index).ffill().mean())
        else:
            rf_ann = 0.0

        sharpe = (ann_return - rf_ann) / ann_vol if ann_vol > 0 else np.nan

        running_max = nav_series.expanding().max()
        drawdown = (nav_series - running_max) / running_max
        mdd = drawdown.min()

        return {
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': mdd,
            'Final NAV': nav_series.iloc[-1],
            'Total Return': (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
        }
