"""Backtesting engine"""
from strategies import (
    BuyWriteStrategy,
    EnhancedCollarStrategy,
    ForwardStartStrategy,
    VolTargetStrategy,
    ExpOUCollarStrategy
)


class BacktestEngine:

    STRATEGY_CLASSES = {
        'Buy-Write': BuyWriteStrategy,
        'Enhanced Collar': EnhancedCollarStrategy,
        'Forward-Start': ForwardStartStrategy,
        'Vol-Target': VolTargetStrategy,
        'ExpOU-Collar': ExpOUCollarStrategy,
    }

    @staticmethod
    def get_available_strategies():
        return list(BacktestEngine.STRATEGY_CLASSES.keys())

    @staticmethod
    def get_default_params(strategy_name):
        cls = BacktestEngine.STRATEGY_CLASSES.get(strategy_name)
        if cls:
            return cls().default_params
        return {}

    @staticmethod
    def run_backtest(strategy_name, price_data, vix_data, rf_data, params):
        cls = BacktestEngine.STRATEGY_CLASSES.get(strategy_name)
        if not cls:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = cls()

        nav_series, metrics = strategy.run_backtest(price_data, vix_data, rf_data, params)

        benchmark = 100 * price_data / price_data.iloc[0]
        benchmark_metrics = {
            'Annualized Return': ((price_data.iloc[-1] / price_data.iloc[0]) ** (252 / len(price_data)) - 1),
            'Final NAV': benchmark.iloc[-1]
        }

        return nav_series, metrics, benchmark, benchmark_metrics

    @staticmethod
    def run_comparison(price_data, vix_data, rf_data, params):
        """
        Runs Enhanced Collar (BS) vs ExpOU-Collar side by side.
        Returns both NAV series + the pricing diff at each roll date.
        """
        bs_strat = EnhancedCollarStrategy()
        ou_strat = ExpOUCollarStrategy()

        nav_bs, metrics_bs = bs_strat.run_backtest(price_data, vix_data, rf_data, params)
        nav_ou, metrics_ou = ou_strat.run_backtest(price_data, vix_data, rf_data, params)

        nav_diff  = nav_ou - nav_bs
        cum_alpha = (nav_ou / nav_bs - 1) * 100

        return {
            'nav_bs':     nav_bs,
            'nav_ou':     nav_ou,
            'metrics_bs': metrics_bs,
            'metrics_ou': metrics_ou,
            'nav_diff':   nav_diff,
            'cum_alpha':  cum_alpha,
        }
