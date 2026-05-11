"""Walk-forward validation and parameter sensitivity."""
import numpy as np
import pandas as pd
from config import PARAM_BOUNDS


def walk_forward(engine, strategy_name, price_data, vix_data, rf_data,
                 params, n_splits=5):
    """
    Splits the backtest period into n_splits folds.
    Runs on the OOS slice (last 30% of each fold) with fixed params.
    Returns OOS Sharpe, return, and max drawdown per fold.
    """
    n = len(price_data)
    fold_size = n // n_splits
    results = []

    for i in range(n_splits):
        start = i * fold_size
        end   = start + fold_size
        split = start + int(0.7 * fold_size)

        p_test = price_data.iloc[split:end]
        v_test = vix_data.iloc[split:end]
        r_test = rf_data.iloc[split:end]

        try:
            _, metrics, _, _ = engine.run_backtest(
                strategy_name, p_test, v_test, r_test, params
            )
            results.append({
                'fold':       i + 1,
                'oos_start':  p_test.index[0],
                'oos_end':    p_test.index[-1],
                'oos_sharpe': metrics['Sharpe Ratio'],
                'oos_return': metrics['Annualized Return'],
                'oos_mdd':    metrics['Max Drawdown'],
            })
        except Exception:
            results.append({
                'fold': i + 1,
                'oos_start': p_test.index[0],
                'oos_end':   p_test.index[-1],
                'oos_sharpe': np.nan,
                'oos_return': np.nan,
                'oos_mdd':    np.nan,
            })

    return pd.DataFrame(results)


def sensitivity_grid(engine, strategy_name, price_data, vix_data, rf_data,
                     base_params, param1, param2, n_steps=8):
    """
    Grid search over (param1, param2). Returns a Sharpe heatmap as a DataFrame
    indexed by param1 values with param2 values as columns.
    """
    bounds = PARAM_BOUNDS.get(strategy_name, {})
    if param1 not in bounds or param2 not in bounds:
        raise ValueError(f"Params '{param1}', '{param2}' not in PARAM_BOUNDS['{strategy_name}']")

    p1_min, p1_max, _ = bounds[param1]
    p2_min, p2_max, _ = bounds[param2]

    p1_vals = np.linspace(p1_min, p1_max, n_steps)
    p2_vals = np.linspace(p2_min, p2_max, n_steps)

    sharpe_matrix = np.full((n_steps, n_steps), np.nan)

    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            run_params = {**base_params,
                          param1: round(float(v1), 6),
                          param2: round(float(v2), 6)}
            try:
                _, metrics, _, _ = engine.run_backtest(
                    strategy_name, price_data, vix_data, rf_data, run_params
                )
                sharpe_matrix[i, j] = metrics['Sharpe Ratio']
            except Exception:
                pass

    return pd.DataFrame(
        sharpe_matrix,
        index=np.round(p1_vals, 4),
        columns=np.round(p2_vals, 4),
    )
