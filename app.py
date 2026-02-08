"""Equity Strategies Backtesting Platform"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from utils import DataFetcher
from backtest import BacktestEngine

st.set_page_config(
    page_title="Equity Strategies Platform",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #10b981;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .metric-delta-positive {
        color: #10b981;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .metric-delta-negative {
        color: #ef4444;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        background-color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    
    .info-card h3 {
        color: #1e293b;
        margin-top: 0;
    }
    
    .section-header {
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e5e7eb;
    }
    
    .section-header h2 {
        color: #1f2937;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


def format_metric(value, format_type='number'):
    if isinstance(value, pd.Series):
        value = value.iloc[0] if len(value) > 0 else np.nan
    
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    try:
        if format_type == 'percentage':
            return f"{float(value):.2%}"
        elif format_type == 'currency':
            return f"${float(value):,.2f}"
        elif format_type == 'number':
            return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"
    
    return str(value)


def create_metric_card(label, value, delta=None, delta_positive=False):
    delta_html = ""
    if delta is not None:
        delta_class = "metric-delta-positive" if delta_positive else "metric-delta-negative"
        delta_symbol = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{delta_class}">{delta_symbol} {abs(delta):.2f}%</div>'
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    return card_html


def safe_to_float(val, default=None):
    try:
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else default
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except:
        return default


def main():
    st.markdown("""
    <div class="main-header">
        <h1>Equity Strategies Platform</h1>
        <p>Advanced Backtesting Platform with Professional Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Configuration Panel")
        
        st.markdown("#### Strategy Selection")
        strategy = st.selectbox(
            "Choose Strategy",
            BacktestEngine.get_available_strategies(),
            help="Select the trading strategy to backtest"
        )
        
        st.markdown("#### Asset Selection")
        asset = st.selectbox(
            "Choose Asset",
            DataFetcher.get_available_assets(),
            help="Select the underlying asset"
        )
        
        st.markdown("#### Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=datetime.now() - timedelta(days=365*5),
                help="Backtest start date"
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=datetime.now(),
                help="Backtest end date"
            )
        
        st.markdown("#### Parameters")
        
        default_params = BacktestEngine.get_default_params(strategy)
        params = {}
        
        for param_name, default_value in default_params.items():
            if isinstance(default_value, int):
                params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=int(default_value * 0.5),
                    max_value=int(default_value * 2),
                    value=default_value
                )
            elif isinstance(default_value, float):
                params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=float(default_value * 0.5),
                    max_value=float(default_value * 2),
                    value=default_value,
                    step=0.001
                )
        
        st.markdown("---")
        run_button = st.button("Run Backtest", use_container_width=True, type="primary", key="run_backtest_btn")
    
    if run_button:
        with st.spinner("Running backtest..."):
            try:
                price_data, vix_data, rf_data = DataFetcher.fetch_data(
                    asset, start_date, end_date
                )
                
                nav_series, metrics, benchmark, benchmark_metrics = BacktestEngine.run_backtest(
                    strategy, price_data, vix_data, rf_data, params
                )
                if benchmark is None or len(benchmark) == 0:
                    benchmark = 100 * price_data / price_data.iloc[0]
                if isinstance(benchmark, pd.DataFrame):
                    if 'Close' in benchmark.columns:
                        benchmark = benchmark['Close']
                    else:
                        benchmark = benchmark.iloc[:, 0]
                benchmark = benchmark.reindex(nav_series.index)
                benchmark = pd.to_numeric(benchmark, errors='coerce').ffill().bfill()
                benchmark_metrics = dict(benchmark_metrics)
                if len(benchmark) > 1:
                    benchmark_metrics['Final NAV'] = float(benchmark.iloc[-1])
                    benchmark_metrics['Annualized Return'] = (
                        (benchmark.iloc[-1] / benchmark.iloc[0]) ** (252 / max(len(benchmark) - 1, 1)) - 1
                    )
                
                st.success("Backtest completed successfully!")
                
                st.markdown('<div class="section-header"><h2>Performance Summary</h2></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    final_nav = safe_to_float(metrics['Final NAV'], 100)
                    nav_delta = final_nav - 100
                    st.markdown(create_metric_card(
                        "Final Portfolio Value",
                        f"${final_nav:,.2f}",
                        nav_delta,
                        nav_delta >= 0
                    ), unsafe_allow_html=True)
                
                with col2:
                    total_ret = safe_to_float(metrics['Total Return'], 0)
                    st.markdown(create_metric_card(
                        "Total Return",
                        f"{total_ret:.2%}",
                        None
                    ), unsafe_allow_html=True)
                
                with col3:
                    ann_ret = safe_to_float(metrics['Annualized Return'], 0)
                    st.markdown(create_metric_card(
                        "Annualized Return",
                        f"{ann_ret:.2%}",
                        None
                    ), unsafe_allow_html=True)
                
                with col4:
                    sharpe_val = safe_to_float(metrics['Sharpe Ratio'], None)
                    sharpe_value = f"{sharpe_val:.2f}" if sharpe_val is not None else "N/A"
                    st.markdown(create_metric_card(
                        "Sharpe Ratio",
                        sharpe_value,
                        None
                    ), unsafe_allow_html=True)
                
                st.markdown("---")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    ann_vol = safe_to_float(metrics['Annualized Volatility'], 0)
                    st.markdown(create_metric_card(
                        "Annualized Volatility",
                        f"{ann_vol:.2%}",
                        None
                    ), unsafe_allow_html=True)
                
                with col6:
                    max_dd = safe_to_float(metrics['Max Drawdown'], 0)
                    st.markdown(create_metric_card(
                        "Maximum Drawdown",
                        f"{max_dd:.2%}",
                        None
                    ), unsafe_allow_html=True)
                
                with col7:
                    final_nav_strat = safe_to_float(metrics['Final NAV'], 100)
                    final_nav_bench = safe_to_float(benchmark_metrics.get('Final NAV'), 100)
                    outperformance = final_nav_strat - final_nav_bench
                    st.markdown(create_metric_card(
                        "vs Buy & Hold",
                        f"{outperformance:+.2f}",
                        outperformance,
                        outperformance >= 0
                    ), unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown('<div class="section-header"><h2>Performance Analysis</h2></div>', unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "NAV Comparison", 
                    "Drawdown Analysis", 
                    "Monthly Returns",
                    "Detailed Metrics"
                ])
                
                with tab1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=nav_series.index,
                        y=nav_series.values,
                        name=strategy,
                        line=dict(color='#667eea', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.1)'
                    ))
                    fig.add_trace(go.Scatter(
                        x=benchmark.index,
                        y=benchmark.values,
                        name="Buy & Hold",
                        line=dict(color='#f59e0b', width=2, dash='dash')
                    ))
                    fig.update_layout(
                        title="Portfolio Growth: Strategy vs Buy & Hold",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        hovermode='x unified',
                        height=550,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    cumulative = nav_series / 100
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values,
                        fill='tozeroy',
                        name='Drawdown',
                        line=dict(color='#ef4444', width=2),
                        fillcolor='rgba(239, 68, 68, 0.3)'
                    ))
                    fig.update_layout(
                        title="Drawdown Analysis - Risk Assessment",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        hovermode='x unified',
                        height=550,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12)
                    )
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Drawdown", f"{drawdown.min():.2f}%")
                    with col2:
                        st.metric("Avg Drawdown", f"{drawdown.mean():.2f}%")
                    with col3:
                        underwater = (drawdown < -5).sum() / len(drawdown) * 100
                        st.metric("Time in Drawdown >5%", f"{underwater:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    returns = nav_series.pct_change()
                    returns_monthly = returns.resample('ME').sum()
                    
                    returns_monthly.index = returns_monthly.index.strftime('%Y-%m')
                    colors = ['#10b981' if x > 0 else '#ef4444' for x in returns_monthly.values]
                    
                    fig = go.Figure(data=go.Bar(
                        x=returns_monthly.index,
                        y=returns_monthly.values * 100,
                        marker=dict(
                            color=colors,
                            line=dict(color='rgba(0,0,0,0.2)', width=1)
                        ),
                        text=[f"{x*100:.1f}%" for x in returns_monthly.values],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Monthly Returns Distribution",
                        xaxis_title="Month",
                        yaxis_title="Return (%)",
                        height=550,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        showlegend=False
                    )
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.2)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        positive_months = (returns_monthly.astype(float) > 0).sum()
                        st.metric("Positive Months", f"{positive_months}/{len(returns_monthly)}")
                    with col2:
                        win_rate = positive_months / len(returns_monthly) * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with col3:
                        best_month = returns_monthly.astype(float).max() * 100
                        st.metric("Best Month", f"+{best_month:.2f}%")
                    with col4:
                        worst_month = returns_monthly.astype(float).min() * 100
                        st.metric("Worst Month", f"{worst_month:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab4:
                    st.markdown('<div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.07);">', unsafe_allow_html=True)
                    st.subheader("Strategy vs Benchmark Comparison")
                    
                    fnav = safe_to_float(metrics['Final NAV'], 100)
                    tret = safe_to_float(metrics['Total Return'], 0)
                    aret = safe_to_float(metrics['Annualized Return'], 0)
                    avol = safe_to_float(metrics['Annualized Volatility'], 0)
                    shr = safe_to_float(metrics['Sharpe Ratio'], None)
                    mdd = safe_to_float(metrics['Max Drawdown'], 0)
                    
                    bnav = safe_to_float(benchmark_metrics.get('Final NAV'), 100)
                    baret = safe_to_float(benchmark_metrics.get('Annualized Return'), 0)
                    btret = safe_to_float(benchmark_metrics.get('Total Return'), (bnav / 100 - 1) if bnav else 0)
                    
                    comparison_data = {
                        'Metric': [
                            'Final Portfolio Value',
                            'Total Return',
                            'Annualized Return',
                            'Annualized Volatility',
                            'Sharpe Ratio',
                            'Max Drawdown'
                        ],
                        'Strategy': [
                            f"${fnav:,.2f}",
                            f"{tret:.2%}",
                            f"{aret:.2%}",
                            f"{avol:.2%}",
                            f"{shr:.3f}" if shr is not None else "N/A",
                            f"{mdd:.2%}"
                        ],
                        'Buy & Hold': [
                            f"${bnav:,.2f}",
                            f"{btret:.2%}",
                            f"{baret:.2%}",
                            "N/A",
                            "N/A",
                            "N/A"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown('<div class="section-header"><h2>Strategy Configuration</h2></div>', unsafe_allow_html=True)
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h3>Backtest Configuration</h3>
                        <p><strong>Strategy:</strong> {strategy}</p>
                        <p><strong>Asset:</strong> {asset}</p>
                        <p><strong>Period:</strong> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>
                        <p><strong>Data Points:</strong> {len(nav_series)} trading days</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_col2:
                    if params:
                        params_html = "<h3>Strategy Parameters</h3>"
                        for key, value in params.items():
                            params_html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                        st.markdown(f'<div class="info-card">{params_html}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown('<div class="section-header"><h2>Export Results</h2></div>', unsafe_allow_html=True)
                
                try:
                    export_data = {
                        'Date': nav_series.index.astype(str),
                        'Strategy NAV': nav_series.values.flatten(),
                        'Benchmark NAV': benchmark.values.flatten()
                    }
                    
                    strat_ret = nav_series.pct_change().fillna(0) * 100
                    bench_ret = benchmark.pct_change().fillna(0) * 100
                    
                    export_data['Strategy Return (%)'] = strat_ret.values.flatten()
                    export_data['Benchmark Return (%)'] = bench_ret.values.flatten()
                    
                    export_df = pd.DataFrame(export_data)
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"backtest_{strategy.replace(' ', '_')}_{asset.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.warning(f"Could not export results: {str(e)}")
            
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                st.info("Please check your inputs and try again.")
    
    else:
        st.markdown("""
        <div class="info-card">
            <h3>Welcome to Equity Strategies Platform</h3>
            <p>Your professional backtesting platform for equity options strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>Quick Start Guide</h3>
                <ol>
                    <li>Select a strategy from the sidebar</li>
                    <li>Choose an underlying asset</li>
                    <li>Set your preferred date range</li>
                    <li>Adjust strategy parameters</li>
                    <li>Click "Run Backtest" to analyze</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>Available Strategies</h3>
                <p><strong>Buy-Write:</strong> Generate income by selling calls against long stock positions</p>
                <p><strong>Enhanced Collar:</strong> Protect downside with puts while selling calls for income</p>
                <p><strong>Forward-Start:</strong> Income generation using forward-start call options</p>
                <p><strong>Vol-Target:</strong> Dynamically adjust exposure to maintain target volatility</p>
                <p><strong>ExpOU-Collar:</strong> Enhanced collar with exponential OU volatility model</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>Key Metrics Explained</h3>
                <p><strong>Sharpe Ratio:</strong> Risk-adjusted returns (higher is better, >1 is good)</p>
                <p><strong>Max Drawdown:</strong> Largest peak-to-trough decline</p>
                <p><strong>Annualized Return:</strong> Average yearly performance</p>
                <p><strong>Volatility:</strong> Strategy risk level (lower is more stable)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>Pro Tips</h3>
                <ul>
                    <li>Test strategies over different market conditions</li>
                    <li>Compare Sharpe ratios to assess risk-adjusted performance</li>
                    <li>Monitor drawdowns for risk management</li>
                    <li>Export results for deeper analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
