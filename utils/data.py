"""Data utilities for fetching market data"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch market data from Yahoo Finance."""
    
    TICKER_MAP = {
        'S&P 500': '^GSPC',
        'Nasdaq 100': '^NDX',
        'Russell 2000': '^RUT',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'Amazon': 'AMZN',
        'Google': 'GOOGL',
    }
    
    @staticmethod
    def get_available_assets():
        """Return list of available assets."""
        return list(DataFetcher.TICKER_MAP.keys())
    
    @staticmethod
    def fetch_data(asset_name, start_date, end_date):
        """Fetch price, VIX, and risk-free rate data."""
        ticker = DataFetcher.TICKER_MAP.get(asset_name, asset_name)
        
        try:
            price_data = yf.download(ticker, start=start_date, end=end_date, 
                                     auto_adjust=True, progress=False)['Close']
            price_data = price_data.dropna()
            
            vix_data = yf.download('^VIX', start=start_date, end=end_date,
                                   auto_adjust=True, progress=False)['Close']
            vix_data = vix_data.dropna()
            
            rf_data = yf.download('^IRX', start=start_date, end=end_date,
                                  auto_adjust=True, progress=False)['Close']
            rf_data = (rf_data / 10000).dropna()
            
            common_idx = price_data.index.intersection(vix_data.index).intersection(rf_data.index)
            
            return (price_data.loc[common_idx], 
                    vix_data.loc[common_idx], 
                    rf_data.loc[common_idx])
        
        except Exception as e:
            raise ValueError(f"Error fetching data for {asset_name}: {str(e)}")
    
    @staticmethod
    def validate_date_range(start_date, end_date):
        """Validate and return date range."""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > datetime.now():
            end_date = datetime.now()
        
        return start_date, end_date
