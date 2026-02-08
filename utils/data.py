import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataFetcher:
    
    TICKER_MAP = {
        'S&P 500': '^GSPC',
        'Nasdaq 100': '^NDX',
        'Russell 2000': '^RUT',
        'Dow Jones': '^DJI',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'Nikkei 225': '^N225',
        'Euro Stoxx 50': '^STOXX50E',
        'Hang Seng': '^HSI',
        'CAC 40': '^FCHI',
        'S&P 500 Equal Weight': '^SPXEW',
        'S&P 400 MidCap': '^MID',
        'S&P 500 Low Volatility': '^SPLV',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'Amazon': 'AMZN',
        'Google': 'GOOGL',
        'Meta': 'META',
        'NVIDIA': 'NVDA',
        'Alphabet (Class C)': 'GOOG',
        'Berkshire Hathaway B': 'BRK-B',
        'JPMorgan Chase': 'JPM',
        'Johnson & Johnson': 'JNJ',
        'Procter & Gamble': 'PG',
        'Visa': 'V',
        'Mastercard': 'MA',
        'Coca-Cola': 'KO',
        'PepsiCo': 'PEP',
        'Exxon Mobil': 'XOM',
        'Chevron': 'CVX',
        'Walmart': 'WMT',
        'Costco': 'COST',
        'Netflix': 'NFLX',
        'Adobe': 'ADBE',
        'Salesforce': 'CRM',
        'Intel': 'INTC',
        'AMD': 'AMD',
        'Cisco': 'CSCO',
        'Pfizer': 'PFE',
        'UnitedHealth': 'UNH',
        'Boeing': 'BA',
        'Gold': 'GC=F',
        'Oil (WTI)': 'CL=F',
        'US Dollar Index': 'DX-Y.NYB',
        'US 10Y Yield': '^TNX',
    }
    
    @staticmethod
    def get_available_assets():
        return list(DataFetcher.TICKER_MAP.keys())
    
    @staticmethod
    def fetch_data(asset_name, start_date, end_date):
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
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > datetime.now():
            end_date = datetime.now()
        
        return start_date, end_date
