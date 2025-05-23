import yfinance as yf
import pandas as pd

def fetch_data(ticker: str = "AAPL", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker using yfinance.
    Returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
    """
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data