# strategies/heikin_ashi/heikin_ashi_revised.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .heikin_ashi_baseline import compute_heikin_ashi, plot_performance

def fetch_data(ticker: str,
               period: str = "2y",
               interval: str = "1d"
              ) -> pd.DataFrame:
    """
    Download and clean data, flattening any MultiIndex.
    """
    df = yf.download(ticker, period=period, interval=interval)
    # If columns are a MultiIndex, flatten to the first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def backtest_heikin_ashi_ma(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    ma_window: int = 50
) -> pd.DataFrame:
    """
    Revised Heikin-Ashi + MA50 strategy:
      - compute Heikin-Ashi candles
      - apply 50-day moving average filter
      - go long when HA_Close > HA_Open AND Close > MA50
      - shift signal into position for next-day execution
      - compute returns with .squeeze()
    """
    # 1) Prepare data
    df = fetch_data(ticker, period=period, interval=interval)

    # 2) Compute Heikin-Ashi candles
    df = compute_heikin_ashi(df)

    # 3) Apply 50-day moving average filter
    df['MA50'] = df['Close'].rolling(window=ma_window).mean()

    # 4) Generate trading signal
    df['Signal'] = 0
    mask = (df['HA_Close'] > df['HA_Open']) & (df['Close'] > df['MA50'])
    df.loc[mask, 'Signal'] = 1

    # 5) Shift signal into position for next-day execution
    df['Position'] = df['Signal'].shift().fillna(0).astype(int)

    # 6) Compute daily returns (retain .squeeze())
    df['Return'] = df['Close'].pct_change().squeeze() * df['Position'].shift().fillna(0)
    df['Return'].fillna(0, inplace=True)

    # 7) Compute cumulative returns
    df['Cumulative'] = (1 + df['Return']).cumprod()

    return df

def main():
    ticker = "AAPL"
    df_bt  = backtest_heikin_ashi_ma(ticker, ma_window=50)

    sharpe = df_bt['Return'].mean() / df_bt['Return'].std(ddof=0) * np.sqrt(252)
    mdd    = (df_bt['Cumulative'].cummax() - df_bt['Cumulative']).max()
    print(f"Sharpe: {sharpe:.2f}, Max Drawdown: {mdd:.2%}")

    plot_performance(df_bt, ticker)

if __name__ == "__main__":
    main()
