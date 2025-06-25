# strategies/heikin_ashi/heikin_ashi_baseline.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_data(ticker: str,
               period: str = "2y",
               interval: str = "1d") -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker.
    """
    return yf.download(ticker, period=period, interval=interval).dropna()


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with ['Open','High','Low','Close'],
    compute Heikin-Ashi columns in-place and return the new DataFrame.
    """
    df = df.copy()
    # HA_Close = average of O/H/L/C
    df['HA_Close'] = df[['Open','High','Low','Close']].sum(axis=1) / 4

    # HA_Open: first bar = avg(Open, Close); thereafter = avg(prev HA_Open, prev HA_Close)
    first_ha_open = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2

    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = first_ha_open
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + df['HA_Close'].iloc[i-1]) / 2

    df['HA_Open'] = ha_open


    # HA_High = max(High, HA_Open, HA_Close); HA_Low = min(Low, HA_Open, HA_Close)
    df['HA_High'] = df[['High','HA_Open','HA_Close']].max(axis=1)
    df['HA_Low']  = df[['Low','HA_Open','HA_Close']].min(axis=1)

    return df


def backtest_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals and backtest a simple HA strategy:
      - long if HA_Close > HA_Open, flat otherwise
    Returns DataFrame with ['Return','Cumulative'] added.
    """
    df = compute_heikin_ashi(df)
    # signal: long (1) when HA_Close > HA_Open, else 0
    df['Position'] = np.where(df['HA_Close'] > df['HA_Open'], 1, 0)

    # daily returns (keep .squeeze())
    df['Return'] = df['Close'].pct_change().squeeze() * df['Position'].shift().fillna(0)
    df['Return'].fillna(0, inplace=True)

    # cumulative return
    df['Cumulative'] = (1 + df['Return']).cumprod()
    return df


def plot_performance(df: pd.DataFrame, ticker: str):
    """
    Plot strategy cumulative return vs buy & hold.
    """
    cum_strat = df['Cumulative']
    cum_bh    = (1 + df['Close'].pct_change().fillna(0)).cumprod()

    plt.figure(figsize=(10,6))
    plt.plot(cum_strat, label='HA Strategy')
    plt.plot(cum_bh,    '--', label='Buy & Hold')
    plt.title(f'{ticker} Heikin-Ashi Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # demo run
    ticker = "AAPL"
    df     = fetch_data(ticker)
    df_bt  = backtest_heikin_ashi(df)

    sharpe = df_bt['Return'].mean() / df_bt['Return'].std(ddof=0) * np.sqrt(252)
    maxdd  = (df_bt['Cumulative'].cummax() - df_bt['Cumulative']).max()

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {maxdd:.2%}")

    plot_performance(df_bt, ticker)
