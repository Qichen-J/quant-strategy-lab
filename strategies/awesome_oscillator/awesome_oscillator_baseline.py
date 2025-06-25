# strategies/awesome_oscillator/awesome_oscillator_baseline.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_data(ticker: str,
               period: str = "2y",
               interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data from yfinance and drop NaNs.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def compute_ao(df: pd.DataFrame,
               fast_window: int = 5,
               slow_window: int = 34) -> pd.DataFrame:
    """
    Compute the Awesome Oscillator (AO) components:
      - median_price = (High + Low) / 2
      - ao_fast = SMA(median_price, fast_window)
      - ao_slow = SMA(median_price, slow_window)
      - AO = ao_fast - ao_slow
    """
    df = df.copy()
    df['median_price'] = (df['High'] + df['Low']) / 2
    df['ao_fast'] = df['median_price'].rolling(window=fast_window).mean()
    df['ao_slow'] = df['median_price'].rolling(window=slow_window).mean()
    df['AO']      = df['ao_fast'] - df['ao_slow']
    df.dropna(inplace=True)
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on AO crossing zero:
      - signal =  1 when AO crosses above zero
      - signal = -1 when AO crosses below zero
    """
    df = df.copy()
    df['AO_prev'] = df['AO'].shift(1)
    df['signal']  = 0
    df.loc[(df['AO'] > 0) & (df['AO_prev'] <= 0), 'signal'] =  1
    df.loc[(df['AO'] < 0) & (df['AO_prev'] >= 0), 'signal'] = -1
    return df

def backtest_ao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest AO strategy:
      - position = forward-filled signal, applied to next day's return
      - strat_ret = position * daily_return
      - equity_curve and buy_hold_curve
      - drawdown metrics
    """
    df = df.copy()
    # carry forward last non-zero signal and shift for next-day execution
    df['position'] = df['signal'].replace(0, method='ffill').shift(1).fillna(0)

    # daily returns
    df['returns']   = df['Close'].pct_change()
    df['strat_ret'] = df['position'] * df['returns']
    df.dropna(inplace=True)

    # equity curves
    df['equity_curve']   = (1 + df['strat_ret']).cumprod()
    df['buy_hold_curve'] = (1 + df['returns']).cumprod()

    # performance metrics
    sharpe_ratio = df['strat_ret'].mean() / df['strat_ret'].std(ddof=0) * np.sqrt(252)
    df['sharpe_ratio'] = sharpe_ratio
    df['cum_max']      = df['equity_curve'].cummax()
    df['drawdown']     = df['equity_curve'] / df['cum_max'] - 1
    df['max_drawdown'] = df['drawdown'].min()

    return df

def plot_performance(df: pd.DataFrame, ticker: str):
    """
    Plot strategy equity curve vs. buy-and-hold benchmark.
    """
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['buy_hold_curve'], label='Buy & Hold Equity Curve')
    plt.plot(df.index, df['equity_curve'],    label='AO Strategy Equity Curve')
    sharpe = df['sharpe_ratio'].iloc[0]
    maxdd  = df['max_drawdown'].iloc[0]
    plt.title(
        f"{ticker} AO Strategy Backtest "
        f"(Sharpe={sharpe:.2f}, MaxDD={maxdd:.2%})"
    )
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # demo run
    ticker = "AAPL"
    df     = fetch_data(ticker)
    df     = compute_ao(df)
    df     = generate_signals(df)
    df_bt  = backtest_ao(df)

    # extract scalar metrics
    sharpe = df_bt["sharpe_ratio"].iloc[0]
    maxdd  = df_bt["max_drawdown"].iloc[0]

    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown:         {maxdd:.2%}")

    plot_performance(df_bt, ticker)