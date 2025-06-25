# strategies/awesome_oscillator/awesome_oscillator_revised.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .awesome_oscillator_baseline import (
    fetch_data, compute_ao, generate_signals, plot_performance
)

def backtest_ao_sl_tp(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.05
) -> pd.DataFrame:
    """
    Revised AO strategy with stop-loss and take-profit.
      1) fetch and prepare data
      2) compute AO and generate entry/exit signals
      3) manage position with SL/TP in a daily loop using numpy arrays
      4) build equity curves and attach performance metrics
    """
    # 1) Download and prepare
    df = fetch_data(ticker, period=period, interval=interval)
    df = compute_ao(df)
    df = generate_signals(df)
    df = df.copy()
    df['strategy_return'] = 0.0

    # 2) Extract numpy arrays for fast, unambiguous indexing
    close_arr  = df['Close'].values
    open_arr   = df['Open'].values
    high_arr   = df['High'].values
    low_arr    = df['Low'].values
    signal_arr = df['signal'].values
    dates      = df.index

    position    = 0
    entry_price = np.nan

    # 3) Daily loop
    for i in range(1, len(df)):
        date       = dates[i]
        prev_close = close_arr[i - 1]
        today_open = open_arr[i]
        high_price = high_arr[i]
        low_price  = low_arr[i]
        sig        = int(signal_arr[i])

        # open new position
        if position == 0 and sig != 0:
            position    = sig
            entry_price = today_open
            if position == 1:
                ret = close_arr[i] / entry_price - 1
            else:
                ret = entry_price / close_arr[i] - 1
            df.at[date, 'strategy_return'] = ret

        # long position: apply SL/TP
        elif position == 1:
            sl = entry_price * (1 - stop_loss_pct)
            tp = entry_price * (1 + take_profit_pct)
            if low_price <= sl:
                df.at[date, 'strategy_return'] = sl / prev_close - 1
                position = 0
            elif high_price >= tp:
                df.at[date, 'strategy_return'] = tp / prev_close - 1
                position = 0
            else:
                df.at[date, 'strategy_return'] = close_arr[i] / prev_close - 1

        # short position: apply SL/TP
        elif position == -1:
            sl = entry_price * (1 + stop_loss_pct)
            tp = entry_price * (1 - take_profit_pct)
            if high_price >= sl:
                df.at[date, 'strategy_return'] = prev_close / sl - 1
                position = 0
            elif low_price <= tp:
                df.at[date, 'strategy_return'] = prev_close / tp - 1
                position = 0
            else:
                df.at[date, 'strategy_return'] = -(close_arr[i] / prev_close - 1)

    # 4) Build equity curves
    df['equity_curve']   = (1 + df['strategy_return']).cumprod()
    df['buy_hold_curve'] = (1 + df['Close'].pct_change().squeeze()).cumprod().fillna(1)

    # 5) Compute performance metrics
    sharpe  = df['strategy_return'].mean() / df['strategy_return'].std(ddof=0) * np.sqrt(252)
    cum_max = df['equity_curve'].cummax()
    drawdown= df['equity_curve'] / cum_max - 1
    maxdd   = drawdown.min()

    # attach scalar metrics for easy access in plotting
    df['sharpe_ratio'] = sharpe
    df['max_drawdown'] = maxdd

    return df

def main():
    ticker = "AAPL"
    df_bt  = backtest_ao_sl_tp(ticker, stop_loss_pct=0.02, take_profit_pct=0.05)

    # extract scalar metrics
    sharpe = df_bt['sharpe_ratio'].iloc[0]
    maxdd  = df_bt['max_drawdown'].iloc[0]
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown:         {maxdd:.2%}")

    plot_performance(df_bt, ticker)

if __name__ == "__main__":
    main()
