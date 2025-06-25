# strategies/rsi_macd_baseline.py

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

def plot_price(df: pd.DataFrame, ticker: str):
    """
    Plot closing price.
    """
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df["Close"])
    plt.title(f"{ticker} Closing Price (recent {df.shape[0]} days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.show()

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(close: pd.Series,
                 span_short: int = 12,
                 span_long: int = 26,
                 signal_span: int = 9
                ) -> (pd.Series, pd.Series):
    """
    Compute MACD line and signal line.
    """
    exp_short  = close.ewm(span=span_short, adjust=False).mean()
    exp_long   = close.ewm(span=span_long,  adjust=False).mean()
    macd_line  = exp_short - exp_long
    signal_line= macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line

def plot_rsi(df: pd.DataFrame):
    """
    Plot RSI with overbought/oversold lines.
    """
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["RSI"], label="RSI")
    plt.axhline(70, linestyle="--", label="Overbought")
    plt.axhline(30, linestyle="--", label="Oversold")
    plt.legend()
    plt.show()

def plot_macd(df: pd.DataFrame):
    """
    Plot MACD line and signal line.
    """
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["MACD"], label="MACD")
    plt.plot(df.index, df["Signal"], label="Signal")
    plt.legend()
    plt.show()

def backtest_rsi_macd(df: pd.DataFrame,
                     rsi_low: float = 30,
                     rsi_high: float = 70,
                     stop_loss: float = 0.02
                    ) -> pd.DataFrame:
    """
    Generate positions, compute returns & cumulative returns.
    Returns DataFrame with columns:
      ['Close','RSI','MACD','Signal','Position','Return','Cumulative']
    """
    df = df.copy()
    # indicators
    df["RSI"]   = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])

    # positions
    df["Position"] = np.nan
    buy_mask  = (df["RSI"] < rsi_low)  & (df["MACD"] > df["Signal"])
    sell_mask = (df["RSI"] > rsi_high) & (df["MACD"] < df["Signal"])
    df.loc[buy_mask,  "Position"] = 1
    df.loc[sell_mask, "Position"] = 0
    df["Position"].ffill(inplace=True)
    df["Position"].fillna(0, inplace=True)

    # returns with stop-loss
    ret = df["Close"].pct_change().fillna(0) * df["Position"].shift().fillna(0)
    ret = ret.clip(lower=-stop_loss)
    df["Return"]     = ret
    df["Cumulative"] = (1 + df["Return"]).cumprod()

    return df

def plot_performance(df: pd.DataFrame):
    """
    Plot strategy vs buy & hold performance.
    """
    cum_strategy = df["Cumulative"]
    cum_bh       = (1 + df["Close"].pct_change().fillna(0)).cumprod()

    plt.figure(figsize=(10,6))
    plt.plot(cum_strategy, label="Strategy Return")
    plt.plot(cum_bh,       label="Buy & Hold", linestyle="--")
    plt.legend()
    plt.title("Strategy vs Buy & Hold Performance")
    plt.show()

if __name__ == "__main__":
    # demo run
    ticker = "AAPL"
    df     = fetch_data(ticker)
    plot_price(df, ticker)
    df_bt  = backtest_rsi_macd(df)
    print("Sharpe ratio:", df_bt["Return"].mean() / df_bt["Return"].std() * np.sqrt(252))
    print("Max drawdown:", (df_bt["Cumulative"].cummax() - df_bt["Cumulative"]).max())
    plot_rsi(df_bt)
    plot_macd(df_bt)
    plot_performance(df_bt)
