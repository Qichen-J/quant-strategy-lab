# strategies/rsi_macd_revised.py

from .rsi_macd_baseline import fetch_data, compute_rsi, compute_macd, plot_performance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_rsi_macd_revised(
    tickers: list[str],
    rsi_low: float = 40,
    rsi_high: float = 60,
    stop_loss: float = 0.02,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Multi-asset RSI+MACD strategy with stop-loss (revised thresholds).
    Returns DataFrame of daily returns for each ticker + 'Portfolio'.
    """
    df0 = fetch_data(tickers[0], period=period, interval=interval)
    rets = pd.DataFrame(index=df0.index)
    for t in tickers:
        df = fetch_data(t, period=period, interval=interval)
        df["RSI"] = compute_rsi(df["Close"])
        macd, sig = compute_macd(df["Close"])
        df["MACD"], df["Signal"] = macd, sig
        buy_mask  = (df["RSI"] < rsi_low)  & (df["MACD"] > df["Signal"])
        sell_mask = (df["RSI"] > rsi_high) & (df["MACD"] < df["Signal"])
        df["Position"] = 0
        df.loc[buy_mask,  "Position"] = 1
        df.loc[sell_mask, "Position"] = 0
        df["Position"].ffill(inplace=True)
        raw = df["Close"].pct_change().squeeze().fillna(0)
        raw_ret = raw * df["Position"].shift().fillna(0)
        ret = np.where(raw_ret < -stop_loss, -stop_loss, raw_ret)
        rets[t] = pd.Series(ret, index=df.index)
    rets["Portfolio"] = rets.mean(axis=1)
    return rets

def plot_portfolio_performance(rets: pd.DataFrame, tickers: list[str]):
    cum_strat = (1 + rets["Portfolio"]).cumprod()
    bh = pd.concat({
        t: (1 + fetch_data(t)["Close"].pct_change().fillna(0)).cumprod()
        for t in tickers
    }, axis=1)
    cum_bh = bh.mean(axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(cum_strat, label="Strategy Portfolio")
    plt.plot(cum_bh,    linestyle="--", label="Buy & Hold Portfolio")
    plt.title("Multi-Asset RSI+MACD Revised vs Buy & Hold")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    tickers = ["AAPL", "SPY", "XOM"]
    rets    = backtest_rsi_macd_revised(tickers)
    sharpe  = rets["Portfolio"].mean() / rets["Portfolio"].std(ddof=0) * np.sqrt(252)
    maxdd   = ((1 + rets["Portfolio"]).cumprod().cummax() -
               (1 + rets["Portfolio"]).cumprod()).max()

    print(f"Combined Sharpe:     {sharpe:.2f}")
    print(f"Combined Max Drawdown:{maxdd:.2%}")
    plot_portfolio_performance(rets, tickers)
