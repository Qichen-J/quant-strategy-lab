# strategies/rsi_macd_gridsearch.py

import itertools
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_data(tickers, period="2y", interval="1d"):
    """
    Download data for each ticker and return a dict of DataFrames.
    """
    data = {}
    for t in tickers:
        df = yf.download(t, period=period, interval=interval).dropna()
        data[t] = df
    return data


def compute_indicators(df, window=14):
    """
    Given a DataFrame with 'Close', compute RSI, MACD, Signal in-place.
    """
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.rolling(window).mean()
    avg_loss  = loss.rolling(window).mean()
    df["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)

    exp1      = df["Close"].ewm(span=12, adjust=False).mean()
    exp2      = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    df["MACD"]   = macd_line
    df["Signal"] = macd_line.ewm(span=9, adjust=False).mean()


def backtest_with_params(df, rsi_low, rsi_high, stop_loss):
    """
    Given one ticker's DataFrame (with indicators), generate Position,
    compute daily returns (with .squeeze()) and apply stop_loss.
    Returns a 1D Series of daily returns.
    """
    d = df.copy()
    d["Position"] = np.nan

    buy  = (d["RSI"] <  rsi_low) & (d["MACD"] > d["Signal"])
    sell = (d["RSI"] >  rsi_high) & (d["MACD"] < d["Signal"])
    d.loc[buy,  "Position"] = 1
    d.loc[sell, "Position"] = 0
    d["Position"].ffill(inplace=True)
    d["Position"].fillna(0, inplace=True)

    raw = d["Close"].pct_change().squeeze().fillna(0)
    ret = raw * d["Position"].shift().fillna(0)
    ret = ret.clip(lower=-stop_loss)
    return ret.rename(d.name if hasattr(d, "name") else None)


def grid_search_rsi(
    data_dict,
    tickers,
    rsi_lows,
    rsi_highs,
    stop_loss=0.02
):
    """
    Loop over all low/high combinations, backtest each,
    and collect sharpe & max drawdown.
    """
    results = []
    for low, high in itertools.product(rsi_lows, rsi_highs):
        if low >= high:
            continue

        rets = []
        for t in tickers:
            df = data_dict[t].copy()
            df.name = t
            compute_indicators(df)
            ret = backtest_with_params(df, rsi_low=low, rsi_high=high, stop_loss=stop_loss)
            rets.append(ret)

        df_rets   = pd.concat(rets, axis=1)
        port_rets = df_rets.mean(axis=1)

        cumret = (1 + port_rets).cumprod()
        sharpe = port_rets.mean() / port_rets.std(ddof=0) * np.sqrt(252)
        maxdd  = (cumret.cummax() - cumret).max()

        results.append({
            "rsi_low": low,
            "rsi_high": high,
            "sharpe": sharpe,
            "max_drawdown": maxdd
        })

    return pd.DataFrame(results)


def plot_sharpe_heatmap(df_res):
    """
    Given a DataFrame with columns ['rsi_low','rsi_high','sharpe'],
    plot a heatmap of Sharpe values.
    """
    pivot = df_res.pivot(index="rsi_low", columns="rsi_high", values="sharpe")
    plt.figure(figsize=(6, 5))
    plt.title("Multi‐Asset Sharpe by RSI thresholds")
    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(),   pivot.index.max()
        ]
    )
    plt.colorbar(label="Sharpe")
    plt.xlabel("RSI high")
    plt.ylabel("RSI low")
    plt.xticks(pivot.columns)
    plt.yticks(pivot.index)
    plt.show()


def main():
    # Configuration
    tickers   = ["AAPL", "SPY", "XOM"]
    period    = "2y"
    interval  = "1d"
    stop_loss = 0.02
    rsi_lows  = range(20, 41, 5)
    rsi_highs = range(60, 81, 5)

    # 1) Fetch and prepare data
    data_dict = fetch_data(tickers, period=period, interval=interval)

    # 2) Grid search
    df_res = grid_search_rsi(
        data_dict,
        tickers,
        rsi_lows,
        rsi_highs,
        stop_loss=stop_loss
    )

    # 3) Identify best
    best = df_res.loc[df_res["sharpe"].idxmax()]
    print("Best RSI low/high:", int(best["rsi_low"]), "/", int(best["rsi_high"]))
    print(f"Sharpe: {best['sharpe']:.2f}, MaxDD: {best['max_drawdown']:.2%}")

    # 4) Heatmap
    plot_sharpe_heatmap(df_res)


if __name__ == "__main__":
    main()
