import pandas as pd
import matplotlib.pyplot as plt
import numpy as mp

def run_backtest(data: pd.DataFrame) -> None:
    """
    Generate trading signals, calculate performance metrics, and plot results.
    Signals: buy when RSI<30 and MACD>Signal; sell when RSI>70 and MACD<Signal.
    """
    # Initialize position
    data['Position'] = np.nan
    buy = (data['RSI'] < 30) & (data['MACD'] > data['Signal'])
    sell = (data['RSI'] > 70) & (data['MACD'] < data['Signal'])
    data.loc[buy, 'Position'] = 1
    data.loc[sell, 'Position'] = 0
    data['Position'] = data['Position'].ffill().fillna(0)

    # Calculate returns
    data['Return'] = data['Close'].pct_change() * data['Position'].shift()
    cumulative = (1 + data['Return']).cumprod()
    max_drawdown = (cumulative.cummax() - cumulative).max()
    sharpe = data['Return'].mean() / data['Return'].std() * (252**0.5)

    # Print metrics
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Plot performance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label='Strategy Return')
    plt.plot((1 + data['Close'].pct_change()).cumprod(), label='Buy & Hold')
    plt.legend()
    plt.title('Strategy vs Buy & Hold performance')
    plt.show()
