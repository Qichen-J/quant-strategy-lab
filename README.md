# quant-strategy-lab

## 1. Baseline Strategy
- **Logic**: RSI (30/70) + MACD (12, 26, 9)
- **Asset**: AAPL (daily data over the past 2 years)
- **Performance**: Sharpe ≈ 1.01, Maximum Drawdown 6.88%, see backtest in `notebooks/baseline_strategy.ipynb`

### 2. Parameter Grid Search
- **Logic**: Search RSI low ∈ {20,25,30,35,40}, high ∈ {60,65,70,75,80} on [AAPL, SPY, XOM]  
- **Best**: RSI(35/60), Sharpe ≈ 0.73, MaxDD ≈ 6.2%  
- **Notebook**: `notebooks/grid_search_rsi.ipynb`

### 3. Multi-Asset + Stop-Loss
- **Logic**: Equally-weighted on [AAPL, SPY, XOM], daily stop-loss 2%, RSI(40/75) + MACD(12,26,9)  
- **Result**: Sharpe ≈ 1.84, MaxDD ≈ 9.7%  
- **Notebook**: `notebooks/multi_asset_backtest.ipynb`

## How to Run
```bash
git clone https://github.com/Qichen-J/quant-strategy-lab.git
cd quant-strategy-lab
pip install -r requirements.txt
jupyter notebook notebooks/baseline_strategy.ipynb
jupyter notebook notebooks/grid_search_rsi.ipynb
jupyter notebook otebooks/multi_asset_backtest.ipynb
