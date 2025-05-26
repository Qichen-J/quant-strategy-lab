# quant-strategy-lab

## RSI-MACD

### 1. Baseline Strategy
- **Logic**: RSI(30/70) + MACD(12,26,9) on AAPL  
- **Result**: Sharpe ≈ 1.01, MaxDD ≈ 6.9%  
- **Notebook**: `notebooks/baseline_strategy_RSI_MACD.ipynb`

### 2. Parameter Grid Search
- **Logic**: Search RSI low ∈ {20,25,30,35,40}, high ∈ {60,65,70,75,80} on [AAPL, SPY, XOM]  
- **Best**: RSI(35/60), Sharpe ≈ 1.84, MaxDD ≈ 9.7%  
- **Notebook**: `notebooks/grid_search_rsi.ipynb`

### 3. Multi-Asset + Stop-Loss
- **Logic**: Equally-weighted on [AAPL, SPY, XOM], daily stop-loss 2%, RSI(40/75) + MACD(12,26,9)  
- **Result**: Sharpe ≈ 1.84, MaxDD ≈ 9.7%  
- **Notebook**: `notebooks/multi_asset_backtest.ipynb`

## Heikin-Ashi Candle Stick

### 1. Baseline Strategy
- **Logic**: HA_Close > HA_Open 
- **Result**: Sharpe ≈ 0.88, MaxDD ≈ 28.76%  
- **Notebook**: `notebooks/baseline_HA.ipynb`

### 2. Heikin-Ashi + MA50 Strategy  
   - **Logic:** go long when `HA_Close > HA_Open` **and** `Close > MA50`  
   - **Result:** Sharpe ≈ 0.79, MaxDD ≈ 13.21%  
   - **Notebook:** `revised_HA.ipynb`

## How to Run
```bash
git clone https://github.com/Qichen-J/quant-strategy-lab.git
cd quant-strategy-lab
pip install -r requirements.txt
jupyter notebook notebooks/baseline_strategy.ipynb
jupyter notebook notebooks/grid_search_rsi.ipynb
jupyter notebook otebooks/multi_asset_backtest.ipynb
