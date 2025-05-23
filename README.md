# quant-strategy-lab

## Baseline Strategy
- **Logic**: RSI (30/70) + MACD (12, 26, 9)
- **Asset**: AAPL (daily data over the past 2 years)
- **Performance**: Sharpe ≈ 1.01, Maximum Drawdown 6.88%, see backtest in `notebooks/baseline_strategy.ipynb`

## How to Run
```bash
git clone https://github.com/your-username/quant-strategy-lab.git
cd quant-strategy-lab
pip install -r requirements.txt
jupyter notebook notebooks/baseline_strategy.ipynb
