"""
Strategy Backtesting
This script backtests the optimized portfolio 
"""

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load historical data
# Make sure this file includes the backtesting period (e.g., Aug 1, 2024 – Jul 31, 2025)
data = pd.read_csv("data/processed/historical_prices.csv", parse_dates=["Date"], index_col="Date")

# Step 3: Select backtesting period
start_date = "2024-08-01"
end_date = "2025-07-31"
backtest_data = data.loc[start_date:end_date]

# Step 4: Calculate daily returns
returns = backtest_data.pct_change().dropna()

# Step 5: Define strategy weights (from Task 4's optimal portfolio)
# Example weights: TSLA=0.5, BND=0.3, SPY=0.2 (replace with your Task 4 results)
strategy_weights = np.array([0.5, 0.3, 0.2])  # [TSLA, BND, SPY]

# Step 6: Define benchmark weights (static 60% SPY / 40% BND)
benchmark_weights = np.array([0.0, 0.4, 0.6])  # No TSLA

# Step 7: Calculate portfolio daily returns
strategy_returns = returns.dot(strategy_weights)
benchmark_returns = returns.dot(benchmark_weights)

# Step 8: Calculate cumulative returns
strategy_cum = (1 + strategy_returns).cumprod()
benchmark_cum = (1 + benchmark_returns).cumprod()

# Step 9: Plot cumulative returns comparison
plt.figure(figsize=(10,6))
plt.plot(strategy_cum, label="Strategy Portfolio", linewidth=2)
plt.plot(benchmark_cum, label="Benchmark (60% SPY / 40% BND)", linewidth=2)
plt.title("Backtesting: Strategy vs Benchmark")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Calculate performance metrics
def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    return (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)

strategy_total_return = strategy_cum.iloc[-1] - 1
benchmark_total_return = benchmark_cum.iloc[-1] - 1

strategy_sharpe = sharpe_ratio(strategy_returns)
benchmark_sharpe = sharpe_ratio(benchmark_returns)

# Step 11: Print performance summary
print("=== Backtest Results ===")
print(f"Strategy Total Return: {strategy_total_return:.2%}")
print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")

# Step 12: Interpretation (to be expanded in report)
if strategy_total_return > benchmark_total_return:
    print("\nThe strategy outperformed the benchmark in total return.")
else:
    print("\nThe strategy underperformed the benchmark in total return.")

if strategy_sharpe > benchmark_sharpe:
    print("The strategy delivered better risk-adjusted returns than the benchmark.")
else:
    print("The strategy had worse risk-adjusted returns than the benchmark.")
