
"""
Goal: Use historical data for TSLA, BND, SPY to optimize portfolio allocation 
      using Modern Portfolio Theory (MPT) and visualize the Efficient Frontier.
"""

# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import plotting

# =========================
# Load Historical Data
# =========================
# Download adjusted close prices for TSLA, BND, SPY from 2015-07-01 to 2025-07-31
tickers = ["TSLA", "BND", "SPY"]
data = yf.download(tickers, start="2015-07-01", end="2025-07-31")["Adj Close"]

print("Data Sample:")
print(data.head())

# =========================
#  Calculate Expected Returns & Covariance Matrix
# =========================
# Annualized mean historical returns
mu = mean_historical_return(data)

# Ledoit-Wolf shrinkage covariance matrix for stability
S = CovarianceShrinkage(data).ledoit_wolf()

# =========================
# Optimize Portfolio (Max Sharpe Ratio)
# =========================
# Create Efficient Frontier object
ef = EfficientFrontier(mu, S)

# Find the portfolio with the maximum Sharpe Ratio
weights = ef.max_sharpe()

# Clean weights for easier readability
cleaned_weights = ef.clean_weights()
print("\nOptimal Weights Allocation:")
print(cleaned_weights)

# Display portfolio performance
performance = ef.portfolio_performance(verbose=True)

# =========================
#  Plot Efficient Frontier
# =========================
fig, ax = plt.subplots(figsize=(8, 6))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

plt.title("Efficient Frontier for TSLA, BND, SPY")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.grid(True)
plt.show()

# =========================
# Save Results
# =========================
# Save weights to CSV
pd.Series(cleaned_weights).to_csv("outputs/task3_optimal_weights.csv")

# Save performance metrics to a text file
with open("outputs/task3_performance.txt", "w") as f:
    f.write(str(performance))

print("\nResults saved to outputs/ folder.")
