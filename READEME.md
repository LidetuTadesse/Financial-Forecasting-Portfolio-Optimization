# Financial Forecasting & Portfolio Optimization

## Project Overview
This project applies **time series forecasting** and **Modern Portfolio Theory (MPT)** to optimize investment portfolios.  
It uses **historical data** from `yfinance` for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) from **2015-07-01 to 2025-07-31**.

## Objectives
- Fetch & preprocess historical financial data
- Apply ARIMA/SARIMA & LSTM for forecasting
- Optimize portfolios using MPT and Efficient Frontier
- Backtest strategies against benchmark performance

## Project Structure
financial_forecasting/
│── data/ # Raw & processed datasets (not in repo)
│── notebooks/ # Jupyter notebooks for each step
│── src/ # Python modules
│── tests/ # Unit tests
│── reports/ # Visualizations & results
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── .gitignore # Ignore rules

bash
Copy code


# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

