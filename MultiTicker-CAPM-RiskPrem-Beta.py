

# Import libraries
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

# Define constants
riskfree = 0.025  # Risk-free rate

# Import the data for a group of stocks
def import_stock_data(tickers, start='2014-04-04', end=None):
    # If end date is not provided, use today's date
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    # Dictionary to store adjusted close price data for each ticker
    data = {}
    
    # Fetch stock data for each ticker
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start, end=end)['Adj Close']
    
    return pd.DataFrame(data)

# List of tickers including market index
tickers = ['JNJ', 'UNH', 'PFE', '^GSPC']  # Adding S&P 500 index for market data
data = import_stock_data(tickers)

# Calculate logarithmic daily returns
sec_returns = np.log(data / data.shift(1))

# Annualize the market return to get the risk premium
market_annual_return = sec_returns['^GSPC'].mean() * 252
riskpremium = market_annual_return - riskfree

# Dictionary to store betas and CAPM returns for each stock
betas = {}
capm_returns = {}

# Calculate beta and CAPM return for each stock
for ticker in tickers[:-1]:  # Exclude the market index
    # Calculate covariance between stock and market and annualize it
    cov_with_market = sec_returns[[ticker, '^GSPC']].cov().iloc[0, 1] * 252
    # Calculate market variance and annualize
    market_var = sec_returns['^GSPC'].var() * 252
    
    # Calculate Beta
    beta = cov_with_market / market_var
    betas[ticker] = beta
    
    # Calculate CAPM expected return
    capm_return = riskfree + beta * riskpremium
    capm_returns[ticker] = capm_return

# Display results
for ticker in tickers[:-1]:
    print(f"Ticker: {ticker}, Beta: {betas[ticker]}, CAPM Return: {capm_returns[ticker]}, Risk Premium: {riskpremium}")
