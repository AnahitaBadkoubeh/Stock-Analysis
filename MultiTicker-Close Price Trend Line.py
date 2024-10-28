

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define the stock tickers
tickers = ["UNH", "PFE", "JNJ"]

# Fetch stock data from Yahoo Finance
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")

# Iterate over each ticker to analyze and plot closing prices and trend line
for ticker in tickers:
    # Extract data for the current ticker using `xs` to filter by level in MultiIndex
    ticker_data = data.xs(ticker, axis=1, level=1)
    
    # Extract the closing prices for the ticker
    close_prices = ticker_data['Close']

    # Convert index (date) to a numeric format for regression analysis
    x = np.arange(len(close_prices.index))
    y = close_prices.values

    # Perform linear regression to get the slope and intercept
    slope, intercept, _, _, _ = linregress(x, y)

    # Calculate the best-fit line using the slope and intercept
    best_fit_line = slope * x + intercept

    # Create a plot for each ticker
    plt.figure(figsize=(10, 5))  # Set figure size

    # Plot the original closing prices
    plt.plot(close_prices.index, close_prices, label=f'{ticker} Closing Prices')

    # Plot the trend line
    plt.plot(close_prices.index, best_fit_line, 'r--', label=f'Trend Line (Slope={slope:.2f})')

    # Add title and labels
    plt.title(f'{ticker} Closing Prices and Trend Line')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')

    # Add a legend to the plot
    plt.legend()

    # Show grid and plot
    plt.grid(True)
    plt.show()
