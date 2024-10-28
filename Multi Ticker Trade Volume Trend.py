# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:36:03 2024

@author: Administrator
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# List of tickers you want to download
tickers = ['PFE', 'JNJ', 'UNH']

# Loop to download, process, and plot volume trends for each ticker
for ticker in tickers:
    # Download the stock data
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    # Reset the index to make 'Date' a column and convert 'Date' to datetime format
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Convert 'Date' column from datetime to numeric for regression analysis
    data['Date_ordinal'] = data['Date'].apply(lambda x: x.toordinal())
    
    ### Calculate and plot liner regression ###
    # Perform linear regression on volume data using the ordinal date
    slope, intercept, r_value, p_value, std_err = linregress(data['Date_ordinal'], data['Volume'])
    
    # Calculate the regression line values
    trend_line = data['Date_ordinal'] * slope + intercept

    # Plotting the volume data and the trend line
    plt.figure(figsize=(10, 5))
    plt.scatter(data['Date'], data['Volume'], label='Actual Volume', color='lightblue')
    plt.plot(data['Date'], trend_line, label='Trend Line', color='red', linewidth=2)
    plt.title(f'Volume Trend for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    ### 