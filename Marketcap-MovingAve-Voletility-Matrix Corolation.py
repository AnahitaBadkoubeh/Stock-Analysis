# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:53:02 2024

@author: Administrator
"""

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import yfinance as yf
%matplotlib inline


# List of tickers you want to download
tickers = ['PFE', 'JNJ', 'UNH']

# Loop to download, process, and dynamically assign DataFrames for each ticker
for ticker in tickers:
    # Download the stock data
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    # Dynamically create a variable for each ticker and assign the processed data
    globals()[f"{ticker.lower()}_data"] = data
    
    # Print a message to confirm data download and variable assignment
    print(f"Data for {ticker} processed and stored in variable '{ticker.lower()}_data'.")

# Access and print Stock data seprately using the dynamically created variable
unh_data = globals().get('unh_data')
jnj_data = globals().get('jnj_data')
pfe_data = globals().get('pfe_data')

# Market Capitalisation
unh_data['MarktCap'] = unh_data['Open'] * unh_data['Volume']
jnj_data['MarktCap'] = jnj_data['Open'] * jnj_data['Volume']
pfe_data['MarktCap'] = pfe_data['Open'] * pfe_data['Volume']

a = unh_data['MarktCap'] 
last_row = a.iloc[-1]
print(last_row)

b = jnj_data['MarktCap']
last_rowb = b.iloc[-1]
print(last_rowb)

c = pfe_data['MarktCap']
last_rowc = c.iloc[-1]
print(last_rowc)


# Plot Marketcap
unh_data['MarktCap'].plot(label = 'UNH', figsize = (15,7))
jnj_data['MarktCap'].plot(label = 'JNJ')
pfe_data['MarktCap'].plot(label = 'PFE')
plt.title('Market Cap')
plt.legend()

### Moving average (50 days) ###
# Calculale Moving Average for Stocks
unh_data['MA50'] = unh_data['Close'].rolling(50).mean()
jnj_data['MA50'] = jnj_data['Close'].rolling(50).mean()
pfe_data['MA50'] = pfe_data['Close'].rolling(50).mean()

# Plot Moving Average for UNH
plt.title('UNH Moving Average (50 Days)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
unh_data['Close'].plot(figsize = (15,7))
unh_data['MA50'].plot()
plt.legend()
plt.show()

# Plot Moving Average for JNJ
plt.title('JNJ Moving Average (50 Days)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
jnj_data['Close'].plot(figsize = (15,7))
jnj_data['MA50'].plot()
plt.legend()
plt.show()

# Plot Moving Average for PFE
plt.title('PFE Moving Average (50 Days)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
pfe_data['Close'].plot(figsize = (15,7))
pfe_data['MA50'].plot()
plt.legend()
plt.show()

### Scattered Plot Matrix ###
data = pd.concat([unh_data['Close'],jnj_data['Close'],pfe_data['Close']],axis = 1)
data.columns = ['UNH Close Price','JNJ Close Price','PFE Close Price']
scatter_matrix(data, figsize = (8,8), hist_kwds= {'bins':250})

### Voletility (Percentage Increase in Stock Value) ###
## Voletility Calculation
unh_data['returns'] = (unh_data['Close']/unh_data['Close'].shift(1)) -1
jnj_data['returns'] = (jnj_data['Close']/jnj_data['Close'].shift(1))-1
pfe_data['returns'] = (pfe_data['Close']/pfe_data['Close'].shift(1)) -1

## PFE Vpletility Plottinh Histogram
plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
ax = pfe_data['returns'].hist(bins=100, label='PFE', alpha=0.5, color='blue')

# Setting the title
ax.set_title('PFE Volatility')

# Adding labels
ax.set_xlabel('Returns')
ax.set_ylabel('Frequency')

# Adding a legend
ax.legend()

# Showing the plot
plt.show()

## JNJ Vpletility Plottinh Histogram
plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
ax = jnj_data['returns'].hist(bins=100, label='JNJ', alpha=0.5, color='Red')

# Setting the title
ax.set_title('JNJ Volatility')

# Adding labels
ax.set_xlabel('Returns')
ax.set_ylabel('Frequency')

# Adding a legend
ax.legend()

# Showing the plot
plt.show()

## UNH Vpletility Plottinh Histogram
plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
ax = unh_data['returns'].hist(bins=100, label='UNH', alpha=0.5, color='Green')

# Setting the title
ax.set_title('UNH Volatility')

# Adding labels
ax.set_xlabel('Returns')
ax.set_ylabel('Frequency')

# Adding a legend
ax.legend()

# Showing the plot
plt.show()

## Voletility Comparison
unh_data['returns'].hist(bins = 100, label = 'UNH', alpha = 0.5, color='Green', figsize = (15,7))
jnj_data['returns'].hist(bins = 100, label = 'JNJ', alpha = 0.5, color='Red')
pfe_data['returns'].hist(bins = 100, label = 'PFE', alpha = 0.5, color='Blue')
plt.title('Volatility Comparison')  # Add title to the figure
plt.xlabel('Returns')  # Optional: Add an x-label
plt.ylabel('Frequency')  # Optional: Add a y-label
plt.legend()  # Show legend to identify which histogram corresponds to which stock

plt.show()