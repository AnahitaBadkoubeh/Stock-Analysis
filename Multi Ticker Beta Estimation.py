

import pandas as pd
import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
from scipy.stats import linregress
from plotly.offline import iplot


tickers = ["UNH", "PFE", "JNJ", "^SP500-35"]

# Fetch data
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")['Adj Close']

print(data)


# List the columns names
print("Column Names:", data.columns.tolist())

#Normalizing Stock Prices

def normalize_prices(df):

    df_ = df.copy()

    for stock in df_.columns:

        df_[stock] = df_[stock]/df_[stock][0]

    return df_

norm_df = normalize_prices(data)



####  Comparison normalized closing price of all stocks #### 
plt.figure(figsize=(14, 7))  # Set figure size for better readability
for ticker in tickers:
    norm_df.plot(label=ticker)
plt.title('Stocks Normalized Adj Close Prices Comparison')
plt.xlabel('Date')
plt.ylabel('Normalized Adj Close Price')
plt.legend()
plt.grid(False)
plt.show()




#### Comparison closing price with trends of all stocks  ####
plt.figure(figsize=(14, 7))

# Loop through each ticker to process its data
for ticker in tickers:
    # Convert index (date) to a numeric format for regression analysis
    x = np.arange(len(norm_df.index))
    y = norm_df[ticker].values  # Get only the values for the current ticker
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Calculate the best-fit line
    best_fit_line = intercept + slope * x  # Correct formula for the line
    
    # Plot the normalized closing prices
    plt.plot(norm_df.index, y, label=f'{ticker} Normalized Adj Close Price')  # Use 'y' as it's already the price data
    
    # Plot the best-fit line
    plt.plot(norm_df.index, best_fit_line, label=f'{ticker} Trend (slope={slope:.5f})', linestyle='--')

# Add labels and legend to the plot


plt.title('Normalized Adj Close Prices & Trend Lines')
plt.xlabel('Date')
plt.ylabel('Normalized Adj Close Price')
plt.legend()
plt.grid(False)
plt.show()


# Measure of Volatility
# Calculating Daily % change in stock prices

daily_returns = norm_df.pct_change()

daily_returns.iloc[0,:] = 0

# Boxplot of daily returns (in %)

daily_returns.boxplot(figsize=(6, 5), grid=False)

plt.title("Daily returns of the stocks")

### CAPM Model


# Initializing empty dictionaries to save results

beta,alpha = dict(), dict()

# Make a subplot

fig, axes = plt.subplots(1,3, dpi=150, figsize=(15,8))

axes = axes.flatten()

# Loop on every daily stock return

for idx, stock in enumerate(daily_returns.columns.values[:-1]):

    # scatter plot between stocks and the NSE

    daily_returns.plot(kind = "scatter", x = "^SP500-35", y = stock, ax=axes[idx])

    # Fit a line (regression using polyfit of degree 1)

    b_, a_ = np.polyfit(daily_returns["^SP500-35"] ,daily_returns[stock], 1)

    regression_line = b_ * daily_returns["^SP500-35"] + a_

    axes[idx].plot(daily_returns["^SP500-35"], regression_line, "-", color = "r")

    # save the regression coeeficient for the current stock

    beta[stock] = b_

    alpha[stock] = a_

plt.suptitle("Beta estimation: regression between ^SP500-35 and individual stock daily performance", size=20)

plt.show()







