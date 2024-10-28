

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

%matplotlib inline

tickers = ["UNH", "PFE", "JNJ"]

# Fetch data
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")

print(data)


# List the columns names
print("Column Names:", data.columns.tolist())

# Checking for null values
data.isna().sum()

# Description of Data in the Dataframe
data.describe().round(2)

# Converting the “Date” column (now is an index) dtype from object to date (numeric format)
data.reset_index(inplace=True)

# Now 'Date' is a column
data['Date'] = pd.to_datetime(data['Date'])



####  Comparison closing price of all stocks #### 
plt.figure(figsize=(14, 7))  # Set figure size for better readability
for ticker in tickers:
    data['Close', ticker].plot(label=ticker)
plt.title('Closing Prices Comparison')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(False)
plt.show()




#### Trends comparison  ####
plt.figure(figsize=(14, 7))

# Loop through each ticker to process its data
for ticker in tickers:
    # Extract the closing prices for the ticker
    close_prices = data['Close', ticker]
    
    # Convert index (date) to a numeric format for regression analysis
    x = np.arange(len(close_prices.index))
    y = close_prices.values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Calculate the best-fit line
    best_fit_line = slope * x + intercept
    
    # Plot the original closing prices
    plt.plot(close_prices.index, close_prices, label=f'{ticker} Close')
    
    # Plot the best-fit line
    plt.plot(close_prices.index, best_fit_line, label=f'{ticker} Trend (slope={slope:.2f})', linestyle='--')

# Add labels and legend to the plot
plt.title('Closing Prices and Trend Lines')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(False)
plt.show()

#Discription: Over the last year, UNH is clearly outperforming other stocks.
#Hardly can say there is consistent growth due to the slop of the line =0.23.
#The trend of JNJ and PFE is negative. Aquicision news of the companies. Ending 
#pandemic



# Save the data frame to CSV
data.to_csv('stock_data.csv')

print("Current Working Directory:", os.getcwd())


# Change the working directory
os.chdir("C:")
print("New Working Directory:", os.getcwd())