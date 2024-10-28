import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt


# List of tickers you want to download
tickers = ['PFE', 'JNJ', 'UNH']

# Loop to download, process, and dynamically assign DataFrames for each ticker
for ticker in tickers:
    # Download the stock data
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    # Reset the index to make 'Date' a column
    data.reset_index(inplace=True)
    
    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Dynamically create a variable for each ticker and assign the processed data
    globals()[f"{ticker.lower()}_data"] = data
    
    # Print a message to confirm data download and variable assignment
    print(f"Data for {ticker} processed and stored in variable '{ticker.lower()}_data'.")

# Access and print Stock data seprately using the dynamically created variable
unh_data = globals().get('unh_data')
jnj_data = globals().get('jnj_data')
pfe_data = globals().get('pfe_data')

# Example of accessing data through the dynamically created variables
# Now you can use these variables directly to access or manipulate the data
print("Data for JNJ:")
print(jnj_data.head())  # Accessing the data stored in the dynamically named variable 'jnj_data'

print("Data for PFE:")
print(pfe_data.head())  # Accessing the data stored in the dynamically named variable 'pfe_data'

print("Data for UNH:")
print(unh_data.head())  # Accessing the data stored in the dynamically named variabl 'unh_data'

# Save the each ticker data in to seprate data frame to CSV
data.to_csv('jnj_data.csv')

data.to_csv('pfe_data.csv')

data.to_csv('unh_data.csv')

print("Current Working Directory:", os.getcwd())

# Add day column in to each dataframe
unh_data['Day']=unh_data["Date"].dt.day

jnj_data['Day']=jnj_data["Date"].dt.day

pfe_data['Day']=pfe_data["Date"].dt.day


### ROI calculation for stocks seprately ###
## Buy one share of stock on the 30th of each month.
## From January 2023 to January 2024.

## UNH ROI
sumUNH=0 #total amount invested in UNH

s1=0 #number of shares owned by UNH

# Calcuate total amount invested and number of shares owned in UNH
for i in range(len(unh_data)):

    if unh_data.loc[i,'Day']==30:

        sumUNH+=unh_data.loc[i,'Open']

        s1+=1

#displaying basic results
print("Total Invested in UNH = $",round(sumUNH,2))

print("Shares Owned of UNH =",s1)

print("Average Investmentment of 1 share = $",round((sumUNH/s1),2))




unh_end=525.97 #last open price of UNH on 2023-12-29

#obtained by looking at the data or can be seen after executing unh_data.tail()
#calculating investment results
result1=round((unh_end*s1)-sumUNH,2)

roiUNH=round((result1/sumUNH)*100,2)

#displaying investment results
print("nInvestment Result:")

if result1<0:

    print("Net Unrealised Loss = $",result1)

else:

    print("Net Unrealised Profit = $",result1)


print("UNH ROI from 01.01.2023 to 01.01.2024 =",roiUNH,"%")


## JNJ ROI
sumJNJ=0 #total amount invested in JNJ

s2=0 #number of shares owned by JNJ

# Calcuate total amount invested and number of shares owned in JNJ
for i in range(len(jnj_data)):

    if jnj_data.loc[i,'Day']==30:

        sumJNJ+=jnj_data.loc[i,'Open']

        s2+=1

# displaying basic results
print("Total Invested in JNJ = $",round(sumJNJ,2))

print("Shares Owned of JNJ =",s2)

print("Average Investmentment of 1 share = $",round((sumJNJ/s2),2))



jnj_end=156.5 #last open price of JNJ on 01.01.2024

#obtained by looking at the data or can be seen after executing jnj_data.tail()
#calculating investment results
result2=round((jnj_end*s2)-sumUNH,2)

roiJNJ=round((result2/sumJNJ)*100,2)

#displaying investment results
print("nInvestment Result:")

if result2<0:

    print("Net Unrealised Loss = $",result2)

else:

    print("Net Unrealised Profit = $",result2)

print("JNJ ROI from 01.01.2023 to 01.01.2024 =",roiJNJ,"%")


## PFE ROI
sumPFE=0 #total amount invested in PFE

s3=0 #number of shares owned by PFE

# Calcuate total amount invested and number of shares owned in PFE
for i in range(len(pfe_data)):

    if pfe_data.loc[i,'Day']==30:

        sumPFE+=pfe_data.loc[i,'Open']

        s3+=1

# displaying basic results
print("Total Invested in PFE = $",round(sumPFE,2))

print("Shares Owned of PFE =",s3)

print("Average Investmentment of 1 share = $",round((sumPFE/s3),2))



pfe_end=28.78 #last open price of pfe on 01.01.2024

#obtained by looking at the data or can be seen after executing pfe_data.tail()
#calculating investment results
result3=round((pfe_end*s3)-sumPFE,2)

roiPFE=round((result3/sumPFE)*100,2)

#displaying investment results
print("nInvestment Result:")

if result3<0:

    print("Net Unrealised Loss = $",result3)

else:

    print("Net Unrealised Profit = $",result3)


print("PFE ROI from 01.01.2023 to 01.01.2024 =",roiPFE,"%")


###  Plotting Stocks ROI on Bar Graph ###
plt.figure(figsize=(5,7))

stock=['UNH','JNJ','PFE']

ROI=[roiUNH,roiJNJ,roiPFE]

col=['Blue','Grey','Orange']


plt.bar(stock,ROI,color=col)

plt.title("Stocks ROI Comparison")

plt.xlabel("Stocks")

plt.ylabel("Percentage")


### Plotting Stocks Profit/Loss Amount on Bar Graph ###
plt.figure(figsize=(5,7))

stock=['UNH','JNJ','PFE']

amt=[result1,result2,result3]

col=['Blue','Grey','Orange']

plt.bar(stock,amt,color=col)

plt.title("Stocks Profit/Loss Comparison")

plt.xlabel("Stocks")

plt.ylabel("Amount")

### Portfolio Allocation ###

plt.figure(figsize=(5,7))

stock=['UNH','JNJ','PFE']

shares=[s1,s2,s3]

col=['Blue','Grey','Orange']


plt.pie(shares,labels=stock,autopct="%1.2f%%",colors=col)

plt.legend(title="",loc="upper left")

plt.title("Portfolio Allocation")

















