#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:46:44 2026

@author: joshuataylor
"""

import pandas as pd
from fredapi import Fred
from dotenv.main import load_dotenv
import os

# This file is the less polished but provides a more thorough look at what I did at each step
# It will contain more prints and more eda that didn't make the final version so that file
# can be less messy. Here I am probably going to over explain my thought process so I 
# have a clear understanding of what I was doing in the moment


# Pulls consumer sentiment data from FRED and S&P 500 price data from Yahoo Finance,
# cleans and merges both datasets, computes derived analytical columns,
# and saves the result to df_sent_stock.csv for use in the dashboard.

#%% Creating the consumer sentiment df


# Load API credentials
load_dotenv()
api_key = os.getenv('FRED_API_KEY') # Read the key from the .env file

# Fail clearly if the key is missing, rather than producing a cryptic error later
if not api_key:
    raise ValueError("FRED_API_KEY not found. Check your .env file locally or your GitHub Secret in the cloud.")


# Connect to FRED via an API
fred = Fred(api_key = api_key)

# Pull University of Michigan's Consumer Sentiment Index
sentiment = fred.get_series('UMCSENT')

df_sentiment = pd.DataFrame(sentiment, columns=['sentiment'])

print(df_sentiment.head(10))
print(df_sentiment.shape)

# There are quite a few NaN values so I need to figure out what to do with them
# How many NaN values are there total? 210
print(df_sentiment.isnull().sum())

# When does monthly coverage become consistent? There are zero NaNs past 1978-01-01
# I'm able to tell through glancing through the raw data
print(df_sentiment.loc['1978-01-01':].isnull().sum())

# Filter to 1978 where data becomes consistent monthly
df_sentiment = df_sentiment.loc['1978-01-01':]

print(df_sentiment.shape)
print(df_sentiment.tail(5))
print(df_sentiment.isnull().sum())


#%% Creating and cleaning the stock market df

# I need to see when I have good stock market data to determine the cutoff
# Running !pip install yfinance for stock market data
# Import needed packages
import yfinance as yf

# ^GSPC is Yahoo Finance's ticker symbol for the S&P 500
sp500 = yf.download('^GSPC', start='1950-01-01')

print(sp500.head(10))
print(sp500.tail(10))
print(sp500.shape)

# Open — the price at the very start of the trading day when the market opened
# High — the highest price the S&P 500 reached at any point during that day
# Low — the lowest price reached during that day
# Close — the price at the end of the trading day when the market closed. 
# Close is what I want since it represents the market's inal verdict on value for the day.
# Volume — the total number of shares traded that day. High volume on a big price move 
# suggests conviction; the same move on low volume is considered less meaningful.

# Since this data runs reliably bcak to 1950 I will set the cutoff at 1978 when
# the consumer sentiment data becomes reliable

# One thing I just ran into worth remembering: the Sentiment data is dated on the first 
# of the month for each month. Meaning, Jan 1st of 1990's value reflects how consumers
# felt throughout the entirety of January, not the previous month.
# And, the stock market data is gathered every day. So I am going to use the close price
# of the last day of the month to pair as my indicator of how the stock market reacted
# for that month. I will also have other derived values as well to let the user
# toggle between consumer sentiment and other valeus of the SP 500's performance.

# Downloading all daily S&P 500 data from Jan 1 1978 to today
sp500 = yf.download('^GSPC', start='1978-01-01')

# Flatten the MultiIndex columns down to just the metric
# e.g. ('Close', '^GSPC') becomes just 'Close'
sp500.columns = sp500.columns.get_level_values('Price')

# Resample takes daily rows and groups them into monthly buckets
# 'ME' means month-end — each bucket ends on the last trading day of that month
# .agg() lets you apply a different calculation to each column simultaneously
# rather than calling resample separately for each statistic
df_sp500 = sp500.resample('ME').agg(
    # 'first' takes the opening price of the very first trading day of the month
    sp500_open=('Open', 'first'),
    # 'max' scans every daily high in that month and returns the highest one
    sp500_high=('High', 'max'),
    # 'min' scans every daily low in that month and returns the lowest one
    sp500_low=('Low', 'min'),
    # 'last' takes the closing price of the very last trading day of the month
    sp500_close=('Close', 'last'),
    # 'mean' averages all daily closing prices across every trading day that month
    sp500_avg_close=('Close', 'mean'),
    # 'mean' averages daily volume — how many shares traded each day that month
    sp500_avg_volume=('Volume', 'mean')
)

# pct_change() computes (current_month - prior_month) / prior_month * 100
# giving you the percentage gain or loss from one month-end close to the next
# the first row will be NaN because there is no prior month to compare against
df_sp500['sp500_monthly_return'] = df_sp500['sp500_close'].pct_change() * 100

# This measures how wide the price range was within the month
# (high - low) / low gives the swing as a fraction of the low price
# multiplying by 100 converts it to a percentage
# a high number means a volatile month, a low number means a calm month
df_sp500['sp500_intra_volatility'] = ((df_sp500['sp500_high'] - df_sp500['sp500_low']) / df_sp500['sp500_low']) * 100

print(df_sp500.head(10))
print(df_sp500.shape)
print(df_sp500.columns.tolist())


#%% Cleaning the dataframes so we can merge them

# Since my sentiment data is marked for the beginning of the month whereas the correlating
# stock market data is the closing price on the last day of the month we need to make sure
# the indexes are aligned properly
# Example: Sentiment = 2026-01-01 is paired with Stock Market data dated 2026-01-31
# Remember that the Consumer Survery is collected throughout the month (January in this case)
# but it is dated to the first of the month. Although it says the first of the month, it
# represents consumer sentiment for the entirety of January
# So I am going to remove the day part of the date and have the date just be the year and 
# month so we can merge the datasets cleanly
print(df_sentiment.index[:5])
print(df_sp500.index[:5])

# Removing the day from the dates in each dataframe
df_sentiment.index = df_sentiment.index.to_period('M')
df_sp500.index = df_sp500.index.to_period('M')

print(df_sentiment.index[:3])
print(df_sp500.index[:3])

# Merging the two cleaned dataframes - running an inner join on the year-month index
# I'm only keeps months where both datasets have data
df_merged = df_sentiment.join(df_sp500, how='inner')

print(df_merged.shape)
print(df_merged.head(5))
print(df_merged.isnull().sum())


#%% Adding derived columns for analysis

# Z-scores: transforms both series to standard deviations from their mean
# allows plotting sentiment and sp500 on the same axis for honest comparison
df_merged['sentiment_z'] = (df_merged['sentiment'] - df_merged['sentiment'].mean()) / df_merged['sentiment'].std()
df_merged['sp500_close_z'] = (df_merged['sp500_close'] - df_merged['sp500_close'].mean()) / df_merged['sp500_close'].std()

# Month-over-month percent change in sentiment
# sp500_monthly_return already exists, this is the equivalent for sentiment
df_merged['sentiment_pct_change'] = df_merged['sentiment'].pct_change() * 100

# Rolling 12-month correlation between sentiment and sp500 closing price
# min_periods=6 means we need at least 6 months of data to compute a value
df_merged['rolling_corr_12m'] = df_merged['sentiment'].rolling(window=12, min_periods=6).corr(df_merged['sp500_close'])

print(df_merged.columns.tolist())
print(df_merged.isnull().sum())
print(df_merged.tail(5))


#%% Saving the merged dataframe to a CSV in your project folder so I don't have to re-pull
# from both APIs each time I run the dashboard
# Round to 4 decimals so tiny floating-point differences between runs
# don't register as changes and trigger redundant commits. I have a line in my workflow file
# that is supposed to ignore new commits if nothing changes from a data the automated data pull. 
# Meaning, if there is no new data yet and my automated data pull still runs, Github won't note that 
# I ran a commit. I did this because I only want commits to show when new data is added.
# If no new data is added then it muddies which commits are worthwhile to look at / to include
# Not rounding the dervied columns might produce ever so slightly different values
# each time they are calculated so we round them now to keep them stable and not to get
# Github to think that something changed if it's only the dervied columns' trailing decimal
# numbers
df_merged = df_merged.round(4)
df_merged.to_csv('df_sent_stock.csv')
print(os.getcwd())





























