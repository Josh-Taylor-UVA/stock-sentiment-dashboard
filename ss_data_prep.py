#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:28:23 2026

@author: joshuataylor
"""
# Pulls consumer sentiment data from FRED and S&P 500 price data from Yahoo Finance,
# cleans and merges both datasets, computes derived analytical columns,
# and saves the result to df_sent_stock.csv for use in the dashboard.

import pandas as pd
from fredapi import Fred
from dotenv.main import load_dotenv
import yfinance as yf
import os

# ── Load API credentials ───────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv('FRED_API_KEY')

# Fail clearly if the key is missing so we don't produce an unknown error later
if not api_key:
    raise ValueError("FRED_API_KEY not found. Check your .env file locally or your GitHub Secret in the cloud.")

# ── Pull consumer sentiment data from FRED ─────────────────────────────────────
fred = Fred(api_key=api_key)
sentiment = fred.get_series('UMCSENT')
df_sentiment = pd.DataFrame(sentiment, columns=['sentiment'])

# 210 NaN values exist prior to 1978 due to irregular survey cadence
# Data becomes consistently monthly from 1978 onward
df_sentiment = df_sentiment.loc['1978-01-01':]

# ── Pull S&P 500 data from Yahoo Finance ───────────────────────────────────────
sp500 = yf.download('^GSPC', start='1978-01-01')

# Flatten MultiIndex columns returned by yfinance
sp500.columns = sp500.columns.get_level_values('Price')

# Resample daily data to monthly, capturing key price statistics
df_sp500 = sp500.resample('ME').agg(
    sp500_open=('Open', 'first'),
    sp500_high=('High', 'max'),
    sp500_low=('Low', 'min'),
    sp500_close=('Close', 'last'),
    sp500_avg_close=('Close', 'mean'),
    sp500_avg_volume=('Volume', 'mean')
)

# Month-over-month return and intra-month volatility
df_sp500['sp500_monthly_return'] = df_sp500['sp500_close'].pct_change() * 100
df_sp500['sp500_intra_volatility'] = ((df_sp500['sp500_high'] - df_sp500['sp500_low']) / df_sp500['sp500_low']) * 100

# ── Align indexes and merge ────────────────────────────────────────────────────
# Sentiment is dated month-start, S&P 500 is dated month-end
# Converting both to year-month periods removes the day and allows a clean merge
df_sentiment.index = df_sentiment.index.to_period('M')
df_sp500.index = df_sp500.index.to_period('M')

# Inner join keeps only months where both datasets have data
df_merged = df_sentiment.join(df_sp500, how='inner')

# ── Compute derived analytical columns ────────────────────────────────────────
# Z-scores: puts sentiment and S&P 500 on the same scale for direct comparison
df_merged['sentiment_z'] = (df_merged['sentiment'] - df_merged['sentiment'].mean()) / df_merged['sentiment'].std()
df_merged['sp500_close_z'] = (df_merged['sp500_close'] - df_merged['sp500_close'].mean()) / df_merged['sp500_close'].std()

# Month-over-month percent change in sentiment
df_merged['sentiment_pct_change'] = df_merged['sentiment'].pct_change() * 100

# Rolling 12-month correlation — quantifies how tightly coupled the two series are over time
df_merged['rolling_corr_12m'] = df_merged['sentiment'].rolling(window=12, min_periods=6).corr(df_merged['sp500_close'])

# ── Save to CSV ────────────────────────────────────────────────────────────────
df_merged.to_csv('df_sent_stock.csv')

print(df_merged.shape)
print(df_merged.columns.tolist())
print(df_merged.isnull().sum())
print(df_merged.tail(5))