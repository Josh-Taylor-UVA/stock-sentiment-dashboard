#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:35:50 2026

@author: joshuataylor
"""
# ss_app.py
# Main dashboard application for the Stock-Sentiment project

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Sentiment vs S&P 500",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('df_sent_stock.csv', index_col=0)
    return df

df = load_data()

# ── Get min and max dates from the data ───────────────────────────────────────
# Converting the index to a list of strings so the slider can use them
all_dates = df.index.tolist()
min_date = all_dates[0]
max_date = all_dates[-1]

# ── Helper function ────────────────────────────────────────────────────────────
def filter_df(start, end):
    return df.loc[start:end]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Consumer Sentiment vs S&P 500")
st.markdown("Exploring the relationship between the University of Michigan Consumer Sentiment Index and S&P 500 equity prices from 1978 to present.")

# ── Global filter controls ─────────────────────────────────────────────────────
sync_all = st.checkbox("Apply global date filter to all charts", value=False)

if sync_all:
    global_start, global_end = st.select_slider(
        "Global date range",
        options=all_dates,
        value=(min_date, max_date),
        key="global"
    )

# ── Metric cards ───────────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current sentiment index",
        value=f"{latest['sentiment']:.1f}",
        delta=f"{latest['sentiment'] - prev['sentiment']:.1f} vs last month"
    )
with col2:
    st.metric(
        label="S&P 500 close",
        value=f"{latest['sp500_close']:,.0f}",
        delta=f"{latest['sp500_monthly_return']:.1f}% this month"
    )
with col3:
    st.metric(
        label="12-month correlation",
        value=f"{latest['rolling_corr_12m']:.2f}",
    )
with col4:
    st.metric(
        label="Sentiment % change",
        value=f"{latest['sentiment_pct_change']:.1f}%",
        delta=f"{latest['sentiment_pct_change'] - prev['sentiment_pct_change']:.1f} vs last month"
    )

# ── Normalized comparison chart ────────────────────────────────────────────────
st.subheader("Normalized comparison (z-scores)")
st.markdown("Both series transformed to standard deviations from their mean — allows direct visual comparison despite different scales.")

if sync_all:
    df_z = filter_df(global_start, global_end)
else:
    z_start, z_end = st.select_slider(
        "Date range",
        options=all_dates,
        value=(min_date, max_date),
        key="z_range"
    )
    df_z = filter_df(z_start, z_end)

fig_z = go.Figure()
fig_z.add_trace(go.Scatter(
    x=df_z.index.astype(str),
    y=df_z['sentiment_z'],
    name='Consumer Sentiment',
    line=dict(color='#0F6E56', width=1.5)
))
fig_z.add_trace(go.Scatter(
    x=df_z.index.astype(str),
    y=df_z['sp500_close_z'],
    name='S&P 500 Close',
    line=dict(color='#185FA5', width=1.5)
))
fig_z.add_hline(y=0, line_dash='dash', line_color='gray', line_width=0.8)
fig_z.update_layout(
    height=450,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode='x unified',
    yaxis=dict(autorange=True)
)
st.plotly_chart(fig_z, use_container_width=True)

# ── Rolling 12-month correlation ───────────────────────────────────────────────
st.subheader("Rolling 12-month correlation")
st.markdown("How tightly coupled are sentiment and the S&P 500 over time? Values near 1.0 indicate strong positive correlation, near 0 means decoupled, negative means inverse relationship.")

if sync_all:
    df_corr = filter_df(global_start, global_end)
else:
    corr_start, corr_end = st.select_slider(
        "Date range",
        options=all_dates,
        value=(min_date, max_date),
        key="corr_range"
    )
    df_corr = filter_df(corr_start, corr_end)

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(
    x=df_corr.index.astype(str),
    y=df_corr['rolling_corr_12m'],
    name='12-month rolling correlation',
    line=dict(color='#533AB7', width=1.5),
    fill='tozeroy',
    fillcolor='rgba(83, 58, 183, 0.1)'
))
fig_corr.add_hline(y=0, line_dash='dash', line_color='gray', line_width=0.8)
fig_corr.add_hline(y=0.5, line_dash='dot', line_color='green', line_width=0.8)
fig_corr.add_hline(y=-0.5, line_dash='dot', line_color='red', line_width=0.8)
fig_corr.update_layout(
    height=400,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(l=0, r=0, t=30, b=0),
    yaxis=dict(range=[-1, 1], autorange=False),
    hovermode='x unified'
)
st.plotly_chart(fig_corr, use_container_width=True)

# ── Month-over-month % change ──────────────────────────────────────────────────
st.subheader("Month-over-month % change")
st.markdown("Do swings in consumer sentiment correspond to swings in S&P 500 returns?")

if sync_all:
    df_pct = filter_df(global_start, global_end)
else:
    pct_start, pct_end = st.select_slider(
        "Date range",
        options=all_dates,
        value=(min_date, max_date),
        key="pct_range"
    )
    df_pct = filter_df(pct_start, pct_end)

fig_pct = go.Figure()
fig_pct.add_trace(go.Bar(
    x=df_pct.index.astype(str),
    y=df_pct['sp500_monthly_return'],
    name='S&P 500 monthly return %',
    marker_color='#185FA5',
    opacity=0.7
))
fig_pct.add_trace(go.Bar(
    x=df_pct.index.astype(str),
    y=df_pct['sentiment_pct_change'],
    name='Sentiment % change',
    marker_color='#0F6E56',
    opacity=0.7
))
fig_pct.update_layout(
    height=450,
    barmode='overlay',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode='x unified',
    yaxis=dict(autorange=True)
)
st.plotly_chart(fig_pct, use_container_width=True)

# ── Raw data ───────────────────────────────────────────────────────────────────
st.subheader("Raw data")
if sync_all:
    st.dataframe(filter_df(global_start, global_end).tail(20))
else:
    st.dataframe(df.tail(20))