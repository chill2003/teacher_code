# 載入必要模組
import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import indicator_f_Lo2_short
import indicator_forKBar_short

# 設置標題
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

# 讀取資料
df_original = pd.read_pickle('2201.pkl')

# 選擇日期區間
st.sidebar.header("選擇開始與結束的日期, 區間:2022-01-03 至 2024-06-07")
start_date = st.sidebar.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date = st.sidebar.text_input('選擇結束日期 (日期格式: 2024-06-07)', '2024-06-07')
start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

# 轉換為字典
KBar_dic = df.to_dict()

# 將時間單位轉換為天
KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime().date() for i in KBar_time_list]
KBar_dic['time'] = np.array(KBar_time_list)

# 選擇K棒的時間長度
cycle_duration = st.sidebar.number_input('輸入一根 K 棒的時間長度(天數)', value=1)
cycle_duration = int(cycle_duration)

# 計算K棒
Date = start_date.date()
KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

# 填充K棒數據
for i in range(KBar_dic['time'].size):
    time = KBar_dic['time'][i]
    open_price = KBar_dic['open'][i]
    close_price = KBar_dic['close'][i]
    low_price = KBar_dic['low'][i]
    high_price = KBar_dic['high'][i]
    qty = KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

# 計算移動平均線
LongMAPeriod = st.sidebar.slider('設定計算長移動平均線(MA)的 K 棒數目', 0, 100, 10)
ShortMAPeriod = st.sidebar.slider('設定計算短移動平均線(MA)的 K 棒數目', 0, 100, 2)
KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

# 計算RSI指標
LongRSIPeriod = st.sidebar.slider('設定計算長RSI的 K 棒數目', 0, 1000, 10)
ShortRSIPeriod = st.sidebar.slider('設定計算短RSI的 K 棒數目', 0, 1000, 2)

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

# 繪製圖表
st.subheader("畫圖")
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Candlestick(x=KBar_df['time'], open=KBar_df['open'], high=KBar_df['high'], low=KBar_df['low'], close=KBar_df['close'], name='K線'), secondary_y=True)
fig1.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
fig1.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_long'], mode='lines', line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
fig1.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_short'], mode='lines', line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
fig1.layout.yaxis2.showgrid = True
st.plotly_chart(fig1, use_container_width=True)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Candlestick(x=KBar_df['time'], open=KBar_df['open'], high=KBar_df['high'], low=KBar_df['low'], close=KBar_df['close'], name='K線'), secondary_y=True)
fig2.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_long'], mode='lines', line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), secondary_y=False)
fig2.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_short'], mode='lines', line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), secondary_y=False)
fig2.layout.yaxis2.showgrid = True
st.plotly_chart(fig2, use_container_width=True)










import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 讀取資料（假設您已經有一個名為 KBar_df 的 DataFrame，其中包含了時間、開盤價、最高價、最低價和收盤價等資料）
# 這裡假設 KBar_df 是您的 DataFrame

# 假設這是您的 DataFrame
# KBar_df = pd.read_csv("your_data.csv")

# 設定計算唐琪安通道的 K 棒數目
dc_window = st.slider('設定計算唐琪安通道的 K 棒數目(整數, 例如 20)', 0, 1000, 20, key="dc_slider")

# 設定計算布林通道的窗口大小
bb_window = st.slider('設定計算布林通道的窗口大小(整數, 例如 20)', 0, 1000, 20, key="bb_slider")

def calculate_donchian_channel(df, window):
    df['upper_dc'] = df['Close'].rolling(window=window).max()
    df['lower_dc'] = df['Close'].rolling(window=window).min()
    return df

# 計算布林通道
def calculate_bollinger_bands(df, window):
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['middle_bb'] = rolling_mean
    df['upper_bb'] = rolling_mean + (rolling_std * 2)
    df['lower_bb'] = rolling_mean - (rolling_std * 2)
    return df

# 計算唐琪安通道
KBar_df = calculate_donchian_channel(KBar_df, window=dc_window)

# 計算布林通道
KBar_df = calculate_bollinger_bands(KBar_df, window=bb_window)

# 唐奇安通道圖表
with st.expander("唐奇安通道圖"):
    fig_dc = go.Figure()
    fig_dc.add_trace(go.Candlestick(x=KBar_df['Time'], open=KBar_df['Open'], high=KBar_df['High'], low=KBar_df['Low'], close=KBar_df['Close'], name='K線'))
    fig_dc.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_dc'], mode='lines', line=dict(color='green'), name='Upper Donchian Channel'))
    fig_dc.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_dc'], mode='lines', line=dict(color='green'), name='Lower Donchian Channel'))
    fig_dc.update_layout(height=600, title_text="唐奇安通道")
    st.plotly_chart(fig_dc, use_container_width=True)

# 布林通道圖表
with st.expander("布林通道圖"):
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Candlestick(x=KBar_df['Time'], open=KBar_df['Open'], high=KBar_df['High'], low=KBar_df['Low'], close=KBar_df['Close'], name='K線'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_bb'], mode='lines', line=dict(color='blue'), name='Upper Bollinger Band'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_bb'], mode='lines', line=dict(color='blue'), name='Lower Bollinger Band'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['middle_bb'], mode='lines', line=dict(color='red'), name='Middle Bollinger Band'))
    fig_bb.update_layout(height=600, title_text="布林通道")
    st.plotly_chart(fig_bb, use_container_width=True)
