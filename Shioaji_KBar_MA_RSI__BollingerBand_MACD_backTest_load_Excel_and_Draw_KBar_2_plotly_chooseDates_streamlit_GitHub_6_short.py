# 載入必要模組
import os
# os.chdir(r'C:\Users\user\Dropbox\系務\專題實作\112\金融看板\for students')
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 


###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

#df = pd.read_excel("kbars_台積電_1100701_1100708_2.xlsx")
#df = pd.read_excel("kbars_2330_2022-07-01-2022-07-31.xlsx")

# ## 讀取 excel 檔
# df_original = pd.read_excel("kbars_2330_2022-01-01-2022-11-18.xlsx")

# ## 保存为Pickle文件:
# df_original.to_pickle('kbars_2330_2022-01-01-2022-11-18.pkl')

## 读取Pickle文件
df_original = pd.read_pickle('2201.pkl')


#df.columns  ## Index(['Unnamed: 0', 'time', 'open', 'low', 'high', 'close', 'volume','amount'], dtype='object')
df_original = df_original.drop('Unnamed: 0',axis=1)
#df.columns  ## Index(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'], dtype='object')
#df['time']
#type(df['time'])  ## pandas.core.series.Series
#df['time'][11]
#df.head()
#df.tail()
#type(df['time'][0])


##### 選擇資料區間
#st.set_page_config(layout = 'wide',initial_sidebar_state='expanded')
st.sidebar.header("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
Date = start_date.strftime("%Y-%m-%d")

st.sidebar.subheader("設定一根 K 棒的時間長度(分鐘)")
cycle_duration = st.sidebar.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)',value=1440, key="KBar_duration")
cycle_duration = int(cycle_duration)
#cycle_duration = 1440   ## 可以改成你想要的 KBar 週期
#KBar = indicator_f_Lo2.KBar(Date,'time',2)
KBar = indicator_forKBar_short.KBar(Date,cycle_duration) 

###### (2) 轉化為字典 ######:
KBar_dic = df.to_dict()
#type(KBar_dic)
#KBar_dic.keys()  ## dict_keys(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'])
#KBar_dic['open']
#type(KBar_dic['open'])  ## dict
#KBar_dic['open'].values()
#type(KBar_dic['open'].values())  ## dict_values
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open']=np.array(KBar_open_list)
#type(KBar_dic['open'])  ## numpy.ndarray
#KBar_dic['open'].shape  ## (1596,)
#KBar_dic['open'].size   ##  1596

KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
#KBar_dic['product'].size   ## 1596
#KBar_dic['product'][0]      ## 'tsmc'

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
KBar_dic['time']=np.array(KBar_time_list)

# KBar_time_list[0]        ## Timestamp('2022-07-01 09:01:00')
# type(KBar_time_list[0])  ## pandas._libs.tslibs.timestamps.Timestamp
#KBar_time_list[0].to_pydatetime() ## datetime.datetime(2022, 7, 1, 9, 1)
#KBar_time_list[0].to_numpy()      ## numpy.datetime64('2022-07-01T09:01:00.000000000')
#KBar_dic['time']=np.array(KBar_time_list)
#KBar_dic['time'][80]   ## Timestamp('2022-09-01 23:02:00')

KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low']=np.array(KBar_low_list)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high']=np.array(KBar_high_list)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close']=np.array(KBar_close_list)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume']=np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount']=np.array(KBar_amount_list)


######  (3) 改變 KBar 時間長度 (以下)  ########
# Product_array = np.array([])
# Time_array = np.array([])
# Open_array = np.array([])
# High_array = np.array([])
# Low_array = np.array([])
# Close_array = np.array([])
# Volume_array = np.array([])

Date = start_date.strftime("%Y-%m-%d")

st.sidebar.subheader("設定一根 K 棒的時間長度(天)")
cycle_duration_days = st.sidebar.number_input('輸入一根 K 棒的時間長度(單位:天)', value=1, key="KBar_duration_days")
cycle_duration = int(cycle_duration_days)  # 将天数转换为整数
# cycle_duration = 1  # 这里是默认值，你也可以直接设置为1，表示一天为一个K棒

KBar = indicator_forKBar_short.KBar(Date, cycle_duration)  # 创建KBar对象，将天数作为周期长度

#KBar_dic['amount'].shape   ##(5585,)
#KBar_dic['amount'].size    ##5585
#KBar_dic['time'].size    ##5585

for i in range(KBar_dic['time'].size):
    
    #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
    time = KBar_dic['time'][i]
    #prod = KBar_dic['product'][i]
    open_price= KBar_dic['open'][i]
    close_price= KBar_dic['close'][i]
    low_price= KBar_dic['low'][i]
    high_price= KBar_dic['high'][i]
    qty =  KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    #tag=KBar.TimeAdd(time,price,qty,prod)
    tag=KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    # 更新K棒才判斷，若要逐筆判斷則 註解下面兩行, 因為計算 MA是利用收盤價, 而在 KBar class 中的 "TimeAdd"函數方法中, 收盤價只是一直附加最新的 price 而已.
    #if tag != 1:
        #continue
    #print(KBar.Time,KBar.GetOpen(),KBar.GetHigh(),KBar.GetLow(),KBar.GetClose(),KBar.GetVolume()) 
    
    
        
# #type(KBar.Time[1:-1]) ##numpy.ndarray       
# Time_array =  np.append(Time_array, KBar.Time[1:-1])    
# Open_array =  np.append(Open_array,KBar.Open[1:-1])
# High_array =  np.append(High_array,KBar.High[1:-1])
# Low_array =  np.append(Low_array,KBar.Low[1:-1])
# Close_array =  np.append(Close_array,KBar.Close[1:-1])
# Volume_array =  np.append(Volume_array,KBar.Volume[1:-1])
# Product_array = np.append(Product_array,KBar.Prod[1:-1])

KBar_dic = {}

# ## 形成 KBar 字典:
# KBar_dic['time'] =  Time_array   
# KBar_dic['product'] =  Product_array
# KBar_dic['open'] =  Open_array
# KBar_dic['high'] =  High_array
# KBar_dic['low'] =  Low_array
# KBar_dic['close'] =  Close_array
# KBar_dic['volume'] =  Volume_array

 ## 形成 KBar 字典 (新週期的):
KBar_dic['time'] =  KBar.TAKBar['time']   
#KBar_dic['product'] =  KBar.TAKBar['product']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] =  KBar.TAKBar['high']
KBar_dic['low'] =  KBar.TAKBar['low']
KBar_dic['close'] =  KBar.TAKBar['close']
KBar_dic['volume'] =  KBar.TAKBar['volume']
# KBar_dic['time'].shape  ## (2814,)
# KBar_dic['open'].shape  ## (2814,)
# KBar_dic['high'].shape  ## (2814,)
# KBar_dic['low'].shape  ## (2814,)
# KBar_dic['close'].shape  ## (2814,)
# KBar_dic['volume'].shape  ## (2814,)
#KBar_dic['time'][536]
######  改變 KBar 時間長度 (以上)  ########



###### (4) 計算各種技術指標 ######
##### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)


#####  (i) 移動平均線策略   #####
####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
#LongMAPeriod=st.number_input('輸入一個整數', key="Long_MA")
#LongMAPeriod=int(LongMAPeriod)
LongMAPeriod=st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
#ShortMAPeriod=st.number_input('輸入一個整數', key="Short_MA")
#ShortMAPeriod=int(ShortMAPeriod)
ShortMAPeriod=st.slider('選擇一個整數', 0, 100, 2)

#### 計算長短移動平均線
KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

#### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]



#####  (ii) RSI 策略   #####
#### 順勢策略
### 設定長短 RSI 的 K棒 長度:
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10, key='long_rsi_period')

st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2, key='short_rsi_period')

### 計算 RSI指標長短線, 以及定義中線
## 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle']=np.array([50]*len(KBar_dic['time']))

### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

# 設定計算唐琪安通道的 K 棒數目
dc_window = st.slider('設定計算唐琪安通道的 K 棒數目(整數, 例如 20)', 0, 1000, 20)

# #### 逆勢策略
# ### 建立部位管理物件
# OrderRecord=Record() 
# ### 計算 RSI指標, 天花板與地板
# RSIPeriod=5
# Ceil=80
# Floor=20
# MoveStopLoss=30
# KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
# KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
# KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))

# ### 將K線 Dictionary 轉換成 Dataframe
# KBar_RSI_df=pd.DataFrame(KBar_dic)


###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]


###### (6) 畫圖 ######
st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
import plotly.offline as pyoff


##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)


##### K線圖, RSI
with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    #### include candlestick with rangeselector
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
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
    fig_dc.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_dc'], mode='lines', line=dict(color='green'), name='Upper Donchian Channel'))
    fig_dc.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_dc'], mode='lines', line=dict(color='green'), name='Lower Donchian Channel'))
    fig_dc.update_layout(height=600, title_text="唐奇安通道")
    st.plotly_chart(fig_dc, use_container_width=True)

# 布林通道圖表
with st.expander("布林通道圖"):
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_bb'], mode='lines', line=dict(color='blue'), name='Upper Bollinger Band'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_bb'], mode='lines', line=dict(color='blue'), name='Lower Bollinger Band'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['middle_bb'], mode='lines', line=dict(color='red'), name='Middle Bollinger Band'))
    fig_bb.update_layout(height=600, title_text="布林通道")
    st.plotly_chart(fig_bb, use_container_width=True)
ChatGPT
ChatGPT
您想要將唐奇安通道和布林通道分開成兩张图。为此，我将更新代码以分别绘制唐奇安通道和布林通道的图表。以下是更新后的代码：

python
複製程式碼
# 計算唐琪安通道
def calculate_donchian_channel(df, window):
    df['upper_dc'] = df['Close'].rolling(window=window).max()
    df['lower_dc'] = df['Close'].rolling
你
# 載入必要模組
import os
# os.chdir(r'C:\Users\user\Dropbox\系務\專題實作\112\金融看板\for students')
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 


###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

#df = pd.read_excel("kbars_台積電_1100701_1100708_2.xlsx")
#df = pd.read_excel("kbars_2330_2022-07-01-2022-07-31.xlsx")

# ## 讀取 excel 檔
# df_original = pd.read_excel("kbars_2330_2022-01-01-2022-11-18.xlsx")

# ## 保存为Pickle文件:
# df_original.to_pickle('kbars_2330_2022-01-01-2022-11-18.pkl')

## 读取Pickle文件
df_original = pd.read_pickle('2201.pkl')


#df.columns  ## Index(['Unnamed: 0', 'time', 'open', 'low', 'high', 'close', 'volume','amount'], dtype='object')
df_original = df_original.drop('Unnamed: 0',axis=1)
#df.columns  ## Index(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'], dtype='object')
#df['time']
#type(df['time'])  ## pandas.core.series.Series
#df['time'][11]
#df.head()
#df.tail()
#type(df['time'][0])


##### 選擇資料區間
#st.set_page_config(layout = 'wide',initial_sidebar_state='expanded')
st.sidebar.header("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
import datetime

# 用户输入的字符串日期
start_date_str = st.sidebar.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date_str = st.sidebar.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')

# 将字符串日期转换为 datetime.datetime 对象
start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

# 确保比较的对象都是 datetime.datetime 类型
if start_date <= end_date:
    st.write("Start date is before or equal to end date.")
else:
    st.write("Start date is after end date.")


###### (2) 轉化為字典 ######:
KBar_dic = df.to_dict()
#type(KBar_dic)
#KBar_dic.keys()  ## dict_keys(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'])
#KBar_dic['open']
#type(KBar_dic['open'])  ## dict
#KBar_dic['open'].values()
#type(KBar_dic['open'].values())  ## dict_values
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open']=np.array(KBar_open_list)
#type(KBar_dic['open'])  ## numpy.ndarray
#KBar_dic['open'].shape  ## (1596,)
#KBar_dic['open'].size   ##  1596

KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
#KBar_dic['product'].size   ## 1596
#KBar_dic['product'][0]      ## 'tsmc'

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
KBar_dic['time']=np.array(KBar_time_list)

# KBar_time_list[0]        ## Timestamp('2022-07-01 09:01:00')
# type(KBar_time_list[0])  ## pandas._libs.tslibs.timestamps.Timestamp
#KBar_time_list[0].to_pydatetime() ## datetime.datetime(2022, 7, 1, 9, 1)
#KBar_time_list[0].to_numpy()      ## numpy.datetime64('2022-07-01T09:01:00.000000000')
#KBar_dic['time']=np.array(KBar_time_list)
#KBar_dic['time'][80]   ## Timestamp('2022-09-01 23:02:00')

KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low']=np.array(KBar_low_list)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high']=np.array(KBar_high_list)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close']=np.array(KBar_close_list)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume']=np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount']=np.array(KBar_amount_list)


######  (3) 改變 KBar 時間長度 (以下)  ########
# Product_array = np.array([])
# Time_array = np.array([])
# Open_array = np.array([])
# High_array = np.array([])
# Low_array = np.array([])
# Close_array = np.array([])
# Volume_array = np.array([])

Date = start_date.strftime("%Y-%m-%d")

st.sidebar.subheader("設定一根 K 棒的時間長度(天)")
cycle_duration_days = st.sidebar.number_input('輸入一根 K 棒的時間長度(單位:天)', value=1, key="KBar_duration_days")
cycle_duration = int(cycle_duration_days)  # 将天数转换为整数
# cycle_duration = 1  # 这里是默认值，你也可以直接设置为1，表示一天为一个K棒

KBar = indicator_forKBar_short.KBar(Date, cycle_duration)  # 创建KBar对象，将天数作为周期长度

#KBar_dic['amount'].shape   ##(5585,)
#KBar_dic['amount'].size    ##5585
#KBar_dic['time'].size    ##5585

for i in range(KBar_dic['time'].size):
    
    #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
    time = KBar_dic['time'][i]
    #prod = KBar_dic['product'][i]
    open_price= KBar_dic['open'][i]
    close_price= KBar_dic['close'][i]
    low_price= KBar_dic['low'][i]
    high_price= KBar_dic['high'][i]
    qty =  KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    #tag=KBar.TimeAdd(time,price,qty,prod)
    tag=KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    # 更新K棒才判斷，若要逐筆判斷則 註解下面兩行, 因為計算 MA是利用收盤價, 而在 KBar class 中的 "TimeAdd"函數方法中, 收盤價只是一直附加最新的 price 而已.
    #if tag != 1:
        #continue
    #print(KBar.Time,KBar.GetOpen(),KBar.GetHigh(),KBar.GetLow(),KBar.GetClose(),KBar.GetVolume()) 
    
    
        
# #type(KBar.Time[1:-1]) ##numpy.ndarray       
# Time_array =  np.append(Time_array, KBar.Time[1:-1])    
# Open_array =  np.append(Open_array,KBar.Open[1:-1])
# High_array =  np.append(High_array,KBar.High[1:-1])
# Low_array =  np.append(Low_array,KBar.Low[1:-1])
# Close_array =  np.append(Close_array,KBar.Close[1:-1])
# Volume_array =  np.append(Volume_array,KBar.Volume[1:-1])
# Product_array = np.append(Product_array,KBar.Prod[1:-1])

KBar_dic = {}

# ## 形成 KBar 字典:
# KBar_dic['time'] =  Time_array   
# KBar_dic['product'] =  Product_array
# KBar_dic['open'] =  Open_array
# KBar_dic['high'] =  High_array
# KBar_dic['low'] =  Low_array
# KBar_dic['close'] =  Close_array
# KBar_dic['volume'] =  Volume_array

 ## 形成 KBar 字典 (新週期的):
KBar_dic['time'] =  KBar.TAKBar['time']   
#KBar_dic['product'] =  KBar.TAKBar['product']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] =  KBar.TAKBar['high']
KBar_dic['low'] =  KBar.TAKBar['low']
KBar_dic['close'] =  KBar.TAKBar['close']
KBar_dic['volume'] =  KBar.TAKBar['volume']
# KBar_dic['time'].shape  ## (2814,)
# KBar_dic['open'].shape  ## (2814,)
# KBar_dic['high'].shape  ## (2814,)
# KBar_dic['low'].shape  ## (2814,)
# KBar_dic['close'].shape  ## (2814,)
# KBar_dic['volume'].shape  ## (2814,)
#KBar_dic['time'][536]
######  改變 KBar 時間長度 (以上)  ########



###### (4) 計算各種技術指標 ######
##### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)


#####  (i) 移動平均線策略   #####
####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
#LongMAPeriod=st.number_input('輸入一個整數', key="Long_MA")
#LongMAPeriod=int(LongMAPeriod)
LongMAPeriod=st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
#ShortMAPeriod=st.number_input('輸入一個整數', key="Short_MA")
#ShortMAPeriod=int(ShortMAPeriod)
ShortMAPeriod=st.slider('選擇一個整數', 0, 100, 2)

#### 計算長短移動平均線
KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

#### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]



#####  (ii) RSI 策略   #####
#### 順勢策略
### 設定長短 RSI 的 K棒 長度:
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10, key='long_rsi_period')

st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2, key='short_rsi_period')

### 計算 RSI指標長短線, 以及定義中線
## 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle']=np.array([50]*len(KBar_dic['time']))

### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

# 設定計算唐琪安通道的 K 棒數目
dc_window = st.slider('設定計算唐琪安通道的 K 棒數目(整數, 例如 20)', 0, 1000, 20)

# #### 逆勢策略
# ### 建立部位管理物件
# OrderRecord=Record() 
# ### 計算 RSI指標, 天花板與地板
# RSIPeriod=5
# Ceil=80
# Floor=20
# MoveStopLoss=30
# KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
# KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
# KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))

# ### 將K線 Dictionary 轉換成 Dataframe
# KBar_RSI_df=pd.DataFrame(KBar_dic)


###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]


###### (6) 畫圖 ######
st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
import plotly.offline as pyoff


##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)


##### K線圖, RSI
with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    #### include candlestick with rangeselector
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
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
    fig_dc.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_dc'], mode='lines', line=dict(color='green'), name='Upper Donchian Channel'))
    fig_dc.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_dc'], mode='lines', line=dict(color='green'), name='Lower Donchian Channel'))
    fig_dc.update_layout(height=600, title_text="唐奇安通道")
    st.plotly_chart(fig_dc, use_container_width=True)

# 布林通道圖表
with st.expander("布林通道圖"):
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['upper_bb'], mode='lines', line=dict(color='blue'), name='Upper Bollinger Band'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['lower_bb'], mode='lines', line=dict(color='blue'), name='Lower Bollinger Band'))
    fig_bb.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['middle_bb'], mode='lines', line=dict(color='red'), name='Middle Bollinger Band'))
    fig_bb.update_layout(height=600, title_text="布林通道")
    st.plotly_chart(fig_bb, use_container_width=True)
