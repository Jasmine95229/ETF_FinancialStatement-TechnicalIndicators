#財精四C 07155328白祐瑄
#%%
#爬蟲抓 0050.TW 成分股
import requests
from bs4 import BeautifulSoup
import pandas as pd
res=requests.get('https://www.taifex.com.tw/cht/9/futuresQADetail')
soup=BeautifulSoup(res.text,'lxml')
data=soup.select_one('#printhere')
l=pd.read_html(data.prettify())

df=l[0]
new_col={x:y for x,y in zip(df.iloc[:,4:8].columns,df.iloc[:,0:4].columns)}
df0=df.iloc[:,0:4].append(df.iloc[:,4:8].rename(columns=new_col),ignore_index=True)
df0.columns=['rank','code','name','percent']
df0=df0.iloc[0:50,:]
df0['code']=[int(i) for i in df0['code']]
dict0={'成分股':df0}
#%%
import yfinance as yf

start='2009-04-01'
end='2021-04-30'
ticker_l=[str(i)+'.TW' for i in dict0['成分股']['code']]
df_yf=yf.download(ticker_l,start=start,end=end,auto_adjust=True,interval='1d')

df_yf.isna().mean()
nan_threshold = 0.01
[df_yf.isna().mean() <= nan_threshold]
df_yf = df_yf.iloc[:, list(df_yf.isna().mean() <= nan_threshold)]

df_yf.dropna(axis=0,inplace=True)

dict_yf={'close':df_yf['Close'],
         'high':df_yf['High'],
         'low':df_yf['Low'],
         'open':df_yf['Open'],
         'volume':df_yf['Volume']}

#%% TA-Lib
#https://mrjbq7.github.io/ta-lib/
#https://www.bookstack.cn/read/talib-zh/func_groups-overlap_studies.md
import talib
dict_talib_table=talib.get_function_groups()

#輸入不可以有空缺值！
df0=pd.DataFrame()
df0['close']=dict_yf['close']['2330.TW']

#MA(Moving Average)移動平均
#matype:計算平均線方法
#timeperiod:週期
df0['MA']=talib.MA(dict_yf['close']['2330.TW'],timeperiod=30,matype=0)

#SMA(Simple Moving Average)簡單移動平均
df0['SMA']=talib.SMA(dict_yf['close']['2330.TW'],timeperiod=60)

#WMA(Weighted Moving Average)加權移動平均
df0['WMA']=talib.WMA(dict_yf['close']['2330.TW'],timeperiod=60)

#EMA(Exponential Moving Average)指數移動平均
df0['EMA']=talib.EMA(dict_yf['close']['2330.TW'],timeperiod=60)

import matplotlib.pyplot as plt

df0[['close','SMA','WMA','EMA']].head(200).plot()
df0[['close','MA','WMA','EMA']].head(200).plot()

#MACD(Moving Average Convergence/Divergence)指數平滑異同移動平均
#fastperiod:短期指數移動平均線，slowperiod:長期指數移動平均線
#signalperiod:離差值DIF(fastperiod-slowperiod)的n指數移動平均線->DEM
#hist->DIF-DEM
macd,macdsignal,macdhist=talib.MACD(dict_yf['close']['2330.TW'],fastperiod=12,slowperiod=26,signalperiod=9)
df0['MACD'],df0['MACD_signal'],df0['MACD_hist']=macd,macdsignal,macdhist
fig,axes=plt.subplots(nrows=2)
df0['close'][100:200].plot(ax=axes[0])
df0[['MACD','MACD_signal','MACD_hist']].iloc[100:200,:].plot(ax=axes[1])

#RSI(Relative Strength Index)相對強弱指數
df0['RSI']=talib.RSI(dict_yf['close']['2330.TW'],timeperiod=30)
fig,axes=plt.subplots(nrows=2)
df0['close'].head(200).plot(ax=axes[0])
df0['RSI'].head(200).plot(ax=axes[1])

#BBANDS(Bollinger Bands)布林帶
#matype:計算平均線方法(bolling線的middle線)
upper,middle,lower=talib.BBANDS(dict_yf['close']['2330.TW'],timeperiod=10,nbdevup=2,nbdevdn=2,matype=0)
df0['BBU'],df0['BBM'],df0['BBL']=upper,middle,lower
df0[['close','BBU','BBM','BBL']].iloc[100:300,:].plot()

#MOM(Momentum)動量
df0['MOM']=talib.MOM(dict_yf['close']['2330.TW'],timeperiod=30)
df0['MOM'][100:300].plot()

#BETA(Beta)
df0['beta']=talib.BETA(dict_yf['high']['2330.TW'],dict_yf['low']['2330.TW'],timeperiod=5)
df0['beta'][100:300].plot()

#ADX(Average Directional Movement Index)平均動向指數
df0['ADX']=talib.ADX(dict_yf['high']['2330.TW'],
              dict_yf['low']['2330.TW'],
              dict_yf['close']['2330.TW'],timeperiod=14)

#STOCH(Stochastic)隨機指標
#https://rich01.com/what-is-kd-indicator/
#fastk_period=N
slowk,slowd=talib.STOCH(dict_yf['high']['2330.TW'],dict_yf['low']['2330.TW'],dict_yf['close']['2330.TW'],
                        fastk_period=5,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
df0['slow_K'],df0['slow_D']=slowk,slowd
fig,axes=plt.subplots(nrows=2)
df0['close'][100:200].plot(ax=axes[0])
df0[['slow_K','slow_D']].iloc[100:200,:].plot(ax=axes[1])

#%%

def ta_lib(ticker,func):
    if func=='sma':
        df=pd.DataFrame(talib.SMA(dict_yf['close'][ticker],timeperiod=60))
        df.columns=[ticker]
        return df
    elif func=='wma':
        df=pd.DataFrame(talib.WMA(dict_yf['close'][ticker],timeperiod=60))
        df.columns=[ticker]
        return df
    elif func=='ema':
        df=pd.DataFrame(talib.EMA(dict_yf['close'][ticker],timeperiod=60))
        df.columns=[ticker]
        return df
    elif func=='macd':
        macd,macdsignal,macdhist=talib.MACD(dict_yf['close'][ticker],fastperiod=12,slowperiod=26,signalperiod=9)
        macd=pd.DataFrame(macd,columns=[ticker])
        macdsignal=pd.DataFrame(macdsignal,columns=[ticker])
        macdhist=pd.DataFrame(macdhist,columns=[ticker])
        return macd, macdsignal,macdhist
    elif func=='rsi':
        df=pd.DataFrame(talib.RSI(dict_yf['close'][ticker],timeperiod=60))
        df.columns=[ticker]
        return df
    elif func=='bbands':
        upper,middle,lower=talib.BBANDS(dict_yf['close'][ticker],timeperiod=10,nbdevup=2,nbdevdn=2,matype=0)
        upper=pd.DataFrame(upper,columns=[ticker])
        middle=pd.DataFrame(middle,columns=[ticker])
        lower=pd.DataFrame(lower,columns=[ticker])
        return upper,middle,lower
    elif func=='mom':
        df=pd.DataFrame(talib.SMA(dict_yf['close'][ticker],timeperiod=60))
        df.columns=[ticker]
        return df
    elif func=='beta':
        df=pd.DataFrame(talib.BETA(dict_yf['high'][ticker],dict_yf['low'][ticker],timeperiod=5))
        df.columns=[ticker]
        return df
    elif func=='adx':
        df=pd.DataFrame(talib.ADX(dict_yf['high'][ticker],
                                  dict_yf['low'][ticker],
                                  dict_yf['close'][ticker],timeperiod=14))
        df.columns=[ticker]
        return df
    elif func=='kd':
        slowk,slowd=talib.STOCH(dict_yf['high'][ticker],dict_yf['low'][ticker],dict_yf['close'][ticker],
                                fastk_period=5,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
        slowk=pd.DataFrame(slowk,columns=[ticker])
        slowd=pd.DataFrame(slowd,columns=[ticker])
        return slowk,slowd

#%%  
func_l=['sma','wma','ema','macd','rsi','bbands','mom','beta','adx','kd']
ticker_l=dict_yf['close'].columns

for func in func_l:
    df0=pd.DataFrame()
    df11=pd.DataFrame()
    df22=pd.DataFrame()
    df33=pd.DataFrame()
    for ticker in ticker_l:
        if func in ['macd','bbands']:
            df1,df2,df3=ta_lib(ticker,func)
            df11=pd.concat([df11,df1],axis=1)
            df22=pd.concat([df22,df2],axis=1)
            df33=pd.concat([df33,df3],axis=1)
        elif func=='kd':
            df1,df2=ta_lib(ticker,func)
            df11=pd.concat([df11,df1],axis=1)
            df22=pd.concat([df22,df2],axis=1)
        else:
            df=ta_lib(ticker,func)
            df0=pd.concat([df0,df],axis=1)
    if func=='macd':
        dict0['macd'],dict0['macdsignal'],dict0['macdhist']=df11,df22,df33
    elif func=='bbands':
        dict0['bb_upper'],dict0['bb_middle'],dict0['bb_lower']=df11,df22,df33
    elif func=='kd':
        dict0['kd_k'],dict0['kd_d']=df11,df22
    else:
        dict0[func]=df0

import os
#os.makedirs('/Users/JasminePai/python/FINANCIAL_BIG_DATA/table')
os.chdir('/Users/JasminePai/python/FINANCIAL_BIG_DATA/table')
for i in dict0.keys():
    dict0[str(i)].to_csv(str(i)+'.csv')
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
